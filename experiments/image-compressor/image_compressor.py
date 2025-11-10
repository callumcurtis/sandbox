import sys
from typing import BinaryIO, Iterator

import numpy as np

width, height = 352, 288
chroma_ratio_width, chroma_ratio_height = 2, 2
Y_size = width*height
Cb_size = width*height//chroma_ratio_width//chroma_ratio_height
Cr_size = width*height//chroma_ratio_width//chroma_ratio_height
chroma_height = height//chroma_ratio_height
chroma_width = width//chroma_ratio_width
block_size = 8
num_planes = 3

class Config:
    quantization_enabled: bool = True
    # Delta-ing the DCT DC and AC coefficients within a block is not effective as the coefficients are independent.
    intra_block_delta_dct_coeff_enabled: bool = False
    # Delta-ing the DCT DC coefficients between blocks is effective as nearby blocks are likely to have similar values.
    inter_block_delta_dct_dc_coeff_enabled: bool = True
    # Enabling truncation will introduce ringing artifacts, commonly seen as "blockiness"
    #
    # From https://commons.und.edu/cgi/viewcontent.cgi?article=4090&context=theses
    # """
    # The high frequency component makes the right-side wave more
    # square shaped. The high frequencies represent sharp contrast edges where this is a
    # lot of pixel intensity variation. So, if we lose or truncate high frequencies during
    # JPEG compression then the image will have ringing artifacts.
    # """
    truncate_to: int = 10
    truncate_enabled: bool = False


config = Config()
if not config.truncate_enabled:
    config.truncate_to = block_size**2


# Read the YUV file from stdin in bytes
yuv_bytes = sys.stdin.buffer.read()
assert len(yuv_bytes) == Y_size + Cb_size + Cr_size

# Create Y, Cb, and Cr arrays
Y = np.frombuffer(yuv_bytes[:Y_size], dtype=np.uint8)
Cb = np.frombuffer(yuv_bytes[Y_size:Y_size+Cb_size], dtype=np.uint8)
Cr = np.frombuffer(yuv_bytes[Y_size+Cb_size:Y_size+Cb_size+Cr_size], dtype=np.uint8)

# Resize the planes to be 2D
Y = Y.reshape(height, width)
Cb = Cb.reshape(chroma_height, chroma_width)
Cr = Cr.reshape(chroma_height, chroma_width)

def do_dct(block: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    return dct_matrix @ block @ dct_matrix.T

def undo_dct(block: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    return dct_matrix.T @ block @ dct_matrix

def do_blockify(plane: np.ndarray, block_size: int) -> list[list[np.ndarray]]:
    # Pad the width to be divisible by the block size by repeating the last column
    padded_plane = np.pad(plane, ((0, 0), (0, block_size - (plane.shape[1] % block_size) if plane.shape[1] % block_size != 0 else 0)), mode='edge')
    # Pad the height to be divisible by the block size by repeating the last row
    padded_plane = np.pad(padded_plane, ((0, block_size - (plane.shape[0] % block_size) if plane.shape[0] % block_size != 0 else 0), (0, 0)), mode='edge')
    blocks_wide = padded_plane.shape[1] // block_size
    blocks_high = padded_plane.shape[0] // block_size
    blocks = np.split(padded_plane, blocks_high, axis=0)
    blocks = [np.split(block, blocks_wide, axis=1) for block in blocks]
    return blocks

def undo_blockify(blocks: list[list[np.ndarray | None]]) -> np.ndarray:
    return np.concatenate([np.concatenate(row, axis=1) for row in blocks], axis=0)[:height,:width]

def do_flatten_blocks(blocks: list[list[np.ndarray]]) -> list[np.ndarray]:
    x = [block for sublist in blocks for block in sublist]
    return x

def make_unflattened_block_container(width: int, height: int, block_size: int) -> list[list[None]]:
    blocks_wide = (width+block_size-1)//block_size
    blocks_high = (height+block_size-1)//block_size
    return [[None] * blocks_wide for _ in range(blocks_high)]

def undo_flatten_block(unflattened_blocks: list[list[np.ndarray | None]], block: np.ndarray, block_index: int) -> list[list[np.ndarray | None]]:
    block_row = block_index // len(unflattened_blocks[0])
    block_col = block_index % len(unflattened_blocks[0])
    unflattened_blocks[block_row][block_col] = block

def make_dct_matrix(size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if i == 0:
                matrix[i, j] = 1/np.sqrt(size)
            else:
                matrix[i, j] = np.cos((2*j+1)*i*np.pi/(2*size)) * np.sqrt(2/size)
    return matrix

def do_quantize(block: np.ndarray, quantization_matrix: np.ndarray | int) -> np.ndarray:
    return np.round(block / quantization_matrix)

def clip_and_convert_to_dtype(block: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.clip(block, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

def do_map_to_unsigned_int(block: np.ndarray) -> np.ndarray:
    # Interleave negative and positive values into a single unsigned int8.
    # All negative numbers are multiplied by -2 and subtracted by 1. All positive numbers are multiplied by 2.
    # We are using uint8 to represent the result, so the allowed range of the original data is 7 bits.
    assert block.dtype == np.int8
    unsigned_block = np.zeros_like(block, dtype=np.uint8)
    unsigned_block[block > 0] = block[block > 0] * 2
    unsigned_block[block < 0] = block[block < 0] * -2 - 1
    return unsigned_block

def undo_map_to_unsigned_int(block: np.ndarray) -> np.ndarray:
    # All odd non-zero numbers are actually negative, so are added with one and then divided by -2.
    # All even non-zero numbers are actually positive, so are divided by 2.
    assert block.dtype == np.uint8
    signed_block = np.zeros_like(block, dtype=np.int8)
    signed_block[block % 2 == 1] = block[block % 2 == 1] // -2
    signed_block[block % 2 == 0] = block[block % 2 == 0] // 2
    return signed_block

def undo_quantize(block: np.ndarray, quantization_matrix: np.ndarray) -> np.ndarray:
    return block * quantization_matrix

def do_zigzag(block: np.ndarray) -> np.ndarray:
    row, col = 0, 0
    inserted = 0
    zigzagged = np.zeros(block.size, dtype=block.dtype)

    for _ in range(block.size):
        zigzagged[inserted] = block[row, col]
        inserted += 1
        if (row + col) % 2 == 0:
            if col == block.shape[1] - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:
            if row == block.shape[0] - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1

    return zigzagged

def undo_zigzag(zigzagged: np.ndarray) -> np.ndarray:
    row, col = 0, 0
    height = width = np.sqrt(zigzagged.size).astype(int)
    unzigzagged = np.zeros((height, width), dtype=zigzagged.dtype)
    for i in range(zigzagged.size):
        unzigzagged[row, col] = zigzagged[i]
        if (row + col) % 2 == 0:
            if col == unzigzagged.shape[1] - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:
            if row == unzigzagged.shape[0] - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1
    return unzigzagged

def do_truncate(sequence: np.ndarray, truncate_to: int) -> np.ndarray:
    return sequence[:truncate_to]

def undo_truncate(sequence: np.ndarray, truncate_to: int, block_size: int) -> np.ndarray:
    assert len(sequence) == truncate_to
    padding_size = block_size**2 - truncate_to
    assert len(sequence) + padding_size == block_size**2
    return np.concatenate([sequence, np.zeros(padding_size, dtype=sequence.dtype)])

def do_delta_within_block(sequence: np.ndarray) -> np.ndarray:
    # Use the previous value to predict the next value
    return np.concatenate([[sequence[0]], sequence[1:] - sequence[:-1]])

def undo_delta_within_block(delta: np.ndarray) -> np.ndarray:
    return np.cumsum(delta)

def get_nearby_block_ind(block_ind: int, blocks_wide: int) -> int:
    return block_ind - 1 if block_ind % blocks_wide != 0 else block_ind - blocks_wide

def do_delta_between_blocks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return second - first

def undo_delta_between_blocks(first: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return first + delta

class PrefixCode:

    def __init__(self, prefix_code: int, num_bits: int):
        self.prefix_code = prefix_code
        self.num_bits = num_bits

    def __repr__(self) -> str:
        return f"PrefixCode(prefix_code={bin(self.prefix_code)}, num_bits={self.num_bits})"

class Offset:

    def __init__(self, start: int, num_bits: int):
        self.start = start
        self.num_bits = num_bits

    def __repr__(self) -> str:
        return f"Offset(start={self.start}, num_bits={self.num_bits})"

class ByteWriter:

    def __init__(self, f: BinaryIO):
        self.f = f
        self.byte = 0
        self.bit_index = 0
        self.total_bytes_written = 0

    def write_integral(self, value: int, num_bits: int):
        assert 0 <= value < 2**num_bits
        for i in range(num_bits):
            self.byte |= ((value >> (num_bits - i - 1)) & 1) << (7 - self.bit_index)
            self.bit_index += 1
            if self.bit_index == 8:
                self.f.write(bytes([self.byte]))
                self.byte = 0
                self.bit_index = 0
                self.total_bytes_written += 1
        return self

    def flush(self):
        if self.bit_index > 0:
            self.f.write(bytes([self.byte]))
            self.byte = 0
            self.bit_index = 0
            self.total_bytes_written += 1
        return self

class ByteReader:

    def __init__(self, f: BinaryIO):
        self.f = f
        self.byte = 0
        self.bit_index = 0
        self.total_bytes_read = 0
        self.cache_next_byte_()
        self.exhausted = False

    def read_bits(self, num_bits: int) -> int:
        assert not self.exhausted
        out = 0
        for i in range(num_bits):
            out |= ((self.byte >> (7 - self.bit_index)) & 1) << (num_bits - i - 1)
            self.bit_index += 1
            if self.bit_index == 8:
                self.cache_next_byte_()
        return out

    def cache_next_byte_(self) -> None:
        byte = self.f.read(1)
        if not byte:
            self.exhausted = True
            return
        # TODO: enforce byteorder for portability
        self.byte = int.from_bytes(byte, byteorder=sys.byteorder)
        self.bit_index = 0
        self.total_bytes_read += 1

    def read_byte(self) -> int:
        return self.read_bits(8)


def expand_prefix_codes_and_offsets(prefix_codes_and_offsets: list[tuple[PrefixCode, Offset]]) -> list[tuple[PrefixCode, Offset]]:
    out = []
    for prefix_code, offset in prefix_codes_and_offsets:
        for i in range(offset.start, offset.start + 2**offset.num_bits):
            out.append((prefix_code, offset))
    return out


prefix_codes_and_offsets_for_runs_of_zeros = [
    (PrefixCode(prefix_code=0b0, num_bits=1), Offset(start=0, num_bits=0)),
    (PrefixCode(prefix_code=0b10, num_bits=2), Offset(start=1, num_bits=1)),
    (PrefixCode(prefix_code=0b1110, num_bits=4), Offset(start=3, num_bits=2)),
    (PrefixCode(prefix_code=0b111110, num_bits=6), Offset(start=7, num_bits=3)),
    (PrefixCode(prefix_code=0b111111, num_bits=6), Offset(start=15, num_bits=4)),
    (PrefixCode(prefix_code=0b110, num_bits=3), Offset(start=31, num_bits=5)),
    (PrefixCode(prefix_code=0b11110, num_bits=5), Offset(start=63, num_bits=1)),
]
prefix_code_and_offset_by_length_of_run_of_zeros = expand_prefix_codes_and_offsets(prefix_codes_and_offsets_for_runs_of_zeros)
run_of_zeros_offset_by_prefix_code = {
    prefix_code: offset
    for prefix_code, offset in prefix_codes_and_offsets_for_runs_of_zeros
}
bins_for_prefix_codes_for_lengths_of_runs_of_zeros = {prefix_code: 0 for prefix_code, _ in prefix_codes_and_offsets_for_runs_of_zeros}


prefix_codes_and_offsets_for_non_zero_values = [
    (PrefixCode(prefix_code=0b1100, num_bits=4), Offset(start=0, num_bits=0)),
    (PrefixCode(prefix_code=0b0, num_bits=1), Offset(start=1, num_bits=1)),
    (PrefixCode(prefix_code=0b10, num_bits=2), Offset(start=3, num_bits=2)),
    (PrefixCode(prefix_code=0b1101, num_bits=4), Offset(start=7, num_bits=3)),
    (PrefixCode(prefix_code=0b11110, num_bits=5), Offset(start=15, num_bits=4)),
    (PrefixCode(prefix_code=0b111110, num_bits=6), Offset(start=31, num_bits=5)),
    (PrefixCode(prefix_code=0b111111, num_bits=6), Offset(start=63, num_bits=6)),
    (PrefixCode(prefix_code=0b1110, num_bits=4), Offset(start=127, num_bits=7)),
]
prefix_code_and_offset_by_non_zero_value = expand_prefix_codes_and_offsets(prefix_codes_and_offsets_for_non_zero_values)
non_zero_value_offset_by_prefix_code = {
    prefix_code: offset
    for prefix_code, offset in prefix_codes_and_offsets_for_non_zero_values
}
bins_for_prefix_codes_for_non_zero_values = {prefix_code: 0 for prefix_code, _ in prefix_codes_and_offsets_for_non_zero_values}


def do_clip_to_range_of_non_zero_values(block: np.ndarray) -> np.ndarray:
    largest_offset = prefix_codes_and_offsets_for_non_zero_values[-1][1]
    maximum = largest_offset.start + 2**largest_offset.num_bits - 1
    difference = block[block > maximum] - maximum
    # Make sure to subtract by an even number since we are in the interleaved/unsigned int world
    difference += (difference % 2)
    block[block > maximum] -= difference
    return block

class PrefixCodeNode:

    def __init__(self, prefix_code: PrefixCode | None = None):
        self.zero = None
        self.one = None
        self.prefix_code = prefix_code

def build_prefix_code_tree(prefix_codes: list[PrefixCode]) -> PrefixCodeNode:
    root = PrefixCodeNode()

    def add_prefix_code(prefix_code: PrefixCode) -> None:
        curr = root
        for i in range(prefix_code.num_bits-1, -1, -1):
            is_one = bool(prefix_code.prefix_code >> i & 1)
            if is_one:
                if not curr.one:
                    curr.one = PrefixCodeNode(prefix_code=prefix_code if i == 0 else None)
                curr = curr.one
            else:
                if not curr.zero:
                    curr.zero = PrefixCodeNode(prefix_code=prefix_code if i == 0 else None)
                curr = curr.zero

    for prefix_code in prefix_codes:
        add_prefix_code(prefix_code)

    return root

non_zero_value_prefix_code_tree = build_prefix_code_tree([prefix_code for prefix_code, _ in prefix_codes_and_offsets_for_non_zero_values])

def decode_prefix_code(byte_reader: ByteReader, prefix_tree_root: PrefixCodeNode) -> PrefixCode:
    curr = prefix_tree_root
    while curr and not curr.prefix_code:
        bit = byte_reader.read_bits(1)
        if bit & 1:
            curr = curr.one
        else:
            curr = curr.zero
    assert curr and curr.prefix_code is not None
    return curr.prefix_code

run_of_zeros_prefix_code_tree = build_prefix_code_tree([
    prefix_code for prefix_code in run_of_zeros_offset_by_prefix_code
])

def write_entropy_encoded_block(block: np.ndarray, byte_writer: ByteWriter):
    assert block.dtype == np.uint8
    assert block.shape == (block_size ** 2,)

    def prefix_code_and_offset_for_non_zero_value(value: int) -> tuple[PrefixCode, Offset]:
        prefix_code, offset = prefix_code_and_offset_by_non_zero_value[value]
        bins_for_prefix_codes_for_non_zero_values[prefix_code] += 1
        return prefix_code, offset

    def prefix_code_and_offset_for_run_of_zeros(length: int) -> tuple[PrefixCode, Offset]:
        prefix_code, offset = prefix_code_and_offset_by_length_of_run_of_zeros[length]
        bins_for_prefix_codes_for_lengths_of_runs_of_zeros[prefix_code] += 1
        return prefix_code, offset

    num_coeffs_written = 0
    while num_coeffs_written < block.size:
        non_zero_value = block[num_coeffs_written]
        prefix_code, offset = prefix_code_and_offset_for_non_zero_value(non_zero_value)
        byte_writer.write_integral(prefix_code.prefix_code, prefix_code.num_bits)
        if offset.num_bits > 0:
            byte_writer.write_integral(non_zero_value - offset.start, offset.num_bits)
        num_coeffs_written += 1
        if num_coeffs_written == block.size:
            break
        run_of_zeros_length = 0
        while num_coeffs_written < block.size and block[num_coeffs_written] == 0:
            run_of_zeros_length += 1
            num_coeffs_written += 1
        prefix_code, offset = prefix_code_and_offset_for_run_of_zeros(run_of_zeros_length)
        byte_writer.write_integral(prefix_code.prefix_code, prefix_code.num_bits)
        if offset.num_bits > 0:
            byte_writer.write_integral(run_of_zeros_length - offset.start, offset.num_bits)

def write_plain_block(block: np.ndarray, byte_writer: ByteWriter):
    assert block.shape == (block_size**2,)
    assert block.dtype == np.uint8
    for value in block:
        byte_writer.write_integral(value, 8)

def read_entropy_encoded_blocks(byte_reader: ByteReader, num_blocks: int) -> Iterator[np.ndarray]:
    block_ind = 0
    while block_ind < num_blocks:
        block = np.zeros((block_size ** 2), dtype=np.uint8)
        num_bytes_in_block = 0
        while num_bytes_in_block < block_size ** 2:
            # Read non-zero value
            non_zero_value_prefix_code = decode_prefix_code(byte_reader, non_zero_value_prefix_code_tree)
            non_zero_value_offset = non_zero_value_offset_by_prefix_code[non_zero_value_prefix_code]
            if non_zero_value_offset.num_bits > 0:
                non_zero_value = byte_reader.read_bits(non_zero_value_offset.num_bits) + non_zero_value_offset.start
            else:
                non_zero_value = non_zero_value_offset.start
            block[num_bytes_in_block] = non_zero_value
            num_bytes_in_block += 1
            if num_bytes_in_block >= block_size ** 2:
                break
            # Read run of zeros
            run_of_zeros_prefix_code = decode_prefix_code(byte_reader, run_of_zeros_prefix_code_tree)
            run_of_zeros_offset = run_of_zeros_offset_by_prefix_code[run_of_zeros_prefix_code]
            if run_of_zeros_offset.num_bits > 0:
                run_of_zeros_length = byte_reader.read_bits(run_of_zeros_offset.num_bits) + run_of_zeros_offset.start
            else:
                run_of_zeros_length = run_of_zeros_offset.start
            # The numpy array is already zero-initialize
            num_bytes_in_block += run_of_zeros_length
        yield block
        block_ind += 1

def read_plain_blocks(byte_reader: ByteReader, num_blocks: int) -> Iterator[np.ndarray]:
    block = np.zeros((block_size ** 2), dtype=np.uint8)
    num_bytes_in_block = 0
    current_block = 0
    while byte := byte_reader.read_byte():
        block[num_bytes_in_block] = byte
        num_bytes_in_block += 1
        if num_bytes_in_block == block_size ** 2:
            yield block
            current_block += 1
            if current_block == num_blocks:
                return
            num_bytes_in_block = 0
            block = np.zeros((block_size ** 2), dtype=np.uint8)
    assert num_bytes_in_block == 0

dct_matrix = make_dct_matrix(block_size)
luminance_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32) * 2
chroma_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)

fwrite = open("compressed", "wb")

byte_writer = ByteWriter(fwrite)

# Compress
for i, (plane, quantization_matrix) in enumerate([
    (Y, luminance_quantization_matrix),
    (Cb, chroma_quantization_matrix),
    (Cr, chroma_quantization_matrix),
]):
    transmitted_dc_coeffs = np.zeros((plane.shape[0]*plane.shape[1])//(block_size**2), dtype=np.int32)
    for j, block in enumerate(do_flatten_blocks(do_blockify(plane, block_size))):
        do_print = i == 0 and j == 0
        if do_print:
            print(f"(Compression) original block:\n{block}")
        block = do_dct(block, dct_matrix)
        if do_print:
            print(f"(Compression) DCTed block:\n{block}")
        if config.quantization_enabled:
            block = do_quantize(block, quantization_matrix)
            if do_print:
                print(f"(Compression) quantized block:\n{block}")
        block = clip_and_convert_to_dtype(block, np.int8)
        if do_print:
            print(f"(Compression) clipped and converted to int block:\n{block}")
        block = do_map_to_unsigned_int(block)
        if do_print:
            print(f"(Compression) converted to unsigned block:\n{block}")
        block = do_clip_to_range_of_non_zero_values(block)
        if do_print:
            print(f"(Compression) clipped to range of non-zero values block:\n{block}")
        if config.inter_block_delta_dct_dc_coeff_enabled and j > 0:
            transmitted_dc_coeffs[j] = block[0][0]
            nearby_block_ind = get_nearby_block_ind(j, plane.shape[1]//block_size)
            # TODO: try multiple nearby blocks (above and to the right)
            block[0][0] = do_delta_between_blocks(transmitted_dc_coeffs[nearby_block_ind], transmitted_dc_coeffs[j])
            if do_print:
                print(f"(Compression) inter-delta'd DC coefficient block:\n{block}")
        block = do_zigzag(block)
        if do_print:
            print(f"(Compression) zigzagged block:\n{block}")
        block = do_truncate(block, config.truncate_to)
        if do_print:
            print(f"(Compression) truncated block:\n{block}")
        if config.intra_block_delta_dct_coeff_enabled:
            block = do_delta_within_block(block)
            if do_print:
                print("(Compression) intra-delta'd AC coefficients block:")
        write_entropy_encoded_block(block, byte_writer)
        # write_plain_block(block, byte_writer)

byte_writer.flush()
fwrite.close()

print("\n\n")
print(f"Total bytes written: {byte_writer.total_bytes_written}")
print(f"Frequencies of prefix codes for non-zero values: {bins_for_prefix_codes_for_non_zero_values}")
print(f"Frequencies of prefix codes for lengths of runs of zeros: {bins_for_prefix_codes_for_lengths_of_runs_of_zeros}")
print("\n\n")

fread = open("compressed", "rb")
byte_reader = ByteReader(fread)

# Decompress
decompressed_planes = []
for i, (w, h, quantization_matrix) in enumerate([
    (width, height, luminance_quantization_matrix),
    (chroma_width, chroma_height, chroma_quantization_matrix),
    (chroma_width, chroma_height, chroma_quantization_matrix)
]):
    num_blocks = (w*h)//(block_size**2)
    transmitted_dc_coeffs = np.zeros(num_blocks, dtype=np.int32)
    unflattened_blocks = make_unflattened_block_container(w, h, block_size)
    for j, block in enumerate(read_entropy_encoded_blocks(byte_reader, num_blocks)):
        do_print = i == 0 and j == 0
        if do_print:
            print(f"(Decompression) transmitted block:\n{block}")
        if config.intra_block_delta_dct_coeff_enabled:
            if do_print:
                print(f"(Decompression) intra-delta'd AC coefficients block:\n{block}")
            block = undo_delta_within_block(block)
        if do_print:
            print(f"(Decompression) truncated block:\n{block}")
        block = undo_truncate(block, config.truncate_to, block_size)
        if do_print:
            print(f"(Decompression) zigzagged block:\n{block}")
        block = undo_zigzag(block)
        if config.inter_block_delta_dct_dc_coeff_enabled and j > 0:
            if do_print:
                print(f"(Decompression) inter-delta'd DC coefficient block:\n{block}")
            nearby_block_ind = get_nearby_block_ind(j, w//block_size)
            # TODO: try multiple nearby blocks (above and to the right)
            block[0][0] = undo_delta_between_blocks(transmitted_dc_coeffs[nearby_block_ind], block[0][0])
            transmitted_dc_coeffs[j] = block[0][0]
        if do_print:
            print(f"(Decompression) converted to unsigned int block:\n{block}")
        block = undo_map_to_unsigned_int(block)
        if config.quantization_enabled:
            if do_print:
                print(f"(Decompression) quantized block:\n{block}")
            block = undo_quantize(block, quantization_matrix)
        if do_print:
            print(f"(Decompression) DCTed block:\n{block}")
        block = undo_dct(block, dct_matrix)
        block = clip_and_convert_to_dtype(block, np.uint8)
        if do_print:
            print(f"(Decompression) original block:\n{block}")
        undo_flatten_block(unflattened_blocks, block, j)
    decompressed_planes.append(undo_blockify(unflattened_blocks))

fread.close()

with open("decompressed", "wb") as f:
    for plane in decompressed_planes:
        plane.tofile(f)

