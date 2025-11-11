import sys
from typing import BinaryIO, Callable, Iterator

import numpy as np
import pydantic

def do_dct(block: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    return dct_matrix @ block @ dct_matrix.T

def undo_dct(block: np.ndarray, dct_matrix: np.ndarray) -> np.ndarray:
    return dct_matrix.T @ block @ dct_matrix

def blockify_frame_plane(frame_plane: np.ndarray, block_size: int) -> list[list[np.ndarray]]:
    # Pad the width to be divisible by the block size by repeating the last column
    padded_plane = np.pad(frame_plane, ((0, 0), (0, block_size - (frame_plane.shape[1] % block_size) if frame_plane.shape[1] % block_size != 0 else 0)), mode='edge')
    # Pad the height to be divisible by the block size by repeating the last row
    padded_plane = np.pad(padded_plane, ((0, block_size - (frame_plane.shape[0] % block_size) if frame_plane.shape[0] % block_size != 0 else 0), (0, 0)), mode='edge')
    blocks_wide = padded_plane.shape[1] // block_size
    blocks_high = padded_plane.shape[0] // block_size
    blocks = padded_plane.reshape(blocks_high, block_size, blocks_wide, block_size).transpose(0, 2, 1, 3)
    return blocks

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
    signed_block[block % 2 == 1] = block[block % 2 == 1].astype(np.int32) // -2
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

def get_nearby_block_pos(block_pos: tuple[int, int], blocks_shape: tuple[int, int]) -> int:
    block_row, block_col = block_pos
    if block_col <= 0:
        return (block_row - 1, 0)
    return (block_row, block_col - 1)

def do_delta_between_blocks(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dtype = np.int8
    assert first.dtype == second.dtype == dtype
    dtype_range = np.iinfo(dtype).max - np.iinfo(dtype).min + 1
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max
    half_dtype_range = dtype_range // 2
    if second == first:
        # Case 1: second is equal to first
        delta = 0
    elif second < first:
        if first - second >= half_dtype_range:
            # Case 2: second is much smaller than first
            # Then use a positive delta that will overflow first to go back to second
            delta = (second - dtype_min) + (dtype_max + 1 - first)
        else:
            # Case 3: second is a little smaller than first
            # Then use a negative delta that will go back to second
            delta = second - first
    else:
        if second - first >= half_dtype_range:
            # Case 4: second is much larger than first
            # Then use a negative delta that will underflow first to go forward to second
            delta = (dtype_min - first) - (dtype_max + 1 - second)
        else:
            # Case 5: second is a little larger than first
            # Then use a positive delta that will go forward to second
            delta = second - first
    return delta

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

class OutputBitStream:

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

class InputBitStream:

    def __init__(self, f: BinaryIO):
        self.f = f
        self.byte = 0
        self.bit_index = 0
        self.total_bytes_read = 0
        self.cache_next_byte_()
        self.exhausted = False

    def read_bits(self, num_bits: int) -> int:
        if self.exhausted:
            raise ValueError("InputBitStream is exhausted")
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

def expand_prefix_codes_and_offsets(prefix_codes_and_offsets: list[tuple[PrefixCode, Offset]]) -> list[tuple[PrefixCode, Offset]]:
    out = []
    for prefix_code, offset in prefix_codes_and_offsets:
        for i in range(offset.start, offset.start + 2**offset.num_bits):
            out.append((prefix_code, offset))
    return out

def do_clip_to_range_of_values(block: np.ndarray, maximum_offset: Offset) -> np.ndarray:
    maximum = maximum_offset.start + 2**maximum_offset.num_bits - 1
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

def decode_prefix_code(input_bit_stream: InputBitStream, prefix_tree_root: PrefixCodeNode) -> PrefixCode:
    curr = prefix_tree_root
    while curr and not curr.prefix_code:
        bit = input_bit_stream.read_bits(1)
        if bit & 1:
            curr = curr.one
        else:
            curr = curr.zero
    assert curr and curr.prefix_code is not None
    return curr.prefix_code

def write_entropy_encoded_flattened_block(
    block: np.ndarray,
    block_size: int,
    prefix_code_and_offset_by_value: list[tuple[PrefixCode, Offset]],
    prefix_code_and_offset_by_length_of_run_of_zeros: list[tuple[PrefixCode, Offset]],
    output_bit_stream: OutputBitStream,
):
    assert block.dtype == np.uint8
    assert block.shape == (block_size ** 2,)

    num_coeffs_written = 0
    while num_coeffs_written < block.size:
        value = block[num_coeffs_written]
        prefix_code, offset = prefix_code_and_offset_by_value[value]
        output_bit_stream.write_integral(prefix_code.prefix_code, prefix_code.num_bits)
        if offset.num_bits > 0:
            output_bit_stream.write_integral(value - offset.start, offset.num_bits)
        num_coeffs_written += 1
        if num_coeffs_written == block.size:
            break
        run_of_zeros_length = 0
        while num_coeffs_written < block.size and block[num_coeffs_written] == 0:
            run_of_zeros_length += 1
            num_coeffs_written += 1
        prefix_code, offset = prefix_code_and_offset_by_length_of_run_of_zeros[run_of_zeros_length]
        output_bit_stream.write_integral(prefix_code.prefix_code, prefix_code.num_bits)
        if offset.num_bits > 0:
            output_bit_stream.write_integral(run_of_zeros_length - offset.start, offset.num_bits)

def read_entropy_encoded_flattened_blocks(
    input_bit_stream: InputBitStream,
    num_blocks: int,
    block_size: int,
    prefix_tree_root_for_values: PrefixCodeNode,
    prefix_tree_root_for_lengths_of_runs_of_zeros: PrefixCodeNode,
    value_offset_by_prefix_code: dict[PrefixCode, Offset],
    length_of_run_of_zeros_offset_by_prefix_code: dict[PrefixCode, Offset],
) -> Iterator[np.ndarray]:
    block_ind = 0
    while block_ind < num_blocks:
        block = np.zeros((block_size ** 2), dtype=np.uint8)
        num_bytes_in_block = 0
        while num_bytes_in_block < block_size ** 2:
            # Read non-zero value
            value_prefix_code = decode_prefix_code(input_bit_stream, prefix_tree_root_for_values)
            value_offset = value_offset_by_prefix_code[value_prefix_code]
            if value_offset.num_bits > 0:
                value = input_bit_stream.read_bits(value_offset.num_bits) + value_offset.start
            else:
                value = value_offset.start
            block[num_bytes_in_block] = value
            num_bytes_in_block += 1
            if num_bytes_in_block >= block_size ** 2:
                break
            # Read run of zeros
            run_of_zeros_prefix_code = decode_prefix_code(input_bit_stream, prefix_tree_root_for_lengths_of_runs_of_zeros)
            run_of_zeros_offset = length_of_run_of_zeros_offset_by_prefix_code[run_of_zeros_prefix_code]
            if run_of_zeros_offset.num_bits > 0:
                run_of_zeros_length = input_bit_stream.read_bits(run_of_zeros_offset.num_bits) + run_of_zeros_offset.start
            else:
                run_of_zeros_length = run_of_zeros_offset.start
            # The numpy array is already zero-initialize
            num_bytes_in_block += run_of_zeros_length
        yield block
        block_ind += 1

def read_frame_plane_from_file(
    input_buffer: BinaryIO,
    height: int,
    width: int,
) -> np.ndarray | None:
    length = height * width
    b = input_buffer.read(length)
    if len(b) < length:
        return None
    return np.frombuffer(b, dtype=np.uint8).reshape((height, width))

def read_frame_planes_from_file(
    input_buffer: BinaryIO,
    height: int,
    width: int,
    chrominance_subsampling_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    Y = read_frame_plane_from_file(input_buffer, height, width)
    Cb = read_frame_plane_from_file(input_buffer, height // chrominance_subsampling_factor, width // chrominance_subsampling_factor)
    Cr = read_frame_plane_from_file(input_buffer, height // chrominance_subsampling_factor, width // chrominance_subsampling_factor)
    planes = (Y, Cb, Cr)
    if any(plane is None for plane in planes):
        return None
    return planes

def compress_blockified_frame_plane(
    blockified_frame_plane: np.ndarray, # (blocks_tall, blocks_wide, block_size, block_size)
    dct_matrix: np.ndarray,
    quantization_matrix: np.ndarray,
    maximum_value_offset: Offset,
) -> np.ndarray: # (blocks_tall, blocks_wide, block_size, block_size)
    compressed_blocks = np.zeros((
        blockified_frame_plane.shape[0],
        blockified_frame_plane.shape[1],
        blockified_frame_plane.shape[2] * blockified_frame_plane.shape[3],
    ), dtype=np.uint8)
    dc_coefficient_by_block = np.zeros(blockified_frame_plane.shape[:-2], dtype=np.int8)
    for block_row in range(blockified_frame_plane.shape[0]):
        for block_col in range(blockified_frame_plane.shape[1]):
            block = blockified_frame_plane[block_row, block_col]
            block = do_dct(block, dct_matrix)
            block = do_quantize(block, quantization_matrix)
            block = clip_and_convert_to_dtype(block, np.int8)
            if not (block_row == 0 and block_col == 0):
                dc_coefficient_by_block[block_row, block_col] = block[0, 0]
                nearby_block_row, nearby_block_col = get_nearby_block_pos((block_row, block_col), blockified_frame_plane.shape[:-2])
                block[0, 0] = do_delta_between_blocks(dc_coefficient_by_block[nearby_block_row, nearby_block_col], dc_coefficient_by_block[block_row, block_col])
            block = do_map_to_unsigned_int(block)
            block = do_clip_to_range_of_values(block, maximum_value_offset)
            block = do_zigzag(block)
            compressed_blocks[block_row, block_col] = block
    return compressed_blocks

def compress_blockified_frame_planes(
    blockified_frame_planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    dct_matrix: np.ndarray,
    luminance_quantization_matrix: np.ndarray,
    chrominance_quantization_matrix: np.ndarray,
    maximum_value_offset: Offset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        compress_blockified_frame_plane(
            blockified_frame_planes[0],
            dct_matrix,
            luminance_quantization_matrix,
            maximum_value_offset,
        ),
        compress_blockified_frame_plane(
            blockified_frame_planes[1],
            dct_matrix,
            chrominance_quantization_matrix,
            maximum_value_offset,
        ),
        compress_blockified_frame_plane(
            blockified_frame_planes[2],
            dct_matrix,
            chrominance_quantization_matrix,
            maximum_value_offset,
        ),
    )

def write_blockified_frame_planes(
    blockified_frame_planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    block_writer: Callable[[np.ndarray], None],
) -> None:
    for blockified_frame_plane in blockified_frame_planes:
        for block_row in range(blockified_frame_plane.shape[0]):
            for block_col in range(blockified_frame_plane.shape[1]):
                block_writer(blockified_frame_plane[block_row, block_col])

def blockify_frame_planes(
    frame_planes: tuple[np.ndarray, np.ndarray, np.ndarray],
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(blockify_frame_plane(frame_plane, block_size) for frame_plane in frame_planes)

def compress(
    input_buffer: BinaryIO,
    height: int,
    width: int,
    block_size: int,
    dct_matrix: np.ndarray,
    luminance_quantization_matrix: np.ndarray,
    chrominance_quantization_matrix: np.ndarray,
    chrominance_subsampling_factor: int,
    prefix_code_and_offset_by_value: list[tuple[PrefixCode, Offset]],
    prefix_code_and_offset_by_length_of_run_of_zeros: list[tuple[PrefixCode, Offset]],
    output_bit_stream: OutputBitStream,
) -> None:
    frame_ind = 0
    while True:
        print(f"Compressing frame {frame_ind}", file=sys.stderr)
        frame_planes = read_frame_planes_from_file(
            input_buffer,
            height,
            width,
            chrominance_subsampling_factor,
        )
        if frame_planes is None:
            print(f"No more frames to compress; number of frames compressed: {frame_ind}", file=sys.stderr)
            break
        blockified_frame_planes = blockify_frame_planes(frame_planes, block_size)
        # TODO: encode resolution
        # TODO: encode quality level
        # TODO: B-frames (0.015 bits per pixel) and P-frames (0.1 bits per pixel)
        # TODO: motion vectors
        # TODO: sync frames
        # TODO: dynamic huffman codes for each saga
        compressed_blockified_frame_planes = compress_blockified_frame_planes(
            blockified_frame_planes,
            dct_matrix,
            luminance_quantization_matrix,
            chrominance_quantization_matrix,
            prefix_code_and_offset_by_value[-1][1],
        )
        write_blockified_frame_planes(
            compressed_blockified_frame_planes,
            lambda block: write_entropy_encoded_flattened_block(
                block,
                block_size,
                prefix_code_and_offset_by_value,
                prefix_code_and_offset_by_length_of_run_of_zeros,
                output_bit_stream,
            )
        )
        frame_ind += 1

def decompress(
    input_bit_stream: InputBitStream,
    height: int,
    width: int,
    block_size: int,
    dct_matrix: np.ndarray,
    luminance_quantization_matrix: np.ndarray,
    chrominance_quantization_matrix: np.ndarray,
    chrominance_subsampling_factor: int,
    prefix_tree_root_for_values: PrefixCodeNode,
    prefix_tree_root_for_lengths_of_runs_of_zeros: PrefixCodeNode,
    value_offset_by_prefix_code: dict[PrefixCode, Offset],
    length_of_run_of_zeros_offset_by_prefix_code: dict[PrefixCode, Offset],
    output_buffer: BinaryIO,
) -> None:
    frame_ind = 0
    is_last_frame = False
    while not is_last_frame:
        print(f"Decompressing frame {frame_ind}", file=sys.stderr)
        unflattened_blocks_by_plane = []
        # TODO: only print after decompressing all planes
        for (plane_width, plane_height, quantization_matrix) in [
            (width, height, luminance_quantization_matrix),
            (width // chrominance_subsampling_factor, height // chrominance_subsampling_factor, chrominance_quantization_matrix),
            (width // chrominance_subsampling_factor, height // chrominance_subsampling_factor, chrominance_quantization_matrix),
        ]:
            num_blocks = plane_width * plane_height // (block_size ** 2)
            blocks_wide = (plane_width+block_size-1)//block_size
            blocks_high = (plane_height+block_size-1)//block_size
            dc_coefficients_by_block = np.zeros((blocks_high, blocks_wide), dtype=np.int8)
            unflattened_blocks = np.zeros((blocks_high, blocks_wide, block_size, block_size), dtype=np.uint8)
            unflattened_blocks_by_plane.append(unflattened_blocks)
            try:
                blocks = list(read_entropy_encoded_flattened_blocks(
                    input_bit_stream,
                    num_blocks,
                    block_size,
                    prefix_tree_root_for_values,
                    prefix_tree_root_for_lengths_of_runs_of_zeros,
                    value_offset_by_prefix_code,
                    length_of_run_of_zeros_offset_by_prefix_code,
                ))
            except ValueError:
                print(f"No more frames to decompress; number of frames decompressed: {frame_ind}", file=sys.stderr)
                is_last_frame = True
                break
            for block_ind, block in enumerate(blocks):
                block_row = block_ind // blocks_wide
                block_col = block_ind % blocks_wide
                block = undo_zigzag(block)
                block = undo_map_to_unsigned_int(block)
                if block_ind > 0:
                    nearby_block_row, nearby_block_col = get_nearby_block_pos((block_row, block_col), (blocks_high, blocks_wide))
                    block[0, 0] = undo_delta_between_blocks(dc_coefficients_by_block[nearby_block_row, nearby_block_col], block[0, 0])
                    dc_coefficients_by_block[block_row, block_col] = block[0, 0]
                block = undo_quantize(block, quantization_matrix)
                block = undo_dct(block, dct_matrix)
                block = clip_and_convert_to_dtype(block, np.uint8)
                unflattened_blocks[block_row, block_col] = block
            row = 0
            for block_row in range(blocks_high):
                for row_within_block in range(block_size):
                    col = 0
                    for block_col in range(blocks_wide):
                        for col_within_block in range(block_size):
                            output_buffer.write(unflattened_blocks[block_row, block_col, row_within_block, col_within_block])
                            col += 1
                            if col >= plane_width:
                                break
                    row += 1
                    if row >= plane_height:
                        break
        frame_ind += 1

class Meta(pydantic.BaseModel):
    height: int
    width: int
    block_size: int
    chrominance_subsampling_factor: int
    dct_matrix: np.ndarray
    luminance_quantization_matrix: np.ndarray
    chrominance_quantization_matrix: np.ndarray
    prefix_code_and_offset_by_value: list[tuple[PrefixCode, Offset]]
    prefix_code_and_offset_by_length_of_run_of_zeros: list[tuple[PrefixCode, Offset]]
    prefix_tree_root_for_values: PrefixCodeNode
    prefix_tree_root_for_lengths_of_runs_of_zeros: PrefixCodeNode
    value_offset_by_prefix_code: dict[PrefixCode, Offset]
    length_of_run_of_zeros_offset_by_prefix_code: dict[PrefixCode, Offset]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

def make_meta(height: int, width: int) -> Meta:
    block_size = 8
    prefix_codes_and_offsets_for_lengths_of_runs_of_zeros = [
        (PrefixCode(prefix_code=0b0, num_bits=1), Offset(start=0, num_bits=0)),
        (PrefixCode(prefix_code=0b10, num_bits=2), Offset(start=1, num_bits=1)),
        (PrefixCode(prefix_code=0b1110, num_bits=4), Offset(start=3, num_bits=2)),
        (PrefixCode(prefix_code=0b111110, num_bits=6), Offset(start=7, num_bits=3)),
        (PrefixCode(prefix_code=0b111111, num_bits=6), Offset(start=15, num_bits=4)),
        (PrefixCode(prefix_code=0b110, num_bits=3), Offset(start=31, num_bits=5)),
        (PrefixCode(prefix_code=0b11110, num_bits=5), Offset(start=63, num_bits=1)),
    ]
    prefix_codes_and_offsets_for_values = [
        (PrefixCode(prefix_code=0b1100, num_bits=4), Offset(start=0, num_bits=0)),
        (PrefixCode(prefix_code=0b0, num_bits=1), Offset(start=1, num_bits=1)),
        (PrefixCode(prefix_code=0b10, num_bits=2), Offset(start=3, num_bits=2)),
        (PrefixCode(prefix_code=0b1101, num_bits=4), Offset(start=7, num_bits=3)),
        (PrefixCode(prefix_code=0b11110, num_bits=5), Offset(start=15, num_bits=4)),
        (PrefixCode(prefix_code=0b111110, num_bits=6), Offset(start=31, num_bits=5)),
        (PrefixCode(prefix_code=0b111111, num_bits=6), Offset(start=63, num_bits=6)),
        (PrefixCode(prefix_code=0b1110, num_bits=4), Offset(start=127, num_bits=7)),
    ]

    return Meta(
        height=height,
        width=width,
        block_size=block_size,
        chrominance_subsampling_factor=2,
        dct_matrix=make_dct_matrix(block_size),
        luminance_quantization_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32) * 2,
        chrominance_quantization_matrix = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=np.float32),
        prefix_code_and_offset_by_value=expand_prefix_codes_and_offsets(prefix_codes_and_offsets_for_values),
        prefix_code_and_offset_by_length_of_run_of_zeros=expand_prefix_codes_and_offsets(prefix_codes_and_offsets_for_lengths_of_runs_of_zeros),
        prefix_tree_root_for_values=build_prefix_code_tree([p for p, _ in prefix_codes_and_offsets_for_values]),
        prefix_tree_root_for_lengths_of_runs_of_zeros=build_prefix_code_tree([p for p, _ in prefix_codes_and_offsets_for_lengths_of_runs_of_zeros]),
        value_offset_by_prefix_code={p: o for p, o in prefix_codes_and_offsets_for_values},
        length_of_run_of_zeros_offset_by_prefix_code={p: o for p, o in prefix_codes_and_offsets_for_lengths_of_runs_of_zeros},
    )

if __name__ == "__main__":
    meta = make_meta(288, 352)
    if len(sys.argv) != 2:
        print("Usage: python video_compressor.py <compress|decompress>", file=sys.stderr)
        sys.exit(1)
    if sys.argv[1] not in ["compress", "decompress"]:
        print("Usage: python video_compressor.py <compress|decompress>", file=sys.stderr)
        sys.exit(1)
    is_compressing = sys.argv[1] == "compress"
    if is_compressing:
        output_bit_stream = OutputBitStream(sys.stdout.buffer)
        compress(
            input_buffer=sys.stdin.buffer,
            height=meta.height,
            width=meta.width,
            block_size=meta.block_size,
            dct_matrix=meta.dct_matrix,
            luminance_quantization_matrix=meta.luminance_quantization_matrix,
            chrominance_quantization_matrix=meta.chrominance_quantization_matrix,
            chrominance_subsampling_factor=meta.chrominance_subsampling_factor,
            prefix_code_and_offset_by_value=meta.prefix_code_and_offset_by_value,
            prefix_code_and_offset_by_length_of_run_of_zeros=meta.prefix_code_and_offset_by_length_of_run_of_zeros,
            output_bit_stream=output_bit_stream,
        )
        output_bit_stream.flush()
    else:
        decompress(
            input_bit_stream=InputBitStream(sys.stdin.buffer),
            height=meta.height,
            width=meta.width,
            block_size=meta.block_size,
            dct_matrix=meta.dct_matrix,
            luminance_quantization_matrix=meta.luminance_quantization_matrix,
            chrominance_quantization_matrix=meta.chrominance_quantization_matrix,
            chrominance_subsampling_factor=meta.chrominance_subsampling_factor,
            prefix_tree_root_for_values=meta.prefix_tree_root_for_values,
            prefix_tree_root_for_lengths_of_runs_of_zeros=meta.prefix_tree_root_for_lengths_of_runs_of_zeros,
            value_offset_by_prefix_code=meta.value_offset_by_prefix_code,
            length_of_run_of_zeros_offset_by_prefix_code=meta.length_of_run_of_zeros_offset_by_prefix_code,
            output_buffer=sys.stdout.buffer,
        )
