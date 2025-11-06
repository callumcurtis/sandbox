import sys

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


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
    max_bits_per_dct_coeff_in_transmission: int = 9
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
], dtype=np.float32)
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

blocks_by_plane_ind = []

# Compress
for i, (plane, quantization_matrix) in enumerate([
    (Y, luminance_quantization_matrix),
    (Cr, chroma_quantization_matrix),
    (Cb, chroma_quantization_matrix),
]):
    blocks_by_plane_ind.append([])
    transmitted_dc_coeffs = np.zeros((plane.shape[0]*plane.shape[1])//(block_size**2), dtype=np.int32)
    for j, block in enumerate(do_flatten_blocks(do_blockify(plane, block_size))):
        do_print = i == 0 and j == 41
        if do_print:
            print("Original block:")
            print(block)
        block = do_dct(block, dct_matrix)
        if do_print:
            print("DCTed block:")
            print(block)
        if config.quantization_enabled:
            block = do_quantize(block, quantization_matrix)
            if do_print:
                print("Quantized block:")
                print(block)
        block = clip_and_convert_to_dtype(block, np.int8)
        if do_print:
            print("Clipped and converted to int8 block:")
            print(block)
        block = do_map_to_unsigned_int(block)
        if do_print:
            print("Mapped to unsigned int8 block:")
            print(block)
        if config.inter_block_delta_dct_dc_coeff_enabled and j > 0:
            transmitted_dc_coeffs[j] = block[0][0]
            nearby_block_ind = get_nearby_block_ind(j, plane.shape[1]//block_size)
            # TODO: try multiple nearby blocks (above and to the right)
            # TODO: handle the case where the delta overflows the transmission dtype
            block[0][0] = do_delta_between_blocks(transmitted_dc_coeffs[nearby_block_ind], transmitted_dc_coeffs[j])
            if do_print:
                print("Delta-ed between blocks:")
                print(block)
        block = do_zigzag(block)
        if do_print:
            print("Zigzagged block:")
            print(block)
        block = do_truncate(block, config.truncate_to)
        if do_print:
            print("Truncated block:")
            print(block)
        if config.intra_block_delta_dct_coeff_enabled:
            block = do_delta_within_block(block)
            if do_print:
                print("Delta-ed within block:")
                print(block)
        blocks_by_plane_ind[i].append(block)

print("\n\n")

# Decompress
decompressed_planes = []
for i, (w, h, quantization_matrix) in enumerate([
    (width, height, luminance_quantization_matrix),
    (chroma_width, chroma_height, chroma_quantization_matrix),
    (chroma_width, chroma_height, chroma_quantization_matrix)
]):
    transmitted_dc_coeffs = np.zeros((w*h)//(block_size**2), dtype=np.int32)
    unflattened_blocks = make_unflattened_block_container(w, h, block_size)
    for j, block in enumerate(blocks_by_plane_ind[i]):
        do_print = i == 0 and j == 41
        if do_print:
            print("Transmitted block:")
            print(block)
        if config.intra_block_delta_dct_coeff_enabled:
            block = undo_delta_within_block(block)
            if do_print:
                print("Un-delta-ed within block:")
                print(block)
        block = undo_truncate(block, config.truncate_to, block_size)
        if do_print:
            print("Un-truncated block:")
            print(block)
        block = undo_zigzag(block)
        if do_print:
            print("Un-zigzagged block:")
            print(block)
        if config.inter_block_delta_dct_dc_coeff_enabled and j > 0:
            nearby_block_ind = get_nearby_block_ind(j, w//block_size)
            # TODO: try multiple nearby blocks (above and to the right)
            block[0][0] = undo_delta_between_blocks(transmitted_dc_coeffs[nearby_block_ind], block[0][0])
            transmitted_dc_coeffs[j] = block[0][0]
            if do_print:
                print("Un-delta-ed between blocks:")
                print(block)
        block = undo_map_to_unsigned_int(block)
        if do_print:
            print("Un-mapped to unsigned int8 block:")
            print(block)
        if config.quantization_enabled:
            block = undo_quantize(block, quantization_matrix)
            if do_print:
                print("Un-quantized block:")
                print(block)
        block = undo_dct(block, dct_matrix)
        if do_print:
            print("Un-DCTed block:")
            print(block)
        undo_flatten_block(unflattened_blocks, block, j)
    decompressed_planes.append(undo_blockify(unflattened_blocks))


## Display


def YCrCb420_to_YCrCb440(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Cb_upsampled = ndimage.zoom(Cb, 2, order=1)
    Cr_upsampled = ndimage.zoom(Cr, 2, order=1)
    h, w = Y.shape
    Cb_upsampled = Cb_upsampled[:h, :w]
    Cr_upsampled = Cr_upsampled[:h, :w]
    return Y, Cr_upsampled, Cb_upsampled

def YCrCb440_to_RGB(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    R = R.astype(np.uint8)
    G = G.astype(np.uint8)
    B = B.astype(np.uint8)
    return R, G, B


decompressed_RGB = np.stack(YCrCb440_to_RGB(*YCrCb420_to_YCrCb440(*decompressed_planes)), axis=-1)


plt.figure()
plt.imshow(decompressed_RGB)
plt.title("Decompressed RGB")
plt.axis("off")
plt.show()
