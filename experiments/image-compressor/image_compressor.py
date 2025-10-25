import sys

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
    quantization_dtype: np.dtype = np.int32
    quantization_enabled: bool = True
    delta_enabled: bool = False
    truncate_to: int = block_size**2 // 2
    truncate_enabled: bool = True


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

def do_quantize(block: np.ndarray, quantization_matrix: np.ndarray | int, dtype: np.dtype = np.int32) -> np.ndarray:
    return np.round(block / quantization_matrix).astype(dtype)

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
    return np.concatenate([sequence, np.zeros(block_size**2 - truncate_to)])

def do_delta(sequence: np.ndarray) -> np.ndarray:
    # Use the previous value to predict the next value
    return np.concatenate([[sequence[0]], sequence[1:] - sequence[:-1]])

def undo_delta(delta: np.ndarray) -> np.ndarray:
    return np.cumsum(delta)

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
    (Cb, chroma_quantization_matrix),
    (Cr, chroma_quantization_matrix)
]):
    blocks_by_plane_ind.append([])
    for block in do_flatten_blocks(do_blockify(plane, block_size)):
        block = do_dct(block, dct_matrix)
        block = do_quantize(block, quantization_matrix if config.quantization_enabled else 1, config.quantization_dtype)
        block = do_zigzag(block)
        block = do_truncate(block, config.truncate_to)
        if config.delta_enabled:
            block = do_delta(block)
        blocks_by_plane_ind[i].append(block)


assert len(blocks_by_plane_ind) == num_planes

# Decompress
decompressed_by_plane_ind = []
for i, (w, h, quantization_matrix) in enumerate([
    (width, height, luminance_quantization_matrix),
    (chroma_width, chroma_height, chroma_quantization_matrix),
    (chroma_width, chroma_height, chroma_quantization_matrix)
]):
    unflattened_blocks = make_unflattened_block_container(w, h, block_size)
    for j, block in enumerate(blocks_by_plane_ind[i]):
        if config.delta_enabled:
            block = undo_delta(block)
        block = undo_truncate(block, config.truncate_to, block_size)
        block = undo_zigzag(block)
        block = undo_quantize(block, quantization_matrix if config.quantization_enabled else 1)
        block = undo_dct(block, dct_matrix)
        undo_flatten_block(unflattened_blocks, block, j)
    plane = undo_blockify(unflattened_blocks)
    print(plane)
    print("\n\n")
    print(Y)
    print("\n\n")
    exit()
