## 2. Definitions

- block: a frame encapsulates one or more blocks; each block has arbitrary content, a header, and guaranteed maximum content size; blocks depend on previous blocks in the same frame for decoding, but not decompression

## 3. Compression Algorithm

- Does not support random access to compressed data
- Supports arbitrary content lengths using bounded intermediate storage

### 3.1. Frames

- Compressed content is made up of one or more frames
- Each frame is independent and can be decompressed independently of other frames
- Frame has associated parameters for decoding
- Original content corresponds to concatenation of decompressed frames
- Two types of frames: Zstandard frames and skippable frames

#### 3.1.1. Zstandard Frames

- Contain compressed data

```
                    +--------------------+------------+
                    | Magic_Number       | 4 bytes    | <-- 0xFD2FB528
                    +--------------------+------------+
                    | Frame_Header       | 2-14 bytes | <-- see Section 3.1.1.1.
                    +--------------------+------------+
                    | Data_Block         | n bytes    | <-- see Section 3.1.1.2.
                    +--------------------+------------+
                    | [More Data_Blocks] |            | <-- exclude from MVP
                    +--------------------+------------+
                    | [Content_Checksum] | 4 bytes    | <-- exclude from MVP
                    +--------------------+------------+
```

##### 3.1.1.1. Frame_Header

```
                  +-------------------------+-----------+
                  | Frame_Header_Descriptor | 1 byte    |
                  +-------------------------+-----------+
                  | [Window_Descriptor]     | 0-1 byte  | <-- skipped in MVP
                  +-------------------------+-----------+
                  | [Dictionary_ID]         | 0-4 bytes | <-- skipped in MVP
                  +-------------------------+-----------+
                  | [Frame_Content_Size]    | 0-8 bytes | <-- see Section 3.1.1.1.4
                  +-------------------------+-----------+
```

###### 3.1.1.1.1. Frame_Header_Descriptor

```
                 +============+=========================+
                 | Bit Number | Field Name              |
                 +============+=========================+
                 | 7-6        | Frame_Content_Size_Flag | <-- see Section 3.1.1.1.1.1.
                 +------------+-------------------------+
                 | 5          | Single_Segment_Flag     | <-- 0b1 for MVP
                 +------------+-------------------------+
                 | 4          | (unused)                | <-- 0b0
                 +------------+-------------------------+
                 | 3          | (reserved)              | <-- 0b0
                 +------------+-------------------------+
                 | 2          | Content_Checksum_Flag   | <-- 0b0 for MVP
                 +------------+-------------------------+
                 | 1-0        | Dictionary_ID_Flag      | <-- 0b00 for MVP
                 +------------+-------------------------+
```

####### 3.1.1.1.1.1. Frame_Content_Size_Flag

```
             +-------------------------+--------+---+---+---+
             | Frame_Content_Size_Flag |   0    | 1 | 2 | 3 |
             +-------------------------+--------+---+---+---+
             | FCS_Field_Size          | 0 or 1 | 2 | 4 | 8 |
             +-------------------------+--------+---+---+---+
```

###### 3.1.1.1.4. Frame_Content_Size

- Original (uncompressed) size

```
                    +================+================+
                    | FCS_Field_Size | Range          |
                    +================+================+
                    |       0        | unknown        |
                    +----------------+----------------+
                    |       1        | 0 - 255        |
                    +----------------+----------------+
                    |       2        | 256 - 65791    |
                    +----------------+----------------+
                    |       4        | 0 - 2^(32) - 1 |
                    +----------------+----------------+
                    |       8        | 0 - 2^(64) - 1 |
                    +----------------+----------------+
```

- Little-endian
- If `FCS_Field_Size` is 2, an offset of 256 is added

##### 3.1.1.2. Data_Block

```
                 +============+============+============+===============+
                 | Last_Block | Block_Type | Block_Size | Block_Content |
                 +============+============+============+===============+
                 |   bit 0    |  bits 1-2  | bits 3-23  |    n bytes    |
                 +------------+------------+------------+---------------+
```

- `Block_Size`: size of `Block_Content`, which is the compressed content

###### 3.1.1.2.1. Last_Block

```
                       +=======+==================+
                       | Value | Block_Type       |
                       +=======+==================+
                       |   0   |    Raw_Block     |
                       +-------+------------------+
                       |   1   |    RLE_Block     |
                       +-------+------------------+
                       |   2   | Compressed_Block |
                       +-------+------------------+
                       |   3   |     Reserved     |
                       +-------+------------------+
```

- `Raw_Block`: `Block_Content` contains `Block_Size` bytes
- `RLE_Block`: `Block_Content` contains a single byte, repeated `Block_Size` times
- `Compressed_Block`: see section 3.1.1.3.; `Block_Size` is the length of `Block_Content`, which is the compressed content

####### 3.1.1.2.4. Block_Content and Block_Maximum_Size

- `Block_Content` is upper-bounded by `Block_Maximum_Size`, which is the smaller of 128 KB and `Window_Size`

#### 3.1.2. Skippable Frames

- Contain custom user metadata


