import math
import sys
import itertools
import os
import random
from dataclasses import dataclass
from typing import BinaryIO

R = 4
r = R + 1
L = 1 << R
A = [0, 1, 2]

def Ls_from_ss(ss: list[int]):
    # Ls[s] represents the number of appearances of symbol s in ss.
    Ls = [0] * len(A)
    for s in ss:
        Ls[s] += 1
    return Ls

def starts_from_Ls(Ls: list[int]) -> list[int]:
    # start[s] represents the starting index of symbol s in the encoding table.
    starts = [-l for l in Ls]
    cum_Ls = itertools.accumulate(Ls)
    for i, cum_start in enumerate(cum_Ls):
        if i == len(A) - 1:
            break
        starts[i+1] += cum_start
    return starts

def encoding_table_from_Ls_and_symbols_and_starts(Ls: list[int], symbols: list[int], starts: list[int]) -> list[int]:
    nexts = Ls.copy()
    encoding_table = [0] * L
    for x in range(L, 2*L):
        s = symbols[x - L]
        encoding_table[starts[s] + nexts[s]] = x
        nexts[s] += 1
    return encoding_table

def spread_symbols_from_Ls(Ls: list[int]) -> list[int]:
    X = 0
    symbols = [0] * L
    step = ((5 / 8) * L) + 3
    for s in range(len(A)):
        for _ in range(Ls[s]):
            symbols[int(X)] = s
            X = (X + step) % L
    return symbols

def nbs_from_Ls(Ls: list[int]) -> list[int]:
    ks = [R - math.floor(math.log2(Ls[s])) for s in A]
    nbs = [
        (ks[s] << r) - (Ls[s] << ks[s])
        for s in A
    ]
    return nbs

@dataclass
class DecodingTableEntry:
    symbol: int
    nbBits: int
    newX: int

def decoding_table_from_Ls_and_symbols(Ls: list[int], symbols: list[int]) -> list[DecodingTableEntry]:
    decoding_table = []
    nexts = Ls.copy()
    for X in range(L):
        symbol = symbols[X]
        x = nexts[symbol]
        nexts[symbol] += 1
        nbBits = R - math.floor(math.log2(x))
        newX = (x << nbBits) - L
        t = DecodingTableEntry(
            symbol=symbol,
            nbBits = nbBits,
            newX = newX
        )
        decoding_table.append(t)
    return decoding_table

class ByteWriter:

    def __init__(self, f: BinaryIO):
        self.f = f
        self.byte = 0
        self.bit_index = 0
        self.total_bytes_written = 0

    def write_lower_bits(self, value: int, num_bits: int):
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

class ReverseByteReader:

    def __init__(self, f: BinaryIO):
        self.f = f
        self.byte = 0
        self.bit_index = 0
        self.total_bytes_read = 0
        self.f.seek(0, os.SEEK_END)
        self.position = self.f.tell()
        if self.position > 0:
            self.position -= 1
            self.cache_next_byte_()

    @property
    def exhausted(self) -> bool:
        return self.position < 0 and self.bit_index >= 8

    def read_bits(self, num_bits: int) -> int:
        out = 0
        for i in range(num_bits):
            if self.bit_index == 8:
                self.cache_next_byte_()
            out |= ((self.byte >> self.bit_index) & 1) << i
            self.bit_index += 1
        return out

    def cache_next_byte_(self) -> None:
        self.f.seek(self.position)
        byte = self.f.read(1)
        # TODO: enforce byteorder for portability
        self.byte = int.from_bytes(byte, byteorder=sys.byteorder)
        self.bit_index = 0
        self.total_bytes_read += 1
        self.position -= 1

    def read_byte(self) -> int:
        return self.read_bits(8)

def encode(x: int, s: int, starts: list[int], encoding_table: list[int], nbs: list[int], byte_writer: ByteWriter) -> int:
    nbBits = (x + nbs[s]) >> r
    byte_writer.write_lower_bits(x, nbBits)
    x = encoding_table[starts[s] + (x >> nbBits)]
    return x

def decode(x: int, decoding_table: list[DecodingTableEntry], byte_reader: ReverseByteReader) -> tuple[int, int]:
    X = x - L
    t = decoding_table[X]
    X = t.newX + byte_reader.read_bits(t.nbBits)
    return X + L, t.symbol

def main():
    ss = [0] * 3 + [1] * 8 + [2] * 5
    Ls = Ls_from_ss(ss)
    symbols = spread_symbols_from_Ls(Ls)
    starts = starts_from_Ls(Ls)
    encoding_table = encoding_table_from_Ls_and_symbols_and_starts(Ls, symbols, starts)
    nbs = nbs_from_Ls(Ls)
    decoding_table = decoding_table_from_Ls_and_symbols(Ls, symbols)

    print(f"Ls: {Ls}")
    print(f"symbols: {symbols}")
    print(f"encoding table: {encoding_table}")
    print(f"nbs: {nbs}")
    print(f"decoding table: {decoding_table}")

    xinit = L
    x = xinit

    to_encode = random.choices(
        A,
        weights=Ls,
        k=10000
    )
    decoded = []

    with open("buffer", "wb") as f:
        byte_writer = ByteWriter(f)
        for s in to_encode:
            x = encode(x, s, starts, encoding_table, nbs, byte_writer)
        padded_bits = (8 - byte_writer.bit_index) if byte_writer.bit_index > 0 else 0
        byte_writer.flush()

    with open("buffer", "rb") as f:
        byte_reader = ReverseByteReader(f)
        byte_reader.read_bits(padded_bits)
        while not byte_reader.exhausted:
            x, s = decode(x, decoding_table, byte_reader)
            decoded.append(s)

    assert x == xinit
    assert list(reversed(decoded)) == to_encode

if __name__ == "__main__":
    main()

