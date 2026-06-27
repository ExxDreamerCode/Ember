#!/usr/bin/env python3
"""Compact an Ember NNUE by omitting all-zero feature rows.

The compact format is intended for the embedded network. It keeps the dense
NNUE header and dense tail unchanged, stores a virtual-row to physical-row
u16 table, then stores only nonzero feature-transformer rows.
"""

import struct
import sys
from pathlib import Path

NNUE_MAGIC = 0x4E4E5545
COMPACT_MAGIC = b"ECN1"
UINT16_MAX = 0xFFFF
PSQ_INPUTS_PER_BUCKET = 768


def get_header(data):
    if len(data) < 8:
        raise ValueError("NNUE header too small")
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != NNUE_MAGIC:
        raise ValueError("bad NNUE magic")

    if version == 5:
        header_size = 8
        rows = 16 * PSQ_INPUTS_PER_BUCKET
        body = len(data) - header_size
        numerator = body - 32
        denominator = 2 * (12288 + 1 + 16)
        if numerator % denominator != 0:
            raise ValueError("cannot infer v5 hidden size")
        hidden_size = numerator // denominator
    elif version == 6:
        if len(data) < 9:
            raise ValueError("NNUE v6 header too small")
        header_size = 9
        rows = 16 * PSQ_INPUTS_PER_BUCKET
        pairwise = data[8] & 2 != 0
        output_multiplier = 8 if pairwise else 16
        body = len(data) - header_size
        numerator = body - 32
        denominator = 2 * (12288 + 1 + output_multiplier)
        if numerator % denominator != 0:
            raise ValueError("cannot infer v6 hidden size")
        hidden_size = numerator // denominator
    elif version >= 7:
        if len(data) < 15:
            raise ValueError("NNUE v7+ header too small")
        flags = data[8]
        hidden_size = struct.unpack_from("<H", data, 9)[0]
        header_size = 15
        king_buckets = 16
        if flags & 0x80:
            if len(data) < 17:
                raise ValueError("NNUE extended header too small")
            king_buckets = data[15]
            header_size += 2
        if version >= 10:
            header_size += 1
        rows = king_buckets * PSQ_INPUTS_PER_BUCKET
    else:
        raise ValueError(f"unsupported NNUE version {version}")

    if rows > UINT16_MAX:
        raise ValueError(f"virtual row count {rows} does not fit in u16")
    if hidden_size == 0:
        raise ValueError("hidden size must be nonzero")

    feature_bytes = rows * hidden_size * 2
    feature_end = header_size + feature_bytes
    if feature_end > len(data):
        raise ValueError("NNUE feature matrix extends past file end")

    return header_size, rows, hidden_size


def compact(in_path, out_path):
    data = Path(in_path).read_bytes()
    header_size, virtual_rows, hidden_size = get_header(data)
    row_bytes = hidden_size * 2
    feature_start = header_size
    feature_end = feature_start + virtual_rows * row_bytes
    zero_row = b"\0" * row_bytes

    row_map = []
    rows = []
    for row in range(virtual_rows):
        start = feature_start + row * row_bytes
        payload = data[start : start + row_bytes]
        if payload == zero_row:
            row_map.append(UINT16_MAX)
        else:
            physical = len(rows)
            if physical >= UINT16_MAX:
                raise ValueError(f"physical row count {physical + 1} does not fit in u16")
            row_map.append(physical)
            rows.append(payload)

    physical_rows = len(rows)
    if physical_rows >= UINT16_MAX:
        raise ValueError(f"physical row count {physical_rows} does not fit in u16")

    header = bytearray()
    header += COMPACT_MAGIC
    header += struct.pack("<I", 1)
    header += struct.pack("<Q", len(data))
    header += struct.pack("<IIII", header_size, virtual_rows, physical_rows, hidden_size)

    out = bytearray(header)
    out += data[:header_size]
    out += struct.pack(f"<{virtual_rows}H", *row_map)
    out += b"".join(rows)
    out += data[feature_end:]
    Path(out_path).write_bytes(out)

    dense_feature_bytes = virtual_rows * row_bytes
    compact_feature_bytes = physical_rows * row_bytes + virtual_rows * 2
    print(f"Input: {in_path} ({len(data)} bytes)")
    print(f"Output: {out_path} ({len(out)} bytes)")
    print(f"Rows: {physical_rows}/{virtual_rows} nonzero")
    print(f"Feature bytes: {dense_feature_bytes} -> {compact_feature_bytes}")
    print(f"Saved: {len(data) - len(out)} bytes")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.nnue> <output.compact.nnue>")
        return 2
    compact(sys.argv[1], sys.argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
