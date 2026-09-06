#!/usr/bin/env python3
"""
Convert bullet quantised.bin -> .nnue for Ember engine (v7).

Bullet quantised format (raw, NO padding between layers):
  l0w: (num_king_buckets * 768) x hidden_size  i16  -> [12288 x 1024]
  l0b: hidden_size i16                             -> [1024]
  l1w: (2*hidden_size) x num_output_buckets i16    -> [2048 x 8]  (row-major!)
  l1b: num_output_buckets i32                      -> [8]

Ember expects output_weights: [num_out x (2*hs)] i16  = [8 x 2048]  (row-major)
  - Bucket 0: weights[0..2047]
  - Bucket 1: weights[2048..4095]
  - etc.

But raw Bullet stores l1w as [2*hs x num_out] = [2048 x 8] row-major:
  - Neuron 0, bucket 0..7
  - Neuron 1, bucket 0..7
  - etc.

TRANSPOSITION REQUIRED: [2048][8] -> [8][2048]

If input is already .nnue (v6/v7), assume l1w is already in Ember format [8][2048].

Usage:
  python convert_bullet_to_nnue.py quantised.bin net.nnue     (bullet raw)
  python convert_bullet_to_nnue.py v6.nnue net.nnue      (re-header)
"""

import struct
import sys
import os

QA = 255
QB = 64
NNUE_MAGIC = 0x4E4E5545

NUM_KB = 16
PSQ = NUM_KB * 768   # = 12288
HS = 1024
NUM_OUT = 8


def read_payload(data):
    """Parse raw weights (with or without padding). Returns (l0w, l0b, l1w, l1b, need_transpose)."""
    total = len(data)

    l0w_bytes = PSQ * HS * 2
    l0b_bytes = HS * 2
    l1w_bytes = 2 * HS * NUM_OUT * 2
    l1b_bytes = NUM_OUT * 4
    expected = l0w_bytes + l0b_bytes + l1w_bytes + l1b_bytes

    if total < expected:
        print(f"[ERROR] Payload too small: {total} bytes, expected >= {expected}")
        sys.exit(1)

    off = 0
    l0w = list(struct.unpack_from(f'<{PSQ * HS}h', data, off))
    off += l0w_bytes
    print(f"  l0w: {len(l0w)} i16 [{PSQ}x{HS}], offset={off}")

    l0b = list(struct.unpack_from(f'<{HS}h', data, off))
    off += l0b_bytes
    print(f"  l0b: {len(l0b)} i16 [{HS}], offset={off}")

    l1w = list(struct.unpack_from(f'<{2 * HS * NUM_OUT}h', data, off))
    off += l1w_bytes
    print(f"  l1w: {len(l1w)} i16 [{2*HS}x{NUM_OUT}], offset={off}")

    l1b = list(struct.unpack_from(f'<{NUM_OUT}i', data, off))
    off += l1b_bytes
    print(f"  l1b: {len(l1b)} i32 [{NUM_OUT}], offset={off}")

    print(f"  Consumed: {off}/{total} bytes (trailing={total - off})")
    return l0w, l0b, l1w, l1b


def transpose_l1w(l1w):
    """Transpose l1w from [2048][8] to [8][2048]."""
    hs2 = 2 * HS
    transposed = [0] * (hs2 * NUM_OUT)
    for r in range(hs2):
        for c in range(NUM_OUT):
            transposed[c * hs2 + r] = l1w[r * NUM_OUT + c]
    return transposed


def has_magic(data):
    return len(data) >= 4 and struct.unpack('<I', data[:4])[0] == NNUE_MAGIC


def get_header_size(data):
    if len(data) < 8:
        raise ValueError("NNUE header too small")
    ver = struct.unpack('<I', data[4:8])[0]
    if ver == 5:
        return 8
    if ver == 6:
        return 9
    if ver >= 7:
        if len(data) < 15:
            raise ValueError("NNUE v7+ header too small")
        flags = data[8]
        size = 15
        if flags & 0x80:
            size += 2
        if ver >= 10:
            size += 1
        if len(data) < size:
            raise ValueError(f"NNUE v{ver} header too small for flags {flags:#04x}")
        return size
    raise ValueError(f"Unsupported NNUE version: {ver}")


def dump_header(data, label=""):
    if not has_magic(data):
        print(f"[{label}] No magic")
        return
    ver = struct.unpack('<I', data[4:8])[0]
    flags = data[8] if ver >= 6 else 0
    ft = struct.unpack('<H', data[9:11])[0] if ver >= 7 else 0
    l1s = struct.unpack('<H', data[11:13])[0] if ver >= 7 else 0
    l2s = struct.unpack('<H', data[13:15])[0] if ver >= 7 else 0

    print(f"[{label}] v{ver} ft={ft} L1={l1s} L2={l2s} flags={flags:#010b}")
    print(f"  SCReLU={flags&1} l1sc64={(flags>>2)&1} dual={(flags>>4)&1}")


def write_nnue_v7(out_path, l0w, l0b, l1w, l1b):
    """Write NNUE v7 header + weights."""
    flags = 0b00000001  # SCReLU only
    header = struct.pack('<II', NNUE_MAGIC, 7)
    header += struct.pack('B', flags)
    header += struct.pack('<HHH', HS, 0, 0)

    body = struct.pack(f'<{len(l0w)}h', *l0w)
    body += struct.pack(f'<{len(l0b)}h', *l0b)
    body += struct.pack(f'<{len(l1w)}h', *l1w)
    body += struct.pack(f'<{len(l1b)}i', *l1b)

    expected = len(header) + len(body)
    with open(out_path, 'wb') as f:
        f.write(header + body)
    sz = os.path.getsize(out_path)
    print(f"\nWritten: {out_path}")
    print(f"  Size: {sz} bytes ({sz/1e6:.1f} MB)")
    print(f"  [OK]" if sz == expected else f"  [WARN] Expected {expected}")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input> <output.nnue>")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]

    with open(inp, 'rb') as f:
        data = f.read()

    print(f"Input: {inp}  ({len(data)} bytes, {len(data)/1e6:.1f} MB)")

    if has_magic(data):
        dump_header(data, "input")
        hdr = get_header_size(data)
        print(f"  Header: {hdr} bytes (weights already in Ember format, NO transpose)")
        payload = data[hdr:]
        l0w, l0b, l1w, l1b = read_payload(payload)
        # l1w already in [8][2048] - do NOT transpose
    else:
        print("  Raw bullet quantised.bin - NEEDS transpose!")
        l0w, l0b, l1w, l1b = read_payload(data)
        print("  Transposing l1w: [2048x8] -> [8x2048] ...")
        l1w = transpose_l1w(l1w)
        print("  Done")

    write_nnue_v7(out, l0w, l0b, l1w, l1b)


if __name__ == '__main__':
    main()
