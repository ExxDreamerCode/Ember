#!/usr/bin/env python3

import argparse
import base64
import json
import re
import urllib.request
from pathlib import Path


BASE_URL = "https://tablebase.lichess.ovh/tables/standard"
TABLE_NAME = re.compile(r"^(K[PNBRQ]*vK[PNBRQ]*)\.(rtbw|rtbz)$")


def read_text(source: str) -> str:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            return response.read().decode("utf-8")
    return Path(source).read_text(encoding="utf-8")


def parse_hashes(text: str) -> dict[str, str]:
    hashes = {}
    for line in text.splitlines():
        digest, name = line.split(maxsplit=1)
        hashes[name] = "sha256-" + base64.b64encode(bytes.fromhex(digest)).decode()
    return hashes


def parse_sizes(text: str) -> dict[str, int]:
    sizes = {}
    for line in text.splitlines():
        size, name = line.split(maxsplit=1)
        sizes[name] = int(size)
    return sizes


def piece_count(name: str) -> int | None:
    match = TABLE_NAME.fullmatch(name)
    if match is None:
        return None
    return len(match.group(1)) - 1  # Ignore the material separator, "v".


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a pinned Syzygy manifest from the Lichess mirror metadata."
    )
    parser.add_argument("--pieces", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256-source", default=f"{BASE_URL}/sha256")
    parser.add_argument("--bytes-source", default=f"{BASE_URL}/bytes.tsv")
    args = parser.parse_args()

    wanted = set(args.pieces)
    hashes = parse_hashes(read_text(args.sha256_source))
    sizes = parse_sizes(read_text(args.bytes_source))
    names = sorted(
        (name for name in hashes if piece_count(name) in wanted),
        key=lambda name: (piece_count(name), name),
    )

    missing_sizes = [name for name in names if name not in sizes]
    if missing_sizes:
        raise SystemExit(f"missing byte sizes for: {', '.join(missing_sizes)}")
    if not names:
        raise SystemExit("no matching Syzygy files found")

    entries = [
        {"name": name, "hash": hashes[name], "bytes": sizes[name]} for name in names
    ]
    args.output.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(entry["bytes"] for entry in entries)
    print(f"wrote {len(entries)} files ({total_bytes} bytes) to {args.output}")


if __name__ == "__main__":
    main()
