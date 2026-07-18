#!/usr/bin/env python3
"""Verify the immutable files in the extracted Windows bundle."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        try:
            expected, relative_text = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid manifest line {line_number}") from exc
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path on line {line_number}: {relative_text}")
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError(f"invalid SHA-256 on line {line_number}")
        entries.append((expected, Path(*relative.parts)))
    if not entries:
        raise ValueError("checksum manifest is empty")
    return entries


def verify(manifest: Path) -> list[str]:
    root = manifest.resolve().parent
    failures: list[str] = []
    for expected, relative in parse_manifest(manifest):
        candidate = root / relative
        if not candidate.is_file():
            failures.append(f"MISSING {relative.as_posix()}")
            continue
        actual = sha256_file(candidate)
        if actual != expected:
            failures.append(f"CHANGED {relative.as_posix()}")
    return failures


def main(argv: list[str]) -> int:
    manifest = Path(argv[1] if len(argv) > 1 else "SHA256SUMS.txt")
    try:
        failures = verify(manifest)
    except (OSError, ValueError) as exc:
        print(f"Verification error: {exc}", file=sys.stderr)
        return 2
    if failures:
        print("Bundle verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("Bundle verification passed.")
    print("battle.toml and results/ are user data and are intentionally not checksummed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
