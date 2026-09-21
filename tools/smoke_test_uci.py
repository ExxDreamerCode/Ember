#!/usr/bin/env python3
"""Run a short UCI exchange against a packaged Ember executable."""

from __future__ import annotations

import argparse
import queue
import subprocess
import time
import tomllib
from pathlib import Path
from typing import TextIO

from stress_test_uci import start_process


COMMANDS = "\n".join(
    [
        "uci",
        "isready",
        "setoption name Book value",
        "position startpos",
        "go depth 1",
        "",
    ]
)


def package_version(cargo_toml: Path) -> str:
    with cargo_toml.open("rb") as stream:
        document = tomllib.load(stream)
    version = document.get("package", {}).get("version")
    if not isinstance(version, str):
        raise ValueError(f"missing package version in {cargo_toml}")
    return version


def validate_uci_output(output: str, expected_version: str) -> None:
    lines = output.splitlines()
    required = [
        f"id name Ember {expected_version}",
        "uciok",
        "readyok",
    ]
    for line in required:
        if line not in lines:
            raise ValueError(f"UCI smoke output is missing {line!r}")
    if not any(line.startswith("bestmove ") and line != "bestmove 0000" for line in lines):
        raise ValueError("UCI smoke output has no legal best move")


def run_smoke(command: list[str], cargo_toml: Path, timeout: float,
              transcript: TextIO | None = None) -> str:
    process, lines, reader = start_process(command, transcript)
    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(COMMANDS)
    process.stdin.flush()

    captured: list[str] = []
    deadline = time.monotonic() + timeout
    bestmove_seen = False
    while not bestmove_seen:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            line = lines.get(timeout=min(0.25, remaining))
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        captured.append(line)
        bestmove_seen = line.startswith("bestmove ") and line.rstrip() != "bestmove 0000"

    if not bestmove_seen:
        process.kill()
        process.wait()
        reader.join(timeout=1.0)
        process.stdin.close()
        process.stdout.close()
        output = "".join(captured)
        raise TimeoutError(f"UCI smoke did not produce a best move:\n{output}")

    try:
        process.stdin.write("quit\n")
        process.stdin.flush()
    except BrokenPipeError:
        pass
    try:
        process.wait(timeout=min(5.0, timeout))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        reader.join(timeout=1.0)
        process.stdin.close()
        process.stdout.close()
        raise TimeoutError("UCI smoke process did not exit after quit") from None
    reader.join(timeout=1.0)
    while not lines.empty():
        captured.append(lines.get_nowait())
    process.stdin.close()
    process.stdout.close()
    reader.check_capture()

    output = "".join(captured)
    if process.returncode != 0:
        raise RuntimeError(f"UCI smoke command exited with {process.returncode}:\n{output}")
    validate_uci_output(output, package_version(cargo_toml))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo-toml", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="executable command, optionally preceded by an emulator",
    )
    args = parser.parse_args()

    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    output = run_smoke(command, args.cargo_toml, args.timeout)
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
