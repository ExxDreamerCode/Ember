#!/usr/bin/env python3
"""Replay FENs through a UCI engine and report tablebase response timing."""

from __future__ import annotations

import argparse
import json
import subprocess
import time


def read_until(process: subprocess.Popen[str], prefix: str) -> list[str]:
    lines: list[str] = []
    assert process.stdout is not None
    while line := process.stdout.readline():
        line = line.rstrip()
        lines.append(line)
        if line.startswith(prefix):
            return lines
    raise RuntimeError(f"engine exited before {prefix!r}: {lines[-10:]}")


def send(process: subprocess.Popen[str], command: str) -> None:
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="target/release/ember")
    parser.add_argument("--tables", required=True)
    parser.add_argument("--clock-ms", type=int, default=150)
    parser.add_argument("--increment-ms", type=int, default=80)
    parser.add_argument("fen", nargs="+")
    args = parser.parse_args()

    process = subprocess.Popen(
        [args.engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        send(process, "uci")
        read_until(process, "uciok")
        send(process, f"setoption name SyzygyPath value {args.tables}")
        send(process, "setoption name Threads value 1")
        send(process, "isready")
        read_until(process, "readyok")

        results = []
        for fen in args.fen:
            send(process, "ucinewgame")
            send(process, f"position fen {fen}")
            send(process, "isready")
            read_until(process, "readyok")
            start = time.monotonic()
            send(
                process,
                "go "
                f"wtime {args.clock_ms} btime {args.clock_ms} "
                f"winc {args.increment_ms} binc {args.increment_ms}",
            )
            lines = read_until(process, "bestmove")
            results.append(
                {
                    "fen": fen,
                    "wall_ms": round((time.monotonic() - start) * 1000, 3),
                    "bestmove": lines[-1].split(maxsplit=1)[1],
                    "info": next(
                        (line for line in reversed(lines[:-1]) if line.startswith("info ")),
                        None,
                    ),
                }
            )
        print(json.dumps(results, indent=2))
    finally:
        if process.poll() is None:
            send(process, "quit")
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
