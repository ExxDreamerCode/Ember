#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


DEFAULT_POSITIONS = [
    ("startpos", "startpos"),
]

INFO_RE = re.compile(
    r"^info .*?\bdepth\s+(\d+).*?\bnodes\s+(\d+)"
    r".*?\bnps\s+(\d+).*?\btime\s+(\d+)\b"
)


def parse_binary_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path).expanduser()
    if not label:
        raise argparse.ArgumentTypeError("binary label cannot be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"binary path does not exist: {path}")
    return label, path


def load_positions(path):
    if path is None:
        return DEFAULT_POSITIONS

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    positions = []
    for item in data:
        label = item["label"]
        command = item["position"]
        if command != "startpos" and not command.startswith("fen "):
            raise SystemExit(f"bad position command for {label}: {command}")
        positions.append((label, command))
    if not positions:
        raise SystemExit("position file must contain at least one position")
    return positions


def send(proc, command):
    proc.stdin.write(command + "\n")
    proc.stdin.flush()


def read_until(proc, predicate, timeout_at):
    lines = []
    while True:
        if time.monotonic() > timeout_at:
            raise TimeoutError("engine did not finish before timeout")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("engine exited before expected output")
        line = line.rstrip("\n")
        lines.append(line)
        if predicate(line):
            return lines


def run_one(binary, position, go_command, hash_mb, threads, disable_book, timeout):
    label, position_command = position
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(
        [str(binary)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    infos = []
    try:
        send(proc, "uci")
        read_until(proc, lambda line: line == "uciok", deadline)
        send(proc, f"setoption name Hash value {hash_mb}")
        send(proc, f"setoption name Threads value {threads}")
        if disable_book:
            send(proc, "setoption name Book value")
        send(proc, "isready")
        read_until(proc, lambda line: line == "readyok", deadline)
        send(proc, "ucinewgame")
        send(proc, f"position {position_command}")
        send(proc, go_command)
        start = time.perf_counter()
        lines = read_until(proc, lambda line: line.startswith("bestmove "), deadline)
        wall = time.perf_counter() - start
        for line in lines:
            match = INFO_RE.search(line)
            if match:
                infos.append(
                    {
                        "depth": int(match.group(1)),
                        "nodes": int(match.group(2)),
                        "nps": int(match.group(3)),
                        "time_ms": int(match.group(4)),
                    }
                )
        bestmove = lines[-1].split()[1]
    finally:
        if proc.poll() is None:
            try:
                send(proc, "quit")
            except BrokenPipeError:
                pass
            proc.wait(timeout=5)

    if not infos:
        raise RuntimeError(f"no info lines parsed for {binary} on {label}")

    last = infos[-1]
    return {
        "position": label,
        "depth": last["depth"],
        "nodes": last["nodes"],
        "nps": last["nps"],
        "time_ms": last["time_ms"],
        "wall_seconds": wall,
        "bestmove": bestmove,
    }


def summarize(rows):
    return {
        "median_depth": statistics.median(row["depth"] for row in rows),
        "median_nodes": statistics.median(row["nodes"] for row in rows),
        "median_nps": statistics.median(row["nps"] for row in rows),
        "median_time_ms": statistics.median(row["time_ms"] for row in rows),
        "samples": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure completed depth and node shape for UCI searches."
    )
    parser.add_argument("binaries", nargs="+", type=parse_binary_arg)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--go-command", default="go")
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--keep-book", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    positions = load_positions(args.positions)
    samples = []
    for repeat in range(1, args.repeats + 1):
        for label, binary in args.binaries:
            for position in positions:
                row = run_one(
                    binary,
                    position,
                    args.go_command,
                    args.hash,
                    args.threads,
                    not args.keep_book,
                    args.timeout,
                )
                row["label"] = label
                row["repeat"] = repeat
                samples.append(row)
                print(
                    f"{label} repeat={repeat} position={row['position']} "
                    f"depth={row['depth']} nodes={row['nodes']} "
                    f"nps={row['nps']} time_ms={row['time_ms']} "
                    f"bestmove={row['bestmove']}",
                    flush=True,
                )

    result = {
        "go_command": args.go_command,
        "hash_mb": args.hash,
        "threads": args.threads,
        "book_disabled": not args.keep_book,
        "samples": samples,
        "summaries": {},
    }
    for label, _ in args.binaries:
        rows = [row for row in samples if row["label"] == label]
        result["summaries"][label] = summarize(rows)

    print(json.dumps(result["summaries"], indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
