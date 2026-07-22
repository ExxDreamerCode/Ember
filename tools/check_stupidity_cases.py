#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from benchmark_search import run_engine as run_uci_search


def run_ember(binary, fen, depth, timeout):
    cmds = [
        "uci",
        "isready",
        "setoption name Book value",
        f"position fen {fen}",
        f"go depth {depth}",
        "quit",
    ]
    proc, _ = run_uci_search(
        Path(binary),
        "\n".join(cmds) + "\n",
        timeout,
    )
    bestmove = None
    for line in proc.stdout.splitlines():
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
    if proc.returncode != 0 or bestmove is None:
        raise RuntimeError(f"Ember failed for depth {depth} on {fen}\n{proc.stdout}")
    return bestmove, proc.stdout


def expected_rows(case):
    bad = {case["bad_move"], *case.get("also_bad_moves", [])}
    for depth, expected in sorted(case["fix"]["depth_moves"].items(), key=lambda row: int(row[0])):
        yield case["case_id"], case["fen"], int(depth), expected, bad
    for idx, variant in enumerate(case.get("variants", []), start=1):
        variant_bad = {variant["bad_move"]}
        for depth, expected in sorted(variant["after_depth_moves"].items(), key=lambda row: int(row[0])):
            yield f"{case['case_id']}:variant-{idx}", variant["fen"], int(depth), expected, variant_bad


def check_case(case_path, binary, timeout):
    case = json.loads(Path(case_path).read_text(encoding="utf-8"))
    failures = []
    for label, fen, depth, expected, bad_moves in expected_rows(case):
        bestmove, _ = run_ember(binary, fen, depth, timeout)
        ok = bestmove not in bad_moves
        status = "ok" if ok else "FAIL"
        print(f"{status} {label} depth={depth} bestmove={bestmove} reference={expected}")
        if not ok:
            failures.append((label, depth, bestmove, expected, sorted(bad_moves)))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Replay recorded stupidity regression cases.")
    parser.add_argument("--binary", default="target/release/ember")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    for case_path in args.cases:
        check_case(case_path, args.binary, args.timeout)


if __name__ == "__main__":
    main()
