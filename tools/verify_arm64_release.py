#!/usr/bin/env python3
"""Check generic ARM64 dispatch and plain/PGO parity under explicit QEMU CPUs.

QEMU provides correctness evidence only. Its elapsed times are not benchmarks.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from benchmark_search import sha256_file, validate_option_transcript
from smoke_test_uci import run_smoke
from stress_test_uci import run_stress, start_process, wait_for_exit, wait_for_line


CPU_MODELS = ("cortex-a53", "max")  # baseline NEON; DOTPROD available


def bench_behavior(output, depth):
    """Retain nodes, scores, PVs and signatures; discard only elapsed time/NPS."""
    totals = re.findall(
        r"^info string bench total: (\d+) positions, depth (\d+), nodes (\d+), "
        r"time \d+ms, nps \d+, signature ([0-9a-f]{16})$", output, re.MULTILINE,
    )
    positions = re.findall(r"^info string bench (\d+)/(\d+) ", output, re.MULTILINE)
    if len(totals) != 1 or totals[0][:2] != ("8", str(depth)):
        raise ValueError("missing or unexpected eight-position bench total")
    if positions != [(str(i), "8") for i in range(1, 9)]:
        raise ValueError("incomplete or reordered bench positions")
    last_info = None
    total_nodes = 0
    for line in output.splitlines():
        if line.startswith("info depth "):
            last_info = re.match(r"info depth (\d+) score (cp|mate) -?\d+ nodes (\d+) .*\bpv \S+", line)
            if last_info is None:
                raise ValueError("malformed bench search information")
        elif re.match(r"info string bench \d+/", line):
            if last_info is None or int(last_info[1]) != depth:
                raise ValueError("bench position did not complete the requested depth")
            nodes = re.search(r"\bnodes (\d+)", line)
            if nodes is None or nodes[1] != last_info[3]:
                raise ValueError("bench position node count disagrees with search")
            total_nodes += int(nodes[1])
            last_info = None
    if total_nodes != int(totals[0][2]):
        raise ValueError("bench total disagrees with position node counts")
    return [
        " ".join(re.sub(r"\b(?:time|nps) \d+(?:ms)?[,]?", "", line).split())
        for line in output.splitlines()
        if line.startswith("info depth ") or re.match(r"info string bench (?:\d+/|total:)", line)
    ]


def run_bench(command, backend, depth, timeout, log_path):
    commands = (
        "uci\nsetoption name Threads value 1\nsetoption name Hash value 64\n"
        "setoption name Book value\n"
        f"setoption name NNUEBackend value {backend}\nisready\nbench depth {depth}\nquit\n"
    )
    env = dict(os.environ)
    env.pop("EMBER_SEARCH_BACKEND", None)
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, input=commands, text=True, stdout=log,
                       stderr=subprocess.STDOUT, env=env, timeout=timeout, check=True)
    output = log_path.read_text(encoding="utf-8")
    options = validate_option_transcript(output, [("NNUEBackend", backend)])
    expected = "auto (aarch64-simd256)" if backend == "auto" else "scalar"
    if options["NNUEBackend"] != expected:
        raise ValueError(f"unexpected ARM64 backend: {options}")
    return bench_behavior(output, depth)


def run_smp_search(command, timeout, log_path):
    with log_path.open("x", encoding="utf-8") as transcript:
        _run_smp_search(command, timeout, transcript)


def _run_smp_search(command, timeout, transcript):
    process, lines, reader = start_process(command, transcript)
    captured = []
    try:
        process.stdin.write(
            "uci\nsetoption name Book value\nsetoption name Hash value 16\n"
            "setoption name Threads value 2\nsetoption name NNUEBackend value auto\nisready\n"
        )
        process.stdin.flush()
        wait_for_line(process, lines, captured, lambda line: line.startswith("readyok"), timeout)
        for moves in ("", " moves e2e4 e7e5"):
            start = len(captured)
            process.stdin.write(f"position startpos{moves}\ngo depth 4\n")
            process.stdin.flush()
            bestmove = wait_for_line(process, lines, captured,
                                     lambda line: line.startswith("bestmove "), timeout)
            if bestmove.strip() == "bestmove 0000" or not any(
                line.startswith("info depth 4 ") for line in captured[start:]
            ):
                raise RuntimeError("SMP search did not complete depth 4")
            process.stdin.write("isready\n")
            process.stdin.flush()
            wait_for_line(process, lines, captured, lambda line: line.startswith("readyok"), timeout)
        process.stdin.write("quit\n")
        process.stdin.flush()
        process.stdin.close()
        captured.append(wait_for_exit(process, lines, reader, timeout))
        options = validate_option_transcript("".join(captured), [("NNUEBackend", "auto")])
        if options["NNUEBackend"] != "auto (aarch64-simd256)":
            raise RuntimeError("SMP did not select the native ARM backend")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=1)
        while not lines.empty():
            captured.append(lines.get_nowait())
        process.stdin.close()
        process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain", type=Path, required=True)
    parser.add_argument("--pgo", type=Path, required=True)
    parser.add_argument("--qemu", default="qemu-aarch64")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.depth <= 64:
        parser.error("--depth must be between 1 and 64")
    binaries = {name: path.resolve(strict=True) for name, path in
                [("plain", args.plain), ("pgo", args.pgo)]}
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "binaries": {name: {"path": str(path), "sha256": sha256_file(path)}
                     for name, path in binaries.items()},
        "qemu_version": subprocess.check_output([args.qemu, "--version"], text=True),
        "cpu_models": CPU_MODELS, "depth": args.depth, "threads": 1, "hash_mb": 64,
        "timeout": args.timeout,
    }
    (args.out_dir / "invocation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reference = None
    os.environ.pop("EMBER_SEARCH_BACKEND", None)
    for cpu in CPU_MODELS:
        for name, binary in binaries.items():
            command = [args.qemu, "-cpu", cpu, str(binary)]
            for backend in ("scalar", "auto"):
                label = f"{cpu}-{name}-{backend}"
                behavior = run_bench(command, backend, args.depth, args.timeout,
                                     args.out_dir / f"{label}.log")
                if reference is None:
                    reference = behavior
                elif behavior != reference:
                    raise RuntimeError(f"deterministic search mismatch: {label}; see raw logs")
                print(f"{label}: exact nodes, scores, PVs and signature", flush=True)
            # Exercise the shipped automatic backend, including two-thread
            # cancellation/EOF cleanup, on both feature sets.
            with (args.out_dir / f"{cpu}-{name}-smoke.log").open("x", encoding="utf-8") as log:
                run_smoke(command, Path("Cargo.toml"), args.timeout, log)
            run_smp_search(command, args.timeout, args.out_dir / f"{cpu}-{name}-smp.log")
            run_stress(command, args.timeout, args.out_dir, f"{cpu}-{name}-")
    (args.out_dir / "passed.json").write_text(json.dumps({"behavior": reference}, indent=2) + "\n")


if __name__ == "__main__":
    main()
