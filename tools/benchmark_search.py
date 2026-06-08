#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


DEFAULT_POSITIONS = [
    ("startpos", "startpos"),
    ("kiwipete", "fen r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QKB1R w KQkq - 0 1"),
    ("sicilian", "fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 8"),
    ("queenless-middlegame", "fen 2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14"),
    ("tactical", "fen r2q1rk1/ppp2ppp/2n1bn2/3pp3/1b2P3/2NP1N2/PPPBBPPP/R2Q1RK1 w - - 0 8"),
    ("endgame-rooks", "fen 8/2p2pk1/1p4p1/p2Pp3/P1P1P1P1/1P3K2/8/8 w - - 0 40"),
    ("minor-piece-endgame", "fen 8/5pk1/6p1/3N4/3P4/5P2/6PK/8 w - - 0 45"),
    ("promotion-race", "fen 8/1P6/8/8/8/8/6p1/6Kk w - - 0 1"),
]


INFO_RE = re.compile(r"^info .*?\bdepth\s+(\d+).*?\bnodes\s+(\d+).*?\bnps\s+(\d+)\b")


def now_id():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_engine(binary, input_text, timeout):
    start = time.perf_counter()
    proc = subprocess.run(
        [str(binary)],
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc, time.perf_counter() - start


def parse_last_info(output):
    infos = []
    for line in output.splitlines():
        match = INFO_RE.search(line)
        if match:
            infos.append({
                "depth": int(match.group(1)),
                "nodes": int(match.group(2)),
                "nps": int(match.group(3)),
            })
    return infos[-1] if infos else None


def parse_binary_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path).expanduser()
    if not label:
        raise argparse.ArgumentTypeError("binary label cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"binary does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"binary path is not a file: {path}")
    return label, path


def load_positions(path):
    if path is None:
        return DEFAULT_POSITIONS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    positions = []
    for item in data:
        try:
            label = item["label"]
            command = item["position"]
        except KeyError as exc:
            raise SystemExit(f"position file item missing key: {exc}") from exc
        if not (command == "startpos" or command.startswith("fen ")):
            raise SystemExit(f"position must be 'startpos' or start with 'fen ': {label}")
        positions.append((label, command))
    if not positions:
        raise SystemExit("position file must contain at least one position")
    return positions


def uci_input(position_command, depth, hash_mb, disable_book):
    commands = [
        "uci",
        "isready",
        f"setoption name Hash value {hash_mb}",
    ]
    if disable_book:
        commands.append("setoption name Book value")
    commands.extend([
        "ucinewgame",
        f"position {position_command}",
        f"go depth {depth}",
        "quit",
        "",
    ])
    return "\n".join(commands)


def bench_once(binary, position, depth, hash_mb, timeout, disable_book):
    label, command = position
    proc, wall = run_engine(binary, uci_input(command, depth, hash_mb, disable_book), timeout)
    parsed = parse_last_info(proc.stdout)
    if proc.returncode != 0 or parsed is None or "bestmove " not in proc.stdout:
        raise RuntimeError(f"benchmark failed for {binary} on {label}\n{proc.stdout}")
    return {
        "position": label,
        "reported_depth": parsed["depth"],
        "nodes": parsed["nodes"],
        "nps": parsed["nps"],
        "wall_seconds": wall,
    }


def summarize(samples):
    nps_values = [row["nps"] for row in samples if row["nps"] > 0]
    return {
        "samples": len(samples),
        "median_nps": statistics.median(nps_values) if nps_values else 0,
        "mean_nps": statistics.fmean(nps_values) if nps_values else 0,
        "min_nps": min(nps_values) if nps_values else 0,
        "max_nps": max(nps_values) if nps_values else 0,
        "mean_wall_seconds": statistics.fmean(row["wall_seconds"] for row in samples) if samples else 0,
        "total_nodes": sum(row["nodes"] for row in samples),
    }


def position_medians(samples):
    grouped = {}
    for row in samples:
        if row["nps"] > 0:
            grouped.setdefault(row["position"], []).append(row["nps"])
    return {position: statistics.median(values) for position, values in grouped.items()}


def geometric_mean(values):
    values = [value for value in values if value > 0]
    return statistics.geometric_mean(values) if values else 0


def paired_speedup(samples, label, baseline_label):
    baseline_positions = position_medians([row for row in samples if row["label"] == baseline_label])
    label_positions = position_medians([row for row in samples if row["label"] == label])
    ratios = [
        label_positions[position] / baseline_nps
        for position, baseline_nps in baseline_positions.items()
        if baseline_nps > 0 and position in label_positions
    ]
    return geometric_mean(ratios)


def write_report(path, result):
    lines = [
        "# Search Benchmark",
        "",
        f"- Run id: `{result['run_id']}`",
        f"- Depth: `{result['depth']}`",
        f"- Repeats: `{result['repeats']}`",
        f"- Hash: `{result['hash_mb']} MB`",
        f"- Book disabled: `{str(result['disable_book']).lower()}`",
        f"- Baseline: `{result['baseline']}`",
        "",
        "## Summary",
        "",
        "| Binary | Paired speedup | Pooled median nps | Mean nps | Mean wall s | Binary bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for binary in result["binaries"]:
        label = binary["label"]
        summary = result["summaries"][label]
        lines.append(
            f"| {label} | {paired_speedup(result['samples'], label, result['baseline']):.3f}x | "
            f"{summary['median_nps']:.0f} | {summary['mean_nps']:.0f} | "
            f"{summary['mean_wall_seconds']:.3f} | {binary['bytes']} |"
        )

    lines.extend([
        "",
        "## Binaries",
        "",
        "| Label | Path | SHA256 |",
        "| --- | --- | --- |",
    ])
    for binary in result["binaries"]:
        lines.append(f"| {binary['label']} | `{binary['path']}` | `{binary['sha256']}` |")

    lines.extend([
        "",
        "## Per-position nps",
        "",
        "| Binary | Position | Repeat | Depth | Nodes | NPS | Wall s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in result["samples"]:
        lines.append(
            f"| {row['label']} | {row['position']} | {row['repeat']} | "
            f"{row['reported_depth']} | {row['nodes']} | {row['nps']} | {row['wall_seconds']:.3f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark UCI chess-engine search throughput for existing binaries.")
    parser.add_argument("--binary", action="append", type=parse_binary_arg, required=True, metavar="LABEL=PATH")
    parser.add_argument("--baseline", default=None, help="Binary label used for paired speedups. Defaults to first --binary.")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--positions", default=None, help="Optional JSON file with [{label, position}] entries.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", default="results/search-bench")
    parser.add_argument("--keep-book", action="store_true", help="Do not send an empty Book option before searching.")
    args = parser.parse_args()

    labels = [label for label, _ in args.binary]
    if len(labels) != len(set(labels)):
        raise SystemExit("binary labels must be unique")
    baseline = args.baseline or labels[0]
    if baseline not in labels:
        raise SystemExit(f"baseline {baseline!r} is not one of: {', '.join(labels)}")
    if args.depth < 1:
        raise SystemExit("--depth must be >= 1")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    positions = load_positions(args.positions)
    run_id = args.run_id or now_id()
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "depth": args.depth,
        "repeats": args.repeats,
        "hash_mb": args.hash_mb,
        "disable_book": not args.keep_book,
        "baseline": baseline,
        "positions": [{"label": label, "position": command} for label, command in positions],
        "binaries": [],
        "samples": [],
        "summaries": {},
    }

    for label, path in args.binary:
        resolved = path.resolve()
        result["binaries"].append({
            "label": label,
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        })
        print(f"benchmarking {label} ({resolved})...", flush=True)
        for repeat in range(1, args.repeats + 1):
            for position in positions:
                sample = bench_once(resolved, position, args.depth, args.hash_mb, args.timeout, not args.keep_book)
                sample["label"] = label
                sample["repeat"] = repeat
                result["samples"].append(sample)
                print(
                    f"{label} repeat={repeat} {sample['position']} "
                    f"depth={sample['reported_depth']} nps={sample['nps']}",
                    flush=True,
                )
        result["summaries"][label] = summarize([row for row in result["samples"] if row["label"] == label])

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(out_dir / "summary.md", result)
    print(out_dir / "summary.md")


if __name__ == "__main__":
    main()
