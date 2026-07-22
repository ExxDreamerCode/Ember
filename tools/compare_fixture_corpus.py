#!/usr/bin/env python3

import argparse
import concurrent.futures
import hashlib
import json
import queue
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


STANDARD_HEADER = [
    "id",
    "depth",
    "fen_before_blunder",
    "setup_move",
    "expected_move",
    "themes",
    "rating",
    "popularity",
    "plays",
]
MINED_HEADER = [
    "failed_id",
    "fen_before_blunder",
    "setup_move",
    "expected_move",
    "got_depth2",
    "got_depth3",
    "got_depth4",
    "themes",
    "rating",
    "popularity",
    "plays",
]
DEFAULT_HASH_MB = 256


@dataclass(frozen=True)
class FixtureCheck:
    fixture: str
    line_number: int
    activation: str
    fixture_format: str
    case_id: str
    depth: int
    fen: str
    setup_move: str
    expected_move: str

    @property
    def key(self):
        return (self.fixture, self.line_number, self.depth)


def _standard_check(path, line_number, activation, columns):
    try:
        depth = int(columns[1])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid depth {columns[1]!r}") from error
    if not 1 <= depth <= 64:
        raise ValueError(f"{path}:{line_number}: depth must be in 1..=64")
    return FixtureCheck(
        fixture=path.name,
        line_number=line_number,
        activation=activation,
        fixture_format="standard",
        case_id=columns[0],
        depth=depth,
        fen=columns[2],
        setup_move=columns[3],
        expected_move=columns[4],
    )


def parse_fixture(path):
    path = Path(path)
    checks = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue

        if raw.startswith("# "):
            columns = raw[2:].split("\t")
            if columns == MINED_HEADER:
                continue
            if len(columns) == len(STANDARD_HEADER):
                checks.append(_standard_check(path, line_number, "disabled", columns))
            elif len(columns) == len(MINED_HEADER):
                for depth in (2, 3, 4):
                    checks.append(
                        FixtureCheck(
                            fixture=path.name,
                            line_number=line_number,
                            activation="disabled",
                            fixture_format="mined",
                            case_id=columns[0],
                            depth=depth,
                            fen=columns[1],
                            setup_move=columns[2],
                            expected_move=columns[3],
                        )
                    )
            continue

        if raw.startswith("#"):
            continue
        columns = raw.split("\t")
        if columns == STANDARD_HEADER:
            continue
        if len(columns) != len(STANDARD_HEADER):
            raise ValueError(
                f"{path}:{line_number}: expected {len(STANDARD_HEADER)} columns, "
                f"got {len(columns)}"
            )
        checks.append(_standard_check(path, line_number, "active", columns))

    return checks


def load_checks(fixture_dir):
    paths = sorted(Path(fixture_dir).glob("*.tsv"))
    if not paths:
        raise ValueError(f"no TSV fixtures found in {fixture_dir}")
    checks = [check for path in paths for check in parse_fixture(path)]
    keys = [check.key for check in checks]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate fixture/line/depth check key")
    return checks


def move_matches(actual, expected):
    if expected.startswith("!"):
        return actual not in expected[1:].split("|")
    return actual in expected.split("|")


def _read_until(lines, prefix, deadline, output):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {prefix}")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            raise TimeoutError(f"timed out waiting for {prefix}")
        if line is None:
            raise RuntimeError(f"engine exited while waiting for {prefix}")
        output.append(line)
        if line.startswith(prefix):
            return line


def uci_setup_commands(hash_mb):
    return (
        "uci",
        "setoption name Threads value 1",
        f"setoption name Hash value {hash_mb}",
        "setoption name Book value",
        "isready",
    )


def run_check(binary, check, timeout, hash_mb):
    started = time.monotonic()
    deadline = started + timeout
    process = subprocess.Popen(
        [binary],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output = []
    lines = queue.Queue()

    def collect_output():
        if process.stdout is None:
            lines.put(None)
            return
        for line in process.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    reader = threading.Thread(target=collect_output, daemon=True)
    reader.start()

    def send(command):
        if process.stdin is None:
            raise RuntimeError("engine stdin is unavailable")
        process.stdin.write(command + "\n")
        process.stdin.flush()

    try:
        setup_commands = uci_setup_commands(hash_mb)
        send(setup_commands[0])
        _read_until(lines, "uciok", deadline, output)
        for command in setup_commands[1:]:
            send(command)
        _read_until(lines, "readyok", deadline, output)
        position = f"position fen {check.fen}"
        if check.setup_move != "-":
            position += f" moves {check.setup_move}"
        send(position)
        send(f"go depth {check.depth}")
        bestmove_line = _read_until(lines, "bestmove ", deadline, output)
        bestmove = bestmove_line.split()[1]
        send("quit")
        process.wait(timeout=max(1.0, deadline - time.monotonic()))
        reader.join(timeout=1.0)
        return {
            "bestmove": bestmove,
            "passed": move_matches(bestmove, check.expected_move),
            "elapsed_seconds": time.monotonic() - started,
            "error": None,
        }
    except Exception as error:  # Preserve every failed check in the comparison report.
        if process.poll() is None:
            process.kill()
        process.wait()
        reader.join(timeout=1.0)
        return {
            "bestmove": None,
            "passed": False,
            "elapsed_seconds": time.monotonic() - started,
            "error": str(error),
            "output_tail": output[-40:],
        }


def run_binary(label, binary, checks, workers, timeout, hash_mb):
    results = {}
    completed = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_check, binary, check, timeout, hash_mb): check
            for check in checks
        }
        for future in concurrent.futures.as_completed(futures):
            check = futures[future]
            results[check.key] = future.result()
            completed += 1
            if completed % 25 == 0 or completed == len(checks):
                passed = sum(result["passed"] for result in results.values())
                errors = sum(result["error"] is not None for result in results.values())
                print(
                    f"{label}: {completed}/{len(checks)} checks, "
                    f"passed={passed}, errors={errors}",
                    flush=True,
                )
    return results, time.monotonic() - started


def direction(baseline_passed, candidate_passed):
    if baseline_passed and candidate_passed:
        return "both-pass"
    if baseline_passed:
        return "baseline-only"
    if candidate_passed:
        return "candidate-only"
    return "neither-pass"


def summarize(rows):
    groups = {}
    for row in rows:
        keys = [
            ("all", "all"),
            (row["check"]["fixture"], "all"),
            (row["check"]["fixture"], row["check"]["activation"]),
        ]
        for key in keys:
            summary = groups.setdefault(
                "/".join(key),
                {
                    "checks": 0,
                    "position_scores": {},
                    "both-pass": 0,
                    "baseline-only": 0,
                    "candidate-only": 0,
                    "neither-pass": 0,
                    "baseline-errors": 0,
                    "candidate-errors": 0,
                },
            )
            summary["checks"] += 1
            position = (
                row["check"]["fixture"],
                row["check"]["line_number"],
            )
            scores = summary["position_scores"].setdefault(position, [0, 0])
            scores[0] += row["baseline"]["passed"]
            scores[1] += row["candidate"]["passed"]
            summary[row["direction"]] += 1
            summary["baseline-errors"] += row["baseline"]["error"] is not None
            summary["candidate-errors"] += row["candidate"]["error"] is not None

    for summary in groups.values():
        position_scores = summary.pop("position_scores")
        summary["positions"] = len(position_scores)
        summary["baseline-better-positions"] = sum(
            baseline > candidate for baseline, candidate in position_scores.values()
        )
        summary["candidate-better-positions"] = sum(
            candidate > baseline for baseline, candidate in position_scores.values()
        )
        summary["equal-positions"] = sum(
            baseline == candidate for baseline, candidate in position_scores.values()
        )
        summary["baseline-passes"] = summary["both-pass"] + summary["baseline-only"]
        summary["candidate-passes"] = summary["both-pass"] + summary["candidate-only"]
    return groups


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path, rows):
    columns = [
        "fixture",
        "line",
        "activation",
        "format",
        "id",
        "depth",
        "expected",
        "baseline_move",
        "baseline_passed",
        "candidate_move",
        "candidate_passed",
        "direction",
        "baseline_error",
        "candidate_error",
    ]
    lines = ["\t".join(columns)]
    for row in rows:
        check = row["check"]
        values = [
            check["fixture"],
            str(check["line_number"]),
            check["activation"],
            check["fixture_format"],
            check["case_id"],
            str(check["depth"]),
            check["expected_move"],
            row["baseline"]["bestmove"] or "-",
            str(row["baseline"]["passed"]).lower(),
            row["candidate"]["bestmove"] or "-",
            str(row["candidate"]["passed"]).lower(),
            row["direction"],
            row["baseline"]["error"] or "-",
            row["candidate"]["error"] or "-",
        ]
        lines.append("\t".join(value.replace("\t", " ") for value in values))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Ember binaries across active and disabled TSV fixtures."
    )
    parser.add_argument("--fixtures", default="tests/fixtures")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--hash-mb",
        type=int,
        default=DEFAULT_HASH_MB,
        help=f"UCI Hash value for both engines (default: {DEFAULT_HASH_MB})",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-tsv", required=True)
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.hash_mb < 1:
        parser.error("--hash-mb must be positive")
    checks = load_checks(args.fixtures)
    print(
        f"loaded {len(checks)} checks across "
        f"{len({(check.fixture, check.line_number) for check in checks})} positions",
        flush=True,
    )

    baseline, baseline_seconds = run_binary(
        args.baseline_label,
        args.baseline,
        checks,
        args.workers,
        args.timeout,
        args.hash_mb,
    )
    candidate, candidate_seconds = run_binary(
        args.candidate_label,
        args.candidate,
        checks,
        args.workers,
        args.timeout,
        args.hash_mb,
    )

    rows = []
    for check in checks:
        baseline_result = baseline[check.key]
        candidate_result = candidate[check.key]
        rows.append(
            {
                "check": asdict(check),
                "baseline": baseline_result,
                "candidate": candidate_result,
                "direction": direction(
                    baseline_result["passed"], candidate_result["passed"]
                ),
            }
        )

    payload = {
        "metadata": {
            "baseline_label": args.baseline_label,
            "baseline_binary": str(Path(args.baseline).resolve()),
            "baseline_sha256": sha256(args.baseline),
            "candidate_label": args.candidate_label,
            "candidate_binary": str(Path(args.candidate).resolve()),
            "candidate_sha256": sha256(args.candidate),
            "fixture_sha256": {
                path.name: sha256(path)
                for path in sorted(Path(args.fixtures).glob("*.tsv"))
            },
            "workers": args.workers,
            "hash_mb": args.hash_mb,
            "timeout_seconds": args.timeout,
            "baseline_wall_seconds": baseline_seconds,
            "candidate_wall_seconds": candidate_seconds,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.output_tsv, rows)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
