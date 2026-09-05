#!/usr/bin/env python3

import argparse
import concurrent.futures
import hashlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess


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

# Exit code used when a --gate comparison fails its acceptance rules.
GATE_EXIT_FAIL = 2

POSITION_ERROR_PREFIXES = (
    "info string Ignoring invalid FEN:",
    "info string Stopping position move list at illegal move:",
)


@dataclass(frozen=True)
class FixtureCheck:
    fixture: str
    line_number: int
    activation: str
    fixture_format: str
    variant: str
    case_id: str
    depth: int
    fen: str
    setup_move: str
    expected_move: str

    @property
    def key(self):
        return (self.fixture, self.line_number, self.depth)


def _parse_variant(path, line_number, raw):
    variant = raw.strip().lower()
    if variant in ("standard", "chess960", "fischerandom", "fischer random"):
        return "chess960" if variant != "standard" else "standard"
    raise ValueError(f"{path}:{line_number}: invalid fixture variant {raw!r}")


def _validate_fen_board(path, line_number, fen):
    fields = fen.split()
    if not fields:
        raise ValueError(f"{path}:{line_number}: empty FEN")
    ranks = fields[0].split("/")
    if len(ranks) != 8:
        raise ValueError(f"{path}:{line_number}: FEN board must contain 8 ranks")
    for rank_number, rank in enumerate(ranks, 1):
        width = 0
        for symbol in rank:
            if symbol in "12345678":
                width += int(symbol)
            elif symbol in "pnbrqkPNBRQK":
                width += 1
            else:
                raise ValueError(
                    f"{path}:{line_number}: invalid FEN board symbol {symbol!r}"
                )
        if width != 8:
            raise ValueError(
                f"{path}:{line_number}: FEN rank {rank_number} expands to {width} squares"
            )


def _standard_check(path, line_number, activation, columns, variant):
    try:
        depth = int(columns[1])
    except ValueError as error:
        raise ValueError(f"{path}:{line_number}: invalid depth {columns[1]!r}") from error
    if not 0 <= depth <= 64:
        raise ValueError(f"{path}:{line_number}: depth must be in 0..=64")
    _validate_fen_board(path, line_number, columns[2])
    return FixtureCheck(
        fixture=path.name,
        line_number=line_number,
        activation=activation,
        fixture_format="standard",
        variant=variant,
        case_id=columns[0],
        depth=depth,
        fen=columns[2],
        setup_move=columns[3],
        expected_move=columns[4],
    )


def parse_fixture(path):
    path = Path(path)
    checks = []
    variant = "standard"
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue

        if raw.startswith("# Variant:"):
            variant = _parse_variant(path, line_number, raw[len("# Variant:") :])
            continue

        if raw.startswith("# "):
            columns = raw[2:].split("\t")
            if columns == MINED_HEADER:
                continue
            if len(columns) == len(STANDARD_HEADER):
                checks.append(
                    _standard_check(path, line_number, "disabled", columns, variant)
                )
            elif len(columns) == len(MINED_HEADER):
                _validate_fen_board(path, line_number, columns[1])
                for depth in (2, 3, 4):
                    checks.append(
                        FixtureCheck(
                            fixture=path.name,
                            line_number=line_number,
                            activation="disabled",
                            fixture_format="mined",
                            variant=variant,
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
        checks.append(_standard_check(path, line_number, "active", columns, variant))

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


def validate_bestmove(check, actual):
    if actual in {"0000", "(none)"}:
        raise RuntimeError(f"engine returned null bestmove {actual}")
    try:
        board = chess.Board(check.fen, chess960=check.variant == "chess960")
        if check.setup_move != "-":
            for move in check.setup_move.split():
                board.push_uci(move)
        board.parse_uci(actual)
    except ValueError as error:
        raise RuntimeError(f"engine returned illegal bestmove {actual!r}") from error


def parse_uci_option(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    name, option_value = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("option name cannot be empty")
    return name, option_value


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


def uci_setup_commands(hash_mb, use_embedded_book=False, chess960=False, options=()):
    commands = [
        "uci",
        "setoption name Threads value 1",
        f"setoption name Hash value {hash_mb}",
        "setoption name UCI_Chess960 value true" if chess960 else None,
        "setoption name Book value <embedded>"
        if use_embedded_book
        else "setoption name Book value",
    ]
    commands.extend(f"setoption name {name} value {value}" for name, value in options)
    commands.append("isready")
    return tuple(commands)


def search_observations(output):
    depths = []
    nodes_at_depth = {}
    reported_nodes = None
    for line in output:
        if not line.startswith("info "):
            continue
        depth_match = re.search(r"(?:^|\s)depth\s+(\d+)(?:\s|$)", line)
        nodes_match = re.search(r"(?:^|\s)nodes\s+(\d+)(?:\s|$)", line)
        if depth_match is not None:
            depth = int(depth_match.group(1))
            depths.append(depth)
        if nodes_match is not None:
            reported_nodes = int(nodes_match.group(1))
            if depth_match is not None:
                nodes_at_depth[depth] = reported_nodes
    reported_depth = max(depths, default=None)
    if reported_depth in nodes_at_depth:
        reported_nodes = nodes_at_depth[reported_depth]
    return reported_depth, reported_nodes


def validate_uci_contract(
    check,
    position_output,
    reported_depth,
    reported_nodes,
    allow_unreported_book_stats=False,
):
    diagnostics = [
        line
        for line in position_output
        if line.startswith(POSITION_ERROR_PREFIXES)
    ]
    if diagnostics:
        raise RuntimeError(f"position setup was rejected: {diagnostics[0]}")
    if (
        check.depth == 0
        and reported_depth is None
        and reported_nodes is None
        and allow_unreported_book_stats
    ):
        return
    if reported_depth is None or reported_nodes is None:
        raise RuntimeError("engine did not report search depth and nodes")
    if check.depth == 0 and reported_nodes != 0:
        raise RuntimeError(
            f"depth-zero book check searched {reported_nodes} node(s)"
        )
    if check.depth > 0 and reported_depth < check.depth:
        raise RuntimeError(
            f"engine stopped at depth {reported_depth}, below requested depth {check.depth}"
        )


def run_check(
    binary,
    check,
    timeout,
    hash_mb,
    options=(),
    allow_unreported_book_stats=False,
):
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
        setup_commands = uci_setup_commands(
            hash_mb, check.depth == 0, check.variant == "chess960", options
        )
        send(setup_commands[0])
        _read_until(lines, "uciok", deadline, output)
        for command in setup_commands[1:]:
            if command is None:
                continue
            send(command)
        _read_until(lines, "readyok", deadline, output)
        position = f"position fen {check.fen}"
        if check.setup_move != "-":
            position += f" moves {check.setup_move}"
        position_output_start = len(output)
        send(position)
        send(f"go depth {check.depth}")
        bestmove_line = _read_until(lines, "bestmove ", deadline, output)
        fields = bestmove_line.split()
        if len(fields) < 2:
            raise RuntimeError("engine returned malformed bestmove")
        bestmove = fields[1]
        position_output = output[position_output_start:]
        reported_depth, reported_nodes = search_observations(position_output)
        validate_uci_contract(
            check,
            position_output,
            reported_depth,
            reported_nodes,
            allow_unreported_book_stats,
        )
        validate_bestmove(check, bestmove)
        send("quit")
        return_code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
        reader.join(timeout=1.0)
        if return_code != 0:
            raise RuntimeError(f"engine exited with status {return_code}")
        return {
            "bestmove": bestmove,
            "passed": move_matches(bestmove, check.expected_move),
            "reported_depth": reported_depth,
            "reported_nodes": reported_nodes,
            "elapsed_seconds": time.monotonic() - started,
            "error": None,
        }
    except Exception as error:  # Preserve every failed check in the comparison report.
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5.0)
        except Exception:
            pass
        reader.join(timeout=1.0)
        return {
            "bestmove": None,
            "passed": False,
            "elapsed_seconds": time.monotonic() - started,
            "error": str(error),
            "output_tail": output[-40:],
        }
    finally:
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


def run_binary(
    label,
    binary,
    checks,
    workers,
    timeout,
    hash_mb,
    options=(),
    allow_unreported_book_stats=False,
):
    results = {}
    completed = 0
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_check,
                binary,
                check,
                timeout,
                hash_mb,
                options,
                allow_unreported_book_stats,
            ): check
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


def disabled_status(row):
    if row["check"]["activation"] != "disabled":
        return "-"
    baseline_passed = row["baseline"]["passed"]
    candidate_passed = row["candidate"]["passed"]
    if baseline_passed and candidate_passed:
        return "stale-passes-in-both"
    if candidate_passed:
        return "fixed-by-candidate"
    if baseline_passed:
        return "lost-while-disabled"
    return "still-red"


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
                    "disabled-status": {},
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
            status = disabled_status(row)
            if status != "-":
                summary["disabled-status"][status] = (
                    summary["disabled-status"].get(status, 0) + 1
                )
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
        "variant",
        "id",
        "depth",
        "expected",
        "baseline_move",
        "baseline_passed",
        "candidate_move",
        "candidate_passed",
        "direction",
        "disabled_status",
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
            check["variant"],
            check["case_id"],
            str(check["depth"]),
            check["expected_move"],
            row["baseline"]["bestmove"] or "-",
            str(row["baseline"]["passed"]).lower(),
            row["candidate"]["bestmove"] or "-",
            str(row["candidate"]["passed"]).lower(),
            row["direction"],
            disabled_status(row),
            row["baseline"]["error"] or "-",
            row["candidate"]["error"] or "-",
        ]
        lines.append("\t".join(value.replace("\t", " ") for value in values))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gate(
    rows,
    hard_fixtures=(),
    profile="strict",
    net_tolerance_permille=10,
    floor_ratio_permille=800,
):
    if profile not in ("strict", "rearchitecture"):
        raise ValueError(f"unknown fixture gate profile {profile!r}")

    active = [row for row in rows if row["check"]["activation"] == "active"]
    baseline_passes = sum(1 for row in active if row["baseline"]["passed"])
    candidate_passes = sum(1 for row in active if row["candidate"]["passed"])
    baseline_errors = sum(1 for row in rows if row["baseline"]["error"] is not None)
    candidate_errors = sum(1 for row in rows if row["candidate"]["error"] is not None)
    baseline_only = [row for row in active if row["direction"] == "baseline-only"]
    candidate_only = [row for row in active if row["direction"] == "candidate-only"]

    hard = {name for name in hard_fixtures}
    active_fixture_names = {row["check"]["fixture"] for row in active}
    missing_hard_fixtures = sorted(hard - active_fixture_names)
    hard_failed = [
        row
        for row in active
        if row["check"]["fixture"] in hard and not row["candidate"]["passed"]
    ]

    net_loss = baseline_passes - candidate_passes
    net_tolerance = baseline_passes * net_tolerance_permille // 1000
    floor = (baseline_passes * floor_ratio_permille + 999) // 1000

    reasons = []
    if missing_hard_fixtures:
        reasons.append(
            "hard fixture(s) have no active checks: "
            f"{', '.join(missing_hard_fixtures)}"
        )
    if hard_failed:
        reasons.append(
            f"{len(hard_failed)} hard-layer candidate failure(s) in "
            f"{', '.join(sorted(hard))}"
        )
    if baseline_errors or candidate_errors:
        reasons.append(
            "engine errors invalidate the comparison "
            f"(baseline {baseline_errors}, candidate {candidate_errors})"
        )
    if (
        profile == "strict"
        and net_loss * 1000 > baseline_passes * net_tolerance_permille
    ):
        reasons.append(
            f"net loss {net_loss} exceeds exact tolerance "
            f"({net_tolerance_permille} permille of {baseline_passes} baseline passes)"
        )
    if (
        profile == "rearchitecture"
        and candidate_passes * 1000 < baseline_passes * floor_ratio_permille
    ):
        reasons.append(
            f"candidate {candidate_passes} below proportional floor {floor} "
            f"({floor_ratio_permille} permille of baseline {baseline_passes})"
        )

    def summarize_rows(rows_list):
        items = []
        for row in rows_list:
            check = row["check"]
            items.append(
                {
                    "fixture": check["fixture"],
                    "line": check["line_number"],
                    "id": check["case_id"],
                    "depth": check["depth"],
                    "expected": check["expected_move"],
                    "baseline_move": row["baseline"]["bestmove"] or "-",
                    "candidate_move": row["candidate"]["bestmove"] or "-",
                }
            )
        return items

    return {
        "passed": not reasons,
        "profile": profile,
        "active_checks": len(active),
        "baseline_passes": baseline_passes,
        "candidate_passes": candidate_passes,
        "net_loss": net_loss,
        "net_tolerance": net_tolerance,
        "floor": floor,
        "baseline_errors": baseline_errors,
        "candidate_errors": candidate_errors,
        "hard_fixtures": sorted(hard),
        "missing_hard_fixtures": missing_hard_fixtures,
        "hard_failed": summarize_rows(hard_failed),
        # Retain this field for consumers of the first gate-report schema.
        "hard_regressed": summarize_rows(hard_failed),
        "regressed": summarize_rows(baseline_only),
        "fixed": summarize_rows(candidate_only),
        "reasons": reasons,
    }


def format_gate_report(gate):
    lines = []
    lines.append("=== Fixture gate ===")
    lines.append(
        f"active checks: {gate['active_checks']}; "
        f"baseline passes: {gate['baseline_passes']}; "
        f"candidate passes: {gate['candidate_passes']}"
    )
    lines.append(f"profile: {gate['profile']}")
    if gate["profile"] == "strict":
        lines.append(
            f"net loss: {gate['net_loss']} "
            f"(whole-case tolerance {gate['net_tolerance']})"
        )
    else:
        lines.append(f"proportional candidate floor: {gate['floor']}")
    if gate["baseline_errors"] or gate["candidate_errors"]:
        lines.append(
            "INVALID: engine errors -> "
            f"baseline {gate['baseline_errors']}, candidate {gate['candidate_errors']} "
            "(comparison rejected)"
        )
    for header, items in (
        ("HARD-layer candidate failures", gate["hard_failed"]),
        ("Regressed (baseline-only, active)", gate["regressed"]),
        ("Fixed (candidate-only, active)", gate["fixed"]),
    ):
        if not items:
            continue
        lines.append(f"{header} ({len(items)}):")
        for item in items:
            lines.append(
                f"  {item['fixture']}:{item['line']} {item['id']} "
                f"expected={item['expected']} baseline={item['baseline_move']} "
                f"candidate={item['candidate_move']}"
            )
    lines.append(f"gate verdict: {'PASS' if gate['passed'] else 'FAIL'}")
    for reason in gate["reasons"]:
        lines.append(f"  rejected: {reason}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Ember binaries across active and disabled TSV fixtures."
    )
    parser.add_argument("--fixtures", default="tests/fixtures")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument(
        "--baseline-option",
        action="append",
        type=parse_uci_option,
        default=[],
        metavar="NAME=VALUE",
    )
    parser.add_argument(
        "--baseline-allow-unreported-book-stats",
        action="store_true",
        help=(
            "allow a historical baseline to omit depth/node telemetry for "
            "depth-zero book results; observed nonzero nodes still fail"
        ),
    )
    parser.add_argument(
        "--candidate-option",
        action="append",
        type=parse_uci_option,
        default=[],
        metavar="NAME=VALUE",
    )
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
    parser.add_argument(
        "--gate",
        action="store_true",
        help=(
            "enforce the two-binary fixture gate: hard-layer regressions, "
            "net-loss tolerance, and absolute floor over active cases"
        ),
    )
    parser.add_argument(
        "--gate-profile",
        choices=("strict", "rearchitecture"),
        default="strict",
        help=(
            "strict enforces the net-loss tolerance; rearchitecture enforces "
            "the proportional candidate floor (default: strict)"
        ),
    )
    parser.add_argument(
        "--gate-hard-fixtures",
        default="engine_regressions.tsv",
        metavar="FIXTURE[,FIXTURE...]",
        help="fixtures whose active pass->fail flips always fail the gate",
    )
    parser.add_argument(
        "--gate-net-tolerance-permille",
        type=int,
        default=10,
        help=(
            "allowed net active-case loss in permille of baseline passes "
            "(default: 10 = 1%%)"
        ),
    )
    parser.add_argument(
        "--gate-floor-ratio-permille",
        type=int,
        default=800,
        help=(
            "candidate must solve at least this permille of baseline active "
            "passes as an absolute floor (default: 800 = 80%%)"
        ),
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.hash_mb < 1:
        parser.error("--hash-mb must be positive")
    for permille_name, value in (
        ("net-tolerance", args.gate_net_tolerance_permille),
        ("floor-ratio", args.gate_floor_ratio_permille),
    ):
        if not 0 <= value <= 1000:
            parser.error(f"--gate-{permille_name}-permille must be in 0..=1000")
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
        args.baseline_option,
        args.baseline_allow_unreported_book_stats,
    )
    candidate, candidate_seconds = run_binary(
        args.candidate_label,
        args.candidate,
        checks,
        args.workers,
        args.timeout,
        args.hash_mb,
        args.candidate_option,
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
            "baseline_options": args.baseline_option,
            "baseline_allow_unreported_book_stats": (
                args.baseline_allow_unreported_book_stats
            ),
            "candidate_label": args.candidate_label,
            "candidate_binary": str(Path(args.candidate).resolve()),
            "candidate_sha256": sha256(args.candidate),
            "candidate_options": args.candidate_option,
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

    if not args.gate:
        print(json.dumps(payload["summary"], indent=2), flush=True)
        return 0

    hard_fixtures = [
        name.strip()
        for name in args.gate_hard_fixtures.split(",")
        if name.strip()
    ]
    gate = evaluate_gate(
        rows,
        hard_fixtures=hard_fixtures,
        profile=args.gate_profile,
        net_tolerance_permille=args.gate_net_tolerance_permille,
        floor_ratio_permille=args.gate_floor_ratio_permille,
    )
    payload["gate"] = gate
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(format_gate_report(gate), flush=True)
    return GATE_EXIT_FAIL if not gate["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
