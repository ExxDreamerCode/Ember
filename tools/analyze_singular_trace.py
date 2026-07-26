#!/usr/bin/env python3
"""Summarize Singular Extension candidates and optionally label them with an oracle."""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys
from collections import Counter, defaultdict


TRACE_PREFIX = "info string search-debug singular-event "
MATE_SCORE = 100_000


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_trace_line(line):
    marker = line.find(TRACE_PREFIX)
    if marker < 0:
        return None
    payload = line[marker + len(TRACE_PREFIX) :].strip()
    event = json.loads(payload)
    required = {
        "fen",
        "move",
        "depth",
        "pv",
        "tt_flag",
        "verification_nodes",
        "verification_score",
        "threshold",
        "outcome",
    }
    missing = required.difference(event)
    if missing:
        raise ValueError(f"singular event is missing fields: {sorted(missing)}")
    return event


def read_events(paths):
    events = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    event = parse_trace_line(line)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                if event is not None:
                    event["_source"] = str(path)
                    event["_line"] = line_number
                    events.append(event)
    return events


def percentile(values, percentage):
    if not values:
        return 0
    ordered = sorted(values)
    rank = math.ceil((percentage / 100.0) * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def summarize_group(events):
    nodes = [int(event["verification_nodes"]) for event in events]
    outcomes = Counter(event["outcome"] for event in events)
    useful = outcomes["extended"]
    total_nodes = sum(nodes)
    return {
        "candidates": len(events),
        "outcomes": dict(sorted(outcomes.items())),
        "extension_rate": useful / len(events) if events else 0.0,
        "verification_nodes": {
            "total": total_nodes,
            "median": statistics.median(nodes) if nodes else 0,
            "p95": percentile(nodes, 95),
            "maximum": max(nodes, default=0),
        },
        "nodes_per_extension": total_nodes / useful if useful else None,
    }


def group_summary(events, key):
    groups = defaultdict(list)
    for event in events:
        groups[str(key(event))].append(event)
    return {name: summarize_group(group) for name, group in sorted(groups.items())}


def summarize(events):
    return {
        "overall": summarize_group(events),
        "by_depth": group_summary(events, lambda event: event["depth"]),
        "by_pv": group_summary(events, lambda event: event["pv"]),
        "by_tt_flag": group_summary(events, lambda event: event["tt_flag"]),
        "by_tt_pv": group_summary(events, lambda event: event.get("tt_pv", "unknown")),
        "by_tt_age": group_summary(events, lambda event: event.get("tt_age", "unknown")),
        "by_capture": group_summary(events, lambda event: event["capture"]),
        "by_promotion": group_summary(events, lambda event: event["promotion"]),
    }


def parse_uci_info(line):
    if not line.startswith("info "):
        return None
    tokens = line.split()
    result = {"multipv": 1}
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "multipv" and index + 1 < len(tokens):
            result["multipv"] = int(tokens[index + 1])
            index += 2
        elif token == "depth" and index + 1 < len(tokens):
            result["depth"] = int(tokens[index + 1])
            index += 2
        elif token == "nodes" and index + 1 < len(tokens):
            result["nodes"] = int(tokens[index + 1])
            index += 2
        elif token == "score" and index + 2 < len(tokens):
            score_kind = tokens[index + 1]
            score_value = int(tokens[index + 2])
            if score_kind == "cp":
                result["score_cp"] = score_value
            elif score_kind == "mate":
                result["score_cp"] = (
                    MATE_SCORE - score_value
                    if score_value > 0
                    else -MATE_SCORE - score_value
                )
            index += 3
        elif token == "pv":
            result["pv"] = tokens[index + 1 :]
            break
        else:
            index += 1
    if "score_cp" not in result or not result.get("pv"):
        return None
    return result


class UciOracle:
    def __init__(self, binary, hash_mb):
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name Threads value 1")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("setoption name Ponder value false")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, command):
        if self.process.stdin is None:
            raise RuntimeError("oracle stdin is unavailable")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, terminator):
        if self.process.stdout is None:
            raise RuntimeError("oracle stdout is unavailable")
        lines = []
        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\r\n")
            lines.append(line)
            if line.startswith(terminator):
                return lines
        stderr = self.process.stderr.read() if self.process.stderr else ""
        raise RuntimeError(f"oracle stopped before {terminator}: {stderr}")

    def search(self, fen, nodes, multipv=1, searchmoves=None):
        self._send(f"setoption name MultiPV value {multipv}")
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self._send(f"position fen {fen}")
        command = f"go nodes {nodes}"
        if searchmoves:
            command += " searchmoves " + " ".join(searchmoves)
        self._send(command)
        lines = self._read_until("bestmove")
        latest = {}
        for line in lines:
            parsed = parse_uci_info(line)
            if parsed is not None:
                latest[parsed["multipv"]] = parsed
        return [latest[index] for index in sorted(latest)]

    def close(self):
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def label_position(oracle, event, nodes, gap_cp):
    candidate = event["move"]
    unrestricted = oracle.search(event["fen"], nodes, multipv=2)
    candidate_lines = oracle.search(
        event["fen"], nodes, multipv=1, searchmoves=[candidate]
    )
    if not candidate_lines:
        raise RuntimeError(f"oracle returned no score for candidate {candidate}")
    candidate_line = candidate_lines[0]
    alternatives = [
        line for line in unrestricted if line["pv"] and line["pv"][0] != candidate
    ]
    if not alternatives:
        raise RuntimeError(f"oracle returned no alternative to candidate {candidate}")
    alternative = max(alternatives, key=lambda line: line["score_cp"])
    candidate_score = candidate_line["score_cp"]
    alternative_score = alternative["score_cp"]
    gap = candidate_score - alternative_score
    best_line = max(unrestricted, key=lambda line: line["score_cp"])
    return {
        "candidate_score_cp": candidate_score,
        "best_alternative_score_cp": alternative_score,
        "candidate_gap_cp": gap,
        "oracle_singular": gap >= gap_cp,
        "oracle_best_move": best_line["pv"][0],
        "oracle_alternative_move": alternative["pv"][0],
        "candidate_pv": candidate_line["pv"],
        "alternative_pv": alternative["pv"],
    }


def label_events(events, oracle, nodes, gap_cp, max_positions=None):
    labels = {}
    for event in events:
        key = (event["fen"], event["move"])
        if key in labels:
            continue
        if max_positions is not None and len(labels) >= max_positions:
            break
        labels[key] = label_position(oracle, event, nodes, gap_cp)

    labeled = []
    for event in events:
        key = (event["fen"], event["move"])
        if key not in labels:
            continue
        item = dict(event)
        item["oracle"] = labels[key]
        labeled.append(item)
    return labeled


def oracle_summary(events):
    matrix = defaultdict(Counter)
    gaps = []
    for event in events:
        oracle = event["oracle"]
        classification = "singular" if oracle["oracle_singular"] else "not_singular"
        matrix[event["outcome"]][classification] += 1
        gaps.append(oracle["candidate_gap_cp"])
    return {
        "positions": len({(event["fen"], event["move"]) for event in events}),
        "events": len(events),
        "decision_matrix": {
            outcome: dict(sorted(counts.items()))
            for outcome, counts in sorted(matrix.items())
        },
        "candidate_gap_cp": {
            "median": statistics.median(gaps) if gaps else None,
            "p05": percentile(gaps, 5) if gaps else None,
            "p95": percentile(gaps, 95) if gaps else None,
        },
    }


def render_markdown(report):
    overall = report["summary"]["overall"]
    nodes = overall["verification_nodes"]
    lines = [
        "# Singular verification trace",
        "",
        f"- Candidates: {overall['candidates']}",
        f"- Outcomes: `{json.dumps(overall['outcomes'], sort_keys=True)}`",
        f"- Extension rate: {overall['extension_rate']:.1%}",
        f"- Verification nodes: {nodes['total']} total, {nodes['median']} median, {nodes['p95']} p95",
        (
            "- Nodes per extension: "
            + (
                f"{overall['nodes_per_extension']:.1f}"
                if overall["nodes_per_extension"] is not None
                else "n/a"
            )
        ),
    ]
    oracle = report.get("oracle")
    if oracle is not None:
        lines.extend(
            [
                "",
                "## Oracle labels",
                "",
                f"- Unique positions: {oracle['summary']['positions']}",
                f"- Labeled events: {oracle['summary']['events']}",
                f"- Decision matrix: `{json.dumps(oracle['summary']['decision_matrix'], sort_keys=True)}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize EMBER_TRACE_SINGULAR_CANDIDATES output and optionally "
            "compare candidates with a deeper UCI oracle."
        )
    )
    parser.add_argument("traces", nargs="+", type=pathlib.Path)
    parser.add_argument("--oracle", type=pathlib.Path)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--gap-cp", type=int, default=80)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.nodes <= 0 or args.hash_mb <= 0 or args.gap_cp < 0:
        raise SystemExit("--nodes and --hash-mb must be positive; --gap-cp cannot be negative")
    if args.max_positions is not None and args.max_positions <= 0:
        raise SystemExit("--max-positions must be positive")
    for path in args.traces:
        if not path.is_file():
            raise SystemExit(f"trace does not exist: {path}")
    if args.oracle is not None and not args.oracle.is_file():
        raise SystemExit(f"oracle does not exist: {args.oracle}")

    events = read_events(args.traces)
    report = {
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.traces
        ],
        "summary": summarize(events),
    }
    if args.oracle is not None:
        with UciOracle(args.oracle, args.hash_mb) as oracle:
            labeled = label_events(
                events,
                oracle,
                args.nodes,
                args.gap_cp,
                args.max_positions,
            )
        report["oracle"] = {
            "binary": str(args.oracle.resolve()),
            "sha256": sha256_file(args.oracle),
            "nodes_per_search": args.nodes,
            "hash_mb": args.hash_mb,
            "singular_gap_cp": args.gap_cp,
            "summary": oracle_summary(labeled),
            "events": labeled,
        }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
