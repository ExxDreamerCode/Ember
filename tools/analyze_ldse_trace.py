#!/usr/bin/env python3
"""Summarize LDSE probes and label their cutoffs with a deeper UCI oracle."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import analyze_singular_trace
from analyze_singular_trace import UciOracle

PREFIX = "info string search-debug ldse-event "
MATE_THRESHOLD = 90_000
REQUIRED = {
    "hash", "fen", "move", "depth", "alpha", "beta", "eval", "tt_score",
    "line_uses", "probe_depth", "probe_score", "probe_nodes", "outcome",
}


class TraceOracle(UciOracle):
    def __init__(self, binary, hash_mb, transcript=None, chess960=False):
        # Inherit stderr into the caller's transcript; an undrained pipe can
        # block a long-running oracle before it finishes its stdout response.
        self.transcript = transcript
        self.logging_error = None
        self.process = subprocess.Popen(
            [str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self._send("uci")
            options = self._read_until("uciok")
            if chess960:
                if not any(line.startswith("option name UCI_Chess960 ") for line in options):
                    raise RuntimeError("oracle does not advertise UCI_Chess960")
                self._send("setoption name UCI_Chess960 value true")
            self._send("setoption name Threads value 1")
            self._send(f"setoption name Hash value {hash_mb}")
            self._send("setoption name Ponder value false")
            self._send("isready")
            self._read_until("readyok")
        except BaseException:
            self.close()
            raise

    def _log(self, line):
        if self.transcript is not None and self.logging_error is None:
            try:
                self.transcript.write(line + "\n")
                self.transcript.flush()
            except Exception as error:
                self.logging_error = error

    def _send(self, command):
        super()._send(command)
        self._log("> " + command)

    def _read_until(self, terminator):
        if self.process.stdout is None:
            raise RuntimeError("oracle stdout is unavailable")
        lines = []
        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\r\n")
            lines.append(line)
            self._log("< " + line)
            if line.startswith(terminator):
                if self.logging_error is not None:
                    raise RuntimeError("oracle transcript write failed") from self.logging_error
                return lines
        raise RuntimeError(f"oracle stopped before {terminator}")

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self._send("quit")
                except BrokenPipeError:
                    pass
            try:
                status = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired as error:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("oracle did not exit after quit") from error
            if status != 0:
                raise RuntimeError(f"oracle exited with status {status}")
            if self.logging_error is not None:
                raise RuntimeError("oracle transcript write failed") from self.logging_error
        finally:
            if self.process.stdin:
                self.process.stdin.close()
            if self.process.stdout:
                self.process.stdout.close()


def read_events(paths):
    events = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if PREFIX not in line:
                    continue
                try:
                    event = json.loads(line.split(PREFIX, 1)[1])
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid LDSE event") from error
                if not isinstance(event, dict):
                    raise ValueError(f"{path}:{line_number}: LDSE event must be an object")
                missing = REQUIRED.difference(event)
                if missing or event["outcome"] not in {"cutoff", "rejected", "stopped"}:
                    raise ValueError(f"{path}:{line_number}: invalid LDSE fields: {sorted(missing)}")
                if not isinstance(event["probe_nodes"], int) or event["probe_nodes"] < 0:
                    raise ValueError(f"{path}:{line_number}: invalid probe node count")
                event["source"] = str(path)
                event["line"] = line_number
                events.append(event)
    return events


def summarize(events):
    outcomes = Counter(event["outcome"] for event in events)
    nodes = sum(int(event["probe_nodes"]) for event in events)
    cutoffs = outcomes["cutoff"]
    return {
        "probes": len(events),
        "outcomes": dict(sorted(outcomes.items())),
        "cutoff_rate": cutoffs / len(events) if events else 0.0,
        "probe_nodes": nodes,
        "nodes_per_cutoff": nodes / cutoffs if cutoffs else None,
    }


def label_events(events, oracle, nodes, max_positions):
    labels = {}
    unique = {(event["fen"], event["move"]): event for event in events}
    cutoffs = {(event["fen"], event["move"]) for event in events if event["outcome"] == "cutoff"}
    rank = lambda key: hashlib.sha256((key[0] + "\0" + key[1]).encode("utf-8")).digest()
    selected_cutoffs = sorted(cutoffs, key=rank)[: (max_positions + 1) // 2]
    other = sorted(unique.keys() - cutoffs, key=rank)
    selected_other = other[: max_positions - len(selected_cutoffs)]
    remaining = sorted(cutoffs - set(selected_cutoffs), key=rank)
    selected = selected_cutoffs + selected_other
    selected.extend(remaining[: max_positions - len(selected)])
    for key in selected:
        event = unique[key]
        unrestricted = oracle.search(event["fen"], nodes, multipv=1)
        candidate = oracle.search(event["fen"], nodes, multipv=1, searchmoves=[event["move"]])
        if not unrestricted or not candidate:
            raise RuntimeError(f"oracle returned no score for {event['fen']} {event['move']}")
        labels[key] = {
            "best_score_cp": unrestricted[0]["score_cp"],
            "best_pv": unrestricted[0]["pv"],
            "move_score_cp": candidate[0]["score_cp"],
            "move_pv": candidate[0]["pv"],
        }
    result = []
    for event in events:
        label = labels.get((event["fen"], event["move"]))
        if label is not None:
            comparable = abs(label["move_score_cp"]) < MATE_THRESHOLD and abs(event["beta"]) < MATE_THRESHOLD
            result.append({
                **event,
                "oracle": label,
                "oracle_move_reaches_beta": (
                    label["move_score_cp"] >= event["beta"] if comparable else None
                ),
            })
    return result


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--chess960", action="store_true", help="analyze trace positions in Chess960 mode")
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--max-positions", type=int, default=100)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.nodes <= 0 or args.hash_mb <= 0 or args.max_positions <= 0:
        parser.error("nodes, hash-mb and max-positions must be positive")
    if args.oracle and not args.transcript:
        parser.error("--transcript is required with --oracle")
    if args.oracle and not args.oracle.is_file():
        parser.error("--oracle must name an existing executable")
    events = read_events(args.traces)
    report = {
        "inputs": [{"path": str(p.resolve()), "sha256": sha256_file(p)} for p in args.traces],
        "summary": summarize(events),
    }
    if args.oracle:
        oracle_hash = sha256_file(args.oracle)
        runner_hashes = {
            "analyze_ldse_trace": sha256_file(Path(__file__)),
            "analyze_singular_trace": sha256_file(Path(analyze_singular_trace.__file__)),
        }
        with args.transcript.open("x", encoding="utf-8", newline="\n") as transcript:
            transcript.write("# " + json.dumps({
                "argv": sys.argv,
                "oracle_sha256": oracle_hash,
                "runner_sha256": runner_hashes,
                "inputs": report["inputs"],
            }, sort_keys=True) + "\n")
            transcript.flush()
            with TraceOracle(args.oracle, args.hash_mb, transcript, chess960=args.chess960) as oracle:
                labels = label_events(events, oracle, args.nodes, args.max_positions)
        report["oracle"] = {
            "path": str(args.oracle.resolve()), "sha256": oracle_hash,
            "runner_sha256": runner_hashes,
            "nodes_per_search": args.nodes, "hash_mb": args.hash_mb,
            "variant": "Chess960" if args.chess960 else "standard",
            "transcript": str(args.transcript.resolve()),
            "transcript_sha256": sha256_file(args.transcript),
            "unique_positions": len({(e["fen"], e["move"]) for e in labels}),
            "sampling": "distinct FEN and move pairs, stratified by cutoff and ranked by SHA-256",
            "interpretation": (
                "Centipawn thresholds are approximate across engines; mate windows "
                "are not compared because root-relative distances differ."
            ),
            "events": labels,
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
