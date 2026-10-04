#!/usr/bin/env python3
"""Check Ember's Syzygy adapter against the maintained Pyrrhic oracle."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from pyrrhic_tools import pyrrhic_root



def compare_ember(request, reference, candidate, ember):
    errors = []
    if ember.get("wdl") != reference["wdl"]:
        errors.append("ember_wdl")
    if request["operation"] == "wdl":
        return errors
    if ember.get("dtz") != reference["dtz"]["value"]:
        errors.append("ember_dtz")
    reference_wdl50 = reference.get("wdl50")
    if not isinstance(reference_wdl50, str):
        errors.append("reference_wdl50_missing")
        return errors
    try:
        halfmoves = int(request["fen"].split()[4])
    except (IndexError, ValueError):
        return errors + ["invalid_fen_clock"]
    distance = reference["dtz"]["value"]
    # Ember withholds an exact search bound near the fifty-move edge.
    expected_wdl50 = (
        None if distance and 99 <= halfmoves + abs(distance) <= 101
        else reference_wdl50
    )
    if ember.get("wdl50") != expected_wdl50:
        errors.append("ember_wdl50")
    score = ember.get("search_score")
    if expected_wdl50 is None:
        if score is not None:
            errors.append("ember_ambiguous_score")
    elif not isinstance(score, int):
        errors.append("ember_score_missing")
    elif expected_wdl50 in ("Draw", "CursedWin", "BlessedLoss") and score != 0:
        errors.append("ember_draw_score")
    elif expected_wdl50 == "Win" and score <= 0:
        errors.append("ember_win_score")
    elif expected_wdl50 == "Loss" and score >= 0:
        errors.append("ember_loss_score")
    if request["operation"] == "root" and ember.get("selected_move") != candidate.get("selected_move"):
        errors.append("ember_selected_move")
    return errors


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_requests(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def read_replies(path, requests):
    lines = path.read_bytes().splitlines()
    if len(lines) != len(requests):
        raise RuntimeError(f"{path} returned {len(lines)} of {len(requests)} replies")
    replies = [json.loads(line) for line in lines]
    for request, reply in zip(requests, replies):
        if (reply.get("version"), reply.get("id"), reply.get("status")) != (
            1, request["id"], "ok"
        ):
            raise RuntimeError(f"invalid reply for {request['id']} in {path}")
    return replies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyrrhic-root", type=Path)
    parser.add_argument("--ember", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        fork = (args.pyrrhic_root or pyrrhic_root()).resolve(strict=True)
        ember = args.ember.resolve(strict=True)
        tables = args.tables.resolve(strict=True)
        comparator = fork / "tools/compare_syzygy.py"
        candidate = fork / "target/debug/examples/syzygy_probe"
        reference = fork / "tools/syzygy-reference/target/debug/pyrrhic-syzygy-reference"
        cases = fork / "tools/syzygy_compact_cases.jsonl"
        manifest = fork / "nix/syzygy-3-4-5.json"
        inputs = (comparator, candidate, reference, cases, manifest, ember)
        if not tables.is_dir() or any(not path.is_file() for path in inputs):
            raise ValueError("build the fork probes and provide complete table inputs")
        expected_requests = read_requests(cases)
        if not expected_requests:
            raise ValueError("compact case file is empty")
        expected_count = len(expected_requests)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        invocation = {
            "fork": str(fork), "ember": str(ember), "tables": str(tables),
            "timeout": args.timeout, "case_count": expected_count,
            "inputs_sha256": {str(path): digest(path) for path in inputs},
        }
        (args.output_dir / "invocation.json").write_text(
            json.dumps(invocation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        backend = args.output_dir / "backend"
        command = [
            sys.executable, str(comparator), "--candidate", str(candidate),
            "--reference", str(reference), "--tables", str(tables),
            "--manifest", str(manifest), "--cases", str(cases),
            "--output-dir", str(backend), "--timeout", str(args.timeout),
        ]
        checked = subprocess.run(command, capture_output=True, timeout=args.timeout, check=False)
        (args.output_dir / "backend.stdout.log").write_bytes(checked.stdout)
        (args.output_dir / "backend.stderr.log").write_bytes(checked.stderr)
        if checked.returncode:
            raise RuntimeError(f"backend comparison failed with exit {checked.returncode}")
        backend_summary = json.loads((backend / "summary.json").read_text(encoding="utf-8"))
        if backend_summary["mismatches"]:
            raise RuntimeError(f"backend reported {backend_summary['mismatches']} mismatches")
        if backend_summary["compared"] != expected_count:
            raise RuntimeError(
                f"backend compared {backend_summary['compared']} of "
                f"{expected_count} compact cases"
            )
        batches = sorted(backend.glob("batch-*/requests.jsonl"))
        if not batches:
            raise RuntimeError("backend comparison produced no request batches")
        request_batches = [(path, read_requests(path)) for path in batches]
        recorded_requests = [request for _, requests in request_batches for request in requests]
        if recorded_requests != expected_requests:
            raise RuntimeError(
                "backend request batches do not match the compact corpus "
                f"({len(recorded_requests)} of {expected_count} requests)"
            )
        mismatches = []
        compared = 0
        for request_file, requests in request_batches:
            batch_dir = request_file.parent
            reference_replies = read_replies(batch_dir / "reference.stdout.jsonl", requests)
            candidate_replies = read_replies(batch_dir / "candidate.stdout.jsonl", requests)
            command = [str(ember), "--tables", str(tables)]
            result = subprocess.run(
                command, input=request_file.read_bytes(), capture_output=True,
                timeout=args.timeout, check=False,
            )
            output = batch_dir / "ember.stdout.jsonl"
            output.write_bytes(result.stdout)
            (batch_dir / "ember.stderr.log").write_bytes(result.stderr)
            if result.returncode:
                raise RuntimeError(f"Ember probe exited {result.returncode}")
            ember_replies = read_replies(output, requests)
            for request, reference_reply, candidate_reply, ember_reply in zip(
                requests, reference_replies, candidate_replies, ember_replies
            ):
                compared += 1
                errors = compare_ember(
                    request, reference_reply, candidate_reply, ember_reply
                )
                if errors:
                    mismatches.append({"id": request["id"], "errors": errors})
        if compared != expected_count:
            raise RuntimeError(f"Ember compared {compared} of {expected_count} compact cases")
        (args.output_dir / "mismatches.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in mismatches),
            encoding="utf-8",
        )
        (args.output_dir / "summary.json").write_text(
            json.dumps({"compared": compared, "mismatches": len(mismatches)}, indent=2)
            + "\n", encoding="utf-8",
        )
        print(json.dumps({"compared": compared, "mismatches": len(mismatches)}))
        return 2 if mismatches else 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"Ember Syzygy comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
