#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from seek import (
    DEFAULT_BEST_JSON,
    DEFAULT_JOURNAL,
    DEFAULT_PENDING_JSON,
    DEFAULT_TUNE_TOML,
    HEAD_TO_HEAD,
    ROOT,
    append_journal,
    engine_binary,
    engine_block,
    incumbent_tune_option,
    journal_has_run,
    load_best,
    load_pending,
    now_utc,
    pending_entries,
    read_summary,
    read_toml,
    resolve_repo_path,
    sha256_file,
    validate_best,
    validate_engine_params,
    validate_recheck_config,
    validate_runtime_options,
    validate_tune_config,
    value_for,
    write_json,
    write_toml_config,
)


def generate_recheck_config(cfg, best, params, engine_cmd, name, value, binary_sha256, time_control=None, workers=None, worker_multiplier=None):
    common = cfg["common"]
    recheck = cfg["recheck"]
    run_cfg = {
        "name": f"recheck-{name.lower()}-{value}",
        "time_control": time_control or recheck["time_control"],
        "timemargin_ms": common.get("timemargin_ms", 50),
        "workers": workers if workers is not None else "auto",
        "worker_multiplier": (
            worker_multiplier
            if worker_multiplier is not None
            else common.get("worker_multiplier", 1.0)
        ),
        "max_pairs": recheck["max_pairs"],
        "min_pairs": recheck["min_pairs"],
        "batch_pairs": recheck["batch_pairs"],
        "alpha": 0.05,
        "alternative": "greater",
        "opening_source": common["opening_source"],
        "polyglot_book": common["polyglot_book"],
        "book_min_plies": common["book_min_plies"],
        "book_max_plies": common["book_max_plies"],
        "opening_format": "epd",
        "seed": recheck["seed"],
        "results_dir": str(Path(cfg["results_dir"]) / "rechecks"),
        "rating_interval": 20,
        "max_moves": common["max_moves"],
        "cutechess_cmd": common.get("cutechess_cmd", "cutechess-cli"),
    }
    candidate_option = ",".join(
        f"{spec['name']}={value if spec['name'] == name else value_for(best, params, spec['name'])}"
        for spec in params
    )
    return {
        "run": run_cfg,
        "engine_a": engine_block(
            "RecheckCandidate", engine_cmd, candidate_option, common
        ),
        "engine_b": engine_block(
            "Incumbent", engine_cmd, incumbent_tune_option(best, params), common
        ),
        "recheck_meta": {
            "binary_sha256": binary_sha256,
            "param": name,
            "value": value,
        },
    }


def recheck_run_id(config):
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"recheck-{hashlib.sha256(encoded).hexdigest()[:24]}"


def validate_run_id(run_id):
    if (
        not isinstance(run_id, str)
        or len(run_id) > 128
        or run_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None
    ):
        raise ValueError(f"unsafe recheck run ID: {run_id!r}")


def prepare_config(config, run_id):
    validate_run_id(run_id)
    run_dir = resolve_repo_path(config["run"]["results_dir"]) / run_id
    config_path = run_dir / "match.toml"
    if config_path.exists():
        if read_toml(config_path) != config:
            raise RuntimeError(
                f"recheck config does not match existing run {run_id}"
            )
        return config_path, True
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"recheck run directory exists without its config: {run_dir}"
        )
    write_toml_config(config_path, config)
    return config_path, False


def run_recheck(config_path, run_id):
    command = [
        sys.executable,
        str(HEAD_TO_HEAD),
        "all",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(
            f"head_to_head recheck failed with exit {completed.returncode}"
        )


def write_recheck_result(config, config_path, run_id, summary, entry, status):
    result_dir = (
        resolve_repo_path(config["run"]["results_dir"]) / run_id
    )
    result_path = result_dir / "recheck.json"
    result = {
        "version": 1,
        "run_id": run_id,
        "completed_at": now_utc(),
        "param": entry["param"],
        "value": entry["value"],
        "summary": summary,
        "decision": status,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
    }
    write_json(result_path, result)
    return result_path, result


def update_pending_after_recheck(pending_path, key, status, summary, run_id):
    pending = load_pending(str(pending_path))
    if key not in pending["candidates"]:
        raise RuntimeError(f"pending candidate missing: {key}")
    entry = pending["candidates"][key]
    entry["status"] = status
    entry["recheck_run_id"] = run_id
    entry["recheck_elo"] = summary.get("elo")
    entry["recheck_pairs"] = summary.get("pairs", 0)
    entry["recheck_timestamp"] = now_utc()
    write_json(pending_path, pending)
    return entry


def recheck_candidate(entry, cfg, best, params, engine_cmd, journal_path, pending_path, time_control=None, workers=None, worker_multiplier=None):
    name = entry["param"]
    value = entry["value"]
    binary = engine_binary(engine_cmd)
    binary_sha256 = sha256_file(binary)
    config = generate_recheck_config(
        cfg,
        best,
        params,
        engine_cmd,
        name,
        value,
        binary_sha256,
        time_control,
        workers,
        worker_multiplier,
    )
    run_id = recheck_run_id(config)
    key = f"{name}={value}"
    if entry.get("recheck_run_id") == run_id and entry.get("status") != "pending":
        return entry
    print(f"[recheck] {name}={value} run {run_id}")
    config_path, _resumed = prepare_config(config, run_id)
    run_recheck(config_path, run_id)
    summary = read_summary(config_path, run_id)
    elo = summary.get("elo")
    accept_elo_ge = cfg["recheck"]["accept_elo_ge"]
    verdict = summary.get("verdict")
    if elo is not None and elo > accept_elo_ge and verdict != "engine_b_better":
        best["values"][name] = value
        status = "accepted"
    else:
        status = "rejected"
    print(f"[recheck] {name}={value}: elo={elo}, verdict={verdict}, status={status}")
    record = {
        "run_id": run_id,
        "timestamp": now_utc(),
        "phase": "recheck",
        "param": name,
        "old_value": value_for(best, params, name) if status != "accepted" else value,
        "new_value": value,
        "verdict": verdict,
        "accepted": status == "accepted",
        "elo": elo,
        "score_rate": summary.get("score_rate"),
        "pairs": summary.get("pairs", 0),
        "games": summary.get("games", 0),
        "binary_sha256": binary_sha256,
        "time_control": config["run"]["time_control"],
        "accept_elo_ge": accept_elo_ge,
    }
    if not journal_has_run(journal_path, run_id):
        append_journal(journal_path, record)
    write_recheck_result(config, config_path, run_id, summary, entry, status)
    return update_pending_after_recheck(pending_path, key, status, summary, run_id)


def main():
    parser = argparse.ArgumentParser(description="Recheck pending auto-tune candidates")
    parser.add_argument("--config", default=str(DEFAULT_TUNE_TOML))
    parser.add_argument("--best", default=str(DEFAULT_BEST_JSON))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument("--pending", default=str(DEFAULT_PENDING_JSON))
    parser.add_argument(
        "--engine",
        default="target/release/ember",
        help="path to the Ember release binary",
    )
    parser.add_argument(
        "--time-control",
        default=None,
        help="override recheck.time_control",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of parallel games (default: auto from CPU count)",
    )
    parser.add_argument(
        "--worker-multiplier",
        type=float,
        default=None,
        help="fraction of logical CPUs to use for workers (default: 1.0)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        validate_runtime_options(
            args.time_control,
            args.workers,
            args.worker_multiplier,
        )
        cfg = read_toml(args.config)
        params = validate_tune_config(cfg)
        recheck = validate_recheck_config(cfg)
        best = load_best(args.best)
        validate_best(best, params)
        pending = load_pending(args.pending)
        entries = pending_entries(pending)
        if not entries:
            print("[recheck] no pending candidates")
            return 0
        if not args.dry_run:
            validate_engine_params(args.engine, best, params)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))

    if args.dry_run:
        print(f"[recheck] dry-run: {len(entries)} candidate(s) would be rechecked")
        for entry in entries:
            print(
                f"[recheck] dry-run: {entry['param']}={entry['value']} "
                f"(discovery elo {entry['discovery_elo']:+.1f})"
            )
        return 0

    binary_path = engine_binary(args.engine)
    binary_sha256 = sha256_file(binary_path)
    for entry in entries:
        recheck_candidate(
            entry,
            cfg,
            best,
            params,
            args.engine,
            args.journal,
            args.pending or str(DEFAULT_PENDING_JSON),
            args.time_control,
            args.workers,
            args.worker_multiplier,
        )
    write_json(args.best, best)
    print(f"[recheck] best.json updated: {json.dumps(best['values'], sort_keys=True)}")


if __name__ == "__main__":
    sys.exit(main())