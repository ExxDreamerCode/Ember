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
    DEFAULT_TUNE_TOML,
    HEAD_TO_HEAD,
    ROOT,
    engine_binary,
    engine_block,
    incumbent_tune_option,
    load_best,
    now_utc,
    read_summary,
    read_toml,
    resolve_repo_path,
    sha256_file,
    validate_best,
    validate_engine_params,
    validate_runtime_options,
    validate_tune_config,
    value_for,
    write_json,
    write_toml_config,
)


def tuned_changes(best, params):
    return {
        spec["name"]: {
            "default": spec["base"],
            "tuned": value_for(best, params, spec["name"]),
        }
        for spec in params
        if value_for(best, params, spec["name"]) != spec["base"]
    }


def generate_confirmation_config(
    cfg,
    best,
    params,
    engine_cmd,
    binary_sha256,
    time_control=None,
    workers=None,
    worker_multiplier=None,
):
    changes = tuned_changes(best, params)
    if not changes:
        raise ValueError("best.json has no changes to confirm")
    if not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
        raise ValueError("binary SHA-256 must be 64 lowercase hexadecimal digits")

    common = cfg["common"]
    confirmation = cfg["confirmation"]
    run_cfg = {
        "name": "auto-tune-final-confirmation",
        "time_control": time_control or confirmation["time_control"],
        "timemargin_ms": common.get("timemargin_ms", 50),
        "workers": workers if workers is not None else "auto",
        "worker_multiplier": (
            worker_multiplier
            if worker_multiplier is not None
            else common.get("worker_multiplier", 1.0)
        ),
        "max_pairs": confirmation["max_pairs"],
        "min_pairs": confirmation["min_pairs"],
        "batch_pairs": confirmation["batch_pairs"],
        "alpha": confirmation["alpha"],
        "alternative": "greater",
        "opening_source": common["opening_source"],
        "polyglot_book": common["polyglot_book"],
        "book_min_plies": common["book_min_plies"],
        "book_max_plies": common["book_max_plies"],
        "opening_format": "epd",
        "seed": confirmation["seed"],
        "results_dir": str(Path(cfg["results_dir"]) / "confirmations"),
        "rating_interval": 20,
        "max_moves": common["max_moves"],
        "cutechess_cmd": common.get("cutechess_cmd", "cutechess-cli"),
    }
    defaults = {"values": {}}
    return {
        "run": run_cfg,
        "sprt": {
            "enabled": confirmation["enabled"],
            "elo0": confirmation["elo0"],
            "elo1": confirmation["elo1"],
            "alpha": confirmation["alpha"],
            "beta": confirmation["beta"],
        },
        "engine_a": engine_block(
            "Tuned",
            engine_cmd,
            incumbent_tune_option(best, params),
            common,
        ),
        "engine_b": engine_block(
            "Defaults",
            engine_cmd,
            incumbent_tune_option(defaults, params),
            common,
        ),
        "confirmation_meta": {
            "binary_sha256": binary_sha256,
            "candidate_changes": changes,
        },
    }


def confirmation_run_id(config):
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"confirm-{hashlib.sha256(encoded).hexdigest()[:24]}"


def validate_run_id(run_id):
    if (
        not isinstance(run_id, str)
        or len(run_id) > 128
        or run_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9_.-]+", run_id) is None
    ):
        raise ValueError(f"unsafe confirmation run ID: {run_id!r}")


def prepare_confirmation_config(config, run_id):
    validate_run_id(run_id)
    run_dir = resolve_repo_path(config["run"]["results_dir"]) / run_id
    config_path = run_dir / "match.toml"
    if config_path.exists():
        if read_toml(config_path) != config:
            raise RuntimeError(
                f"confirmation config does not match existing run {run_id}"
            )
        return config_path, True
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"confirmation run directory exists without its config: {run_dir}"
        )
    write_toml_config(config_path, config)
    return config_path, False


def run_confirmation(config_path, run_id):
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
            f"head_to_head confirmation failed with exit {completed.returncode}"
        )


def write_confirmation_result(
    config,
    config_path,
    run_id,
    binary_path,
    binary_sha256,
    changes,
    summary,
):
    result_path = (
        resolve_repo_path(config["run"]["results_dir"])
        / run_id
        / "confirmation.json"
    )
    result = {
        "version": 1,
        "run_id": run_id,
        "completed_at": now_utc(),
        "confirmed": summary["verdict"] == "engine_a_better",
        "verdict": summary["verdict"],
        "candidate_changes": changes,
        "binary_path": str(Path(binary_path).resolve()),
        "binary_sha256": binary_sha256,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "summary": summary,
    }
    write_json(result_path, result)
    return result_path, result


def main():
    parser = argparse.ArgumentParser(
        description="Confirm the final auto-tuned vector with an independent SPRT"
    )
    parser.add_argument("--config", default=str(DEFAULT_TUNE_TOML))
    parser.add_argument("--best", default=str(DEFAULT_BEST_JSON))
    parser.add_argument(
        "--engine",
        default="target/release/ember",
        help="path to the Ember release binary",
    )
    parser.add_argument(
        "--time-control",
        default=None,
        help="override confirmation.time_control",
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
        best = load_best(args.best)
        validate_best(best, params)
        changes = tuned_changes(best, params)
        if not changes:
            raise ValueError("best.json has no changes to confirm")

        if args.dry_run:
            print("[confirm] dry-run: independent confirmation is configured")
            print("[confirm] changes: " + json.dumps(changes, sort_keys=True))
            print(
                "[confirm] settings: "
                + json.dumps(
                    {
                        "time_control": (
                            args.time_control
                            or cfg["confirmation"]["time_control"]
                        ),
                        "seed": cfg["confirmation"]["seed"],
                        "elo0": cfg["confirmation"]["elo0"],
                        "elo1": cfg["confirmation"]["elo1"],
                        "alpha": cfg["confirmation"]["alpha"],
                        "beta": cfg["confirmation"]["beta"],
                        "max_pairs": cfg["confirmation"]["max_pairs"],
                    },
                    sort_keys=True,
                )
            )
            print("[confirm] dry-run: binary not probed and no files written")
            return 0

        validate_engine_params(args.engine, best, params)
        binary_path = engine_binary(args.engine)
        binary_sha256 = sha256_file(binary_path)
        config = generate_confirmation_config(
            cfg,
            best,
            params,
            args.engine,
            binary_sha256,
            args.time_control,
            args.workers,
            args.worker_multiplier,
        )
        run_id = confirmation_run_id(config)
        config_path, resumed = prepare_confirmation_config(config, run_id)
        action = "resuming" if resumed else "starting"
        print(f"[confirm] {action} deterministic run {run_id}")
        print(f"[confirm] config: {config_path}")
        run_confirmation(config_path, run_id)
        if sha256_file(binary_path) != binary_sha256:
            raise RuntimeError("engine binary changed during confirmation")
        summary = read_summary(config_path, run_id)
        result_path, result = write_confirmation_result(
            config,
            config_path,
            run_id,
            binary_path,
            binary_sha256,
            changes,
            summary,
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))

    if result["confirmed"]:
        print(f"[confirm] confirmed: tuned vector accepted H1 ({result_path})")
        return 0
    print(
        "[confirm] not confirmed: "
        f"verdict={result['verdict']} ({result_path})"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
