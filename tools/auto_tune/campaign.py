#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

from seek import (
    DEFAULT_BEST_JSON,
    DEFAULT_JOURNAL,
    DEFAULT_TUNE_TOML,
    HERE,
    ROOT,
    engine_binary,
    load_best,
    now_utc,
    read_json,
    read_toml,
    sha256_file,
    validate_best,
    validate_runtime_options,
    validate_tune_config,
    write_json,
)

CAMPAIGN_VERSION = 1
DEFAULT_CAMPAIGN_STATE = HERE / "campaign.json"
SEEK = HERE / "seek.py"
RECHECK = HERE / "recheck.py"


def pass_paths(state_path, pass_index):
    state_path = Path(state_path)
    prefix = f"{state_path.stem}.pass-{pass_index + 1}"
    return (
        state_path.with_name(f"{prefix}.seek.json"),
        state_path.with_name(f"{prefix}.pending.json"),
    )


def pass_seeds(cfg, pass_index):
    return (
        cfg["common"]["seed"] + pass_index,
        cfg["recheck"]["seed"] + pass_index,
    )


def validate_seed_schedule(cfg, max_passes):
    confirmation_seed = cfg["confirmation"]["seed"]
    seeds = []
    for pass_index in range(max_passes):
        discovery_seed, recheck_seed = pass_seeds(cfg, pass_index)
        if discovery_seed == recheck_seed:
            raise ValueError(f"pass {pass_index + 1} reuses its discovery seed")
        if confirmation_seed in (discovery_seed, recheck_seed):
            raise ValueError(
                f"pass {pass_index + 1} reuses the confirmation seed"
            )
        seeds.extend((discovery_seed, recheck_seed))
    if len(seeds) != len(set(seeds)):
        raise ValueError("campaign discovery and recheck seeds must be unique")


def advance_campaign(data, after_values):
    before_values = data["before_values"]
    changed = after_values != before_values
    pass_index = data["pass_index"]
    discovery_seed, recheck_seed = data["active_seeds"]
    data["completed_passes"].append(
        {
            "pass": pass_index + 1,
            "discovery_seed": discovery_seed,
            "recheck_seed": recheck_seed,
            "before_values": before_values,
            "after_values": dict(after_values),
            "changed": changed,
            "completed_at": now_utc(),
        }
    )
    if not changed:
        return "converged"
    if pass_index + 1 >= data["session"]["max_passes"]:
        return "pass_limit"
    data["pass_index"] = pass_index + 1
    data["phase"] = "discovery"
    data["before_values"] = dict(after_values)
    data["active_seeds"] = None
    return "continue"


def run_command(command):
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(
            f"campaign command failed with exit {completed.returncode}: "
            + " ".join(command)
        )


def add_runtime_args(command, args):
    if args.time_control:
        command.extend(["--time-control", args.time_control])
    if args.workers is not None:
        command.extend(["--workers", str(args.workers)])
    if args.worker_multiplier is not None:
        command.extend(["--worker-multiplier", str(args.worker_multiplier)])


def campaign_session(args, cfg):
    binary = engine_binary(args.engine)
    return {
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "best_path": str(Path(args.best).resolve()),
        "journal_path": str(Path(args.journal).resolve()),
        "engine_cmd": args.engine,
        "binary_path": str(Path(binary).resolve()),
        "binary_sha256": sha256_file(binary),
        "time_control": args.time_control,
        "workers": args.workers,
        "worker_multiplier": args.worker_multiplier,
        "params": args.params,
        "max_passes": args.max_passes,
        "base_discovery_seed": cfg["common"]["seed"],
        "base_recheck_seed": cfg["recheck"]["seed"],
    }


def load_campaign_state(path, session, best):
    data = read_json(path)
    if data is None:
        data = {
            "version": CAMPAIGN_VERSION,
            "session": session,
            "pass_index": 0,
            "phase": "discovery",
            "before_values": dict(best["values"]),
            "active_seeds": None,
            "completed_passes": [],
        }
        write_json(path, data)
        return data
    if data.get("version") != CAMPAIGN_VERSION:
        raise RuntimeError(f"unsupported campaign state version in {path}")
    if data.get("session") != session:
        raise RuntimeError(
            f"campaign state {path} belongs to a different invocation"
        )
    return data


def write_campaign_result(state_path, data, best, reason):
    state_path = Path(state_path)
    result_path = state_path.with_name(f"{state_path.stem}.result.json")
    result = {
        "version": CAMPAIGN_VERSION,
        "completed_at": now_utc(),
        "reason": reason,
        "converged": reason == "converged",
        "final_values": dict(best["values"]),
        "session": data["session"],
        "passes": data["completed_passes"],
    }
    write_json(result_path, result)
    state_path.unlink(missing_ok=True)
    return result_path


def main():
    parser = argparse.ArgumentParser(
        description="Run bounded auto-tune discovery and recheck passes"
    )
    parser.add_argument("--config", default=str(DEFAULT_TUNE_TOML))
    parser.add_argument("--best", default=str(DEFAULT_BEST_JSON))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument("--state", default=str(DEFAULT_CAMPAIGN_STATE))
    parser.add_argument("--engine", default="target/release/ember")
    parser.add_argument("--params", default=None)
    parser.add_argument("--time-control", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--worker-multiplier", type=float, default=None)
    parser.add_argument("--max-passes", type=int, default=3)
    args = parser.parse_args()

    try:
        validate_runtime_options(
            args.time_control,
            args.workers,
            args.worker_multiplier,
        )
        if args.max_passes < 1:
            raise ValueError("--max-passes must be positive")
        cfg = read_toml(args.config)
        params = validate_tune_config(cfg)
        validate_seed_schedule(cfg, args.max_passes)
        best = load_best(args.best)
        validate_best(best, params)
        session = campaign_session(args, cfg)
        data = load_campaign_state(args.state, session, best)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))

    try:
        while True:
            pass_index = data["pass_index"]
            seek_state, pending_path = pass_paths(args.state, pass_index)
            if data["active_seeds"] is None:
                data["active_seeds"] = list(pass_seeds(cfg, pass_index))
                write_json(args.state, data)
            discovery_seed, recheck_seed = data["active_seeds"]

            if data["phase"] == "discovery":
                print(
                    f"[campaign] pass {pass_index + 1}: discovery "
                    f"seed={discovery_seed}",
                    flush=True,
                )
                command = [
                    sys.executable,
                    str(SEEK),
                    "--config",
                    args.config,
                    "--best",
                    args.best,
                    "--journal",
                    args.journal,
                    "--state",
                    str(seek_state),
                    "--pending",
                    str(pending_path),
                    "--engine",
                    args.engine,
                    "--seed",
                    str(discovery_seed),
                    "--no-recheck",
                ]
                if args.params:
                    command.extend(["--params", args.params])
                add_runtime_args(command, args)
                run_command(command)
                data["phase"] = "recheck"
                write_json(args.state, data)

            if data["phase"] == "recheck":
                print(
                    f"[campaign] pass {pass_index + 1}: recheck "
                    f"seed={recheck_seed}",
                    flush=True,
                )
                command = [
                    sys.executable,
                    str(RECHECK),
                    "--config",
                    args.config,
                    "--best",
                    args.best,
                    "--journal",
                    args.journal,
                    "--pending",
                    str(pending_path),
                    "--engine",
                    args.engine,
                    "--seed",
                    str(recheck_seed),
                ]
                add_runtime_args(command, args)
                run_command(command)
                best = load_best(args.best)
                validate_best(best, params)
                reason = advance_campaign(data, best["values"])
                if reason != "continue":
                    result_path = write_campaign_result(
                        args.state,
                        data,
                        best,
                        reason,
                    )
                    print(
                        f"[campaign] finished: reason={reason} "
                        f"values={json.dumps(best['values'], sort_keys=True)} "
                        f"result={result_path}",
                        flush=True,
                    )
                    return 0
                write_json(args.state, data)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
