#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HEAD_TO_HEAD = ROOT / "tools" / "head_to_head.py"
DEFAULT_TUNE_TOML = HERE / "tune.toml"
DEFAULT_BEST_JSON = HERE / "best.json"
DEFAULT_JOURNAL = HERE / "journal.jsonl"
EMPTY_BEST = {"values": {}}


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_json(path):
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def append_journal(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_best(path):
    data = read_json(path) or EMPTY_BEST
    data.setdefault("values", {})
    return data


def param_specs(cfg):
    return list(cfg["params"])


def value_for(best, params, name):
    if name in best["values"]:
        return best["values"][name]
    for spec in params:
        if spec["name"] == name:
            return spec["base"]
    raise RuntimeError(f"no base value for {name}")


def incumbent_tune_option(best, params):
    overrides = []
    for spec in params:
        value = value_for(best, params, spec["name"])
        if value != spec["base"]:
            overrides.append(f"{spec['name']}={value}")
    return ",".join(overrides)


def candidate_tune_option(best, params, candidate_name, candidate_value):
    overrides = []
    for spec in params:
        value = candidate_value if spec["name"] == candidate_name else value_for(
            best, params, spec["name"]
        )
        if value != spec["base"]:
            overrides.append(f"{spec['name']}={value}")
    return ",".join(overrides)


def in_range(spec, value):
    return spec["min"] <= value <= spec["max"]


def engine_block(name, cmd, tune_value, common):
    options = {
        "Hash": str(common["hash_mb"]),
        "Threads": str(common["threads"]),
        "Book": "",
    }
    if tune_value:
        options["Tune"] = tune_value
    return {
        "name": name,
        "cmd": cmd,
        "proto": "uci",
        "options": options,
    }


def toml_key(key):
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_-]+", key):
        return key
    return json.dumps(str(key))


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{toml_key(key)} = {toml_value(item)}" for key, item in value.items()
        )
        return "{ " + inner + " }"
    raise TypeError(f"unsupported TOML value: {value!r}")


def write_toml_config(path, config):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, body in config.items():
        lines.append(f"[{section}]")
        for key, value in body.items():
            lines.append(f"{key} = {toml_value(value)}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_match_config(
    cfg, common, sprt, candidate_name, candidate_value, best, params, engine_cmd, time_control, workers, worker_multiplier
):
    run_cfg = {
        "name": f"tune-{candidate_name.lower()}-{candidate_value}",
        "time_control": time_control,
        "timemargin_ms": 2000,
        "workers": workers if workers is not None else "auto",
        "worker_multiplier": worker_multiplier if worker_multiplier is not None else common.get("worker_multiplier", 1.0),
        "max_pairs": common["max_pairs"],
        "min_pairs": common["min_pairs"],
        "batch_pairs": common["batch_pairs"],
        "alpha": sprt.get("alpha", 0.05),
        "alternative": "greater",
        "opening_source": common["opening_source"],
        "polyglot_book": common["polyglot_book"],
        "book_min_plies": common["book_min_plies"],
        "book_max_plies": common["book_max_plies"],
        "opening_format": "epd",
        "seed": common["seed"],
        "results_dir": str(Path(cfg["results_dir"]) / "runs"),
        "rating_interval": 20,
        "max_moves": common["max_moves"],
        "cutechess_cmd": common.get("cutechess_cmd", "cutechess-cli"),
    }
    return {
        "run": run_cfg,
        "sprt": {
            "enabled": True,
            "elo0": sprt["elo0"],
            "elo1": sprt["elo1"],
            "alpha": sprt.get("alpha", 0.05),
            "beta": sprt.get("beta", 0.05),
        },
        # The SPRT expresses Elo as engine A minus engine B.  Put the
        # candidate in A so accepting H1 (elo1 > 0) is positive evidence for
        # the change, rather than merely a failure to show that it is worse.
        "engine_a": engine_block(
            "Candidate",
            engine_cmd,
            candidate_tune_option(best, params, candidate_name, candidate_value),
            common,
        ),
        "engine_b": engine_block(
            "Incumbent", engine_cmd, incumbent_tune_option(best, params), common
        ),
    }


def run_match(config_path, run_id):
    cmd = [
        sys.executable,
        str(HEAD_TO_HEAD),
        "run",
        "--config",
        str(config_path),
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"head_to_head failed ({proc.returncode}): {proc.stdout[-2000:]}"
        )
    return proc.stdout


def read_summary(config_path, run_id):
    cfg = read_toml(config_path)
    rd = Path(cfg["run"]["results_dir"]) / run_id
    summary_path = rd / "estimates" / "summary.json"
    summary = read_json(summary_path)
    if summary is None:
        raise RuntimeError(f"no summary at {summary_path}")
    return summary


def read_verdict(config_path, run_id):
    return read_summary(config_path, run_id)["verdict"]


def engine_binary(engine_cmd):
    exe = shlex.split(engine_cmd)[0]
    path = shutil.which(exe)
    if path is None:
        candidate = Path(exe)
        if candidate.is_file():
            path = str(candidate.resolve())
        elif os.name == "nt":
            for suffix in (".exe", ".cmd", ".bat", ".com"):
                with_suffix = candidate.with_name(candidate.name + suffix)
                if with_suffix.is_file():
                    path = str(with_suffix.resolve())
                    break
    if path is None:
        raise RuntimeError(f"cannot find engine binary: {engine_cmd}")
    return path


def write_match_report(results_root, run_id, record, summary):
    report_dir = Path(results_root) / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "report.json", {"record": record, "summary": summary})
    accepted = record["accepted"]
    verdict_label = {
        "engine_a_better": "accepted (candidate is better)",
        "engine_b_better": "rejected (candidate improvement not established)",
        "inconclusive": "inconclusive",
        "continue": "continuing",
    }.get(record["verdict"], record["verdict"])
    elo = record["elo"]
    elo_text = f"{elo:+.1f}" if elo is not None else "n/a"
    score_rate = summary.get("score_rate")
    score_text = f"{100.0 * score_rate:.2f}%" if score_rate is not None else "n/a"
    pairs = summary.get("pairs", 0)
    games = summary.get("games", 0)
    llr = None
    if summary.get("sprt"):
        llr = summary["sprt"].get("llr")
    llr_text = f"{llr:.3f}" if llr is not None else "n/a"
    lines = [
        f"# Tune report: {record['param']} {record['old_value']} -> {record['new_value']}",
        "",
        f"- **Verdict**: {verdict_label}",
        f"- **Accepted**: {'Yes' if accepted else 'No'}",
        f"- **Elo (candidate - incumbent)**: {elo_text}",
        f"- **Score rate**: {score_text}",
        f"- **Pairs / games**: {pairs} / {games}",
        f"- **LLR**: {llr_text}",
        f"- **Time control**: {record['time_control']}",
        f"- **SPRT**: elo0={record['sprt_elo0']}, elo1={record['sprt_elo1']}, "
        f"alpha={record['sprt_alpha']}, beta={record['sprt_beta']}",
        f"- **Binary**: sha256 {record['binary_sha256']}",
        f"- **Time**: {record['timestamp']}",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def journal_match_count(journal_path):
    if not Path(journal_path).exists():
        return 0
    count = 0
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def resolve_time_control(common, journal_path):
    if common.get("time_control"):
        return common["time_control"]
    time_controls = common.get("time_controls") or []
    if not time_controls:
        raise RuntimeError(
            "tune.toml must define common.time_controls or common.time_control"
        )
    return time_controls[journal_match_count(journal_path) % len(time_controls)]


def run_single_match(cfg, best, params, name, value, engine_cmd, journal_path, workers, worker_multiplier):
    common = cfg["common"]
    sprt = cfg["sprt"]
    time_control = resolve_time_control(common, journal_path)
    tmp = tempfile.mkdtemp(prefix="auto-tune-")
    try:
        config = generate_match_config(
            cfg, common, sprt, name, value, best, params, engine_cmd, time_control, workers, worker_multiplier
        )
        config_path = Path(tmp) / "match.toml"
        write_toml_config(config_path, config)
        run_id = (
            f"tune-{name.lower()}-{value}-"
            f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        )
        run_match(config_path, run_id)
        summary = read_summary(config_path, run_id)
        verdict = summary["verdict"]
        binary = engine_binary(engine_cmd)
        old_value = value_for(best, params, name)
        accepted = verdict == "engine_a_better"
        elo = summary.get("elo")
        record = {
            "timestamp": now_utc(),
            "param": name,
            "old_value": old_value,
            "new_value": value,
            "verdict": verdict,
            "accepted": accepted,
            "elo": elo if elo is not None else None,
            "score_rate": summary.get("score_rate"),
            "pairs": summary.get("pairs", 0),
            "games": summary.get("games", 0),
            "llr": summary["sprt"].get("llr") if summary.get("sprt") else None,
            "binary_sha256": sha256_file(binary),
            "time_control": time_control,
            "sprt_elo0": sprt["elo0"],
            "sprt_elo1": sprt["elo1"],
            "sprt_alpha": sprt.get("alpha", 0.05),
            "sprt_beta": sprt.get("beta", 0.05),
        }
        append_journal(journal_path, record)
        write_match_report(cfg["results_dir"], run_id, record, summary)
        return summary, run_id, record
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def try_candidate(cfg, best, params, name, candidate, engine_cmd, journal_path, best_path, dry_run, workers, worker_multiplier):
    if not in_range(next(s for s in params if s["name"] == name), candidate):
        return False
    print(f"[tune] try {name}={candidate}")
    if dry_run:
        print(f"[tune] dry-run: would match {name}={candidate}")
        return False
    summary, _run_id, record = run_single_match(
        cfg, best, params, name, candidate, engine_cmd, journal_path, workers, worker_multiplier
    )
    verdict = summary["verdict"]
    print(f"[tune] verdict for {name}={candidate}: {verdict}")
    if verdict == "engine_a_better":
        best["values"][name] = candidate
        write_best(best_path, best)
        return True
    return False


def tune_parameter(cfg, best, params, spec, engine_cmd, journal_path, best_path, dry_run, workers, worker_multiplier):
    name = spec["name"]
    current = value_for(best, params, name)
    print(f"[tune] parameter {name}: current={current}")

    improved = True
    while improved:
        improved = False
        for step in [spec["step"], -spec["step"]]:
            candidate = current + step
            if try_candidate(cfg, best, params, name, candidate, engine_cmd, journal_path, best_path, dry_run, workers, worker_multiplier):
                current = candidate
                improved = True
                break
        if not improved:
            wider = current + 2 * spec["step"]
            if wider != current:
                if try_candidate(cfg, best, params, name, wider, engine_cmd, journal_path, best_path, dry_run, workers, worker_multiplier):
                    current = wider
                    improved = True
    print(f"[tune] parameter {name} settled at {current}")
    return current


def write_best(best_path, best):
    write_json(best_path, best)


def main():
    parser = argparse.ArgumentParser(description="Coordinate-descent SPRT tuner")
    parser.add_argument("--config", default=str(DEFAULT_TUNE_TOML))
    parser.add_argument("--best", default=str(DEFAULT_BEST_JSON))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument(
        "--engine",
        default="target/release/ember",
        help="path to the Ember release binary",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="comma-separated subset of parameter names to tune",
    )
    parser.add_argument(
        "--time-control",
        default=None,
        help="override the time control for all matches",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="number of parallel games per match (default: auto from CPU count)",
    )
    parser.add_argument(
        "--worker-multiplier",
        type=float,
        default=None,
        help="fraction of logical CPUs to use for workers (default: 1.0)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = read_toml(args.config)
    best = load_best(args.best)
    params = param_specs(cfg)
    if args.params:
        selected = [p.strip() for p in args.params.split(",")]
        params = [spec for spec in params if spec["name"] in selected]
    if args.time_control:
        cfg["common"]["time_control"] = args.time_control

    for spec in params:
        tune_parameter(
            cfg,
            best,
            params,
            spec,
            args.engine,
            args.journal,
            args.best,
            args.dry_run,
            args.workers,
            args.worker_multiplier,
        )

    if args.dry_run:
        print("[tune] dry-run: best.json and journal left untouched")
    else:
        write_best(args.best, best)
        print(f"[tune] best.json updated: {json.dumps(best['values'], sort_keys=True)}")


if __name__ == "__main__":
    main()

# python tools/auto_tune/seek.py --time-control 1+0.01
