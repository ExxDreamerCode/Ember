#!/usr/bin/env python3
"""Compare two Ember revisions on an identical seeded opponent schedule."""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

from head_to_head import sample_polyglot_openings


RESULT_SCORE = {
    "1-0": (1.0, 0.0),
    "0-1": (0.0, 1.0),
    "1/2-1/2": (0.5, 0.5),
}


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def shell_quote(value):
    if re.match(r"^[A-Za-z0-9_./:=+@,%^-]+$", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def load_config(path):
    cfg = read_toml(path)
    required = ["run", "build", "baseline", "candidate", "selection"]
    missing = [section for section in required if section not in cfg]
    if missing:
        raise ValueError(f"missing config sections: {', '.join(missing)}")
    if not cfg["run"].get("time_controls"):
        raise ValueError("run.time_controls must not be empty")
    if not cfg["selection"].get("weaker_opponents"):
        raise ValueError("selection.weaker_opponents must not be empty")
    if not cfg["selection"].get("stronger_opponents"):
        raise ValueError("selection.stronger_opponents must not be empty")
    return cfg


def make_run_id(cfg):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{safe_name(cfg['run'].get('name', 'version-opponents'))}"


def run_dir_for(cfg, run_id):
    return Path(cfg["run"].get("results_dir", "results/version-opponents")) / run_id


def run_logged(args, log_path, cwd=None, env=None, check=True):
    started = time.time()
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(shell_quote(str(arg)) for arg in args) + "\n")
        f.write(proc.stdout)
        if proc.stdout and not proc.stdout.endswith("\n"):
            f.write("\n")
        f.write(f"[exit={proc.returncode} elapsed={elapsed:.3f}s]\n\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}")
    return proc


def git_output(repo, *args):
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def resolve_revision(repo, revision):
    return git_output(repo, "rev-parse", f"{revision}^{{commit}}")


def build_one(cfg, rd, version_id, revision_override=None):
    build_cfg = cfg["build"]
    version_cfg = cfg[version_id]
    repo = Path(build_cfg.get("repo", ".")).resolve()
    revision = revision_override or version_cfg["revision"]
    resolved = resolve_revision(repo, revision)
    root = rd / "builds" / version_id
    metadata_path = root / "metadata.json"
    installed = root / "bin" / "ember"

    if metadata_path.exists() and installed.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("revision") == resolved and metadata.get("sha256") == sha256_file(installed):
            return metadata

    source = root / "source"
    archive = root / "source.tar"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)

    run_logged(
        ["git", "-C", str(repo), "archive", "--format=tar", f"--output={archive}", resolved],
        root / "archive.log",
    )
    run_logged(["tar", "-xf", str(archive), "-C", str(source)], root / "archive.log")

    command = list(build_cfg.get("command", ["cargo", "build", "--locked", "--release", "--bin", "ember"]))
    run_logged(command, root / "build.log", cwd=source)
    built = source / build_cfg.get("binary", "target/release/ember")
    if not built.is_file():
        raise RuntimeError(f"build did not produce {built}")
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, installed)
    installed.chmod(0o755)

    metadata = {
        "version_id": version_id,
        "configured_revision": revision,
        "revision": resolved,
        "binary": str(installed.resolve()),
        "sha256": sha256_file(installed),
        "built_at": now_utc(),
        "command": command,
    }
    write_json(metadata_path, metadata)
    archive.unlink(missing_ok=True)
    return metadata


def build_versions(cfg, rd, baseline_revision=None, candidate_revision=None):
    rd.mkdir(parents=True, exist_ok=True)
    builds = {
        "baseline": build_one(cfg, rd, "baseline", baseline_revision),
        "candidate": build_one(cfg, rd, "candidate", candidate_revision),
    }
    write_json(rd / "builds" / "metadata.json", builds)
    return builds


def probe_uci(binary, cwd=None, timeout=10):
    proc = subprocess.Popen(
        [str(binary)],
        cwd=cwd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        output, _ = proc.communicate("uci\nquit\n", timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
        raise RuntimeError(f"UCI probe timed out: {binary}")
    if "uciok" not in output:
        raise RuntimeError(f"UCI probe did not receive uciok: {binary}")
    return {
        "uci": True,
        "ponder": bool(re.search(r"^option name Ponder type check\b", output, re.MULTILINE | re.IGNORECASE)),
        "id_name": next(
            (line[len("id name ") :] for line in output.splitlines() if line.startswith("id name ")),
            "",
        ),
    }


def command_path(command):
    command = str(command)
    if "/" in command:
        path = Path(command).expanduser().resolve()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(command)


def load_opponents(path):
    data = read_toml(path)
    opponents = data.get("engine", [])
    by_name = {engine["name"]: engine for engine in opponents}
    if len(by_name) != len(opponents):
        raise RuntimeError(f"duplicate opponent names in {path}")
    return by_name


def opponent_availability(engine):
    path = command_path(engine["cmd"])
    if path is None:
        return False, f"command not found: {engine['cmd']}", None
    required_env = engine.get("required_env")
    if required_env:
        value = os.environ.get(required_env, "")
        if not value:
            return False, f"environment variable not set: {required_env}", path
        if required_env == "RYBKA4_EXE" and not Path(value).expanduser().is_file():
            return False, f"{required_env} does not name a file", path
    return True, "", path


def select_opponents(cfg, opponent_file):
    configured = load_opponents(opponent_file)
    selected = {"weaker": [], "stronger": []}
    unavailable = []
    seen = set()
    for band, key in (("weaker", "weaker_opponents"), ("stronger", "stronger_opponents")):
        for name in cfg["selection"][key]:
            if name in seen:
                raise RuntimeError(f"opponent appears in both pools or twice: {name}")
            seen.add(name)
            if name not in configured:
                raise RuntimeError(f"unknown opponent in {key}: {name}")
            engine = dict(configured[name])
            available, reason, path = opponent_availability(engine)
            if not available:
                unavailable.append({"name": name, "band": band, "reason": reason})
                continue
            engine["cmd"] = path
            engine["band"] = band
            selected[band].append(engine)
    for band in ("weaker", "stronger"):
        if not selected[band]:
            reasons = "; ".join(item["reason"] for item in unavailable if item["band"] == band)
            raise RuntimeError(f"no available {band} opponents ({reasons})")
    return selected, unavailable


def make_scenario_specs(opponents, time_controls, ponder_modes, count, seed, ember_can_ponder):
    if count < 1:
        raise ValueError("scenario count must be positive")
    rng = random.Random(seed)
    pools = {}
    for band in ("weaker", "stronger"):
        candidates = []
        for opponent in opponents[band]:
            for time_control in time_controls:
                for ponder in ponder_modes:
                    if ponder and not (ember_can_ponder and opponent.get("supports_ponder", False)):
                        continue
                    candidates.append((opponent, str(time_control), bool(ponder)))
        if not candidates:
            raise RuntimeError(f"no valid scenario combinations in {band} pool")
        rng.shuffle(candidates)
        pools[band] = {"original": candidates, "current": list(candidates), "index": 0}

    band_order = ["weaker", "stronger"]
    rng.shuffle(band_order)
    specs = []
    for index in range(count):
        band = band_order[index % len(band_order)]
        pool = pools[band]
        if pool["index"] >= len(pool["current"]):
            pool["current"] = list(pool["original"])
            rng.shuffle(pool["current"])
            pool["index"] = 0
        opponent, time_control, ponder = pool["current"][pool["index"]]
        pool["index"] += 1
        version_order = ["baseline", "candidate"]
        rng.shuffle(version_order)
        specs.append(
            {
                "id": f"scenario-{index + 1:04d}",
                "index": index + 1,
                "band": band,
                "opponent": dict(opponent),
                "time_control": time_control,
                "ponder": ponder,
                "version_order": version_order,
            }
        )
    return specs


def schedule_fingerprint(cfg, config_path, builds, opponent_file, scenario_count):
    run_cfg = cfg["run"]
    inputs = {
        "config_sha256": sha256_file(config_path),
        "opponents_sha256": sha256_file(opponent_file),
        "book_sha256": sha256_file(Path(run_cfg.get("polyglot_book", "src/book.bin"))),
        "baseline_revision": builds["baseline"]["revision"],
        "candidate_revision": builds["candidate"]["revision"],
        "scenario_count": scenario_count,
    }
    return stable_hash(inputs), inputs


def prepare_schedule(cfg, config_path, rd, scenario_count=None):
    builds_path = rd / "builds" / "metadata.json"
    if not builds_path.exists():
        raise RuntimeError("build metadata is missing; run the build phase first")
    builds = json.loads(builds_path.read_text(encoding="utf-8"))
    count = int(scenario_count or cfg["run"].get("scenario_count", 48))
    opponent_file = Path(cfg["selection"].get("opponent_file", "ratings/opponents.toml"))
    fingerprint, fingerprint_inputs = schedule_fingerprint(
        cfg, config_path, builds, opponent_file, count
    )
    manifest_path = rd / "schedule.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("input_fingerprint") != fingerprint:
            raise RuntimeError("existing schedule was generated from different inputs; use a new run id")
        return existing

    repo = Path(cfg["build"].get("repo", ".")).resolve()
    capabilities = {
        version_id: probe_uci(builds[version_id]["binary"], cwd=repo)
        for version_id in ("baseline", "candidate")
    }
    ember_can_ponder = all(capabilities[version_id]["ponder"] for version_id in capabilities)
    selected, unavailable = select_opponents(cfg, opponent_file)
    run_cfg = cfg["run"]
    ponder_modes = [bool(value) for value in run_cfg.get("ponder_modes", [False])]
    specs = make_scenario_specs(
        selected,
        run_cfg["time_controls"],
        ponder_modes,
        count,
        int(run_cfg.get("seed", 20260719)),
        ember_can_ponder,
    )
    openings = sample_polyglot_openings(cfg, count)
    openings_dir = rd / "openings"
    openings_dir.mkdir(parents=True, exist_ok=True)
    for spec, opening in zip(specs, openings):
        spec["opening_epd"] = opening
        opening_path = openings_dir / f"{spec['id']}.epd"
        opening_path.write_text(opening + "\n", encoding="utf-8")
        spec["opening_file"] = str(opening_path.resolve())

    effective_ponder_modes = sorted({spec["ponder"] for spec in specs})
    manifest = {
        "schema_version": 1,
        "created_at": now_utc(),
        "input_fingerprint": fingerprint,
        "fingerprint_inputs": fingerprint_inputs,
        "seed": int(run_cfg.get("seed", 20260719)),
        "scenario_count": count,
        "games_per_version": count * 2,
        "total_games": count * 4,
        "versions": {
            version_id: {
                "name": cfg[version_id]["name"],
                "revision": builds[version_id]["revision"],
                "binary": builds[version_id]["binary"],
                "sha256": builds[version_id]["sha256"],
                "options": cfg[version_id].get("options", {}),
            }
            for version_id in ("baseline", "candidate")
        },
        "capabilities": capabilities,
        "requested_ponder_modes": ponder_modes,
        "effective_ponder_modes": effective_ponder_modes,
        "ponder_note": (
            "enabled only where both Ember revisions and the opponent advertise support"
            if True in effective_ponder_modes
            else "ponder-on scenarios omitted because at least one Ember revision does not advertise support"
        ),
        "unavailable_opponents": unavailable,
        "available_opponents": {
            band: [engine["name"] for engine in selected[band]]
            for band in ("weaker", "stronger")
        },
        "scenarios": specs,
    }
    write_json(manifest_path, manifest)
    return manifest


def engine_args(engine, ponder=False):
    args = [
        "-engine",
        f"name={engine['name']}",
        f"cmd={engine['cmd']}",
        f"proto={engine.get('proto', 'uci')}",
    ]
    if engine.get("dir"):
        args.append(f"dir={engine['dir']}")
    for arg in engine.get("args", []):
        args.append(f"arg={arg}")
    for key, value in engine.get("options", {}).items():
        args.append(f"option.{key}={value}")
    if ponder:
        args.append("ponder")
    return args


def count_games(path):
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.startswith("[Event "))


def match_command(cfg, manifest, rd, scenario, version_id):
    run_cfg = cfg["run"]
    version = manifest["versions"][version_id]
    work_root = (rd / "work" / scenario["id"] / version_id).resolve()
    ember_dir = work_root / "ember"
    opponent_dir = work_root / "opponent"
    ember_dir.mkdir(parents=True, exist_ok=True)
    opponent_dir.mkdir(parents=True, exist_ok=True)
    ember = {
        "name": version["name"],
        "cmd": version["binary"],
        "proto": "uci",
        "dir": str(ember_dir),
        "options": version.get("options", {}),
    }
    opponent = dict(scenario["opponent"])
    opponent["dir"] = str(opponent_dir)
    pgn = rd / "games" / version_id / f"{scenario['id']}.pgn"
    args = [str(run_cfg.get("cutechess_cmd", "cutechess-cli"))]
    args.extend(engine_args(ember, scenario["ponder"]))
    args.extend(engine_args(opponent, scenario["ponder"]))
    args.extend(
        [
            "-each",
            f"tc={scenario['time_control']}",
            f"timemargin={int(run_cfg.get('timemargin_ms', 50))}",
            "-openings",
            f"file={scenario['opening_file']}",
            f"format={run_cfg.get('opening_format', 'epd')}",
            "order=sequential",
            "policy=round",
            "-games",
            "2",
            "-rounds",
            "1",
            "-repeat",
            "-concurrency",
            "1",
            "-pgnout",
            str(pgn.resolve()),
            "-recover",
        ]
    )
    max_moves = int(run_cfg.get("max_moves", 0))
    if max_moves > 0:
        args.extend(["-maxmoves", str(max_moves)])
    return args, pgn


def run_scenario(cfg, manifest, rd, scenario):
    completed = []
    for version_id in scenario["version_order"]:
        args, pgn = match_command(cfg, manifest, rd, scenario, version_id)
        pgn.parent.mkdir(parents=True, exist_ok=True)
        games = count_games(pgn)
        if games == 2:
            completed.append({"version": version_id, "status": "already-complete"})
            continue
        if games != 0:
            pgn.unlink()
        log = rd / "logs" / version_id / f"{scenario['id']}.log"
        run_logged(args, log, cwd=Path(cfg["build"].get("repo", ".")).resolve())
        games = count_games(pgn)
        if games != 2:
            raise RuntimeError(
                f"{scenario['id']} {version_id} produced {games} games instead of 2"
            )
        completed.append({"version": version_id, "status": "played"})
    return {"scenario": scenario["id"], "versions": completed}


def detect_workers(cfg, explicit_workers=None):
    cores = os.cpu_count() or 1
    if explicit_workers is not None:
        return max(1, int(explicit_workers)), cores, "cli"
    configured = str(cfg["run"].get("workers", "auto"))
    if configured != "auto":
        return max(1, int(configured)), cores, "config"
    multiplier = float(cfg["run"].get("worker_multiplier", 0.5))
    return max(1, int(math.floor(cores * multiplier))), cores, "auto"


def run_schedule(cfg, rd, explicit_workers=None):
    manifest_path = rd / "schedule.json"
    if not manifest_path.exists():
        raise RuntimeError("schedule is missing; run the schedule phase first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workers, cores, source = detect_workers(cfg, explicit_workers)
    state = {
        "phase": "running",
        "started_at": now_utc(),
        "workers": workers,
        "cpu_count": cores,
        "worker_source": source,
        "scenario_count": len(manifest["scenarios"]),
    }
    write_json(rd / "state.json", state)
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_scenario, cfg, manifest, rd, scenario): scenario["id"]
            for scenario in manifest["scenarios"]
        }
        for future in as_completed(futures):
            scenario_id = futures[future]
            try:
                result = future.result()
                statuses = ", ".join(
                    f"{item['version']}={item['status']}" for item in result["versions"]
                )
                print(f"[{scenario_id}] {statuses}", flush=True)
            except Exception as error:  # keep all independent matches running
                failures.append({"scenario": scenario_id, "error": str(error)})
                print(f"[{scenario_id}] FAILED: {error}", file=sys.stderr, flush=True)
    if failures:
        state.update({"phase": "failed", "finished_at": now_utc(), "failures": failures})
        write_json(rd / "state.json", state)
        raise RuntimeError(f"{len(failures)} scenarios failed; rerun to resume")
    state.update({"phase": "games-complete", "finished_at": now_utc()})
    write_json(rd / "state.json", state)
    return manifest


def parse_pgn_headers(path):
    games = []
    tags = {}
    order = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("[Event ") and tags:
                order += 1
                games.append({**tags, "pgn": str(path), "order": order})
                tags = {}
            if line.startswith("[") and line.endswith("]"):
                match = re.match(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$', line)
                if match:
                    tags[match.group(1)] = match.group(2)
    if tags:
        order += 1
        games.append({**tags, "pgn": str(path), "order": order})
    return [game for game in games if game.get("Result") in RESULT_SCORE]


def outcome_for_game(game, ember_name):
    result = game["Result"]
    white_score, black_score = RESULT_SCORE[result]
    if game.get("White") == ember_name:
        color = "white"
        score = white_score
    elif game.get("Black") == ember_name:
        color = "black"
        score = black_score
    else:
        raise RuntimeError(f"{ember_name} is absent from {game['pgn']} game {game['order']}")
    outcome = {1.0: "win", 0.5: "draw", 0.0: "loss"}[score]
    return {
        "ember_color": color,
        "score": score,
        "outcome": outcome,
        "result": result,
        "termination": game.get("Termination", ""),
        "pgn": game["pgn"],
        "pgn_game": game["order"],
    }


def collect_outcomes(manifest, rd, version_id):
    ember_name = manifest["versions"][version_id]["name"]
    outcomes = {}
    for scenario in manifest["scenarios"]:
        path = rd / "games" / version_id / f"{scenario['id']}.pgn"
        games = parse_pgn_headers(path)
        if len(games) != 2:
            raise RuntimeError(f"expected 2 games in {path}, found {len(games)}")
        for game in games:
            outcome = outcome_for_game(game, ember_name)
            key = (scenario["id"], outcome["ember_color"])
            if key in outcomes:
                raise RuntimeError(f"duplicate Ember color in {path}: {outcome['ember_color']}")
            outcomes[key] = outcome
    return outcomes


def compare_outcome_records(manifest, baseline, candidate):
    rows = []
    for scenario in manifest["scenarios"]:
        for color in ("white", "black"):
            key = (scenario["id"], color)
            if key not in baseline or key not in candidate:
                raise RuntimeError(f"missing aligned game: {scenario['id']} {color}")
            before = baseline[key]
            after = candidate[key]
            delta = after["score"] - before["score"]
            rows.append(
                {
                    "scenario": scenario["id"],
                    "band": scenario["band"],
                    "opponent": scenario["opponent"]["name"],
                    "time_control": scenario["time_control"],
                    "ponder": scenario["ponder"],
                    "opening_epd": scenario["opening_epd"],
                    "ember_color": color,
                    "baseline_outcome": before["outcome"],
                    "candidate_outcome": after["outcome"],
                    "baseline_score": before["score"],
                    "candidate_score": after["score"],
                    "score_delta": delta,
                    "change": "regression" if delta < 0 else "improvement" if delta > 0 else "unchanged",
                    "transition": f"{before['outcome']} -> {after['outcome']}",
                    "baseline_termination": before["termination"],
                    "candidate_termination": after["termination"],
                    "baseline_pgn": before["pgn"],
                    "baseline_pgn_game": before["pgn_game"],
                    "candidate_pgn": after["pgn"],
                    "candidate_pgn_game": after["pgn_game"],
                }
            )
    return rows


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def score_to_elo(score_rate):
    clipped = min(1.0 - 1e-6, max(1e-6, score_rate))
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def clustered_samples(rows):
    scenarios = {}
    for row in rows:
        item = scenarios.setdefault(
            row["scenario"], {"baseline_score": 0.0, "candidate_score": 0.0, "games": 0}
        )
        item["baseline_score"] += row["baseline_score"]
        item["candidate_score"] += row["candidate_score"]
        item["games"] += 1
    return [scenarios[key] for key in sorted(scenarios)]


def bootstrap_comparison(rows, seed, samples):
    clusters = clustered_samples(rows)
    if not clusters or samples < 1:
        return {}
    rng = random.Random(seed ^ 0xB00757A9)
    rate_deltas = []
    elo_deltas = []
    for _ in range(samples):
        picked = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        games = sum(item["games"] for item in picked)
        baseline_rate = sum(item["baseline_score"] for item in picked) / games
        candidate_rate = sum(item["candidate_score"] for item in picked) / games
        rate_deltas.append(candidate_rate - baseline_rate)
        elo_deltas.append(score_to_elo(candidate_rate) - score_to_elo(baseline_rate))
    return {
        "samples": samples,
        "score_rate_delta_ci95": [
            percentile(rate_deltas, 0.025),
            percentile(rate_deltas, 0.975),
        ],
        "matched_elo_delta_ci95": [
            percentile(elo_deltas, 0.025),
            percentile(elo_deltas, 0.975),
        ],
    }


def paired_randomization_p(rows, seed, samples):
    deltas = [item["candidate_score"] - item["baseline_score"] for item in clustered_samples(rows)]
    if not deltas:
        return None
    observed = abs(sum(deltas))
    if len(deltas) <= 18:
        total = 1 << len(deltas)
        extreme = 0
        for mask in range(total):
            value = sum(delta if mask & (1 << index) else -delta for index, delta in enumerate(deltas))
            if abs(value) >= observed - 1e-12:
                extreme += 1
        return extreme / total
    rng = random.Random(seed ^ 0x51A9F11F)
    extreme = 0
    for _ in range(samples):
        value = sum(delta if rng.getrandbits(1) else -delta for delta in deltas)
        if abs(value) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def wdl(rows, prefix):
    return {
        outcome: sum(1 for row in rows if row[f"{prefix}_outcome"] == outcome)
        for outcome in ("win", "draw", "loss")
    }


def is_time_forfeit(termination):
    lowered = termination.lower()
    return "time" in lowered or "flag" in lowered


def summarize(rows, seed, bootstrap_samples, randomization_samples):
    games = len(rows)
    baseline_score = sum(row["baseline_score"] for row in rows)
    candidate_score = sum(row["candidate_score"] for row in rows)
    baseline_rate = baseline_score / games
    candidate_rate = candidate_score / games
    transitions = {}
    for row in rows:
        transitions[row["transition"]] = transitions.get(row["transition"], 0) + 1
    summary = {
        "scenarios": len({row["scenario"] for row in rows}),
        "games_per_version": games,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "baseline_score_rate": baseline_rate,
        "candidate_score_rate": candidate_rate,
        "score_rate_delta": candidate_rate - baseline_rate,
        "matched_elo_delta": score_to_elo(candidate_rate) - score_to_elo(baseline_rate),
        "baseline_wdl": wdl(rows, "baseline"),
        "candidate_wdl": wdl(rows, "candidate"),
        "regressions": sum(row["change"] == "regression" for row in rows),
        "improvements": sum(row["change"] == "improvement" for row in rows),
        "unchanged": sum(row["change"] == "unchanged" for row in rows),
        "transitions": dict(sorted(transitions.items())),
        "baseline_time_forfeit_losses": sum(
            row["baseline_outcome"] == "loss" and is_time_forfeit(row["baseline_termination"])
            for row in rows
        ),
        "candidate_time_forfeit_losses": sum(
            row["candidate_outcome"] == "loss" and is_time_forfeit(row["candidate_termination"])
            for row in rows
        ),
        "paired_randomization_p_two_sided": paired_randomization_p(
            rows, seed, randomization_samples
        ),
        "breakdowns": make_breakdowns(
            rows, seed, bootstrap_samples, randomization_samples
        ),
    }
    summary.update(bootstrap_comparison(rows, seed, bootstrap_samples))
    return summary


def make_breakdowns(rows, seed, bootstrap_samples, randomization_samples):
    breakdowns = []
    dimensions = (
        ("band", lambda row: row["band"]),
        ("opponent", lambda row: row["opponent"]),
        ("time_control", lambda row: row["time_control"]),
        ("ponder", lambda row: str(row["ponder"]).lower()),
    )
    for dimension, getter in dimensions:
        values = sorted({getter(row) for row in rows})
        for value in values:
            group = [row for row in rows if getter(row) == value]
            games = len(group)
            baseline_score = sum(row["baseline_score"] for row in group)
            candidate_score = sum(row["candidate_score"] for row in group)
            baseline_rate = baseline_score / games
            candidate_rate = candidate_score / games
            group_seed = seed ^ int(
                hashlib.sha256(f"{dimension}\0{value}".encode("utf-8")).hexdigest()[:16],
                16,
            )
            intervals = bootstrap_comparison(group, group_seed, bootstrap_samples)
            score_ci = intervals.get("score_rate_delta_ci95", [None, None])
            elo_ci = intervals.get("matched_elo_delta_ci95", [None, None])
            breakdowns.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "games_per_version": games,
                    "baseline_score_rate": baseline_rate,
                    "candidate_score_rate": candidate_rate,
                    "score_rate_delta": candidate_rate - baseline_rate,
                    "score_rate_delta_ci95_low": score_ci[0],
                    "score_rate_delta_ci95_high": score_ci[1],
                    "matched_elo_delta": score_to_elo(candidate_rate) - score_to_elo(baseline_rate),
                    "matched_elo_delta_ci95_low": elo_ci[0],
                    "matched_elo_delta_ci95_high": elo_ci[1],
                    "paired_randomization_p_two_sided": paired_randomization_p(
                        group, group_seed, randomization_samples
                    ),
                    "regressions": sum(row["change"] == "regression" for row in group),
                    "improvements": sum(row["change"] == "improvement" for row in group),
                }
            )
    return breakdowns


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_result_path(path, rd):
    try:
        return str(Path(path).resolve().relative_to(rd.resolve()))
    except ValueError:
        return str(path)


def write_changed_artifacts(rd, rows):
    changed = [row for row in rows if row["change"] != "unchanged"]
    changed.sort(
        key=lambda row: (
            row["change"] != "regression",
            row["score_delta"],
            row["scenario"],
            row["ember_color"],
        )
    )
    write_csv(rd / "estimates" / "changed-outcomes.csv", changed)
    write_json(rd / "estimates" / "changed-outcomes.json", changed)

    lines = [
        "# Changed outcomes",
        "",
        "Regressions are listed before improvements. Each row links the exact paired PGNs for deeper analysis.",
        "",
        "| Change | Scenario | Opponent | TC | Ponder | Color | Outcome | Baseline PGN | Candidate PGN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in changed:
        baseline_path = relative_result_path(row["baseline_pgn"], rd)
        candidate_path = relative_result_path(row["candidate_pgn"], rd)
        lines.append(
            f"| {row['change']} | {row['scenario']} | {row['opponent']} | "
            f"{row['time_control']} | {str(row['ponder']).lower()} | {row['ember_color']} | "
            f"{row['transition']} | [{row['baseline_pgn_game']}](../{baseline_path}) | "
            f"[{row['candidate_pgn_game']}](../{candidate_path}) |"
        )
    (rd / "estimates" / "changed-outcomes.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    pgn_paths = []
    seen = set()
    for row in changed:
        for key in ("baseline_pgn", "candidate_pgn"):
            path = row[key]
            if path not in seen:
                seen.add(path)
                pgn_paths.append(path)
    with open(rd / "estimates" / "changed-games.pgn", "w", encoding="utf-8") as output:
        for path in pgn_paths:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            output.write(text.rstrip() + "\n\n")


def format_ci(values, scale=1.0, digits=2):
    if not values or values[0] is None:
        return "n/a"
    return f"{values[0] * scale:.{digits}f} to {values[1] * scale:.{digits}f}"


def write_report(rd, manifest, summary):
    baseline = manifest["versions"]["baseline"]
    candidate = manifest["versions"]["candidate"]
    skipped = manifest.get("unavailable_opponents", [])
    skipped_text = ", ".join(f"{item['name']} ({item['reason']})" for item in skipped) or "none"
    lines = [
        "# Cross-version opponent comparison",
        "",
        f"Baseline: **{baseline['name']}** `{baseline['revision']}`",
        f"Candidate: **{candidate['name']}** `{candidate['revision']}`",
        f"Seed: `{manifest['seed']}`",
        f"Schedule: {manifest['scenario_count']} scenarios, two color-swapped games per version and scenario.",
        f"Pondering: {manifest['ponder_note']}.",
        f"Unavailable configured opponents: {skipped_text}.",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Score | {summary['baseline_score']:.1f}/{summary['games_per_version']} | "
        f"{summary['candidate_score']:.1f}/{summary['games_per_version']} | "
        f"{summary['candidate_score'] - summary['baseline_score']:+.1f} |",
        f"| Score rate | {100 * summary['baseline_score_rate']:.2f}% | "
        f"{100 * summary['candidate_score_rate']:.2f}% | "
        f"{100 * summary['score_rate_delta']:+.2f} pp |",
        f"| Win/draw/loss | {summary['baseline_wdl']['win']}/{summary['baseline_wdl']['draw']}/{summary['baseline_wdl']['loss']} | "
        f"{summary['candidate_wdl']['win']}/{summary['candidate_wdl']['draw']}/{summary['candidate_wdl']['loss']} | |",
        f"| Time-forfeit losses | {summary['baseline_time_forfeit_losses']} | "
        f"{summary['candidate_time_forfeit_losses']} | "
        f"{summary['candidate_time_forfeit_losses'] - summary['baseline_time_forfeit_losses']:+d} |",
        "",
        f"Matched-panel Elo delta (candidate - baseline): **{summary['matched_elo_delta']:+.1f}**",
        f"Clustered 95% Elo interval: **{format_ci(summary.get('matched_elo_delta_ci95'), digits=1)}**",
        f"Clustered 95% score-rate delta: **{format_ci(summary.get('score_rate_delta_ci95'), 100.0)} percentage points**",
        f"Paired randomization p-value, two-sided: **{summary['paired_randomization_p_two_sided']:.6g}**",
        "",
        f"Changed outcomes: {summary['regressions']} regressions, {summary['improvements']} improvements, "
        f"{summary['unchanged']} unchanged.",
        "See [changed-outcomes.md](estimates/changed-outcomes.md) for the exact games.",
        "",
        "The Elo delta is a matched result on this seeded mixture of opponents, clocks, openings, and pondering modes. It is not a standalone CCRL rating.",
        "",
    ]
    (rd / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(cfg, rd):
    manifest = json.loads((rd / "schedule.json").read_text(encoding="utf-8"))
    baseline = collect_outcomes(manifest, rd, "baseline")
    candidate = collect_outcomes(manifest, rd, "candidate")
    rows = compare_outcome_records(manifest, baseline, candidate)
    run_cfg = cfg["run"]
    summary = summarize(
        rows,
        int(manifest["seed"]),
        int(run_cfg.get("bootstrap_samples", 20000)),
        int(run_cfg.get("randomization_samples", 50000)),
    )
    write_csv(rd / "estimates" / "game-outcomes.csv", rows)
    write_csv(rd / "estimates" / "breakdowns.csv", summary["breakdowns"])
    write_json(rd / "estimates" / "comparison.json", summary)
    write_changed_artifacts(rd, rows)
    write_report(rd, manifest, summary)
    state = {
        "phase": "finished",
        "finished_at": now_utc(),
        "summary": summary,
    }
    write_json(rd / "state.json", state)
    return summary


def apply_overrides(cfg, args):
    if args.baseline_revision:
        cfg["baseline"]["revision"] = args.baseline_revision
    if args.candidate_revision:
        cfg["candidate"]["revision"] = args.candidate_revision
    if args.scenarios is not None:
        cfg["run"]["scenario_count"] = int(args.scenarios)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Compare two Ember revisions against one deterministic mixed opponent schedule"
    )
    parser.add_argument("command", choices=["build", "schedule", "run", "analyze", "all"])
    parser.add_argument("--config", default="configs/version-opponents/default.toml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--baseline-revision", default=None)
    parser.add_argument("--candidate-revision", default=None)
    parser.add_argument("--scenarios", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = apply_overrides(load_config(config_path), args)
    run_id = args.run_id or make_run_id(cfg)
    rd = run_dir_for(cfg, run_id)
    rd.mkdir(parents=True, exist_ok=True)

    if args.command in {"build", "all"}:
        build_versions(
            cfg,
            rd,
            baseline_revision=cfg["baseline"]["revision"],
            candidate_revision=cfg["candidate"]["revision"],
        )
    if args.command in {"schedule", "all"}:
        prepare_schedule(cfg, config_path, rd, cfg["run"].get("scenario_count"))
    if args.command in {"run", "all"}:
        run_schedule(cfg, rd, args.workers)
    if args.command in {"analyze", "all"}:
        summary = analyze(cfg, rd)
        print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"results: {rd}")


if __name__ == "__main__":
    main()
