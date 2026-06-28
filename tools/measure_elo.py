#!/usr/bin/env python3
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
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


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


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_exists(cmd):
    return shutil.which(cmd) is not None


def run_cmd(args, log_path=None, cwd=None, check=True, timeout=None, env=None):
    start = time.time()
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.time() - start
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("$ " + " ".join(shell_quote(a) for a in args) + "\n")
            f.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                f.write("\n")
            f.write(f"[exit={proc.returncode} elapsed={elapsed:.3f}s]\n\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}")
    return proc


def shell_quote(s):
    if re.match(r"^[A-Za-z0-9_./:=+@,%^-]+$", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def load_config(config_path):
    cfg = read_toml(config_path)
    opponent_file = Path(cfg["selection"]["opponent_file"])
    opponents = read_toml(opponent_file).get("engine", [])
    return cfg, opponents


def make_run_id():
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    suffix = git.stdout.strip() if git.returncode == 0 else "nogit"
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if dirty.returncode == 0 and dirty.stdout.strip():
        suffix += "-dirty"
    return f"{stamp}-{suffix}"


def run_dir_for(run_id):
    return Path("results") / run_id


def detect_workers(cfg, explicit_workers):
    cores = os.cpu_count() or 1
    if explicit_workers:
        return max(1, int(explicit_workers)), cores, "cli"
    configured = str(cfg["run"].get("workers", "auto"))
    if configured != "auto":
        return max(1, int(configured)), cores, "config"
    multiplier = float(cfg["run"].get("worker_multiplier", 1.5))
    workers = max(1, int(math.ceil(cores * multiplier)))
    return workers, cores, "auto"


def effective_max_games(cfg, explicit_max_games):
    if explicit_max_games is not None:
        return max(2, int(explicit_max_games)), "cli"
    return max(2, int(cfg["run"].get("max_games", 10_000))), "config"


def metadata_base(
    cfg,
    config_path,
    run_id,
    workers,
    cores,
    worker_source,
    explicit_max_games=None,
):
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    nix = subprocess.run(
        ["nix", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    env_commit = os.environ.get("EMBER_ELO_GIT_COMMIT") or None
    env_dirty = os.environ.get("EMBER_ELO_GIT_DIRTY")
    max_games, max_games_source = effective_max_games(cfg, explicit_max_games)
    return {
        "run_id": run_id,
        "started_at": now_utc(),
        "hostname": socket.gethostname(),
        "cpu_count": cores,
        "workers": workers,
        "worker_source": worker_source,
        "max_games": max_games,
        "max_games_source": max_games_source,
        "measurement_mode": cfg["run"].get("mode", "mixed-prior"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "git_commit": env_commit or (git_commit.stdout.strip() if git_commit.returncode == 0 else None),
        "git_dirty": (env_dirty == "true") if env_dirty is not None else (bool(dirty.stdout.strip()) if dirty.returncode == 0 else None),
        "nix_version": nix.stdout.strip() if nix.returncode == 0 else None,
        "time_control": cfg["run"]["time_control"],
        "ember_book": cfg["ember"].get("book", "<embedded>"),
        "ember_book_name": cfg["ember"].get("book_name", cfg["ember"].get("book", "<embedded>")),
    }


def probe(config_path, run_id, explicit_workers=None, explicit_max_games=None):
    cfg, opponents = load_config(config_path)
    workers, cores, worker_source = detect_workers(cfg, explicit_workers)
    rd = run_dir_for(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    meta = metadata_base(
        cfg,
        config_path,
        run_id,
        workers,
        cores,
        worker_source,
        explicit_max_games,
    )

    tools = {}
    for cmd in ["cargo", "rustc", "cutechess-cli", "python3", "tar", "zstd"]:
        tools[cmd] = {
            "path": shutil.which(cmd),
            "available": command_exists(cmd),
        }

    engine_probe = []
    missing_required = []
    for engine in opponents:
        available = command_exists(engine["cmd"])
        item = dict(engine)
        item["available"] = available
        item["path"] = shutil.which(engine["cmd"])
        engine_probe.append(item)
        if engine.get("required", False) and not available:
            missing_required.append(engine["name"])

    if cfg["run"].get("mode") == "stockfish-adaptive":
        sf_cfg = cfg.get("stockfish_adaptive", {})
        sf_cmd = sf_cfg.get("cmd", "stockfish")
        available = command_exists(sf_cmd)
        engine_probe.append({
            "name": "Stockfish adaptive",
            "cmd": sf_cmd,
            "available": available,
            "path": shutil.which(sf_cmd),
        })
        if not available:
            missing_required.append("Stockfish adaptive")

    meta["tools"] = tools
    meta["opponents"] = engine_probe
    meta["missing_required_opponents"] = missing_required
    write_json(rd / "metadata.json", meta)
    write_json(rd / "state.json", {"run_id": run_id, "phase": "probe", "metadata": meta})

    print(json.dumps({
        "run_id": run_id,
        "workers": workers,
        "cpu_count": cores,
        "missing_required_opponents": missing_required,
    }, indent=2))
    if missing_required:
        raise SystemExit(2)


def build(config_path, run_id):
    cfg, _ = load_config(config_path)
    rd = run_dir_for(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    run_cmd(["cargo", "build", "--release", "--bin", "ember"], log_path=rd / "build.log")
    ember_bin = Path(cfg["ember"]["binary"])
    if not ember_bin.exists():
        raise RuntimeError(f"Ember binary was not created: {ember_bin}")
    meta_path = rd / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["ember_binary"] = str(ember_bin)
    meta["ember_binary_sha256"] = sha256_file(ember_bin)
    meta["built_at"] = now_utc()
    write_json(meta_path, meta)


def smoke(config_path, run_id):
    cfg, _ = load_config(config_path)
    rd = run_dir_for(run_id)
    ember_bin = Path(cfg["ember"]["binary"])
    cmds = [
        "uci",
        "isready",
        f"setoption name Hash value {cfg['ember']['options'].get('Hash', '64')}",
        f"setoption name Threads value {cfg['ember']['options'].get('Threads', '1')}",
        f"setoption name Book value {cfg['ember'].get('book', '<embedded>')}",
        "ucinewgame",
        "position startpos",
        "go movetime 100",
        "quit",
    ]
    proc = subprocess.run(
        [str(ember_bin)],
        input="\n".join(cmds) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    log = rd / "smoke.log"
    log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 or "uciok" not in proc.stdout or "readyok" not in proc.stdout or "bestmove" not in proc.stdout:
        raise RuntimeError("UCI smoke check failed; see smoke.log")
    meta_path = rd / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["smoke_ok"] = True
    meta["smoke_at"] = now_utc()
    write_json(meta_path, meta)


def select_opponents(opponents, cfg):
    include_roles = set(cfg["selection"].get("include_roles", ["final", "bracket"]))
    selected = []
    for engine in opponents:
        if engine.get("role") not in include_roles:
            continue
        if not command_exists(engine["cmd"]):
            if engine.get("required", False):
                raise RuntimeError(f"required opponent missing: {engine['name']} ({engine['cmd']})")
            continue
        selected.append(engine)
    selected.sort(key=lambda e: (e.get("role") != "final", int(e.get("rating", 0))))
    return selected


def engine_args(engine, ember_bin=None):
    args = ["-engine", f"name={engine['name']}", f"cmd={ember_bin or engine['cmd']}", f"proto={engine['proto']}"]
    for key, value in engine.get("options", {}).items():
        args.append(f"option.{key}={value}")
    return args


def ember_engine(cfg):
    return {
        "name": cfg["ember"]["name"],
        "cmd": cfg["ember"]["binary"],
        "proto": cfg["ember"]["proto"],
        "options": cfg["ember"].get("options", {}),
        "family": "ember",
    }


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def expected_games(rounds):
    return max(1, int(rounds)) * 2


def count_games_in_pgn(path):
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[Event "):
                count += 1
    return count


def run_match(cfg, rd, whiteish, blackish, rounds, workers, pgn_name):
    games_dir = rd / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    pgn = games_dir / pgn_name
    log = games_dir / (pgn_name + ".cutechess.log")
    want = expected_games(rounds)
    before = count_games_in_pgn(pgn)
    if before >= want:
        with open(rd / "commands.log", "a", encoding="utf-8") as f:
            f.write(f"skip existing {pgn} ({want} games)\n")
        return 0

    opening_file = cfg["run"]["opening_file"]
    args = ["cutechess-cli"]
    args.extend(engine_args(whiteish, ember_bin=cfg["ember"]["binary"] if whiteish["name"] == cfg["ember"]["name"] else None))
    args.extend(engine_args(blackish, ember_bin=cfg["ember"]["binary"] if blackish["name"] == cfg["ember"]["name"] else None))
    args.extend([
        "-each",
        f"tc={cfg['run']['time_control']}",
        f"timemargin={int(cfg['run'].get('timemargin_ms', 2000))}",
        "-openings",
        f"file={opening_file}",
        f"format={cfg['run'].get('opening_format', 'epd')}",
        "order=random",
        "-games",
        "2",
        "-rounds",
        str(int(rounds)),
        "-repeat",
        "-concurrency",
        str(workers),
        "-pgnout",
        str(pgn),
        "-recover",
        "-ratinginterval",
        "10",
    ])
    run_cmd(args, log_path=log, check=True)
    with open(rd / "commands.log", "a", encoding="utf-8") as f:
        f.write(" ".join(shell_quote(a) for a in args) + "\n")
    after = count_games_in_pgn(pgn)
    return max(0, min(after, want) - before)


def run_matches(config_path, run_id, explicit_workers=None, explicit_max_games=None):
    cfg, opponents = load_config(config_path)
    if cfg["run"].get("mode") == "stockfish-adaptive":
        run_stockfish_adaptive_matches(config_path, run_id, explicit_workers, explicit_max_games)
        return

    rd = run_dir_for(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    workers, cores, worker_source = detect_workers(cfg, explicit_workers)
    selected = select_opponents(opponents, cfg)
    if len(selected) < int(cfg["selection"].get("min_final_opponents", 1)):
        print(f"warning: selected only {len(selected)} opponents", file=sys.stderr)

    ember = ember_engine(cfg)
    state = {
        "run_id": run_id,
        "phase": "run",
        "started_at": now_utc(),
        "workers": workers,
        "cpu_count": cores,
        "selected_opponents": [e["name"] for e in selected],
    }
    write_json(rd / "state.json", state)

    rounds = int(cfg["run"].get("rounds_per_pair", 10))
    cal_rounds = int(cfg["run"].get("calibration_rounds_per_pair", 4))
    max_games, _ = effective_max_games(cfg, explicit_max_games)
    scheduled_games = 0

    for opponent in selected:
        if scheduled_games + expected_games(rounds) > max_games:
            break
        scheduled_games += run_match(
            cfg,
            rd,
            ember,
            opponent,
            rounds,
            workers,
            f"{safe_name(ember['name'])}-vs-{safe_name(opponent['name'])}.pgn",
        )

    by_rating = sorted(selected, key=lambda e: int(e.get("rating", 0)))
    seen = set()
    for idx, a in enumerate(by_rating):
        for b in by_rating[idx + 1:idx + 3]:
            key = tuple(sorted([a["name"], b["name"]]))
            if key in seen:
                continue
            seen.add(key)
            if scheduled_games + expected_games(cal_rounds) > max_games:
                break
            scheduled_games += run_match(
                cfg,
                rd,
                a,
                b,
                cal_rounds,
                workers,
                f"{safe_name(a['name'])}-vs-{safe_name(b['name'])}.pgn",
            )

    state["finished_at"] = now_utc()
    state["scheduled_games"] = scheduled_games
    state["max_games"] = max_games
    write_json(rd / "state.json", state)


def stockfish_engine(cfg, level):
    sf_cfg = cfg.get("stockfish_adaptive", {})
    options = dict(sf_cfg.get("options", {}))
    options["UCI_LimitStrength"] = "true"
    options["UCI_Elo"] = str(int(level))
    options.setdefault("Threads", "1")
    options.setdefault("Hash", "64")
    return {
        "name": f"Stockfish-UCI-{int(level)}",
        "cmd": sf_cfg.get("cmd", "stockfish"),
        "proto": sf_cfg.get("proto", "uci"),
        "options": options,
        "family": "stockfish-limited",
        "rating": int(level),
    }


def stockfish_level_from_name(name):
    match = re.search(r"Stockfish-UCI-(\d+)", name)
    return int(match.group(1)) if match else None


def ember_observations_vs_stockfish(games, ember_name):
    observations = []
    for game in games:
        white_level = stockfish_level_from_name(game["white"])
        black_level = stockfish_level_from_name(game["black"])
        if game["white"] == ember_name and black_level is not None:
            observations.append((black_level, game["white_score"]))
        elif game["black"] == ember_name and white_level is not None:
            observations.append((white_level, game["black_score"]))
    return observations


def fit_stockfish_equivalent(cfg, observations):
    if not observations:
        return {
            "rating": None,
            "ci95_low": None,
            "ci95_high": None,
            "standard_error": None,
            "observations": 0,
        }

    levels = [level for level, _ in observations]
    sf_cfg = cfg.get("stockfish_adaptive", {})
    prior = float(sf_cfg.get("rating_prior", sum(levels) / len(levels)))
    prior_sigma = float(sf_cfg.get("rating_prior_sigma", 1000))
    rating = prior
    c = math.log(10.0) / 400.0

    for _ in range(80):
        grad = (rating - prior) / (prior_sigma * prior_sigma)
        hess = 1.0 / (prior_sigma * prior_sigma)
        for level, score in observations:
            p = 1.0 / (1.0 + math.exp(-c * (rating - level)))
            grad += c * (p - score)
            hess += c * c * p * (1.0 - p)
        if hess <= 0.0:
            break
        step = grad / hess
        rating -= step
        if abs(step) < 1e-5:
            break

    hess = 1.0 / (prior_sigma * prior_sigma)
    for level, _ in observations:
        p = 1.0 / (1.0 + math.exp(-c * (rating - level)))
        hess += c * c * p * (1.0 - p)
    se = math.sqrt(1.0 / hess) if hess > 0.0 else None
    return {
        "rating": rating,
        "ci95_low": rating - 1.96 * se if se is not None else None,
        "ci95_high": rating + 1.96 * se if se is not None else None,
        "standard_error": se,
        "observations": len(observations),
    }


def clamp_level(cfg, level):
    sf_cfg = cfg.get("stockfish_adaptive", {})
    lo = int(sf_cfg.get("uci_elo_min", 1320))
    hi = int(sf_cfg.get("uci_elo_max", 3190))
    step = max(1, int(sf_cfg.get("target_step", 10)))
    rounded = int(round(level / step) * step)
    return max(lo, min(hi, rounded))


def run_stockfish_adaptive_matches(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_games=None,
):
    cfg, _ = load_config(config_path)
    rd = run_dir_for(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    workers, cores, _ = detect_workers(cfg, explicit_workers)
    max_games, max_games_source = effective_max_games(cfg, explicit_max_games)
    ember = ember_engine(cfg)
    sf_cfg = cfg.get("stockfish_adaptive", {})
    levels = [int(level) for level in sf_cfg.get("levels", [2500, 2600, 2750, 2850, 2950])]
    if not levels:
        raise RuntimeError("stockfish adaptive mode needs at least one level")

    state = {
        "run_id": run_id,
        "phase": "run",
        "mode": "stockfish-adaptive",
        "started_at": now_utc(),
        "workers": workers,
        "cpu_count": cores,
        "max_games": max_games,
        "max_games_source": max_games_source,
        "pilot_levels": levels,
    }
    write_json(rd / "state.json", state)

    scheduled_games = 0
    pilot_games_per_level = int(sf_cfg.get("pilot_games_per_level", 24))
    pilot_games_per_level = max(2, pilot_games_per_level - (pilot_games_per_level % 2))
    if max_games < pilot_games_per_level * len(levels):
        pilot_games_per_level = max(2, (max_games // max(1, len(levels))) // 2 * 2)

    for level in levels:
        if scheduled_games + 2 > max_games:
            break
        games_for_level = min(pilot_games_per_level, max_games - scheduled_games)
        games_for_level -= games_for_level % 2
        if games_for_level < 2:
            break
        scheduled_games += run_match(
            cfg,
            rd,
            ember,
            stockfish_engine(cfg, level),
            games_for_level // 2,
            workers,
            f"{safe_name(ember['name'])}-vs-Stockfish-UCI-{level}.pgn",
        )

    pilot_games = parse_all_games(rd)
    pilot_fit = fit_stockfish_equivalent(
        cfg,
        ember_observations_vs_stockfish(pilot_games, ember["name"]),
    )
    if pilot_fit["rating"] is None:
        target = clamp_level(cfg, sum(levels) / len(levels))
    else:
        target = clamp_level(cfg, pilot_fit["rating"])

    state["pilot_finished_at"] = now_utc()
    state["pilot_scheduled_games"] = scheduled_games
    state["pilot_estimate"] = pilot_fit
    state["target_level"] = target
    write_json(rd / "state.json", state)

    remaining = max_games - scheduled_games
    offsets = [int(offset) for offset in sf_cfg.get("main_level_offsets", [0])]
    if remaining >= 2 and offsets:
        remaining_budget = remaining
        for idx, offset in enumerate(offsets):
            slots_left = len(offsets) - idx
            games_for_level = remaining_budget // slots_left
            games_for_level -= games_for_level % 2
            if games_for_level < 2:
                continue
            level = clamp_level(cfg, target + offset)
            before = count_games_in_pgn(
                rd / "games" / f"{safe_name(ember['name'])}-vs-Stockfish-UCI-{level}.pgn"
            )
            total_games = before + games_for_level
            scheduled_games += run_match(
                cfg,
                rd,
                ember,
                stockfish_engine(cfg, level),
                total_games // 2,
                workers,
                f"{safe_name(ember['name'])}-vs-Stockfish-UCI-{level}.pgn",
            )
            remaining_budget = max(0, max_games - scheduled_games)

    state["finished_at"] = now_utc()
    state["scheduled_games"] = scheduled_games
    write_json(rd / "state.json", state)


def parse_pgn_file(path):
    games = []
    tags = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("[Event ") and tags:
                maybe_add_game(games, tags, path)
                tags = {}
            if line.startswith("[") and line.endswith("]"):
                m = re.match(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$', line)
                if m:
                    tags[m.group(1)] = m.group(2)
    if tags:
        maybe_add_game(games, tags, path)
    return games


def maybe_add_game(games, tags, path):
    result = tags.get("Result")
    if result not in RESULT_SCORE:
        return
    white = tags.get("White")
    black = tags.get("Black")
    if not white or not black:
        return
    ws, bs = RESULT_SCORE[result]
    games.append({
        "white": white,
        "black": black,
        "result": result,
        "white_score": ws,
        "black_score": bs,
        "termination": tags.get("Termination", ""),
        "pgn": str(path),
        "opening": tags.get("Opening", ""),
        "event": tags.get("Event", ""),
    })


def parse_all_games(rd):
    games = []
    for pgn in sorted((rd / "games").glob("*.pgn")):
        games.extend(parse_pgn_file(pgn))
    out = rd / "estimates" / "game-table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["white", "black", "result", "white_score", "black_score", "termination", "pgn", "opening", "event"])
        writer.writeheader()
        writer.writerows(games)
    shutil.copyfile(out, rd / "parsed-games.csv")
    return games


def build_priors(cfg, opponents, games):
    priors = {}
    sigmas = {}
    for engine in opponents:
        name = engine["name"]
        priors[name] = float(engine["rating"])
        sigmas[name] = float(engine.get("rating_sigma", cfg["estimator"].get("default_sigma", 250)))
    players = sorted({g["white"] for g in games} | {g["black"] for g in games})
    known_values = [priors[p] for p in players if p in priors]
    center = sum(known_values) / len(known_values) if known_values else 1800.0
    ember_name = cfg["ember"]["name"]
    if ember_name in players:
        priors[ember_name] = center
        sigmas[ember_name] = float(cfg["estimator"].get("ember_prior_sigma", 10000))
    return priors, sigmas


def nll(ratings, games, priors, sigmas):
    c = math.log(10.0) / 400.0
    total = 0.0
    eps = 1e-12
    for g in games:
        rw = ratings[g["white"]]
        rb = ratings[g["black"]]
        p = 1.0 / (1.0 + math.exp(-c * (rw - rb)))
        p = min(1.0 - eps, max(eps, p))
        s = g["white_score"]
        total -= s * math.log(p) + (1.0 - s) * math.log(1.0 - p)
    for player, prior in priors.items():
        if player in ratings:
            sigma = sigmas[player]
            total += ((ratings[player] - prior) ** 2) / (2.0 * sigma * sigma)
    return total


def solve_linear(a, b):
    n = len(b)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-15:
            mat[pivot][col] += 1e-9
        mat[col], mat[pivot] = mat[pivot], mat[col]
        div = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


def fit_ratings(games, priors, sigmas):
    players = sorted({g["white"] for g in games} | {g["black"] for g in games})
    if not games:
        raise RuntimeError("no games parsed")
    known_values = [priors[p] for p in players if p in priors]
    center = sum(known_values) / len(known_values) if known_values else 1800.0
    ratings = {p: float(priors.get(p, center)) for p in players}
    idx = {p: i for i, p in enumerate(players)}
    c = math.log(10.0) / 400.0

    for _ in range(80):
        grad = [0.0] * len(players)
        hess = [[0.0] * len(players) for _ in players]
        for g in games:
            wi = idx[g["white"]]
            bi = idx[g["black"]]
            rw = ratings[g["white"]]
            rb = ratings[g["black"]]
            p = 1.0 / (1.0 + math.exp(-c * (rw - rb)))
            s = g["white_score"]
            diff = c * (p - s)
            grad[wi] += diff
            grad[bi] -= diff
            w = c * c * p * (1.0 - p)
            hess[wi][wi] += w
            hess[bi][bi] += w
            hess[wi][bi] -= w
            hess[bi][wi] -= w

        for p_name, prior in priors.items():
            if p_name not in idx:
                continue
            i = idx[p_name]
            sigma = max(1e-6, sigmas[p_name])
            grad[i] += (ratings[p_name] - prior) / (sigma * sigma)
            hess[i][i] += 1.0 / (sigma * sigma)

        for i in range(len(players)):
            hess[i][i] += 1e-8

        delta = solve_linear(hess, grad)
        old = nll(ratings, games, priors, sigmas)
        step = 1.0
        improved = False
        for _ in range(12):
            trial = {p: ratings[p] - step * delta[idx[p]] for p in players}
            new = nll(trial, games, priors, sigmas)
            if new <= old:
                ratings = trial
                improved = True
                break
            step *= 0.5
        if not improved or max(abs(step * d) for d in delta) < 1e-4:
            break
    return ratings


def summarize_scores(games, ember_name):
    table = {}
    for g in games:
        for player, opponent, score in [
            (g["white"], g["black"], g["white_score"]),
            (g["black"], g["white"], g["black_score"]),
        ]:
            row = table.setdefault((player, opponent), {"player": player, "opponent": opponent, "games": 0, "score": 0.0})
            row["games"] += 1
            row["score"] += score
    return table


def termination_summary(games):
    summary = {}
    for g in games:
        term = (g.get("termination") or "not recorded").strip() or "not recorded"
        item = summary.setdefault(term, {"termination": term, "games": 0, "players": {}})
        item["games"] += 1
        lower = term.lower()
        culprit = None
        if "illegal move" in lower:
            if g["result"] == "1-0":
                culprit = g["black"]
            elif g["result"] == "0-1":
                culprit = g["white"]
            else:
                culprit = "unknown"
        elif "time" in lower:
            if g["result"] == "1-0":
                culprit = g["black"]
            elif g["result"] == "0-1":
                culprit = g["white"]
            else:
                culprit = "unknown"
        if culprit:
            item["players"][culprit] = item["players"].get(culprit, 0) + 1
    rows = []
    for item in summary.values():
        rows.append({
            "termination": item["termination"],
            "games": item["games"],
            "players": item["players"],
        })
    rows.sort(key=lambda r: (-r["games"], r["termination"]))
    return rows


def bootstrap_ember(games, priors, sigmas, ember_name, samples, seed):
    rng = random.Random(seed)
    values = []
    if not games or samples <= 0:
        return values
    for _ in range(samples):
        sampled = [rng.choice(games) for _ in games]
        try:
            ratings = fit_ratings(sampled, priors, sigmas)
            if ember_name in ratings:
                values.append(ratings[ember_name])
        except Exception:
            continue
    return values


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def analyze_stockfish_adaptive(config_path, run_id):
    cfg, _ = load_config(config_path)
    rd = run_dir_for(run_id)
    games = parse_all_games(rd)
    ember_name = cfg["ember"]["name"]
    observations = ember_observations_vs_stockfish(games, ember_name)
    fit = fit_stockfish_equivalent(cfg, observations)
    score_table = summarize_scores(games, ember_name)
    ember_scores = [row for (player, _), row in score_table.items() if player == ember_name]

    estimates_dir = rd / "estimates"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    with open(estimates_dir / "stockfish-observations.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["opponent_uci_elo", "ember_score"])
        writer.writerows(observations)

    state_path = rd / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    players = [{
        "player": ember_name,
        "rating": fit["rating"],
        "starting_rating": None,
        "rating_delta": None,
        "prior_sigma": None,
        "family": "ember",
        "rating_source": "Stockfish adaptive fit",
    }]
    seen_levels = sorted({level for level, _ in observations})
    for level in seen_levels:
        players.append({
            "player": f"Stockfish-UCI-{level}",
            "rating": float(level),
            "starting_rating": float(level),
            "rating_delta": 0.0,
            "prior_sigma": 0.0,
            "family": "stockfish-limited",
            "rating_source": "Stockfish UCI_Elo setting",
        })

    with open(estimates_dir / "fitted-ratings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "player",
                "rating",
                "starting_rating",
                "rating_delta",
                "prior_sigma",
                "family",
                "rating_source",
            ],
        )
        writer.writeheader()
        writer.writerows(players)

    estimate = {
        "run_id": run_id,
        "estimated_at": now_utc(),
        "game_count": len(games),
        "ember": {
            "name": ember_name,
            "rating": fit["rating"],
            "ci95_low": fit["ci95_low"],
            "ci95_high": fit["ci95_high"],
            "bootstrap_samples": 0,
        },
        "players": players,
        "ember_scores": ember_scores,
        "terminations": termination_summary(games),
        "rating_scale": cfg["estimator"].get(
            "primary_scale",
            "Stockfish UCI_Elo equivalent",
        ),
        "ci95_method": "1D logistic Hessian",
        "stockfish_adaptive": {
            "observations": fit["observations"],
            "standard_error": fit["standard_error"],
            "target_level": state.get("target_level"),
            "pilot_levels": state.get("pilot_levels", []),
            "pilot_estimate": state.get("pilot_estimate"),
            "max_games": state.get("max_games"),
            "scheduled_games": state.get("scheduled_games"),
        },
    }
    write_json(estimates_dir / "estimate.json", estimate)
    write_estimate_md(estimates_dir / "estimate.md", estimate)
    return estimate


def analyze(config_path, run_id):
    cfg, opponents = load_config(config_path)
    if cfg["run"].get("mode") == "stockfish-adaptive":
        return analyze_stockfish_adaptive(config_path, run_id)

    rd = run_dir_for(run_id)
    games = parse_all_games(rd)
    priors, sigmas = build_priors(cfg, opponents, games)
    ratings = fit_ratings(games, priors, sigmas)
    ember_name = cfg["ember"]["name"]
    samples = bootstrap_ember(
        games,
        priors,
        sigmas,
        ember_name,
        int(cfg["run"].get("bootstrap_samples", 200)),
        int(cfg["run"].get("bootstrap_seed", 1)),
    )
    ci_low = percentile(samples, 0.025)
    ci_high = percentile(samples, 0.975)

    estimates_dir = rd / "estimates"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    with open(estimates_dir / "bootstrap-samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "ember_rating"])
        for i, value in enumerate(samples):
            writer.writerow([i, value])
    run_cmd(["zstd", "-f", str(estimates_dir / "bootstrap-samples.csv")], log_path=rd / "commands.log", check=False)

    opponent_by_name = {o["name"]: o for o in opponents}
    fitted_rows = []
    for player in sorted(ratings, key=lambda p: ratings[p], reverse=True):
        opp = opponent_by_name.get(player, {})
        fitted_rows.append({
            "player": player,
            "rating": ratings[player],
            "starting_rating": priors.get(player),
            "rating_delta": ratings[player] - priors[player] if player in priors else None,
            "prior_sigma": sigmas.get(player),
            "family": opp.get("family", "ember" if player == ember_name else ""),
            "rating_source": opp.get("rating_source", ""),
        })
    with open(estimates_dir / "fitted-ratings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["player", "rating", "starting_rating", "rating_delta", "prior_sigma", "family", "rating_source"])
        writer.writeheader()
        writer.writerows(fitted_rows)

    score_table = summarize_scores(games, ember_name)
    ember_scores = [row for (player, _), row in score_table.items() if player == ember_name]
    estimate = {
        "run_id": run_id,
        "estimated_at": now_utc(),
        "game_count": len(games),
        "ember": {
            "name": ember_name,
            "rating": ratings.get(ember_name),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_samples": len(samples),
        },
        "players": fitted_rows,
        "ember_scores": ember_scores,
        "terminations": termination_summary(games),
        "rating_scale": cfg["estimator"].get("primary_scale", "unspecified"),
        "ci95_method": "bootstrap",
    }
    write_json(estimates_dir / "estimate.json", estimate)
    write_estimate_md(estimates_dir / "estimate.md", estimate)
    return estimate


def write_estimate_md(path, estimate):
    ember = estimate["ember"]
    lines = [
        "# Elo estimate",
        "",
        f"Ember rating: {ember['rating']:.1f}" if ember["rating"] is not None else "Ember rating: unavailable",
    ]
    if ember["ci95_low"] is not None and ember["ci95_high"] is not None:
        lines.append(f"95% CI: {ember['ci95_low']:.1f} to {ember['ci95_high']:.1f}")
    if estimate.get("ci95_method"):
        lines.append(f"95% CI method: {estimate['ci95_method']}")
    lines.extend([
        f"Games: {estimate['game_count']}",
        f"Rating scale: {estimate['rating_scale']}",
        "",
        "## Ember scores",
        "",
        "| Opponent | Games | Score | Score % |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in sorted(estimate["ember_scores"], key=lambda r: r["opponent"]):
        pct = 100.0 * row["score"] / row["games"] if row["games"] else 0.0
        lines.append(f"| {row['opponent']} | {row['games']} | {row['score']:.1f} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Terminations",
        "",
        "| Termination | Games | Attributed players |",
        "| --- | ---: | --- |",
    ])
    for row in estimate.get("terminations", []):
        players = ", ".join(f"{k}: {v}" for k, v in sorted(row.get("players", {}).items())) or ""
        lines.append(f"| {row['termination']} | {row['games']} | {players} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_archive(rd, artifact_name):
    artifact = rd / artifact_name
    if artifact.exists():
        artifact.unlink()
    include = []
    for name in ["config.toml", "state.json", "metadata.json", "commands.log", "build.log", "smoke.log", "parsed-games.csv"]:
        p = rd / name
        if p.exists():
            include.append(p.name)
    for sub in ["games", "estimates"]:
        if (rd / sub).exists():
            include.append(sub)
    if not include:
        return None
    run_cmd(["tar", "-I", "zstd -19", "-cf", artifact.name] + include, log_path=rd / "commands.log", cwd=rd)
    return artifact


def report(config_path, run_id):
    cfg, opponents = load_config(config_path)
    rd = run_dir_for(run_id)
    estimate_path = rd / "estimates" / "estimate.json"
    if not estimate_path.exists():
        estimate = analyze(config_path, run_id)
    else:
        estimate = json.loads(estimate_path.read_text(encoding="utf-8"))
    meta_path = rd / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    artifact = make_archive(rd, cfg["run"].get("artifact_name", "artifacts.tar.zst"))
    artifact_hash = sha256_file(artifact) if artifact and artifact.exists() else None

    manifest = []
    if artifact and artifact.exists():
        proc = run_cmd(["tar", "-tf", str(artifact)], check=False)
        manifest = [line for line in proc.stdout.splitlines() if line]

    ember = estimate["ember"]
    lines = [
        "# Ember Elo measurement report",
        "",
        f"Run id: `{run_id}`",
        f"Generated: {now_utc()}",
        "",
        "## Result",
        "",
    ]
    if ember["rating"] is not None:
        lines.append(f"Current measured Ember score: **{ember['rating']:.0f} Elo**")
    else:
        lines.append("Current measured Ember score: unavailable")
    if ember["ci95_low"] is not None and ember["ci95_high"] is not None:
        method = estimate.get("ci95_method", "bootstrap")
        lines.append(f"95% {method} CI: **{ember['ci95_low']:.0f} to {ember['ci95_high']:.0f} Elo**")
    lines.extend([
        f"Games parsed: {estimate['game_count']}",
        f"Rating scale: {estimate['rating_scale']}",
        "",
        "## Run metadata",
        "",
        f"Time control: `{cfg['run']['time_control']}`",
        f"Workers: `{meta.get('workers', 'unknown')}` from `{meta.get('worker_source', 'unknown')}`; CPU cores detected: `{meta.get('cpu_count', 'unknown')}`",
        f"Game budget: `{meta.get('max_games', cfg['run'].get('max_games', 'unknown'))}` from `{meta.get('max_games_source', 'config')}`",
        f"Measurement mode: `{meta.get('measurement_mode', cfg['run'].get('mode', 'mixed-prior'))}`",
        f"Execution host captured at runtime: `{meta.get('hostname', 'unknown')}`",
        f"Git commit: `{meta.get('git_commit', 'unknown')}`; dirty: `{meta.get('git_dirty', 'unknown')}`",
        f"Nix: `{meta.get('nix_version', 'unknown')}`",
        f"Ember binary SHA256: `{meta.get('ember_binary_sha256', 'unknown')}`",
        f"Ember book: `{cfg['ember'].get('book', '<embedded>')}`",
        f"Ember book name/source: `{cfg['ember'].get('book_name', cfg['ember'].get('book', '<embedded>'))}`",
        "",
        "## Ember scores",
        "",
        "| Opponent | Games | Score | Score % |",
        "| --- | ---: | ---: | ---: |",
    ])
    for row in sorted(estimate["ember_scores"], key=lambda r: r["opponent"]):
        pct = 100.0 * row["score"] / row["games"] if row["games"] else 0.0
        lines.append(f"| {row['opponent']} | {row['games']} | {row['score']:.1f} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Terminations",
        "",
        "| Termination | Games | Attributed players |",
        "| --- | ---: | --- |",
    ])
    for row in estimate.get("terminations", []):
        players = ", ".join(f"{k}: {v}" for k, v in sorted(row.get("players", {}).items())) or ""
        lines.append(f"| {row['termination']} | {row['games']} | {players} |")

    lines.extend([
        "",
        "## Fitted ratings",
        "",
        "| Player | Fitted | Start | Delta | Sigma | Family |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in estimate["players"]:
        rating = "" if row["rating"] is None else f"{row['rating']:.0f}"
        start = "" if row["starting_rating"] is None else f"{row['starting_rating']:.0f}"
        delta = "" if row["rating_delta"] is None else f"{row['rating_delta']:+.0f}"
        sigma = "" if row["prior_sigma"] is None else f"{row['prior_sigma']:.0f}"
        lines.append(f"| {row['player']} | {rating} | {start} | {delta} | {sigma} | {row['family']} |")

    if "stockfish_adaptive" in estimate:
        adaptive = estimate["stockfish_adaptive"]
        se = adaptive.get("standard_error")
        lines.extend([
            "",
            "## Stockfish adaptive",
            "",
            f"Target UCI_Elo after pilot: `{adaptive.get('target_level', 'unknown')}`",
            f"Stockfish observations: `{adaptive.get('observations', 0)}`",
            f"Standard error: `{se:.2f} Elo`" if se is not None else "Standard error: unavailable",
            f"Scheduled games: `{adaptive.get('scheduled_games', 'unknown')}` of budget `{adaptive.get('max_games', 'unknown')}`",
            "",
            "This mode reports a Stockfish-UCI_Elo-equivalent rating under the run conditions, not a fixed external CCRL rating.",
        ])

    lines.extend([
        "",
        "## Validation artifacts",
        "",
        f"Archive: `{artifact.name if artifact else 'none'}`",
        f"Archive SHA256: `{artifact_hash or 'none'}`",
        "",
        "The archive contains the raw PGNs, Cute Chess logs, commands, parsed game table, fitted ratings, estimator data, config, and metadata needed to validate this report.",
        "",
        "### Archive manifest",
        "",
    ])
    for item in manifest[:200]:
        lines.append(f"- `{item}`")
    if len(manifest) > 200:
        lines.append(f"- ... {len(manifest) - 200} more entries")

    lines.extend(["", "## Caveats", ""])
    if "stockfish_adaptive" in estimate:
        lines.extend([
            "- This is a Stockfish-UCI_Elo-equivalent estimate under this exact time control, opening set, book setting, and Stockfish build.",
            "- It is useful for repeatable development comparisons, but it is not a fixed external CCRL rating.",
            "- Longer runs and a larger opening set are needed for narrow confidence intervals.",
        ])
    else:
        lines.extend([
            "- Opponent ratings are priors, not fixed truth; the fitted table shows how much they moved.",
            "- The default pool mixes external CCRL-style priors and Stockfish limited-strength diagnostics, so this first score should be treated as a current calibrated estimate with visible uncertainty.",
            "- Longer runs with more independent Nix-available external engines will narrow the confidence interval and reduce dependence on Stockfish-limited opponents.",
        ])
    (rd / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta["artifact"] = str(artifact) if artifact else None
    meta["artifact_sha256"] = artifact_hash
    meta["finished_at"] = now_utc()
    write_json(meta_path, meta)
    print(rd / "report.md")


def copy_config(config_path, run_id):
    rd = run_dir_for(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, rd / "config.toml")


def all_steps(config_path, run_id, explicit_workers=None, explicit_max_games=None):
    copy_config(config_path, run_id)
    probe(config_path, run_id, explicit_workers, explicit_max_games)
    build(config_path, run_id)
    smoke(config_path, run_id)
    run_matches(config_path, run_id, explicit_workers, explicit_max_games)
    analyze(config_path, run_id)
    report(config_path, run_id)


def main():
    parser = argparse.ArgumentParser(description="Measure Ember Elo with Cute Chess and a joint Elo estimator.")
    parser.add_argument("command", choices=["probe", "build", "smoke", "run", "analyze", "report", "all"])
    parser.add_argument("--config", default="configs/elo/default.toml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", default=None)
    parser.add_argument("--max-games", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    run_id = args.run_id or make_run_id()

    if args.command == "probe":
        copy_config(config_path, run_id)
        probe(config_path, run_id, args.workers, args.max_games)
    elif args.command == "build":
        build(config_path, run_id)
    elif args.command == "smoke":
        smoke(config_path, run_id)
    elif args.command == "run":
        run_matches(config_path, run_id, args.workers, args.max_games)
    elif args.command == "analyze":
        analyze(config_path, run_id)
    elif args.command == "report":
        report(config_path, run_id)
    elif args.command == "all":
        all_steps(config_path, run_id, args.workers, args.max_games)


if __name__ == "__main__":
    main()
