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

from sprt import pentanomial_counts, pentanomial_sprt

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


def shell_quote(s):
    if re.match(r"^[A-Za-z0-9_./:=+@,%^-]+$", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_cmd(args, log_path=None, check=True, env=None, cwd=None):
    start = time.time()
    proc = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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


def load_config(path):
    cfg = read_toml(path)
    if "engine_a" not in cfg or "engine_b" not in cfg:
        raise ValueError("config must contain [engine_a] and [engine_b]")
    return cfg


def make_run_id(cfg):
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = safe_name(cfg["run"].get("name", "head-to-head"))
    return f"{stamp}-{name}"


def run_dir_for(cfg, run_id):
    return Path(cfg["run"].get("results_dir", "results/head-to-head")) / run_id


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


def materialize_revision_commands(cfg, rd):
    for engine_id in ["engine_a", "engine_b"]:
        if "revision" in cfg[engine_id]:
            cfg[engine_id]["cmd"] = str(
                (rd / "builds" / engine_id / "bin" / "ember").resolve()
            )


def build_revision(cfg, rd, engine_id):
    build_cfg = cfg.get("build", {})
    engine = cfg[engine_id]
    repo = Path(build_cfg.get("repo", ".")).resolve()
    revision = git_output(repo, "rev-parse", f"{engine['revision']}^{{commit}}")
    root = rd / "builds" / engine_id
    installed = root / "bin" / "ember"
    metadata_path = root / "metadata.json"
    if installed.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("revision") == revision and metadata.get(
            "sha256"
        ) == sha256_file(installed):
            return metadata

    source = root / "source"
    archive = root / "source.tar"
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=tar",
            f"--output={archive}",
            revision,
        ],
        log_path=root / "build.log",
    )
    run_cmd(
        ["tar", "-xf", str(archive), "-C", str(source)],
        log_path=root / "build.log",
    )
    command = list(
        build_cfg.get(
            "command",
            ["cargo", "build", "--locked", "--release", "--bin", "ember"],
        )
    )
    run_cmd(command, log_path=root / "build.log", cwd=source)
    built = source / build_cfg.get("binary", "target/release/ember")
    if not built.is_file():
        raise RuntimeError(f"build did not produce {built}")
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, installed)
    installed.chmod(0o755)
    metadata = {
        "engine": engine_id,
        "configured_revision": engine["revision"],
        "revision": revision,
        "binary": str(installed.resolve()),
        "sha256": sha256_file(installed),
        "command": command,
    }
    write_json(metadata_path, metadata)
    return metadata


def record_revision_metadata(metadata, cfg, revision_metadata):
    metadata.setdefault("engine_binaries", {})
    metadata.setdefault("tools", {})
    for engine_id, binary in revision_metadata.items():
        metadata["engine_binaries"][cfg[engine_id]["name"]] = binary
        path = binary["binary"]
        metadata["tools"][path] = {"path": path, "available": True}


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def detect_workers(cfg, explicit_workers):
    cores = os.cpu_count() or 1
    if explicit_workers is not None:
        return max(1, int(explicit_workers)), cores, "cli"
    configured = str(cfg["run"].get("workers", "auto"))
    if configured != "auto":
        return max(1, int(configured)), cores, "config"
    multiplier = float(cfg["run"].get("worker_multiplier", 1.0))
    return max(1, int(math.ceil(cores * multiplier))), cores, "auto"


def engine_args(engine):
    args = [
        "-engine",
        f"name={engine['name']}",
        f"cmd={engine['cmd']}",
        f"proto={engine.get('proto', 'uci')}",
    ]
    if "dir" in engine:
        args.append(f"dir={engine['dir']}")
    for arg in engine.get("args", []):
        args.append(f"arg={arg}")
    for key, value in engine.get("options", {}).items():
        if key.lower() == "syzygypath" and value and value.lower() != "<empty>":
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = path.resolve()
            value = str(path)
        args.append(f"option.{key}={value}")
    return args


def engine_limit_args(run_cfg):
    limits = []
    if "time_control" in run_cfg:
        limits.append(("tc", run_cfg["time_control"]))
    if "nodes" in run_cfg:
        limits.append(("nodes", int(run_cfg["nodes"])))
    if "depth" in run_cfg:
        limits.append(("depth", int(run_cfg["depth"])))
    if "move_time" in run_cfg:
        limits.append(("st", run_cfg["move_time"]))
    if len(limits) != 1:
        raise RuntimeError(
            "configure exactly one of time_control, nodes, depth, or move_time"
        )

    name, value = limits[0]
    args = [f"{name}={value}"]
    if name in {"nodes", "depth"}:
        args.insert(0, "tc=inf")
    if name in {"tc", "st"}:
        args.append(f"timemargin={int(run_cfg.get('timemargin_ms', 2000))}")
    return args


def count_games_in_pgn(path):
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("[Event "):
                count += 1
    return count


def read_opening_lines(path):
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    if not lines:
        raise RuntimeError(f"no openings in {path}")
    return lines


def sample_polyglot_openings(cfg, count):
    try:
        import chess
        import chess.polyglot
    except ImportError as e:
        raise RuntimeError(
            "polyglot opening sampling requires python-chess"
        ) from e

    run_cfg = cfg["run"]
    book = Path(run_cfg.get("polyglot_book", "src/book.bin"))
    if not book.exists():
        raise RuntimeError(f"polyglot book not found: {book}")

    seed = int(run_cfg.get("seed", 20260714))
    min_plies = int(run_cfg.get("book_min_plies", 8))
    max_plies = int(run_cfg.get("book_max_plies", 20))
    if min_plies > max_plies:
        raise RuntimeError("book_min_plies must be <= book_max_plies")

    rng = random.Random(seed)
    openings = []
    seen = set()
    attempts = max(count * 80, 2000)
    with chess.polyglot.open_reader(str(book)) as reader:
        for _ in range(attempts):
            board = chess.Board()
            target_plies = rng.randint(min_plies, max_plies)
            for _ply in range(target_plies):
                entries = list(reader.find_all(board))
                if not entries:
                    break
                weights = [max(1, entry.weight) for entry in entries]
                entry = rng.choices(entries, weights=weights, k=1)[0]
                board.push(entry.move)
                if board.is_game_over(claim_draw=True):
                    break
            if board.ply() < min_plies or board.is_game_over(claim_draw=True):
                continue
            epd = board.epd()
            if epd not in seen:
                seen.add(epd)
                openings.append(epd)
                if len(openings) >= count:
                    break
    if len(openings) < count:
        raise RuntimeError(f"sampled only {len(openings)} unique book openings, wanted {count}")
    return openings


def prepare_openings(cfg, rd, max_pairs):
    openings_path = rd / "openings" / "sampled.epd"
    if openings_path.exists():
        return openings_path, len(read_opening_lines(openings_path))

    source = cfg["run"].get("opening_source", "file")
    if source == "polyglot":
        lines = sample_polyglot_openings(cfg, max_pairs)
    elif source == "file":
        opening_file = Path(cfg["run"]["opening_file"])
        lines = read_opening_lines(opening_file)
        rng = random.Random(int(cfg["run"].get("seed", 20260714)))
        rng.shuffle(lines)
        lines = lines[:max_pairs]
    else:
        raise RuntimeError(f"unknown opening_source: {source}")

    openings_path.parent.mkdir(parents=True, exist_ok=True)
    openings_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return openings_path, len(lines)


def write_batch_openings(rd, all_openings, start, count):
    path = rd / "openings" / f"batch-{start + 1:06d}-{start + count:06d}.epd"
    if not path.exists():
        path.write_text("\n".join(all_openings[start : start + count]) + "\n", encoding="utf-8")
    return path


def parse_pgn_file(path):
    games = []
    tags = {}
    order = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("[Event ") and tags:
                order += 1
                maybe_add_game(games, tags, path, order)
                tags = {}
            if line.startswith("[") and line.endswith("]"):
                m = re.match(r'^\[([A-Za-z0-9_]+)\s+"(.*)"\]$', line)
                if m:
                    tags[m.group(1)] = m.group(2)
    if tags:
        order += 1
        maybe_add_game(games, tags, path, order)
    return games


def maybe_add_game(games, tags, path, order):
    result = tags.get("Result")
    if result not in RESULT_SCORE:
        return
    white = tags.get("White")
    black = tags.get("Black")
    if not white or not black:
        return
    ws, bs = RESULT_SCORE[result]
    games.append(
        {
            "white": white,
            "black": black,
            "result": result,
            "white_score": ws,
            "black_score": bs,
            "termination": tags.get("Termination", ""),
            "fen": tags.get("FEN", ""),
            "round": tags.get("Round", ""),
            "event": tags.get("Event", ""),
            "pgn": str(path),
            "order": order,
        }
    )


def parse_all_games(rd):
    games = []
    for pgn in sorted((rd / "games").glob("*.pgn")):
        games.extend(parse_pgn_file(pgn))
    return games


def score_for(game, player):
    if game["white"] == player:
        return game["white_score"]
    if game["black"] == player:
        return game["black_score"]
    raise ValueError(f"{player} not in game {game}")


def pair_games(games, engine_a, engine_b):
    pairs = []
    buckets = {}
    fallback = []
    ordered = sorted(games, key=lambda g: (g["pgn"], g["order"]))
    for game in ordered:
        if {game["white"], game["black"]} != {engine_a, engine_b}:
            continue
        if game.get("fen"):
            buckets.setdefault((game["pgn"], game["fen"]), []).append(game)
        else:
            fallback.append(game)

    chunks = []
    for chunk in buckets.values():
        chunks.extend(chunk[i : i + 2] for i in range(0, len(chunk) - 1, 2))
    chunks.extend(fallback[i : i + 2] for i in range(0, len(fallback) - 1, 2))

    for chunk in chunks:
        if len(chunk) != 2:
            continue
        colors_ok = (
            chunk[0]["white"] != chunk[1]["white"]
            and chunk[0]["black"] != chunk[1]["black"]
        )
        if colors_ok:
            a_score = sum(score_for(game, engine_a) for game in chunk)
            pairs.append(
                {
                    "pair": len(pairs) + 1,
                    "a_score": a_score,
                    "delta": a_score - 1.0,
                    "results": " ".join(game["result"] for game in chunk),
                    "pgn": chunk[0]["pgn"],
                    "rounds": " ".join(game["round"] for game in chunk),
                }
            )
    return pairs


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def score_to_elo(score):
    eps = 1e-6
    score = min(1.0 - eps, max(eps, score))
    return 400.0 * math.log10(score / (1.0 - score))


def analyze_pairs(pairs, alpha):
    n = len(pairs)
    if n == 0:
        return {
            "pairs": 0,
            "games": 0,
            "score": 0.0,
            "score_rate": 0.0,
            "elo": None,
            "p_greater": None,
            "p_less": None,
            "p_two_sided": None,
            "los": None,
            "significant": False,
        }

    deltas = [pair["delta"] for pair in pairs]
    total_score = sum(pair["a_score"] for pair in pairs)
    mean_delta = sum(deltas) / n
    if n > 1:
        variance = sum((d - mean_delta) ** 2 for d in deltas) / (n - 1)
    else:
        variance = 0.0
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n) if n > 0 else float("inf")
    if se == 0.0:
        if mean_delta > 0:
            z = float("inf")
        elif mean_delta < 0:
            z = float("-inf")
        else:
            z = 0.0
    else:
        z = mean_delta / se

    if math.isinf(z):
        cdf = 1.0 if z > 0 else 0.0
    else:
        cdf = normal_cdf(z)
    p_less = cdf
    p_greater = 1.0 - cdf
    p_two = min(1.0, 2.0 * min(p_less, p_greater))
    score_rate = total_score / (2.0 * n)
    ci_delta = 1.959963984540054 * se
    score_ci_low = max(0.0, (mean_delta - ci_delta + 1.0) / 2.0)
    score_ci_high = min(1.0, (mean_delta + ci_delta + 1.0) / 2.0)

    a_wins = sum(1 for p in pairs if p["a_score"] > 1.0)
    a_losses = sum(1 for p in pairs if p["a_score"] < 1.0)
    tied_pairs = n - a_wins - a_losses
    return {
        "pairs": n,
        "games": 2 * n,
        "score": total_score,
        "score_rate": score_rate,
        "pair_mean_delta": mean_delta,
        "pair_sd": sd,
        "pair_se": se,
        "z": z,
        "p_greater": p_greater,
        "p_less": p_less,
        "p_two_sided": p_two,
        "los": cdf,
        "elo": score_to_elo(score_rate),
        "elo_ci95_low": score_to_elo(score_ci_low),
        "elo_ci95_high": score_to_elo(score_ci_high),
        "a_won_pairs": a_wins,
        "a_lost_pairs": a_losses,
        "tied_pairs": tied_pairs,
        "significant": min(p_greater, p_less) <= alpha,
    }


def selected_p_value(stats, alternative):
    if alternative == "greater":
        return stats["p_greater"]
    if alternative == "less":
        return stats["p_less"]
    if alternative == "two-sided":
        return stats["p_two_sided"]
    raise RuntimeError(f"unknown alternative: {alternative}")


def decision(stats, cfg):
    run_cfg = cfg["run"]
    sprt_cfg = cfg.get("sprt", {})
    if sprt_cfg.get("enabled", False):
        min_pairs = int(sprt_cfg.get("min_pairs", run_cfg.get("min_pairs", 1)))
        if stats["pairs"] < min_pairs:
            return "continue"
        state = stats["sprt"]["state"]
        if state == "accept_h1":
            return "engine_a_better"
        if state == "accept_h0":
            return "engine_b_better"
        return "continue"

    alpha = float(run_cfg.get("alpha", 0.05))
    min_pairs = int(run_cfg.get("min_pairs", 30))
    if stats["pairs"] < min_pairs:
        return "continue"
    if stats["p_greater"] is not None and stats["p_greater"] <= alpha:
        return "engine_a_better"
    if stats["p_less"] is not None and stats["p_less"] <= alpha:
        return "engine_b_better"
    return "continue"


def capped_verdict(stats, verdict, max_pairs):
    if verdict == "continue" and stats["pairs"] >= max_pairs:
        return "inconclusive"
    return verdict


def write_pair_csv(rd, pairs):
    out = rd / "estimates" / "paired-openings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair", "a_score", "delta", "results", "pgn", "rounds"]
        )
        writer.writeheader()
        writer.writerows(pairs)


def fmt_float(value, digits=1, suffix=""):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def fmt_sig(value):
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def write_report(rd, cfg, stats, verdict):
    a = cfg["engine_a"]["name"]
    b = cfg["engine_b"]["name"]
    p = selected_p_value(stats, cfg["run"].get("alternative", "greater"))
    los = None if stats["los"] is None else 100.0 * stats["los"]
    lines = [
        "# Head-to-head report",
        "",
        f"Engine A: **{a}**",
        f"Engine B: **{b}**",
        f"Opening design: unique random openings, two games per opening, colors swapped.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Pairs | {stats['pairs']} |",
        f"| Games | {stats['games']} |",
        f"| A score | {stats['score']:.1f} / {stats['games']} |",
        f"| A score rate | {100.0 * stats['score_rate']:.2f}% |",
        f"| Elo A-B | {fmt_float(stats['elo'])} |",
        f"| 95% Elo CI | {fmt_float(stats.get('elo_ci95_low'))} to {fmt_float(stats.get('elo_ci95_high'))} |",
        f"| Paired p(A>B) one-sided | {fmt_sig(stats['p_greater'])} |",
        f"| Paired p(A<B) one-sided | {fmt_sig(stats['p_less'])} |",
        f"| Paired p two-sided | {fmt_sig(stats['p_two_sided'])} |",
        f"| LOS-like P(A>B) | {fmt_float(los, 2, '%')} |",
        f"| A won/lost/tied pairs | {stats.get('a_won_pairs', 0)} / {stats.get('a_lost_pairs', 0)} / {stats.get('tied_pairs', 0)} |",
        "",
        f"Configured alpha: `{float(cfg['run'].get('alpha', 0.05))}`",
        f"Selected alternative: `{cfg['run'].get('alternative', 'greater')}`",
        f"Selected p-value: `{fmt_sig(p)}`",
        f"Verdict: **{verdict}**",
        "",
    ]
    if "sprt" in stats:
        sprt = stats["sprt"]
        lines.extend(
            [
                "## Sequential test",
                "",
                f"Pentanomial: `{sprt['pentanomial']}`",
                f"Hypotheses: `{sprt['elo0']:.1f}` versus `{sprt['elo1']:.1f}` Elo",
                f"Alpha / beta: `{sprt['alpha']:.3f}` / `{sprt['beta']:.3f}`",
                f"LLR: `{sprt['llr']:.6f}`",
                f"Bounds: `{sprt['lower_bound']:.6f}` to `{sprt['upper_bound']:.6f}`",
                f"State: **{sprt['state']}**",
                "",
            ]
        )
    (rd / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(config_path, run_id, cfg_override=None):
    cfg = cfg_override or load_config(config_path)
    rd = run_dir_for(cfg, run_id)
    games = parse_all_games(rd)
    pairs = pair_games(games, cfg["engine_a"]["name"], cfg["engine_b"]["name"])
    alpha = float(cfg["run"].get("alpha", 0.05))
    stats = analyze_pairs(pairs, alpha)
    sprt_cfg = cfg.get("sprt", {})
    if sprt_cfg.get("enabled", False) and pairs:
        stats["sprt"] = pentanomial_sprt(
            pentanomial_counts(pairs),
            float(sprt_cfg["elo0"]),
            float(sprt_cfg["elo1"]),
            float(sprt_cfg.get("alpha", 0.05)),
            float(sprt_cfg.get("beta", 0.05)),
        )
    verdict = decision(stats, cfg)
    write_pair_csv(rd, pairs)
    summary = dict(stats)
    summary["verdict"] = verdict
    write_json(rd / "estimates" / "summary.json", summary)
    write_report(rd, cfg, stats, verdict)
    return stats, verdict


def probe(config_path, run_id, explicit_workers=None):
    cfg = load_config(config_path)
    rd = run_dir_for(cfg, run_id)
    materialize_revision_commands(cfg, rd)
    workers, cores, worker_source = detect_workers(cfg, explicit_workers)
    meta = {
        "run_id": run_id,
        "started_at": now_utc(),
        "hostname": socket.gethostname(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "cpu_count": cores,
        "workers": workers,
        "worker_source": worker_source,
        "engine_a": cfg["engine_a"],
        "engine_b": cfg["engine_b"],
        "tools": {},
    }
    for cmd in ["python3", cfg["run"].get("cutechess_cmd", "cutechess-cli")]:
        exe = cmd.split()[0]
        meta["tools"][cmd] = {"path": shutil.which(exe), "available": shutil.which(exe) is not None}
    for engine in [cfg["engine_a"], cfg["engine_b"]]:
        exe = str(engine["cmd"]).split()[0]
        meta["tools"][engine["cmd"]] = {"path": shutil.which(exe), "available": shutil.which(exe) is not None}
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "metadata.json", meta)
    write_json(rd / "state.json", {"phase": "probe", "metadata": meta})
    print(json.dumps({"run_id": run_id, "workers": workers, "cpu_count": cores}, indent=2))


def build(config_path, run_id):
    cfg = load_config(config_path)
    rd = run_dir_for(cfg, run_id)
    revision_engines = [
        engine_id
        for engine_id in ["engine_a", "engine_b"]
        if "revision" in cfg[engine_id]
    ]
    if revision_engines:
        revision_metadata = {}
        for engine_id in revision_engines:
            revision_metadata[engine_id] = build_revision(cfg, rd, engine_id)
        materialize_revision_commands(cfg, rd)
        metadata_path = rd / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        record_revision_metadata(metadata, cfg, revision_metadata)
        write_json(metadata_path, metadata)
        if len(revision_engines) == 2:
            return

    command = cfg.get("build", {}).get("command")
    if not command:
        return
    run_cmd(command, log_path=rd / "build.log")
    for engine in [cfg["engine_a"], cfg["engine_b"]]:
        cmd = Path(engine["cmd"])
        if cmd.exists() and cmd.is_file():
            meta_path = rd / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            meta.setdefault("engine_binaries", {})[engine["name"]] = {
                "path": str(cmd),
                "sha256": sha256_file(cmd),
            }
            write_json(meta_path, meta)


def run_batches(
    config_path,
    run_id,
    explicit_workers=None,
    explicit_max_pairs=None,
    explicit_alpha=None,
):
    cfg = load_config(config_path)
    materialize_revision_commands(cfg, run_dir_for(cfg, run_id))
    if explicit_alpha is not None:
        cfg["run"]["alpha"] = float(explicit_alpha)
    rd = run_dir_for(cfg, run_id)
    rd.mkdir(parents=True, exist_ok=True)
    workers, _cores, _source = detect_workers(cfg, explicit_workers)
    max_pairs = int(explicit_max_pairs or cfg["run"].get("max_pairs", 200))
    batch_pairs = max(1, int(cfg["run"].get("batch_pairs", 16)))
    openings_path, opening_count = prepare_openings(cfg, rd, max_pairs)
    max_pairs = min(max_pairs, opening_count)
    all_openings = read_opening_lines(openings_path)

    start_pair = 0
    cutechess = cfg["run"].get("cutechess_cmd", "cutechess-cli")
    if isinstance(cutechess, str):
        cutechess_cmd = [cutechess]
    else:
        cutechess_cmd = list(cutechess)

    while start_pair < max_pairs:
        stats, verdict = analyze(config_path, run_id, cfg)
        if verdict != "continue":
            break
        start_pair = stats["pairs"]
        if start_pair >= max_pairs:
            break
        pairs_this_batch = min(batch_pairs, max_pairs - start_pair)
        batch_openings = write_batch_openings(rd, all_openings, start_pair, pairs_this_batch)
        pgn = rd / "games" / f"batch-{start_pair + 1:06d}-{start_pair + pairs_this_batch:06d}.pgn"
        pgn.parent.mkdir(parents=True, exist_ok=True)
        if count_games_in_pgn(pgn) >= pairs_this_batch * 2:
            start_pair += pairs_this_batch
            continue

        args = []
        args.extend(cutechess_cmd)
        args.extend(engine_args(cfg["engine_a"]))
        args.extend(engine_args(cfg["engine_b"]))
        args.extend(
            [
                "-each",
                *engine_limit_args(cfg["run"]),
                "-openings",
                f"file={batch_openings}",
                f"format={cfg['run'].get('opening_format', 'epd')}",
                "order=sequential",
                "policy=round",
                "-games",
                "2",
                "-rounds",
                str(pairs_this_batch),
                "-repeat",
                "-concurrency",
                str(workers),
                "-pgnout",
                str(pgn),
                "-recover",
                "-ratinginterval",
                str(int(cfg["run"].get("rating_interval", 20))),
            ]
        )
        if int(cfg["run"].get("max_moves", 0)) > 0:
            args.extend(["-maxmoves", str(int(cfg["run"]["max_moves"]))])
        run_cmd(args, log_path=rd / "games" / (pgn.name + ".cutechess.log"))
        with open(rd / "commands.log", "a", encoding="utf-8") as f:
            f.write(" ".join(shell_quote(a) for a in args) + "\n")

    stats, verdict = analyze(config_path, run_id, cfg)
    verdict = capped_verdict(stats, verdict, max_pairs)
    if verdict == "inconclusive":
        summary = dict(stats)
        summary["verdict"] = verdict
        write_json(rd / "estimates" / "summary.json", summary)
        write_report(rd, cfg, stats, verdict)
    state = {
        "phase": "finished",
        "finished_at": now_utc(),
        "stats": stats,
        "verdict": verdict,
        "max_pairs": max_pairs,
    }
    write_json(rd / "state.json", state)
    print(json.dumps(state, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description="Paired-opening head-to-head runner")
    parser.add_argument("command", choices=["probe", "build", "run", "analyze", "all"])
    parser.add_argument("--config", default="configs/head-to-head/ember-syzygy.toml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", default=None)
    parser.add_argument("--max-pairs", default=None)
    parser.add_argument("--alpha", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    run_id = args.run_id or make_run_id(cfg)

    if args.command in {"probe", "all"}:
        probe(config_path, run_id, args.workers)
    if args.command in {"build", "all"}:
        build(config_path, run_id)
    if args.command in {"run", "all"}:
        run_batches(config_path, run_id, args.workers, args.max_pairs, args.alpha)
    if args.command in {"analyze"}:
        stats, verdict = analyze(config_path, run_id)
        print(json.dumps({"stats": stats, "verdict": verdict}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
