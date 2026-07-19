#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib

try:
    import chess
    import chess.svg
except ImportError:  # pragma: no cover
    chess = None

try:
    import cairosvg
except ImportError:  # pragma: no cover
    cairosvg = None


MATE_CP = 100_000


def now_utc():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shell_quote(value):
    if re.match(r"^[A-Za-z0-9_./:=+@,%^-]*$", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_cmd(args, log_path=None, cwd=None, env=None, check=True, timeout=None):
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
            f.write("$ " + " ".join(shell_quote(str(a)) for a in args) + "\n")
            f.write(proc.stdout)
            if proc.stdout and not proc.stdout.endswith("\n"):
                f.write("\n")
            f.write(f"[exit={proc.returncode} elapsed={elapsed:.3f}s]\n\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, args))}")
    return proc


def command_exists(cmd):
    return shutil.which(cmd) is not None


def make_run_id():
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        suffix = git.stdout.strip() if git.returncode == 0 else "nogit"
        dirty = subprocess.run(["git", "status", "--porcelain"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if dirty.returncode == 0 and dirty.stdout.strip():
            suffix += "-dirty"
    except FileNotFoundError:
        suffix = "nogit"
    return f"{stamp}-{suffix}"


def run_dir_for(run_id):
    return Path("results") / "stupidity" / run_id


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def detect_workers(cfg, explicit_workers):
    if explicit_workers:
        return max(1, int(explicit_workers))
    configured = str(cfg["run"].get("workers", "auto"))
    if configured != "auto":
        return max(1, int(configured))
    cores = os.cpu_count() or 1
    return max(1, int(math.ceil(cores * float(cfg["run"].get("worker_multiplier", 1.25)))))


def expected_games(rounds):
    return max(1, int(rounds)) * 2


def engine_args(engine):
    args = ["-engine", f"name={engine['name']}", f"cmd={engine['cmd']}", f"proto={engine.get('proto', 'uci')}"]
    for key, value in engine.get("options", {}).items():
        args.append(f"option.{key}={value}")
    return args


def ember_engine(cfg, name=None):
    return {
        "name": name or cfg["ember"]["name"],
        "cmd": cfg["ember"]["binary"],
        "proto": cfg["ember"].get("proto", "uci"),
        "options": cfg["ember"].get("options", {}),
    }


def stockfish_engine(cfg):
    sf = cfg["stockfish"]
    options = dict(sf.get("options", {}))
    options.setdefault("UCI_LimitStrength", "true")
    options.setdefault("UCI_Elo", str(sf.get("uci_elo", 1800)))
    options.setdefault("Threads", "1")
    options.setdefault("Hash", "64")
    return {
        "name": sf.get("name", f"Stockfish-UCI-{options['UCI_Elo']}"),
        "cmd": sf.get("cmd", "stockfish"),
        "proto": sf.get("proto", "uci"),
        "options": options,
    }


def opponent_engine(cfg):
    if "opponent" not in cfg:
        return stockfish_engine(cfg)
    opponent = cfg["opponent"]
    return {
        "name": opponent["name"],
        "cmd": opponent["cmd"],
        "proto": opponent.get("proto", "uci"),
        "options": dict(opponent.get("options", {})),
    }


def copy_config(config_path, rd):
    rd.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, rd / "config.toml")


def probe(config_path, run_id, explicit_workers=None):
    cfg = read_toml(config_path)
    rd = run_dir_for(run_id)
    copy_config(config_path, rd)
    opponent = opponent_engine(cfg)
    tools = {}
    for cmd in ["cargo", "rustc", "cutechess-cli", cfg["stockfish"].get("cmd", "stockfish"), opponent["cmd"], "ffmpeg"]:
        tools[cmd] = {"path": shutil.which(cmd), "available": command_exists(cmd)}
    meta = {
        "run_id": run_id,
        "started_at": now_utc(),
        "config_path": str(config_path),
        "workers": detect_workers(cfg, explicit_workers),
        "tools": tools,
    }
    write_json(rd / "metadata.json", meta)
    required = ["cargo", "cutechess-cli", cfg["stockfish"].get("cmd", "stockfish"), opponent["cmd"]]
    missing = [name for name, item in tools.items() if not item["available"] and name in required]
    print(json.dumps({"run_id": run_id, "missing": missing, "workers": meta["workers"]}, indent=2))
    if missing:
        raise SystemExit(2)


def build(config_path, run_id):
    cfg = read_toml(config_path)
    rd = run_dir_for(run_id)
    rd.mkdir(parents=True, exist_ok=True)
    args = ["cargo", "build", "--release", "--bin", "ember"]
    features = cfg["ember"].get("features", [])
    if isinstance(features, str):
        features = [features]
    if features:
        args.extend(["--features", " ".join(features)])
    run_cmd(args, log_path=rd / "build.log")
    binary = Path(cfg["ember"]["binary"])
    if not binary.exists():
        raise RuntimeError(f"missing Ember binary: {binary}")


def smoke(config_path, run_id):
    cfg = read_toml(config_path)
    rd = run_dir_for(run_id)
    trace = rd / "smoke-trace.jsonl"
    cmds = [
        "uci",
        "isready",
        f"setoption name TraceFile value {trace}",
        "setoption name Book value",
        "position startpos",
        "go depth 1",
        "quit",
    ]
    proc = subprocess.run(
        [cfg["ember"]["binary"]],
        input="\n".join(cmds) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    (rd / "smoke.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0 or "uciok" not in proc.stdout or "bestmove" not in proc.stdout or not trace.exists():
        raise RuntimeError("smoke failed")


def run_match(cfg, rd, white, black, rounds, workers, pgn_name):
    games_dir = rd / "games"
    traces_dir = rd / "traces"
    games_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if cfg["run"].get("trace", True):
        env["EMBER_TRACE_DIR"] = str(traces_dir.resolve())
    pgn = games_dir / pgn_name
    args = ["cutechess-cli"]
    args.extend(engine_args(white))
    args.extend(engine_args(black))
    args.extend([
        "-each",
        f"tc={cfg['run']['time_control']}",
        f"timemargin={int(cfg['run'].get('timemargin_ms', 2000))}",
        "-openings",
        f"file={cfg['run']['opening_file']}",
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
    ])
    run_cmd(args, log_path=games_dir / (pgn_name + ".cutechess.log"), env=env)


def run_matches(config_path, run_id, explicit_workers=None, explicit_max_games=None):
    cfg = read_toml(config_path)
    rd = run_dir_for(run_id)
    workers = detect_workers(cfg, explicit_workers)
    max_games = int(explicit_max_games or cfg["run"].get("max_games", 48))
    rounds = int(cfg["run"].get("rounds_per_pair", 12))
    scheduled = 0
    opponent = opponent_engine(cfg)
    if scheduled + expected_games(rounds) <= max_games:
        pgn_name = f"Ember-vs-{safe_name(opponent['name'])}.pgn"
        run_match(cfg, rd, ember_engine(cfg), opponent, rounds, workers, pgn_name)
        scheduled += expected_games(rounds)
    self_rounds = max(1, min(rounds, (max_games - scheduled) // 2))
    if self_rounds > 0:
        run_match(cfg, rd, ember_engine(cfg, "Ember-A"), ember_engine(cfg, "Ember-B"), self_rounds, workers, "Ember-A-vs-Ember-B.pgn")
        scheduled += expected_games(self_rounds)
    state = {
        "run_id": run_id,
        "finished_at": now_utc(),
        "scheduled_games": scheduled,
        "workers": workers,
    }
    write_json(rd / "state.json", state)


def load_traces(rd):
    rows = []
    for path in sorted((rd / "traces").glob("*.jsonl")):
        for idx, row in enumerate(read_jsonl(path), start=1):
            if row.get("event") != "ember_decision":
                continue
            row["trace_file"] = str(path)
            row["trace_line"] = idx
            rows.append(row)
    return rows


class UciEngine:
    def __init__(self, cmd):
        self.proc = subprocess.Popen(
            [cmd],
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        self.send("uci")
        self.read_until("uciok")

    def send(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def read_until(self, token):
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)
            if line.startswith(token) or token in line:
                break
        return lines

    def setoption(self, name, value):
        self.send(f"setoption name {name} value {value}")

    def ready(self):
        self.send("isready")
        self.read_until("readyok")

    def quit(self):
        if self.proc.poll() is None:
            self.send("quit")
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


INFO_DEPTH_RE = re.compile(r"\bdepth\s+(\d+)\b")
INFO_MULTIPV_RE = re.compile(r"\bmultipv\s+(\d+)\b")
INFO_SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


def score_to_cp(kind, value):
    value = int(value)
    if kind == "cp":
        return value
    sign = 1 if value > 0 else -1
    return sign * (MATE_CP - min(abs(value), 999))


def parse_stockfish(lines):
    infos = {}
    bestmove = None
    for line in lines:
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
            continue
        depth_match = INFO_DEPTH_RE.search(line)
        score_match = INFO_SCORE_RE.search(line)
        pv_offset = line.find(" pv ")
        if not depth_match or not score_match or pv_offset < 0:
            continue
        pv = line[pv_offset + 4 :].split()
        if not pv:
            continue
        depth = int(depth_match.group(1))
        multipv_match = INFO_MULTIPV_RE.search(line)
        multipv = int(multipv_match.group(1)) if multipv_match else 1
        score_cp = score_to_cp(score_match.group(1), score_match.group(2))
        infos[multipv] = {
            "depth": depth,
            "multipv": multipv,
            "score_cp": score_cp,
            "move": pv[0],
            "pv": pv,
        }
    return bestmove, [infos[key] for key in sorted(infos)]


def stockfish_analyze(sf, fen, depth, multipv=1, searchmoves=None):
    sf.setoption("MultiPV", str(max(1, multipv)))
    sf.ready()
    sf.send(f"position fen {fen}")
    if searchmoves:
        sf.send("go depth {} searchmoves {}".format(depth, " ".join(searchmoves)))
    else:
        sf.send(f"go depth {depth}")
    lines = sf.read_until("bestmove")
    bestmove, infos = parse_stockfish(lines)
    return {"bestmove": bestmove, "infos": infos, "raw": lines[-20:]}


def mine(config_path, run_id, limit=None, depth_override=None, multipv_override=None):
    cfg = read_toml(config_path)
    rd = run_dir_for(run_id)
    rows = [row for row in load_traces(rd) if row.get("source") == "search" and row.get("chosen_move") != "0000"]
    if limit:
        rows = rows[: int(limit)]
    depth = int(depth_override or cfg["stockfish"].get("analysis_depth", 12))
    multipv = int(multipv_override or cfg["stockfish"].get("multipv", 5))
    sf = UciEngine(cfg["stockfish"].get("cmd", "stockfish"))
    for key, value in cfg["stockfish"].get("analysis_options", {}).items():
        sf.setoption(key, value)
    sf.setoption("Threads", cfg["stockfish"].get("options", {}).get("Threads", "1"))
    sf.setoption("Hash", cfg["stockfish"].get("options", {}).get("Hash", "128"))
    sf.ready()

    candidates = []
    seen = set()
    try:
        analyzed = 0
        skipped_duplicates = 0
        for idx, row in enumerate(rows, start=1):
            fen = row["fen"]
            move = row["chosen_move"]
            key = (fen, move)
            if key in seen:
                skipped_duplicates += 1
                continue
            seen.add(key)
            top = stockfish_analyze(sf, fen, depth, multipv=multipv)
            chosen = stockfish_analyze(sf, fen, depth, multipv=1, searchmoves=[move])
            analyzed += 1
            if analyzed % 10 == 0:
                print(
                    f"analyzed={analyzed} duplicates={skipped_duplicates} "
                    f"trace_rows={idx}/{len(rows)} candidates={len(candidates)}",
                    file=sys.stderr,
                    flush=True,
                )
                write_jsonl(rd / "candidates.partial.jsonl", sorted(candidates, key=lambda item: item["severity"], reverse=True))
            if not top["infos"] or not chosen["infos"]:
                continue
            top_info = top["infos"][0]
            chosen_info = chosen["infos"][0]
            if top_info["move"] == move:
                continue
            gap = top_info["score_cp"] - chosen_info["score_cp"]
            if gap < 150:
                continue
            severity = gap
            if abs(top_info["score_cp"]) >= 90_000 or abs(chosen_info["score_cp"]) >= 90_000:
                severity += 20_000
            candidates.append({
                "schema": 1,
                "candidate_id": f"C-{len(candidates)+1:04d}",
                "run_id": run_id,
                "fen": fen,
                "side": row.get("side"),
                "ember_move": move,
                "ember_score_cp": row.get("score_cp"),
                "ember_depth": row.get("depth_reached"),
                "trace_file": row.get("trace_file"),
                "trace_line": row.get("trace_line"),
                "stockfish_depth": depth,
                "stockfish_best_move": top_info["move"],
                "stockfish_best_score_cp": top_info["score_cp"],
                "stockfish_chosen_score_cp": chosen_info["score_cp"],
                "eval_loss_cp": gap,
                "severity": severity,
                "multipv": top["infos"],
                "chosen_pv": chosen_info["pv"],
            })
    finally:
        sf.quit()

    candidates.sort(key=lambda row: row["severity"], reverse=True)
    for idx, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"C-{idx:04d}"
    write_jsonl(rd / "candidates.jsonl", candidates)
    write_candidates_md(rd / "candidates.md", candidates[:50])
    print(json.dumps({"candidates": len(candidates), "depth": depth, "multipv": multipv, "path": str(rd / "candidates.jsonl")}, indent=2))


def write_candidates_md(path, rows):
    lines = [
        "# Stupidity candidates",
        "",
        "| Id | Loss cp | Ember | Stockfish | FEN |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        fen = row["fen"].replace("|", " ")
        lines.append(f"| {row['candidate_id']} | {row['eval_loss_cp']} | {row['ember_move']} | {row['stockfish_best_move']} | `{fen}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ember_once(binary, fen, depth, hash_mb="64"):
    cmds = [
        "uci",
        "isready",
        f"setoption name Hash value {hash_mb}",
        "setoption name Book value",
        f"position fen {fen}",
        f"go depth {depth}",
        "quit",
    ]
    proc = subprocess.run(
        [binary],
        input="\n".join(cmds) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    bestmove = None
    infos = []
    for line in proc.stdout.splitlines():
        if line.startswith("bestmove "):
            bestmove = line.split()[1]
        elif line.startswith("info "):
            infos.append(line)
    return {"bestmove": bestmove, "stdout": proc.stdout, "infos": infos}


def mirror_fen(fen):
    if chess is None:
        return None
    try:
        return chess.Board(fen).mirror().fen()
    except Exception:
        return None


def verify(config_path, run_id, candidate_id=None, top=5, repeats=3, create_case=False):
    cfg = read_toml(config_path)
    rd = run_dir_for(run_id)
    candidates = read_jsonl(rd / "candidates.jsonl")
    if candidate_id:
        candidates = [row for row in candidates if row["candidate_id"] == candidate_id]
    else:
        candidates = candidates[: int(top)]
    if not candidates:
        raise RuntimeError("no candidates to verify")

    sf = UciEngine(cfg["stockfish"].get("cmd", "stockfish"))
    sf.setoption("Threads", cfg["stockfish"].get("options", {}).get("Threads", "1"))
    sf.setoption("Hash", cfg["stockfish"].get("options", {}).get("Hash", "128"))
    sf.ready()
    verified = []
    try:
        for row in candidates:
            fen = row["fen"]
            depth = int(cfg["stockfish"].get("verify_depth", 8))
            ember_runs = [run_ember_once(cfg["ember"]["binary"], fen, depth, cfg["ember"].get("options", {}).get("Hash", "64")) for _ in range(int(repeats))]
            exact_moves = [run["bestmove"] for run in ember_runs]
            exact_repeats_bad = exact_moves.count(row["ember_move"])
            fresh_top = stockfish_analyze(sf, fen, int(cfg["stockfish"].get("analysis_depth", 12)), multipv=int(cfg["stockfish"].get("multipv", 5)))
            mirror = mirror_fen(fen)
            mirror_result = None
            if mirror:
                mirror_ember = run_ember_once(cfg["ember"]["binary"], mirror, depth, cfg["ember"].get("options", {}).get("Hash", "64"))
                mirror_top = stockfish_analyze(sf, mirror, int(cfg["stockfish"].get("analysis_depth", 12)), multipv=3)
                mirror_chosen = stockfish_analyze(sf, mirror, int(cfg["stockfish"].get("analysis_depth", 12)), multipv=1, searchmoves=[mirror_ember["bestmove"]])
                mirror_gap = None
                if mirror_top["infos"] and mirror_chosen["infos"]:
                    mirror_gap = mirror_top["infos"][0]["score_cp"] - mirror_chosen["infos"][0]["score_cp"]
                mirror_result = {
                    "fen": mirror,
                    "ember_move": mirror_ember["bestmove"],
                    "stockfish_best_move": mirror_top["infos"][0]["move"] if mirror_top["infos"] else None,
                    "eval_loss_cp": mirror_gap,
                }
            item = dict(row)
            item.update({
                "verified_at": now_utc(),
                "ember_replay_depth": depth,
                "exact_replay_moves": exact_moves,
                "exact_repeats_bad": exact_repeats_bad,
                "fresh_stockfish": fresh_top["infos"],
                "mirror": mirror_result,
            })
            verified.append(item)
            if create_case:
                write_case(item)
    finally:
        sf.quit()
    write_jsonl(rd / "verified.jsonl", verified)
    print(json.dumps({"verified": len(verified), "path": str(rd / "verified.jsonl")}, indent=2))


def write_case(row):
    case_id = "S-" + row["candidate_id"].split("-")[-1]
    case_dir = Path("stupidities") / "cases" / case_id
    case = {
        "schema": 1,
        "case_id": case_id,
        "title": f"Ember chooses {row['ember_move']} instead of {row['stockfish_best_move']}",
        "status": "candidate",
        "fen": row["fen"],
        "side": row.get("side"),
        "bad_move": row["ember_move"],
        "reference_best_move": row["stockfish_best_move"],
        "reference_eval_loss_cp": row["eval_loss_cp"],
        "stockfish_depth": row["stockfish_depth"],
        "stockfish_best_score_cp": row["stockfish_best_score_cp"],
        "stockfish_chosen_score_cp": row["stockfish_chosen_score_cp"],
        "reference_pv": row.get("multipv", [{}])[0].get("pv", []),
        "chosen_pv": row.get("chosen_pv", []),
        "exact_replay_moves": row.get("exact_replay_moves", []),
        "mirror": row.get("mirror"),
        "fix": {},
    }
    write_json(case_dir / "case.json", case)


def render_case(case_path):
    if chess is None or cairosvg is None:
        raise RuntimeError("render needs python chess and cairosvg")
    case_path = Path(case_path)
    if case_path.is_dir():
        case_path = case_path / "case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case_dir = case_path.parent
    demo_dir = case_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    render_position(
        case.get("case_id", "case"),
        case["fen"],
        demo_dir,
        "base",
        case.get("bad_move"),
        case.get("reference_best_move"),
        case.get("fix", {}).get("fixed_move"),
    )
    for idx, variant in enumerate(case.get("variants", []), start=1):
        render_position(
            f"{case.get('case_id', 'case')} variant {idx}",
            variant["fen"],
            demo_dir,
            f"variant-{idx}",
            variant.get("bad_move"),
            variant.get("reference_best_move"),
            variant.get("fixed_move"),
        )


def render_position(case_id, fen, demo_dir, prefix, bad_move, reference_move, fixed_move):
    render_line(case_id, fen, demo_dir, f"{prefix}-before-bad", bad_move, "Bad Ember move", "#d95f02")
    render_line(case_id, fen, demo_dir, f"{prefix}-reference", reference_move, "Reference move", "#1b9e77")
    if fixed_move:
        render_line(case_id, fen, demo_dir, f"{prefix}-after-fixed", fixed_move, "Fixed Ember move", "#377eb8")


def render_line(case_id, fen, demo_dir, name, move_uci, label, color):
    if not move_uci:
        return
    board = chess.Board(fen)
    frames = []
    frames.append((board.copy(), [], f"{case_id}: position before move"))
    move = chess.Move.from_uci(move_uci)
    arrows = []
    if move in board.legal_moves:
        arrows = [chess.svg.Arrow(move.from_square, move.to_square, color=color)]
        board.push(move)
    frames.append((board.copy(), arrows, f"{label}: {move_uci}"))
    for idx, (frame_board, frame_arrows, frame_label) in enumerate(frames):
        svg = chess.svg.board(frame_board, size=720, arrows=frame_arrows, coordinates=True)
        svg_path = demo_dir / f"{name}-{idx:03d}.svg"
        png_path = demo_dir / f"{name}-{idx:03d}.png"
        svg_path.write_text(svg, encoding="utf-8")
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), output_width=720, output_height=720)
        frames[idx] = (png_path, frame_label)
    concat = demo_dir / f"{name}.concat"
    with open(concat, "w", encoding="utf-8") as f:
        for png_path, _ in frames:
            f.write(f"file '{png_path.name}'\n")
            f.write("duration 2\n")
        f.write(f"file '{frames[-1][0].name}'\n")
    run_cmd(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat.name,
            "-vf", "fps=1,format=yuv420p", f"{name}.mp4",
        ],
        cwd=demo_dir,
        log_path=demo_dir / "render.log",
    )


def report(run_id):
    rd = run_dir_for(run_id)
    candidates = read_jsonl(rd / "candidates.jsonl")
    verified = read_jsonl(rd / "verified.jsonl")
    lines = [
        "# Stupidity hunt report",
        "",
        f"Run id: `{run_id}`",
        f"Generated: {now_utc()}",
        "",
        f"Candidates: `{len(candidates)}`",
        f"Verified: `{len(verified)}`",
        "",
    ]
    if candidates:
        top = candidates[0]
        lines.extend([
            "## Top candidate",
            "",
            f"- Id: `{top['candidate_id']}`",
            f"- FEN: `{top['fen']}`",
            f"- Ember move: `{top['ember_move']}`",
            f"- Reference move: `{top['stockfish_best_move']}`",
            f"- Eval loss: `{top['eval_loss_cp']}` cp",
            "",
        ])
    (rd / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(rd / "report.md")


def all_steps(config_path, run_id, explicit_workers=None, explicit_max_games=None):
    probe(config_path, run_id, explicit_workers)
    build(config_path, run_id)
    smoke(config_path, run_id)
    run_matches(config_path, run_id, explicit_workers, explicit_max_games)
    mine(config_path, run_id)
    verify(config_path, run_id, top=3, repeats=3, create_case=True)
    report(run_id)


def main():
    parser = argparse.ArgumentParser(description="Find and verify high-profile Ember bad decisions.")
    parser.add_argument("command", choices=["probe", "build", "smoke", "run", "mine", "verify", "render", "report", "all"])
    parser.add_argument("--config", default="configs/stupidity/default.toml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--workers", default=None)
    parser.add_argument("--max-games", default=None)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--multipv", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--top", default=5, type=int)
    parser.add_argument("--repeats", default=3, type=int)
    parser.add_argument("--create-case", action="store_true")
    parser.add_argument("--case", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    run_id = args.run_id or make_run_id()
    if args.command == "probe":
        probe(config_path, run_id, args.workers)
    elif args.command == "build":
        build(config_path, run_id)
    elif args.command == "smoke":
        smoke(config_path, run_id)
    elif args.command == "run":
        run_matches(config_path, run_id, args.workers, args.max_games)
    elif args.command == "mine":
        mine(config_path, run_id, args.limit, args.depth, args.multipv)
    elif args.command == "verify":
        verify(config_path, run_id, args.candidate_id, args.top, args.repeats, args.create_case)
    elif args.command == "render":
        if not args.case:
            raise SystemExit("--case is required for render")
        render_case(args.case)
    elif args.command == "report":
        report(run_id)
    elif args.command == "all":
        all_steps(config_path, run_id, args.workers, args.max_games)


if __name__ == "__main__":
    main()

#python3 tools/hunt_stupidities.py all --config configs/stupidity/default.toml --max-games 500 --workers 6
