#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import queue
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

import chess
import chess.pgn


CASE_RE = re.compile(
    r"CcrlCase\s*\{\s*"
    r'label:\s*"(?P<label>[^"]+)",\s*'
    r'bad_move:\s*"(?P<bad_move>[^"]+)",\s*'
    r'history:\s*"(?P<history>[^"]*)",\s*'
    r"\}",
    re.MULTILINE | re.DOTALL,
)

OPTION_RE = re.compile(r"^option name (?P<name>.*?) type\b", re.IGNORECASE)

OPPONENT_BY_GAME = {
    15: "ccrl-seawall-20250322",
    24: "ccrl-pawnstar-0.13.593",
    34: "ccrl-revolver-2.0",
    38: "ccrl-pawnstar-0.13.593",
    46: "ccrl-puffin-5.0",
    60: "ccrl-knightx-4.92",
}


def now_id():
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def load_cases(path):
    text = Path(path).read_text(encoding="utf-8")
    cases = []
    for match in CASE_RE.finditer(text):
        label = match.group("label")
        game_match = re.search(r"CCRL game (\d+)", label)
        if not game_match:
            raise SystemExit(f"cannot find CCRL game number in label: {label}")
        cases.append(
            {
                "game": int(game_match.group(1)),
                "label": label,
                "bad_move": match.group("bad_move"),
                "history": match.group("history").split(),
            }
        )
    if not cases:
        raise SystemExit(f"no CCRL cases found in {path}")
    return cases


def find_meta(meta_dir, game_number):
    matches = sorted(Path(meta_dir).glob(f"{game_number}_*.meta.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected one metadata file for game {game_number}, found {len(matches)}")
    return matches[0]


def engine_color(meta):
    if meta["white"]["name"].startswith("Ember"):
        return chess.WHITE
    if meta["black"]["name"].startswith("Ember"):
        return chess.BLACK
    raise SystemExit("metadata does not contain Ember as white or black")


def initial_board(history):
    board = chess.Board()
    for uci in history:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise SystemExit(f"illegal historical move {uci} at ply {board.ply() + 1}: {board.fen()}")
        board.push(move)
    return board


def clocks_after_history(meta, history_len, initial_ms, increment_ms):
    clocks = {chess.WHITE: initial_ms, chess.BLACK: initial_ms}
    for move in meta["moves"][:history_len]:
        color = chess.WHITE if move["color"] == "w" else chess.BLACK
        used = move.get("time")
        if used is not None:
            clocks[color] -= int(round(float(used) * 1000))
        clocks[color] += increment_ms
    return {color: max(1000, int(value)) for color, value in clocks.items()}


def clock_text(ms):
    ms = max(0, int(ms))
    minutes, rem = divmod(ms // 1000, 60)
    return f"{minutes}:{rem:02d}.{ms % 1000:03d}"


def parse_options(lines):
    options = {}
    for line in lines:
        match = OPTION_RE.match(line)
        if match:
            name = match.group("name").strip()
            options[name.lower()] = name
    return options


class UciEngine:
    def __init__(self, name, command, log):
        self.name = name
        self.command = command
        self.log = log
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.options = {}

    def _read_stdout(self):
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip("\r\n")
            self.log.write(f"{self.name}< {line}\n")
            self.log.flush()
            self.lines.put(line)

    def send(self, command):
        if self.proc.poll() is not None:
            raise RuntimeError(f"{self.name} exited before command: {command}")
        self.log.write(f"{self.name}> {command}\n")
        self.log.flush()
        assert self.proc.stdin is not None
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def read_until(self, predicate, timeout, context):
        deadline = time.monotonic() + timeout
        lines = []
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            lines.append(line)
            if predicate(line):
                return lines
        raise TimeoutError(f"{self.name} timed out waiting for {context}")

    def initialize(self, hash_mb, threads, disable_book):
        self.send("uci")
        lines = self.read_until(lambda line: line == "uciok", 20, "uciok")
        self.options = parse_options(lines)
        self.set_option("Hash", str(hash_mb))
        self.set_option("Threads", str(threads))
        if disable_book:
            self.set_option("Book", "")
            self.set_option("OwnBook", "false")
        self.ready()

    def set_option(self, name, value):
        actual = self.options.get(name.lower())
        if actual is None:
            return
        if value == "":
            self.send(f"setoption name {actual} value")
        else:
            self.send(f"setoption name {actual} value {value}")

    def ready(self):
        self.send("isready")
        self.read_until(lambda line: line == "readyok", 20, "readyok")

    def new_game(self):
        self.send("ucinewgame")
        self.ready()

    def choose_move(self, moves, clocks, increment_ms):
        position = "position startpos"
        if moves:
            position += " moves " + " ".join(moves)
        self.send(position)
        go = (
            f"go wtime {max(1, int(clocks[chess.WHITE]))} "
            f"btime {max(1, int(clocks[chess.BLACK]))} "
            f"winc {increment_ms} binc {increment_ms}"
        )
        start = time.monotonic()
        self.send(go)
        side_clock = clocks[chess.WHITE] if len(moves) % 2 == 0 else clocks[chess.BLACK]
        timeout = max(60.0, side_clock / 1000.0 + increment_ms / 1000.0 + 30.0)
        lines = self.read_until(lambda line: line.startswith("bestmove "), timeout, "bestmove")
        elapsed_ms = int(round((time.monotonic() - start) * 1000))
        best = lines[-1].split()[1]
        return best, elapsed_ms, lines

    def close(self):
        if self.proc.poll() is not None:
            return
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)


def write_pgn(path, meta, moves, result, start_ply, bad_move, replacement_move):
    game = chess.pgn.Game()
    game.headers["Event"] = "CCRL bad-move replay"
    game.headers["Site"] = meta["site"]
    game.headers["Date"] = dt.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = str(start_ply)
    game.headers["White"] = meta["white"]["name"].replace("Ember 1.1.1", "Ember current")
    game.headers["Black"] = meta["black"]["name"].replace("Ember 1.1.1", "Ember current")
    game.headers["Result"] = result
    game.headers["TimeControl"] = "600+10"
    game.headers["OriginalResult"] = meta["result"]
    game.headers["StartPly"] = str(start_ply)
    game.headers["ObservedBadMove"] = bad_move
    game.headers["ReplacementMove"] = replacement_move

    board = chess.Board()
    node = game
    for idx, uci in enumerate(moves, start=1):
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
        if idx == start_ply:
            node.comment = f"Replay replacement for observed {bad_move}."
    path.write_text(str(game) + "\n", encoding="utf-8")


def run_case(case, args, out_dir):
    meta_path = find_meta(args.meta_dir, case["game"])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    board = initial_board(case["history"])
    ember_color = engine_color(meta)
    if board.turn != ember_color:
        raise SystemExit(f"{case['label']}: side to move is not Ember after history")
    if chess.Move.from_uci(case["bad_move"]) not in board.legal_moves:
        raise SystemExit(f"{case['label']}: observed bad move is not legal in the replay position")

    opponent_cmd_name = OPPONENT_BY_GAME[case["game"]]
    opponent_binary = Path(args.opponents_dir) / opponent_cmd_name
    if not opponent_binary.exists():
        raise SystemExit(f"opponent binary not found: {opponent_binary}")
    if not Path(args.ember).exists():
        raise SystemExit(f"Ember binary not found: {args.ember}")

    clocks = clocks_after_history(meta, len(case["history"]), args.initial_ms, args.increment_ms)
    moves = list(case["history"])
    start_ply = len(moves) + 1
    log_path = out_dir / f"{case['game']}_uci.log"
    result = "*"
    termination = "max plies reached"
    replacement = None
    move_records = []

    with log_path.open("w", encoding="utf-8") as log:
        ember = UciEngine("Ember", shlex.split(args.ember), log)
        opponent = UciEngine(opponent_cmd_name, [str(opponent_binary)], log)
        engines = {
            ember_color: ember,
            not ember_color: opponent,
        }
        try:
            for engine in engines.values():
                engine.initialize(args.hash_mb, args.threads, args.disable_book)
                engine.new_game()

            for _ in range(args.max_plies):
                if board.is_game_over(claim_draw=True):
                    result = board.result(claim_draw=True)
                    termination = board.outcome(claim_draw=True).termination.name.lower()
                    break

                color = board.turn
                engine = engines[color]
                before_clock = clocks[color]
                best, elapsed_ms, _ = engine.choose_move(moves, clocks, args.increment_ms)
                if best == "0000":
                    result = "1-0" if color == chess.BLACK else "0-1"
                    termination = f"{engine.name} returned 0000"
                    break
                move = chess.Move.from_uci(best)
                if move not in board.legal_moves:
                    result = "1-0" if color == chess.BLACK else "0-1"
                    termination = f"{engine.name} returned illegal move {best}"
                    break

                board.push(move)
                moves.append(best)
                clocks[color] = clocks[color] - elapsed_ms + args.increment_ms
                if clocks[color] <= -args.timemargin_ms:
                    result = "1-0" if color == chess.BLACK else "0-1"
                    termination = f"{engine.name} lost on time"
                    break

                if replacement is None:
                    replacement = best
                move_records.append(
                    {
                        "ply": len(moves),
                        "engine": engine.name,
                        "move": best,
                        "elapsed_ms": elapsed_ms,
                        "clock_before_ms": before_clock,
                        "clock_after_ms": clocks[color],
                        "fen": board.fen(),
                    }
                )

            else:
                result = "*"
                termination = "max plies reached"
        finally:
            ember.close()
            opponent.close()

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        termination = board.outcome(claim_draw=True).termination.name.lower()

    replacement = replacement or "none"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", case["label"]).strip("_")
    pgn_path = out_dir / f"{case['game']}_{safe}.pgn"
    write_pgn(pgn_path, meta, moves, result, start_ply, case["bad_move"], replacement)

    summary = {
        "game": case["game"],
        "label": case["label"],
        "original_result": meta["result"],
        "white": meta["white"]["name"],
        "black": meta["black"]["name"],
        "ember_color": "white" if ember_color == chess.WHITE else "black",
        "start_ply": start_ply,
        "bad_move": case["bad_move"],
        "replacement_move": replacement,
        "result": result,
        "termination": termination,
        "final_fen": board.fen(),
        "played_plies": len(moves) - len(case["history"]),
        "final_clocks": {
            "white": clock_text(clocks[chess.WHITE]),
            "black": clock_text(clocks[chess.BLACK]),
        },
        "pgn": str(pgn_path),
        "log": str(log_path),
        "moves": move_records,
    }
    (out_dir / f"{case['game']}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"game {case['game']}: {case['bad_move']} -> {replacement}, "
        f"result {result}, {termination}, {summary['played_plies']} plies"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Replay CCRL repetition blunders with current Ember.")
    parser.add_argument("--cases", default="tests/ccrl_repetition_regression.rs")
    parser.add_argument("--meta-dir", default="ratings/ccrl/125th_Amateur_D11/ember_1.1.1/meta")
    parser.add_argument("--ember", default="target/release/ember")
    parser.add_argument("--opponents-dir", default="result/bin")
    parser.add_argument("--output-dir", default=f"ratings/ccrl/replays/{now_id()}")
    parser.add_argument("--game", type=int, action="append", help="only replay this CCRL game number")
    parser.add_argument("--initial-ms", type=int, default=600_000)
    parser.add_argument("--increment-ms", type=int, default=10_000)
    parser.add_argument("--hash-mb", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timemargin-ms", type=int, default=2000)
    parser.add_argument("--max-plies", type=int, default=240)
    parser.add_argument("--disable-book", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.cases)
    if args.game:
        selected = set(args.game)
        cases = [case for case in cases if case["game"] in selected]
    if not cases:
        raise SystemExit("no cases selected")

    summaries = [run_case(case, args, out_dir) for case in cases]
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
