#!/usr/bin/env python3
"""Find positions where Ember fails to preserve an existing advantage.

The harness first lets a strong Stockfish play against Ember from varied book
starts. When Stockfish obtains a large advantage immediately after an Ember
move, the position is replayed with colors swapped: Ember receives the
advantage and Stockfish defends with a larger movetime. Cases where Ember's
advantage collapses are written as disabled TSV fixture rows for later triage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import chess.polyglot


STARTPOS_FEN = chess.STARTING_FEN
FIXTURE_HEADER = (
    "id\tdepth\tfen_before_blunder\tsetup_move\texpected_move"
    "\tthemes\trating\tpopularity\tplays"
)
MATE_CP = 100_000


@dataclass
class SearchResult:
    bestmove: str
    score_cp: int | None
    depth: int | None
    nodes: int | None
    elapsed_s: float
    infos: list[str] = field(default_factory=list)


@dataclass
class Anchor:
    seed: int
    source_game_index: int
    opening_plies: int
    stockfish_color: bool
    history_after_blunder: list[str]
    blunder_move: str
    blunder_ply: int
    advantage_cp: int
    previous_advantage_cp: int | None
    fen_after_blunder: str


@dataclass
class LostAdvantageCase:
    case_id: str
    seed: int
    source_game_index: int
    stockfish_color: str
    anchor_ply: int
    anchor_move: str
    anchor_advantage_cp: int
    previous_advantage_cp: int | None
    replay_start_fen: str
    bad_ply: int
    bad_fen: str
    bad_move: str
    bad_depth: int
    bad_score_cp_before: int | None
    bad_score_cp_after: int | None
    stockfish_best_move: str | None
    stockfish_best_score_cp: int | None
    eval_loss_cp: int | None
    replay_result: str
    termination: str
    bucket: str
    history_before_bad: list[str]
    replay_moves: list[str]
    source_pgn: str
    replay_pgn: str


class UciEngine:
    def __init__(self, label: str, cmd: str, options: dict[str, str], log_path: Path):
        self.label = label
        self.log = log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok", 30.0)
        for name, value in options.items():
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.wait_for("readyok", 30.0)

    def send(self, line: str) -> None:
        self.log.write(f"> {line}\n")
        self.log.flush()
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def readline(self, timeout: float) -> str:
        assert self.proc.stdout is not None
        start = time.monotonic()
        while True:
            if self.proc.poll() is not None:
                rest = self.proc.stdout.read() or ""
                if rest:
                    self.log.write(rest)
                raise RuntimeError(f"{self.label} exited with code {self.proc.returncode}")
            line = self.proc.stdout.readline()
            if line:
                self.log.write(line)
                self.log.flush()
                return line.rstrip("\n")
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"timed out reading from {self.label}")

    def wait_for(self, token: str, timeout: float) -> None:
        start = time.monotonic()
        while True:
            line = self.readline(max(0.1, timeout - (time.monotonic() - start)))
            if line == token or line.endswith(token):
                return
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"timed out waiting for {token} from {self.label}")

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 30.0)

    def search(
        self,
        moves: list[str],
        movetime_ms: int,
        searchmoves: list[str] | None = None,
    ) -> SearchResult:
        self.send("position startpos moves " + " ".join(moves))
        command = f"go movetime {movetime_ms}"
        if searchmoves:
            # Stockfish's UCI parser consumes every token after searchmoves as
            # a move, so limits such as movetime must come first.
            command += " searchmoves " + " ".join(searchmoves)
        self.send(command)
        infos: list[str] = []
        start = time.monotonic()
        timeout = max(30.0, movetime_ms / 1000.0 + 20.0)
        while True:
            line = self.readline(timeout)
            if line.startswith("info "):
                infos.append(line)
            elif line.startswith("bestmove "):
                return SearchResult(
                    bestmove=line.split()[1],
                    score_cp=parse_last_score(infos),
                    depth=parse_last_number(infos, "depth"),
                    nodes=parse_last_number(infos, "nodes"),
                    elapsed_s=time.monotonic() - start,
                    infos=infos,
                )

    def close(self) -> None:
        try:
            self.send("quit")
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self.log.close()


def parse_last_score(infos: list[str]) -> int | None:
    for line in reversed(infos):
        match = re.search(r" score (cp|mate) (-?\d+)", line)
        if not match:
            continue
        value = int(match.group(2))
        if match.group(1) == "mate":
            return (MATE_CP - min(abs(value), 1000)) * (1 if value > 0 else -1)
        return value
    return None


def parse_last_number(infos: list[str], field: str) -> int | None:
    for line in reversed(infos):
        parts = line.split()
        for left, right in zip(parts, parts[1:]):
            if left == field:
                try:
                    return int(right)
                except ValueError:
                    return None
    return None


def score_for_side(board: chess.Board, score_cp_for_turn: int | None, side: bool) -> int | None:
    if score_cp_for_turn is None:
        return None
    return score_cp_for_turn if board.turn == side else -score_cp_for_turn


def board_from_moves(moves: Iterable[str]) -> chess.Board:
    board = chess.Board()
    for move in moves:
        board.push(chess.Move.from_uci(move))
    return board


def legal_or_none(board: chess.Board, bestmove: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(bestmove)
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def game_result(board: chess.Board) -> tuple[str | None, str | None]:
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0", "checkmate")
    if board.is_stalemate():
        return "1/2-1/2", "stalemate"
    if board.can_claim_fifty_moves() or board.is_fifty_moves():
        return "1/2-1/2", "fifty-move draw"
    if board.can_claim_threefold_repetition():
        return "1/2-1/2", "threefold draw"
    if board.is_insufficient_material():
        return "1/2-1/2", "insufficient material"
    return None, None


def result_for_side(result: str, side: bool) -> str:
    if result == "1/2-1/2" or result == "*":
        return "draw"
    if result == "1-0":
        return "win" if side == chess.WHITE else "loss"
    if result == "0-1":
        return "win" if side == chess.BLACK else "loss"
    return result


def sample_opening(book_path: Path | None, rng: random.Random, min_plies: int, max_plies: int) -> list[str]:
    board = chess.Board()
    target = rng.randint(min_plies, max_plies)
    moves: list[str] = []
    reader = chess.polyglot.open_reader(str(book_path)) if book_path and book_path.exists() else None
    try:
        for _ in range(target):
            if reader is not None:
                entries = list(reader.find_all(board))
                legal_entries = [entry for entry in entries if entry.move in board.legal_moves]
                if legal_entries:
                    total = sum(max(1, entry.weight) for entry in legal_entries)
                    pick = rng.randrange(total)
                    cumulative = 0
                    chosen = legal_entries[0]
                    for entry in legal_entries:
                        cumulative += max(1, entry.weight)
                        if pick < cumulative:
                            chosen = entry
                            break
                    move = chosen.move
                else:
                    move = rng.choice(list(board.legal_moves))
            else:
                move = rng.choice(list(board.legal_moves))
            board.push(move)
            moves.append(move.uci())
            if board.is_game_over(claim_draw=True):
                break
    finally:
        if reader is not None:
            reader.close()
    return moves


def play_source_until_anchor(
    game_index: int,
    seed: int,
    opening: list[str],
    ember_color: bool,
    ember: UciEngine,
    stockfish: UciEngine,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[Anchor | None, list[str], str]:
    board = board_from_moves(opening)
    moves = list(opening)
    previous_advantage: int | None = None
    stockfish_color = not ember_color
    rows = []

    for _ in range(args.source_max_plies):
        result, termination = game_result(board)
        if result is not None:
            break

        engine = ember if board.turn == ember_color else stockfish
        movetime = args.source_ember_ms if board.turn == ember_color else args.source_stockfish_ms
        searched = engine.search(moves, movetime)
        move = legal_or_none(board, searched.bestmove)
        if move is None:
            break

        before = board.fen()
        board.push(move)
        moves.append(move.uci())
        rows.append(
            {
                "ply": len(moves),
                "side": "ember" if board.turn == stockfish_color else "stockfish",
                "move": move.uci(),
                "fen_before": before,
                "depth": searched.depth,
                "score_cp": searched.score_cp,
            }
        )

        if not board.turn == stockfish_color:
            continue

        analyzed = stockfish.search(moves, args.anchor_analysis_ms)
        advantage = score_for_side(board, analyzed.score_cp, stockfish_color)
        if advantage is None:
            continue
        if advantage >= args.anchor_cp and (
            previous_advantage is None
            or previous_advantage < args.anchor_cp
            or advantage - previous_advantage >= args.anchor_swing_cp
        ):
            write_source_pgn(out_dir / f"source-{game_index:05d}.pgn", opening, moves, result="*")
            (out_dir / f"source-{game_index:05d}.json").write_text(
                json.dumps(rows, indent=2), encoding="utf-8"
            )
            return (
                Anchor(
                    seed=seed,
                    source_game_index=game_index,
                    opening_plies=len(opening),
                    stockfish_color=stockfish_color,
                    history_after_blunder=moves.copy(),
                    blunder_move=move.uci(),
                    blunder_ply=len(moves),
                    advantage_cp=advantage,
                    previous_advantage_cp=previous_advantage,
                    fen_after_blunder=board.fen(),
                ),
                moves,
                str(out_dir / f"source-{game_index:05d}.pgn"),
            )
        previous_advantage = advantage

    write_source_pgn(out_dir / f"source-{game_index:05d}.pgn", opening, moves, result="*")
    return None, moves, str(out_dir / f"source-{game_index:05d}.pgn")


def play_replay(
    case_index: int,
    anchor: Anchor,
    ember: UciEngine,
    stockfish: UciEngine,
    args: argparse.Namespace,
    out_dir: Path,
    source_pgn: str,
) -> LostAdvantageCase | None:
    strong_side = anchor.stockfish_color
    board = board_from_moves(anchor.history_after_blunder)
    moves = anchor.history_after_blunder.copy()
    replay_moves: list[str] = []
    rows = []
    last_advantage = anchor.advantage_cp
    first_bad: dict | None = None
    result = "*"
    termination = "max plies reached"

    for _ in range(args.replay_max_plies):
        maybe_result, maybe_termination = game_result(board)
        if maybe_result is not None:
            result = maybe_result
            termination = maybe_termination or "game over"
            break

        if board.turn == strong_side:
            before = board.fen()
            before_moves = moves.copy()
            before_eval = stockfish.search(moves, args.replay_analysis_ms)
            before_advantage = score_for_side(board, before_eval.score_cp, strong_side)
            if before_advantage is not None:
                last_advantage = before_advantage
            searched = ember.search(moves, args.replay_ember_ms)
            engine_name = "ember"
            movetime = args.replay_ember_ms
        else:
            before = board.fen()
            before_moves = moves.copy()
            searched = stockfish.search(moves, args.replay_stockfish_ms)
            engine_name = "stockfish"
            movetime = args.replay_stockfish_ms

        move = legal_or_none(board, searched.bestmove)
        if move is None:
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            termination = f"illegal bestmove {searched.bestmove} by {engine_name}"
            break

        board.push(move)
        moves.append(move.uci())
        replay_moves.append(move.uci())
        row = {
            "ply": len(moves),
            "engine": engine_name,
            "move": move.uci(),
            "fen_before": before,
            "movetime_ms": movetime,
            "depth": searched.depth,
            "score_cp": searched.score_cp,
            "elapsed_s": searched.elapsed_s,
        }
        rows.append(row)

        if engine_name != "ember" or first_bad is not None:
            continue

        after_eval = stockfish.search(moves, args.replay_analysis_ms)
        after_advantage = score_for_side(board, after_eval.score_cp, strong_side)
        if after_advantage is None:
            continue
        row["stockfish_after_cp"] = after_advantage
        lost_enough = last_advantage - after_advantage >= args.replay_drop_cp
        fell_below_floor = after_advantage < args.replay_floor_cp
        if lost_enough and fell_below_floor:
            best = stockfish.search(before_moves, args.label_analysis_ms)
            bad = stockfish.search(before_moves, args.label_analysis_ms, searchmoves=[move.uci()])
            best_score = score_for_side(board_from_moves(before_moves), best.score_cp, strong_side)
            bad_score = score_for_side(board_from_moves(before_moves), bad.score_cp, strong_side)
            first_bad = {
                "bad_ply": len(moves),
                "bad_fen": before,
                "bad_move": move.uci(),
                "bad_depth": searched.depth or args.fixture_depth_fallback,
                "before_cp": last_advantage,
                "after_cp": after_advantage,
                "stockfish_best_move": best.bestmove,
                "stockfish_best_score_cp": best_score,
                "bad_score_cp": bad_score,
                "history_before_bad": before_moves,
            }

    if first_bad is None and result != "*":
        side_result = result_for_side(result, strong_side)
        if side_result != "win":
            first_bad = infer_late_failure(rows, anchor, moves, result, termination, strong_side, args)
            if first_bad is not None:
                label_failure(first_bad, stockfish, strong_side, args)

    replay_pgn = out_dir / f"replay-{case_index:05d}.pgn"
    write_replay_pgn(replay_pgn, anchor.history_after_blunder, moves, result, termination)
    (out_dir / f"replay-{case_index:05d}.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    if result_for_side(result, strong_side) == "win":
        return None
    if first_bad is None:
        return None

    bucket = classify_case(first_bad["bad_fen"], first_bad["bad_move"], first_bad["stockfish_best_move"], result, termination)
    eval_loss = None
    if first_bad.get("stockfish_best_score_cp") is not None and first_bad.get("bad_score_cp") is not None:
        eval_loss = first_bad["stockfish_best_score_cp"] - first_bad["bad_score_cp"]
    return LostAdvantageCase(
        case_id=f"advpres-{case_index:04d}",
        seed=anchor.seed,
        source_game_index=anchor.source_game_index,
        stockfish_color="white" if anchor.stockfish_color == chess.WHITE else "black",
        anchor_ply=anchor.blunder_ply,
        anchor_move=anchor.blunder_move,
        anchor_advantage_cp=anchor.advantage_cp,
        previous_advantage_cp=anchor.previous_advantage_cp,
        replay_start_fen=anchor.fen_after_blunder,
        bad_ply=first_bad["bad_ply"],
        bad_fen=first_bad["bad_fen"],
        bad_move=first_bad["bad_move"],
        bad_depth=max(1, min(64, int(first_bad["bad_depth"]))),
        bad_score_cp_before=first_bad.get("before_cp"),
        bad_score_cp_after=first_bad.get("after_cp"),
        stockfish_best_move=first_bad.get("stockfish_best_move"),
        stockfish_best_score_cp=first_bad.get("stockfish_best_score_cp"),
        eval_loss_cp=eval_loss,
        replay_result=result,
        termination=termination,
        bucket=bucket,
        history_before_bad=first_bad["history_before_bad"],
        replay_moves=replay_moves,
        source_pgn=source_pgn,
        replay_pgn=str(replay_pgn),
    )


def infer_late_failure(
    rows: list[dict],
    anchor: Anchor,
    moves: list[str],
    result: str,
    termination: str,
    strong_side: bool,
    args: argparse.Namespace,
) -> dict | None:
    ember_rows = [row for row in rows if row.get("engine") == "ember"]
    if not ember_rows:
        return None
    selected = first_repeating_ember_row(rows, anchor, moves, strong_side) or ember_rows[-1]
    history_before = moves[: selected["ply"] - 1]
    return {
        "bad_ply": selected["ply"],
        "bad_fen": selected["fen_before"],
        "bad_move": selected["move"],
        "bad_depth": selected.get("depth") or args.fixture_depth_fallback,
        "before_cp": None,
        "after_cp": None,
        "stockfish_best_move": None,
        "stockfish_best_score_cp": None,
        "bad_score_cp": None,
        "history_before_bad": history_before,
        "late_failure_result": result,
        "late_failure_termination": termination,
    }


def first_repeating_ember_row(
    rows: list[dict],
    anchor: Anchor,
    moves: list[str],
    strong_side: bool,
) -> dict | None:
    row_by_ply = {row["ply"]: row for row in rows if row.get("engine") == "ember"}
    board = chess.Board()
    replay_start = len(anchor.history_after_blunder)
    for ply, uci in enumerate(moves, start=1):
        mover = board.turn
        board.push(chess.Move.from_uci(uci))
        if ply <= replay_start or mover != strong_side:
            continue
        row = row_by_ply.get(ply)
        if row is not None and board.is_repetition(2):
            return row
    return None


def label_failure(
    failure: dict,
    stockfish: UciEngine,
    strong_side: bool,
    args: argparse.Namespace,
) -> None:
    board = board_from_moves(failure["history_before_bad"])
    try:
        bad_move = chess.Move.from_uci(failure["bad_move"])
    except ValueError:
        return
    if bad_move not in board.legal_moves:
        return
    best = stockfish.search(failure["history_before_bad"], args.label_analysis_ms)
    bad = stockfish.search(
        failure["history_before_bad"],
        args.label_analysis_ms,
        searchmoves=[failure["bad_move"]],
    )
    failure["stockfish_best_move"] = best.bestmove
    failure["stockfish_best_score_cp"] = score_for_side(board, best.score_cp, strong_side)
    failure["bad_score_cp"] = score_for_side(board, bad.score_cp, strong_side)


def classify_case(fen: str, bad_move: str, stockfish_best: str | None, result: str, termination: str) -> str:
    board = chess.Board(fen)
    piece_count = len(board.piece_map())
    try:
        bad = chess.Move.from_uci(bad_move)
    except ValueError:
        bad = None
    try:
        best = chess.Move.from_uci(stockfish_best) if stockfish_best else None
    except ValueError:
        best = None
    if "threefold" in termination:
        return "repetition-conversion"
    if "fifty" in termination or board.halfmove_clock >= 70:
        return "fifty-move-conversion"
    if piece_count <= 7:
        return "low-material-conversion"
    if best and board.is_capture(best) and (bad is None or not board.is_capture(bad)):
        return "missed-capture-or-tactic"
    if bad and board.gives_check(bad):
        return "checking-move-backfires"
    if bad and board.piece_at(bad.from_square) and board.piece_at(bad.from_square).piece_type == chess.KING:
        return "king-move-conversion"
    return "quiet-advantage-loss"


def write_source_pgn(path: Path, opening: list[str], moves: list[str], result: str) -> None:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "Ember lost-advantage source game"
    game.headers["Result"] = result
    node = game
    for uci in moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    path.write_text(str(game) + "\n", encoding="utf-8")


def write_replay_pgn(path: Path, start_history: list[str], moves: list[str], result: str, termination: str) -> None:
    start = board_from_moves(start_history)
    game = chess.pgn.Game()
    game.setup(start)
    game.headers["Event"] = "Ember advantage-preservation replay"
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    node = game
    board = start.copy(stack=False)
    for uci in moves[len(start_history) :]:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
        board.push(move)
    path.write_text(str(game) + "\n", encoding="utf-8")


def fixture_comment(case: LostAdvantageCase) -> str:
    best = case.stockfish_best_move or "unknown"
    loss = "unknown" if case.eval_loss_cp is None else str(case.eval_loss_cp)
    before = "unknown" if case.bad_score_cp_before is None else str(case.bad_score_cp_before)
    after = "unknown" if case.bad_score_cp_after is None else str(case.bad_score_cp_after)
    return (
        f"# Advantage-preservation hunt {case.case_id}. Source PGN: {case.source_pgn}; "
        f"replay PGN: {case.replay_pgn}. Ember was given a {case.anchor_advantage_cp} cp "
        f"Stockfish-confirmed advantage after original ply {case.anchor_ply} "
        f"({case.anchor_move}), then lost advantage bucket={case.bucket} at replay ply "
        f"{case.bad_ply}: played {case.bad_move}, Stockfish preferred {best}, "
        f"eval before/after={before}/{after}, labeled loss={loss} cp."
    )


def fixture_row(case: LostAdvantageCase) -> str:
    setup = " ".join(case.history_before_bad) if case.history_before_bad else "-"
    themes = f"advantagePreservation,stockfishHunt,{case.bucket}"
    return "\t".join(
        [
            case.case_id,
            str(case.bad_depth),
            STARTPOS_FEN,
            setup,
            "!" + case.bad_move,
            themes,
            "0",
            "0",
            "0",
        ]
    )


def write_fixture(path: Path, cases: list[LostAdvantageCase]) -> None:
    lines = [
        "# Advantage-preservation cases generated by tools/hunt_lost_advantage.py.",
        "# Rows stay disabled until the responsible bucket is understood and fixed.",
        FIXTURE_HEADER,
    ]
    for case in cases:
        lines.extend(["", fixture_comment(case), "# DISABLED: collected for bucket triage; not fixed yet.", "# " + fixture_row(case)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, cases: list[LostAdvantageCase], args: argparse.Namespace) -> None:
    buckets: dict[str, int] = {}
    for case in cases:
        buckets[case.bucket] = buckets.get(case.bucket, 0) + 1
    lines = ["# Lost-advantage hunt", ""]
    lines.append(f"- Cases: {len(cases)}")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Ember: `{args.ember}`")
    lines.append(f"- Stockfish: `{args.stockfish}`")
    lines.append(f"- Threads: Ember={args.ember_threads}, Stockfish={args.stockfish_threads}")
    lines.append(f"- Hash: {args.hash_mb} MiB")
    lines.append(f"- Source movetime: Ember={args.source_ember_ms} ms, Stockfish={args.source_stockfish_ms} ms")
    lines.append(f"- Replay movetime: Ember={args.replay_ember_ms} ms, Stockfish={args.replay_stockfish_ms} ms")
    lines.append(
        f"- Analysis movetime: anchor={args.anchor_analysis_ms} ms, "
        f"replay={args.replay_analysis_ms} ms, label={args.label_analysis_ms} ms"
    )
    lines.append(f"- Anchor threshold: {args.anchor_cp} cp")
    lines.append(f"- Replay drop threshold: {args.replay_drop_cp} cp")
    lines.append("")
    lines.append("## Buckets")
    lines.append("")
    lines.append("| Bucket | Cases |")
    lines.append("| --- | ---: |")
    for bucket, count in sorted(buckets.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Id | Bucket | Anchor cp | Bad move | Stockfish | Loss cp | Result |")
    lines.append("| --- | --- | ---: | --- | --- | ---: | --- |")
    for case in cases:
        loss = "" if case.eval_loss_cp is None else str(case.eval_loss_cp)
        lines.append(
            f"| {case.case_id} | {case.bucket} | {case.anchor_advantage_cp} | "
            f"{case.bad_move} | {case.stockfish_best_move or ''} | {loss} | "
            f"{case.replay_result} {case.termination} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ember", required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--book", default="src/book.bin")
    parser.add_argument("--output-dir", default="results/lost-advantage")
    parser.add_argument("--fixture-output", default="tests/fixtures/advantage_preservation.tsv")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--max-source-games", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opening-min-plies", type=int, default=6)
    parser.add_argument("--opening-max-plies", type=int, default=16)
    parser.add_argument("--source-max-plies", type=int, default=120)
    parser.add_argument("--replay-max-plies", type=int, default=160)
    parser.add_argument("--source-ember-ms", type=int, default=40)
    parser.add_argument("--source-stockfish-ms", type=int, default=120)
    parser.add_argument("--anchor-analysis-ms", type=int, default=80)
    parser.add_argument("--replay-ember-ms", type=int, default=80)
    parser.add_argument("--replay-stockfish-ms", type=int, default=240)
    parser.add_argument("--replay-analysis-ms", type=int, default=80)
    parser.add_argument("--label-analysis-ms", type=int, default=300)
    parser.add_argument("--ember-threads", type=int, default=4)
    parser.add_argument("--stockfish-threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--anchor-cp", type=int, default=350)
    parser.add_argument("--anchor-swing-cp", type=int, default=150)
    parser.add_argument("--replay-drop-cp", type=int, default=250)
    parser.add_argument("--replay-floor-cp", type=int, default=200)
    parser.add_argument("--fixture-depth-fallback", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_root = Path(args.output_dir) / time.strftime("%Y%m%d-%H%M%S")
    games_dir = run_root / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    book = Path(args.book) if args.book else None

    ember_options = {"Threads": str(args.ember_threads), "Hash": str(args.hash_mb), "Book": ""}
    stockfish_options = {"Threads": str(args.stockfish_threads), "Hash": str(args.hash_mb)}
    ember = UciEngine("Ember", args.ember, ember_options, logs_dir / "ember.log")
    stockfish = UciEngine("Stockfish", args.stockfish, stockfish_options, logs_dir / "stockfish.log")

    cases: list[LostAdvantageCase] = []
    seen: set[tuple[str, str]] = set()
    try:
        for game_index in range(1, args.max_source_games + 1):
            if len(cases) >= args.cases:
                break
            ember.new_game()
            stockfish.new_game()
            opening = sample_opening(book, rng, args.opening_min_plies, args.opening_max_plies)
            ember_color = chess.WHITE if game_index % 2 else chess.BLACK
            anchor, _moves, source_pgn = play_source_until_anchor(
                game_index, args.seed, opening, ember_color, ember, stockfish, args, games_dir
            )
            if anchor is None:
                continue
            replay = play_replay(len(cases) + 1, anchor, ember, stockfish, args, games_dir, source_pgn)
            if replay is None:
                continue
            key = (replay.bad_fen, replay.bad_move)
            if key in seen:
                continue
            seen.add(key)
            cases.append(replay)
            print(
                f"case {len(cases)}/{args.cases}: {replay.case_id} "
                f"{replay.bucket} {replay.bad_move} loss={replay.eval_loss_cp}",
                flush=True,
            )
    finally:
        ember.close()
        stockfish.close()

    (run_root / "cases.json").write_text(
        json.dumps([asdict(case) for case in cases], indent=2), encoding="utf-8"
    )
    write_summary(run_root / "summary.md", cases, args)
    write_fixture(Path(args.fixture_output), cases)
    print(f"wrote {len(cases)} cases to {run_root}", flush=True)
    print(f"fixture: {args.fixture_output}", flush=True)
    if len(cases) < args.cases:
        print(f"warning: requested {args.cases} cases but collected {len(cases)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
