#!/usr/bin/env python3
"""Audit WDL/DTZ choices made by an engine in one or more PGN files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import chess.syzygy


DEPTH_RE = re.compile(r"/[#-]?(\d+)\b")


@dataclass(frozen=True)
class RootProbe:
    move: chess.Move
    child_wdl: int
    child_dtz: int
    zeroing: bool
    immediate_loss: bool

    @property
    def root_wdl(self) -> int:
        return -self.child_wdl

    @property
    def dtz_key(self) -> tuple[int, int, int]:
        # Mirrors shakmaty-syzygy Tablebase::best_move(): immediate mates,
        # then whether zeroing helps the winning/losing side, then child DTZ.
        return (
            -int(self.immediate_loss),
            int(self.zeroing ^ (self.child_dtz < 0)),
            -self.child_dtz,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--max-pieces", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("pgn", nargs="+", type=Path)
    return parser.parse_args()


def probe_root(
    tablebase: chess.syzygy.Tablebase, board: chess.Board
) -> dict[chess.Move, RootProbe]:
    probes: dict[chess.Move, RootProbe] = {}
    for move in board.legal_moves:
        zeroing = board.is_zeroing(move)
        board.push(move)
        try:
            child_wdl = tablebase.probe_wdl(board)
            child_dtz = tablebase.probe_dtz(board)
            probes[move] = RootProbe(
                move=move,
                child_wdl=child_wdl,
                child_dtz=child_dtz,
                zeroing=zeroing,
                immediate_loss=child_dtz == -1 and board.is_checkmate(),
            )
        finally:
            board.pop()
    return probes


def optimal_root_moves(probes: dict[chess.Move, RootProbe]) -> tuple[int, set[chess.Move]]:
    best_child_wdl = min(probe.child_wdl for probe in probes.values())
    wdl_best = [probe for probe in probes.values() if probe.child_wdl == best_child_wdl]
    best_key = min(probe.dtz_key for probe in wdl_best)
    return -best_child_wdl, {probe.move for probe in wdl_best if probe.dtz_key == best_key}


def wdl_50_from_dtz(dtz: int, halfmoves: int) -> int | None:
    """Conservative WDL50; ±100 is ambiguous when DTZ may be rounded."""
    adjusted = dtz + halfmoves if dtz > 0 else dtz - halfmoves if dtz < 0 else 0
    if adjusted == 0:
        return 0
    if adjusted == 100 or adjusted == -100:
        return None
    if adjusted > 100:
        return 1
    if adjusted > 0:
        return 2
    if adjusted < -100:
        return -1
    return -2


def main() -> None:
    args = parse_args()
    pgn_paths = [
        path
        for pattern in args.pgn
        for path in (sorted(pattern.glob("*.pgn")) if pattern.is_dir() else [pattern])
    ]
    stats: Counter[str] = Counter()
    by_piece_count: dict[int, Counter[str]] = {
        count: Counter() for count in range(3, args.max_pieces + 1)
    }
    wdl_downgrades: list[dict[str, object]] = []
    dtz_downgrades: list[dict[str, object]] = []
    failed_conversions: list[dict[str, object]] = []
    probe_errors: list[dict[str, object]] = []
    terminations: Counter[str] = Counter()

    with chess.syzygy.open_tablebase(str(args.tables)) as tablebase:
        for pgn_path in pgn_paths:
            with pgn_path.open(encoding="utf-8", errors="replace") as handle:
                while game := chess.pgn.read_game(handle):
                    stats["games"] += 1
                    game_number = stats["games"]
                    white_is_engine = game.headers.get("White") == args.engine
                    black_is_engine = game.headers.get("Black") == args.engine
                    if not (white_is_engine or black_is_engine):
                        continue

                    result = game.headers.get("Result", "*")
                    if result == "1-0":
                        engine_result = "win" if white_is_engine else "loss"
                    elif result == "0-1":
                        engine_result = "loss" if white_is_engine else "win"
                    elif result == "1/2-1/2":
                        engine_result = "draw"
                    else:
                        engine_result = "unfinished"
                    stats[
                        {
                            "win": "engine_wins",
                            "loss": "engine_losses",
                            "draw": "engine_draws",
                            "unfinished": "unfinished_games",
                        }[engine_result]
                    ] += 1
                    terminations[game.headers.get("Termination", "unspecified")] += 1

                    first_forced_win: dict[str, object] | None = None
                    board = game.board()
                    for ply, node in enumerate(game.mainline(), start=1):
                        move = node.move
                        engine_to_move = (
                            white_is_engine if board.turn == chess.WHITE else black_is_engine
                        )
                        if engine_to_move:
                            stats["engine_moves"] += 1
                            pieces = len(board.piece_map())
                            if 3 <= pieces <= args.max_pieces:
                                stats["eligible_moves"] += 1
                                bucket = by_piece_count[pieces]
                                bucket["eligible_moves"] += 1
                                match = DEPTH_RE.search(node.comment)
                                if match and int(match.group(1)) == 1:
                                    stats["depth1_moves"] += 1
                                    bucket["depth1_moves"] += 1
                                try:
                                    probes = probe_root(tablebase, board)
                                    chosen = probes[move]
                                    best_wdl, optimal_moves = optimal_root_moves(probes)
                                    root_dtz = tablebase.probe_dtz(board)
                                    root_wdl_50 = wdl_50_from_dtz(
                                        root_dtz, board.halfmove_clock
                                    )
                                except (KeyError, ValueError) as exc:
                                    stats["probe_errors"] += 1
                                    bucket["probe_errors"] += 1
                                    probe_errors.append(
                                        {
                                            "file": str(pgn_path),
                                            "game": game_number,
                                            "ply": ply,
                                            "fen": board.fen(),
                                            "move": move.uci(),
                                            "error": repr(exc),
                                        }
                                    )
                                else:
                                    stats["probed_moves"] += 1
                                    bucket["probed_moves"] += 1
                                    stats[f"root_wdl_{best_wdl}"] += 1
                                    bucket[f"root_wdl_{best_wdl}"] += 1
                                    if root_wdl_50 is not None:
                                        stats[f"root_wdl50_{root_wdl_50}"] += 1
                                        bucket[f"root_wdl50_{root_wdl_50}"] += 1
                                    if root_wdl_50 == 2 and first_forced_win is None:
                                        first_forced_win = {
                                            "ply": ply,
                                            "fen": board.fen(),
                                            "move": move.uci(),
                                            "root_dtz": root_dtz,
                                        }

                                    if chosen.root_wdl != best_wdl:
                                        stats["wdl_downgrades"] += 1
                                        bucket["wdl_downgrades"] += 1
                                        wdl_downgrades.append(
                                            {
                                                "file": str(pgn_path),
                                                "game": game_number,
                                                "ply": ply,
                                                "fen": board.fen(),
                                                "move": move.uci(),
                                                "chosen_wdl": chosen.root_wdl,
                                                "best_wdl": best_wdl,
                                                "best_moves": sorted(m.uci() for m in optimal_moves),
                                                "comment": node.comment,
                                            }
                                        )
                                    else:
                                        stats["optimal_wdl_moves"] += 1
                                        bucket["optimal_wdl_moves"] += 1
                                        if move in optimal_moves:
                                            stats["optimal_dtz_moves"] += 1
                                            bucket["optimal_dtz_moves"] += 1
                                        else:
                                            stats["dtz_downgrades"] += 1
                                            bucket["dtz_downgrades"] += 1
                                            dtz_downgrades.append(
                                                {
                                                    "file": str(pgn_path),
                                                    "game": game_number,
                                                    "ply": ply,
                                                    "fen": board.fen(),
                                                    "move": move.uci(),
                                                    "chosen_child_dtz": chosen.child_dtz,
                                                    "chosen_zeroing": chosen.zeroing,
                                                    "best_moves": sorted(
                                                        m.uci() for m in optimal_moves
                                                    ),
                                                    "comment": node.comment,
                                                }
                                            )
                        board.push(move)

                    if first_forced_win is not None and engine_result != "win":
                        stats["failed_forced_win_conversions"] += 1
                        failed_conversions.append(
                            {
                                "file": str(pgn_path),
                                "game": game_number,
                                "result": result,
                                "engine_result": engine_result,
                                "termination": game.headers.get("Termination", "unspecified"),
                                "first_forced_win": first_forced_win,
                            }
                        )

    report = {
        "engine": args.engine,
        "tables": str(args.tables),
        "pgn_files": [str(path) for path in pgn_paths],
        "stats": dict(sorted(stats.items())),
        "by_piece_count": {
            str(count): dict(sorted(counts.items()))
            for count, counts in by_piece_count.items()
        },
        "terminations": dict(sorted(terminations.items())),
        "wdl_downgrades": wdl_downgrades,
        "dtz_downgrades": dtz_downgrades,
        "failed_conversions": failed_conversions,
        "probe_errors": probe_errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
