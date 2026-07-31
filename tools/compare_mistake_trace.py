#!/usr/bin/env python3
"""Compare a Stockfish witness line with Ember's traced search DAG.

The tool is intended for cases where Ember chose a suspicious root move and a
stronger reference engine provides a concrete alternative. It runs Stockfish to
extract a witness PV, runs an Ember fixed-depth search with
`EMBER_TRACE_SEARCH_DAG` enabled for the suspicious and witness root moves, and
reports whether Ember visited the witness positions or diverged before seeing
them.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


try:
    import chess  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal CI envs.
    chess = None  # type: ignore[assignment]


MATE_CP = 100_000
DEFAULT_FIXTURE = Path("tests/fixtures/advantage_preservation.tsv")


@dataclass(frozen=True)
class FixtureCase:
    case_id: str
    depth: int
    fen_before_blunder: str
    setup_moves: list[str]
    expected_move: str
    themes: list[str]
    rating: int
    popularity: int
    plays: int
    disabled: bool
    line_number: int

    @property
    def forbidden_moves(self) -> list[str]:
        if not self.expected_move.startswith("!"):
            return []
        return [move for move in self.expected_move[1:].split("|") if move]


@dataclass(frozen=True)
class UciSearchResult:
    bestmove: str | None
    score_cp: int | None
    depth: int | None
    nodes: int | None
    pv: list[str]
    elapsed_s: float


@dataclass(frozen=True)
class TraceNode:
    key: str
    fen: str
    data: dict[str, Any]


@dataclass(frozen=True)
class RootDag:
    root: dict[str, Any]
    nodes_by_key: dict[str, TraceNode]


class UciEngine:
    def __init__(
        self,
        label: str,
        command: str,
        log_path: Path,
        options: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.label = label
        self.log = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.send("uci")
        self.wait_for("uciok", 30.0)
        for name, value in (options or {}).items():
            self.send(f"setoption name {name} value {value}")
        self.send("isready")
        self.wait_for("readyok", 30.0)

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def send(self, line: str) -> None:
        self.log.write(f"> {line}\n")
        self.log.flush()
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def readline(self, timeout: float) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.label} exited with code {self.process.returncode}"
                ) from exc
            raise TimeoutError(f"timed out reading from {self.label}") from exc
        if line is None:
            raise RuntimeError(f"{self.label} exited with code {self.process.returncode}")
        self.log.write(line + "\n")
        self.log.flush()
        return line

    def wait_for(self, token: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for {token} from {self.label}")
            line = self.readline(min(remaining, 1.0))
            if line == token or line.endswith(token):
                return

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.send("isready")
        self.wait_for("readyok", 30.0)

    def search(
        self,
        fen: str,
        setup_moves: list[str],
        go: str,
        timeout: float,
        searchmoves: list[str] | None = None,
    ) -> UciSearchResult:
        position = f"position fen {fen}"
        if setup_moves:
            position += " moves " + " ".join(setup_moves)
        self.send(position)
        command = f"go {go}"
        if searchmoves:
            # Stockfish consumes every token after `searchmoves` as a move, so
            # limits such as movetime/depth must come first.
            command += " searchmoves " + " ".join(searchmoves)
        self.send(command)
        infos: list[str] = []
        start = time.monotonic()
        deadline = start + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for bestmove from {self.label}")
            line = self.readline(remaining)
            if line.startswith("info "):
                infos.append(line)
            elif line.startswith("bestmove "):
                parts = line.split()
                bestmove = parts[1] if len(parts) > 1 and parts[1] != "(none)" else None
                return UciSearchResult(
                    bestmove=bestmove,
                    score_cp=parse_last_score(infos),
                    depth=parse_last_number(infos, "depth"),
                    nodes=parse_last_number(infos, "nodes"),
                    pv=parse_last_pv(infos),
                    elapsed_s=time.monotonic() - start,
                )

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.send("quit")
                self.process.wait(timeout=5.0)
        except Exception:
            if self.process.poll() is None:
                self.process.kill()
        finally:
            self.reader.join(timeout=1.0)
            self.log.close()


def require_chess() -> Any:
    if chess is None:
        raise SystemExit(
            "python-chess is required; run this from the repository's "
            "Nix elo-runner shell"
        )
    return chess


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


def parse_last_pv(infos: list[str]) -> list[str]:
    for line in reversed(infos):
        parts = line.split()
        if " pv " not in f" {line} ":
            continue
        try:
            index = parts.index("pv")
        except ValueError:
            continue
        return [part for part in parts[index + 1 :] if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", part)]
    return []


def position_key(fen: str) -> str:
    """Return the part of FEN covered by normal transposition identity."""

    return " ".join(fen.split()[:4])


def parse_fixture_rows(path: Path, include_disabled: bool = True) -> list[FixtureCase]:
    rows: list[FixtureCase] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        disabled = False
        if stripped.startswith("#"):
            if not include_disabled:
                continue
            stripped = stripped[1:].strip()
            disabled = True
            if not stripped or stripped.startswith("DISABLED:") or "\t" not in stripped:
                continue
        if stripped.startswith("id\t"):
            continue
        parts = stripped.split("\t")
        if len(parts) != 9:
            continue
        setup = [] if parts[3] == "-" else parts[3].split()
        rows.append(
            FixtureCase(
                case_id=parts[0],
                depth=int(parts[1]),
                fen_before_blunder=parts[2],
                setup_moves=setup,
                expected_move=parts[4],
                themes=[theme for theme in parts[5].split(",") if theme],
                rating=int(parts[6]),
                popularity=int(parts[7]),
                plays=int(parts[8]),
                disabled=disabled,
                line_number=line_number,
            )
        )
    return rows


def board_for_case(case: FixtureCase) -> Any:
    chess_module = require_chess()
    board = chess_module.Board(case.fen_before_blunder)
    for move_uci in case.setup_moves:
        board.push(chess_module.Move.from_uci(move_uci))
    return board


def move_repeats(case: FixtureCase, move_uci: str) -> bool:
    chess_module = require_chess()
    board = board_for_case(case)
    move = chess_module.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        return False
    board.push(move)
    return board.is_repetition(2)


def score_for_case_side(case: FixtureCase, score_for_turn: int | None) -> int | None:
    if score_for_turn is None:
        return None
    board = board_for_case(case)
    return score_for_turn if board.turn else -score_for_turn


def pv_positions(fen: str, setup_moves: list[str], pv: list[str]) -> list[dict[str, Any]]:
    chess_module = require_chess()
    board = chess_module.Board(fen)
    for move_uci in setup_moves:
        board.push(chess_module.Move.from_uci(move_uci))
    positions: list[dict[str, Any]] = []
    for ply, move_uci in enumerate(pv, start=1):
        move = chess_module.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            positions.append(
                {
                    "ply": ply,
                    "move": move_uci,
                    "fen": board.fen(),
                    "key": position_key(board.fen()),
                    "illegal": True,
                }
            )
            break
        board.push(move)
        fen_after = board.fen()
        positions.append(
            {
                "ply": ply,
                "move": move_uci,
                "fen": fen_after,
                "key": position_key(fen_after),
                "illegal": False,
            }
        )
    return positions


def load_root_dags(path: Path) -> dict[str, RootDag]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    roots = [record for record in records if record.get("type") == "root"]
    latest_by_move: dict[str, dict[str, Any]] = {}
    for root in roots:
        latest_by_move[root["move"]] = root
    dags: dict[str, RootDag] = {}
    for move, root in latest_by_move.items():
        sequence = root["sequence"]
        nodes_by_key: dict[str, TraceNode] = {}
        for record in records:
            if record.get("type") != "node" or record.get("sequence") != sequence:
                continue
            key = position_key(record["fen"])
            nodes_by_key.setdefault(key, TraceNode(key=key, fen=record["fen"], data=record))
        dags[move] = RootDag(root=root, nodes_by_key=nodes_by_key)
    return dags


def summarize_root(dag: RootDag | None) -> dict[str, Any] | None:
    if dag is None:
        return None
    draw_keys = ("search_cycle_returns", "claimable_draw_returns", "automatic_draw_returns")
    draw_totals = {
        key: sum(int(node.data.get(key, 0)) for node in dag.nodes_by_key.values())
        for key in draw_keys
    }
    return {
        "depth": dag.root.get("depth"),
        "move": dag.root.get("move"),
        "score": dag.root.get("score"),
        "searched_nodes": dag.root.get("searched_nodes"),
        "positions": dag.root.get("positions"),
        "edges": dag.root.get("edges"),
        "truncated": dag.root.get("truncated"),
        **draw_totals,
    }


def compare_witness_positions(
    dag: RootDag | None,
    witness_positions: list[dict[str, Any]],
) -> dict[str, Any]:
    if dag is None:
        return {
            "visited_prefix_plies": 0,
            "first_missing": witness_positions[0] if witness_positions else None,
            "positions": [],
        }
    visited_prefix = 0
    rows = []
    first_missing = None
    for position in witness_positions:
        node = dag.nodes_by_key.get(position["key"])
        visited = node is not None
        if visited and first_missing is None:
            visited_prefix = position["ply"]
        elif first_missing is None:
            first_missing = position
        rows.append(
            {
                **position,
                "visited": visited,
                "node": summarize_node(node.data) if node is not None else None,
            }
        )
    return {
        "visited_prefix_plies": visited_prefix,
        "first_missing": first_missing,
        "positions": rows,
    }


def summarize_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "main_visits": node.get("main_visits", 0),
        "qsearch_visits": node.get("qsearch_visits", 0),
        "min_depth": node.get("min_depth", 0),
        "max_depth": node.get("max_depth", 0),
        "eval_visits": node.get("eval_visits", 0),
        "min_eval": node.get("min_eval", 0),
        "max_eval": node.get("max_eval", 0),
        "tt_visits": node.get("tt_visits", 0),
        "tt_exact": node.get("tt_exact", 0),
        "tt_alpha": node.get("tt_alpha", 0),
        "tt_beta": node.get("tt_beta", 0),
        "search_cycle_returns": node.get("search_cycle_returns", 0),
        "claimable_draw_returns": node.get("claimable_draw_returns", 0),
        "automatic_draw_returns": node.get("automatic_draw_returns", 0),
    }


def select_cases(args: argparse.Namespace) -> list[FixtureCase]:
    cases = parse_fixture_rows(args.fixture, include_disabled=True)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case.case_id in wanted]
    if args.bucket:
        cases = [case for case in cases if args.bucket in case.themes]
    if args.disabled_only:
        cases = [case for case in cases if case.disabled]
    if args.direct_repetition_only:
        cases = [
            case
            for case in cases
            if case.forbidden_moves and move_repeats(case, case.forbidden_moves[0])
        ]
    if args.limit is not None:
        cases = cases[: args.limit]
    return cases


def analyze_case(
    case: FixtureCase,
    ember_cmd: str,
    stockfish: UciEngine,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any] | None:
    bad_move = case.forbidden_moves[0] if case.forbidden_moves else None
    if not bad_move:
        return None

    stockfish.new_game()
    best = stockfish.search(
        case.fen_before_blunder,
        case.setup_moves,
        f"movetime {args.stockfish_ms}",
        timeout=max(30.0, args.stockfish_ms / 1000.0 + 20.0),
    )
    if best.bestmove is None:
        return None
    bad_repeats = move_repeats(case, bad_move)
    best_repeats = move_repeats(case, best.bestmove)
    if args.stockfish_nonrepeat_only and best_repeats:
        return None

    bad_analysis = stockfish.search(
        case.fen_before_blunder,
        case.setup_moves,
        f"movetime {args.stockfish_ms}",
        timeout=max(30.0, args.stockfish_ms / 1000.0 + 20.0),
        searchmoves=[bad_move],
    )
    best_analysis = stockfish.search(
        case.fen_before_blunder,
        case.setup_moves,
        f"movetime {args.stockfish_ms}",
        timeout=max(30.0, args.stockfish_ms / 1000.0 + 20.0),
        searchmoves=[best.bestmove],
    )

    depth = args.depth if args.depth is not None else case.depth
    roots = sorted({bad_move, best.bestmove})
    trace_path = out_dir / f"{case.case_id}.ember-dag.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    ember_env = os.environ.copy()
    ember_env.update(
        {
            "EMBER_TRACE_SEARCH_DAG": str(trace_path),
            "EMBER_TRACE_SEARCH_DAG_DEPTH": str(depth),
            "EMBER_TRACE_SEARCH_DAG_ROOTS": ",".join(roots),
            "EMBER_TRACE_SEARCH_DAG_MAX_PLY": str(args.trace_max_ply),
            "EMBER_TRACE_SEARCH_DAG_MAX_POSITIONS": str(args.trace_max_positions),
        }
    )
    ember = UciEngine(
        "ember",
        ember_cmd,
        out_dir / f"{case.case_id}.ember.uci.log",
        options={
            "Threads": str(args.threads),
            "Hash": str(args.hash),
            "Book": "",
        },
        env=ember_env,
    )
    try:
        ember.new_game()
        ember_result = ember.search(
            case.fen_before_blunder,
            case.setup_moves,
            f"depth {depth}",
            timeout=max(60.0, args.ember_timeout),
        )
    finally:
        ember.close()

    dags = load_root_dags(trace_path) if trace_path.exists() else {}
    best_pv = best_analysis.pv or best.pv
    witness_positions = pv_positions(case.fen_before_blunder, case.setup_moves, best_pv)
    bad_positions = pv_positions(case.fen_before_blunder, case.setup_moves, bad_analysis.pv)
    best_dag = dags.get(best.bestmove)
    bad_dag = dags.get(bad_move)

    return {
        "case": asdict(case),
        "depth": depth,
        "bad_move": bad_move,
        "stockfish_best_move": best.bestmove,
        "bad_move_repeats": bad_repeats,
        "stockfish_best_repeats": best_repeats,
        "ember_bestmove": ember_result.bestmove,
        "stockfish": {
            "best": asdict(best),
            "bad_move": asdict(bad_analysis),
            "best_move": asdict(best_analysis),
            "bad_score_for_case_side": score_for_case_side(case, bad_analysis.score_cp),
            "best_score_for_case_side": score_for_case_side(case, best_analysis.score_cp),
        },
        "ember_roots": {
            "bad": summarize_root(bad_dag),
            "stockfish_best": summarize_root(best_dag),
        },
        "witness": {
            "stockfish_best_pv": best_pv,
            "stockfish_best_trace": compare_witness_positions(best_dag, witness_positions),
            "bad_move_pv": bad_analysis.pv,
            "bad_move_trace": compare_witness_positions(bad_dag, bad_positions),
        },
        "artifacts": {
            "ember_dag": str(trace_path),
            "ember_log": str(out_dir / f"{case.case_id}.ember.uci.log"),
        },
    }


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Comparative mistake trace report",
        "",
        "| Case | Bad | SF best | Direct repeat | Ember best | SF cp bad/best | Ember root scores bad/best | Witness visited | First missing |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        case_id = result["case"]["case_id"]
        bad = result["bad_move"]
        best = result["stockfish_best_move"]
        repeat = f"{result['bad_move_repeats']}/{result['stockfish_best_repeats']}"
        ember_best = result["ember_bestmove"]
        sf_bad = result["stockfish"]["bad_score_for_case_side"]
        sf_best = result["stockfish"]["best_score_for_case_side"]
        root_bad = root_score(result["ember_roots"]["bad"])
        root_best = root_score(result["ember_roots"]["stockfish_best"])
        trace = result["witness"]["stockfish_best_trace"]
        visited = f"{trace['visited_prefix_plies']}/{len(trace['positions'])}"
        first = trace["first_missing"]
        first_text = "-" if first is None else f"ply {first['ply']} after {first['move']}"
        lines.append(
            f"| {case_id} | {bad} | {best} | {repeat} | {ember_best} | "
            f"{sf_bad}/{sf_best} | {root_bad}/{root_best} | {visited} | {first_text} |"
        )
    lines.append("")
    lines.append("`Direct repeat` is `bad move repeats / Stockfish best repeats`.")
    lines.append("Scores are from the side to move in the fixture position.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def root_score(root: dict[str, Any] | None) -> str:
    if root is None:
        return "missing"
    return str(root.get("score"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--bucket")
    parser.add_argument("--disabled-only", action="store_true", default=True)
    parser.add_argument("--include-active", action="store_false", dest="disabled_only")
    parser.add_argument("--direct-repetition-only", action="store_true")
    parser.add_argument("--stockfish-nonrepeat-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ember", required=True)
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash", type=int, default=64)
    parser.add_argument("--stockfish-ms", type=int, default=1000)
    parser.add_argument("--trace-max-ply", type=int, default=20)
    parser.add_argument("--trace-max-positions", type=int, default=500_000)
    parser.add_argument("--ember-timeout", type=float, default=120.0)
    args = parser.parse_args()

    require_chess()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args)
    results: list[dict[str, Any]] = []
    stockfish = UciEngine(
        "stockfish",
        args.stockfish,
        args.out_dir / "stockfish.uci.log",
        options={"Threads": "1", "Hash": "256"},
    )
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] tracing {case.case_id}", flush=True)
            result = analyze_case(case, args.ember, stockfish, args, args.out_dir)
            if result is not None:
                results.append(result)
    finally:
        stockfish.close()

    json_path = args.out_dir / "comparative-traces.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_dir / "comparative-traces.md", results)
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
