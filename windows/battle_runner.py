#!/usr/bin/env python3
"""Run an explicit, sequential Ember challenge series through lichess-bot."""

from __future__ import annotations

import argparse
import ast
import ctypes
import datetime as dt
import getpass
import hashlib
import itertools
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
import yaml


MAX_THREADS = 256
VALID_COLORS = {"random", "white", "black"}
VALID_MODES = {"casual", "rated"}
VALID_VARIANTS = {
    "standard",
    "chess960",
    "fromPosition",
    "antichess",
    "atomic",
    "crazyhouse",
    "horde",
    "kingOfTheHill",
    "racingKings",
    "threeCheck",
}
INFO_RE = re.compile(r"\binfo\b.*?\bnodes\s+(\d+).*?\bnps\s+(\d+)\b", re.IGNORECASE)
BESTMOVE_RE = re.compile(r"\bbestmove\b", re.IGNORECASE)
RETRY_AT_RE = re.compile(r"\buntil\s+([0-9T:.+\-]+Z?)\b", re.IGNORECASE)
CONTROL_READY_EVENT = "ember_control_ready"
CONTROL_READY_TIMEOUT_SECONDS = 180
TERMINAL_PGN_RESULTS = {"1-0", "0-1", "1/2-1/2"}

BENCHMARK_POSITIONS = [
    ("startpos", "startpos"),
    ("kiwipete", "fen r3k2r/p1ppqpb1/bn2pnp1/2P5/1p2P3/2N2N2/PP1PBPPP/R2QKB1R w KQkq - 0 1"),
    ("sicilian", "fen r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQ1RK1 w - - 0 8"),
    ("queenless", "fen 2r2rk1/1b2bppp/p3pn2/1p1p4/3P4/1BN1PN2/PP3PPP/2R2RK1 w - - 0 14"),
    ("tactical", "fen r2q1rk1/ppp2ppp/2n1bn2/3pp3/1b2P3/2NP1N2/PPPBBPPP/R2Q1RK1 w - - 0 8"),
    ("endgame", "fen 8/2p2pk1/1p4p1/p2Pp3/P1P1P1P1/1P3K2/8/8 w - - 0 40"),
]


@dataclass(frozen=True)
class GameSpec:
    opponents: list[str]
    variant: str
    base_seconds: int
    increment_seconds: int
    mode: str
    color: str
    scoring: bool
    tags: list[str]

    @property
    def opponent(self) -> str:
        """The preferred opponent; retained for old callers and configs."""
        return self.opponents[0]


@dataclass(frozen=True)
class BattleConfig:
    hash_mb: int
    ponder: bool
    book: str
    benchmark_depth: int
    benchmark_repeats: int
    benchmark_timeout_seconds: int
    lichess_url: str
    challenge_timeout_seconds: int
    availability_poll_seconds: int
    opponent_retry_seconds: int
    opponent_wait_timeout_seconds: int
    game_poll_seconds: int
    game_timeout_seconds: int
    games: list[GameSpec]


class RunnerError(RuntimeError):
    pass


class ChallengeRejected(RunnerError):
    def __init__(self, opponent: str, status_code: int, detail: str) -> None:
        self.opponent = opponent
        self.status_code = status_code
        self.detail = detail
        super().__init__(
            f"challenge to {opponent} was rejected by Lichess "
            f"(HTTP {status_code}): {detail}"
        )


class TransientLichessError(RunnerError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def timestamp_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_game_count(value: str) -> int | None:
    normalized = value.strip().upper()
    if normalized == "INF":
        return None
    try:
        count = int(normalized)
    except ValueError as exc:
        raise ValueError("enter a positive integer or INF") from exc
    if count < 1:
        raise ValueError("game count must be positive")
    return count


def prompt_game_count(input_fn: Callable[[str], str] = input) -> int | None:
    while True:
        try:
            return parse_game_count(input_fn("Games to play (positive integer or INF): "))
        except ValueError as exc:
            print(f"Invalid game count: {exc}.")


def scheduled_games(
    templates: list[GameSpec], game_count: int | None
) -> Iterable[tuple[int, GameSpec]]:
    for index, game in enumerate(itertools.cycle(templates), 1):
        if game_count is not None and index > game_count:
            return
        yield index, game


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json_line(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def require_int(mapping: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunnerError(f"{key} must be an integer")
    if not minimum <= value <= maximum:
        raise RunnerError(f"{key} must be in {minimum}..{maximum}")
    return value


def retry_after_seconds(value: str | None) -> int:
    try:
        return max(60, int(value or "60"))
    except ValueError:
        return 60


def retry_at_from_lichess_error(detail: str) -> dt.datetime | None:
    match = RETRY_AT_RE.search(detail)
    if not match:
        return None
    value = match.group(1)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def lichess_error_detail(response: requests.Response) -> str:
    """Extract a bounded, single-line error without exposing request headers."""
    detail: Any = None
    try:
        payload = response.json()
    except (requests.JSONDecodeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("message")
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("error") or json.dumps(
                detail, ensure_ascii=False, sort_keys=True
            )
        if not detail:
            detail = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    elif isinstance(payload, list):
        detail = json.dumps(payload, ensure_ascii=False)

    if not isinstance(detail, str) or not detail.strip():
        detail = response.text or response.reason or "no error details returned"
    detail = " ".join(detail.split())
    return detail[:1000] + ("..." if len(detail) > 1000 else "")


def raise_for_lichess_error(response: requests.Response, action: str) -> None:
    if response.status_code < 400:
        return
    detail = lichess_error_detail(response)
    raise RunnerError(f"{action}: Lichess returned HTTP {response.status_code}: {detail}")


def load_config(path: Path) -> BattleConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RunnerError(f"cannot read {path}: {exc}") from exc

    engine = raw.get("engine")
    benchmark = raw.get("benchmark")
    lichess = raw.get("lichess")
    raw_games = raw.get("games")
    if not isinstance(engine, dict) or not isinstance(benchmark, dict) or not isinstance(lichess, dict):
        raise RunnerError("battle.toml requires [engine], [benchmark], and [lichess] tables")
    if not isinstance(raw_games, list) or not raw_games:
        raise RunnerError("battle.toml requires at least one [[games]] entry")

    hash_mb = require_int(engine, "hash_mb", 1, 4096)
    ponder = engine.get("ponder", True)
    book = engine.get("book", "<embedded>")
    if not isinstance(ponder, bool) or not isinstance(book, str):
        raise RunnerError("engine.ponder must be boolean and engine.book must be a string")

    url = lichess.get("url", "https://lichess.org")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise RunnerError("lichess.url must be an HTTP(S) URL")
    lichess_with_defaults = {
        "availability_poll_seconds": 15,
        "opponent_retry_seconds": 300,
        "opponent_wait_timeout_seconds": 0,
        **lichess,
    }

    games: list[GameSpec] = []
    for index, item in enumerate(raw_games, 1):
        if not isinstance(item, dict):
            raise RunnerError(f"games entry {index} must be a table")
        opponent = item.get("opponent")
        opponents_value = item.get("opponents")
        if opponent is not None and opponents_value is not None:
            raise RunnerError(f"games entry {index}: use opponent or opponents, not both")
        if opponents_value is not None:
            if (
                not isinstance(opponents_value, list)
                or not opponents_value
                or any(not isinstance(value, str) or not value.strip() for value in opponents_value)
            ):
                raise RunnerError(f"games entry {index}: opponents must be a non-empty list of names")
            opponents = [value.strip() for value in opponents_value]
        elif isinstance(opponent, str) and opponent.strip():
            opponents = [opponent.strip()]
        else:
            raise RunnerError(f"games entry {index}: opponent or opponents is required")
        if len({value.casefold() for value in opponents}) != len(opponents):
            raise RunnerError(f"games entry {index}: opponents contains duplicate names")
        variant = item.get("variant", "standard")
        mode = item.get("mode", "casual")
        color = item.get("color", "random")
        scoring = item.get("scoring", False)
        tags = item.get("tags", [])
        if variant not in VALID_VARIANTS:
            raise RunnerError(f"games entry {index}: unsupported variant {variant!r}")
        if mode not in VALID_MODES:
            raise RunnerError(f"games entry {index}: mode must be casual or rated")
        if color not in VALID_COLORS:
            raise RunnerError(f"games entry {index}: color must be random, white, or black")
        if not isinstance(scoring, bool):
            raise RunnerError(f"games entry {index}: scoring must be boolean")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise RunnerError(f"games entry {index}: tags must be a list of strings")
        base_seconds = require_int(item, "base_seconds", 0, 10800)
        if base_seconds not in {0, 15, 30, 45, 60, 90} and base_seconds % 60:
            raise RunnerError(
                f"games entry {index}: base_seconds must be 0, 15, 30, 45, 60, 90, or a multiple of 60"
            )
        games.append(
            GameSpec(
                opponents=opponents,
                variant=variant,
                base_seconds=base_seconds,
                increment_seconds=require_int(item, "increment_seconds", 0, 60),
                mode=mode,
                color=color,
                scoring=scoring,
                tags=tags,
            )
        )

    return BattleConfig(
        hash_mb=hash_mb,
        ponder=ponder,
        book=book,
        benchmark_depth=require_int(benchmark, "depth", 1, 64),
        benchmark_repeats=require_int(benchmark, "repeats", 1, 20),
        benchmark_timeout_seconds=require_int(benchmark, "timeout_seconds", 10, 3600),
        lichess_url=url.rstrip("/"),
        challenge_timeout_seconds=require_int(lichess, "challenge_timeout_seconds", 5, 3600),
        availability_poll_seconds=require_int(
            lichess_with_defaults, "availability_poll_seconds", 5, 300
        ),
        opponent_retry_seconds=require_int(
            lichess_with_defaults, "opponent_retry_seconds", 30, 3600
        ),
        opponent_wait_timeout_seconds=require_int(
            lichess_with_defaults, "opponent_wait_timeout_seconds", 0, 604800
        ),
        game_poll_seconds=require_int(lichess, "game_poll_seconds", 1, 300),
        game_timeout_seconds=require_int(lichess, "game_timeout_seconds", 60, 86400),
        games=games,
    )


def logical_cpu_count() -> int:
    count = 0
    if os.name == "nt":
        try:
            all_processor_groups = 0xFFFF
            count = int(ctypes.windll.kernel32.GetActiveProcessorCount(all_processor_groups))
        except (AttributeError, OSError, ValueError):
            count = 0
    if count < 1:
        count = os.cpu_count() or 1
    return min(count, MAX_THREADS)


def engine_uci_probe(engine: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(engine)],
            input="uci\nisready\nquit\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunnerError(f"cannot start Ember: {exc}") from exc
    output = completed.stdout
    if completed.returncode != 0 or "uciok" not in output or "readyok" not in output:
        raise RunnerError(f"Ember UCI probe failed (exit {completed.returncode})")
    identity = next((line for line in output.splitlines() if line.startswith("id name ")), "id name unknown")
    return {"identity": identity.removeprefix("id name "), "sha256": sha256_file(engine)}


def run_uci_search(
    engine: Path, commands: Iterable[str], timeout: float
) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    assert process.stdout is not None

    lines: queue.Queue[str] = queue.Queue()

    def collect_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)

    reader = threading.Thread(target=collect_stdout, daemon=True)
    reader.start()
    output: list[str] = []
    deadline = time.monotonic() + timeout

    def drain_output() -> None:
        while not lines.empty():
            output.append(lines.get_nowait())

    try:
        for command in commands:
            try:
                process.stdin.write(command + "\n")
            except BrokenPipeError:
                break
        try:
            process.stdin.flush()
        except BrokenPipeError:
            pass

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired([str(engine)], timeout)
            try:
                line = lines.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            output.append(line)
            if line.startswith("bestmove "):
                break

        if process.poll() is None:
            try:
                process.stdin.write("quit\n")
                process.stdin.flush()
            except BrokenPipeError:
                pass
            if process.poll() is None:
                process.wait(timeout=max(1.0, deadline - time.monotonic()))
        reader.join(timeout=1.0)
        drain_output()
    except subprocess.TimeoutExpired as exc:
        if process.poll() is None:
            process.kill()
            process.wait()
        reader.join(timeout=1.0)
        drain_output()
        exc.output = "".join(output)
        raise
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.stdout.close()

    completed = subprocess.CompletedProcess(
        [str(engine)], process.returncode, stdout="".join(output)
    )
    return completed, time.perf_counter() - started


def run_benchmark(engine: Path, config: BattleConfig, threads: int, output_dir: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    raw_log = output_dir / "benchmark-raw.log"

    def search(label: str, position: str, depth: int) -> dict[str, Any]:
        commands = [
            "uci",
            "isready",
            f"setoption name Hash value {config.hash_mb}",
            f"setoption name Threads value {threads}",
            "setoption name Book value",
            "setoption name SyzygyPath value",
            "ucinewgame",
            f"position {position}",
            f"go depth {depth}",
        ]
        try:
            completed, elapsed = run_uci_search(
                engine, commands, config.benchmark_timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            output = exc.output if isinstance(exc.output, str) else ""
            with raw_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"===== {label} depth={depth} timeout={exc.timeout} =====\n{output}\n"
                )
            raise RunnerError(f"benchmark timed out on {label}; see {raw_log}") from exc
        except OSError as exc:
            with raw_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"===== {label} depth={depth} launch-error =====\n"
                    f"{type(exc).__name__}: {exc}\n"
                )
            raise RunnerError(f"benchmark could not run on {label}; see {raw_log}") from exc

        with raw_log.open("a", encoding="utf-8") as stream:
            stream.write(
                f"===== {label} depth={depth} returncode={completed.returncode} =====\n"
                f"{completed.stdout}\n"
            )
        info_matches = INFO_RE.findall(completed.stdout)
        if completed.returncode != 0 or not info_matches or "bestmove " not in completed.stdout:
            raise RunnerError(f"benchmark failed on {label}; see {raw_log}")
        nodes, nps = (int(value) for value in info_matches[-1])
        return {"position": label, "depth": depth, "nodes": nodes, "nps": nps, "wall_seconds": elapsed}

    search("warmup", "startpos", min(5, config.benchmark_depth))
    for repeat in range(1, config.benchmark_repeats + 1):
        for label, position in BENCHMARK_POSITIONS:
            sample = search(label, position, config.benchmark_depth)
            sample["repeat"] = repeat
            samples.append(sample)

    nps_values = sorted(sample["nps"] for sample in samples)
    median = (nps_values[(len(nps_values) - 1) // 2] + nps_values[len(nps_values) // 2]) / 2
    result = {
        "created_at": utc_now(),
        "threads": threads,
        "hash_mb": config.hash_mb,
        "book": "disabled",
        "syzygy": "disabled",
        "depth": config.benchmark_depth,
        "repeats": config.benchmark_repeats,
        "median_nps": median,
        "mean_nps": sum(nps_values) / len(nps_values),
        "min_nps": min(nps_values),
        "max_nps": max(nps_values),
        "samples": samples,
    }
    write_json_atomic(output_dir / "benchmark.json", result)
    return result


def generate_lichess_config(
    template: Path,
    destination: Path,
    engine: Path,
    run_dir: Path,
    config: BattleConfig,
    threads: int,
) -> None:
    value = yaml.safe_load(template.read_text(encoding="utf-8"))
    value["token"] = ""
    value["url"] = config.lichess_url + "/"
    engine_config = value["engine"]
    engine_config["dir"] = str(engine.parent)
    engine_config["name"] = engine.name
    engine_config["working_dir"] = str(engine.parent)
    engine_config["protocol"] = "uci"
    engine_config["debug"] = True
    engine_config["ponder"] = config.ponder
    engine_config["polyglot"]["enabled"] = False
    engine_config["online_moves"]["chessdb_book"]["enabled"] = False
    engine_config["online_moves"]["lichess_cloud_analysis"]["enabled"] = False
    engine_config["online_moves"]["lichess_opening_explorer"]["enabled"] = False
    engine_config["online_moves"]["online_egtb"]["enabled"] = False
    engine_config["lichess_bot_tbs"]["syzygy"]["enabled"] = False
    engine_config["lichess_bot_tbs"]["gaviota"]["enabled"] = False
    engine_config["uci_options"] = {
        "Threads": threads,
        "Hash": config.hash_mb,
        "Book": config.book,
        "SyzygyPath": "",
    }
    value["challenge"]["concurrency"] = 1
    value["challenge"]["allow_list"] = ["__ember_scheduled_outgoing_only__"]
    value["matchmaking"]["allow_matchmaking"] = False
    value["quit_after_all_games_finish"] = True
    value["pgn_directory"] = str(run_dir / "pgn")
    value["pgn_file_grouping"] = "game"
    destination.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")
    rendered = destination.read_text(encoding="utf-8")
    if "token: ''" not in rendered and 'token: ""' not in rendered:
        raise RunnerError("generated lichess-bot config did not retain an empty token")


class LichessClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": "Ember-Windows-Battle/1.0",
            }
        )

    def request(self, method: str, path: str, *, attempts: int = 3, **kwargs: Any) -> requests.Response:
        response: requests.Response | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.request(method, self.base_url + path, **kwargs)
            except requests.RequestException as exc:
                if attempt == attempts:
                    if attempts == 1:
                        raise
                    raise TransientLichessError(
                        f"temporary Lichess connection failure during {method} {path}: {exc}"
                    ) from exc
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"Temporary Lichess connection failure; retrying in {wait_seconds} seconds "
                    f"({attempt}/{attempts})."
                )
                time.sleep(wait_seconds)
                continue
            if response.status_code == 429:
                wait_seconds = retry_after_seconds(response.headers.get("Retry-After"))
                if attempt == attempts:
                    return response
                print(f"Lichess rate limit reached; waiting {wait_seconds} seconds.")
                time.sleep(wait_seconds)
                continue
            if response.status_code in {500, 502, 503, 504}:
                if attempt == attempts:
                    raise TransientLichessError(
                        f"temporary Lichess server error during {method} {path}: "
                        f"HTTP {response.status_code}"
                    )
                wait_seconds = min(2 ** attempt, 30)
                print(
                    f"Temporary Lichess server error HTTP {response.status_code}; retrying in "
                    f"{wait_seconds} seconds ({attempt}/{attempts})."
                )
                time.sleep(wait_seconds)
                continue
            return response
        assert response is not None
        return response

    def account(self) -> dict[str, Any]:
        response = self.request("GET", "/api/account", timeout=30)
        raise_for_lichess_error(response, "authentication check failed")
        return response.json()

    def opponent_status(self, username: str) -> dict[str, Any] | None:
        return self.opponents_status([username]).get(username.casefold())

    def opponents_status(self, usernames: list[str]) -> dict[str, dict[str, Any]]:
        response = self.request(
            "GET",
            "/api/users/status",
            params={"ids": ",".join(usernames), "withGameIds": "true"},
            timeout=30,
        )
        raise_for_lichess_error(response, "could not check opponent availability")
        values = response.json()
        return {
            str(value.get("id") or value.get("name") or "").casefold(): value
            for value in values
            if isinstance(value, dict) and (value.get("id") or value.get("name"))
        }

    def create_challenge(
        self, game: GameSpec, timeout_seconds: int, opponent: str | None = None
    ) -> tuple[str | None, str, dict[str, Any]]:
        opponent = opponent or game.opponent
        deadline = time.monotonic() + timeout_seconds
        payload = {
            "rated": "true" if game.mode == "rated" else "false",
            "clock.limit": str(game.base_seconds),
            "clock.increment": str(game.increment_seconds),
            "color": game.color,
            "variant": game.variant,
            "keepAliveStream": "true",
        }
        response = self.request(
            "POST",
            f"/api/challenge/{opponent}",
            data=payload,
            stream=True,
            timeout=(30, timeout_seconds),
            attempts=1,
        )
        if response.status_code in {400, 404, 409}:
            try:
                detail = lichess_error_detail(response)
            finally:
                response.close()
            raise ChallengeRejected(opponent, response.status_code, detail)
        try:
            raise_for_lichess_error(response, f"could not challenge {opponent}")
        except RunnerError:
            response.close()
            raise
        challenge_id: str | None = None
        initial: dict[str, Any] = {}
        done = "unknown"
        try:
            lines = response.iter_lines(decode_unicode=True)
            while True:
                if time.monotonic() >= deadline:
                    done = "timeout"
                    break
                try:
                    raw_line = next(lines)
                except StopIteration:
                    break
                except requests.RequestException:
                    if time.monotonic() >= deadline:
                        done = "timeout"
                        break
                    raise
                if not raw_line:
                    continue
                message = json.loads(raw_line)
                if not initial and isinstance(message, dict) and "done" not in message:
                    initial = message
                candidate = message.get("id") or message.get("challenge", {}).get("id")
                if isinstance(candidate, str):
                    challenge_id = candidate
                if message.get("done"):
                    done = str(message["done"])
                    break
        finally:
            response.close()
        return challenge_id, done, initial

    def cancel_challenge(self, challenge_id: str) -> None:
        response = self.request("POST", f"/api/challenge/{challenge_id}/cancel", timeout=30)
        if response.status_code not in {200, 404}:
            raise_for_lichess_error(response, f"could not cancel challenge {challenge_id}")

    def ongoing_games(self) -> list[dict[str, Any]]:
        response = self.request("GET", "/api/account/playing", timeout=30)
        raise_for_lichess_error(response, "could not check the bot's ongoing games")
        value = response.json()
        games = value.get("nowPlaying", []) if isinstance(value, dict) else []
        return [game for game in games if isinstance(game, dict)]

def parse_search_nps(lines: Iterable[str]) -> dict[str, Any]:
    final_searches: list[tuple[int, int]] = []
    last_info: tuple[int, int] | None = None
    for line in lines:
        match = INFO_RE.search(line)
        if match:
            last_info = (int(match.group(1)), int(match.group(2)))
        if BESTMOVE_RE.search(line) and last_info is not None:
            final_searches.append(last_info)
            last_info = None
    total_nodes = sum(nodes for nodes, _ in final_searches)
    inferred_seconds = sum(nodes / nps for nodes, nps in final_searches if nps > 0)
    return {
        "searches": len(final_searches),
        "total_nodes": total_nodes,
        "weighted_nps": total_nodes / inferred_seconds if inferred_seconds > 0 else None,
        "final_info": [{"nodes": nodes, "nps": nps} for nodes, nps in final_searches],
    }


def wait_for_opponent_and_challenge(
    client: LichessClient,
    game: GameSpec,
    config: BattleConfig,
    bot_process: subprocess.Popen[Any],
    progress: str,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    rng: random.Random | random.SystemRandom | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = (
        started + config.opponent_wait_timeout_seconds
        if config.opponent_wait_timeout_seconds > 0
        else None
    )
    cooldowns: dict[str, dt.datetime] = {}
    attempts: list[dict[str, Any]] = []
    previous_summary = ""

    def emit(value: dict[str, Any]) -> None:
        if event_sink is not None:
            event_sink({"at": utc_now(), **value})

    while deadline is None or time.monotonic() < deadline:
        if bot_process.poll() is not None:
            raise RunnerError(f"lichess-bot stopped unexpectedly with exit code {bot_process.returncode}")

        try:
            statuses = client.opponents_status(game.opponents)
        except (TransientLichessError, requests.RequestException) as exc:
            print(f"{progress} Temporary error checking opponents: {exc}; continuing to wait.")
            emit({"status": "AVAILABILITY_CHECK_RETRY", "error": str(exc)})
            time.sleep(config.availability_poll_seconds)
            continue

        now = dt.datetime.now(dt.timezone.utc)
        ready: list[str] = []
        availability: list[str] = []
        for opponent in game.opponents:
            key = opponent.casefold()
            status = statuses.get(key)
            retry_at = cooldowns.get(key)
            if retry_at is not None and retry_at > now:
                availability.append(f"{opponent}=cooldown-until-{retry_at.isoformat()}")
            elif status is None or not status.get("online", False):
                availability.append(f"{opponent}=offline")
            elif status.get("playing", False) or status.get("playingId"):
                availability.append(f"{opponent}=busy")
            else:
                availability.append(f"{opponent}=ready")
                ready.append(opponent)

        (rng or random.SystemRandom()).shuffle(ready)
        for opponent in ready:
            print(f"{progress} Challenging {opponent}...")
            attempt: dict[str, Any] = {"opponent": opponent, "started_at": utc_now()}
            try:
                challenge_id, outcome, challenge = client.create_challenge(
                    game, config.challenge_timeout_seconds, opponent=opponent
                )
            except ChallengeRejected as exc:
                retry_at = retry_at_from_lichess_error(exc.detail)
                if retry_at is None or retry_at <= now:
                    retry_at = now + dt.timedelta(seconds=config.opponent_retry_seconds)
                cooldowns[opponent.casefold()] = retry_at
                attempt.update(
                    status="REJECTED",
                    finished_at=utc_now(),
                    http_status=exc.status_code,
                    error=exc.detail,
                    retry_at=retry_at.isoformat(),
                )
                attempts.append(attempt)
                emit({"status": "CHALLENGE_REJECTED", **attempt})
                print(f"  {exc}; trying another opponent.")
                continue
            except requests.RequestException as exc:
                # A challenge POST is not safe to repeat immediately: it may
                # have been accepted before its streaming response broke.
                print(f"  Challenge connection was interrupted: {exc}; checking for an active game.")
                time.sleep(5)
                active_game: dict[str, Any] | None = None
                try:
                    for candidate in client.ongoing_games():
                        opponent_value = candidate.get("opponent", {})
                        username = (
                            str(opponent_value.get("username", ""))
                            if isinstance(opponent_value, dict)
                            else str(opponent_value)
                        )
                        if username.casefold() == opponent.casefold():
                            active_game = candidate
                            break
                except (TransientLichessError, requests.RequestException):
                    active_game = None
                if active_game is not None:
                    challenge_id = str(active_game.get("gameId") or "")
                    if not challenge_id:
                        active_game = None
                if active_game is not None:
                    outcome, challenge = "accepted", {}
                    attempt.update(
                        status="ACCEPTED_AFTER_CONNECTION_ERROR",
                        finished_at=utc_now(),
                        challenge_id=challenge_id,
                        error=str(exc),
                    )
                    attempts.append(attempt)
                    emit({"status": "CHALLENGE_ACCEPTED", **attempt})
                    return {
                        "accepted": True,
                        "opponent": opponent,
                        "challenge_id": challenge_id,
                        "outcome": outcome,
                        "challenge": challenge,
                        "attempts": attempts,
                        "waited_seconds": time.monotonic() - started,
                    }
                retry_at = now + dt.timedelta(seconds=config.opponent_retry_seconds)
                cooldowns[opponent.casefold()] = retry_at
                attempt.update(
                    status="CONNECTION_ERROR",
                    finished_at=utc_now(),
                    error=str(exc),
                    retry_at=retry_at.isoformat(),
                )
                attempts.append(attempt)
                emit({"status": "CHALLENGE_CONNECTION_ERROR", **attempt})
                continue

            attempt.update(
                status=outcome.upper(),
                finished_at=utc_now(),
                challenge_id=challenge_id,
            )
            attempts.append(attempt)
            emit({"status": f"CHALLENGE_{outcome.upper()}", **attempt})
            if outcome == "accepted" and challenge_id:
                return {
                    "accepted": True,
                    "opponent": opponent,
                    "challenge_id": challenge_id,
                    "outcome": outcome,
                    "challenge": challenge,
                    "attempts": attempts,
                    "waited_seconds": time.monotonic() - started,
                }
            if challenge_id and outcome == "timeout":
                client.cancel_challenge(challenge_id)
            cooldowns[opponent.casefold()] = now + dt.timedelta(
                seconds=config.opponent_retry_seconds
            )
            print(f"  Challenge was not accepted ({outcome}); trying another opponent.")

        summary = ", ".join(availability)
        if summary != previous_summary:
            print(
                f"{progress} No opponent is currently available; waiting "
                f"{config.availability_poll_seconds}s. {summary}"
            )
            emit({"status": "WAITING_FOR_OPPONENT", "availability": availability})
            previous_summary = summary
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            break
        time.sleep(
            min(config.availability_poll_seconds, remaining)
            if remaining is not None
            else config.availability_poll_seconds
        )

    return {
        "accepted": False,
        "opponent": None,
        "challenge_id": None,
        "outcome": "availability_timeout",
        "challenge": {},
        "attempts": attempts,
        "waited_seconds": time.monotonic() - started,
    }


def lichess_bot_literal(line: str, marker: str) -> dict[str, Any] | None:
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    try:
        value = ast.literal_eval(line[marker_index + len(marker) :].strip())
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def lichess_bot_event(line: str) -> dict[str, Any] | None:
    return lichess_bot_literal(line, "Event: ")


def lichess_bot_game_state(line: str) -> dict[str, Any] | None:
    return lichess_bot_literal(line, "Game state: ")


PGN_HEADER_RE = re.compile(r'^\[([^ ]+)\s+"(.*)"\]$')


def pgn_headers(pgn: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in pgn.splitlines():
        match = PGN_HEADER_RE.match(line)
        if match:
            headers[match.group(1)] = match.group(2)
        elif headers and line.strip():
            break
    return headers


def completed_local_pgn(pgn_dir: Path, game_id: str) -> tuple[Path, str] | None:
    for path in sorted(pgn_dir.glob(f"*{game_id}*.pgn")):
        try:
            pgn = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        headers = pgn_headers(pgn)
        if headers.get("Result") in TERMINAL_PGN_RESULTS:
            return path, pgn
    return None


def normalized_game_result(game_id: str, game: dict[str, Any], pgn: str) -> dict[str, Any]:
    headers = pgn_headers(pgn)
    status_value = game.get("status")
    status = status_value.get("name") if isinstance(status_value, dict) else status_value
    winner = {"1-0": "white", "0-1": "black"}.get(headers.get("Result", ""))
    return {
        "id": game_id,
        "status": status or "unknown",
        "winner": winner,
        "game": game,
        "pgn_headers": headers,
    }


def wait_for_bot_ready(
    bot_log: Path,
    bot_process: subprocess.Popen[Any],
    timeout_seconds: int = CONTROL_READY_TIMEOUT_SECONDS,
    poll_seconds: float = 0.25,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    position = 0
    while time.monotonic() < deadline:
        if bot_log.exists():
            with bot_log.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(position)
                while line := stream.readline():
                    event = lichess_bot_event(line)
                    if event is not None and event.get("type") == CONTROL_READY_EVENT:
                        return
                position = stream.tell()
        if bot_process.poll() is not None:
            raise RunnerError(
                f"lichess-bot stopped before its control stream was ready "
                f"(exit {bot_process.returncode})"
            )
        time.sleep(poll_seconds)
    raise RunnerError(
        "lichess-bot control stream did not become ready; another bot process "
        "may still be using this token"
    )


def wait_for_game(
    bot_log: Path,
    pgn_dir: Path,
    game_id: str,
    bot_process: subprocess.Popen[Any],
    poll_seconds: int,
    timeout_seconds: int,
    log_offset: int = 0,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + timeout_seconds
    final_game: dict[str, Any] | None = None
    terminal_game_state: dict[str, Any] | None = None
    local_game_done_at: float | None = None
    position = log_offset
    while time.monotonic() < deadline:
        if bot_log.exists():
            with bot_log.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(position)
                while line := stream.readline():
                    game_state = lichess_bot_game_state(line)
                    if (
                        game_state is not None
                        and game_state.get("status") != "started"
                    ):
                        terminal_game_state = game_state
                    event = lichess_bot_event(line)
                    if event is None:
                        continue
                    event_game = event.get("game")
                    if not isinstance(event_game, dict):
                        continue
                    event_game_id = str(event_game.get("gameId") or event_game.get("id") or "")
                    if event_game_id != game_id:
                        continue
                    if event.get("type") == "gameFinish":
                        if final_game is None and event_sink is not None:
                            event_sink(
                                {
                                    "at": utc_now(),
                                    "status": "GAME_FINISH_OBSERVED",
                                    "game_id": game_id,
                                }
                            )
                        final_game = event_game
                    elif event.get("type") == "local_game_done" and final_game is None:
                        local_game_done_at = time.monotonic()
                position = stream.tell()

        local_pgn = completed_local_pgn(pgn_dir, game_id)
        if final_game is not None and local_pgn is not None:
            _, pgn = local_pgn
            return normalized_game_result(game_id, final_game, pgn), pgn
        if (
            terminal_game_state is not None
            and local_game_done_at is not None
            and local_pgn is not None
        ):
            _, pgn = local_pgn
            return normalized_game_result(game_id, terminal_game_state, pgn), pgn
        if local_game_done_at is not None and terminal_game_state is None:
            raise RunnerError(
                f"lichess-bot worker for game {game_id} ended without a terminal PGN; "
                "inspect lichess-bot.log for the underlying stream error"
            )
        if (
            local_game_done_at is not None
            and local_pgn is None
            and time.monotonic() - local_game_done_at >= 30
        ):
            raise RunnerError(
                f"lichess-bot did not write the terminal PGN for game {game_id}"
            )
        if bot_process.poll() is not None:
            raise RunnerError(f"lichess-bot stopped unexpectedly with exit code {bot_process.returncode}")
        time.sleep(poll_seconds)
    raise RunnerError(f"game {game_id} did not finish within {timeout_seconds} seconds")


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def print_plan(config: BattleConfig, threads: int, engine_info: dict[str, Any]) -> None:
    print("\nEmber Lichess battle plan")
    print(f"  Engine: {engine_info['identity']}")
    print(f"  Engine SHA-256: {engine_info['sha256']}")
    print(f"  Logical CPU threads: {threads} (automatic; Ember cap {MAX_THREADS})")
    print(f"  Hash: {config.hash_mb} MiB")
    print("  Syzygy: disabled")
    print(f"  Configured game templates: {len(config.games)}; sequential only")
    for index, game in enumerate(config.games, 1):
        score_text = "scoring" if game.scoring else "non-scoring"
        opponents = (
            ", ".join(game.opponents)
            if len(game.opponents) <= 5
            else f"{len(game.opponents)} configured opponents"
        )
        print(
            f"    {index}. [{opponents}]: "
            f"{game.variant} {game.base_seconds}+{game.increment_seconds} "
            f"{game.mode} color={game.color} {score_text}"
        )
    wait_text = (
        "without a timeout"
        if config.opponent_wait_timeout_seconds == 0
        else f"for up to {config.opponent_wait_timeout_seconds}s"
    )
    print(
        f"  Opponent selection: random ready bot from each pool; poll every "
        f"{config.availability_poll_seconds}s and wait {wait_text}"
    )
    print(f"  Challenge accept timeout: {config.challenge_timeout_seconds}s per opponent")
    print("\nThis will use all detected logical CPUs and can make the home machine busy.")
    print("No service, autostart, power, sleep, registry, PATH, or firewall setting will be changed.")


def run(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parent
    config_path = Path(args.config).resolve()
    engine = root / "engine" / "ember.exe"
    lichess_bot_dir = root / "lichess-bot"
    template = lichess_bot_dir / "config.yml.default"
    if not engine.is_file() or not template.is_file():
        raise RunnerError("portable bundle is incomplete: Ember or lichess-bot config template is missing")

    config = load_config(config_path)
    threads = logical_cpu_count()
    engine_info = engine_uci_probe(engine)
    print_plan(config, threads, engine_info)
    if args.dry_run:
        print("\nDry run complete; no token was read and no network request was made.")
        return 0
    if input("\nType YES to benchmark Ember and begin issuing challenges: ").strip() != "YES":
        print("Canceled; no challenge was issued.")
        return 0
    game_count = prompt_game_count()
    token = os.environ.get("LICHESS_BOT_TOKEN") or getpass.getpass("Lichess bot token: ")
    if not token:
        raise RunnerError("no Lichess token supplied")

    run_dir = root / "results" / timestamp_id()
    run_dir.mkdir(parents=True)
    (run_dir / "pgn").mkdir()
    events_path = run_dir / "events.ndjson"
    state_path = run_dir / "state.json"
    state: dict[str, Any] = {
        "phase": "PREPARING",
        "started_at": utc_now(),
        "threads": threads,
        "hash_mb": config.hash_mb,
        "engine": engine_info,
        "requested_games": game_count if game_count is not None else "INF",
        "completed": [],
    }
    write_json_atomic(state_path, state)

    print("Running pre-battle NPS benchmark...")
    state["phase"] = "BENCHMARKING"
    write_json_atomic(state_path, state)
    benchmark = run_benchmark(engine, config, threads, run_dir)
    print(f"Benchmark median NPS: {benchmark['median_nps']:.0f}")

    generated_config = run_dir / "lichess-bot.generated.yml"
    generate_lichess_config(template, generated_config, engine, run_dir, config, threads)
    bot_log = run_dir / "lichess-bot.log"
    bot_console = (run_dir / "lichess-bot-console.log").open("w", encoding="utf-8")
    child_env = os.environ.copy()
    child_env["LICHESS_BOT_TOKEN"] = token
    child_env["PYTHONUNBUFFERED"] = "1"

    client = LichessClient(config.lichess_url, token)
    account = client.account()
    print(f"Authenticated as {account.get('username') or account.get('id') or 'unknown bot'}.")

    state["phase"] = "STARTING_BOT"
    write_json_atomic(state_path, state)
    bot_process = subprocess.Popen(
        [
            sys.executable,
            str(lichess_bot_dir / "lichess-bot.py"),
            "--config",
            str(generated_config),
            "--logfile",
            str(bot_log),
            "-v",
        ],
        cwd=lichess_bot_dir,
        env=child_env,
        stdout=bot_console,
        stderr=subprocess.STDOUT,
    )
    token = ""
    child_env["LICHESS_BOT_TOKEN"] = ""

    interrupted = False
    try:
        wait_for_bot_ready(bot_log, bot_process)
        state["phase"] = "PLAYING"
        write_json_atomic(state_path, state)
        total_text = str(game_count) if game_count is not None else "INF"
        for index, game in scheduled_games(config.games, game_count):
            record: dict[str, Any] = {
                "index": index,
                "game": asdict(game),
                "started_at": utc_now(),
                "status": "WAITING_FOR_OPPONENT",
            }
            append_json_line(events_path, record)
            log_offset = bot_log.stat().st_size if bot_log.exists() else 0
            progress = f"[{index}/{total_text}]"

            def record_runtime_event(value: dict[str, Any]) -> None:
                append_json_line(events_path, {"index": index, **value})

            selection = wait_for_opponent_and_challenge(
                client,
                game,
                config,
                bot_process,
                progress,
                event_sink=record_runtime_event,
            )
            challenge_id = selection["challenge_id"]
            outcome = selection["outcome"]
            challenge = selection["challenge"]
            record.update(
                selected_opponent=selection["opponent"],
                opponent_wait_seconds=selection["waited_seconds"],
                challenge_attempts=selection["attempts"],
                challenge_id=challenge_id,
                challenge=challenge,
                challenge_outcome=outcome,
            )
            if not selection["accepted"] or not challenge_id:
                record.update(status=f"SKIPPED_{outcome.upper()}", finished_at=utc_now())
                print(f"{progress} No opponent became available before the configured timeout; skipped.")
                append_json_line(events_path, record)
                state["completed"].append(record)
                write_json_atomic(state_path, state)
                continue

            game_id = challenge_id
            record.update(status="PLAYING", game_id=game_id, accepted_at=utc_now())
            append_json_line(events_path, record)
            print(f"  Accepted: https://lichess.org/{game_id}")
            final_json, pgn = wait_for_game(
                bot_log,
                run_dir / "pgn",
                game_id,
                bot_process,
                config.game_poll_seconds,
                config.game_timeout_seconds,
                log_offset=log_offset,
                event_sink=record_runtime_event,
            )
            (run_dir / f"game-{index:03d}-{game_id}.json").write_text(
                json.dumps(final_json, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (run_dir / f"game-{index:03d}-{game_id}.pgn").write_text(pgn, encoding="utf-8")
            time.sleep(1)
            if bot_log.exists():
                with bot_log.open("r", encoding="utf-8", errors="replace") as stream:
                    stream.seek(log_offset)
                    nps = parse_search_nps(stream)
            else:
                nps = parse_search_nps([])
            record.update(
                status="FINISHED",
                finished_at=utc_now(),
                lichess_status=final_json.get("status"),
                winner=final_json.get("winner"),
                nps=nps,
            )
            append_json_line(events_path, record)
            state["completed"].append(record)
            write_json_atomic(state_path, state)
            print(f"  Finished: {final_json.get('status')}; weighted game NPS: {nps['weighted_nps']}")
    except KeyboardInterrupt:
        interrupted = True
        state["phase"] = "INTERRUPTED"
        state["finished_at"] = utc_now()
        write_json_atomic(state_path, state)
        print("\nBattle stopped by user; shutting down lichess-bot.")
    finally:
        stop_process(bot_process)
        bot_console.close()

    if not interrupted:
        state["phase"] = "COMPLETE"
        state["finished_at"] = utc_now()
    write_json_atomic(state_path, state)
    summary = {
        "run": run_dir.name,
        "threads": threads,
        "hash_mb": config.hash_mb,
        "benchmark_median_nps": benchmark["median_nps"],
        "requested_games": game_count if game_count is not None else "INF",
        "games": state["completed"],
    }
    write_json_atomic(run_dir / "summary.json", summary)
    outcome = "stopped" if interrupted else "complete"
    print(f"\nBattle {outcome}. Results: {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="battle.toml")
    parser.add_argument("--dry-run", action="store_true", help="validate and print without reading a token")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (RunnerError, requests.RequestException, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
