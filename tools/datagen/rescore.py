#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

CHUNK_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024
DEFAULT_NODES = 5000
DEFAULT_TIMEOUT_S = 60.0

SCORE_CP_RE = re.compile(r"\bscore\s+cp\s+(-?\d+)\b")
SCORE_MATE_RE = re.compile(r"\bscore\s+mate\s+(-?\d+)\b")
BESTMOVE_RE = re.compile(r"\bbestmove\s+([^\s]+)")
READYOK_RE = re.compile(r"\breadyok\b")
UCIOK_RE = re.compile(r"\buciok\b")


def log(msg: str) -> None:
    print(f"[rescore] {msg}", flush=True)


class EmberWorker:
    def __init__(
        self,
        exe: str,
        threads: int,
        nnue_off: bool,
        hash_mb: int,
        nodes: int,
        timeout_s: float,
        worker_id: int,
    ):
        self.exe = exe
        self.threads = threads
        self.nnue_off = nnue_off
        self.hash_mb = hash_mb
        self.nodes = nodes
        self.timeout_s = timeout_s
        self.worker_id = worker_id
        self.lock = threading.RLock()
        self.proc: Optional[subprocess.Popen] = None
        self.read_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self.dead = False
        self._start()

    def _start(self) -> None:
        log(f"worker {self.worker_id}: starting {self.exe} (Threads={self.threads}, "
            f"NNUE={'off' if self.nnue_off else 'on'})")
        self.proc = subprocess.Popen(
            [self.exe],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        assert self.proc.stderr is not None

        threading.Thread(target=self._drain_stderr, daemon=True).start()

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

        self._uci_handshake()

    def _drain_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        try:
            for _ in self.proc.stderr:
                pass
        except (ValueError, OSError):
            pass

    def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        try:
            for line in self.proc.stdout:
                self.read_q.put(line.rstrip("\n"))
        except (ValueError, OSError):
            pass
        finally:
            self.read_q.put(None)

    def _send(self, cmd: str) -> None:
        assert self.proc and self.proc.stdin
        with self.lock:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()

    def _readline(self, timeout: float) -> Optional[str]:
        try:
            return self.read_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _wait_for(self, pattern: re.Pattern, timeout: float) -> Optional[str]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = deadline - time.time()
            line = self._readline(min(remaining, 0.5))
            if line is None:
                continue
            if pattern.search(line):
                return line
        return None

    def _uci_handshake(self) -> None:
        self._send("uci")
        if not self._wait_for(UCIOK_RE, 15.0):
            raise RuntimeError(f"worker {self.worker_id}: no uciok")
        self._send("isready")
        if not self._wait_for(READYOK_RE, 15.0):
            raise RuntimeError(f"worker {self.worker_id}: no readyok")
        if self.nnue_off:
            self._send("setoption name NNUE value")
        self._send(f"setoption name Threads {self.threads}")
        self._send(f"setoption name Hash {self.hash_mb}")
        self._send("isready")
        if not self._wait_for(READYOK_RE, 15.0):
            raise RuntimeError(f"worker {self.worker_id}: no readyok after options")

    def score_fen(self, fen: str):
        if self.dead or self.proc is None:
            return None, None, None, None

        try:
            with self.lock:
                self._send(f"position fen {fen}")
                self._send(f"go nodes {self.nodes}")

            best_score_cp = None
            best_score_mate = None
            best_move = None
            nodes = 0
            bestmove_seen = False

            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                remaining = deadline - time.time()
                line = self._readline(min(remaining, 0.5))
                if line is None:
                    continue

                m = SCORE_CP_RE.search(line)
                if m:
                    best_score_cp = int(m.group(1))
                    best_score_mate = None
                    continue
                m = SCORE_MATE_RE.search(line)
                if m:
                    best_score_mate = int(m.group(1))
                    best_score_cp = None
                    continue

                nm = re.search(r"\bnodes\s+(\d+)", line)
                if nm:
                    nodes = int(nm.group(1))

                bm = BESTMOVE_RE.search(line)
                if bm:
                    best_move = bm.group(1)
                    bestmove_seen = True
                    break

            if not bestmove_seen:
                self.dead = True
                self._kill()
                return None, None, None, None

            if best_score_cp is not None:
                return "cp", best_score_cp, nodes, best_move
            if best_score_mate is not None:
                return "mate", best_score_mate, nodes, best_move
            try:
                with self.lock:
                    self._send(f"position fen {fen}")
                    self._send(f"go nodes {self.nodes}")
            except Exception:
                self.dead = True
                self._kill()
                return None, None, None, None
            deadline2 = time.time() + self.timeout_s
            bestmove_seen2 = False
            while time.time() < deadline2:
                remaining = deadline2 - time.time()
                line = self._readline(min(remaining, 0.5))
                if line is None:
                    continue
                m = SCORE_CP_RE.search(line)
                if m:
                    return "cp", int(m.group(1)), nodes, best_move
                m = SCORE_MATE_RE.search(line)
                if m:
                    return "mate", int(m.group(1)), nodes, best_move
                bm = BESTMOVE_RE.search(line)
                if bm:
                    best_move = bm.group(1)
                    bestmove_seen2 = True
                    break
            if bestmove_seen2:
                return "cp", 0, nodes, best_move
            self.dead = True
            self._kill()
            return None, None, None, None

        except Exception:
            self.dead = True
            self._kill()
            return None, None, None, None

    def _kill(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.kill()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self.proc.kill()
                self.proc.wait()
            except OSError:
                pass

    def restart(self) -> None:
        self._kill()
        self.dead = False
        self.read_q = queue.Queue()
        self._start()


def iter_fens(input_globs: list[str]):
    files = []
    for g in input_globs:
        files.extend(glob.glob(g, recursive=True))
    if not files:
        sys.exit(f"rescore: no input files matched: {input_globs}")
    files.sort()
    for path in files:
        log(f"reading {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, 1):
                fen = line.rstrip("\n")
                if fen:
                    yield path, i, fen


class ChunkWriter:
    def __init__(self, out_dir: Path, chunk_bytes: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_bytes = chunk_bytes
        self.part = 0
        self.current_size = 0
        self.current: Optional[object] = None
        self.written = 0
        self.lock = threading.Lock()
        self._open_next()

    def _open_next(self) -> None:
        path = self.out_dir / f"scored_{self.part:05d}.tsv"
        self.current = open(path, "w", encoding="utf-8")
        self.current_size = 0
        log(f"opened output chunk {path}")

    def write(self, record: str) -> None:
        with self.lock:
            data = record.encode("utf-8")
            if self.current is None or self.current_size >= self.chunk_bytes:
                if self.current is not None:
                    self.current.close()
                self.part += 1
                self._open_next()
            assert self.current is not None
            self.current.write(record)
            self.current.write("\n")
            self.current_size += len(data) + 1
            self.written += 1

    def close(self) -> None:
        with self.lock:
            if self.current is not None:
                self.current.close()
                self.current = None


def worker_loop(
    worker_id: int,
    exe: str,
    threads: int,
    nnue_off: bool,
    hash_mb: int,
    nodes: int,
    timeout_s: float,
    task_queue: "deque[tuple[str, int, str]]",
    queue_lock: threading.Lock,
    writer: ChunkWriter,
    stats: dict,
    stats_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    ember = EmberWorker(exe, threads, nnue_off, hash_mb, nodes, timeout_s, worker_id)
    processed = 0
    failed = 0

    while not stop_event.is_set():
        with queue_lock:
            if not task_queue:
                break
            task = task_queue.popleft()
        src, lineno, fen = task

        score_type, score_val, nodes, best_move = ember.score_fen(fen)
        if score_type is None:
            failed += 1
            try:
                ember.restart()
            except Exception:
                log(f"worker {worker_id}: engine restart failed, giving up")
                break
            score_type, score_val, nodes, best_move = ember.score_fen(fen)
            if score_type is None:
                failed += 1
                with stats_lock:
                    stats["failed"] = stats.get("failed", 0) + 1
                log(f"worker {worker_id}: FAILED position {src}:{lineno}")
                continue

        bm = best_move if best_move else "0000"
        record = f"{fen}\t{score_type}\t{score_val}\t{bm}"
        writer.write(record)
        processed += 1
        with stats_lock:
            stats["done"] = stats.get("done", 0) + 1
        if processed % 1000 == 0:
            log(f"worker {worker_id}: {processed} scored, {failed} failed")

    with stats_lock:
        stats[f"w{worker_id}_done"] = processed
        stats[f"w{worker_id}_failed"] = failed
    log(f"worker {worker_id}: finished {processed} scored, {failed} failed")


def cmd_rescore(args: argparse.Namespace) -> int:
    input_globs = args.input
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers
    threads = args.threads
    nodes = args.nodes
    timeout_s = args.timeout
    chunk_bytes = int(args.chunk_gb * 1024 * 1024 * 1024)

    log("building task queue from input files ...")
    task_queue: deque = deque()
    queue_lock = threading.Lock()
    gen = iter_fens(input_globs)

    warmup = workers * 8
    for _ in range(warmup):
        try:
            task_queue.append(next(gen))
        except StopIteration:
            break

    stop_feed = threading.Event()

    def feeder() -> None:
        while not stop_feed.is_set():
            with queue_lock:
                if len(task_queue) < workers * 4:
                    try:
                        task_queue.append(next(gen))
                    except StopIteration:
                        stop_feed.set()
                        return
            time.sleep(0.001)

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    writer = ChunkWriter(out_dir, chunk_bytes)
    stats: dict = {}
    stats_lock = threading.Lock()
    stop_event = threading.Event()

    log(f"starting {workers} workers, threads={threads}, nodes={nodes}, "
        f"timeout={timeout_s}s, chunk_gb={args.chunk_gb}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                worker_loop,
                i,
                args.ember,
                threads,
                args.nnue_off,
                args.hash,
                nodes,
                timeout_s,
                task_queue,
                queue_lock,
                writer,
                stats,
                stats_lock,
                stop_event,
            )
            for i in range(workers)
        ]
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:
                log(f"worker raised exception: {exc}")

    stop_feed.set()
    feeder_thread.join(timeout=5)
    writer.close()

    total_done = stats.get("done", 0)
    total_failed = stats.get("failed", 0)
    log(f"RESCORE DONE: {total_done} scored, {total_failed} failed, "
        f"written to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multithreaded UCI rescorer for Ember clean-data pipeline"
    )
    default_out = str(Path("results/datagen/scored"))
    p.add_argument("--ember", required=True, help="Path to ember executable")
    p.add_argument("--input", required=True, nargs="+",
                   help="Input FEN file glob(s), e.g. 'data/fens/positions_*.fen'")
    p.add_argument("--out", default=default_out,
                   help=f"Output directory for scored TSV chunks (default: {default_out})")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 8,
                   help="Number of Ember subprocess workers (default: CPU count)")
    p.add_argument("--threads", type=int, default=1,
                   help="Engine UCI Threads option per worker (SMP inside Ember)")
    p.add_argument("--nodes", type=int, default=DEFAULT_NODES,
                   help="go nodes limit per position (default: 5000)")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                   help="Per-position timeout in seconds (default: 60)")
    p.add_argument("--hash", type=int, default=256, help="Engine Hash in MB (default: 256)")
    p.add_argument("--chunk-gb", type=float, default=2.0,
                   help="Output chunk size in GiB (default: 2)")
    p.add_argument("--nnue-off", action="store_true", default=True,
                   help="Switch off NNUE before scoring (clean data). Default: on.")
    p.add_argument("--nnue-on", action="store_false", dest="nnue_off",
                   help="Keep NNUE enabled (not recommended for clean data).")
    p.set_defaults(func=cmd_rescore)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
