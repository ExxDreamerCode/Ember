#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional

try:
    import chess
    import chess.pgn
except ImportError as exc:
    sys.exit(f"python-chess is required: pip install python-chess  ({exc})")

CHUNK_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024
MAX_PLY_DEFAULT = 256
BATCH_LINES = 200_000


def log(msg: str) -> None:
    print(f"[make_fen_db] {msg}", flush=True)


def strip_clock_fields(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 6:
        return " ".join(parts[:4] + ["-", "-"])
    return fen


def fen_key(fen: str, strip_clocks: bool) -> str:
    if strip_clocks:
        return strip_clock_fields(fen)
    return fen


def epd_from_fen(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 6:
        return " ".join(parts[:4] + [parts[4], parts[5]])
    return fen


def legal_plies(board: chess.Board, max_ply: int) -> Iterator[str]:
    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_ply:
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(moves[0])
        yield board.fen()
        ply += 1


def scan_pgn_file(path: Path, out_dir: Path, strip_clocks: bool, max_ply: int,
                  batch_lines: int = BATCH_LINES) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    batch: list[str] = []
    batch_idx = 0

    def flush() -> None:
        nonlocal batch, batch_idx
        if not batch:
            return
        chunk_path = out_dir / f"chunk_{os.getpid()}_{threading.get_ident()}_{batch_idx:06d}.fen"
        with open(chunk_path, "w", encoding="utf-8") as f:
            for fen in batch:
                f.write(fen)
                f.write("\n")
        batch_idx += 1
        batch = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    fen = board.fen()
                    key = fen_key(fen, strip_clocks)
                    batch.append(key)
                    total += 1
                    if len(batch) >= batch_lines:
                        flush()
                    if board.ply() >= max_ply:
                        break
    except Exception as exc:
        log(f"WARNING: failed to scan {path}: {exc}")
    flush()
    return total


def scan_worker(args: tuple[str, str, bool, int, int]) -> int:
    path, out_dir, strip_clocks, max_ply, batch_lines = args
    return scan_pgn_file(Path(path), Path(out_dir), strip_clocks, max_ply, batch_lines)


def cmd_scan(args: argparse.Namespace) -> int:
    src = Path(args.src)
    tmp = Path(args.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    pgn_files = sorted(src.rglob("*.pgn")) + sorted(src.rglob("*.epd"))
    if not pgn_files:
        sys.exit(f"No .pgn/.epd files found under {src}")

    log(f"Found {len(pgn_files)} PGN/EPD files; workers={args.workers}")

    tasks = [
        (str(p), str(tmp), args.strip_clocks, args.max_ply, BATCH_LINES)
        for p in pgn_files
    ]

    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(scan_worker, t) for t in tasks]
        for fut in as_completed(futures):
            n = fut.result()
            total += n
            log(f"Scanned positions so far: {total:,}")

    log(f"SCAN done: {total:,} raw positions written to {tmp}")
    return 0


def external_sort_files(files: list[Path], out_path: Path, chunk_lines: int = 5_000_000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_files: list[Path] = []

    for src in files:
        log(f"Sorting runs from {src.name}")
        chunk: list[str] = []
        run_idx = 0
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                fen = line.rstrip("\n")
                if not fen:
                    continue
                chunk.append(fen)
                if len(chunk) >= chunk_lines:
                    chunk.sort()
                    run = src.parent / f"run_{src.stem}_{run_idx:06d}.tmp"
                    with open(run, "w", encoding="utf-8") as rf:
                        rf.write("\n".join(chunk))
                        rf.write("\n")
                    run_files.append(run)
                    run_idx += 1
                    chunk = []
            if chunk:
                chunk.sort()
                run = src.parent / f"run_{src.stem}_{run_idx:06d}.tmp"
                with open(run, "w", encoding="utf-8") as rf:
                    rf.write("\n".join(chunk))
                    rf.write("\n")
                run_files.append(run)

    log(f"Merging {len(run_files)} sorted runs -> {out_path}")
    seen: Optional[str] = None
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        handles = [open(r, "r", encoding="utf-8") for r in run_files]
        try:
            for fen in heapq.merge(*handles, key=lambda s: s.rstrip("\n")):
                fen = fen.rstrip("\n")
                if fen == seen:
                    continue
                seen = fen
                out.write(fen)
                out.write("\n")
                written += 1
        finally:
            for h in handles:
                h.close()
        for r in run_files:
            try:
                r.unlink()
            except OSError:
                pass

    log(f"SORT+UNIQ done: {written:,} unique positions in {out_path}")


def cmd_sort(args: argparse.Namespace) -> int:
    tmp = Path(args.tmp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(tmp.glob("*.fen"))
    if not chunk_files:
        sys.exit(f"No .fen chunk files found in {tmp}")

    merged = tmp / "merged_unique.fen"
    external_sort_files(chunk_files, merged, chunk_lines=5_000_000)

    chunk_gb = args.chunk_gb
    chunk_bytes = int(chunk_gb * 1024 * 1024 * 1024)
    part_idx = 0
    current_size = 0
    current: Optional[object] = None
    written = 0

    def open_part(idx: int):
        p = out / f"positions_{idx:05d}.fen"
        return open(p, "w", encoding="utf-8")

    try:
        with open(merged, "r", encoding="utf-8") as f:
            for line in f:
                fen = line.rstrip("\n")
                if not fen:
                    continue
                if current is None:
                    current = open_part(part_idx)
                    part_idx += 1
                    current_size = 0
                    log(f"Writing chunk {part_idx-1}")
                if current_size >= chunk_bytes:
                    current.close()
                    current = open_part(part_idx)
                    part_idx += 1
                    current_size = 0
                    log(f"Writing chunk {part_idx-1}")
                current.write(fen)
                current.write("\n")
                current_size += len(fen.encode("utf-8")) + 1
    finally:
        if current is not None:
            current.close()
    try:
        merged.unlink()
    except OSError:
        pass

    log(f"SORT done: {written:,} unique positions in {part_idx} chunk(s) under {out}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_scan(args)
    if rc != 0:
        return rc
    class SortArgs:
        pass
    sa = SortArgs()
    sa.tmp = args.tmp
    sa.out = args.out
    sa.chunk_gb = args.chunk_gb
    sa.strip_clocks = args.strip_clocks
    return cmd_sort(sa)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build deduplicated FEN/EPD database from PGN sources")
    sub = p.add_subparsers(dest="command", required=True)

    default_out = str(Path("results/Datagen/fens"))

    sp_scan = sub.add_parser("scan", help="Scan PGN/EPD files, emit raw FEN chunks")
    sp_scan.add_argument("--src", required=True, help="Directory with .pgn/.epd files")
    sp_scan.add_argument("--tmp", required=True, help="Temp dir for chunk files")
    sp_scan.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    sp_scan.add_argument("--strip-clocks", action="store_true",
                         help="Ignore halfmove/fullmove counters for dedup key")
    sp_scan.add_argument("--max-ply", type=int, default=MAX_PLY_DEFAULT)
    sp_scan.set_defaults(func=cmd_scan)

    sp_sort = sub.add_parser("sort", help="Sort + dedup chunks into 2 GB output files")
    sp_sort.add_argument("--tmp", required=True, help="Temp dir with .fen chunks")
    sp_sort.add_argument("--out", default=default_out,
                         help=f"Output dir for position chunks (default: {default_out})")
    sp_sort.add_argument("--chunk-gb", type=float, default=2.0, help="Output chunk size in GiB")
    sp_sort.set_defaults(func=cmd_sort)

    sp_all = sub.add_parser("all", help="One-shot scan + sort + dedup")
    sp_all.add_argument("--src", required=True)
    sp_all.add_argument("--out", default=default_out,
                        help=f"Output dir for position chunks (default: {default_out})")
    sp_all.add_argument("--tmp", required=True)
    sp_all.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    sp_all.add_argument("--strip-clocks", action="store_true")
    sp_all.add_argument("--max-ply", type=int, default=MAX_PLY_DEFAULT)
    sp_all.add_argument("--chunk-gb", type=float, default=2.0)
    sp_all.set_defaults(func=cmd_all)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
