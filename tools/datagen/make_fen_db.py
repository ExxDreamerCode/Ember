#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

try:
    import chess
    import chess.pgn
except ImportError as exc:
    sys.exit(f"python-chess is required: pip install python-chess  ({exc})")

CHUNK_BYTES_DEFAULT = 2 * 1024 * 1024 * 1024
MAX_PLY_DEFAULT = 256
BATCH_LINES = 200_000
RUN_LINES = 5_000_000


def log(msg: str) -> None:
    print(f"[make_fen_db] {msg}", flush=True)


def strip_clock_fields(fen: str) -> str:
    parts = fen.split()
    if len(parts) >= 6:
        return " ".join(parts[:4])
    return fen


def fen_key(fen: str, strip_clocks: bool) -> str:
    return strip_clock_fields(fen) if strip_clocks else fen


def result_to_stm(result_tag: str, stm_white: bool) -> Optional[int]:
    if result_tag == "1-0":
        return 1 if stm_white else -1
    if result_tag == "0-1":
        return -1 if stm_white else 1
    if result_tag == "1/2-1/2":
        return 0
    return None


def _flush_batch(batch: list[str], batch_idx: int, out_dir: Path) -> int:
    if not batch:
        return batch_idx
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = out_dir / f"chunk_{os.getpid()}_{threading.get_ident()}_{batch_idx:06d}.fen"
    with open(chunk_path, "w", encoding="utf-8") as f:
        for line in batch:
            f.write(line)
            f.write("\n")
    return batch_idx + 1


def scan_pgn_file(path: Path, out_dir: Path, strip_clocks: bool, max_ply: int,
                  batch_lines: int = BATCH_LINES) -> tuple[int, int]:
    total = 0
    skipped = 0
    batch: list[str] = []
    batch_idx = 0

    def process_fh(fh) -> None:
        nonlocal total, skipped, batch, batch_idx
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            result_tag = game.headers.get("Result", "*")
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                stm_white = board.turn
                result = result_to_stm(result_tag, stm_white)
                if result is None:
                    skipped += 1
                    continue
                batch.append(f"{fen}\t{board.ply()}\t{result}")
                total += 1
                if len(batch) >= batch_lines:
                    batch_idx = _flush_batch(batch, batch_idx, out_dir)
                    batch = []
                if board.ply() >= max_ply:
                    break

    try:
        if path.suffix.lower() == ".zst":
            import zstandard
            import io
            with open(path, "rb") as fh:
                dctx = zstandard.ZstdDecompressor()
                with dctx.stream_reader(fh) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                    process_fh(text_stream)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                process_fh(fh)
    except Exception as exc:
        log(f"WARNING: failed to scan {path}: {exc}")
    batch_idx = _flush_batch(batch, batch_idx, out_dir)
    return total, skipped


def scan_worker(args: tuple[str, str, bool, int, int]) -> tuple[int, int]:
    path, out_dir, strip_clocks, max_ply, batch_lines = args
    return scan_pgn_file(Path(path), Path(out_dir), strip_clocks, max_ply, batch_lines)


def cmd_scan(args: argparse.Namespace) -> int:
    src = Path(args.src)
    tmp_dirs = [Path(args.tmp)]
    if args.tmp2:
        tmp_dirs.append(Path(args.tmp2))
    for d in tmp_dirs:
        d.mkdir(parents=True, exist_ok=True)

    pgn_files = sorted(src.rglob("*.pgn")) + sorted(src.rglob("*.epd")) + sorted(src.rglob("*.zst"))
    if not pgn_files:
        sys.exit(f"No .pgn/.epd/.zst files found under {src}")

    log(f"Found {len(pgn_files)} PGN/EPD/zst files; workers={args.workers}; tmp_dirs={[str(d) for d in tmp_dirs]}")

    tasks = [
        (str(p), str(tmp_dirs[i % len(tmp_dirs)]), args.strip_clocks, args.max_ply, BATCH_LINES)
        for i, p in enumerate(pgn_files)
    ]

    total = 0
    skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(scan_worker, t) for t in tasks]
        for fut in as_completed(futures):
            n, s = fut.result()
            total += n
            skipped += s
            log(f"Scanned positions so far: {total:,}")

    log(f"SCAN done: {total:,} raw positions across {len(tmp_dirs)} tmp dir(s) ({skipped:,} skipped without result)")
    return 0


def _collect_chunk_files(tmp_dirs: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for d in tmp_dirs:
        files.extend(sorted(d.glob("*.fen")))
    return files


def _write_sorted_run(chunk: list[str], run: Path, strip_clocks: bool) -> None:
    chunk.sort(key=lambda s: fen_key(s.split("\t")[0], strip_clocks))
    with open(run, "w", encoding="utf-8") as rf:
        rf.write("\n".join(chunk))
        rf.write("\n")


def external_sort_files(files: list[Path], tmp_dirs: Sequence[Path], out_dir: Path, strip_clocks: bool,
                        run_lines: int = RUN_LINES, chunk_bytes: int = CHUNK_BYTES_DEFAULT) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_files: list[Path] = []

    run_idx_global = 0
    for file_idx, src in enumerate(files):
        log(f"Sorting runs from {src.name} ({file_idx + 1}/{len(files)})")
        chunk: list[str] = []
        run_idx = 0
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                fen = line.rstrip("\n")
                if not fen:
                    continue
                chunk.append(fen)
                if len(chunk) >= run_lines:
                    target_dir = tmp_dirs[run_idx_global % len(tmp_dirs)]
                    run_idx_global += 1
                    run = target_dir / f"run_{src.stem}_{run_idx:06d}.tmp"
                    _write_sorted_run(chunk, run, strip_clocks)
                    run_files.append(run)
                    run_idx += 1
                    chunk = []
            if chunk:
                target_dir = tmp_dirs[run_idx_global % len(tmp_dirs)]
                run_idx_global += 1
                run = target_dir / f"run_{src.stem}_{run_idx:06d}.tmp"
                _write_sorted_run(chunk, run, strip_clocks)
                run_files.append(run)
                run_idx += 1

    log(f"Produced {len(run_files)} sorted runs across {len(tmp_dirs)} tmp dir(s)")

    log(f"Merging {len(run_files)} sorted runs -> output chunks under {out_dir}")
    handles = [open(r, "r", encoding="utf-8") for r in run_files]
    written = 0
    skipped_dup = 0
    part_idx = 0
    current_size = 0
    current: Optional[object] = None
    seen_key: Optional[str] = None

    def open_part(idx: int):
        p = out_dir / f"positions_{idx:05d}.fen"
        return open(p, "w", encoding="utf-8")

    try:
        for line in heapq.merge(*handles,
                                key=lambda s: fen_key(s.split("\t")[0], strip_clocks)):
            line = line.rstrip("\n")
            if not line:
                continue
            key = fen_key(line.split("\t")[0], strip_clocks)
            if key == seen_key:
                skipped_dup += 1
                continue
            seen_key = key

            if current is None or current_size >= chunk_bytes:
                if current is not None:
                    current.close()
                current = open_part(part_idx)
                part_idx += 1
                current_size = 0
                log(f"Writing chunk {part_idx - 1}")
            current.write(line)
            current.write("\n")
            current_size += len(line.encode("utf-8")) + 1
            written += 1
    finally:
        if current is not None:
            current.close()
        for h in handles:
            h.close()

    for r in run_files:
        try:
            r.unlink()
        except OSError:
            pass

    log(f"SORT+UNIQ done: {written:,} unique positions in {part_idx} chunk(s) under {out_dir} ({skipped_dup:,} dupes removed)")
    return written


def cmd_sort(args: argparse.Namespace) -> int:
    tmp_dirs = [Path(args.tmp)]
    if args.tmp2:
        tmp_dirs.append(Path(args.tmp2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    chunk_files = _collect_chunk_files(tmp_dirs)
    if not chunk_files:
        sys.exit(f"No .fen chunk files found in {tmp_dirs}")

    chunk_bytes = int(args.chunk_gb * 1024 * 1024 * 1024)
    external_sort_files(chunk_files, tmp_dirs, out, args.strip_clocks,
                        run_lines=RUN_LINES, chunk_bytes=chunk_bytes)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    rc = cmd_scan(args)
    if rc != 0:
        return rc

    class SortArgs:
        pass

    sa = SortArgs()
    sa.tmp = args.tmp
    sa.tmp2 = args.tmp2
    sa.out = args.out
    sa.chunk_gb = args.chunk_gb
    sa.strip_clocks = args.strip_clocks
    return cmd_sort(sa)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build deduplicated FEN/EPD database from PGN sources")
    sub = p.add_subparsers(dest="command", required=True)

    default_out = str(Path("results/Datagen/fens"))

    sp_scan = sub.add_parser("scan", help="Scan PGN/EPD files, emit raw FEN chunks with ply and result")
    sp_scan.add_argument("--src", required=True, help="Directory with .pgn/.epd files")
    sp_scan.add_argument("--tmp", required=True, help="Scratch dir #1 for chunk files")
    sp_scan.add_argument("--tmp2", default=None, help="Scratch dir #2 for chunk files (second disk), optional")
    sp_scan.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    sp_scan.add_argument("--strip-clocks", action="store_true",
                         help="Ignore halfmove/fullmove counters for dedup key")
    sp_scan.add_argument("--max-ply", type=int, default=MAX_PLY_DEFAULT)
    sp_scan.set_defaults(func=cmd_scan)

    sp_sort = sub.add_parser("sort", help="Sort + dedup chunks into output position files")
    sp_sort.add_argument("--tmp", required=True, help="Scratch dir #1 with .fen chunks")
    sp_sort.add_argument("--tmp2", default=None, help="Scratch dir #2 with .fen chunks (second disk), optional")
    sp_sort.add_argument("--out", default=default_out,
                         help=f"Output dir for position chunks (default: {default_out})")
    sp_sort.add_argument("--chunk-gb", type=float, default=2.0, help="Output chunk size in GiB")
    sp_sort.add_argument("--strip-clocks", action="store_true",
                         help="Ignore halfmove/fullmove counters for dedup key")
    sp_sort.set_defaults(func=cmd_sort)

    sp_all = sub.add_parser("all", help="One-shot scan + sort + dedup")
    sp_all.add_argument("--src", required=True)
    sp_all.add_argument("--out", default=default_out,
                        help=f"Output dir for position chunks (default: {default_out})")
    sp_all.add_argument("--tmp", required=True, help="Scratch dir #1")
    sp_all.add_argument("--tmp2", default=None, help="Scratch dir #2 (second disk), optional")
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