# fen_db — PGN → unique sorted position database

Two-stage pipeline: scan PGN/EPD/zst files to extract positions, then sort and
deduplicate them into a position database. Output is chunked `.fen` files
(`positions_00000.fen`, ...) ready for scoring by `score_db`.

## Usage

```bash
cargo build --release --bin fen_db --manifest-path tools/datagen/fen_db/Cargo.toml

# Stage 1: scan PGN files into chunked .fen files in tmp dirs
target/release/fen_db scan --src data/pgn --tmp data/tmp_fen --workers 8

# Stage 2: sort, dedupe, and write unique positions to output dir
target/release/fen_db sort --tmp data/tmp_fen --tmp2 data/tmp_fen2 \
  --out results/datagen/fens --chunk-gb 2 --delete-chunks
```

## Subcommands

| Command | Description |
|---------|-------------|
| `scan` | Parse PGN/EPD/zst files, extract every position with ply and result, write to chunked `.fen` files in tmp dirs |
| `sort` | Sort all tmp chunks, deduplicate by FEN key, merge into unique sorted output chunks |

## `scan` options

| Option | Default | Description |
|--------|---------|-------------|
| `--src <dir>` | required | Directory containing `.pgn`, `.epd`, `.zst` files |
| `--tmp <dir>` | required | Temporary directory for intermediate chunk files |
| `--tmp2 <dir>` | — | Second tmp dir (spreads I/O across two disks/directories) |
| `--workers N` | CPU parallelism | Number of parallel scan threads (one per PGN file) |
| `--strip-clocks` | — | Strip halfmove/fullmove clock from FEN before writing |
| `--max-ply N` | 256 | Stop scanning a game after this many plies |

### Input format

PGN files are parsed move-by-move. Each position after a legal move is written as:

```
<FEN>\t<ply>\t<result>
```

where `result` is `1`, `0`, or `-1` from the side-to-move perspective. Games
without a valid result tag (`*`) have their positions skipped.

### Output (tmp)

Chunked `.fen` files named `chunk_<file_id>_<batch_idx:06>.fen`, distributed
round-robin across `--tmp` and `--tmp2` directories. Each chunk contains up to
200,000 lines.

## `sort` options

| Option | Default | Description |
|--------|---------|-------------|
| `--tmp <dir>` | required | Temporary directory containing scan chunks |
| `--tmp2 <dir>` | — | Second tmp dir (used if scan used two) |
| `--out <dir>` | `results/datagen/fens` | Output directory for unique position chunks |
| `--chunk-gb N` | 2 | Output chunk size in GiB |
| `--strip-clocks` | — | Strip clocks from FEN when computing sort/dedup key |
| `--delete-chunks` | — | Delete source `.fen` chunks after processing them |

## Sort details

1. **Phase 1 — Runs:** Each `.fen` chunk is read, sorted by FEN key, and written
   as a sorted `.tmp` run file. Progress is tracked in `<out>/sort.state` for
   resume support.
2. **Phase 2 — Batched merge:** Runs are merged in batches of 200 using a
   k-way merge heap. Within each batch, duplicate keys are removed.
3. **Phase 3 — Global merge:** All batch files are merged with global deduplication
   into final `positions_NNNNN.fen` output chunks.

### Resume

Sort progress is persisted to `<out>/sort.state`. On restart:
- Already-processed chunks are skipped
- Merge batches resume from the last completed batch
- The state file is deleted on successful completion

## Examples

```bash
# Two-disk scan (spreads I/O)
target/release/fen_db scan --src data/pgn --tmp /disk1/fen_tmp --tmp2 /disk2/fen_tmp \
  --workers 12 --strip-clocks --max-ply 128

# Sort and deduplicate
target/release/fen_db sort --tmp /disk1/fen_tmp --tmp2 /disk2/fen_tmp \
  --out results/datagen/fens --chunk-gb 4 --delete-chunks

# Score the resulting unique positions
target/release/score_db rescore --input "results/datagen/fens/positions_*.fen" \
  --out results/datagen/scored --nnue-on --eval-only --workers 8
```
