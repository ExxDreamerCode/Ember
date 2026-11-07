# score_db — Rescore PGN positions with Ember

Batch rescores positions from TSV/FEN input using the Ember engine.
Output is written as chunked TSV files (`scored_00000.tsv`, ...) ready for
NNUE training.

## Usage

```bash
cargo build --release --bin score_db

# Rescore positions from a glob pattern
target/release/score_db rescore --input "results/positions_*.fen" --out results/datagen/scored

# With NNUE, eval-only (static evaluation, no search)
target/release/score_db rescore --input "results/positions_*.fen" --out results/datagen/scored \
  --nnue-on --eval-only --workers 8

# Classic evaluation with search
target/release/score_db rescore --input "results/positions_*.fen" --out results/datagen/scored \
  --nnue-off --nodes 5000 --depth 64 --workers 8
```

## Subcommands

| Command | Description |
|---------|-------------|
| `rescore` | Read positions from input files, score each with Ember, write chunked TSV output |
| `score` | Alias for `rescore` (same behavior) |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input <glob>` | required | Glob pattern for input position files (e.g. `results/positions_*.fen`) |
| `--out <dir>` | `results/datagen/scored` | Output directory for chunked TSV files |
| `--workers N` | CPU parallelism | Number of worker threads, each runs its own Ember Engine |
| `--nodes N` | 5000 | Engine node limit per position (search mode only) |
| `--depth N` | 64 | Fixed search depth per position (search mode only) |
| `--chunk-gb N` | 2 | Output chunk size in GiB; a new `scored_NNNNN.tsv` is started when exceeded |
| `--nnue-off` | — | Disable NNUE (classic evaluation) |
| `--nnue-on` | — | Force embedded NNUE |
| `--eval-only` | — | Use static `corrected_eval` instead of search (much faster, no bestmove) |
| `--quiet` | — | Reduce per-worker log frequency |

## Input format

Each input line is expected to be tab-separated. The feeder parses:

| Field | Index | Description | Example |
|-------|-------|-------------|---------|
| FEN | 0 | Position in FEN notation | `rnbqkbnr/... w KQkq - 0 1` |
| (ignored) | 1–3 | — | — |
| ply | 4 | Game ply number | `5` |
| result | 5 | Game result (`1-0`, `0-1`, `1/2-1/2`, `*`) | `1-0` |

Lines that are empty or have an empty FEN are skipped.

## Output format

Each output line is tab-separated:

| Field | Description | Example |
|-------|-------------|---------|
| FEN | Position FEN | `rnbqkbnr/... w KQkq - 0 1` |
| score_type | `"ep"` for eval-only, `"cp"` for search | `cp` |
| score_val | Score in centipawns, relative to side to move | `56` |
| best_move | Best move in UCI format (`0000` for eval-only) | `e2e4` |
| ply | Game ply | `1` |
| result | Game result from side-to-move perspective (`1`, `0`, `-1`) | `1` |

Output is written to chunked files: `scored_00000.tsv`, `scored_00001.tsv`, ...
A new chunk is started when the accumulated byte size exceeds `--chunk-gb`.

## Resume

Progress is persisted to `<out>/score_db.state` after every `STATE_SAVE_EVERY`
records (currently 1000). On restart:

- The feeder skips positions with index `< saved_index`
- Output chunks resume from `find_next_part` (continues `scored_NNNNN.tsv` numbering)
- The state file is deleted on successful completion

## Examples

```bash
# Full rescore of all positions_*.fen files, eval-only, 8 workers
target/release/score_db rescore --input "data/positions_*.fen" --out data/scored \
  --nnue-on --eval-only --workers 8 --chunk-gb 4

# Classic eval (no NNUE), search at depth 10
target/release/score_db rescore --input "data/positions_*.fen" --out data/scored \
  --nnue-off --depth 10 --workers 4

# Resume an interrupted run (state is in <out>/score_db.state)
target/release/score_db rescore --input "data/positions_*.fen" --out data/scored \
  --workers 8
```
