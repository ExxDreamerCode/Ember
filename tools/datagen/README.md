## Stage 1 — Unique FEN

```bash
tools/datagen/fen_db/target/release/fen_db scan \
  --src /data/lichess_pgn \
  --out results/datagen/fens \
  --tmp results/datagen/tmp \
  --tmp2 results/datagen/tmp2 \
  --workers 16 \
  --strip-clocks \
  --chunk-gb 2
```

Parameters:
- `--src` — directory with .pgn/.epd/.zst
- `--tmp` — temporary directory #1
- `--tmp2` — temporary directory #2 (second disk), optional
- `--out` — output directory (default: results/Datagen/fens)
- `--workers` — parallel scanners (default: CPU count)
- `--strip-clocks` — ignore halfmove/fullmove during dedup
- `--max-ply` — max plies per game (default: 256)
- `--chunk-gb` — output chunk size in GiB (default: 2)

Resume: progress of both sort stages is saved to `sort.state` in `--out`.
- After every phase-1 run file is flushed, the chunk path and next run index are appended, so a re-run skips already-processed chunks and continues run numbering instead of overwriting `run_*.tmp` files left on disk.
- After every phase-2 merged batch is written, the batch index is appended, so a re-run skips already-produced `merged_batch_*.tmp` files and continues the batched k-way merge from where it stopped.

State is cleared on full completion.

Scans PGN/EPD/zst files, deduplicates positions by FEN key, and writes `positions_*.fen` chunks.

## Stage 2 — Rescorer (Rust, `score_db`)

```bash
target/release/score_db score \
  --input "results/datagen/fens/positions_*.fen" \
  --out results/datagen/scored \
  --workers 16 \
  --nodes 5000 \
  --chunk-gb 2 \
  --nnue-off
```

Parameters:
- `--input` — glob pattern for `positions_*.fen` chunks (can be repeated)
- `--out` — output directory for `scored_*.tsv` chunks (default: results/datagen/scored)
- `--workers` — parallel scoring workers (default: available CPU parallelism)
- `--nodes` — engine node limit per position (default: 5000)
- `--chunk-gb` — output chunk size in GiB (default: 2)
- `--nnue-off` — disable NNUE for clean classic-eval data
- `--state` — custom path for resume state file (default: `<out>/score_db.state`)

Output format: `FEN\tscore_type\tscore_value\tbestmove\tply\tresult`

Resume: progress is saved to `score_db.state` every 10,000 positions. If the run is interrupted, re-running the same command skips already-processed lines and continues from the next chunk part. State is cleared on full completion.

## Stage 3 — binpack_writer

```bash
cargo build --release --bin binpack_writer
target/release/binpack_writer results/datagen/scored/scored_*.tsv results/datagen/binpack/output.binpack
```

Parameters:
- Positional args — one or more `scored_*.tsv` input files, followed by the output `.binpack` path
- `--no-shuffle` — disable Fisher-Yates shuffle (shuffle is on by default)

Output: `.binpack` binary file ready for NNUE training. Entries are filtered (no check, no tactical moves) and shuffled by default so training batches contain diverse positions.

### Example full pipeline

```bash
# Stage 1: scan + dedup (spread across two disks)
tools/datagen/fen_db/target/release/fen_db scan \
  --src /data/lichess_pgn \
  --tmp /disk1/tmp \
  --tmp2 /disk2/tmp \
  --out results/datagen/fens \
  --workers 16 --strip-clocks

# Stage 2: rescore with resume support
target/release/score_db score \
  --input "results/datagen/fens/positions_*.fen" \
  --out results/datagen/scored \
  --workers 16 --nodes 5000 --nnue-off

# Stage 3: shuffle + pack into single .binpack
target/release/binpack_writer \
  results/datagen/scored/scored_*.tsv \
  results/datagen/binpack/output.binpack
```
