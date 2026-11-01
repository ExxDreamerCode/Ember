## Stage 1 — Unique FEN

```bash
python tools/Datagen/make_fen_db.py all \
  --src /data/lichess_pgn \
  --out results/datagen/fens \
  --tmp results/datagen/tmp \
  --workers 16 \
  --strip-clocks \
  --chunk-gb 2
```

Parameters:
- `--src` — directory with .pgn/.epd
- `--tmp` — temporary directory
- `--out` — output directory (default: results/Datagen/fens)
- `--workers` — parallel scanners (default: CPU count)
- `--strip-clocks` — ignore halfmove/fullmove during dedup
- `--max-ply` — max plies per game (default: 256)
- `--chunk-gb` — chunk size in GiB (default: 2)

## Stage 2 — Rescorer

```bash
python tools/datagen/rescore.py \
  --ember target/release/ember.exe \
  --input "results/datagen/fens/positions_*.fen" \
  --out results/Datagen/scored \
  --workers 16 \
  --threads 1 \
  --nodes 5000 \
  --timeout 60 \
  --chunk-gb 2 \
  --nnue-off
```

Output: `FEN\tscore_type\tscore_value\tbestmove`

## Stage 3 — binpack_writer

```bash
cargo build --release --bin binpack_writer
for f in results/datagen/scored/scored_*.tsv; do
  target/release/binpack_writer "$f" "results/datagen/binpack/$(basename "$f" .tsv).binpack"
done
```
