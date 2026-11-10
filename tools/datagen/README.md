# Datagen — PGN → .binpack / position database

The `datagen` directory contains tools for generating chess training data:

| Tool | Purpose |
|------|---------|
| **chainpack** | PGN/zst → engine evaluation → chain-based `.binpack` (compressed-binpack compatible) |
| **score_db** | TSV/FEN positions → Ember evaluation → chunked scored TSV (`scored_NNNNN.tsv`) |
| **fen_db** | PGN/EPD/zst → extract positions → sort + dedupe → unique `.fen` chunks |

Each tool has its own README with usage details. See the subdirectory for the tool you want to use.

---

## chainpack — PGN/zst → .binpack

Single-tool pipeline: PGN/zst → engine evaluation → chain-based `.binpack`.

## Usage

```bash
cargo build --release --bin chainpack
target/release/chainpack [options] <input.pgn|zst> [<input.pgn|zst> ...] <output.binpack>
```

### Options

- `--threads N` — number of worker threads (default: CPU parallelism). Each worker runs its own Ember Engine instance.
- `--nodes N` — engine node limit per position (default: 5000)
- `--depth D` — fixed search depth per position (default: 64)
- `--eval-only` — use static corrected_eval instead of search (much faster, no bestmove search)
- `--nnue-off` — disable NNUE for classic-eval data
- `--nnue-on` — force embedded NNUE
- `--batch-size N` — entries per write batch (default: 1000000)

### Resume

If the run is interrupted, a `<output>.chainpack.state` file records progress
(`positions_written`, `games_completed`, `current_file`). On restart,
`chainpack` skips already-processed games and resumes from the middle of the current input file.
The state file is removed automatically on successful completion.

### Input format

`.pgn` or `.pgn.zst` files. Each game is parsed move-by-move from standard chess history.
Every position in the game is evaluated by the Ember engine and written as a chain link
in the output `.binpack`.

### Output format

Chunks are written with `BINP` magic + u32 LE size header, target ~32 MiB per chunk.
Each chunk payload is a sequence of chains in the standard compressed training-data
binpack layout:

- **Stem** (32 bytes): compressed position (24 B) + compressed move (2 B) + zigzag score
  (2 B) + ply+result (2 B) + rule50 (2 B). The first entry of each game is a stem.
- **Count** (2 bytes, big-endian): number of continuation entries that follow
- **Movetext**: bit-packed continuations, one `piece_id + move_id + VLE score delta`
  triple per ply. Positions, plies, results, and rule50 counters of continuations are
  recovered by replaying the moves from the stem.

The layout is read by Ember's own codec in the `ember_chess::chainpack` module and by
third-party readers of the compressed binpack format. Files produced by older chainpack
builds (flat stems with an always-zero count) can be re-chained by regrouping consecutive
entries whose position equals the previous entry's position after its move; no
re-evaluation is needed.

## Example

```bash
target/release/chainpack --threads 8 --nodes 5000 --depth 10 game.pgn output.binpack

target/release/chainpack --threads 8 --eval-only game.pgn.zst output.binpack

target/release/chainpack --threads 8 --eval-only --nnue-off game.pgn output.binpack

target/release/chainpack --threads 8 game1.pgn game2.pgn.zst output.binpack
```