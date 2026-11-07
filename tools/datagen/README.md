# Datagen — PGN → .binpack / position database

The `datagen` directory contains tools for generating chess training data:

| Tool | Purpose |
|------|---------|
| **chainpack** | PGN/zst → engine evaluation → chain-based `.binpack` (sfbinpack-compatible) |
| **score_db** | TSV/FEN positions → Ember evaluation → chunked scored TSV (`scored_NNNNN.tsv`) |
| **fen_db** | PGN/EPD/zst → extract positions → sort + dedupe → unique `.fen` chunks |

Each tool has its own README with usage details. See the subdirectory for the tool you want to use.

---

## chainpack — PGN/zst → .binpack

Single-tool pipeline: PGN/zst → engine evaluation → chain-based `.binpack`.
Output is byte-compatible with the sfbinpack format (Stem + Count + MoveText).

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
- `--no-shuffle` — disable Fisher-Yates shuffle of chain entries (shuffle is on by default)
- `--batch-size N` — entries per write batch (default: 1000000)
- `--seen-bits N` — Bloom filter size in bits for position deduplication (default: 256000000 = 32 MiB).
  Fixed memory regardless of position count. Higher values reduce false positives where
  a new position is wrongly skipped as a duplicate. No false negatives: every repeated
  position is always detected.

### Filters

The following positions/moves are skipped before evaluation:

- **Check**: side to move is in check
- **Tactical**: promotions, captures, en passant, castling
- **Duplicate**: position with Zobrist hash already seen (Bloom filter)

Skipped counters are printed per worker to stderr.

### Resume

If the run is interrupted, a `<output>.chainpack.state` file records progress. On restart,
`chainpack` skips already-processed games and resumes from the middle of the current input file.
The state file is removed automatically on successful completion.

### Input format

`.pgn` or `.pgn.zst` files. Each game is parsed move-by-move. Every position in the game is evaluated by the Ember engine and written as a chain link in the output `.binpack`.

### Output format

- **Stem** (32 bytes): compressed position (24 B) + compressed move (2 B) + zigzag score (2 B) + ply+result (2 B) + rule50 (2 B)
- **Count** (2 bytes, big-endian): number of continuation plies
- **MoveText**: per continuation — piece_id + move_id (bit-packed, LSB-first) + score delta (VLE, 4-bit blocks)

Chunks are written with `BINP` magic + u32 LE size header, target ~32 MiB per chunk.

## Example

```bash
target/release/chainpack --threads 8 --nodes 5000 --depth 10 game.pgn output.binpack

target/release/chainpack --threads 8 --eval-only game.pgn.zst output.binpack

target/release/chainpack --threads 8 --eval-only --nnue-off game.pgn output.binpack

target/release/chainpack --threads 8 --seen-bits 512000000 game.pgn output.binpack

target/release/chainpack --threads 8 game1.pgn game2.pgn.zst output.binpack
```