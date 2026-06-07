# S-0002 Report: Futility Pruning in Sparse Promotion Endgames

## Summary

The broad stupidity hunt found a repeated sparse-endgame failure. In this
position, Black is worse but can avoid immediate mate with active rook defense:

```text
7R/1P4p1/4Kpk1/3N4/4P3/7P/8/4r3 b - - 0 72
```

The old engine chose `e1b1` at depths 6 and 8. Stockfish depth 16 refutes that
move as mate in 3. The fixed engine chooses `e1e4`, Stockfish's depth-16 best
defensive move for the position.

## Discovery

The broad run used `configs/stupidity/broad.toml`:

- 400 games were generated.
- 35,155 Ember search decisions were traced.
- Depth-8 Stockfish triage produced 186 candidates.
- The top 30 candidates were replay-verified.

This case was candidate `C-0016` from run `stupidity-002-broad-01`.

I rejected several higher-ranked tablebase-like candidates because they did not
replay from a clean FEN or were only missed faster mates while already winning.
S-0002 was kept because the bad move repeats from the clean FEN and turns a
defensible position into a forced mate.

## Bad Behavior

Original position:

| Depth | Old move | Old score |
| ---: | --- | ---: |
| 6 | `e1b1` | -831 cp |
| 8 | `e1b1` | -942 cp |

Horizontal mirror:

```text
R7/1p4P1/1kpK4/4N3/3P4/P7/8/3r4 b - - 0 72
```

| Depth | Old move | Old score |
| ---: | --- | ---: |
| 6 | `d1g1` | -830 cp |
| 8 | `d1g1` | -947 cp |

The old moves are not strategic trade-offs. Stockfish depth 16 constrained
search reports `e1b1` as mate in 3:

```text
e1b1 h3h4 b1b6 d5b6 f6f5 e4f5
```

For comparison, Stockfish depth 16 rates `e1e4` around -918 cp rather than a
mate score. Black is still worse, but the immediate forced mate is gone.

## Root Cause

The ablation matrix isolated futility pruning:

| Build | Depth 6 | Depth 8 |
| --- | --- | --- |
| Baseline | `e1b1` | `e1b1` |
| `EMBER_DISABLE_FUTILITY=1` | `e1e4` | `e1e4` |
| Disable history pruning | `e1b1` | `e1b1` |
| Disable LMP | `e1b1` | `e1b1` |
| Disable null move | `e1b1` | `e1b1` |
| Disable reverse futility | `e1b1` | `e1b1` |
| Disable SEE pruning | `e1b1` | `e1b1` |
| Disable IID reduction | `e1b1` | `e1b1` |

LMR also changes the depth-8 move, but not the depth-6 move. Futility is the
only single toggle that removes the bad move at both reproduced depths.

The specific bad assumption was using static evaluation plus a small futility
margin in sparse promotion endgames. In those positions quiet promotion and
mating resources can dominate the result, so returning alpha before searching
them is not sound enough.

## Fix

Futility pruning remains enabled in normal play. The fix only skips depth-2
and depth-3 futility cutoffs in positions where the static bound is
structurally unsafe:

- promotion-race positions with pawns on the sixth or seventh rank;
- sparse endgames with eight or fewer non-king pieces.

Depth-1 futility remains active, as does futility in ordinary middlegame and
quiet positions. This keeps the fix focused on the cutoff depth that caused
S-0002.

## Fixed Behavior

Original position:

| Depth | Fixed move | Fixed score |
| ---: | --- | ---: |
| 6 | `e1e4` | -1173 cp |
| 8 | `e1e4` | -1282 cp |
| 10 | `e1e4` | -1437 cp |

Horizontal mirror:

| Depth | Fixed move | Fixed score |
| ---: | --- | ---: |
| 6 | `d1d4` | -1177 cp |
| 8 | `d1d4` | -1329 cp |

## Elo Check

I measured the pre-fix and fixed engines with
`configs/elo/stockfish-adaptive.toml`. Both runs used 500 games, the `8+0.08`
time control, the embedded mini book, and the Stockfish UCI_Elo-equivalent
rating scale.

| Engine | Run id | Games | Elo | 95% CI | SE |
| --- | --- | ---: | ---: | --- | ---: |
| Pre-fix baseline | `s0002-before-manual` | 500 | 2370 | 2338-2402 | 16.41 |
| Fixed engine | `s0002-after-19d9940` | 500 | 2382 | 2350-2414 | 16.32 |

The fixed run is +12 Elo relative to the pre-fix baseline. The 95% intervals
overlap, so this measurement does not prove a strength gain, but it also does
not show a strength regression.

## Demo Videos

Base position:

- Bad move: `stupidities/cases/S-0002/demo/base-before-bad.mp4`
- Reference move: `stupidities/cases/S-0002/demo/base-reference.mp4`
- Fixed move: `stupidities/cases/S-0002/demo/base-after-fixed.mp4`

Horizontal mirror:

- Bad move: `stupidities/cases/S-0002/demo/variant-1-before-bad.mp4`
- Reference move: `stupidities/cases/S-0002/demo/variant-1-reference.mp4`
- Fixed move: `stupidities/cases/S-0002/demo/variant-1-after-fixed.mp4`

## Verification

Commands used:

```text
cargo build --release --bin ember --features "decision-trace search-debug"
tools/check_stupidity_cases.py stupidities/cases/S-0001/case.json stupidities/cases/S-0002/case.json
python3 tools/hunt_stupidities.py render --case stupidities/cases/S-0002
```

Strength check:

```text
tools/measure_elo.py all --config configs/elo/stockfish-adaptive.toml --run-id s0002-before-manual
tools/measure_elo.py all --config configs/elo/stockfish-adaptive.toml --run-id s0002-after-19d9940
```
