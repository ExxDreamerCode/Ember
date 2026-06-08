# S-0003 Report: Selective Reductions in Sparse Promotion Races

## Summary

The Blunder 7.2.0 hunt found a repeated sparse-endgame failure:

```text
3b4/8/5k2/8/5K1p/p6P/Pp6/5R2 w - - 0 85
```

At the traced depth 8, the old engine chose `f4g4`. Stockfish triage rated
that move about 416 cp worse than a rook-file defense such as `f1e1` or
`f1d1`. The move is dominated in the practical sense we care about here: it
abandons the blockade while Black has connected passed pawns on `a3` and `b2`.

The fixed engine no longer repeats `f4g4` at the traced depth. It chooses
`f4e4` at depth 8. The color-swapped verification variant also stops repeating
its bad king move and chooses the reference blockade move `f5e5`.

## Discovery

The run used `configs/stupidity/blunder-7.2.0.toml`.

- 400 games were generated.
- 200 games were Ember vs Blunder 7.2.0.
- 200 games were Ember self-play.
- 39,944 traced decisions were collected.
- A 5,000-decision triage slice produced 28 candidates.
- The top 8 candidates were replay-verified with five exact repeats.

Several higher-ranked mate-scale candidates were rejected after deeper
inspection because they were tablebase-like lost positions where every move was
losing or where the shallow Stockfish triage preferred a move that did not hold
up at deeper depth. Candidate `C-0007` was kept because it repeated exactly,
had a clear non-mate centipawn loss, and reproduced in a transformed
promotion-race position.

## Bad Behavior

Original position:

| Depth | Old move |
| ---: | --- |
| 6 | `f4g4` |
| 8 | `f4g4` |
| 10 | `f4g4` |
| 12 | `f1b1` |

Verifier result:

- exact replay at depth 8 repeated `f4g4` five out of five times;
- Stockfish triage: `f1e1` scored 0 cp, while `f4g4` scored -416 cp.

Color-swapped verification variant:

```text
5r2/pP6/P6p/5k1P/8/5K2/8/3B4 b - - 0 85
```

| Depth | Old move |
| ---: | --- |
| 6 | `f5g5` |
| 8 | `f5g5` |
| 10 | `f5g5` |
| 12 | `f8d8` |

The verifier measured the transformed bad move as about 399 cp worse than
`f5e5`.

## Root Cause

This case had two concrete code-level issues.

First, passed-pawn blocker detection in `eval_pawns` mixed side-relative ranks
with actual board rows. White pawns were stored as `7 - row`, but blocker scans
then compared those values as if they were board rows. That made advanced pawn
positions less reliable, especially on same-file and adjacent-file blocker
tests. The fix stores both sides in actual board rows and uses side-specific
scan directions.

Second, the search allowed selective pruning/reduction in the same structural
class already marked unsafe for futility pruning: promotion races and sparse
endgames. In these positions, quiet late moves are often not disposable. They
can be the only way to blockade a near-promotion pawn.

Ablation isolated the selective-search part:

| Position | Toggle | Depth 6 | Depth 8 |
| --- | --- | --- | --- |
| Original | Baseline | `f4g4` | `f4g4` |
| Original | Disable LMP | `f1h1` | `f1d1` |
| Original | Disable LMR | `f4g4` | `f4g4` |
| Variant | Baseline | `f5g5` | `f5g5` |
| Variant | Disable LMR | `f5e5` | `f5e5` |
| Variant | Disable LMP | `f5g5` | `f5g5` |

The final fix does not globally disable LMP or LMR. It skips them only when
`promotion_race(st) || sparse_endgame(st)`, matching the existing futility
safety guard.

## Fixed Behavior

Original position:

| Depth | Fixed move |
| ---: | --- |
| 8 | `f4e4` |

The very shallow depth-6 search can still choose `f4g4` in the base position,
so the regression case intentionally locks the traced depth-8 failure. The
color-swapped variant is fixed at both depth 6 and depth 8:

| Depth | Fixed move |
| ---: | --- |
| 6 | `f5e5` |
| 8 | `f5e5` |
| 10 | `f5e5` |

## Elo Check

Both runs used `configs/elo/default.toml`: 240 games, `8+0.08`, the embedded
mini book, the same mixed Blunder/GNU-Chess/Stockfish-limited opponent pool,
and 200 bootstrap samples.

| Engine | Run id | Games | Elo | 95% CI |
| --- | --- | ---: | ---: | --- |
| Pre-fix baseline | `elo-blunder-7.2.0-240g-v2` | 240 | 2231 | 2155-2315 |
| Fixed engine | `elo-blunder-s0003-240g` | 240 | 2240 | 2138-2333 |

The fixed run is +9 Elo relative to the pre-fix baseline. The confidence
intervals overlap widely, so this is not evidence of a strength gain, but it
does not show a strength regression.

Ember's Blunder 7.2.0 score moved from 4.0/24 to 5.0/24 in the fixed run.

## Verification

Commands used:

```text
cargo build --release --bin ember
python3 tools/check_stupidity_cases.py stupidities/cases/S-0001/case.json stupidities/cases/S-0002/case.json stupidities/cases/S-0003/case.json
tools/measure_elo.py all --config configs/elo/default.toml --run-id elo-blunder-7.2.0-240g-v2
tools/measure_elo.py all --config configs/elo/default.toml --run-id elo-blunder-s0003-240g
```
