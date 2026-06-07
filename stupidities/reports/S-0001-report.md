# S-0001 Report: Passive Rook Moves Under Mating Attack

## Summary

Ember reached a position where the side to move was under a concrete mating
attack. The old search chose passive rook moves (`b2b1`, `b2a2`) and evaluated
the position as roughly equal or favorable. Deeper reference analysis shows
those rook moves allow forced mating lines.

The fixed engine no longer chooses those moves in the original position or in a
color-swapped mirror variant. The case is stored in:

- `stupidities/cases/S-0001/case.json`
- `stupidities/cases/S-0001/notes.md`

## Discovery

The first hunt run used `tools/hunt_stupidities.py` with
`configs/stupidity/default.toml`:

- 48 short games were generated.
- 4,891 Ember decisions were traced as JSONL.
- A depth-8 Stockfish triage pass found 42 candidates.

The chosen case came from this FEN:

```text
4r1k1/1p3pp1/1p1p3p/1Pr5/4Ppn1/1B1P1N1q/1RP2P2/3QR1K1 w - - 0 23
```

Shallow human assessment: White's king is exposed, Black has queen on `h3`,
rook pressure, and a knight near the king. A quiet rook move that does not
address the attack is suspicious. That made the candidate worth deeper
analysis.

## Bad Behavior

Old behavior, reproduced at the pre-fix baseline commit `3099825`, before any
search changes in this branch. Correction history, LMP, and null move pruning
were all enabled by default in that baseline.

Original position:

| Depth | Old move | Old score |
| ---: | --- | ---: |
| 6 | `b2b1` | +97 cp |
| 8 | `b2a2` | +93 cp |
| 10 | `b2a2` | +47 cp |

Color-swapped mirror:

```text
3qr1k1/1rp2p2/1b1p1n1Q/4pPN1/1pR5/1P1P3P/1P3PP1/4R1K1 b - - 0 23
```

| Depth | Old move | Old score |
| ---: | --- | ---: |
| 6 | `e8f8` | +98 cp |
| 8 | `b7a7` | +125 cp |
| 10 | `b6d4` | +31 cp |

The worst old moves are not merely suboptimal. Stockfish depth 14 constrained
search reports:

| Position | Move | Reference result |
| --- | --- | ---: |
| Original | `b2b1` | mate score, about `-99995` cp |
| Original | `b2a2` | mate score, about `-99995` cp |
| Mirror | `b7a7` | mate score, about `-99995` cp |
| Mirror | `e8f8` | mate score, about `-99996` cp |

Example refutation for `b2b1`:

```text
b2b1 c5h5 b3f7 g8f7 f3e5 d6e5 d1f3 h3f3 c2c3 f3f2
```

## Root Cause

The ablation matrix isolated three interacting search heuristics:

1. Correction history was updated inside recursive search. That makes sibling
   evaluations order-dependent during the same search.
2. Late move pruning removed quiet defensive resources in the mirrored
   position.
3. Null move pruning hid the shallow defensive move in the original position
   after the first two issues were addressed.

These heuristics are useful and should remain available in normal play, but the
old implementation let them act in positions where their assumptions were not
valid.

## Fix

The final fix keeps all three heuristics enabled in normal builds:

1. Correction history still adjusts static evaluation, but its table is updated
   only once after a completed root search. Recursive negamax no longer mutates
   corrected evaluation while sibling branches are still being searched.
2. Late move pruning remains enabled, but it is skipped when either king has
   substantial local attack-zone pressure.
3. Null move pruning remains enabled, but it is skipped under the same tactical
   king-pressure condition.

The env-based ablation toggles remain available only in `search-debug` builds:

```text
EMBER_DISABLE_CORR_HIST=1
EMBER_DISABLE_LMP=1
EMBER_DISABLE_NULL_MOVE=1
EMBER_DISABLE_LMR=1
EMBER_DISABLE_FUTILITY=1
EMBER_DISABLE_REVERSE_FUTILITY=1
EMBER_DISABLE_SEE_PRUNING=1
EMBER_DISABLE_HISTORY_PRUNING=1
EMBER_DISABLE_IID_REDUCTION=1
```

Relevant implementation commits:

- `Fix correction history update timing`
- `Guard LMP in tactical king positions`
- `Guard null move in tactical king positions`

## Fixed Behavior

Original position:

| Depth | Fixed move | Fixed score |
| ---: | --- | ---: |
| 6 | `b3c4` | +34 cp |
| 8 | `b3d5` | +11 cp |
| 10 | `b3d5` | +28 cp |

Mirror:

| Depth | Fixed move | Fixed score |
| ---: | --- | ---: |
| 6 | `b6d4` | +11 cp |
| 8 | `b6d4` | +11 cp |
| 10 | `b6d4` | +28 cp |

Reference depth 14 constrained searches rate the fixed moves as non-mating
defenses:

| Position | Fixed move | Reference score |
| --- | --- | ---: |
| Original | `b3d5` | about `-589` cp |
| Mirror | `b6d4` | about `-550` cp |

The fixed moves are not claiming equality; they avoid the forced mate blunder.
At depth 6 the fixed engine currently chooses `b3c4`, which is also an active
non-rook defense and does not repeat either recorded mate-losing rook move.

## Elo Check

To make sure the fix did not simply remove useful search strength, I measured
the pre-fix and fixed engines with `configs/elo/stockfish-adaptive.toml`.
Both runs used 500 games, the `8+0.08` time control, the embedded mini book,
and the Stockfish UCI_Elo-equivalent rating scale.

| Engine | Run id | Games | Elo | 95% CI | SE |
| --- | --- | ---: | ---: | --- | ---: |
| Pre-fix baseline | `s0001-before-3099825-fg` | 500 | 2238 | 2207-2269 | 15.89 |
| Fixed engine | `s0001-after-c9eed16-fg` | 500 | 2364 | 2332-2395 | 16.22 |

The fixed run is +126 Elo relative to the pre-fix baseline. The 95% intervals
do not overlap, so this measurement does not show a strength regression; it
shows a statistically clear improvement under these run conditions.

## Demo Videos

Base position:

- Bad move: `stupidities/cases/S-0001/demo/base-before-bad.mp4`
- Reference move: `stupidities/cases/S-0001/demo/base-reference.mp4`
- Fixed move: `stupidities/cases/S-0001/demo/base-after-fixed.mp4`

Mirror variant:

- Bad move: `stupidities/cases/S-0001/demo/variant-1-before-bad.mp4`
- Reference move: `stupidities/cases/S-0001/demo/variant-1-reference.mp4`
- Fixed move: `stupidities/cases/S-0001/demo/variant-1-after-fixed.mp4`

## Verification

Commands used:

```text
cargo test
cargo test --features "decision-trace search-debug"
cargo build --release --bin ember
tools/check_stupidity_cases.py stupidities/cases/S-0001/case.json
python3 tools/hunt_stupidities.py mine --config configs/stupidity/default.toml --run-id hunt-001 --depth 8 --multipv 4
python3 tools/hunt_stupidities.py verify --config configs/stupidity/default.toml --run-id hunt-001 --top 5 --repeats 3 --create-case
python3 tools/hunt_stupidities.py render --case stupidities/cases/S-0001
```

The match and analysis runs used the Nix flake runner environment.
