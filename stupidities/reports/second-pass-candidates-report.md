# Second-Pass Candidate Report

## Summary

I investigated four saved candidates from `stupidity-002-broad-01`:

- `C-0004`
- `C-0030`
- `C-0027`
- `C-0026`

No source fix was accepted. Two candidates had plausible IID-related fixes, but
every fixing source variant failed the no-regression Elo rule. The source tree
therefore stays unchanged.

The accepted baseline for comparison is `s0002-after-19d9940`: 2382 Elo,
95% CI 2350-2414, 500 games.

## C-0004

```text
r7/pp1k1p2/2p3p1/q2pBP1p/2nP3P/2b3P1/PPP1PRB1/1K1R4 w - - 0 21
```

Current behavior:

| Depth | Ember move |
| ---: | --- |
| 6 | `f5g6` |
| 8 | `f2f3` |
| 10 | `f2f3` |

Stockfish depth 14 refutes `f2f3` as mate in 3:

```text
f2f3 a5b4 b2b3 b4a3 b3c4 a3b2
```

I rejected this candidate for this pass because the ablation did not isolate a
single root cause. Full selective-search removal changes the move, but
individual toggles do not produce a clean fixed behavior. This remains
suspicious and should be revisited as a compound-search failure.

## C-0030

```text
6k1/1p4pp/4p3/pP2P3/2B3P1/r1P4q/5P1P/3Q2K1 b - - 0 38
```

Current behavior:

| Depth | Ember move |
| ---: | --- |
| 8 | `h3h4` |
| 10 | `h3h4` |

Stockfish depth 16 ranks `h3h6` first:

| Move | Stockfish depth 16 score |
| --- | ---: |
| `h3h6` | -443 cp |
| `h3h4` | -954 cp |

The ablation matrix isolated IID reduction:

| Build | Depth 8 move |
| --- | --- |
| Baseline | `h3h4` |
| `EMBER_DISABLE_IID_REDUCTION=1` | `h3h6` |
| Other single toggles | `h3h4` |

However, normal-build source variants that fixed C-0030 needed to guard IID
reduction throughout substantially more of the PV, and those variants lowered
the Elo point estimate:

| Variant | Run id | Games | Elo | 95% CI |
| --- | --- | ---: | ---: | --- |
| All king-pressure PV nodes | `s0003-iid-guard` | 500 | 2375 | 2342-2407 |
| All queen/king-pressure PV nodes | `s0003-queen-iid-guard` | 500 | 2342 | 2310-2373 |

Both were rejected under the no-regression rule.

## C-0027

```text
8/1k6/1P4p1/1K3p2/2N2P1P/8/2r5/8 b - - 0 53
```

Current behavior:

| Depth | Ember move |
| ---: | --- |
| 8 | `c2c4` |
| 10 | `c2h2` |

Stockfish depth 14 scores `c2c4` at -721 cp and prefers active rook moves
such as `c2f2`. I rejected this candidate because the bad move disappears at
depth 10, and every single selective-search toggle keeps `c2c4` at depth 8.
This looks like a horizon/evaluation weakness rather than a clean dominated
heuristic bug.

## C-0026

```text
5Q2/8/2p1p1k1/1p1p1b2/8/4q3/8/3K4 b - - 0 85
```

Current behavior:

| Depth | Ember move |
| ---: | --- |
| 8 | `d5d4` |
| 10 | `d5d4` |

Stockfish depth 16 scores `d5d4` as 0 cp, while active queen moves keep a
winning advantage:

| Move | Stockfish depth 16 score |
| --- | ---: |
| `d5d4` | 0 cp |
| `e3d3` | 1087 cp |

A narrow root-child IID guard fixed this candidate:

```text
ply <= 1 && (king_pressure >= 3 || queen_pressure >= 2)
```

It changed Ember to `e3d3` at depths 8 and 10 and preserved S-0001/S-0002.
But the 500-game Elo run still fell below the accepted baseline:

| Variant | Run id | Games | Elo | 95% CI |
| --- | --- | ---: | ---: | --- |
| Root-child queen/king guard | `s0003-root-iid-guard` | 500 | 2361 | 2329-2393 |

This variant was rejected under the no-regression rule.

## Conclusion

No S-0003 fix was accepted. The best future direction is `C-0004`, because it
is a concrete mate miss that still reproduces, but it needs a deeper compound
search investigation rather than another local guard.
