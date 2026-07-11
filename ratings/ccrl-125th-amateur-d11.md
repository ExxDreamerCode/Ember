# CCRL 125th Amateur D11 Notes

Ember 1.1.1 was accepted into CCRL's 125th Amateur Division 11 event.
That was useful coverage because the games were produced by an external
tournament harness, against engines we do not control, with real time
management and long game histories. It exposed a class of bugs that was
easy to miss in isolated position tests.

## Recreate The Data

Downloaded game archives are ignored under `ratings/ccrl/`. Recreate the
local dump with:

```sh
python3 tools/download_ccrl_ember_games.py \
  --archive-url https://ccrl.live/pgns/125th_Amateur_D11/ \
  --engine "Ember 1.1.1" \
  --filename-match ember_1.1.1 \
  --scan-mode filename \
  --out-dir ratings/ccrl/125th_Amateur_D11/ember_1.1.1
```

The same event later included Ember 1.1.2. Recreate that local dump with:

```sh
python3 tools/download_ccrl_ember_games.py \
  --archive-url https://ccrl.live/pgns/125th_Amateur_D11/ \
  --engine "Ember 1.1.2" \
  --filename-match ember_1.1.2 \
  --scan-mode filename \
  --out-dir ratings/ccrl/125th_Amateur_D11/ember_1.1.2
```

Build or fetch all packaged CCRL replay opponents with:

```sh
nix build .#ccrl-opponents
```

The aggregate output exposes all replay wrappers under `result/bin/`:

```sh
ls -1 result/bin/ccrl-*
```

For an interactive shell with those wrappers on `PATH`:

```sh
nix shell .#ccrl-opponents
```

If flakes are not enabled globally, add:

```sh
--extra-experimental-features "nix-command flakes"
```

## Mistake Reproduction And Replay

Ember 1.1.1 was rebuilt from commit `bd752d9`; its UCI banner identifies
as `Ember 1.1.1`. All six mistakes reproduced with full UCI move history,
not from bare FEN. That detail matters because these are repetition-state
bugs: the same piece placement without the same history can search
normally.

The current branch was then replayed from each bad-move position with the
pinned CCRL opponent packages. Replay settings were close to the
tournament harness: `600+10`, `Threads=1`, `Hash=64`, and opening books
disabled. The replay harness feeds both engines the full
`position startpos moves ...` history on every move.

Run the replay harness after building current Ember and the pinned
opponents:

```sh
nix build .#ccrl-opponents
nix develop .#elo-runner --command cargo build --release --bin ember
nix develop .#elo-runner --command python3 tools/replay_ccrl_bad_moves.py \
  --ember target/release/ember \
  --opponents-dir result/bin
```

The harness writes PGNs, UCI logs, and JSON summaries under the ignored
`ratings/ccrl/replays/` tree.

| Game | Ply | Ember 1.1.1 bad move | Repetition move | Ember 1.1.1 output | Stockfish check | Current move | Replay result |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| 15 Seawall | 42... | `Kf8` / `f7f8` | `Kg8` / `f7g8` | `depth 64 score cp 0` | repeat `-91`, actual `-513` | `f7g8` | `1-0`, checkmate after 94 plies; still lost |
| 24 PawnStar | 33... | `a5` / `a6a5` | `Bf6` / `d4f6` | `depth 64 score cp 0` | repeat `0`, actual `-249` | `d4f6` | `1/2-1/2`, threefold after 2 plies; saved draw |
| 34 Revolver | 111... | `Qh4+` / `e4h4` | `Qd4` / `e4d4` | `depth 64 score cp 0` | repeat `0`, actual `-1147` | `e4h7` | `1/2-1/2`, threefold after 31 plies; saved draw |
| 38 PawnStar | 37... | `a5` / `a6a5` | `Rh8` / `e8h8` | `depth 64 score cp 0` | repeat `-53`, actual `-213` | `e8h8` | `1-0`, checkmate after 104 plies; still lost |
| 46 Puffin | 55. | `Rb6+` / `a6b6` | `Ra7` / `a6a7` | `depth 64 score cp 0` | repeat `0`, actual `-1461` | `a6a7` | `1/2-1/2`, threefold after 57 plies; saved draw |
| 60 KnightX | 71. | `Rg8+` / `g7g8` | `Rh7` / `g7h7` | `depth 64 score cp 0` | repeat `0`, actual `-1309` | `g7h7` | `1/2-1/2`, threefold after 3 plies; saved draw |

Stockfish scores are centipawns from Ember's side to move, so lower is
worse for Ember in each row. The key finding is that the chat hypothesis
was right: these were 0.00 repetition-exit blunders. Ember 1.1.1 saw a
drawn repeating move, but selected a non-repeating losing move and still
reported `cp 0` at depth 64 almost instantly. This reproduced without
warming the transposition table as long as the full move history was
supplied.

The Revolver current replay did not choose the exact historical
repetition move: `e4d4` and current `e4h7` both evaluate as drawn, while
the old `e4h4` is mate-losing. Current searches also no longer jump
straight to the suspicious depth-64 `cp 0` pattern; they finish at normal
depths with normal node counts.

## Local Snapshot

The downloaded snapshot contains 11 Ember PGNs against 8 unique opponent
versions:

| Opponent | Games | Ember Score |
| --- | ---: | ---: |
| byte-knight 4.0.0 | 1 | 1.0 |
| KnightX 4.92 | 1 | 0.0 |
| OliThink 5.11.9 | 1 | 0.5 |
| Pawnstar 0.13.593 | 2 | 0.0 |
| Puffin 5.0 | 1 | 0.0 |
| Rengar v2.1.1 | 1 | 1.0 |
| Revolver 2.0 | 2 | 0.0 |
| seawall 20250322 | 2 | 1.0 |

Total in the local PGN snapshot: 3.5/11. The downloaded
`tournament-results.json` was live data and still reported 8 games when it
was captured, so treat the PGNs as the source of truth for this local
analysis snapshot.

## Conclusions

The main tournament lesson was not the raw score. The important result was
that Ember 1.1.1 repeatedly preferred losing non-repetition moves in drawn
repetition positions while still assigning them a draw score. Those games
gave us concrete, externally generated regression cases.

The root cause was not visible from a final FEN alone. It depended on the
engine's history state: `Engine::new` called `set_fen`, which already seeded
the repetition stack, then pushed the initial hash again while leaving
`rep_stack_len` at 1. Later moves were appended after the stale entry, so
the active repetition stack lagged the board by one ply. Root search then
tested repetition against the wrong history state and could mark unrelated
losing child moves as draws.

The fix was commit `efb5237` (`Fix initial repetition stack seeding`), and
the regression coverage now replays the full UCI move histories instead of
checking only reduced positions.

Replaying the games with the fixed engine recovered four likely draws:
PawnStar game 24, Revolver game 34, Puffin game 46, and KnightX game 60.
Games 15 and 38 still lost after the corrected move under the replay
settings, so not every observed blunder changed the final result.

## Lessons Learned

- Keep full move histories for tournament bugs. FEN-only reproducers can
  hide bugs in repetition, fifty-move, castling, en-passant, and other
  state carried outside the piece placement.
- Regression tests for game-state bugs should drive the same public
  protocol path used by tournament managers. For these cases that means
  replaying UCI `position ... moves ...` histories before searching.
- Package exact opponent versions early. Replaying the games later is much
  easier when source tags, binary archives, NuGet/Cargo dependencies, and
  checksums are frozen at the same time as the PGNs are analyzed.
- Do not trust engine banners as version proof. The packaged
  `Revolver_2.0` tag still reports `id name Revolver 1.0`; the package is
  pinned to the 2.0 source tag and intentionally leaves that banner
  unpatched.
- Keep downloaded tournament data out of git, but keep the downloader and
  exact commands tracked. That gives reproducibility without committing
  third-party PGN archives.
- Separate deployed-version analysis from current-branch analysis. Ember
  1.1.1 was the version in the CCRL event; the current branch can be fixed
  while still needing tests that reproduce the deployed failure.

## Follow-up Opponents

The second pass added the remaining requested source-built opponent:
Eidolon 1.0.0 from `Daniel729/Eidolon` tag `v1.0.0`. The package removes
one obsolete stable Rust feature gate and builds the upstream source with
`RUSTC_BOOTSTRAP=1` for `portable_simd`. Rengar 2.1.1, Puffin 5.0, and
Revolver 2.0 were already packaged from source tags, so no binary archive
or Wine wrapper is needed for these four opponents.

Exact target archive links:

| Game | Result | Archive |
| --- | --- | --- |
| Revolver 2.0 - Ember 1.1.1 | `1-0` | <https://ccrl.live/pgns/125th_Amateur_D11/6_revolver_2.0_vs_ember_1.1.1.pgn> |
| Revolver 2.0 - Ember 1.1.1 | `1-0` | <https://ccrl.live/pgns/125th_Amateur_D11/34_revolver_2.0_vs_ember_1.1.1.pgn> |
| Ember 1.1.1 - Puffin 5.0 | `0-1` | <https://ccrl.live/pgns/125th_Amateur_D11/46_ember_1.1.1_vs_puffin_5.0.pgn> |
| Eidolon 1.0.0 - Ember 1.1.1 | `1-0` | <https://ccrl.live/pgns/125th_Amateur_D11/108_eidolon_1.0.0_vs_ember_1.1.1.pgn> |
| Ember 1.1.2 - Rengar v2.1.1 | `0-1` | <https://ccrl.live/pgns/125th_Amateur_D11/117_ember_1.1.2_vs_rengar_v2.1.1.pgn> |
| Eidolon 1.0.0 - Ember 1.1.2 | `1/2-1/2` | <https://ccrl.live/pgns/125th_Amateur_D11/124_eidolon_1.0.0_vs_ember_1.1.2.pgn> |
| Ember 1.1.2 - Revolver 2.0 | `0-1` | <https://ccrl.live/pgns/125th_Amateur_D11/126_ember_1.1.2_vs_revolver_2.0.pgn> |
| Revolver 2.0 - Ember 1.1.2 | `1/2-1/2` | <https://ccrl.live/pgns/125th_Amateur_D11/126_revolver_2.0_vs_ember_1.1.2.pgn> |

The bad-move scan used Stockfish depth 12 from Ember's side of the board
to rank evaluation drops, then probed exact release binaries built from
`V1.1.1` (`bd752d9`) and `V1.1.2` (`7c191a5`) with full UCI histories and
book disabled.

| Game | Candidate | Stockfish drop | Exact release reproduction | Read |
| --- | --- | ---: | --- | --- |
| 6 Revolver-Ember 1.1.1 | `22...b6` / `b7b6` | ~81 cp | Not reproduced from 50-5000 ms; exact 1.1.1 preferred `...c5` or `...Be8`. | No stable one-move blunder found; the loss looks like accumulated strategic drift. |
| 34 Revolver-Ember 1.1.1 | `111...Qh4+` / `e4h4` | ~638 cp | Repeats instantly at all tested movetimes with `depth 64 score cp 0`. | Same stale repetition-stack bug: a losing non-repetition move was scored as a draw. A one-ply Stockfish replacement did not save the result with exact 1.1.1 continuing. |
| 46 Ember-Puffin 1.1.1 | `55.Rb6+` / `a6b6` | ~662 cp | Repeats instantly at all tested movetimes with `depth 64 score cp 0`. | Same stale repetition-stack bug. Replacing only this move with Stockfish's rook move changed the continuation from a loss to a draw. |
| 108 Eidolon-Ember 1.1.1 | `64...f2` / `f3f2` | ~77 cp | Repeats from 50-5000 ms. | Already losing rook/pawn ending; a one-ply Stockfish replacement still lost. |
| 117 Ember-Rengar 1.1.2 | `27.Ng4` / `e3g4` | ~89 cp | Repeats in bounded search at the tournament-like short time. | Tactical miss. Replacing only this move with Stockfish's `Ra7` changed the continuation from a loss to a win. |
| 124 Eidolon-Ember 1.1.2 | `20...Bb7` / `c8b7` | ~63 cp | Repeats only in short searches, then switches to `...Rxb7`. | Drawn game; the one-ply replacement did not produce a completed win inside the cap. |
| 126 Ember-Revolver 1.1.2 | several candidates | -- | `44.e6` repeats, but Stockfish does not consistently reject it at higher depth. | No tested one-ply replacement, including earlier candidates such as `12.Rc1`, `13.h4`, `20.Ka1`, and `24.c3`, improved the result. |
| 126 Revolver-Ember 1.1.2 | `48...Qd6` / `d8d6` | ~118 cp | Time-sensitive; not consistently reproduced. | Drawn game; the Stockfish replacement line did not improve the result. |

The one-ply replacement test confirmed only two result-changing root
moves in this follow-up set: Puffin game 46 and Rengar game 117. The
Revolver game 34 move is still a clear bug-level evaluation failure, but
exact Ember 1.1.1 can still lose after a single corrected move. The
Revolver 1.1.2 loss appears to be an already bad game state rather than a
single isolated move in the tested candidate set.
