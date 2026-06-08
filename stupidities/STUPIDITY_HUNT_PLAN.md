# Stupidity Hunt Plan

This plan describes how to turn Ember game runs into a repeatable workflow for
finding high-profile bad decisions, proving that they are systemic, fixing their
root causes, and producing human-readable demos.

## Goal

Find decisions where Ember clearly worsens its position in a way that is not a
reasonable strategic trade-off, prove the issue across related positions, fix the
engine code, and keep each fixed case as a replayable regression.

The authoritative evidence should be machine-readable:

- The exact game history and position before the decision.
- The move Ember chose.
- The legal alternatives and reference engine analysis.
- The later game continuation or refutation showing the damage.
- The fixed Ember decision on the same state and variants.

Video or GUI output is useful for demos, but it should be generated from those
records rather than treated as primary evidence.

## Current Project Hooks

The existing Elo runner is the best starting point:

- `tools/measure_elo.py` already builds Ember, runs Cute Chess matches, stores
  PGNs/logs, writes metadata, and archives artifacts.
- `configs/elo/default.toml` and `configs/elo/stockfish-adaptive.toml` already
  define repeatable match settings.
- `flake.nix` already provides the intended runner shell with `cutechess`,
  `stockfish`, `gnuchess`, `fairymax`, `cargo`, and `python3`.

The main engine hooks are:

- `src/main.rs`: UCI parsing, especially `position` and `go`.
- `src/engine.rs`: root move selection in `Engine::find_best_move`.
- `src/search.rs`: likely root-cause surface for search behavior, including
  TT use, qsearch, null move pruning, futility pruning, LMP, SEE pruning, LMR,
  aspiration windows, history/counter moves, and correction history.
- `src/book.rs`: book move selection currently uses wall-clock time, so book
  usage must be disabled or made deterministic for reproducible hunting.

## Phase 1: Reproducible Hunt Runs

Add a separate stupidity-hunt configuration instead of overloading Elo configs:

- `configs/stupidity/default.toml`
- fixed Ember binary path
- fixed hash size
- fixed opening file
- deterministic book setting, preferably disabled for discovery unless the
  specific goal is to audit the book
- separate run directory under `results/stupidity/<run-id>`
- configurable opponent modes:
  - Ember vs Ember
  - Ember vs limited Stockfish
  - Ember vs GNU Chess
  - Ember vs full Stockfish at short controls for tactical pressure

Use the existing `measure_elo.py` structure for build/probe/smoke/report, but
create a separate tool so Elo reports stay focused.

Proposed tool:

```text
tools/hunt_stupidities.py
```

Commands:

```text
probe
build
smoke
run
mine
rank
verify
render
report
all
```

## Phase 2: Structured Decision Tracing

Add structured tracing outside normal UCI stdout. UCI stdout must stay
protocol-clean for Cute Chess.

Preferred control:

- environment variable: `EMBER_TRACE_DIR`
- optional UCI option: `TraceFile`

Decision tracing should be compiled only for hunt builds with the
`decision-trace` Cargo feature so normal play pays no tracing overhead.

Each Ember decision should emit one JSONL record containing:

- run id, game id, engine side, ply number
- full UCI position command or reconstructed move history
- FEN before the decision
- side to move
- legal moves
- chosen move
- whether the move came from book or search
- depth limit or time budget
- final score, nodes, elapsed time
- per-depth best move, score, nodes, PV
- root move score table when explicitly enabled
- current engine options and build metadata

Minimal record shape:

```json
{
  "schema": 1,
  "event": "ember_decision",
  "game_id": "run-001/game-003",
  "ply": 27,
  "fen": "...",
  "position_command": "position startpos moves ...",
  "side": "white",
  "legal_moves": ["..."],
  "chosen_move": "e2e4",
  "source": "search",
  "depth": 8,
  "score_cp": 42,
  "nodes": 123456,
  "elapsed_ms": 95,
  "pv": ["e2e4", "..."]
}
```

For candidate analysis, add an optional expensive root analysis mode that records
scores for all legal root moves at the requested depth. This should be disabled
by default in match play and enabled during verification.

## Phase 3: Candidate Mining

Mine candidates from PGNs plus trace records.

Use several triggers:

- Ember loses after a sharp evaluation drop.
- Stockfish says the chosen move is much worse than the best legal move.
- Ember misses a forced mate or allows a forced mate.
- Ember hangs major material without compensation.
- Ember enters a repeated pattern of dominated choices across similar positions.
- Ember's own search score is stable or positive, but strong reference analysis
  immediately refutes the move.

Candidate score components:

- eval loss in centipawns
- mate swing severity
- material loss
- game outcome impact
- reproducibility count
- simplicity and human-demo value
- root-cause diagnosability

Initial high-profile threshold:

- chosen move is worse than best legal move by at least 300 cp, or
- chosen move changes a non-losing position into a losing or mate-threatened
  position, or
- chosen move allows a forced mate that at least one alternative avoids.

Store mined candidates as JSONL:

```text
results/stupidity/<run-id>/candidates.jsonl
```

Each candidate should include:

- source PGN path
- trace record id
- FEN before move
- move history
- Ember move
- reference best move
- reference eval before and after
- refutation PV
- short explanation generated from the facts

## Phase 4: Prove It Is Stupidity

A candidate should not enter the improvement collection until it passes two
tests.

First, prove the move is actually bad:

- Run strong Stockfish MultiPV on the exact FEN.
- Compare the chosen move with the top alternatives.
- Replay the chosen move and the reference move for several plies.
- Require a clear, stable eval gap or mate proof.
- Reject cases where the reference engine shows the move is a plausible
  sacrifice, fortress attempt, repetition attempt, time-management choice, or
  mixed-strategy trade-off.

Second, prove the issue is systemic:

- Replay the exact state repeatedly with the same fixed depth/time.
- Generate related positions:
  - color-swapped and board-mirrored variants when legal
  - nearby legal continuations before the bad move
  - similar positions mined from other games
  - neutral mutations that preserve the core tactic or strategic feature
- Keep the case only if Ember repeats the bad decision or same class of decision
  across multiple related states.

Verification output:

```text
results/stupidity/<run-id>/verified/<case-id>.json
```

## Phase 5: Root-Cause Analysis

Add search ablation controls. These can be UCI options, environment variables,
or a dedicated debug build feature.

Useful toggles:

- disable transposition table
- disable aspiration windows
- disable null move pruning
- disable reverse futility pruning
- disable futility pruning
- disable late move pruning
- disable SEE pruning
- disable late move reductions
- disable correction history
- disable qsearch SEE filtering
- force full-width root move scoring

For each verified candidate, run an ablation matrix:

```text
baseline
no_tt
no_null_move
no_lmr
no_lmp
no_see_pruning
no_futility
no_correction_history
full_width_root
```

The desired output is a root-cause statement such as:

```text
Case S-0007 reproduces at depths 5-8. It disappears when LMR is disabled and
when full-depth re-search is forced for checking quiet moves. The bad move is
ranked first because the refuting quiet move is reduced and never re-searched.
```

Only after this step should the engine fix be attempted.

## Phase 6: Fix and Regression Collection

Create a persistent case collection:

```text
stupidities/
  cases/
    S-0001/
      case.json
      before.pgn
      after.pgn
      notes.md
      demo/
```

Each `case.json` should contain:

- schema version
- short title
- root-cause category
- original FEN
- move history
- bad move
- acceptable moves after fix
- rejected moves after fix
- reference engine version and settings
- reference eval and PV
- verification variants
- exact replay settings for Ember

Regression command:

```text
python3 tools/hunt_stupidities.py verify --cases stupidities/cases
```

Pass criteria:

- Ember no longer chooses the rejected move on the exact original state.
- Ember no longer chooses equivalent rejected moves on the variants.
- The new move is within an accepted eval gap from the reference move.
- No new high-profile candidate appears in a short smoke hunt.

## Phase 7: Human Demo Output

Generate demos from verified case data.

Suggested outputs:

- annotated PGN
- static board SVG or PNG frames
- HTML replay page
- MP4 generated from frames

Demo contents:

- the game segment leading to the bad decision
- the bad Ember move highlighted
- reference best move and refutation PV
- evaluation bar before and after the decision
- the fixed Ember decision on the same position
- a small set of verified related positions showing the pattern is gone

Rendering should be deterministic and scriptable:

```text
python3 tools/hunt_stupidities.py render --case stupidities/cases/S-0001
```

Add `ffmpeg` and a board rendering dependency to the Nix shell when this phase
starts.

## Immediate Implementation Backlog

1. Add `tools/hunt_stupidities.py` with `probe`, `build`, `smoke`, and `run`
   by borrowing from `tools/measure_elo.py`.
2. Add `configs/stupidity/default.toml`.
3. Add FEN serialization to the Rust board layer.
4. Add JSONL decision tracing at `Engine::find_best_move`.
5. Disable or make deterministic book moves during hunt runs.
6. Parse PGN move text and pair it with Ember trace records.
7. Add Stockfish analysis for candidate ranking.
8. Add exact-state replay verification.
9. Add search ablation controls.
10. Add the first persistent case under `stupidities/cases/`.

## Notes From Initial Inspection

- `cargo test` passes, but there are currently no Rust tests.
- Direct PATH in the inspected shell did not include `stockfish`,
  `cutechess-cli`, or `gnuchess`; the intended way to run match tooling is
  through `nix develop .#elo-runner`.
- Keep the Elo runner intact. Stupidity hunting needs richer move-level
  telemetry and replay, while Elo measurement needs clean aggregate results.
