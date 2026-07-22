# Ember development guide

This file applies to the whole repository. It records the working practices that have
proved useful while debugging search, SMP, time management, Syzygy, opening-book, and
deployment behavior.

Treat this as a living document. When a repeatable development, debugging, or
quality-assurance practice proves useful, update this file as part of the same work. Keep
the guidance general and actionable rather than tied to one incident, and revise or remove
advice when the repository's workflow changes.

## Guiding principle

Treat a bad game, a suspicious move, an NPS change, and an Elo result as different kinds
of evidence. Reproduce the behavior, identify the responsible subsystem, add the narrowest
useful regression, and only then change the engine. A plausible chess explanation is not
yet a code bug, and a faster benchmark is not proof of greater playing strength.

Use this dependency chain as the default mental model:

```text
observation
  -> preserve artifacts and exact configuration
  -> locate the first meaningful divergence
  -> reproduce it deterministically
  -> classify the responsible subsystem
  -> add regression coverage
  -> make the smallest causal change
  -> pass correctness checks
  -> compare NPS and search shape
  -> compare playing strength and clock safety
  -> build and smoke-test deliverables
```

Do not skip directly from an anecdotal game result to a broad search heuristic change.

### Stable behavioral baseline

Treat the `V1.1.2` release as Ember's stable behavioral baseline. For every proposed change
that can affect move choice, compare the relevant fixtures and game scenarios against both
the immediate parent and `V1.1.2`. A position that only `V1.1.2` solves is a regression to
investigate before accepting the change.

The baseline is a floor, not an oracle. Do not restore a `V1.1.2` move when strong analysis
shows that the newer move is better, and do not preserve a known old bug. Record the
evidence whenever an intentional change breaks a previously passing `V1.1.2` case. Use
`tools/compare_fixture_corpus.py` to compare active and disabled position regressions across
two binaries. This UCI-level comparison supplements rather than replaces the in-process
fixture suite; investigate any difference between those paths instead of silently choosing
the more convenient result.

## Workspace and reproducibility

- Inspect the branch, worktree, and recent history before editing. Preserve unrelated
  tracked changes and all user-owned untracked files.
- Never use destructive Git operations to clean a shared worktree. Do not rewrite history
  below a base the user has declared stable.
- Record exact revisions, build flags, CPU model, logical CPU count, thread count, hash size,
  books, tablebase paths, time controls, seeds, and commands. A result without this context
  is hard to compare or reproduce.
- Identify any uncommitted experiment explicitly in its result directory; a revision name
  alone does not identify a dirty tree.
- Preserve raw logs, PGNs, JSON summaries, engine traces, and benchmark output needed to
  audit a conclusion.
- Do not run two CPU-bound comparisons concurrently on the same machine. They contaminate
  timing and NPS results.

## Regression policy

### Chess move regressions belong in TSV fixtures

Represent a position in `tests/fixtures/*.tsv` whenever the assertion is fundamentally
"in this chess history or position, Ember should choose or avoid these moves." Do not add a
one-off Rust test for a case that can be represented losslessly in this format.

The fixture runner automatically discovers every regular `.tsv` file directly under
`tests/fixtures/`. Adding a row or another TSV fixture must not require a Rust change. All
fixture files use this header:

```text
id	depth	fen_before_blunder	setup_move	expected_move	themes	rating	popularity	plays
```

Fixture conventions:

- IDs must be descriptive and globally unique across all fixture files.
- Put a comment immediately before every hand-written regression. Explain the source,
  link to the game or report when possible, state what went wrong, and state the intended
  invariant. Separate hand-written cases with an empty line.
- Use `-` for no setup move. Otherwise use one UCI move or a space-separated UCI move
  history.
- Preserve the full move history when repetition, the fifty-move counter, castling rights,
  en-passant state, or other history-sensitive behavior matters. A final FEN alone is not
  always an equivalent reproducer.
- `expected_move` accepts one exact move, alternatives separated by `|`, or forbidden
  moves after `!`. Prefer an invariant such as "do not play the losing move" when several
  continuations are sound.
- Choose the lowest stable depth that still exercises the bug, unless the failure itself
  depends on a documented deployment-like depth. Depth-based cases should remain
  deterministic and reasonably cheap.
- Keep source metadata in `themes`, `rating`, `popularity`, and `plays` when it exists. Use
  neutral zero values for hand-written cases where it does not.
- If a valuable position still fails and no generally safe fix exists, keep a commented-out
  row with a `DISABLED` explanation. Do not weaken an active assertion merely to make an
  unsafe engine change appear green.
- Before prioritizing mined or externally sourced disabled cases, verify their expected
  moves with a current strong reference engine at a recorded search budget. Keep supported
  cases separate from disagreements and near-ties so Ember is not tuned to an obsolete or
  subjective label.
- Do not duplicate a TSV move regression in Rust.

Use Rust tests for behavior the TSV schema cannot express: UCI protocol ordering, stop and
ponder lifecycle, clock budgets, thread coordination, node accounting, option handling,
resource cleanup, parser behavior, and other subsystem invariants. Python tooling and the
deployment tooling should have their regressions in their existing Python test suites.

The fast fixture test validates the TSV schema, numeric fields, and cross-file ID uniqueness.
The ignored release fixture test runs every active move case and is exercised by its
dedicated CI job.

## Diagnosing a bad game or move

1. Preserve the original PGN and engine artifacts. Prefer raw UCI output and clocks over a
   reconstructed account from the board alone.
2. Confirm the exact Ember binary or revision and all UCI options. Check book, Threads,
   Hash, Ponder, SyzygyPath, and the real time control.
3. Find the first meaningful Ember divergence, not merely the move after the evaluation has
   already collapsed. Record the evaluation before and after each candidate from the same
   side-to-move convention.
4. Analyze the suspicious position with a strong Stockfish build at a stable, explicitly
   recorded node or depth budget. Compare Ember's move with the best alternative, then
   follow both lines for several moves. A shallow one-ply comparison is often misleading.
5. Reproduce Ember's choice with the original move history and deployment settings. Then
   vary one dimension at a time: book on/off, one thread versus deployment threads, fixed
   depth versus clocked search, clean versus reused process, and tablebases on/off.
6. Compare the same position on known-good and candidate revisions with identical binaries,
   settings, and hardware. Use a targeted history search or bisect when the first bad
   revision is unknown.
7. Classify the failure before editing:

   - **Book:** Was the position actually in the book? Was the selected move legal, within
     the configured quality window, and evaluated from the correct side? Did the engine
     intentionally leave book or silently fail to load it?
   - **Time management:** Distinguish allocated search time from wall-clock time and UCI
     overhead. Inspect increment reserve, move overhead, soft/hard stops, predicted next
     iteration cost, ponder transitions, and time remaining after `bestmove`.
   - **SMP:** Check leader ownership of the final root move, worker stop propagation, stale
     results, root-lane assignment, aggregate node accounting, and whether worker activity
     ends promptly after `bestmove`.
   - **Search/evaluation:** If the same bad move is stable with one and many threads, suspect
     evaluation, selectivity, extensions/reductions, quiescence, transposition reuse, or
     horizon effects before blaming SMP.
   - **Persistent state:** Compare a fresh process with a process that has played the full
     game. Inspect history aging, cached root ordering, repetition state, transposition
     tables, and NNUE incremental state.
   - **Syzygy:** Verify the material count, complete WDL/DTZ availability, path contents,
     root probe result, fifty-move semantics, and the transition into smaller tablebases.
     A six-piece set is not a replacement for the three-to-five-piece files.
   - **Deployment infrastructure:** Separate engine failures from match scheduling, input
     stream termination, game aborts, harness state, and subprocess lifecycle failures.

Only fix behavior when the evidence identifies a bug or a defensible generally better
decision. If the root cause remains a broad finite-depth weakness, record the position and
the competing hypotheses instead of tuning narrowly to one game.

## Correctness gates

Run checks in increasing cost order and stop on a real failure:

1. `cargo fmt --all --check`
2. The narrow unit, integration, fixture, or Python test covering the change
3. `cargo check --locked --all-features`
4. `cargo clippy --locked --all-targets --all-features -- -D warnings`
5. `cargo test --locked --all-features -- --test-threads=1` with the repository's documented
   stack limits
6. The ignored release move-fixture suite when chess behavior changed
7. Relevant old-CPU, cross-architecture, packaging, or deployment tests

Use the Nix `ci` shell where CI does. Match `.github/workflows/ci.yml` rather than inventing
a subtly different command.

Every bug fix should have a regression at the narrowest useful layer. A regression proves
the causal invariant, not just that the final game happens to end differently.

When adding a special search ordering or extension, test both the intended motif and nearby
counterexamples that must not qualify. Prefer predicates that describe the candidate move
itself over a position-wide trigger such as "some rook check exists"; a broad trigger can
change the search of every unrelated root move. Run the complete move-fixture corpus after
changing eligibility, because several individually reasonable heuristics can overlap.

## Performance and playing-strength gates

Correctness, speed, search shape, clock safety, and Elo are separate gates. Report all
relevant ones; do not use one as a proxy for another.

### Reproducible comparisons

- Build baseline and candidate from explicit revisions with the same toolchain and release
  flags.
- Run them on the same otherwise-idle machine. Keep Hash, Threads, books, tablebases,
  openings, seeds, opponents, ponder mode, and time controls identical.
- Treat fixed-depth move choices as configuration-dependent results. In particular,
  transposition-table size changes replacement collisions and can change a principal
  variation without any UCI/library bug. Match the in-process fixture defaults when
  cross-checking through UCI, make deliberate overrides explicit, and record Hash in the
  result artifact.
- Use paired openings and swap colors. Fixed seeds make a rerun diagnostic rather than a
  new experiment.
- Warm up before timing and use multiple repetitions. Prefer medians and distributions over
  a single sample.
- Save the complete result directory, not just a summary copied into chat or a PR.

### NPS and search shape

Use `tools/benchmark_search.py` for throughput and
`nix run .#search-shape-benchmark` for depth, nodes, elapsed time, and tree-shape changes.
Disable the opening book unless book behavior is the subject of the test.

For SMP work, cover `Threads=1,2,4,8,12` when the machine has at least 12 logical CPUs. Do not
request more active threads than the hardware can execute when judging scaling. Record both
total NPS and scaling relative to one thread. Also inspect reached depth and node count:
higher NPS can accompany a worse tree, and lower NPS can accompany better pruning.

Use at least three repeats for meaningful before/after measurements. If the delta is close
to run-to-run noise, rerun rather than declaring a regression or improvement. CI's quick NPS
job is a smoke test, not an Elo or performance proof.

### Elo and game comparisons

Choose the harness that matches the question:

- `tools/head_to_head.py` compares two Ember configurations directly with paired book
  starts and colors.
- `tools/compare_versions.py` compares two Ember revisions against identical seeded
  scenarios drawn from stronger and weaker external opponents, real time controls, opening
  starts, and ponder settings. Its changed-outcome list is a triage queue for deeper
  analysis.
- `tools/measure_elo.py` estimates strength against the configured opponent pool or a
  calibrated Stockfish level.

Always report games, wins/draws/losses, score, Elo estimate, confidence interval, LOS when
available, color split, and termination reasons. Do not call a small WDL difference an
improvement without statistical support. For paired external-opponent tests, compute
uncertainty over paired scenarios rather than pretending every game is independent.

Use pentanomial SPRT when a head-to-head match should stop sequentially. Define the Elo
indifference interval, alpha, beta, minimum pair count, and maximum pair count before the
match. Count paired-opening outcomes from 0 through 2 points and inspect the recorded LLR
and bounds. Repeatedly checking an ordinary fixed-sample p-value after each batch does not
preserve its advertised false-positive rate and must not be presented as an SPRT result.

When a candidate has worsened outcomes, locate the first Ember move that differs from the
baseline and analyze both choices with strong Stockfish. Compare the immediate balance and
several subsequent moves. Look for a repeated signature across games before changing a
general heuristic.

### Clock safety

Time-management and SMP changes require clocked matches in addition to fixed-depth tests.
Include an extreme increment control such as `1+0.01`, a representative short control such
as `8+0.08`, and a less compressed control when practical. Inspect time forfeits separately
from chess losses.

For selected games, record time spent and time remaining per move. Check that search stops
within its hard budget, workers become idle promptly after `bestmove`, ponder transitions do
not leak work, and obvious forced replies do not receive pathological budgets. Opponent time
may inform strategy only through explicit, tested policy; never assume the opponent clock is
already incorporated.

## Commit and history discipline

- Make each commit one accomplished, reviewable part of the work. Separate fixtures/tests,
  engine behavior, tooling, Nix opponents, packaging, and documentation when they are
  independently meaningful.
- Put regression coverage in a separate commit from the behavioral fix when practical.
  Keep submitted history coherent and buildable. For a known move weakness that cannot yet
  be fixed safely, add a commented fixture rather than making every intermediate commit red.
- If the starting code is unformatted, format it in a dedicated first commit. Do not hide
  logic changes in formatting noise.
- Fold late compile, CI, or packaging corrections into the commit that introduced the
  problem before publication. Avoid a visible back-and-forth sequence when the final design
  can be expressed directly.
- Write imperative subjects. Use the body to explain the invariant, cause, and important
  trade-off, not to narrate every edit. Wrap commit-description lines.
- Before committing, inspect the staged diff and verify that the description matches it.
- Do not commit PR prose, scratch plans, downloaded reports, PGNs/results, tablebase
  archives, torrents, build outputs, or generated packages unless the repository explicitly
  tracks that artifact.
- Never rewrite commits below a user-specified boundary. After a rebase, add new commits
  unless the user explicitly authorizes another history rewrite.

## Nix, opponents, and Syzygy

- Keep Nix inputs reproducible: pin exact upstream revisions and hashes. Do not silently
  replace an opponent binary or source release under the same package definition.
- Add opponent packages separately from the comparison or test that consumes them. This
  keeps licensing/build review distinct from experimental methodology.
- Treat Syzygy manifests as exact datasets. Verify file counts, WDL/DTZ pairing, store paths,
  and material coverage. Test `3-4-5-6` against `3-4-5` or no Syzygy as complete
  configurations, not as a misleading six-piece-only directory.

## Definition of done

A change is done when:

- the cause is understood well enough to justify the implementation;
- the appropriate regression exists and passes;
- formatting, tests, and relevant platform checks pass;
- NPS/search shape and Elo/clock gates appropriate to the risk show no unexplained
  degradation;
- raw evidence is preserved and the exact comparison configuration is documented;
- commits are atomic, accurately described, and free of unrelated files;
- requested outputs are produced and verified.

If a gate cannot be run, say exactly which one and why. Do not replace missing evidence with
confidence language.
