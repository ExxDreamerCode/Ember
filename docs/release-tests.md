# Release acceptance tests

This checklist describes the release decision process for comparing a candidate
release against the previous released version. It is intentionally about
evidence and decision criteria, not about one particular release.

Use the same machine and the same settings for both binaries whenever a
comparison is involved. Record the exact candidate revision, previous release
revision or tag, build flags, CPU model, logical CPU count, thread count, hash
size, books, tablebases, time controls, seeds, commands, and result directory.
Keep raw PGNs, logs, JSON summaries, benchmark outputs, and built binary
checksums.

Suggested placeholders:

```text
PREVIOUS_RELEASE=<previous release tag or commit>
CANDIDATE_RELEASE=<candidate commit>
RUN_ID=<release-name>-acceptance
```

## 1. Version and artifact identity

Before any strength testing, confirm that every user-visible version-bearing
place names the candidate release:

- `Cargo.toml` `[package].version`
- `Cargo.lock` entry for `ember-chess`
- UCI `id name Ember <version>`
- release archive metadata
- portable bundle metadata, when producing a portable bundle
- README or release-facing documentation links

Build the candidate from the exact release commit. Run a UCI smoke test against
each produced binary and verify that the UCI id matches the Cargo version:

```bash
python3 tools/smoke_test_uci.py -- path/to/ember
```

Acceptance criteria:

- every built binary reports the candidate version;
- archive metadata and binary checksums refer to the same commit;
- no binary or bundle still identifies itself as the previous release.

## 2. Correctness gates

Run the normal correctness gates first. Stop and fix any real failure before
starting performance or Elo work.

```bash
cargo fmt --all --check
cargo check --locked --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
ulimit -s 65536
export RUST_MIN_STACK=16777216
cargo test --locked --all-features -- --test-threads=1
```

Run the release move-fixture corpus:

```bash
ulimit -s 65536
export RUST_MIN_STACK=16777216
export EMBER_LICHESS_CORPUS_THREADS=<worker-count>
cargo test --locked --release --all-features --test lichess_puzzle_corpus -- --ignored --test-threads=<worker-count>
```

Acceptance criteria:

- formatting, check, clippy, and tests pass;
- every active TSV move fixture passes;
- if a fixture outcome differs from the previous release, the change is
  understood and documented;
- no public move regression is hidden in a Rust-only test or an untracked note.

## 3. Native NPS comparison

Compare previous release and candidate throughput with the opening book disabled.
Use explicit binary paths and the same Hash/Threads/depth/repeat settings.

Recommended coverage:

- Threads: `1,2,4,8,12` when the machine has at least 12 logical CPUs;
- depth: at least `14` for release acceptance;
- repeats: at least `3`;
- Hash: fixed, for example `128 MiB`;
- book disabled.

Example:

```bash
python3 tools/benchmark_search.py \
  --binary previous=path/to/previous/ember \
  --binary candidate=path/to/candidate/ember \
  --baseline previous \
  --depth 14 \
  --repeats 3 \
  --hash-mb 128 \
  --threads <threads> \
  --timeout 180 \
  --out-dir results/release-acceptance/nps-native \
  --run-id threads-<threads>
```

Acceptance criteria:

- no unexplained pooled median NPS regression at any important thread count;
- small deltas near run-to-run noise are rerun before judgment;
- a lower NPS result is acceptable only if search-shape/Elo evidence explains
  and justifies it;
- record scaling relative to one thread for the candidate.

## 4. Windows-cross artifact NPS smoke

Build the Windows amd64 artifact through the release path and smoke-test it. If
the artifact is checked on a non-Windows OS through Wine, treat the result as a
compatibility smoke, not as the final Windows performance truth.

Recommended coverage:

- compare candidate native binary and candidate Windows-cross binary;
- Threads: `1,4,8,12`;
- depth: at least `12`;
- repeats: at least `2`;
- Hash fixed, book disabled.

Acceptance criteria:

- Windows artifact starts and completes the UCI smoke;
- binary format and runtime-linkage checks pass;
- NPS is not catastrophically below the same candidate native build;
- any large difference is followed up with a real Windows benchmark before
  making user-facing performance claims.

## 5. Normal chess head-to-head against previous release

Run a paired-opening head-to-head match between candidate and previous release.
Use random opening starts, color swaps, identical UCI settings, and a short
representative clock.

Recommended default:

- 100 paired openings = 200 games;
- `8+0.08`;
- Threads `1`;
- Hash `64`;
- book disabled unless the release specifically changes the embedded book;
- paired colors;
- fixed seed;
- SPRT enabled with a narrow indifference interval, for example `[-5,+5] Elo`.

Example with the head-to-head harness:

```bash
python3 tools/head_to_head.py \
  --config configs/head-to-head/<release-config>.toml \
  --run-id "$RUN_ID-normal-h2h" \
  --max-pairs 100 \
  all
```

Acceptance criteria:

- candidate is not weaker than the previous release by the configured SPRT or
  fixed-sample paired result;
- no time forfeits;
- color split is not obviously pathological;
- changed outcomes that look suspicious are triaged by finding the first
  candidate move that differs from the previous release;
- if SPRT remains formally inconclusive at the cap, the fixed-sample result,
  confidence interval, and other acceptance tests must still support release.

Report:

- games, WDL, score, score rate;
- Elo estimate and confidence interval;
- LOS or paired probability when available;
- paired openings won/lost/tied;
- SPRT LLR, bounds, pentanomial, and state;
- termination reasons.

## 6. Chess960 head-to-head against previous release

Run the same kind of paired head-to-head test for Chess960. Generate deterministic
Chess960 starts from a recorded seed and play two color-swapped games per start.

Recommended default:

- 100 seeded Chess960 starts = 200 games;
- `8+0.08`;
- Threads `1`;
- Hash `64`;
- book disabled;
- fixed seed;
- record all generated FRC start IDs or FEN/EPD rows.

Acceptance criteria:

- candidate is not weaker than the previous release;
- no time forfeits or protocol aborts;
- both sides castling and FRC position setup work throughout the match;
- color split is not obviously pathological.

Report:

- candidate WDL, score, and score rate;
- candidate result by color;
- Elo estimate and confidence interval when available;
- time forfeits and abnormal terminations;
- path to the generated starts, PGN, and cutechess log.

## 7. Calibrated Elo with static opponents

Run the calibrated opponent-pool Elo measurement for the candidate. This answers
"is the candidate in the expected strength band against known external
opponents?" rather than "is the candidate better than the previous release?"

Recommended default:

```bash
python3 tools/measure_elo.py \
  --config configs/elo/default.toml \
  --run-id "$RUN_ID-elo" \
  --max-games 240 \
  all
```

Acceptance criteria:

- fitted Elo is in the expected release range for the project;
- confidence interval does not indicate a major unexplained drop;
- no time forfeits;
- per-opponent results look plausible, especially around the candidate's
  expected strength;
- any large deviation from recent releases is investigated before tagging.

Report:

- fitted Elo and bootstrap confidence interval;
- games parsed;
- per-opponent scores;
- fitted opponent ratings and caveats;
- archive hash for the raw artifacts.

## 8. Seeded mixed-opponent comparison

Run the deterministic side-by-side comparison where the previous release and
candidate play the same seeded scenarios against external opponents.

Recommended default:

```bash
python3 tools/compare_versions.py \
  --config configs/version-opponents/default.toml \
  --run-id "$RUN_ID-mixed" \
  --baseline-revision "$PREVIOUS_RELEASE" \
  --candidate-revision "$CANDIDATE_RELEASE" \
  --scenarios 64 \
  all
```

Acceptance criteria:

- candidate score rate is not lower than the previous release on the matched
  panel;
- time-forfeit losses do not increase;
- changed outcomes are mostly improvements or are statistically consistent with
  noise;
- every regression is triaged enough to decide whether it is a real blocker;
- time-control breakdowns do not reveal a clock-safety regression.

Report:

- previous and candidate score, score rate, and WDL;
- matched Elo delta and confidence interval;
- score-rate delta and confidence interval;
- paired randomization p-value;
- improved, worsened, and unchanged outcome counts;
- time-forfeit losses by version;
- unavailable configured opponents;
- whether ponder scenarios were included or omitted.

## 9. Long-control ponder smoke

Run a small long-control smoke against a stable reference opponent with pondering
enabled. This is not an Elo measurement. It checks clock handling, ponder
transitions, book interaction, and obvious deployment-like behavior.

Recommended coverage:

- normal chess from the initial position with the embedded book enabled;
- Chess960 from seeded FRC starts;
- `60+0.6` or another clearly longer control than the short statistical tests;
- Threads `1`;
- Hash `64`;
- ponder enabled for both engines when supported;
- at least 8 normal games and 8 Chess960 games.

Acceptance criteria:

- no time forfeits;
- no protocol aborts;
- no stuck games or orphaned engine processes after match completion;
- candidate does not show obvious catastrophic blunders in the sampled games;
- the embedded book does not delay the first normal move.

Report:

- WDL and color split;
- termination reasons;
- any warnings emitted by the harness;
- PGN and log paths.

## 10. Embedded-book immediate-move smoke

Run a direct UCI smoke from `startpos` with the default embedded book. The exact
book move may change across releases; the invariant is that a valid book move is
returned without search nodes.

Example:

```text
uci
isready
ucinewgame
position startpos
go movetime 1
quit
```

Acceptance criteria:

- output contains a legal `bestmove`;
- reported search nodes are `0`;
- no unexpected delay before the book move;
- if random book mode is enabled for a release-specific check, verify that it
  still returns through the book path rather than falling into normal search.

## 11. Release-binary CI checks

Before accepting the release, confirm that CI builds every intended release
artifact and smoke-tests the runnable ones.

Expected coverage:

- Linux amd64;
- Linux arm64;
- Windows amd64;
- Windows arm64;
- macOS amd64;
- macOS arm64;
- native Windows MSVC amd64 build-smoke, if enabled in CI.

Acceptance criteria:

- CI passes on all required jobs;
- produced archives use the expected version, platform, architecture, and commit
  hash prefix in their names and metadata;
- Linux static-linkage checks pass;
- Windows PE format and static CRT checks pass;
- macOS architecture and system-library dependency checks pass;
- packaged UCI smoke tests pass.

## 12. Final release decision

The release is in good shape when:

- all correctness gates pass;
- every binary and archive identifies the candidate version;
- native NPS has no unexplained regression versus the previous release;
- Windows-cross artifact smoke is clean;
- normal chess and Chess960 head-to-head tests show no strength regression;
- calibrated Elo remains in the expected range;
- mixed-opponent comparison does not reveal a release-blocking regression;
- long-control ponder smoke is clean;
- embedded-book immediate-move smoke is clean;
- raw evidence is preserved and the final report records exact configuration.

Block or delay the release when:

- any required correctness gate fails;
- a binary reports the wrong version;
- the candidate has repeated time forfeits;
- normal chess or Chess960 shows a statistically meaningful regression;
- mixed-opponent regressions point to a real search, time-management, book,
  Syzygy, or protocol bug;
- NPS drops materially without compensating search-quality evidence;
- release archives cannot be reproduced or do not match their metadata.

If a result is inconclusive rather than bad, increase the relevant sample size or
rerun the noisy measurement. Do not replace missing evidence with confidence
language in the release notes.

## 13. Report template

Create a release acceptance report and include:

```text
Candidate tested:
Previous release:
Host and CPU:
Date:
Raw artifact directory:

Candidate binary hashes:
Previous release binary hashes:

Correctness gates:
Native NPS table:
Windows-cross NPS smoke:
Normal chess H2H:
Chess960 H2H:
Calibrated Elo:
Mixed-opponent comparison:
Long-control ponder smoke:
Embedded-book smoke:
CI release artifact status:

Conclusion:
Known caveats before tagging:
```

Keep the conclusion factual: say which gates passed, which gates were
inconclusive, which gates could not be run, and why the remaining risk is or is
not acceptable for the release.
