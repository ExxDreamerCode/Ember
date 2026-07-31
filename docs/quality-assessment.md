# Quality assessment

This document collects the repository tooling used to assess Ember's playing strength,
search behavior, and regression risk. Commands below assume they are run from the repository
root.

## Elo measurement

The repository includes a Nix environment and scripts for automated matches through Cute
Chess:

```bash
nix develop .#elo-runner
python3 tools/measure_elo.py all --config configs/elo/default.toml
```

For a more accurate strength estimate against Stockfish, use the adaptive mode. It first
plays a short pilot match at several `UCI_Elo` values, then picks the Stockfish level
closest to a 50% Ember score and spends the remaining game budget there:

```bash
python3 tools/measure_elo.py all \
  --config configs/elo/stockfish-adaptive.toml \
  --max-games 500
```

`--max-games` sets the upper bound on the number of games that may be scheduled. The
adaptive result is the `Stockfish UCI_Elo equivalent` for the selected time control, book,
and opening set, not an exact external CCRL rating.

To look for regressions between two versions, use the paired comparison against a shared
opponent pool:

```bash
python3 tools/compare_versions.py all \
  --config configs/version-opponents/default.toml \
  --baseline-revision OLD \
  --candidate-revision NEW
```

From the initial seed, it reproducibly selects the opening, time control, opponent, and
ponder mode. Each version plays the same two-game mini-match with colors swapped. The report
lists games where the outcome changed, and the Elo-difference estimate and confidence
interval are computed over shared scenarios rather than treating every game as independent.

The rough cost of precision below is estimated for an 8-core CPU where automatic mode uses
`ceil(8 * 1.5) = 12` workers, time control `8+0.08`, and an opponent calibrated to a roughly
50% result.

| 95% CI | Interval width | Games | Approximate time |
| ---: | ---: | ---: | ---: |
| ±50 Elo | 100 Elo | ~185 | ~8–12 min |
| ±40 Elo | 80 Elo | ~290 | ~13–18 min |
| ±30 Elo | 60 Elo | ~515 | ~22–33 min |
| ±20 Elo | 40 Elo | ~1,160 | ~50–75 min |
| ±15 Elo | 30 Elo | ~2,060 | ~1.5–2.2 h |
| ±10 Elo | 20 Elo | ~4,640 | ~3.3–5 h |
| ±7.5 Elo | 15 Elo | ~8,250 | ~6–9 h |
| ±5 Elo | 10 Elo | ~18,550 | ~13–20 h |

The latest tested CCRL rating for Ember is **3040** ± 70 Elo in single-threaded mode.

## Search-shape benchmark

For regressions where not only NPS matters, but also reached depth, node count, and tree
shape, use the dedicated UCI benchmark:

```bash
cargo build --release
nix run .#search-shape-benchmark -- \
  current=./target/release/ember \
  --repeats 3
```

Several binaries can be compared in one run:

```bash
nix run .#search-shape-benchmark -- \
  good=/path/to/good-ember \
  bad=/path/to/bad-ember \
  --repeats 3 \
  --go-command "go depth 20"
```

By default the script disables the opening book with `setoption name Book value`, uses the
starting position, `Hash=64`, and `Threads=1`. A custom position set can be passed as JSON
through `--positions`.

## Advantage-defense matches

When a suspicious move occurs in a position where Ember already has a large advantage,
analyze more than the root move. Let Ember play the position out against a much stronger
Stockfish that has a larger time budget. This checks whether Ember can preserve the
advantage against an opponent that actively searches for the most inconvenient defensive
resources.

Use this for won endgames, conversion problems, horizon suspicions, and positions where a
move looks odd but a static engine evaluation still says Ember is winning. Run the match
from the exact suspicious position and also from one or two earlier positions. The earlier
positions are important: a move can be individually defensible only because a previous
decision already allowed the opponent's saving resource.

Keep the setup paired and explicit:

- test every candidate move or policy branch from the same position;
- test both the baseline and candidate Ember binaries;
- give Stockfish at least twice Ember's time when the goal is to find Ember's conversion
  weakness rather than to estimate match strength;
- disable books unless book behavior is the subject of the test;
- record the FEN or full move history, side to move, UCI options, Stockfish version, time
  control, result, final termination, and the first move where the advantage materially
  changes.

This technique is not an Elo test. A single saved or spoiled win is diagnostic evidence. If
Stockfish repeatedly holds or wins from a position that should be technically won for Ember,
reduce the game to the earliest failing decision and add the narrowest regression that
captures the underlying invariant.

## Lost-advantage hunting

`tools/hunt_lost_advantage.py` scales the advantage-defense idea into a repeatable corpus
mining pass. It first lets Ember play against stronger Stockfish from randomized book
starts. When Stockfish gets a large advantage immediately after an Ember move, the tool
starts a second game from that position with colors swapped: Ember receives the advantaged
side and Stockfish defends with more time. If Ember loses the advantage, the run records the
first large evaluation drop, writes PGNs and JSON traces, classifies the case, and emits a
disabled TSV row in `tests/fixtures/advantage_preservation.tsv`.

Example:

```bash
python3 tools/hunt_lost_advantage.py \
  --ember ./target/release/ember \
  --stockfish stockfish \
  --cases 100 \
  --seed 20260730 \
  --output-dir results/lost-advantage
```

The generated TSV rows are deliberately disabled. Treat them as a triage queue, not as a
green test suite to satisfy immediately. First inspect the bucket summary, pick the largest
or most clearly causal class, and verify a representative sample with deeper analysis. Only
uncomment a row in the same commit that fixes the underlying class or establishes a narrow
invariant that Ember should already satisfy.

The run is diagnostic rather than statistical. Its value is the preserved artifact set:
source PGNs, replay PGNs, raw UCI logs, per-move JSON, the generated fixture rows, the seed,
engine paths, thread counts, hash, movetimes, and thresholds. Keep those artifacts available
when making a search change from the corpus, because many collected positions are broad
finite-depth conversion weaknesses rather than isolated code defects.

## Comparative witness tracing

Use comparative witness tracing after a corpus or game analysis identifies a concrete Ember
mistake and a strong reference engine has a plausible better move. The goal is not to trace
Stockfish's full tree. First extract one high-quality Stockfish witness line, then ask a
smaller question: did Ember's own fixed-depth search visit the same line, and if it did,
where did Ember evaluate or prune it differently?

`tools/compare_mistake_trace.py` automates this first pass for TSV-backed positions. It
parses active or disabled fixture rows, reconstructs the full move history, labels the root
with Stockfish, runs Ember with `EMBER_TRACE_SEARCH_DAG` restricted to the suspicious root
move and the witness root move, and writes both JSON and Markdown summaries. For repetition
conversion triage, combine:

```bash
python3 tools/compare_mistake_trace.py \
  --ember ./target/release/ember \
  --stockfish stockfish \
  --out-dir results/comparative-traces/repetition \
  --fixture tests/fixtures/advantage_preservation.tsv \
  --bucket repetition-conversion \
  --direct-repetition-only \
  --stockfish-nonrepeat-only
```

Read the report by classifying the first missing witness position, not just the root move.
If the witness root itself is missing, the issue is root move generation, legality, or root
filtering. If only the first reply is missing, the likely cause is ordering, null-window
search, pruning, or a cutoff before the witness line becomes visible. If the witness line
is visited but evaluated differently, inspect the recorded node summaries: static eval
visits, TT flags, qsearch visits, search-cycle returns, claimable draw returns, and automatic
draw returns. If Ember and the reference move have equal root scores, treat that as a
tie-breaking/order hypothesis, not as proof that any non-repeating move is safe.

Use the trace result to propose the narrowest policy, then run the normal quality gates. A
policy that fixes disabled fixture rows but loses a paired head-to-head gate is rejected;
keep the trace artifact and leave the rows disabled until a narrower cause is found. The
technique is evidence for where to look next, not a substitute for Elo, NPS, active fixture,
and clock-safety checks.
