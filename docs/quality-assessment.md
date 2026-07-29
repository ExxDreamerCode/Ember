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
