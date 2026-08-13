# Auto-tuning search constants

The auto-tuning system searches for stronger values of search constants through
sequential SPRT matches against the current best set. It does not recompile the
engine for each candidate: a single release binary uses one UCI option `Tune`
to distinguish the "current best" from the "candidate" on the fly.

## How it works

### Engine-side infrastructure

The `src/tune.rs` module keeps global atomic overrides for the marked
constants. In normal games (when `Tune` is not set) the hot-path read does a
single `load(Relaxed)` and returns the compile-time default — practically no
overhead.

The tunable constants and their defaults:

### Selectivity (src/search/selectivity.rs)

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `PROBCUT_MIN_DEPTH` | 8 | Minimum depth for ProbCut |
| `PROBCUT_MARGIN_CP` | 350 | ProbCut margin in centipawns |
| `ROOT_REPETITION_TIE_MIN_SCORE` | 300 | Minimum score to prefer a non-repeating root move |

### Reverse futility / futility pruning (src/search/negamax.rs)

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `REVERSE_FUTILITY_BASE_CP` | 80 | RFP margin base |
| `REVERSE_FUTILITY_PER_DEPTH_CP` | 65 | RFP margin per depth |
| `REVERSE_FUTILITY_MAX_DEPTH` | 8 | Maximum RFP depth |
| `FUTILITY_MARGIN_PER_DEPTH_CP` | 150 | Futility margin per depth |
| `FUTILITY_MAX_DEPTH` | 3 | Maximum futility depth |

### Null move

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `NULL_MOVE_MIN_DEPTH` | 3 | Minimum depth for null move |
| `NULL_MOVE_REDUCTION_BASE` | 3 | Reduction base |
| `NULL_MOVE_REDUCTION_DIVISOR` | 4 | Depth divisor in reduction |
| `NULL_MOVE_MARGIN_DIVISOR` | 200 | (eval − beta) divisor in reduction |
| `NULL_MOVE_MARGIN_CAP` | 3 | Cap on the margin part of the reduction |
| `NULL_MOVE_KING_PRESSURE_LIMIT` | 3 | Max king pressure allowed for null move |
| `NULL_MOVE_NON_PAWN_LIMIT` | 4 | Minimum non-pawn material for null move |

### Pruning (negamax.rs)

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `SEE_MARGIN_PER_DEPTH_CP` | 80 | SEE pruning margin per depth |
| `HISTORY_PRUNE_MARGIN_PER_DEPTH` | 1024 | History pruning margin per depth |
| `HISTORY_PRUNE_MAX_DEPTH` | 5 | Maximum history pruning depth |

### Selectivity (negamax.rs)

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `CHECK_EXTENSION_MAX_DEPTH` | 16 | Maximum depth for check extension |
| `LMP_MAX_DEPTH` | 8 | Maximum late move pruning depth |
| `IID_MIN_DEPTH` | 4 | Minimum depth for internal iterative deepening |
| `LMR_DIVISOR_MILLIS` | 1800 | LMR reduction divisor (ln(move)·ln(depth)·1000 / divisor) |

### UCI interface

```text
option name Tune type string default <empty>
```

Applying one or more overrides:

```text
setoption name Tune value "PROBCUT_MARGIN_CP=400,PROBCUT_MIN_DEPTH=12"
```

`Tune` value formats:

- CSV: `NAME=VALUE,NAME2=VALUE2` — recommended for cutechess, since a single
  `option.Tune=...` contains no spaces;
- whitespace: `NAME VALUE` — one parameter per call.

Viewing active overrides:

```text
tune
```

Example response:

```text
info string tune PROBCUT_MIN_DEPTH = 12
info string tune PROBCUT_MARGIN_CP = 400
```

Clearing:

```text
setoption name Tune value ""
```

Important: the `ucinewgame` command does **not** reset overrides. This is so the
setting survives the transition between games inside a match.

## The auto-tuner

The tools live in `tools/auto_tune/`:

| File | Purpose |
| --- | --- |
| `tune.toml` | Parameter descriptions, ranges, SPRT and openings |
| `seek.py` | Coordinate descent: iterates neighbours of each parameter, runs SPRT against the current best, keeps a journal |
| `apply.py` | Shows values from `best.json` that differ from the defaults, for manual porting into the code |
| `best.json` | Current optimal values (created automatically) |
| `journal.jsonl` | Full journal of every SPRT match |

### Running

`seek.py` is a plain Python script; it does not have to run inside Nix. It
launches matches through `head_to_head.py`, so real tuning needs the same
dependencies as the head-to-head runner:

- Python 3.11+ with the standard `tomllib` module;
- the `python-chess` package (see `requirements.txt`);
- `cutechess-cli` on `PATH`;
- a release engine binary (the `--engine` argument).

All of this is provided by the `nix develop .#elo-runner` dev-shell, or you can
install the dependencies directly (`pip install -r requirements.txt` +
`cutechess-cli`):

```bash
# Build the release binary (only needed once)
cargo build --release --bin ember

# Tune all parameters with the given time control
python tools/auto_tune/seek.py --time-control 8+0.08

# Tune selected parameters
python tools/auto_tune/seek.py --params PROBCUT_MIN_DEPTH,PROBCUT_MARGIN_CP

# Show the found values
python tools/auto_tune/apply.py

# Rehearse without real matches (no cutechess or binary needed)
python tools/auto_tune/seek.py --dry-run

# Limit parallel games (default auto = all logical CPUs)
python tools/auto_tune/seek.py --workers 4

# Use half of the logical CPUs
python tools/auto_tune/seek.py --worker-multiplier 0.5
```

Controlling parallelism inside a match:

- `--workers N` — number of parallel games (cutechess `-concurrency`).
  Default `auto`: `floor(logical_cpus * worker_multiplier / threads)`.
- `--worker-multiplier X` — fraction of logical CPUs used for workers
  (default `1.0`). For example `0.5` on an 8-core machine gives 4 workers.

Reducing the workers **slows down** the match but frees CPU for other tasks.

If you work in the Linux `nix develop .#elo-runner` dev-shell, then
`cutechess-cli`, `python-chess` and the toolchain are already available, and
`--engine` defaults to `target/release/ember`, which builds there.

### How decisions are made

1. The current value of a parameter is taken (from `best.json`, or the default
   from `tune.toml`).
2. The neighbour `current + step` is tried, then `current - step`.
3. Each candidate is compared with the current best via `head_to_head.py run`
   with pentanomial SPRT enabled (elo0=0, elo1=5, alpha=beta=0.05).
   `engine_a` is the incumbent, `engine_b` is the candidate.
4. The candidate is accepted only when SPRT rejects the null hypothesis
   (`engine_b_better` — "candidate is better"). The `engine_a_better` verdict
   ("incumbent is better") means the candidate is rejected, and
   `inconclusive`/`continue` means not enough data. After acceptance the value
   becomes the new best and the process repeats in the same direction.
5. When both neighbours are rejected, a wider step `current + 2*step` is tried.
6. The parameter settles when no neighbour passes — the result is marked
   "settled".

Time controls from `common.time_controls` rotate between SPRT matches in a
round-robin fashion (the counter comes from the number of entries in
`journal.jsonl`). The `--time-control` flag overrides the whole set. After each
accepted value `best.json` is rewritten immediately, not only at the end of the
run, so an interrupted tuning can be resumed.

All matches are recorded in `journal.jsonl`: parameter, old/new value, verdict,
accepted or not, Elo, score rate, pairs/games, LLR, SHA-256 of the binary,
time control, SPRT parameters. This makes any result reproducible and explains
why a value was accepted or rejected.

### Run reports

For each match, two files are created in `results/tune/<run_id>/`:

- **`report.md`** — human-readable report: verdict (accepted/rejected/
  inconclusive), Elo (candidate − incumbent), score rate, pairs/games, LLR,
  time control, SPRT parameters, SHA-256 of the binary, time.
- **`report.json`** — machine-readable version: `record` (all journal fields) +
  `summary` (full statistics from `head_to_head.py`).

`run_id` has the form `tune-<param>-<value>-<timestamp>`, so each run is easy
to find and match against a `journal.jsonl` entry.

### `tune.toml` configuration

```toml
results_dir = "results/tune"

[common]
time_controls = ["8+0.08", "1+0.01"]   # which time controls to play
max_pairs = 1000                        # pair limit per SPRT
min_pairs = 20
seed = 20260714
opening_source = "polyglot"
polyglot_book = "src/book.bin"
hash_mb = 64
threads = 1
cutechess_cmd = "cutechess-cli"

[sprt]
enabled = true
elo0 = 0
elo1 = 3
alpha = 0.10
beta = 0.05

[[params]]
name = "PROBCUT_MIN_DEPTH"
base = 8
min = 4
max = 16
step = 1
```

`seek.py` builds a temporary head-to-head TOML config for each match:
`engine_a` is the incumbent with values from `best.json`, `engine_b` is the
candidate differing only in the tuned parameter. The other parameters already
improved earlier are passed to both sides identically, so each match measures
only one parameter.

## Important limitations

- The auto-tuner does **not** change code — it only writes `best.json` and
  `journal.jsonl`. Values from `apply.py` must be ported into `src/` manually
  and confirmed with your own SPRT before committing.
- It is recommended to run on an idle machine and not keep parallel
  CPU-bound processes — timings and NPS will be distorted.
- SPRT with `elo0=0, elo1=3` is a strict test: small improvements may need many
  pairs. `max_pairs` limits the time spent on unpromising candidates.
- Changing a parameter value affects the search tree shape, so after accepting
  a value you should always run the usual correctness checks
  (`cargo test --all-features`, `cargo clippy`) and compare NPS/search shape.

## Adding a new parameter

1. Add a variant to `TuneParam` in `src/tune.rs` (name, index, `from_name`).
2. At the usage site of the constant, replace the read with
   `tune::get_int(TuneParam::NewParam, DEFAULT)`.
3. Add a `[[params]]` entry to `tools/auto_tune/tune.toml`.
4. Run `cargo test --all-features` and make sure the default path is unchanged.