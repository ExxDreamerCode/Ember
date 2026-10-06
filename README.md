<p align="center">
  <img src="logo.png" alt="Ember Logo" width="200">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/rust-nightly--2026--10--01-orange" alt="Rust Version">
  <img src="https://img.shields.io/github/actions/workflow/status/ExxDreamerCode/Ember/ci.yml?branch=main&label=CI" alt="CI Status">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

# 🔥 Ember — a chess engine written in Rust

**Author:** [D.r.e.A.m.e.R](https://github.com/ExxDreamerCode) · **Co-author:** [Boris Nagaev (@starius)](https://github.com/starius)

**Ember** is a UCI-compatible chess engine written in Rust. It is built for learning, experimentation, and steady engine work. The project is under active development and is regularly refined and improved.

Russian version: [docs/README.ru.md](docs/README.ru.md).

## 📋 Requirements

- **Rust nightly** to build from source. Use the pinned toolchain in
  `rust-toolchain.toml` (`nightly-2026-10-01`). The oldest nightly empirically
  verified for this revision is `nightly-2026-02-08`.
- A UCI-compatible chess interface, for example [Arena](http://www.playwitharena.de/), [Cute Chess](https://cutechess.com/), or [Lichess](https://lichess.org/)

## 🔧 Installation

- Download the [latest release](https://github.com/ExxDreamerCode/Ember/releases/latest)

Detailed instructions for reproducible Nix builds, release archives, and the portable Windows bundle are in [BUILD.md](BUILD.md).

## ♟️ Usage

### With a graphical interface

1. Open your UCI-compatible chess program.
2. Add the engine and point it to the downloaded binary.
3. Start playing.

### Command line

```bash
# Interactive mode
cargo run --release

# Or send UCI commands directly
echo -e "uci\nisready\nquit" | cargo run --release
```

### UCI options

| Option | Type | Default | Range | Description |
| --- | --- | --- | --- | --- |
| `Hash` | spin | 256 | 1–4096 | Transposition table size in megabytes |
| `Threads` | spin | 1 | 1–256 | Number of search threads |
| `MultiPV` | spin | 1 | 1–256 | Number of root moves analyzed and reported independently per depth |
| `Move Overhead` | spin | 7 | 0–5000 | Milliseconds reserved per move for GUI and protocol latency when the clock is converted into a search budget |
| `Ponder` | check | false | — | Let the GUI enable pondering with `go ponder` and `ponderhit` |
| `OwnBook` | check | false | — | Let the engine use its own opening book; when false the GUI provides book moves |
| `Book` | string | `<embedded>` | — | Path to a `.bin` opening book used when `OwnBook` is enabled; an empty value disables the book |
| `RandomBookMove` | check | false | — | Pick uniformly among safe book moves within 5 centipawns of the best static evaluation |
| `BookMinMoveWeight` | spin | 2 | 1–65535 | Minimum absolute book move weight |
| `BookMinMoveWeightPermille` | spin | 10 | 0–1000 | Minimum move weight share in permille |
| `NNUE` | string | `<embedded>` | — | Path to an `.nnue` network file |
| `NNUEBackend` | combo | `auto` | `auto`, available backends | Backend used for Ember V1/V2 NNUE search |
| `TraceFile` | string | `<empty>` | — | Path to a `.jsonl` traceback file; advertised only by builds with the `decision-trace` feature |
| `SyzygyPath` | string | `<empty>` | — | Path to a Syzygy tablebase directory with DTZ files |
| `UCI_Chess960` | check | `false` | — | Enable or disable Chess960 |
| `Tune` | string | `<empty>` | — | Runtime overrides for tunable search constants (see [docs/auto-tuning.md](docs/auto-tuning.md)) |

`TraceFile` is compiled in only with the `decision-trace` Cargo feature, which the default
feature set does not include, so the packaged release binaries do not list it. Opt in with
`cargo build --release --features decision-trace` (or
`nix run .#windows-release -- --features decision-trace` for a traced Windows build, see
[BUILD.md](BUILD.md)). Every other option above is available in all builds.

### Syzygy through Nix

The repository contains a Nix target for the complete Syzygy 3-4-5 WDL+DTZ set from the Lichess mirror:

```bash
nix build .#syzygy
```

All 290 files are downloaded as fixed-output derivations with SHA-256 hashes from `nix/syzygy-3-4-5.json`. The resulting path can be passed to the engine:

```text
setoption name SyzygyPath value ./result/share/syzygy/3-4-5
```

This is the up-to-5-piece set and is 983957920 bytes. The `syzygy` alias intentionally points to this smaller set.

For focused tests, `nix build .#syzygy-ci` provides ten pinned WDL/DTZ files
(774944 bytes) under `result/share/syzygy/ci`. This subset covers recorded
positions only; it is not a complete replacement for the 3-4-5 set. Keep a
loaded tablebase directory unchanged while any search may use it. Changing
`SyzygyPath` publishes a new tablebase generation for the next search while
an active search finishes with its existing generation.

Use a separate target for the complete up-to-6-piece set:

```bash
nix build .#syzygy-6
setoption name SyzygyPath value ./result/share/syzygy/3-4-5-6
```

`syzygy-6`, also available as `syzygy-3-4-5-6`, combines the 3-5-piece and 6-piece tables in one directory so transitions after captures can also be probed through Syzygy. The set contains 1020 files and takes 161209573952 bytes, about 150 GiB, so the Nix store needs a large amount of free space. SHA-256 hashes and sizes for the 6-piece files are pinned in `nix/syzygy-6.json`; the manifest can be reproduced with `nix/generate-syzygy-manifest.py` from the Lichess mirror metadata.

### Opening book

The engine supports opening books in Polyglot `.bin` format. A default book is
**embedded** in the binary and is loaded automatically at startup, but it is
only consulted after the standard UCI `OwnBook` option is enabled:

```text
setoption name OwnBook value true
```

`OwnBook` defaults to **false** so rating-list and GUI testing behaves
predictably: the GUI provides book moves and start positions. 
Enabling `OwnBook` selects the book chosen by the `Book` option (`<embedded>` by default).

Ember does not auto-discover `book.bin` next to the executable or in the current
working directory. External books are used only after an explicit UCI `Book`
option combined with `OwnBook = true`.

You can set a book path through UCI:

```text
setoption name Book value C:\path\to\book.bin
```

If the book is in the same directory as the engine, the file name is enough:

```text
setoption name Book value book.bin
```

To **disable** the book, pass an empty value (this overrides an enabled
`OwnBook` as well):

```text
setoption name Book value
```

To return to the embedded book:

```text
setoption name Book value <embedded>
```

Any Polyglot-compatible book is supported, including Stockfish books.

When the engine picks a move from the book, it reports the choice with zeroed
search telemetry and a tag identifying its origin:

```text
info depth 0 score cp 0 nodes 0 nps 0 time 0 pv e2e4 string book move
```

### Neural network (NNUE)

An NNUE network is **embedded** in the binary and loads automatically at startup. An external `net.nnue` file next to the executable is **not required**.

The embedded network is used by default. It is controlled through the `NNUE` UCI option:

```text
setoption name NNUE value                      # disable NNUE and fall back to classic eval
setoption name NNUE value <embedded>            # return to the embedded network
setoption name NNUE value C:\path\to\file.nnue  # load an external network
```

If the file is next to the engine, you can specify only the file name:

```text
setoption name NNUE value my-net.nnue
```

The NNUE backend is selected automatically based on the CPU and applies equally to
Ember V1 and Ember V2 networks. For testing and benchmarking it can be overridden:

```text
setoption name NNUEBackend value scalar
setoption name NNUEBackend value x86-v3
setoption name NNUEBackend value x86-avx512
setoption name NNUEBackend value aarch64-simd512
setoption name NNUEBackend value auto
```

A backend that is not available on the current CPU will be ignored.
On AArch64, `auto` selects `aarch64-simd256`; its V2 dense layers use
runtime-detected DOTPROD instructions when the CPU provides them and baseline
NEON otherwise.

When an external network is loaded, the engine prints its version and architecture:

```text
info string Loaded NNUE v6 my-net.nnue SCReLU (FT=1024 L1=0 L2=0)
```

The `NNUE` option identifies the network format from the file header and supports
the following NNUE architectures:

- **Ember V1** — Ember's own v1 `.nnue` format. A half-based feature transformer with
  king-bucket inputs, optional threat features, `CReLU`/`SCReLU`/pairwise activations, and
  optional hidden layers `L1`/`L2`. Shown as `Loaded NNUE <version> <name> <activation> (...)`.
- **Ember V2** — A container format for Ember v2 architecture with a fixed
  function structure (a semi-transformer with PSQ and threat functions, followed by several 
  `Affine` stacks), whose dimensions are validated at load. Shown as
  `Loaded Ember V2 net <path> (arch hash=... desc="..." ...)`.
  These files already use the compression produced by the NNUE PyTorch tooling;
  they do not need conversion to Ember's compact `ECN1` format.
- **Compact (ECN1)** — Ember's packed compact variant of the native format for V1 nets (magic
  `ECN1`, generated by `training/en/v1/compact_nnue_v1.py`). It shrinks the feature transformer
  to a dense basis plus a correction map, so it is not an independent architecture, just
  a storage layout of the native network. External `ECN1` files load through the `NNUE`
  option; the binary itself now embeds an Ember V2 network.
- **Classic `HalfKP(Friend)`** — Stockfish's classic network format with the
  `HalfKP(Friend)` feature and `AffineTransform` + `ClippedReLU` layers. Shown as
  `Loaded legacy HalfKP net <path>`.

### Archived networks

The repository does not track the archived Ember V1 and V2 networks used by tests and
benchmarks. They are attached to the dedicated network releases
[`v1.1`](https://github.com/ExxDreamerCode/Ember/releases/tag/v1.1) (Ember V1) and
[`v2.2`](https://github.com/ExxDreamerCode/Ember/releases/tag/v2.2) (Ember V2);
Run `python tools/fetch_networks.py` to restore the `networks/` layout locally. Tests and
benchmarks that need these files check for them at run time and skip with a message when
the local copy is absent, so a fresh checkout without networks still builds and passes.

## ⚙️ Configuration

Engine parameters are changed through the UCI `setoption` command:

```text
setoption name Hash value 256
setoption name OwnBook value true
setoption name Book value book.bin
setoption name Move Overhead value 20
setoption name TraceFile value Trace.jsonl   # decision-trace builds only
```

The engine also understands the built-in UCI `bench` command: a fixed-depth search over a
small embedded position corpus that reports nodes, NPS, and a reproducible node-count
signature per run:

```text
bench              # full corpus at depth 10
bench 12           # full corpus at depth 12
bench depth 12 positions 4   # first 4 corpus positions at depth 12
```

It is meant for quick in-process comparisons and release-build smoke tests; the full
search-shape benchmark is described in [docs/quality-assessment.md](docs/quality-assessment.md).

## 📊 Quality assessment

Ratings are not comparable between the three lists.

**[CCRL 40/15](https://computerchess.org.uk/ccrl/4040/)** — Computed 2026-10-03

| Version | Rating | Games |
| --- | ---: | ---: |
| Ember 1.3.1 64-bit | 3361 ± 98 | 25 |
| Ember 1.3.0 64-bit | 3154 ± 20 | 541 |

**[CCRL Blitz (2m+1s)](https://computerchess.org.uk/ccrl/404/)** — Computed 2026-10-03

| Version | Rating | Games |
| --- | ---: | ---: |
| Ember 1.3.1 64-bit | 3389 ± 16 | 1024 |
| Ember 1.1.2 64-bit | 3020 ± 17 | 1056 |
| Ember 0.9.2 64-bit | 1928 ± 20 | 906 |

**[CCRL FRC](https://computerchess.org.uk/ccrl/404FRC/)** — Computed 2026-10-03

| Version | Rating | Games |
| --- | ---: | ---: |
| Ember 1.3.1 64-bit | 3395 ± 16 | 1115 |
| Ember 1.3.0 64-bit | 3200 ± 21 | 750 |

Elo measurement, paired version comparisons, and search-shape benchmarks are documented in
[docs/quality-assessment.md](docs/quality-assessment.md).

## 🛠️ Development

```bash
# Run tests
cargo test

# Check for errors
cargo check

# Run with optimizations
cargo run --release

# Build in release mode
cargo build --release
```

## 🤝 Contributing

Found a bug or have an idea? Open an issue or PR — help and feedback are welcome.

## 📄 License

This project is distributed under the MIT license.

Runtime Syzygy probing uses the maintained MIT
[`pyrrhic-rs` fork](https://github.com/starius/pyrrhic-rs).
Its original Pyrrhic notice is preserved in
[`licenses/PYRRHIC-LICENSE`](licenses/PYRRHIC-LICENSE),
and the locked Rust dependency notices are in
[`licenses/THIRD-PARTY-LICENSES.html`](licenses/THIRD-PARTY-LICENSES.html).
The Syzygy parser and probe use safe Rust; the crate's file-mapping module has
one unsafe operation and requires table files to remain unchanged while mapped.
The fork's GPL Syzygy reference is a separate test process and is excluded
from Ember's runtime dependencies and release packages.

Ember is built, tested, and trained with other people's work, and each of those keeps its own
license, separate from the MIT license above:

- the v1 trainer runs on [Bullet](https://github.com/jw1912/bullet) (MIT) by Jamie Whiting, and the
  v2 trainer on [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch) (GPL-3.0);
- the training data are the publicly available binpacks from
  [official-stockfish/master-binpacks](https://huggingface.co/datasets/official-stockfish/master-binpacks)
  (ODbL);
- the Python tools under `tools/` and `training/` use
  [python-chess](https://github.com/niklasf/python-chess) (GPL-3.0-or-later).
