<p align="center">
  <img src="logo.png" alt="Ember Logo" width="200">
</p>

# 🔥 Ember — a chess engine written in Rust

<p align="center">
  <img src="https://img.shields.io/badge/rust-1.70%2B-orange" alt="Rust Version">
  <img src="https://img.shields.io/badge/UCI-compatible-brightgreen" alt="UCI Compatible">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
</p>

**Ember** is a UCI-compatible chess engine written in Rust. I build it for learning, experimentation, and steady engine work. The project is under active development and is regularly refined and improved.

Russian version: [docs/README.ru.md](docs/README.ru.md).

## 📋 Requirements

- **Rust** 1.70 or newer
- A UCI-compatible chess interface, for example [Arena](http://www.playwitharena.com/), [Cute Chess](https://cutechess.com/), or [Lichess](https://lichess.org/)

## 🔧 Installation

- Download the [latest release](https://github.com/ExxDreamerCode/Ember/releases/tag/V1.2.0)

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
| `Book` | string | `<embedded>` | — | Path to a `.bin` opening book |
| `RandomBookMove` | check | false | — | Pick uniformly among safe book moves within 5 centipawns of the best static evaluation |
| `BookMinMoveWeight` | spin | 2 | 1–65535 | Minimum absolute book move weight |
| `BookMinMoveWeightPermille` | spin | 10 | 0–1000 | Minimum move weight share in permille |
| `NNUE` | string | `<embedded>` | — | Path to an `.nnue` network file |
| `NNUEBackend` | combo | `auto` | `auto`, available backends | Backend used for NNUE search |
| `TraceFile` | string | `<empty>` | — | Path to a `.jsonl` traceback file |
| `SyzygyPath` | string | `<empty>` | — | Path to a Syzygy tablebase directory with DTZ files |
| `UCI_Chess960` | string | `false` | — | Enable or disable Chess960 |

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

Use a separate target for the complete up-to-6-piece set:

```bash
nix build .#syzygy-6
setoption name SyzygyPath value ./result/share/syzygy/3-4-5-6
```

`syzygy-6`, also available as `syzygy-3-4-5-6`, combines the 3-5-piece and 6-piece tables in one directory so transitions after captures can also be probed through Syzygy. The set contains 1020 files and takes 161209573952 bytes, about 150 GiB, so the Nix store needs a large amount of free space. SHA-256 hashes and sizes for the 6-piece files are pinned in `nix/syzygy-6.json`; the manifest can be reproduced with `nix/generate-syzygy-manifest.py` from the Lichess mirror metadata.

### Opening book

The engine supports opening books in Polyglot `.bin` format. A default book is **embedded** in the binary and is loaded automatically when no `book.bin` is found next to the executable.

Load priority:

1. `book.bin` next to the executable
2. `book.bin` in the current working directory
3. The **embedded book** if no external book is found

You can set a book path through UCI:

```text
setoption name Book value C:\path\to\book.bin
```

If the book is in the same directory as the engine, the file name is enough:

```text
setoption name Book value book.bin
```

To **disable** the book, pass an empty value:

```text
setoption name Book value
```

To return to the embedded book:

```text
setoption name Book value <embedded>
```

Any Polyglot-compatible book is supported, including Stockfish books.

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

The NNUE backend is selected automatically based on the CPU. For testing and benchmarking it can be overridden:

```text
setoption name NNUEBackend value scalar
setoption name NNUEBackend value x86-v3
setoption name NNUEBackend value x86-avx512
setoption name NNUEBackend value aarch64-simd512
setoption name NNUEBackend value auto
```

A backend that is not available on the current CPU will be ignored.

When an external network is loaded, the engine prints its version and architecture:

```text
info string Loaded NNUE v6 my-net.nnue SCReLU (FT=1024 L1=0 L2=0)
```

## ⚙️ Configuration

Engine parameters are changed through the UCI `setoption` command:

```text
setoption name Hash value 256
setoption name Book value book.bin
setoption name TraceFile value Trace.jsonl
```

## 📊 Quality assessment

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
