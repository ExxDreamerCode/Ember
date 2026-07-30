# Build and release guide

This document describes the reproducible build and release workflows for Ember.
The main `README.md` stays focused on installation and engine usage.

## Release binaries with Nix

Each supported operating system and architecture has an individual Nix target:

```bash
# On Linux
nix build .#ember-linux-amd64
nix build .#ember-linux-arm64
nix build .#ember-windows-amd64
nix build .#ember-windows-arm64

# On macOS
nix build .#ember-macos-amd64
nix build .#ember-macos-arm64
```

Linux binaries are fully static and use musl. Windows binaries statically link
the MSVC CRT. macOS always keeps the system `libSystem` dynamic, but release
binaries must not depend on Homebrew, MacPorts, the Nix store, or any other
third-party runtime libraries.

x86-64 builds keep runtime dispatch for accelerated AVX2 and AVX-512
implementations on capable processors. The macOS x86-64 build also runs under
Rosetta 2.

The release CI builds all six variants in two jobs. The Linux job builds Linux
and Windows artifacts. The native ARM64 macOS job runs the full test suite and
builds both macOS architectures. Before upload, Linux and both macOS binaries run
a short UCI smoke test, Windows amd64 runs under Wine, and Windows arm64 is
checked as a valid PE ARM64 binary with a statically linked CRT. The workflow can
also be started manually with `Run workflow` on the GitHub Actions page.

Each binary is packaged separately. Windows uses ZIP; Linux and macOS use
`tar.gz`. Example names:

```text
ember-1.2.0-01234567-linux-amd64.tar.gz
ember-1.2.0-01234567-windows-arm64.zip
```

The directory inside the archive uses the same name as the archive without its
extension. It contains the binary, `BUILD-INFO.txt` with the platform,
architecture, version, full commit hash, and binary SHA-256, and
`SHA256SUMS.txt` with the binary checksum.

## Download release archives from CI

To download all six release archives from one GitHub Actions run into
`release/`, pass the workflow run ID from the run URL:

```bash
python3 tools/download_release_archives.py RUN_ID
```

The downloader fetches both CI artifacts, extracts the ready-to-publish release
archives, and validates that the complete platform set belongs to one version
and one commit.

For a private repository, or when GitHub requires authentication, pass a token in
`GH_TOKEN` or `GITHUB_TOKEN`. To download artifacts from a repository other than
the current Git remote, pass `--repo OWNER/REPOSITORY`.

### Create a GitHub token for artifact downloads

Use a fine-grained personal access token:

1. Open GitHub and click your avatar in the upper-right corner.
2. Go to `Settings` -> `Developer settings` -> `Personal access tokens` ->
   `Fine-grained tokens`.
3. Click `Generate new token`.
4. Use a clear name, such as `download Ember CI artifacts`, and choose a short
   expiration.
5. Set `Resource owner` to the owner of the repository.
6. Set `Repository access` to `Only select repositories` and select the Ember
   repository.
7. In `Repository permissions`, set `Actions` to `Read-only` and `Contents` to
   `Read-only`.
8. Generate the token and save it immediately. GitHub shows it only once.

Example using a token saved in a local file:

```bash
GH_TOKEN="$(tr -d '\r\n' < ../gh-token.txt)" \
  python3 tools/download_release_archives.py RUN_ID --repo OWNER/REPOSITORY
```

## Windows engine binaries with Nix

On Linux, the reproducible MSVC-ABI Windows binary is built as a regular Nix
package using `cargo-xwin`, Clang, and LLD:

```bash
nix build .#windows-ember
```

The resulting binary is:

```text
result/bin/ember.exe
```

This target builds a self-contained binary with the MSVC CRT statically linked
and the `x86-64-v3` CPU baseline. It is intended for modern x86-64 processors
with AVX2, BMI2, and FMA. The portable ZIP uses the same package, and all xwin
settings live in `nix/windows-ember.nix`.

On x86-64 Windows and Linux, Ember uses statically linked `mimalloc` by default.
ARM64, macOS, and other targets keep the system allocator. To build an x86-64
comparison binary with the system allocator, pass Cargo `--no-default-features`.

For iterative development, a compatibility frontend is still available. It
builds into `target/xwin/x86_64-pc-windows-msvc/release/ember.exe` and allows an
older-CPU baseline to be selected:

```bash
EMBER_WINDOWS_TARGET_CPU=x86-64 nix run .#windows-release
```

Additional `rustc` flags are passed through `EMBER_WINDOWS_RUSTFLAGS`.
Additional Cargo arguments go after `--`:

```bash
EMBER_WINDOWS_RUSTFLAGS="-C debuginfo=1" \
  nix run .#windows-release -- --features decision-trace
```

Both Windows paths use the same Nix definition for the Windows SDK, CRT, and
Cargo arguments. The iterative frontend caches downloaded `cargo-xwin` files in
`$HOME/.cache/cargo-xwin`. Using the Microsoft CRT and Windows SDK implies
acceptance of their licenses.

## Portable Ember + lichess-bot for Windows

The fully self-contained Windows ZIP includes Ember, embedded Python,
`lichess-bot`, and the sequential challenge runner. It is built with one Nix
target:

```bash
nix build .#windows-portable
```

The result is:

```text
result/ember-lichess-windows.zip
result/ember-lichess-windows.zip.sha256
```

All downloaded build inputs are pinned in Nix and verified by cryptographic
hashes. The ZIP also contains `SHA256SUMS.txt`, which is checked by `Verify.cmd`
and automatically before each run. User-owned `battle.toml` and the `results/`
directory are intentionally not part of the internal checksum manifest. Python
`__pycache__` directories and bytecode files are also ignored because they can
be regenerated by the local interpreter during normal use.

Short Windows usage flow:

1. Extract the whole ZIP into a writable directory, for example
   `C:\EmberBattle`.
2. Open `battle.toml` in Notepad and configure the games. The bundled default
   configuration is ready for casual 3+2 games against random available bots
   from the opponent pool.
3. Run `Verify.cmd`.
4. Run `Run Battle.cmd`, review the printed plan, type `YES`, enter the number
   of games or `INF` for continuous play, then enter the Lichess token in the
   hidden prompt.

The token is never embedded into source files, Nix derivations, the ZIP, or
`battle.toml`. It is read only from the `LICHESS_BOT_TOKEN` process environment
or from the hidden prompt and is passed to `lichess-bot` in memory. The bot
account token must have `bot:play` and `challenge:write` permissions.

The default opponent pool contains 50 community bots of varying strength. Their
ratings and availability change over time. The runner checks online/busy status
for the whole pool in one request, shuffles the available opponents, and
challenges one of them. If the challenge is rejected or no answer arrives within
`challenge_timeout_seconds`, the runner cancels it and tries the next opponent.
A Lichess response with an exact `please wait until ...` timestamp becomes a
cooldown for that bot only.

`Threads` is resolved automatically from the number of logical Windows CPUs,
capped by Ember's supported maximum of 256, and is used consistently by the
pre-game NPS benchmark and the actual games. `hash_mb` in `battle.toml` is the
Ember transposition-table size in MiB, not a checksum. The allowed range is
1-4096; the default is 1024.

Games are direct challenges and are run strictly sequentially. The chosen game
count repeats configured templates in a loop, and `INF` keeps playing until
Ctrl-C. One `lichess-bot` process and one controller stream are used for the
whole series. Each `[[games]]` entry may use the legacy single
`opponent = "name"` field or a pool with `opponents = ["bot1", "bot2"]`.

If the whole pool is offline, busy, or on cooldown, the runner waits without a
timeout by default and polls status every 15 seconds. This is expressed as
`opponent_wait_timeout_seconds = 0`. Each individual challenge waits 15 seconds
by default (`challenge_timeout_seconds = 15`) before being cancelled. Attempts,
rejections, cooldowns, wait times, and selected opponents are logged. Token and
permission errors remain fatal.

`scoring` and `tags` are labels for later analysis. Casual/rated mode is
controlled by `mode`. Matchmaking and Syzygy are disabled. The runner stays in a
visible window and does not install a service, configure autostart, create a
scheduled task, or change Windows power/sleep settings, registry, `PATH`, or
firewall rules.

After a challenge is accepted, only `lichess-bot` reads the live game stream.
The runner follows the local log and waits for the saved PGN, so two processes
do not compete for request limits on the same token. HTTP 429 during the stream
waits for the required backoff and retries the same request. If the game worker
exits before `gameFinish`, the run fails with an explicit error instead of
reporting an interrupted game as successfully finished.
