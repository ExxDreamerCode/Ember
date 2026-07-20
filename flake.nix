{
  description = "Ember Elo measurement runner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs {
            inherit system;
            overlays = [ (import rust-overlay) ];
          }));
    in
    {
      apps = forAllSystems (pkgs:
        let
          rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          windowsEmber = import ./nix/windows-ember.nix {
            inherit pkgs;
            lib = pkgs.lib;
          };
          crossAarch64 = pkgs.pkgsCross.aarch64-multiplatform.stdenv.cc;
          crossAarch64Libc = pkgs.pkgsCross.aarch64-multiplatform.stdenv.cc.libc;

          search-shape-benchmark = pkgs.writeShellApplication {
            name = "search-shape-benchmark";
            runtimeInputs = with pkgs; [
              python3
            ];
            text = ''
              exec python3 tools/benchmark_search_shape.py "$@"
            '';
          };


          x86_64-qemu-oldcpu-smoke = pkgs.writeShellApplication {
            name = "x86_64-qemu-oldcpu-smoke";
            runtimeInputs = with pkgs; [
              coreutils
              gnugrep
              python3
              qemu
              rustToolchain
              stdenv.cc
            ];
            text = ''
              if [ "$(uname -m)" != "x86_64" ]; then
                echo "skipping x86_64 QEMU smoke on $(uname -m)"
                exit 0
              fi

              unset RUSTFLAGS
              cargo build --locked --release --bin ember

              python3 - target/release/ember <<'PY'
              import os
              import queue
              import subprocess
              import sys
              import threading
              import time

              env = os.environ.copy()
              env["EMBER_SEARCH_BACKEND"] = "auto"
              process = subprocess.Popen(
                  ["qemu-x86_64", "-cpu", "Nehalem", sys.argv[1]],
                  stdin=subprocess.PIPE,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  text=True,
                  bufsize=1,
                  env=env,
              )
              assert process.stdin is not None
              assert process.stdout is not None

              lines: queue.Queue[str] = queue.Queue()

              def collect_stdout() -> None:
                  assert process.stdout is not None
                  for line in process.stdout:
                      lines.put(line)

              reader = threading.Thread(target=collect_stdout, daemon=True)
              reader.start()

              commands = [
                  "uci",
                  "isready",
                  "setoption name Book value",
                  "position startpos",
                  "go depth 1",
              ]
              for command in commands:
                  process.stdin.write(command + "\n")
              process.stdin.flush()

              deadline = time.monotonic() + 30.0
              bestmove_seen = False
              while True:
                  remaining = deadline - time.monotonic()
                  if remaining <= 0:
                      break
                  try:
                      line = lines.get(timeout=min(0.5, remaining))
                  except queue.Empty:
                      if process.poll() is not None:
                          break
                      continue
                  print(line, end="")
                  sys.stdout.flush()
                  if line.startswith("bestmove "):
                      bestmove_seen = True
                      break

              if process.poll() is None:
                  try:
                      process.stdin.write("quit\n")
                      process.stdin.flush()
                  except BrokenPipeError:
                      pass

              if not bestmove_seen:
                  if process.poll() is None:
                      process.kill()
                  process.wait()
                  reader.join(timeout=1.0)
                  sys.exit("old x86 QEMU smoke did not produce bestmove")

              try:
                  process.wait(timeout=5.0)
              except subprocess.TimeoutExpired:
                  process.kill()
                  process.wait()
                  sys.exit("old x86 QEMU smoke did not exit after quit")
              reader.join(timeout=1.0)
              sys.exit(process.returncode)
              PY
            '';
          };

          aarch64-qemu-tests = pkgs.writeShellApplication {
            name = "aarch64-qemu-tests";
            runtimeInputs = with pkgs; [
              coreutils
              crossAarch64
              qemu
              rustToolchain
              stdenv.cc
            ];
            text = ''
              export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="${crossAarch64}/bin/aarch64-unknown-linux-gnu-gcc"
              export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="${pkgs.qemu}/bin/qemu-aarch64 -L ${crossAarch64Libc}"
              export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=generic"
              unset RUSTFLAGS

              ulimit -s 65536 || true
              export RUST_MIN_STACK=16777216

              exec cargo test --locked --target aarch64-unknown-linux-gnu --all-features -- --test-threads=1 "$@"
            '';
          };
        in
        {

          windows-release = {
            type = "app";
            program = "${windowsEmber.releaseApp}/bin/windows-release";
          };

          x86_64-qemu-oldcpu-smoke = {
            type = "app";
            program = "${x86_64-qemu-oldcpu-smoke}/bin/x86_64-qemu-oldcpu-smoke";
          };

          aarch64-qemu-tests = {
            type = "app";
            program = "${aarch64-qemu-tests}/bin/aarch64-qemu-tests";
          };

          search-shape-benchmark = {
            type = "app";
            program = "${search-shape-benchmark}/bin/search-shape-benchmark";
          };
        });

      packages = forAllSystems (pkgs:
        let
          windowsEmber = import ./nix/windows-ember.nix {
            inherit pkgs;
            lib = pkgs.lib;
          };
        in
        (import ./nix/ccrl-opponents.nix { inherit pkgs; })
        // (import ./nix/syzygy-tablebases.nix { inherit pkgs; })
        // {
          windows-ember = windowsEmber.package;
          windows-portable = import ./nix/windows-portable.nix {
            inherit pkgs;
            lib = pkgs.lib;
            emberWindows = windowsEmber.package;
          };
        });

      devShells = forAllSystems (pkgs:
        let
          rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

          blunder-7-2-0 = pkgs.buildGoModule {
            pname = "blunder";
            version = "7.2.0";

            src = pkgs.fetchFromGitHub {
              owner = "deanmchris";
              repo = "blunder";
              rev = "v7.2.0";
              sha256 = "18phz1nakggx6rf5yv71nj45lbr0jcbhdxf2pyzm4xvgdm428xy2";
            };

            subPackages = [ "blunder" ];
            vendorHash = null;

            postInstall = ''
              mv "$out/bin/blunder" "$out/bin/blunder-7.2.0"
            '';
          };
        in
        {
          elo-runner = pkgs.mkShell {
            packages = with pkgs; [
              bash
              coreutils
              rustToolchain
              go
              git
              gnugrep
              gnused
              gnutar
              gzip
              rsync
              zstd
              tmux
              (python3.withPackages (ps: [
                ps.chess
                ps.cairosvg
              ]))
              cutechess
              stockfish
              gnuchess
              fairymax
              ffmpeg
              blunder-7-2-0
            ];

            shellHook = ''
              export EMBER_ELO_NIX_SHELL=1
            '';
          };

          ci = pkgs.mkShell {
            packages = with pkgs; [
              bash
              coreutils
              rustToolchain
              (python3.withPackages (ps: [
                ps.pyyaml
                ps.requests
              ]))
            ];
          };

          default = self.devShells.${pkgs.system}.elo-runner;
        });
    };
}
