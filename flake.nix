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

              output=$(printf 'uci
isready
setoption name Book value
position startpos
go depth 1
quit
'                 | EMBER_SEARCH_BACKEND=auto qemu-x86_64 -cpu Nehalem target/release/ember)
              printf '%s
' "$output"
              grep -q '^bestmove ' <<<"$output"
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
              python3
            ];
          };

          default = self.devShells.${pkgs.system}.elo-runner;
        });
    };
}
