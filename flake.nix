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

          aarch64-qemu-tests = pkgs.writeShellApplication {
            name = "aarch64-qemu-tests";
            runtimeInputs = with pkgs; [
              coreutils
              crossAarch64
              qemu
              rustToolchain
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
        import ./nix/ccrl-opponents.nix { inherit pkgs; });

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
            RUSTFLAGS = "-C target-cpu=native";

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
            RUSTFLAGS = "-C target-cpu=native";

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
