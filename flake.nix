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
          search-shape-benchmark = pkgs.writeShellApplication {
            name = "search-shape-benchmark";
            runtimeInputs = with pkgs; [
              python3
            ];
            text = ''
              exec python3 tools/benchmark_search_shape.py "$@"
            '';
          };
        in
        {
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
