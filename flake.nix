{
  description = "Ember Elo measurement runner";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system:
          f (import nixpkgs {
            inherit system;
          }));
    in
    {
      devShells = forAllSystems (pkgs:
        let
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
              cargo
              rustc
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

          default = self.devShells.${pkgs.system}.elo-runner;
        });
    };
}
