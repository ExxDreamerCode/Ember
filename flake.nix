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
      devShells = forAllSystems (pkgs: {
        elo-runner = pkgs.mkShell {
          packages = with pkgs; [
            bash
            coreutils
            cargo
            rustc
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
          ];

          shellHook = ''
            export EMBER_ELO_NIX_SHELL=1
          '';
        };

        default = self.devShells.${pkgs.system}.elo-runner;
      });
    };
}
