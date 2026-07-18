{ pkgs }:

let
  inherit (pkgs) lib;

  baseUrl = "https://tablebase.lichess.ovh/tables/standard";
  entries3To5 = builtins.fromJSON (builtins.readFile ./syzygy-3-4-5.json);
  entries6 = builtins.fromJSON (builtins.readFile ./syzygy-6.json);

  fileSource =
    sourcePrefix: entry:
    pkgs.fetchurl {
      inherit (entry) name hash;
      url = "${baseUrl}/${sourcePrefix}-${
        if lib.hasSuffix ".rtbw" entry.name then "wdl" else "dtz"
      }/${entry.name}";
    };

  addSources = sourcePrefix: map (entry: entry // { src = fileSource sourcePrefix entry; });

  sourced3To5 = addSources "3-4-5" entries3To5;
  sourced6 = addSources "6" entries6;

  mkSyzygy =
    {
      name,
      directory,
      pieceLabel,
      maxPieces,
      entries,
    }:
    let
      fileCount = builtins.length entries;
      totalBytes = lib.foldl' (total: entry: total + entry.bytes) 0 entries;
    in
    pkgs.runCommand name
      {
        preferLocalBuild = true;
        passthru = {
          inherit fileCount maxPieces totalBytes;
        };
        meta = {
          description = "Syzygy ${pieceLabel} WDL and DTZ tablebases from the Lichess mirror";
          homepage = "https://tablebase.lichess.ovh/tables/standard/";
          platforms = lib.platforms.all;
        };
      }
      ''
              table_dir="$out/share/syzygy/${directory}"
              mkdir -p "$table_dir"
              ${lib.concatMapStringsSep "\n" (entry: ''
                ln -s ${entry.src} "$table_dir/${entry.name}"
              '') entries}

              cat > "$table_dir/README.txt" <<'EOF'
        Syzygy ${pieceLabel} WDL+DTZ tablebases.

        Source: https://tablebase.lichess.ovh/tables/standard/
        Files: ${toString fileCount}
        Bytes: ${toString totalBytes}

        Each file is fetched by Nix as a fixed-output derivation with its SHA-256
        hash pinned in the corresponding manifest under nix/.
        Use this directory as Ember's UCI SyzygyPath.
        EOF
      '';

  syzygy-3-4-5 = mkSyzygy {
    name = "syzygy-3-4-5";
    directory = "3-4-5";
    pieceLabel = "3-4-5 piece";
    maxPieces = 5;
    entries = sourced3To5;
  };

  syzygy-3-4-5-6 = mkSyzygy {
    name = "syzygy-3-4-5-6";
    directory = "3-4-5-6";
    pieceLabel = "3-4-5-6 piece";
    maxPieces = 6;
    entries = sourced3To5 ++ sourced6;
  };
in
{
  inherit syzygy-3-4-5 syzygy-3-4-5-6;
  syzygy-6 = syzygy-3-4-5-6;
  syzygy = syzygy-3-4-5;
}
