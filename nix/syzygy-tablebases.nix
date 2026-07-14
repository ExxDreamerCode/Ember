{ pkgs }:

let
  inherit (pkgs) lib;

  baseUrl = "https://tablebase.lichess.ovh/tables/standard";
  entries = builtins.fromJSON (builtins.readFile ./syzygy-3-4-5.json);

  fileSource = entry:
    pkgs.fetchurl {
      name = entry.name;
      url = "${baseUrl}/${if lib.hasSuffix ".rtbw" entry.name then "3-4-5-wdl" else "3-4-5-dtz"}/${entry.name}";
      hash = entry.hash;
    };

  linkedFiles = map (entry: entry // { src = fileSource entry; }) entries;
  totalBytes = lib.foldl' (total: entry: total + entry.bytes) 0 entries;

  syzygy-3-4-5 = pkgs.runCommand "syzygy-3-4-5" {
    preferLocalBuild = true;
    passthru = {
      fileCount = builtins.length entries;
      inherit totalBytes;
    };
    meta = {
      description = "Syzygy 3-4-5 piece WDL and DTZ tablebases from the Lichess mirror";
      homepage = "https://tablebase.lichess.ovh/tables/standard/";
      platforms = lib.platforms.all;
    };
  } ''
    table_dir="$out/share/syzygy/3-4-5"
    mkdir -p "$table_dir"
    ${lib.concatMapStringsSep "\n" (entry: ''
      ln -s ${entry.src} "$table_dir/${entry.name}"
    '') linkedFiles}

    cat > "$table_dir/README.txt" <<'EOF'
Syzygy 3-4-5 piece WDL+DTZ tablebases.

Source: https://tablebase.lichess.ovh/tables/standard/
Files: 290
Bytes: 983957920

Each file is fetched by Nix as a fixed-output derivation with the SHA-256
hash pinned in nix/syzygy-3-4-5.json.
Use this directory as Ember's UCI SyzygyPath.
EOF
  '';
in
{
  inherit syzygy-3-4-5;
  syzygy = syzygy-3-4-5;
}
