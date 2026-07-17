{
  pkgs,
  lib,
  emberWindows,
}:

let
  pythonVersion = "3.12.10";
  richVersion = "15.0.0";
  lichessBotRevision = "b95d251725ee30051fe9eb0ca83a20127c8dbaa6";

  pythonEmbed = pkgs.fetchurl {
    url = "https://www.python.org/ftp/python/${pythonVersion}/python-${pythonVersion}-embed-amd64.zip";
    hash = "sha256-SsvtbdHHRLA3bjsc9XzpBvncnpXmiCRYTICZpjAlo8M=";
  };

  # The pinned lichess-bot revision requires rich~=15.0, which is newer than
  # the Rich package in this flake's nixpkgs revision. The universal wheel is
  # pure Python and is still fetched and hash-checked by Nix.
  richWheel = pkgs.fetchurl {
    url = "https://files.pythonhosted.org/packages/82/3b/64d4899d73f91ba49a8c18a8ff3f0ea8f1c1d75481760df8c68ef5235bf5/rich-${richVersion}-py3-none-any.whl";
    hash = "sha256-M71O90Iy+3P+knmiV3GEB/FpwJt4qHrT0pb1SOJ94Ls=";
  };

  lichessBot = pkgs.fetchFromGitHub {
    owner = "lichess-bot-devs";
    repo = "lichess-bot";
    rev = lichessBotRevision;
    hash = "sha256-xzi8lVy8fwi9pF5Ucj9/HOXfuL0Q/Xk+TRHUjzMxjzM=";
  };

  pythonDependencies = pkgs.python312.withPackages (ps: with ps; [
    backoff
    chess
    pyyaml
    requests
  ]);

in
pkgs.stdenvNoCC.mkDerivation {
  pname = "ember-lichess-windows-portable";
  version = "1.1.2";

  dontUnpack = true;
  nativeBuildInputs = [
    pkgs.coreutils
    pkgs.findutils
    pkgs.gnused
    pkgs.python3
    pkgs.unzip
    pkgs.zip
  ];

  installPhase = ''
    runHook preInstall

    bundle="$TMPDIR/Ember-Lichess"
    runtime="$bundle/runtime"
    site="$runtime/Lib/site-packages"
    mkdir -p "$bundle/engine" "$bundle/lichess-bot" "$runtime" "$site" "$out"

    cp "${emberWindows}/bin/ember.exe" "$bundle/engine/ember.exe"
    unzip -q "${pythonEmbed}" -d "$runtime"

    dependency_site="${pythonDependencies}/${pkgs.python312.sitePackages}"
    for name in \
      backoff certifi charset_normalizer chess idna markdown_it mdurl pygments \
      requests urllib3 yaml
    do
      if test -e "$dependency_site/$name"; then
        cp -LR "$dependency_site/$name" "$site/"
      fi
    done
    for metadata in \
      backoff certifi charset_normalizer chess idna markdown_it_py mdurl \
      pygments pyyaml requests urllib3
    do
      for path in "$dependency_site/''${metadata}"-*.dist-info; do
        if test -e "$path"; then
          cp -LR "$path" "$site/"
        fi
      done
    done
    unzip -q "${richWheel}" -d "$site"
    chmod -R u+w "$site"
    find "$site" -type f \( -name '*.so' -o -name '*.pyc' \) -delete
    find "$site" -type d -name __pycache__ -prune -exec rm -rf {} +

    cat > "$runtime/python312._pth" <<'EOF'
python312.zip
.
..
..\lichess-bot
Lib\site-packages
import site
EOF

    cp -R "${lichessBot}/lib" "$bundle/lichess-bot/lib"
    cp "${lichessBot}/lichess-bot.py" "$bundle/lichess-bot/lichess-bot.py"
    cp "${lichessBot}/config.yml.default" "$bundle/lichess-bot/config.yml.default"
    cp "${lichessBot}/requirements.txt" "$bundle/lichess-bot/requirements.txt"
    cp "${lichessBot}/LICENSE" "$bundle/lichess-bot/LICENSE"
    for source in "${lichessBot}"/*.py; do
      case "$(basename "$source")" in
        lichess-bot.py) ;;
        *) cp "$source" "$bundle/lichess-bot/" ;;
      esac
    done

    cp "${../windows/battle_runner.py}" "$bundle/battle_runner.py"
    cp "${../windows/verify_bundle.py}" "$bundle/verify_bundle.py"
    cp "${../windows/battle.toml}" "$bundle/battle.toml"
    cp "${../windows}/Run Battle.cmd" "$bundle/Run Battle.cmd"
    cp "${../windows/Verify.cmd}" "$bundle/Verify.cmd"

    cat > "$bundle/VERSIONS.txt" <<EOF
Ember: 1.1.2 (x86-64-v3, static MSVC CRT)
Python: ${pythonVersion} embeddable x86-64
Rich: ${richVersion}
lichess-bot revision: ${lichessBotRevision}
Nixpkgs revision: ${pkgs.lib.version or "pinned by flake.lock"}
EOF

    cat > "$bundle/README.txt" <<'EOF'
Ember Lichess Windows portable bundle

1. Extract the complete ZIP to a writable directory.
2. Edit battle.toml with Notepad. It contains no Lichess token.
3. Double-click Verify.cmd (Run Battle.cmd also verifies automatically).
4. Double-click Run Battle.cmd, review the printed plan, type YES, and enter
   the Lichess token at the masked prompt.

The bot token needs the bot:play and challenge:write permissions. If one
opponent rejects a challenge, the exact Lichess reason is recorded and the
runner continues with the next configured game.

The runner uses all logical CPUs automatically (maximum 256) and can make the
machine busy. It changes no Windows service, autostart, scheduled task, power,
sleep, registry, PATH, or firewall setting. Keep the console and machine awake.

The default battle is one casual, non-scoring 3+2 standard game against the
first ready bot in a configured opponent pool. Busy, offline, rate-limited, or
declining bots are bypassed. If the whole pool is unavailable, the runner waits
and polls until one is ready. Games are direct challenges and strictly
sequential. Temporary monitoring disconnects are retried without stopping an
active game. The scoring flag and tags are analysis metadata; mode selects
casual or rated. Results are written below results/. Syzygy and lichess-bot
matchmaking are off.
EOF

    chmod -R u+w "$bundle"

    # battle.toml is intentionally user-editable. The manifest covers every
    # other shipped file and never covers future results.
    (
      cd "$bundle"
      find . -type f \
        ! -path './battle.toml' \
        ! -path './SHA256SUMS.txt' \
        ! -path './results/*' \
        -printf '%P\0' \
        | sort -z \
        | xargs -0 sha256sum > SHA256SUMS.txt
    )

    find "$bundle" -exec touch -h -d '@1' {} +
    (
      cd "$TMPDIR"
      zip -X -9 -q -r "$out/ember-lichess-windows.zip" Ember-Lichess
    )
    (
      cd "$out"
      sha256sum ember-lichess-windows.zip > ember-lichess-windows.zip.sha256
    )

    runHook postInstall
  '';

  passthru = {
    inherit emberWindows lichessBot pythonEmbed;
    xwinSdk = emberWindows.xwinSdk;
  };

  meta = {
    description = "Portable native-Windows Ember and lichess-bot challenge runner";
    platforms = lib.platforms.linux;
  };
}
