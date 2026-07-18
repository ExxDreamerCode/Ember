#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "usage: $0 /path/to/ember-lichess-windows.zip" >&2
  exit 2
fi

archive=$(realpath "$1")
sidecar="$archive.sha256"
if [[ ! -f "$archive" || ! -f "$sidecar" ]]; then
  echo "archive or checksum sidecar is missing" >&2
  exit 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

(
  cd "$(dirname "$archive")"
  sha256sum -c "$(basename "$sidecar")"
)
unzip -q "$archive" -d "$work"
cd "$work/Ember-Lichess"

export WINEDEBUG=-all
export WINEPREFIX="$work/wine-prefix"
wineboot --init >/dev/null 2>&1

wine cmd /d /c Verify.cmd
wine runtime/python.exe -c \
  "import backoff, certifi, chess, requests, rich, yaml; import battle_runner; print('Python imports: OK')"
wine runtime/python.exe lichess-bot/lichess-bot.py --help > "$work/lichess-bot-help.log"
grep -q 'Play on Lichess with a bot' "$work/lichess-bot-help.log"
echo "lichess-bot startup: OK"

printf 'uci\nisready\nsetoption name Threads value 1\nposition startpos\ngo depth 4\nquit\n' \
  | timeout 60 wine engine/ember.exe > "$work/uci.log"
grep -q '^uciok' "$work/uci.log"
grep -q '^readyok' "$work/uci.log"
grep -q '^bestmove ' "$work/uci.log"
echo "Ember UCI search: OK"

wine runtime/python.exe battle_runner.py --config battle.toml --dry-run
echo "Portable Windows smoke test: OK"
