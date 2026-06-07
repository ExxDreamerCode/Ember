#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  tools/remote-measure-elo.sh --host HOST [options] run
  tools/remote-measure-elo.sh --host HOST [options] detach
  tools/remote-measure-elo.sh --host HOST [options] status RUN_ID
  tools/remote-measure-elo.sh --host HOST [options] tail RUN_ID
  tools/remote-measure-elo.sh --host HOST [options] fetch RUN_ID
  tools/remote-measure-elo.sh --host HOST [options] stop RUN_ID
  tools/remote-measure-elo.sh --host HOST [options] cleanup --older-than 14d

Options:
  --host HOST           SSH host alias. Can also use EMBER_ELO_HOST.
  --remote-root PATH    Remote work root. Default: ember-elo-runs
  --config PATH         Config path. Default: configs/elo/default.toml
  --run-id ID           Run id. Default: generated locally
  --workers N           Override worker count
  --max-games N         Override maximum games scheduled by the runner
  --older-than AGE      Cleanup age for cleanup command, such as 14d
EOF
}

host="${EMBER_ELO_HOST:-}"
remote_root="${EMBER_ELO_REMOTE_ROOT:-ember-elo-runs}"
config="configs/elo/default.toml"
run_id=""
workers=""
max_games=""
older_than=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host="$2"
      shift 2
      ;;
    --remote-root)
      remote_root="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --run-id)
      run_id="$2"
      shift 2
      ;;
    --workers)
      workers="$2"
      shift 2
      ;;
    --max-games)
      max_games="$2"
      shift 2
      ;;
    --older-than)
      older_than="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

command="${1:-}"
shift || true

if [[ -z "$host" ]]; then
  echo "missing --host or EMBER_ELO_HOST" >&2
  exit 2
fi

if [[ -z "$command" ]]; then
  usage >&2
  exit 2
fi

if [[ -z "$run_id" ]]; then
  stamp="$(date +%Y%m%d-%H%M%S)"
  git_suffix="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
  if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
    git_suffix="${git_suffix}-dirty"
  fi
  run_id="${stamp}-${git_suffix}"
fi

ssh_cmd=(ssh "$host")
rsync_cmd=(rsync -az --delete
  --exclude .git
  --exclude .git/
  --exclude target/
  --exclude results/
  --exclude .elo-remote/
)

remote_source="${remote_root}/${run_id}/source"
remote_results="${remote_source}/results/${run_id}"
nix_flags=(--extra-experimental-features nix-command --extra-experimental-features flakes)
git_commit="$(git rev-parse HEAD 2>/dev/null || true)"
git_dirty="false"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  git_dirty="true"
fi

remote_measure_command() {
  local extra_workers=()
  local extra_max_games=()
  if [[ -n "$workers" ]]; then
    extra_workers=(--workers "$workers")
  fi
  if [[ -n "$max_games" ]]; then
    extra_max_games=(--max-games "$max_games")
  fi
  printf 'cd %q && EMBER_ELO_GIT_COMMIT=%q EMBER_ELO_GIT_DIRTY=%q nix %q %q %q %q develop .#elo-runner --command python3 tools/measure_elo.py all --config %q --run-id %q' \
    "$remote_source" \
    "$git_commit" "$git_dirty" \
    "${nix_flags[0]}" "${nix_flags[1]}" "${nix_flags[2]}" "${nix_flags[3]}" \
    "$config" "$run_id"
  if [[ ${#extra_workers[@]} -gt 0 ]]; then
    printf ' %q %q' "${extra_workers[0]}" "${extra_workers[1]}"
  fi
  if [[ ${#extra_max_games[@]} -gt 0 ]]; then
    printf ' %q %q' "${extra_max_games[0]}" "${extra_max_games[1]}"
  fi
}

fetch_results() {
  mkdir -p "results/${run_id}"
  rsync -az "$host:${remote_results}/report.md" "results/${run_id}/" 2>/dev/null || true
  rsync -az "$host:${remote_results}/metadata.json" "results/${run_id}/" 2>/dev/null || true
  rsync -az "$host:${remote_results}/estimates/estimate.json" "results/${run_id}/" 2>/dev/null || true
  rsync -az "$host:${remote_results}/artifacts.tar.zst" "results/${run_id}/" 2>/dev/null || true
}

case "$command" in
  run)
    "${ssh_cmd[@]}" "mkdir -p '$remote_source'"
    "${rsync_cmd[@]}" ./ "$host:$remote_source/"
    "${ssh_cmd[@]}" "$(remote_measure_command)"
    fetch_results
    echo "Fetched results/${run_id}/"
    ;;
  detach)
    "${ssh_cmd[@]}" "mkdir -p '$remote_source'"
    "${rsync_cmd[@]}" ./ "$host:$remote_source/"
    remote_cmd="$(remote_measure_command)"
    "${ssh_cmd[@]}" "cd '$remote_source' && tmux new-session -d -s 'ember-elo-${run_id}' '$remote_cmd'"
    echo "$run_id"
    ;;
  status)
    run_id="${1:-$run_id}"
    "${ssh_cmd[@]}" "tmux has-session -t 'ember-elo-${run_id}' 2>/dev/null && echo running || echo not-running"
    ;;
  tail)
    run_id="${1:-$run_id}"
    "${ssh_cmd[@]}" "tail -n 120 '$remote_results/commands.log' 2>/dev/null || tail -n 120 '$remote_results/build.log' 2>/dev/null || true"
    ;;
  fetch)
    run_id="${1:-$run_id}"
    remote_source="${remote_root}/${run_id}/source"
    remote_results="${remote_source}/results/${run_id}"
    fetch_results
    echo "Fetched results/${run_id}/"
    ;;
  stop)
    run_id="${1:-$run_id}"
    "${ssh_cmd[@]}" "tmux kill-session -t 'ember-elo-${run_id}' 2>/dev/null || true"
    ;;
  cleanup)
    if [[ -z "$older_than" ]]; then
      echo "cleanup requires --older-than" >&2
      exit 2
    fi
    if [[ ! "$older_than" =~ ^[0-9]+d$ ]]; then
      echo "--older-than currently supports Nd, for example 14d" >&2
      exit 2
    fi
    days="${older_than%d}"
    "${ssh_cmd[@]}" "mkdir -p '$remote_root' && find '$remote_root' -mindepth 1 -maxdepth 1 -type d -mtime '+$days' -print -exec rm -rf {} +"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
