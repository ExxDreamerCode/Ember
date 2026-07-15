#!/usr/bin/env bash
set -euo pipefail

exec nix --extra-experimental-features nix-command \
  --extra-experimental-features flakes \
  run .#x86_64-qemu-oldcpu-smoke -- "$@"
