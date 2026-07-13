#!/usr/bin/env bash
set -euo pipefail

exec nix --extra-experimental-features nix-command \
  --extra-experimental-features flakes \
  run .#aarch64-qemu-tests -- "$@"
