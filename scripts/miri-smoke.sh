#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-strict-provenance -Zmiri-symbolic-alignment-check}"
export RUSTFLAGS="${RUSTFLAGS:-}"

run cargo +nightly miri test --lib slim_ptr -- --nocapture
run cargo +nightly miri test --lib root_value -- --nocapture
run cargo +nightly miri test --lib write_zipper_movement_preserves -- --nocapture
