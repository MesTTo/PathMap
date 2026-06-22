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

run cargo +nightly fmt --check
run cargo +nightly perf-lints
run cargo +nightly test --all-features --lib
run cargo +nightly check --manifest-path fuzz/Cargo.toml

if [[ "${RUN_FUZZ_SMOKE:-0}" == "1" ]]; then
  if cargo fuzz --version >/dev/null 2>&1; then
    run scripts/fuzz-smoke.sh
  else
    printf '\n- skip fuzz smoke: cargo-fuzz unavailable\n'
  fi
fi

if [[ "${RUN_MIRI_SMOKE:-0}" == "1" ]]; then
  if cargo +nightly miri --version >/dev/null 2>&1; then
    run scripts/miri-smoke.sh
  else
    printf '\n- skip miri smoke: cargo +nightly miri unavailable\n'
  fi
fi

if cargo machete --version >/dev/null 2>&1; then
  run cargo machete --with-metadata --skip-target-dir
else
  printf '\n- skip unused dependency scan: cargo machete unavailable\n'
fi

if cargo deny --version >/dev/null 2>&1; then
  run cargo deny check
else
  printf '\n- skip dependency policy: cargo deny unavailable\n'
fi

if cargo +nightly udeps --version >/dev/null 2>&1; then
  run cargo +nightly udeps --all-targets --all-features
else
  printf '\n- skip nightly unused dependency scan: cargo +nightly udeps unavailable\n'
fi
