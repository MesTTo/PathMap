#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

target="${FUZZ_TARGET:-pathmap_ops}"
runs="${FUZZ_RUNS:-1000}"

# cargo-fuzz supplies sanitizer RUSTFLAGS itself, which can bypass the repo's
# target rustflags. Keep gxhash's x86 AES/SSE2 requirements explicit here.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+aes,+sse2"

cargo +nightly fuzz run "$target" -- -runs="$runs" "$@"
