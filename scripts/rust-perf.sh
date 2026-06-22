#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

usage() {
  cat <<'USAGE'
usage: scripts/rust-perf.sh <command> [args...]

commands:
  bench [bench-name] [divan-args...]      Run a Divan benchmark.
  flamegraph [bench-name] [divan-args...] Profile a Divan benchmark to target/flamegraphs.
  samply [bench-name] [divan-args...]     Profile a Divan benchmark with samply.
  heaptrack [bench-name] [divan-args...]  Profile a Divan benchmark invocation with heaptrack.
  hyperfine <command...>                  Run a whole-command timing benchmark.
  bloat [cargo-bloat-args...]             Show largest release symbols.
  llvm-lines [cargo-llvm-lines-args...]   Show monomorphization/codegen size.
  asm <rust-path> [cargo-asm-args...]     Show assembly for a function.

examples:
  scripts/rust-perf.sh bench write_zipper_scout --sample-count 10
  scripts/rust-perf.sh flamegraph write_zipper_scout --sample-count 10
  scripts/rust-perf.sh samply write_zipper_scout --sample-count 10
  scripts/rust-perf.sh heaptrack write_zipper_scout --sample-count 10
  scripts/rust-perf.sh hyperfine 'cargo bench-scout'
  scripts/rust-perf.sh bloat -n 20
  scripts/rust-perf.sh llvm-lines --release
USAGE
}

command="${1:-}"
if [[ -z "$command" || "$command" == "-h" || "$command" == "--help" ]]; then
  usage
  exit 0
fi
shift

case "$command" in
  bench)
    bench_name="${1:-write_zipper_scout}"
    [[ $# -gt 0 ]] && shift
    cargo bench --bench "$bench_name" --features counters -- "$@"
    ;;
  flamegraph)
    bench_name="${1:-write_zipper_scout}"
    [[ $# -gt 0 ]] && shift
    mkdir -p target/flamegraphs
    cargo flamegraph --bench "$bench_name" --features counters \
      --output "target/flamegraphs/${bench_name}.svg" -- "$@"
    ;;
  samply)
    bench_name="${1:-write_zipper_scout}"
    [[ $# -gt 0 ]] && shift
    cargo samply --no-profile-inject --bench "$bench_name" --features counters -- "$@"
    ;;
  heaptrack)
    bench_name="${1:-write_zipper_scout}"
    [[ $# -gt 0 ]] && shift
    mkdir -p target/heaptrack
    heaptrack --output "target/heaptrack/${bench_name}.gz" \
      cargo bench --bench "$bench_name" --features counters -- "$@"
    ;;
  hyperfine)
    if [[ $# -eq 0 ]]; then
      usage >&2
      exit 2
    fi
    hyperfine "$@"
    ;;
  bloat)
    cargo bloat --release "$@"
    ;;
  llvm-lines)
    cargo llvm-lines "$@"
    ;;
  asm)
    rust_path="${1:-}"
    if [[ -z "$rust_path" ]]; then
      usage >&2
      exit 2
    fi
    shift
    cargo asm "$rust_path" --rust "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
