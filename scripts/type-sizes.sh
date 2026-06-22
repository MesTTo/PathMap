#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${TYPE_SIZES_DIR:-target/type-sizes}"
mkdir -p "$out_dir"

full_out="$out_dir/full.txt"
hot_out="$out_dir/hot.txt"
build_dir="$out_dir/build"
filter="${TYPE_SIZES_FILTER:-^print-type-size type: \`((zipper::read_zipper_core::ReadZipperCore)|(write_zipper::WriteZipperCore)|(write_zipper::KeyFields)|(write_zipper::mut_node_stack::MutNodeStack)|(line_list_node::LineListNode)|(dense_byte_node::ByteNode)|(tiny_node::TinyRefNode))}"

# Keep target-cpu native explicit for gxhash while adding nightly layout output.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-Zprint-type-sizes -C target-cpu=native"
export CARGO_TARGET_DIR="$build_dir"

cargo clean --target-dir "$build_dir" >/dev/null 2>&1 || true
cargo +nightly test --lib --no-run >"$full_out" 2>&1
rg "$filter" "$full_out" >"$hot_out"

if command -v top-type-sizes >/dev/null 2>&1; then
  top-type-sizes "$full_out" >"$out_dir/top.txt" || true
fi

cat "$hot_out"
printf '\nFull type-size output: %s\n' "$full_out"
printf 'Hot type-size output:  %s\n' "$hot_out"
if [[ -f "$out_dir/top.txt" ]]; then
  printf 'Top type-size output:  %s\n' "$out_dir/top.txt"
fi
