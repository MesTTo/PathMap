# Performance And Hygiene

PathMap already uses Divan for microbenchmarks. The commands below are the local
tooling layer for avoidable performance loss, dependency waste, profiling, and
binary-size checks.

## Cheap Checks

Run these before claiming a performance or dependency-cleanup improvement:

```bash
cargo +nightly perf-lints
scripts/rust-hygiene.sh
RUN_FUZZ_SMOKE=1 scripts/rust-hygiene.sh
```

`cargo +nightly perf-lints` is intentionally scoped to deny Clippy's performance
category. The repo has unrelated Clippy debt, so this alias allows the broader
Clippy backlog while still failing on `clippy::perf` regressions.

The GitHub Actions equivalent is `.github/workflows/rust-hygiene.yml`. It runs
formatting, Clippy perf lints, all-features library tests, `cargo machete`,
fuzz target compilation, `cargo deny check`, `cargo tree --duplicates`, and
benchmark compilation.

For a direct fuzz smoke run:

```bash
scripts/fuzz-smoke.sh
FUZZ_RUNS=10000 scripts/fuzz-smoke.sh
```

The script keeps `-C target-feature=+aes,+sse2` in `RUSTFLAGS` because
`cargo-fuzz` injects sanitizer flags and can otherwise bypass the repo's normal
x86 target flags, which `gxhash` requires.

For strict-provenance unsafe-boundary smoke tests:

```bash
scripts/miri-smoke.sh
RUN_MIRI_SMOKE=1 scripts/rust-hygiene.sh
```

This runs focused Miri coverage for slim pointer tagging, root-value paths, and
write-zipper stale-scout-frame movement. Keep it as a focused gate unless a
change touches broader unsafe/refcount/drop behavior.

## Benchmarks

Use the existing Divan benches:

```bash
cargo bench
cargo bench-scout
cargo +nightly bench-cata-context
cargo +nightly bench-node-layout
cargo +nightly bench --bench prefix_zipper
cargo +nightly bench --bench product_zipper_criterion -- --noplot
cargo +nightly bench --bench restrict_zipper_criterion -- --noplot
python3 benches/bench_avg.py --runs 5 --bench write_zipper_scout
python3 scripts/node-layout-table.py --sample-count 20
python3 scripts/node-layout-ab.py --sample-count 20
```

For CLI-level timing outside Cargo benchmarks, use `hyperfine`.

`scripts/node-layout-table.py` runs the `node_layout` bench and renders Divan
output as a Markdown table annotated with representation, fixture, child count,
path depth, and bytes/key. Use it before changing `LineListNode` or dense-node
lookup strategy.

`scripts/node-layout-ab.py` renders pairwise comparisons from the same benchmark
output: current direct point operations versus the older zipper-shaped routes,
nested line-list versus dense walks at equal depth, and root threshold-adjacent
line-list fanout 2 versus dense fanout 3/4. Use those A/B rows before changing
small-node representation thresholds or lookup strategy. It also includes a
node-local child-hit kernel comparing linear key scans, `ByteMask` test/rank,
binary search, and branchless-linear scans across fanouts; that kernel isolates
child-selection cost from path depth and mutation behavior. The same bench also
includes `ByteMask::indexed_bit` forward/backward select kernels, with
benchmark-only legacy rows, so indexed child selection changes can be measured
before touching production traversal code.

`PathMap::get_val_at`, `PathMap::contains`, and `PathMap::path_exists_at` use
direct `node_along_path` descent rather than building a read zipper. Keep
point-query optimizations on that direct path unless they need
traversal/forking semantics that only the zipper provides. For path-existence
checks, avoid recursive partial-key checks at each level; descend first and
perform one final `node_contains_partial_key` check on the residual node.

`PathMap::set_val_at` also uses direct mutable descent for ordinary point
sets/updates, then delegates insertion and node upgrades to `node_set_val`.
`PathMap::join_val_at` uses one mutable descent for the existing-value case;
`PathMap::get_val_or_set_mut_with_at` uses direct mutable residual descent for
existing values and node-level insertion for missing values. `PathMap::remove_val_at`
also uses direct mutable residual descent when `prune == false`, because value
removal without pruning preserves the dangling path and needs no ancestor
repair. `PathMap::remove_branches_at` uses the same direct mutable residual
descent when `prune == false`, because cutting branches below the focus does
not need to repair ancestors and preserves the focus path/value. The `old_join`,
`old_get_or_set`, `old_remove_no_prune`, `old_remove_branches_no_prune`, and
`old_create_path` rows in `scripts/node-layout-table.py` preserve zipper-route
A/B comparisons. `PathMap::create_path` uses direct residual descent for
ordinary missing-path creation, but falls back to the write zipper if the
node-level create reports that no path bytes were created; that fallback
preserves compressed-edge split semantics. `PathMap::prune_path` stays on the
write zipper route: a direct node-local trial was lawful, but loaded
node-layout A/B samples showed nested dangling-path regressions because path
pruning is dominated by ancestor repair and path re-creation costs.

Cached catamorphisms should stay structural by default. Use
`PathScope::None`/`PathScope::Suffix(0)` when the fold result depends only on
the residual subtrie; those scopes keep the old node-id cache behavior. Use a
non-zero suffix or `PathScope::Full` only when the result actually depends on
that path context, because those scopes deliberately widen the cache key and
reduce reuse for shared subtries reached through different prefixes.
The `cata_context` Divan bench measures this tradeoff on a shared residual DAG:
on the 2026-06-18 loaded-workstation sample, `Suffix(0)` was about 5.9x faster
than `Suffix(1)` and about 137x faster than `Full` for the 100-repeat case.

The `prefix_zipper` Divan bench compares the current `PrefixZipper` iteration
fast path with a benchmark-only wrapper that forces the old default
`ZipperIteration` behavior through the public movement surface. Use it before
changing prefix-action traversal or adding new blind iteration APIs. On the
2026-06-18 loaded-workstation sample in
`/tmp/pathmap-prefix-zipper-bench-20260618.txt`, skipping the artificial prefix
reduced a 256-byte prefixed root-value step from 464.8 ns median to 12.38 ns,
and reduced 256-value prefixed leaf iteration from 30.11 us median to 4.779 us.

The `streaming_iteration_criterion` Criterion bench compares three value
traversal shapes: the owned-key `PathMap::iter()` convenience, the borrowed-path
`PathMap::for_each_value` callback, and a manual read-zipper loop. Use it before
changing public iteration surfaces or adding a lending iterator abstraction. On
the 2026-06-18 loaded-workstation sample at load average 6.48/6.37/6.31, the
borrowed callback roughly matched manual zipper traversal and owned `iter()` on
short keys: about 2.67-2.73 us for 128 values, 21.6-21.8 us for 1024 values,
and 86.5-86.8 us for 4096 values. Treat this as allocation-shape evidence, not
a large timing speedup claim. The API is still useful because it exposes the hot
borrowed path/value shape without requiring callers to write zipper loops or
allocate and then discard owned key buffers.

The `graft_child_maps_criterion` Criterion bench compares the grouped
`ZipperWriting::graft_child_maps` API with the equivalent manual child loop over
`descend_to_byte`, `graft_map`, and `ascend_byte`. Use it before changing
multi-branch graft APIs or importing PR #38-style grouped write code. On the
2026-06-18 loaded-workstation sample at load average about 6.17/6.07/6.03,
`remove_unset=true` consistently favored the grouped path: contiguous masks were
about 86-88 ns vs 144-148 ns at 1 child, 216-221 ns vs 313-321 ns at 4
children, 846-866 ns vs 1.21-1.25 us at 16 children, and 3.23-3.25 us vs
4.58-4.71 us at 64 children. `remove_unset=false` stayed close to the manual
loop, about 6.2 us at 1-16 children and 7.7-8.0 us at 64 children, because that
mode still uses the slow selected-child update path while preserving unmasked
siblings. The benchmark also exposed and now protects a root-dense keep-unset
edge case in `WriteZipperCore::with_node_at_path`.

The `product_zipper_criterion` Criterion bench isolates constructor costs for
secondary factor normalization. It measures the borrowed node-root fast path,
focused secondary factors that materialize a hidden root path, and virtual
secondary factors such as `OverlayZipper` that materialize through
`try_make_map`. On the 2026-06-18 loaded-workstation sample at load average
6.26/6.11/5.27, borrowed secondaries were ~67-69 ns, focused materialized
secondaries were ~105-117 ns, and virtual overlay materialization scaled from
~2.1 us at one path to ~20 us at 128 paths. Treat these as local
shape-of-cost evidence, not a clean baseline. After `materialize_zipper` was
changed to reuse one relative path buffer, a loaded-workstation sample at load
average 5.83/6.47/6.46 measured virtual overlay materialization at ~2.00 us
for one path, ~3.73-3.81 us for 16 paths, and ~18.8-19.2 us for 128 paths.

The `restrict_zipper_criterion` Criterion bench isolates the lazy guard
automaton used by `RestrictZipper` and the heterotyped
`PathMap::restrict_by_paths` API. It measures inactive root child-mask
intersection, lazy value iteration, lazy materialization, same-typed eager
`PathMap::restrict`, and heterotyped `restrict_by_paths`. On the 2026-06-18
loaded-workstation sample at load average 5.57/5.70/5.60, lazy root child
count stayed ~18-19 ns across 16, 128, and 512 guard groups. Before direct
specialization, heterotyped `restrict_by_paths` followed the generic lazy
materialization path: ~30 us at 16 groups, ~248 us at 128 groups, and ~1.01 ms
at 512 groups. After the direct residual traversal slice, a loaded-workstation
sample at load average 8.63/7.77/6.56 measured heterotyped `restrict_by_paths`
at ~1.3 us, ~1.4 us, and ~1.5 us for the same group counts. Same-typed eager
`restrict` remains the relevant node-algebra baseline at ~1.6 us, ~14 us, and
~58 us on the same second run. Treat these as local shape-of-cost evidence, not
a clean baseline. The path-buffer materialization slice then measured lazy
materialization at 28.5-28.9 us for 16 groups, 238.5-242.5 us for 128 groups,
and 984.8-1001.8 us for 512 groups under load average 5.83/6.47/6.46.

With the `counters` feature enabled, `VirtualZipperCounters` records product
factor entries for `ProductZipper` and `ProductZipperG`, plus overlay value
mappings and child-mask unions for `OverlayZipper`. It also records
dependent-product enroll calls and successful factor entries for
`DependentProductZipperG`, separating boundary checks that run user enroll code
from checks that actually add a virtual factor. `RestrictZipper` records value
filters and inactive child-mask intersections, and is tested against eager
`PathMap::restrict`. The debug `DiffZipper` records observation-comparison
checks when it compares two zippers for residual equivalence. Use the focused
`product_zippers_count_lazy_factor_entries`,
`overlay_counter_records_virtual_observations`, and
`dependent_product_counts_enroll_attempts_and_factor_entries`,
`restrict_zipper_matches_eager_restrict`, and
`restrict_zipper_counts_value_and_child_mask_filters`, plus
`diff_zipper_counts_observation_checks`, tests before changing product,
overlay, dependent-product, restrict, or debug diff observation behavior.

The default release profile is configured for throughput-oriented optimized
builds:

```toml
[profile.release]
lto = true
codegen-units = 1
opt-level = 3
```

There is also a size-oriented profile for cache-footprint experiments:

```bash
cargo build --profile release-z
cargo bench --profile release-z
```

Treat `release-z` as a measurement option, not a default speed claim.

## Profiling And Size

These are opt-in because they can be slow or need platform permissions:

```bash
cargo flamegraph-scout
cargo samply-scout
scripts/rust-perf.sh samply write_zipper_scout --sample-count 10
scripts/rust-perf.sh heaptrack write_zipper_scout --sample-count 10
cargo bloat-top
cargo llvm-lines-top
cargo asm pathmap::module::function --rust
scripts/type-sizes.sh
perf record --call-graph dwarf cargo bench-scout
```

Use `heaptrack`, `valgrind --tool=dhat`, or `bytehound` for allocation-heavy
workloads when allocator behavior is the actual question.

Use `scripts/type-sizes.sh` when a change touches hot structs, enum variants,
or zipper/node storage. It runs nightly `-Zprint-type-sizes`, stores the full
report under `target/type-sizes/full.txt`, and prints a filtered report for
`LineListNode`, `ByteNode`, `ReadZipperCore`, `WriteZipperCore`, `KeyFields`,
`MutNodeStack`, and `TinyRefNode`. If `top-type-sizes` is installed, it also
writes a compact summary to `target/type-sizes/top.txt`. The old `BridgeNode`
backend is currently compiled out with `#[cfg(any())]`, so re-enabling it should
include a separate layout audit.

Hot layout regressions should be guarded with static size assertions where the
size is part of the design. The current x86_64 guards cover the compact node
types and zipper core fields that regressed during root-value storage work.

## Layout And Branches

The current root-value design intentionally keeps `Option<V>` out of common trie
node structs. A previous physical-node storage experiment inflated hot node
sizes and tripped layout-sensitive tests, so root values stay in the parent-edge
slot until a measured design preserves cache footprint.

For branch-prediction-sensitive code, prefer simple predictable hot paths first:
measure before adding `#[inline]`, `#[cold]`, lookup-table rewrites, or manual
branch rearrangements. Branch misses should be validated with `perf stat`,
`perf record`, Cachegrind/Callgrind, or an equivalent profiler before changing
readability or layout.

For zipper performance, keep Oleg Kiselyov's zipper-as-continuation framing in
mind: movement is traversal context, mutation is a materialization boundary.
That means optimizing write zippers should first preserve read-only scout
movement and only then reduce measured materialization work.

For node layout work, prefer compact contiguous scans when profiling shows the
branch fanout is cache-sensitive. ART/START, Rust's `BTreeMap` background, and
CHAMP-style bitmap tries are the current comparison set; do not switch layouts
without checking cache behavior, path-copying cost, and type-size regressions.

## Installed Local Tools

The current workstation has the main cargo subcommands and standalone profiler
commands installed:

```bash
cargo udeps --version
cargo bloat --version
cargo llvm-lines --version
cargo asm --version
cargo flamegraph --version
cargo machete --version
cargo deny --version
hyperfine --version
perf --version
heaptrack --version
valgrind --version
samply --version
cargo samply --version
```

Verified versions on this workstation:

- `cargo-udeps 0.1.61`
- `cargo-bloat 0.12.1`
- `cargo-llvm-lines 0.4.46`
- `cargo-asm 0.1.16`
- `flamegraph 0.6.13`
- `cargo-machete 0.9.2`
- `cargo-deny 0.19.7`
- `hyperfine 1.19.0`
- `perf 7.0.0`
- `heaptrack 1.5.0`
- `valgrind 3.26.0`
- `samply 0.13.1`
- `cargo-samply 0.4.2`

`cargo +nightly udeps` should stay a separate, explicit dependency check because
it uses nightly compiler internals and can be noisier than `cargo machete`.

`deny.toml` currently records a reasoned ignore for the unmaintained `paste`
advisory so `cargo deny check` can run in CI while that cleanup remains explicit.

`bytehound` is not installed on this workstation; it is not available here as an
apt package or a crates.io binary named `bytehound`. Use `heaptrack`, DHAT via
Valgrind, or `samply` first for allocation and sampling work.
