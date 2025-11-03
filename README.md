
# pathmap

This crate provides a key-value store with prefix compression, structural sharing, and powerful algebraic operations.

PathMap is optimized for large data sets and can be used efficiently in a multi-threaded environment.

This crate provides the low-level data structure for [MORK](https://github.com/trueagi-io/MORK/)

## Usage

Check out the [book](https://pathmap-rs.github.io/).

## kvmap

The crate name **`pathmap`** was previously used by **Canmi** for a different project, 
[`kvmap`](https://github.com/canmi21/kvmap) *(formerly published as `pathmap`)*, 
which is a **SQL-driven key-value store**.

If you are looking for Canmi’s SQL-based KVMap project, please visit:  
[https://github.com/canmi21/kvmap](https://github.com/canmi21/kvmap)

## Getting Started

Add the following to your Cargo.toml:

```toml
pathmap = ">=0.2.0-alpha0, <0.3.0"
```

**NOTE** This is pre-release software and there is likely to be further API churn.  Specify an exact version to be insulated from the churn until the `0.2.0` release.

## Optional Cargo features

- `nightly`: Uses nightly-only features including support for a custom [`Allocator`](https://doc.rust-lang.org/std/alloc/trait.Allocator.html), better SIMD optimizations, etc.  Requires the *nightly* tool-chain.

- `arena_compact`: Exposes an additional read-only trie format and read-zipper type that is more efficient in memory and supports mapping a large file from disk.

- `jemalloc`: Enables [jemalloc](https://jemalloc.net/) as the default allocator.  This dramatically improves scaling for write-heavy workloads and is generally recommended.  The only reason it is not the default is to avoid interference with the host application's allocator.

- `zipper_tracking`: Exports the `zipper_tracking` module publicly, allowing the host application to use the conflict-checking logic independently of zipper creation.

- `viz`: Provide APIs to inspect and visualize pathmap trie structures.  Useful to observe structural sharing.

Other cargo features in this crate are intended for use by the developers of `pathmap` itself.
