use divan::{black_box, Bencher, Divan};
use pathmap::ring::{AlgebraicResult, AlgebraicStatus, Lattice, COUNTER_IDENT, SELF_IDENT};
use pathmap::utils::{BitMask, ByteMask};
use pathmap::zipper::{ZipperMoving, ZipperWriting};
use pathmap::PathMap;

const LOOKUPS: usize = 512;

fn main() {
    Divan::from_args().main();
}

fn root_fanout_map(fanout: usize) -> (PathMap<u64>, Vec<[u8; 1]>) {
    let mut map = PathMap::new();
    let keys: Vec<[u8; 1]> = (0..fanout).map(|idx| [idx as u8]).collect();
    for (idx, key) in keys.iter().enumerate() {
        map.set_val_at(key, idx as u64);
    }
    (map, keys)
}

fn compressed_key_map(byte_len: usize) -> (PathMap<u64>, Vec<u8>) {
    let mut map = PathMap::new();
    let key: Vec<u8> = (0..byte_len)
        .map(|idx| b'a'.wrapping_add((idx % 23) as u8))
        .collect();
    map.set_val_at(&key, 1);
    (map, key)
}

fn compressed_branch_map(byte_len: usize) -> (PathMap<u64>, Vec<u8>, Vec<u8>) {
    let (mut map, key) = compressed_key_map(byte_len);
    let mut child_key = key.clone();
    child_key.push(0xff);
    map.set_val_at(&child_key, 2);
    (map, key, child_key)
}

fn compressed_missing_path_map(byte_len: usize) -> (PathMap<u64>, Vec<u8>) {
    let (map, _key) = compressed_key_map(byte_len);
    let key: Vec<u8> = (0..byte_len)
        .map(|idx| 0xf0_u8.wrapping_sub(idx as u8))
        .collect();
    (map, key)
}

fn nested_fanout_map(depth: usize, fanout: usize) -> (PathMap<u64>, Vec<u8>) {
    let mut map = PathMap::new();
    let mut prefix = Vec::with_capacity(depth);
    let continuation = (fanout - 1) as u8;
    let mut val = 0_u64;

    for level in 0..depth {
        for sibling in 0..continuation {
            let mut key = prefix.clone();
            key.push(sibling);
            map.set_val_at(&key, val);
            val += 1;
        }
        prefix.push(continuation);
        if level + 1 == depth {
            map.set_val_at(&prefix, val);
        }
    }

    (map, prefix)
}

fn nested_branch_map(depth: usize, fanout: usize) -> (PathMap<u64>, Vec<u8>, Vec<u8>) {
    let (mut map, key) = nested_fanout_map(depth, fanout);
    let mut child_key = key.clone();
    child_key.push(fanout as u8);
    map.set_val_at(&child_key, 1_000);
    (map, key, child_key)
}

fn nested_missing_path_map(depth: usize, fanout: usize) -> (PathMap<u64>, Vec<u8>) {
    let (map, mut key) = nested_fanout_map(depth, fanout);
    key.push(fanout as u8);
    (map, key)
}

fn local_lookup_keys(fanout: usize) -> (Vec<u8>, ByteMask) {
    let keys: Vec<u8> = (0..fanout)
        .map(|idx| ((idx * 37 + 11) % 251) as u8)
        .collect();
    let mut mask = ByteMask::EMPTY;
    for &key in &keys {
        mask.set_bit(key);
    }
    (keys, mask)
}

fn repeated_linear_child_index(keys: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = black_box(keys[idx % keys.len()]);
        let found = keys
            .iter()
            .position(|&candidate| candidate == key)
            .expect("benchmark key should exist in the linear scan set");
        sum += found as u64;
    }
    sum
}

fn repeated_mask_rank_index(keys: &[u8], mask: ByteMask) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = black_box(keys[idx % keys.len()]);
        if mask.test_bit(key) {
            sum += mask.index_of(key) as u64;
        }
    }
    sum
}

fn repeated_binary_child_index(keys: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = black_box(keys[idx % keys.len()]);
        let found = keys
            .binary_search(&key)
            .expect("benchmark key should exist in the binary search set");
        sum += found as u64;
    }
    sum
}

fn repeated_branchless_linear_child_index(keys: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = black_box(keys[idx % keys.len()]);
        let mut found = 0_usize;
        for (candidate_idx, &candidate) in keys.iter().enumerate() {
            found += candidate_idx * usize::from(candidate == key);
        }
        sum += found as u64;
    }
    sum
}

fn repeated_indexed_bit_forward(mask: ByteMask, fanout: usize) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let bit = mask
            .indexed_bit::<true>(black_box(idx % fanout))
            .expect("benchmark mask should contain enough forward bits");
        sum += bit as u64;
    }
    sum
}

fn repeated_indexed_bit_backward(mask: ByteMask, fanout: usize) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let bit = mask
            .indexed_bit::<false>(black_box(idx % fanout))
            .expect("benchmark mask should contain enough backward bits");
        sum += bit as u64;
    }
    sum
}

fn legacy_indexed_bit<const FORWARD: bool>(mask: ByteMask, idx: usize) -> u8 {
    let words = mask.into_inner();
    let mut i = if FORWARD { 0 } else { 3 };
    let mut m = words[i];
    let mut c = 0;
    let mut c_ahead = m.count_ones() as usize;
    loop {
        if idx < c_ahead {
            break;
        }
        if FORWARD {
            i += 1
        } else {
            i -= 1
        };
        assert!(i <= 3, "benchmark index should be in range");
        m = words[i];
        c = c_ahead;
        c_ahead += m.count_ones() as usize;
    }

    let mut loc;
    if !FORWARD {
        loc = 63 - m.leading_zeros();
        while c < idx {
            m ^= 1u64 << loc;
            loc = 63 - m.leading_zeros();
            c += 1;
        }
    } else {
        loc = m.trailing_zeros();
        while c < idx {
            m ^= 1u64 << loc;
            loc = m.trailing_zeros();
            c += 1;
        }
    }

    (i << 6 | (loc as usize)) as u8
}

fn repeated_legacy_indexed_bit_forward(mask: ByteMask, fanout: usize) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let bit = legacy_indexed_bit::<true>(mask, black_box(idx % fanout));
        sum += bit as u64;
    }
    sum
}

fn repeated_legacy_indexed_bit_backward(mask: ByteMask, fanout: usize) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let bit = legacy_indexed_bit::<false>(mask, black_box(idx % fanout));
        sum += bit as u64;
    }
    sum
}

fn repeated_root_lookup(map: &PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        sum += *map.get_val_at(key).unwrap();
    }
    sum
}

fn repeated_root_path_exists(map: &PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        sum += map.path_exists_at(key) as u64;
    }
    sum
}

fn repeated_lookup(map: &PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for _ in 0..LOOKUPS {
        sum += *map.get_val_at(key).unwrap();
    }
    sum
}

fn repeated_path_exists(map: &PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for _ in 0..LOOKUPS {
        sum += map.path_exists_at(key) as u64;
    }
    sum
}

fn repeated_root_update(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        sum += map.set_val_at(key, idx as u64).unwrap_or(0);
    }
    sum
}

fn repeated_update(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        sum += map.set_val_at(key, idx as u64).unwrap_or(0);
    }
    sum
}

fn repeated_root_join(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        sum += old_join_val_at(map, key, idx as u64) as u64;
    }
    sum
}

fn repeated_join(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        sum += old_join_val_at(map, key, idx as u64) as u64;
    }
    sum
}

fn old_join_val_at(map: &mut PathMap<u64>, path: &[u8], v: u64) -> AlgebraicStatus {
    if map.get_val_at(path).is_none() {
        map.set_val_at(path, v);
        return AlgebraicStatus::Element;
    }

    let mut remove = false;
    let status = {
        let existing = map
            .get_val_mut_at(path)
            .expect("value existence was checked before mutable access");
        match existing.pjoin(&v) {
            AlgebraicResult::None => {
                remove = true;
                AlgebraicStatus::None
            }
            AlgebraicResult::Element(joined) => {
                *existing = joined;
                AlgebraicStatus::Element
            }
            AlgebraicResult::Identity(mask) if mask & SELF_IDENT > 0 => AlgebraicStatus::Identity,
            AlgebraicResult::Identity(mask) => {
                debug_assert!(mask & COUNTER_IDENT > 0);
                *existing = v;
                AlgebraicStatus::Element
            }
        }
    };

    if remove {
        let _ = map.remove_val_at(path, true);
    }
    status
}

fn repeated_root_old_join(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        sum += old_join_val_at(map, key, idx as u64) as u64;
    }
    sum
}

fn repeated_old_join(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        sum += old_join_val_at(map, key, idx as u64) as u64;
    }
    sum
}

fn repeated_root_get_or_set(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        let val = map.get_val_or_set_mut_at(key, 0);
        *val = val.wrapping_add(idx as u64);
        sum += *val;
    }
    sum
}

fn repeated_get_or_set(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let val = map.get_val_or_set_mut_at(key, 0);
        *val = val.wrapping_add(idx as u64);
        sum += *val;
    }
    sum
}

fn repeated_root_old_get_or_set(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        let mut zipper = map.write_zipper_at_path(key);
        let val = zipper.get_val_or_set_mut(0);
        *val = val.wrapping_add(idx as u64);
        sum += *val;
    }
    sum
}

fn repeated_old_get_or_set(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let mut zipper = map.write_zipper_at_path(key);
        let val = zipper.get_val_or_set_mut(0);
        *val = val.wrapping_add(idx as u64);
        sum += *val;
    }
    sum
}

fn repeated_root_remove_no_prune(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        let removed = map.remove_val_at(key, false).unwrap_or(0);
        sum += removed;
        map.set_val_at(key, removed.wrapping_add(idx as u64));
    }
    sum
}

fn repeated_remove_no_prune(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let removed = map.remove_val_at(key, false).unwrap_or(0);
        sum += removed;
        map.set_val_at(key, removed.wrapping_add(idx as u64));
    }
    sum
}

fn old_remove_val_at_no_prune(map: &mut PathMap<u64>, path: &[u8]) -> Option<u64> {
    let mut zipper = map.write_zipper();
    zipper.descend_to(path);
    zipper.remove_val(false)
}

fn repeated_root_old_remove_no_prune(map: &mut PathMap<u64>, keys: &[[u8; 1]]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let key = &keys[idx % keys.len()];
        let removed = old_remove_val_at_no_prune(map, key).unwrap_or(0);
        sum += removed;
        map.set_val_at(key, removed.wrapping_add(idx as u64));
    }
    sum
}

fn repeated_old_remove_no_prune(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        let removed = old_remove_val_at_no_prune(map, key).unwrap_or(0);
        sum += removed;
        map.set_val_at(key, removed.wrapping_add(idx as u64));
    }
    sum
}

fn old_remove_branches_at_no_prune(map: &mut PathMap<u64>, path: &[u8]) -> bool {
    let mut zipper = map.write_zipper();
    zipper.descend_to(path);
    zipper.remove_branches(false)
}

fn repeated_remove_branches_no_prune(map: &mut PathMap<u64>, key: &[u8], child_key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        sum += map.remove_branches_at(key, false) as u64;
        map.set_val_at(child_key, idx as u64);
    }
    sum
}

fn repeated_old_remove_branches_no_prune(
    map: &mut PathMap<u64>,
    key: &[u8],
    child_key: &[u8],
) -> u64 {
    let mut sum = 0_u64;
    for idx in 0..LOOKUPS {
        sum += old_remove_branches_at_no_prune(map, key) as u64;
        map.set_val_at(child_key, idx as u64);
    }
    sum
}

fn old_create_path(map: &mut PathMap<u64>, path: &[u8]) -> bool {
    let mut zipper = map.write_zipper();
    zipper.descend_to(path);
    zipper.create_path()
}

fn repeated_create_path(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for _ in 0..LOOKUPS {
        sum += map.create_path(key) as u64;
        map.prune_path(key);
    }
    sum
}

fn repeated_old_create_path(map: &mut PathMap<u64>, key: &[u8]) -> u64 {
    let mut sum = 0_u64;
    for _ in 0..LOOKUPS {
        sum += old_create_path(map, key) as u64;
        map.prune_path(key);
    }
    sum
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_linear_child_hit(bencher: Bencher, fanout: usize) {
    let (keys, _mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_linear_child_index(&keys));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_mask_rank_hit(bencher: Bencher, fanout: usize) {
    let (keys, mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_mask_rank_index(&keys, mask));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_binary_child_hit(bencher: Bencher, fanout: usize) {
    let (mut keys, _mask) = local_lookup_keys(fanout);
    keys.sort_unstable();

    bencher.bench_local(|| {
        black_box(repeated_binary_child_index(&keys));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_branchless_linear_child_hit(bencher: Bencher, fanout: usize) {
    let (keys, _mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_branchless_linear_child_index(&keys));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_legacy_indexed_bit_forward(bencher: Bencher, fanout: usize) {
    let (_keys, mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_legacy_indexed_bit_forward(mask, fanout));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_legacy_indexed_bit_backward(bencher: Bencher, fanout: usize) {
    let (_keys, mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_legacy_indexed_bit_backward(mask, fanout));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_indexed_bit_forward(bencher: Bencher, fanout: usize) {
    let (_keys, mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_indexed_bit_forward(mask, fanout));
    });
}

#[divan::bench(args = [1usize, 2, 3, 4, 8, 16, 32, 64, 128])]
fn node_layout_kernel_indexed_bit_backward(bencher: Bencher, fanout: usize) {
    let (_keys, mask) = local_lookup_keys(fanout);

    bencher.bench_local(|| {
        black_box(repeated_indexed_bit_backward(mask, fanout));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_lookup(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        black_box(repeated_root_lookup(&map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_lookup(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        black_box(repeated_root_lookup(&map, &keys));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_path_exists(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        black_box(repeated_root_path_exists(&map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_path_exists(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        black_box(repeated_root_path_exists(&map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_path_exists(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        black_box(repeated_path_exists(&map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_path_exists(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        black_box(repeated_path_exists(&map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_path_exists(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        black_box(repeated_path_exists(&map, &key));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_update(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_update(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_update(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_update(&mut map, &keys));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_join(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_join(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_join(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_join(&mut map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_join(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_join(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_join(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_join(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_join(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_join(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_old_join(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_old_join(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_old_join(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_old_join(&mut map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_old_join(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_join(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_old_join(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_join(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_old_join(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_join(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_get_or_set(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_get_or_set(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_get_or_set(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_get_or_set(&mut map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_get_or_set(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_get_or_set(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_get_or_set(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_get_or_set(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_get_or_set(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_get_or_set(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_old_get_or_set(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_old_get_or_set(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_old_get_or_set(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_old_get_or_set(&mut map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_old_get_or_set(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_get_or_set(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_old_get_or_set(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_get_or_set(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_old_get_or_set(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_get_or_set(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_remove_no_prune(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_remove_no_prune(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_remove_no_prune(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_remove_no_prune(&mut map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_remove_no_prune(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_remove_no_prune(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_remove_no_prune(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_remove_no_prune(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_remove_no_prune(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_remove_no_prune(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2])]
fn node_layout_line_list_root_old_remove_no_prune(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_old_remove_no_prune(&mut map, &keys));
    });
}

#[divan::bench(args = [3usize, 4, 8, 16, 32, 64, 128])]
fn node_layout_dense_root_old_remove_no_prune(bencher: Bencher, fanout: usize) {
    let (map, keys) = root_fanout_map(fanout);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_root_old_remove_no_prune(&mut map, &keys));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_old_remove_no_prune(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_remove_no_prune(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_old_remove_no_prune(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_remove_no_prune(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_old_remove_no_prune(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_remove_no_prune(&mut map, &key));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_remove_branches_no_prune(bencher: Bencher, byte_len: usize) {
    let (map, key, child_key) = compressed_branch_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_remove_branches_no_prune(
            &mut map, &key, &child_key,
        ));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_remove_branches_no_prune(bencher: Bencher, depth: usize) {
    let (map, key, child_key) = nested_branch_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_remove_branches_no_prune(
            &mut map, &key, &child_key,
        ));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_remove_branches_no_prune(bencher: Bencher, depth: usize) {
    let (map, key, child_key) = nested_branch_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_remove_branches_no_prune(
            &mut map, &key, &child_key,
        ));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_old_remove_branches_no_prune(bencher: Bencher, byte_len: usize) {
    let (map, key, child_key) = compressed_branch_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_remove_branches_no_prune(
            &mut map, &key, &child_key,
        ));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_old_remove_branches_no_prune(bencher: Bencher, depth: usize) {
    let (map, key, child_key) = nested_branch_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_remove_branches_no_prune(
            &mut map, &key, &child_key,
        ));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_old_remove_branches_no_prune(bencher: Bencher, depth: usize) {
    let (map, key, child_key) = nested_branch_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_remove_branches_no_prune(
            &mut map, &key, &child_key,
        ));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_create_path(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_missing_path_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_create_path(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_create_path(bencher: Bencher, depth: usize) {
    let (map, key) = nested_missing_path_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_create_path(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_create_path(bencher: Bencher, depth: usize) {
    let (map, key) = nested_missing_path_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_create_path(&mut map, &key));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_old_create_path(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_missing_path_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_create_path(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_old_create_path(bencher: Bencher, depth: usize) {
    let (map, key) = nested_missing_path_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_create_path(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_old_create_path(bencher: Bencher, depth: usize) {
    let (map, key) = nested_missing_path_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_old_create_path(&mut map, &key));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_update(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_update(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_update(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_update(&mut map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_update(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        let mut map = map.clone();
        black_box(repeated_update(&mut map, &key));
    });
}

#[divan::bench(args = [4usize, 8, 16, 24, 32])]
fn node_layout_compressed_key_lookup(bencher: Bencher, byte_len: usize) {
    let (map, key) = compressed_key_map(byte_len);

    bencher.bench_local(|| {
        black_box(repeated_lookup(&map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_line_list_lookup(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 2);

    bencher.bench_local(|| {
        black_box(repeated_lookup(&map, &key));
    });
}

#[divan::bench(args = [1usize, 2, 4, 8, 16])]
fn node_layout_nested_dense_lookup(bencher: Bencher, depth: usize) {
    let (map, key) = nested_fanout_map(depth, 3);

    bencher.bench_local(|| {
        black_box(repeated_lookup(&map, &key));
    });
}
