use divan::{Bencher, Divan};

use pathmap::PathMap;
use pathmap::utils::ByteMask;
use pathmap::zipper::*;

fn main() {
    let divan = Divan::from_args().sample_count(1000);

    divan.main();
}

const BRANCH_COUNTS: [usize; 11] = [1, 2, 3, 4, 6, 8, 15, 30, 60, 100, 200];

fn make_case(masked_branch_count: usize) -> (PathMap<u32>, PathMap<u32>, ByteMask) {
    let masked_children: Vec<u8> = (0..masked_branch_count).map(|idx| idx as u8).collect();
    let extra_branch_count = (masked_branch_count / 2).max(2);
    let extra_children: Vec<u8> = (masked_branch_count..(masked_branch_count + extra_branch_count))
        .map(|idx| idx as u8)
        .collect();

    let mut src = PathMap::new();
    let mut dst = PathMap::new();

    src.set_val_at(b"", 0);
    dst.set_val_at(b"", 1);

    for (idx, child) in masked_children.iter().copied().enumerate() {
        let idx = idx as u32;

        dst.set_val_at(&[child], 10_000 + idx);
        dst.set_val_at(&[child, 1], 10_001 + idx);
        dst.set_val_at(&[child, 2, 3], 10_002 + idx);

        src.set_val_at(&[child], 20_000 + idx);
        src.set_val_at(&[child, 4], 20_001 + idx);
        src.set_val_at(&[child, 5, 6], 20_002 + idx);
        src.set_val_at(&[child, 7, 8, 9], 20_003 + idx);
    }

    for (idx, child) in extra_children.iter().copied().enumerate() {
        let idx = idx as u32;
        dst.set_val_at(&[child], 30_000 + idx);
        dst.set_val_at(&[child, 11], 30_001 + idx);
        dst.set_val_at(&[child, 12, 13], 30_002 + idx);

        src.set_val_at(&[child], 40_000 + idx);
        src.set_val_at(&[child, 14], 40_001 + idx);
    }

    (src, dst, ByteMask::from_iter(masked_children))
}

fn make_partial_source_case(masked_branch_count: usize) -> (PathMap<u32>, PathMap<u32>, ByteMask) {
    let masked_children: Vec<u8> = (0..masked_branch_count).map(|idx| idx as u8).collect();
    let extra_branch_count = (masked_branch_count / 2).max(2);
    let extra_children: Vec<u8> = (masked_branch_count..(masked_branch_count + extra_branch_count))
        .map(|idx| idx as u8)
        .collect();

    let mut src = PathMap::new();
    let mut dst = PathMap::new();

    src.set_val_at(b"", 0);
    dst.set_val_at(b"", 1);

    for (idx, child) in masked_children.iter().copied().enumerate() {
        let idx = idx as u32;

        dst.set_val_at(&[child], 10_000 + idx);
        dst.set_val_at(&[child, 1], 10_001 + idx);
        dst.set_val_at(&[child, 2, 3], 10_002 + idx);

        //Having both a value and onward branch forces an early upgrade from the PairNode to a ByteNode
        // src.set_val_at(&[child], 20_000 + idx);
        src.set_val_at(&[child, 4], 20_001 + idx);
        src.set_val_at(&[child, 5, 6], 20_002 + idx);
        src.set_val_at(&[child, 7, 8, 9], 20_003 + idx);
    }

    for (idx, child) in extra_children.iter().copied().enumerate() {
        let idx = idx as u32;
        dst.set_val_at(&[child], 30_000 + idx);
        dst.set_val_at(&[child, 11], 30_001 + idx);
        dst.set_val_at(&[child, 12, 13], 30_002 + idx);
    }

    (src, dst, ByteMask::from_iter(masked_children))
}

/// `remove_unset = false`, `src` contains hundreds of branches, regardless of mask
#[divan::bench(sample_size = 1, args = BRANCH_COUNTS)]
fn graft_masked_branches_keep_full_src(bencher: Bencher, masked_branch_count: usize) {
    let (src_template, dst_template, child_mask) = make_case(masked_branch_count);

    let out = bencher
        .with_inputs(|| (src_template.clone(), dst_template.clone()))
        .bench_local_values(|(src, mut dst)| {
            let mut wz = dst.write_zipper_at_path(b"");
            let rz = src.read_zipper_at_path(b"");
            wz.graft_masked_branches(&rz, child_mask, false);
            drop(wz);
            dst
        });

    divan::black_box_drop(out);
}

/// `remove_unset = true`, `src` contains hundreds of branches, regardless of mask
#[divan::bench(sample_size = 1, args = BRANCH_COUNTS)]
fn graft_masked_branches_remove_full_src(bencher: Bencher, masked_branch_count: usize) {
    let (src_template, dst_template, child_mask) = make_case(masked_branch_count);

    let out = bencher
        .with_inputs(|| (src_template.clone(), dst_template.clone()))
        .bench_local_values(|(src, mut dst)| {
            let mut wz = dst.write_zipper_at_path(b"");
            let rz = src.read_zipper_at_path(b"");
            wz.graft_masked_branches(&rz, child_mask, true);
            drop(wz);
            dst
        });

    divan::black_box_drop(out);
}

/// `remove_unset = false`, `src` contains only the branches we're grafting
#[divan::bench(sample_size = 1, args = BRANCH_COUNTS)]
fn graft_masked_branches_keep_part_src(
    bencher: Bencher,
    masked_branch_count: usize,
) {
    let (src_template, dst_template, child_mask) = make_partial_source_case(masked_branch_count);

    let out = bencher
        .with_inputs(|| (src_template.clone(), dst_template.clone()))
        .bench_local_values(|(src, mut dst)| {
            let mut wz = dst.write_zipper_at_path(b"");
            let rz = src.read_zipper_at_path(b"");
            wz.graft_masked_branches(&rz, child_mask, false);
            drop(wz);
            dst
        });

    divan::black_box_drop(out);
}

/// `remove_unset = true`, `src` contains only the branches we're grafting
#[divan::bench(sample_size = 1, args = BRANCH_COUNTS)]
fn graft_masked_branches_remove_part_src(
    bencher: Bencher,
    masked_branch_count: usize,
) {
    let (src_template, dst_template, child_mask) = make_partial_source_case(masked_branch_count);

    let out = bencher
        .with_inputs(|| (src_template.clone(), dst_template.clone()))
        .bench_local_values(|(src, mut dst)| {
            let mut wz = dst.write_zipper_at_path(b"");
            let rz = src.read_zipper_at_path(b"");
            wz.graft_masked_branches(&rz, child_mask, true);
            drop(wz);
            dst
        });

    divan::black_box_drop(out);
}
