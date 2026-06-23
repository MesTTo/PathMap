use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use pathmap::utils::ByteMask;
use pathmap::zipper::{ZipperMoving, ZipperWriting};
use pathmap::PathMap;

const BRANCH_COUNTS: [usize; 4] = [1, 4, 16, 64];

#[derive(Clone, Copy)]
enum MaskShape {
    Contiguous,
    Dispersed,
}

impl MaskShape {
    fn name(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Dispersed => "dispersed",
        }
    }
}

struct GraftFixture {
    dst: PathMap<u32>,
    maps: Vec<PathMap<u32>>,
    child_mask: ByteMask,
}

fn child_bytes(count: usize, shape: MaskShape) -> Vec<u8> {
    debug_assert!(count <= 256);
    let mut bytes: Vec<u8> = (0u8..=255).collect();
    if matches!(shape, MaskShape::Dispersed) {
        bytes.sort_by_key(|byte| byte.wrapping_mul(73).wrapping_add(19));
    }
    bytes.truncate(count);
    bytes
}

fn make_child_map(child: u8, idx: u32) -> PathMap<u32> {
    let mut map = PathMap::new();
    map.set_val_at([], 20_000 + idx);
    map.set_val_at([b':', child, b':', b'a'], 20_001 + idx);
    map.set_val_at([b':', child, b':', b'b', b':', b'c'], 20_002 + idx);
    map
}

fn make_fixture(branch_count: usize, shape: MaskShape, dst_extra_count: usize) -> GraftFixture {
    let graft_children = child_bytes(branch_count, shape);
    let child_mask = ByteMask::from_iter(graft_children.iter().copied());

    let mut dst = PathMap::new();
    dst.set_val_at([], 1);

    for (idx, child) in child_bytes(dst_extra_count.max(branch_count), shape)
        .into_iter()
        .enumerate()
    {
        let idx = idx as u32;
        dst.set_val_at([child], 10_000 + idx);
        dst.set_val_at([child, b':', b'o', b'l', b'd'], 10_001 + idx);
    }

    let maps = graft_children
        .into_iter()
        .enumerate()
        .map(|(idx, child)| make_child_map(child, idx as u32))
        .collect();

    GraftFixture {
        dst,
        maps,
        child_mask,
    }
}

fn grouped_graft(
    mut dst: PathMap<u32>,
    maps: Vec<PathMap<u32>>,
    child_mask: ByteMask,
    remove_unset: bool,
) -> usize {
    {
        let mut zipper = dst.write_zipper();
        zipper.graft_child_maps(child_mask, maps, remove_unset);
    }
    dst.val_count()
}

fn manual_graft(
    mut dst: PathMap<u32>,
    maps: Vec<PathMap<u32>>,
    child_mask: ByteMask,
    remove_unset: bool,
) -> usize {
    {
        let mut zipper = dst.write_zipper();
        if remove_unset {
            zipper.remove_unmasked_branches(child_mask, false);
        }
        let mut maps = maps.into_iter();
        for child in child_mask.iter() {
            let map = maps
                .next()
                .expect("fixture should contain one map for each child-mask bit");
            zipper.descend_to_byte(child);
            zipper.graft_map(map);
            zipper.ascend_byte();
        }
    }
    dst.val_count()
}

fn grouped_child_map_grafting(c: &mut Criterion) {
    let mut group = c.benchmark_group("graft_child_maps");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));

    for branch_count in BRANCH_COUNTS {
        for shape in [MaskShape::Contiguous, MaskShape::Dispersed] {
            let remove_fixture = make_fixture(branch_count, shape, branch_count);
            let keep_fixture = make_fixture(branch_count, shape, 150);

            let id = format!("{}/{}", shape.name(), branch_count);

            group.bench_with_input(
                BenchmarkId::new("grouped_remove_unset", &id),
                &remove_fixture,
                |b, fixture| {
                    b.iter_batched(
                        || {
                            (
                                fixture.dst.clone(),
                                fixture.maps.clone(),
                                fixture.child_mask,
                            )
                        },
                        |(dst, maps, child_mask)| {
                            black_box(grouped_graft(dst, maps, child_mask, true))
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new("manual_remove_unset", &id),
                &remove_fixture,
                |b, fixture| {
                    b.iter_batched(
                        || {
                            (
                                fixture.dst.clone(),
                                fixture.maps.clone(),
                                fixture.child_mask,
                            )
                        },
                        |(dst, maps, child_mask)| {
                            black_box(manual_graft(dst, maps, child_mask, true))
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new("grouped_keep_unset", &id),
                &keep_fixture,
                |b, fixture| {
                    b.iter_batched(
                        || {
                            (
                                fixture.dst.clone(),
                                fixture.maps.clone(),
                                fixture.child_mask,
                            )
                        },
                        |(dst, maps, child_mask)| {
                            black_box(grouped_graft(dst, maps, child_mask, false))
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new("manual_keep_unset", &id),
                &keep_fixture,
                |b, fixture| {
                    b.iter_batched(
                        || {
                            (
                                fixture.dst.clone(),
                                fixture.maps.clone(),
                                fixture.child_mask,
                            )
                        },
                        |(dst, maps, child_mask)| {
                            black_box(manual_graft(dst, maps, child_mask, false))
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, grouped_child_map_grafting);
criterion_main!(benches);
