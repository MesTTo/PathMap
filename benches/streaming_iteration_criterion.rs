use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pathmap::zipper::{ZipperIteration, ZipperMoving, ZipperValues};
use pathmap::PathMap;

fn path(n: usize) -> Vec<u8> {
    let mut path = Vec::with_capacity(32);
    path.extend_from_slice(b"stream:");
    path.extend_from_slice(format!("{n:08x}").as_bytes());
    path.extend_from_slice(b":value");
    path
}

fn make_map(path_count: usize) -> PathMap<usize> {
    let mut map = PathMap::new();
    for i in 0..path_count {
        map.set_val_at(path(i), i);
    }
    map
}

fn owned_iter_sum(map: &PathMap<usize>) -> usize {
    map.iter()
        .map(|(path, value)| path.len().wrapping_add(*value))
        .fold(0usize, usize::wrapping_add)
}

fn manual_zipper_sum(map: &PathMap<usize>) -> usize {
    let mut sum = 0usize;
    let mut zipper = map.read_zipper();
    if let Some(value) = zipper.val() {
        sum = sum.wrapping_add(zipper.path().len()).wrapping_add(*value);
    }
    while zipper.to_next_val() {
        if let Some(value) = zipper.val() {
            sum = sum.wrapping_add(zipper.path().len()).wrapping_add(*value);
        }
    }
    sum
}

fn streaming_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathmap_streaming_iteration");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));

    for path_count in [128usize, 1024, 4096] {
        let map = make_map(path_count);

        group.bench_with_input(
            BenchmarkId::new("owned_iter_vec_keys", path_count),
            &map,
            |b, map| b.iter(|| black_box(owned_iter_sum(map))),
        );

        group.bench_with_input(
            BenchmarkId::new("manual_zipper", path_count),
            &map,
            |b, map| b.iter(|| black_box(manual_zipper_sum(map))),
        );
    }

    group.finish();
}

criterion_group!(benches, streaming_iteration);
criterion_main!(benches);
