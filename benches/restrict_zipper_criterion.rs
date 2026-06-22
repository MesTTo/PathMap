use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pathmap::PathMap;
use pathmap::zipper::{RestrictZipper, Zipper, ZipperIteration, ZipperSubtries};

struct RestrictFixture {
    data: PathMap<usize>,
    guard: PathMap<usize>,
    heterotyped_guard: PathMap<&'static str>,
}

fn path(prefix: &[u8], n: usize, suffix: &[u8]) -> Vec<u8> {
    let mut path = Vec::with_capacity(prefix.len() + 8 + suffix.len());
    path.extend_from_slice(prefix);
    path.extend_from_slice(format!("{n:08x}").as_bytes());
    path.extend_from_slice(suffix);
    path
}

fn make_fixture(path_count: usize) -> RestrictFixture {
    let mut data = PathMap::new();
    let mut guard = PathMap::new();
    let mut heterotyped_guard = PathMap::new();

    for i in 0..path_count {
        data.set_val_at(path(b"active:", i, b":leaf/a"), i);
        data.set_val_at(path(b"active:", i, b":leaf/b"), i);
        data.create_path(path(b"active:", i, b":dangling"));

        data.set_val_at(path(b"shared:", i, b":value"), i);
        data.create_path(path(b"shared:", i, b":dangling"));

        data.set_val_at(path(b"blocked:", i, b":value"), i);
        data.create_path(path(b"blocked:", i, b":dangling"));

        guard.set_val_at(path(b"active:", i, b""), i);
        guard.create_path(path(b"shared:", i, b""));
        guard.set_val_at(path(b"shared:", i, b":value"), i);

        heterotyped_guard.set_val_at(path(b"active:", i, b""), "guard");
        heterotyped_guard.create_path(path(b"shared:", i, b""));
        heterotyped_guard.set_val_at(path(b"shared:", i, b":value"), "guard");
    }

    RestrictFixture {
        data,
        guard,
        heterotyped_guard,
    }
}

fn count_values<Z>(mut zipper: Z) -> usize
where
    Z: ZipperIteration,
{
    let mut count = usize::from(zipper.is_val());
    while zipper.to_next_val() {
        count += 1;
    }
    count
}

fn restrict_guard_observation(c: &mut Criterion) {
    let mut group = c.benchmark_group("restrict_zipper_guard_observation");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));

    for path_count in [16usize, 128, 512] {
        let fixture = make_fixture(path_count);

        group.bench_with_input(
            BenchmarkId::new("lazy_root_child_count", path_count),
            &fixture,
            |b, fixture| {
                b.iter(|| {
                    let zipper = RestrictZipper::new(
                        fixture.data.read_zipper(),
                        fixture.guard.read_zipper(),
                    );
                    black_box(zipper.child_count())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lazy_value_iteration", path_count),
            &fixture,
            |b, fixture| {
                b.iter(|| {
                    let zipper = RestrictZipper::new(
                        fixture.data.read_zipper(),
                        fixture.guard.read_zipper(),
                    );
                    black_box(count_values(zipper))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lazy_materialize", path_count),
            &fixture,
            |b, fixture| {
                b.iter(|| {
                    let zipper = RestrictZipper::new(
                        fixture.data.read_zipper(),
                        fixture.guard.read_zipper(),
                    );
                    black_box(zipper.try_make_map())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eager_restrict", path_count),
            &fixture,
            |b, fixture| b.iter(|| black_box(fixture.data.restrict(&fixture.guard))),
        );

        group.bench_with_input(
            BenchmarkId::new("restrict_by_paths_heterotyped", path_count),
            &fixture,
            |b, fixture| {
                b.iter(|| black_box(fixture.data.restrict_by_paths(&fixture.heterotyped_guard)))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, restrict_guard_observation);
criterion_main!(benches);
