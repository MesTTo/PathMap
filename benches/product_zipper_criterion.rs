use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use pathmap::PathMap;
use pathmap::zipper::{OverlayZipper, ProductZipper, Zipper};

struct ProductFixture {
    primary: PathMap<usize>,
    secondary: PathMap<usize>,
    overlay_left: PathMap<usize>,
    overlay_right: PathMap<usize>,
}

fn path(prefix: &[u8], n: usize, suffix: &[u8]) -> Vec<u8> {
    let mut path = Vec::with_capacity(prefix.len() + 8 + suffix.len());
    path.extend_from_slice(prefix);
    path.extend_from_slice(format!("{n:08x}").as_bytes());
    path.extend_from_slice(suffix);
    path
}

fn make_fixture(path_count: usize) -> ProductFixture {
    let mut primary = PathMap::new();
    let mut secondary = PathMap::new();
    let mut overlay_left = PathMap::new();
    let mut overlay_right = PathMap::new();

    for i in 0..path_count {
        primary.set_val_at(path(b"primary:", i, b":terminal"), i);
        secondary.set_val_at(path(b"scope:", i, b":secondary"), i);
        overlay_left.set_val_at(path(b"overlay:left:", i, b""), i);
        overlay_right.set_val_at(path(b"overlay:right:", i, b""), i + path_count);
    }

    ProductFixture {
        primary,
        secondary,
        overlay_left,
        overlay_right,
    }
}

fn product_secondary_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_zipper_secondary_construction");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(500));

    for path_count in [1usize, 16, 128] {
        let fixture = make_fixture(path_count);

        group.bench_with_input(
            BenchmarkId::new("borrowed_node_root_secondary", path_count),
            &fixture,
            |b, fixture| {
                b.iter(|| {
                    let zipper = ProductZipper::new(
                        fixture.primary.read_zipper(),
                        [fixture.secondary.read_zipper()],
                    );
                    black_box(zipper.child_count())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("focused_secondary_materialized", path_count),
            &fixture,
            |b, fixture| {
                b.iter(|| {
                    let zipper = ProductZipper::new(
                        fixture.primary.read_zipper(),
                        [fixture.secondary.read_zipper_at_path(b"sco")],
                    );
                    black_box(zipper.child_count())
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("virtual_overlay_secondary_materialized", path_count),
            &fixture,
            |b, fixture| {
                b.iter_batched(
                    || {
                        OverlayZipper::new(
                            fixture.overlay_left.read_zipper(),
                            fixture.overlay_right.read_zipper(),
                        )
                    },
                    |overlay| {
                        let zipper = ProductZipper::new(fixture.primary.read_zipper(), [overlay]);
                        black_box(zipper.child_count())
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, product_secondary_construction);
criterion_main!(benches);
