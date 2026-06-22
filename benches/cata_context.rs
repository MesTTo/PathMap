use divan::{Bencher, Divan, black_box};
use pathmap::PathMap;
use pathmap::morphisms::{Catamorphism, PathScope};

fn main() {
    Divan::from_args().sample_count(20).main();
}

fn shared_leaf_map() -> PathMap<u32> {
    let mut map = PathMap::new();
    map.set_val_at(b"leaf", 1);
    map.set_val_at(b"limb", 2);
    map.set_val_at(b"twig", 3);
    map
}

fn heavily_shared_map(levels: usize, fanout: u8) -> PathMap<u32> {
    let mut map = shared_leaf_map();
    for _level in 0..levels {
        let shared = map.read_zipper();
        let next = PathMap::new_from_ana(false, |quit, _val, children, _path| {
            if quit {
                return;
            }
            for byte in 0..fanout {
                children.graft_at_byte(byte, &shared);
            }
        });
        drop(shared);
        map = next;
    }
    map
}

fn contextual_score(map: &PathMap<u32>, scope: PathScope) -> u64 {
    map.read_zipper().into_cata_jumping_contextual_cached(
        scope,
        |_mask, children: &mut [u64], value, sub_path, path_context| {
            children.iter().copied().sum::<u64>()
                + value.map_or(0, |value| *value as u64)
                + sub_path.len() as u64
                + path_context.len() as u64
        },
    )
}

fn repeated_contextual_score(map: &PathMap<u32>, scope: PathScope, repeats: usize) -> u64 {
    let mut total = 0_u64;
    for _ in 0..repeats {
        total = total.wrapping_add(contextual_score(map, scope));
    }
    total
}

#[divan::bench(args = [1usize, 10, 100])]
fn cata_context_suffix_zero(bencher: Bencher, repeats: usize) {
    let map = heavily_shared_map(4, 4);

    bencher.bench_local(|| {
        black_box(repeated_contextual_score(
            black_box(&map),
            PathScope::Suffix(0),
            repeats,
        ));
    });
}

#[divan::bench(args = [1usize, 10, 100])]
fn cata_context_suffix_one(bencher: Bencher, repeats: usize) {
    let map = heavily_shared_map(4, 4);

    bencher.bench_local(|| {
        black_box(repeated_contextual_score(
            black_box(&map),
            PathScope::Suffix(1),
            repeats,
        ));
    });
}

#[divan::bench(args = [1usize, 10, 100])]
fn cata_context_full_path(bencher: Bencher, repeats: usize) {
    let map = heavily_shared_map(4, 4);

    bencher.bench_local(|| {
        black_box(repeated_contextual_score(
            black_box(&map),
            PathScope::Full,
            repeats,
        ));
    });
}
