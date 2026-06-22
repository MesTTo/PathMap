use divan::{Bencher, Divan, black_box};
use pathmap::PathMap;
use pathmap::zipper::*;

fn main() {
    Divan::from_args().sample_count(100).main();
}

fn shared_compounds() -> PathMap<()> {
    let mut shared = PathMap::new();
    shared.set_val_at(b"compounds:atropine", ());
    shared.set_val_at(b"compounds:botox", ());
    shared.set_val_at(b"compounds:colchicine", ());
    shared.set_val_at(b"compounds:digitalis", ());
    shared
}

fn shared_compound_space() -> PathMap<()> {
    let shared = shared_compounds();
    let mut map = PathMap::new();
    {
        let mut zipper = map.write_zipper();
        zipper.descend_to(b"keep_in_the_pharmacy:");
        zipper.graft_map(shared.clone());
        zipper.move_to_path(b"handle_with_care:");
        zipper.graft_map(shared);
    }
    map
}

#[divan::bench(args = [1usize, 10, 100])]
fn write_zipper_shared_subtrie_movement(bencher: Bencher, repeats: usize) {
    let mut map = shared_compound_space();

    bencher.bench_local(|| {
        let mut hits = 0usize;
        for _ in 0..repeats {
            let mut zipper = black_box(&mut map).write_zipper();
            zipper.descend_to(b"handle_with_care:compounds:colchicine");
            if zipper.is_val() {
                hits += 1;
            }
        }
        black_box(hits);
    });
}

#[divan::bench(args = [1usize, 10, 100])]
fn zipper_head_sibling_reader_writer_movement(bencher: Bencher, repeats: usize) {
    let mut map = shared_compound_space();

    bencher.bench_local(|| {
        let zh = black_box(&mut map).zipper_head();
        let mut hits = 0usize;
        for _ in 0..repeats {
            let reader = zh
                .read_zipper_at_borrowed_path(b"keep_in_the_pharmacy:compounds:")
                .unwrap();
            let mut writer = zh
                .write_zipper_at_exclusive_path(b"handle_with_care:")
                .unwrap();
            writer.descend_to(b"compounds:colchicine");
            if writer.is_val() && reader.val_count() == 4 {
                hits += 1;
            }
        }
        black_box(hits);
    });
}

#[divan::bench(args = [1usize, 10, 100])]
fn write_zipper_scouted_get_val_or_set(bencher: Bencher, repeats: usize) {
    bencher.bench_local(|| {
        let mut writes = 0usize;
        for _ in 0..repeats {
            let mut map = shared_compound_space();
            let mut zipper = black_box(&mut map).write_zipper();
            zipper.descend_to(b"handle_with_care:compounds:endrin");
            *zipper.get_val_or_set_mut_with(|| ()) = ();
            writes += 1;
        }
        black_box(writes);
    });
}

#[divan::bench(args = [1usize, 10, 100])]
fn write_zipper_scouted_graft_map(bencher: Bencher, repeats: usize) {
    bencher.bench_local(|| {
        let mut grafts = 0usize;
        for _ in 0..repeats {
            let mut replacement = PathMap::new();
            replacement.write_zipper().set_val(());
            replacement.set_val_at(b":replacement", ());

            let mut map = shared_compound_space();
            let mut zipper = black_box(&mut map).write_zipper();
            zipper.descend_to(b"handle_with_care:compounds:endrin");
            zipper.graft_map(replacement);
            grafts += 1;
        }
        black_box(grafts);
    });
}

#[divan::bench(args = [1usize, 10, 100])]
fn write_zipper_scouted_val_count(bencher: Bencher, repeats: usize) {
    let mut map = shared_compound_space();

    bencher.bench_local(|| {
        let mut values = 0usize;
        for _ in 0..repeats {
            let mut zipper = black_box(&mut map).write_zipper();
            zipper.descend_to(b"handle_with_care:compounds:");
            values += zipper.val_count();
        }
        black_box(values);
    });
}
