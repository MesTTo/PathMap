use divan::{Bencher, Divan, black_box};
use pathmap::PathMap;
use pathmap::utils::ByteMask;
use pathmap::zipper::{PrefixZipper, Zipper, ZipperIteration, ZipperMoving};

fn main() {
    Divan::from_args().sample_count(16).main();
}

struct DefaultIter<Z>(Z);

impl<Z> Zipper for DefaultIter<Z>
where
    Z: Zipper,
{
    fn path_exists(&self) -> bool {
        self.0.path_exists()
    }

    fn is_val(&self) -> bool {
        self.0.is_val()
    }

    fn child_count(&self) -> usize {
        self.0.child_count()
    }

    fn child_mask(&self) -> ByteMask {
        self.0.child_mask()
    }
}

impl<Z> ZipperMoving for DefaultIter<Z>
where
    Z: ZipperMoving,
{
    fn at_root(&self) -> bool {
        self.0.at_root()
    }

    fn reset(&mut self) {
        self.0.reset()
    }

    fn path(&self) -> &[u8] {
        self.0.path()
    }

    fn val_count(&self) -> usize {
        self.0.val_count()
    }

    fn move_to_path<K: AsRef<[u8]>>(&mut self, path: K) -> usize {
        self.0.move_to_path(path)
    }

    fn descend_to<K: AsRef<[u8]>>(&mut self, path: K) {
        self.0.descend_to(path)
    }

    fn descend_to_existing<K: AsRef<[u8]>>(&mut self, path: K) -> usize {
        self.0.descend_to_existing(path)
    }

    fn descend_to_byte(&mut self, byte: u8) {
        self.0.descend_to_byte(byte)
    }

    fn descend_indexed_byte(&mut self, idx: usize) -> bool {
        self.0.descend_indexed_byte(idx)
    }

    fn descend_first_byte(&mut self) -> bool {
        self.0.descend_first_byte()
    }

    fn descend_until(&mut self) -> bool {
        self.0.descend_until()
    }

    fn ascend(&mut self, steps: usize) -> bool {
        self.0.ascend(steps)
    }

    fn ascend_byte(&mut self) -> bool {
        self.0.ascend_byte()
    }

    fn ascend_until(&mut self) -> bool {
        self.0.ascend_until()
    }

    fn ascend_until_branch(&mut self) -> bool {
        self.0.ascend_until_branch()
    }

    fn to_next_sibling_byte(&mut self) -> bool {
        self.0.to_next_sibling_byte()
    }

    fn to_next_step(&mut self) -> bool {
        self.0.to_next_step()
    }
}

impl<Z> ZipperIteration for DefaultIter<Z> where Z: ZipperMoving {}

fn prefix(prefix_len: usize) -> Vec<u8> {
    (0..prefix_len)
        .map(|idx| b'a'.wrapping_add((idx % 23) as u8))
        .collect()
}

fn source_with_root_value() -> PathMap<u64> {
    let mut map = PathMap::new();
    map.set_val_at(b"", 1);
    map
}

fn source_with_leaf_values(count: usize) -> PathMap<u64> {
    let mut map = PathMap::new();
    for idx in 0..count {
        let key = [((idx >> 8) & 0xff) as u8, (idx & 0xff) as u8, b':', b'v'];
        map.set_val_at(&key, idx as u64);
    }
    map
}

#[divan::bench(args = [0, 8, 64, 256])]
fn default_iteration_first_prefixed_root_value(bencher: Bencher, prefix_len: usize) {
    let map = source_with_root_value();
    let prefix = prefix(prefix_len);

    bencher.bench_local(|| {
        let mut zipper = DefaultIter(PrefixZipper::new(
            black_box(prefix.as_slice()),
            map.read_zipper(),
        ));
        let moved = zipper.to_next_val();
        black_box((moved, zipper.path().len()));
    });
}

#[divan::bench(args = [0, 8, 64, 256])]
fn optimized_iteration_first_prefixed_root_value(bencher: Bencher, prefix_len: usize) {
    let map = source_with_root_value();
    let prefix = prefix(prefix_len);

    bencher.bench_local(|| {
        let mut zipper = PrefixZipper::new(black_box(prefix.as_slice()), map.read_zipper());
        let moved = zipper.to_next_val();
        black_box((moved, zipper.path().len()));
    });
}

#[divan::bench(args = [8, 64, 256])]
fn default_iteration_prefixed_leaf_values(bencher: Bencher, prefix_len: usize) {
    let map = source_with_leaf_values(256);
    let prefix = prefix(prefix_len);

    bencher.bench_local(|| {
        let mut zipper = DefaultIter(PrefixZipper::new(
            black_box(prefix.as_slice()),
            map.read_zipper(),
        ));
        let mut count = 0_u64;
        while zipper.to_next_val() {
            count += 1;
        }
        black_box(count);
    });
}

#[divan::bench(args = [8, 64, 256])]
fn optimized_iteration_prefixed_leaf_values(bencher: Bencher, prefix_len: usize) {
    let map = source_with_leaf_values(256);
    let prefix = prefix(prefix_len);

    bencher.bench_local(|| {
        let mut zipper = PrefixZipper::new(black_box(prefix.as_slice()), map.read_zipper());
        let mut count = 0_u64;
        while zipper.to_next_val() {
            count += 1;
        }
        black_box(count);
    });
}
