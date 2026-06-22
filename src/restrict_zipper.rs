use crate::PathMap;
use crate::alloc::{GlobalAlloc, global_alloc};
use crate::utils::{BitMask, ByteMask};
use crate::zipper::*;

/// A read-only virtual zipper that restricts data paths by prefixes that lead
/// to values in a guard zipper.
///
/// If the guard has a value at the current focus, that value validates the
/// current focus and every descendant. Before validation, traversal is limited
/// to branches that exist in both the data and guard spaces.
#[derive(Clone)]
pub struct RestrictZipper<DataZ, GuardZ> {
    data: DataZ,
    guard: GuardZ,
    active: bool,
    active_stack: Vec<bool>,
}

impl<DataZ, GuardZ> RestrictZipper<DataZ, GuardZ>
where
    DataZ: ZipperMoving,
    GuardZ: ZipperMoving,
{
    pub fn new(mut data: DataZ, mut guard: GuardZ) -> Self {
        data.reset();
        guard.reset();
        Self {
            data,
            guard,
            active: false,
            active_stack: Vec::new(),
        }
    }
}

impl<DataZ, GuardZ> RestrictZipper<DataZ, GuardZ>
where
    GuardZ: Zipper,
{
    #[inline]
    fn active_here(&self) -> bool {
        self.active || self.guard.is_val()
    }
}

impl<DataZ, GuardZ> RestrictZipper<DataZ, GuardZ>
where
    DataZ: ZipperMoving,
    GuardZ: ZipperMoving,
{
    #[inline]
    fn push_descend_state(&mut self) {
        self.active_stack.push(self.active);
        self.active = self.active_here();
    }

    fn pop_ascend_state(&mut self, steps: usize) {
        for _ in 0..steps {
            if let Some(active) = self.active_stack.pop() {
                self.active = active;
            } else {
                self.active = false;
                break;
            }
        }
    }
}

impl<DataZ, GuardZ, V> ZipperValues<V> for RestrictZipper<DataZ, GuardZ>
where
    DataZ: ZipperValues<V>,
    GuardZ: Zipper,
{
    fn val(&self) -> Option<&V> {
        #[cfg(feature = "counters")]
        crate::counters::record_restrict_value_filter();
        if self.active_here() {
            self.data.val()
        } else {
            None
        }
    }
}

impl<DataZ, GuardZ> Zipper for RestrictZipper<DataZ, GuardZ>
where
    DataZ: Zipper,
    GuardZ: Zipper,
{
    fn path_exists(&self) -> bool {
        self.data.path_exists() && (self.active_here() || self.guard.path_exists())
    }

    fn is_val(&self) -> bool {
        self.active_here() && self.data.is_val()
    }

    fn child_count(&self) -> usize {
        self.child_mask().count_bits()
    }

    fn child_mask(&self) -> ByteMask {
        let data_mask = self.data.child_mask();
        if self.active_here() {
            data_mask
        } else {
            #[cfg(feature = "counters")]
            crate::counters::record_restrict_child_mask_filter();
            data_mask & self.guard.child_mask()
        }
    }
}

impl<DataZ, GuardZ> ZipperMoving for RestrictZipper<DataZ, GuardZ>
where
    DataZ: ZipperMoving + Clone,
    GuardZ: ZipperMoving + Clone,
{
    fn at_root(&self) -> bool {
        self.data.at_root()
    }

    fn reset(&mut self) {
        self.data.reset();
        self.guard.reset();
        self.active = false;
        self.active_stack.clear();
    }

    fn path(&self) -> &[u8] {
        self.data.path()
    }

    fn val_count(&self) -> usize {
        let mut cursor = self.clone();
        cursor.reset();
        let mut count = cursor.is_val() as usize;
        while cursor.to_next_step() {
            if cursor.is_val() {
                count += 1;
            }
        }
        count
    }

    fn descend_to<K: AsRef<[u8]>>(&mut self, path: K) {
        for &byte in path.as_ref() {
            self.descend_to_byte(byte);
        }
    }

    fn descend_to_byte(&mut self, k: u8) {
        self.push_descend_state();
        self.data.descend_to_byte(k);
        self.guard.descend_to_byte(k);
    }

    fn ascend(&mut self, steps: usize) -> bool {
        let before = self.data.path().len();
        let data_ok = self.data.ascend(steps);
        let guard_ok = self.guard.ascend(steps);
        let ascended = before - self.data.path().len();
        self.pop_ascend_state(ascended);
        data_ok && guard_ok
    }

    fn ascend_until(&mut self) -> bool {
        let mut moved = false;
        while !self.at_root() {
            self.ascend_byte();
            moved = true;
            if self.at_root() || self.child_count() != 1 || self.is_val() {
                break;
            }
        }
        moved
    }

    fn ascend_until_branch(&mut self) -> bool {
        let mut moved = false;
        while !self.at_root() {
            self.ascend_byte();
            moved = true;
            if self.at_root() || self.child_count() != 1 {
                break;
            }
        }
        moved
    }
}

impl<DataZ, GuardZ> ZipperIteration for RestrictZipper<DataZ, GuardZ>
where
    DataZ: ZipperMoving + Clone,
    GuardZ: ZipperMoving + Clone,
{
}

impl<DataZ, GuardZ> ZipperAbsolutePath for RestrictZipper<DataZ, GuardZ>
where
    DataZ: ZipperAbsolutePath + Clone,
    GuardZ: ZipperMoving + Clone,
{
    fn origin_path(&self) -> &[u8] {
        self.data.origin_path()
    }

    fn root_prefix_path(&self) -> &[u8] {
        self.data.root_prefix_path()
    }
}

impl<DataZ, GuardZ, V> ZipperSubtries<V, GlobalAlloc> for RestrictZipper<DataZ, GuardZ>
where
    V: Clone + Send + Sync + Unpin,
    DataZ: ZipperMoving + ZipperValues<V> + Clone,
    GuardZ: ZipperMoving + Clone,
{
    fn native_subtries(&self) -> bool {
        false
    }

    fn try_make_map(&self) -> Option<PathMap<V, GlobalAlloc>> {
        Some(materialize_zipper(self.clone(), global_alloc()))
    }

    fn trie_ref(&self) -> Option<TrieRef<'_, V, GlobalAlloc>> {
        None
    }

    fn alloc(&self) -> GlobalAlloc {
        global_alloc()
    }
}

crate::impl_name_only_debug!(
    impl<DataZ, GuardZ> core::fmt::Debug for RestrictZipper<DataZ, GuardZ>
);

#[cfg(test)]
mod tests {
    use super::RestrictZipper;
    use crate::PathMap;
    use crate::zipper::{
        ZipperInfallibleSubtries, ZipperIteration, ZipperMoving, ZipperSubtries, ZipperValues,
    };

    fn value_paths<Z: ZipperIteration>(mut zipper: Z) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        if zipper.is_val() {
            out.push(zipper.path().to_vec());
        }
        while zipper.to_next_val() {
            out.push(zipper.path().to_vec());
        }
        out
    }

    fn collect_observed_pathspace<Z, V>(zipper: &mut Z, out: &mut Vec<(Vec<u8>, Option<V>)>)
    where
        Z: ZipperMoving + ZipperValues<V>,
        V: Clone,
    {
        if zipper.path_exists() {
            out.push((zipper.path().to_vec(), zipper.val().cloned()));
        }

        let child_count = zipper.child_count();
        for child_idx in 0..child_count {
            assert!(zipper.descend_indexed_byte(child_idx));
            collect_observed_pathspace(zipper, out);
            assert!(zipper.ascend_byte());
        }
    }

    fn observed_pathspace<Z, V>(mut zipper: Z) -> Vec<(Vec<u8>, Option<V>)>
    where
        Z: ZipperMoving + ZipperValues<V>,
        V: Clone + Ord,
    {
        let mut out = Vec::new();
        collect_observed_pathspace(&mut zipper, &mut out);
        out.sort();
        out
    }

    fn assert_same_observed_pathspace<ExpectedZ, ActualZ, V>(expected: ExpectedZ, actual: ActualZ)
    where
        ExpectedZ: ZipperMoving + ZipperValues<V>,
        ActualZ: ZipperMoving + ZipperValues<V>,
        V: Clone + Ord + core::fmt::Debug,
    {
        assert_eq!(observed_pathspace(expected), observed_pathspace(actual));
    }

    #[test]
    fn restrict_zipper_matches_eager_restrict() {
        let data = PathMap::from_iter([
            (&b"aa"[..], 1usize),
            (&b"ab"[..], 2),
            (&b"b"[..], 3),
            (&b"ba"[..], 4),
            (&b"c"[..], 5),
        ]);
        let guard = PathMap::from_iter([(&b"a"[..], 0usize), (&b"ba"[..], 0)]);

        let eager = data.restrict(&guard);
        let lazy = RestrictZipper::new(data.read_zipper(), guard.read_zipper());

        assert_eq!(value_paths(lazy), value_paths(eager.read_zipper()));
    }

    #[test]
    fn dpa_restrict_zipper_matches_eager_restrict_pathspace() {
        let mut data = PathMap::<usize>::new();
        data.set_val_at(b"", 0);
        data.set_val_at(b"active/value", 1);
        data.create_path(b"active/dangling");
        data.set_val_at(b"active/deep/value", 2);
        data.create_path(b"active/deep/dangling");
        data.set_val_at(b"prefix/value", 3);
        data.create_path(b"prefix/dangling");
        data.set_val_at(b"shared/value", 4);
        data.create_path(b"shared/dangling");
        data.set_val_at(b"blocked/value", 5);
        data.create_path(b"blocked/dangling");

        let mut guard = PathMap::<usize>::new();
        guard.set_val_at(b"active", 10);
        guard.create_path(b"prefix");
        guard.set_val_at(b"prefix/value", 11);
        guard.create_path(b"shared");
        guard.set_val_at(b"shared/value", 12);
        guard.create_path(b"guard-only");

        let eager = data.restrict(&guard);
        let lazy = RestrictZipper::new(data.read_zipper(), guard.read_zipper());
        assert_same_observed_pathspace(eager.read_zipper(), lazy);

        assert_eq!(eager.get_val_at(b""), None);
        assert!(eager.path_exists_at(b"active/dangling"));
        assert_eq!(eager.get_val_at(b"active/deep/value"), Some(&2));
        assert_eq!(eager.get_val_at(b"prefix/value"), Some(&3));
        assert!(!eager.path_exists_at(b"prefix/dangling"));
        assert_eq!(eager.get_val_at(b"shared/value"), Some(&4));
        assert!(!eager.path_exists_at(b"shared/dangling"));
        assert!(!eager.path_exists_at(b"blocked/value"));

        let focus = b"prefix/";
        let focused_eager = eager.read_zipper_at_path(focus).make_map();
        let focused_lazy = RestrictZipper::new(
            data.read_zipper_at_path(focus),
            guard.read_zipper_at_path(focus),
        );
        assert_same_observed_pathspace(focused_eager.read_zipper(), focused_lazy);

        let mut root_guard = PathMap::<usize>::new();
        root_guard.set_val_at(b"", 99);
        let root_eager = data.restrict(&root_guard);
        let root_lazy = RestrictZipper::new(data.read_zipper(), root_guard.read_zipper());
        assert_same_observed_pathspace(root_eager.read_zipper(), root_lazy);
    }

    #[test]
    fn pathmap_restrict_by_paths_accepts_heterotyped_guard_values() {
        let mut data = PathMap::<usize>::new();
        data.set_val_at(b"active/value", 1);
        data.create_path(b"active/dangling");
        data.set_val_at(b"prefix/value", 2);
        data.create_path(b"prefix/dangling");
        data.set_val_at(b"blocked/value", 3);

        let mut same_typed_guard = PathMap::<usize>::new();
        same_typed_guard.set_val_at(b"active", 10);
        same_typed_guard.create_path(b"prefix");
        same_typed_guard.set_val_at(b"prefix/value", 11);

        let mut heterotyped_guard = PathMap::<&'static str>::new();
        heterotyped_guard.set_val_at(b"active", "guard-payload-is-ignored");
        heterotyped_guard.create_path(b"prefix");
        heterotyped_guard.set_val_at(b"prefix/value", "guard-payload-is-ignored");

        let expected = data.restrict(&same_typed_guard);
        let actual = data.restrict_by_paths(&heterotyped_guard);

        assert_same_observed_pathspace(expected.read_zipper(), actual.read_zipper());
        assert_eq!(actual.get_val_at(b"active/value"), Some(&1));
        assert_eq!(actual.get_val_at(b"prefix/value"), Some(&2));
        assert!(!actual.path_exists_at(b"prefix/dangling"));
        assert!(!actual.path_exists_at(b"blocked/value"));

        let mut root_guard = PathMap::<&'static str>::new();
        root_guard.set_val_at(b"", "everything");
        let root_actual = data.restrict_by_paths(&root_guard);
        assert_same_observed_pathspace(data.read_zipper(), root_actual.read_zipper());
    }

    #[test]
    fn pathmap_restrict_by_paths_matches_lazy_restrict_for_many_compressed_groups() {
        fn group_path(prefix: &[u8], n: usize, suffix: &[u8]) -> Vec<u8> {
            let mut path = Vec::new();
            path.extend_from_slice(prefix);
            path.extend_from_slice(format!("{n:08x}").as_bytes());
            path.extend_from_slice(suffix);
            path
        }

        let mut data = PathMap::<usize>::new();
        let mut guard = PathMap::<&'static str>::new();
        for i in 0..32 {
            data.set_val_at(group_path(b"active:", i, b":leaf/a"), i);
            data.set_val_at(group_path(b"active:", i, b":leaf/b"), i);
            data.create_path(group_path(b"active:", i, b":dangling"));
            data.set_val_at(group_path(b"shared:", i, b":value"), i);
            data.create_path(group_path(b"shared:", i, b":dangling"));
            data.set_val_at(group_path(b"blocked:", i, b":value"), i);

            guard.set_val_at(group_path(b"active:", i, b""), "guard");
            guard.create_path(group_path(b"shared:", i, b""));
            guard.set_val_at(group_path(b"shared:", i, b":value"), "guard");
        }

        let actual = data.restrict_by_paths(&guard);
        let expected = RestrictZipper::new(data.read_zipper(), guard.read_zipper())
            .try_make_map()
            .unwrap();

        assert_same_observed_pathspace(expected.read_zipper(), actual.read_zipper());
        assert_eq!(actual.val_count(), 96);
        assert!(!actual.path_exists_at(b"blocked:00000000:value"));
        assert!(!actual.path_exists_at(b"shared:00000000:dangling"));
        assert!(actual.path_exists_at(b"active:00000000:dangling"));
    }

    #[test]
    fn restrict_zipper_materializes_eager_restrict_pathspace() {
        let mut data = PathMap::<usize>::new();
        data.set_val_at(b"active/value", 1);
        data.create_path(b"active/dangling");
        data.set_val_at(b"prefix/value", 2);
        data.create_path(b"prefix/dangling");
        data.set_val_at(b"blocked/value", 3);

        let mut guard = PathMap::<usize>::new();
        guard.set_val_at(b"active", 0);
        guard.create_path(b"prefix");
        guard.set_val_at(b"prefix/value", 0);

        let eager = data.restrict(&guard);
        let lazy = RestrictZipper::new(data.read_zipper(), guard.read_zipper());
        let materialized = lazy.try_make_map().unwrap();

        assert!(!lazy.native_subtries());
        assert_same_observed_pathspace(eager.read_zipper(), materialized.read_zipper());
        assert!(materialized.path_exists_at(b"active/dangling"));
        assert!(!materialized.path_exists_at(b"prefix/dangling"));
        assert!(!materialized.path_exists_at(b"blocked/value"));
    }

    #[test]
    fn restrict_zipper_materializes_current_focus_relative_pathspace() {
        let mut data = PathMap::<usize>::new();
        data.set_val_at(b"active/value", 1);
        data.create_path(b"active/dangling");
        data.set_val_at(b"prefix/value", 2);
        data.create_path(b"prefix/dangling");
        data.set_val_at(b"blocked/value", 3);

        let mut guard = PathMap::<usize>::new();
        guard.set_val_at(b"active", 0);
        guard.create_path(b"prefix");
        guard.set_val_at(b"prefix/value", 0);

        let eager = data.restrict(&guard);
        let focus = b"prefix/";
        let focused_eager = eager.read_zipper_at_path(focus).make_map();

        let mut lazy = RestrictZipper::new(data.read_zipper(), guard.read_zipper());
        lazy.descend_to(focus);
        let materialized = lazy.try_make_map().unwrap();

        assert_same_observed_pathspace(focused_eager.read_zipper(), materialized.read_zipper());
        assert_eq!(materialized.get_val_at(b"value"), Some(&2));
        assert!(!materialized.path_exists_at(b"prefix/value"));
        assert!(!materialized.path_exists_at(b"dangling"));
    }

    #[cfg(feature = "counters")]
    #[test]
    fn materialize_zipper_counts_observer_output() {
        use crate::counters::{reset_virtual_zipper_counters, virtual_zipper_counters};

        let _guard = crate::counters::counter_test_guard();
        let mut data = PathMap::<usize>::new();
        data.set_val_at(b"a", 1);
        data.create_path(b"b");

        let mut guard = PathMap::<usize>::new();
        guard.set_val_at(b"", 0);
        let lazy = RestrictZipper::new(data.read_zipper(), guard.read_zipper());

        reset_virtual_zipper_counters();
        let materialized = lazy.try_make_map().unwrap();

        assert_eq!(materialized.get_val_at(b"a"), Some(&1));
        assert!(materialized.path_exists_at(b"b"));

        let counters = virtual_zipper_counters();
        assert_eq!(counters.observer_materializations, 1);
        assert_eq!(counters.observer_materialized_paths, 2);
        assert_eq!(counters.observer_materialized_values, 1);
    }

    #[test]
    fn restrict_zipper_root_guard_value_validates_all_paths() {
        let data = PathMap::from_iter([(&b"a"[..], 1usize), (&b"b"[..], 2)]);
        let guard = PathMap::from_iter([(&b""[..], ())]);
        let lazy = RestrictZipper::new(data.read_zipper(), guard.read_zipper());

        assert_eq!(value_paths(lazy), vec![b"a".to_vec(), b"b".to_vec()]);
    }

    #[cfg(feature = "counters")]
    #[test]
    fn restrict_zipper_counts_value_and_child_mask_filters() {
        use crate::counters::{reset_virtual_zipper_counters, virtual_zipper_counters};
        use crate::zipper::{Zipper, ZipperValues};

        let _guard = crate::counters::counter_test_guard();
        let data = PathMap::from_iter([(&b"a"[..], 1usize)]);
        let guard = PathMap::from_iter([(&b"a"[..], ())]);
        let zipper = RestrictZipper::new(data.read_zipper(), guard.read_zipper());

        reset_virtual_zipper_counters();
        assert_eq!(zipper.child_count(), 1);
        assert!(!zipper.val().is_some());

        let counters = virtual_zipper_counters();
        assert_eq!(counters.restrict_child_mask_filters, 1);
        assert_eq!(counters.restrict_value_filters, 1);
    }
}
