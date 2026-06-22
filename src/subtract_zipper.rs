use crate::PathMap;
use crate::alloc::{GlobalAlloc, global_alloc};
use crate::ring::{AlgebraicResult, DistributiveLattice, DistributiveLatticeRef, SELF_IDENT};
use crate::utils::{BitMask, ByteMask};
use crate::zipper::*;

#[derive(Clone)]
enum SubtractValue<V> {
    None,
    LeftIdentity,
    Owned(V),
}

/// A read-only virtual zipper over the pathspace difference of two zippers.
///
/// The left and right zippers are kept at the same relative path. Values are
/// subtracted with [`DistributiveLattice::psubtract`]. If value subtraction
/// synthesizes a new value, the zipper caches that value at the current focus so
/// it can still satisfy the borrowed [`ZipperValues`] interface.
#[derive(Clone)]
pub struct SubtractZipper<LeftZ, RightZ, V> {
    left: LeftZ,
    right: RightZ,
    value: SubtractValue<V>,
}

impl<LeftZ, RightZ, V> SubtractZipper<LeftZ, RightZ, V>
where
    LeftZ: ZipperMoving + ZipperValues<V>,
    RightZ: ZipperMoving + ZipperValues<V>,
    V: DistributiveLattice + Clone,
{
    /// Create a lazy difference zipper from two aligned source zippers.
    pub fn new(mut left: LeftZ, mut right: RightZ) -> Self {
        left.reset();
        right.reset();
        let mut this = Self {
            left,
            right,
            value: SubtractValue::None,
        };
        this.refresh_value();
        this
    }

    fn refresh_value(&mut self) {
        self.value = match self.left.val().psubtract(&self.right.val()) {
            AlgebraicResult::None => SubtractValue::None,
            AlgebraicResult::Identity(mask) if mask & SELF_IDENT != 0 => {
                SubtractValue::LeftIdentity
            }
            AlgebraicResult::Identity(_) => SubtractValue::None,
            AlgebraicResult::Element(Some(value)) => SubtractValue::Owned(value),
            AlgebraicResult::Element(None) => SubtractValue::None,
        };
    }

    fn residual_exists(&self) -> bool
    where
        LeftZ: Clone,
        RightZ: Clone,
    {
        self.path_exists()
    }

    fn right_residual_is_empty(&self) -> bool {
        !self.right.path_exists()
            && self.right.val().is_none()
            && self.right.child_mask().is_empty_mask()
    }
}

impl<LeftZ, RightZ, V> ZipperValues<V> for SubtractZipper<LeftZ, RightZ, V>
where
    LeftZ: ZipperValues<V>,
    V: Clone,
{
    fn val(&self) -> Option<&V> {
        #[cfg(feature = "counters")]
        crate::counters::record_subtract_value_map();
        match &self.value {
            SubtractValue::None => None,
            SubtractValue::LeftIdentity => self.left.val(),
            SubtractValue::Owned(value) => Some(value),
        }
    }
}

impl<LeftZ, RightZ, V> Zipper for SubtractZipper<LeftZ, RightZ, V>
where
    LeftZ: ZipperMoving + ZipperValues<V> + Clone,
    RightZ: ZipperMoving + ZipperValues<V> + Clone,
    V: DistributiveLattice + Clone,
{
    fn path_exists(&self) -> bool {
        if !self.left.path_exists() {
            return false;
        }
        if self.right_residual_is_empty() || !self.right.path_exists() {
            return true;
        }
        self.val().is_some() || !self.child_mask().is_empty_mask()
    }

    fn is_val(&self) -> bool {
        self.val().is_some()
    }

    fn child_count(&self) -> usize {
        self.child_mask().count_bits()
    }

    fn child_mask(&self) -> ByteMask {
        let left_mask = self.left.child_mask();
        if left_mask.is_empty_mask() || self.right_residual_is_empty() {
            return left_mask;
        }

        let mut out = ByteMask::EMPTY;
        for child_idx in 0..left_mask.count_bits() {
            let Some(byte) = left_mask.indexed_bit::<true>(child_idx) else {
                continue;
            };
            #[cfg(feature = "counters")]
            crate::counters::record_subtract_child_probe();
            let mut child = self.clone();
            child.descend_to_byte(byte);
            if child.residual_exists() {
                out.set_bit(byte);
            }
        }
        out
    }
}

impl<LeftZ, RightZ, V> ZipperMoving for SubtractZipper<LeftZ, RightZ, V>
where
    LeftZ: ZipperMoving + ZipperValues<V> + Clone,
    RightZ: ZipperMoving + ZipperValues<V> + Clone,
    V: DistributiveLattice + Clone,
{
    fn at_root(&self) -> bool {
        self.left.at_root()
    }

    fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
        self.refresh_value();
    }

    fn path(&self) -> &[u8] {
        self.left.path()
    }

    fn val_count(&self) -> usize {
        let mut cursor = self.clone();
        cursor.reset();
        let mut count = cursor.is_val() as usize;
        while cursor.to_next_val() {
            count += 1;
        }
        count
    }

    fn descend_to<K: AsRef<[u8]>>(&mut self, path: K) {
        for &byte in path.as_ref() {
            self.descend_to_byte(byte);
        }
    }

    fn descend_to_byte(&mut self, k: u8) {
        self.left.descend_to_byte(k);
        self.right.descend_to_byte(k);
        self.refresh_value();
    }

    fn descend_indexed_byte(&mut self, idx: usize) -> bool {
        let Some(byte) = self.child_mask().indexed_bit::<true>(idx) else {
            return false;
        };
        self.descend_to_byte(byte);
        true
    }

    fn ascend(&mut self, steps: usize) -> bool {
        let left_ok = self.left.ascend(steps);
        let right_ok = self.right.ascend(steps);
        self.refresh_value();
        left_ok && right_ok
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

impl<LeftZ, RightZ, V> ZipperIteration for SubtractZipper<LeftZ, RightZ, V>
where
    LeftZ: ZipperMoving + ZipperValues<V> + Clone,
    RightZ: ZipperMoving + ZipperValues<V> + Clone,
    V: DistributiveLattice + Clone,
{
}

impl<LeftZ, RightZ, V> ZipperAbsolutePath for SubtractZipper<LeftZ, RightZ, V>
where
    LeftZ: ZipperAbsolutePath + ZipperValues<V> + Clone,
    RightZ: ZipperMoving + ZipperValues<V> + Clone,
    V: DistributiveLattice + Clone,
{
    fn origin_path(&self) -> &[u8] {
        self.left.origin_path()
    }

    fn root_prefix_path(&self) -> &[u8] {
        self.left.root_prefix_path()
    }
}

impl<LeftZ, RightZ, V> ZipperSubtries<V, GlobalAlloc> for SubtractZipper<LeftZ, RightZ, V>
where
    V: DistributiveLattice + Clone + Send + Sync + Unpin,
    LeftZ: ZipperMoving + ZipperValues<V> + Clone,
    RightZ: ZipperMoving + ZipperValues<V> + Clone,
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
    impl<LeftZ, RightZ, V> core::fmt::Debug for SubtractZipper<LeftZ, RightZ, V>
);

#[cfg(test)]
mod tests {
    use super::SubtractZipper;
    use crate::PathMap;
    use crate::ring::{AlgebraicResult, DistributiveLattice, Lattice, SELF_IDENT};
    use crate::zipper::{
        Zipper, ZipperInfallibleSubtries, ZipperIteration, ZipperMoving, ZipperSubtries,
        ZipperValues,
    };

    #[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
    struct DiffValue(u8);

    impl Lattice for DiffValue {
        fn pjoin(&self, other: &Self) -> AlgebraicResult<Self> {
            if self == other {
                AlgebraicResult::Identity(SELF_IDENT)
            } else {
                AlgebraicResult::Element(DiffValue(self.0.max(other.0)))
            }
        }

        fn pmeet(&self, other: &Self) -> AlgebraicResult<Self> {
            if self == other {
                AlgebraicResult::Identity(SELF_IDENT)
            } else {
                AlgebraicResult::None
            }
        }
    }

    impl DistributiveLattice for DiffValue {
        fn psubtract(&self, other: &Self) -> AlgebraicResult<Self> {
            match self.0.cmp(&other.0) {
                core::cmp::Ordering::Equal => AlgebraicResult::None,
                core::cmp::Ordering::Greater => {
                    AlgebraicResult::Element(DiffValue(self.0 - other.0))
                }
                core::cmp::Ordering::Less => AlgebraicResult::Identity(SELF_IDENT),
            }
        }
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
    fn dpa_subtract_zipper_matches_eager_subtract_pathspace() {
        let mut left = PathMap::<DiffValue>::new();
        left.set_val_at(b"", DiffValue(10));
        left.set_val_at(b"left-only/value", DiffValue(7));
        left.create_path(b"dangling/preserved");
        left.create_path(b"dangling/left-only");
        left.create_path(b"dangling/shared");
        left.set_val_at(b"shared/remove", DiffValue(4));
        left.set_val_at(b"shared/owned", DiffValue(9));
        left.set_val_at(b"shared/identity", DiffValue(2));
        left.set_val_at(b"branch/mixed/a", DiffValue(1));
        left.set_val_at(b"branch/mixed/c", DiffValue(8));

        let mut right = PathMap::<DiffValue>::new();
        right.set_val_at(b"", DiffValue(10));
        right.create_path(b"dangling/shared");
        right.create_path(b"dangling/left-only/child");
        right.set_val_at(b"shared/remove", DiffValue(4));
        right.set_val_at(b"shared/owned", DiffValue(3));
        right.set_val_at(b"shared/identity", DiffValue(5));
        right.set_val_at(b"branch/mixed/a", DiffValue(1));
        right.set_val_at(b"branch/mixed/b", DiffValue(9));
        right.set_val_at(b"right-only/value", DiffValue(1));

        let eager = left.subtract(&right);
        let lazy = SubtractZipper::new(left.read_zipper(), right.read_zipper());
        assert_same_observed_pathspace(eager.read_zipper(), lazy);

        assert_eq!(eager.get_val_at(b""), None);
        assert_eq!(eager.get_val_at(b"left-only/value"), Some(&DiffValue(7)));
        assert!(eager.path_exists_at(b"dangling/preserved"));
        assert!(!eager.path_exists_at(b"dangling/left-only"));
        assert!(!eager.path_exists_at(b"dangling/shared"));
        assert!(!eager.path_exists_at(b"shared/remove"));
        assert_eq!(eager.get_val_at(b"shared/owned"), Some(&DiffValue(6)));
        assert_eq!(eager.get_val_at(b"shared/identity"), Some(&DiffValue(2)));
        assert!(!eager.path_exists_at(b"branch/mixed/a"));
        assert_eq!(eager.get_val_at(b"branch/mixed/c"), Some(&DiffValue(8)));
        assert!(!eager.path_exists_at(b"right-only/value"));

        let focus = b"shared/";
        let focused_eager = eager.read_zipper_at_path(focus);
        let focused_lazy = SubtractZipper::new(
            left.read_zipper_at_path(focus),
            right.read_zipper_at_path(focus),
        );
        assert_same_observed_pathspace(focused_eager, focused_lazy);
    }

    #[test]
    fn subtract_zipper_materializes_eager_subtract_pathspace() {
        let mut left = PathMap::<DiffValue>::new();
        left.set_val_at(b"left-only/value", DiffValue(7));
        left.create_path(b"dangling/preserved");
        left.create_path(b"dangling/shared");
        left.set_val_at(b"shared/remove", DiffValue(4));
        left.set_val_at(b"shared/owned", DiffValue(9));
        left.set_val_at(b"shared/identity", DiffValue(2));

        let mut right = PathMap::<DiffValue>::new();
        right.create_path(b"dangling/shared");
        right.set_val_at(b"shared/remove", DiffValue(4));
        right.set_val_at(b"shared/owned", DiffValue(3));
        right.set_val_at(b"shared/identity", DiffValue(5));
        right.set_val_at(b"right-only/value", DiffValue(1));

        let eager = left.subtract(&right);
        let lazy = SubtractZipper::new(left.read_zipper(), right.read_zipper());
        let materialized = lazy.try_make_map().unwrap();

        assert!(!lazy.native_subtries());
        assert_same_observed_pathspace(eager.read_zipper(), materialized.read_zipper());
        assert!(materialized.path_exists_at(b"dangling/preserved"));
        assert!(!materialized.path_exists_at(b"dangling/shared"));
        assert_eq!(
            materialized.get_val_at(b"shared/owned"),
            Some(&DiffValue(6))
        );
    }

    #[test]
    fn subtract_zipper_materializes_current_focus_relative_pathspace() {
        let mut left = PathMap::<DiffValue>::new();
        left.set_val_at(b"left-only/value", DiffValue(7));
        left.create_path(b"dangling/preserved");
        left.create_path(b"dangling/shared");
        left.set_val_at(b"shared/remove", DiffValue(4));
        left.set_val_at(b"shared/owned", DiffValue(9));
        left.set_val_at(b"shared/identity", DiffValue(2));

        let mut right = PathMap::<DiffValue>::new();
        right.create_path(b"dangling/shared");
        right.set_val_at(b"shared/remove", DiffValue(4));
        right.set_val_at(b"shared/owned", DiffValue(3));
        right.set_val_at(b"shared/identity", DiffValue(5));
        right.set_val_at(b"right-only/value", DiffValue(1));

        let eager = left.subtract(&right);
        let focus = b"shared/";
        let focused_eager = eager.read_zipper_at_path(focus).make_map();

        let mut lazy = SubtractZipper::new(left.read_zipper(), right.read_zipper());
        lazy.descend_to(focus);
        let materialized = lazy.try_make_map().unwrap();

        assert_same_observed_pathspace(focused_eager.read_zipper(), materialized.read_zipper());
        assert_eq!(materialized.get_val_at(b"owned"), Some(&DiffValue(6)));
        assert_eq!(materialized.get_val_at(b"identity"), Some(&DiffValue(2)));
        assert!(!materialized.path_exists_at(b"shared/owned"));
        assert!(!materialized.path_exists_at(b"remove"));
    }

    #[test]
    fn subtract_zipper_iterates_values() {
        let left = PathMap::from_iter([(&b"a"[..], true), (&b"b"[..], false), (&b"c"[..], true)]);
        let right = PathMap::from_iter([(&b"a"[..], true), (&b"b"[..], true)]);

        let mut lazy = SubtractZipper::new(left.read_zipper(), right.read_zipper());
        let mut paths = Vec::new();
        if lazy.is_val() {
            paths.push(lazy.path().to_vec());
        }
        while lazy.to_next_val() {
            paths.push(lazy.path().to_vec());
        }

        assert_eq!(paths, vec![b"b".to_vec(), b"c".to_vec()]);
    }

    #[cfg(feature = "counters")]
    #[test]
    fn subtract_zipper_counts_value_and_child_observations() {
        use crate::counters::{reset_virtual_zipper_counters, virtual_zipper_counters};

        let _guard = crate::counters::counter_test_guard();
        let left = PathMap::from_iter([(&b"a"[..], ()), (&b"b"[..], ())]);
        let right = PathMap::from_iter([(&b"a"[..], ())]);
        let zipper = SubtractZipper::new(left.read_zipper(), right.read_zipper());

        reset_virtual_zipper_counters();
        assert_eq!(zipper.val(), None);
        assert_eq!(zipper.child_count(), 1);

        let counters = virtual_zipper_counters();
        assert_eq!(counters.subtract_value_maps, 2);
        assert_eq!(counters.subtract_child_probes, 2);
    }
}
