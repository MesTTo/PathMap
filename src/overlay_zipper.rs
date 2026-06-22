//! Overlay zippers are virtual unions, not lattice joins.
//!
//! The value mapping function returns borrowed values from the source zippers.
//! That keeps traversal allocation-free, but it also means this type cannot
//! synthesize a new joined value. A future join zipper would need an owned value
//! policy and a place to store created values, so it should be designed with the
//! algebra policy API rather than folded into this observer.

use crate::PathMap;
use crate::alloc::{GlobalAlloc, global_alloc};
use crate::utils::{BitMask, ByteMask};
use crate::zipper::{
    TrieRef, Zipper, ZipperIteration, ZipperMoving, ZipperSubtries, ZipperValues,
    materialize_zipper,
};
use fast_slice_utils::find_prefix_overlap;

/// Zipper that traverses a virtual trie formed by fusing the tries of two other zippers
pub struct OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    a: AZipper,
    b: BZipper,
    mapping: Mapping,
    _marker: core::marker::PhantomData<(AV, BV, OutV)>,
}

fn identity_ref<'a, V>(a_val: Option<&'a V>, b_val: Option<&'a V>) -> Option<&'a V> {
    a_val.or(b_val)
}

impl<V, AZipper, BZipper>
    OverlayZipper<
        V,
        V,
        V,
        AZipper,
        BZipper,
        for<'a> fn(Option<&'a V>, Option<&'a V>) -> Option<&'a V>,
    >
where
    AZipper: ZipperMoving,
    BZipper: ZipperMoving,
{
    /// Create a new `OverlayZipper` from two other zippers, using a default value mapping function
    ///
    /// In cases where both source zippers supply a value, the value from `AZipper` will be supplied by
    /// the `OverlayZipper`.
    pub fn new(a: AZipper, b: BZipper) -> Self {
        Self::with_mapping(a, b, identity_ref)
    }
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    AZipper: ZipperMoving,
    BZipper: ZipperMoving,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    /// Create a new `OverlayZipper` from two other zippers, using a the supplied value mapping function
    pub fn with_mapping(mut a: AZipper, mut b: BZipper, mapping: Mapping) -> Self {
        a.reset();
        b.reset();
        Self {
            a,
            b,
            mapping,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    AZipper: ZipperMoving + ZipperValues<AV> + Clone,
    BZipper: ZipperMoving + ZipperValues<BV> + Clone,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    fn to_sibling(&mut self, next: bool) -> bool {
        let path = self.path();
        let Some(&last) = path.last() else {
            return false;
        };
        self.ascend(1);
        let child_mask = self.child_mask();
        let maybe_child = if next {
            child_mask.next_bit(last)
        } else {
            child_mask.prev_bit(last)
        };
        let Some(child) = maybe_child else {
            self.descend_to_byte(last);
            return false;
        };
        self.descend_to_byte(child);
        true
    }

    fn realign_after_independent_ascend(&mut self) {
        let depth_a = self.a.path().len();
        let depth_b = self.b.path().len();
        if depth_b > depth_a {
            self.a.descend_to(&self.b.path()[depth_a..]);
        } else if depth_a > depth_b {
            self.b.descend_to(&self.a.path()[depth_b..]);
        }
    }
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> ZipperValues<OutV>
    for OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    AZipper: ZipperValues<AV>,
    BZipper: ZipperValues<BV>,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    fn val(&self) -> Option<&OutV> {
        #[cfg(feature = "counters")]
        crate::counters::record_overlay_value_map();
        (self.mapping)(self.a.val(), self.b.val())
    }
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> Zipper
    for OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    AZipper: Zipper + ZipperValues<AV>,
    BZipper: Zipper + ZipperValues<BV>,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    fn path_exists(&self) -> bool {
        self.a.path_exists() || self.b.path_exists()
    }
    fn is_val(&self) -> bool {
        //NOTE: the mapping function has the ability to nullify the value, so we need ZipperValues to implement this correctly
        // self.a.is_val() || self.b.is_val()
        self.val().is_some()
    }
    fn child_count(&self) -> usize {
        self.child_mask().count_bits()
    }
    fn child_mask(&self) -> ByteMask {
        #[cfg(feature = "counters")]
        crate::counters::record_overlay_child_mask_union();
        self.a.child_mask() | self.b.child_mask()
    }
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> ZipperMoving
    for OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    AZipper: ZipperMoving + ZipperValues<AV> + Clone,
    BZipper: ZipperMoving + ZipperValues<BV> + Clone,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    fn at_root(&self) -> bool {
        self.a.at_root() || self.b.at_root()
    }

    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
    }

    fn path(&self) -> &[u8] {
        self.a.path()
    }

    fn val_count(&self) -> usize {
        let mut cursor = OverlayZipper::with_mapping(self.a.clone(), self.b.clone(), &self.mapping);
        cursor.reset();
        let mut count = cursor.is_val() as usize;
        while cursor.to_next_val() {
            count += 1;
        }
        count
    }

    fn descend_to<P: AsRef<[u8]>>(&mut self, path: P) {
        let path = path.as_ref();
        self.a.descend_to(path);
        self.b.descend_to(path);
    }

    fn descend_to_existing<P: AsRef<[u8]>>(&mut self, path: P) -> usize {
        let path = path.as_ref();
        let depth_a = self.a.descend_to_existing(path);
        let depth_b = self.b.descend_to_existing(path);
        if depth_a > depth_b {
            self.b.descend_to(&path[depth_b..depth_a]);
            depth_a
        } else if depth_b > depth_a {
            self.a.descend_to(&path[depth_a..depth_b]);
            depth_b
        } else {
            depth_a
        }
    }

    fn descend_to_val<K: AsRef<[u8]>>(&mut self, path: K) -> usize {
        let path = path.as_ref();
        let depth_a = self.a.descend_to_val(path);
        let depth_o = self.b.descend_to_val(path);
        if depth_a < depth_o {
            if self.a.is_val() {
                self.b.ascend(depth_o - depth_a);
                depth_a
            } else {
                self.a.descend_to(&path[depth_a..depth_o]);
                depth_o
            }
        } else if depth_o < depth_a {
            if self.b.is_val() {
                self.a.ascend(depth_a - depth_o);
                depth_o
            } else {
                self.a.descend_to(&path[depth_o..depth_a]);
                depth_a
            }
        } else {
            depth_a
        }
    }

    fn descend_to_byte(&mut self, k: u8) {
        self.a.descend_to(&[k]);
        self.b.descend_to(&[k]);
    }

    fn descend_first_byte(&mut self) -> bool {
        self.descend_indexed_byte(0)
    }

    fn descend_indexed_byte(&mut self, idx: usize) -> bool {
        let child_mask = self.child_mask();
        let Some(byte) = child_mask.indexed_bit::<true>(idx) else {
            return false;
        };
        self.descend_to_byte(byte);
        true
    }

    fn descend_until(&mut self) -> bool {
        let start_depth = self.a.path().len();
        let desc_a = self.a.descend_until();
        let desc_b = self.b.descend_until();
        let path_a = &self.a.path()[start_depth..];
        let path_b = &self.b.path()[start_depth..];
        if !desc_a && !desc_b {
            return false;
        }
        if !desc_a && desc_b {
            if self.a.child_count() == 0 {
                self.a.descend_to(path_b);
                return true;
            } else {
                self.b.ascend(self.b.path().len() - start_depth);
                return false;
            }
        }
        if desc_a && !desc_b {
            if self.b.child_count() == 0 {
                self.b.descend_to(path_a);
                return true;
            } else {
                self.a.ascend(self.a.path().len() - start_depth);
                return false;
            }
        }
        let overlap = find_prefix_overlap(path_a, path_b);
        if path_a.len() > overlap {
            self.a.ascend(path_a.len() - overlap);
        }
        if path_b.len() > overlap {
            self.b.ascend(path_b.len() - overlap);
        }
        overlap > 0
    }

    fn ascend(&mut self, steps: usize) -> bool {
        self.a.ascend(steps) | self.b.ascend(steps)
    }

    fn ascend_byte(&mut self) -> bool {
        self.ascend(1)
    }

    fn ascend_until(&mut self) -> bool {
        debug_assert_eq!(self.a.path(), self.b.path());
        let asc_a = self.a.ascend_until();
        let asc_b = self.b.ascend_until();
        if !(asc_b || asc_a) {
            return false;
        }
        self.realign_after_independent_ascend();
        true
    }

    fn ascend_until_branch(&mut self) -> bool {
        let asc_a = self.a.ascend_until_branch();
        let asc_b = self.b.ascend_until_branch();
        self.realign_after_independent_ascend();
        asc_a || asc_b
    }

    fn to_next_sibling_byte(&mut self) -> bool {
        self.to_sibling(true)
    }

    fn to_prev_sibling_byte(&mut self) -> bool {
        self.to_sibling(false)
    }
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> ZipperIteration
    for OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    AZipper: ZipperMoving + ZipperValues<AV> + Clone,
    BZipper: ZipperMoving + ZipperValues<BV> + Clone,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
}

impl<AV, BV, OutV, AZipper, BZipper, Mapping> ZipperSubtries<OutV, GlobalAlloc>
    for OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
where
    OutV: Clone + Send + Sync + Unpin,
    AZipper: ZipperMoving + ZipperValues<AV> + Clone,
    BZipper: ZipperMoving + ZipperValues<BV> + Clone,
    Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
{
    fn native_subtries(&self) -> bool {
        false
    }

    fn try_make_map(&self) -> Option<PathMap<OutV, GlobalAlloc>> {
        let zipper = OverlayZipper {
            a: self.a.clone(),
            b: self.b.clone(),
            mapping: &self.mapping,
            _marker: core::marker::PhantomData,
        };
        Some(materialize_zipper(zipper, global_alloc()))
    }

    fn trie_ref(&self) -> Option<TrieRef<'_, OutV, GlobalAlloc>> {
        None
    }

    fn alloc(&self) -> GlobalAlloc {
        global_alloc()
    }
}

crate::impl_name_only_debug!(
    impl<AV, BV, OutV, AZipper, BZipper, Mapping> core::fmt::Debug for OverlayZipper<AV, BV, OutV, AZipper, BZipper, Mapping>
        where
        Mapping: for<'a> Fn(Option<&'a AV>, Option<&'a BV>) -> Option<&'a OutV>,
);

#[cfg(test)]
mod tests {
    use super::OverlayZipper;
    use crate::alloc::GlobalAlloc;
    use crate::{
        PathMap,
        zipper::{
            ReadZipperUntracked,
            ZipperInfallibleSubtries,
            ZipperMoving,
            ZipperSubtries,
            ZipperValues,
            zipper_iteration_tests,
            zipper_moving_tests,
            // ZipperIteration,
            // ZipperMoving,
            // ZipperValues
        },
    };

    fn collect_observed_pathspace<Z, V>(zipper: &mut Z, paths: &mut Vec<(Vec<u8>, Option<V>)>)
    where
        Z: ZipperMoving + ZipperValues<V>,
        V: Clone,
    {
        if zipper.path_exists() {
            paths.push((zipper.path().to_vec(), zipper.val().cloned()));
        }

        let child_count = zipper.child_count();
        for child_idx in 0..child_count {
            assert!(zipper.descend_indexed_byte(child_idx));
            collect_observed_pathspace(zipper, paths);
            assert!(zipper.ascend_byte());
        }
    }

    fn observed_pathspace<Z, V>(zipper: &mut Z) -> Vec<(Vec<u8>, Option<V>)>
    where
        Z: ZipperMoving + ZipperValues<V>,
        V: Clone + Ord,
    {
        let mut paths = Vec::new();
        collect_observed_pathspace(zipper, &mut paths);
        paths.sort();
        paths
    }

    fn assert_same_observed_pathspace<Z, V>(expected: &PathMap<V>, actual: &mut Z)
    where
        Z: ZipperMoving + ZipperValues<V>,
        V: Clone + Send + Sync + Unpin + Ord + core::fmt::Debug,
    {
        let mut expected_zipper = expected.read_zipper();
        assert_eq!(
            observed_pathspace(actual),
            observed_pathspace(&mut expected_zipper)
        );
    }

    // #[test]
    // fn overlay_preserves_keys() {
    //     // base: ACT { "aaa" -> 1, "bbb" -> 3 }
    //     // overlay: PathMap { "aaa" -> 2, "ccc" -> 4 }
    //     // result: Overlay { "aaa" -> 2, "bbb" -> 3, "ccc" -> 4 }
    //     let keys: &[&[u8]] = &[b"a", b"aa", b"ab", b"b", b"ba", b"bb"];
    //     let trie_a = keys[..3].into_iter().map(|k| (k, ())).collect::<PathMap<()>>();
    //     let trie_b = keys[3..].into_iter().map(|k| (k, ())).collect::<PathMap<()>>();
    //     let mut oz = OverlayZipper::new(trie_a.read_zipper(), trie_b.read_zipper());
    //     assert_eq!(oz.keys(), keys);
    // }

    type Mapping = for<'a> fn(Option<&'a ()>, Option<&'a ()>) -> Option<&'a ()>;
    type OZ<'a, V, A = GlobalAlloc> = OverlayZipper<
        V,
        V,
        V,
        ReadZipperUntracked<'a, 'static, V, A>,
        ReadZipperUntracked<'a, 'static, V, A>,
        Mapping,
    >;

    fn split_overlay_maps(keys: &[&[u8]]) -> (PathMap<()>, PathMap<()>) {
        let cutoff = keys.len() / 3 * 2;
        let a = keys[..cutoff]
            .into_iter()
            .map(|k| (k, ()))
            .collect::<PathMap<()>>();
        let b = keys[cutoff..]
            .into_iter()
            .map(|k| (k, ()))
            .collect::<PathMap<()>>();
        (a, b)
    }

    fn overlay_from_maps<'a>(
        trie: &'a mut (PathMap<()>, PathMap<()>),
        path: &[u8],
    ) -> OZ<'a, ()> {
        OverlayZipper::new(
            trie.0.read_zipper_at_path(path),
            trie.1.read_zipper_at_path(path),
        )
    }

    #[cfg(feature = "counters")]
    #[test]
    fn overlay_counter_records_virtual_observations() {
        use crate::counters::{reset_virtual_zipper_counters, virtual_zipper_counters};

        let _guard = crate::counters::counter_test_guard();
        let mut a = PathMap::<()>::new();
        a.insert(b"a", ());
        let mut b = PathMap::<()>::new();
        b.insert(b"b", ());

        let overlay = OverlayZipper::new(a.read_zipper(), b.read_zipper());

        reset_virtual_zipper_counters();
        assert_eq!(crate::zipper::Zipper::child_count(&overlay), 2);
        let counters = virtual_zipper_counters();
        assert_eq!(counters.overlay_child_mask_unions, 1);
        assert_eq!(counters.overlay_value_maps, 0);

        assert!(!crate::zipper::Zipper::is_val(&overlay));
        let counters = virtual_zipper_counters();
        assert_eq!(counters.overlay_child_mask_unions, 1);
        assert_eq!(counters.overlay_value_maps, 1);
    }

    #[test]
    fn overlay_val_count_counts_virtual_union_once() {
        let mut a = PathMap::<()>::new();
        a.insert(b"", ());
        a.insert(b"a", ());
        a.insert(b"aa", ());
        a.insert(b"b", ());

        let mut b = PathMap::<()>::new();
        b.insert(b"", ());
        b.insert(b"a", ());
        b.insert(b"ab", ());
        b.insert(b"c", ());

        let overlay = OverlayZipper::new(a.read_zipper(), b.read_zipper());

        assert_eq!(overlay.val_count(), 6);
    }

    #[test]
    fn overlay_val_count_respects_current_focus() {
        let mut a = PathMap::<()>::new();
        a.insert(b"a", ());
        a.insert(b"aa", ());
        a.insert(b"ax", ());
        a.insert(b"b", ());

        let mut b = PathMap::<()>::new();
        b.insert(b"a", ());
        b.insert(b"ab", ());
        b.insert(b"ay", ());
        b.insert(b"c", ());

        let overlay = OverlayZipper::new(a.read_zipper_at_path(b"a"), b.read_zipper_at_path(b"a"));

        assert_eq!(overlay.val_count(), 5);
    }

    #[test]
    fn overlay_val_count_uses_the_value_mapping() {
        fn right_only<'a>(_: Option<&'a u8>, b_val: Option<&'a u8>) -> Option<&'a u8> {
            b_val
        }

        let mut a = PathMap::<u8>::new();
        a.insert(b"", 1);
        a.insert(b"a", 1);
        a.insert(b"b", 1);

        let mut b = PathMap::<u8>::new();
        b.insert(b"a", 2);
        b.insert(b"c", 2);

        let overlay = OverlayZipper::with_mapping(a.read_zipper(), b.read_zipper(), right_only);

        assert_eq!(overlay.val_count(), 2);
    }

    #[test]
    fn dpa_overlay_zipper_matches_eager_join_observations() {
        let mut left = PathMap::<()>::new();
        left.insert(b"", ());
        left.insert(b"shared:value", ());
        left.insert(b"left:value", ());
        left.create_path(b"shared:dangling");
        left.create_path(b"left:dangling");

        let mut right = PathMap::<()>::new();
        right.insert(b"", ());
        right.insert(b"shared:value", ());
        right.insert(b"right:value", ());
        right.create_path(b"shared:dangling");
        right.create_path(b"right:dangling");

        let joined = left.join(&right);
        let mut overlay = OverlayZipper::new(left.read_zipper(), right.read_zipper());
        assert_same_observed_pathspace(&joined, &mut overlay);

        let focus = b"shared:";
        let expected_focus = joined.read_zipper_at_path(focus).make_map();
        let mut overlay_focus = OverlayZipper::new(
            left.read_zipper_at_path(focus),
            right.read_zipper_at_path(focus),
        );
        assert_same_observed_pathspace(&expected_focus, &mut overlay_focus);
    }

    #[test]
    fn overlay_try_make_map_materializes_eager_join_observations() {
        let mut left = PathMap::<u8>::new();
        left.set_val_at(b"shared:value", 1);
        left.set_val_at(b"left:value", 2);
        left.create_path(b"shared:dangling");

        let mut right = PathMap::<u8>::new();
        right.set_val_at(b"shared:value", 9);
        right.set_val_at(b"right:value", 3);
        right.create_path(b"right:dangling");

        let expected = left.join(&right);
        let overlay = OverlayZipper::new(left.read_zipper(), right.read_zipper());

        assert!(!overlay.native_subtries());
        let materialized = overlay
            .try_make_map()
            .expect("overlay zipper should materialize by observation");
        let mut materialized_zipper = materialized.read_zipper();
        assert_same_observed_pathspace(&expected, &mut materialized_zipper);

        let focus = b"missing";
        let focused_expected = expected.read_zipper_at_path(focus).make_map();
        let mut focused_overlay = OverlayZipper::new(left.read_zipper(), right.read_zipper());
        focused_overlay.descend_to(focus);
        let focused_materialized = focused_overlay
            .try_make_map()
            .expect("focused overlay zipper should materialize by observation");
        let mut focused_materialized_zipper = focused_materialized.read_zipper();
        assert_same_observed_pathspace(&focused_expected, &mut focused_materialized_zipper);
    }

    zipper_moving_tests::zipper_moving_tests!(
        overlay_zipper,
        split_overlay_maps,
        overlay_from_maps
    );

    zipper_iteration_tests::zipper_iteration_tests!(
        overlay_zipper,
        split_overlay_maps,
        overlay_from_maps
    );
}
