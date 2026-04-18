use pathmap_derive::PolyZipperExplicit;

use crate::{
    alloc::{Allocator, GlobalAlloc},
    ring::{AlgebraicResult, COUNTER_IDENT, DistributiveLattice, Lattice, SELF_IDENT},
    utils::ByteMask,
    zipper::{
        ReadZipperTracked, ReadZipperUntracked, Zipper, ZipperInfallibleSubtries, ZipperMoving,
        ZipperValues, ZipperWriting,
    },
};

/// Extension trait providing algebraic merge operations on radix-256 trie zippers.
///
/// This trait exposes high-level operations such as [`join`](Self::join),
/// [`meet`](Self::meet), and [`subtract`](Self::subtract) directly on zipper
/// instances, allowing them to be invoked in a method-oriented style:
///
/// ```ignore
/// lhs.join(&mut rhs, &mut out);
/// lhs.meet(&mut rhs, &mut out);
/// lhs.subtract(&mut rhs, &mut out);
/// ```
///
/// # Overview
///
/// All operations are implemented as *lockstep traversals* over two tries using
/// zipper navigation. They exploit the lexicographic ordering of child edges
/// (bytes `0..=255`) to efficiently merge, intersect, or subtract subtries
/// without visiting unrelated regions.
///
/// Each method delegates to a corresponding free function ([`zipper_join`],
/// [`zipper_meet`], [`zipper_subtract`]), preserving their performance
/// characteristics and semantics.
///
/// # Semantics
///
/// The provided operations correspond to common lattice and set-like behaviors:
///
/// - [`join`](Self::join): least upper bound (union-like merge),
/// - [`meet`](Self::meet): greatest lower bound (intersection),
/// - [`subtract`](Self::subtract): asymmetric difference (`lhs \ rhs`).
///
/// All operations write their result into a separate output zipper implementing
/// [`ZipperWriting`].
///
/// # Notes
///
/// - The operations are *asymmetric* with respect to the receiver (`self`) and
///   the `rhs` argument, which is particularly relevant for
///   [`subtract`](Self::subtract).
/// - The output zipper is written incrementally during traversal and must be
///   positioned consistently with the input zippers.
///
/// # See also
///
/// - [`zipper_join`]
/// - [`zipper_meet`]
/// - [`zipper_subtract`]
pub trait ZipperAlgebraExt<V: Clone + Send + Sync, A: Allocator = GlobalAlloc>:
    ZipperInfallibleSubtries<V, A> + ZipperMoving + Sized
{
    #[inline]
    fn join<ZR, Out>(&mut self, rhs: &mut ZR, out: &mut Out)
    where
        V: Lattice,
        ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        zipper_join(self, rhs, out);
    }

    #[inline]
    fn meet<ZR, Out>(&mut self, rhs: &mut ZR, out: &mut Out)
    where
        V: Lattice,
        ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        zipper_meet(self, rhs, out);
    }

    #[inline]
    fn subtract<ZR, Out>(&mut self, rhs: &mut ZR, out: &mut Out)
    where
        V: DistributiveLattice,
        ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        zipper_subtract(self, rhs, out);
    }
}

impl<V: Clone + Send + Sync + Unpin, A: Allocator> ZipperAlgebraExt<V, A>
    for ReadZipperUntracked<'_, '_, V, A>
{
}

impl<V: Clone + Send + Sync + Unpin, A: Allocator> ZipperAlgebraExt<V, A>
    for ReadZipperTracked<'_, '_, V, A>
{
}

/// Performs an ordered join (least upper bound) of two radix-256 tries using zipper traversal.
///
/// This function merges two tries by simultaneously traversing them in lexicographic order,
/// exploiting the ordering of child edges (bytes `0..=255`) to avoid unnecessary descent.
///
/// # Value semantics
///
/// When both tries contain a value at the same key, they are merged using the [`Lattice`]
/// operation [`Lattice::pjoin`]. The result is interpreted as follows:
///
/// - [`AlgebraicResult::None`] → no value is written,
/// - [`AlgebraicResult::Identity`] → one of the inputs is reused (based on identity mask),
/// - [`AlgebraicResult::Element`] → the computed value is written.
///
/// If only one side contains a value, it is propagated unchanged.
///
/// # Complexity
///
/// Let:
/// - `h` be the maximum key length,
/// - `d` be the size of overlapping subtries,
/// - `f` be the size of the frontier (distinct child edges encountered).
///
/// Then:
///
/// - Best case (disjoint tries): **O(h)**
/// - Typical case: **O(h + f)**
/// - Worst case (identical structure): **O(n)**
///
/// The algorithm avoids visiting entire disjoint subtrees by grafting them directly.
///
/// # Notes
///
/// This is a stackless depth-first traversal implemented via zippers.
/// The outer loop manages ascent (unwinding), while the inner loop
/// exhausts the current node's children.
///
pub fn zipper_join<V, ZL, ZR, Out, A>(lhs: &mut ZL, rhs: &mut ZR, out: &mut Out)
where
    V: Lattice + Clone + Send + Sync,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    zipper_merge::<Join, V, ZL, ZR, Out, A>(lhs, rhs, out);
}

/// Performs an ordered join (least upper bound) of three radix-256 tries using zipper traversal.
/// That is, it performs: (`lhs` ⊔ `mid`) ⊔ `rhs`, where `⊔` = [`zipper_join`]
///
/// # See also
///
/// [`zipper_join`]
///
pub fn zipper_join3<V, ZL, ZM, ZR, Out, A>(lhs: &mut ZL, mid: &mut ZM, rhs: &mut ZR, out: &mut Out)
where
    V: Lattice + Clone + Send + Sync,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZM: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    zipper_merge3::<Join, V, ZL, ZM, ZR, Out, A>(lhs, mid, rhs, out);
}

/// Performs an ordered meet (greatest lower bound) of two radix-256 tries using zipper traversal.
///
/// This function intersects two tries by simultaneously traversing them in lexicographic order,
/// exploiting the ordering of child edges (bytes `0..=255`) to avoid unnecessary descent.
///
/// # Value semantics
///
/// When both tries contain a value at the same key, they are merged using the [`Lattice`]
/// operation [`Lattice::pmeet`]. The result is interpreted as follows:
///
/// - [`AlgebraicResult::None`] → no value is written,
/// - [`AlgebraicResult::Identity`] → one of the inputs is reused (based on identity mask),
/// - [`AlgebraicResult::Element`] → the computed value is written.
///
/// If a value is present on only one side, it is discarded.
///
/// # Complexity
///
/// Let:
/// - `h` be the maximum key length,
/// - `d` be the size of overlapping subtries,
/// - `f` be the size of the shared frontier (common child edges).
///
/// Then:
///
/// - Best case (disjoint tries): **O(h)**
/// - Typical case: **O(h + f)**
/// - Worst case (identical structure): **O(n)**
///
/// The algorithm avoids visiting disjoint subtrees by skipping them entirely.
///
/// # Notes
///
/// This is a stackless depth-first traversal implemented via zippers.
/// The outer loop manages ascent (unwinding), while the inner loop
/// processes only shared child edges at each node.
///
/// Unlike [`zipper_join`], this operation descends exclusively into edges
/// present in *both* tries, forming the intersection of their structures.
///
pub fn zipper_meet<V, ZL, ZR, Out, A>(lhs: &mut ZL, rhs: &mut ZR, out: &mut Out)
where
    V: Lattice + Clone + Send + Sync,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    zipper_merge::<Meet, V, ZL, ZR, Out, A>(lhs, rhs, out);
}

/// Performs an ordered meet (greater lower bound) of three radix-256 tries using zipper traversal.
/// That is, it performs: (`lhs` ⊓ `mid`) ⊓ `rhs`, where `⊓` = [`zipper_meet`]
///
/// # See also
///
/// [`zipper_meet`]
///
pub fn zipper_meet3<V, ZL, ZM, ZR, Out, A>(lhs: &mut ZL, mid: &mut ZM, rhs: &mut ZR, out: &mut Out)
where
    V: Lattice + Clone + Send + Sync,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZM: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    zipper_merge3::<Meet, V, ZL, ZM, ZR, Out, A>(lhs, mid, rhs, out);
}

/// Performs an ordered subtraction (set difference, `lhs \ rhs`) of two radix-256 tries
/// using zipper traversal.
///
/// This function subtracts the structure and values of `rhs` from `lhs` by simultaneously
/// traversing both tries in lexicographic order. It exploits the ordering of child edges
/// (bytes `0..=255`) to avoid unnecessary descent.
///
/// # Value semantics
///
/// Values are handled asymmetrically:
///
/// - If only `lhs` contains a value, it is preserved unchanged.
/// - If only `rhs` contains a value, it is ignored.
/// - If both tries contain a value at the same key, they are comined using the [`DistributiveLattice`]
///   operation [`DistributiveLattice::psubtract`]. The result is interpreted as follows:
///
///   - [`AlgebraicResult::None`] → no value is written,
///   - [`AlgebraicResult::Identity`] → only lhs is preserved (based on identity mask),
///   - [`AlgebraicResult::Element`] → the computed value is written.
///
/// # Complexity
///
/// Let:
/// - `h` be the maximum key length,
/// - `d` be the size of overlapping subtries,
/// - `f` be the size of the frontier (distinct child edges encountered).
///
/// Then:
///
/// - Best case (disjoint tries): **O(h)**
/// - Typical case: **O(h + f)**
/// - Worst case (identical structure): **O(n)**
///
/// The algorithm avoids visiting disjoint subtrees by grafting `lhs`-only regions
/// and skipping `rhs`-only regions entirely.
///
/// # Notes
///
/// This is a stackless depth-first traversal implemented via zippers.
/// The outer loop manages ascent (unwinding), while the inner loop
/// exhausts the current node's children.
///
/// Unlike [`zipper_join`] and [`zipper_meet`], this operation is asymmetric:
/// it preserves only the parts of `lhs` that are not overlapped by `rhs`,
/// effectively removing any shared structure or values.
///
pub fn zipper_subtract<V, ZL, ZR, Out, A>(lhs: &mut ZL, rhs: &mut ZR, out: &mut Out)
where
    V: DistributiveLattice + Clone + Send + Sync,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    zipper_merge::<Subtract, V, ZL, ZR, Out, A>(lhs, rhs, out);
}

/// Performs an ordered subtraction (set difference of three radix-256 tries using zipper traversal.
/// That is, it performs: (`lhs` \ `mid`) \ `rhs`, where `\` = [`zipper_subtract`]
///
/// # See also
///
/// [`zipper_subtract`]
///
pub fn zipper_subtract3<V, ZL, ZM, ZR, Out, A>(
    lhs: &mut ZL,
    mid: &mut ZM,
    rhs: &mut ZR,
    out: &mut Out,
) where
    V: DistributiveLattice + Clone + Send + Sync,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZM: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    zipper_merge3::<Subtract, V, ZL, ZM, ZR, Out, A>(lhs, mid, rhs, out);
}

trait MergePolicy<V: Clone + Send + Sync> {
    #[inline]
    fn on_left_only<Z, Out, A>(z: &mut Z, range: ByteMask, out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        Self::on_single(z, 0b01, range, out);
    }

    #[inline]
    fn on_right_only<Z, Out, A>(z: &mut Z, range: ByteMask, out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        Self::on_single(z, 0b10, range, out);
    }

    fn on_single<Z, Out, A>(z: &mut Z, mask: u64, range: ByteMask, out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>;

    fn descend_on_some_equal(mask: u64) -> bool;
}

trait ValuePolicy<V> {
    fn combine(l: Option<&V>, r: Option<&V>) -> Option<V>;
    fn combine3(l: Option<&V>, m: Option<&V>, r: Option<&V>) -> Option<V> {
        // (l op m) op r
        Self::combine(Self::combine(l, m).as_ref(), r)
    }
    fn combine4(a: Option<&V>, b: Option<&V>, c: Option<&V>, d: Option<&V>) -> Option<V> {
        // ((a op b) op c) op d
        Self::combine(Self::combine(Self::combine(a, b).as_ref(), c).as_ref(), d)
    }
    fn combine_n<'a, I>(vals: I) -> Option<V>
    where
        I: Iterator<Item = Option<&'a V>>,
        V: 'a,
    {
        vals.fold(None, |acc, v| Self::combine(acc.as_ref(), v))
    }
}

fn zipper_merge<P, V, ZL, ZR, Out, A>(lhs: &mut ZL, rhs: &mut ZR, out: &mut Out)
where
    V: Clone + Send + Sync,
    P: MergePolicy<V> + ValuePolicy<V>,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    // merge root values before descending
    if let Some(v) = P::combine(lhs.val(), rhs.val()) {
        out.set_val(v);
    }

    let mut k = 0;
    let mut lhs_mask = lhs.child_mask();
    let mut rhs_mask = rhs.child_mask();
    let mut lhs_idx = 0;
    let mut rhs_idx = 0;

    // At each node, the algorithm treats the sets of child edges of `lhs` and `rhs` as two sorted
    // sequences and performs a merge-like traversal:
    //
    // - If a range of edges exists only in one side, the corresponding merge policy method is
    //   called.
    // - If both sides contain the same edge, the algorithm descends into that child, recursively
    //   merging the corresponding subtries.
    // - Descent is simulated iteratively using zipper movement (`descend_to_byte` / `ascend_byte`)
    //   and an explicit depth counter (`k`).
    'ascend: loop {
        'merge_level: loop {
            let lhs_next = lhs_mask.indexed_bit::<true>(lhs_idx as usize);
            let rhs_next = rhs_mask.indexed_bit::<true>(rhs_idx as usize);

            match lhs_next {
                Some(lhs_byte) => match rhs_next {
                    Some(rhs_byte) if lhs_byte < rhs_byte => {
                        P::on_left_only(lhs, ByteMask::from_range(lhs_byte..rhs_byte), out);
                        lhs_idx = lhs_mask.index_of(rhs_byte);
                    }
                    Some(rhs_byte) if lhs_byte > rhs_byte => {
                        P::on_right_only(rhs, ByteMask::from_range(rhs_byte..lhs_byte), out);
                        rhs_idx = rhs_mask.index_of(lhs_byte);
                    }
                    Some(rhs_byte) => {
                        // equal → descend
                        out.descend_to_byte(lhs_byte);

                        lhs.descend_to_byte(lhs_byte);
                        rhs.descend_to_byte(lhs_byte);

                        if let Some(v) = P::combine(lhs.val(), rhs.val()) {
                            out.set_val(v);
                        }

                        lhs_mask = lhs.child_mask();
                        rhs_mask = rhs.child_mask();

                        lhs_idx = 0;
                        rhs_idx = 0;

                        k += 1;
                        continue 'merge_level;
                    }
                    None => {
                        P::on_left_only(lhs, ByteMask::from_range(lhs_byte..), out);
                        break 'merge_level;
                    }
                },
                None => match rhs_next {
                    Some(rhs_byte) => {
                        P::on_right_only(rhs, ByteMask::from_range(rhs_byte..), out);
                        break 'merge_level;
                    }
                    None => break 'merge_level,
                },
            }
        }

        // If we are at root and no deeper recursion pending, we're done
        if k == 0 {
            break 'ascend;
        }

        let byte_from = *lhs.path().last().expect("non-empty path when k > 0");

        rhs.ascend_byte();
        rhs_mask = rhs.child_mask();
        rhs_idx = rhs_mask.index_of(byte_from) + 1;

        lhs.ascend_byte();
        lhs_mask = lhs.child_mask();
        lhs_idx = lhs_mask.index_of(byte_from) + 1;

        out.ascend_byte();
        k -= 1;
    }
}

#[inline(always)]
fn cmp_swap(a: &mut Option<u8>, b: &mut Option<u8>) {
    if let Some(x) = *a {
        if let Some(y) = *b {
            if y < x {
                std::mem::swap(a, b);
            }
        }
    } else {
        std::mem::swap(a, b);
    }
}

fn zipper_merge3<P, V, ZL, ZM, ZR, Out, A>(lhs: &mut ZL, mid: &mut ZM, rhs: &mut ZR, out: &mut Out)
where
    V: Clone + Send + Sync,
    P: MergePolicy<V> + ValuePolicy<V>,
    A: Allocator,
    ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZM: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    const L: u8 = 0b001;
    const M: u8 = 0b010;
    const R: u8 = 0b100;
    const LM: u8 = L | M;
    const LR: u8 = L | R;
    const MR: u8 = M | R;
    const LMR: u8 = L | M | R;

    fn descend2<P, V, ZL, ZR, Out, A>(b: u8, lhs: &mut ZL, rhs: &mut ZR, out: &mut Out)
    where
        V: Clone + Send + Sync,
        P: MergePolicy<V> + ValuePolicy<V>,
        A: Allocator,
        ZL: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        ZR: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        out.descend_to_byte(b);
        lhs.descend_to_byte(b);
        rhs.descend_to_byte(b);

        zipper_merge::<P, V, ZL, ZR, Out, A>(lhs, rhs, out);

        rhs.ascend_byte();
        lhs.ascend_byte();
        out.ascend_byte();
    }

    // merge root values before descending
    if let Some(v) = P::combine3(lhs.val(), mid.val(), rhs.val()) {
        out.set_val(v);
    }

    let mut k = 0;
    let mut lhs_mask = lhs.child_mask();
    let mut mid_mask = mid.child_mask();
    let mut rhs_mask = rhs.child_mask();
    let mut lhs_idx = 0;
    let mut mid_idx = 0;
    let mut rhs_idx = 0;

    'ascend: loop {
        'merge_level: loop {
            let l = lhs_mask.indexed_bit::<true>(lhs_idx as usize);
            let m = mid_mask.indexed_bit::<true>(mid_idx as usize);
            let r = rhs_mask.indexed_bit::<true>(rhs_idx as usize);

            let mut a = l;
            let mut b = m;
            let mut c = r;

            cmp_swap(&mut a, &mut b);
            cmp_swap(&mut a, &mut c);

            if let Some(min) = a {
                let mut frontier = 0;
                if a == l {
                    frontier = L;
                }
                if a == m {
                    frontier |= M;
                }
                if a == r {
                    frontier |= R;
                }

                match frontier {
                    // single → graft
                    L => {
                        cmp_swap(&mut b, &mut c);
                        if let Some(next) = b {
                            P::on_single(lhs, L as u64, ByteMask::from_range(min..next), out);
                            lhs_idx = lhs_mask.index_of(next)
                        } else {
                            P::on_single(lhs, L as u64, ByteMask::from_range(min..), out);
                            break 'merge_level;
                        }
                    }
                    M => {
                        cmp_swap(&mut b, &mut c);
                        if let Some(next) = b {
                            P::on_single(mid, M as u64, ByteMask::from_range(min..next), out);
                            mid_idx = mid_mask.index_of(next);
                        } else {
                            P::on_single(mid, M as u64, ByteMask::from_range(min..), out);
                            break 'merge_level;
                        }
                    }
                    R => {
                        cmp_swap(&mut b, &mut c);
                        if let Some(next) = b {
                            P::on_single(rhs, R as u64, ByteMask::from_range(min..next), out);
                            rhs_idx = rhs_mask.index_of(next);
                        } else {
                            P::on_single(rhs, R as u64, ByteMask::from_range(min..), out);
                            break 'merge_level;
                        }
                    }
                    // two-way → descend 2
                    LM => {
                        if P::descend_on_some_equal(LM as u64) {
                            descend2::<P, V, ZL, ZM, Out, A>(min, lhs, mid, out);
                        }
                        lhs_idx += 1;
                        mid_idx += 1;
                    }
                    MR => {
                        if P::descend_on_some_equal(MR as u64) {
                            descend2::<P, V, ZM, ZR, Out, A>(min, mid, rhs, out);
                        }
                        mid_idx += 1;
                        rhs_idx += 1;
                    }
                    LR => {
                        if P::descend_on_some_equal(LR as u64) {
                            descend2::<P, V, ZL, ZR, Out, A>(min, lhs, rhs, out);
                        }
                        lhs_idx += 1;
                        rhs_idx += 1;
                    }
                    // full 3-way
                    LMR => {
                        out.descend_to_byte(min);

                        lhs.descend_to_byte(min);
                        mid.descend_to_byte(min);
                        rhs.descend_to_byte(min);

                        if let Some(val) = P::combine3(lhs.val(), mid.val(), rhs.val()) {
                            out.set_val(val);
                        }

                        lhs_mask = lhs.child_mask();
                        mid_mask = mid.child_mask();
                        rhs_mask = rhs.child_mask();

                        lhs_idx = 0;
                        mid_idx = 0;
                        rhs_idx = 0;

                        k += 1;
                        continue 'merge_level;
                    }
                    _ => unreachable!(),
                }
            } else {
                break 'merge_level;
            }
        }

        // If we are at root and no deeper recursion pending, we're done
        if k == 0 {
            break 'ascend;
        }

        let byte_from = *lhs.path().last().expect("non-empty path when k > 0");

        rhs.ascend_byte();
        rhs_mask = rhs.child_mask();
        rhs_idx = rhs_mask.index_of(byte_from) + 1;

        mid.ascend_byte();
        mid_mask = mid.child_mask();
        mid_idx = mid_mask.index_of(byte_from) + 1;

        lhs.ascend_byte();
        lhs_mask = lhs.child_mask();
        lhs_idx = lhs_mask.index_of(byte_from) + 1;

        out.ascend_byte();
        k -= 1;
    }
}

// semi-unrolled (bitmask-driven)
// Beyond 4, the combinatorics start to creak, but k = 4 is a sweet spot:
// - still manageable (16 frontier cases)
// - still branch-predictable
// - still worth it for hot paths
fn zipper_merge4<P, V, Z0, Z1, Z2, Z3, Out, A>(
    z0: &mut Z0,
    z1: &mut Z1,
    z2: &mut Z2,
    z3: &mut Z3,
    out: &mut Out,
) where
    V: Clone + Send + Sync,
    P: MergePolicy<V> + ValuePolicy<V>,
    A: Allocator,
    Z0: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Z1: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Z2: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Z3: ZipperInfallibleSubtries<V, A> + ZipperMoving,
    Out: ZipperWriting<V, A>,
{
    // merge root values before descending
    if let Some(v) = P::combine4(z0.val(), z1.val(), z2.val(), z3.val()) {
        out.set_val(v);
    }

    let mut k = 0;
    // state (fully unrolled)
    let mut m0 = z0.child_mask();
    let mut m1 = z1.child_mask();
    let mut m2 = z2.child_mask();
    let mut m3 = z3.child_mask();

    let mut i0 = 0;
    let mut i1 = 0;
    let mut i2 = 0;
    let mut i3 = 0;

    'ascend: loop {
        'merge_level: loop {
            // min selection
            let mut b0 = m0.indexed_bit::<true>(i0 as usize);
            let mut b1 = m1.indexed_bit::<true>(i1 as usize);
            let mut b2 = m2.indexed_bit::<true>(i2 as usize);
            let mut b3 = m3.indexed_bit::<true>(i3 as usize);

            let mut a = b0;
            let mut b = b1;
            let mut c = b2;
            let mut d = b3;

            cmp_swap(&mut a, &mut b);
            cmp_swap(&mut a, &mut c);
            cmp_swap(&mut a, &mut d);

            if let Some(min) = a {
                let mut frontier = 0u8;
                if b0 == a {
                    frontier |= 0b0001;
                }
                if b1 == a {
                    frontier |= 0b0010;
                }
                if b2 == a {
                    frontier |= 0b0100;
                }
                if b3 == a {
                    frontier |= 0b1000;
                }

                // full match
                if frontier == 0b1111 {
                    out.descend_to_byte(min);

                    z0.descend_to_byte(min);
                    z1.descend_to_byte(min);
                    z2.descend_to_byte(min);
                    z3.descend_to_byte(min);

                    if let Some(v) = P::combine4(z0.val(), z1.val(), z2.val(), z3.val()) {
                        out.set_val(v);
                    }

                    m0 = z0.child_mask();
                    i0 = 0;
                    m1 = z1.child_mask();
                    i1 = 0;
                    m2 = z2.child_mask();
                    i2 = 0;
                    m3 = z3.child_mask();
                    i3 = 0;

                    k += 1;
                    continue 'merge_level;
                }

                let cnt = frontier.count_ones();
                // singleton
                if cnt == 1 {
                    cmp_swap(&mut b, &mut c);
                    cmp_swap(&mut b, &mut d);

                    match frontier {
                        0b0001 => {
                            if let Some(next) = b {
                                P::on_single(z0, 0b0001, ByteMask::from_range(min..next), out);
                                i0 = m0.index_of(next);
                            } else {
                                P::on_single(z0, 0b0001, ByteMask::from_range(min..), out);
                                break 'merge_level;
                            }
                        }
                        0b0010 => {
                            if let Some(next) = b {
                                P::on_single(z1, 0b0010, ByteMask::from_range(min..next), out);
                                i1 = m1.index_of(next);
                            } else {
                                P::on_single(z1, 0b0010, ByteMask::from_range(min..), out);
                                break 'merge_level;
                            }
                        }
                        0b0100 => {
                            if let Some(next) = b {
                                P::on_single(z2, 0b0100, ByteMask::from_range(min..next), out);
                                i2 = m2.index_of(next);
                            } else {
                                P::on_single(z2, 0b0100, ByteMask::from_range(min..), out);
                                break 'merge_level;
                            }
                        }
                        0b1000 => {
                            if let Some(next) = b {
                                P::on_single(z3, 0b1000, ByteMask::from_range(min..next), out);
                                i3 = m3.index_of(next);
                            } else {
                                P::on_single(z3, 0b1000, ByteMask::from_range(min..), out);
                                break 'merge_level;
                            }
                        }
                        _ => unreachable!(),
                    };
                } else {
                    // partial overlap (2 or 3)

                    // avoid 16 match arms and duplicated logic
                    if P::descend_on_some_equal(frontier as u64) {
                        out.descend_to_byte(min);

                        if frontier & 0b0001 != 0 {
                            z0.descend_to_byte(min);
                        }
                        if frontier & 0b0010 != 0 {
                            z1.descend_to_byte(min);
                        }
                        if frontier & 0b0100 != 0 {
                            z2.descend_to_byte(min);
                        }
                        if frontier & 0b1000 != 0 {
                            z3.descend_to_byte(min);
                        }

                        // recurse on subset (still using 4-way function, but inactive ones won't match)
                        if (cnt == 2) {
                            let i = frontier.trailing_zeros();
                            let j = (frontier & !(1 << i)).trailing_zeros();
                            match (i, j) {
                                (0, 1) => {
                                    zipper_merge::<P, V, Z0, Z1, Out, A>(z0, z1, out);
                                }
                                (0, 2) => {
                                    zipper_merge::<P, V, Z0, Z2, Out, A>(z0, z2, out);
                                }
                                (0, 3) => {
                                    zipper_merge::<P, V, Z0, Z3, Out, A>(z0, z3, out);
                                }
                                (1, 2) => {
                                    zipper_merge::<P, V, Z1, Z2, Out, A>(z1, z2, out);
                                }
                                (1, 3) => {
                                    zipper_merge::<P, V, Z1, Z3, Out, A>(z1, z3, out);
                                }
                                (2, 3) => {
                                    zipper_merge::<P, V, Z2, Z3, Out, A>(z2, z3, out);
                                }
                                _ => unreachable!(),
                            }
                        } else {
                            // cnt == 3
                            let mut bits = frontier;
                            let i = bits.trailing_zeros();
                            bits &= bits - 1; // trick: it removes the lowest bit set from a bitmask
                            let j = bits.trailing_zeros();
                            bits &= bits - 1;
                            let k = bits.trailing_zeros();
                            match (i, j, k) {
                                (0, 1, 2) => {
                                    zipper_merge3::<P, V, Z0, Z1, Z2, Out, A>(z0, z1, z2, out);
                                }
                                (0, 1, 3) => {
                                    zipper_merge3::<P, V, Z0, Z1, Z3, Out, A>(z0, z1, z3, out);
                                }
                                (0, 2, 3) => {
                                    zipper_merge3::<P, V, Z0, Z2, Z3, Out, A>(z0, z2, z3, out);
                                }
                                (1, 2, 3) => {
                                    zipper_merge3::<P, V, Z1, Z2, Z3, Out, A>(z1, z2, z3, out);
                                }
                                _ => unreachable!(),
                            }
                        }

                        if frontier & 0b0001 != 0 {
                            z0.ascend_byte();
                        }
                        if frontier & 0b0010 != 0 {
                            z1.ascend_byte();
                        }
                        if frontier & 0b0100 != 0 {
                            z2.ascend_byte();
                        }
                        if frontier & 0b1000 != 0 {
                            z3.ascend_byte();
                        }

                        out.ascend_byte();
                    }
                    // then advance
                    if frontier & 0b0001 != 0 {
                        i0 += 1;
                    }
                    if frontier & 0b0010 != 0 {
                        i1 += 1;
                    }
                    if frontier & 0b0100 != 0 {
                        i2 += 1;
                    }
                    if frontier & 0b1000 != 0 {
                        i3 += 1;
                    }
                }
            } else {
                break 'merge_level;
            }
        }

        // If we are at root and no deeper recursion pending, we're done
        if k == 0 {
            break 'ascend;
        }

        let byte_from = *z0.path().last().expect("non-empty path when k > 0");

        z0.ascend_byte();
        m0 = z0.child_mask();
        i0 = m0.index_of(byte_from) + 1;

        z1.ascend_byte();
        m1 = z1.child_mask();
        i1 = m1.index_of(byte_from) + 1;

        z2.ascend_byte();
        m2 = z2.child_mask();
        i2 = m2.index_of(byte_from) + 1;

        z3.ascend_byte();
        m3 = z3.child_mask();
        i3 = m3.index_of(byte_from) + 1;

        out.ascend_byte();
        k -= 1;
    }
}

use zipper_algebra_poly::SomeZ as Z;

// - The function is fully monomorphized over `N` and uses a bitmask (`active`)
//   to track participating zippers.
// - Small frontiers (`k ≤ 4`) are dispatched to specialized implementations
//   for improved performance.
// - Requires `N ≤ 64`.
fn zipper_merge_n_mono<P, V, Out, A, const N: usize>(
    zs: &mut [Z<'_, '_, V, A>; N],
    active: u64,
    out: &mut Out,
) where
    V: Clone + Send + Sync + Unpin,
    P: MergePolicy<V> + ValuePolicy<V>,
    A: Allocator,
    Out: ZipperWriting<V, A>,
{
    debug_assert!(N > 0 && N <= 64);

    #[inline]
    fn active_bits<const N: usize>(active: u64) -> impl Iterator<Item = usize> {
        (0..N).filter(move |i| (active >> i) & 1 != 0)
    }

    fn zippers<'a, 'trie, 'path, V, A, const N: usize>(
        zs: &'a [Z<'trie, 'path, V, A>; N],
        active: u64,
    ) -> impl Iterator<Item = (usize, &'a Z<'trie, 'path, V, A>)>
    where
        V: Clone + Send + Sync + Unpin,
        A: Allocator,
    {
        active_bits::<N>(active).map(|i| (i, &zs[i]))
    }

    fn values<'a, 'trie, 'path, V, A, const N: usize>(
        zs: &'a [Z<'trie, 'path, V, A>; N],
        active: u64,
    ) -> impl Iterator<Item = Option<&'a V>>
    where
        V: Clone + Send + Sync + Unpin,
        A: Allocator,
    {
        zippers(zs, active).map(|(_, z)| z.val())
    }

    // small micro-helpers
    #[inline(always)]
    fn for_each_bit(mut bits: u64, mut f: impl FnMut(usize)) {
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            f(i);
        }
    }

    #[inline(always)]
    fn with_k<const K: usize, T, R>(
        xs: &mut [T],
        mut bits: u64,
        f: impl FnOnce([&mut T; K]) -> R,
    ) -> R {
        debug_assert!(bits.count_ones() as usize >= K);

        // collect raw pointers first (safe)
        let mut ptrs: [*mut T; K] = [std::ptr::null_mut(); K];

        let mut i = 0;
        while i < K {
            let idx = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            ptrs[i] = unsafe { xs.as_mut_ptr().add(idx) };
            i += 1;
        }

        // SAFETY:
        // - indices are distinct (bitmask)
        // - derived from same slice

        // should be zero-cost after inlining
        let refs = unsafe { ptrs.map(|p| &mut *p) };

        f(refs)
    }

    // combine root values
    if let Some(v) = P::combine_n(values(zs, active)) {
        out.set_val(v);
    }

    let mut idxs = [0; N];
    let mut masks = [ByteMask::EMPTY; N];
    for (i, z) in zippers(zs, active) {
        masks[i] = z.child_mask();
    }

    // At each node, the algorithm:
    //
    // - Treats the child edges of all active zippers as sorted byte sequences,
    // - Computes the minimal byte `a` across all inputs,
    // - Forms the *frontier* — the subset of zippers containing `a`,
    // - Dispatches based on frontier size:
    //
    //   - **Full match (`frontier == active`)**
    //     Descends into all zippers without recursion (fast path).
    //
    //   - **Singleton (`|frontier| = 1`)**
    //     Grafts the corresponding subtrie directly into the output.
    //
    //   - **Partial overlap (`1 < |frontier| < N`)**
    //     Optionally descends into the subset, dispatching to specialized
    //     implementations for small arities (`k ≤ 4`) or recursively invoking
    //     this function on the subset.
    //
    // The traversal is performed iteratively using zipper movements
    // (`descend_to_byte` / `ascend_byte`) and an explicit depth counter,
    // avoiding recursion in the common case.
    let mut k = 0;
    debug_assert!(active.count_ones() > 0);
    'ascend: loop {
        'merge_level: loop {
            let mut min = None;
            let mut frontier = 0u64;
            let mut next = None;

            for i in active_bits::<N>(active) {
                if let Some(b) = masks[i].indexed_bit::<true>(idxs[i] as usize) {
                    match min {
                        None => {
                            min = Some(b);
                            frontier = 1 << i;
                        }
                        Some(m) if b < m => {
                            next = Some(m);
                            min = Some(b);
                            frontier = 1 << i;
                        }
                        Some(m) if b == m => {
                            frontier |= 1 << i;
                        }
                        Some(m) => {
                            next = match next {
                                Some(n) if n <= b => Some(n),
                                _ => Some(b),
                            };
                        }
                    }
                }
            }

            match min {
                None => {
                    break 'merge_level;
                }
                Some(a) => {
                    // Dispatch

                    // - Case A: full match (frontier == all bits)
                    if frontier == active {
                        out.descend_to_byte(a);

                        // descend and refresh masks and indices
                        for_each_bit(active, |i| {
                            let mut z = &mut zs[i];
                            z.descend_to_byte(a);
                            masks[i] = z.child_mask();
                            idxs[i] = 0;
                        });

                        if let Some(v) = P::combine_n(values(zs, active)) {
                            out.set_val(v);
                        }

                        k += 1;
                        continue 'merge_level;
                    }

                    let cnt = frontier.count_ones();
                    // - Case B: singleton (|frontier| = 1)
                    if (cnt == 1) {
                        let i = frontier.trailing_zeros() as usize;
                        match next {
                            None => {
                                P::on_single(&mut zs[i], frontier, ByteMask::from_range(a..), out);
                                break 'merge_level;
                            }
                            Some(b) => {
                                P::on_single(&mut zs[i], frontier, ByteMask::from_range(a..b), out);
                                // advance
                                idxs[i] = masks[i].index_of(b);
                            }
                        }
                    } else {
                        // - Case C: subset (1 < k < N)
                        if P::descend_on_some_equal(frontier) {
                            out.descend_to_byte(a);
                            match cnt {
                                2 => with_k::<2, _, _>(zs, frontier, |[lhs, rhs]| {
                                    lhs.descend_to_byte(a);
                                    rhs.descend_to_byte(a);

                                    zipper_merge::<P, _, _, _, _, _>(lhs, rhs, out);

                                    rhs.ascend_byte();
                                    lhs.ascend_byte();
                                }),
                                3 => with_k::<3, _, _>(zs, frontier, |[lhs, mid, rhs]| {
                                    lhs.descend_to_byte(a);
                                    mid.descend_to_byte(a);
                                    rhs.descend_to_byte(a);

                                    zipper_merge3::<P, _, _, _, _, _, _>(lhs, mid, rhs, out);

                                    rhs.ascend_byte();
                                    mid.ascend_byte();
                                    lhs.ascend_byte();
                                }),
                                4 => with_k::<4, _, _>(zs, frontier, |[z0, z1, z2, z3]| {
                                    z0.descend_to_byte(a);
                                    z1.descend_to_byte(a);
                                    z2.descend_to_byte(a);
                                    z3.descend_to_byte(a);

                                    zipper_merge4::<P, _, _, _, _, _, _, _>(z0, z1, z2, z3, out);

                                    z3.ascend_byte();
                                    z2.ascend_byte();
                                    z1.ascend_byte();
                                    z0.ascend_byte();
                                }),
                                _ => {
                                    // descend all active in the frontier
                                    for_each_bit(frontier, |i| zs[i].descend_to_byte(a));

                                    // recursive call with SAME array, smaller mask
                                    zipper_merge_n_mono::<P, V, Out, A, N>(zs, frontier, out);

                                    //ascend
                                    for_each_bit(frontier, |i| {
                                        zs[i].ascend_byte();
                                    });
                                }
                            }

                            out.ascend_byte();
                        }

                        // advance indices
                        for_each_bit(frontier, |i| {
                            idxs[i] += 1;
                        });
                    }
                }
            }
        }

        if (k == 0) {
            break 'ascend;
        }

        let i0 = active.trailing_zeros() as usize;
        let byte_from = *zs[i0].path().last().expect("non-empty path when k > 0");

        // ascend
        for_each_bit(active, |i| {
            let mut z = &mut zs[i];
            z.ascend_byte();
            masks[i] = z.child_mask();
            idxs[i] = masks[i].index_of(byte_from) + 1;
        });

        out.ascend_byte();
        k -= 1;
    }
}

// ==================== JOIN ====================

struct Join;
impl<V: Clone + Send + Sync> MergePolicy<V> for Join {
    #[inline]
    fn on_single<Z, Out, A>(z: &mut Z, _mask: u64, range: ByteMask, out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        out.graft_children(z, range);
    }

    #[inline]
    fn descend_on_some_equal(_mask: u64) -> bool {
        true
    }
}

impl<V: Lattice + Clone> ValuePolicy<V> for Join {
    fn combine(l: Option<&V>, r: Option<&V>) -> Option<V> {
        if let Some(lv) = l {
            if let Some(rv) = r {
                match lv.pjoin(rv) {
                    AlgebraicResult::None => None,
                    AlgebraicResult::Identity(mask) => {
                        if mask & SELF_IDENT != 0 {
                            l.cloned()
                        } else {
                            r.cloned()
                        }
                    }
                    AlgebraicResult::Element(v) => Some(v),
                }
            } else {
                l.cloned()
            }
        } else {
            r.cloned()
        }
    }
}

// ==================== MEET ====================

struct Meet;
impl<V: Clone + Send + Sync> MergePolicy<V> for Meet {
    #[inline(always)]
    fn on_single<Z, Out, A>(_z: &mut Z, _mask: u64, _range: ByteMask, _out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
    }

    #[inline(always)]
    fn descend_on_some_equal(_mask: u64) -> bool {
        false
    }
}

impl<V: Lattice + Clone> ValuePolicy<V> for Meet {
    fn combine(l: Option<&V>, r: Option<&V>) -> Option<V> {
        l.and_then(|lv| r.and_then(|rv| meet_refs(lv, rv)))
    }

    fn combine3(l: Option<&V>, m: Option<&V>, r: Option<&V>) -> Option<V> {
        l.and_then(|x| m.and_then(|y| r.and_then(|z| meet_acc(meet_refs(x, y)?, z))))
    }

    fn combine4(a: Option<&V>, b: Option<&V>, c: Option<&V>, d: Option<&V>) -> Option<V> {
        a.and_then(|w| {
            b.and_then(|x| {
                c.and_then(|y| d.and_then(|z| meet_acc(meet_acc(meet_refs(w, x)?, y)?, z)))
            })
        })
    }

    fn combine_n<'a, I>(vals: I) -> Option<V>
    where
        I: Iterator<Item = Option<&'a V>>,
        V: 'a,
    {
        let mut it = vals;
        let z = it.next()?.cloned()?;
        it.try_fold(z, |acc, v| {
            let rv = v?;
            meet_acc(acc, rv)
        })
    }
}

#[inline]
fn meet_refs<V: Lattice + Clone>(a: &V, b: &V) -> Option<V> {
    match a.pmeet(b) {
        AlgebraicResult::None => None,
        AlgebraicResult::Identity(mask) => {
            if mask & SELF_IDENT != 0 {
                Some(a.clone())
            } else {
                Some(b.clone())
            }
        }
        AlgebraicResult::Element(v) => Some(v),
    }
}
#[inline]
fn meet_acc<V: Lattice + Clone>(a: V, b: &V) -> Option<V> {
    match a.pmeet(b) {
        AlgebraicResult::None => None,
        AlgebraicResult::Identity(mask) => {
            if mask & SELF_IDENT != 0 {
                Some(a)
            } else {
                Some(b.clone())
            }
        }
        AlgebraicResult::Element(v) => Some(v),
    }
}

// ==================== SUBTRACT ====================

struct Subtract;
impl<V: Clone + Send + Sync> MergePolicy<V> for Subtract {
    #[inline]
    fn on_left_only<Z, Out, A>(z: &mut Z, range: ByteMask, out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        out.graft_children(z, range);
    }

    #[inline]
    fn on_right_only<Z, Out, A>(_z: &mut Z, _range: ByteMask, _out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
    }

    #[inline]
    fn on_single<Z, Out, A>(z: &mut Z, mask: u64, range: ByteMask, out: &mut Out)
    where
        A: Allocator,
        Z: ZipperInfallibleSubtries<V, A> + ZipperMoving,
        Out: ZipperWriting<V, A>,
    {
        if mask == 1 {
            out.graft_children(z, range);
        }
    }

    #[inline]
    fn descend_on_some_equal(mask: u64) -> bool {
        mask & 1 != 0
    }
}

impl<V: DistributiveLattice + Clone> ValuePolicy<V> for Subtract {
    fn combine(l: Option<&V>, r: Option<&V>) -> Option<V> {
        l.and_then(|lv| {
            if let Some(rv) = r {
                match lv.psubtract(rv) {
                    AlgebraicResult::None => None,
                    AlgebraicResult::Identity(mask) => {
                        if mask & SELF_IDENT != 0 {
                            Some(lv.clone())
                        } else {
                            None
                        }
                    }
                    AlgebraicResult::Element(v) => Some(v),
                }
            } else {
                // lhs-only → keep
                Some(lv.clone())
            }
        })
    }

    fn combine_n<'a, I>(vals: I) -> Option<V>
    where
        I: Iterator<Item = Option<&'a V>>,
        V: 'a,
    {
        let mut it = vals;
        let z = it.next()?.cloned()?;
        it.try_fold(z, |acc, v| match v {
            Some(rv) => match acc.psubtract(rv) {
                AlgebraicResult::None => None,
                AlgebraicResult::Identity(mask) => {
                    if mask & SELF_IDENT != 0 {
                        Some(acc)
                    } else {
                        None
                    }
                }
                AlgebraicResult::Element(x) => Some(x),
            },
            None => Some(acc),
        })
    }
}

mod zipper_algebra_poly {
    // ==================== Machinery for zipper_merge_n ====================
    use crate as pathmap;
    use crate::PathMap;
    use crate::alloc::Allocator;
    use crate::trie_node::*;
    use crate::zipper::*;

    #[derive(PolyZipperExplicit)]
    #[poly_zipper_explicit(traits(ZipperMoving, ZipperValues))]
    pub(super) enum SomeZ<'trie, 'path, V: Clone + Send + Sync + Unpin, A: Allocator> {
        RZ(ReadZipperUntracked<'trie, 'path, V, A>),
        RZT(ReadZipperTracked<'trie, 'path, V, A>),
    }

    impl<V: Clone + Send + Sync + Unpin, A: Allocator> zipper_priv::ZipperPriv for SomeZ<'_, '_, V, A> {
        type V = V;

        type A = A;

        #[inline]
        fn get_focus(&self) -> AbstractNodeRef<'_, Self::V, Self::A> {
            match self {
                SomeZ::RZ(inner) => inner.get_focus(),
                SomeZ::RZT(inner) => inner.get_focus(),
            }
        }

        #[inline]
        fn try_borrow_focus(&self) -> Option<&TrieNodeODRc<Self::V, Self::A>> {
            match self {
                SomeZ::RZ(inner) => inner.try_borrow_focus(),
                SomeZ::RZT(inner) => inner.try_borrow_focus(),
            }
        }
    }

    impl<V: Clone + Send + Sync + Unpin, A: Allocator> ZipperInfallibleSubtries<V, A>
        for SomeZ<'_, '_, V, A>
    {
        fn make_map(&self) -> PathMap<V, A> {
            match self {
                SomeZ::RZ(inner) => inner.make_map(),
                SomeZ::RZT(inner) => inner.make_map(),
            }
        }

        fn get_trie_ref(&self) -> TrieRef<'_, V, A> {
            match self {
                SomeZ::RZ(inner) => inner.get_trie_ref(),
                SomeZ::RZT(inner) => inner.get_trie_ref(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        PathMap,
        zipper::{ReadZipperUntracked, WriteZipperUntracked},
    };

    type Paths = &'static [(&'static [u8], u64)];
    type BinaryTest = (Paths, Paths);
    type TernaryTest = (Paths, Paths, Paths);

    fn mk_binary_test(test: &BinaryTest) -> (PathMap<u64>, PathMap<u64>) {
        (PathMap::from_iter(test.0), PathMap::from_iter(test.1))
    }

    fn mk_ternary_test(test: &TernaryTest) -> (PathMap<u64>, PathMap<u64>, PathMap<u64>) {
        (
            PathMap::from_iter(test.0),
            PathMap::from_iter(test.1),
            PathMap::from_iter(test.2),
        )
    }

    fn check2<
        'x,
        T: IntoIterator<Item = &'x (&'x [u8], u64)>,
        F: for<'a> FnOnce(
            &mut ReadZipperUntracked<'a, 'x, u64>,
            &mut ReadZipperUntracked<'a, 'x, u64>,
            &mut WriteZipperUntracked<'a, 'x, u64>,
        ),
    >(
        test: &BinaryTest,
        expected: T,
        op: F,
    ) {
        let (left, right) = mk_binary_test(test);

        let mut result = PathMap::new();

        let mut lhs = left.read_zipper();
        let mut rhs = right.read_zipper();
        let mut out = result.write_zipper();

        op(&mut lhs, &mut rhs, &mut out);

        assert_trie(expected, result);
    }

    fn check3<
        'x,
        T: IntoIterator<Item = &'x (&'x [u8], u64)>,
        F: for<'a> FnOnce(
            &mut ReadZipperUntracked<'a, 'x, u64>,
            &mut ReadZipperUntracked<'a, 'x, u64>,
            &mut ReadZipperUntracked<'a, 'x, u64>,
            &mut WriteZipperUntracked<'a, 'x, u64>,
        ),
    >(
        test: &TernaryTest,
        expected: T,
        op: F,
    ) {
        let (left, middle, right) = mk_ternary_test(test);

        let mut result = PathMap::new();

        let mut lhs = left.read_zipper();
        let mut mid = middle.read_zipper();
        let mut rhs = right.read_zipper();
        let mut out = result.write_zipper();

        op(&mut lhs, &mut mid, &mut rhs, &mut out);

        assert_trie(expected, result);
    }

    fn assert_trie<'a, T: IntoIterator<Item = &'a (&'a [u8], u64)>>(
        expected: T,
        result: PathMap<u64>,
    ) {
        let mut result_copy = result.clone();

        for (expected_path, expected_val) in expected {
            assert!(
                result.path_exists_at(expected_path),
                "Path {expected_path:#?} does NOT exist in {result:#?}"
            );
            let actual_val = result.get_val_at(expected_path);
            assert_eq!(
                actual_val,
                Some(expected_val),
                "Value at {expected_path:#?}"
            );

            result_copy.remove_val_at(expected_path, true);
        }

        assert!(
            result_copy.is_empty(),
            "Paths unaccounted for are present in the result: {result_copy:#?}"
        );
    }

    const DISJOINT_PATHS: BinaryTest = (
        &[
            (&[0x00], 0),
            (&[0x00, 0x00], 1),
            (&[0x00, 0x00, 0x00], 2),
            (&[0x00, 0x00, 0x00, 0x00], 3),
        ],
        &[
            (&[0xFF], 0),
            (&[0xFF, 0x00], 1),
            (&[0xFF, 0x00, 0x00], 2),
            (&[0xFF, 0x00, 0x00, 0x00], 3),
        ],
    );

    const DISJOINT_PATHS_3: TernaryTest = (
        &[
            (&[0x00], 0),
            (&[0x00, 0x00], 1),
            (&[0x00, 0x00, 0x00], 2),
            (&[0x00, 0x00, 0x00, 0x00], 3),
        ],
        &[
            (&[0xF0], 0),
            (&[0xF0, 0x00], 1),
            (&[0xF0, 0x00, 0x00], 2),
            (&[0xF0, 0x00, 0x00, 0x00], 3),
        ],
        &[
            (&[0xFF], 0),
            (&[0xFF, 0x00], 1),
            (&[0xFF, 0x00, 0x00], 2),
            (&[0xFF, 0x00, 0x00, 0x00], 3),
        ],
    );

    const PATHS_WITH_SHARED_PREFIX: BinaryTest = (
        &[(b"aaaaa0", 0), (b"bbbbbbbb0", 1)],
        &[(b"aaaaa1", 0), (b"bbbbb1", 1), (b"bbbbbbbb1", 2)],
    );

    const PATHS_WITH_SHARED_PREFIX_3: TernaryTest = (
        &[(b"aaaaa0", 0), (b"bbbbbbbb0", 1)],
        &[(b"aaaaa1", 0), (b"bbbbb1", 1), (b"bbbbbbbb1", 2)],
        &[(b"aaaaa2", 0), (b"bbbbb2", 1), (b"bbbbbbbb2", 2)],
    );

    const INTERLEAVING_PATHS: BinaryTest = (
        &[(&[0], 0), (&[2], 1), (&[4], 2), (&[6], 3)],
        &[(&[1], 0), (&[3], 1), (&[5], 2), (&[7], 3)],
    );

    const INTERLEAVING_PATHS_3: TernaryTest = (
        &[(&[0], 0), (&[3], 1), (&[6], 2), (&[9], 3)],
        &[(&[1], 0), (&[4], 1), (&[7], 2), (&[10], 3)],
        &[(&[2], 0), (&[5], 1), (&[8], 2), (&[11], 3)],
    );

    const ONE_SIDED_PATHS: BinaryTest = (
        &[
            (&[0x00], 0),
            (&[0x00, 0x01], 1),
            (&[0x00, 0x01, 0x02], 2),
            (&[0x00, 0x01, 0x02, 0x03], 3),
            (&[0x01], 4),
            (&[0x01, 0x02], 5),
            (&[0x01, 0x02, 0x03], 6),
            (&[0x01, 0x02, 0x03, 0x04], 7),
            (&[0x01, 0x02, 0x03, 0x04, 0x05], 8),
            (&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 9),
        ],
        &[
            (&[0x00], 0),
            (&[0x00, 0x01, 0x02, 0x03], 1),
            (&[0x01, 0x02, 0x03, 0x04, 0x05], 2),
        ],
    );

    const ONE_SIDED_PATHS_3: TernaryTest = (
        &[
            (&[0x00], 0),
            (&[0x00, 0x01], 1),
            (&[0x00, 0x01, 0x02], 2),
            (&[0x00, 0x01, 0x02, 0x03], 3),
            (&[0x01], 4),
            (&[0x01, 0x02], 5),
            (&[0x01, 0x02, 0x03], 6),
            (&[0x01, 0x02, 0x03, 0x04], 7),
            (&[0x01, 0x02, 0x03, 0x04, 0x05], 8),
            (&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 9),
        ],
        &[
            (&[0x00], 0),
            (&[0x00, 0x01, 0x02, 0x03], 1),
            (&[0x01, 0x02, 0x03, 0x04, 0x05], 2),
        ],
        &[(&[0x00], 0), (&[0x00, 0x01, 0x02], 1)],
    );

    const ALMOST_IDENTICAL_PATHS: BinaryTest = (
        &[
            (b"abcdefg", 0),
            (b"hijklmnop", 1),
            (b"qrstuwvxyz", 2),
            (b"0", 3),
            (b"1", 4),
            (b"2", 5),
            (b"3", 6),
            (b"4", 7),
            (b"5", 8),
            (b"6789", 9),
        ],
        &[
            (b"abcdefg", 0),
            (b"qrstuwvxyz", 2),
            (b"0", 3),
            (b"1", 4),
            (b"4", 7),
            (b"5", 8),
            (b"6789", 9),
        ],
    );

    const ALMOST_IDENTICAL_PATHS_3: TernaryTest = (
        &[
            (b"abcdefg", 0),
            (b"hijklmnop", 1),
            (b"qrstuwvxyz", 2),
            (b"0", 3),
            (b"1", 4),
            (b"2", 5),
            (b"3", 6),
            (b"4", 7),
            (b"5", 8),
            (b"6789", 9),
        ],
        &[
            (b"abcdefg", 0),
            (b"qrstuwvxyz", 2),
            (b"0", 3),
            (b"1", 4),
            (b"4", 7),
            (b"5", 8),
            (b"6789", 9),
        ],
        &[
            (b"abcdefg", 0),
            (b"hijklmnop", 1),
            (b"1", 4),
            (b"2", 5),
            (b"3", 6),
            (b"4", 7),
            (b"5", 8),
        ],
    );

    const LHS_EMPTY: BinaryTest = (&[], &[(&[1], 0), (&[2], 1)]);
    const LHS_EMPTY_3: TernaryTest = (&[], &[(&[1], 0), (&[2], 1)], &[(&[3], 0), (&[4], 1)]);

    const RHS_EMPTY: BinaryTest = (&[(&[1], 0), (&[2], 1)], &[]);
    const RHS_EMPTY_3: TernaryTest = (&[(&[1], 0), (&[2], 1)], &[(&[3], 0), (&[4], 1)], &[]);

    const MID_EMPTY: TernaryTest = (&[(&[1], 0), (&[2], 1)], &[], &[(&[3], 0), (&[4], 1)]);

    const PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN: BinaryTest = (
        &[
            (&[1, 2, 3], 0),
            (&[1, 2, 3, 4], 1),
            (&[1, 2, 3, 10, 11, 12], 2),
        ],
        &[
            (&[1, 2, 3], 10),
            (&[1, 2, 3, 5], 11),
            (&[1, 2, 3, 10, 11, 0], 12),
        ],
    );

    const PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN_3: TernaryTest = (
        &[
            (&[1, 2, 3], 0),
            (&[1, 2, 3, 4], 1),
            (&[1, 2, 3, 10, 11, 12], 2),
        ],
        &[
            (&[1, 2, 3], 10),
            (&[1, 2, 3, 5], 11),
            (&[1, 2, 3, 10, 11, 0], 12),
        ],
        &[
            (&[1, 2, 3], 20),
            (&[1, 2, 3, 6], 21),
            (&[1, 2, 3, 10, 11, 1], 22),
        ],
    );

    const ZIGZAG_PATHS: BinaryTest = (
        &[
            (&[1, 1], 0),
            (&[2], 1),
            (&[2, 1], 2),
            (&[3], 3),
            (&[3, 2, 1], 4),
            (&[4], 4),
            (&[4, 3, 2, 1], 5),
        ],
        &[
            (&[1], 0),
            (&[1, 2], 1),
            (&[2, 1], 2),
            (&[3], 3),
            (&[3, 4], 4),
            (&[4, 3], 5),
        ],
    );

    const ZIGZAG_PATHS_3: TernaryTest = (
        &[
            (&[1, 1], 0),
            (&[2], 1),
            (&[2, 1], 2),
            (&[3], 3),
            (&[3, 2, 1], 4),
            (&[4], 4),
            (&[4, 3, 2, 1], 5),
        ],
        &[
            (&[1], 0),
            (&[1, 2], 1),
            (&[2, 1], 2),
            (&[3], 3),
            (&[3, 4], 4),
            (&[4, 3], 5),
        ],
        &[
            (&[1], 0),
            (&[2], 1),
            (&[2, 1], 2),
            (&[3, 2, 1, 0], 3),
            (&[4, 3, 2, 1, 0], 4),
            (&[4, 3, 2, 1], 5),
        ],
    );

    const PATHS_WITH_ROOT_VALS_AND_CHILDREN: BinaryTest =
        (&[(&[], 1), (&[1], 10)], &[(&[], 2), (&[1], 20)]);

    const PATHS_WITH_ROOT_VALS_AND_CHILDREN_3: TernaryTest = (
        &[(&[], 1), (&[1], 10)],
        &[(&[], 2), (&[1], 20)],
        &[(&[], 3), (&[1], 30)],
    );

    mod join {
        use super::*;
        use crate::experimental::zipper_algebra::{ZipperAlgebraExt, zipper_join, zipper_join3};

        #[test]
        fn test_disjoint() {
            check2(
                &DISJOINT_PATHS,
                &[DISJOINT_PATHS.0, DISJOINT_PATHS.1].concat(),
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_disjoint3() {
            check3(
                &DISJOINT_PATHS_3,
                &[DISJOINT_PATHS_3.0, DISJOINT_PATHS_3.1, DISJOINT_PATHS_3.2].concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_deep_shared_prefix_then_split() {
            check2(
                &PATHS_WITH_SHARED_PREFIX,
                &[PATHS_WITH_SHARED_PREFIX.0, PATHS_WITH_SHARED_PREFIX.1].concat(),
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_deep_shared_prefix_then_split3() {
            check3(
                &PATHS_WITH_SHARED_PREFIX_3,
                &[
                    PATHS_WITH_SHARED_PREFIX_3.0,
                    PATHS_WITH_SHARED_PREFIX_3.1,
                    PATHS_WITH_SHARED_PREFIX_3.2,
                ]
                .concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_interleaving_paths() {
            check2(
                &INTERLEAVING_PATHS,
                &[INTERLEAVING_PATHS.0, INTERLEAVING_PATHS.1].concat(),
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_interleaving_paths3() {
            check3(
                &INTERLEAVING_PATHS_3,
                &[
                    INTERLEAVING_PATHS_3.0,
                    INTERLEAVING_PATHS_3.1,
                    INTERLEAVING_PATHS_3.2,
                ]
                .concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_one_side_empty_at_many_levels() {
            check2(&ONE_SIDED_PATHS, ONE_SIDED_PATHS.0, |lhs, rhs, out| {
                lhs.join(rhs, out)
            });
        }

        #[test]
        fn test_one_side_empty_at_many_levels3() {
            check3(
                &ONE_SIDED_PATHS_3,
                ONE_SIDED_PATHS_3.0,
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_almost_identical_paths() {
            check2(
                &ALMOST_IDENTICAL_PATHS,
                ALMOST_IDENTICAL_PATHS.0,
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_almost_identical_paths3() {
            check3(
                &ALMOST_IDENTICAL_PATHS_3,
                ALMOST_IDENTICAL_PATHS_3.0,
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_one_side_empty() {
            check2(&LHS_EMPTY, LHS_EMPTY.1, |lhs, rhs, out| lhs.join(rhs, out));
            check2(&RHS_EMPTY, RHS_EMPTY.0, |lhs, rhs, out| lhs.join(rhs, out));
        }

        #[test]
        fn test_one_side_empty3() {
            check3(
                &LHS_EMPTY_3,
                &[LHS_EMPTY_3.1, LHS_EMPTY_3.2].concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
            check3(
                &MID_EMPTY,
                &[MID_EMPTY.0, MID_EMPTY.2].concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
            check3(
                &RHS_EMPTY_3,
                &[RHS_EMPTY_3.0, RHS_EMPTY_3.1].concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_exact_overlap_divergent_subtries() {
            let expected: Paths = &[
                (&[1, 2, 3], 0),
                (&[1, 2, 3, 4], 1),
                (&[1, 2, 3, 5], 11),
                (&[1, 2, 3, 10, 11, 0], 12),
                (&[1, 2, 3, 10, 11, 12], 2),
            ];
            check2(
                &PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN,
                expected,
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_exact_overlap_divergent_subtries3() {
            let expected: Paths = &[
                (&[1, 2, 3], 0),
                (&[1, 2, 3, 4], 1),
                (&[1, 2, 3, 5], 11),
                (&[1, 2, 3, 6], 21),
                (&[1, 2, 3, 10, 11, 0], 12),
                (&[1, 2, 3, 10, 11, 1], 22),
                (&[1, 2, 3, 10, 11, 12], 2),
            ];
            check3(
                &PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN_3,
                expected,
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_zigzag() {
            check2(
                &ZIGZAG_PATHS,
                &[ZIGZAG_PATHS.0, ZIGZAG_PATHS.1].concat(),
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_zigzag3() {
            check3(
                &ZIGZAG_PATHS_3,
                &[ZIGZAG_PATHS.0, ZIGZAG_PATHS.1, ZIGZAG_PATHS_3.2].concat(),
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_root_values() {
            check2(
                &PATHS_WITH_ROOT_VALS_AND_CHILDREN,
                PATHS_WITH_ROOT_VALS_AND_CHILDREN.0,
                |lhs, rhs, out| lhs.join(rhs, out),
            );
        }

        #[test]
        fn test_root_values3() {
            check3(
                &PATHS_WITH_ROOT_VALS_AND_CHILDREN_3,
                PATHS_WITH_ROOT_VALS_AND_CHILDREN_3.0,
                |lhs, mid, rhs, out| zipper_join3(lhs, mid, rhs, out),
            );
        }
    }

    mod meet {
        use super::*;
        use crate::experimental::zipper_algebra::{ZipperAlgebraExt, zipper_meet, zipper_meet3};

        #[test]
        fn test_disjoint() {
            check2(&DISJOINT_PATHS, [], |lhs, rhs, out| {
                lhs.meet(rhs, out);
            });
        }

        #[test]
        fn test_disjoint3() {
            check3(&DISJOINT_PATHS_3, [], |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_deep_shared_prefix_then_split() {
            check2(&PATHS_WITH_SHARED_PREFIX, [], |lhs, rhs, out| {
                lhs.meet(rhs, out);
            });
        }

        #[test]
        fn test_deep_shared_prefix_then_split3() {
            check3(&PATHS_WITH_SHARED_PREFIX_3, [], |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_interleaving_paths() {
            check2(&INTERLEAVING_PATHS, [], |lhs, rhs, out| {
                lhs.meet(rhs, out);
            });
        }

        #[test]
        fn test_interleaving_paths3() {
            check3(&INTERLEAVING_PATHS_3, [], |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_one_side_empty_at_many_levels() {
            let expected: Paths = &[
                (&[0x00], 0),
                (&[0x00, 0x01, 0x02, 0x03], 3),
                (&[0x01, 0x02, 0x03, 0x04, 0x05], 8),
            ];
            check2(&ONE_SIDED_PATHS, expected, |lhs, rhs, out| {
                lhs.meet(rhs, out);
            });
        }

        #[test]
        fn test_one_side_empty_at_many_levels3() {
            let expected: Paths = &[(&[0x00], 0)];
            check3(&ONE_SIDED_PATHS_3, expected, |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_almost_identical_paths() {
            check2(
                &ALMOST_IDENTICAL_PATHS,
                ALMOST_IDENTICAL_PATHS.1,
                |lhs, rhs, out| lhs.meet(rhs, out),
            );
        }

        #[test]
        fn test_almost_identical_paths3() {
            let expected: Paths = &[(b"abcdefg", 0), (b"1", 4), (b"4", 7), (b"5", 8)];
            check3(&ALMOST_IDENTICAL_PATHS_3, expected, |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_one_side_empty() {
            check2(&LHS_EMPTY, [], |lhs, rhs, out| lhs.meet(rhs, out));
            check2(&RHS_EMPTY, [], |lhs, rhs, out| lhs.meet(rhs, out));
        }

        #[test]
        fn test_one_side_empty3() {
            check3(&LHS_EMPTY_3, [], |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out)
            });
            check3(&MID_EMPTY, [], |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out)
            });
            check3(&RHS_EMPTY_3, [], |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out)
            });
        }

        #[test]
        fn test_exact_overlap_divergent_subtries() {
            let expected: Paths = &[(&[1, 2, 3], 0)];
            check2(
                &PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN,
                expected,
                |lhs, rhs, out| lhs.meet(rhs, out),
            );
        }

        #[test]
        fn test_exact_overlap_divergent_subtries3() {
            let expected: Paths = &[(&[1, 2, 3], 0)];
            check3(
                &PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN_3,
                expected,
                |lhs, mid, rhs, out| zipper_meet3(lhs, mid, rhs, out),
            );
        }

        #[test]
        fn test_zigzag() {
            let expected: Paths = &[(&[2, 1], 2), (&[3], 3)];
            check2(&ZIGZAG_PATHS, expected, |lhs, rhs, out| {
                lhs.meet(rhs, out);
            });
        }

        #[test]
        fn test_zigzag3() {
            let expected: Paths = &[(&[2, 1], 2)];
            check3(&ZIGZAG_PATHS_3, expected, |lhs, mid, rhs, out| {
                zipper_meet3(lhs, mid, rhs, out)
            });
        }

        #[test]
        fn test_root_values() {
            check2(
                &PATHS_WITH_ROOT_VALS_AND_CHILDREN,
                PATHS_WITH_ROOT_VALS_AND_CHILDREN.0,
                |lhs, rhs, out| lhs.meet(rhs, out),
            );
        }

        #[test]
        fn test_root_values3() {
            check3(
                &PATHS_WITH_ROOT_VALS_AND_CHILDREN_3,
                PATHS_WITH_ROOT_VALS_AND_CHILDREN.0,
                |lhs, mid, rhs, out| zipper_meet3(lhs, mid, rhs, out),
            );
        }
    }

    mod subtract {
        use super::*;
        use crate::experimental::zipper_algebra::{
            ZipperAlgebraExt, zipper_subtract, zipper_subtract3,
        };

        #[test]
        fn test_disjoint() {
            check2(&DISJOINT_PATHS, DISJOINT_PATHS.0, |lhs, rhs, out| {
                lhs.subtract(rhs, out);
            });
        }

        #[test]
        fn test_disjoint3() {
            check3(
                &DISJOINT_PATHS_3,
                DISJOINT_PATHS_3.0,
                |lhs, mid, rhs, out| {
                    zipper_subtract3(lhs, mid, rhs, out);
                },
            );
        }

        #[test]
        fn test_deep_shared_prefix_then_split() {
            check2(
                &PATHS_WITH_SHARED_PREFIX,
                PATHS_WITH_SHARED_PREFIX.0,
                |lhs, rhs, out| lhs.subtract(rhs, out),
            );
        }

        #[test]
        fn test_deep_shared_prefix_then_split3() {
            check3(
                &PATHS_WITH_SHARED_PREFIX_3,
                PATHS_WITH_SHARED_PREFIX_3.0,
                |lhs, mid, rhs, out| {
                    zipper_subtract3(lhs, mid, rhs, out);
                },
            );
        }

        #[test]
        fn test_interleaving_paths() {
            check2(
                &INTERLEAVING_PATHS,
                INTERLEAVING_PATHS.0,
                |lhs, rhs, out| lhs.subtract(rhs, out),
            );
        }

        #[test]
        fn test_interleaving_paths3() {
            check3(
                &INTERLEAVING_PATHS_3,
                INTERLEAVING_PATHS_3.0,
                |lhs, mid, rhs, out| {
                    zipper_subtract3(lhs, mid, rhs, out);
                },
            );
        }

        #[test]
        fn test_one_side_empty_at_many_levels() {
            let expected: Paths = &[
                (&[0x00, 0x01], 1),
                (&[0x00, 0x01, 0x02], 2),
                (&[0x00, 0x01, 0x02, 0x03], 3),
                (&[0x01], 4),
                (&[0x01, 0x02], 5),
                (&[0x01, 0x02, 0x03], 6),
                (&[0x01, 0x02, 0x03, 0x04], 7),
                (&[0x01, 0x02, 0x03, 0x04, 0x05], 8),
                (&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 9),
            ];
            check2(&ONE_SIDED_PATHS, expected, |lhs, rhs, out| {
                lhs.subtract(rhs, out)
            });
        }

        #[test]
        fn test_one_side_empty_at_many_levels3() {
            let expected: Paths = &[
                (&[0x00, 0x01], 1),
                (&[0x00, 0x01, 0x02], 2),
                (&[0x00, 0x01, 0x02, 0x03], 3),
                (&[0x01], 4),
                (&[0x01, 0x02], 5),
                (&[0x01, 0x02, 0x03], 6),
                (&[0x01, 0x02, 0x03, 0x04], 7),
                (&[0x01, 0x02, 0x03, 0x04, 0x05], 8),
                (&[0x01, 0x02, 0x03, 0x04, 0x05, 0x06], 9),
            ];
            check3(&ONE_SIDED_PATHS_3, expected, |lhs, mid, rhs, out| {
                zipper_subtract3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_almost_identical_paths() {
            let expected: Paths = &[(b"hijklmnop", 1), (b"2", 5), (b"3", 6)];
            check2(&ALMOST_IDENTICAL_PATHS, expected, |lhs, rhs, out| {
                lhs.subtract(rhs, out)
            });
        }

        #[test]
        fn test_almost_identical_paths3() {
            check3(&ALMOST_IDENTICAL_PATHS_3, [], |lhs, mid, rhs, out| {
                zipper_subtract3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_one_side_empty() {
            check2(&LHS_EMPTY, [], |lhs, rhs, out| lhs.subtract(rhs, out));
            check2(&RHS_EMPTY, RHS_EMPTY.0, |lhs, rhs, out| {
                lhs.subtract(rhs, out)
            });
        }

        #[test]
        fn test_one_side_empty3() {
            check3(&LHS_EMPTY_3, [], |lhs, mid, rhs, out| {
                zipper_subtract3(lhs, mid, rhs, out);
            });
            check3(&MID_EMPTY, MID_EMPTY.0, |lhs, mid, rhs, out| {
                zipper_subtract3(lhs, mid, rhs, out);
            });
            check3(&RHS_EMPTY_3, RHS_EMPTY_3.0, |lhs, mid, rhs, out| {
                zipper_subtract3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_exact_overlap_divergent_subtries() {
            check2(
                &PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN,
                PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN.0,
                |lhs, rhs, out| lhs.subtract(rhs, out),
            );
        }

        #[test]
        fn test_exact_overlap_divergent_subtries3() {
            check3(
                &PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN_3,
                PATHS_WITH_SAME_PREFIX_DIFFERENT_CHILDREN_3.0,
                |lhs, mid, rhs, out| {
                    zipper_subtract3(lhs, mid, rhs, out);
                },
            );
        }

        #[test]
        fn test_zigzag() {
            let expected: Paths = &[
                (&[1, 1], 0),
                (&[2], 1),
                (&[3, 2, 1], 4),
                (&[4], 4),
                (&[4, 3, 2, 1], 5),
            ];
            check2(&ZIGZAG_PATHS, expected, |lhs, rhs, out| {
                lhs.subtract(rhs, out)
            });
        }

        #[test]
        fn test_zigzag3() {
            let expected: Paths = &[(&[1, 1], 0), (&[3, 2, 1], 4), (&[4], 4)];
            check3(&ZIGZAG_PATHS_3, expected, |lhs, mid, rhs, out| {
                zipper_subtract3(lhs, mid, rhs, out);
            });
        }

        #[test]
        fn test_root_values() {
            check2(
                &PATHS_WITH_ROOT_VALS_AND_CHILDREN,
                PATHS_WITH_ROOT_VALS_AND_CHILDREN.0,
                |lhs, rhs, out| lhs.subtract(rhs, out),
            );
        }

        #[test]
        fn test_root_values3() {
            check3(
                &PATHS_WITH_ROOT_VALS_AND_CHILDREN_3,
                PATHS_WITH_ROOT_VALS_AND_CHILDREN_3.0,
                |lhs, mid, rhs, out| {
                    zipper_subtract3(lhs, mid, rhs, out);
                },
            );
        }
    }
}
