use crate::{
    alloc::{Allocator, GlobalAlloc},
    ring::{AlgebraicResult, COUNTER_IDENT, DistributiveLattice, Lattice, SELF_IDENT},
    utils::ByteMask,
    zipper::{
        ReadZipperUntracked, Zipper, ZipperInfallibleSubtries, ZipperMoving, ZipperValues,
        ZipperWriting,
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
