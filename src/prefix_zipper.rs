use crate::PathMap;
use crate::alloc::Allocator;
use crate::trie_node::TaggedNodeRef;
use crate::utils::ByteMask;
use crate::zipper::*;
use fast_slice_utils::{find_prefix_overlap, starts_with};
use std::borrow::Cow;

#[derive(Clone)]
enum PrefixPos {
    Prefix { valid: usize },
    PrefixOff { valid: usize, invalid: usize },
    Source,
}

impl PrefixPos {
    // #[inline]
    // fn is_prefix(&self) -> bool {
    //     matches!(self, PrefixPos::Prefix {..})
    // }
    #[inline]
    fn is_invalid(&self) -> bool {
        matches!(self, PrefixPos::PrefixOff { .. })
    }
    #[inline]
    fn is_source(&self) -> bool {
        matches!(self, PrefixPos::Source)
    }
    #[inline]
    fn prefixed_depth(&self) -> Option<usize> {
        match self {
            PrefixPos::Prefix { valid } => Some(*valid),
            PrefixPos::PrefixOff { valid, invalid } => Some(*valid + *invalid),
            PrefixPos::Source => None,
        }
    }
}

/// A [Zipper] type that wrapps another `Zipper`, and allows an arbitrary path to prepend the
/// wrapped zipper's space
///
/// ```
/// use pathmap::{PathMap, zipper::*};
///
/// let map: PathMap<()> = [(b"A", ()), (b"B", ())].into_iter().collect();
/// let mut rz = PrefixZipper::new(b"origin.prefix.", map.read_zipper());
/// rz.set_root_prefix_path(b"origin.").unwrap();
///
/// rz.descend_to(b"prefix.A");
/// assert_eq!(rz.path_exists(), true);
/// assert_eq!(rz.origin_path(), b"origin.prefix.A");
/// assert_eq!(rz.path(), b"prefix.A");
/// assert_eq!(rz.root_prefix_path(), b"origin.");
/// ```
#[derive(Clone)]
pub struct PrefixZipper<'prefix, Z> {
    path: Vec<u8>,
    source: Z,
    prefix: Cow<'prefix, [u8]>,
    origin_depth: usize,
    position: PrefixPos,
}

impl<'prefix, Z> PrefixZipper<'prefix, Z>
where
    Z: ZipperMoving,
{
    /// Creates a new `PrefixZipper` wrapping the supplied `source` zipper and prepending the
    /// supplied `prefix`
    pub fn new<P>(prefix: P, mut source: Z) -> Self
    where
        P: Into<Cow<'prefix, [u8]>>,
    {
        let prefix = prefix.into();
        source.reset();
        let position = if prefix.is_empty() {
            PrefixPos::Source
        } else {
            PrefixPos::Prefix { valid: 0 }
        };
        Self {
            path: Vec::new(),
            source,
            prefix,
            origin_depth: 0,
            position,
        }
    }

    pub fn with_origin(mut self, origin: &[u8]) -> Result<Self, &'static str> {
        if !starts_with(&*self.prefix, origin) {
            return Err("set_origin must be called within prefix");
        }
        self.origin_depth = origin.len();
        self.reset();
        Ok(self)
    }

    /// Sets the portion of the zipper's `prefix` to treat as the [`root_prefix_path`](ZipperAbsolutePath::root_prefix_path)
    ///
    /// The remaining portion of the `prefix` will be part of the [`path`](ZipperMoving::path).
    /// This method resets the zipper, and typically it is called immediately after creating the `PrefixZipper`.
    pub fn set_root_prefix_path(&mut self, root_prefix_path: &[u8]) -> Result<(), &'static str> {
        if !starts_with(&*self.prefix, root_prefix_path) {
            return Err("zipper's prefix must begin with root_prefix_path");
        }
        self.origin_depth = root_prefix_path.len();
        self.reset();
        Ok(())
    }

    fn set_valid(&mut self, valid: usize) {
        debug_assert!(
            valid <= self.prefix.len(),
            "valid prefix can't be outside prefix"
        );
        self.position = if valid == self.prefix.len() - self.origin_depth {
            PrefixPos::Source
        } else {
            PrefixPos::Prefix { valid }
        };
    }

    #[inline]
    fn descend_prefix_to_source_root(&mut self) {
        let PrefixPos::Prefix { valid } = self.position else {
            return;
        };

        self.path
            .extend_from_slice(&self.prefix[self.origin_depth + valid..]);
        self.position = PrefixPos::Source;
        debug_assert!(self.source.at_root());
    }

    #[inline]
    fn sync_path_to_source_focus(&mut self) {
        if self.path.len() >= self.prefix.len() {
            self.path.truncate(self.prefix.len());
        } else {
            self.path.clear();
            self.path.extend_from_slice(&self.prefix);
        }
        self.path.extend_from_slice(self.source.path());
    }

    #[inline]
    fn reset_after_iteration_exhausted(&mut self) {
        self.source.reset();
        self.set_valid(0);
        self.path.truncate(self.origin_depth);
    }

    fn to_next_val_slow(&mut self) -> bool
    where
        Self: ZipperMoving + Zipper,
    {
        loop {
            if self.descend_first_byte() {
                if self.is_val() {
                    return true;
                }
                if self.descend_until() && self.is_val() {
                    return true;
                }
            } else {
                loop {
                    if self.to_next_sibling_byte() {
                        if self.is_val() {
                            return true;
                        }
                        break;
                    }
                    self.ascend_byte();
                    if self.at_root() {
                        return false;
                    }
                }
            }
        }
    }

    fn k_path_slow(&mut self, k: usize, base_idx: usize) -> bool
    where
        Self: ZipperMoving,
    {
        loop {
            if self.path().len() < base_idx + k {
                while self.descend_first_byte() {
                    if self.path().len() == base_idx + k {
                        return true;
                    }
                }
            }
            if self.to_next_sibling_byte() {
                if self.path().len() == base_idx + k {
                    return true;
                }
                continue;
            }
            while self.path().len() > base_idx {
                self.ascend_byte();
                if self.path().len() == base_idx {
                    return false;
                }
                if self.to_next_sibling_byte() {
                    break;
                }
            }
        }
    }

    fn ascend_n(&mut self, mut steps: usize) -> Result<(), usize> {
        if let PrefixPos::PrefixOff { valid, mut invalid } = self.position {
            if invalid > steps {
                invalid -= steps;
                self.position = PrefixPos::PrefixOff { valid, invalid };
                return Ok(());
            }
            steps -= invalid;
            self.set_valid(valid.saturating_sub(steps));
            return if let Some(remaining) = steps.checked_sub(valid) {
                Err(remaining)
            } else {
                Ok(())
            };
        }
        if self.position.is_source() {
            // let Err(remaining) = self.source.ascend(steps) else {
            //     return Ok(());
            // };
            let len_before = self.source.path().len();
            if self.source.ascend(steps) {
                return Ok(());
            }
            let len_after = self.source.path().len();
            steps -= len_before - len_after;
            self.position = PrefixPos::Prefix {
                valid: self.prefix.len() - self.origin_depth,
            };
            // Intermediate state: self.position points one off
        }
        if let PrefixPos::Prefix { valid } = self.position {
            self.set_valid(valid.saturating_sub(steps));
            return if let Some(remaining) = steps.checked_sub(valid) {
                Err(remaining)
            } else {
                Ok(())
            };
        }
        Err(steps)
    }
    fn ascend_until_n<const VAL: bool>(&mut self) -> Option<usize> {
        if self.at_root() {
            return None;
        }
        let mut ascended = 0;
        if self.position.is_source() {
            // if let Some(moved) = self.source.ascend_until() {
            //     return Some(moved);
            // }
            let len_before = self.source.path().len();
            let was_good = if VAL {
                self.source.ascend_until()
            } else {
                self.source.ascend_until_branch()
            };
            if was_good && ((VAL && self.source.is_val()) || self.source.child_count() > 1) {
                let len_after = self.source.path().len();
                return Some(len_before - len_after);
            }
            ascended += len_before;
            let valid = self.prefix.len() - self.origin_depth;
            self.position = PrefixPos::Prefix { valid };
        }
        ascended += self
            .position
            .prefixed_depth()
            .expect("we should no longer pointe at source at this point");
        self.set_valid(0);
        Some(ascended)
    }
}

impl<'prefix, Z> PrefixZipper<'prefix, Z> {
    /// Returns the path that must be descended before the PrefixZipper's focus is at the root of the inner zipper, or
    /// `None` if the focus is no longer along the prefix path
    #[inline]
    pub fn prefix_path_below_focus(&self) -> Option<&[u8]> {
        match self.position {
            PrefixPos::Prefix { valid } => Some(&self.prefix[self.origin_depth + valid..]),
            PrefixPos::PrefixOff {
                valid: _,
                invalid: _,
            } => None,
            PrefixPos::Source => Some(&[]),
        }
    }
}

impl<'prefix, Z> ZipperConcrete for PrefixZipper<'prefix, Z>
where
    Z: ZipperConcrete,
{
    fn shared_node_id(&self) -> Option<u64> {
        match self.position {
            PrefixPos::Source => self.source.shared_node_id(),
            _ => None,
        }
    }
    fn is_shared(&self) -> bool {
        match self.position {
            PrefixPos::Source => self.source.is_shared(),
            _ => false,
        }
    }
}

impl<'prefix, Z, V> ZipperValues<V> for PrefixZipper<'prefix, Z>
where
    Z: ZipperValues<V>,
{
    fn val(&self) -> Option<&V> {
        if !self.position.is_source() {
            return None;
        }
        self.source.val()
    }
}

impl<'prefix, 'source, Z, V> ZipperReadOnlyValues<'source, V> for PrefixZipper<'prefix, Z>
where
    Z: ZipperReadOnlyValues<'source, V>,
{
    fn get_val(&self) -> Option<&'source V> {
        if !self.position.is_source() {
            return None;
        }
        self.source.get_val()
    }
}

impl<'prefix, 'source, Z, V> ZipperReadOnlyConditionalValues<'source, V>
    for PrefixZipper<'prefix, Z>
where
    Z: ZipperReadOnlyConditionalValues<'source, V>,
{
    type WitnessT = Z::WitnessT;
    fn witness<'w>(&self) -> Self::WitnessT {
        self.source.witness()
    }
    fn get_val_with_witness<'w>(&self, witness: &'w Self::WitnessT) -> Option<&'w V>
    where
        'source: 'w,
    {
        if !self.position.is_source() {
            return None;
        }
        self.source.get_val_with_witness(witness)
    }
}

impl<'prefix, Z> ZipperPathBuffer for PrefixZipper<'prefix, Z>
where
    Z: ZipperMoving,
{
    unsafe fn origin_path_assert_len(&self, len: usize) -> &[u8] {
        assert!(self.path.capacity() >= len);
        unsafe { core::slice::from_raw_parts(self.path.as_ptr(), len) }
    }
    fn prepare_buffers(&mut self) {
        if self.path.len() < self.origin_depth {
            self.prepare_path_buf_cold()
        }
        debug_assert_eq!(
            &self.prefix[..self.origin_depth],
            &self.path[..self.origin_depth]
        );
    }
    fn reserve_buffers(&mut self, path_len: usize, _stack_depth: usize) {
        self.path.reserve(path_len);
    }
}

impl<'prefix, Z> PrefixZipper<'prefix, Z> {
    #[cold]
    fn prepare_path_buf_cold(&mut self) {
        self.path.clear();
        self.path
            .extend_from_slice(&self.prefix[..self.origin_depth]);
    }
}

impl<'prefix, Z> Zipper for PrefixZipper<'prefix, Z>
where
    Z: Zipper,
{
    fn path_exists(&self) -> bool {
        match self.position {
            PrefixPos::Prefix { .. } => true,
            PrefixPos::PrefixOff { .. } => false,
            PrefixPos::Source => self.source.path_exists(),
        }
    }
    fn is_val(&self) -> bool {
        match self.position {
            PrefixPos::Source => self.source.is_val(),
            _ => false,
        }
    }
    fn child_count(&self) -> usize {
        match self.position {
            PrefixPos::Prefix { .. } => 1,
            PrefixPos::PrefixOff { .. } => 0,
            PrefixPos::Source => self.source.child_count(),
        }
    }
    fn child_mask(&self) -> ByteMask {
        match self.position {
            PrefixPos::Prefix { valid } => {
                let byte = self.prefix[self.origin_depth + valid];
                ByteMask::from(byte)
            }
            PrefixPos::PrefixOff { .. } => ByteMask::EMPTY,
            PrefixPos::Source => self.source.child_mask(),
        }
    }
}

impl<'prefix, Z> ZipperMoving for PrefixZipper<'prefix, Z>
where
    Z: ZipperMoving,
{
    fn at_root(&self) -> bool {
        match self.position {
            PrefixPos::Prefix { valid } => valid == 0,
            PrefixPos::PrefixOff { .. } => false,
            PrefixPos::Source => self.prefix.len() <= self.origin_depth && self.source.at_root(),
        }
    }

    fn reset(&mut self) {
        self.prepare_buffers();
        self.path.truncate(self.origin_depth);
        debug_assert_eq!(self.path, &self.prefix[..self.origin_depth]);
        self.source.reset();
        self.set_valid(0);
    }

    #[inline]
    fn path(&self) -> &[u8] {
        &self.path[self.origin_depth..]
    }

    fn val_count(&self) -> usize {
        self.source.val_count()
    }

    fn descend_to_existing<K: AsRef<[u8]>>(&mut self, patho: K) -> usize {
        if self.position.is_invalid() {
            return 0;
        }
        let mut descended = 0;
        let mut path = patho.as_ref();
        if let PrefixPos::Prefix { valid } = &self.position {
            let valid = *valid;
            let rest_prefix = &self.prefix[self.origin_depth + valid..];
            let overlap = find_prefix_overlap(rest_prefix, path);
            path = &path[overlap..];
            self.set_valid(valid + overlap);
            descended += overlap;
        }
        if self.position.is_source() {
            descended += self.source.descend_to_existing(path);
        }
        self.path.extend_from_slice(&patho.as_ref()[..descended]);
        descended
    }

    fn descend_to<K: AsRef<[u8]>>(&mut self, path: K) {
        let mut path = path.as_ref();
        let existing = self.descend_to_existing(path);
        path = &path[existing..];
        if path.is_empty() {
            return;
        }
        self.path.extend_from_slice(&path);
        self.position = match self.position {
            PrefixPos::Prefix { valid } => PrefixPos::PrefixOff {
                valid,
                invalid: path.len(),
            },
            PrefixPos::PrefixOff { valid, invalid } => PrefixPos::PrefixOff {
                valid,
                invalid: invalid + path.len(),
            },
            PrefixPos::Source => {
                self.source.descend_to(path);
                PrefixPos::Source
            }
        };
    }

    #[inline]
    fn descend_to_byte(&mut self, k: u8) {
        self.descend_to([k])
    }

    fn descend_indexed_byte(&mut self, child_idx: usize) -> bool {
        let mask = self.child_mask();
        let Some(byte) = mask.indexed_bit::<true>(child_idx) else {
            return false;
        };
        self.descend_to_byte(byte);
        debug_assert!(self.path_exists());
        true
    }

    #[inline]
    fn descend_first_byte(&mut self) -> bool {
        self.descend_indexed_byte(0)
    }

    fn descend_until(&mut self) -> bool {
        if self.position.is_invalid() {
            return false;
        }
        if let Some(prefixed_depth) = self.position.prefixed_depth() {
            self.path
                .extend_from_slice(&self.prefix[self.origin_depth + prefixed_depth..]);
            self.position = PrefixPos::Source;
        }
        let len_before = self.source.path().len();
        if !self.source.descend_until() {
            return false;
        }
        let path = self.source.path();
        self.path.extend_from_slice(&path[len_before..]);
        true
    }

    #[inline]
    fn to_next_sibling_byte(&mut self) -> bool {
        if !self.position.is_source() {
            return false;
        }
        if !self.source.to_next_sibling_byte() {
            return false;
        }
        let byte = *self.source.path().last().unwrap();
        *self.path.last_mut().unwrap() = byte;
        true
    }

    #[inline]
    fn to_prev_sibling_byte(&mut self) -> bool {
        if !self.position.is_source() {
            return false;
        }
        if !self.source.to_prev_sibling_byte() {
            return false;
        }
        let byte = *self.source.path().last().unwrap();
        *self.path.last_mut().unwrap() = byte;
        true
    }
    fn ascend(&mut self, steps: usize) -> bool {
        let ascended = match self.ascend_n(steps) {
            Err(remaining) => steps - remaining,
            Ok(()) => steps,
        };
        self.path.truncate(self.path.len() - ascended);
        ascended == steps
    }
    #[inline]
    fn ascend_byte(&mut self) -> bool {
        self.ascend(1)
    }
    #[inline]
    fn ascend_until(&mut self) -> bool {
        let Some(ascended) = self.ascend_until_n::<true>() else {
            return false;
        };
        self.path.truncate(self.path.len() - ascended);
        true
    }
    #[inline]
    fn ascend_until_branch(&mut self) -> bool {
        let Some(ascended) = self.ascend_until_n::<false>() else {
            return false;
        };
        self.path.truncate(self.path.len() - ascended);
        true
    }
}

/// An interface for a [Zipper] to support accessing the full path buffer used to create the zipper
impl<'prefix, Z> ZipperAbsolutePath for PrefixZipper<'prefix, Z>
where
    Z: ZipperAbsolutePath,
{
    fn origin_path(&self) -> &[u8] {
        &self.path
    }
    fn root_prefix_path(&self) -> &[u8] {
        &self.path[..self.origin_depth]
    }
}

impl<'prefix, Z> ZipperIteration for PrefixZipper<'prefix, Z>
where
    Z: ZipperIteration,
{
    fn to_next_val(&mut self) -> bool {
        match self.position {
            PrefixPos::PrefixOff { .. } => self.to_next_val_slow(),
            PrefixPos::Prefix { .. } => {
                self.descend_prefix_to_source_root();
                if self.source.is_val() {
                    return true;
                }
                if self.source.to_next_val() {
                    self.sync_path_to_source_focus();
                    true
                } else {
                    self.reset_after_iteration_exhausted();
                    false
                }
            }
            PrefixPos::Source => {
                if self.source.to_next_val() {
                    self.sync_path_to_source_focus();
                    true
                } else {
                    self.reset_after_iteration_exhausted();
                    false
                }
            }
        }
    }

    fn descend_first_k_path(&mut self, k: usize) -> bool {
        match self.position {
            PrefixPos::PrefixOff { .. } => self.k_path_slow(k, self.path().len()),
            PrefixPos::Prefix { valid } => {
                let original_path_len = self.path.len();
                let original_position = self.position.clone();
                let remaining_prefix = self.prefix.len() - self.origin_depth - valid;

                if k <= remaining_prefix {
                    self.path.extend_from_slice(
                        &self.prefix[self.origin_depth + valid..self.origin_depth + valid + k],
                    );
                    self.set_valid(valid + k);
                    return true;
                }

                self.descend_prefix_to_source_root();
                if self.source.descend_first_k_path(k - remaining_prefix) {
                    self.sync_path_to_source_focus();
                    true
                } else {
                    self.path.truncate(original_path_len);
                    self.position = original_position;
                    false
                }
            }
            PrefixPos::Source => {
                if self.source.descend_first_k_path(k) {
                    self.sync_path_to_source_focus();
                    true
                } else {
                    false
                }
            }
        }
    }

    fn to_next_k_path(&mut self, k: usize) -> bool {
        if let PrefixPos::Source = self.position
            && self.source.path().len() >= k
        {
            let moved = self.source.to_next_k_path(k);
            self.sync_path_to_source_focus();
            return moved;
        }

        let base_idx = if self.path().len() >= k {
            self.path().len() - k
        } else {
            return false;
        };
        self.k_path_slow(k, base_idx)
    }
}

impl<'prefix, 'a, V, Z> ZipperReadOnlyIteration<'a, V> for PrefixZipper<'prefix, Z>
where
    Z: ZipperReadOnlyIteration<'a, V>,
    Self: ZipperReadOnlyValues<'a, V> + ZipperIteration,
{
}

impl<'prefix, 'a, V, Z> ZipperReadOnlyConditionalIteration<'a, V> for PrefixZipper<'prefix, Z>
where
    Z: ZipperReadOnlyConditionalIteration<'a, V>,
    Self: ZipperReadOnlyConditionalValues<'a, V, WitnessT = Z::WitnessT> + ZipperIteration,
{
}

impl<'prefix, Z, V> ZipperForking<V> for PrefixZipper<'prefix, Z>
where
    Z: ZipperIteration + ZipperForking<V>,
{
    type ReadZipperT<'a>
        = PrefixZipper<'prefix, Z::ReadZipperT<'a>>
    where
        Self: 'a;
    fn fork_read_zipper<'a>(&'a self) -> <Self as ZipperForking<V>>::ReadZipperT<'a> {
        PrefixZipper {
            path: Vec::new(),
            position: PrefixPos::Prefix { valid: 0 },
            source: self.source.fork_read_zipper(),
            prefix: self.prefix.clone(),
            origin_depth: 0,
        }
    }
}

// The virtual-prefix fallback below materializes a temporary map, wraps it in `ReadZipperOwned`,
// and extracts a `'static` core from that owned zipper. This impl inherits the `V` and `A`
// `'static` bounds until `ReadZipperOwned` no longer stores a boxed map behind a static core.
impl<'prefix, 'a, V: Clone + Send + Sync + Unpin + 'static, Z, A: Allocator + 'static>
    zipper_priv::ZipperReadOnlyPriv<'a, V, A> for PrefixZipper<'prefix, Z>
where
    Z: zipper_priv::ZipperReadOnlyPriv<'a, V, A>,
    Self: ZipperSubtries<V, A>,
{
    fn borrow_raw_parts<'z>(&'z self) -> (TaggedNodeRef<'z, V, A>, &'z [u8], Option<&'z V>) {
        panic!()
    } //Not sure how we'd implement borrow_raw_parts for a PrefixZipper, in the general case
    fn take_core(&mut self) -> Option<read_zipper_core::ReadZipperCore<'a, 'static, V, A>> {
        if let Some(prefix_path) = self.prefix_path_below_focus() {
            if prefix_path.len() > 0 {
                let temp_map = self.try_make_map();
                return temp_map.and_then(|map| {
                    let mut owned_z: ReadZipperOwned<V, A> = map.into_read_zipper(b"");
                    owned_z.take_core()
                });
            } else {
                self.source.take_core()
            }
        } else {
            self.source.take_core()
        }
    }
}

impl<'prefix, V: Clone + Send + Sync + Unpin, Z, A: Allocator> ZipperSubtries<V, A>
    for PrefixZipper<'prefix, Z>
where
    Z: ZipperSubtries<V, A>,
{
    fn native_subtries(&self) -> bool {
        self.source.native_subtries()
    }
    fn try_make_map(&self) -> Option<PathMap<V, A>> {
        match self.prefix_path_below_focus() {
            Some(prefix_path) => {
                let leaf_map = self.source.try_make_map()?;
                if prefix_path.len() > 0 {
                    let mut new_map = PathMap::new_in(self.source.alloc());
                    let mut wz = new_map.write_zipper_at_path(prefix_path);
                    wz.graft_map(leaf_map);
                    Some(new_map)
                } else {
                    Some(leaf_map)
                }
            }
            None => Some(PathMap::new_in(self.source.alloc())),
        }
    }
    fn trie_ref(&self) -> Option<TrieRef<'_, V, A>> {
        if !self.native_subtries() {
            return None;
        }
        if let Some(prefix_path) = self.prefix_path_below_focus() {
            if prefix_path.len() > 0 {
                return self.try_make_map().map(|temp_map| TrieRef::from(temp_map));
            } else {
                return self.source.trie_ref();
            }
        } else {
            Some(TrieRefOwned::new_invalid_in(self.source.alloc()).into())
        }
    }
    fn alloc(&self) -> A {
        self.source.alloc()
    }
}

impl<'prefix, V: Clone + Send + Sync + Unpin, Z, A: Allocator> ZipperInfallibleSubtries<V, A>
    for PrefixZipper<'prefix, Z>
where
    Z: ZipperInfallibleSubtries<V, A> + ZipperSubtries<V, A>,
{
    fn make_map(&self) -> PathMap<V, A> {
        match self.prefix_path_below_focus() {
            Some(prefix_path) => {
                let leaf_map = self.source.make_map();
                if prefix_path.len() > 0 {
                    let mut new_map = PathMap::new_in(leaf_map.alloc.clone());
                    let mut wz = new_map.write_zipper_at_path(prefix_path);
                    wz.graft_map(leaf_map);
                    new_map
                } else {
                    leaf_map
                }
            }
            None => PathMap::new_in(self.source.alloc()),
        }
    }
    fn get_trie_ref(&self) -> TrieRef<'_, V, A> {
        if let Some(prefix_path) = self.prefix_path_below_focus() {
            if prefix_path.len() > 0 {
                let temp_map = self.make_map();
                return TrieRef::from(temp_map);
            } else {
                return self.source.get_trie_ref();
            }
        } else {
            TrieRefOwned::new_invalid_in(self.source.alloc()).into()
        }
    }
    fn get_focus(&self) -> OpaqueAbstractNodeRef<'_, V, A> {
        self.source.get_focus()
    }
    fn try_borrow_focus(&self) -> Option<OpaqueTrieNodeRef<'_, V, A>> {
        self.source.try_borrow_focus()
    }
}

impl<'prefix, 'a, V: Clone + Send + Sync + 'a, Z, A: Allocator + 'a>
    ZipperReadOnlySubtries<'a, V, A> for PrefixZipper<'prefix, Z>
where
    Z: ZipperReadOnlySubtries<'a, V, A>,
    Self: zipper_priv::ZipperReadOnlyPriv<'a, V, A> + ZipperSubtries<V, A>,
{
    type TrieRefT = <Z as ZipperReadOnlySubtries<'a, V, A>>::TrieRefT;
    fn trie_ref_at_path<K: AsRef<[u8]>>(&self, path: K) -> Self::TrieRefT {
        self.source.trie_ref_at_path(path)
    }
}

crate::zipper::impl_zipper_debug!(
    impl<Z> core::fmt::Debug for PrefixZipper<'_, Z>
        where Z: ZipperAbsolutePath
);

#[cfg(test)]
mod tests {
    use super::PrefixZipper;
    use crate::overlay_zipper::OverlayZipper;
    use crate::trie_map::PathMap;
    use crate::zipper::{
        Zipper, ZipperAbsolutePath, ZipperInfallibleSubtries, ZipperIteration, ZipperMoving,
        ZipperReadOnlyIteration, ZipperSubtries, ZipperWriting,
    };
    const PATHS1: &[(&[u8], u64)] = &[
        (b"0000", 0),
        (b"00000", 1),
        (b"00011", 2),
        (b"11111", 3),
        (b"11222", 4),
    ];
    const PATHS2: &[(&[u8], u64)] = &[(b"000", 0), (b"00000", 0), (b"00111", 1)];

    fn collect_existing_paths<Z>(zipper: &mut Z, paths: &mut Vec<Vec<u8>>)
    where
        Z: ZipperMoving,
    {
        if zipper.path_exists() {
            paths.push(zipper.path().to_vec());
        }

        let child_count = zipper.child_count();
        for child_idx in 0..child_count {
            assert!(zipper.descend_indexed_byte(child_idx));
            collect_existing_paths(zipper, paths);
            assert!(zipper.ascend_byte());
        }
    }

    fn existing_paths<V>(map: &PathMap<V>) -> Vec<Vec<u8>>
    where
        V: Clone + Send + Sync + Unpin,
    {
        let mut zipper = map.read_zipper();
        let mut paths = Vec::new();
        collect_existing_paths(&mut zipper, &mut paths);
        paths.sort();
        paths
    }

    fn assert_same_pathspace<V>(expected: &PathMap<V>, actual: &PathMap<V>)
    where
        V: Clone + Send + Sync + Unpin + Eq + core::fmt::Debug,
    {
        let expected_paths = existing_paths(expected);
        let actual_paths = existing_paths(actual);

        assert_eq!(
            actual_paths, expected_paths,
            "path-existence sets differ; expected {expected:#?}, actual {actual:#?}"
        );

        for path in expected_paths {
            assert_eq!(
                actual.get_val_at(&path),
                expected.get_val_at(&path),
                "value differs at path {path:?}"
            );
        }
    }

    #[test]
    fn dpa_prefix_zipper_materializes_like_insert_prefix_and_derivative_recovers_source() {
        let prefix = b"virtual:";
        let mut source = PathMap::<bool>::new();
        source.set_val_at(b"alpha", true);
        source.set_val_at(b"branch:leaf", false);
        source.create_path(b"dangling:path");

        let materialized = PrefixZipper::new(prefix, source.read_zipper()).make_map();
        let mut inserted = source.clone();
        {
            let mut wz = inserted.write_zipper();
            assert!(wz.insert_prefix(prefix));
        }
        assert_same_pathspace(&inserted, &materialized);

        let mut rooted_source = source.clone();
        rooted_source.set_val_at(b"", false);
        let mut prefixed = PrefixZipper::new(prefix, rooted_source.read_zipper());
        assert_eq!(prefixed.descend_to_existing(prefix), prefix.len());
        assert!(prefixed.path_exists());
        assert_same_pathspace(&rooted_source, &prefixed.make_map());
    }

    #[test]
    fn prefix_try_make_map_materializes_virtual_source() {
        let prefix = b"prefixed:";
        let mut left = PathMap::<bool>::new();
        left.set_val_at(b"shared", true);
        left.set_val_at(b"left", false);
        left.create_path(b"left:dangling");

        let mut right = PathMap::<bool>::new();
        right.set_val_at(b"shared", false);
        right.set_val_at(b"right", true);
        right.create_path(b"right:dangling");

        let joined = left.join(&right);
        let mut expected = joined.clone();
        {
            let mut wz = expected.write_zipper();
            assert!(wz.insert_prefix(prefix));
        }

        let overlay = OverlayZipper::new(left.read_zipper(), right.read_zipper());
        let prefixed = PrefixZipper::new(prefix.as_slice(), overlay);
        assert!(!prefixed.native_subtries());
        let materialized = prefixed
            .try_make_map()
            .expect("prefix over materializable virtual source should materialize");
        assert_same_pathspace(&expected, &materialized);

        let overlay = OverlayZipper::new(left.read_zipper(), right.read_zipper());
        let mut focused = PrefixZipper::new(prefix.as_slice(), overlay);
        assert_eq!(focused.descend_to_existing(prefix), prefix.len());
        let focused_materialized = focused
            .try_make_map()
            .expect("focused prefix over materializable virtual source should materialize");
        assert_same_pathspace(&joined, &focused_materialized);
    }

    #[test]
    fn prefix_materialization_is_empty_after_prefix_divergence() {
        let mut source = PathMap::<bool>::new();
        source.set_val_at(b"alpha", true);
        source.create_path(b"dangling");

        let mut prefixed = PrefixZipper::new(b"abc", source.read_zipper());
        prefixed.descend_to(b"abd");
        assert!(!prefixed.path_exists());

        let expected = PathMap::<bool>::new();
        assert_same_pathspace(&expected, &prefixed.make_map());
        let try_materialized = prefixed
            .try_make_map()
            .expect("diverged prefix residual should still materialize as empty");
        assert_same_pathspace(&expected, &try_materialized);
    }

    #[test]
    fn prefix_iteration_jumps_virtual_prefix_and_visits_source_values() {
        let prefix = b"prefix/";
        let mut source = PathMap::<u64>::new();
        source.set_val_at(b"", 0);
        source.set_val_at(b"alpha", 1);
        source.set_val_at(b"beta/gamma", 2);

        let mut prefixed = PrefixZipper::new(prefix, source.read_zipper());
        let mut visited = Vec::new();
        while let Some(&value) = prefixed.to_next_get_val() {
            visited.push((prefixed.path().to_vec(), value));
        }

        assert_eq!(
            visited,
            vec![
                (b"prefix/".to_vec(), 0),
                (b"prefix/alpha".to_vec(), 1),
                (b"prefix/beta/gamma".to_vec(), 2),
            ]
        );
        assert!(prefixed.at_root());
    }

    #[test]
    fn prefix_k_path_iteration_spans_virtual_prefix_and_source() {
        let prefix = b"pre/";
        let mut source = PathMap::<u64>::new();
        source.set_val_at(b"ab", 1);
        source.set_val_at(b"ac", 2);

        let mut prefixed = PrefixZipper::new(prefix, source.read_zipper());
        assert!(prefixed.descend_first_k_path(6));
        assert_eq!(prefixed.path(), b"pre/ab");
        assert!(prefixed.to_next_k_path(6));
        assert_eq!(prefixed.path(), b"pre/ac");
        assert!(!prefixed.to_next_k_path(6));
        assert!(prefixed.at_root());
    }

    #[test]
    fn test_prefix_zipper1() {
        let map = PathMap::from_iter(PATHS1.iter().map(|&x| x));
        let mut rz = PrefixZipper::new(b"prefix", map.read_zipper());
        rz.set_root_prefix_path(b"pre").unwrap();
        assert_eq!(rz.descend_to_existing(b"fix00000"), 8);
        assert_eq!(rz.ascend_until(), true);
        assert_eq!(rz.path(), b"fix0000");
        assert_eq!(rz.origin_path(), b"prefix0000");
        assert_eq!(rz.descend_to_existing(b"0"), 1);
        assert_eq!(rz.ascend_until_branch(), true);
        assert_eq!(rz.path(), b"fix000");
        assert_eq!(rz.ascend_until_branch(), true);
        assert_eq!(rz.path(), b"fix");
        assert_eq!(rz.ascend_until_branch(), true);
        assert_eq!(rz.path(), b"");
        assert_eq!(rz.origin_path(), b"pre");
        assert_eq!(rz.ascend_until_branch(), false);
    }

    #[test]
    fn test_prefix_zipper2() {
        let map = PathMap::from_iter(PATHS2.iter().map(|&x| x));
        let mut rz = PrefixZipper::new(b"prefix", map.read_zipper());
        rz.set_root_prefix_path(b"pre").unwrap();
        assert_eq!(rz.descend_to_existing(b"fix00000"), 8);
        assert_eq!(rz.ascend_until(), true);
        assert_eq!(rz.path(), b"fix000");
        assert_eq!(rz.origin_path(), b"prefix000");
        assert_eq!(rz.ascend_until(), true);
        assert_eq!(rz.path(), b"fix00");
        assert_eq!(rz.ascend_until(), true);
        assert_eq!(rz.path(), b"");
        assert_eq!(rz.ascend_until(), false);
        assert_eq!(rz.descend_to_existing(b"fix00000"), 8);
        assert_eq!(rz.ascend_until_branch(), true);
        assert_eq!(rz.path(), b"fix00");
        assert_eq!(rz.ascend_until_branch(), true);
        assert_eq!(rz.path(), b"");
        assert_eq!(rz.origin_path(), b"pre");
        assert_eq!(rz.ascend_until_branch(), false);
    }
}
