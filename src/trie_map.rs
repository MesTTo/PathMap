use crate::alloc::{global_alloc, Allocator, GlobalAlloc};
use crate::merkleization::{merkleize_impl, MerkleizeResult};
use crate::morphisms::{new_map_from_ana_in, TrieBuilder};
use crate::ring::{
    AlgebraicResult, AlgebraicStatus, DistributiveLattice, DistributiveLatticeRef, Lattice,
    LatticeRef, Quantale, COUNTER_IDENT, SELF_IDENT,
};
use crate::trie_node::*;
use crate::zipper::*;
use core::cell::UnsafeCell;

use crate::gxhash;

/// A map type that uses a trie based on byte slices (`&[u8]`) known as "paths"
///
/// This type is implemented using some of the approaches explained in the
/// ["Bitwise trie with bitmap" Wikipedia article](https://en.wikipedia.org/wiki/Bitwise_trie_with_bitmap).
///
/// ```
/// # use pathmap::PathMap;
/// let mut map = PathMap::<String>::new();
/// map.set_val_at("one", "1".to_string());
/// map.set_val_at("two", "2".to_string());
///
/// assert!(map.contains("one"));
/// assert_eq!(map.get("two"), Some(&"2".to_string()));
/// assert!(!map.contains("three"));
/// ```
pub struct PathMap<V: Clone + Send + Sync, A: Allocator = GlobalAlloc> {
    pub(crate) root: UnsafeCell<Option<TrieNodeODRc<V, A>>>,
    pub(crate) root_val: UnsafeCell<Option<V>>,
    pub(crate) alloc: A,
}

unsafe impl<V: Clone + Send + Sync, A: Allocator> Send for PathMap<V, A> {}
unsafe impl<V: Clone + Send + Sync, A: Allocator> Sync for PathMap<V, A> {}

impl<V: Clone + Send + Sync + Unpin, A: Allocator> Clone for PathMap<V, A> {
    fn clone(&self) -> Self {
        let root_ref = unsafe { &*self.root.get() };
        let root_val_ref = unsafe { &*self.root_val.get() };
        Self::new_with_root_in(root_ref.clone(), root_val_ref.clone(), self.alloc.clone())
    }
}

impl<V: Clone + Send + Sync + Unpin + core::fmt::Debug, A: Allocator> core::fmt::Debug
    for PathMap<V, A>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        const MAX_DEBUG_PATHS: usize = 100;

        let mut rz = self.read_zipper();

        //Try first assuming the paths are all ascii
        let mut contains_all_ascii = true;
        let mut dbg_map = f.debug_map();
        let mut path_cnt = 0;
        while rz.to_next_val() && path_cnt < MAX_DEBUG_PATHS {
            if let Some(key) = crate::utils::debug::render_debug_path(
                rz.path(),
                crate::utils::debug::PathRenderMode::RequireAscii,
            ) {
                dbg_map.entry(&key, rz.val().unwrap());
                path_cnt += 1;
            } else {
                contains_all_ascii = false;
                break;
            }
        }
        if contains_all_ascii {
            return dbg_map.finish();
        }

        //If that failed, render them again with non-ascii paths
        rz.reset();
        let mut dbg_struct = f.debug_struct("PathMap");
        let mut path_cnt = 0;
        while rz.to_next_val() && path_cnt < MAX_DEBUG_PATHS {
            let key = crate::utils::debug::render_debug_path(
                rz.path(),
                crate::utils::debug::PathRenderMode::ByteList,
            )
            .unwrap();
            dbg_struct.field(&key, rz.val().unwrap());
            path_cnt += 1;
        }
        dbg_struct.finish()
    }
}

impl<V: Clone + Send + Sync + Unpin> PathMap<V, GlobalAlloc> {
    /// Creates a new empty map
    #[inline]
    pub const fn new() -> Self {
        Self::new_with_root_in(None, None, global_alloc())
    }

    /// Creates a new single-element pathmap
    #[inline]
    pub fn single<P: AsRef<[u8]>>(path: P, val: V) -> Self {
        Self::from((path, val))
    }

    /// Creates a new `PathMap` by evaluating the specified anamorphism
    ///
    /// `alg_f`: `alg(w: W, val: &mut Option<V>, children: &mut ChildBuilder<W>, path: &[u8])`
    /// generates the value downstream and downstream children from a path
    ///
    /// Setting the `val` option to `Some` within the closure sets the value at the current path.
    ///
    /// The example below creates a trie with binary tree, 3 levels deep, where each level has a 'L'
    /// and an 'R' branch, and the leaves have a unit value.
    /// ```
    /// # use pathmap::PathMap;
    /// let map = PathMap::<()>::new_from_ana(3, |idx, val, children, _path| {
    ///     if idx > 0 {
    ///         children.push(b"L", idx - 1);
    ///         children.push(b"R", idx - 1);
    ///     } else {
    ///         *val = Some(());
    ///     }
    /// });
    /// ```
    pub fn new_from_ana<W, AlgF>(w: W, alg_f: AlgF) -> Self
    where
        V: 'static,
        AlgF: FnMut(W, &mut Option<V>, &mut TrieBuilder<V, W, GlobalAlloc>, &[u8]),
    {
        Self::new_from_ana_in(w, alg_f, global_alloc())
    }
}

impl<V: Clone + Send + Sync + Unpin, A: Allocator> PathMap<V, A> {
    #[inline]
    pub(crate) fn root(&self) -> Option<&TrieNodeODRc<V, A>> {
        unsafe { &*self.root.get() }.as_ref()
    }
    #[inline]
    pub(crate) fn root_val(&self) -> Option<&V> {
        unsafe { &*self.root_val.get() }.as_ref()
    }
    fn root_val_mut_ref(&mut self) -> Option<&mut V> {
        self.root_val.get_mut().as_mut()
    }
    #[inline]
    fn set_root_val(&mut self, val: V) -> Option<V> {
        self.root_val.get_mut().replace(val)
    }
    #[inline]
    fn remove_root_val(&mut self) -> Option<V> {
        self.root_val.get_mut().take()
    }
    #[inline]
    fn root_val_or_set_mut_with<F>(&mut self, func: F) -> &mut V
    where
        F: FnOnce() -> V,
    {
        if self.root_val().is_none() {
            self.set_root_val(func());
        }
        self.root_val_mut_ref()
            .expect("root value was inserted before mutable access")
    }
    #[inline]
    pub(crate) fn get_or_init_root_mut(&mut self) -> &mut TrieNodeODRc<V, A> {
        self.ensure_root();
        self.root.get_mut().as_mut().unwrap()
    }
    /// Internal method to ensure there is a valid node at the root of the map
    #[inline]
    pub(crate) fn ensure_root(&self) {
        let root_ref = unsafe { &*self.root.get() };
        if root_ref.is_some() {
            return;
        }
        self.do_init_root();
    }
    #[cold]
    fn do_init_root(&self) {
        #[cfg(feature = "all_dense_nodes")]
        let root = TrieNodeODRc::new_in(
            crate::dense_byte_node::DenseByteNode::<V, A>::new_in(self.alloc.clone()),
            self.alloc.clone(),
        );
        #[cfg(any())]
        let root = TrieNodeODRc::new_in(
            crate::bridge_node::BridgeNode::new_empty(),
            self.alloc.clone(),
        );
        #[cfg(not(feature = "all_dense_nodes"))]
        let root = TrieNodeODRc::new_in(
            crate::line_list_node::LineListNode::new_in(self.alloc.clone()),
            self.alloc.clone(),
        );

        let root_ref = unsafe { &mut *self.root.get() };
        *root_ref = Some(root);
    }

    /// Creates a new empty map in the specified allocator
    #[inline]
    pub const fn new_in(alloc: A) -> Self {
        Self::new_with_root_in(None, None, alloc)
    }

    /// Creates a new single-element pathmap in the specified allocator
    #[inline]
    pub fn single_in<P: AsRef<[u8]>>(path: P, val: V, alloc: A) -> Self {
        let mut map = Self::new_in(alloc);
        map.set_val_at(path, val);
        map
    }

    /// See [`new_from_ana`](Self::new_from_ana) for description of behavior
    pub fn new_from_ana_in<W, AlgF>(w: W, alg_f: AlgF, alloc: A) -> Self
    where
        V: 'static,
        AlgF: FnMut(W, &mut Option<V>, &mut TrieBuilder<V, W, A>, &[u8]),
    {
        new_map_from_ana_in(w, alg_f, alloc)
    }

    /// Internal Method.  Creates a new `PathMap` with the supplied root node
    #[inline]
    pub(crate) const fn new_with_root_in(
        root_node: Option<TrieNodeODRc<V, A>>,
        root_val: Option<V>,
        alloc: A,
    ) -> Self {
        Self {
            root: UnsafeCell::new(root_node),
            root_val: UnsafeCell::new(root_val),
            alloc,
        }
    }

    /// Internal Method.  Removes and returns the root node and root_val from a `PathMap`
    #[inline]
    pub(crate) fn into_root(self) -> (Option<TrieNodeODRc<V, A>>, Option<V>) {
        let root_node = match self.root() {
            Some(root) => {
                if !root.as_tagged().node_is_empty() {
                    self.root.into_inner()
                } else {
                    None
                }
            }
            None => None,
        };
        let root_val = self.root_val.into_inner();
        (root_node, root_val)
    }

    /// Creates a new [TrieRef], referring to a position from the root of the `PathMap`
    pub fn trie_ref_at_path<K: AsRef<[u8]>>(&self, path: K) -> TrieRefBorrowed<'_, V, A> {
        self.ensure_root();
        let path = path.as_ref();
        TrieRefBorrowed::new_with_key_and_path_in(
            self.root().unwrap(),
            self.root_val(),
            &[],
            path,
            self.alloc.clone(),
        )
    }

    /// Creates a new read-only [Zipper], starting at the root of a `PathMap`
    pub fn read_zipper<'a>(&'a self) -> ReadZipperUntracked<'a, 'static, V, A> {
        self.ensure_root();
        ReadZipperUntracked::new_with_node_and_path_internal_in(
            self.root().unwrap(),
            &[],
            0,
            self.root_val(),
            self.alloc.clone(),
        )
    }

    /// Creates a new read-only [Zipper], with the specified path from the root of the map; This method is much more
    /// efficient than [read_zipper_at_path](Self::read_zipper_at_path), but means the resulting zipper is bound by
    /// the `'path` lifetime
    pub fn read_zipper_at_borrowed_path<'path>(
        &self,
        path: &'path [u8],
    ) -> ReadZipperUntracked<'_, 'path, V, A> {
        self.ensure_root();
        let root_val = match path.len() == 0 {
            true => self.root_val(),
            false => None,
        };
        ReadZipperUntracked::new_with_node_and_path_in(
            self.root().unwrap(),
            path.as_ref(),
            path.len(),
            0,
            root_val,
            self.alloc.clone(),
        )
    }

    /// Creates a new read-only [Zipper], with the `path` specified from the root of the map
    pub fn read_zipper_at_path<K: AsRef<[u8]>>(
        &self,
        path: K,
    ) -> ReadZipperUntracked<'_, 'static, V, A> {
        self.ensure_root();
        let path = path.as_ref();
        let root_val = match path.len() == 0 {
            true => self.root_val(),
            false => None,
        };
        ReadZipperUntracked::new_with_node_and_cloned_path_in(
            self.root().unwrap(),
            path,
            path.len(),
            0,
            root_val,
            self.alloc.clone(),
        )
    }

    /// Creates a new [write zipper](ZipperWriting) starting at the root of the `PathMap`
    pub fn write_zipper(&mut self) -> WriteZipperUntracked<'_, 'static, V, A> {
        self.ensure_root();
        let root_node = self.root.get_mut().as_mut().unwrap();
        let root_val = self.root_val.get_mut();
        WriteZipperUntracked::new_with_node_and_path_internal_in(
            root_node,
            Some(root_val),
            &[],
            0,
            self.alloc.clone(),
        )
    }

    /// Creates a new [write zipper](ZipperWriting) with the specified path from the root of the map
    pub fn write_zipper_at_path<'a, 'path>(
        &'a mut self,
        path: &'path [u8],
    ) -> WriteZipperUntracked<'a, 'path, V, A> {
        self.ensure_root();
        let root_node = self.root.get_mut().as_mut().unwrap();
        let root_val = match path.len() == 0 {
            true => Some(self.root_val.get_mut()),
            false => None,
        };
        WriteZipperUntracked::new_with_node_and_path_in(
            root_node,
            root_val,
            path,
            path.len(),
            0,
            self.alloc.clone(),
        )
    }

    /// Creates a [ZipperHead] at the root of the map
    pub fn zipper_head(&mut self) -> ZipperHead<'_, '_, V, A> {
        self.ensure_root();
        let root_node = self.root.get_mut().as_mut().unwrap();
        let root_val = self.root_val.get_mut();
        let z = WriteZipperCore::new_with_node_and_path_internal_in(
            root_node,
            Some(root_val),
            &[],
            0,
            self.alloc.clone(),
        );
        z.into_zipper_head()
    }

    /// Transforms the map into a [Zipper], which is handy when you need to embed the zipper in another
    /// struct without a lifetime parameter
    pub fn into_read_zipper<K: AsRef<[u8]>>(self, path: K) -> ReadZipperOwned<V, A> {
        ReadZipperOwned::new_with_map(self, path)
    }

    /// Transforms the map into a [WriteZipperOwned], which is handy when you need to embed the zipper
    /// in another struct without a lifetime parameter
    pub fn into_write_zipper<K: AsRef<[u8]>>(self, path: K) -> WriteZipperOwned<V, A> {
        WriteZipperOwned::new_with_map(self, path)
    }

    /// Transforms the map into a [ZipperHead] that owns the map's contents.  This is handy when the
    /// ZipperHead needs to be part of another structure
    pub fn into_zipper_head<K: AsRef<[u8]>>(self, path: K) -> ZipperHeadOwned<V, A> {
        let path = path.as_ref();
        let mut wz = self.into_write_zipper(&path);
        if path.len() > 0 {
            wz.core().key.prepare_buffers();
        }
        ZipperHeadOwned::new(wz)
    }

    /// Returns an iterator over all key-value pairs within the map
    ///
    /// NOTE: This is much less efficient than using the [read_zipper](Self::read_zipper) method
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (Vec<u8>, &'a V)> + 'a {
        self.read_zipper().into_iter()
    }

    /// Calls `visit` for every key-value pair in the map without allocating key
    /// buffers.
    ///
    /// The path slice is borrowed from an internal read zipper and is valid only
    /// for the duration of the callback. Use [`PathMap::iter`] if an owned
    /// `Vec<u8>` key is required.
    pub fn for_each_value<F>(&self, mut visit: F)
    where
        F: FnMut(&[u8], &V),
    {
        let mut zipper = self.read_zipper();
        if let Some(value) = zipper.val() {
            visit(zipper.path(), value);
        }
        while zipper.to_next_val() {
            if let Some(value) = zipper.val() {
                visit(zipper.path(), value);
            }
        }
    }

    /// Fallible variant of [`PathMap::for_each_value`].
    ///
    /// Returning `Err` from `visit` stops traversal immediately and returns that
    /// error to the caller.
    pub fn try_for_each_value<F, E>(&self, mut visit: F) -> Result<(), E>
    where
        F: FnMut(&[u8], &V) -> Result<(), E>,
    {
        let mut zipper = self.read_zipper();
        if let Some(value) = zipper.val() {
            visit(zipper.path(), value)?;
        }
        while zipper.to_next_val() {
            if let Some(value) = zipper.val() {
                visit(zipper.path(), value)?;
            }
        }
        Ok(())
    }

    /// Returns `true` if the map contains a value at the specified key, otherwise returns `false`
    pub fn contains<K: AsRef<[u8]>>(&self, k: K) -> bool {
        self.get_val_at(k).is_some()
    }

    /// Returns `true` if `path` is contained within the map, or `false` otherwise
    pub fn path_exists_at<K: AsRef<[u8]>>(&self, path: K) -> bool {
        let path = path.as_ref();
        if path.is_empty() {
            return true;
        }

        self.ensure_root();
        let (node, remaining_key, _val) = node_along_path(self.root().unwrap(), path, None, false);
        remaining_key.is_empty() || node.as_tagged().node_contains_partial_key(remaining_key)
    }

    /// Deprecated alias for [`PathMap::path_exists_at`]
    #[deprecated]
    pub fn contains_path<K: AsRef<[u8]>>(&self, k: K) -> bool {
        self.path_exists_at(k)
    }

    /// Inserts `v` into the map at `path`.  Panics if `path` has a zero length
    ///
    /// Returns `Some(replaced_val)` if an existing value was replaced, otherwise returns `None` if
    /// the value was added to the map without replacing anything.
    pub fn set_val_at<K: AsRef<[u8]>>(&mut self, path: K, v: V) -> Option<V> {
        let path = path.as_ref();
        if path.len() == 0 {
            return self.set_root_val(v);
        }

        {
            let root_node = self.get_or_init_root_mut();
            let (node_key, node) = node_along_path_mut(root_node, path, true);
            if !node.is_empty() {
                return match node.make_mut().node_set_val(node_key, v) {
                    Ok((old_val, _created_subnode)) => old_val,
                    Err(replacement_node) => {
                        *node = replacement_node;
                        None
                    }
                };
            }
        }

        let mut zipper = self.write_zipper();
        zipper.descend_to(path);
        zipper.set_val(v)
    }

    /// Alias for [Self::set_val_at], so `PathMap` "feels" like other Rust collections
    pub fn insert<K: AsRef<[u8]>>(&mut self, k: K, v: V) -> Option<V> {
        self.set_val_at(k, v)
    }

    /// Joins `v` with the value at `path`, inserting it if no value exists.
    ///
    /// Unlike [`PathMap::set_val_at`], this preserves the existing value when
    /// the lattice join reports it as the identity, replaces it when the new
    /// value is the identity, and removes the value when the join annihilates.
    pub fn join_val_at<K: AsRef<[u8]>>(&mut self, path: K, v: V) -> AlgebraicStatus
    where
        V: Lattice,
    {
        let path = path.as_ref();
        let mut remove = false;
        let status = if let Some(existing) = self.get_val_mut_at(path) {
            match existing.pjoin(&v) {
                AlgebraicResult::None => {
                    remove = true;
                    AlgebraicStatus::None
                }
                AlgebraicResult::Element(joined) => {
                    *existing = joined;
                    AlgebraicStatus::Element
                }
                AlgebraicResult::Identity(mask) if mask & SELF_IDENT > 0 => {
                    AlgebraicStatus::Identity
                }
                AlgebraicResult::Identity(mask) => {
                    debug_assert!(mask & COUNTER_IDENT > 0);
                    *existing = v;
                    AlgebraicStatus::Element
                }
            }
        } else {
            self.set_val_at(path, v);
            AlgebraicStatus::Element
        };

        if remove {
            let _ = self.remove_val_at(path, true);
        }
        status
    }

    /// Removes the value at `path` from the map and returns it, or returns `None` if there was no value at `path`
    ///
    /// If `prune` is `true`, the path will be pruned, otherwise it will be left dangling.
    pub fn remove_val_at<K: AsRef<[u8]>>(&mut self, path: K, prune: bool) -> Option<V> {
        let path = path.as_ref();
        if path.len() == 0 {
            return self.remove_root_val();
        }

        if !prune {
            let root_node = self.root.get_mut().as_mut()?;
            let (node_key, node) = node_along_path_mut(root_node, path, true);
            return node.make_mut().node_remove_val(node_key, false);
        }

        //NOTE: we're descending the zipper rather than creating it at the path so it will be allowed to
        // prune the branches.  A WriteZipper can't move above its root, so it couldn't prune otherwise
        let mut zipper = self.write_zipper();
        zipper.descend_to(path);
        zipper.remove_val(prune)
    }

    /// Alias for [Self::remove_val_at], so `PathMap` "feels" like other Rust collections
    pub fn remove<K: AsRef<[u8]>>(&mut self, path: K) -> Option<V> {
        self.remove_val_at(path, true)
    }

    /// Returns a reference to the value at the specified `path`, or `None` if no value exists
    pub fn get_val_at<K: AsRef<[u8]>>(&self, path: K) -> Option<&V> {
        let path = path.as_ref();
        if path.is_empty() {
            return self.root_val();
        }

        self.ensure_root();
        let (node, remaining_key, val) = node_along_path(self.root().unwrap(), path, None, false);
        if remaining_key.is_empty() {
            val
        } else {
            node.as_tagged().node_get_val(remaining_key)
        }
    }

    /// Alias for [Self::get_val_at], so `PathMap` "feels" like other Rust collections
    pub fn get<K: AsRef<[u8]>>(&self, path: K) -> Option<&V> {
        self.get_val_at(path)
    }

    /// Returns a mutable reference to the value at the specified `path` in the `PathMap`, if it exists
    pub fn get_val_mut_at<K: AsRef<[u8]>>(&mut self, path: K) -> Option<&mut V> {
        let path = path.as_ref();
        if path.len() == 0 {
            return self.root_val_mut_ref();
        }

        self.ensure_root();
        let root_node = self.root.get_mut().as_mut().unwrap();
        let (node_key, node) = node_along_path_mut(root_node, path, true);
        node.make_mut().node_into_val_ref_mut(node_key)
    }

    /// Alias for [Self::get_val_mut_at], so `PathMap` "feels" like other Rust collections
    pub fn get_mut<K: AsRef<[u8]>>(&mut self, path: K) -> Option<&mut V> {
        self.get_val_mut_at(path)
    }

    /// Returns a mutable reference to the value at the specified `path`, inserting the result
    /// of `func` if no value exists
    pub fn get_val_or_set_mut_with_at<F, K>(&mut self, path: K, func: F) -> &mut V
    where
        F: FnOnce() -> V,
        K: AsRef<[u8]>,
    {
        let path = path.as_ref();
        if path.len() == 0 {
            return self.root_val_or_set_mut_with(func);
        }

        let root_node = self.get_or_init_root_mut();
        let (node_key, node) = node_along_path_mut(root_node, path, true);
        let needs_redescend = if node.as_tagged().node_contains_val(node_key) {
            false
        } else {
            match node.make_mut().node_set_val(node_key, func()) {
                Ok((_old_val, created_subnode)) => created_subnode,
                Err(replacement_node) => {
                    *node = replacement_node;
                    true
                }
            }
        };

        if needs_redescend {
            let (final_key, final_node) = node_along_path_mut(node, node_key, true);
            final_node
                .make_mut()
                .node_into_val_ref_mut(final_key)
                .expect("value was inserted before mutable access")
        } else {
            node.make_mut()
                .node_into_val_ref_mut(node_key)
                .expect("value existed or was inserted before mutable access")
        }
    }

    /// Returns a mutable reference to the value at the specified `path`, inserting `default`
    /// if no value already exists
    pub fn get_val_or_set_mut_at<K: AsRef<[u8]>>(&mut self, path: K, default: V) -> &mut V {
        self.get_val_or_set_mut_with_at(path, || default)
    }

    /// Removes all downstream branches below `path`.  Does not affect a value at `path`
    ///
    /// Returns `true` if at least one branch was removed.
    ///
    /// If `prune` is `true`, the path will be pruned, otherwise it will be left dangling.
    pub fn remove_branches_at<K: AsRef<[u8]>>(&mut self, path: K, prune: bool) -> bool {
        let path = path.as_ref();
        if !prune {
            if path.len() == 0 {
                let root = self.root.get_mut();
                let root_node = match root.as_mut() {
                    Some(root_node) => root_node,
                    None => return false,
                };
                if root_node.as_tagged().node_is_empty() {
                    return false;
                }
                *root_node = TrieNodeODRc::new_allocated_in(0, 0, self.alloc.clone());
                return true;
            }

            let root_node = match self.root.get_mut().as_mut() {
                Some(root_node) => root_node,
                None => return false,
            };
            let (node_key, node) = node_along_path_mut(root_node, path, true);
            return node.make_mut().node_remove_all_branches(node_key, false);
        }

        //NOTE: we're descending the zipper rather than creating it at the path so it will be allowed to
        // prune the branches.  A WriteZipper can't move above its root, so it couldn't prune otherwise
        let mut zipper = self.write_zipper();
        zipper.descend_to(path);
        zipper.remove_branches(prune)
    }

    /// Returns `true` if the map is empty, otherwise returns `false`
    pub fn is_empty(&self) -> bool {
        (match self.root() {
            Some(root) => root.as_tagged().node_is_empty(),
            None => true,
        } && self.root_val().is_none())
    }

    /// Prunes the dangling `path` specified up to the first upstream value or fork in the path, and
    /// returns the number of path bytes removed
    pub fn prune_path<K: AsRef<[u8]>>(&mut self, path: K) -> usize {
        let mut zipper = self.write_zipper();
        zipper.descend_to(path.as_ref());
        zipper.prune_path()
    }

    /// Creates the `path` specified as a dangling path.  Returns `true` if new path bytes were created,
    /// or `false` if the path already existed
    pub fn create_path<K: AsRef<[u8]>>(&mut self, path: K) -> bool {
        let path = path.as_ref();
        if path.len() == 0 {
            return false;
        }

        let root_node = self.get_or_init_root_mut();
        let (node_key, node) = node_along_path_mut(root_node, path, true);
        if node.is_empty() {
            let mut zipper = self.write_zipper();
            zipper.descend_to(path);
            return zipper.create_path();
        }

        let created_path = match node.make_mut().node_create_dangling(node_key) {
            Ok((created_path, _created_subnode)) => created_path,
            Err(replacement_node) => {
                *node = replacement_node;
                true
            }
        };
        if created_path {
            return true;
        }

        let mut zipper = self.write_zipper();
        zipper.descend_to(path);
        zipper.create_path()
    }

    /// Returns the total number of values contained within the map
    ///
    /// Cost note: this is not a cheap method. It may have order-N cost.
    pub fn val_count(&self) -> usize {
        let root_val = self.root_val().is_some() as usize;
        match self.root() {
            Some(root) => val_count_below_root(root.as_tagged()) + root_val,
            None => root_val,
        }
    }

    /// Returns a new `PathMap` containing the union of the paths in `self` and the paths in `other`
    pub fn join(&self, other: &Self) -> Self
    where
        V: Lattice,
    {
        result_into_map(self.pjoin(other), self, other, self.alloc.clone())
    }

    /// Returns a new `PathMap` containing the intersection of the paths in `self` and the paths in `other`
    pub fn meet(&self, other: &Self) -> Self
    where
        V: Lattice,
    {
        result_into_map(self.pmeet(other), self, other, self.alloc.clone())
    }

    /// Returns a new `PathMap` where the paths in `self` are restricted by the paths leading to
    /// values in `other`
    ///
    /// NOTE: if `other` has a root value, this function returns a clone of `self` because other's root
    /// value validates all paths.  If `other` does not have a root value, the returned map won't have
    /// one either.
    pub fn restrict(&self, other: &Self) -> Self {
        if other.root_val().is_some() {
            return self.clone();
        }
        let self_root = self.root();
        let other_root = other.root();
        if self_root.is_none() || other_root.is_none() {
            Self::new_in(self.alloc.clone())
        } else {
            match self_root
                .unwrap()
                .as_tagged()
                .prestrict_dyn(other_root.unwrap().as_tagged())
            {
                AlgebraicResult::Element(new_root) => {
                    Self::new_with_root_in(Some(new_root), None, self.alloc.clone())
                }
                AlgebraicResult::None => Self::new_in(self.alloc.clone()),
                AlgebraicResult::Identity(mask) => {
                    debug_assert_eq!(mask, SELF_IDENT);
                    Self::new_with_root_in(
                        Some(self.root().cloned().unwrap()),
                        None,
                        self.alloc.clone(),
                    )
                }
            }
        }
    }

    /// Returns a new `PathMap` where the paths in `self` are restricted by
    /// paths that lead to values in `guard`.
    ///
    /// Unlike [`restrict`](Self::restrict), this method accepts a guard map
    /// with a different value type. Guard values are used only as validation
    /// marks; their payloads are not read or copied into the returned map.
    pub fn restrict_by_paths<GV, GA>(&self, guard: &PathMap<GV, GA>) -> Self
    where
        GV: Clone + Send + Sync + Unpin,
        GA: Allocator,
    {
        if guard.root_val().is_some() {
            return self.clone();
        }

        let mut data = self.read_zipper();
        let mut guard = guard.read_zipper();
        let mut out = Self::new_in(self.alloc.clone());
        let mut path = Vec::new();
        restrict_by_paths_into(&mut data, &mut guard, false, &mut out, &mut path);
        out
    }

    /// Returns a new `PathMap` containing the contents from `self` minus the contents of `other`
    pub fn subtract(&self, other: &Self) -> Self
    where
        V: DistributiveLattice,
    {
        let subtracted_root_val = match self.root_val().psubtract(&other.root_val()) {
            AlgebraicResult::Element(new_val) => new_val,
            AlgebraicResult::Identity(mask) => {
                debug_assert_eq!(mask, SELF_IDENT);
                self.root_val().cloned()
            }
            AlgebraicResult::None => None,
        };

        let subtracted_root_node = match self.root().psubtract(&other.root()) {
            AlgebraicResult::Element(subtracted_node) => subtracted_node,
            AlgebraicResult::Identity(mask) => {
                debug_assert_eq!(mask, SELF_IDENT);
                self.root().cloned()
            }
            AlgebraicResult::None => None,
        };

        Self::new_with_root_in(
            subtracted_root_node,
            subtracted_root_val,
            self.alloc.clone(),
        )
    }

    /// Returns `self` minus `other`, skipping subtrees the two maps share by
    /// structure (COW clones of the same trie).
    ///
    /// Semantically identical to [`subtract`](Self::subtract): the result is the
    /// set of paths in `self` not in `other`. The difference is cost. When `self`
    /// and `other` are copy-on-write clones of a common trie that differ by only a
    /// few facts, most of their nodes are the SAME node in memory. For any such
    /// shared node, everything under it in `self` is also under it in `other`, so
    /// its contribution to `self \ other` is empty and the whole subtree is
    /// pruned without descending. That turns a near-identical diff from O(size of
    /// the trie) into O(the paths that actually differ).
    ///
    /// The pruning rides on the same node-identity short-circuit `subtract`
    /// already performs at the trie root (`TrieNodeODRc::psubtract`'s `ptr_eq`):
    /// it is now also applied at every shared child reached during the descent.
    /// Equal node identity means equal memory address means identical contents,
    /// so this never changes the result, only the work done to compute it.
    pub fn subtract_cow(&self, other: &Self) -> Self
    where
        V: DistributiveLattice,
    {
        self.subtract(other)
    }

    /// Optimize the `PathMap` by factoring shared subtries using a temporary [Merkle Tree](https://en.wikipedia.org/wiki/Merkle_tree)
    pub fn merkleize(&mut self) -> MerkleizeResult
    where
        V: core::hash::Hash,
    {
        let Some(root) = self.root() else {
            return MerkleizeResult::default();
        };
        let mut result = MerkleizeResult::default();
        let mut memo = gxhash::HashMap::default();
        let (hash, new_root) = merkleize_impl(&mut result, &mut memo, root, self.root_val());
        result.hash = hash;
        if let Some(new_root) = new_root {
            *self.root.get_mut() = Some(new_root);
        }
        result
    }
}

#[cfg(feature = "old_cursor")]
impl<V: Clone + Send + Sync + Unpin> PathMap<V> {
    /// Returns a [crate::old_cursor::PathMapCursor] to traverse all key-value pairs within the map. This
    /// is more efficient than using [iter](Self::iter), but is not compatible with the [Iterator] trait
    ///
    /// NOTE: This API is deprecated in favor of the [read_zipper](Self::read_zipper) method
    #[deprecated]
    pub fn cursor<'a>(&'a self) -> crate::old_cursor::PathMapCursor<'a, V> {
        crate::old_cursor::PathMapCursor::new(self)
    }

    /// Returns an [crate::old_cursor::AllDenseCursor], which behaves exactly like a [crate::old_cursor::PathMapCursor],
    /// but is only available with the `all_dense_nodes` feature.  This is mainly kept for benchmarking.
    #[deprecated]
    pub fn all_dense_cursor<'a>(&'a self) -> crate::old_cursor::AllDenseCursor<'a, V> {
        crate::old_cursor::AllDenseCursor::new(self)
    }
}

impl<V: Clone + Send + Sync + Unpin, K: AsRef<[u8]>> FromIterator<(K, V)> for PathMap<V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        for (key, val) in iter {
            map.set_val_at(key, val);
        }
        map
    }
}

impl<'a, V: Clone + Send + Sync + Unpin, K: AsRef<[u8]>> FromIterator<&'a (K, V)> for PathMap<V> {
    fn from_iter<I: IntoIterator<Item = &'a (K, V)>>(iter: I) -> Self {
        let mut map = Self::new();
        for (key, val) in iter {
            map.set_val_at(key, val.clone());
        }
        map
    }
}

impl<'a> FromIterator<&'a [u8]> for PathMap<()> {
    fn from_iter<I: IntoIterator<Item = &'a [u8]>>(iter: I) -> Self {
        let mut map = Self::new();
        for key in iter {
            map.set_val_at(key, ());
        }
        map
    }
}

impl<V: Clone + Send + Sync + Unpin, K: AsRef<[u8]>> From<(K, V)> for PathMap<V> {
    fn from(pair: (K, V)) -> Self {
        let mut map = Self::new();
        map.set_val_at(pair.0, pair.1);
        map
    }
}

impl<V: Clone + Send + Sync + Unpin + 'static, A: Allocator + 'static> std::iter::IntoIterator
    for PathMap<V, A>
{
    type Item = (Vec<u8>, V);
    type IntoIter = OwnedZipperIter<V, A>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_write_zipper(&[]).into_iter()
    }
}

/// Internal function to convert an [AlgebraicResult] (partial lattice result) into a `PathMap`
fn result_into_map<V: Clone + Send + Sync + Unpin, A: Allocator>(
    result: AlgebraicResult<PathMap<V, A>>,
    self_map: &PathMap<V, A>,
    other_map: &PathMap<V, A>,
    result_region: A,
) -> PathMap<V, A> {
    match result {
        AlgebraicResult::Element(new_map) => new_map,
        AlgebraicResult::None => PathMap::new_in(result_region),
        AlgebraicResult::Identity(mask) => {
            if mask & SELF_IDENT > 0 {
                self_map.clone()
            } else {
                debug_assert_eq!(mask, COUNTER_IDENT);
                other_map.clone()
            }
        }
    }
}

impl<V: Clone + Lattice + Send + Sync + Unpin, A: Allocator> Lattice for PathMap<V, A> {
    fn pjoin(&self, other: &Self) -> AlgebraicResult<Self> {
        let joined_node = self.root().pjoin(&other.root());
        let joined_root_val = self.root_val().pjoin(&other.root_val());
        joined_node.merge(
            joined_root_val,
            |which_arg| match which_arg {
                0 => Some(self.root().cloned()),
                1 => Some(other.root().cloned()),
                _ => unreachable!(),
            },
            |which_arg| match which_arg {
                0 => Some(self.root_val().cloned()),
                1 => Some(other.root_val().cloned()),
                _ => unreachable!(),
            },
            |root_node, root_val| {
                AlgebraicResult::Element(Self::new_with_root_in(
                    root_node.flatten(),
                    root_val.flatten(),
                    self.alloc.clone(),
                ))
            },
        )
    }
    fn join_into(&mut self, other: Self) -> AlgebraicStatus {
        let (other_root_node, other_root_val) = other.into_root();

        let root_node_status = if let Some(other_root) = other_root_node {
            let (status, result) = self
                .get_or_init_root_mut()
                .make_mut()
                .join_into_dyn(other_root);
            match result {
                Ok(()) => {}
                Err(replacement) => {
                    *self.get_or_init_root_mut() = replacement;
                }
            }
            status
        } else {
            if self.is_empty() {
                AlgebraicStatus::None
            } else {
                AlgebraicStatus::Identity
            }
        };

        let root_val_status = match other_root_val {
            Some(other_root_val) => match self.root_val_mut_ref() {
                Some(self_root_val) => self_root_val.join_into(other_root_val),
                None => {
                    self.set_root_val(other_root_val);
                    AlgebraicStatus::Element
                }
            },
            None => {
                if self.root_val().is_some() {
                    AlgebraicStatus::Identity
                } else {
                    AlgebraicStatus::None
                }
            }
        };
        root_node_status.merge(root_val_status, true, true)
    }
    fn pmeet(&self, other: &Self) -> AlgebraicResult<Self> {
        let meet_node = self.root().pmeet(&other.root());
        let meet_root_val = self.root_val().pmeet(&other.root_val());
        meet_node.merge(
            meet_root_val,
            |which_arg| match which_arg {
                0 => Some(self.root().cloned()),
                1 => Some(other.root().cloned()),
                _ => unreachable!(),
            },
            |which_arg| match which_arg {
                0 => Some(self.root_val().cloned()),
                1 => Some(other.root_val().cloned()),
                _ => unreachable!(),
            },
            |root_node, root_val| {
                AlgebraicResult::Element(Self::new_with_root_in(
                    root_node.flatten(),
                    root_val.flatten(),
                    self.alloc.clone(),
                ))
            },
        )
    }
    fn meet_into(&mut self, other: &Self) -> AlgebraicStatus {
        self.write_zipper().meet_into(&other.read_zipper(), true)
    }
}

impl<V: Clone + Send + Sync + Unpin + DistributiveLattice, A: Allocator> DistributiveLattice
    for PathMap<V, A>
{
    fn psubtract(&self, other: &Self) -> AlgebraicResult<Self> {
        let subtract_node = self.root().psubtract(&other.root());
        let subtract_root_val = self.root_val().psubtract(&other.root_val());
        subtract_node.merge(
            subtract_root_val,
            |which_arg| match which_arg {
                0 => Some(self.root().cloned()),
                1 => Some(other.root().cloned()),
                _ => unreachable!(),
            },
            |which_arg| match which_arg {
                0 => Some(self.root_val().cloned()),
                1 => Some(other.root_val().cloned()),
                _ => unreachable!(),
            },
            |root_node, root_val| {
                AlgebraicResult::Element(Self::new_with_root_in(
                    root_node.flatten(),
                    root_val.flatten(),
                    self.alloc.clone(),
                ))
            },
        )
    }
    fn subtract_into(&mut self, other: &Self) -> AlgebraicStatus {
        self.write_zipper()
            .subtract_into(&other.read_zipper(), true)
    }
}

fn restrict_by_paths_into<V, A, DataZ, GuardZ>(
    data: &mut DataZ,
    guard: &mut GuardZ,
    active: bool,
    out: &mut PathMap<V, A>,
    path: &mut Vec<u8>,
) where
    V: Clone + Send + Sync + Unpin,
    A: Allocator,
    DataZ: ZipperMoving + ZipperValues<V>,
    GuardZ: ZipperMoving,
{
    let active_here = active || guard.is_val();
    let visible = data.path_exists() && (active_here || guard.path_exists());
    if visible {
        if active_here {
            if let Some(value) = data.val().cloned() {
                out.set_val_at(&*path, value);
            } else if !path.is_empty() {
                out.create_path(&*path);
            }
        } else if !path.is_empty() {
            out.create_path(&*path);
        }
    }

    let child_mask = if active_here {
        data.child_mask()
    } else {
        data.child_mask() & guard.child_mask()
    };

    for child in child_mask.iter() {
        data.descend_to_byte(child);
        guard.descend_to_byte(child);
        path.push(child);
        restrict_by_paths_into(data, guard, active_here, out, path);
        path.pop();
        debug_assert!(data.ascend_byte());
        let _ = guard.ascend_byte();
    }
}

impl<V: Clone + Send + Sync + Unpin, A: Allocator> Quantale for PathMap<V, A> {
    fn prestrict(&self, other: &Self) -> AlgebraicResult<Self> {
        if other.root_val().is_some() {
            return AlgebraicResult::Identity(SELF_IDENT);
        }
        match (self.root(), other.root()) {
            (Some(self_root), Some(other_root)) => match self_root.prestrict(other_root) {
                AlgebraicResult::Element(new_root) => AlgebraicResult::Element(
                    Self::new_with_root_in(Some(new_root), None, self.alloc.clone()),
                ),
                AlgebraicResult::Identity(mask) => {
                    debug_assert_eq!(mask, SELF_IDENT);
                    if self.root_val().is_some() {
                        AlgebraicResult::Element(Self::new_with_root_in(
                            Some(self_root.clone()),
                            None,
                            self.alloc.clone(),
                        ))
                    } else {
                        AlgebraicResult::Identity(SELF_IDENT)
                    }
                }
                AlgebraicResult::None => AlgebraicResult::None,
            },
            _ => AlgebraicResult::None,
        }
    }
}

impl<V: Clone + Send + Sync + Unpin> Default for PathMap<V> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use crate::ring::{AlgebraicResult, AlgebraicStatus, DistributiveLattice, Lattice, SELF_IDENT};
    use crate::trie_map::*;
    use crate::zipper::{Zipper, ZipperInfallibleSubtries, ZipperMoving, ZipperWriting};

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

    fn put_map_at_focus<V>(mut source: PathMap<V>, path: &[u8], view: &PathMap<V>) -> PathMap<V>
    where
        V: Clone + Send + Sync + Unpin,
    {
        source.write_zipper_at_path(path).graft_map(view.clone());
        source
    }

    #[test]
    fn pathmap_for_each_value_matches_owned_iter_and_visits_root() {
        let mut map = PathMap::new();
        map.set_val_at([], 0);
        map.set_val_at(b"alpha", 1);
        map.set_val_at(b"beta/gamma", 2);

        let mut streamed = Vec::new();
        map.for_each_value(|path, value| streamed.push((path.to_vec(), *value)));
        streamed.sort();

        let mut owned = map
            .iter()
            .map(|(path, value)| (path, *value))
            .collect::<Vec<_>>();
        owned.sort();

        assert_eq!(streamed, owned);
    }

    #[test]
    fn pathmap_try_for_each_value_stops_on_callback_error() {
        let mut map = PathMap::new();
        map.set_val_at(b"alpha", 1);
        map.set_val_at(b"beta", 2);
        map.set_val_at(b"gamma", 3);

        let mut visited = Vec::new();
        let result = map.try_for_each_value(|path, value| {
            visited.push((path.to_vec(), *value));
            if *value == 2 {
                Err("stop")
            } else {
                Ok(())
            }
        });

        assert_eq!(result, Err("stop"));
        assert!(visited.iter().any(|(_, value)| *value == 2));
        assert!(visited.len() < map.val_count());
    }

    #[test]
    fn get_from_map_test() {
        let mut map = PathMap::new();
        //NOW: map contains an empty ListNode

        map.set_val_at("aaaaa", "aaaaa");
        assert_eq!(map.get_val_at("aaaaa").unwrap(), &"aaaaa");
        //NOW: map contains a ListNode with slot_0 filled by a value

        map.set_val_at("bbbbb", "bbbbb");
        assert_eq!(map.get_val_at("bbbbb").unwrap(), &"bbbbb");
        //NOW: map contains a ListNode with slot_0 and slot_1 filled by values

        map.set_val_at("ccccc", "ccccc");
        assert_eq!(map.get_val_at("aaaaa").unwrap(), &"aaaaa");
        assert_eq!(map.get_val_at("bbbbb").unwrap(), &"bbbbb");
        assert_eq!(map.get_val_at("ccccc").unwrap(), &"ccccc");
        //NOW: map contains a DenseByteNode, with 3 separate ListNodes, each containing one value

        map.set_val_at("ddddd", "ddddd");
        assert_eq!(map.get_val_at("ddddd").unwrap(), &"ddddd");
        //NOW: map contains a DenseByteNode, with 4 separate ListNodes, each containing one value

        map.set_val_at("abbbb", "abbbb");
        assert_eq!(map.get_val_at("abbbb").unwrap(), &"abbbb");
        //NOW: Dense("a"..) -> List("aaaa", "bbbb")

        map.set_val_at("aaaab", "aaaab");
        assert_eq!(map.get_val_at("aaaaa").unwrap(), &"aaaaa");
        assert_eq!(map.get_val_at("bbbbb").unwrap(), &"bbbbb");
        assert_eq!(map.get_val_at("abbbb").unwrap(), &"abbbb");
        assert_eq!(map.get_val_at("aaaab").unwrap(), &"aaaab");
        //NOW: Dense("a"..) -> List("aaa", "bbbb") -> List("a", "b")

        map.set_val_at("aaaac", "aaaac");
        assert_eq!(map.get_val_at("aaaaa").unwrap(), &"aaaaa");
        assert_eq!(map.get_val_at("aaaab").unwrap(), &"aaaab");
        assert_eq!(map.get_val_at("aaaac").unwrap(), &"aaaac");
        //NOW: Dense("a"..) -> List("aaa", "bbbb") -> Dense("a", "b", "c")

        map.set_val_at("acaaa", "acaaa");
        assert_eq!(map.get_val_at("aaaaa").unwrap(), &"aaaaa");
        assert_eq!(map.get_val_at("aaaab").unwrap(), &"aaaab");
        assert_eq!(map.get_val_at("aaaac").unwrap(), &"aaaac");
        assert_eq!(map.get_val_at("abbbb").unwrap(), &"abbbb");
        assert_eq!(map.get_val_at("acaaa").unwrap(), &"acaaa");
        //NOW: Dense("a"..) -> Dense("a", "b", "c") a-> List("aa") -> Dense("a", "b", "c")
        //                                          b-> List("bbb")
        //                                          c-> List("aaa")
    }

    #[test]
    fn map_insert_test() {
        let keys = [
            vec![
                75, 104, 119, 196, 129, 106, 97, 104, 32, 68, 197, 171, 32, 75, 197, 141, 104,
            ],
            vec![75, 104, 111, 100, 106, 97, 45, 66, 117, 110, 97, 107],
            vec![
                75, 104, 111, 100, 122, 104, 97, 45, 68, 111, 107, 117, 107, 104,
            ],
            vec![
                75, 104, 118, 97, 106, 101, 104, 32, 68, 111, 32, 75, 117, 104,
            ],
            vec![
                75, 104, 118, 196, 129, 106, 101, 104, 32, 68, 111, 32, 75, 197, 171, 104,
            ],
            vec![75, 104, 119, 97, 106, 97, 32, 68, 111, 32, 75, 111, 104],
            vec![75, 104, 119, 97, 106, 97, 32, 68, 117, 32, 75, 111, 104],
            vec![
                75, 104, 119, 97, 106, 97, 104, 32, 68, 111, 32, 75, 111, 104,
            ],
            vec![
                75, 104, 119, 97, 106, 97, 104, 32, 68, 117, 32, 75, 111, 104,
            ],
            vec![
                75, 104, 119, 97, 106, 97, 104, 45, 121, 101, 32, 68, 111, 32, 75, 117,
            ],
            vec![
                75, 104, 119, 97, 106, 97, 104, 45, 121, 101, 32, 68, 111, 32, 75, 197, 171,
            ],
            vec![
                75, 104, 119, 196, 129, 106, 97, 32, 68, 111, 32, 75, 111, 104,
            ],
            vec![
                75, 104, 119, 196, 129, 106, 97, 104, 32, 68, 197, 141, 32, 75, 197, 141, 104,
            ],
            vec![
                75, 104, 119, 196, 129, 106, 196, 129, 32, 68, 117, 32, 75, 111, 104,
            ],
            vec![107, 104, 119, 97, 106, 104, 32, 100, 119, 32, 107, 119, 104],
            vec![
                216, 174, 217, 136, 216, 167, 216, 172, 217, 135, 32, 216, 175, 217, 136, 32, 218,
                169, 217, 136, 217, 135,
            ],
            vec![
                73, 109, 196, 129, 109, 32, 197, 158, 196, 129, 225, 184, 169, 105, 98,
            ],
            vec![69, 109, 97, 109, 32, 83, 97, 104, 101, 98],
            vec![
                69, 109, 196, 129, 109, 32, 197, 158, 196, 129, 225, 184, 169, 101, 98,
            ],
            vec![72, 97, 122, 114, 97, 116],
            vec![73, 109, 97, 109, 32, 83, 97, 104, 101, 98],
            vec![73, 109, 97, 109, 32, 83, 97, 104, 105, 98],
            vec![73, 109, 97, 109, 115, 97, 107, 104, 105, 98],
            vec![
                73, 109, 196, 129, 109, 32, 83, 196, 129, 225, 186, 150, 101, 98,
            ],
            vec![75, 104, 119, 97, 106, 97],
            vec![
                75, 104, 119, 97, 106, 97, 32, 73, 109, 97, 109, 32, 83, 97, 105, 121, 105, 100,
            ],
        ];
        let mut map = PathMap::new();
        for (i, key) in keys.iter().enumerate() {
            map.set_val_at(key, i);
        }
        for (i, key) in keys.iter().enumerate() {
            assert_eq!(map.get_val_at(key), Some(&i));
        }
    }

    #[test]
    fn long_key_map_test() {
        let mut map = PathMap::new();

        map.set_val_at("aaaaaaaaaa01234567890123456789", 30);
        assert_eq!(
            map.get_val_at("aaaaaaaaaa01234567890123456789").unwrap(),
            &30
        );

        map.set_val_at("bbbbbbbbbb012345678901234567891", 31);
        assert_eq!(
            map.get_val_at("bbbbbbbbbb012345678901234567891").unwrap(),
            &31
        );

        map.set_val_at("cccccccccc012345678901234567890123456789", 40);
        assert_eq!(
            map.get_val_at("cccccccccc012345678901234567890123456789")
                .unwrap(),
            &40
        );

        map.set_val_at("dddddddddd01234567890123456789012345678901234", 45);
        assert_eq!(
            map.get_val_at("dddddddddd01234567890123456789012345678901234")
                .unwrap(),
            &45
        );

        map.set_val_at("eeeeeeeeee01234567890123456789012345678901234567890123456789012345678901234567890123456789", 90);
        assert_eq!(map.get_val_at("eeeeeeeeee01234567890123456789012345678901234567890123456789012345678901234567890123456789").unwrap(), &90);
    }

    #[test]
    fn map_contains_path_test() {
        let mut btm = PathMap::new();
        let rs = [
            "arrow",
            "bow",
            "cannon",
            "roman",
            "romane",
            "romanus",
            "romulus",
            "rubens",
            "ruber",
            "rubicon",
            "rubicundus",
            "rom'i",
        ];
        rs.iter().enumerate().for_each(|(i, r)| {
            btm.set_val_at(r.as_bytes(), i);
        });

        assert_eq!(btm.path_exists_at(b"can"), true);
        assert_eq!(btm.path_exists_at(b"cannon"), true);
        assert_eq!(btm.path_exists_at(b"cannonade"), false);
        assert_eq!(btm.path_exists_at(b""), true);
    }

    #[test]
    fn map_get_mut_test() {
        let mut map = PathMap::<usize>::new();

        //Test root value with `get_val_mut_at`
        assert_eq!(map.get_val_mut_at(b""), None);
        assert_eq!(map.set_val_at(b"", 42), None);
        assert_eq!(map.get_val_mut_at(b""), Some(&mut 42));
        *map.get_val_mut_at(b"").unwrap() = 24;
        assert_eq!(map.get_val_mut_at(b""), Some(&mut 24));

        //Test non-root value with `get_val_mut_at`
        const PATH: &[u8] = b"This is a long path to somewhere, hopefully far enough away that we end up creating more than one node";
        assert_eq!(map.get_val_mut_at(PATH), None);
        assert_eq!(map.set_val_at(PATH, 42), None);
        assert_eq!(map.get_val_mut_at(PATH), Some(&mut 42));
        *map.get_val_mut_at(PATH).unwrap() = 24;
        assert_eq!(map.get_val_mut_at(PATH), Some(&mut 24));
    }

    #[test]
    fn map_get_val_mut_or_set_test() {
        let mut map = PathMap::<usize>::new();

        //Test root value
        assert_eq!(map.get_val_or_set_mut_at(b"", 42), &mut 42);
        assert_eq!(map.get_val_mut_at(b""), Some(&mut 42));
        assert_eq!(map.remove_val_at(b"", true), Some(42));
        *map.get_val_or_set_mut_at(b"", 42) = 24;
        assert_eq!(map.get_val_mut_at(b""), Some(&mut 24));

        //Test non-root value
        const PATH: &[u8] = b"This is a long path to somewhere, hopefully far enough away that we end up creating more than one node";
        assert_eq!(map.get_val_mut_at(PATH), None);
        assert_eq!(map.get_val_or_set_mut_at(PATH, 42), &mut 42);
        assert_eq!(map.get_val_mut_at(PATH), Some(&mut 42));
        assert_eq!(map.remove_val_at(PATH, true), Some(42));
        *map.get_val_or_set_mut_at(PATH, 42) = 24;
        assert_eq!(map.get_val_mut_at(PATH), Some(&mut 24));
    }

    #[test]
    fn map_remove_test1() {
        let mut map = PathMap::new();
        map.set_val_at("aaaaa", "aaaaa");
        map.set_val_at("bbbbb", "bbbbb");
        map.set_val_at("ccccc", "ccccc");
        map.set_val_at("ddddd", "ddddd");
        map.set_val_at("abbbb", "abbbb");
        map.set_val_at("aaaab", "aaaab");
        map.set_val_at("aaaac", "aaaac");
        map.set_val_at("acaaa", "acaaa");
        assert_eq!(map.val_count(), 8);

        assert_eq!(map.remove_val_at(b"aaaaa", true), Some("aaaaa"));
        assert_eq!(map.val_count(), 7);
        assert_eq!(map.remove_val_at(b"acaaa", true), Some("acaaa"));
        assert_eq!(map.val_count(), 6);
        assert_eq!(map.remove_val_at(b"cccccnot-a-real-key", true), None);
        assert_eq!(map.val_count(), 6);
        assert_eq!(map.remove_val_at(b"aaaac", true), Some("aaaac"));
        assert_eq!(map.val_count(), 5);
        assert_eq!(map.remove_val_at(b"aaaab", true), Some("aaaab"));
        assert_eq!(map.val_count(), 4);
        assert_eq!(map.remove_val_at(b"abbbb", true), Some("abbbb"));
        assert_eq!(map.val_count(), 3);
        assert_eq!(map.remove_val_at(b"ddddd", true), Some("ddddd"));
        assert_eq!(map.val_count(), 2);
        assert_eq!(map.remove_val_at(b"ccccc", true), Some("ccccc"));
        assert_eq!(map.val_count(), 1);
        assert_eq!(map.remove_val_at(b"bbbbb", true), Some("bbbbb"));
        assert_eq!(map.val_count(), 0);
        assert!(map.is_empty());
    }

    #[test]
    fn map_remove_test2() {
        let mut btm = PathMap::from_iter(
            [("abbb", ()), ("b", ()), ("bba", ())]
                .iter()
                .map(|(p, v)| (p.as_bytes(), v)),
        );
        btm.remove_val_at("abbb".as_bytes(), true);
        btm.remove_val_at("a".as_bytes(), true);
    }

    #[test]
    fn map_update_test() {
        let rs = [
            "arrow",
            "bow",
            "cannon",
            "roman",
            "romane",
            "romanus",
            "romulus",
            "rubens",
            "ruber",
            "rubicon",
            "rubicundus",
            "rom'i",
        ];
        let mut btm: PathMap<u64> = rs
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k, i as u64))
            .collect();

        let mut zipper = btm.write_zipper_at_path(b"cannon");
        assert_eq!(zipper.get_val_or_set_mut(42), &2);
        drop(zipper);

        let mut zipper = btm.write_zipper_at_path(b"dagger");
        assert_eq!(zipper.get_val_or_set_mut(42), &42);
    }

    #[test]
    fn map_join_test() {
        let mut a = PathMap::<usize>::new();
        let mut b = PathMap::<usize>::new();
        let rs = [
            "Abbotsford",
            "Abbottabad",
            "Abcoude",
            "Abdul Hakim",
            "Abdulino",
            "Abdullahnagar",
            "Abdurahmoni Jomi",
            "Abejorral",
            "Abelardo Luz",
        ];
        for (i, path) in rs.into_iter().enumerate() {
            if i % 2 == 0 {
                a.set_val_at(path, i);
            } else {
                b.set_val_at(path, i);
            }
        }

        let joined = a.join(&b);
        for (path, i) in joined.iter() {
            // println!("{} {}", std::str::from_utf8(&path).unwrap(), i);
            assert_eq!(rs[*i].as_bytes(), &path);
        }
        assert_eq!(joined.val_count(), rs.len());
    }

    #[test]
    fn dpa_path_derivative_commutes_with_map_algebra() {
        let mut left = PathMap::<bool>::new();
        left.set_val_at(b"scope:a", false);
        left.set_val_at(b"scope:both-different", false);
        left.set_val_at(b"scope:both-same", true);
        left.set_val_at(b"scope:left-only", true);
        left.create_path(b"scope:dangling:left");
        left.create_path(b"scope:dangling:shared");
        left.set_val_at(b"outside:left", true);

        let mut right = PathMap::<bool>::new();
        right.set_val_at(b"scope:b", true);
        right.set_val_at(b"scope:both-different", true);
        right.set_val_at(b"scope:both-same", true);
        right.create_path(b"scope:dangling:right");
        right.create_path(b"scope:dangling:shared");
        right.set_val_at(b"outside:right", false);

        let prefix = b"scope:";
        let left_focus = left.read_zipper_at_path(prefix).make_map();
        let right_focus = right.read_zipper_at_path(prefix).make_map();

        let joined_then_focused = left.join(&right).read_zipper_at_path(prefix).make_map();
        let focused_then_joined = left_focus.join(&right_focus);
        assert_same_pathspace(&joined_then_focused, &focused_then_joined);

        let met_then_focused = left.meet(&right).read_zipper_at_path(prefix).make_map();
        let focused_then_met = left_focus.meet(&right_focus);
        assert_same_pathspace(&met_then_focused, &focused_then_met);

        let subtracted_then_focused = left.subtract(&right).read_zipper_at_path(prefix).make_map();
        let focused_then_subtracted = left_focus.subtract(&right_focus);
        assert_same_pathspace(&subtracted_then_focused, &focused_then_subtracted);
    }

    #[test]
    fn map_join_preserves_shared_dangling_path_in_line_list_promotion() {
        let mut left = PathMap::<bool>::new();
        left.create_path(b"scope:dangling:left");
        left.create_path(b"scope:dangling:shared");

        let mut right = PathMap::<bool>::new();
        right.create_path(b"scope:dangling:right");
        right.create_path(b"scope:dangling:shared");

        let joined = left.join(&right);
        assert!(joined.path_exists_at(b"scope:dangling:left"));
        assert!(joined.path_exists_at(b"scope:dangling:right"));
        assert!(joined.path_exists_at(b"scope:dangling:shared"));
    }

    #[test]
    fn dpa_graft_map_satisfies_full_subtree_lens_laws() {
        let mut source = PathMap::<bool>::new();
        source.set_val_at(b"focus", false);
        source.set_val_at(b"focus:old", true);
        source.create_path(b"focus:dangling:old");
        source.set_val_at(b"sibling", true);
        source.create_path(b"sibling:dangling");

        let mut view = PathMap::<bool>::new();
        view.set_val_at(b"", true);
        view.set_val_at(b":new", false);
        view.create_path(b":dangling:new");

        let mut alt_view = PathMap::<bool>::new();
        alt_view.set_val_at(b"", false);
        alt_view.set_val_at(b":alt", true);
        alt_view.create_path(b":dangling:alt");

        let focus = b"focus";

        let replaced = put_map_at_focus(source.clone(), focus, &view);
        let got = replaced.read_zipper_at_path(focus).make_map();
        assert_same_pathspace(&view, &got);
        assert_eq!(replaced.get_val_at(b"sibling"), Some(&true));
        assert!(replaced.path_exists_at(b"sibling:dangling"));

        let original_focus = source.read_zipper_at_path(focus).make_map();
        let restored = put_map_at_focus(source.clone(), focus, &original_focus);
        assert_same_pathspace(&source, &restored);

        let sequential = put_map_at_focus(
            put_map_at_focus(source.clone(), focus, &view),
            focus,
            &alt_view,
        );
        let direct = put_map_at_focus(source, focus, &alt_view);
        assert_same_pathspace(&direct, &sequential);
    }

    #[test]
    fn dpa_drop_head_projects_multi_byte_residuals_into_focus() {
        let mut source = PathMap::<bool>::new();
        source.set_val_at([0, 0, b'x'], false);
        source.set_val_at([0, 1, b'x'], true);
        source.set_val_at([2, 2, b'y', b'z'], false);
        source.create_path([3, 3, b'd']);
        source.create_path([4, 4, b'd', b'e']);

        let mut expected = source.read_zipper_at_path([0, 0]).make_map();
        for prefix in [[0, 1], [2, 2], [3, 3], [4, 4]] {
            expected = expected.join(&source.read_zipper_at_path(prefix).make_map());
        }

        let mut projected = source.clone();
        assert!(projected.write_zipper().join_k_path_into(2, true));

        assert_same_pathspace(&expected, &projected);
        assert_eq!(projected.get_val_at(b"x"), Some(&true));
        assert_eq!(projected.get_val_at(b"yz"), Some(&false));
        assert!(projected.path_exists_at(b"d"));
        assert!(projected.path_exists_at(b"de"));
    }

    #[test]
    fn map_join_into_test() {
        let mut a = PathMap::<usize>::new();
        let mut b = PathMap::<usize>::new();
        let rs = [
            "Abbotsford",
            "Abbottabad",
            "Abcoude",
            "Abdul Hakim",
            "Abdulino",
            "Abdullahnagar",
            "Abdurahmoni Jomi",
            "Abejorral",
            "Abelardo Luz",
        ];
        for (i, path) in rs.into_iter().enumerate() {
            if i % 2 == 0 {
                a.set_val_at(path, i);
            } else {
                b.set_val_at(path, i);
            }
        }

        a.join_into(b);
        for (path, i) in a.iter() {
            // println!("{} {}", std::str::from_utf8(&path).unwrap(), i);
            assert_eq!(rs[*i].as_bytes(), &path);
        }
        assert_eq!(a.val_count(), rs.len());
    }

    #[test]
    fn map_meet_and_subtract_into_trait_overrides() {
        let mut a = PathMap::<()>::new();
        a.set_val_at([], ());
        a.set_val_at("AA", ());
        a.set_val_at("AB", ());

        let mut b = PathMap::<()>::new();
        b.set_val_at("AA", ());
        b.set_val_at("BB", ());

        let mut met = a.clone();
        assert_eq!(Lattice::meet_into(&mut met, &b), AlgebraicStatus::Element);
        assert_eq!(met.get_val_at([]), None);
        assert_eq!(met.get_val_at("AA"), Some(&()));
        assert_eq!(met.get_val_at("AB"), None);
        assert_eq!(met.get_val_at("BB"), None);
        assert_eq!(Lattice::meet_into(&mut met, &b), AlgebraicStatus::Identity);

        let mut subtracted = a.clone();
        assert_eq!(
            DistributiveLattice::subtract_into(&mut subtracted, &b),
            AlgebraicStatus::Element
        );
        assert_eq!(subtracted.get_val_at([]), Some(&()));
        assert_eq!(subtracted.get_val_at("AA"), None);
        assert_eq!(subtracted.get_val_at("AB"), Some(&()));
        assert_eq!(subtracted.get_val_at("BB"), None);
        assert_eq!(
            DistributiveLattice::subtract_into(&mut subtracted, &b),
            AlgebraicStatus::Identity
        );

        let mut emptying = a.clone();
        assert_eq!(
            DistributiveLattice::subtract_into(&mut emptying, &a),
            AlgebraicStatus::None
        );
        assert!(emptying.is_empty());
    }

    #[test]
    fn map_join_val_at_inserts_and_merges_values() {
        let mut map = PathMap::<bool>::new();

        assert_eq!(
            map.join_val_at(b"feature:enabled", false),
            AlgebraicStatus::Element
        );
        assert_eq!(map.get_val_at(b"feature:enabled"), Some(&false));

        assert_eq!(
            map.join_val_at(b"feature:enabled", true),
            AlgebraicStatus::Element
        );
        assert_eq!(map.get_val_at(b"feature:enabled"), Some(&true));

        assert_eq!(
            map.join_val_at(b"feature:enabled", false),
            AlgebraicStatus::Identity
        );
        assert_eq!(map.get_val_at(b"feature:enabled"), Some(&true));
    }

    #[test]
    fn map_join_val_at_handles_root_values() {
        let mut map = PathMap::<bool>::new();

        assert_eq!(map.join_val_at([], false), AlgebraicStatus::Element);
        assert_eq!(map.get_val_at([]), Some(&false));

        assert_eq!(map.join_val_at([], true), AlgebraicStatus::Element);
        assert_eq!(map.get_val_at([]), Some(&true));
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CancelOnEqual(u8);

    impl Lattice for CancelOnEqual {
        fn pjoin(&self, other: &Self) -> AlgebraicResult<Self> {
            if self == other {
                AlgebraicResult::None
            } else {
                AlgebraicResult::Identity(SELF_IDENT)
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

    #[test]
    fn map_join_val_at_removes_when_join_annihilates() {
        let mut map = PathMap::<CancelOnEqual>::new();

        assert_eq!(
            map.join_val_at(b"cancel", CancelOnEqual(7)),
            AlgebraicStatus::Element
        );
        assert_eq!(map.val_count(), 1);

        assert_eq!(
            map.join_val_at(b"cancel", CancelOnEqual(7)),
            AlgebraicStatus::None
        );
        assert_eq!(map.get_val_at(b"cancel"), None);
        assert_eq!(map.val_count(), 0);
        assert!(map.is_empty());
    }

    #[cfg(feature = "old_cursor")]
    #[test]
    #[allow(
        deprecated,
        reason = "this test intentionally covers the deprecated old_cursor feature"
    )]
    fn cursor_test() {
        let table = ["A", "Bcdef", "Ghij", "Klmnopqrst"];
        let btm: PathMap<usize> = table.iter().enumerate().map(|(n, s)| (s, n)).collect();
        let mut cursor = btm.cursor();
        while let Some((k, v)) = cursor.next() {
            // println!("{}, {v}", std::str::from_utf8(k).unwrap());
            assert_eq!(k, table[*v].as_bytes());
        }
    }

    #[test]
    fn map_root_value_test1() {
        let mut map = PathMap::<usize>::new();

        //Direct-map operations on root value
        assert_eq!(map.get_val_at([]), None);
        assert_eq!(map.set_val_at([], 1), None);
        assert_eq!(map.get_val_at([]), Some(&1));
        assert_eq!(map.remove_val_at([], true), Some(1));
        assert_eq!(map.get_val_at([]), None);

        //Through a WriteZipper, created at the root
        let mut z = map.write_zipper();
        assert_eq!(z.val(), None);
        assert_eq!(z.set_val(1), None);
        assert_eq!(z.val(), Some(&1));
        *z.get_val_mut().unwrap() = 2;
        assert_eq!(z.remove_val(true), Some(2));
        assert_eq!(z.val(), None);
        drop(z);

        //A root value written before root allocation must not reappear after the
        //same root is overwritten through native node storage.
        assert_eq!(map.set_val_at([], 10), None);
        assert_eq!(map.set_val_at(b"legacy/allocates-root", 1), None);
        let mut z = map.write_zipper();
        assert_eq!(z.val(), Some(&10));
        assert_eq!(z.set_val(20), Some(10));
        assert_eq!(z.remove_val(true), Some(20));
        assert_eq!(z.val(), None);
        drop(z);
        assert_eq!(map.get_val_at([]), None);

        //A native root value must survive concrete root-node upgrades.
        assert_eq!(map.set_val_at([], 30), None);
        assert_eq!(map.set_val_at(b"upgrade/a", 1), None);
        assert_eq!(map.set_val_at(b"upgrade/b", 2), None);
        assert_eq!(map.set_val_at(b"upgrade/c", 3), None);
        assert_eq!(map.get_val_at([]), Some(&30));
        assert_eq!(map.remove_val_at([], true), Some(30));
        assert_eq!(map.get_val_at([]), None);

        //Through a WriteZipper, created at a zero-length path
        let mut z = map.write_zipper_at_path(&[]);
        assert_eq!(z.val(), None);
        assert_eq!(z.set_val(1), None);
        assert_eq!(z.val(), Some(&1));
        *z.get_val_mut().unwrap() = 2;
        assert_eq!(z.remove_val(true), Some(2));
        assert_eq!(z.val(), None);
        drop(z);

        //Through read zippers
        assert_eq!(map.read_zipper().get_val(), None);
        assert_eq!(map.set_val_at([], 1), None);
        assert_eq!(map.read_zipper().get_val(), Some(&1));
        assert_eq!(map.read_zipper_at_borrowed_path(&[]).get_val(), Some(&1));
        assert_eq!(map.read_zipper_at_path([]).get_val(), Some(&1));
        assert_eq!(map.remove_val_at([], true), Some(1));
        assert_eq!(map.read_zipper_at_borrowed_path(&[]).get_val(), None);
        assert_eq!(map.read_zipper_at_path([]).get_val(), None);

        //Through ZipperHeads
        let map_head = map.zipper_head();
        let mut z = map_head.write_zipper_at_exclusive_path([]).unwrap();
        assert_eq!(z.val(), None);
        assert_eq!(z.set_val(1), None);
        assert_eq!(z.val(), Some(&1));
        *z.get_val_mut().unwrap() = 2;
        drop(z);
        drop(map_head);
        assert_eq!(map.get_val_at([]), Some(&2));
    }

    #[test]
    fn remove_val_at_without_prune_preserves_dangling_path() {
        let mut empty = PathMap::<u8>::new();
        assert_eq!(empty.remove_val_at(b"missing", false), None);

        let mut map = PathMap::new();
        assert_eq!(map.set_val_at(b"alpha", 1), None);
        assert_eq!(map.remove_val_at(b"alpha", false), Some(1));
        assert_eq!(map.get_val_at(b"alpha"), None);
        assert!(map.path_exists_at(b"alpha"));

        assert_eq!(map.set_val_at(b"alpha", 2), None);
        assert_eq!(map.remove_val_at(b"alpha", true), Some(2));
        assert_eq!(map.get_val_at(b"alpha"), None);
        assert!(!map.path_exists_at(b"alpha"));
    }

    #[test]
    fn remove_branches_at_without_prune_matches_zipper_prefix_cut() {
        let mut direct = PathMap::new();
        direct.set_val_at(b"alpha", 1);
        direct.set_val_at(b"alphabet", 2);
        direct.set_val_at(b"alpine", 3);
        direct.set_val_at(b"beta", 4);

        let mut zipper_shaped = direct.clone();
        {
            let mut zipper = zipper_shaped.write_zipper();
            zipper.descend_to(b"alpha");
            assert_eq!(zipper.remove_branches(false), true);
        }

        assert_eq!(direct.remove_branches_at(b"alpha", false), true);
        for path in [
            b"alpha".as_slice(),
            b"alphabet".as_slice(),
            b"alpine".as_slice(),
            b"beta".as_slice(),
        ] {
            assert_eq!(direct.get_val_at(path), zipper_shaped.get_val_at(path));
            assert_eq!(
                direct.path_exists_at(path),
                zipper_shaped.path_exists_at(path)
            );
        }
    }

    #[test]
    fn remove_branches_at_without_prune_matches_zipper_root_cut() {
        let mut empty = PathMap::<u8>::new();
        assert_eq!(empty.remove_branches_at(b"", false), false);

        let mut direct = PathMap::new();
        direct.set_val_at(b"", 0);
        direct.set_val_at(b"a", 1);
        direct.set_val_at(b"b", 2);

        let mut zipper_shaped = direct.clone();
        {
            let mut zipper = zipper_shaped.write_zipper();
            zipper.descend_to(b"");
            assert_eq!(zipper.remove_branches(false), true);
        }

        assert_eq!(direct.remove_branches_at(b"", false), true);
        for path in [b"".as_slice(), b"a".as_slice(), b"b".as_slice()] {
            assert_eq!(direct.get_val_at(path), zipper_shaped.get_val_at(path));
            assert_eq!(
                direct.path_exists_at(path),
                zipper_shaped.path_exists_at(path)
            );
        }
    }

    fn old_create_path<K: AsRef<[u8]>>(map: &mut PathMap<u8>, path: K) -> bool {
        let mut zipper = map.write_zipper();
        zipper.descend_to(path.as_ref());
        zipper.create_path()
    }

    #[test]
    fn create_path_matches_zipper_for_missing_compressed_prefix() {
        let mut direct = PathMap::new();
        direct.set_val_at(b"alphabet", 1);
        direct.set_val_at(b"alpine", 2);

        let mut zipper_shaped = direct.clone();
        let old_created = old_create_path(&mut zipper_shaped, b"alpha");

        assert_eq!(direct.create_path(b"alpha"), old_created);
        for path in [
            b"alpha".as_slice(),
            b"alphabet".as_slice(),
            b"alpine".as_slice(),
        ] {
            assert_eq!(direct.get_val_at(path), zipper_shaped.get_val_at(path));
            assert_eq!(
                direct.path_exists_at(path),
                zipper_shaped.path_exists_at(path)
            );
        }
    }

    #[test]
    fn create_path_matches_zipper_for_missing_leaf_path() {
        let mut direct = PathMap::new();
        direct.set_val_at(b"alpha", 1);

        let mut zipper_shaped = direct.clone();
        assert_eq!(old_create_path(&mut zipper_shaped, b"beta"), true);

        assert_eq!(direct.create_path(b"beta"), true);
        for path in [b"alpha".as_slice(), b"beta".as_slice()] {
            assert_eq!(direct.get_val_at(path), zipper_shaped.get_val_at(path));
            assert_eq!(
                direct.path_exists_at(path),
                zipper_shaped.path_exists_at(path)
            );
        }
    }

    #[test]
    fn create_path_matches_zipper_for_existing_and_empty_paths() {
        let mut direct = PathMap::new();
        direct.set_val_at(b"alpha", 1);

        let mut zipper_shaped = direct.clone();
        assert_eq!(old_create_path(&mut zipper_shaped, b""), false);
        assert_eq!(old_create_path(&mut zipper_shaped, b"alpha"), false);
        assert_eq!(old_create_path(&mut zipper_shaped, b"alphabet"), true);

        assert_eq!(direct.create_path(b""), false);
        assert_eq!(direct.create_path(b"alpha"), false);
        assert_eq!(direct.create_path(b"alphabet"), true);
        for path in [b"".as_slice(), b"alpha".as_slice(), b"alphabet".as_slice()] {
            assert_eq!(direct.get_val_at(path), zipper_shaped.get_val_at(path));
            assert_eq!(
                direct.path_exists_at(path),
                zipper_shaped.path_exists_at(path)
            );
        }
    }

    #[test]
    fn create_path_extends_existing_dangling_prefix() {
        let mut map = PathMap::<u8>::new();
        assert_eq!(map.create_path(b"root"), true);
        assert_eq!(map.create_path(b"root:child"), true);

        assert!(map.path_exists_at(b"root"));
        assert!(map.path_exists_at(b"root:child"));
    }

    #[test]
    fn set_val_extends_binary_dangling_prefix() {
        let mut map = PathMap::<u8>::new();
        assert_eq!(map.create_path([217]), true);

        assert_eq!(map.set_val_at([217, 0], 42), None);

        assert!(map.path_exists_at([217]));
        assert!(map.path_exists_at([217, 0]));
        assert_eq!(map.get_val_at([217, 0]), Some(&42));
    }

    /// Tests algebraic ops on maps with root values, but no trie
    #[test]
    fn map_root_value_test2() {
        let mut map_a = PathMap::<()>::new();
        assert_eq!(map_a.get_val_at([]), None);
        assert_eq!(map_a.set_val_at([], ()), None);
        assert_eq!(map_a.get_val_at([]), Some(&()));
        let map_b = PathMap::<()>::new();

        let joined = map_a.join(&map_b);
        assert_eq!(joined.get_val_at([]), Some(&()));

        let mut cloned = map_b.clone();
        cloned.join_into(map_a.clone());
        assert_eq!(cloned.get_val_at([]), Some(&()));

        let meet = map_a.meet(&map_b);
        assert_eq!(meet.get_val_at([]), None);

        let meet = map_a.meet(&map_a);
        assert_eq!(meet.get_val_at([]), Some(&()));

        let subtract = map_a.subtract(&map_b);
        assert_eq!(subtract.get_val_at([]), Some(&()));

        let subtract = map_a.subtract(&map_a);
        assert_eq!(subtract.get_val_at([]), None);

        let subtract = map_a.subtract(&map_a);
        assert_eq!(subtract.get_val_at([]), None);

        let restrict = map_a.restrict(&map_a);
        assert_eq!(restrict.get_val_at([]), Some(&()));

        let restrict = map_a.restrict(&map_b);
        assert_eq!(restrict.get_val_at([]), None);
    }

    /// Tests algebraic ops on maps with root values and a downstream trie
    #[test]
    fn map_root_value_test3() {
        //Both a root val and a trie
        let mut map_a = PathMap::<()>::new();
        assert_eq!(map_a.set_val_at([], ()), None);
        assert_eq!(map_a.set_val_at("AA", ()), None);

        //Trie different from map_a, but no root val
        let mut map_b = PathMap::<()>::new();
        assert_eq!(map_b.set_val_at("BB", ()), None);

        //Trie same as map_a, but no root val
        let mut map_c = PathMap::<()>::new();
        assert_eq!(map_c.set_val_at("AA", ()), None);

        //Root val but no trie
        let mut map_d = PathMap::<()>::new();
        assert_eq!(map_d.set_val_at([], ()), None);

        //pjoin
        let joined_result = map_a.pjoin(&map_b);
        assert!(joined_result.is_element());
        let joined = joined_result.unwrap([&map_a, &map_b]);
        assert_eq!(joined.get_val_at([]), Some(&()));
        assert_eq!(joined.get_val_at("AA"), Some(&()));
        assert_eq!(joined.get_val_at("BB"), Some(&()));

        let joined_result = map_a.pjoin(&map_c);
        assert!(joined_result.is_identity());

        let joined_result = map_a.pjoin(&map_d);
        assert!(joined_result.is_identity());

        //pmeet
        let meet_result = map_a.pmeet(&map_a);
        assert!(meet_result.is_identity());

        let meet_result = map_a.pmeet(&map_b);
        assert!(meet_result.is_none());

        let meet_result = map_a.pmeet(&map_c);
        assert!(meet_result.is_element());
        let meet = meet_result.unwrap([&map_a, &map_c]);
        assert_eq!(meet.get_val_at([]), None);
        assert_eq!(meet.get_val_at("AA"), Some(&()));
        assert_eq!(meet.get_val_at("BB"), None);

        let meet_result = map_a.pmeet(&map_d);
        assert!(meet_result.is_element());
        let meet = meet_result.unwrap([&map_a, &map_d]);
        assert_eq!(meet.get_val_at([]), Some(&()));
        assert_eq!(meet.get_val_at("AA"), None);

        //psubtract
        let subtract_result = map_a.psubtract(&map_a);
        assert!(subtract_result.is_none());

        let subtract_result = map_a.psubtract(&map_b);
        assert!(subtract_result.is_identity());

        let subtract_result = map_a.psubtract(&map_c);
        assert!(subtract_result.is_element());
        let subtract = subtract_result.unwrap([&map_a, &map_c]);
        assert_eq!(subtract.get_val_at([]), Some(&()));
        assert_eq!(subtract.get_val_at("AA"), None);

        let subtract_result = map_a.psubtract(&map_d);
        assert!(subtract_result.is_element());
        let subtract = subtract_result.unwrap([&map_a, &map_d]);
        assert_eq!(subtract.get_val_at([]), None);
        assert_eq!(subtract.get_val_at("AA"), Some(&()));

        //prestrict
        let restrict_result = map_a.prestrict(&map_b);
        assert!(restrict_result.is_none());

        let restrict_result = map_a.prestrict(&map_c);
        assert!(restrict_result.is_element());
        let restrict = restrict_result.unwrap([&map_a, &map_c]);
        assert_eq!(restrict.get_val_at([]), None);
        assert_eq!(restrict.get_val_at("AA"), Some(&()));

        let restrict_result = map_a.prestrict(&map_d);
        assert!(restrict_result.is_identity());
    }

    #[test]
    fn map_root_value_test4() {
        let mut map0 = PathMap::<usize>::new();
        let mut map1 = PathMap::<usize>::new();
        map1.set_val_at([], 0);

        let mut wz = map0.write_zipper();
        wz.graft(&map1.read_zipper());
        drop(wz);

        assert_eq!(map0.get_val_at([]), Some(&0));
    }

    #[test]
    fn map_root_value_survives_zipper_head_take_graft_roundtrip() {
        let mut map = PathMap::<u8>::new();
        assert_eq!(map.set_val_at([], 7), None);
        assert_eq!(map.set_val_at(b"child", 1), None);

        assert_eq!(map.get_val_at([]), Some(&7));
        assert_eq!(map.read_zipper().get_val(), Some(&7));
        assert_eq!(map.write_zipper().val(), Some(&7));

        {
            let map_head = map.zipper_head();
            {
                let rz = map_head.read_zipper_at_path([]).unwrap();
                assert_eq!(rz.val(), Some(&7));
            }
            {
                let mut wz = map_head.write_zipper_at_exclusive_path([]).unwrap();
                assert_eq!(wz.set_val(8), Some(7));
                map_head.cleanup_write_zipper(wz);
            }
        }

        assert_eq!(map.get_val_at([]), Some(&8));
        assert_eq!(map.get_val_at(b"child"), Some(&1));

        assert_eq!(map.set_val_at(b"branch", 10), None);
        assert_eq!(map.set_val_at(b"branch/leaf", 11), None);
        let val_count_before = map.val_count();

        let taken = {
            let mut wz = map.write_zipper_at_path(b"branch");
            wz.take_map(true).expect("branch submap should exist")
        };

        assert_eq!(map.get_val_at(b"branch"), None);
        assert_eq!(map.get_val_at(b"branch/leaf"), None);

        assert_eq!(taken.get_val_at([]), Some(&10));
        assert_eq!(taken.get_val_at(b"/leaf"), Some(&11));

        {
            let mut wz = map.write_zipper_at_path(b"branch");
            wz.graft_map(taken);
        }

        assert_eq!(map.get_val_at([]), Some(&8));
        assert_eq!(map.get_val_at(b"child"), Some(&1));
        assert_eq!(map.get_val_at(b"branch"), Some(&10));
        assert_eq!(map.get_val_at(b"branch/leaf"), Some(&11));
        assert_eq!(map.val_count(), val_count_before);
    }

    #[test]
    fn map_root_value_graft_map_replaces_branches_with_expected_focus_value_policy() {
        let mut dst = PathMap::<u8>::new();
        assert_eq!(dst.set_val_at([], 1), None);
        assert_eq!(dst.set_val_at(b"old", 2), None);

        let mut src = PathMap::<u8>::new();
        assert_eq!(src.set_val_at([], 3), None);
        assert_eq!(src.set_val_at(b"new", 4), None);

        {
            let mut wz = dst.write_zipper();
            wz.graft_map(src);
        }

        assert_eq!(dst.get_val_at([]), Some(&3));
        assert_eq!(dst.get_val_at(b"old"), None);
        assert_eq!(dst.get_val_at(b"new"), Some(&4));
        assert_eq!(dst.val_count(), 2);
    }

    #[test]
    fn owned_read_zipper_test() {
        let table = ["A", "AB", "Ab", "ABC", "ABc", "ABCD", "B"];
        let map: PathMap<usize> = table.iter().enumerate().map(|(n, s)| (s, n)).collect();
        let mut zipper = map.into_read_zipper(b"AB");

        let expected = [3, 5, 4];
        let mut i = 0;
        let witness = zipper.witness();
        while let Some(val) = zipper.to_next_get_val_with_witness(&witness) {
            assert_eq!(*val, expected[i]);
            i += 1;
        }

        let map = zipper.into_map();
        assert_eq!(map.val_count(), 7);
    }
    /// This tests [WriteZipper]s with starting paths inside the map
    #[test]
    fn map_write_zipper_test1() {
        let mut map = PathMap::<isize>::new();
        map.set_val_at(b"start:0000:hello", 0);

        let mut z = map.write_zipper_at_path(b"start:0000:");
        z.descend_to(b"goodbye");
        z.set_val(0);
        drop(z);

        assert_eq!(map.val_count(), 2);
        assert_eq!(map.get_val_at(b"start:0000:hello"), Some(&0));
        assert_eq!(map.get_val_at(b"start:0000:goodbye"), Some(&0));

        let mut map = PathMap::<isize>::new();
        map.set_val_at(b"start:0000:hello", 0);
        map.set_val_at(b"start:0001:hello", 1);
        map.set_val_at(b"start:0002:hello", 2);
        map.set_val_at(b"start:0003:hello", 3);

        let mut z = map.write_zipper_at_path(b"start:0000:");
        z.descend_to(b"goodbye");
        z.set_val(0);
        drop(z);

        let mut z = map.write_zipper_at_path(b"start:0001:");
        z.descend_to(b"goodbye");
        z.set_val(1);
        drop(z);

        let mut z = map.write_zipper_at_path(b"start:0002:");
        z.descend_to(b"goodbye");
        z.set_val(2);
        drop(z);

        let mut z = map.write_zipper_at_path(b"start:0003:");
        z.descend_to(b"goodbye");
        z.set_val(3);
        drop(z);

        assert_eq!(map.val_count(), 8);
        assert_eq!(map.get_val_at(b"start:0000:hello"), Some(&0));
        assert_eq!(map.get_val_at(b"start:0000:goodbye"), Some(&0));
        assert_eq!(map.get_val_at(b"start:0003:hello"), Some(&3));
        assert_eq!(map.get_val_at(b"start:0003:goodbye"), Some(&3));
    }
    /// Identical logic to `map_write_zipper_test2`, but tests [WriteZipperOwned]
    #[test]
    fn map_write_zipper_test2() {
        let mut map = PathMap::<isize>::new();
        map.set_val_at(b"start:0000:hello", 0);

        let mut z = map.into_write_zipper(b"start:0000:");
        z.descend_to(b"goodbye");
        z.set_val(0);
        let map = z.into_map();

        assert_eq!(map.val_count(), 2);
        assert_eq!(map.get_val_at(b"start:0000:hello"), Some(&0));
        assert_eq!(map.get_val_at(b"start:0000:goodbye"), Some(&0));

        let mut map = PathMap::<isize>::new();
        map.set_val_at(b"start:0000:hello", 0);
        map.set_val_at(b"start:0001:hello", 1);
        map.set_val_at(b"start:0002:hello", 2);
        map.set_val_at(b"start:0003:hello", 3);

        let mut z = map.into_write_zipper(b"start:0000:");
        z.descend_to(b"goodbye");
        z.set_val(0);
        let map = z.into_map();

        let mut z = map.into_write_zipper(b"start:0001:");
        z.descend_to(b"goodbye");
        z.set_val(1);
        let map = z.into_map();

        let mut z = map.into_write_zipper(b"start:0002:");
        z.descend_to(b"goodbye");
        z.set_val(2);
        let map = z.into_map();

        let mut z = map.into_write_zipper(b"start:0003:");
        z.descend_to(b"goodbye");
        z.set_val(3);
        let map = z.into_map();

        assert_eq!(map.val_count(), 8);
        assert_eq!(map.get_val_at(b"start:0000:hello"), Some(&0));
        assert_eq!(map.get_val_at(b"start:0000:goodbye"), Some(&0));
        assert_eq!(map.get_val_at(b"start:0003:hello"), Some(&3));
        assert_eq!(map.get_val_at(b"start:0003:goodbye"), Some(&3));
    }

    /// Makes a PathMap with a value type that must be dropped, to ensure we don't leak memory
    #[test]
    fn map_values_drop_test() {
        let mut map = PathMap::<String>::new();

        //We want at least one pair node and at least one byte node.
        // "h" is the byte node, "how" is the pair node
        map.set_val_at("hello", "hello".to_string());
        map.set_val_at("howdy", "howdy".to_string());
        map.set_val_at("how do you do", "how do you do".to_string());
        map.set_val_at("hi there", "hi there".to_string());

        assert_eq!(
            map.remove_val_at("how do you do", true),
            Some("how do you do".to_string())
        );
        assert_eq!(map.remove_val_at("hello", true), Some("hello".to_string()));
    }

    #[test]
    fn val_count_root_value() {
        let mut map = PathMap::new();
        map.insert(b"", ());
        map.insert(b"a", ());
        assert_eq!(map.val_count(), 2);
    }

    /// Validates that alg ops on whole maps do the right thing WRT the existence of the root value
    #[test]
    fn map_root_val_test1() {
        let mut map = PathMap::new();
        map.insert(b"b", ());
        map.insert(b"a", ());
        map.insert(b"", ());

        //Validate subtract of identity clears the root val
        let ident_map = map.clone();
        let result_map = map.subtract(&ident_map);
        assert_eq!(result_map.iter().count(), 0);

        //Validate subtract of empty keeps the root val
        let empty_map = PathMap::new();
        let result_map = map.subtract(&empty_map);
        assert_eq!(result_map.iter().count(), 3);

        //Validate subtract of just_root clears it
        let mut just_root_map = PathMap::new();
        just_root_map.insert(b"", ());
        let result_map = map.subtract(&just_root_map);
        assert_eq!(result_map.iter().count(), 2);

        //Validate subtract of all_but_root keeps it
        let mut all_but_root_map = PathMap::new();
        all_but_root_map.insert(b"b", ());
        all_but_root_map.insert(b"a", ());
        let result_map = map.subtract(&all_but_root_map);
        assert_eq!(result_map.iter().count(), 1);

        //Validate meet with empty clears the root val
        let result_map = map.meet(&empty_map);
        assert_eq!(result_map.iter().count(), 0);

        //Validate meet with identity leaves the root val
        let result_map = map.meet(&ident_map);
        assert_eq!(result_map.iter().count(), 3);

        //Validate meet with just_root keeps it
        let result_map = map.meet(&just_root_map);
        assert_eq!(result_map.iter().count(), 1);

        //Validate meet with all_but_root removes it
        let result_map = map.meet(&all_but_root_map);
        assert_eq!(result_map.iter().count(), 2);
    }
}

// PathMap currently keeps its absolute-path methods concrete. A public
// `PathReader`/`PathWriter` split would be API factoring, not a local
// speed fix, because map writes already bypass zipper movement when the
// direct node path is available and fall back to a zipper only for pruning or
// compressed-path repair. Add the traits when a generic caller needs one API
// over both maps and zippers.
