//! Temporary structural deduplication for `PathMap`.
//!
//! The current API is intentionally narrow: [`PathMap::merkleize`](crate::PathMap::merkleize)
//! runs one in-memory pass, uses a temporary hash memo, and drops that memo before returning. A
//! persistent or content-addressed memo needs separate ownership and eviction rules, because keeping
//! node references in a long-lived memo changes refcounts and can force copies during later writes.
//! MORK's content-addressed persistence track should build that as a separate design.

use crate::alloc::Allocator;
use crate::gxhash;
use crate::trie_node::*;

/// Statistics created after merkleization
#[derive(Default, Debug)]
pub struct MerkleizeResult {
    /// The hash of the entire trie beneath the root
    pub hash: u128,
    /// The number of shared node references that replaced identical copies during the merkleization
    pub reused: usize,
    /// The number of nodes cloned so descendants could be replaced without mutating existing aliases
    pub cloned: usize,
    /// The number of child links rewritten to point at memoized equivalent subtries
    pub replaced: usize,
}

pub(crate) fn merkleize_impl<V, A>(
    counters: &mut MerkleizeResult,
    memo: &mut gxhash::HashMap<u128, TrieNodeODRc<V, A>>,
    node: &TrieNodeODRc<V, A>,
    value: Option<&V>,
) -> (u128, Option<TrieNodeODRc<V, A>>)
where
    V: Clone + Send + Sync + std::hash::Hash,
    A: Allocator,
{
    // hash = (value, [(path, child_hash)])
    use std::collections::hash_map::Entry;
    use std::hash::Hash;
    const INITIAL_SEED: i64 = 0;
    let mut hasher = gxhash::GxHasher::with_seed(INITIAL_SEED);
    value.hash(&mut hasher);
    let mut replacement = None;

    let node_ref = node.as_tagged();
    let mut it = node_ref.new_iter_token();
    while it != NODE_ITER_FINISHED {
        let (next, path, child, val) = node_ref.next_items(it);
        it = next;
        path.hash(&mut hasher);
        let (child_hash, replace);
        if let Some(child) = child {
            (child_hash, replace) = merkleize_impl(counters, memo, child, val);
            if let Some(replace) = replace {
                let node = replacement.get_or_insert_with(|| {
                    counters.cloned += 1;
                    node.clone()
                });
                counters.replaced += 1;
                node.make_mut().node_replace_child(path, replace);
            }
        } else {
            // value and no child -> pretend there's an empty node
            let mut hasher = gxhash::GxHasher::with_seed(INITIAL_SEED);
            val.hash(&mut hasher);
            child_hash = hasher.finish_u128();
        }
        child_hash.hash(&mut hasher);
    }
    let hash = hasher.finish_u128();
    match memo.entry(hash) {
        Entry::Vacant(entry) => {
            counters.cloned += 1;
            if let Some(replacement) = &replacement {
                entry.insert(replacement.clone());
            } else {
                entry.insert(node.clone());
            }
        }
        // if we've seen the hash before, do the replacement
        Entry::Occupied(entry) => {
            counters.reused += 1;
            replacement = Some(entry.get().clone());
        }
    }
    (hash, replacement)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_btm_merkleize() {
        let paths: &[&[u8]] = &[
            b"axx", b"ayy", b"bxx", b"byy", b"cxx", b"cyy", b"ddxx", b"ddyy",
        ];
        let paths = paths.iter().map(|&path| (path, ()));
        let mut btm = crate::PathMap::from_iter(paths);
        #[cfg(feature = "viz")]
        {
            let mut before = Vec::new();
            use crate::viz::{viz_maps, DrawConfig};
            viz_maps(&[btm.clone()], &DrawConfig::default(), &mut before).unwrap();
            eprintln!("before:");
            eprintln!("```mermaid\n{}```", std::str::from_utf8(&before).unwrap());
        }
        let result = btm.merkleize();
        eprintln!("merkleize result: {result:?}\n");
        #[cfg(feature = "viz")]
        {
            use crate::viz::{viz_maps, DrawConfig};
            let mut after = Vec::new();
            viz_maps(&[btm], &DrawConfig::default(), &mut after).unwrap();
            eprintln!("after:");
            eprintln!("```mermaid\n{}```", std::str::from_utf8(&after).unwrap());
        }
    }
}
