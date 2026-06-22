#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pathmap::{
    alloc::global_alloc,
    zipper::{
        DependentProductZipperG, OverlayZipper, PrefixZipper, ProductZipper, ProductZipperG,
        RestrictZipper, SubtractZipper, ZipperCreation, ZipperInfallibleSubtries,
        ZipperIteration, ZipperMoving, ZipperReadOnlyValues, ZipperSubtries, ZipperValues,
        ZipperWriting, materialize_zipper,
    },
    PathMap,
};
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
enum Op {
    Insert {
        key: Vec<u8>,
        val: u8,
    },
    Remove {
        key: Vec<u8>,
    },
    Get {
        key: Vec<u8>,
    },
    GetMutOverwrite {
        key: Vec<u8>,
        val: u8,
    },
    GetOrSetMutIncrement {
        key: Vec<u8>,
        val: u8,
    },
    ReadZipperGet {
        root: Vec<u8>,
        suffix: Vec<u8>,
    },
    WriteZipperSet {
        root: Vec<u8>,
        suffix: Vec<u8>,
        val: u8,
    },
    WriteZipperRemove {
        root: Vec<u8>,
        suffix: Vec<u8>,
    },
    WriteRootMoveSet {
        key: Vec<u8>,
        val: u8,
    },
    ReadRootMoveGet {
        key: Vec<u8>,
    },
    ZipperHeadReadGet {
        key: Vec<u8>,
    },
    ZipperHeadWriteSet {
        key: Vec<u8>,
        val: u8,
    },
    ZipperHeadWriteRemove {
        key: Vec<u8>,
    },
    TakeGraftRoundTrip {
        root: Vec<u8>,
    },
    GraftGeneratedChildMap {
        root: Vec<u8>,
        child: Vec<u8>,
        child_val: u8,
        root_val: Option<u8>,
    },
    RestrictMaterialize {
        focus: Vec<u8>,
        guard_path: Vec<u8>,
        guard_val: u8,
        guard_root_val: Option<u8>,
    },
    OverlayMaterialize {
        focus: Vec<u8>,
        other_path: Vec<u8>,
        other_val: u8,
        other_root_val: Option<u8>,
        other_dangling_path: Vec<u8>,
    },
    PrefixMaterialize {
        prefix: Vec<u8>,
        source_path: Vec<u8>,
        source_val: u8,
        source_root_val: Option<u8>,
        source_dangling_path: Vec<u8>,
    },
    ProductMaterialize {
        primary_path: Vec<u8>,
        primary_val: u8,
        primary_root_val: Option<u8>,
        primary_dangling_path: Vec<u8>,
        secondary_path: Vec<u8>,
        secondary_val: u8,
        secondary_root_val: Option<u8>,
        secondary_dangling_path: Vec<u8>,
    },
    DependentProductMaterialize {
        primary_path: Vec<u8>,
        primary_val: u8,
        primary_root_val: Option<u8>,
        primary_dangling_path: Vec<u8>,
        secondary_path: Vec<u8>,
        secondary_val: u8,
        secondary_root_val: Option<u8>,
        secondary_dangling_path: Vec<u8>,
    },
    DropHeadProject {
        focus: Vec<u8>,
        byte_cnt: u8,
        prefix_a: Vec<u8>,
        suffix_a: Vec<u8>,
        val_a: u8,
        prefix_b: Vec<u8>,
        suffix_b: Vec<u8>,
        val_b: u8,
        dangling_prefix: Vec<u8>,
        dangling_suffix: Vec<u8>,
    },
    SubtractMaterialize {
        focus: Vec<u8>,
        left_a: Vec<u8>,
        left_b: Vec<u8>,
        right_a: Vec<u8>,
        right_b: Vec<u8>,
        left_root_val: Option<bool>,
        right_root_val: Option<bool>,
    },
    CheckLen,
}

fn joined_path(root: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut path = Vec::with_capacity(root.len() + suffix.len());
    path.extend_from_slice(root);
    path.extend_from_slice(suffix);
    path
}

fn non_empty_path(mut path: Vec<u8>) -> Vec<u8> {
    if path.is_empty() {
        path.push(0);
    }
    path
}

fn exact_len_path(mut path: Vec<u8>, len: usize) -> Vec<u8> {
    path.resize(len, 0);
    path.truncate(len);
    path
}

fn has_strict_prefix(path: &[u8], prefix: &[u8]) -> bool {
    path.len() > prefix.len() && path.starts_with(prefix)
}

fn assert_model(map: &PathMap<u8>, model: &HashMap<Vec<u8>, u8>) {
    assert_eq!(map.val_count(), model.len(), "Length mismatch");
    assert_eq!(map.is_empty(), model.is_empty(), "Empty mismatch");
    for (key, val) in model {
        assert_eq!(
            map.get(key).copied(),
            Some(*val),
            "Model value mismatch for key: {:?}",
            key
        );
    }
}

fn zipper_values<Z, V>(mut zipper: Z) -> Vec<(Vec<u8>, V)>
where
    Z: ZipperIteration + ZipperValues<V>,
    V: Copy + Ord,
{
    let mut values = Vec::new();
    if let Some(value) = zipper.val() {
        values.push((zipper.path().to_vec(), *value));
    }
    while zipper.to_next_val() {
        values.push((zipper.path().to_vec(), *zipper.val().unwrap()));
    }
    values.sort();
    values
}

fn collect_observed_pathspace<Z, V>(zipper: &mut Z, paths: &mut Vec<(Vec<u8>, Option<V>)>)
where
    Z: ZipperMoving + ZipperValues<V>,
    V: Copy + Ord,
{
    if zipper.path_exists() {
        paths.push((zipper.path().to_vec(), zipper.val().copied()));
    }

    let child_count = zipper.child_count();
    for child_idx in 0..child_count {
        assert!(zipper.descend_indexed_byte(child_idx));
        collect_observed_pathspace(zipper, paths);
        assert!(zipper.ascend_byte());
    }
}

fn zipper_observed_pathspace<Z, V>(mut zipper: Z) -> Vec<(Vec<u8>, Option<V>)>
where
    Z: ZipperMoving + ZipperValues<V>,
    V: Copy + Ord,
{
    let mut paths = Vec::new();
    collect_observed_pathspace(&mut zipper, &mut paths);
    paths.sort();
    paths
}

fn terminal_existing_paths<V>(map: &PathMap<V>) -> Vec<Vec<u8>>
where
    V: Clone + Send + Sync + Unpin,
{
    fn collect<Z>(zipper: &mut Z, terminals: &mut Vec<Vec<u8>>)
    where
        Z: ZipperMoving,
    {
        if zipper.path_exists() && !zipper.path().is_empty() && zipper.child_count() == 0 {
            terminals.push(zipper.path().to_vec());
            return;
        }

        let child_count = zipper.child_count();
        for child_idx in 0..child_count {
            assert!(zipper.descend_indexed_byte(child_idx));
            collect(zipper, terminals);
            assert!(zipper.ascend_byte());
        }
    }

    let mut zipper = map.read_zipper();
    let mut terminals = Vec::new();
    collect(&mut zipper, &mut terminals);
    terminals
}

fn expected_terminal_product<V>(primary: &PathMap<V>, secondary: &PathMap<V>) -> PathMap<V>
where
    V: Copy + Send + Sync + Unpin + Ord,
{
    let mut expected = primary.clone();
    let mut secondary_paths = zipper_observed_pathspace(secondary.read_zipper());
    secondary_paths.retain(|(path, _)| !path.is_empty());

    for prefix in terminal_existing_paths(primary) {
        for (suffix, val) in &secondary_paths {
            let mut path = prefix.clone();
            path.extend_from_slice(suffix);
            match val {
                Some(val) => {
                    expected.set_val_at(path, *val);
                }
                None => {
                    expected.create_path(path);
                }
            };
        }
    }

    expected
}

fn expected_drop_head_projection(source: &PathMap<u8>, focus: &[u8], byte_cnt: usize) -> PathMap<u8> {
    let focus_map = source.read_zipper_at_path(focus).make_map();
    let mut expected = PathMap::new();

    for (path, val) in zipper_observed_pathspace(focus_map.read_zipper()) {
        if path.len() < byte_cnt {
            continue;
        }

        let suffix = &path[byte_cnt..];
        match val {
            Some(val) => {
                expected.join_val_at(suffix, val);
            }
            None if !suffix.is_empty() => {
                expected.create_path(suffix);
            }
            None => {}
        }
    }

    expected
}

fuzz_target!(|ops: Vec<Op>| {
    let mut map = PathMap::new();
    let mut model = HashMap::new();

    for op in ops {
        match op {
            Op::Insert { key, val } => {
                let res = map.insert(&key, val);
                let model_res = model.insert(key, val);
                assert_eq!(res, model_res, "Insert mismatch");
            }
            Op::Remove { key } => {
                let res = map.remove(&key);
                let model_res = model.remove(&key);
                assert_eq!(res, model_res, "Remove mismatch for key: {:?}", key);
            }
            Op::Get { key } => {
                let res = map.get(&key).copied();
                let model_res = model.get(&key).copied();
                assert_eq!(res, model_res, "Get mismatch for key: {:?}", key);
            }
            Op::GetMutOverwrite { key, val } => {
                let res = map.get_val_mut_at(&key);
                let model_res = model.get_mut(&key);
                match (res, model_res) {
                    (Some(res), Some(model_res)) => {
                        *res = val;
                        *model_res = val;
                    }
                    (None, None) => {}
                    _ => panic!("Mutable get presence mismatch for key: {:?}", key),
                }
            }
            Op::GetOrSetMutIncrement { key, val } => {
                let res = map.get_val_or_set_mut_at(&key, val);
                let model_res = model.entry(key.clone()).or_insert(val);
                assert_eq!(
                    *res, *model_res,
                    "Get-or-set mutable value mismatch for key: {:?}",
                    key
                );

                let next = res.wrapping_add(1);
                *res = next;
                *model_res = next;
            }
            Op::ReadZipperGet { root, suffix } => {
                let full_path = joined_path(&root, &suffix);
                let mut zipper = map.read_zipper_at_path(&root);
                zipper.descend_to(&suffix);
                let res = zipper.get_val().copied();
                let model_res = model.get(&full_path).copied();
                assert_eq!(
                    res, model_res,
                    "Read zipper get mismatch for key: {:?}",
                    full_path
                );
            }
            Op::WriteZipperSet { root, suffix, val } => {
                let full_path = joined_path(&root, &suffix);
                let mut zipper = map.write_zipper_at_path(&root);
                zipper.descend_to(&suffix);
                let res = zipper.set_val(val);
                let model_res = model.insert(full_path.clone(), val);
                assert_eq!(
                    res, model_res,
                    "Write zipper set mismatch for key: {:?}",
                    full_path
                );
            }
            Op::WriteZipperRemove { root, suffix } => {
                let full_path = joined_path(&root, &suffix);
                let mut zipper = map.write_zipper_at_path(&root);
                zipper.descend_to(&suffix);
                let res = zipper.remove_val(true);
                let model_res = model.remove(&full_path);
                assert_eq!(
                    res, model_res,
                    "Write zipper remove mismatch for key: {:?}",
                    full_path
                );
            }
            Op::WriteRootMoveSet { key, val } => {
                let mut zipper = map.write_zipper();
                zipper.move_to_path(&key);
                let res = zipper.set_val(val);
                let model_res = model.insert(key.clone(), val);
                assert_eq!(
                    res, model_res,
                    "Root write zipper move/set mismatch for key: {:?}",
                    key
                );
            }
            Op::ReadRootMoveGet { key } => {
                let mut zipper = map.read_zipper();
                zipper.move_to_path(&key);
                let res = zipper.get_val().copied();
                let model_res = model.get(&key).copied();
                assert_eq!(
                    res, model_res,
                    "Root read zipper move/get mismatch for key: {:?}",
                    key
                );
            }
            Op::ZipperHeadReadGet { key } => {
                let map_head = map.zipper_head();
                let zipper = map_head
                    .read_zipper_at_path(&key)
                    .expect("fresh ZipperHead should not have path conflicts");
                let res = zipper.val().copied();
                let model_res = model.get(&key).copied();
                assert_eq!(
                    res, model_res,
                    "ZipperHead read zipper get mismatch for key: {:?}",
                    key
                );
            }
            Op::ZipperHeadWriteSet { key, val } => {
                let map_head = map.zipper_head();
                let mut zipper = map_head
                    .write_zipper_at_exclusive_path(&key)
                    .expect("fresh ZipperHead should not have path conflicts");
                let res = zipper.set_val(val);
                let model_res = model.insert(key.clone(), val);
                assert_eq!(
                    res, model_res,
                    "ZipperHead write zipper set mismatch for key: {:?}",
                    key
                );
                map_head.cleanup_write_zipper(zipper);
            }
            Op::ZipperHeadWriteRemove { key } => {
                let map_head = map.zipper_head();
                let mut zipper = map_head
                    .write_zipper_at_exclusive_path(&key)
                    .expect("fresh ZipperHead should not have path conflicts");
                let res = zipper.remove_val(true);
                let model_res = model.remove(&key);
                assert_eq!(
                    res, model_res,
                    "ZipperHead write zipper remove mismatch for key: {:?}",
                    key
                );
                map_head.cleanup_write_zipper(zipper);
            }
            Op::TakeGraftRoundTrip { root } => {
                let taken = {
                    let mut zipper = map.write_zipper_at_path(&root);
                    zipper.take_map(true)
                };

                if let Some(taken) = taken {
                    let mut zipper = map.write_zipper_at_path(&root);
                    zipper.graft_map(taken);
                }

                assert_model(&map, &model);
            }
            Op::GraftGeneratedChildMap {
                root,
                child,
                child_val,
                root_val,
            } => {
                let child = non_empty_path(child);
                let mut grafted = PathMap::new();
                grafted.set_val_at(&child, child_val);
                if let Some(root_val) = root_val {
                    grafted.set_val_at([], root_val);
                }

                {
                    let mut zipper = map.write_zipper_at_path(&root);
                    zipper.graft_map(grafted);
                }

                model.retain(|key, _| !has_strict_prefix(key, &root));
                let full_child_path = joined_path(&root, &child);
                model.insert(full_child_path, child_val);

                match root_val {
                    Some(root_val) => {
                        model.insert(root, root_val);
                    }
                    None => {
                        model.remove(&root);
                    }
                }

                assert_model(&map, &model);
            }
            Op::RestrictMaterialize {
                focus,
                guard_path,
                guard_val,
                guard_root_val,
            } => {
                let mut guard = PathMap::new();
                let guard_path = non_empty_path(guard_path);
                guard.set_val_at(&guard_path, guard_val);
                if let Some(root_val) = guard_root_val {
                    guard.set_val_at([], root_val);
                }

                let eager = map.restrict(&guard);
                let expected = eager.read_zipper_at_path(&focus).make_map();
                let mut lazy = RestrictZipper::new(map.read_zipper(), guard.read_zipper());
                lazy.descend_to(&focus);
                let materialized = lazy
                    .try_make_map()
                    .expect("lazy restrict materialization should be infallible");

                assert_eq!(
                    zipper_values(expected.read_zipper()),
                    zipper_values(materialized.read_zipper()),
                    "Restrict materialization mismatch at focus: {:?}",
                    focus
                );
                assert_model(&map, &model);
            }
            Op::OverlayMaterialize {
                focus,
                other_path,
                other_val,
                other_root_val,
                other_dangling_path,
            } => {
                let mut other = PathMap::new();
                other.set_val_at(&non_empty_path(other_path), other_val);
                other.create_path(&non_empty_path(other_dangling_path));
                if let Some(other_root_val) = other_root_val {
                    other.set_val_at([], other_root_val);
                }

                let eager = map.join(&other);
                let expected = eager.read_zipper_at_path(&focus).make_map();
                let mut lazy = OverlayZipper::new(map.read_zipper(), other.read_zipper());
                lazy.descend_to(&focus);
                let materialized = lazy
                    .try_make_map()
                    .expect("overlay materialization should be infallible");

                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(materialized.read_zipper()),
                    "Overlay materialization mismatch at focus: {:?}",
                    focus
                );
                assert_model(&map, &model);
            }
            Op::PrefixMaterialize {
                prefix,
                source_path,
                source_val,
                source_root_val,
                source_dangling_path,
            } => {
                let prefix = non_empty_path(prefix);
                let mut source = PathMap::new();
                source.set_val_at(&non_empty_path(source_path), source_val);
                source.create_path(&non_empty_path(source_dangling_path));

                let materialized = PrefixZipper::new(prefix.as_slice(), source.read_zipper())
                    .try_make_map()
                    .expect("prefix materialization should be infallible for read zippers");
                let mut expected = source.clone();
                {
                    let mut zipper = expected.write_zipper();
                    assert!(zipper.insert_prefix(&prefix));
                }

                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(materialized.read_zipper()),
                    "Prefix materialization mismatch for prefix: {:?}",
                    prefix
                );

                let mut rooted_source = source.clone();
                if let Some(source_root_val) = source_root_val {
                    rooted_source.set_val_at([], source_root_val);
                }
                let mut focused =
                    PrefixZipper::new(prefix.as_slice(), rooted_source.read_zipper());
                focused.descend_to(&prefix);
                let recovered = focused
                    .try_make_map()
                    .expect("focused prefix materialization should be infallible for read zippers");

                assert_eq!(
                    zipper_observed_pathspace(rooted_source.read_zipper()),
                    zipper_observed_pathspace(recovered.read_zipper()),
                    "Prefix derivative recovery mismatch for prefix: {:?}",
                    prefix
                );
                assert_model(&map, &model);
            }
            Op::ProductMaterialize {
                primary_path,
                primary_val,
                primary_root_val,
                primary_dangling_path,
                secondary_path,
                secondary_val,
                secondary_root_val,
                secondary_dangling_path,
            } => {
                let mut primary = PathMap::new();
                primary.set_val_at(&non_empty_path(primary_path), primary_val);
                primary.create_path(&non_empty_path(primary_dangling_path));
                if let Some(primary_root_val) = primary_root_val {
                    primary.set_val_at([], primary_root_val);
                }

                let mut secondary = PathMap::new();
                secondary.set_val_at(&non_empty_path(secondary_path), secondary_val);
                secondary.create_path(&non_empty_path(secondary_dangling_path));
                if let Some(secondary_root_val) = secondary_root_val {
                    secondary.set_val_at([], secondary_root_val);
                }

                let expected = expected_terminal_product(&primary, &secondary);
                let concrete = materialize_zipper(
                    ProductZipper::new(primary.read_zipper(), [secondary.read_zipper()]),
                    global_alloc(),
                );
                let generic_zipper =
                    ProductZipperG::new(primary.read_zipper(), [secondary.read_zipper()]);
                let generic = materialize_zipper(generic_zipper.clone(), global_alloc());
                let generic_try_map = generic_zipper
                    .try_make_map()
                    .expect("generic product materialization should be infallible");

                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(concrete.read_zipper()),
                    "Concrete product materialization mismatch"
                );
                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(generic.read_zipper()),
                    "Generic product materialization mismatch"
                );
                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(generic_try_map.read_zipper()),
                    "Generic product try_make_map mismatch"
                );
                assert_model(&map, &model);
            }
            Op::DependentProductMaterialize {
                primary_path,
                primary_val,
                primary_root_val,
                primary_dangling_path,
                secondary_path,
                secondary_val,
                secondary_root_val,
                secondary_dangling_path,
            } => {
                let mut primary = PathMap::new();
                primary.set_val_at(&non_empty_path(primary_path), primary_val);
                primary.create_path(&non_empty_path(primary_dangling_path));
                if let Some(primary_root_val) = primary_root_val {
                    primary.set_val_at([], primary_root_val);
                }

                let mut secondary = PathMap::new();
                secondary.set_val_at(&non_empty_path(secondary_path), secondary_val);
                secondary.create_path(&non_empty_path(secondary_dangling_path));
                if let Some(secondary_root_val) = secondary_root_val {
                    secondary.set_val_at([], secondary_root_val);
                }

                let expected = expected_terminal_product(&primary, &secondary);
                let dependent_zipper = DependentProductZipperG::new_enroll(
                    primary.read_zipper(),
                    (),
                    |_, _, factor_idx| {
                        if factor_idx == 0 {
                            ((), Some(secondary.read_zipper()))
                        } else {
                            ((), None)
                        }
                    },
                );
                let dependent = materialize_zipper(dependent_zipper.clone(), global_alloc());
                let dependent_try_map = dependent_zipper
                    .try_make_map()
                    .expect("dependent product materialization should be infallible");

                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(dependent.read_zipper()),
                    "Dependent product materialization mismatch"
                );
                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(dependent_try_map.read_zipper()),
                    "Dependent product try_make_map mismatch"
                );
                assert_model(&map, &model);
            }
            Op::DropHeadProject {
                focus,
                byte_cnt,
                prefix_a,
                suffix_a,
                val_a,
                prefix_b,
                suffix_b,
                val_b,
                dangling_prefix,
                dangling_suffix,
            } => {
                let byte_cnt = 1 + (byte_cnt as usize % 3);
                let mut source = map.clone();
                source.remove_val_at(&focus, false);

                let path_a = joined_path(
                    &joined_path(&focus, &exact_len_path(prefix_a, byte_cnt)),
                    &suffix_a,
                );
                let path_b = joined_path(
                    &joined_path(&focus, &exact_len_path(prefix_b, byte_cnt)),
                    &suffix_b,
                );
                let dangling_path = joined_path(
                    &joined_path(&focus, &exact_len_path(dangling_prefix, byte_cnt)),
                    &dangling_suffix,
                );

                source.set_val_at(path_a, val_a);
                source.set_val_at(path_b, val_b);
                source.create_path(dangling_path);

                let expected = expected_drop_head_projection(&source, &focus, byte_cnt);
                let mut actual = source.clone();
                {
                    let mut zipper = actual.write_zipper_at_path(&focus);
                    let _ = zipper.join_k_path_into(byte_cnt, true);
                }
                let actual_focus = actual.read_zipper_at_path(&focus).make_map();

                assert_eq!(
                    zipper_observed_pathspace(expected.read_zipper()),
                    zipper_observed_pathspace(actual_focus.read_zipper()),
                    "Drop-head projection mismatch at focus {:?} byte_cnt {}",
                    focus,
                    byte_cnt
                );
                assert_model(&map, &model);
            }
            Op::SubtractMaterialize {
                focus,
                left_a,
                left_b,
                right_a,
                right_b,
                left_root_val,
                right_root_val,
            } => {
                let mut left = PathMap::<bool>::new();
                left.set_val_at(&non_empty_path(left_a), true);
                left.set_val_at(&non_empty_path(left_b), false);
                if let Some(left_root_val) = left_root_val {
                    left.set_val_at([], left_root_val);
                }

                let mut right = PathMap::<bool>::new();
                right.set_val_at(&non_empty_path(right_a), true);
                right.set_val_at(&non_empty_path(right_b), false);
                if let Some(right_root_val) = right_root_val {
                    right.set_val_at([], right_root_val);
                }

                let eager = left.subtract(&right);
                let expected = eager.read_zipper_at_path(&focus).make_map();
                let mut lazy = SubtractZipper::new(left.read_zipper(), right.read_zipper());
                lazy.descend_to(&focus);
                let materialized = lazy
                    .try_make_map()
                    .expect("lazy subtract materialization should be infallible");

                assert_eq!(
                    zipper_values(expected.read_zipper()),
                    zipper_values(materialized.read_zipper()),
                    "Subtract materialization mismatch at focus: {:?}",
                    focus
                );
                assert_model(&map, &model);
            }
            Op::CheckLen => {
                // Note: PathMap::val_count() might be O(N), so use sparingly in heavy fuzzing
                // but good for correctness checks.
                assert_model(&map, &model);
            }
        }
    }
});
