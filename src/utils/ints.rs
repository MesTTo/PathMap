//! Utilities for encoding, decoding, and working with integers represented within paths.  Including a
//! range generator for making efficient ranges to use as arguments to some space-wide operations
//!

use std::collections::HashMap;

use crate::alloc::{global_alloc, Allocator};
use crate::write_zipper::ZipperWriting;
use crate::PathMap;

// Integer ranges are unsigned and half-open: `start <= x < stop`.
// Signed integer paths need a separate order-preserving encoding design.

/// Implemented on integer types that may be encoded as path elements by this code
pub trait PathInteger<const N: usize>:
    num_traits::PrimInt
    + num_traits::ops::saturating::SaturatingAdd
    + num_traits::SaturatingMul
    + std::ops::Mul
    + std::ops::Add
    + std::ops::AddAssign
    + std::ops::BitOrAssign
    + num_traits::FromPrimitive
    + num_traits::ToPrimitive
    + num_traits::ToBytes
    + num_traits::FromBytes<Bytes = [u8; N]>
    + core::hash::Hash
    + core::fmt::Debug
{
}
impl PathInteger<1> for u8 {}
impl PathInteger<2> for u16 {}
impl PathInteger<4> for u32 {}
impl PathInteger<8> for u64 {}
#[cfg(target_pointer_width = "64")]
impl PathInteger<8> for usize {}
impl PathInteger<16> for u128 {}

/// "Bits of Byte". Encode up to 8 natural numbers into a path by combining the bits into the path bytes.
/// Does not pad to number bit length.
/// ```rs,ignore
/// let is = [10, 30, 100];
/// let bob = vec![];
/// > is.iter().map(|x| format!("{:b}", x)).collect::<Vec<_>>()
/// ["1010", "11110", "1100100"]
/// > indices_to_bob(&is[..], &mut bob);
/// > bob.iter().map(|x| format!("{:b}", x)).collect::<Vec<_>>()
/// ["0", "11", "110", "11", "10", "100", "100"]
/// ```
pub fn indices_to_bob<const NUM_SIZE: usize, R: PathInteger<NUM_SIZE>>(
    xs: &[R],
    bob: &mut Vec<u8>,
) -> usize {
    assert!(xs.len() <= 8);
    let steps = xs
        .into_iter()
        .map(|x| (NUM_SIZE * 8) - (x.leading_zeros() as usize))
        .max()
        .unwrap_or(0);
    for c in (0..steps).rev() {
        bob.push(0);
        for i in 0..xs.len() {
            unsafe {
                *bob.last_mut().unwrap_unchecked() |=
                    ((xs[i] >> c) & R::one()).to_u8().unwrap_unchecked() << i;
            }
        }
    }
    steps
}

/// Decodes a "Bits of Byte" path.
/// Requires `xs` to be zeroed.
pub fn bob_to_indices<const NUM_SIZE: usize, R: PathInteger<NUM_SIZE>>(bob: &[u8], xs: &mut [R]) {
    assert!(xs.len() <= 8 && bob.len() <= NUM_SIZE * 8);
    for i in 0..bob.len() {
        for k in 0..xs.len() {
            unsafe {
                xs[k] |= R::from_u8((bob[i] >> k) & 1).unwrap_unchecked() << (bob.len() - 1 - i);
            }
        }
    }
}

/// Encode multiple integers big-endian round-robin wise into a byte path.
/// Does not pad to number byte length.
pub fn indices_to_weave<const NUM_SIZE: usize, R: PathInteger<NUM_SIZE>>(
    xs: &[usize],
    weave: &mut Vec<u8>,
) {
    // let steps = xs.into_iter().map(|x| (NUM_SIZE*8 - (x.leading_zeros() as usize)).div_ceil(8).max(1)).max().unwrap_or(0);
    for c in (0..NUM_SIZE).rev() {
        for i in 0..xs.len() {
            weave.push((xs[i] >> c * 8) as u8)
        }
    }
}

/// Decodes a weave path.
/// Requires `xs` to be zeroed.
pub fn weave_to_indices<const NUM_SIZE: usize, R: PathInteger<NUM_SIZE>>(
    weave: &[u8],
    xs: &mut [R],
) {
    let n = xs.len();
    if n == 0 {
        return;
    }
    assert_eq!(weave.len() % n, 0);
    let steps = weave.len() / n;
    for c in (0..steps).rev() {
        for i in 0..xs.len() {
            unsafe {
                xs[i] |=
                    R::from_u8(weave[n * c + i]).unwrap_unchecked() << (8 * steps - (c + 1) * 8);
            }
        }
    }
}

/// Creates a map that represents an encoded integer range specified by `start`, `stop`, and `step`,
/// with copies of the provided `value` at every path
pub fn gen_int_range<V, const NUM_SIZE: usize, R>(
    start: R,
    stop: R,
    step: R,
    value: V,
) -> PathMap<V>
where
    V: Clone + Send + Sync + Unpin,
    R: PathInteger<NUM_SIZE>,
{
    gen_int_range_in(start, stop, step, value, global_alloc())
}

/// Creates a range as described by [gen_int_range], using the allocator provided
pub fn gen_int_range_in<V, const NUM_SIZE: usize, R, A: Allocator>(
    start: R,
    stop: R,
    step: R,
    value: V,
    alloc: A,
) -> PathMap<V, A>
where
    V: Clone + Send + Sync + Unpin,
    R: PathInteger<NUM_SIZE>,
{
    //Special case for u8s
    if NUM_SIZE == 1 {
        let mut map = PathMap::<V, A>::new_in(alloc);
        let mut i = start;
        while i < stop {
            map.set_val_at(i.to_be_bytes(), value.clone());
            i = i.saturating_add(step);
        }
        return map;
    }

    let mut cache: Vec<HashMap<(R, R), PathMap<V, A>>> = Vec::with_capacity(NUM_SIZE - 1);
    cache.resize(NUM_SIZE - 1, HashMap::new());

    gen_child_level_in(
        NUM_SIZE - 1,
        &mut cache,
        start,
        stop,
        step,
        value.clone(),
        alloc,
    )
}

type Cache<R, V, A> = Vec<HashMap<(R, R), PathMap<V, A>>>;

fn gen_value_level_in<V, const NUM_SIZE: usize, R, A>(
    start: R,
    stop: R,
    step: R,
    value: V,
    alloc: A,
) -> PathMap<V, A>
where
    V: Clone + Send + Sync + Unpin,
    R: PathInteger<NUM_SIZE>,
    A: Allocator,
{
    let mut map = PathMap::<V, A>::new_in(alloc);
    let mut i = start;
    while i < stop {
        let byte = i.to_u8().unwrap();
        map.set_val_at(&[byte], value.clone());
        i = i.saturating_add(step);
    }
    map
}

fn get_from_cache<V, const NUM_SIZE: usize, R, A>(
    level: usize,
    cache: &mut Cache<R, V, A>,
    start: R,
    stop: R,
    step: R,
    value: V,
    alloc: A,
) -> PathMap<V, A>
where
    V: Clone + Send + Sync + Unpin,
    R: PathInteger<NUM_SIZE>,
    A: Allocator,
{
    match cache[level].get(&(start, stop)) {
        Some(map) => {
            // println!("hit level={level} {start:?}-{stop:?}");
            map.clone()
        }
        None => {
            // println!("MISS level={level} {start:?}-{stop:?}");
            let new_map = if level == 0 {
                gen_value_level_in(start, stop, step, value.clone(), alloc)
            } else {
                gen_child_level_in(level, cache, start, stop, step, value.clone(), alloc)
            };
            cache[level].insert((start, stop), new_map.clone());
            new_map
        }
    }
}

pub(crate) fn gen_child_level_in<V, const NUM_SIZE: usize, R, A>(
    level: usize,
    cache: &mut Cache<R, V, A>,
    start: R,
    stop: R,
    step: R,
    value: V,
    alloc: A,
) -> PathMap<V, A>
where
    V: Clone + Send + Sync + Unpin,
    R: PathInteger<NUM_SIZE>,
    A: Allocator,
{
    debug_assert!(start < stop);

    let base = R::from(R::from(256).unwrap().pow(level as u32)).unwrap();
    let one = R::from(1).unwrap();

    let mut map = PathMap::<V, A>::new_in(alloc.clone());

    let mut i = start;
    while i < stop {
        let next_byte_end = ((i / base) + one).saturating_mul(&base);

        //We want a multiple of `step` that gets us to the end of the range, unless one step takes us
        // out of the range
        let jump = ((next_byte_end - i).max(step) / step) * step;
        let range_end = i.saturating_add(jump).min(stop - one);

        //Transer the range in the outer number-space to a range relative to the inner number space
        let child_start = i % base;
        let child_stop = (range_end - i)
            .saturating_add(child_start)
            .saturating_add(one)
            .min(base);

        //Generate the child node, or retrieve it from the cache
        let child_map = get_from_cache(
            level - 1,
            cache,
            child_start,
            child_stop,
            step,
            value.clone(),
            alloc.clone(),
        );
        let higher_byte = (i / base).to_u8().unwrap();
        let path = &[higher_byte];

        let mut wz = map.write_zipper_at_path(path);
        wz.graft_map(child_map);
        drop(wz);

        //Move to the next byte
        i = i.saturating_add(jump);
        if i < next_byte_end {
            i = i.saturating_add(step);
        }
    }

    map
}

#[test]
fn int_range_generator_0() {
    let params: Vec<(u8, u8, u8)> = vec![
        (0, 255, 1),     //Standard step-by-one, fill the whole range
        (2, 16, 3),      //Step by 3, non-zero starting point
        (135, 255, 150), //Step should not cause an overflow
    ];

    for &(start, stop, step) in params.iter() {
        let mut i = start;
        let map = gen_int_range(start, stop, step, ());

        let mut it = map.iter();
        while let Some((path, _)) = it.next() {
            let cn = u8::from_be_bytes(path.try_into().unwrap());
            assert_eq!(cn, i);
            // println!("{cn:?} vs {i:?}");
            i = i.saturating_add(step);
        }
        assert!(i >= stop);
        assert!(i - step < stop);
    }
}

#[test]
fn int_range_generator_1() {
    let params: Vec<(u16, u16, u16)> = vec![
        (0, 20, 1),    //Standard short step-by-one, confined to least-byte
        (500, 530, 1), //Spill across the least-byte boundary
        #[cfg(not(miri))]
        (240, 770, 1), //Span multiple least-byte ranges
        (2, 219, 9),   //A step size that isn't 1
        (175, 751, 25), //A step size that isn't 1, spanning multiple bytes
        (175, 750, 25), //Same as above test, but stop is an even multiple of step so must be excluded
        #[cfg(not(miri))]
        (371, 65535, 101), //A big range with an awkward step
        #[cfg(not(miri))]
        (0, 65535, 1), //The whole range of u16 (minus the last one because ranges are exclusive of end)
    ];

    for &(start, stop, step) in params.iter() {
        let mut i = start;
        let map = gen_int_range(start, stop, step, ());

        let mut it = map.iter();
        while let Some((path, _)) = it.next() {
            let cn = u16::from_be_bytes(path.try_into().unwrap());
            assert_eq!(cn, i);
            // println!("{cn:?} vs {i:?}");
            i = i.saturating_add(step);
        }
        assert!(i >= stop);
        assert!(i - step < stop);
    }
}

#[test]
fn int_range_generator_2() {
    let params: Vec<(u32, u32, u32)> = vec![
        (0, 20, 1),    //Standard short step-by-one, confined to least-byte
        (500, 530, 1), //Spill across the least-byte boundary
        #[cfg(not(miri))]
        (1000, 100000, 1), //Spill across two byte boundaries
        #[cfg(not(miri))]
        (0, 1000000, 3), //A friendly step
        (1234567, 4294967295, 227022703), //A very awkward step (9-digit prime)
                       // (0, 4294967295, 1), //The full range of u32 (disabled because it takes too long to validate)
    ];

    for &(start, stop, step) in params.iter() {
        let mut i = start;
        let map = gen_int_range(start, stop, step, ());

        let mut it = map.iter().enumerate();
        while let Some((_counter, (path, _))) = it.next() {
            let cn = u32::from_be_bytes(path.try_into().unwrap());
            assert_eq!(cn, i);
            // if _counter % 1_000_000 == 0 {
            //     println!("{cn:?} vs {i:?}");
            // }
            i = i.saturating_add(step);
        }
        assert!(i >= stop);
        assert!(i - step < stop);
    }
}

#[cfg(not(miri))]
#[test]
fn int_range_generator_3() {
    //Just doing spot validation becaue validating every entry is too expensive at this level
    let params: Vec<(u64, u64, u64, Vec<u64>, Vec<u64>)> = vec![
        (
            0,
            0xFFFFFFFFFFFFFFFF,
            1,
            vec![0xFFFFFFFFFFFFFFFE, 0, 255, 256, 257, 0x0123456789ABCDEF],
            vec![],
        ), //The whole range
        (
            0xFFF0000000000000,
            0xFFFFFFFFFFFFFFFF,
            0x4000000000000,
            vec![
                0xFFF0000000000000,
                0xFFF4000000000000,
                0xFFF8000000000000,
                0xFFFC000000000000,
            ],
            vec![],
        ),
    ];

    for (start, stop, step, good_list, bad_list) in params.into_iter() {
        let map = gen_int_range(start, stop, step, ());

        // let mut it = map.iter().enumerate();
        // while let Some((_counter, (path, _))) = it.next() {
        //     let cn = u64::from_be_bytes(path.try_into().unwrap());
        //     println!("{cn:x}");
        // }

        for num in good_list {
            assert_eq!(map.get_val_at(num.to_be_bytes()), Some(&()));
        }
        for num in bad_list {
            assert_eq!(map.get_val_at(num.to_be_bytes()), None);
        }
    }
}

#[cfg(not(miri))]
#[test]
fn int_range_generator_4() {
    let start = 2u128.pow(58);
    let end = 2u128.pow(63);
    let step = 3u128 * 7u128 * 11u128 * 2u128.pow(32);
    let map = gen_int_range(start, end, step, ());

    assert_eq!(map.val_count(), 9_005_952);
}

/// This was a failure isolated from one of the benchmarks, but it's been further-simplified into a
/// zipper_head test.  However there is no such thing as a worthless test, so I'll leave it here
#[cfg(not(miri))]
#[test]
fn int_range_generator_5() {
    use crate::zipper::*;

    const K: u64 = 1_000_000_000;

    let mut map = PathMap::new();
    let zh = map.zipper_head();

    let mut buildz = zh.write_zipper_at_exclusive_path(&[0]).unwrap();
    buildz.graft_map(gen_int_range(0, K, 1, ()));
    drop(buildz);
    let mut z = zh.read_zipper_at_path(&[0]).unwrap();

    z.descend_until();
    z.descend_first_byte();
    let _z2 = zh.read_zipper_at_path(z.origin_path()).unwrap();

    z.to_next_sibling_byte();
    z.ascend_byte();
}

#[cfg(not(miri))]
#[test]
fn bob_and_weave_simple() {
    fn assert_bob_and_weave_round_trip(is: [usize; 3]) {
        let mut is_ = [0, 0, 0];
        let mut weave = vec![];
        indices_to_weave::<8, usize>(&is[..], &mut weave);
        weave_to_indices(&weave[..], &mut is_[..]);
        assert_eq!(is, is_);

        let mut is_ = [0, 0, 0];
        let mut bob = vec![];
        indices_to_bob(&is[..], &mut bob);
        bob_to_indices(&bob[..], &mut is_[..]);
        assert_eq!(is, is_);
    }

    assert_bob_and_weave_round_trip([10usize, 30, 100]);
    assert_bob_and_weave_round_trip([3333, 30, 1000]);
}
