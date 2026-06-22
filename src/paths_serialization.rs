//! Functionality for working with the `.paths` data format
//!
//! `.paths` is a compressed trie-based representation suitable for writing to a file.
//!
//! `.paths` data does not contain values, so the `_auxdata` functions allow values to
//! be associated with path indices.

use crate::alloc::Allocator;
use crate::zipper::{
    ZipperIteration, ZipperReadOnlyConditionalIteration, ZipperValues, ZipperWriting,
};
use crate::PathMap;
use crate::TrieValue;
use libz_ng_sys::*;

#[cfg(feature = "nightly")]
#[path = "paths_serialization_nightly.rs"]
mod paths_serialization_nightly;
#[cfg(feature = "nightly")]
pub use paths_serialization_nightly::*;

/// Statistics from a `serialize` operation
#[derive(Debug, Clone, Copy)]
pub struct SerializationStats {
    /// The number of output bytes written to the target
    pub bytes_out: usize,
    /// The number of serialized (uncompressed) bytes
    pub bytes_in: usize,
    /// The total number of paths that were serialized
    pub path_count: usize,
}

/// Statistics from a `deserialize` operation
#[derive(Debug, Clone, Copy)]
pub struct DeserializationStats {
    /// The number of input bytes read from the source
    pub bytes_in: usize,
    /// The number of deserialized (uncompressed) path bytes
    pub bytes_out: usize,
    /// The total number of path insert attempts (i.e. paths in the source)
    pub path_count: usize,
}

/// Serializes each value's path from the focus of `rz` into `.paths` data written to `target`
pub fn serialize_paths<'a, V, W, RZ>(rz: RZ, target: &mut W) -> std::io::Result<SerializationStats>
where
    V: TrieValue,
    RZ: ZipperReadOnlyConditionalIteration<'a, V>,
    W: std::io::Write,
{
    serialize_paths_with_auxdata(rz, target, |_, _, _| {})
}

/// Serializes each value's path from the focus of `rz` into `.paths` data written to `target`
///
/// The `fv` closure is called for each path, permitting values to be serialized separately
/// and associated with path indices
pub fn serialize_paths_with_auxdata<
    'a,
    V: TrieValue,
    RZ: ZipperValues<V> + ZipperIteration,
    W: std::io::Write,
    F: FnMut(usize, &[u8], &V) -> (),
>(
    mut rz: RZ,
    target: &mut W,
    mut fv: F,
) -> std::io::Result<SerializationStats> {
    let mut k = 0;
    serialize_paths_from_funcs(
        target,
        &mut rz,
        |rz| Ok(rz.to_next_val()),
        |rz| {
            let path = rz.path();
            fv(k, path, rz.val().unwrap());
            k += 1;
            Some(path)
        },
    )
}

const PATHS_SERIALIZATION_CHUNK: usize = 4096;

#[allow(invalid_value)]
fn init_deflate_stream(strm: &mut z_stream) {
    // zlib uses a default allocator if the function pointer is null.
    *strm = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let ret = unsafe { zng_deflateInit(strm, 7) };
    assert_eq!(ret, Z_OK);
}

fn write_deflated_input<W: std::io::Write>(
    strm: &mut z_stream,
    output: &mut [u8],
    target: &mut W,
    input: &[u8],
) -> std::io::Result<()> {
    strm.avail_in = input.len() as _;
    strm.next_in = input.as_ptr().cast_mut();

    loop {
        strm.avail_out = output.len() as _;
        strm.next_out = output.as_mut_ptr();
        let ret = unsafe { deflate(strm, Z_NO_FLUSH) };
        assert_ne!(ret, Z_STREAM_ERROR);
        let have = output.len() - strm.avail_out as usize;
        target.write_all(&output[..have])?;
        if strm.avail_out != 0 {
            break;
        }
    }
    assert_eq!(strm.avail_in, 0);
    Ok(())
}

fn write_deflated_path_record<W: std::io::Write>(
    strm: &mut z_stream,
    output: &mut [u8],
    record: &mut [u8],
    target: &mut W,
    path: &[u8],
) -> std::io::Result<()> {
    let len_bytes = (path.len() as u32).to_le_bytes();
    if path.len() <= record.len().saturating_sub(4) {
        record[..4].copy_from_slice(&len_bytes);
        record[4..4 + path.len()].copy_from_slice(path);
        write_deflated_input(strm, output, target, &record[..4 + path.len()])
    } else {
        write_deflated_input(strm, output, target, &len_bytes)?;
        write_deflated_input(strm, output, target, path)
    }
}

fn finish_deflated_output<W: std::io::Write>(
    strm: &mut z_stream,
    output: &mut [u8],
    target: &mut W,
    total_paths: usize,
) -> std::io::Result<SerializationStats> {
    loop {
        strm.avail_out = output.len() as _;
        strm.next_out = output.as_mut_ptr();
        let ret = unsafe { deflate(strm, Z_FINISH) };
        let have = output.len() - strm.avail_out as usize;
        target.write_all(&output[..have])?;
        if ret == Z_STREAM_END {
            break;
        }
        assert_eq!(ret, Z_OK);
    }
    let ret = unsafe { deflateEnd(strm) };
    assert_eq!(ret, Z_OK);

    Ok(SerializationStats {
        bytes_out: strm.total_out,
        bytes_in: strm.total_in,
        path_count: total_paths,
    })
}

/// Generates `.paths` data by invoking arbitrary closures
///
/// The size of the individual path serialization can be double exponential in the size of the PathMap.
///
///NOTE: This function takes two closures because of a limitation in the borrow checker.
/// borrowck isn't smart enough to allow one closure to take a mutable borrow of the `PathSrc`
/// object, and return a const reborrow, and then allow the original mutable reference to
/// be used again after the reborrow was dropped.  Doing the reborrow in the loop works around
/// this limitation.
///
///When the borrow checker becomes more capable, we can try to collapse this function to take
/// one closure instead of two.
pub fn serialize_paths_from_funcs<PathSrc, AdvanceF, PathF, W>(
    target: &mut W,
    src: &mut PathSrc,
    mut advance_f: AdvanceF,
    mut path_f: PathF,
) -> std::io::Result<SerializationStats>
where
    AdvanceF: FnMut(&mut PathSrc) -> std::io::Result<bool>,
    PathF: FnMut(&PathSrc) -> Option<&[u8]>,
    W: std::io::Write,
{
    let mut buffer = [0u8; PATHS_SERIALIZATION_CHUNK];
    let mut path_record = [0u8; PATHS_SERIALIZATION_CHUNK];
    #[allow(invalid_value)]
    let mut strm: z_stream = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    init_deflate_stream(&mut strm);

    let mut total_paths: usize = 0;
    while advance_f(src)? {
        let p = match path_f(src) {
            Some(p) => p,
            None => continue,
        };

        write_deflated_path_record(&mut strm, &mut buffer, &mut path_record, target, p)?;
        total_paths += 1;
    }
    finish_deflated_output(&mut strm, &mut buffer, target, total_paths)
}

/// Deserializes each path from the `.paths` data in `source`, and grafts the resulting data at
/// the focus of `wz`
pub fn deserialize_paths<V: TrieValue, A: Allocator, WZ: ZipperWriting<V, A>, R: std::io::Read>(
    wz: WZ,
    source: R,
    v: V,
) -> std::io::Result<DeserializationStats> {
    deserialize_paths_with_auxdata(wz, source, |_, _| v.clone())
}

/// Deserializes each path from the `.paths` data in `source`, and grafts the resulting data at
/// the focus of `wz`
///
/// Values are constructed with the supplied `fv` closure.
/// See [serialize_paths_with_auxdata]
pub fn deserialize_paths_with_auxdata<
    V: TrieValue,
    A: Allocator,
    WZ: ZipperWriting<V, A>,
    R: std::io::Read,
    F: Fn(usize, &[u8]) -> V,
>(
    mut wz: WZ,
    source: R,
    fv: F,
) -> std::io::Result<DeserializationStats> {
    let mut submap = PathMap::new_in(wz.alloc());
    let r = for_each_deserialized_path(source, |k, p| {
        let v = fv(k, p);
        submap.set_val_at(p, v);
        Ok(())
    });
    wz.graft_map(submap);
    r
}

/// Deserializes each path from the `.paths` data in `source`, calling `f` for each path
pub fn for_each_deserialized_path<
    R: std::io::Read,
    F: FnMut(usize, &[u8]) -> std::io::Result<()>,
>(
    mut source: R,
    mut f: F,
) -> std::io::Result<DeserializationStats> {
    use libz_ng_sys::*;
    const IN: usize = 1024;
    const OUT: usize = 2048;
    let mut ibuffer = [0u8; IN];
    let mut obuffer = [0u8; OUT];
    let mut l = 0u32;
    let mut lbuf = [0u8; 4];
    let mut lbuf_offset = 0;
    let mut finished_path = true;
    let mut total_paths: usize = 0usize;
    #[allow(invalid_value)]
    // zlib uses a default allocator if the function pointer is null.
    let mut strm: z_stream = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
    let mut ret = unsafe { zng_inflateInit(&mut strm) };
    if ret != Z_OK {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "failed to init zlib-ng inflate",
        ));
    }
    let mut wz_buf = vec![];
    // if statement in loop that emulates goto for the many to many ibuffer-obuffer relation
    'reading: loop {
        strm.avail_in = source.read(&mut ibuffer)? as _;
        if strm.avail_in == 0 {
            break;
        }
        strm.next_in = &mut ibuffer as _;

        'decompressing: loop {
            strm.avail_out = OUT as _;
            strm.next_out = obuffer.as_mut_ptr();
            let mut pos = 0usize;

            ret = unsafe { inflate(&mut strm, Z_NO_FLUSH) };
            if ret == Z_STREAM_ERROR {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Z_STREAM_ERROR",
                ));
            }
            if strm.avail_out as usize == OUT {
                if ret == Z_STREAM_END {
                    break 'reading;
                } else {
                    continue 'reading;
                }
            }
            let end = OUT - strm.avail_out as usize;

            'descending: loop {
                if finished_path {
                    let have = (end - pos).min(4 - lbuf_offset);
                    lbuf[lbuf_offset..lbuf_offset + have]
                        .copy_from_slice(&obuffer[pos..pos + have]);
                    pos += have;
                    lbuf_offset += have;
                    if lbuf_offset == 4 {
                        l = u32::from_le_bytes(lbuf);
                        lbuf_offset = 0;
                    } else {
                        if strm.avail_in == 0 {
                            continue 'reading;
                        } else {
                            continue 'decompressing;
                        }
                    }
                }

                if pos + l as usize <= end {
                    wz_buf.extend(&obuffer[pos..pos + l as usize]);
                    f(total_paths, &wz_buf[..])?;
                    wz_buf.clear();
                    total_paths += 1;
                    pos += l as usize;
                    finished_path = true;
                    if pos == end {
                        continue 'decompressing;
                    } else {
                        continue 'descending;
                    }
                } else {
                    wz_buf.extend(&obuffer[pos..end]);
                    finished_path = false;
                    l -= (end - pos) as u32;
                    if strm.avail_in == 0 {
                        continue 'reading;
                    } else {
                        continue 'decompressing;
                    }
                }
            }
        }
    }

    unsafe { inflateEnd(&mut strm) };

    Ok(DeserializationStats {
        bytes_in: strm.total_in,
        bytes_out: strm.total_out,
        path_count: total_paths,
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::zipper::{ZipperIteration, ZipperMoving, ZipperValues};

    fn unit_path_map_from_paths<P>(paths: impl IntoIterator<Item = P>) -> PathMap<()>
    where
        P: AsRef<[u8]>,
    {
        let mut btm = PathMap::new();
        for path in paths {
            btm.set_val_at(path.as_ref(), ());
        }
        btm
    }

    fn deterministic_path(len: usize, mut state: u64) -> Vec<u8> {
        let mut path = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            path.push((state >> 32) as u8);
        }
        path
    }

    fn count_unit_paths(btm: &PathMap<()>) -> usize {
        let mut count = 0;
        let mut rz = btm.read_zipper();
        while rz.to_next_val() {
            count += 1;
        }
        count
    }

    fn assert_unit_paths_round_trip(
        btm: &PathMap<()>,
    ) -> (SerializationStats, DeserializationStats) {
        let expected_path_count = count_unit_paths(btm);
        let mut v = vec![];
        let ser_stats @ SerializationStats {
            bytes_out: c,
            bytes_in: bw,
            path_count: pw,
        } = serialize_paths(btm.read_zipper(), &mut v).expect("serialize paths");
        assert_eq!(pw, expected_path_count);
        println!("ser {} {} {}", c, bw, pw);
        println!("vlen {}", v.len());

        let mut restored_btm = PathMap::new();
        let de_stats @ DeserializationStats {
            bytes_in: c,
            bytes_out: bw,
            path_count: pw,
        } = deserialize_paths(restored_btm.write_zipper(), v.as_slice(), ())
            .expect("deserialize paths");
        assert_eq!(pw, expected_path_count);
        println!("de {} {} {}", c, bw, pw);

        let mut lrz = restored_btm.read_zipper();
        while lrz.to_next_val() {
            assert!(btm.contains(lrz.path()), "{:?}", lrz.path());
        }

        let mut rrz = btm.read_zipper();
        while rrz.to_next_val() {
            assert!(restored_btm.contains(rrz.path()), "{:?}", rrz.path());
        }

        (ser_stats, de_stats)
    }

    #[cfg(not(miri))] // miri really hates the zlib-ng-sys C API
    #[test]
    fn path_serialize_deserialize() {
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
        let btm = unit_path_map_from_paths(rs);
        assert_unit_paths_round_trip(&btm);
    }

    #[cfg(not(miri))] // miri really hates the zlib-ng-sys C API
    #[test]
    fn path_serialize_deserialize_blow_out_buffer() {
        for zeros in 0..10 {
            println!("{zeros} zeros");
            let mut rs = vec![];
            for i in 0..400 {
                rs.push(format!(
                    "{}{}{}{}",
                    "0".repeat(zeros),
                    i / 100,
                    (i / 10) % 10,
                    i % 10
                ))
            }
            let btm = unit_path_map_from_paths(rs);
            assert_unit_paths_round_trip(&btm);
        }
    }

    #[cfg(not(miri))] // miri really hates the zlib-ng-sys C API
    #[test]
    fn path_serialize_deserialize_long_paths() {
        let paths = [
            deterministic_path(2049, 0x4d4f_524b_0000_0001),
            deterministic_path(4097, 0x4d4f_524b_0000_0002),
            deterministic_path(24_000, 0x4d4f_524b_0000_0003),
        ];
        let btm = unit_path_map_from_paths(paths.iter().map(Vec::as_slice));

        let (ser_stats, de_stats) = assert_unit_paths_round_trip(&btm);
        assert!(ser_stats.bytes_in > 4096);
        assert!(ser_stats.bytes_out > 4096);
        assert!(de_stats.bytes_out > 4096);
    }

    #[cfg(not(miri))] // miri really hates the zlib-ng-sys C API
    #[test]
    fn path_serialize_deserialize_values() {
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
        let mut values = vec![];
        let mut v = vec![];
        match serialize_paths_with_auxdata(btm.read_zipper(), &mut v, |c, _p, value| {
            assert_eq!(values.len(), c);
            values.push(*value)
        }) {
            Ok(SerializationStats {
                bytes_out: c,
                bytes_in: bw,
                path_count: pw,
            }) => {
                println!("ser {} {} {}", c, bw, pw);
                println!("vlen {}", v.len());

                let mut restored_btm = PathMap::new();
                match deserialize_paths_with_auxdata(
                    restored_btm.write_zipper(),
                    v.as_slice(),
                    |c, _p| values[c],
                ) {
                    Ok(DeserializationStats {
                        bytes_in: c,
                        bytes_out: bw,
                        path_count: pw,
                    }) => {
                        println!("de {} {} {}", c, bw, pw);

                        let mut lrz = restored_btm.read_zipper();
                        while lrz.to_next_val() {
                            assert_eq!(btm.get_val_at(lrz.path()), Some(lrz.val().unwrap()));
                        }

                        let mut rrz = btm.read_zipper();
                        while rrz.to_next_val() {
                            assert_eq!(
                                restored_btm.get_val_at(rrz.path()),
                                Some(rrz.val().unwrap())
                            );
                        }
                    }
                    Err(e) => {
                        println!("de e {}", e)
                    }
                }
            }
            Err(e) => {
                println!("ser e {}", e)
            }
        }
    }
}
