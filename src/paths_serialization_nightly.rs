use super::*;

/// Returns a coroutine to incrementally serialize into `.paths` data and write to `target`
///
/// Passing `None` signals the end of input.
/// The size of the individual path serialization can be double exponential in the size of the PathMap.
pub fn paths_serialization_sink<'p, W: std::io::Write>(
    target: &mut W,
) -> impl std::ops::Coroutine<Option<&'p [u8]>, Yield = (), Return = std::io::Result<SerializationStats>>
{
    #[coroutine]
    move |i: Option<&'p [u8]>| {
        let mut buffer = [0u8; PATHS_SERIALIZATION_CHUNK];
        let mut path_record = [0u8; PATHS_SERIALIZATION_CHUNK];
        #[allow(invalid_value)]
        let mut strm: z_stream = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        init_deflate_stream(&mut strm);

        let mut total_paths: usize = 0;
        if let Some(mut p) = i {
            loop {
                write_deflated_path_record(&mut strm, &mut buffer, &mut path_record, target, p)?;
                total_paths += 1;
                match yield () {
                    Some(np) => p = np,
                    None => break,
                }
            }
        }
        finish_deflated_output(&mut strm, &mut buffer, target, total_paths)
    }
}

/// Returns a coroutine to incrementally serialize owned path buffers into `.paths` data.
///
/// Passing `None` signals the end of input. Unlike `paths_serialization_sink`, this
/// variant owns the path that remains live while the coroutine is suspended.
pub fn paths_serialization_owned_sink<W: std::io::Write>(
    target: &mut W,
) -> impl std::ops::Coroutine<Option<Vec<u8>>, Yield = (), Return = std::io::Result<SerializationStats>>
{
    #[coroutine]
    move |i: Option<Vec<u8>>| {
        let mut buffer = [0u8; PATHS_SERIALIZATION_CHUNK];
        let mut path_record = [0u8; PATHS_SERIALIZATION_CHUNK];
        #[allow(invalid_value)]
        let mut strm: z_stream = unsafe { std::mem::MaybeUninit::zeroed().assume_init() };
        init_deflate_stream(&mut strm);

        let mut total_paths: usize = 0;
        if let Some(mut p) = i {
            loop {
                let path = p.as_slice();
                write_deflated_path_record(&mut strm, &mut buffer, &mut path_record, target, path)?;
                total_paths += 1;
                match yield () {
                    Some(np) => p = np,
                    None => break,
                }
            }
        }
        finish_deflated_output(&mut strm, &mut buffer, target, total_paths)
    }
}
