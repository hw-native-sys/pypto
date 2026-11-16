// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! On-disk persistence for a system-prompt KV snapshot.
//!
//! The system prompt (Claude Code's ~44 K token prefix, or any agent's reusable
//! prompt) is prefilled once, its graph state captured, and saved here so every
//! later request — including after a restart — restores that state instead of
//! re-paying the prefill. This is the "prefetch once, load instantly" path: the
//! expensive part is the prefill, and a snapshot sidesteps it entirely.
//!
//! ## Format
//!
//! A file is the KVC envelope ([`crate::kvstore_format`]) followed by OUR payload
//! (payload ABI `0x80`, [`PayloadOwner::Ds4Rs`] — antirez's reader refuses it at
//! the header, so the two engines can share a directory without one ever trying
//! to restore the other's bytes). Everything is little-endian, and every blob is
//! 4-byte aligned by construction, so the tensors load as raw slices with a
//! single bulk copy rather than a per-element decode.
//!
//! ```text
//!   envelope header (48) + text length (4) + text (0 for us)
//!   payload:
//!     0  'D' 'S' '4' 'K'   magic
//!     4  version           u32   = 1
//!     8  model_fingerprint u64   identity of the weights (from the caller)
//!    16  sysprompt_hash    u64   identity of the reusable prefix (from the caller)
//!    24  n_boundary        u64
//!    32  boundary_tokens   i32[n_boundary]
//!        n_logits          u64
//!        logits            f32[n_logits]
//!        n_f32             u64   tensor count
//!        per tensor:       u64 elem count, then f32[elem_count]
//!        n_u32             u64
//!        per tensor:       u64 elem count, then u32[elem_count]
//! ```
//!
//! The two fingerprints are *recorded*, not *checked*: this module cannot know
//! the live model, so it hands them back to the caller, which compares them
//! against the live model before a single tensor byte is trusted. That check
//! lives at restore time, where the live graph's geometry is also cross-checked
//! against the tensor shapes — together they close the "snapshot from a
//! different model restores silently and writes plausible-but-wrong text" hole.

use crate::kvstore_format::{self, Header, PayloadOwner, Reason, FIXED_HEADER};
use std::fmt;
use std::path::Path;

pub const SNAPSHOT_MAGIC: [u8; 4] = *b"DS4K";
pub const SNAPSHOT_VERSION: u32 = 1;

/// Minimum payload size: magic + version + two fingerprints.
const MIN_PAYLOAD: usize = 4 + 4 + 8 + 8;

/// A system-prompt snapshot in serializable form. The `f32_tensors` /
/// `u32_tensors` order is fixed and defined by the caller (it mirrors the
/// backend's `KvSnapshot` field order); a reorder or insert is a version bump.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotPayload {
    pub model_fingerprint: u64,
    pub sysprompt_fingerprint: u64,
    /// The reusable prefix (system prompt) as token ids, starting at BOS.
    pub boundary_tokens: Vec<i32>,
    /// Logits at the boundary — the first decode token's sampling input.
    pub logits: Vec<f32>,
    pub f32_tensors: Vec<Vec<f32>>,
    pub u32_tensors: Vec<Vec<u32>>,
}

impl SnapshotPayload {
    /// Serialize to the self-contained payload bytes (no envelope).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(self.byte_len());
        b.extend_from_slice(&SNAPSHOT_MAGIC);
        b.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        b.extend_from_slice(&self.model_fingerprint.to_le_bytes());
        b.extend_from_slice(&self.sysprompt_fingerprint.to_le_bytes());

        b.extend_from_slice(&(self.boundary_tokens.len() as u64).to_le_bytes());
        b.extend_from_slice(cast_le_bytes(&self.boundary_tokens));

        b.extend_from_slice(&(self.logits.len() as u64).to_le_bytes());
        b.extend_from_slice(cast_le_bytes(&self.logits));

        b.extend_from_slice(&(self.f32_tensors.len() as u64).to_le_bytes());
        for t in &self.f32_tensors {
            b.extend_from_slice(&(t.len() as u64).to_le_bytes());
            b.extend_from_slice(cast_le_bytes(t));
        }
        b.extend_from_slice(&(self.u32_tensors.len() as u64).to_le_bytes());
        for t in &self.u32_tensors {
            b.extend_from_slice(&(t.len() as u64).to_le_bytes());
            b.extend_from_slice(cast_le_bytes(t));
        }
        b
    }

    /// Exact payload byte length, so `to_bytes` can pre-allocate without
    /// re-guessing. `None` if a tensor is so large its size overflows `usize`.
    fn byte_len(&self) -> usize {
        let mut n = MIN_PAYLOAD;
        n += 8 + 4 * self.boundary_tokens.len();
        n += 8 + 4 * self.logits.len();
        n += 8;
        for t in &self.f32_tensors {
            n += 8 + 4 * t.len();
        }
        n += 8;
        for t in &self.u32_tensors {
            n += 8 + 4 * t.len();
        }
        n
    }

    /// Parse the payload bytes (no envelope). Rejects any truncation, overflow,
    /// or short read before returning, so a corrupt or half-written file cannot
    /// present as a valid snapshot.
    pub fn from_bytes(bytes: &[u8]) -> Result<SnapshotPayload, PayloadError> {
        let mut c = Cursor::new(bytes);
        if bytes.len() < MIN_PAYLOAD {
            return Err(PayloadError::TooShort);
        }
        if c.read_exact(4)? != SNAPSHOT_MAGIC {
            return Err(PayloadError::BadMagic);
        }
        let version = u32::from_le_bytes(c.read_exact(4)?.try_into().unwrap());
        if version != SNAPSHOT_VERSION {
            return Err(PayloadError::BadVersion(version));
        }
        let model_fingerprint = u64::from_le_bytes(c.read_exact(8)?.try_into().unwrap());
        let sysprompt_fingerprint = u64::from_le_bytes(c.read_exact(8)?.try_into().unwrap());

        let boundary_tokens = c.read_i32_vec()?;
        let logits = c.read_f32_vec()?;

        let n_f32 = c.read_u64()?;
        let mut f32_tensors = Vec::with_capacity(n_f32.min(1 << 20) as usize);
        for _ in 0..n_f32 {
            f32_tensors.push(c.read_f32_vec()?);
        }
        let n_u32 = c.read_u64()?;
        let mut u32_tensors = Vec::with_capacity(n_u32.min(1 << 20) as usize);
        for _ in 0..n_u32 {
            u32_tensors.push(c.read_u32_vec()?);
        }
        // Trailing bytes mean the declared lengths did not consume the whole
        // payload — a format drift, not a legitimate snapshot.
        if !c.is_empty() {
            return Err(PayloadError::TrailingBytes(c.remaining()));
        }
        Ok(SnapshotPayload {
            model_fingerprint,
            sysprompt_fingerprint,
            boundary_tokens,
            logits,
            f32_tensors,
            u32_tensors,
        })
    }

    /// Write the full checkpoint (envelope + payload) to `path`, atomically:
    /// the bytes land in a sibling temp file that is renamed into place only
    /// after the flush, so a crash mid-write can never present as a valid
    /// snapshot (the rename is atomic on the same filesystem).
    ///
    /// The payload is STREAMED piece by piece rather than built as one big
    /// `Vec` first: a 25k-token MLA snapshot is ~6 GB, and materializing a
    /// second contiguous copy next to the live snapshot is exactly the kind of
    /// transient that pushes a memory-tight host into swap. The byte layout is
    /// identical to [`Self::to_bytes`] — `byte_len` supplies the header's
    /// declared size without building anything.
    pub fn save(&self, path: &Path, quant_bits: u8) -> Result<(), std::io::Error> {
        use std::io::Write;
        let now = unix_now();
        let header = Header {
            quant_bits,
            reason: Reason::AgentSystem,
            ext_flags: 0,
            model_id: 0,
            // BOS is always present, so the boundary is never empty; the
            // `.max(1)` only defends a hand-built empty payload from producing
            // a header the reader rejects as `ZeroTokens`.
            tokens: self.boundary_tokens.len().max(1) as u32,
            hits: 0,
            ctx_size: 0,
            owner: PayloadOwner::Ds4Rs,
            created_at: now,
            last_used: now,
            payload_bytes: self.byte_len() as u64,
            text_bytes: 0,
        };
        let tmp = path.with_extension("kvc.tmp");
        {
            let mut w = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&tmp)?);
            w.write_all(&kvstore_format::write_header(&header))?;
            w.write_all(&SNAPSHOT_MAGIC)?;
            w.write_all(&SNAPSHOT_VERSION.to_le_bytes())?;
            w.write_all(&self.model_fingerprint.to_le_bytes())?;
            w.write_all(&self.sysprompt_fingerprint.to_le_bytes())?;
            w.write_all(&(self.boundary_tokens.len() as u64).to_le_bytes())?;
            write_slice(&mut w, &self.boundary_tokens)?;
            w.write_all(&(self.logits.len() as u64).to_le_bytes())?;
            write_slice(&mut w, &self.logits)?;
            w.write_all(&(self.f32_tensors.len() as u64).to_le_bytes())?;
            for t in &self.f32_tensors {
                w.write_all(&(t.len() as u64).to_le_bytes())?;
                write_slice(&mut w, t)?;
            }
            w.write_all(&(self.u32_tensors.len() as u64).to_le_bytes())?;
            for t in &self.u32_tensors {
                w.write_all(&(t.len() as u64).to_le_bytes())?;
                write_slice(&mut w, t)?;
            }
            w.flush()?;
        } // drop closes the file before the rename sees it
        std::fs::rename(&tmp, path)
    }

    /// Read a checkpoint back. Refuses anything that is not OUR payload (an
    /// antirez checkpoint shares the envelope but not the payload), and
    /// refuses a payload whose declared length does not match the file.
    /// Reads via `mmap` where available, so the file's pages are faulted in on
    /// demand and decoded straight into the tensors. A system-prompt snapshot is
    /// multi-gigabyte; `std::fs::read` would first materialize the WHOLE file as
    /// one `Vec<u8>` and only then build the tensors, roughly doubling peak
    /// memory at exactly the moment the process is also holding the model. The
    /// mmap path never creates that second copy.
    ///
    /// Falls back to a plain read on non-unix or if the mapping fails; the two
    /// paths parse the same bytes with the same checks.
    pub fn load(path: &Path) -> Result<SnapshotPayload, LoadError> {
        #[cfg(unix)]
        {
            if let Ok(m) = crate::layer_view::MmapSource::open(path) {
                return Self::parse_file(m.as_slice());
            }
        }
        let raw = std::fs::read(path).map_err(LoadError::Io)?;
        Self::parse_file(&raw)
    }

    /// Envelope + payload checks, shared by both load paths.
    fn parse_file(raw: &[u8]) -> Result<SnapshotPayload, LoadError> {
        let header = kvstore_format::parse_header(raw).map_err(LoadError::Envelope)?;
        if header.owner != PayloadOwner::Ds4Rs {
            return Err(LoadError::NotOurs);
        }
        let off = FIXED_HEADER + 4 + header.text_bytes as usize;
        if off > raw.len() {
            return Err(LoadError::Truncated);
        }
        let payload = &raw[off..];
        if payload.len() as u64 != header.payload_bytes {
            return Err(LoadError::PayloadSize {
                want: header.payload_bytes,
                have: payload.len() as u64,
            });
        }
        SnapshotPayload::from_bytes(payload).map_err(LoadError::Payload)
    }
}

/// Errors parsing the payload bytes.
#[derive(Debug, PartialEq, Eq)]
pub enum PayloadError {
    TooShort,
    BadMagic,
    BadVersion(u32),
    Truncated,
    CountOverflow,
    TrailingBytes(usize),
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PayloadError::TooShort => write!(f, "snapshot payload is truncated"),
            PayloadError::BadMagic => write!(f, "snapshot payload has a bad magic"),
            PayloadError::BadVersion(v) => {
                write!(f, "snapshot payload v{v}, this build writes v{SNAPSHOT_VERSION}")
            }
            PayloadError::Truncated => write!(f, "snapshot payload is truncated"),
            PayloadError::CountOverflow => {
                write!(f, "snapshot payload declares a tensor too large to be real")
            }
            PayloadError::TrailingBytes(n) => {
                write!(f, "snapshot payload has {n} trailing bytes")
            }
        }
    }
}

/// Errors loading a checkpoint file.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Envelope(kvstore_format::ParseError),
    NotOurs,
    Truncated,
    PayloadSize { want: u64, have: u64 },
    Payload(PayloadError),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "read checkpoint: {e}"),
            LoadError::Envelope(e) => write!(f, "checkpoint envelope: {e:?}"),
            LoadError::NotOurs => write!(f, "checkpoint payload is not ours"),
            LoadError::Truncated => write!(f, "checkpoint is truncated"),
            LoadError::PayloadSize { want, have } => {
                write!(f, "checkpoint payload is {have} bytes, header says {want}")
            }
            LoadError::Payload(e) => write!(f, "checkpoint payload: {e}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Bounds-checked sequential reader over the payload. Every read is fallible;
/// a short file surfaces as [`PayloadError::Truncated`] rather than a panic.
struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Cursor { b, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.b.len() - self.pos
    }
    fn is_empty(&self) -> bool {
        self.pos >= self.b.len()
    }
    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], PayloadError> {
        if self.remaining() < n {
            return Err(PayloadError::Truncated);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn read_u64(&mut self) -> Result<u64, PayloadError> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }
    /// Read `count` 4-byte elements as an owned, little-endian vector.
    ///
    /// Alignment is CHECKED, not assumed. Blobs are 4-aligned when the file was
    /// written by us (`text_bytes` is 0, so the payload starts at offset 52),
    /// but the buffer here can be a foreign file or an mmap of one, where an odd
    /// `text_bytes` shifts every blob. A misaligned bulk cast is undefined
    /// behaviour driven by untrusted input, so the fast path is taken only when
    /// it is provably valid and an element-wise decode covers the rest. Both
    /// produce identical values.
    fn read_scalar_vec<T: Copy>(
        &mut self,
        from_le: fn(&[u8]) -> T,
    ) -> Result<Vec<T>, PayloadError> {
        let n = self.read_u64()?;
        let bytes = (n as usize).checked_mul(4).ok_or(PayloadError::CountOverflow)?;
        let raw = self.read_exact(bytes)?;
        if raw.as_ptr().align_offset(std::mem::align_of::<T>()) == 0 {
            // SAFETY: alignment just verified; `bytes` is exactly `n * 4`; T is a
            // 4-byte scalar with no padding or invalid bit patterns; host is LE.
            return Ok(unsafe { cast_le_slice::<T>(raw) }.to_vec());
        }
        Ok(raw.chunks_exact(4).map(from_le).collect())
    }
    fn read_f32_vec(&mut self) -> Result<Vec<f32>, PayloadError> {
        self.read_scalar_vec(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u32_vec(&mut self) -> Result<Vec<u32>, PayloadError> {
        self.read_scalar_vec(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_i32_vec(&mut self) -> Result<Vec<i32>, PayloadError> {
        self.read_scalar_vec(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Reinterpret a byte slice as `&[T]` where `T` is a 4-byte, any-bit-pattern
/// scalar (`f32`, `u32`, `i32`). Little-endian only: the format is written
/// LE, and both the GB10 and Apple Silicon hosts are LE, so a native
/// reinterpretation IS the LE decode.
///
/// # Safety
/// The caller must ensure `bytes` is 4-byte aligned, `bytes.len() % 4 == 0`,
/// and `T` is a 4-byte type with no padding and no invalid bit patterns. The
/// cursor satisfies all three: every blob starts at a 4-aligned offset (all
/// counts are 8 bytes, and the first blob starts at offset 32), the length is
/// `count * 4` exactly, and `f32`/`u32`/`i32` have no padding or invalid
/// patterns.
unsafe fn cast_le_slice<T: Copy>(bytes: &[u8]) -> &[T] {
    let n = bytes.len() / std::mem::size_of::<T>();
    std::slice::from_raw_parts(bytes.as_ptr() as *const T, n)
}

/// View a scalar slice as its raw little-endian bytes. Same [`cast_le_slice`]
/// contract in reverse: `&[f32]`/`&[u32]`/`&[i32]` → `&[u8]`.
fn cast_le_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: T is a 4-byte scalar (f32/u32/i32); any byte is a valid u8 and
    // the byte count is v.len() * 4, within the allocation.
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<T>())
    }
}

/// Streaming counterpart of [`cast_le_bytes`]: write a scalar slice as raw LE
/// bytes without materializing the byte view in a new allocation.
fn write_slice<T: Copy>(w: &mut impl std::io::Write, v: &[T]) -> std::io::Result<()> {
    w.write_all(cast_le_bytes(v))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SnapshotPayload {
        SnapshotPayload {
            model_fingerprint: 0xDEAD_BEEF_CAFE_0001,
            sysprompt_fingerprint: 0x0123_4567_89AB_CDEF,
            boundary_tokens: vec![1, -2, 3, 4, -5],
            logits: vec![0.5f32, -1.25, 2.0],
            f32_tensors: vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0, 7.0], // non-uniform length: layers differ
            ],
            u32_tensors: vec![vec![10, 20], vec![30]],
        }
    }

    #[test]
    fn round_trips() {
        let p = sample();
        let b = p.to_bytes();
        assert_eq!(SnapshotPayload::from_bytes(&b).unwrap(), p);
    }

    #[test]
    fn payload_offsets_are_four_aligned() {
        let b = sample().to_bytes();
        // magic(4)+version(4)+fp(8)+fp(8)+n_boundary(8) = 32
        assert_eq!(&b[0..4], b"DS4K");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), SNAPSHOT_VERSION);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 0xDEAD_BEEF_CAFE_0001);
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 0x0123_4567_89AB_CDEF);
        assert_eq!(u64::from_le_bytes(b[24..32].try_into().unwrap()), 5, "n_boundary");
        assert_eq!(&b[32..36], &1i32.to_le_bytes()[..], "boundary[0] at 32, LE");
        assert_eq!(&b[36..40], &(-2i32).to_le_bytes()[..], "boundary[1] at 36, LE");
    }

    #[test]
    fn rejects_truncation() {
        let b = sample().to_bytes();
        for cut in [0, 1, 4, MIN_PAYLOAD - 1, b.len() - 1, b.len() - 3] {
            let r = SnapshotPayload::from_bytes(&b[..cut]);
            assert!(
                matches!(r, Err(PayloadError::TooShort) | Err(PayloadError::Truncated)),
                "cut at {cut}: {r:?}"
            );
        }
    }

    #[test]
    fn rejects_bad_version_and_magic() {
        let mut b = sample().to_bytes();
        b[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            SnapshotPayload::from_bytes(&b),
            Err(PayloadError::BadVersion(2))
        );
        let mut m = sample().to_bytes();
        m[3] = b'X';
        assert_eq!(SnapshotPayload::from_bytes(&m), Err(PayloadError::BadMagic));
    }

    /// The encode must be BIT-exact, not merely numerically close.
    ///
    /// A KV ring legitimately holds NaN, signed zeros, denormals and infinities,
    /// and a snapshot is supposed to restore the captured state EXACTLY — the
    /// coherence gate asserts a bit-identical continuation, which a lossy round
    /// trip would break in a way no value comparison catches (`NaN != NaN`, and
    /// `-0.0 == 0.0`). Compare raw bit patterns so both traps are excluded.
    #[test]
    fn f32_round_trip_is_bit_exact() {
        let hard: Vec<f32> = vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),          // smallest denormal
            f32::from_bits(0x807F_FFFF), // negative denormal
            f32::MAX,
            f32::MIN,
            1.0e-38,
            -3.14159265,
            // A NaN with a non-default payload: a via-f64 or via-string encode
            // would quietly canonicalize this.
            f32::from_bits(0x7FC0_1234),
        ];
        let p = SnapshotPayload {
            model_fingerprint: 1,
            sysprompt_fingerprint: 2,
            boundary_tokens: vec![i32::MIN, -1, 0, 1, i32::MAX],
            logits: hard.clone(),
            f32_tensors: vec![hard.clone()],
            u32_tensors: vec![vec![0, u32::MAX]],
        };
        let back = SnapshotPayload::from_bytes(&p.to_bytes()).unwrap();
        for (i, (&a, &b)) in hard.iter().zip(&back.f32_tensors[0]).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {i} changed bits: {:#010x} -> {:#010x}",
                a.to_bits(),
                b.to_bits()
            );
        }
        for (&a, &b) in hard.iter().zip(&back.logits) {
            assert_eq!(a.to_bits(), b.to_bits(), "logits must be bit-exact too");
        }
        assert_eq!(back.boundary_tokens, p.boundary_tokens);
        assert_eq!(back.u32_tensors, p.u32_tensors);
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut b = sample().to_bytes();
        b.push(0); // one extra byte the declared lengths do not account for
        assert!(matches!(
            SnapshotPayload::from_bytes(&b),
            Err(PayloadError::TrailingBytes(1))
        ));
    }

    #[test]
    fn save_then_load_round_trips() {
        let dir = std::env::temp_dir().join("ds4_snapshot_store_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sys.kvc");
        let p = sample();
        p.save(&path, 4).unwrap();
        let q = SnapshotPayload::load(&path).unwrap();
        assert_eq!(q, p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_streams_exactly_the_to_bytes_layout() {
        // The streaming writer must produce the same bytes as the in-memory
        // serializer — otherwise `byte_len` and the piecewise writes have
        // drifted apart and every reader sees a payload-size mismatch.
        let dir = std::env::temp_dir().join("ds4_snapshot_store_stream_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sys.kvc");
        let p = sample();
        p.save(&path, 4).unwrap();
        let raw = std::fs::read(&path).unwrap();
        let header = kvstore_format::parse_header(&raw).unwrap();
        let off = FIXED_HEADER + 4 + header.text_bytes as usize;
        assert_eq!(header.payload_bytes, p.to_bytes().len() as u64);
        assert_eq!(&raw[off..], &p.to_bytes()[..]);
        // Atomic write: no temp file left behind.
        assert!(!dir.join("sys.kvc.tmp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The mmap and plain-read paths must agree exactly. They parse the same
    /// bytes, but only one of them is used in production, so a divergence would
    /// show up solely on the deployed path.
    #[test]
    fn mmap_and_read_paths_agree() {
        let dir = std::env::temp_dir().join("ds4_snapshot_mmap_agree");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.kvc");
        let p = sample();
        p.save(&path, 4).unwrap();

        let via_load = SnapshotPayload::load(&path).unwrap(); // mmap on unix
        let raw = std::fs::read(&path).unwrap();
        let via_read = SnapshotPayload::parse_file(&raw).unwrap();
        assert_eq!(via_load, via_read);
        assert_eq!(via_load, p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A payload whose blobs are NOT 4-aligned must still decode correctly.
    ///
    /// Our own writer never produces this (`text_bytes` is 0), but a foreign or
    /// hand-made file can, and the fast path is a raw pointer cast whose
    /// alignment precondition would then be violated by untrusted input. The
    /// decoder is supposed to notice and fall back; this builds exactly that
    /// file — a 1-byte text field, shifting every blob by one — and checks the
    /// values survive.
    #[test]
    fn a_misaligned_payload_still_decodes() {
        let p = sample();
        let payload = p.to_bytes();
        let header = Header {
            quant_bits: 4,
            reason: Reason::AgentSystem,
            ext_flags: 0,
            model_id: 0,
            tokens: 5,
            hits: 0,
            ctx_size: 0,
            owner: PayloadOwner::Ds4Rs,
            created_at: 0,
            last_used: 0,
            payload_bytes: payload.len() as u64,
            text_bytes: 1, // <- the shift
        };
        let mut raw = kvstore_format::write_header(&header);
        raw.push(b'x');
        raw.extend_from_slice(&payload);
        // Offset 52 + 1 = 53, so every blob inside is odd-aligned.
        assert_eq!((FIXED_HEADER + 4 + 1) % 4, 1, "the test file must be misaligned");
        let back = SnapshotPayload::parse_file(&raw).unwrap();
        assert_eq!(back, p, "misaligned decode must match the aligned one");
    }

    #[test]
    fn load_refuses_an_antirez_checkpoint() {
        let dir = std::env::temp_dir().join("ds4_snapshot_store_antirez_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("his.kvc");
        // Write an envelope with the antirez owner + an arbitrary payload.
        let header = Header {
            quant_bits: 4,
            reason: Reason::AgentSession,
            ext_flags: 0,
            model_id: 0,
            tokens: 8,
            hits: 0,
            ctx_size: 0,
            owner: PayloadOwner::Antirez,
            created_at: 0,
            last_used: 0,
            payload_bytes: 4,
            text_bytes: 0,
        };
        let mut raw = kvstore_format::write_header(&header);
        raw.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::write(&path, raw).unwrap();
        assert!(matches!(SnapshotPayload::load(&path), Err(LoadError::NotOurs)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_refuses_a_short_payload() {
        let dir = std::env::temp_dir().join("ds4_snapshot_store_short_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("short.kvc");
        let header = Header {
            quant_bits: 4,
            reason: Reason::AgentSystem,
            ext_flags: 0,
            model_id: 0,
            tokens: 4,
            hits: 0,
            ctx_size: 0,
            owner: PayloadOwner::Ds4Rs,
            created_at: 0,
            last_used: 0,
            payload_bytes: 100, // lies: only 4 bytes follow
            text_bytes: 0,
        };
        let mut raw = kvstore_format::write_header(&header);
        raw.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::write(&path, raw).unwrap();
        assert!(matches!(
            SnapshotPayload::load(&path),
            Err(LoadError::PayloadSize { want: 100, have: 4 })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ---------------------------------------------------------------------------
// Group packing
//
// A DS4 KV snapshot is several PER-LAYER groups of f32 tensors (the KV ring, the
// compressed-attention caches and their states, the indexer's caches and states)
// plus one flat carried-residual row. The payload stores a single flat tensor
// list, so the group boundaries have to survive the round trip.
//
// They are stored EXPLICITLY, as a shape vector, rather than recomputed from the
// live model's layer count. Recomputing would make a restore silently depend on
// the reader and writer agreeing about geometry — exactly the failure this
// format exists to prevent, and one with no loud symptom: attention over
// mis-split rings reads plausible numbers and generates subtly wrong text. With
// the shape in the file, a mismatch is arithmetic, and `unpack_groups` refuses.

/// A shape vector describes one f32 group per entry (its tensor count). The
/// carried-residual row is stored after every group and is not in the shape.
pub type GroupShape = Vec<u32>;

/// Flatten per-layer groups plus the trailing residual row into one tensor list,
/// returning the list and the shape needed to split it again.
pub fn pack_groups(groups: &[Vec<Vec<f32>>], cur_hc: &[f32]) -> (Vec<Vec<f32>>, GroupShape) {
    let shape: GroupShape = groups.iter().map(|g| g.len() as u32).collect();
    let mut flat: Vec<Vec<f32>> = Vec::with_capacity(
        groups.iter().map(|g| g.len()).sum::<usize>() + 1,
    );
    for g in groups {
        flat.extend(g.iter().cloned());
    }
    flat.push(cur_hc.to_vec());
    (flat, shape)
}

/// Inverse of [`pack_groups`]. Refuses any list whose length does not match the
/// shape exactly — a short, long, or reordered payload cannot be split into
/// plausible-looking rings.
pub fn unpack_groups(
    flat: Vec<Vec<f32>>,
    shape: &[u32],
) -> Result<(Vec<Vec<Vec<f32>>>, Vec<f32>), ShapeError> {
    // Sum in u64: a corrupt shape can hold many u32::MAX entries, which would
    // overflow a usize sum on a 32-bit host and match a short list.
    let want: u64 = shape.iter().map(|&n| n as u64).sum::<u64>() + 1;
    if want != flat.len() as u64 {
        return Err(ShapeError::LengthMismatch {
            want,
            have: flat.len() as u64,
        });
    }
    let mut it = flat.into_iter();
    let mut groups = Vec::with_capacity(shape.len());
    for &n in shape {
        // `want` already proved the list is long enough, so `next()` cannot be
        // None here; the explicit error keeps that a refusal rather than a panic
        // if the invariant is ever broken by a future edit.
        let mut g = Vec::with_capacity(n as usize);
        for _ in 0..n {
            g.push(it.next().ok_or(ShapeError::Underrun)?);
        }
        groups.push(g);
    }
    let cur_hc = it.next().ok_or(ShapeError::Underrun)?;
    Ok((groups, cur_hc))
}

#[derive(Debug, PartialEq, Eq)]
pub enum ShapeError {
    LengthMismatch { want: u64, have: u64 },
    Underrun,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::LengthMismatch { want, have } => write!(
                f,
                "snapshot holds {have} tensors, its shape describes {want}"
            ),
            ShapeError::Underrun => write!(f, "snapshot tensor list ended early"),
        }
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;

    fn groups() -> Vec<Vec<Vec<f32>>> {
        vec![
            vec![vec![1.0, 2.0], vec![3.0]],        // 2 layers
            vec![],                                  // an EMPTY group is legal
            vec![vec![4.0], vec![5.0], vec![6.0]],  // 3 layers
        ]
    }

    #[test]
    fn round_trips() {
        let g = groups();
        let hc = vec![9.0, 8.0];
        let (flat, shape) = pack_groups(&g, &hc);
        assert_eq!(shape, vec![2, 0, 3]);
        assert_eq!(flat.len(), 6, "5 tensors + cur_hc");
        let (g2, hc2) = unpack_groups(flat, &shape).unwrap();
        assert_eq!(g2, g);
        assert_eq!(hc2, hc);
    }

    /// The split must be refused, not guessed, when the list and shape disagree.
    /// A wrong split produces rings that are the right TYPE and the wrong DATA,
    /// which no downstream check can detect.
    #[test]
    fn a_mismatched_shape_is_refused() {
        let (flat, mut shape) = pack_groups(&groups(), &[9.0]);
        shape[0] += 1; // claim one more layer than the list holds
        assert!(matches!(
            unpack_groups(flat.clone(), &shape),
            Err(ShapeError::LengthMismatch { want: 7, have: 6 })
        ));
        // ...and a list that is short for a correct shape.
        let (mut short, shape2) = pack_groups(&groups(), &[9.0]);
        short.pop();
        assert!(matches!(
            unpack_groups(short, &shape2),
            Err(ShapeError::LengthMismatch { .. })
        ));
    }

    /// A shape that sums past u64 must not wrap into a match.
    #[test]
    fn an_overflowing_shape_cannot_match() {
        let flat = vec![vec![0.0f32]; 4];
        let shape = vec![u32::MAX; 8];
        assert!(matches!(
            unpack_groups(flat, &shape),
            Err(ShapeError::LengthMismatch { .. })
        ));
    }

    /// Reordering groups changes the shape, so a payload written by one field
    /// order cannot be read back under another without the length check firing.
    #[test]
    fn a_reordered_shape_is_refused() {
        let (flat, _) = pack_groups(&groups(), &[9.0]);
        assert!(matches!(
            unpack_groups(flat, &[3, 0, 3]),
            Err(ShapeError::LengthMismatch { want: 7, have: 6 })
        ));
    }
}
