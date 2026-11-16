// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Engram n-gram memory: row layout, decode, and hashing.
//!
//! V4.1 Flash carries a hash-addressed n-gram table alongside the weights. It
//! is by far the largest thing in the file — two layers of ~384 M rows at 264
//! bytes each, ~203 GiB of a 365 GiB model — and it is never resident. A decode
//! step reads 24 rows per engram layer by index, one 264-byte row at a time.
//!
//! Row layout, from the GGUF's own `engram.encoding = "e4m3_e8m0_32_row264"`:
//! 256 E4M3 value bytes followed by 8 E8M0 scale bytes, one scale per 32
//! values. Grouped, not interleaved.
//!
//! The decode has a step that is easy to leave out and hard to see missing: a
//! scaled value is rounded to BF16 precision with round-to-nearest-even before
//! it reaches the model. Dropping that leaves every value slightly too precise,
//! which no shape or range check notices.

use anyhow::{Result, bail};

/// Values per engram row.
pub const ENGRAM_DIM: usize = 256;
/// Bytes per stored row: `ENGRAM_DIM` codes + `ENGRAM_DIM / 32` scales.
pub const ENGRAM_ROW_BYTES: usize = 264;
/// Values sharing one E8M0 scale.
pub const ENGRAM_SCALE_GROUP: usize = 32;
/// Engram-carrying layers in the model.
pub const ENGRAM_LAYERS: usize = 2;
/// Tokens of history that address a row.
pub const ENGRAM_NGRAM: usize = 4;
/// Hash heads.
pub const ENGRAM_HEADS: usize = 8;
/// Rows fetched per engram layer per step.
pub const ENGRAM_COLS: usize = 24;
/// History slot holding no token yet.
pub const ENGRAM_DEAD: i32 = -1;

const _: () = assert!(ENGRAM_ROW_BYTES == ENGRAM_DIM + ENGRAM_DIM / ENGRAM_SCALE_GROUP);

/// One FP8-E4M3FN byte as f32, sign included.
///
/// Same grid as [`crate::attn_dispatch`]'s magnitude table, written over the
/// whole byte so the sign bit is handled in one place: for a non-zero exponent
/// the value is `(8 + mantissa) * 2^(exponent - 10)`, which is the same number
/// as `(1 + mantissa/8) * 2^(exponent - 7)`.
#[inline]
pub fn engram_e4m3(byte: u8) -> f32 {
    let exponent = ((byte >> 3) & 15) as i32;
    let mantissa = (byte & 7) as f32;
    let value = if exponent != 0 {
        (8.0 + mantissa) * exp2i(exponent - 10)
    } else {
        mantissa * exp2i(-9)
    };
    if byte & 128 != 0 { -value } else { value }
}

/// `2^n` for the small exponents this decode uses.
#[inline]
fn exp2i(n: i32) -> f32 {
    f32::from_bits((((n + 127) as u32) & 0xff) << 23)
}

/// Round `x` to BF16 precision, round-to-nearest-even, keeping f32 storage.
///
/// The engram table was produced from BF16 activations, so a decoded value that
/// keeps full f32 mantissa is not the number the model was trained against.
#[inline]
pub fn round_to_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// Decode one stored row into `ENGRAM_DIM` f32 values.
///
/// Rejects the codings the table never contains rather than letting them
/// through as NaN: code magnitude 127 is E4M3's NaN and scale 255 is E8M0's,
/// and either means the row was not written by the quantiser.
pub fn decode_engram_row(raw: &[u8], out: &mut [f32]) -> Result<()> {
    if raw.len() != ENGRAM_ROW_BYTES {
        bail!(
            "engram row is {} bytes, expected {ENGRAM_ROW_BYTES}",
            raw.len()
        );
    }
    if out.len() != ENGRAM_DIM {
        bail!("engram output is {} values, expected {ENGRAM_DIM}", out.len());
    }
    for j in 0..ENGRAM_DIM {
        let code = raw[j];
        let scale = raw[ENGRAM_DIM + j / ENGRAM_SCALE_GROUP];
        if (code & 127) == 127 || scale == 255 {
            bail!("engram row holds a NaN coding at value {j} (code {code:#04x}, scale {scale})");
        }
        let value = round_to_bf16(engram_e4m3(code) * exp2i(scale as i32 - 127));
        if !value.is_finite() {
            bail!("engram value {j} decoded to {value}");
        }
        out[j] = value;
    }
    Ok(())
}

/// Addressing side of the engram table: the token map and the per-layer hash
/// constants that turn a token history into row indices.
#[derive(Clone, Debug)]
pub struct EngramLayout {
    /// Full vocabulary id -> compressed id. Length is the model's vocab size.
    pub token_map: Vec<u32>,
    pub compressed_vocab_size: u32,
    /// Compressed id standing in for "no token here".
    pub pad_id: u32,
    /// Table row count per layer.
    pub rows: [u32; ENGRAM_LAYERS],
    pub multipliers: [[u64; ENGRAM_NGRAM]; ENGRAM_LAYERS],
    pub primes: [[u32; ENGRAM_COLS]; ENGRAM_LAYERS],
}

/// Rolling token history, newest first. Copying it snapshots the hash state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramHistory {
    pub tail: [i32; ENGRAM_NGRAM - 1],
}

impl Default for EngramHistory {
    fn default() -> Self {
        Self { tail: [ENGRAM_DEAD; ENGRAM_NGRAM - 1] }
    }
}

impl EngramHistory {
    /// Forget the prefix. Every n-gram crossing the reset is padded.
    pub fn reset(&mut self) {
        self.tail = [ENGRAM_DEAD; ENGRAM_NGRAM - 1];
    }
}

/// Rows addressed per token: every layer's every column.
pub const ENGRAM_ROWS_PER_TOKEN: usize = ENGRAM_LAYERS * ENGRAM_COLS;

impl EngramLayout {
    /// Read the layout out of a model file's metadata.
    ///
    /// The arrays are stored flat and layer-major: `primes` is 48 entries as
    /// `[layer][col]` and `multipliers` 8 as `[layer][ngram]`. Nothing in the
    /// file labels them, so `validate` checks the one invariant that would
    /// catch a wrong split -- see there.
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> Result<Self> {
        let token_map = crate::gguf::meta_u32_array(g, "deepseek41.engram.token_map")
            .ok_or_else(|| anyhow::anyhow!("no deepseek41.engram.token_map"))?;
        let rows_v = crate::gguf::meta_u32_array(g, "deepseek41.engram.rows")
            .ok_or_else(|| anyhow::anyhow!("no deepseek41.engram.rows"))?;
        let primes_v = crate::gguf::meta_u32_array(g, "deepseek41.engram.primes")
            .ok_or_else(|| anyhow::anyhow!("no deepseek41.engram.primes"))?;
        let mult_v = crate::gguf::meta_u64_array(g, "deepseek41.engram.multipliers")
            .ok_or_else(|| anyhow::anyhow!("no deepseek41.engram.multipliers"))?;
        let compressed_vocab_size =
            crate::gguf::meta_u32(g, &["deepseek41.engram.compressed_vocab_size"])
                .ok_or_else(|| anyhow::anyhow!("no engram.compressed_vocab_size"))?;
        let pad_id = crate::gguf::meta_u32(g, &["deepseek41.engram.pad_id"])
            .ok_or_else(|| anyhow::anyhow!("no engram.pad_id"))?;

        if rows_v.len() != ENGRAM_LAYERS {
            bail!("engram.rows has {} entries, expected {ENGRAM_LAYERS}", rows_v.len());
        }
        if primes_v.len() != ENGRAM_LAYERS * ENGRAM_COLS {
            bail!(
                "engram.primes has {} entries, expected {}",
                primes_v.len(),
                ENGRAM_LAYERS * ENGRAM_COLS
            );
        }
        if mult_v.len() != ENGRAM_LAYERS * ENGRAM_NGRAM {
            bail!(
                "engram.multipliers has {} entries, expected {}",
                mult_v.len(),
                ENGRAM_LAYERS * ENGRAM_NGRAM
            );
        }

        let mut primes = [[0u32; ENGRAM_COLS]; ENGRAM_LAYERS];
        let mut multipliers = [[0u64; ENGRAM_NGRAM]; ENGRAM_LAYERS];
        for l in 0..ENGRAM_LAYERS {
            primes[l].copy_from_slice(&primes_v[l * ENGRAM_COLS..(l + 1) * ENGRAM_COLS]);
            multipliers[l].copy_from_slice(&mult_v[l * ENGRAM_NGRAM..(l + 1) * ENGRAM_NGRAM]);
        }
        let out = Self {
            token_map,
            compressed_vocab_size,
            pad_id,
            rows: [rows_v[0], rows_v[1]],
            multipliers,
            primes,
        };
        out.validate()?;
        Ok(out)
    }

    /// Check the layout is self-consistent.
    ///
    /// The load-bearing one is `sum(primes[layer]) == rows[layer]`: each column
    /// owns a disjoint prime-sized band of the table and the bands tile it
    /// exactly. That makes it a real check on the flat arrays' layer-major
    /// split -- a transposed read fails it -- and it also proves every row index
    /// this layout can produce is in range, which is why hashing afterwards
    /// needs no bounds test.
    ///
    /// Multipliers must be odd (an even multiplier loses the low bit of every
    /// id) and small enough that `id * multiplier` cannot overflow into the
    /// sign bit.
    pub fn validate(&self) -> Result<()> {
        if self.token_map.is_empty() {
            bail!("engram token_map is empty");
        }
        if self.compressed_vocab_size == 0 || self.compressed_vocab_size > i32::MAX as u32 {
            bail!("engram compressed_vocab_size {} out of range", self.compressed_vocab_size);
        }
        if self.pad_id >= self.compressed_vocab_size {
            bail!("engram pad_id {} >= compressed vocab {}", self.pad_id, self.compressed_vocab_size);
        }
        for (tok, &c) in self.token_map.iter().enumerate() {
            if c >= self.compressed_vocab_size {
                bail!("token_map[{tok}] = {c} >= compressed vocab {}", self.compressed_vocab_size);
            }
        }
        for l in 0..ENGRAM_LAYERS {
            for (j, &m) in self.multipliers[l].iter().enumerate() {
                if m & 1 == 0 {
                    bail!("engram multiplier[{l}][{j}] = {m} is even");
                }
                if m > i64::MAX as u64 / self.compressed_vocab_size as u64 {
                    bail!("engram multiplier[{l}][{j}] = {m} can overflow");
                }
            }
            let mut total: u64 = 0;
            for (c, &prime) in self.primes[l].iter().enumerate() {
                if prime < 2 {
                    bail!("engram prime[{l}][{c}] = {prime} < 2");
                }
                total += prime as u64;
            }
            if total != self.rows[l] as u64 {
                bail!(
                    "engram layer {l}: primes sum to {total} but the table has {} rows",
                    self.rows[l]
                );
            }
        }
        Ok(())
    }

    /// Row indices for `tokens`, advancing `history`.
    ///
    /// Output is `[token][layer][column]`, `ENGRAM_ROWS_PER_TOKEN` per token,
    /// and INCLUDES masked positions -- the caller suppresses engram there
    /// rather than this producing a short result, so the stride stays fixed.
    ///
    /// `mask`, when given, is one byte per token; zero means the position is
    /// masked out and contributes no token to the history.
    pub fn hash(
        &self,
        history: &mut EngramHistory,
        tokens: &[i32],
        mask: Option<&[u8]>,
        rows: &mut [u32],
    ) -> Result<()> {
        if let Some(m) = mask {
            if m.len() != tokens.len() {
                bail!("engram mask is {} bytes for {} tokens", m.len(), tokens.len());
            }
        }
        let want = tokens.len() * ENGRAM_ROWS_PER_TOKEN;
        if rows.len() != want {
            bail!("engram row buffer is {} entries, expected {want}", rows.len());
        }
        for &t in &history.tail {
            if t < ENGRAM_DEAD || (t >= 0 && t as u32 >= self.compressed_vocab_size) {
                bail!("engram history holds {t}, outside the compressed vocab");
            }
        }
        for (i, &t) in tokens.iter().enumerate() {
            if t < 0 || t as usize >= self.token_map.len() {
                bail!("token {i} = {t} is outside the vocabulary");
            }
        }

        let mut w = 0usize;
        for (i, &tok) in tokens.iter().enumerate() {
            let current = if mask.is_some_and(|m| m[i] == 0) {
                ENGRAM_DEAD
            } else {
                self.token_map[tok as usize] as i32
            };

            // Build the n-gram newest-first. `blocked` is STICKY: once a dead
            // slot is reached everything older is padded too, so a prefix that
            // has not filled up yet cannot alias a real n-gram by reusing
            // whatever happens to sit in the older slots.
            let mut ids = [0u32; ENGRAM_NGRAM];
            let mut blocked = false;
            for j in 0..ENGRAM_NGRAM {
                let id = if j == 0 { current } else { history.tail[j - 1] };
                blocked |= id == ENGRAM_DEAD;
                ids[j] = if blocked { self.pad_id } else { id as u32 };
            }

            for l in 0..ENGRAM_LAYERS {
                let mut hash = (ids[0] as u64).wrapping_mul(self.multipliers[l][0]);
                let mut offset: u32 = 0;
                for j in 1..ENGRAM_NGRAM {
                    hash ^= (ids[j] as u64).wrapping_mul(self.multipliers[l][j]);
                    // Every head in this step shares the one hash; they differ
                    // only by modulus and by which band of the table they
                    // address.
                    for head in 0..ENGRAM_HEADS {
                        let col = (j - 1) * ENGRAM_HEADS + head;
                        let prime = self.primes[l][col];
                        rows[w] = (hash % prime as u64) as u32 + offset;
                        w += 1;
                        offset += prime;
                    }
                }
            }

            for j in (1..ENGRAM_NGRAM - 1).rev() {
                history.tail[j] = history.tail[j - 1];
            }
            history.tail[0] = current;
        }
        debug_assert_eq!(w, want);
        Ok(())
    }
}


/// An open engram table: one file descriptor dedicated to row reads.
///
/// **Deliberately NOT an mmap and not the model's Metal view.** The table is
/// ~101 GiB per layer and every read is a cold random row, so letting it through
/// the page cache would evict resident weights to cache bytes that are never
/// read twice. On macOS the descriptor is opened with `F_NOCACHE` and
/// `F_RDAHEAD(0)` for exactly that reason -- and it is also why a throughput
/// number measured on this path must come from a cold fd, since a cached read
/// reports the cache's bandwidth and not the device's.
pub struct EngramTable {
    file: std::fs::File,
    offset: u64,
    rows: u32,
}

impl EngramTable {
    /// Open `path` as a table of `rows` rows beginning at `offset`.
    ///
    /// Checks the extent lies inside the file, so a row index validated against
    /// `rows` can never read past the end.
    pub fn open(path: &std::path::Path, offset: u64, rows: u32) -> Result<Self> {
        if rows == 0 {
            bail!("engram table has no rows");
        }
        let bytes = rows as u64 * ENGRAM_ROW_BYTES as u64;
        if offset > i64::MAX as u64 || bytes > i64::MAX as u64 - offset {
            bail!("engram table extent {offset}+{bytes} is not addressable");
        }
        let file = std::fs::File::open(path)?;
        let meta = file.metadata()?;
        if !meta.is_file() {
            bail!("engram table {} is not a regular file", path.display());
        }
        if offset + bytes > meta.len() {
            bail!(
                "engram table wants {} bytes at {offset} but {} is {} bytes",
                bytes,
                path.display(),
                meta.len()
            );
        }
        let t = Self { file, offset, rows };
        t.set_uncached()?;
        Ok(t)
    }

    /// Take the descriptor out of the page cache and turn read-ahead off.
    ///
    /// Read-ahead is worse than useless here: rows are hashed, so the bytes
    /// after one row are not the next row wanted.
    #[cfg(target_os = "macos")]
    fn set_uncached(&self) -> Result<()> {
        use std::os::unix::io::AsRawFd;
        let fd = self.file.as_raw_fd();
        // SAFETY: fd is owned by self.file and both commands take an int arg.
        unsafe {
            if libc::fcntl(fd, libc::F_NOCACHE, 1) != 0 {
                bail!("F_NOCACHE on the engram table: {}", std::io::Error::last_os_error());
            }
            if libc::fcntl(fd, libc::F_RDAHEAD, 0) != 0 {
                bail!("F_RDAHEAD on the engram table: {}", std::io::Error::last_os_error());
            }
        }
        Ok(())
    }

    #[cfg(not(target_os = "macos"))]
    fn set_uncached(&self) -> Result<()> {
        // POSIX has no per-descriptor equivalent; posix_fadvise(RANDOM) is the
        // closest and is advisory. Left to the caller's io scheduler.
        Ok(())
    }

    /// Rows in the table.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// Read one row's raw bytes, retrying short and interrupted reads.
    pub fn read_row_raw(&self, row: u32, into: &mut [u8]) -> Result<()> {
        if row >= self.rows {
            bail!("engram row {row} is past the table's {} rows", self.rows);
        }
        if into.len() != ENGRAM_ROW_BYTES {
            bail!("engram row buffer is {} bytes, expected {ENGRAM_ROW_BYTES}", into.len());
        }
        use std::os::unix::fs::FileExt;
        let base = self.offset + row as u64 * ENGRAM_ROW_BYTES as u64;
        let mut done = 0usize;
        while done < ENGRAM_ROW_BYTES {
            match self.file.read_at(&mut into[done..], base + done as u64) {
                Ok(0) => bail!("engram row {row} hit EOF after {done} bytes"),
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    /// Read and decode one row into `ENGRAM_DIM` f32 values.
    pub fn read_row(&self, row: u32, out: &mut [f32]) -> Result<()> {
        let mut raw = [0u8; ENGRAM_ROW_BYTES];
        self.read_row_raw(row, &mut raw)?;
        decode_engram_row(&raw, out)
    }

    /// Read and decode `rows` in order into `out`, which is `[row][ENGRAM_DIM]`.
    ///
    /// Temporary storage is one row, independent of how many rows are asked
    /// for -- the table must never be staged whole.
    pub fn read_rows(&self, rows: &[u32], out: &mut [f32]) -> Result<()> {
        let want = rows.len() * ENGRAM_DIM;
        if out.len() != want {
            bail!("engram output is {} values, expected {want}", out.len());
        }
        // Validate every index first, so a bad request writes nothing.
        for (i, &r) in rows.iter().enumerate() {
            if r >= self.rows {
                bail!("engram row request {i} = {r} is past the table's {} rows", self.rows);
            }
        }
        let mut raw = [0u8; ENGRAM_ROW_BYTES];
        for (i, &r) in rows.iter().enumerate() {
            self.read_row_raw(r, &mut raw)?;
            decode_engram_row(&raw, &mut out[i * ENGRAM_DIM..(i + 1) * ENGRAM_DIM])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row of zero codes decodes to zeros whatever the scale.
    #[test]
    fn zero_codes_decode_to_zero() {
        let mut raw = [0u8; ENGRAM_ROW_BYTES];
        raw[ENGRAM_DIM..].fill(120);
        let mut out = [1.0f32; ENGRAM_DIM];
        decode_engram_row(&raw, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 0.0), "{:?}", &out[..8]);
    }

    /// The E4M3 grid matches the magnitude table the attention path already
    /// uses, so the two decoders cannot drift apart silently.
    #[test]
    fn e4m3_matches_the_attention_grid() {
        for i in 0..127u8 {
            let mine = engram_e4m3(i);
            // (1 + mant/8) * 2^(exp-7), the attention spelling.
            let exp = ((i >> 3) & 15) as i32;
            let mant = (i & 7) as f32;
            let theirs = if exp == 0 {
                mant * 0.001_953_125
            } else {
                (1.0 + mant * 0.125) * 2f32.powi(exp - 7)
            };
            assert_eq!(mine, theirs, "code {i}");
            assert_eq!(engram_e4m3(i | 128), -theirs, "code {} (negative)", i | 128);
        }
    }

    /// The BF16 rounding step is applied, and it is round-to-nearest-EVEN.
    ///
    /// This is the step a reimplementation drops. A decoded value that keeps
    /// its full f32 mantissa is within tolerance of the right answer on every
    /// magnitude check, so only a bit-level assertion catches its absence.
    #[test]
    fn decode_rounds_to_bf16_not_to_full_f32() {
        // BF16 keeps the upper 16 bits, so the tie is lower-half == 0x8000 --
        // NOT 0x4000, which is only a quarter of the way and rounds down under
        // every rule. Both directions of nearest-even, and both are cases where
        // truncation gives a different answer than the reference:
        //   0x3f81_8000  tie, upper 0x3f81 is ODD  -> up to 0x3f82 (trunc: 0x3f81)
        //   0x3f80_8000  tie, upper 0x3f80 is EVEN -> down to 0x3f80
        assert_eq!(round_to_bf16(f32::from_bits(0x3f81_8000)).to_bits(), 0x3f82_0000);
        assert_eq!(round_to_bf16(f32::from_bits(0x3f80_8000)).to_bits(), 0x3f80_0000);
        // either side of the tie goes the ordinary way
        assert_eq!(round_to_bf16(f32::from_bits(0x3f80_4000)).to_bits(), 0x3f80_0000);
        assert_eq!(round_to_bf16(f32::from_bits(0x3f80_c000)).to_bits(), 0x3f81_0000);
        // and a value needing no rounding is untouched
        assert_eq!(round_to_bf16(1.0).to_bits(), 1.0f32.to_bits());

        // End to end: every decoded value must already be a BF16 number.
        let mut raw = [0u8; ENGRAM_ROW_BYTES];
        for (j, b) in raw[..ENGRAM_DIM].iter_mut().enumerate() {
            *b = (j % 120) as u8;
        }
        raw[ENGRAM_DIM..].fill(130);
        let mut out = [0.0f32; ENGRAM_DIM];
        decode_engram_row(&raw, &mut out).unwrap();
        for (j, v) in out.iter().enumerate() {
            assert_eq!(v.to_bits() & 0xffff, 0, "value {j} = {v} kept f32 mantissa bits");
        }
    }

    /// Each scale covers exactly 32 values, so bumping one scale moves one
    /// contiguous group of 32 and nothing else.
    #[test]
    fn one_scale_covers_thirty_two_values() {
        let mut raw = [0u8; ENGRAM_ROW_BYTES];
        raw[..ENGRAM_DIM].fill(0x38); // 1.0
        raw[ENGRAM_DIM..].fill(127); // 2^0
        let mut base = [0.0f32; ENGRAM_DIM];
        decode_engram_row(&raw, &mut base).unwrap();
        assert!(base.iter().all(|&v| v == 1.0));

        raw[ENGRAM_DIM + 3] = 128; // 2^1 over values 96..128
        let mut bumped = [0.0f32; ENGRAM_DIM];
        decode_engram_row(&raw, &mut bumped).unwrap();
        for j in 0..ENGRAM_DIM {
            let want = if (96..128).contains(&j) { 2.0 } else { 1.0 };
            assert_eq!(bumped[j], want, "value {j}");
        }
    }

    /// NaN codings are refused rather than propagated.
    #[test]
    fn nan_codings_are_refused() {
        let mut raw = [0u8; ENGRAM_ROW_BYTES];
        raw[ENGRAM_DIM..].fill(127);
        raw[5] = 127;
        let mut out = [0.0f32; ENGRAM_DIM];
        assert!(decode_engram_row(&raw, &mut out).is_err(), "code 127 is E4M3 NaN");
        raw[5] = 255;
        assert!(decode_engram_row(&raw, &mut out).is_err(), "code 255 is -NaN");

        let mut raw2 = [0u8; ENGRAM_ROW_BYTES];
        raw2[ENGRAM_DIM..].fill(127);
        raw2[ENGRAM_DIM + 1] = 255;
        assert!(decode_engram_row(&raw2, &mut out).is_err(), "scale 255 is E8M0 NaN");
    }


    /// A layout whose primes tile each layer's table exactly, for hash tests.
    fn toy_layout() -> EngramLayout {
        let mut primes = [[0u32; ENGRAM_COLS]; ENGRAM_LAYERS];
        // 24 distinct small primes per layer; the two layers differ so a
        // layer mix-up is visible.
        const P: [u32; 24] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
            83, 89,
        ];
        const Q: [u32; 24] = [
            97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
            181, 191, 193, 197, 199, 211, 223,
        ];
        primes[0] = P;
        primes[1] = Q;
        let rows = [P.iter().sum::<u32>(), Q.iter().sum::<u32>()];
        EngramLayout {
            token_map: (0..64u32).map(|t| t % 16).collect(),
            compressed_vocab_size: 16,
            pad_id: 2,
            rows,
            // Odd, distinct, and inside the validator's overflow bound:
            // i64::MAX / compressed_vocab_size, which is ~2^59 for a vocab of
            // 16. The 64-bit mixing constants do NOT fit here -- the real file's
            // multipliers are smaller because its compressed vocab is 99092.
            multipliers: [
                [0x0123_4567_89ab_cdef, 0x0234_5678_9abc_def1, 0x0345_6789_abcd_ef03, 0x0456_789a_bcde_f005],
                [0x0567_89ab_cdef_0007, 0x0678_9abc_def0_1009, 0x0789_abcd_ef01_200b, 0x079a_bcde_f012_300d],
            ],
            primes,
        }
    }

    /// Every row a layout can produce lands inside its own layer's table, and
    /// each column owns a disjoint band. This is what the prime-sum invariant
    /// buys, so hashing needs no per-row bounds check.
    #[test]
    fn rows_stay_inside_each_layers_table() {
        let l = toy_layout();
        l.validate().unwrap();
        let tokens: Vec<i32> = (0..40).collect();
        let mut rows = vec![0u32; tokens.len() * ENGRAM_ROWS_PER_TOKEN];
        let mut h = EngramHistory::default();
        l.hash(&mut h, &tokens, None, &mut rows).unwrap();

        for (t, chunk) in rows.chunks(ENGRAM_ROWS_PER_TOKEN).enumerate() {
            for layer in 0..ENGRAM_LAYERS {
                let mut offset = 0u32;
                for col in 0..ENGRAM_COLS {
                    let prime = l.primes[layer][col];
                    let r = chunk[layer * ENGRAM_COLS + col];
                    assert!(
                        r >= offset && r < offset + prime,
                        "token {t} layer {layer} col {col}: row {r} outside band [{offset}, {})",
                        offset + prime
                    );
                    assert!(r < l.rows[layer], "row {r} past layer {layer} table");
                    offset += prime;
                }
                assert_eq!(offset, l.rows[layer], "bands must tile the table");
            }
        }
    }

    /// `blocked` is sticky: with a cold history only the newest slot is real,
    /// and everything older pads -- so a 1-token prefix cannot hash like a
    /// 4-token one that happens to share its newest token.
    #[test]
    fn a_cold_history_pads_everything_older() {
        let l = toy_layout();
        let mut cold = EngramHistory::default();
        let mut a = vec![0u32; ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut cold, &[7], None, &mut a).unwrap();

        // Same newest token, but a WARM history: must hash differently.
        let mut warm = EngramHistory { tail: [1, 2, 3] };
        let mut b = vec![0u32; ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut warm, &[7], None, &mut b).unwrap();
        assert_ne!(a, b, "history must participate in the hash");

        // And an explicitly all-pad history reproduces the cold result, which
        // is what "sticky" means: the dead slots became pad_id, not garbage.
        let pad = l.pad_id as i32;
        let mut padded = EngramHistory { tail: [pad, pad, pad] };
        let mut c = vec![0u32; ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut padded, &[7], None, &mut c).unwrap();
        assert_eq!(a, c, "dead slots must hash as pad_id");
    }

    /// The history shifts newest-first, so the same 4-gram reached by different
    /// routes gives the same rows.
    #[test]
    fn history_advances_newest_first() {
        let l = toy_layout();
        let mut stepwise = EngramHistory::default();
        let mut sink = vec![0u32; 3 * ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut stepwise, &[5, 6, 7], None, &mut sink).unwrap();
        // After 5,6,7 the tail is newest-first 7,6,5.
        assert_eq!(stepwise.tail, [l.token_map[7] as i32, l.token_map[6] as i32, l.token_map[5] as i32]);

        let mut one_shot = EngramHistory::default();
        let mut all = vec![0u32; 4 * ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut one_shot, &[5, 6, 7, 8], None, &mut all).unwrap();
        let mut tail_only = vec![0u32; ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut stepwise, &[8], None, &mut tail_only).unwrap();
        assert_eq!(
            &all[3 * ENGRAM_ROWS_PER_TOKEN..],
            &tail_only[..],
            "hashing in one call or three must agree on the 4th token"
        );
    }

    /// A masked position contributes nothing to the history, and still emits a
    /// full stride of rows so the caller's indexing does not shift.
    #[test]
    fn a_masked_position_keeps_the_stride_and_kills_the_ngram() {
        let l = toy_layout();
        let mut h = EngramHistory::default();
        let tokens = [5i32, 6, 7, 8];
        let mask = [1u8, 0, 1, 1];
        let mut rows = vec![0u32; tokens.len() * ENGRAM_ROWS_PER_TOKEN];
        l.hash(&mut h, &tokens, Some(&mask), &mut rows).unwrap();
        assert_eq!(rows.len(), tokens.len() * ENGRAM_ROWS_PER_TOKEN);
        // Token 1 was masked, so it entered the history as DEAD and then aged
        // like any other slot. After 4 tokens the tail is newest-first
        // [8, 7, DEAD] -- the dead slot has reached position 2, and every
        // n-gram that still reaches it is padded from there back.
        assert_eq!(
            h.tail,
            [l.token_map[8] as i32, l.token_map[7] as i32, ENGRAM_DEAD],
            "the masked position ages through the history as dead"
        );

        // The masked position's own rows are the all-pad n-gram: current is
        // DEAD, so sticky blocking pads every slot.
        let mut pad_only = EngramHistory::default();
        let mut expect = vec![0u32; ENGRAM_ROWS_PER_TOKEN];
        let pad = l.pad_id as i32;
        pad_only.tail = [pad, pad, pad];
        l.hash(&mut pad_only, &[l.pad_id as i32], None, &mut expect).unwrap();
        assert_eq!(
            &rows[ENGRAM_ROWS_PER_TOKEN..2 * ENGRAM_ROWS_PER_TOKEN],
            &expect[..],
            "a masked position hashes as the all-pad n-gram"
        );
    }

    /// The prime-sum invariant is what catches a wrong read of the flat arrays.
    /// A layer-transposed split does not tile the table and must be refused.
    #[test]
    fn a_transposed_prime_split_is_refused() {
        let good = toy_layout();
        good.validate().unwrap();

        let mut bad = good.clone();
        let (a, b) = (bad.primes[0][0], bad.primes[1][0]);
        bad.primes[0][0] = b;
        bad.primes[1][0] = a;
        assert!(
            bad.validate().is_err(),
            "moving a prime between layers breaks the tiling and must be caught"
        );

        let mut even = good.clone();
        even.multipliers[0][2] += 1; // make it even
        assert!(even.validate().is_err(), "an even multiplier must be refused");

        let mut small = good.clone();
        small.primes[1][5] = 1;
        assert!(small.validate().is_err(), "a prime below 2 must be refused");

        let mut oob = good.clone();
        oob.token_map[3] = oob.compressed_vocab_size;
        assert!(oob.validate().is_err(), "a token_map entry past the compressed vocab");
    }

    /// Out-of-range input is refused rather than indexing past the map.
    #[test]
    fn hash_refuses_bad_input() {
        let l = toy_layout();
        let mut h = EngramHistory::default();
        let mut rows = vec![0u32; ENGRAM_ROWS_PER_TOKEN];
        assert!(l.hash(&mut h, &[-1], None, &mut rows).is_err(), "negative token");
        assert!(l.hash(&mut h, &[9999], None, &mut rows).is_err(), "token past vocab");
        let mut short = vec![0u32; ENGRAM_ROWS_PER_TOKEN - 1];
        assert!(l.hash(&mut h, &[1], None, &mut short).is_err(), "short row buffer");
        assert!(
            l.hash(&mut h, &[1, 2], Some(&[1]), &mut rows).is_err(),
            "mask length must match"
        );
    }

    /// Against the real model file, when one is pointed at by DS4_V41_GGUF.
    ///
    /// Reads the layout out of the metadata and validates it, which exercises
    /// the layer-major split of `primes`/`multipliers` against the file that
    /// motivated it -- the prime-sum invariant is the check that the split is
    /// the right way round.
    #[test]
    fn real_v41_layout_validates() {
        let Ok(path) = std::env::var("DS4_V41_GGUF") else {
            eprintln!("DS4_V41_GGUF unset; skipping the real-file engram layout check");
            return;
        };
        let g = crate::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open");
        let l = EngramLayout::from_gguf(&g).expect("engram layout");
        assert_eq!(l.token_map.len(), 129_280, "V4.1 vocab");
        assert_eq!(l.compressed_vocab_size, 99_092);
        assert_eq!(l.pad_id, 2);
        for layer in 0..ENGRAM_LAYERS {
            let sum: u64 = l.primes[layer].iter().map(|&p| p as u64).sum();
            assert_eq!(sum, l.rows[layer] as u64, "layer {layer} primes tile the table");
            eprintln!("layer {layer}: {} rows from 24 primes", l.rows[layer]);
        }
        // And the row count matches the engram tensor's own second dimension.
        for (i, t) in g.tensors.iter().filter(|t| t.name.contains("engram_embd")).enumerate() {
            assert_eq!(t.dims[0], ENGRAM_ROW_BYTES as u64, "{}", t.name);
            assert_eq!(t.dims[1], l.rows[i] as u64, "{} rows vs metadata", t.name);
        }
    }


    /// Build a table file with `rows` distinguishable rows after `pad` bytes of
    /// leading junk, so the reader's offset arithmetic has something to get
    /// wrong. Row `i` holds code `i % 120` in every slot and scale 127 (2^0).
    fn write_toy_table(pad: usize, rows: u32) -> (tempfile_shim::TempPath, Vec<Vec<f32>>) {
        use std::io::Write;
        let path = tempfile_shim::TempPath::new("engram_toy");
        let mut f = std::fs::File::create(&path.0).unwrap();
        f.write_all(&vec![0xABu8; pad]).unwrap();
        let mut expect = Vec::new();
        for i in 0..rows {
            let mut raw = [0u8; ENGRAM_ROW_BYTES];
            raw[..ENGRAM_DIM].fill((i % 120) as u8);
            raw[ENGRAM_DIM..].fill(127);
            f.write_all(&raw).unwrap();
            let mut want = vec![0.0f32; ENGRAM_DIM];
            decode_engram_row(&raw, &mut want).unwrap();
            expect.push(want);
        }
        f.flush().unwrap();
        (path, expect)
    }

    /// Rows come back from the offset they were written to, in order, decoded.
    #[test]
    fn table_reads_the_row_it_was_asked_for() {
        let (path, expect) = write_toy_table(4096, 40);
        let t = EngramTable::open(&path.0, 4096, 40).unwrap();
        assert_eq!(t.rows(), 40);

        // Individually, out of order -- a reader that ignored the index would
        // pass an in-order check.
        let mut got = vec![0.0f32; ENGRAM_DIM];
        for &r in &[39u32, 0, 17, 1, 38] {
            t.read_row(r, &mut got).unwrap();
            assert_eq!(got, expect[r as usize], "row {r}");
        }

        // And as a batch, which must preserve request order including repeats.
        let req = [7u32, 7, 0, 39, 12];
        let mut out = vec![0.0f32; req.len() * ENGRAM_DIM];
        t.read_rows(&req, &mut out).unwrap();
        for (i, &r) in req.iter().enumerate() {
            assert_eq!(
                &out[i * ENGRAM_DIM..(i + 1) * ENGRAM_DIM],
                &expect[r as usize][..],
                "batch slot {i} (row {r})"
            );
        }
    }

    /// The extent check is what makes a validated row index safe, so a file
    /// too short for the claimed rows must be refused at open.
    #[test]
    fn table_open_refuses_a_short_file() {
        let (path, _) = write_toy_table(0, 8);
        assert!(EngramTable::open(&path.0, 0, 8).is_ok());
        assert!(
            EngramTable::open(&path.0, 0, 9).is_err(),
            "9 rows do not fit in an 8-row file"
        );
        assert!(
            EngramTable::open(&path.0, ENGRAM_ROW_BYTES as u64, 8).is_err(),
            "8 rows do not fit after skipping one"
        );
        assert!(EngramTable::open(&path.0, 0, 0).is_err(), "a table needs rows");
    }

    /// An out-of-range request is refused, and a bad batch writes nothing.
    #[test]
    fn table_refuses_rows_past_the_end() {
        let (path, _) = write_toy_table(0, 8);
        let t = EngramTable::open(&path.0, 0, 8).unwrap();
        let mut one = vec![0.0f32; ENGRAM_DIM];
        assert!(t.read_row(8, &mut one).is_err(), "row 8 of an 8-row table");

        let mut out = vec![9.0f32; 2 * ENGRAM_DIM];
        assert!(t.read_rows(&[0, 8], &mut out).is_err());
        assert!(
            out.iter().all(|&v| v == 9.0),
            "a rejected batch must not have written a partial result"
        );
    }

    /// Against the real table, when DS4_V41_GGUF points at a model.
    ///
    /// Opens the engram tensor at its real offset and reads rows the hash
    /// actually produces, which ties the three pieces together: layout -> row
    /// indices -> decoded values.
    #[test]
    fn real_v41_table_reads_hashed_rows() {
        let Ok(path) = std::env::var("DS4_V41_GGUF") else {
            eprintln!("DS4_V41_GGUF unset; skipping the real-file engram table check");
            return;
        };
        let p = std::path::Path::new(&path);
        let g = crate::gguf::GgufFile::open(p).expect("open");
        let layout = EngramLayout::from_gguf(&g).expect("layout");

        let t0 = g
            .tensors
            .iter()
            .find(|t| t.name == "blk.1.engram_embd.weight")
            .expect("layer-0 engram tensor");
        let table = EngramTable::open(p, g.tensor_data_offset + t0.offset, layout.rows[0])
            .expect("open table");
        assert_eq!(table.rows(), layout.rows[0]);

        // Hash a short prompt and read layer 0's 24 rows for its last token.
        let tokens: Vec<i32> = vec![100, 200, 300, 400, 500];
        let mut rows = vec![0u32; tokens.len() * ENGRAM_ROWS_PER_TOKEN];
        let mut h = EngramHistory::default();
        layout.hash(&mut h, &tokens, None, &mut rows).expect("hash");
        let last = &rows[(tokens.len() - 1) * ENGRAM_ROWS_PER_TOKEN..];
        let layer0 = &last[..ENGRAM_COLS];

        let mut out = vec![0.0f32; ENGRAM_COLS * ENGRAM_DIM];
        table.read_rows(layer0, &mut out).expect("read hashed rows");
        let finite = out.iter().filter(|v| v.is_finite()).count();
        assert_eq!(finite, out.len(), "every decoded engram value must be finite");
        let peak = out.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.0 && peak < 1e4, "decoded engram peak {peak} is implausible");
        eprintln!(
            "layer 0: read {} hashed rows, peak |v| {peak:.4}, rows {:?}..",
            ENGRAM_COLS,
            &layer0[..4]
        );
    }

    /// Minimal temp-path helper: this crate has no dev-dependency on tempfile
    /// and one test file does not justify adding one.
    mod tempfile_shim {
        pub struct TempPath(pub std::path::PathBuf);
        impl TempPath {
            pub fn new(tag: &str) -> Self {
                let mut p = std::env::temp_dir();
                let n = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                p.push(format!("{tag}_{}_{n}.bin", std::process::id()));
                Self(p)
            }
        }
        impl Drop for TempPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    /// The row is 264 bytes for the reason the constants say it is.
    #[test]
    fn row_layout_is_values_then_scales() {
        assert_eq!(ENGRAM_DIM + ENGRAM_DIM / ENGRAM_SCALE_GROUP, ENGRAM_ROW_BYTES);
        assert_eq!(ENGRAM_DIM / ENGRAM_SCALE_GROUP, 8);
        let mut out = [0.0f32; ENGRAM_DIM];
        assert!(decode_engram_row(&[0u8; 263], &mut out).is_err());
        assert!(decode_engram_row(&[0u8; 265], &mut out).is_err());
    }
}
