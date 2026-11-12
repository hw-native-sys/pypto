// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Per-layer quantized weight handles.
//!
//! `ModelManifest` validates the GGUF layout but doesn't expose per-tensor
//! handles. `LayerView` does: one struct per layer index, holding a
//! `TensorHandle` per `classify_tensor` role this layer uses (mla_q_a,
//! mla_kv_a, attn_output_a, ffn_gate_exps, …) plus the underlying bytes.
//!
//! The view is the boundary between `AttentionDispatcher` (which wants
//! `&[f32]` activations) and the on-disk GGUF (which holds Q8_0 / F16 /
//! IQ2_XXS / Q2_K blocks). Three on-ramps:
//!
//! 1. **Linux/test path** — `LayerView::from_bytes(...)` over an
//!    in-memory `Vec<u8>` of the GGUF; works for synthetic tiny files
//!    in unit tests. Will NOT scale to 81 GB.
//! 2. **macOS production path (added in #214)** — `LayerView::from_mmap(...)`
//!    over a `memmap2::Mmap`. Same `TensorHandle` shape; only the byte
//!    backing differs. `as_f32(...)` dequantizes on demand.
//! 3. **Test seeds** — `LayerView::synthetic(...)` builds an all-F32 view
//!    with caller-supplied weights for the CPU oracle path.
//!
//! No f32 materialization of full DS4 (≈1 TB). All quantized accessors
//! return `Vec<f32>` for *one* logical tensor at a time — gate/up/down
//! per expert, q_a/q_b/kv_a/kv_b per layer.

#![allow(dead_code)]

use crate::gguf::{classify_tensor, GgmlType, GgufFile};
use anyhow::{anyhow, bail, Context, Result};
use std::collections::BTreeMap;
use std::ops::Deref;
use std::path::Path;

/// Backing byte store for a `LayerViews`. Either an owned `Vec<u8>` (the
/// test/Linux path used by the synthetic GGUF fixtures) or a read-only
/// mmap of the on-disk GGUF (the macOS production path; lazy-paged so
/// the 81 GB DS4 file does not double-count against antirez's mmap).
///
/// Auto-derefs to `&[u8]` so existing call sites that did
/// `&views.bytes` keep compiling — `&ByteBuf` coerces to `&[u8]`.
pub enum ByteBuf {
    Owned(Vec<u8>),
    #[cfg(unix)]
    Mmap(MmapSource),
}

impl ByteBuf {
    pub fn len(&self) -> usize {
        self.as_ref().len()
    }
    pub fn is_empty(&self) -> bool {
        self.as_ref().is_empty()
    }
}

impl AsRef<[u8]> for ByteBuf {
    fn as_ref(&self) -> &[u8] {
        match self {
            ByteBuf::Owned(v) => v.as_slice(),
            #[cfg(unix)]
            ByteBuf::Mmap(m) => m.as_slice(),
        }
    }
}

impl Deref for ByteBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_ref()
    }
}

impl From<Vec<u8>> for ByteBuf {
    fn from(v: Vec<u8>) -> Self {
        ByteBuf::Owned(v)
    }
}

/// Minimal `libc::mmap` wrapper — `PROT_READ` + `MAP_PRIVATE` over an
/// `open(2)`'d file. Unmaps on drop. Held by `ByteBuf::Mmap`; the OS
/// page cache deduplicates these pages with antirez's own mmap of the
/// same path so the 81 GB DS4 GGUF is held exactly once.
#[cfg(unix)]
pub struct MmapSource {
    ptr: *const u8,
    len: usize,
}

#[cfg(unix)]
unsafe impl Send for MmapSource {}
#[cfg(unix)]
unsafe impl Sync for MmapSource {}

#[cfg(unix)]
impl MmapSource {
    /// Open `path` read-only and map the entire file `PROT_READ` /
    /// `MAP_SHARED`. Errors propagate `errno` from `open` or `mmap`.
    ///
    /// MAP_SHARED (not MAP_PRIVATE): when the model bytes are wrapped as
    /// `newBufferWithBytesNoCopy(StorageModeShared)` and pinned GPU-resident,
    /// a MAP_PRIVATE (copy-on-write) region gets its pages instantiated/dirtied
    /// (anonymous, swap-backed) by the GPU residency path — measured as ~72 GB
    /// of dirty+swapped `mapped file` in vmmap, counted in phys_footprint and
    /// thrashing the compressor under pressure. MAP_SHARED keeps the pages
    /// clean file-backed (reclaimable, footprint-excluded) — matching antirez,
    /// whose identical mmap shows 0 dirty / 2.6 GB footprint. We only ever read
    /// (PROT_READ), so MAP_SHARED never writes back to the file.
    pub fn open(path: &Path) -> Result<Self> {
        use std::ffi::CString;
        use std::os::raw::c_int;

        let c_path = CString::new(path.as_os_str().to_string_lossy().as_bytes())
            .with_context(|| format!("path contains NUL: {}", path.display()))?;
        let fd: c_int = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY) };
        if fd < 0 {
            let err = std::io::Error::last_os_error();
            bail!("open({}): {}", path.display(), err);
        }
        let mut st: libc::stat = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::fstat(fd, &mut st) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            unsafe { libc::close(fd) };
            bail!("fstat({}): {}", path.display(), err);
        }
        let len = st.st_size as usize;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        unsafe { libc::close(fd) };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            bail!("mmap({}, len={}): {}", path.display(), len, err);
        }
        Ok(MmapSource {
            ptr: ptr as *const u8,
            len,
        })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

#[cfg(unix)]
impl Drop for MmapSource {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        }
    }
}

/// Handle to one tensor inside the GGUF. Offsets are byte-absolute
/// (i.e. `file_data_offset + tensor_info.offset`); the view stores the
/// absolute offset to skip the addition on every access.
#[derive(Debug, Clone)]
pub struct TensorHandle {
    pub name: String,
    pub role: &'static str,
    pub ttype: GgmlType,
    pub dims: Vec<u64>,
    pub abs_offset: u64,
    pub byte_size: u64,
}

impl TensorHandle {
    /// `Π dims` — total element count across all dims.
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Last-dim length; for 2-D `[out, in]` this is `in` (matvec input dim).
    pub fn last_dim(&self) -> u64 {
        *self.dims.last().unwrap_or(&1)
    }
}

/// Per-layer weight handle bundle. Roles are populated only when the
/// underlying tensor exists in the GGUF (early layers may be dense, ratio-4
/// layers may have indexer tensors, etc.).
#[derive(Debug, Clone, Default)]
pub struct LayerView {
    pub layer_idx: u32,
    pub handles: BTreeMap<&'static str, TensorHandle>,
}

impl LayerView {
    /// Convenience lookup.
    pub fn get(&self, role: &str) -> Option<&TensorHandle> {
        self.handles.get(role)
    }

    /// Asserting variant — bails with a helpful error when the role is
    /// missing. Use for roles the call site requires (`mla_q_a`, `mla_kv_a`)
    /// vs `get` for optional roles (`indexer_*`).
    pub fn require(&self, role: &str) -> Result<&TensorHandle> {
        self.handles
            .get(role)
            .ok_or_else(|| anyhow!("layer {}: missing required role `{}`", self.layer_idx, role))
    }

    /// True iff this layer carries the MLA-compressed KV machinery.
    pub fn is_compressed(&self) -> bool {
        self.handles.contains_key("attn_compressor_kv")
    }

    /// True iff this layer routes through the indexer top-k.
    pub fn uses_indexer(&self) -> bool {
        self.handles.contains_key("indexer_attn_q_b")
    }
}

/// Full per-layer view of the GGUF + a byte-backing source. The byte
/// source can be either an owned `Vec<u8>` (test path) or a future
/// `Mmap` handle (`#214`).
pub struct LayerViews {
    /// File data offset (`GgufFile::tensor_data_offset`) — added to each
    /// `TensorInfo::offset` to form `TensorHandle::abs_offset`.
    pub data_offset: u64,
    /// Backing bytes — owned `Vec<u8>` for unit tests, or a `MAP_PRIVATE`
    /// mmap of the GGUF for production. Both deref to `&[u8]`.
    pub bytes: ByteBuf,
    /// One per layer index in [0, n_layers).
    pub per_layer: Vec<LayerView>,
    /// Global (non-layered) handles: embed, lm_head, final_norm, etc.
    pub global: BTreeMap<&'static str, TensorHandle>,
}

impl LayerViews {
    /// Build views from a parsed GGUF + the underlying byte buffer.
    /// `n_layers` is taken from the manifest; tensors whose name embeds
    /// a `blk.N.` prefix are routed to layer N, everything else goes
    /// into `global`.
    pub fn from_gguf(g: &GgufFile, bytes: impl Into<ByteBuf>, n_layers: u32) -> Result<Self> {
        let bytes = bytes.into();
        let mut per_layer: Vec<LayerView> = (0..n_layers)
            .map(|i| LayerView {
                layer_idx: i,
                handles: BTreeMap::new(),
            })
            .collect();
        let mut global: BTreeMap<&'static str, TensorHandle> = BTreeMap::new();

        for t in &g.tensors {
            let role = classify_tensor(&t.name);
            let layer_idx = parse_blk_idx(&t.name);
            let handle = TensorHandle {
                name: t.name.clone(),
                role,
                ttype: t.ttype,
                dims: t.dims.clone(),
                // Both terms are untrusted u64s from the GGUF header. Saturating
                // add: a forged `t.offset` near u64::MAX must NOT wrap into a small
                // in-range `abs_offset` (which would slice valid-but-wrong bytes);
                // saturating to u64::MAX makes `bytes_for`'s `end > len` check
                // reject it cleanly. Also avoids a debug-build overflow panic.
                abs_offset: g.tensor_data_offset.saturating_add(t.offset),
                byte_size: g.tensor_byte_size(t),
            };
            match layer_idx {
                Some(idx) if idx < n_layers => {
                    per_layer[idx as usize].handles.insert(role, handle);
                }
                Some(idx) => {
                    bail!("tensor {:?}: blk.{} but n_layers={}", t.name, idx, n_layers);
                }
                None => {
                    global.insert(role, handle);
                }
            }
        }

        Ok(Self {
            data_offset: g.tensor_data_offset,
            bytes,
            per_layer,
            global,
        })
    }

    /// One-shot open: mmap the GGUF and build the views. Used by both
    /// test and production paths — `MAP_PRIVATE` is lazy-paged so a
    /// header-only build (small GGUF tests) doesn't fault the tail.
    #[cfg(unix)]
    pub fn open(path: &Path, n_layers: u32) -> Result<Self> {
        let mmap = MmapSource::open(path)?;
        let g = GgufFile::open(path)?;
        Self::from_gguf(&g, ByteBuf::Mmap(mmap), n_layers)
    }

    /// Fallback for non-Unix builds: read the whole file. Only used by
    /// in-repo unit tests, which use tiny synthetic GGUFs.
    #[cfg(not(unix))]
    pub fn open(path: &Path, n_layers: u32) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let g = GgufFile::open(path)?;
        Self::from_gguf(&g, bytes, n_layers)
    }

    /// Return the layer view; panics in debug if `idx` is out of range.
    pub fn layer(&self, idx: u32) -> &LayerView {
        &self.per_layer[idx as usize]
    }

    /// Raw byte slice for a handle. Bounds-checked.
    /// The tensor's bytes.
    ///
    /// `Cow` rather than `&[u8]`: a GGUF view BORROWS from its single mmap, while a W8A8 package
    /// spans 74 files and can only OWN what it returns. Returning `Cow::Borrowed` here keeps the
    /// GGUF path allocation-free, and lets one decoder loop serve both containers.
    pub fn bytes_for(&self, h: &TensorHandle) -> Result<std::borrow::Cow<'_, [u8]>> {
        let start = h.abs_offset as usize;
        let end = start
            .checked_add(h.byte_size as usize)
            .ok_or_else(|| anyhow!("byte_size overflow for {:?}", h.name))?;
        if end > self.bytes.len() {
            bail!(
                "tensor {:?} byte range {}..{} out of file size {}",
                h.name,
                start,
                end,
                self.bytes.len()
            );
        }
        Ok(std::borrow::Cow::Borrowed(&self.bytes[start..end]))
    }

    /// Dequantize an entire tensor to f32. Supports F32 / F16 directly
    /// in this crate; defers heavy quants (Q4_K / Q2_K / IQ2_XXS) to a
    /// caller-provided dequantizer (typically the host-side `cpu_via_dequant` helper).
    ///
    /// Returns an error for quant types not supported by this minimal
    /// in-crate path — caller should plumb through to `the accelerator host crate`.
    pub fn dequant_f32_simple(&self, h: &TensorHandle) -> Result<Vec<f32>> {
        let raw = self.bytes_for(h)?;
        match h.ttype {
            GgmlType::F32 => {
                let n = h.n_elems() as usize;
                if raw.len() != n * 4 {
                    bail!(
                        "tensor {:?}: F32 byte size {} != 4×{}",
                        h.name,
                        raw.len(),
                        n
                    );
                }
                let mut out = vec![0.0f32; n];
                for (i, c) in raw.chunks_exact(4).enumerate() {
                    out[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                }
                Ok(out)
            }
            GgmlType::F16 => {
                let n = h.n_elems() as usize;
                if raw.len() != n * 2 {
                    bail!(
                        "tensor {:?}: F16 byte size {} != 2×{}",
                        h.name,
                        raw.len(),
                        n
                    );
                }
                let mut out = vec![0.0f32; n];
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    out[i] = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
                }
                Ok(out)
            }
            // BF16 is F32's top 16 bits, so the widen is a shift — no table, no
            // rounding. The MXFP4 DS4-Flash release stores its MoE router
            // (`ffn_gate_inp`) as BF16 where the Q2 release used F16; without
            // this arm the router is the first tensor the loader cannot read.
            GgmlType::BF16 => {
                let n = h.n_elems() as usize;
                if raw.len() != n * 2 {
                    bail!(
                        "tensor {:?}: BF16 byte size {} != 2×{}",
                        h.name,
                        raw.len(),
                        n
                    );
                }
                let mut out = vec![0.0f32; n];
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    out[i] = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                }
                Ok(out)
            }
            GgmlType::Q8_0 => {
                // 34 B per block: half d (off 0..2), int8 qs[32] (off 2..34); 32 weights per block.
                let n = h.n_elems() as usize;
                if n % 32 != 0 {
                    bail!("tensor {:?}: Q8_0 n_elems {} not a multiple of 32", h.name, n);
                }
                let n_blocks = n / 32;
                if raw.len() != n_blocks * 34 {
                    bail!(
                        "tensor {:?}: Q8_0 byte size {} != 34×{} blocks",
                        h.name,
                        raw.len(),
                        n_blocks
                    );
                }
                let mut out = Vec::with_capacity(n);
                for ib in 0..n_blocks {
                    let blk = &raw[ib * 34..(ib + 1) * 34];
                    let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    for k in 0..32 {
                        let q = blk[2 + k] as i8;
                        out.push(d * q as f32);
                    }
                }
                Ok(out)
            }
            other => bail!(
                "tensor {:?}: dequant_f32_simple does not handle {:?}; route through the accelerator host crate::cpu_via_dequant",
                h.name,
                other,
            ),
        }
    }
}

/// `blk.N.…` → Some(N).
fn parse_blk_idx(name: &str) -> Option<u32> {
    let s = name.strip_prefix("blk.")?;
    let dot = s.find('.')?;
    s[..dot].parse::<u32>().ok()
}

/// IEEE 754 binary16 → binary32. Mirrors `the accelerator host crate::quantized_experts::f16_to_f32`.
/// Public so per-row embedding dequant (decode_runner) is byte-identical to
/// `dequant_f32_simple`'s F16 path.
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let f = if exp == 0 {
        if mant == 0 {
            0.0_f32
        } else {
            (mant as f32) * (2.0_f32).powi(-24)
        }
    } else if exp == 0x1f {
        if mant == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let e = exp as i32 - 15;
        let m = 1.0_f32 + (mant as f32) / 1024.0;
        m * (2.0_f32).powi(e)
    };
    if sign == 1 {
        -f
    } else {
        f
    }
}

/// IEEE 754 binary32 → binary16 bit pattern (round-to-nearest-even). Port of
/// `the accelerator host crate::f16_cast::f32_to_f16_bits`, kept here so `engine_core` can encode
/// the Q4_0 block scale without a `half` dep.
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0xff {
        let m16 = if mant != 0 {
            0x200 | (mant >> 13) as u16
        } else {
            0
        };
        return sign | 0x7c00 | m16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let mant = mant | 0x800000;
        let shift = 14 - new_exp;
        let m16 = (mant >> shift) as u16;
        let round_bit = (mant >> (shift - 1)) & 1;
        return sign | (m16 + round_bit as u16);
    }
    let m16 = (mant >> 13) as u16;
    let round_bit = (mant >> 12) & 1;
    let sticky = (mant & 0xfff) != 0;
    let mut packed = sign | ((new_exp as u16) << 10) | m16;
    if round_bit == 1 && (sticky || (m16 & 1) == 1) {
        packed += 1;
    }
    packed
}

/// Re-quantize GGUF `block_q8_0` bytes (34 B/32w: f16 d + 32×int8) to
/// `block_q4_0` bytes (18 B/32w: f16 d + 16 nibble bytes), halving the resident
/// weight bytes for the low-RAM decode path. Standard ggml q4_0 layout: low
/// nibbles are weights [0,16), high nibbles weights [16,32); the matching
/// dequant is `((nibble & 0xF) - 8) * d` — exactly what
/// `ds4_kernel_mul_mv_q4_0_f32` (bridge_shims/dsv4_mul_mv_q4_0_f32.metal) reads.
pub fn requant_q8_0_to_q4_0(q8_bytes: &[u8]) -> Result<Vec<u8>> {
    if q8_bytes.len() % 34 != 0 {
        bail!(
            "requant_q8_0_to_q4_0: byte len {} not a multiple of 34",
            q8_bytes.len()
        );
    }
    let n_blocks = q8_bytes.len() / 34;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for ib in 0..n_blocks {
        let blk = &q8_bytes[ib * 34..(ib + 1) * 34];
        let d8 = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut x = [0.0f32; 32];
        for (k, xk) in x.iter_mut().enumerate() {
            *xk = (blk[2 + k] as i8) as f32 * d8;
        }
        // ggml q4_0 quantize: scale off the max-|value|, encode to [0,15] via -8 bias.
        let mut amax = 0.0f32;
        let mut vmax = 0.0f32;
        for &v in &x {
            if v.abs() > amax {
                amax = v.abs();
                vmax = v;
            }
        }
        let d4 = vmax / -8.0;
        let id = if d4 != 0.0 { 1.0 / d4 } else { 0.0 };
        let mut qi = [0u8; 32];
        for (k, &xk) in x.iter().enumerate() {
            // (int8)(x*id + 8.5) truncates toward zero (operand ≥ 0); clamp to [0,15].
            let q = (xk * id + 8.5) as i32;
            qi[k] = q.clamp(0, 15) as u8;
        }
        let d4_bits = f32_to_f16_bits(d4);
        out.push((d4_bits & 0xff) as u8);
        out.push((d4_bits >> 8) as u8);
        for j in 0..16 {
            out.push(qi[j] | (qi[j + 16] << 4));
        }
    }
    Ok(out)
}

/// CPU dequant of one `block_q4_0` (18 B) → 32 f32 weights. Mirrors the GPU
/// kernel's dequant exactly; used by the q4_0 matvec parity test.
pub fn dequant_q4_0_block(blk: &[u8]) -> [f32; 32] {
    debug_assert_eq!(blk.len(), 18);
    let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let mut out = [0.0f32; 32];
    for j in 0..16 {
        let b = blk[2 + j];
        out[j] = ((b & 0x0F) as i32 - 8) as f32 * d;
        out[j + 16] = ((b >> 4) as i32 - 8) as f32 * d;
    }
    out
}

/// QUALITY PROBE (DS4_ATTN_Q4_PROBE): re-quantize `block_q8_0` bytes to q4_0 precision
/// and pack the result back into `block_q8_0` bytes — i.e. q4-precision VALUES in a q8
/// CONTAINER. Lets the existing q8_0 prefill/attention kernels run at q4 precision with
/// NO new kernel, so we can needle-gate whether a q8→q4 attention cut is coherent BEFORE
/// committing to the K-batched q4 kernel build. Returns input unchanged on bad length.
pub fn q4_precision_in_q8_container(q8_bytes: &[u8]) -> Result<Vec<u8>> {
    let q4 = requant_q8_0_to_q4_0(q8_bytes)?; // 34B/blk → 18B/blk, q4_0 precision
    let n_blocks = q4.len() / 18;
    let mut out = Vec::with_capacity(n_blocks * 34);
    for ib in 0..n_blocks {
        let f = dequant_q4_0_block(&q4[ib * 18..(ib + 1) * 18]); // q4 values → f32
                                                                 // re-pack as block_q8_0 (f16 d = amax/127 + 32×i8) — q8 container, q4 values
        let amax = f.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let dbits = f32_to_f16_bits(d);
        out.push((dbits & 0xff) as u8);
        out.push((dbits >> 8) as u8);
        for &v in &f {
            let q = (v * id).round_ties_even() as i32;
            out.push(q.clamp(-128, 127) as i8 as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal GGUF v3 with one F16 token_embd and one F32
    /// `blk.0.attn_norm` so LayerViews has both a global and a per-layer
    /// handle to expose.
    fn write_two_tensor_gguf(path: &Path) {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // magic
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&3u64.to_le_bytes()); // n_meta_kv

        let write_str = |buf: &mut Vec<u8>, s: &str| {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        };

        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // String
        write_str(&mut buf, "deepseek4");
        write_str(&mut buf, "deepseek4.block_count");
        buf.extend_from_slice(&4u32.to_le_bytes()); // U32
        buf.extend_from_slice(&1u32.to_le_bytes());
        write_str(&mut buf, "deepseek4.embedding_length");
        buf.extend_from_slice(&4u32.to_le_bytes()); // U32
        buf.extend_from_slice(&4u32.to_le_bytes());

        // token_embd.weight F16 [4]
        write_str(&mut buf, "token_embd.weight");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // F16
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        // blk.0.attn_norm.weight F32 [4]
        write_str(&mut buf, "blk.0.attn_norm.weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        buf.extend_from_slice(&8u64.to_le_bytes()); // offset (after 4×f16=8 bytes)

        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        // token_embd: f16 bits [1.0, 2.0, 3.0, 4.0]
        for v in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        // attn_norm: f32 [0.5, 1.0, 1.5, 2.0]
        for v in [0.5f32, 1.0, 1.5, 2.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn parse_blk_idx_picks_index() {
        assert_eq!(parse_blk_idx("blk.0.attn_norm.weight"), Some(0));
        assert_eq!(parse_blk_idx("blk.42.ffn_gate_exps.weight"), Some(42));
        assert_eq!(parse_blk_idx("token_embd.weight"), None);
        assert_eq!(parse_blk_idx("output.weight"), None);
    }

    #[test]
    fn f16_bits_known_values() {
        assert!((f16_bits_to_f32(0x3c00) - 1.0).abs() < 1e-7);
        assert!((f16_bits_to_f32(0x4000) - 2.0).abs() < 1e-7);
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert!((f16_bits_to_f32(0xbc00) - -1.0).abs() < 1e-7);
    }

    #[test]
    fn from_gguf_splits_global_and_per_layer() {
        let tmp = std::env::temp_dir().join("ds4_layer_view_two_tensor.gguf");
        write_two_tensor_gguf(&tmp);
        let views = LayerViews::open(&tmp, 1).expect("open layer views");
        // Global side: embed handle exists, role classified.
        let embed = views.global.get("embed").expect("embed handle");
        assert_eq!(embed.role, "embed");
        assert_eq!(embed.ttype, GgmlType::F16);
        assert_eq!(embed.dims, vec![4]);
        // Layer side: attn_norm under blk.0.
        let l0 = views.layer(0);
        assert_eq!(l0.layer_idx, 0);
        let an = l0.require("attn_norm").expect("attn_norm handle");
        assert_eq!(an.ttype, GgmlType::F32);
        assert_eq!(an.dims, vec![4]);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn dequant_f32_simple_round_trips_f16_and_f32() {
        let tmp = std::env::temp_dir().join("ds4_layer_view_dequant.gguf");
        write_two_tensor_gguf(&tmp);
        let views = LayerViews::open(&tmp, 1).expect("open");

        let embed = views.global.get("embed").unwrap().clone();
        let v = views.dequant_f32_simple(&embed).unwrap();
        assert_eq!(v.len(), 4);
        assert!((v[0] - 1.0).abs() < 1e-3);
        assert!((v[1] - 2.0).abs() < 1e-3);
        assert!((v[2] - 3.0).abs() < 1e-3);
        assert!((v[3] - 4.0).abs() < 1e-3);

        let attn = views.layer(0).require("attn_norm").unwrap().clone();
        let v = views.dequant_f32_simple(&attn).unwrap();
        assert_eq!(v, vec![0.5, 1.0, 1.5, 2.0]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn dequant_f32_simple_decodes_q8_0_known_values() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        for q in -16i8..16 {
            bytes.push(q as u8);
        }
        let h = TensorHandle {
            name: "blk.0.ffn_down_exps.weight".into(),
            role: "moe_down_experts",
            ttype: GgmlType::Q8_0,
            dims: vec![32],
            abs_offset: 0,
            byte_size: 34,
        };
        let views = LayerViews {
            data_offset: 0,
            bytes: ByteBuf::Owned(bytes),
            per_layer: vec![],
            global: BTreeMap::new(),
        };

        let v = views.dequant_f32_simple(&h).unwrap();
        assert_eq!(v.len(), 32);
        assert_eq!(v[0], -16.0);
        assert_eq!(v[15], -1.0);
        assert_eq!(v[16], 0.0);
        assert_eq!(v[31], 15.0);
    }

    #[test]
    fn dequant_f32_simple_rejects_malformed_q8_0_shapes() {
        let h = TensorHandle {
            name: "blk.0.ffn_gate_exps.weight".into(),
            role: "moe_gate_experts",
            ttype: GgmlType::Q8_0,
            dims: vec![33],
            abs_offset: 0,
            byte_size: 68,
        };
        let views = LayerViews {
            data_offset: 0,
            bytes: ByteBuf::Owned(vec![0u8; 68]),
            per_layer: vec![],
            global: BTreeMap::new(),
        };
        let err = views.dequant_f32_simple(&h).unwrap_err().to_string();
        assert!(err.contains("not a multiple of 32"), "err = {err}");

        let h = TensorHandle {
            dims: vec![32],
            byte_size: 33,
            ..h
        };
        let views = LayerViews {
            data_offset: 0,
            bytes: ByteBuf::Owned(vec![0u8; 33]),
            per_layer: vec![],
            global: BTreeMap::new(),
        };
        let err = views.dequant_f32_simple(&h).unwrap_err().to_string();
        assert!(err.contains("byte size"), "err = {err}");
    }

    #[test]
    fn dequant_f32_simple_rejects_quantized_types() {
        // Build a fake handle with a quant type; we don't need real bytes
        // since the type check fires first.
        let h = TensorHandle {
            name: "blk.0.ffn_gate_exps.weight".into(),
            role: "moe_gate_experts",
            ttype: GgmlType::IQ2_XXS,
            dims: vec![1, 256],
            abs_offset: 0,
            byte_size: 66,
        };
        let views = LayerViews {
            data_offset: 0,
            bytes: ByteBuf::Owned(vec![0u8; 66]),
            per_layer: vec![],
            global: BTreeMap::new(),
        };
        let err = views.dequant_f32_simple(&h).unwrap_err().to_string();
        assert!(err.contains("IQ2_XXS"), "err = {err}");
        assert!(err.contains("cpu_via_dequant"), "err = {err}");
    }

    #[test]
    fn require_returns_clear_error_for_missing_role() {
        let lv = LayerView {
            layer_idx: 7,
            handles: BTreeMap::new(),
        };
        let err = lv.require("mla_q_a").unwrap_err().to_string();
        assert!(err.contains("layer 7"));
        assert!(err.contains("mla_q_a"));
    }

    #[test]
    fn is_compressed_uses_indexer_flag_off_by_default() {
        let lv = LayerView::default();
        assert!(!lv.is_compressed());
        assert!(!lv.uses_indexer());
    }

    #[test]
    fn is_compressed_detects_compressor_kv() {
        let mut lv = LayerView::default();
        lv.handles.insert(
            "attn_compressor_kv",
            TensorHandle {
                name: "blk.0.attn_compressor_kv.weight".into(),
                role: "attn_compressor_kv",
                ttype: GgmlType::F16,
                dims: vec![1],
                abs_offset: 0,
                byte_size: 2,
            },
        );
        assert!(lv.is_compressed());
    }

    // ---- requant_q8_0_to_q4_0 round-trip ----------------------------------

    /// Build `block_q8_0` bytes (34 B/32w: f16 scale `d = amax/127` + 32 i8
    /// quants) from an f32 slice whose length is a multiple of 32. Inverse of
    /// the per-block decode in `requant_q8_0_to_q4_0`; lets the test feed real
    /// values through the q8_0 → q4_0 path without depending on `the accelerator host crate`.
    fn quantize_q8_0(w: &[f32]) -> Vec<u8> {
        assert_eq!(w.len() % 32, 0);
        let mut out = Vec::with_capacity(w.len() / 32 * 34);
        for blk in w.chunks_exact(32) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let dbits = f32_to_f16_bits(d);
            out.push((dbits & 0xff) as u8);
            out.push((dbits >> 8) as u8);
            for &v in blk {
                let q = (v * id).round() as i32;
                out.push(q.clamp(-128, 127) as i8 as u8);
            }
        }
        out
    }

    #[test]
    fn requant_q8_0_to_q4_0_rejects_misaligned_len() {
        // 34 is one valid q8_0 block; 35 is not a multiple of 34.
        let err = requant_q8_0_to_q4_0(&vec![0u8; 35])
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple of 34"), "err = {err}");
        // empty input is a clean zero-block result, not an error.
        assert!(requant_q8_0_to_q4_0(&[]).unwrap().is_empty());
    }

    #[test]
    fn requant_q8_0_to_q4_0_block_size_is_18_bytes() {
        // 32-weight input → one q8_0 block (34 B) → one q4_0 block (18 B).
        let q8 = quantize_q8_0(&[1.0f32; 32]);
        assert_eq!(q8.len(), 34);
        let q4 = requant_q8_0_to_q4_0(&q8).unwrap();
        assert_eq!(q4.len(), 18);
    }

    #[test]
    fn requant_q8_0_to_q4_0_round_trip_within_q4_granularity() {
        // A spread of values across two blocks. After q8_0 → q4_0 → dequant,
        // each weight must land within one q4_0 step (|d4| ≈ amax/8) of the
        // q8_0-grid value it started from.
        let mut w = Vec::with_capacity(64);
        for k in 0..64 {
            // deterministic ramp in [-2.0, ~2.0), nonzero, both signs.
            w.push(((k as f32) - 32.0) * 0.0625);
        }
        let q8 = quantize_q8_0(&w);
        let q4 = requant_q8_0_to_q4_0(&q8).unwrap();
        assert_eq!(q4.len(), 2 * 18);

        for (b, blk) in q4.chunks_exact(18).enumerate() {
            let deq = dequant_q4_0_block(blk);
            // The q4_0 step for this block = |scale|; recover it from the block.
            let d4 = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]])).abs();
            for j in 0..32 {
                let orig = w[b * 32 + j];
                let err = (deq[j] - orig).abs();
                // one q4 step of slack + a small q8 pre-quant epsilon.
                let tol = d4 + orig.abs() / 127.0 + 1e-6;
                assert!(
                    err <= tol,
                    "block {b} weight {j}: orig={orig} deq={} err={err} tol={tol}",
                    deq[j]
                );
            }
        }
    }

    #[test]
    fn requant_q8_0_to_q4_0_recovers_exact_q4_grid_values() {
        // Values already on a q4_0 grid (step 0.25, codes -8..7 → -2.0..1.75)
        // must round-trip near-exactly: q8_0 (step amax/127 ≈ 0.0157) then q4_0
        // both represent them, so the error is bounded by the q8 pre-step only.
        let step = 0.25f32;
        let w: Vec<f32> = (0..32)
            .map(|k| ((k % 16) as i32 - 8) as f32 * step)
            .collect();
        let q8 = quantize_q8_0(&w);
        let q4 = requant_q8_0_to_q4_0(&q8).unwrap();
        let deq = dequant_q4_0_block(&q4);
        for j in 0..32 {
            assert!(
                (deq[j] - w[j]).abs() <= w[j].abs() / 127.0 + 1e-4,
                "weight {j}: orig={} deq={}",
                w[j],
                deq[j]
            );
        }
    }
}
