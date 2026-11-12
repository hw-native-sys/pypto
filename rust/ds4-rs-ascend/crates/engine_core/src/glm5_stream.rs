// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! GLM-5.2 (`glm-dsa`) **streaming** weight source + forward — Phase 1b.
//!
//! Phase 0/1a proved the architecture (`glm5_ref`, cos=1.0 vs mlx-lm) and the
//! real-GGUF *config* port (`DefaultsGlm5::from_gguf`). This module runs the
//! actual ~226 GB `unsloth/GLM-5.2-GGUF` (UD-IQ1_S, 6 shards) by **streaming
//! weights from SSD** rather than materializing everything in RAM:
//!
//!   * Every shard is `mmap`-ed `MAP_SHARED` (clean, file-backed, reclaimable —
//!     the OS page cache *is* the SSD stream; same mechanism as the
//!     `DS4_SSD_STREAM` residency used for the 433 GB PRO model, here on the CPU
//!     side instead of GPU-pinned).
//!   * Tensors are dequantized to f32 **on demand**, one at a time, and dropped
//!     after use. For MoE we dequantize only the **top-k selected experts**
//!     (the union over the prefill tokens), never the full 256-expert tensor.
//!
//! The forward is the same math as `glm5_ref` (validated byte/cos-exact on the
//! tiny config) but pulls weights from a [`ShardedGguf`] instead of an eager
//! HashMap. Gated behind GLM-5.2; does NOT touch DS4-V4-Flash.
//!
//! GGUF tensor names → mlx roles: the GGUF stores `blk.N.attn_q_a.weight`,
//! `attn_kv_a_mqa`, split `attn_k_b`/`attn_v_b`, `ffn_*_exps`, indexer
//! `indexer.attn_q_b`/`attn_k`/`k_norm`/`proj`, etc. (see `gguf::classify_tensor`).
//! [`gguf_name`] maps each logical weight this forward needs to its GGUF name.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

use crate::gguf::{GgmlType, GgufFile};
use crate::glm5_ref::{
    group_expert_select, layer_norm, linear, rms_norm, rope_apply, swiglu_mlp, DefaultsGlm5,
    LayerType,
};
use crate::layer_view::{f16_bits_to_f32, MmapSource};

include!("glm5_iq_grids.rs");

// ============================ multi-shard mmap =============================

struct Shard {
    gguf: GgufFile,
    mmap: MmapSource,
}

/// A located tensor: which shard, and the metadata to slice + dequant it.
#[derive(Clone)]
pub struct StreamHandle {
    pub shard: usize,
    pub ttype: GgmlType,
    pub dims: Vec<u64>,
    pub abs_offset: u64,
    pub byte_size: u64,
}

impl StreamHandle {
    pub fn n_elems(&self) -> u64 {
        self.dims.iter().product()
    }
}

// ── DS4_GLM_PROF: per-decode-step weight-cost profiler ───────────────────────
// Splits each dequant call into READ (mmap page-in / memory touch) vs DEQUANT
// (IQ1_S→f32 CPU). decode_step_metal resets at the top + reports at the bottom;
// the remainder (step_total − read − dequant) is GPU + host glue.
use std::cell::Cell;
thread_local! {
    static PROF_READ_NS: Cell<u128> = const { Cell::new(0) };
    static PROF_DEQ_NS: Cell<u128> = const { Cell::new(0) };
    static PROF_BYTES: Cell<u64> = const { Cell::new(0) };
    static PROF_CALLS: Cell<u64> = const { Cell::new(0) };
    static PROF_GPU_NS: Cell<u128> = const { Cell::new(0) };
    static PROF_GPU_CALLS: Cell<u64> = const { Cell::new(0) };
}
/// Accumulate one GPU commit+wait (ns) — called from `commit_wait_traced` so the
/// profiler can split GPU dispatch (latency × count) out of "GPU/glue".
pub fn glm_prof_add_gpu(ns: u128) {
    if glm_prof_on() {
        PROF_GPU_NS.with(|c| c.set(c.get() + ns));
        PROF_GPU_CALLS.with(|c| c.set(c.get() + 1));
    }
}
/// (gpu_ms, gpu_calls) accumulated since reset.
pub fn glm_prof_gpu() -> (f64, u64) {
    (
        PROF_GPU_NS.with(|c| c.get()) as f64 / 1e6,
        PROF_GPU_CALLS.with(|c| c.get()),
    )
}

thread_local! {
    // op-label → (count, total_ns) for the per-op GPU breakdown (DS4_GLM_PROF).
    static PROF_GPU_LABELS: std::cell::RefCell<std::collections::HashMap<String, (u64, u128)>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}
/// Labeled GPU commit+wait — same as `glm_prof_add_gpu` plus a per-`label` tally so
/// the profiler can rank WHICH ops own the GPU-dispatch time.
pub fn glm_prof_add_gpu_labeled(label: &str, ns: u128) {
    if glm_prof_on() {
        PROF_GPU_NS.with(|c| c.set(c.get() + ns));
        PROF_GPU_CALLS.with(|c| c.set(c.get() + 1));
        PROF_GPU_LABELS.with(|m| {
            let mut m = m.borrow_mut();
            let e = m.entry(label.to_string()).or_insert((0, 0));
            e.0 += 1;
            e.1 += ns;
        });
    }
}
/// Per-op GPU breakdown `(label, count, ms)` sorted by descending total time.
pub fn glm_prof_gpu_breakdown() -> Vec<(String, u64, f64)> {
    PROF_GPU_LABELS.with(|m| {
        let mut v: Vec<(String, u64, f64)> = m
            .borrow()
            .iter()
            .map(|(k, (c, ns))| (k.clone(), *c, *ns as f64 / 1e6))
            .collect();
        v.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        v
    })
}

thread_local! {
    static PROF_ALLOC_NS: Cell<u128> = const { Cell::new(0) };
    static PROF_UPLOAD_NS: Cell<u128> = const { Cell::new(0) };
    static PROF_READBACK_NS: Cell<u128> = const { Cell::new(0) };
}
/// Split host-glue: time spent in output-buffer ALLOC vs input UPLOAD vs READBACK.
pub fn glm_prof_add_alloc(ns: u128) {
    if glm_prof_on() {
        PROF_ALLOC_NS.with(|c| c.set(c.get() + ns));
    }
}
pub fn glm_prof_add_upload(ns: u128) {
    if glm_prof_on() {
        PROF_UPLOAD_NS.with(|c| c.set(c.get() + ns));
    }
}
pub fn glm_prof_add_readback(ns: u128) {
    if glm_prof_on() {
        PROF_READBACK_NS.with(|c| c.set(c.get() + ns));
    }
}
/// (alloc_ms, upload_ms, readback_ms) since reset.
pub fn glm_prof_bufmgmt() -> (f64, f64, f64) {
    (
        PROF_ALLOC_NS.with(|c| c.get()) as f64 / 1e6,
        PROF_UPLOAD_NS.with(|c| c.get()) as f64 / 1e6,
        PROF_READBACK_NS.with(|c| c.get()) as f64 / 1e6,
    )
}
pub fn glm_prof_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("DS4_GLM_PROF").is_ok())
}
pub fn glm_prof_reset() {
    PROF_READ_NS.with(|c| c.set(0));
    PROF_DEQ_NS.with(|c| c.set(0));
    PROF_BYTES.with(|c| c.set(0));
    PROF_CALLS.with(|c| c.set(0));
    PROF_GPU_NS.with(|c| c.set(0));
    PROF_GPU_CALLS.with(|c| c.set(0));
    PROF_GPU_LABELS.with(|m| m.borrow_mut().clear());
    PROF_ALLOC_NS.with(|c| c.set(0));
    PROF_UPLOAD_NS.with(|c| c.set(0));
    PROF_READBACK_NS.with(|c| c.set(0));
}
/// The IQ1_S grid table (2048 × u64, each = 8 packed int8 grid values), for the
/// GPU inline-dequant matvec kernel to upload as a constant buffer. Verbatim ggml.
pub fn iq1s_grid() -> &'static [u64; 2048] {
    iq1s_grid_tbl()
}

/// IQ2_XXS grid (256 × u64, 8 packed int8 lattice points each) for the GPU
/// inline-dequant matvec kernel. Verbatim ggml.
pub fn iq2xxs_grid() -> &'static [u64; 256] {
    iq2xxs_grid_tbl()
}

/// IQ3_XXS grid (256 × u32, 4 packed int8 lattice points each) for the GPU
/// inline-dequant matvec kernel. Verbatim ggml.
pub fn iq3xxs_grid() -> &'static [u32; 256] {
    iq3xxs_grid_tbl()
}

/// ggml ksigns table (128 × u8) — sign-bit lookup shared by IQ2_XXS/IQ3_XXS.
pub fn iq_ksigns() -> &'static [u8; 128] {
    &KSIGNS_IQ2XS_RAW
}

/// Returns (read_ms, dequant_ms, bytes_read, n_calls) accumulated since reset.
pub fn glm_prof_report() -> (f64, f64, u64, u64) {
    (
        PROF_READ_NS.with(|c| c.get()) as f64 / 1e6,
        PROF_DEQ_NS.with(|c| c.get()) as f64 / 1e6,
        PROF_BYTES.with(|c| c.get()),
        PROF_CALLS.with(|c| c.get()),
    )
}

/// All shards of a split GGUF, plus a unified tensor lookup keyed by GGUF name.
pub struct ShardedGguf {
    shards: Vec<Shard>,
    by_name: HashMap<String, StreamHandle>,
}

impl ShardedGguf {
    /// Open a split GGUF given any shard path (`…-00001-of-000NN.gguf` is
    /// canonical). Discovers sibling shards from the filename's `-of-` count.
    pub fn open(first_shard: &Path) -> Result<Self> {
        Self::open_paths(&discover_shards(first_shard)?)
    }

    pub fn open_paths(paths: &[PathBuf]) -> Result<Self> {
        if paths.is_empty() {
            bail!("ShardedGguf: no shard paths");
        }
        let mut shards = Vec::with_capacity(paths.len());
        let mut by_name: HashMap<String, StreamHandle> = HashMap::new();
        for (si, p) in paths.iter().enumerate() {
            let gguf = GgufFile::open(p).with_context(|| format!("open shard {}", p.display()))?;
            let mmap =
                MmapSource::open(p).with_context(|| format!("mmap shard {}", p.display()))?;
            for t in &gguf.tensors {
                by_name.insert(
                    t.name.clone(),
                    StreamHandle {
                        shard: si,
                        ttype: t.ttype,
                        dims: t.dims.clone(),
                        abs_offset: gguf.tensor_data_offset + t.offset,
                        byte_size: gguf.tensor_byte_size(t),
                    },
                );
            }
            shards.push(Shard { gguf, mmap });
        }
        Ok(ShardedGguf { shards, by_name })
    }

    pub fn meta_gguf(&self) -> &GgufFile {
        &self.shards[0].gguf
    }
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }
    /// Release the pages of every tensor whose name starts with `prefix`, back to the OS.
    ///
    /// WHY: the shards are mmapped `MAP_SHARED` and the OS page cache is the stream. That is free while the working set fits
    /// in RAM; Kimi K3's ten shards are ~200 GB, so past a few blocks the cache evicts what the NEXT block needs and every
    /// block re-reads from the SSD. Measured growth: 100s per block at block 1, 172s at 3, 327s at 4.
    ///
    /// `MADV_DONTNEED` on file-backed shared pages drops them without losing data -- the file is the backing store, so a
    /// later read faults them back. Called between blocks with `blk.N.` it keeps only the block being computed resident.
    ///
    /// Returns the bytes advised, so a caller can report what it released rather than assume it worked.
    pub fn advice_release_prefix(&self, prefix: &str) -> u64 {
        #[cfg(unix)]
        {
            const PAGE: usize = 4096;
            let mut released = 0u64;
            for (name, h) in &self.by_name {
                if !name.starts_with(prefix) {
                    continue;
                }
                let shard = &self.shards[h.shard];
                let base = shard.mmap.as_slice().as_ptr() as usize;
                let start = base + h.abs_offset as usize;
                // madvise needs a page-aligned start; the LENGTH may end anywhere.
                let aligned = start & !(PAGE - 1);
                let len = (start + h.byte_size as usize).saturating_sub(aligned);
                // SAFETY: the range lies inside a mapping this struct owns and keeps alive for its lifetime.
                let rc = unsafe {
                    libc::madvise(aligned as *mut libc::c_void, len, libc::MADV_DONTNEED)
                };
                if rc == 0 {
                    released += h.byte_size;
                }
            }
            released
        }
        #[cfg(not(unix))]
        {
            let _ = prefix;
            0
        }
    }

    pub fn handle(&self, name: &str) -> Option<&StreamHandle> {
        self.by_name.get(name)
    }
    pub fn n_tensors(&self) -> usize {
        self.by_name.len()
    }

    /// All tensor names whose name contains `substr` (debug: MTP/nextn discovery).
    pub fn names_containing(&self, substr: &str) -> Vec<String> {
        let mut v: Vec<String> = self
            .by_name
            .keys()
            .filter(|k| k.contains(substr))
            .cloned()
            .collect();
        v.sort();
        v
    }

    /// Correct config fields that `DefaultsGlm5::from_gguf` can only guess from
    /// shard-0 metadata (which carries NO tensor-info in a gguf-split set).
    /// In particular `qk_nope_head_dim` = `attn_k_b` dim[0] (=192 for the real
    /// model; the meta-only fallback wrongly returns key_length_mla=256).
    pub fn fixup_config(&self, cfg: &mut DefaultsGlm5) {
        // find any layer's attn_k_b to read qk_nope from its fastest dim.
        for l in 0..cfg.num_hidden_layers {
            let n = gguf_name(l, "k_b");
            if let Some(h) = self.handle(&n) {
                if let Some(&d0) = h.dims.first() {
                    cfg.qk_nope_head_dim = d0 as usize;
                    return;
                }
            }
        }
    }

    /// Dims of a tensor by GGUF name (for layout-sensitive code).
    pub fn dims_of(&self, name: &str) -> Option<&[u64]> {
        self.handle(name).map(|h| h.dims.as_slice())
    }

    fn bytes_for(&self, h: &StreamHandle) -> Result<&[u8]> {
        let slice = self.shards[h.shard].mmap.as_slice();
        let start = h.abs_offset as usize;
        let end = start
            .checked_add(h.byte_size as usize)
            .ok_or_else(|| anyhow!("byte_size overflow"))?;
        if end > slice.len() {
            bail!(
                "tensor byte range {}..{} out of shard {} size {}",
                start,
                end,
                h.shard,
                slice.len()
            );
        }
        Ok(&slice[start..end])
    }

    /// Dequant + (when DS4_GLM_PROF) split the cost into READ (force the mmap
    /// page-in via a per-page touch) vs DEQUANT (warm IQ1_S→f32 CPU).
    fn dequant_timed(&self, ttype: GgmlType, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
        if !glm_prof_on() {
            return dequant_blocks(ttype, bytes, n_elems);
        }
        let t0 = std::time::Instant::now();
        let mut acc = 0u8;
        let mut i = 0;
        while i < bytes.len() {
            acc ^= bytes[i];
            i += 4096;
        }
        std::hint::black_box(acc);
        let read = t0.elapsed().as_nanos();
        let t1 = std::time::Instant::now();
        let out = dequant_blocks(ttype, bytes, n_elems);
        let deq = t1.elapsed().as_nanos();
        PROF_READ_NS.with(|c| c.set(c.get() + read));
        PROF_DEQ_NS.with(|c| c.set(c.get() + deq));
        PROF_BYTES.with(|c| c.set(c.get() + bytes.len() as u64));
        PROF_CALLS.with(|c| c.set(c.get() + 1));
        out
    }

    /// Dequantize a whole tensor by GGUF name to f32 (on demand; drop promptly).
    pub fn dequant_name(&self, name: &str) -> Result<Vec<f32>> {
        let h = self
            .handle(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        self.dequant_timed(h.ttype, self.bytes_for(h)?, h.n_elems() as usize)
            .with_context(|| format!("dequant {name} ({:?})", h.ttype))
    }

    /// Dequantize a single expert `e` of a 3-D expert tensor `[n_expert, …]`.
    pub fn dequant_expert(&self, name: &str, e: usize, n_expert: usize) -> Result<Vec<f32>> {
        let h = self
            .handle(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let total = h.n_elems() as usize;
        if total % n_expert != 0 {
            bail!("{name}: n_elems {total} not divisible by n_expert {n_expert}");
        }
        let per_expert = total / n_expert;
        let bs = h.ttype.block_size();
        let ts = h.ttype.type_size();
        if per_expert % bs != 0 {
            bail!("{name}: per-expert {per_expert} not a multiple of block {bs}");
        }
        let bytes_per_expert = (per_expert / bs) * ts;
        let bytes = self.bytes_for(h)?;
        let start = e * bytes_per_expert;
        let end = start + bytes_per_expert;
        if end > bytes.len() {
            bail!("{name}: expert {e} byte range out of bounds");
        }
        self.dequant_timed(h.ttype, &bytes[start..end], per_expert)
            .with_context(|| format!("dequant expert {e} of {name} ({:?})", h.ttype))
    }

    /// Raw (un-dequantized) mmap bytes + quant type + element count for a tensor —
    /// the no-copy feed for GPU inline-dequant matvec kernels (e.g. IQ1_S). The
    /// returned slice borrows the model mmap directly (no f32 materialization).
    pub fn raw_handle(&self, name: &str) -> Result<(&[u8], GgmlType, usize)> {
        let h = self
            .handle(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        Ok((self.bytes_for(h)?, h.ttype, h.n_elems() as usize))
    }

    /// Raw bytes + quant type + per-expert element count for one expert `e` of a
    /// 3-D `[n_expert, …]` tensor (no dequant). Twin of `dequant_expert`.
    pub fn raw_expert(
        &self,
        name: &str,
        e: usize,
        n_expert: usize,
    ) -> Result<(&[u8], GgmlType, usize)> {
        let h = self
            .handle(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        let total = h.n_elems() as usize;
        anyhow::ensure!(
            total % n_expert == 0,
            "{name}: n_elems {total} not /{n_expert}"
        );
        let per_expert = total / n_expert;
        let bs = h.ttype.block_size();
        let ts = h.ttype.type_size();
        anyhow::ensure!(
            per_expert % bs == 0,
            "{name}: per-expert {per_expert} not /block {bs}"
        );
        let bytes_per_expert = (per_expert / bs) * ts;
        let bytes = self.bytes_for(h)?;
        let start = e * bytes_per_expert;
        let end = start + bytes_per_expert;
        anyhow::ensure!(end <= bytes.len(), "{name}: expert {e} byte range OOB");
        Ok((&bytes[start..end], h.ttype, per_expert))
    }

    /// The FULL shard mmap slice containing tensor `name`, plus the tensor's byte
    /// offset within it. Lets a caller wrap the WHOLE shard as ONE no-copy GPU buffer
    /// (registered once) and index tensors/experts by offset — so the GPU lazy-faults
    /// only the touched expert pages instead of forcing a per-tensor buffer resident.
    pub fn shard_and_offset(&self, name: &str) -> Result<(&[u8], usize)> {
        let h = self
            .handle(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        Ok((self.shards[h.shard].mmap.as_slice(), h.abs_offset as usize))
    }
}

/// `…-<idx>-of-<count>.gguf` → all `count` ordered shard paths. A non-split
/// name returns the single path.
fn discover_shards(first: &Path) -> Result<Vec<PathBuf>> {
    let name = first
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("bad shard path"))?;
    let Some(of_pos) = name.find("-of-") else {
        return Ok(vec![first.to_path_buf()]);
    };
    let after = &name[of_pos + 4..];
    let count_str: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
    let Ok(count) = count_str.parse::<usize>() else {
        return Ok(vec![first.to_path_buf()]);
    };
    let before = &name[..of_pos];
    let dash = before
        .rfind('-')
        .ok_or_else(|| anyhow!("shard name missing index field: {name}"))?;
    let prefix = &before[..dash];
    let width = count_str.len();
    let dir = first.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::with_capacity(count);
    for i in 1..=count {
        let p = dir.join(format!("{prefix}-{i:0width$}-of-{count_str}.gguf"));
        if !p.exists() {
            bail!("missing shard {} (split.count={count})", p.display());
        }
        out.push(p);
    }
    Ok(out)
}

// ============================ dequantization ===============================
// Reference ggml/antirez block layouts. Covers the Unsloth UD-IQ1_S "dynamic"
// quant mix (most tensors IQ1_S; embeddings/output/attention promoted to
// Q4_K/Q5_K/Q6_K/IQ2_XXS, biases F32). IQ1_S/IQ1_M/IQ2_XXS use the exact ggml
// grid tables (glm5_iq_grids.rs, extracted verbatim from ggml-common.h).

/// Dequantize to f32, parallelizing large block-quant tensors across all cores.
/// Each GGUF block (`block_size` elems in `type_size` bytes) is independent, so
/// we split block-aligned byte ranges over `std::thread::scope` and concatenate.
/// Small tensors / non-block layouts fall through to the serial path. This is the
/// decode-step lever-1 win: the per-token dequant (44% of the step, ~27s) ran on
/// ONE core — IQ1_S→f32 is CPU-heavy and embarrassingly parallel.
pub fn dequant_blocks(ttype: GgmlType, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    let bs = ttype.block_size();
    let ts = ttype.type_size();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    // Only parallelize when it pays: known block layout, exact byte count, big
    // enough to amortize ~N thread spawns (bs*1024 elems ≈ a non-trivial tensor).
    if bs == 0
        || ts == 0
        || threads <= 1
        || n_elems % bs != 0
        || bytes.len() != (n_elems / bs) * ts
        || n_elems < bs * 1024
    {
        return dequant_blocks_serial(ttype, bytes, n_elems);
    }
    let n_blocks = n_elems / bs;
    let chunk_blocks = n_blocks.div_ceil(threads);
    let parts: Result<Vec<Vec<f32>>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut b = 0usize;
        while b < n_blocks {
            let nb = chunk_blocks.min(n_blocks - b);
            let sub = &bytes[b * ts..(b + nb) * ts];
            let sub_n = nb * bs;
            handles.push(s.spawn(move || dequant_blocks_serial(ttype, sub, sub_n)));
            b += nb;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("dequant worker panicked"))
            .collect()
    });
    let parts = parts?;
    let mut out = Vec::with_capacity(n_elems);
    for p in &parts {
        out.extend_from_slice(p);
    }
    Ok(out)
}

/// MX FP4 (E2M1) — ggml type 39, 32 values per 17-byte block.
///
/// Faithful port of antirez's `ds4_vec_dot_mxfp4_f32` layout (ds4.c @1f862bd).
/// Two details are NOT guessable and are the whole reason this was read rather
/// than inferred:
///
/// 1. **Nibble order is split, not interleaved.** The low nibble of byte `j`
///    holds element `j`; the high nibble holds element `j + 16`. Reading them
///    as consecutive pairs decodes every block into noise.
/// 2. **e8m0 has a zero special case.** `e == 0` means 2^-127, encoded as bits
///    `0x0040_0000` — not 2^(0-127), which would be a denormal.
pub fn dequant_mxfp4(bytes: &[u8]) -> Vec<f32> {
    /// The 16 FP4 (E2M1) code points, sign-magnitude ordered.
    const MXFP4_VALUES: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    const QK: usize = 32;
    const BLK: usize = 17;
    let nb = bytes.len() / BLK;
    let mut out = vec![0f32; nb * QK];
    for ib in 0..nb {
        let blk = &bytes[ib * BLK..(ib + 1) * BLK];
        let d = e8m0_to_f32(blk[0]);
        let qs = &blk[1..BLK];
        let base = ib * QK;
        for (j, &q) in qs.iter().enumerate() {
            out[base + j] = d * MXFP4_VALUES[(q & 0x0f) as usize];
            out[base + j + QK / 2] = d * MXFP4_VALUES[(q >> 4) as usize];
        }
    }
    out
}

/// e8m0 shared micro-exponent → f32. `e == 0` is 2^-127, not a denormal.
#[inline]
pub fn e8m0_to_f32(e: u8) -> f32 {
    f32::from_bits(if e == 0 { 0x0040_0000 } else { (e as u32) << 23 })
}

fn dequant_blocks_serial(ttype: GgmlType, bytes: &[u8], n_elems: usize) -> Result<Vec<f32>> {
    let out = match ttype {
        GgmlType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        GgmlType::F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        GgmlType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        GgmlType::Q8_0 => dequant_q8_0(bytes),
        GgmlType::MXFP4 => dequant_mxfp4(bytes),
        GgmlType::Q4_0 => dequant_q4_0(bytes),
        GgmlType::Q4_K => dequant_q4_k(bytes),
        GgmlType::Q5_K => dequant_q5_k(bytes),
        GgmlType::Q6_K => dequant_q6_k(bytes),
        GgmlType::Q2_K => dequant_q2_k(bytes),
        GgmlType::Q3_K => dequant_q3_k(bytes),
        GgmlType::IQ2_XXS => dequant_iq2_xxs(bytes),
        GgmlType::IQ3_XXS => dequant_iq3_xxs(bytes),
        GgmlType::IQ4_XS => dequant_iq4_xs(bytes),
        GgmlType::IQ1_S => dequant_iq1_s(bytes),
        GgmlType::IQ1_M => dequant_iq1_m(bytes),
        other => bail!("dequant_blocks: unsupported type {:?}", other),
    };
    if out.len() != n_elems {
        bail!(
            "dequant_blocks {:?}: produced {} elems, expected {}",
            ttype,
            out.len(),
            n_elems
        );
    }
    Ok(out)
}

fn dequant_q8_0(bytes: &[u8]) -> Vec<f32> {
    let nb = bytes.len() / 34;
    let mut out = Vec::with_capacity(nb * 32);
    for ib in 0..nb {
        let blk = &bytes[ib * 34..ib * 34 + 34];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for k in 0..32 {
            out.push(d * (blk[2 + k] as i8) as f32);
        }
    }
    out
}

fn dequant_q4_0(bytes: &[u8]) -> Vec<f32> {
    let nb = bytes.len() / 18;
    let mut out = vec![0.0f32; nb * 32];
    for ib in 0..nb {
        let blk = &bytes[ib * 18..ib * 18 + 18];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..18];
        let base = ib * 32;
        for k in 0..16 {
            out[base + k] = d * ((qs[k] & 0x0F) as i32 - 8) as f32;
            out[base + 16 + k] = d * ((qs[k] >> 4) as i32 - 8) as f32;
        }
    }
    out
}

// ---- k-quants (super-block of 256) ----

fn dequant_q2_k(bytes: &[u8]) -> Vec<f32> {
    let nb = bytes.len() / 84;
    let mut out = Vec::with_capacity(nb * 256);
    for ib in 0..nb {
        let blk = &bytes[ib * 84..ib * 84 + 84];
        let scales = &blk[0..16];
        let qs = &blk[16..80];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
        for i_sb in 0..16u32 {
            let sc = scales[i_sb as usize];
            let dl = d * (sc & 0x0F) as f32;
            let ml = dmin * (sc >> 4) as f32;
            let q_off = ((i_sb / 8) * 32 + (i_sb & 1) * 16) as usize;
            let bf = ((i_sb / 2) & 3) as u8;
            let mask: u8 = 0x03u8 << (bf * 2);
            let shift = bf * 2;
            for k in 0..16 {
                let v = ((qs[q_off + k] & mask) >> shift) as f32;
                out.push(dl * v - ml);
            }
        }
    }
    out
}

/// `block_q3_K` (110 B, 256 w): hmask[32] qs[64] scales[12] d(half). Per ggml
/// dequantize_row_q3_K: 6-bit signed scales (centered 32), 2-bit qs + 1 high bit.
fn dequant_q3_k(bytes: &[u8]) -> Vec<f32> {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let nb = bytes.len() / 110;
    let mut out = vec![0.0f32; nb * 256];
    for ib in 0..nb {
        let blk = &bytes[ib * 110..ib * 110 + 110];
        let hmask = &blk[0..32];
        let qs = &blk[32..96];
        let scales_raw = &blk[96..108];
        let d_all = f16_bits_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        // unpack 16 6-bit scales, centered at 32 (ggml bit-twiddle).
        let mut aux = [0u32; 4];
        for i in 0..3 {
            aux[i] = u32::from_le_bytes([
                scales_raw[i * 4],
                scales_raw[i * 4 + 1],
                scales_raw[i * 4 + 2],
                scales_raw[i * 4 + 3],
            ]);
        }
        let tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
        aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let sbytes: Vec<u8> = aux.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut sc = [0i8; 16];
        for i in 0..16 {
            sc[i] = sbytes[i] as i8 - 32;
        }
        let base = ib * 256;
        let mut m: u8 = 1;
        let mut isc = 0usize;
        for n in (0..256usize).step_by(128) {
            let mut shift = 0u8;
            for _j in 0..4 {
                let dl = d_all * sc[isc] as f32;
                isc += 1;
                for l in 0..32usize {
                    let qi = (n / 128) * 32 + l;
                    let low = ((qs[qi] >> shift) & 3) as i32;
                    let high = if (hmask[l] & m) != 0 { 0 } else { 4 };
                    out[base + n + (shift as usize / 2) * 32 + l] = dl * (low - high) as f32;
                }
                shift += 2;
            }
            m <<= 1;
        }
    }
    out
}

fn dequant_q4_k(bytes: &[u8]) -> Vec<f32> {
    let nb = bytes.len() / 144;
    let mut out = vec![0.0f32; nb * 256];
    for ib in 0..nb {
        let blk = &bytes[ib * 144..ib * 144 + 144];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let base = ib * 256;
        let mut acc = 0usize;
        for i_sb in 0..8usize {
            let (sc, mn) = q4k_get_scale_min(scales, i_sb);
            let dl = d * sc as f32;
            let ml = dmin * mn as f32;
            let q = &qs[(i_sb / 2) * 32..(i_sb / 2) * 32 + 32];
            let hi = i_sb & 1;
            for k in 0..32 {
                let nib = if hi == 0 { q[k] & 0x0F } else { q[k] >> 4 };
                out[base + acc] = dl * nib as f32 - ml;
                acc += 1;
            }
        }
    }
    out
}

#[inline]
fn q4k_get_scale_min(scales: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, mn)
    }
}

fn dequant_q5_k(bytes: &[u8]) -> Vec<f32> {
    // block_q5_K: d(half) dmin(half) scales[12] qh[32] qs[128] = 176 B.
    let nb = bytes.len() / 176;
    let mut out = vec![0.0f32; nb * 256];
    for ib in 0..nb {
        let blk = &bytes[ib * 176..ib * 176 + 176];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = f16_bits_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let base = ib * 256;
        let mut acc = 0usize;
        for i_sb in 0..8usize {
            let (sc, mn) = q4k_get_scale_min(scales, i_sb);
            let dl = d * sc as f32;
            let ml = dmin * mn as f32;
            let q = &qs[(i_sb / 2) * 32..(i_sb / 2) * 32 + 32];
            let hi = i_sb & 1;
            let hbit = i_sb;
            for k in 0..32 {
                let nib = if hi == 0 { q[k] & 0x0F } else { q[k] >> 4 } as i32;
                let h = ((qh[k] >> hbit) & 1) as i32;
                out[base + acc] = dl * (nib + (h << 4)) as f32 - ml;
                acc += 1;
            }
        }
    }
    out
}

fn dequant_q6_k(bytes: &[u8]) -> Vec<f32> {
    // block_q6_K: ql[128] qh[64] scales[16](i8) d(half) = 210 B.
    let nb = bytes.len() / 210;
    let mut out = vec![0.0f32; nb * 256];
    for ib in 0..nb {
        let blk = &bytes[ib * 210..ib * 210 + 210];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let scales = &blk[192..208];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let base = ib * 256;
        for n in (0..256usize).step_by(128) {
            let ql_h = &ql[(n / 2)..(n / 2) + 64];
            let qh_h = &qh[(n / 4)..(n / 4) + 32];
            let sc = &scales[(n / 16)..(n / 16) + 8];
            for l in 0..32usize {
                let is = l / 16;
                let q1 = ((ql_h[l] & 0x0F) as i32) | ((((qh_h[l]) & 3) as i32) << 4);
                let q2 = ((ql_h[l + 32] & 0x0F) as i32) | ((((qh_h[l] >> 2) & 3) as i32) << 4);
                let q3 = ((ql_h[l] >> 4) as i32) | ((((qh_h[l] >> 4) & 3) as i32) << 4);
                let q4 = ((ql_h[l + 32] >> 4) as i32) | ((((qh_h[l] >> 6) & 3) as i32) << 4);
                out[base + n + l] = d * (sc[is] as i8) as f32 * (q1 - 32) as f32;
                out[base + n + l + 32] = d * (sc[is + 2] as i8) as f32 * (q2 - 32) as f32;
                out[base + n + l + 64] = d * (sc[is + 4] as i8) as f32 * (q3 - 32) as f32;
                out[base + n + l + 96] = d * (sc[is + 6] as i8) as f32 * (q4 - 32) as f32;
            }
        }
    }
    out
}

// ---- i-quants (exact ggml reference; grids in glm5_iq_grids.rs) ----

fn dequant_iq2_xxs(bytes: &[u8]) -> Vec<f32> {
    let grid = iq2xxs_grid_tbl();
    let ksigns = &KSIGNS_IQ2XS_RAW;
    let nb = bytes.len() / 66;
    let mut out = Vec::with_capacity(nb * 256);
    for ib in 0..nb {
        let blk = &bytes[ib * 66..ib * 66 + 66];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut qs = [0u16; 32];
        for i in 0..32 {
            qs[i] = u16::from_le_bytes([blk[2 + 2 * i], blk[2 + 2 * i + 1]]);
        }
        for ib32 in 0..8u32 {
            let q2 = &qs[(4 * ib32) as usize..(4 * ib32 + 4) as usize];
            let aux32_g = (q2[0] as u32) | ((q2[1] as u32) << 16);
            let aux32_s = (q2[2] as u32) | ((q2[3] as u32) << 16);
            let aux8 = aux32_g.to_le_bytes();
            let dl = d * (0.5 + (aux32_s >> 28) as f32) * 0.25;
            for k in 0..4u32 {
                let grid_bytes = grid[aux8[k as usize] as usize].to_le_bytes();
                let signs = ksigns[((aux32_s >> (7 * k)) & 127) as usize];
                for i in 0..8 {
                    let s = if (signs & (1u8 << i)) != 0 { -1.0 } else { 1.0 };
                    out.push(dl * (grid_bytes[i] as i8) as f32 * s);
                }
            }
        }
    }
    out
}

/// `block_iq3_xxs` (98 B, 256 w): d(half,0..2) qs[64](2..66) scales_and_signs[32]
/// (66..98). Per ggml dequantize_row_iq3_xxs: 8 ib32; aux32 = scales_and_signs
/// word; db = d*(0.5+(aux32>>28))*0.5; 4 sub-l each 2 grid bytes (u32→4 u8) with
/// ksigns_iq2xs sign mask. iq3xxs_grid is u32 (4 packed u8 lattice points).
fn dequant_iq3_xxs(bytes: &[u8]) -> Vec<f32> {
    let grid = iq3xxs_grid_tbl();
    let ksigns = &KSIGNS_IQ2XS_RAW;
    let kmask = &KMASK_IQ2XS_RAW;
    let nb = bytes.len() / 98;
    let mut out = Vec::with_capacity(nb * 256);
    for i in 0..nb {
        let blk = &bytes[i * 98..i * 98 + 98];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..66]; // 64 bytes
        let sas = &blk[66..98]; // scales_and_signs, 32 bytes
        for ib32 in 0..8usize {
            let aux32 = u32::from_le_bytes([
                sas[4 * ib32],
                sas[4 * ib32 + 1],
                sas[4 * ib32 + 2],
                sas[4 * ib32 + 3],
            ]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4usize {
                let signs = ksigns[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = grid[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
                let g2 = grid[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
                // ggml writes y[j+0] (g1) and y[j+4] (g2); sequential push order
                // is y[0..4] from g1 then y[4..8] from g2.
                for j in 0..4usize {
                    let s1 = if signs & kmask[j] != 0 { -1.0 } else { 1.0 };
                    out.push(db * g1[j] as f32 * s1);
                }
                for j in 0..4usize {
                    let s2 = if signs & kmask[j + 4] != 0 { -1.0 } else { 1.0 };
                    out.push(db * g2[j] as f32 * s2);
                }
            }
        }
    }
    out
}

/// `block_iq4_xs` (136 B, 256 w): d(half,0..2) scales_h u16(2..4) scales_l[4]
/// (4..8) qs[128](8..136). Per ggml dequantize_row_iq4_xs: 8 ib; ls 6-bit from
/// scales_l nibble + scales_h 2-bit; dl = d*(ls-32); nibbles → kvalues_iq4nl.
fn dequant_iq4_xs(bytes: &[u8]) -> Vec<f32> {
    let kv = &KVALUES_IQ4NL_RAW;
    let nb = bytes.len() / 136;
    let mut out = vec![0.0f32; nb * 256];
    for i in 0..nb {
        let blk = &bytes[i * 136..i * 136 + 136];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let scales_h = u16::from_le_bytes([blk[2], blk[3]]);
        let scales_l = &blk[4..8];
        let qs = &blk[8..136];
        let base = i * 256;
        for ib in 0..8usize {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32)
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            let q = &qs[ib * 16..ib * 16 + 16];
            for j in 0..16usize {
                out[base + ib * 32 + j] = dl * kv[(q[j] & 0xf) as usize] as f32;
                out[base + ib * 32 + 16 + j] = dl * kv[(q[j] >> 4) as usize] as f32;
            }
        }
    }
    out
}

const IQ1S_DELTA: f32 = 0.125;

/// `block_iq1_s` (50 B, 256 w): d(half, 0..2) qs[32](2..34) qh[16]u16(34..50).
/// Per ib (8 of them): dl = d*(2*((qh>>12)&7)+1); delta from bit 15;
/// 4 grids per ib indexed `qs[l] | (((qh>>3l)&7)<<8)` into iq1s_grid (u64 →
/// 8×i8); y = dl*(grid[j]+delta).
fn dequant_iq1_s(bytes: &[u8]) -> Vec<f32> {
    let grid = iq1s_grid_tbl();
    let nb = bytes.len() / 50;
    let mut out = Vec::with_capacity(nb * 256);
    for i in 0..nb {
        let blk = &bytes[i * 50..i * 50 + 50];
        let d = f16_bits_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let qs = &blk[2..34];
        let mut qh = [0u16; 8];
        for j in 0..8 {
            qh[j] = u16::from_le_bytes([blk[34 + 2 * j], blk[34 + 2 * j + 1]]);
        }
        for ib in 0..8usize {
            let dl = d * (2 * ((qh[ib] >> 12) & 7) + 1) as f32;
            let delta = if qh[ib] & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4usize {
                let gi = qs[4 * ib + l] as usize | (((qh[ib] >> (3 * l)) & 7) as usize) << 8;
                let g = grid[gi].to_le_bytes();
                for j in 0..8usize {
                    out.push(dl * ((g[j] as i8) as f32 + delta));
                }
            }
        }
    }
    out
}

/// `block_iq1_m` (56 B, 256 w): qs[32](0..32) qh[16](32..48) scales[4]u16(48..56).
/// Per ggml dequantize_row_iq1_m.
fn dequant_iq1_m(bytes: &[u8]) -> Vec<f32> {
    let grid = iq1s_grid_tbl();
    let nb = bytes.len() / 56;
    let mut out = Vec::with_capacity(nb * 256);
    for i in 0..nb {
        let blk = &bytes[i * 56..i * 56 + 56];
        let qs_all = &blk[0..32];
        let qh_all = &blk[32..48];
        let sc = [
            u16::from_le_bytes([blk[48], blk[49]]),
            u16::from_le_bytes([blk[50], blk[51]]),
            u16::from_le_bytes([blk[52], blk[53]]),
            u16::from_le_bytes([blk[54], blk[55]]),
        ];
        // super-block scale d from the 4×4 high nibbles.
        let dbits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16_bits_to_f32(dbits);
        let mut qs = qs_all;
        let mut qh = qh_all;
        for ib in 0..8usize {
            let dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1) as f32;
            let dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
            let idx = [
                qs[0] as usize | ((qh[0] as usize) << 8 & 0x700),
                qs[1] as usize | ((qh[0] as usize) << 4 & 0x700),
                qs[2] as usize | ((qh[1] as usize) << 8 & 0x700),
                qs[3] as usize | ((qh[1] as usize) << 4 & 0x700),
            ];
            let delta = [
                if qh[0] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[0] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[1] & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh[1] & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];
            for l in 0..2usize {
                let g = grid[idx[l]].to_le_bytes();
                for j in 0..8usize {
                    out.push(dl1 * ((g[j] as i8) as f32 + delta[l]));
                }
            }
            for l in 2..4usize {
                let g = grid[idx[l]].to_le_bytes();
                for j in 0..8usize {
                    out.push(dl2 * ((g[j] as i8) as f32 + delta[l]));
                }
            }
            qs = &qs[4..];
            qh = &qh[2..];
        }
    }
    out
}

// ============================ GGUF name mapping ============================

/// Map a logical weight to its GGUF tensor name. `layer == usize::MAX` for
/// globals. Returns the GGUF name; the streaming forward then dequant-streams it.
pub fn gguf_name(layer: usize, role: &str) -> String {
    match role {
        // globals
        "embed" => "token_embd.weight".into(),
        "final_norm" => "output_norm.weight".into(),
        "lm_head" => "output.weight".into(),
        // per-layer norms
        "input_layernorm" => format!("blk.{layer}.attn_norm.weight"),
        "post_attention_layernorm" => format!("blk.{layer}.ffn_norm.weight"),
        // MLA
        "q_a_proj" => format!("blk.{layer}.attn_q_a.weight"),
        "q_a_layernorm" => format!("blk.{layer}.attn_q_a_norm.weight"),
        "q_b_proj" => format!("blk.{layer}.attn_q_b.weight"),
        "kv_a_proj_with_mqa" => format!("blk.{layer}.attn_kv_a_mqa.weight"),
        "kv_a_layernorm" => format!("blk.{layer}.attn_kv_a_norm.weight"),
        "k_b" => format!("blk.{layer}.attn_k_b.weight"),
        "v_b" => format!("blk.{layer}.attn_v_b.weight"),
        "o_proj" => format!("blk.{layer}.attn_output.weight"),
        // indexer (DSA)
        "indexer_wq_b" => format!("blk.{layer}.indexer.attn_q_b.weight"),
        "indexer_wk" => format!("blk.{layer}.indexer.attn_k.weight"),
        "indexer_k_norm_w" => format!("blk.{layer}.indexer.k_norm.weight"),
        "indexer_k_norm_b" => format!("blk.{layer}.indexer.k_norm.bias"),
        "indexer_weights_proj" => format!("blk.{layer}.indexer.proj.weight"),
        // MoE
        "moe_router" => format!("blk.{layer}.ffn_gate_inp.weight"),
        "moe_router_bias" => format!("blk.{layer}.exp_probs_b.bias"),
        "moe_gate_exps" => format!("blk.{layer}.ffn_gate_exps.weight"),
        "moe_up_exps" => format!("blk.{layer}.ffn_up_exps.weight"),
        "moe_down_exps" => format!("blk.{layer}.ffn_down_exps.weight"),
        "moe_gate_shexp" => format!("blk.{layer}.ffn_gate_shexp.weight"),
        "moe_up_shexp" => format!("blk.{layer}.ffn_up_shexp.weight"),
        "moe_down_shexp" => format!("blk.{layer}.ffn_down_shexp.weight"),
        // dense FFN (first_k_dense_replace layers)
        "ffn_gate" => format!("blk.{layer}.ffn_gate.weight"),
        "ffn_up" => format!("blk.{layer}.ffn_up.weight"),
        "ffn_down" => format!("blk.{layer}.ffn_down.weight"),
        other => panic!("gguf_name: unknown role {other}"),
    }
}

// ============================ streaming forward ===========================
// Same math as glm5_ref but pulling each tensor lazily from the mmap. Single
// sequence prefill over `tokens`; returns logits for the LAST position (decode)
// or all positions if `all_logits`. KV-less recompute prefill (correctness path).

/// Returns logits for the final token position `[vocab]`.
pub fn forward_last_logits(
    tokens: &[usize],
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    verbose: bool,
) -> Result<Vec<f32>> {
    let seq = tokens.len();
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;

    // causal mask t<=s
    let mut causal = vec![false; seq * seq];
    for s in 0..seq {
        for t in 0..=s {
            causal[s * seq + t] = true;
        }
    }

    // embed (gather rows; dequant the whole embedding once — it's promoted to
    // a low-bit-but-not-IQ1 type and is vocab*hidden, large but one-shot).
    let embed = g.dequant_name(&gguf_name(0, "embed"))?;
    let mut h = vec![0.0f32; seq * hidden];
    for (s, &tok) in tokens.iter().enumerate() {
        let src = tok * hidden;
        h[s * hidden..s * hidden + hidden].copy_from_slice(&embed[src..src + hidden]);
    }
    drop(embed);

    // The GGUF block_count (79) includes the trailing MTP/nextn layer (blk.78),
    // which llama.cpp marks "unused" and the MAIN forward must NOT run — it is a
    // separate speculative-decode head. Detect it by its `nextn.eh_proj` tensor
    // and stop before it (run only the real transformer layers, 0..78).
    let n_transformer = (0..cfg.num_hidden_layers)
        .take_while(|&l| !g.contains(&format!("blk.{l}.nextn.eh_proj.weight")))
        .count();
    let mut prev_topk: Option<Vec<Vec<usize>>> = None;
    for layer in 0..n_transformer {
        if verbose {
            eprintln!("  [stream] layer {layer}/{n_transformer}");
        }
        let w_in = g.dequant_name(&gguf_name(layer, "input_layernorm"))?;
        let xin = rms_norm(&h, seq, hidden, &w_in, cfg.rms_norm_eps);
        drop(w_in);

        let (attn_out, topk) = attention_stream(
            &xin,
            seq,
            cfg,
            g,
            layer,
            cfg.indexer_types[layer],
            prev_topk.as_ref(),
            &causal,
        )?;
        if let Some(tk) = topk {
            prev_topk = Some(tk);
        }
        for i in 0..seq * hidden {
            h[i] += attn_out[i];
        }

        let w_pln = g.dequant_name(&gguf_name(layer, "post_attention_layernorm"))?;
        let pln = rms_norm(&h, seq, hidden, &w_pln, cfg.rms_norm_eps);
        drop(w_pln);

        let mlp_out = if layer < cfg.first_k_dense_replace {
            let gate = g.dequant_name(&gguf_name(layer, "ffn_gate"))?;
            let up = g.dequant_name(&gguf_name(layer, "ffn_up"))?;
            let down = g.dequant_name(&gguf_name(layer, "ffn_down"))?;
            swiglu_mlp(&pln, seq, hidden, cfg.intermediate_size, &gate, &up, &down)
        } else {
            moe_stream(&pln, seq, cfg, g, layer)?
        };
        for i in 0..seq * hidden {
            h[i] += mlp_out[i];
        }
    }

    // final norm + lm_head, only for the last position to save a vocab×hidden
    // matvec over every token.
    let w_fn = g.dequant_name(&gguf_name(0, "final_norm"))?;
    let last = &h[(seq - 1) * hidden..seq * hidden];
    let fnorm = rms_norm(last, 1, hidden, &w_fn, cfg.rms_norm_eps);
    drop(w_fn);
    let lm = g.dequant_name(&gguf_name(0, "lm_head"))?;
    let logits = linear(&fnorm, 1, hidden, &lm, vocab);
    Ok(logits)
}

fn attention_stream(
    x: &[f32],
    seq: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    layer: usize,
    layer_type: LayerType,
    prev_topk: Option<&Vec<Vec<usize>>>,
    causal: &[bool],
) -> Result<(Vec<f32>, Option<Vec<Vec<usize>>>)> {
    let hidden = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let qk_nope = cfg.qk_nope_head_dim;
    let qk_rope = cfg.qk_rope_head_dim;
    let qhd = cfg.q_head_dim();
    let kv_lora = cfg.kv_lora_rank;
    let vhd = cfg.v_head_dim;
    let scale = (qhd as f32).powf(-0.5);
    let theta = cfg.rope_theta;
    let conv = cfg.rope_convention;

    let w_qa = g.dequant_name(&gguf_name(layer, "q_a_proj"))?;
    let qa = linear(x, seq, hidden, &w_qa, cfg.q_lora_rank);
    drop(w_qa);
    let w_qan = g.dequant_name(&gguf_name(layer, "q_a_layernorm"))?;
    let qr = rms_norm(&qa, seq, cfg.q_lora_rank, &w_qan, 1e-6);
    drop(w_qan);
    let w_qb = g.dequant_name(&gguf_name(layer, "q_b_proj"))?;
    let q = linear(&qr, seq, cfg.q_lora_rank, &w_qb, nh * qhd);
    drop(w_qb);

    let w_kva = g.dequant_name(&gguf_name(layer, "kv_a_proj_with_mqa"))?;
    let ckv = linear(x, seq, hidden, &w_kva, kv_lora + qk_rope);
    drop(w_kva);
    let mut comp = vec![0.0f32; seq * kv_lora];
    let mut k_pe = vec![0.0f32; seq * qk_rope];
    for s in 0..seq {
        let row = &ckv[s * (kv_lora + qk_rope)..(s + 1) * (kv_lora + qk_rope)];
        comp[s * kv_lora..s * kv_lora + kv_lora].copy_from_slice(&row[..kv_lora]);
        k_pe[s * qk_rope..s * qk_rope + qk_rope].copy_from_slice(&row[kv_lora..]);
    }
    let w_kvan = g.dequant_name(&gguf_name(layer, "kv_a_layernorm"))?;
    let kv_latent = rms_norm(&comp, seq, kv_lora, &w_kvan, 1e-6);
    drop(w_kvan);

    let mut q_nope = vec![0.0f32; seq * nh * qk_nope];
    let mut q_pe = vec![0.0f32; seq * nh * qk_rope];
    for s in 0..seq {
        for hh in 0..nh {
            let src = (s * nh + hh) * qhd;
            q_nope[(s * nh + hh) * qk_nope..(s * nh + hh) * qk_nope + qk_nope]
                .copy_from_slice(&q[src..src + qk_nope]);
            let pe = &mut q_pe[(s * nh + hh) * qk_rope..(s * nh + hh) * qk_rope + qk_rope];
            pe.copy_from_slice(&q[src + qk_nope..src + qhd]);
            rope_apply(pe, qk_rope, s, theta, conv);
        }
        rope_apply(
            &mut k_pe[s * qk_rope..s * qk_rope + qk_rope],
            qk_rope,
            s,
            theta,
            conv,
        );
    }

    let topk_used: Option<Vec<Vec<usize>>> = match layer_type {
        LayerType::Full => indexer_stream(x, &qr, seq, cfg, g, layer, causal)?,
        LayerType::Shared => prev_topk.cloned(),
    };

    let neg = f32::MIN;
    let mut pe_scores = vec![0.0f32; nh * seq * seq];
    for hh in 0..nh {
        for s in 0..seq {
            for t in 0..seq {
                let allowed = causal[s * seq + t]
                    && topk_used
                        .as_ref()
                        .map(|tk| tk[s].contains(&t))
                        .unwrap_or(true);
                if !allowed {
                    pe_scores[(hh * seq + s) * seq + t] = neg;
                    continue;
                }
                let qoff = (s * nh + hh) * qk_rope;
                let koff = t * qk_rope;
                let mut dot = 0.0f32;
                for d in 0..qk_rope {
                    dot += q_pe[qoff + d] * k_pe[koff + d];
                }
                pe_scores[(hh * seq + s) * seq + t] = scale * dot;
            }
        }
    }

    // absorbed K/V via attn_k_b / attn_v_b. GGUF stores these SPLIT (DS4 fuses);
    // k_b dims [qk_nope, kv_lora, nh], v_b dims [kv_lora? …]. We treat:
    //   k_abs[h,t,d] = sum_l kv_latent[t,l] * k_b[h,l,d]   (per-head [kv_lora,qk_nope])
    //   v_abs[h,t,d] = sum_l kv_latent[t,l] * v_b[h,d,l]   (per-head [vhd,kv_lora])
    let k_b = g.dequant_name(&gguf_name(layer, "k_b"))?; // [nh*qk_nope*kv_lora] layout
    let v_b = g.dequant_name(&gguf_name(layer, "v_b"))?;
    // k_b GGUF dims are [qk_nope, kv_lora, nh] (last dim = heads), row-major =>
    // element (d_nope, l, h) at ((d_nope*kv_lora)+l)*nh + h. v_b dims
    // [kv_lora, vhd, nh] => (l, d_v, h) at ((l*vhd)+d_v)*nh + h. We index
    // generically using the handle dims.
    let kb_dims = g.handle(&gguf_name(layer, "k_b")).unwrap().dims.clone();
    let vb_dims = g.handle(&gguf_name(layer, "v_b")).unwrap().dims.clone();
    let k_get = make_indexer3(&kb_dims);
    let v_get = make_indexer3(&vb_dims);

    let mut k_abs = vec![0.0f32; nh * seq * qk_nope];
    let mut v_abs = vec![0.0f32; nh * seq * vhd];
    for hh in 0..nh {
        for t in 0..seq {
            let kl = &kv_latent[t * kv_lora..t * kv_lora + kv_lora];
            for d in 0..qk_nope {
                let mut acc = 0.0f32;
                for l in 0..kv_lora {
                    acc += kl[l] * k_b[k_get(d, l, hh)];
                }
                k_abs[(hh * seq + t) * qk_nope + d] = acc;
            }
            for d in 0..vhd {
                let mut acc = 0.0f32;
                for l in 0..kv_lora {
                    acc += kl[l] * v_b[v_get(l, d, hh)];
                }
                v_abs[(hh * seq + t) * vhd + d] = acc;
            }
        }
    }
    drop(k_b);
    drop(v_b);

    let mut out = vec![0.0f32; seq * nh * vhd];
    for hh in 0..nh {
        for s in 0..seq {
            let qoff = (s * nh + hh) * qk_nope;
            let mut logits = vec![0.0f32; seq];
            let mut maxl = f32::NEG_INFINITY;
            for t in 0..seq {
                let bias = pe_scores[(hh * seq + s) * seq + t];
                if bias == neg {
                    logits[t] = f32::NEG_INFINITY;
                    continue;
                }
                let koff = (hh * seq + t) * qk_nope;
                let mut dot = 0.0f32;
                for d in 0..qk_nope {
                    dot += q_nope[qoff + d] * k_abs[koff + d];
                }
                logits[t] = scale * dot + bias;
                if logits[t] > maxl {
                    maxl = logits[t];
                }
            }
            let mut denom = 0.0f32;
            for t in 0..seq {
                if logits[t] == f32::NEG_INFINITY {
                    logits[t] = 0.0;
                } else {
                    logits[t] = (logits[t] - maxl).exp();
                    denom += logits[t];
                }
            }
            let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            let ob = (s * nh + hh) * vhd;
            for t in 0..seq {
                let p = logits[t] * inv;
                if p == 0.0 {
                    continue;
                }
                let vb = (hh * seq + t) * vhd;
                for d in 0..vhd {
                    out[ob + d] += p * v_abs[vb + d];
                }
            }
        }
    }

    let w_o = g.dequant_name(&gguf_name(layer, "o_proj"))?;
    let attn_out = linear(&out, seq, nh * vhd, &w_o, hidden);
    Ok((attn_out, topk_used))
}

/// Build a closure mapping logical (i,j,k) into a row-major 3-D tensor with the
/// given GGUF dims. GGUF dims are stored fastest-first; element (i,j,k) for
/// dims `[d0,d1,d2]` (d0 fastest) is at `i + j*d0 + k*d0*d1`. Our callers pass
/// indices in "logical math order"; we resolve by matching dim sizes.
fn make_indexer3(dims: &[u64]) -> impl Fn(usize, usize, usize) -> usize {
    // GGUF stores dims fastest-varying first. For a weight saved from a
    // PyTorch/mlx tensor of logical shape [A,B,C], the GGUF dims are [C,B,A].
    // We compute the flat index assuming row-major with dims as given
    // (fastest-first): flat = i0 + i1*dims[0] + i2*dims[0]*dims[1], where
    // (i0,i1,i2) is the access ordered fastest-first. Callers index in
    // logical (slowest..fastest) — but here our three call sites pass exactly
    // (fastest, mid, slowest)=(d, l, h) and (l, d, h) which matches GGUF
    // fastest-first for the absorbed-KV layout. Keep it explicit and simple.
    let d0 = dims.first().copied().unwrap_or(1) as usize;
    let d1 = dims.get(1).copied().unwrap_or(1) as usize;
    move |a: usize, b: usize, c: usize| a + b * d0 + c * d0 * d1
}

fn indexer_stream(
    x: &[f32],
    qr: &[f32],
    seq: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    layer: usize,
    causal: &[bool],
) -> Result<Option<Vec<Vec<usize>>>> {
    if seq <= cfg.index_topk {
        return Ok(None);
    }
    let n_heads = cfg.index_n_heads;
    let hd = cfg.index_head_dim;
    let hidden = cfg.hidden_size;
    let theta = cfg.rope_theta;
    let conv = cfg.rope_convention;

    let wqb = g.dequant_name(&gguf_name(layer, "indexer_wq_b"))?;
    let mut q = linear(qr, seq, cfg.q_lora_rank, &wqb, n_heads * hd);
    drop(wqb);
    let wk = g.dequant_name(&gguf_name(layer, "indexer_wk"))?;
    let k_lin = linear(x, seq, hidden, &wk, hd);
    drop(wk);
    let knw = g.dequant_name(&gguf_name(layer, "indexer_k_norm_w"))?;
    let knb = g.dequant_name(&gguf_name(layer, "indexer_k_norm_b"))?;
    let mut k = layer_norm(&k_lin, seq, hd, &knw, &knb, 1e-5);
    drop(knw);
    drop(knb);

    let wp = g.dequant_name(&gguf_name(layer, "indexer_weights_proj"))?;
    let mut weights = linear(x, seq, hidden, &wp, n_heads);
    drop(wp);
    let wscale = (n_heads as f32).powf(-0.5) * (hd as f32).powf(-0.5);
    for v in weights.iter_mut() {
        *v *= wscale;
    }

    let rope_dim = cfg.qk_rope_head_dim.min(hd);
    for s in 0..seq {
        for hh in 0..n_heads {
            let base = (s * n_heads + hh) * hd;
            rope_apply(&mut q[base..base + hd], rope_dim, s, theta, conv);
        }
        rope_apply(&mut k[s * hd..s * hd + hd], rope_dim, s, theta, conv);
    }

    let mut out = Vec::with_capacity(seq);
    for s in 0..seq {
        let mut row = vec![f32::NEG_INFINITY; seq];
        for t in 0..seq {
            if !causal[s * seq + t] {
                continue;
            }
            let mut acc = 0.0f32;
            for hh in 0..n_heads {
                let qoff = (s * n_heads + hh) * hd;
                let koff = t * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[qoff + d] * k[koff + d];
                }
                acc += dot.max(0.0) * weights[s * n_heads + hh];
            }
            row[t] = acc;
        }
        let mut idx: Vec<usize> = (0..seq).collect();
        idx.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(cfg.index_topk);
        out.push(idx);
    }
    Ok(Some(out))
}

fn moe_stream(
    x: &[f32],
    seq: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    layer: usize,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let n_exp = cfg.n_routed_experts;

    let gate_w = g.dequant_name(&gguf_name(layer, "moe_router"))?;
    let e_bias = g.dequant_name(&gguf_name(layer, "moe_router_bias"))?;

    // route every token, collect the union of selected experts so each expert
    // is dequantized at most ONCE per layer.
    let mut inds_per_tok = Vec::with_capacity(seq);
    let mut scores_per_tok = Vec::with_capacity(seq);
    let mut needed: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for s in 0..seq {
        let xs = &x[s * hidden..s * hidden + hidden];
        let mut gates = vec![0.0f32; n_exp];
        for e in 0..n_exp {
            let we = &gate_w[e * hidden..e * hidden + hidden];
            gates[e] = xs.iter().zip(we).map(|(a, b)| a * b).sum();
        }
        let (inds, scores) = group_expert_select(
            &gates,
            &e_bias,
            cfg.num_experts_per_tok,
            cfg.n_group,
            cfg.topk_group,
            cfg.routed_scaling_factor,
            cfg.norm_topk_prob,
        );
        for &e in &inds {
            needed.insert(e);
        }
        inds_per_tok.push(inds);
        scores_per_tok.push(scores);
    }
    drop(gate_w);
    drop(e_bias);

    let mut out = vec![0.0f32; seq * hidden];
    // For each needed expert: dequant its gate/up/down once, apply to every
    // token that selected it.
    for &e in &needed {
        let gate = g.dequant_expert(&gguf_name(layer, "moe_gate_exps"), e, n_exp)?;
        let up = g.dequant_expert(&gguf_name(layer, "moe_up_exps"), e, n_exp)?;
        let down = g.dequant_expert(&gguf_name(layer, "moe_down_exps"), e, n_exp)?;
        for s in 0..seq {
            if let Some(k) = inds_per_tok[s].iter().position(|&x| x == e) {
                let xs = &x[s * hidden..s * hidden + hidden];
                let ye = swiglu_mlp(xs, 1, hidden, inter, &gate, &up, &down);
                let sc = scores_per_tok[s][k];
                for i in 0..hidden {
                    out[s * hidden + i] += sc * ye[i];
                }
            }
        }
    }

    // shared experts (dense over all tokens)
    if cfg.n_shared_experts > 0 {
        let inter_sh = cfg.moe_intermediate_size * cfg.n_shared_experts;
        let sg = g.dequant_name(&gguf_name(layer, "moe_gate_shexp"))?;
        let su = g.dequant_name(&gguf_name(layer, "moe_up_shexp"))?;
        let sd = g.dequant_name(&gguf_name(layer, "moe_down_shexp"))?;
        let sh = swiglu_mlp(x, seq, hidden, inter_sh, &sg, &su, &sd);
        for i in 0..seq * hidden {
            out[i] += sh[i];
        }
    }
    Ok(out)
}

/// Greedy generation: prefill `prompt`, then generate `n_new` tokens by
/// argmax, re-running a (recompute) prefill each step. Correctness path —
/// O(seq²) per step, but coherence is the goal, not speed.
pub fn generate_greedy(
    prompt: &[usize],
    n_new: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    verbose: bool,
) -> Result<Vec<usize>> {
    let mut tokens = prompt.to_vec();
    let mut generated = Vec::with_capacity(n_new);
    for step in 0..n_new {
        if verbose {
            eprintln!("[gen] step {step}/{n_new}, ctx={}", tokens.len());
        }
        let logits = forward_last_logits(&tokens, cfg, g, verbose)?;
        let next = argmax(&logits);
        tokens.push(next);
        generated.push(next);
    }
    Ok(generated)
}

pub fn argmax(v: &[f32]) -> usize {
    let mut bi = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

// ============================ KV-cache decode =============================
//
// a TRUE token-by-token decode loop with a growing full-KV cache,
// replacing the O(seq²)-per-step recompute prefill (`generate_greedy`) for the
// generation path. GLM-5.2 is V3.2-style FULL-KV MLA (no sliding window, no
// compressor ring): the indexer selects top-`index_topk` over the ENTIRE KV
// history, so the cache must retain, for every past position and every layer:
//   * the compressed MLA latent `kv_latent[kv_lora]` (what k_b/v_b absorb), and
//   * the RoPE'd key positional part `k_pe[qk_rope]`, and
//   * the RoPE'd + normed indexer key `idx_k[index_head_dim]`
// so each new token can (a) run the indexer top-k over the full cached idx_k,
// (b) MLA-attend its q against the absorbed cached latent, with NO recompute of
// the past. The math is line-for-line `forward_last_logits`'s attention; only
// the storage differs (cache vs recompute), so the decoded argmax sequence is
// byte-identical to `generate_greedy` (validated in the engine + metal tests).

/// Per-layer growing full-KV cache. Row-major, one row appended per decode step.
#[derive(Clone, Default)]
pub struct LayerKv {
    /// `[n_pos, kv_lora]` compressed+normed MLA latent (k_b/v_b absorb against).
    pub kv_latent: Vec<f32>,
    /// `[n_pos, qk_rope]` RoPE'd shared key positional part.
    pub k_pe: Vec<f32>,
    /// `[n_pos, index_head_dim]` RoPE'd + layer-normed indexer key (full history).
    pub idx_k: Vec<f32>,
}

/// The full decode cache: one `LayerKv` per transformer layer, plus the number
/// of positions currently held.
pub struct Glm5KvCache {
    pub layers: Vec<LayerKv>,
    pub n_pos: usize,
}

impl Glm5KvCache {
    pub fn new(n_transformer: usize) -> Self {
        Glm5KvCache {
            layers: vec![LayerKv::default(); n_transformer],
            n_pos: 0,
        }
    }
}

/// Number of real transformer layers (stops before the trailing MTP/nextn head).
pub fn n_transformer_layers(cfg: &DefaultsGlm5, g: &ShardedGguf) -> usize {
    (0..cfg.num_hidden_layers)
        .take_while(|&l| !g.contains(&format!("blk.{l}.nextn.eh_proj.weight")))
        .count()
}

/// One KV-cache decode step on the CPU: process token `tok` at absolute position
/// `cache.n_pos`, appending its KV to every layer's cache and returning the
/// next-token logits `[vocab]`. The attention math is identical to
/// `attention_stream`'s last-row computation, but K/V come from the cache (the
/// new token's KV is appended first, so the new row attends to itself + history).
pub fn decode_step(
    tok: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    cache: &mut Glm5KvCache,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let pos = cache.n_pos; // 0-indexed absolute position of this token

    // embed this single token.
    let embed = g.dequant_name(&gguf_name(0, "embed"))?;
    let mut h = vec![0.0f32; hidden];
    h.copy_from_slice(&embed[tok * hidden..tok * hidden + hidden]);
    drop(embed);

    let n_transformer = cache.layers.len();
    let mut prev_topk: Option<Vec<usize>> = None;
    for layer in 0..n_transformer {
        let w_in = g.dequant_name(&gguf_name(layer, "input_layernorm"))?;
        let xin = rms_norm(&h, 1, hidden, &w_in, cfg.rms_norm_eps);
        drop(w_in);

        let (attn_out, topk) = attention_decode_step(
            &xin,
            pos,
            cfg,
            g,
            layer,
            cfg.indexer_types[layer],
            prev_topk.as_ref(),
            &mut cache.layers[layer],
        )?;
        if let Some(tk) = topk {
            prev_topk = Some(tk);
        }
        for i in 0..hidden {
            h[i] += attn_out[i];
        }

        let w_pln = g.dequant_name(&gguf_name(layer, "post_attention_layernorm"))?;
        let pln = rms_norm(&h, 1, hidden, &w_pln, cfg.rms_norm_eps);
        drop(w_pln);

        let mlp_out = if layer < cfg.first_k_dense_replace {
            let gate = g.dequant_name(&gguf_name(layer, "ffn_gate"))?;
            let up = g.dequant_name(&gguf_name(layer, "ffn_up"))?;
            let down = g.dequant_name(&gguf_name(layer, "ffn_down"))?;
            swiglu_mlp(&pln, 1, hidden, cfg.intermediate_size, &gate, &up, &down)
        } else {
            moe_stream(&pln, 1, cfg, g, layer)?
        };
        for i in 0..hidden {
            h[i] += mlp_out[i];
        }
    }
    cache.n_pos += 1;

    let w_fn = g.dequant_name(&gguf_name(0, "final_norm"))?;
    let fnorm = rms_norm(&h, 1, hidden, &w_fn, cfg.rms_norm_eps);
    drop(w_fn);
    let lm = g.dequant_name(&gguf_name(0, "lm_head"))?;
    let logits = linear(&fnorm, 1, hidden, &lm, vocab);
    Ok(logits)
}

/// CPU MTP (nextn) drafter reference — the clean reference for the Metal
/// `mtp_draft_step`, mirroring vLLM's GLM-4.5 MTP forward with the validated CPU
/// primitives (no k_abs cache / idx_k fusion). Given the main model's pre-final-norm
/// hidden `h_final` and the just-produced `next_tok`, predicts the token after that.
/// `mtp_cache` is the MTP layer's own 1-layer KV cache. Returns draft logits.
pub fn mtp_draft_step_cpu(
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    h_final: &[f32],
    next_tok: usize,
    pos: usize,
    mtp_cache: &mut Glm5KvCache,
) -> Result<Vec<f32>> {
    use crate::glm5_ref::{linear, rms_norm};
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let eps = cfg.rms_norm_eps;
    let l = n_transformer_layers(cfg, g);

    // enorm(embed(next_tok)) ; hnorm(h_final) ; eh_proj(cat[embed, hidden]).
    let embed_all = g.dequant_name(&gguf_name(0, "embed"))?;
    let embed = &embed_all[next_tok * hidden..next_tok * hidden + hidden];
    let enorm_w = g.dequant_name(&format!("blk.{l}.nextn.enorm.weight"))?;
    let hnorm_w = g.dequant_name(&format!("blk.{l}.nextn.hnorm.weight"))?;
    let e_norm = rms_norm(embed, 1, hidden, &enorm_w, eps);
    let h_norm = rms_norm(h_final, 1, hidden, &hnorm_w, eps);
    drop(embed_all);
    let mut concat = vec![0.0f32; 2 * hidden];
    concat[..hidden].copy_from_slice(&e_norm);
    concat[hidden..].copy_from_slice(&h_norm);
    let eh = g.dequant_name(&format!("blk.{l}.nextn.eh_proj.weight"))?;
    let x = linear(&concat, 1, 2 * hidden, &eh, hidden);
    drop(eh);

    // blk.L transformer layer (own KV cache).
    let w_in = g.dequant_name(&gguf_name(l, "input_layernorm"))?;
    let xin = rms_norm(&x, 1, hidden, &w_in, eps);
    let lt = *cfg.indexer_types.get(l).unwrap_or(&LayerType::Full);
    let (attn_out, _topk) =
        attention_decode_step(&xin, pos, cfg, g, l, lt, None, &mut mtp_cache.layers[0])?;
    let mut h = x;
    for i in 0..hidden {
        h[i] += attn_out[i];
    }
    let w_pln = g.dequant_name(&gguf_name(l, "post_attention_layernorm"))?;
    let pln = rms_norm(&h, 1, hidden, &w_pln, eps);
    let mlp = moe_stream(&pln, 1, cfg, g, l)?;
    for i in 0..hidden {
        h[i] += mlp[i];
    }

    // shared_head_norm + shared lm_head.
    let shn = g.dequant_name(&format!("blk.{l}.nextn.shared_head_norm.weight"))?;
    let fnorm = rms_norm(&h, 1, hidden, &shn, eps);
    let lm = g.dequant_name(&gguf_name(0, "lm_head"))?;
    Ok(linear(&fnorm, 1, hidden, &lm, vocab))
}

/// One-position attention against the growing cache. `x` is the single new
/// token's RMS-normed hidden `[hidden]` at absolute position `pos`. Appends this
/// token's `(kv_latent, k_pe, idx_k)` to `kv`, then attends the new query over
/// all `pos+1` cached positions. Returns `(attn_out[hidden], indexer_topk)`.
fn attention_decode_step(
    x: &[f32],
    pos: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    layer: usize,
    layer_type: LayerType,
    prev_topk: Option<&Vec<usize>>,
    kv: &mut LayerKv,
) -> Result<(Vec<f32>, Option<Vec<usize>>)> {
    let hidden = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let qk_nope = cfg.qk_nope_head_dim;
    let qk_rope = cfg.qk_rope_head_dim;
    let qhd = cfg.q_head_dim();
    let kv_lora = cfg.kv_lora_rank;
    let vhd = cfg.v_head_dim;
    let scale = (qhd as f32).powf(-0.5);
    let theta = cfg.rope_theta;
    let conv = cfg.rope_convention;

    // q_a → q_a_norm → q_b for the single new token.
    let w_qa = g.dequant_name(&gguf_name(layer, "q_a_proj"))?;
    let qa = linear(x, 1, hidden, &w_qa, cfg.q_lora_rank);
    drop(w_qa);
    let w_qan = g.dequant_name(&gguf_name(layer, "q_a_layernorm"))?;
    let qr = rms_norm(&qa, 1, cfg.q_lora_rank, &w_qan, 1e-6);
    drop(w_qan);
    let w_qb = g.dequant_name(&gguf_name(layer, "q_b_proj"))?;
    let q = linear(&qr, 1, cfg.q_lora_rank, &w_qb, nh * qhd);
    drop(w_qb);

    // kv_a (compressed kv + k_pe) → kv_a_norm for the new token; APPEND to cache.
    let w_kva = g.dequant_name(&gguf_name(layer, "kv_a_proj_with_mqa"))?;
    let ckv = linear(x, 1, hidden, &w_kva, kv_lora + qk_rope);
    drop(w_kva);
    let comp = &ckv[..kv_lora];
    let mut k_pe_new = ckv[kv_lora..kv_lora + qk_rope].to_vec();
    let w_kvan = g.dequant_name(&gguf_name(layer, "kv_a_layernorm"))?;
    let kv_latent_new = rms_norm(comp, 1, kv_lora, &w_kvan, 1e-6);
    drop(w_kvan);
    rope_apply(&mut k_pe_new, qk_rope, pos, theta, conv);
    kv.kv_latent.extend_from_slice(&kv_latent_new);
    kv.k_pe.extend_from_slice(&k_pe_new);
    let n_kv = pos + 1;

    // split new q into per-head nope/pe, RoPE the pe parts at position `pos`.
    let mut q_nope = vec![0.0f32; nh * qk_nope];
    let mut q_pe = vec![0.0f32; nh * qk_rope];
    for hh in 0..nh {
        let src = hh * qhd;
        q_nope[hh * qk_nope..hh * qk_nope + qk_nope].copy_from_slice(&q[src..src + qk_nope]);
        let pe = &mut q_pe[hh * qk_rope..hh * qk_rope + qk_rope];
        pe.copy_from_slice(&q[src + qk_nope..src + qhd]);
        rope_apply(pe, qk_rope, pos, theta, conv);
    }

    // indexer: append the new token's idx_k, then top-k over the full cache for
    // the new query row. Inactive when n_kv <= index_topk (real-needle regime).
    let topk_used: Option<Vec<usize>> =
        indexer_decode_step(x, &qr, pos, cfg, g, layer, layer_type, prev_topk, kv)?;

    // absorbed K/V over the FULL cache (cheap matvecs; k_b/v_b are layer-const).
    let k_b = g.dequant_name(&gguf_name(layer, "k_b"))?;
    let v_b = g.dequant_name(&gguf_name(layer, "v_b"))?;
    let kb_dims = g.handle(&gguf_name(layer, "k_b")).unwrap().dims.clone();
    let vb_dims = g.handle(&gguf_name(layer, "v_b")).unwrap().dims.clone();
    let k_get = make_indexer3(&kb_dims);
    let v_get = make_indexer3(&vb_dims);

    // attention for the single new query row over all n_kv cached positions.
    let mut out = vec![0.0f32; nh * vhd];
    for hh in 0..nh {
        let qn = &q_nope[hh * qk_nope..hh * qk_nope + qk_nope];
        let qp = &q_pe[hh * qk_rope..hh * qk_rope + qk_rope];
        let mut logits = vec![f32::NEG_INFINITY; n_kv];
        let mut maxl = f32::NEG_INFINITY;
        for t in 0..n_kv {
            // causal is implicit (t<=pos always); apply indexer top-k if active.
            if let Some(tk) = topk_used.as_ref() {
                if !tk.contains(&t) {
                    continue;
                }
            }
            // absorbed k for (hh,t): k_abs[d] = Σ_l kv_latent[t,l]·k_b(d,l,hh).
            let kl = &kv.kv_latent[t * kv_lora..t * kv_lora + kv_lora];
            let mut dot = 0.0f32;
            for d in 0..qk_nope {
                let mut acc = 0.0f32;
                for l in 0..kv_lora {
                    acc += kl[l] * k_b[k_get(d, l, hh)];
                }
                dot += qn[d] * acc;
            }
            // pe score against cached k_pe[t].
            let kp = &kv.k_pe[t * qk_rope..t * qk_rope + qk_rope];
            let mut pdot = 0.0f32;
            for d in 0..qk_rope {
                pdot += qp[d] * kp[d];
            }
            let sc = scale * dot + scale * pdot;
            logits[t] = sc;
            if sc > maxl {
                maxl = sc;
            }
        }
        let mut denom = 0.0f32;
        for t in 0..n_kv {
            if logits[t] == f32::NEG_INFINITY {
                logits[t] = 0.0;
            } else {
                logits[t] = (logits[t] - maxl).exp();
                denom += logits[t];
            }
        }
        let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
        let ob = hh * vhd;
        for t in 0..n_kv {
            let p = logits[t] * inv;
            if p == 0.0 {
                continue;
            }
            // absorbed v for (hh,t): v_abs[d] = Σ_l kv_latent[t,l]·v_b(l,d,hh).
            let kl = &kv.kv_latent[t * kv_lora..t * kv_lora + kv_lora];
            for d in 0..vhd {
                let mut acc = 0.0f32;
                for l in 0..kv_lora {
                    acc += kl[l] * v_b[v_get(l, d, hh)];
                }
                out[ob + d] += p * acc;
            }
        }
    }
    drop(k_b);
    drop(v_b);

    let w_o = g.dequant_name(&gguf_name(layer, "o_proj"))?;
    let attn_out = linear(&out, 1, nh * vhd, &w_o, hidden);
    Ok((attn_out, topk_used))
}

/// Indexer top-k for the single new query at `pos`, over the full cached idx_k.
/// Appends the new token's RoPE'd+normed idx_k to `kv.idx_k` (Full layers
/// recompute the key projection; Shared layers reuse `prev_topk` but STILL must
/// append their key — identical to the recompute path where the key is computed
/// every layer but the top-k selection is shared). Returns the selected
/// positions for the new query (or `None` when the indexer is inactive).
#[allow(clippy::too_many_arguments)]
fn indexer_decode_step(
    x: &[f32],
    qr: &[f32],
    pos: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    layer: usize,
    layer_type: LayerType,
    prev_topk: Option<&Vec<usize>>,
    kv: &mut LayerKv,
) -> Result<Option<Vec<usize>>> {
    let n_kv = pos + 1;
    let hd = cfg.index_head_dim;
    let hidden = cfg.hidden_size;
    let theta = cfg.rope_theta;
    let conv = cfg.rope_convention;
    let rope_dim = cfg.qk_rope_head_dim.min(hd);

    // Always append THIS token's indexer key (the cache must grow every layer so
    // a later step's full-KV select sees it), regardless of Full/Shared — this
    // mirrors `indexer_stream` computing k for all positions each recompute.
    let wk = g.dequant_name(&gguf_name(layer, "indexer_wk"))?;
    let k_lin = linear(x, 1, hidden, &wk, hd);
    drop(wk);
    let knw = g.dequant_name(&gguf_name(layer, "indexer_k_norm_w"))?;
    let knb = g.dequant_name(&gguf_name(layer, "indexer_k_norm_b"))?;
    let mut k_new = layer_norm(&k_lin, 1, hd, &knw, &knb, 1e-5);
    drop(knw);
    drop(knb);
    rope_apply(&mut k_new, rope_dim, pos, theta, conv);
    kv.idx_k.extend_from_slice(&k_new);

    // Inactive when the cache fits within top_k: dense causal (the new query
    // attends to everything). Mirrors `indexer_stream`'s `seq <= index_topk`.
    if n_kv <= cfg.index_topk {
        return Ok(None);
    }
    if let LayerType::Shared = layer_type {
        return Ok(prev_topk.cloned());
    }

    let n_heads = cfg.index_n_heads;
    // indexer q for the new token (per indexer head), RoPE'd at `pos`.
    let wqb = g.dequant_name(&gguf_name(layer, "indexer_wq_b"))?;
    let mut q = linear(qr, 1, cfg.q_lora_rank, &wqb, n_heads * hd);
    drop(wqb);
    for hh in 0..n_heads {
        rope_apply(&mut q[hh * hd..hh * hd + hd], rope_dim, pos, theta, conv);
    }
    // per-head weights for the new query.
    let wp = g.dequant_name(&gguf_name(layer, "indexer_weights_proj"))?;
    let mut weights = linear(x, 1, hidden, &wp, n_heads);
    drop(wp);
    let wscale = (n_heads as f32).powf(-0.5) * (hd as f32).powf(-0.5);
    for v in weights.iter_mut() {
        *v *= wscale;
    }

    // score the new query against every cached idx_k (full history) and top-k.
    let mut row = vec![f32::NEG_INFINITY; n_kv];
    for t in 0..n_kv {
        let kt = &kv.idx_k[t * hd..t * hd + hd];
        let mut acc = 0.0f32;
        for hh in 0..n_heads {
            let qh = &q[hh * hd..hh * hd + hd];
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += qh[d] * kt[d];
            }
            acc += dot.max(0.0) * weights[hh];
        }
        row[t] = acc;
    }
    let mut idx: Vec<usize> = (0..n_kv).collect();
    idx.sort_by(|&a, &b| {
        row[b]
            .partial_cmp(&row[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(cfg.index_topk);
    Ok(Some(idx))
}

/// Greedy generation via the KV-cache decode loop: prefill `prompt` token-by-
/// token into the cache, then generate `n_new` tokens by argmax, each step a
/// single-row forward over the growing cache (NO O(seq²) recompute). Produces
/// the SAME argmax sequence as `generate_greedy` (the recompute oracle) — the
/// per-step math is identical, only KV storage differs.
pub fn generate_greedy_kv(
    prompt: &[usize],
    n_new: usize,
    cfg: &DefaultsGlm5,
    g: &ShardedGguf,
    verbose: bool,
) -> Result<Vec<usize>> {
    assert!(!prompt.is_empty(), "generate_greedy_kv: empty prompt");
    let n_transformer = n_transformer_layers(cfg, g);
    let mut cache = Glm5KvCache::new(n_transformer);

    // prefill: stream every prompt token through the cache; the last prompt
    // token's logits give the first generated token.
    let mut last_logits = Vec::new();
    for (i, &tok) in prompt.iter().enumerate() {
        if verbose {
            eprintln!("[gen-kv] prefill {i}/{}", prompt.len());
        }
        last_logits = decode_step(tok, cfg, g, &mut cache)?;
    }

    let mut generated = Vec::with_capacity(n_new);
    for step in 0..n_new {
        let next = argmax(&last_logits);
        generated.push(next);
        if step + 1 == n_new {
            break;
        }
        if verbose {
            eprintln!("[gen-kv] decode {step}/{n_new}, ctx={}", cache.n_pos);
        }
        last_logits = decode_step(next, cfg, g, &mut cache)?;
    }
    Ok(generated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iq1s_grid_first_entry_is_all_minus_one() {
        // 0xffffffffffffffff cast to int8 = eight -1.
        let g = iq1s_grid_tbl();
        let bytes = g[0].to_le_bytes();
        for b in bytes {
            assert_eq!(b as i8, -1);
        }
        assert_eq!(g.len(), 2048);
    }

    #[test]
    fn iq2xxs_and_ksigns_sizes() {
        assert_eq!(iq2xxs_grid_tbl().len(), 256);
        assert_eq!(KSIGNS_IQ2XS_RAW.len(), 128);
    }

    #[test]
    fn dequant_q8_0_roundtrip_simple() {
        // one block: d=1.0 (f16 0x3c00), qs = 0,1,2,...,31
        let mut bytes = vec![0u8; 34];
        bytes[0] = 0x00;
        bytes[1] = 0x3c; // f16 1.0
        for k in 0..32 {
            bytes[2 + k] = k as u8;
        }
        let out = dequant_blocks(GgmlType::Q8_0, &bytes, 32).unwrap();
        for k in 0..32 {
            assert!((out[k] - k as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn iq1s_block_dequant_runs_and_is_finite() {
        // 50-byte IQ1_S block: d=1.0, qs all 0, qh all 0 -> grid index 0 (all -1),
        // ls = 2*0+1 = 1, delta=+0.125 -> every weight = 1*( -1 + 0.125 ) = -0.875.
        let mut bytes = vec![0u8; 50];
        bytes[0] = 0x00;
        bytes[1] = 0x3c; // f16 1.0
        let out = dequant_blocks(GgmlType::IQ1_S, &bytes, 256).unwrap();
        assert_eq!(out.len(), 256);
        for v in &out {
            assert!((v - (-0.875)).abs() < 1e-4, "v={v}");
        }
    }

    #[test]
    fn kv_cache_helpers_basic() {
        // Glm5KvCache grows one row per layer per step; LayerKv default is empty.
        let mut c = Glm5KvCache::new(3);
        assert_eq!(c.layers.len(), 3);
        assert_eq!(c.n_pos, 0);
        for l in &c.layers {
            assert!(l.kv_latent.is_empty() && l.k_pe.is_empty() && l.idx_k.is_empty());
        }
        c.n_pos += 1;
        assert_eq!(c.n_pos, 1);
    }

    #[test]
    fn argmax_picks_first_max() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY, -1.0]), 1);
    }

    #[test]
    fn discover_shards_parses_count() {
        // non-existent paths still parse the pattern (open would fail later).
        let p = Path::new("/tmp/does-not-exist/GLM-5.2-UD-IQ1_S-00001-of-00006.gguf");
        // discover requires files to exist; just exercise the single-file branch.
        let single = Path::new("/tmp/model.gguf");
        assert_eq!(discover_shards(single).unwrap().len(), 1);
        let _ = p;
    }
}

/// MXFP4 format tests, ported from antirez's `tests/test_mxfp4_dot.c` (@1f862bd).
///
/// These deliberately reuse HIS constants, HIS LCG seed and HIS golden value, so
/// agreement is cross-implementation evidence rather than us checking our own
/// arithmetic against itself.
#[cfg(test)]
mod mxfp4_tests {
    use super::{dequant_mxfp4, e8m0_to_f32};
    use crate::gguf::GgmlType;

    /// The lift that lets MXFP4 reuse the PTO backend's PROVEN Q8_0 lowering.
    ///
    /// The Ascend PTO path has no sub-byte dequant primitive — the DS4-Flash-Q2
    /// handover records Q2_K falling back to AIV scalar and IQ2_XXS needing a host
    /// precompute. MXFP4 at 4.25 bits would hit the same wall, which made it the
    /// gating unknown for the 910 port.
    ///
    /// It does not have to. Every one of the 16 FP4 code points is a HALF-INTEGER
    /// (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6), so `value * 2` is an exact integer in
    /// [-12, 12] — comfortably int8. Folding the compensating ½ into the block
    /// scale gives
    ///
    /// ```text
    ///   dst = Σ (value*2 as i8) * (d/2) * x
    /// ```
    ///
    /// which is bit-for-bit the shape the Q8_0 matvec already lowers: si8 weights,
    /// an f32 per-element scale, `tmul`, `trowsum`. No new tile op, no on-device
    /// nibble unpack; the host does the unpack and the LUT, exactly the route
    /// already accepted for IQ2_XXS.
    ///
    /// EXACT, not approximate — hence `assert_eq` on the bits rather than a
    /// tolerance. A tolerance here would hide the very thing being claimed.
    #[test]
    fn mxfp4_folds_exactly_into_int8_times_half_scale() {
        // Every block byte pattern that matters: all 16 codes in both nibbles,
        // across a spread of exponents including the e==0 special case.
        for e in [0u8, 1, 64, 127, 128, 200, 254, 255] {
            let mut blk = vec![0u8; 17];
            blk[0] = e;
            for (j, b) in blk[1..].iter_mut().enumerate() {
                // low nibble walks the codes, high nibble walks them offset, so
                // every (low, high) pairing is exercised across the 16 bytes.
                *b = ((j as u8) & 0x0f) | ((((j as u8) + 7) & 0x0f) << 4);
            }
            let reference = dequant_mxfp4(&blk);
            let d = e8m0_to_f32(e);

            // The host-precompute the PTO kernel would be fed.
            const CODES2: [i8; 16] =
                [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
            let mut w_i8 = vec![0i8; 32];
            for (j, &q) in blk[1..].iter().enumerate() {
                w_i8[j] = CODES2[(q & 0x0f) as usize];
                w_i8[j + 16] = CODES2[(q >> 4) as usize];
            }
            let half_d = d * 0.5;

            for i in 0..32 {
                let folded = w_i8[i] as f32 * half_d;
                if folded.to_bits() == reference[i].to_bits() {
                    continue;
                }
                // The ONE permitted difference, found by this test rather than
                // assumed: code 8 is NEGATIVE zero, and `0i8 as f32 * d` yields
                // POSITIVE zero. Same value, different sign bit.
                //
                // Harmless here, and worth saying why rather than widening the
                // assertion: the kernel consumes these only through a dot product,
                // and +0 and -0 contribute identically to a sum. The sign could
                // only survive as the sign of an exactly-zero result, which cannot
                // change a logit ordering or an argmax.
                assert!(
                    folded == 0.0 && reference[i] == 0.0,
                    "e={e} i={i}: folded {folded} != reference {} and it is not the \
                     benign +-0 case — the int8 x half-scale lift is NOT exact, so \
                     MXFP4 cannot reuse the Q8_0 lowering",
                    reference[i]
                );
            }

            // What actually has to hold: the DOT is bit-identical, because that is
            // the only form the kernel computes. Signed zero cannot survive it.
            let y: Vec<f32> = (0..32).map(|k| (k as f32) * 0.37 - 5.0).collect();
            let want: f32 = reference.iter().zip(&y).map(|(a, b)| a * b).sum();
            let got: f32 = w_i8
                .iter()
                .zip(&y)
                .map(|(w, b)| *w as f32 * half_d * b)
                .sum();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "e={e}: folded dot {got} != reference dot {want}"
            );
        }
    }

    /// A dot product built on our dequant, matching his `vec_dot_mxfp4_f32`.
    fn dot(bytes: &[u8], y: &[f32]) -> f32 {
        dequant_mxfp4(bytes)
            .iter()
            .zip(y)
            .map(|(a, b)| a * b)
            .sum()
    }

    #[test]
    fn block_layout_is_17_bytes_of_32_values() {
        assert_eq!(GgmlType::MXFP4.type_size(), 17);
        assert_eq!(GgmlType::MXFP4.block_size(), 32);
        // ggml_id is the inverse of the private from_u32; 39 is the tag antirez
        // registers in his type table ("[39] = {\"mxfp4\", 32, 17}").
        assert_eq!(GgmlType::MXFP4.ggml_id(), 39);
    }

    #[test]
    fn e8m0_matches_ldexp_with_the_zero_special_case() {
        for e in [0u8, 1, 2, 126, 127, 128, 254] {
            // his: e == 0 ? ldexpf(1, -127) : ldexpf(1, e - 127)
            let expect = if e == 0 {
                (2f64).powi(-127) as f32
            } else {
                (2f64).powi(e as i32 - 127) as f32
            };
            assert_eq!(e8m0_to_f32(e), expect, "e8m0({e})");
        }
    }

    /// His golden constant. qs = 0xf6 everywhere means low nibble 6 (+4.0) and
    /// high nibble 15 (-6.0); with e=127 (scale 1.0) and y all ones the dot is
    /// 16*(4.0) + 16*(-6.0) = -32.0. This is the test that catches a swapped or
    /// interleaved nibble order, which is the easiest thing to get wrong.
    #[test]
    fn nibble_order_gives_his_golden_minus_32() {
        let mut blk = [0u8; 17];
        blk[0] = 127;
        for b in blk[1..].iter_mut() {
            *b = 0xf6;
        }
        let y = [1.0f32; 32];
        assert_eq!(dot(&blk, &y), -32.0);
    }

    /// His `test_random_reference`: same LCG, same seed, same 100 trials, checked
    /// against an independently-derived value (half of a twice-value table times
    /// a ldexp scale) rather than against the decoder under test.
    #[test]
    fn random_blocks_match_his_independent_reference() {
        const TWICE: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        fn reference_value(e: u8, q: u8) -> f32 {
            let scale = if e == 0 {
                (2f64).powi(-127) as f32
            } else {
                (2f64).powi(e as i32 - 127) as f32
            };
            0.5 * TWICE[q as usize] as f32 * scale
        }
        const N_BLOCK: usize = 7;
        const QK: usize = 32;
        let mut state: u32 = 0x1234_5678;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for trial in 0..100 {
            let mut blocks = vec![0u8; N_BLOCK * 17];
            for ib in 0..N_BLOCK {
                blocks[ib * 17] = (110 + next() % 35) as u8;
                for j in 0..16 {
                    blocks[ib * 17 + 1 + j] = (next() >> 24) as u8;
                }
            }
            let y: Vec<f32> = (0..N_BLOCK * QK)
                .map(|_| ((next() >> 16) as u16 as i16) as f32 / 4096.0)
                .collect();

            let got = dot(&blocks, &y);
            // The reference accumulates in f64. His C version sums it in f32 and
            // uses a 2e-6 tolerance calibrated to HIS summation order; ours walks
            // the dequantised elements in index order, and on trial 39 the two
            // f32 orderings differed by 0.5 on a sum of 143262 — 3.5e-6 relative,
            // which is ordering noise, not decode error (his golden -32.0 case
            // already pins the decode exactly). Summing the reference in f64
            // makes it the accurate side rather than a second f32 opinion.
            let mut expect = 0f64;
            for ib in 0..N_BLOCK {
                let e = blocks[ib * 17];
                for j in 0..16 {
                    let q = blocks[ib * 17 + 1 + j];
                    expect += reference_value(e, q & 0x0f) as f64 * y[ib * QK + j] as f64;
                    expect += reference_value(e, q >> 4) as f64 * y[ib * QK + j + QK / 2] as f64;
                }
            }
            let expect = expect as f32;
            // 1e-5 relative: 224 f32 additions of terms up to ~1e3 accumulate
            // worst-case n*eps ~= 2.7e-5 and typically sqrt(n)*eps ~= 1.8e-6, so
            // this bounds the ordering difference without hiding a decode fault —
            // a wrong nibble or scale is wrong by orders of magnitude, not by 1e-5.
            let tol = 1.0e-5 * 1.0f32.max(expect.abs());
            assert!(
                (got - expect).abs() <= tol,
                "trial {trial}: got {got:.9} expected {expect:.9}"
            );
        }
    }
}
