// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! End-to-end decode-step driver (CPU reference).
//!
//! Stacks the per-layer forward + MoE blocks, applies the final RMSNorm +
//! lm_head matvec, and returns either logits or a sampled token. This is the
//! oracle that Metal dispatch (E6 macOS landing) must match.
//!
//! Shapes are deliberately tiny in the unit tests — the real DS4 numbers
//! (61 layers, d_model=7168, d_ffn_moe=2048, etc.) are read from the GGUF
//! metadata at engine init.

#![allow(dead_code)]

use anyhow::{bail, Result};
use std::cell::RefCell;

use crate::attn_dispatch::{
    decode_attn_ffn_post_with, decode_attn_ffn_pre_with, decode_attn_prefix_with,
    AttentionDispatcher, AttnLayerWeights, AttnStepInputs, KvCacheView,
};
use crate::dispatch::{CpuDispatcher, KernelDispatcher};
use crate::kv_cache::KvCache;

/// One per-(layer, slot) HC fingerprint sample captured by the in-process
/// HC recorder (see `HC_SLOT_RECORDER`).
///
/// Mirrors the `HC_SLOT layer=… pos=… slot=… rms=… top3=…` stderr dump
/// emitted by `DS4_DUMP_HC_PER_SLOT=POS`. The recorder gives Linux unit
/// tests programmatic access to those numbers (the offline antirez-
/// fixture replay harness in `crates/engine_core/tests/m4_fixture_replay.rs`).
#[derive(Debug, Clone)]
pub struct HcSlotSample {
    pub layer: usize,
    pub pos: u32,
    pub slot: usize,
    pub rms: f64,
    pub top3: Vec<(usize, f32)>,
}

thread_local! {
    /// Optional in-process recorder. When `Some`, the HC dump block in
    /// `decode_step_with_attn` appends each per-(layer, slot) sample at
    /// the matching target pos. Default is `None` (no-op).
    pub static HC_SLOT_RECORDER: RefCell<Option<Vec<HcSlotSample>>> =
        const { RefCell::new(None) };

    /// Verifier-bisection tap: set `.0` to a target layer index; after that
    /// layer's attn-half `hc_collapse_norm` runs, decode_step stores the
    /// `normed` vector in `.1`. Lets a test compare the verifier's per-layer
    /// `normed_k` against decode_step's at the same layer (DS4 spec-decode
    /// emit-row bisection). `(usize::MAX, _)` = disabled.
    pub static NORMED_CAPTURE: RefCell<(usize, Vec<f32>)> =
        const { RefCell::new((usize::MAX, Vec::new())) };

    /// Verifier-bisection tap: `.0` = target layer; after that layer, stores
    /// `(after_attn_hc, after_ffn_hc)` in `.1`/`.2`. Splits the layer's post-flash
    /// path (attn-half output-proj+hc_expand vs FFN/MoE half) when comparing
    /// the verifier's per-layer residual to decode_step's.
    pub static LAYER_RESID_CAPTURE: RefCell<(usize, Vec<f32>, Vec<f32>)> =
        const { RefCell::new((usize::MAX, Vec::new(), Vec::new())) };

    /// CLEAN per-layer output capture (probe-only). When `Some`, the oracle
    /// `decode_step_with_attn*` pushes each layer's output residual (`after_ffn_hc`
    /// = `state.cur_hc`) — a NON-resident, non-drain reference for comparing the
    /// batched decoder per layer without perturbing a GPU resident chain.
    pub static ORACLE_RESID_CAP: RefCell<Option<Vec<Vec<f32>>>> =
        const { RefCell::new(None) };

    /// DSpark target-layer tap (gated, default-disarmed). When `armed`, the decode
    /// loop in `decode_step_with_attn_to_residual` captures the hc-mean-collapsed
    /// `[d]` residual *after* each layer in `targets` into the matching `captured`
    /// slot (in `targets` order), so `dspark_tap_take_main_hidden` can concat them
    /// into the DSpark `main_hidden` `[targets.len() * d]`. The normal decode path
    /// pays only a single cached-bool read at loop start when disarmed — see the
    /// `dspark_targets` snapshot in `decode_step_with_attn_to_residual`.
    pub static DSPARK_TAP: RefCell<DsparkTap> = const { RefCell::new(DsparkTap::new()) };
}
/// Arm/disarm the oracle per-layer output capture.
pub fn oracle_resid_cap_set(on: bool) {
    ORACLE_RESID_CAP.with(|c| *c.borrow_mut() = if on { Some(Vec::new()) } else { None });
}
/// Take the captured per-layer oracle outputs (clears the capture).
pub fn oracle_resid_cap_take() -> Option<Vec<Vec<f32>>> {
    ORACLE_RESID_CAP.with(|c| c.borrow_mut().take())
}

/// State of the [`DSPARK_TAP`] thread-local. `captured[i]` is the hc-mean-collapsed
/// `[d]` residual after layer `targets[i]` (or `None` until that layer runs).
#[derive(Clone, Debug)]
pub struct DsparkTap {
    pub armed: bool,
    pub targets: Vec<usize>,
    pub captured: Vec<Option<Vec<f32>>>,
}
impl DsparkTap {
    const fn new() -> Self {
        Self {
            armed: false,
            targets: Vec::new(),
            captured: Vec::new(),
        }
    }
}

/// Arm the DSpark target-layer tap for the current thread. After the next
/// `decode_step_with_attn[_to_residual]`, [`dspark_tap_take_main_hidden`] yields
/// `main_hidden` = concat of the hc-mean-collapsed residuals after each `targets`
/// layer (production: `dspark::consts::TARGET_LAYER_IDS` = `[40,41,42]`).
pub fn dspark_tap_arm(targets: &[usize]) {
    DSPARK_TAP.with(|c| {
        *c.borrow_mut() = DsparkTap {
            armed: true,
            targets: targets.to_vec(),
            captured: vec![None; targets.len()],
        };
    });
}

/// Disarm the DSpark tap and drop any captured residuals.
pub fn dspark_tap_disarm() {
    DSPARK_TAP.with(|c| *c.borrow_mut() = DsparkTap::new());
}

/// Drain the DSpark tap into the concatenated `main_hidden` `[targets.len() * d]`,
/// disarming it. Returns `None` if the tap was not armed or any target layer was
/// not reached (e.g. the model has fewer layers than `max(targets)+1`).
pub fn dspark_tap_take_main_hidden() -> Option<Vec<f32>> {
    DSPARK_TAP.with(|c| {
        let mut t = c.borrow_mut();
        if !t.armed {
            return None;
        }
        let caps = std::mem::take(&mut t.captured);
        *t = DsparkTap::new();
        let mut out = Vec::new();
        for cap in caps {
            out.extend(cap?); // any missing target → None
        }
        Some(out)
    })
}

/// DSpark hc-mean collapse: average the `n_hc` hyper-connection copies of an
/// `[n_hc * d]` HC residual into a single `[d]` vector. This is the plain
/// arithmetic mean over the hc dim (`h.mean(dim=2)` in DSpark `inference/model.py`)
/// that `main_proj` consumes — NOT the learned `output_hc_head_one` collapse.
pub fn hc_mean_collapse(cur_hc: &[f32], n_hc: usize, d: usize) -> Vec<f32> {
    debug_assert_eq!(
        cur_hc.len(),
        n_hc * d,
        "hc_mean_collapse: cur_hc must be n_hc*d"
    );
    let mut out = vec![0.0f32; d];
    for h in 0..n_hc {
        let slot = &cur_hc[h * d..(h + 1) * d];
        for (o, &v) in out.iter_mut().zip(slot) {
            *o += v;
        }
    }
    let inv = 1.0 / n_hc as f32;
    for o in out.iter_mut() {
        *o *= inv;
    }
    out
}

/// Arm the in-process HC-slot recorder for the current thread. Returns
/// any previously-armed buffer.
pub fn arm_hc_slot_recorder() -> Option<Vec<HcSlotSample>> {
    HC_SLOT_RECORDER.with(|r| r.replace(Some(Vec::new())))
}

/// Disarm and drain the recorder, returning the accumulated samples.
pub fn drain_hc_slot_recorder() -> Vec<HcSlotSample> {
    HC_SLOT_RECORDER
        .with(|r| r.replace(None))
        .unwrap_or_default()
}

/// Per-layer weights (tiny CPU oracle — real DS4 has 16 named tensors per layer).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub d_model: usize,
    /// MoE expert FFN dim. `None` for the Metal-loaded `.moe` (the real value lives
    /// in QuantizedExpertWeights — MetalDispatcher reads it from QEW); `Some` for the
    /// CPU oracle. NonZero + Option (was a bare `0` sentinel) so the "0 means read
    /// from QEW" contract is type-enforced and a 0 can never silently feed a kernel
    /// dispatch as a real dim. See memory: ffi-boundary-dim-validation-requirement.
    pub d_ffn: Option<std::num::NonZeroUsize>,
    pub n_experts: usize,
    pub n_experts_used: usize,
    // Pre-attention norm
    pub attn_norm_gamma: Vec<f32>, // [d_model]
    // Attention QKV+O — collapsed to a single dense projection for the CPU oracle.
    // Real MLA uses lora-down/lora-up + KV compression; we'll wire that once we have
    // the antirez oracle dump as ground truth.
    pub w_attn: Vec<f32>, // [d_model][d_model] row-major
    // Pre-FFN norm
    pub ffn_norm_gamma: Vec<f32>, // [d_model]
    // MoE router
    pub w_router: Vec<f32>, // [n_experts][d_model]
    /// No-copy F16 mmap bytes for `w_router` (`moe_router` is F16 in the GGUF).
    /// Present only in lean mode for NON-hash-routed layers (the GPU
    /// `encode_router_logits` matvec reads it via `matvec_f16`); empty for
    /// hash-routed layers (which keep the f32 for `run_ffn_half`) and non-lean.
    pub w_router_f16: std::borrow::Cow<'static, [u8]>,
    pub router_bias: Vec<f32>, // [n_experts]
    // hash routing table for early MoE layers (antirez ds4.c:5155-5157).
    // When present, `selected[i] = routing_table[token_id, i]`, bypassing top-k
    // and bias entirely; weights still come from probs via
    // `hash_router_weights_from_probs`. Layers without this table fall through
    // to the standard `router_finalize(probs, bias, k)` path. Layout matches
    // antirez `blk.N.ffn_gate_tid2eid.weight` of shape [n_experts_used, vocab].
    pub routing_table: Option<Vec<i32>>, // [vocab][n_experts_used] row-major
    // Stacked expert weights
    pub w_gate_exps: Vec<f32>, // [n_experts][d_ffn][d_model]
    pub w_up_exps: Vec<f32>,   // [n_experts][d_ffn][d_model]
    pub w_down_exps: Vec<f32>, // [n_experts][d_model][d_ffn]
}

impl LayerWeights {
    /// No-copy F16 router bytes as `Option<&[u8]>` for the GPU `matvec_f16` —
    /// `Some` only when the f16 path applies (lean, non-hash layer), else `None`
    /// (the f32 path runs). See [`Self::w_router_as_f32`] for the CPU/verifier side.
    #[inline]
    pub fn w_router_f16(&self) -> Option<&[u8]> {
        (!self.w_router_f16.is_empty()).then_some(&*self.w_router_f16)
    }

    /// `w_router` as f32: the dequant when present, else reconstructed from the
    /// no-copy F16 bytes (lean non-hash layers). Used by `run_ffn_half` (the
    /// hash/silu-fidelity fallback) and the K-verifier (`base_run_bundle`),
    /// which have no f16 router matvec. Cow::Borrowed when the f32 is present.
    pub fn w_router_as_f32(&self) -> std::borrow::Cow<'_, [f32]> {
        if !self.w_router.is_empty() || self.w_router_f16.is_empty() {
            std::borrow::Cow::Borrowed(&self.w_router)
        } else {
            std::borrow::Cow::Owned(
                self.w_router_f16
                    .chunks_exact(2)
                    .map(|c| crate::layer_view::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(),
            )
        }
    }
}

pub struct ModelWeights {
    pub layers: Vec<LayerWeights>,
    pub final_norm_gamma: Vec<f32>, // [d_model]
    pub lm_head: Vec<f32>,          // [vocab_size][d_model]
    pub vocab_size: usize,
    pub d_model: usize,
}

pub struct DecodeConfig {
    pub eps_rms: f32,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self { eps_rms: 1e-6 }
    }
}

/// One decode layer (CPU reference). Returns the residual-updated hidden state.
///
/// Skeleton:
///   x = x + W_attn · rms_norm(x, γ_attn)
///   x = x + MoE_FFN(rms_norm(x, γ_ffn))
///
/// The attention block is the simplified dense projection above; the real
/// MLA + compressed-KV path lands once we have antirez oracle dumps to A/B
/// against (the kernel registry is already populated for it).
pub fn decode_layer(x: &[f32], w: &LayerWeights, cfg: &DecodeConfig) -> Vec<f32> {
    decode_layer_with(&CpuDispatcher, 0, x, w, cfg)
}

/// Generic decode-layer over any `KernelDispatcher`. The CPU oracle calls this
/// via `decode_layer`; the Metal path will call it with a `MetalDispatcher`.
///
/// `layer_idx` is the 0-based layer position in the stack; backends that
/// route through preloaded per-layer quantized tables (e.g. `MetalDispatcher`)
/// use it to select the right table.
pub fn decode_layer_with<D: KernelDispatcher>(
    dispatch: &D,
    layer_idx: u32,
    x: &[f32],
    w: &LayerWeights,
    cfg: &DecodeConfig,
) -> Vec<f32> {
    let d = w.d_model;
    assert_eq!(x.len(), d);

    // ── Attention sub-block ─────────────────────────────────────────────
    let x_norm = dispatch.rms_norm(x, &w.attn_norm_gamma, cfg.eps_rms);
    let attn_out = dispatch.matvec_f32(&w.w_attn, &x_norm, d);
    let mut h: Vec<f32> = x
        .iter()
        .zip(attn_out.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    // ── MoE FFN sub-block ───────────────────────────────────────────────
    let h_norm = dispatch.rms_norm(&h, &w.ffn_norm_gamma, cfg.eps_rms);
    let logits = dispatch.matvec_f32(&w.w_router, &h_norm, w.n_experts);
    let probs = dispatch.softplus_sqrt(&logits);
    let (selected, weights) = dispatch.router_finalize(&probs, &w.router_bias, w.n_experts_used);
    let ffn_out = dispatch.moe_routed_step(
        layer_idx,
        &h_norm,
        &selected,
        &weights,
        &w.w_gate_exps,
        &w.w_up_exps,
        &w.w_down_exps,
        w.d_ffn.map_or(0, |n| n.get()),
    );
    for (hv, &ov) in h.iter_mut().zip(ffn_out.iter()) {
        *hv += ov;
    }
    h
}

/// Full decode step (CPU reference): stack of layers + final norm + lm_head.
pub fn decode_step(x: Vec<f32>, model: &ModelWeights, cfg: &DecodeConfig) -> Result<Vec<f32>> {
    decode_step_with(&CpuDispatcher, x, model, cfg)
}

/// Generic decode step over any `KernelDispatcher`.
pub fn decode_step_with<D: KernelDispatcher>(
    dispatch: &D,
    mut x: Vec<f32>,
    model: &ModelWeights,
    cfg: &DecodeConfig,
) -> Result<Vec<f32>> {
    if x.len() != model.d_model {
        bail!(
            "decode_step: x has {} elements, expected d_model={}",
            x.len(),
            model.d_model
        );
    }
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        if layer.d_model != model.d_model {
            bail!("layer.d_model mismatch");
        }
        x = decode_layer_with(dispatch, layer_idx as u32, &x, layer, cfg);
    }
    let x = dispatch.rms_norm(&x, &model.final_norm_gamma, cfg.eps_rms);
    anyhow::ensure!(
        !model.lm_head.is_empty(),
        "f32 lm_head was dropped (DS4_LOW_RAM); the CPU reference decode needs it — unset DS4_LOW_RAM"
    );
    Ok(dispatch.matvec_f32(&model.lm_head, &x, model.vocab_size))
}

// ---------------------------------------------------------------------------
// Phase B1 of #225 — composed decode step over (KernelDispatcher,
// AttentionDispatcher) for a single position. Synthetic 2-trait model;
// real GGUF weight load lands in Phase B2.
// ---------------------------------------------------------------------------

/// Per-layer weights for the composed decode step: combines the attention
/// half (`AttnLayerWeights`) with the MoE/router half (`LayerWeights`).
/// Shared shape contract: `attn.params.d_embd == moe.d_model`.
#[derive(Debug, Clone)]
pub struct ComposedLayerWeights {
    pub attn: AttnLayerWeights,
    pub moe: LayerWeights,
}

/// Composed model: stack of attention + MoE layers + final norm + lm_head.
#[derive(Debug, Clone)]
pub struct ComposedModelWeights {
    pub layers: Vec<ComposedLayerWeights>,
    pub final_norm_gamma: Vec<f32>,
    pub lm_head: Vec<f32>,
    /// Raw GGUF `block_q8_0` bytes for the LM head (`output.weight`, natively
    /// Q8_0 in DS4 V4 Flash). Lets the decode tail read 1 byte/weight (562 MB)
    /// instead of dequantizing to f32 and reading 4 bytes/weight (2.1 GB) every
    /// token. Empty when the source tensor isn't Q8_0 (falls back to `lm_head`).
    pub lm_head_q8: std::borrow::Cow<'static, [u8]>,
    /// Re-quantized `block_q4_0` bytes for the LM head (DS4_LOW_RAM + DS4_Q4_LMHEAD).
    /// When non-empty, the decode tail reads it via `ds4_kernel_mul_mv_q4_0_f32`
    /// (~281 MB vs the 562 MB Q8_0) and both `lm_head` (f32) and `lm_head_q8`
    /// are dropped. Empty in the default (Q8_0) configuration.
    pub lm_head_q4: Vec<u8>,
    pub vocab_size: usize,
    pub d_model: usize,
    /// Output-side HC fold weights (antirez `output_hc_head_one`, ds4.c:7654-7681).
    /// Optional because the synthetic test fixtures don't populate them; when
    /// present they replace the "take slot 0" shortcut at the end of decode.
    /// Shapes (antirez ds4.c:2143-2145):
    ///   `output_hc_base ` : F32 [n_hc]
    ///   `output_hc_fn   ` : F16 [hc_dim × n_hc]  where hc_dim = d_model × n_hc
    ///   `output_hc_scale` : F32 [1]
    /// `n_hc` is taken from the first layer's `AttnLayerWeights::params.n_hc`.
    pub output_hc_base: Option<Vec<f32>>,
    pub output_hc_fn: Option<Vec<f32>>,
    pub output_hc_scale: Option<Vec<f32>>,
}

impl ComposedModelWeights {
    /// Build a `ComposedModelWeights` from a `LayerViews` + DS4 defaults
    /// ( Phase B3b). Designed for the **Metal production path** where
    /// `MetalDispatcher` ignores the f32 routed-expert slices and reads from
    /// `QuantizedExpertWeights` instead. For the CPU oracle path (which DOES
    /// consume the f32 routed-expert slices) use the synthetic builders in
    /// `tests/m4_composed_decode.rs` — full Q2_K / IQ2_XXS dequant of all
    /// 61 layers × 256 experts would be ~80 GB and is not what this is for.
    ///
    /// Per-layer attention half goes through `AttnLayerWeights::from_layer_view`.
    /// Per-layer MoE half: norm γ + router weights + router bias are dequanted
    /// from F32/F16; routed-expert slices left empty (length-0 `Vec`s) —
    /// `MetalDispatcher::moe_routed_step` ignores them.
    ///
    /// `final_norm_gamma` and `lm_head` come from `LayerViews::global` (roles
    /// `final_norm` and `lm_head`).
    /// Build composed weights, KEEPING the f32 duplicates of q8-backed weights
    /// (the trait / `run_argmax` decode path reads them).
    pub fn from_views(
        views: &crate::layer_view::LayerViews,
        manifest: &crate::gguf::ModelManifest,
        defaults: crate::attn_dispatch::DefaultsDs4,
    ) -> Result<Self> {
        Self::from_views_impl(views, manifest, defaults, false)
    }

    /// Lean build for the encoder / `DecodeSession` (server) path: SKIP the f32
    /// duplicates of the q8-backed weights (attn q/kv, output, shared, lm_head)
    /// so the ~22 GB is never allocated — vs allocated-then-freed, which leaves
    /// it in malloc's free-cache and inflates phys_footprint. Bit-identical
    /// decode (the encoder reads those weights via q8). The trait/`run_argmax`
    /// f32 path must use `from_views` instead.
    pub fn from_views_lean(
        views: &crate::layer_view::LayerViews,
        manifest: &crate::gguf::ModelManifest,
        defaults: crate::attn_dispatch::DefaultsDs4,
    ) -> Result<Self> {
        Self::from_views_impl(views, manifest, defaults, true)
    }

    fn from_views_impl<S: crate::w8a8_roles::WeightSource>(
        src: &S,
        manifest: &crate::gguf::ModelManifest,
        defaults: crate::attn_dispatch::DefaultsDs4,
        lean: bool,
    ) -> Result<Self> {
        let g = crate::gguf::GgufFile::open(&manifest.path)?;
        let d_model = manifest.d_model as usize;
        let vocab_size = manifest.vocab_size as usize;
        let n_layers = manifest.n_layers as usize;
        let n_experts = manifest.n_experts as usize;
        let n_experts_used = manifest.n_experts_used as usize;
        // Bound once: `globals()` may return an OWNED map for a package, and a reference into a
        // temporary would not outlive the reads below.
        let globals = src.globals()?;

        let final_norm_h = globals
            .get("final_norm")
            .ok_or_else(|| anyhow::anyhow!("missing global tensor: final_norm"))?;
        let final_norm_gamma = src.dequant_f32(final_norm_h)?;

        let lm_head_h = globals
            .get("lm_head")
            .ok_or_else(|| anyhow::anyhow!("missing global tensor: lm_head"))?;
        let lm_head_is_q8 = lm_head_h.ttype == crate::gguf::GgmlType::Q8_0;
        let lm_head_is_q4 = lm_head_h.ttype == crate::gguf::GgmlType::Q4_0;
        // Raw Q8_0 bytes (the default DS4 V4 Flash lm_head type) so the decode
        // tail can read 1 byte/weight instead of the dequantized f32.
        let lm_head_q8_src: std::borrow::Cow<'static, [u8]> = if lm_head_is_q8 {
            // No-copy: borrow the q8 lm_head bytes straight from the mmap (same
            // SAFETY as the attn q8 borrow — DecodeRunner retains the mmap past
            // `composed`; MAP_SHARED keeps the pages clean file-backed).
            let s = src.bytes_for(lm_head_h)?;
            let s: &[u8] = &s;
            std::borrow::Cow::Borrowed(unsafe { std::mem::transmute::<&[u8], &'static [u8]>(s) })
        } else {
            std::borrow::Cow::Borrowed(&[])
        };
        // q4 lm_head, two routes:
        //  (a) ON DISK (offline re-quant tool, requant_gguf_q8_0_to_q4_0):
        //      output.weight is already Q4_0 → mmap it directly, NO load-time
        //      requant, NO transient q8 copy (the production low-RAM path).
        //  (b) LOAD-TIME (DS4_Q4_LMHEAD on a Q8_0 model): re-quantize at load
        //      (~281 MB) and drop the q8 — convenient but transiently holds both.
        let want_q4_requant =
            std::env::var("DS4_Q4_LMHEAD").ok().as_deref() == Some("1") && lm_head_is_q8;
        let lm_head_q4: Vec<u8> = if lm_head_is_q4 {
            let q4 = src.bytes_for(lm_head_h)?.to_vec();
            eprintln!(
                "engine_core: q4 lm_head from disk ({} MB)",
                q4.len() / (1024 * 1024)
            );
            q4
        } else if want_q4_requant {
            let q4 = crate::layer_view::requant_q8_0_to_q4_0(&lm_head_q8_src)?;
            eprintln!(
                "engine_core: DS4_Q4_LMHEAD — lm_head Q8_0→Q4_0 ({} MB → {} MB) at load",
                lm_head_q8_src.len() / (1024 * 1024),
                q4.len() / (1024 * 1024),
            );
            q4
        } else {
            Vec::new()
        };
        let q4_active = !lm_head_q4.is_empty();
        // Keep the q8 bytes only when q8 is the live tail (no q4 in play).
        let lm_head_q8 = if want_q4_requant {
            std::borrow::Cow::Borrowed(&[][..])
        } else {
            lm_head_q8_src
        };

        // f32 lm_head: needed only by full-logits/sampling + the CPU reference.
        // Dropped (~2 GB) when q4 is active, or under DS4_LOW_RAM on a q8 model
        // (the greedy q8 tail reads the q8 bytes directly). A q4-on-disk model
        // has no f32 to dequant anyway. Those paths assert with a pointer here.
        let low_ram = std::env::var("DS4_LOW_RAM").ok().as_deref() == Some("1");
        // Lean: drop the f32 lm_head when the q8 full-logits tail is active
        // (matches free_dead_f32_weights / the server tail's q8 condition).
        let lean_lmhead = lean
            && lm_head_is_q8
            && !want_q4_requant
            && std::env::var("DS4_Q8_LMHEAD").ok().as_deref() != Some("0")
            && std::env::var("DS4_Q8_0_ACT").ok().as_deref() != Some("1");
        // q4-on-disk lm_head: the q4 full-logits tail (`tail_lm_head_full_q4`)
        // reads the ~281 MB q4 bytes directly, so we DROP the ~2 GB f32 dequant.
        // The q4 matvec is the model's NATIVE output-head computation (the f32
        // we used to dequant was itself dequant(q4)) — coherence-equivalent, no
        // quality loss. (Previously kept ~2 GB f32 here because the full-logits
        // tail only had f32/q8 paths.) Like the q8 drop,
        // this disables the CPU-reference decode (which needs the f32), but the
        // server uses the encoder tail, not the CPU reference. DS4_Q4_LMHEAD_FULL=0
        // is the escape hatch: it forces the f32 tail, so we must KEEP the f32 then
        // (matches the encoder's `use_q4` gate on the same env var).
        let q4_full_tail =
            q4_active && std::env::var("DS4_Q4_LMHEAD_FULL").ok().as_deref() != Some("0");
        let lm_head = if q4_full_tail || (low_ram && lm_head_is_q8) || lean_lmhead {
            if lm_head_is_q8 {
                let saved_mb = lm_head_h.n_elems().saturating_mul(4) / (1024 * 1024);
                eprintln!(
                    "engine_core: dropped f32 lm_head (~{saved_mb} MB); \
                     greedy quantized tail only (full-logits/CPU-reference disabled)"
                );
            } else if q4_active {
                let saved_mb = lm_head_q4.len() / 18 * 32 * 4 / (1024 * 1024);
                eprintln!(
                    "engine_core: q4 lm_head — dropped f32 dequant (~{saved_mb} MB); \
                     q4 full-logits tail serves logits ({} MB q4 resident)",
                    lm_head_q4.len() / (1024 * 1024)
                );
            }
            Vec::new()
        } else if q4_active {
            // DS4_Q4_LMHEAD_FULL=0 escape hatch: dequant q4→f32 (~2 GB) so the f32
            // full-logits tail has its weight. dequant_f32_simple bails on Q4_0.
            let nb = lm_head_q4.len() / 18;
            let mut f = Vec::with_capacity(nb * 32);
            for ib in 0..nb {
                f.extend_from_slice(&crate::layer_view::dequant_q4_0_block(
                    &lm_head_q4[ib * 18..(ib + 1) * 18],
                ));
            }
            eprintln!(
                "engine_core: DS4_Q4_LMHEAD_FULL=0 — kept f32 dequant of q4 lm_head (~{} MB)",
                f.len() * 4 / (1024 * 1024)
            );
            f
        } else {
            src.dequant_f32(lm_head_h)?
        };

        // Output-side HC fold weights (antirez ds4.c:7654-7681 `output_hc_head_one`).
        // All three must be present together; missing any is a manifest bug.
        let output_hc_base = match globals.get("output_hc_base") {
            Some(h) => Some(src.dequant_f32(h)?),
            None => None,
        };
        let output_hc_fn = match globals.get("output_hc_fn") {
            Some(h) => Some(src.dequant_f32(h)?),
            None => None,
        };
        let output_hc_scale = match globals.get("output_hc_scale") {
            Some(h) => Some(src.dequant_f32(h)?),
            None => None,
        };

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let view = src.layer(i as u32)?;
            let mut params = crate::attn_dispatch::LayerParams::from_gguf(
                &g,
                manifest.d_model,
                i as u32,
                defaults,
            )?;
            if view.is_compressed() {
                params.compress_ratio = if view.uses_indexer() { 4 } else { 128 };
                // Antirez ds4.c:4596-4607 — compressed layers use a
                // different RoPE base (160000 vs 10000), a freq_scale of
                // 1/DS4_ROPE_SCALE_FACTOR (= 1/16), YaRN ext_factor=1.0,
                // and an attn_factor scaled by 1/(1+0.1·log(SCALE)).
                params.rope_freq_base = 160000.0;
                params.rope_freq_scale = 1.0 / 16.0;
                params.rope_ext_factor = 1.0;
                params.rope_attn_factor = 1.0 / (1.0 + 0.1 * (16.0_f32).ln());
                params.rope_orig_ctx = 65536;
            } else {
                // Layers 0-1 (uncompressed): freq_scale = 1.0 explicitly,
                // freq_base = DS4_ROPE_FREQ_BASE (10000), ext_factor = 0.
                params.rope_freq_base = 10000.0;
                params.rope_freq_scale = 1.0;
                params.rope_ext_factor = 0.0;
                params.rope_attn_factor = 1.0;
            }
            let shared_dim = crate::gguf::meta_u32(&g, &["deepseek4.expert_feed_forward_length"])
                .unwrap_or(params.d_embd);
            // Shared-expert SwiGLU clamp: use what the MODEL declares
            // (`deepseek4.swiglu_clamp_shexp`, per layer), else 0.0 = unclamped.
            //
            // The literal 7.0 that used to sit here was inert — every consumer
            // dropped the value (the CPU `shared_expert` took it as `_clamp`, the
            // GPU chain hardcoded `limit: 0.0`), so the effective clamp has always
            // been "none". Now that both consumers honour the parameter, passing
            // 7.0 for a model that never asked for it would silently change every
            // pre-MXFP4 DS4 result; 0.0 keeps them bit-identical.
            let shared_clamp = crate::gguf::ds4_shared_swiglu_clamp(&g)
                .and_then(|v| v.get(i as usize).copied())
                .unwrap_or(0.0);
            let attn = crate::attn_dispatch::AttnLayerWeights::from_layer_view(
                src, view.as_ref(), params, shared_dim, shared_clamp, lean,
            )?;

            // No-copy F16 router (`moe_router` is F16). Borrow only for NON-hash
            // layers (hash-routed layers 0/1/2 keep the f32 for `run_ffn_half`);
            // gated by DS4_COMPRESSOR_F16. The lean encoder reads it via
            // `encode_router_logits`'s `matvec_f16`. SAFETY: same mmap-borrow
            // invariant as the attn `raw_q8`/`raw_f16`.
            // DS4_FUSED_COMP: force f32 router on the ratio≠4 resident layers too
            // (the f16 router corrupts the resident path — the last f16 source after
            // hc/compressor; see attn_dispatch.rs from_layer_view). Tiny (router is
            // ~0.01 GB), keeps f16 router everywhere else.
            let comp_f16_on = std::env::var("DS4_COMPRESSOR_F16")
                .map(|v| v != "0")
                .unwrap_or(true)
                && !(std::env::var("DS4_FUSED_COMP").ok().as_deref() != Some("0")
                    && params.compress_ratio != 0
                    && params.compress_ratio != 4);
            let is_hash_layer = view.get("moe_routing_table").is_some();
            let w_router_f16: std::borrow::Cow<'static, [u8]> = match view.get("moe_router") {
                Some(h)
                    if comp_f16_on && !is_hash_layer && h.ttype == crate::gguf::GgmlType::F16 =>
                {
                    let s = src.bytes_for(h)?;
                    let s: &[u8] = &s;
                    let s: &'static [u8] =
                        unsafe { std::mem::transmute::<&[u8], &'static [u8]>(s) };
                    std::borrow::Cow::Borrowed(s)
                }
                _ => std::borrow::Cow::Borrowed(&[]),
            };
            // Lean: skip the f32 router dequant when the f16 is borrowed; skip the
            // MoE-side norm gammas entirely (duplicates of the attn-half gammas —
            // the lean encoder never reads moe.{attn,ffn}_norm_gamma; only the
            // trait/verifier path does, and that's non-lean).
            let skip_router_f32 = lean && !w_router_f16.is_empty();
            let ffn_norm_gamma = if lean {
                Vec::new()
            } else {
                match view.get("ffn_norm") {
                    Some(h) => src.dequant_f32(h)?,
                    None => vec![1.0f32; d_model],
                }
            };
            let w_router = if skip_router_f32 {
                Vec::new()
            } else {
                match view.get("moe_router") {
                    Some(h) => src.dequant_f32(h)?,
                    None => Vec::new(),
                }
            };
            let router_bias = match view.get("moe_router_bias") {
                Some(h) => {
                    let v = src.dequant_f32(h)?;
                    if i == 0 && std::env::var("DS4_DUMP_BIAS_LOAD").is_ok() {
                        let ss: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
                        let rms = (ss / v.len() as f64).sqrt();
                        eprintln!(
                            "BIAS_LOAD il=0 found! len={} rms={:.5} first10={:?}",
                            v.len(),
                            rms,
                            v.iter().take(10).collect::<Vec<_>>(),
                        );
                    }
                    v
                }
                None => {
                    if i == 0 && std::env::var("DS4_DUMP_BIAS_LOAD").is_ok() {
                        eprintln!("BIAS_LOAD il=0 MISSING moe_router_bias -> zeros");
                    }
                    vec![0.0f32; n_experts]
                }
            };

            // load hash routing table (`ffn_gate_tid2eid`) for layers
            // where it exists (layers 0/1/2 of DS4 V4 Flash). Antirez switches
            // on presence at ds4.c:5155-5157. Layout: row-major [vocab, k]
            // with k = n_experts_used, I32 element type.
            let routing_table = match view.get("moe_routing_table") {
                Some(h) if h.ttype == crate::gguf::GgmlType::I32 => {
                    let raw = src.bytes_for(h)?;
                    let n = h.n_elems() as usize;
                    if raw.len() != n * 4 {
                        bail!("routing_table il={}: byte size {} != 4*{}", i, raw.len(), n);
                    }
                    let mut tbl = vec![0i32; n];
                    for (j, c) in raw.chunks_exact(4).enumerate() {
                        tbl[j] = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    }
                    if i == 0 && std::env::var("DS4_DUMP_ROUTING_TABLE").is_ok() {
                        eprintln!(
                            "ROUTING_TABLE il=0 found! len={} first6={:?}",
                            tbl.len(),
                            tbl.iter().take(6).collect::<Vec<_>>(),
                        );
                    }
                    Some(tbl)
                }
                _ => None,
            };

            let moe = LayerWeights {
                d_model,
                d_ffn: None, // None means "read it from the quantised expert weights",
                // which is distinct from a 0 sentinel.
                n_experts,
                n_experts_used,
                // MoE-side attn norm gamma — a duplicate of the attn-half
                // `hc_norm_gamma` only the trait/verifier path reads. Lean skips.
                attn_norm_gamma: if lean {
                    Vec::new()
                } else {
                    attn.hc_norm_gamma.clone()
                },
                w_attn: Vec::new(),
                ffn_norm_gamma,
                w_router,
                w_router_f16,
                router_bias,
                routing_table,
                w_gate_exps: Vec::new(),
                w_up_exps: Vec::new(),
                w_down_exps: Vec::new(),
            };
            layers.push(ComposedLayerWeights { attn, moe });
        }

        Ok(Self {
            layers,
            final_norm_gamma,
            lm_head,
            lm_head_q8,
            lm_head_q4,
            vocab_size,
            d_model,
            output_hc_base,
            output_hc_fn,
            output_hc_scale,
        })
    }
}

/// Per-step state threaded across layers (HC residual + KV cache + position).
/// Built once per decode position; passed by `&mut` into
/// `decode_step_with_attn`.
///
/// Note: the HC sinkhorn `hc_split` buffer is NOT carried across layers —
/// it is recomputed inside every `hc_collapse_norm` from the projection
/// weight, matching antirez `ds4.c:8727`.
#[derive(Clone)]
pub struct AttnStepState {
    /// HC residual — flat `[n_hc * d_embd]`. Initialized to the input embd
    /// expanded across HC slots, mutated layer-by-layer.
    pub cur_hc: Vec<f32>,
    /// KV cache storage — `[raw_cap, n_lora_kv]` per layer (== head_dim;
    /// rope tail at `[n_lora_kv-n_rot..n_lora_kv]` inside each row).
    /// Flattened: `kv_storage[layer_idx]` is one layer's cache.
    pub kv_storage: Vec<Vec<f32>>,
    /// Current write position into each layer's KV cache.
    pub kv_pos: Vec<u32>,
    /// Decode position (token index).
    pub pos: u32,
    /// Compressor `state_kv` per layer. Empty Vec for dense (ratio=0) layers.
    /// Shape: `[rows, width]` flat, where `width = coff * head_dim`,
    /// `coff = ratio==4 ? 2 : 1`, `rows = ratio==4 ? 2*ratio : ratio`.
    /// See antirez `compressor_decode_one` (ds4.c:6208).
    pub comp_state_kv: Vec<Vec<f32>>,
    /// Compressor `state_score` per layer — same shape as `comp_state_kv`.
    /// Initialized to DS4_NEG_INF (≈ -1e9 in antirez) so the softmax
    /// over unset rows produces a zero pooled contribution.
    pub comp_state_score: Vec<Vec<f32>>,
    /// Ring of emitted compressed KV rows per layer. Each emit appends one
    /// row of `head_dim` floats. `n_comp[layer]` is the row count.
    pub comp_kv_ring: Vec<Vec<f32>>,
    /// Number of valid compressed rows in `comp_kv_ring[layer]`.
    pub n_comp: Vec<u32>,
    /// Indexer compressor `state_kv` per layer. Same shape rules as
    /// `comp_state_kv` but uses `head_dim = DS4_N_INDEXER_HEAD_DIM = 128`.
    /// Populated only on ratio==4 layers; empty Vec elsewhere.
    pub index_state_kv: Vec<Vec<f32>>,
    /// Indexer compressor `state_score` per layer. Same shape as
    /// `index_state_kv`; initialised to DS4_NEG_INF.
    pub index_state_score: Vec<Vec<f32>>,
    /// Ring of emitted indexer compressed KV rows per layer. Each
    /// emit appends `DS4_N_INDEXER_HEAD_DIM=128` floats.
    pub index_comp_kv_ring: Vec<Vec<f32>>,
    /// Number of valid indexer compressed rows in `index_comp_kv_ring[layer]`.
    pub n_index_comp: Vec<u32>,
    /// How many indexer ring rows have been mirrored into the GPU-resident
    /// `index_comp_ring` buffer (the accelerator host crate, DS4_GPU_INDEXER path). Lazily synced
    /// from `index_comp_kv_ring` (the source of truth, fed by all emit paths);
    /// 0 for the CPU path. Per-sequence, so it resets with the state.
    pub index_comp_ring_gpu_rows: Vec<u32>,
}

impl AttnStepState {
    /// A fully-empty state (no per-layer mirrors). Only useful as a
    /// placeholder for the legacy `BatchSeqState::new(slot)` (raw-flash-only)
    /// path; the comp-aware batched decoder must use
    /// `AttnStepState::new(model, raw_cap)` so the per-layer compressor mirrors
    /// exist. All Vecs are empty / `pos = 0`.
    pub fn empty() -> Self {
        Self {
            cur_hc: Vec::new(),
            kv_storage: Vec::new(),
            kv_pos: Vec::new(),
            pos: 0,
            comp_state_kv: Vec::new(),
            comp_state_score: Vec::new(),
            comp_kv_ring: Vec::new(),
            n_comp: Vec::new(),
            index_state_kv: Vec::new(),
            index_state_score: Vec::new(),
            index_comp_kv_ring: Vec::new(),
            n_index_comp: Vec::new(),
            index_comp_ring_gpu_rows: Vec::new(),
        }
    }

    /// Initialise state for a fresh decode (`pos = 0`, empty KV).
    pub fn new(model: &ComposedModelWeights, raw_cap: u32) -> Self {
        let n_layers = model.layers.len();
        let mut kv_storage = Vec::with_capacity(n_layers);
        let mut comp_state_kv = Vec::with_capacity(n_layers);
        let mut comp_state_score = Vec::with_capacity(n_layers);
        let mut comp_kv_ring = Vec::with_capacity(n_layers);
        let mut index_state_kv = Vec::with_capacity(n_layers);
        let mut index_state_score = Vec::with_capacity(n_layers);
        let mut index_comp_kv_ring = Vec::with_capacity(n_layers);
        // Indexer compressor uses head_dim = DS4_N_INDEXER_HEAD_DIM (= 128),
        // independent of the layer's attention head_dim (typically 512).
        const IDX_HEAD_DIM: usize = crate::attn_dispatch::DS4_N_INDEXER_HEAD_DIM as usize;
        for layer in &model.layers {
            let p = &layer.attn.params;
            let row = p.n_lora_kv as usize;
            kv_storage.push(vec![0.0f32; row * raw_cap as usize]);

            let ratio = p.compress_ratio;
            if ratio == 0 {
                comp_state_kv.push(Vec::new());
                comp_state_score.push(Vec::new());
                comp_kv_ring.push(Vec::new());
                index_state_kv.push(Vec::new());
                index_state_score.push(Vec::new());
                index_comp_kv_ring.push(Vec::new());
            } else {
                let coff = if ratio == 4 { 2 } else { 1 } as usize;
                let width = coff * p.head_dim as usize;
                let rows = if ratio == 4 { 2 * ratio } else { ratio } as usize;
                comp_state_kv.push(vec![0.0f32; rows * width]);
                // antirez seeds state_score = DS4_NEG_INF so any not-yet-written
                // row contributes ~0 to the pooled softmax.
                comp_state_score.push(vec![-1.0e9f32; rows * width]);
                // Ring capacity is bounded by raw_cap / ratio in practice;
                // start empty and grow on emit.
                comp_kv_ring.push(Vec::new());
                // Indexer compressor: only ratio==4 layers have indexer
                // weights. On other compressed layers we still
                // allocate empty Vecs so the per-layer index keeps aligning
                // with comp_state_kv.
                if ratio == 4 {
                    let idx_width = coff * IDX_HEAD_DIM;
                    index_state_kv.push(vec![0.0f32; rows * idx_width]);
                    index_state_score.push(vec![-1.0e9f32; rows * idx_width]);
                    index_comp_kv_ring.push(Vec::new());
                } else {
                    index_state_kv.push(Vec::new());
                    index_state_score.push(Vec::new());
                    index_comp_kv_ring.push(Vec::new());
                }
            }
        }
        let first_layer = &model.layers[0].attn.params;
        let cur_hc = vec![0.0f32; first_layer.hc_dim()];
        Self {
            cur_hc,
            kv_storage,
            kv_pos: vec![0; n_layers],
            pos: 0,
            comp_state_kv,
            comp_state_score,
            comp_kv_ring,
            n_comp: vec![0; n_layers],
            index_state_kv,
            index_state_score,
            index_comp_kv_ring,
            n_index_comp: vec![0; n_layers],
            index_comp_ring_gpu_rows: vec![0; n_layers],
        }
    }
}

/// Composed decode step: drives every layer through both traits, then
/// applies final norm + lm_head matvec.
///
/// Per-layer order (matches antirez `metal_graph_encode_decode_layer`):
///   1. Attention prefix (steps 1–8 of `decode_attn_layer_with`).
///   2. MoE/router half on `normed`: rms with ffn_norm_gamma → router
///      matvec → softplus_sqrt → router_finalize → moe_routed_step.
///   3. Attention suffix (steps 9–10): shared expert + add routed +
///      HC expand.
///
/// Returns logits over `vocab_size`.
pub fn decode_step_with_attn<K, A>(
    k: &K,
    a: &A,
    x: Vec<f32>,
    model: &ComposedModelWeights,
    state: &mut AttnStepState,
    cfg: &DecodeConfig,
    raw_cap: u32,
) -> Result<Vec<f32>>
where
    K: KernelDispatcher,
    A: AttentionDispatcher,
{
    // factor decode into "produce final residual"
    // + "tail (rms_norm + q8_0? + lm_head matvec)". The SingleBufferEncoder
    // reuses the residual half and replaces the tail with
    // a single-`MTLCommandBuffer` `tail_lm_head_batched` call. The trait
    // path keeps its byte-for-byte semantics (residual tap, q8_0 gate,
    // state.pos bump order all unchanged).
    let final_hidden = decode_step_with_attn_to_residual(k, a, x, model, state, cfg, raw_cap)?;
    let normed = k.rms_norm(&final_hidden, &model.final_norm_gamma, cfg.eps_rms);

    // residual taps before the final lm_head matvec.
    // Set DS4_DUMP_RESIDUAL_TAPS=POS to dump cur_hc / final_hidden / normed
    // + their top-10 abs-value dims when state.pos == POS. Always-cheap reduction.
    //
    // NB: by the time we get here state.pos has already been bumped by the
    // residual helper, so `state.pos - 1` is the position of the residual.
    let tap_pos = state.pos.wrapping_sub(1);
    if let Ok(target_str) = std::env::var("DS4_DUMP_RESIDUAL_TAPS") {
        if let Ok(target_pos) = target_str.parse::<u32>() {
            if tap_pos == target_pos {
                dump_residual_tap("cur_hc", &state.cur_hc, target_pos);
                dump_residual_tap("final_hidden", &final_hidden, target_pos);
                dump_residual_tap("normed", &normed, target_pos);
                // Also dump the post-matvec logits' top-10 raw values
                // alongside the lm_head row that produced them — print
                // dim-sum contributions for the antirez-known token 939
                // versus our top pick (token 19) on a synthetic check.
                let logits_preview = k.matvec_f32(&model.lm_head, &normed, model.vocab_size);
                let mut ranked: Vec<(usize, f32)> =
                    logits_preview.iter().copied().enumerate().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                eprintln!(
                    "TAP_LOGITS pos={target_pos} top10 = {:?}",
                    &ranked[..10.min(ranked.len())]
                );
                if model.vocab_size > 939 {
                    let rank_939 = ranked
                        .iter()
                        .position(|&(i, _)| i == 939)
                        .unwrap_or(usize::MAX);
                    eprintln!(
                        "TAP_LOGITS pos={target_pos} token_939 value={:.4} rank={rank_939}",
                        logits_preview[939]
                    );
                }
            }
        }
    }

    // lm_head is Q8_0 in DS4 V4 Flash. Antirez quantizes the activation
    // (`normed`) to block_q8_0 before the int8-dot. Mirror that lossiness when
    // DS4_Q8_0_ACT=1 so the f32 oracle matches antirez's int8 ground truth.
    let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
    let normed_for_lm = if want_q80 {
        crate::forward::q8_0_round_trip(&normed)
    } else {
        normed
    };
    anyhow::ensure!(
        !model.lm_head.is_empty(),
        "f32 lm_head was dropped (DS4_LOW_RAM); this full-logits decode path needs it — unset DS4_LOW_RAM"
    );
    Ok(k.matvec_f32(&model.lm_head, &normed_for_lm, model.vocab_size))
}

/// Phase C.2. Run `decode_step_with_attn` *up to and including*
/// the post-layer-loop HC fold (`output_hc_head_one`), returning the
/// `[d_embd]` residual. `state.pos` is bumped exactly once, the same way
/// `decode_step_with_attn` bumps it. The caller is responsible for the
/// final `rms_norm + q8_0? + lm_head matvec`.
///
/// Used by:
///   * `decode_step_with_attn` (trait path) — does the tail through
///     `k.rms_norm` + `k.matvec_f32` sequentially.
///   * `SingleBufferEncoder::decode_token_batched` — does the tail
///     through `tail_lm_head_batched`, packing the three ops into one
///     `MTLCommandBuffer`.
///
/// Byte-equivalent decode through either caller — `tail_lm_head_batched`
/// is bit-identical to the sequential trait calls ( smoke).
pub fn decode_step_with_attn_to_residual<K, A>(
    k: &K,
    a: &A,
    x: Vec<f32>,
    model: &ComposedModelWeights,
    state: &mut AttnStepState,
    cfg: &DecodeConfig,
    raw_cap: u32,
) -> Result<Vec<f32>>
where
    K: KernelDispatcher,
    A: AttentionDispatcher,
{
    if x.len() != model.d_model {
        bail!(
            "decode_step_with_attn: x has {} elements, expected d_model={}",
            x.len(),
            model.d_model
        );
    }

    // publish state.pos so debug probes inside the dispatcher (which
    // does not have direct access to `state`) can filter on it.
    crate::attn_dispatch::CURRENT_POS_HINT.with(|c| c.set(state.pos as u32));

    // Expand input embd into HC residual: every HC slot gets a copy of x.
    let first = &model.layers[0].attn.params;
    let n_hc = first.n_hc as usize;
    let d_embd = first.d_embd as usize;
    // DS4 residual is `n_hc` slots × `d_embd`, where `d_embd == d_model`
    // (every HC slot gets a full copy of the token embedding — see
    // antirez `ds4.c:4198`).
    if d_embd != model.d_model {
        bail!(
            "d_embd ({}) != model.d_model ({}); composed model is mis-shaped",
            d_embd,
            model.d_model
        );
    }
    let mut cur_hc = vec![0.0f32; n_hc * d_embd];
    for h in 0..n_hc {
        cur_hc[h * d_embd..(h + 1) * d_embd].copy_from_slice(&x);
    }
    state.cur_hc = cur_hc;

    // n_raw grows by 1 every step; raw_start is 0 (no SWA yet — Phase B
    // doesn't drive sliding-window logic).
    let n_raw = (state.pos + 1).min(raw_cap);
    let raw_start = 0u32;

    // DS4_DECODE_N_LAYERS truncates the layer loop for verifier-bisection
    // (compare cur_hc after L layers vs the K-position verifier at the same L).
    // Numerics/logits are invalid when truncated — diagnostic only.
    let decode_n_layers: usize = std::env::var("DS4_DECODE_N_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(model.layers.len())
        .min(model.layers.len());
    // DS4_DISABLE_COMPRESSOR skips the compressor + indexer attention paths so
    // decode_step does PURE raw SWA attention — matching the K-position
    // verifier (flash_attn_k_mla, raw only). Used to test whether the verifier's
    // divergence is the missing compressor/indexer. Diagnostic only.
    let disable_comp = crate::env_flags::flag_enabled("DS4_DISABLE_COMPRESSOR", false);

    // DSpark target-layer tap (gated, default-off). Snapshot the armed target set
    // ONCE before the loop so the per-layer capture only runs when armed — the
    // normal decode path pays a single bool/clone read here, nothing per layer.
    let dspark_targets: Option<Vec<usize>> = DSPARK_TAP.with(|c| {
        let t = c.borrow();
        t.armed.then(|| t.targets.clone())
    });

    for (layer_idx, layer) in model.layers.iter().enumerate().take(decode_n_layers) {
        // Verifier-bisection: publish the current layer for the flash tap.
        crate::attn_dispatch::CURRENT_LAYER_HINT.with(|h| h.set(layer_idx));
        let p = &layer.attn.params;
        let n_lora_q = p.n_lora_q as usize;
        let kv_row = p.n_lora_kv as usize;
        let q_dim = p.q_dim();

        // real Q/K/V projections per antirez `ds4.c:4452-4497`.
        //
        // 1. normed_x = hc_collapse_norm(cur_hc).normed  — single d_embd vector.
        // 2. qr        = attn_q_a · normed_x             — [n_lora_q].
        // 3. qr_normed = rms_norm(qr, qkv_gamma_q)       — [n_lora_q].
        // 4. q_heads   = attn_q_b · qr_normed            — [q_dim].
        // 5. kv_raw    = attn_kv  · normed_x             — [n_lora_kv].
        //
        // The prefix re-runs hc_collapse_norm and the qr norm internally
        // (redundantly) so the existing pipeline downstream is unchanged.
        // When `attn_q_a` is empty (legacy synthetic fixtures), fall back
        // to the Phase-B1 identity stub so the test suite keeps passing.
        let use_real_proj = !layer.attn.attn_q_a.is_empty()
            && !layer.attn.attn_q_b.is_empty()
            && !layer.attn.attn_kv.is_empty();
        let mut normed_for_comp: Option<Vec<f32>> = None;
        // capture the LoRA Q rms_normed value for the indexer.
        // Antirez `layer_attention_raw_swa_one` (ds4.c:6788, 6834-6838) feeds
        // `qr_norm` (== `rms_norm(qr, qkv_gamma_q)`) into the indexer; we
        // surface it here so the call site below doesn't re-rms_norm `qr`.
        let mut qr_norm_for_indexer: Option<Vec<f32>> = None;
        let (kv_raw_row, q_heads) = if use_real_proj {
            let (_collapsed, normed, _split) = crate::op_time!("hc_collapse_norm_entry", {
                a.hc_collapse_norm(
                    p,
                    crate::attn_dispatch::HcKind::Attn,
                    &layer.attn.hc_attn_fn,
                    &layer.attn.hc_attn_scale,
                    &layer.attn.hc_attn_base,
                    &state.cur_hc,
                    Some(&layer.attn.hc_norm_gamma),
                )
            });
            // Verifier-bisection tap: capture this layer's attn-half normed.
            NORMED_CAPTURE.with(|c| {
                let mut cc = c.borrow_mut();
                if cc.0 == layer_idx {
                    cc.1 = normed.clone();
                }
            });
            // antirez attn LoRA weights are Q8_0; mirror its
            // activation-side q8_0 rounding when DS4_Q8_0_ACT=1.
            let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
            let head_rms_f64 =
                std::env::var("DS4_HEAD_RMS_F64_FIDELITY").ok().as_deref() == Some("1");
            let n_head = p.n_head as usize;
            let head_dim = p.head_dim as usize;
            // when neither fidelity gate is on
            // (the default), the entire 5-op attention-half QKV chain
            // (matvec_qa → rms_norm → matvec_qb → head_rms_norm → matvec_kv)
            // is fused into ONE MTLCommandBuffer via
            // `k.attn_qkv_chain_batched`. Saves 1 commit+wait + 2 readbacks
            // per layer per token vs the previous `layer_qa_rms_batched +
            // qkv_b_head_rms_batched` pair. Trait default falls back to that
            // pair on CPU backends so semantics are unchanged.
            //
            // Q8_0 fidelity (DS4_Q8_0_ACT=1) and f64 head-rms fidelity
            // (DS4_HEAD_RMS_F64_FIDELITY=1) both insert CPU ops between the
            // GPU dispatches, so the chain can't fuse in those branches —
            // keep the legacy split path.
            let (qr_normed, q_heads, kv_raw_row) = if !want_q80 && !head_rms_f64 {
                crate::op_time!("attn_qkv_chain_batched", {
                    k.attn_qkv_chain_batched(
                        &normed,
                        &layer.attn.attn_q_a,
                        &layer.attn.qkv_gamma_q,
                        n_lora_q,
                        &layer.attn.attn_q_b,
                        n_head,
                        head_dim,
                        p.rms_eps,
                        &layer.attn.attn_kv,
                        kv_row,
                    )
                })
            } else {
                let normed_q = if want_q80 {
                    crate::forward::q8_0_round_trip(&normed)
                } else {
                    normed.clone()
                };
                let qr_normed = crate::op_time!("layer_qa_rms_batched", {
                    k.layer_qa_rms_batched(
                        &normed_q,
                        &layer.attn.attn_q_a,
                        &layer.attn.qkv_gamma_q,
                        n_lora_q,
                        p.rms_eps,
                    )
                });
                let qr_normed_q = if want_q80 {
                    crate::forward::q8_0_round_trip(&qr_normed)
                } else {
                    qr_normed.clone()
                };
                let normed_kv = if want_q80 {
                    crate::forward::q8_0_round_trip(&normed)
                } else {
                    normed.clone()
                };
                let (q_heads, kv_raw_row) = if head_rms_f64 {
                    let mut q_heads = k.matvec_f32(&layer.attn.attn_q_b, &qr_normed_q, q_dim);
                    for h in 0..n_head {
                        let chunk = &mut q_heads[h * head_dim..(h + 1) * head_dim];
                        let ss: f64 = chunk.iter().map(|&v| (v as f64) * (v as f64)).sum();
                        let scale = 1.0 / ((ss / head_dim as f64) as f32 + p.rms_eps).sqrt();
                        for v in chunk.iter_mut() {
                            *v *= scale;
                        }
                    }
                    let kv_raw_row = k.matvec_f32(&layer.attn.attn_kv, &normed_kv, kv_row);
                    (q_heads, kv_raw_row)
                } else {
                    crate::op_time!("qkv_b_head_rms_batched", {
                        k.qkv_b_head_rms_batched(
                            &qr_normed_q,
                            &layer.attn.attn_q_b,
                            q_dim,
                            n_head,
                            head_dim,
                            p.rms_eps,
                            &normed_kv,
                            &layer.attn.attn_kv,
                            kv_row,
                        )
                    })
                };
                (qr_normed, q_heads, kv_raw_row)
            };
            // surface qr_normed for the indexer call below.
            qr_norm_for_indexer = Some(qr_normed);
            normed_for_comp = Some(normed);
            (kv_raw_row, q_heads)
        } else {
            let h_row = &state.cur_hc[..d_embd];
            let kv_raw_row: Vec<f32> = (0..kv_row).map(|i| h_row[i % d_embd]).collect();
            let q_heads: Vec<f32> = (0..q_dim).map(|i| h_row[i % d_embd] * 0.5).collect();
            (kv_raw_row, q_heads)
        };

        // ── 1a. Compressor. For ratio∈{4,128} non-dense layers
        //       project `normed` through attn_compressor_{kv,gate,ape},
        //       update per-layer state, and on emit (every `ratio` steps)
        //       append a pooled+rms_normed+rope-tailed row to the comp ring.
        //       Antirez `compressor_decode_one` (ds4.c:6208) — called
        //       AFTER kv_fp8_store, but our prefix is opaque; we run the
        //       compressor BEFORE the prefix and feed it via `kv_comp_rows`.
        //       This is bit-equivalent because the compressor doesn't read
        //       the raw KV cache — only `normed` + its own state.
        if !disable_comp && p.compress_ratio != 0 && layer.attn.has_attn_compressor() {
            if let Some(normed_x) = normed_for_comp.as_deref() {
                let comp = crate::attn_dispatch::CompressorInputs {
                    w_kv: &layer.attn.attn_compressor_kv,
                    w_gate: &layer.attn.attn_compressor_gate,
                    // CPU trait path is non-lean — f32 weights are present.
                    w_kv_f16: None,
                    w_gate_f16: None,
                    w_ape: &layer.attn.attn_compressor_ape,
                    w_norm: &layer.attn.attn_compressor_norm,
                    head_dim: p.head_dim,
                    compress_ratio: p.compress_ratio,
                };
                let emit = crate::op_time!("compressor_decode_one", {
                    crate::attn_dispatch::compressor_decode_one(
                        a,
                        p,
                        &comp,
                        normed_x,
                        &mut state.comp_state_kv[layer_idx],
                        &mut state.comp_state_score[layer_idx],
                        state.pos,
                    )
                });
                if let Some(row) = emit {
                    state.comp_kv_ring[layer_idx].extend_from_slice(&row);
                    state.n_comp[layer_idx] += 1;
                }
            }
        }

        // ── 1b. Indexer. On ratio==4 layers, run a parallel
        //       indexer compressor over `indexer_compressor_{kv,gate,ape,
        //       norm}` (head_dim = DS4_N_INDEXER_HEAD_DIM = 128), then call
        //       `indexer_allowed_decode_one` with the resulting compressed
        //       ring + attn_norm + qr_norm to produce a top-k boolean mask
        //       over compressed rows. The mask is fed to `flash_attn_decode`
        //       via AttnStepInputs.comp_selected / n_selected.
        //
        //       Antirez `layer_attention_raw_swa_one` (ds4.c:6816-6838):
        //         if (ratio == 4) {
        //             compressor_decode_one(index_comp, model,
        //                 layer->indexer_compressor_{kv,gate,ape,norm},
        //                 attn_norm, cache->index_state_kv, ...);
        //             comp_allowed = indexer_allowed_decode_one(model, layer,
        //                 attn_norm, qr_norm, cache->index_comp_kv, ...);
        //         }
        let mut indexer_allowed: Option<Vec<u32>> = None;
        if !disable_comp
            && p.compress_ratio == 4
            && layer.attn.has_indexer_compressor()
            && layer.attn.has_indexer_qb()
        {
            if let (Some(normed_x), Some(qr_norm)) =
                (normed_for_comp.as_deref(), qr_norm_for_indexer.as_deref())
            {
                let idx_comp = crate::attn_dispatch::CompressorInputs {
                    w_kv: &layer.attn.indexer_compressor_kv,
                    w_gate: &layer.attn.indexer_compressor_gate,
                    // CPU trait path is non-lean — f32 weights are present.
                    w_kv_f16: None,
                    w_gate_f16: None,
                    w_ape: &layer.attn.indexer_compressor_ape,
                    w_norm: &layer.attn.indexer_compressor_norm,
                    head_dim: crate::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                    compress_ratio: p.compress_ratio,
                };
                let emit = crate::op_time!("indexer_compressor_decode_one", {
                    crate::attn_dispatch::compressor_decode_one(
                        a,
                        p,
                        &idx_comp,
                        normed_x,
                        &mut state.index_state_kv[layer_idx],
                        &mut state.index_state_score[layer_idx],
                        state.pos,
                    )
                });
                if let Some(row) = emit {
                    state.index_comp_kv_ring[layer_idx].extend_from_slice(&row);
                    state.n_index_comp[layer_idx] += 1;
                }
                let n_index_comp = state.n_index_comp[layer_idx];
                if n_index_comp > 0 {
                    let inputs = crate::attn_dispatch::IndexerInputs {
                        w_q_b: &layer.attn.indexer_attn_q_b,
                        w_proj: &layer.attn.indexer_proj,
                        index_comp_kv: &state.index_comp_kv_ring[layer_idx],
                        n_comp: n_index_comp,
                        n_head: crate::attn_dispatch::DS4_N_INDEXER_HEAD,
                        head_dim: crate::attn_dispatch::DS4_N_INDEXER_HEAD_DIM,
                        top_k: crate::attn_dispatch::DS4_N_INDEXER_TOP_K,
                    };
                    let (allowed, _n_sel) = crate::op_time!("indexer_allowed_decode_one", {
                        crate::attn_dispatch::indexer_allowed_decode_one(
                            a, p, &inputs, normed_x, qr_norm, state.pos,
                        )
                    });
                    // Convert allowed[c] bitmap → list of selected indices.
                    let mut selected: Vec<u32> = allowed
                        .iter()
                        .enumerate()
                        .filter_map(|(c, &a)| if a { Some(c as u32) } else { None })
                        .collect();
                    selected.shrink_to_fit();
                    indexer_allowed = Some(selected);
                }
            }
        }

        // ── 1. Attention prefix ─────────────────────────────────────────
        let prefix = {
            // Split-borrow `state` so kv_storage and comp_kv_ring can be
            // borrowed concurrently (one mut, one shared).
            let comp_rows_slice: Option<&[f32]> = if state.n_comp[layer_idx] > 0 {
                Some(&state.comp_kv_ring[layer_idx])
            } else {
                None
            };
            let n_comp = state.n_comp[layer_idx];
            let (comp_sel_slice, n_sel): (Option<&[u32]>, u32) =
                if let Some(ref sel) = indexer_allowed {
                    (Some(sel.as_slice()), sel.len() as u32)
                } else {
                    (None, 0)
                };
            let mut step = AttnStepInputs {
                cur_hc: &state.cur_hc,
                kv_raw_row: &kv_raw_row,
                q_heads: &q_heads,
                pos: state.pos,
                kv_view: KvCacheView {
                    raw: &mut state.kv_storage[layer_idx],
                    raw_cap,
                    pos: state.kv_pos[layer_idx],
                },
                n_raw,
                raw_start,
                routed_out: &[],
                kv_comp_rows: comp_rows_slice,
                n_comp,
                comp_selected: comp_sel_slice,
                n_selected: n_sel,
            };
            let prefix = crate::op_time!("decode_attn_prefix", {
                decode_attn_prefix_with(a, &layer.attn, &mut step)
            });
            state.kv_pos[layer_idx] = step.kv_view.pos;
            prefix
        };

        // ── 2a. FFN-half hc_collapse_norm → ffn_normed ( fix). ──
        // Antirez (ds4.c:9233-9302) feeds the FFN-half `_ffn_norm` (= our
        // `ffn_normed`) directly into router + MoE + shared_expert. The
        // earlier driver fed `rms_norm(prefix.normed, ffn_norm_gamma)` which
        // is the attention-half input — wrong vector at every layer.
        let ffn_pre = crate::op_time!("decode_attn_ffn_pre", {
            decode_attn_ffn_pre_with(a, &layer.attn, &prefix)
        });

        // ── 2b. MoE/router half on `ffn_normed`. ───────────────────────
        let h_norm = &ffn_pre.ffn_normed;
        // C.3e: one batched call replaces
        // `matvec(w_router) → softplus_sqrt`. On Metal this packs both
        // ops into one MTLCommandBuffer with a single readback (saves
        // 1 cwr/layer = 43 cwr/token). Bit-identical to the sequential
        // path on every backend.
        let probs = crate::op_time!("router_logits_batched", {
            k.router_logits_batched(&layer.moe.w_router, h_norm, layer.moe.n_experts)
        });
        // hash-mode routing for layers where `ffn_gate_tid2eid` is
        // present (DS4 V4 Flash layers 0/1/2). Antirez ds4.c:5155-5157 picks
        // `selected[i] = routing_table[token_id*k + i]` and derives weights
        // from probs only (no bias). All other layers fall through to the
        // standard top-k+bias path.
        let (selected, weights) = if let Some(table) = layer.moe.routing_table.as_ref() {
            let k_used = layer.moe.n_experts_used;
            let token_id = crate::attn_dispatch::CURRENT_TOKEN_HINT.with(|c| c.get()) as usize;
            let base = token_id.saturating_mul(k_used);
            let end = base.saturating_add(k_used);
            if end > table.len() {
                bail!(
                    "routing_table il={layer_idx}: token_id={} k={} oob (len={})",
                    token_id,
                    k_used,
                    table.len(),
                );
            }
            let sel: Vec<usize> = table[base..end].iter().map(|&v| v as usize).collect();
            if std::env::var("DS4_DUMP_HASH_ROUTING").is_ok() {
                let target_layer = std::env::var("DS4_DUMP_ROUTER_LAYER")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                if layer_idx == target_layer {
                    eprintln!(
                        "HASH_ROUTE il={layer_idx} pos={} token_id={} sel={:?}",
                        state.pos, token_id, sel,
                    );
                }
            }
            let w = crate::moe::hash_router_weights_from_probs(&probs, &sel, k_used);
            (sel, w)
        } else {
            crate::op_time!("router_finalize", {
                k.router_finalize(&probs, &layer.moe.router_bias, layer.moe.n_experts_used)
            })
        };
        // follow-up: router stage dump for L0 pos bisect.
        // DS4_DUMP_ROUTER=POS — fires at incoming `state.pos == POS` and only
        // for the layer specified in DS4_DUMP_ROUTER_LAYER (default 0).
        {
            let target_pos = std::env::var("DS4_DUMP_ROUTER")
                .ok()
                .and_then(|s| s.parse::<u32>().ok());
            if let Some(p) = target_pos {
                let target_layer = std::env::var("DS4_DUMP_ROUTER_LAYER")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                if state.pos == p && layer_idx == target_layer {
                    let rms_of = |v: &[f32]| -> f64 {
                        let ss: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
                        (ss / v.len() as f64).sqrt()
                    };
                    let top5 = |v: &[f32]| -> Vec<(usize, f32)> {
                        let mut r: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
                        r.sort_by(|a, b| {
                            b.1.abs()
                                .partial_cmp(&a.1.abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        r.into_iter().take(5).collect()
                    };
                    // C.3e: the production path goes through
                    // `router_logits_batched` and only returns `probs`.
                    // Recompute pre-softplus logits here just for the dump.
                    let logits = k.matvec_f32(&layer.moe.w_router, h_norm, layer.moe.n_experts);
                    eprintln!(
                        "ROUTER_LOGITS il={layer_idx} pos={} n={} rms={:.5} top5={:?}",
                        state.pos,
                        logits.len(),
                        rms_of(&logits),
                        top5(&logits),
                    );
                    eprintln!(
                        "ROUTER_PROBS il={layer_idx} pos={} n={} rms={:.5} top5={:?}",
                        state.pos,
                        probs.len(),
                        rms_of(&probs),
                        top5(&probs),
                    );
                    eprintln!(
                        "ROUTER_SELECTED il={layer_idx} pos={} {:?}",
                        state.pos, selected,
                    );
                    eprintln!(
                        "ROUTER_WEIGHTS il={layer_idx} pos={} {:?}",
                        state.pos, weights,
                    );
                    // Bias top-10 by abs value + values at antirez-selected positions.
                    let bias = &layer.moe.router_bias;
                    let bias_top10 = {
                        let mut r: Vec<(usize, f32)> = bias.iter().copied().enumerate().collect();
                        r.sort_by(|a, b| {
                            b.1.abs()
                                .partial_cmp(&a.1.abs())
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        r.into_iter().take(10).collect::<Vec<_>>()
                    };
                    eprintln!(
                        "ROUTER_BIAS il={layer_idx} pos={} n={} rms={:.5} top10={:?}",
                        state.pos,
                        bias.len(),
                        rms_of(bias),
                        bias_top10,
                    );
                    // Bias values at antirez's selected indices for L0 pos=27.
                    let antirez_sel: [usize; 6] = [192, 13, 123, 217, 147, 67];
                    let bias_at_antirez: Vec<(usize, f32)> = antirez_sel
                        .iter()
                        .map(|&i| (i, bias.get(i).copied().unwrap_or(0.0)))
                        .collect();
                    eprintln!(
                        "ROUTER_BIAS_AT_ANTIREZ_SEL il={layer_idx} pos={} {:?}",
                        state.pos, bias_at_antirez,
                    );
                    let bias_at_ours: Vec<(usize, f32)> = selected
                        .iter()
                        .map(|&i| (i, bias.get(i).copied().unwrap_or(0.0)))
                        .collect();
                    eprintln!(
                        "ROUTER_BIAS_AT_OURS_SEL il={layer_idx} pos={} {:?}",
                        state.pos, bias_at_ours,
                    );
                }
            }
        }
        // B Slice 5-redo: when not in
        // DS4_SILU_FIDELITY=1 mode, fuse `moe_routed_step` and
        // `shared_chain_batched` into ONE MTLCommandBuffer on the Metal
        // backend (saves 1 wait/layer ~25-100 ms — see
        // project_m23_path_c_a_batchscope memory). Fidelity mode keeps
        // the legacy path because it uses `shared_expert` +
        // `shared_down_hc_expand_add` instead of `shared_chain_batched`.
        let silu_fidelity = std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
        let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
        let (routed_out, precomputed_shared_out): (Vec<f32>, Option<Vec<f32>>) = if silu_fidelity {
            let r = crate::op_time!("moe_routed_step", {
                k.moe_routed_step(
                    layer_idx as u32,
                    h_norm,
                    &selected,
                    &weights,
                    &layer.moe.w_gate_exps,
                    &layer.moe.w_up_exps,
                    &layer.moe.w_down_exps,
                    layer.moe.d_ffn.map_or(0, |n| n.get()),
                )
            });
            (r, None)
        } else {
            let ffn_in_owned;
            let ffn_in: &[f32] = if want_q80 {
                ffn_in_owned = crate::forward::q8_0_round_trip(&ffn_pre.ffn_normed);
                &ffn_in_owned
            } else {
                &ffn_pre.ffn_normed
            };
            let (r, s) = crate::op_time!("moe_and_shared_chain_batched", {
                k.moe_and_shared_chain_batched(
                    layer_idx as u32,
                    h_norm,
                    &selected,
                    &weights,
                    &layer.moe.w_gate_exps,
                    &layer.moe.w_up_exps,
                    &layer.moe.w_down_exps,
                    layer.moe.d_ffn.map_or(0, |n| n.get()),
                    ffn_in,
                    &layer.attn.w_shared_gate,
                    &layer.attn.w_shared_up,
                    &layer.attn.w_shared_down,
                    layer.attn.shared_dim,
                    want_q80,
                )
            });
            (r, Some(s))
        };

        // per-layer routed_out + ffn_normed RMS at a target pos.
        // Set DS4_DUMP_MOE_TAPS=POS to fire at state.pos == POS.
        if let Ok(t) = std::env::var("DS4_DUMP_MOE_TAPS") {
            if let Ok(target) = t.parse::<u32>() {
                if state.pos == target {
                    let rms = |v: &[f32]| -> f64 {
                        let ss: f64 = v.iter().map(|&x| (x as f64).powi(2)).sum();
                        (ss / v.len() as f64).sqrt()
                    };
                    eprintln!(
                        "MOE_TAP il={layer_idx} pos={} ffn_normed_rms={:.5} routed_out_rms={:.5}",
                        state.pos,
                        rms(h_norm),
                        rms(&routed_out),
                    );
                    if layer_idx == 0 {
                        let r8: Vec<f32> = routed_out.iter().take(8).copied().collect();
                        let n8: Vec<f32> = h_norm.iter().take(8).copied().collect();
                        eprintln!("MOE_TAP_FIRST8 il=0 routed_out={:?}", r8);
                        eprintln!("MOE_TAP_FIRST8 il=0 ffn_normed={:?}", n8);
                    }
                }
            }
        }

        // ── 3. Attention suffix (shared expert + HC expand) ────────────
        // Uses ffn_pre (NOT a fresh hc_collapse_norm) so shared_expert sees
        // the exact same `ffn_normed` we fed to MoE.
        // DS4_DISABLE_FFN: skip the FFN/MoE contribution so cur_hc = attn-only
        // (= prefix.after_attn_hc). Used to split attn-half vs ffn-half when
        // bisecting the verifier — diagnostic only.
        let after_ffn_hc = if crate::env_flags::flag_enabled("DS4_DISABLE_FFN", false) {
            prefix.after_attn_hc.clone()
        } else {
            crate::op_time!("decode_attn_ffn_post", {
                decode_attn_ffn_post_with(
                    k,
                    a,
                    &layer.attn,
                    &prefix,
                    &ffn_pre,
                    &routed_out,
                    state.pos,
                    precomputed_shared_out.as_deref(),
                )
            })
        };

        // Verifier-bisection tap: capture this layer's after_attn / after_ffn.
        LAYER_RESID_CAPTURE.with(|c| {
            let mut cc = c.borrow_mut();
            if cc.0 == layer_idx {
                cc.1 = prefix.after_attn_hc.clone();
                cc.2 = after_ffn_hc.clone();
            }
        });

        // CLEAN per-layer output capture (probe): push this layer's output
        // residual for a non-drain oracle reference (see ORACLE_RESID_CAP).
        ORACLE_RESID_CAP.with(|c| {
            if let Some(v) = c.borrow_mut().as_mut() {
                v.push(after_ffn_hc.clone());
            }
        });

        // Update threaded state for the next layer. hc_split is not
        // carried — antirez recomputes it per layer inside hc_collapse_norm.
        state.cur_hc = after_ffn_hc;

        // DSpark target-layer tap: capture the hc-mean-collapsed `[d]` residual
        // AFTER this layer (= `state.cur_hc`, the output residual of `layer_idx`)
        // when it is one of the DSpark target layers. Runs only when armed.
        if let Some(targets) = dspark_targets.as_ref() {
            if let Some(slot) = targets.iter().position(|&l| l == layer_idx) {
                let collapsed = hc_mean_collapse(&state.cur_hc, n_hc, d_embd);
                DSPARK_TAP.with(|c| {
                    let mut t = c.borrow_mut();
                    if slot < t.captured.len() {
                        t.captured[slot] = Some(collapsed);
                    }
                });
            }
        }

        // Optional per-layer residual diagnostics (env-gated). Prints
        // RMS / min / max / mean of `state.cur_hc` so we can spot anomalous
        // growth or collapse during pos=N bisects. Cheap (one full reduction
        // per layer × 43 layers; ~14M elements once); only on at debug time.
        //
        // DS4_DUMP_HC_RMS=1         — fire at every layer × every pos (noisy)
        // DS4_DUMP_HC_RMS=POS (>=2) — fire only when state.pos == POS
        let hc_rms_fire = match std::env::var("DS4_DUMP_HC_RMS").ok() {
            Some(s) if s == "1" => true,
            Some(s) => s.parse::<u32>().ok().is_some_and(|p| p == state.pos),
            None => false,
        };
        if hc_rms_fire {
            let v = &state.cur_hc;
            let n = v.len() as f64;
            let mut s = 0.0f64;
            let mut ss = 0.0f64;
            let mut lmin = f32::INFINITY;
            let mut lmax = f32::NEG_INFINITY;
            for &x in v {
                let xd = x as f64;
                s += xd;
                ss += xd * xd;
                if x < lmin {
                    lmin = x;
                }
                if x > lmax {
                    lmax = x;
                }
            }
            let mean = s / n;
            let rms = (ss / n).sqrt();
            eprintln!(
                "DUMP_HC layer={layer_idx} pos={} rms={:.4} mean={:.4} min={:.4} max={:.4}",
                state.pos, rms, mean, lmin, lmax,
            );
        }

        // per-slot RMS / top-3 abs / top-3 dims at a target pos.
        // Set DS4_DUMP_HC_PER_SLOT=POS to fire only at state.pos == POS.
        // Goal: find the first layer where one slot's magnitude diverges,
        // pinpointing the kernel responsible for the pos=3 logit drift.
        //
        // in addition to the stderr dump, push each sample into
        // the thread-local `HC_SLOT_RECORDER` when armed. Lets Linux
        // tests assert against captured antirez fixtures without parsing
        // stderr. The env var still drives BOTH paths (recorder is a
        // no-op when unarmed).
        let dump_target_pos: Option<u32> = std::env::var("DS4_DUMP_HC_PER_SLOT")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());
        let recorder_armed = HC_SLOT_RECORDER.with(|r| r.borrow().is_some());
        if dump_target_pos.is_some() || recorder_armed {
            // The recorder fires at the same `state.pos == POS` gate the
            // stderr dump uses. If the recorder is armed but the env var
            // is unset, default the gate to `state.pos` itself so the
            // caller can target any pos by just stepping decode to it.
            let target_pos = dump_target_pos.unwrap_or(state.pos);
            if state.pos == target_pos {
                let n_hc_local = layer.attn.params.n_hc as usize;
                let d_embd_local = layer.attn.params.d_embd as usize;
                for h in 0..n_hc_local {
                    let slot = &state.cur_hc[h * d_embd_local..(h + 1) * d_embd_local];
                    let mut ss = 0.0f64;
                    for &x in slot {
                        ss += (x as f64) * (x as f64);
                    }
                    let rms = (ss / d_embd_local as f64).sqrt();
                    let mut ranked: Vec<(usize, f32)> = slot.iter().copied().enumerate().collect();
                    ranked.sort_by(|a, b| {
                        b.1.abs()
                            .partial_cmp(&a.1.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top3: Vec<(usize, f32)> = ranked.iter().take(3).copied().collect();
                    if dump_target_pos.is_some() {
                        eprintln!(
                            "HC_SLOT layer={layer_idx} pos={} slot={h} rms={rms:.4} top3={:?}",
                            state.pos, top3
                        );
                    }
                    if recorder_armed {
                        HC_SLOT_RECORDER.with(|r| {
                            if let Some(buf) = r.borrow_mut().as_mut() {
                                buf.push(HcSlotSample {
                                    layer: layer_idx,
                                    pos: state.pos,
                                    slot: h,
                                    rms,
                                    top3: top3.clone(),
                                });
                            }
                        });
                    }
                }
            }
        }
    }

    // Reduce HC residual to a single d_embd vector.
    // Real path (antirez `output_hc_head_one`, ds4.c:7654-7681) when the
    // GGUF supplies `output_hc_base / output_hc_fn / output_hc_scale`.
    // Synthetic fallback (used by the small CPU-oracle tests that omit
    // these tensors) takes slot 0 — INCORRECT for production decode but
    // preserves the existing fixture semantics.
    let final_hidden = match (
        model.output_hc_base.as_deref(),
        model.output_hc_fn.as_deref(),
        model.output_hc_scale.as_deref(),
    ) {
        (Some(base), Some(fn_w), Some(scale)) => output_hc_head_one(
            k,
            &state.cur_hc,
            base,
            fn_w,
            scale,
            n_hc,
            d_embd,
            cfg.eps_rms,
        ),
        _ => state.cur_hc[..d_embd].to_vec(),
    };

    state.pos = state.pos.saturating_add(1);
    Ok(final_hidden)
}

/// Print residual statistics + top-10 absolute-value dim indices.
/// Used by `DS4_DUMP_RESIDUAL_TAPS` to fingerprint near-one-hot residuals
/// feeding the lm_head matvec.
fn dump_residual_tap(label: &str, v: &[f32], pos: u32) {
    let n = v.len();
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut lmin = f32::INFINITY;
    let mut lmax = f32::NEG_INFINITY;
    for &x in v {
        let xd = x as f64;
        sum += xd;
        sumsq += xd * xd;
        if x < lmin {
            lmin = x;
        }
        if x > lmax {
            lmax = x;
        }
    }
    let mean = sum / n as f64;
    let rms = (sumsq / n as f64).sqrt();
    let mut ranked: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top10: Vec<(usize, f32)> = ranked.iter().take(10).copied().collect();
    eprintln!(
        "TAP_{label} pos={pos} n={n} rms={rms:.4} mean={mean:.4} min={lmin:.4} max={lmax:.4} top10_abs = {:?}",
        top10
    );
}

/// Output-side HC fold: collapse the `[n_hc × d_embd]` residual into a
/// single `[d_embd]` vector before the final RMSNorm + lm_head matvec.
///
/// Mirrors antirez `output_hc_head_one` (ds4.c:7654-7681):
///   1. `flat = rms_norm_no_weight(inp_hc, hc_dim, DS4_RMS_EPS)` — implemented
///      via `KernelDispatcher::rms_norm` with a unit-γ vector so we reuse the
///      dispatcher (Metal or CPU) instead of forking a no-weight variant.
///   2. `pre  = matvec_f16(output_hc_fn, flat)` → `[n_hc]`.
///   3. `w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS`.
///   4. `out[d] = Σ_h w[h] * inp_hc[h*d_embd + d]` (`hc_weighted_sum_one`).
///
/// Important: step 1 normalises across the FULL `hc_dim` (= n_hc × d_embd) —
/// not per-slot — and uses NO learnt gamma. Step 4 sums the un-normalised
/// `inp_hc`, NOT `flat`.
/// Reduce an HC residual (`[n_hc, d_embd]` flat) to a single `d_embd`
/// vector via the antirez `output_hc_head_one` (`ds4.c:7654-7681`).
/// Used at the end of the per-token decode loop to produce the
/// `final_hidden` that feeds the final norm + lm_head matvec.
///
/// Public so the M5.4.5 unified-cb decoder (`MetalDispatcher::
/// decode_token_unified`) can call it after `encode_first_half`
/// without duplicating the HC-fold logic.
pub fn output_hc_head_one<K: KernelDispatcher>(
    k: &K,
    inp_hc: &[f32],
    base: &[f32],
    fn_w: &[f32],
    scale: &[f32],
    n_hc: usize,
    d_embd: usize,
    eps_rms: f32,
) -> Vec<f32> {
    let hc_dim = n_hc * d_embd;
    debug_assert_eq!(inp_hc.len(), hc_dim);
    debug_assert_eq!(base.len(), n_hc);
    debug_assert_eq!(fn_w.len(), hc_dim * n_hc);
    debug_assert!(!scale.is_empty());

    let pre = k.output_hc_head_batched(inp_hc, fn_w, n_hc, d_embd, eps_rms);

    const DS4_HC_EPS: f32 = 1.0e-6;
    let s0 = scale[0];
    let mut w = vec![0.0f32; n_hc];
    for h in 0..n_hc {
        let x = pre[h] * s0 + base[h];
        let sigm = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        };
        w[h] = sigm + DS4_HC_EPS;
    }

    let mut out = vec![0.0f32; d_embd];
    for d in 0..d_embd {
        let mut acc = 0.0f32;
        for h in 0..n_hc {
            acc += inp_hc[h * d_embd + d] * w[h];
        }
        out[d] = acc;
    }
    out
}

/// Greedy sampling — argmax over logits.
pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Full step + greedy sample. Convenience wrapper used by the unit tests.
pub fn decode_one_token(x: Vec<f32>, model: &ModelWeights, cfg: &DecodeConfig) -> Result<usize> {
    let logits = decode_step(x, model, cfg)?;
    Ok(sample_argmax(&logits))
}

/// KV-cache-aware decode step — placeholder for E6 macOS landing.
///
/// On Linux/CPU we don't actually compress K/V; we keep this stub so the
/// engine surface (call site) stays stable. Once Metal lands the real call
/// site reads from `KvCache` and dispatches the FlashAttn-decode kernel.
pub fn decode_step_with_kv(
    x: Vec<f32>,
    model: &ModelWeights,
    _kv: &mut KvCache,
    cfg: &DecodeConfig,
) -> Result<Vec<f32>> {
    decode_step(x, model, cfg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_matrix(d: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; d * d];
        for i in 0..d {
            w[i * d + i] = 1.0;
        }
        w
    }

    fn one_hot_lm_head(vocab: usize, d: usize) -> Vec<f32> {
        // Row i = unit vector with a 1 at column (i % d) — gives a deterministic
        // logit pattern when we feed a known hidden state in.
        let mut w = vec![0.0f32; vocab * d];
        for v in 0..vocab {
            w[v * d + (v % d)] = 1.0;
        }
        w
    }

    fn tiny_layer(d: usize, d_ffn: usize, n_experts: usize, n_used: usize) -> LayerWeights {
        let identity_attn = identity_matrix(d);
        // Per-expert gate/up = identity (d_ffn=d here for simplicity), down = identity.
        assert_eq!(
            d_ffn, d,
            "tiny_layer requires d_ffn = d for identity packing"
        );
        let exp_w = identity_attn.clone();
        let mut stacked_gu = Vec::with_capacity(n_experts * d * d);
        let mut stacked_d = Vec::with_capacity(n_experts * d * d);
        for _ in 0..n_experts {
            stacked_gu.extend_from_slice(&exp_w);
            stacked_d.extend_from_slice(&exp_w);
        }
        LayerWeights {
            d_model: d,
            d_ffn: std::num::NonZeroUsize::new(d_ffn),
            n_experts,
            n_experts_used: n_used,
            attn_norm_gamma: vec![1.0; d],
            w_attn: vec![0.0; d * d], // zero attention → residual passthrough
            ffn_norm_gamma: vec![1.0; d],
            w_router: vec![0.5; n_experts * d], // uniform routing
            w_router_f16: Vec::new().into(),
            router_bias: vec![0.0; n_experts],
            w_gate_exps: stacked_gu.clone(),
            w_up_exps: stacked_gu,
            w_down_exps: stacked_d,
            routing_table: None,
        }
    }

    #[test]
    fn decode_layer_zero_attn_zero_ffn_is_passthrough() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Attention W=0, FFN router weights uniform but no useful work because
        // the moe block produces silu(x)*x for identity gate=up; we check that
        // the *shape* is preserved and the structural pipeline runs without error.
        let d = 4;
        let layer = LayerWeights {
            d_model: d,
            d_ffn: std::num::NonZeroUsize::new(d),
            n_experts: 4,
            n_experts_used: 2,
            attn_norm_gamma: vec![1.0; d],
            w_attn: vec![0.0; d * d],
            ffn_norm_gamma: vec![1.0; d],
            w_router: vec![0.0; 4 * d],
            w_router_f16: Vec::new().into(),
            router_bias: vec![0.0; 4],
            // Experts have identity gate/up/down, so moe_routed_step adds
            // weighted silu(x_norm)*x_norm back to x.
            w_gate_exps: vec![
                {
                    let mut m = vec![0.0f32; d * d];
                    for i in 0..d {
                        m[i * d + i] = 1.0;
                    }
                    m
                };
                4
            ]
            .concat(),
            w_up_exps: vec![
                {
                    let mut m = vec![0.0f32; d * d];
                    for i in 0..d {
                        m[i * d + i] = 1.0;
                    }
                    m
                };
                4
            ]
            .concat(),
            w_down_exps: vec![
                {
                    let mut m = vec![0.0f32; d * d];
                    for i in 0..d {
                        m[i * d + i] = 1.0;
                    }
                    m
                };
                4
            ]
            .concat(),
            routing_table: None,
        };
        let x = vec![0.5f32, -0.3, 1.2, -0.1];
        let y = decode_layer(&x, &layer, &DecodeConfig::default());
        assert_eq!(y.len(), x.len());
        // x[i] grows monotonically only when silu(x_norm)*x_norm has same sign as x_norm,
        // which is always true (silu(z)*z is nonneg for z >= ~−1.28 and similarly bounded).
        // We just sanity check we got real numbers.
        for &v in &y {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn decode_step_runs_n_layers_without_error() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = 4;
        let model = ModelWeights {
            layers: (0..3).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let x = vec![0.5f32; d];
        let logits = decode_step(x, &model, &DecodeConfig::default()).unwrap();
        assert_eq!(logits.len(), 8);
        for &v in &logits {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn decode_step_rejects_wrong_d_model() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = 4;
        let model = ModelWeights {
            layers: vec![tiny_layer(d, d, 4, 2)],
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let x = vec![0.5f32; d + 1];
        assert!(decode_step(x, &model, &DecodeConfig::default()).is_err());
    }

    #[test]
    fn sample_argmax_returns_index_of_max() {
        let logits = vec![0.1f32, -0.5, 3.2, 1.0, 2.9];
        assert_eq!(sample_argmax(&logits), 2);
    }

    #[test]
    fn sample_argmax_handles_negative_only() {
        let logits = vec![-5.0f32, -1.0, -3.0];
        assert_eq!(sample_argmax(&logits), 1);
    }

    #[test]
    fn decode_one_token_returns_valid_vocab_id() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let d = 4;
        let model = ModelWeights {
            layers: (0..2).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let x = vec![0.5f32, -0.3, 1.2, -0.1];
        let tok = decode_one_token(x, &model, &DecodeConfig::default()).unwrap();
        assert!(tok < 8, "token id {tok} out of vocab range");
    }

    #[test]
    fn decode_step_dispatch_counts_match_expected_pattern() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Each layer dispatches: 2× rms_norm, 2× matvec_f32, 1× softplus_sqrt,
        // 1× router_finalize, 1× moe_routed_step = 7 calls.
        // Full decode_step over N layers adds 1 rms_norm + 1 matvec at the
        // end (final norm + lm_head). So total = 7N + 2.
        use crate::dispatch::RecordingDispatcher;
        let d = 4;
        let n_layers = 3;
        let model = ModelWeights {
            layers: (0..n_layers).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let cpu = CpuDispatcher;
        let rec = RecordingDispatcher::new(&cpu);
        let x = vec![0.5f32; d];
        let _ = decode_step_with(&rec, x, &model, &DecodeConfig::default()).unwrap();
        let c = rec.counts();
        assert_eq!(c.rms_norm, 2 * n_layers + 1);
        assert_eq!(c.matvec_f32, 2 * n_layers + 1);
        assert_eq!(c.softplus_sqrt, n_layers);
        assert_eq!(c.router_finalize, n_layers);
        assert_eq!(c.moe_routed_step, n_layers);
        assert_eq!(c.total(), 7 * n_layers + 2);
    }

    #[test]
    fn tracing_dispatcher_full_decode_layer_event_pattern() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Validate that the trace from one decode_layer matches the expected
        // 7-event sequence: rms_norm, matvec, rms_norm, matvec, softplus_sqrt,
        // router_finalize, moe_routed_step.
        use crate::dispatch::{TraceEvent, TracingDispatcher};
        let d = 4;
        let layer = LayerWeights {
            d_model: d,
            d_ffn: std::num::NonZeroUsize::new(d),
            n_experts: 2,
            n_experts_used: 1,
            attn_norm_gamma: vec![1.0; d],
            w_attn: vec![0.0; d * d],
            ffn_norm_gamma: vec![1.0; d],
            w_router: vec![0.0; 2 * d],
            w_router_f16: Vec::new().into(),
            router_bias: vec![0.0; 2],
            w_gate_exps: vec![0.0; 2 * d * d],
            w_up_exps: vec![0.0; 2 * d * d],
            w_down_exps: vec![0.0; 2 * d * d],
            routing_table: None,
        };
        let cpu = CpuDispatcher;
        let tracer = TracingDispatcher::new(&cpu);
        let _ = decode_layer_with(
            &tracer,
            0,
            &[0.1, 0.2, 0.3, 0.4],
            &layer,
            &DecodeConfig::default(),
        );

        let events = tracer.events();
        let kinds: Vec<&'static str> = events.iter().map(TraceEvent::kind).collect();
        assert_eq!(
            kinds,
            vec![
                "rms_norm",
                "matvec_f32",
                "rms_norm",
                "matvec_f32",
                "softplus_sqrt",
                "router_finalize",
                "moe_routed_step",
            ]
        );
    }

    #[test]
    fn triple_stacked_decorators_in_full_decode_step() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // This is the exact harness shape Mac validation will use:
        //   Recording⊕Tracing⊕Timing wrapping a back-end dispatcher.
        // On Linux the back end is CpuDispatcher; on Mac it'll be
        // MetalDispatcher. The test proves the API composes without
        // needing the device.
        use crate::dispatch::{PerKernelTimingDispatcher, RecordingDispatcher, TracingDispatcher};
        let d = 4;
        let n_layers = 3;
        let model = ModelWeights {
            layers: (0..n_layers).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let cpu = CpuDispatcher;
        let timer = PerKernelTimingDispatcher::new(&cpu);
        let tracer = TracingDispatcher::new(&timer);
        let recorder = RecordingDispatcher::new(&tracer);

        let direct = decode_step(vec![0.5f32; d], &model, &DecodeConfig::default()).unwrap();
        let stacked =
            decode_step_with(&recorder, vec![0.5f32; d], &model, &DecodeConfig::default()).unwrap();

        // (1) result bit-identical to direct path.
        assert_eq!(direct, stacked);

        // (2) all three decorators saw the same 7N+2 calls.
        let counts = recorder.counts();
        let timings = timer.timings();
        let events = tracer.events();
        assert_eq!(counts.total(), 7 * n_layers + 2);
        assert_eq!(timings.total_calls() as usize, 7 * n_layers + 2);
        assert_eq!(events.len(), 7 * n_layers + 2);
    }

    #[test]
    fn per_kernel_timing_dispatcher_records_full_decode_step() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        use crate::dispatch::PerKernelTimingDispatcher;
        let d = 4;
        let n_layers = 3;
        let model = ModelWeights {
            layers: (0..n_layers).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let cpu = CpuDispatcher;
        let timer = PerKernelTimingDispatcher::new(&cpu);
        let _ =
            decode_step_with(&timer, vec![0.5f32; d], &model, &DecodeConfig::default()).unwrap();
        let t = timer.timings();
        // Same 7N+2 pattern as RecordingDispatcher.
        assert_eq!(t.rms_norm_calls as usize, 2 * n_layers + 1);
        assert_eq!(t.matvec_f32_calls as usize, 2 * n_layers + 1);
        assert_eq!(t.softplus_sqrt_calls as usize, n_layers);
        assert_eq!(t.router_finalize_calls as usize, n_layers);
        assert_eq!(t.moe_routed_step_calls as usize, n_layers);
        assert_eq!(t.total_calls() as usize, 7 * n_layers + 2);
        // Each method that ran must have non-zero accumulated ns.
        assert!(t.rms_norm_ns > 0);
        assert!(t.matvec_f32_ns > 0);
        assert!(t.moe_routed_step_ns > 0);
    }

    #[test]
    fn decode_step_with_cpu_dispatcher_matches_direct() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // CpuDispatcher must yield bit-identical logits to the direct path.
        let d = 4;
        let model = ModelWeights {
            layers: (0..2).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let x = vec![0.5f32, -0.3, 1.2, -0.1];
        let cfg = DecodeConfig::default();
        let direct = decode_step(x.clone(), &model, &cfg).unwrap();
        let via_disp = decode_step_with(&CpuDispatcher, x, &model, &cfg).unwrap();
        assert_eq!(direct.len(), via_disp.len());
        for (a, b) in direct.iter().zip(via_disp.iter()) {
            assert_eq!(a, b, "dispatcher path drifted from direct path");
        }
    }

    /// Write a single-layer DS4 GGUF carrying:
    ///   - 10 attention roles + ffn_norm + moe_router + moe_router_bias on blk.0
    ///   - global `output_norm` (final_norm) + `output` (lm_head)
    ///   - 7 metadata keys that `LayerParams::from_gguf` reads
    ///
    /// All tensors are F32 so `dequant_f32_simple` round-trips exactly.
    fn write_from_views_gguf(path: &std::path::Path) -> (u32, u32, u32, u32, u32) {
        // DS4 V4 Flash defaults pin n_hc = 4. d_embd == d_model in DS4
        // (each HC slot carries a full d_model-wide copy).
        let d_model: u32 = 4;
        let vocab: u32 = 6;
        let n_layers: u32 = 1;
        let n_experts: u32 = 4;
        let n_experts_used: u32 = 2;

        let d = d_model as usize;
        let n_hc = 4usize;
        let d_embd = d_model as usize;
        let hc_dim = n_hc * d_embd;
        let mix_hc = 2 * n_hc + n_hc * n_hc;
        let n_head = 2usize;
        let head_dim = 4usize;
        let n_lora_q = 4usize;
        let n_lora_kv = 4usize;
        let q_dim = n_head * head_dim;
        // Grouped output projection: heads[q_dim] → [n_out_group, group_dim]
        // → attn_output_a [out_low_dim, group_dim] (row-major; GGUF order
        // [group_dim, out_low_dim]) → attn_output_b [d_embd, out_low_dim]
        // (row-major; GGUF order [out_low_dim, d_embd]).
        let n_out_group = 2usize;
        let group_dim = q_dim / n_out_group;
        let n_lora_o = 3usize;
        let out_low_dim = n_out_group * n_lora_o;
        let shared_dim = d_embd;

        let mut tensors: Vec<(String, Vec<u64>, f32)> = vec![
            // attention roles on blk.0
            ("blk.0.attn_norm.weight".into(), vec![d_embd as u64], 1.0),
            ("blk.0.ffn_norm.weight".into(), vec![d as u64], 0.95),
            (
                "blk.0.attn_q_a_norm.weight".into(),
                vec![n_lora_q as u64],
                1.1,
            ),
            (
                "blk.0.attn_kv_a_norm.weight".into(),
                vec![n_lora_kv as u64],
                1.2,
            ),
            // MLA Q/KV projections (added 2026-05-13 for Phase B2 surface).
            // GGUF dim order is [in, out]; antirez `ds4.c:2160` ground truth:
            //   attn_kv: [d_embd, n_lora_kv] (n_lora_kv == head_dim == 512).
            (
                "blk.0.attn_q_a.weight".into(),
                vec![d_embd as u64, n_lora_q as u64],
                0.11,
            ),
            (
                "blk.0.attn_q_b.weight".into(),
                vec![n_lora_q as u64, q_dim as u64],
                0.12,
            ),
            (
                "blk.0.attn_kv.weight".into(),
                vec![d_embd as u64, n_lora_kv as u64],
                0.13,
            ),
            (
                "blk.0.attn_output_a.weight".into(),
                vec![group_dim as u64, out_low_dim as u64],
                0.10,
            ),
            (
                "blk.0.attn_output_b.weight".into(),
                vec![out_low_dim as u64, d_embd as u64],
                0.20,
            ),
            ("blk.0.attn_sinks.weight".into(), vec![n_head as u64], -1.0),
            (
                "blk.0.ffn_gate_shexp.weight".into(),
                vec![d_embd as u64, shared_dim as u64],
                0.30,
            ),
            (
                "blk.0.ffn_up_shexp.weight".into(),
                vec![d_embd as u64, shared_dim as u64],
                0.40,
            ),
            (
                "blk.0.ffn_down_shexp.weight".into(),
                vec![shared_dim as u64, d_embd as u64],
                0.50,
            ),
            // HC projection weights (both pre-attn + pre-FFN)
            (
                "blk.0.hc_attn_fn.weight".into(),
                vec![hc_dim as u64, mix_hc as u64],
                0.7,
            ),
            ("blk.0.hc_attn_scale.weight".into(), vec![3u64], 0.06),
            (
                "blk.0.hc_attn_base.weight".into(),
                vec![mix_hc as u64],
                0.05,
            ),
            (
                "blk.0.hc_ffn_fn.weight".into(),
                vec![hc_dim as u64, mix_hc as u64],
                0.71,
            ),
            ("blk.0.hc_ffn_scale.weight".into(), vec![3u64], 0.061),
            (
                "blk.0.hc_ffn_base.weight".into(),
                vec![mix_hc as u64],
                0.051,
            ),
            // skeleton LayerWeights roles
            (
                "blk.0.ffn_gate_inp.weight".into(),
                vec![n_experts as u64, d as u64],
                0.01,
            ),
            (
                "blk.0.exp_probs_b.weight".into(),
                vec![n_experts as u64],
                0.02,
            ),
            // globals
            ("output_norm.weight".into(), vec![d as u64], 0.99),
            ("output.weight".into(), vec![vocab as u64, d as u64], 0.33),
        ];

        // Sort by deterministic name for reproducibility (no semantic effect).
        tensors.sort_by(|a, b| a.0.cmp(&b.0));

        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // 8 keys: architecture + 6 attention/rope keys + output_group_count.
        buf.extend_from_slice(&8u64.to_le_bytes());

        let write_str = |buf: &mut Vec<u8>, s: &str| {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        };
        write_str(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_str(&mut buf, "deepseek4");

        write_str(&mut buf, "deepseek4.attention.head_count");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(n_head as u32).to_le_bytes());

        write_str(&mut buf, "deepseek4.attention.key_length");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(head_dim as u32).to_le_bytes());

        write_str(&mut buf, "deepseek4.attention.value_length");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(head_dim as u32).to_le_bytes());

        write_str(&mut buf, "deepseek4.attention.q_lora_rank");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(n_lora_q as u32).to_le_bytes());

        write_str(&mut buf, "deepseek4.attention.kv_lora_rank");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(n_lora_kv as u32).to_le_bytes());

        write_str(&mut buf, "deepseek4.rope.dimension_count");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes());

        write_str(&mut buf, "deepseek4.attention.output_group_count");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&(n_out_group as u32).to_le_bytes());

        let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
        let mut running = 0u64;
        for (_, dims, _) in &tensors {
            offsets.push(running);
            let n_elem: u64 = dims.iter().product();
            running += n_elem * 4;
        }
        for ((name, dims, _), &off) in tensors.iter().zip(offsets.iter()) {
            write_str(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for dim in dims {
                buf.extend_from_slice(&dim.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // F32
            buf.extend_from_slice(&off.to_le_bytes());
        }
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        for (_, dims, fill) in &tensors {
            let n_elem: u64 = dims.iter().product();
            for _ in 0..n_elem {
                buf.extend_from_slice(&fill.to_le_bytes());
            }
        }
        std::fs::write(path, &buf).unwrap();
        (n_layers, d_model, vocab, n_experts, n_experts_used)
    }

    #[test]
    fn from_views_builds_composed_model_from_synthetic_gguf() {
        let tmp = std::env::temp_dir().join("ds4_compose_from_views.gguf");
        let (n_layers, d_model, vocab, n_experts, n_experts_used) = write_from_views_gguf(&tmp);

        let views = crate::layer_view::LayerViews::open(&tmp, n_layers).expect("open views");

        let manifest = crate::gguf::ModelManifest {
            path: tmp.clone(),
            n_layers,
            d_model,
            vocab_size: vocab,
            n_experts,
            n_experts_used,
            total_tensor_bytes: 0,
            per_type_bytes: std::collections::BTreeMap::new(),
            roles_seen: std::collections::BTreeSet::new(),
        };

        let defaults = crate::attn_dispatch::DefaultsDs4::ds4_v4_flash();
        let composed =
            ComposedModelWeights::from_views(&views, &manifest, defaults).expect("from_views");

        assert_eq!(composed.d_model, d_model as usize);
        assert_eq!(composed.vocab_size, vocab as usize);
        assert_eq!(composed.layers.len(), n_layers as usize);

        // Globals should be the f32 fill values.
        assert!(composed.final_norm_gamma.iter().all(|&v| v == 0.99));
        assert_eq!(
            composed.lm_head.len(),
            (vocab as usize) * (d_model as usize)
        );
        assert!(composed.lm_head.iter().all(|&v| v == 0.33));

        // Layer 0 attention slice: per write_from_views_gguf fill values.
        let l0 = &composed.layers[0];
        assert!(l0.attn.hc_norm_gamma.iter().all(|&v| v == 1.0));
        assert!(l0.attn.qkv_gamma_q.iter().all(|&v| v == 1.1));
        assert!(l0.attn.qkv_gamma_kv.iter().all(|&v| v == 1.2));
        assert!(l0.attn.attn_sinks.iter().all(|&v| v == -1.0));
        assert!(l0.attn.w_o_a.iter().all(|&v| v == 0.10));
        assert!(l0.attn.w_o_b.iter().all(|&v| v == 0.20));
        assert!(l0.attn.w_shared_gate.iter().all(|&v| v == 0.30));
        assert!(l0.attn.w_shared_up.iter().all(|&v| v == 0.40));
        assert!(l0.attn.w_shared_down.iter().all(|&v| v == 0.50));
        assert!(l0.attn.hc_attn_fn.iter().all(|&v| v == 0.7));
        assert!(l0.attn.hc_attn_scale.iter().all(|&v| v == 0.06));
        assert!(l0.attn.hc_attn_base.iter().all(|&v| v == 0.05));
        assert!(l0.attn.hc_ffn_fn.iter().all(|&v| v == 0.71));
        assert!(l0
            .attn
            .hc_ffn_scale
            .iter()
            .all(|&v| (v - 0.061).abs() < 1e-6));
        assert!(l0
            .attn
            .hc_ffn_base
            .iter()
            .all(|&v| (v - 0.051).abs() < 1e-6));
        assert!(l0.attn.hc_ffn_norm_gamma.iter().all(|&v| v == 0.95));

        // LayerParams populated from metadata (head_count=2, key_length=4,
        // q_lora_rank=4, kv_lora_rank=4, rope.dimension_count=2).
        assert_eq!(l0.attn.params.layer_idx, 0);
        assert_eq!(l0.attn.params.n_head, 2);
        assert_eq!(l0.attn.params.head_dim, 4);
        assert_eq!(l0.attn.params.n_rot, 2);
        assert_eq!(l0.attn.params.n_lora_q, 4);
        assert_eq!(l0.attn.params.n_lora_kv, 4);
        // n_hc comes from defaults (DS4 V4 Flash = 4). d_embd == d_model.
        assert_eq!(l0.attn.params.n_hc, defaults.n_hc);
        assert_eq!(l0.attn.params.d_embd, d_model);
        // No attn_compressor_kv tensor → compress_ratio stays at default 1.
        assert_eq!(l0.attn.params.compress_ratio, 1);

        // MoE skeleton slices come from the f32 fill values.
        assert!(l0.moe.ffn_norm_gamma.iter().all(|&v| v == 0.95));
        assert!(l0.moe.w_router.iter().all(|&v| (v - 0.01).abs() < 1e-7));
        assert!(l0.moe.router_bias.iter().all(|&v| (v - 0.02).abs() < 1e-7));
        // attn_norm_gamma was cloned from attn.hc_norm_gamma.
        assert_eq!(l0.moe.attn_norm_gamma, l0.attn.hc_norm_gamma);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn from_views_bails_on_missing_global_final_norm() {
        // Build a GGUF that has all per-layer roles + lm_head but no
        // output_norm — `from_views` must surface a clear error.
        let tmp = std::env::temp_dir().join("ds4_compose_from_views_no_final_norm.gguf");
        write_from_views_gguf(&tmp);
        // Strip output_norm by rewriting without it.
        // Simplest: rebuild minimally.
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x46554747u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 keys
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        std::fs::write(&tmp, &buf).unwrap();
        let views = crate::layer_view::LayerViews::open(&tmp, 0).expect("open");
        let manifest = crate::gguf::ModelManifest {
            path: tmp.clone(),
            n_layers: 0,
            d_model: 8,
            vocab_size: 6,
            n_experts: 0,
            n_experts_used: 0,
            total_tensor_bytes: 0,
            per_type_bytes: std::collections::BTreeMap::new(),
            roles_seen: std::collections::BTreeSet::new(),
        };
        let defaults = crate::attn_dispatch::DefaultsDs4::ds4_v4_flash();
        let err = ComposedModelWeights::from_views(&views, &manifest, defaults)
            .unwrap_err()
            .to_string();
        assert!(err.contains("final_norm"), "err = {err}");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn decode_step_with_kv_matches_plain_decode_step() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // The CPU-side KV-cache placeholder is a no-op pass-through.
        let d = 4;
        let model = ModelWeights {
            layers: (0..2).map(|_| tiny_layer(d, d, 4, 2)).collect(),
            final_norm_gamma: vec![1.0; d],
            lm_head: one_hot_lm_head(8, d),
            vocab_size: 8,
            d_model: d,
        };
        let x = vec![0.1f32, 0.2, 0.3, 0.4];
        let cfg = DecodeConfig::default();
        let logits_plain = decode_step(x.clone(), &model, &cfg).unwrap();
        let mut kv = KvCache::new(crate::kv_cache::KvCacheShape {
            n_layers: 2,
            max_seq_len: 4,
            kv_lora_rank: 8,
            n_kv_heads: 1,
            qk_rope_head_dim: 4,
        });
        let logits_kv = decode_step_with_kv(x, &model, &mut kv, &cfg).unwrap();
        assert_eq!(logits_plain.len(), logits_kv.len());
        for (a, b) in logits_plain.iter().zip(logits_kv.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn output_hc_head_one_matches_hand_computed_reference() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Tiny n_hc=2, d_embd=3 case. Compute the antirez expression by hand
        // and compare against `output_hc_head_one`.
        let n_hc = 2usize;
        let d_embd = 3usize;
        let hc_dim = n_hc * d_embd;
        let inp_hc: Vec<f32> = vec![
            // slot 0
            1.0, 2.0, 3.0, // slot 1
            -1.0, 0.5, 0.25,
        ];
        // Pick fn_w so that pre[h] is something specific. fn_w shape [hc_dim, n_hc].
        // pre[h] = Σ_i fn_w[h*hc_dim + i] * flat[i].
        let fn_w: Vec<f32> = vec![
            // pre[0] row (hc_dim=6 entries)
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0, // pre[1] row
            0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
        ];
        let base: Vec<f32> = vec![0.2, -0.1];
        let scale: Vec<f32> = vec![2.0];

        let cpu = CpuDispatcher;
        let out = output_hc_head_one(&cpu, &inp_hc, &base, &fn_w, &scale, n_hc, d_embd, 1e-5);

        // Reference: replicate the antirez expression.
        let mean_sq: f64 =
            inp_hc.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / hc_dim as f64;
        let r_scale = 1.0 / ((mean_sq as f32) + 1e-5).sqrt();
        let flat: Vec<f32> = inp_hc.iter().map(|&v| v * r_scale).collect();

        let pre0 = 0.1 * flat[0];
        let pre1 = 0.5 * flat[3];

        let sigmoid = |x: f32| {
            if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            }
        };
        let w0 = sigmoid(pre0 * 2.0 + 0.2) + 1.0e-6;
        let w1 = sigmoid(pre1 * 2.0 - 0.1) + 1.0e-6;

        let expected: Vec<f32> = (0..d_embd)
            .map(|d| inp_hc[d] * w0 + inp_hc[d_embd + d] * w1)
            .collect();

        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "out[{i}] = {a} vs expected {b}");
        }
    }
}
