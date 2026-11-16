// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Attention-side kernel dispatch trait + CPU reference ( Option 3).
//!
//! `KernelDispatcher` (in `dispatch.rs`) covers the 5 math primitives used
//! by the MoE/router half of decode_step. This module adds the missing
//! attention-half surface: per-layer kernels needed to drive DS4 (MLA)
//! attention without delegating to antirez at decode time.
//!
//! Two orthogonal pieces:
//!
//! 1. `AttentionDispatcher` trait — 8 new methods, one per antirez call site
//! that the existing 5-method trait doesn't cover. Methods are listed in
//! the order they fire per layer (per `ds4.c:8684 metal_graph_encode_decode_layer`).
//! 2. `LayerParams` struct — the per-layer shape pack threaded through every
//! call. Avoids 20-argument signatures and matches antirez's
//! `the accelerator host crate_args_*` uniform structs.
//!
//! Design choices ( Option 3, see `project_ds4_decode_layer_kernel_map.md`):
//!
//! - **Two traits, not one.** Keeping `KernelDispatcher` stable means existing
//! `decode_step_with` callers (and the 3 Tracing/Recording/Timing decorators)
//! don't get rewritten. `MetalDispatcher` will implement both.
//! - **Methods named for antirez kernels, not abstract math.** `rope_tail`,
//! `flash_attn_decode`, `kv_fp8_store` — the registry already names these,
//! the trait stays the obvious seam.
//! - **CPU first, then Metal.** Every method has a scalar Rust impl on
//! `CpuAttentionDispatcher` good enough for trace-equality testing. The
//! Metal impl lands in #214.
//! - **f32 in, f32 out at the trait boundary.** Quantized weight access
//! (Q8_0 / FP8 / IQ2_XXS) stays inside the impl. CPU dispatcher dequantizes
//! on demand from a `LayerView` (#213); Metal dispatcher reads packed
//! blocks straight from the mmaped GGUF.
//!
//! Optional, deferred to :
//! - `compressor_update` — only fires on compressed layers
//! - `indexer_step` — only fires on ratio==4 layers
//! Both are exposed as `Option<…>` so the trait stays single-shape.

// Trait + types are designed surface for #213 / #214; not yet called from
// decode_step. Suppress the dead-code wave until LayerView lands.
#![allow(dead_code)]

use std::borrow::Cow;
use std::cell::Cell;
use std::f32::consts::PI;

thread_local! {
 /// per-step position hint, set by `decode_step_with_attn` and
 /// read by env-gated diagnostic dumps inside the CPU oracle. NOT part of
 /// the dispatcher contract — it's a debug-only side channel. Default 0.
 pub static CURRENT_POS_HINT: Cell<u32> = const { Cell::new(0) };

 /// Verifier-bisection layer hint: set by decode_step's per-layer loop so
 /// flash capture (below) knows the current layer. `usize::MAX` = unset.
 pub static CURRENT_LAYER_HINT: Cell<usize> = const { Cell::new(usize::MAX) };

 /// Verifier-bisection flash tap: set `.0` to a target layer; after that
 /// layer's `flash_attn_decode` (in `decode_attn_prefix_with`), decode_step
 /// stores the per-head flash output (`heads`, pre-output-proj) in `.1`. Lets
 /// a test compare the verifier's flash_attn_k_mla output against
 /// decode_step's flash at the same layer (spec-decode flash bisection).
 pub static FLASH_CAPTURE: std::cell::RefCell<(usize, Vec<f32>)> =
 const { std::cell::RefCell::new((usize::MAX, Vec::new())) };

 /// Verifier-bisection output-proj tap: `.0` = target layer; after that
 /// layer's `attn_output_proj` computes `attn_out` (W_o·heads, pre-hc_expand),
 /// stores it in `.1`. Splits output_proj vs hc_expand_attn.
 pub static ATTN_OUT_CAPTURE: std::cell::RefCell<(usize, Vec<f32>)> =
 const { std::cell::RefCell::new((usize::MAX, Vec::new())) };

 /// per-step input token id. Set by the driver before each
 /// `decode_step_with_attn` call; consumed by hash-routed MoE selection
 /// for layers with `ffn_gate_tid2eid` (antirez ds4.c:5155-5157). Default 0.
 pub static CURRENT_TOKEN_HINT: Cell<i32> = const { Cell::new(0) };
}

/// Per-layer shape parameters threaded into every attention call.
/// Mirrors the subset of antirez's `the accelerator host crate_args_*` uniforms that
/// the decode hot path reads. Values come from the validated GGUF
/// manifest (`ModelManifest`), not from in-flight tensors.
///
/// Field names match antirez `DS4_N_*` constants where applicable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayerParams {
 /// 0-based layer index. Backends use it to select the right weight
 /// view from `ModelManifest` / `LayerView`.
 pub layer_idx: u32,
 /// `DS4_N_EMBD` — hidden dim per HC slot. In DS4 this equals the
 /// model `d_model` because each HC slot holds a full-width copy of
 /// the token embedding (antirez `ds4.c:4198`).
 pub d_embd: u32,
 /// `DS4_N_HC` — number of HC slots. `flat_hc.len() = d_embd * n_hc`.
 pub n_hc: u32,
 /// `DS4_N_HEAD` — number of attention heads.
 pub n_head: u32,
 /// `DS4_N_HEAD_DIM` — dim per head (including RoPE-tail section).
 pub head_dim: u32,
 /// `DS4_N_ROT` — RoPE-tail rotation dim (subset of `head_dim`).
 pub n_rot: u32,
 /// `DS4_N_LORA_Q` — Q low-rank.
 pub n_lora_q: u32,
 /// `DS4_N_LORA_KV` — KV low-rank (NOPE part). KV-down emits
 /// `n_lora_kv` floats per token. Per antirez `ds4.c:88` (`DS4_N_HEAD_DIM=512`),
 /// the kv-down row has exactly `n_lora_kv` (== head_dim) floats; the rope
 /// tail occupies the last `n_rot` slots `[n_lora_kv - n_rot..n_lora_kv]`.
 pub n_lora_kv: u32,
 /// HC sinkhorn / collapse iterations (`DS4_N_HC_SINKHORN_ITER`).
 pub hc_sinkhorn_iter: u32,
 /// HC collapse epsilon (`DS4_HC_EPS`).
 pub hc_eps: f32,
 /// RMS norm epsilon (`DS4_RMS_EPS`).
 pub rms_eps: f32,
 /// `DS4_ROPE_ORIG_CTX` — base context size used by YaRN scaling.
 pub rope_orig_ctx: u32,
 /// Computed `freq_base`, `freq_scale`, `ext_factor`, `attn_factor`
 /// for the current `pos` — see `ds4.c::metal_graph_encode_decode_layer`.
 pub rope_freq_base: f32,
 pub rope_freq_scale: f32,
 pub rope_ext_factor: f32,
 pub rope_attn_factor: f32,
 /// Layer-specific compression ratio (1 = no compressor, 4 = indexer).
 pub compress_ratio: u32,
 /// `DS4_N_OUT_GROUP` — output-projection group count for grouped
 /// multi-query attention. `heads[q_dim]` is reshaped to
 /// `[n_out_group, group_dim]` where `group_dim = q_dim / n_out_group`,
 /// and `attn_output_a` is row-major `[out_low_dim, group_dim]` where
 /// `out_low_dim = n_out_group * n_lora_o`. For V4 Flash, n_out_group=8.
 pub n_out_group: u32,
}

impl LayerParams {
 /// True iff this layer runs the compressor + (when ratio==4) indexer
 /// path. Used by `decode_step_with` to gate optional calls.
 pub fn is_compressed(&self) -> bool {
 self.compress_ratio > 1
 }

 /// True iff this layer routes attention through the indexer top-k.
 pub fn uses_indexer(&self) -> bool {
 self.compress_ratio == 4
 }

 /// `n_head * head_dim` — flat per-token Q dim out of `attn_q_b`.
 pub fn q_dim(&self) -> usize {
 self.n_head as usize * self.head_dim as usize
 }

 /// `q_dim / n_out_group` — width of each group fed through W_o^a.
 pub fn group_dim(&self) -> usize {
 self.q_dim() / self.n_out_group as usize
 }

 /// Flat HC dim — width of `cur_hc`/`flat_hc` buffers.
 pub fn hc_dim(&self) -> usize {
 self.n_hc as usize * self.d_embd as usize
 }

 /// Build per-layer `LayerParams` from GGUF metadata ( Phase B3a).
 ///
 /// Reads the `deepseek4.*` shape keys (with `deepseek2.*` / `llama.*`
 /// fallbacks where they exist). Each key tried in `meta_u32` order. The
 /// constants for HC / sinkhorn / `compress_ratio` are not in the GGUF —
 /// they come from caller-supplied DS4 defaults (`Defaults::ds4_v4_flash`).
 ///
 /// `layer_idx` is threaded in unchanged. `compress_ratio` defaults to 1
 /// (dense layer); the caller may override for indexer layers.
 pub fn from_gguf(
 g: &crate::gguf::GgufFile,
 d_model: u32,
 layer_idx: u32,
 defaults: DefaultsDs4,
 ) -> anyhow::Result<Self> {
 use crate::gguf::{meta_f32, meta_u32};

 let n_head = meta_u32(
 g,
 &[
 "deepseek4.attention.head_count",
 "deepseek41.num_attention_heads",
 "deepseek2.attention.head_count",
 ],
 )
 .ok_or_else(|| anyhow::anyhow!("missing deepseek4.attention.head_count"))?;
 let head_dim = meta_u32(
 g,
 &[
 "deepseek4.attention.key_length",
 "deepseek41.head_dim",
 "deepseek2.attention.key_length",
 ],
 )
 .ok_or_else(|| anyhow::anyhow!("missing deepseek4.attention.key_length"))?;
 // antirez also exposes `value_length`; for the trace-equality contract
 // we assert they agree (DS4 keeps K and V head dims equal).
 if let Some(v_len) = meta_u32(
 g,
 &[
 "deepseek4.attention.value_length",
 "deepseek2.attention.value_length",
 ],
 ) {
 anyhow::ensure!(
 v_len == head_dim,
 "DS4 requires attention.key_length == value_length, got {head_dim} vs {v_len}"
 );
 }
 let n_rot = meta_u32(g, &["deepseek4.rope.dimension_count"]).unwrap_or(defaults.n_rot);
 // The LoRA ranks are read from the TENSORS, not only from metadata. V4.1 carries no
 // `q_lora_rank` key at all and its rank is 1280 where V4 Flash's is 1024, so a metadata
 // read fell back to the default and every projection then asserted on a 1280-row weight
 // ("matvec dims: W has 1024 rows of 5120"). A GGUF row count cannot disagree with the
 // weights it describes, so it is the authoritative source and metadata is the fallback.
 let rank_of = |suffix: &str| -> Option<u32> {
 g.tensors
 .iter()
 .find(|t| t.name.ends_with(suffix))
 .and_then(|t| t.dims.get(1).copied())
 .map(|v| v as u32)
 };
 let n_lora_q = rank_of("attn_q_a.weight")
 .or_else(|| meta_u32(g, &["deepseek4.attention.q_lora_rank"]))
 .unwrap_or(defaults.n_lora_q);
 let n_lora_kv = rank_of("attn_kv.weight")
 .or_else(|| meta_u32(g, &["deepseek4.attention.kv_lora_rank"]))
 .unwrap_or(defaults.n_lora_kv);
 let rope_freq_base = meta_f32(g, &["deepseek4.rope.freq_base", "deepseek2.rope.freq_base"])
 .unwrap_or(defaults.rope_freq_base);
 let rope_freq_scale = meta_f32(g, &["deepseek4.rope.scaling.factor"])
 .map(|f| 1.0 / f)
 .unwrap_or(defaults.rope_freq_scale);
 let rope_orig_ctx = meta_u32(g, &["deepseek4.rope.scaling.original_context_length"])
 .unwrap_or(defaults.rope_orig_ctx);

 Ok(Self {
 layer_idx,
 // In DS4 every HC slot holds a *full-width* copy of the
 // embedding (antirez `ds4.c:4198` memcpys the entire token
 // embed into each of `n_hc` slots), so `d_embd == d_model`,
 // not `d_model / n_hc`. The residual is `n_hc * d_embd`
 // wide overall.
 d_embd: d_model,
 n_hc: defaults.n_hc,
 n_head,
 head_dim,
 n_rot,
 n_lora_q,
 n_lora_kv,
 hc_sinkhorn_iter: defaults.hc_sinkhorn_iter,
 hc_eps: defaults.hc_eps,
 rms_eps: defaults.rms_eps,
 rope_orig_ctx,
 rope_freq_base,
 rope_freq_scale,
 rope_ext_factor: defaults.rope_ext_factor,
 rope_attn_factor: defaults.rope_attn_factor,
 compress_ratio: 1,
 n_out_group: meta_u32(g, &["deepseek4.attention.output_group_count"])
 .unwrap_or(defaults.n_out_group),
 })
 }
}

/// Per-architecture default constants that aren't carried in the GGUF
/// metadata directly — needed to construct `LayerParams` from a GGUF.
///
/// `ds4_v4_flash()` returns the DS4 V4 Flash 284B defaults; other DS4 sizes
/// can plug in their own values.
#[derive(Debug, Clone, Copy)]
pub struct DefaultsDs4 {
 pub n_hc: u32,
 pub n_rot: u32,
 pub n_lora_q: u32,
 pub n_lora_kv: u32,
 pub hc_sinkhorn_iter: u32,
 pub hc_eps: f32,
 pub rms_eps: f32,
 pub rope_orig_ctx: u32,
 pub rope_freq_base: f32,
 pub rope_freq_scale: f32,
 pub rope_ext_factor: f32,
 pub rope_attn_factor: f32,
 pub n_out_group: u32,
}

impl DefaultsDs4 {
 /// DS4 V4 Flash 284B defaults — match antirez `ds4.c:84-104` constants.
 ///
 /// `n_lora_kv` is `DS4_N_HEAD_DIM` (= 512); antirez does not have a
 /// separate kv-lora rank — it reuses the head dim. The naming inside
 /// `LayerParams` is preserved for trait-shape compatibility but the
 /// value comes from `attention.key_length` GGUF metadata.
 pub fn ds4_v4_flash() -> Self {
 Self {
 n_hc: 4,
 n_rot: 64,
 n_lora_q: 1024,
 n_lora_kv: 512,
 hc_sinkhorn_iter: 20,
 hc_eps: 1e-6,
 rms_eps: 1e-6,
 // DS4-V4-Flash is a YaRN model: the DS4 GGUFs that carry RoPE
 // metadata declare `rope.scaling.factor = 16` and
 // `original_context_length = 65536`. These are the fallback for a
 // file that OMITS those keys — as the MXFP4 release does — so they
 // must state what DS4-Flash actually is rather than a generic
 // unscaled RoPE. A file that genuinely wants no scaling says so in
 // its metadata and overrides this.
 rope_orig_ctx: 65536,
 rope_freq_base: 10000.0,
 rope_freq_scale: 1.0 / 16.0,
 rope_ext_factor: 0.0,
 rope_attn_factor: 1.0,
 n_out_group: 8,
 }
 }
}

/// Output of `flash_attn_decode`. Heads × head_dim, row-major.
pub type AttnHeadsOut = Vec<f32>;

/// KV cache view passed to attention. The trait does not own storage;
/// callers thread a mutable slice (cpu) or a `MTLBuffer` ptr (metal)
/// in a way the impl picks up via associated types.
///
/// For CPU impl we use plain f32 storage; the Metal impl will replace
/// this with an opaque handle that wraps an `MTLBuffer`. Keeping the
/// associated type out for now — concrete f32 slice is enough for the
/// CPU oracle.
pub struct KvCacheView<'a> {
 /// Raw KV cache: `[capacity, n_lora_kv]` flat row-major. Per antirez
 /// `DS4_N_HEAD_DIM=512`, the kv row is exactly `n_lora_kv` (==head_dim)
 /// floats — rope tail lives at `[n_lora_kv-n_rot..n_lora_kv]` *inside*
 /// the row, not appended.
 pub raw: &'a mut [f32],
 /// Logical capacity in rows.
 pub raw_cap: u32,
 /// Current write position (KV row count so far).
 pub pos: u32,
}

/// Decode-time attention primitives. Each method matches one antirez
/// kernel; signature is `&[f32]` in, `Vec<f32>` (or `()`) out, with
/// shape info from `LayerParams`.
///
/// The trait is intentionally NOT a supertrait of `KernelDispatcher` —
/// dispatchers compose by holding both. This keeps the attention
/// surface independently testable and decorator-friendly.
/// Which of the two per-layer HC-collapse calls this is. Antirez runs the
/// pre-attention call (`ds4.c:8727`) with the `hc_attn_*` weight triplet,
/// then later the pre-FFN call (`ds4.c:9234`) with the `hc_ffn_*` triplet.
/// Both share the same kernel surface and CPU oracle — only the weights
/// and the RMS-norm γ differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HcKind {
 /// Pre-attention HC collapse + norm (`hc_attn_fn`/`scale`/`base`, `attn_norm` γ).
 Attn,
 /// Pre-FFN HC collapse + norm (`hc_ffn_fn`/`scale`/`base`, `ffn_norm` γ).
 Ffn,
}

pub trait AttentionDispatcher {
 /// Block A3 / G3 (fused). Sinkhorn-weighted HC collapse + RMS norm
 /// in one call.
 ///
 /// The per-token mix is computed inside: `mix = hc_fn @ rms_norm_plain(prev_hc)`,
 /// then the 24-element buffer is decomposed into `pre[0..4]` (sigmoid scale),
 /// `post[4..8]` (2·sigmoid scale), and `comb[8..24]` (n_hc² → softmax →
 /// sinkhorn-20-iters) per `dsv4_hc.metal:402-474`. The HC residual is
 /// reduced into `cur` using `pre` weights, then RMS-normed with `use_gamma`.
 ///
 /// Inputs:
 /// - `kind`: which of the two per-layer HC calls this is.
 /// - `hc_fn`: F16 projection weight `[hc_dim, mix_hc=24]`, row-major (flat).
 /// - `hc_scale`: `[3]` scalars (pre/post/comb).
 /// - `hc_base`: `[mix_hc=24]` per-section additive base.
 /// - `prev_hc`: residual carry from prior block `[hc_dim]`.
 /// - `use_gamma`: γ for the post-collapse RMS norm.
 ///
 /// Output:
 /// - `(cur, normed, hc_split_out[mix_hc=24])`.
 fn hc_collapse_norm(
 &self,
 params: &LayerParams,
 kind: HcKind,
 hc_fn: &[f32],
 hc_scale: &[f32],
 hc_base: &[f32],
 prev_hc: &[f32],
 use_gamma: Option<&[f32]>,
 ) -> (Vec<f32>, Vec<f32>, Vec<f32>);

 /// Block B3 (fused). Joint RMS norm over (qr[..n_lora_q], kv_raw[..n_lora_kv]).
 /// RoPE-tail section is NOT normalized — kept raw for the subsequent
 /// `rope_tail` call.
 fn qkv_rms_norm_rows(
 &self,
 params: &LayerParams,
 qr: &[f32],
 kv_raw: &[f32],
 gamma_q: &[f32],
 gamma_kv: &[f32],
 ) -> (Vec<f32>, Vec<f32>);

 /// Block B3 (kv-only Phase C.3a.3). The qr arm of
 /// `qkv_rms_norm_rows` is dead in the decode hot path (the caller
 /// supplies qr_normed directly via `KernelDispatcher::layer_qa_rms_batched`
 /// and discards the qkv arm's qr_normed output). This method normalises
 /// only the KV-down row, eliminating the dead qr compute + readback on
 /// the Metal path. Default impl delegates to `qkv_rms_norm_rows` with a
 /// throwaway qr so existing impls (CpuAttentionDispatcher, MetalDispatcher)
 /// keep working until they choose to override.
 fn kv_rms_norm_row(&self, params: &LayerParams, kv_raw: &[f32], gamma_kv: &[f32]) -> Vec<f32> {
 let n = kv_raw.len();
 debug_assert_eq!(gamma_kv.len(), n);
 let ss: f64 = kv_raw.iter().map(|&v| v as f64 * v as f64).sum();
 let scale = 1.0f32 / ((ss / n as f64) as f32 + params.rms_eps).sqrt();
 kv_raw
 .iter()
 .zip(gamma_kv.iter())
 .map(|(&x, &g)| x * scale * g)
 .collect()
 }

 /// Block C3 / C4 / D3c / E2. YaRN-scaled RoPE applied to the trailing
 /// `n_rot` floats of each head (or whatever slice the caller passes).
 /// Mutates `x` in place. `backward = true` selects the post-attention
 /// (kqv_back) rotation which uses the compressed-context flag
 /// (antirez `compressed ? ORIG_CTX : 0`).
 fn rope_tail(&self, params: &LayerParams, x: &mut [f32], pos: u32, backward: bool);

 /// B Slice 4. Fuse `kv_rms_norm_row` and
 /// `rope_tail(KV tail)` into one dispatcher call. Returns the
 /// `[n_lora_kv]` row with the trailing `n_rot` floats rotated.
 /// Default impl runs the two trait methods sequentially so semantics
 /// are unchanged on every backend; `MetalDispatcher` overrides to
 /// pack both ops into one `MTLCommandBuffer` (saves 1 commit+wait
 /// per layer per token, 43 cbs / token on DS4 V4 Flash).
 fn kv_norm_rope_batched(
 &self,
 params: &LayerParams,
 kv_raw_row: &[f32],
 qkv_gamma_kv: &[f32],
 pos: u32,
 ) -> Vec<f32> {
 let n_lora_kv = params.n_lora_kv as usize;
 let n_rot = params.n_rot as usize;
 let mut kv_normed = self.kv_rms_norm_row(params, kv_raw_row, qkv_gamma_kv);
 self.rope_tail(
 params,
 &mut kv_normed[n_lora_kv - n_rot..n_lora_kv],
 pos,
 false,
 );
 kv_normed
 }

 /// Block D1. FP8-quantize the produced KV row and append it to the
 /// raw KV cache at `view.pos`.
 fn kv_fp8_store(&self, params: &LayerParams, kv_row_f32: &[f32], view: &mut KvCacheView<'_>);

 /// Block E1. Decode-time flash attention.
 ///
 /// Reads `q` (heads × head_dim), scans raw cache rows ending at `pos`
 /// (SWA window), optionally merges compressed-cache + indexer top-k
 /// (when `comp_selected` is provided), reads `attn_sinks` for the
 /// stable-softmax sink.
 ///
 /// Returns `heads` (n_head × head_dim, row-major) — input to the
 /// post-attention RoPE + W_o projection.
 fn flash_attn_decode(
 &self,
 params: &LayerParams,
 q: &[f32],
 kv_raw: &[f32],
 n_raw: u32,
 raw_cap: u32,
 raw_start: u32,
 kv_comp: Option<&[f32]>,
 n_comp: u32,
 comp_selected: Option<&[u32]>,
 n_selected: u32,
 attn_sinks: &[f32],
 ) -> AttnHeadsOut;

 /// Block F1 (fused). W_o low-rank (attn_output_a + attn_output_b) +
 /// HC expand, writes the post-attention HC residual.
 ///
 /// Inputs:
 /// - `heads`: output of `flash_attn_decode`.
 /// - `w_o_a`, `w_o_b`: Q8-quantized weight views (CPU dispatcher
 /// dequantizes; Metal reads packed). For trait clarity we pass
 /// f32 here — `LayerView` (#213) supplies them.
 /// - `cur_hc`: pre-attn HC residual to combine.
 /// - `hc_split_post`: `post[n_hc..2*n_hc]` section of the pre-attention
 /// `hc_split` produced by `hc_collapse_norm(HcKind::Attn)`. Matches
 /// antirez `hc_post_one` arg 4 (`post`, ds4.c:4217-4237).
 /// - `hc_split_comb`: `comb[2*n_hc..2*n_hc + n_hc²]` n_hc×n_hc mix
 /// matrix from the same `hc_split`. Antirez `hc_post_one`
 /// (ds4.c:4225-4236) folds residual via `Σ_src comb[dst,src] *
 /// residual_hc[src,d]` rather than identity copy — required for
 /// stable residual norms across 43 layers.
 ///
 /// Output: `after_attn_hc` of shape `(hc_dim,)`.
 fn attn_output_proj(
 &self,
 params: &LayerParams,
 heads: &[f32],
 w_o_a: &[f32],
 w_o_b: &[f32],
 cur_hc: &[f32],
 hc_split_post: &[f32],
 hc_split_comb: &[f32],
 ) -> Vec<f32>;

 /// Block J1 (fused). Shared-expert gate + up + silu·clamp Q8.
 /// Inputs/outputs are f32; `w_gate`, `w_up` come from the layer view.
 fn shared_expert(
 &self,
 params: &LayerParams,
 ffn_norm: &[f32],
 w_gate: &[f32],
 w_up: &[f32],
 shared_dim: u32,
 clamp: f32,
 ) -> Vec<f32>;

 /// Block K1 (fused). Shared-expert down + HC expand + add routed/shared
 /// into the next-layer residual.
 ///
 /// `hc_split_post` is the `post[n_hc..2*n_hc]` slice of the pre-FFN
 /// `hc_split` produced by `hc_collapse_norm(HcKind::Ffn)`.
 /// `hc_split_comb` is the `comb[2*n_hc..2*n_hc + n_hc²]` n_hc×n_hc
 /// mix matrix from the same `hc_split` — same role as in
 /// `attn_output_proj` (antirez `hc_post_one`, ds4.c:4225-4236).
 ///
 /// Returns `after_ffn_hc` of shape `(hc_dim,)`.
 fn shared_down_hc_expand_add(
 &self,
 params: &LayerParams,
 shared_mid: &[f32],
 w_down: &[f32],
 routed_out: &[f32],
 after_attn_hc: &[f32],
 hc_split_post: &[f32],
 hc_split_comb: &[f32],
 ) -> Vec<f32>;

 /// batched form of Span A (the 6-op attention
 /// prefix: `hc_collapse_norm(Attn) → kv_rms_norm_row → rope_tail(kv) →
 /// kv_fp8_store → rope_tail(q) [CPU, precomputed table] →
 /// flash_attn_decode → rope_tail(q back) → attn_output_proj`).
 ///
 /// The default impl runs the same 6 trait calls sequentially in the
 /// same order as `decode_attn_prefix_with`, returning the same tuple.
 /// Backends override this to encode all GPU ops into ONE
 /// `MTLCommandBuffer` (1 commit+wait vs current ~6 per layer).
 ///
 /// Bit-identity gate: `tests/attn_prefix_batched_smoke.rs` asserts the
 /// default impl on `CpuAttentionDispatcher` returns the same outputs
 /// as the sequential body in `decode_attn_prefix_with`.
 #[allow(clippy::too_many_arguments)]
 fn attn_prefix_batched(
 &self,
 params: &LayerParams,
 // hc_collapse_norm(Attn) inputs
 hc_attn_fn: &[f32],
 hc_attn_scale: &[f32],
 hc_attn_base: &[f32],
 cur_hc: &[f32],
 hc_norm_gamma: &[f32],
 // kv inputs
 kv_raw_row: &[f32],
 qkv_gamma_kv: &[f32],
 // rope/kv-store inputs
 pos: u32,
 kv_view: &mut KvCacheView<'_>,
 // q-side inputs
 q_heads: &[f32],
 // flash_attn inputs
 n_raw: u32,
 raw_start: u32,
 kv_comp_rows: Option<&[f32]>,
 n_comp: u32,
 comp_selected: Option<&[u32]>,
 n_selected: u32,
 attn_sinks: &[f32],
 // attn_output_proj inputs
 w_o_a: &[f32],
 w_o_b: &[f32],
 ) -> AttnPrefixBatchedOut {
 let p = params;
 let n_rot = p.n_rot as usize;
 let n_lora_kv = p.n_lora_kv as usize;
 let n_hc = p.n_hc as usize;

 // 1. hc_collapse_norm(Attn)
 let (_cur_collapsed, normed, hc_split_attn) = self.hc_collapse_norm(
 p,
 HcKind::Attn,
 hc_attn_fn,
 hc_attn_scale,
 hc_attn_base,
 cur_hc,
 Some(hc_norm_gamma),
 );
 let hc_split_post_attn = hc_split_attn[n_hc..2 * n_hc].to_vec();
 let hc_split_comb_attn = hc_split_attn[2 * n_hc..2 * n_hc + n_hc * n_hc].to_vec();

 // 2. kv_rms_norm_row
 // NOTE: this stays the trait's default impl (CPU f64 reduction)
 // even on Metal because the row is tiny (n_lora_kv ~ 576) and
 // CPU rms_norm of that size beats the cb-launch overhead of a
 // Metal kernel. The fused `kv_norm_rope_batched` exists as
 // infrastructure (see deferred.rs) but is not on the prefix
 // critical path — benchmarking on DS4 V4 Flash showed using it
 // here REGRESSES tok/s by ~45% (0.367 → 0.195 at n_decode=2),
 // because it forces a Metal rms_norm that's slower than CPU
 // for this row size.
 let mut kv_normed = self.kv_rms_norm_row(p, kv_raw_row, qkv_gamma_kv);

 // 3. rope_tail on KV
 self.rope_tail(p, &mut kv_normed[n_lora_kv - n_rot..n_lora_kv], pos, false);

 // 4. kv_fp8_store
 self.kv_fp8_store(p, &kv_normed, kv_view);

 // 5. rope_tail on Q heads (CPU precomputed table — ).
 let mut q_heads_rot = q_heads.to_vec();
 if n_rot > 0 {
 let head_dim = p.head_dim as usize;
 let table = precompute_rope_tail_table(
 n_rot,
 pos,
 false,
 p.rope_freq_base,
 p.rope_freq_scale,
 p.rope_ext_factor,
 p.rope_attn_factor,
 p.rope_orig_ctx,
 );
 debug_assert_eq!(q_heads_rot.len() % head_dim, 0);
 for head in q_heads_rot.chunks_mut(head_dim) {
 let tail = &mut head[head_dim - n_rot..];
 apply_rope_tail_with_table(tail, &table, n_rot);
 }
 }

 // 6. flash_attn_decode
 let heads = self.flash_attn_decode(
 p,
 &q_heads_rot,
 kv_view.raw,
 n_raw,
 kv_view.raw_cap,
 raw_start,
 kv_comp_rows,
 n_comp,
 comp_selected,
 n_selected,
 attn_sinks,
 );

 // 7. rope_tail backward on output heads.
 let mut heads_back = heads;
 if n_rot > 0 {
 let head_dim = p.head_dim as usize;
 let table = precompute_rope_tail_table(
 n_rot,
 pos,
 true,
 p.rope_freq_base,
 p.rope_freq_scale,
 p.rope_ext_factor,
 p.rope_attn_factor,
 p.rope_orig_ctx,
 );
 for head in heads_back.chunks_mut(head_dim) {
 let tail = &mut head[head_dim - n_rot..];
 apply_rope_tail_with_table(tail, &table, n_rot);
 }
 }

 // 8. attn_output_proj
 let after_attn_hc = self.attn_output_proj(
 p,
 &heads_back,
 w_o_a,
 w_o_b,
 cur_hc,
 &hc_split_post_attn,
 &hc_split_comb_attn,
 );

 AttnPrefixBatchedOut {
 after_attn_hc,
 hc_split_attn,
 normed,
 q_heads_rot,
 heads_back,
 kv_normed,
 }
 }
}

/// Output of `AttentionDispatcher::attn_prefix_batched` ( Phase D
/// Slice 1). Carries all intermediates the caller needs to feed the MoE
/// router half + the suffix path. `q_heads_rot` / `heads_back` /
/// `kv_normed` are included so dump-gated diagnostics in
/// `decode_attn_prefix_with` keep working under the batched path.
#[derive(Debug, Clone)]
pub struct AttnPrefixBatchedOut {
 pub after_attn_hc: Vec<f32>,
 pub hc_split_attn: Vec<f32>,
 pub normed: Vec<f32>,
 pub q_heads_rot: Vec<f32>,
 pub heads_back: Vec<f32>,
 pub kv_normed: Vec<f32>,
}

// ---------------------------------------------------------------------------
// CpuAttentionDispatcher — scalar Rust reference for trace-equality testing.
//
// These impls are deliberately straightforward: they exist to validate the
// trait shape, drive trace harnesses, and act as the oracle the Metal impl
// must match within 1e-5. They do NOT need to be fast.
// ---------------------------------------------------------------------------

/// FP8-E4M3FN nearest representable value for `i ∈ [0, 126]` (antirez
/// `dsv4_e4m3fn_value_cpu`, ds4.c:1441).
#[inline]
fn ds4_e4m3fn_value(i: i32) -> f32 {
 const EXP_SCALE: [f32; 16] = [
 0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
 128.0, 256.0,
 ];
 let exp = (i >> 3) & 0x0f;
 let mant = i & 0x07;
 if exp == 0 {
 (mant as f32) * 0.001_953_125
 } else {
 (1.0 + (mant as f32) * 0.125) * EXP_SCALE[exp as usize]
 }
}

/// Round-trip `x` through the FP8-E4M3FN grid (antirez
/// `dsv4_e4m3fn_dequant_cpu`, ds4.c:1456-1481).
fn ds4_e4m3fn_round_trip(x: f32) -> f32 {
 let sign = if x < 0.0 { -1.0 } else { 1.0 };
 let ax = x.abs().min(448.0);
 let mut lo: i32 = 0;
 let mut hi: i32 = 126;
 while lo < hi {
 let mid = (lo + hi + 1) >> 1;
 if ds4_e4m3fn_value(mid) <= ax {
 lo = mid;
 } else {
 hi = mid - 1;
 }
 }
 let mut best = lo;
 if best < 126 {
 let best_diff = (ax - ds4_e4m3fn_value(best)).abs();
 let next_diff = (ax - ds4_e4m3fn_value(best + 1)).abs();
 if next_diff < best_diff
 || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)
 {
 best += 1;
 }
 }
 sign * ds4_e4m3fn_value(best)
}

/// Per-64-block max-scaled FP8-E4M3FN on the NOPE prefix of a KV row.
/// Mirrors antirez `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:1486-1504).
/// Only operates on `x[..head_dim - n_rot]` (the NOPE region); the rope tail
/// is left as f32.
pub fn ds4_fp8_kv_quantize_row_inplace(x: &mut [f32], head_dim: usize, n_rot: usize) {
 let n_nope = head_dim - n_rot;
 let mut off = 0usize;
 while off < n_nope {
 let end = (off + 64).min(n_nope);
 let mut amax = 0.0f32;
 for i in off..end {
 let av = x[i].abs();
 if av > amax {
 amax = av;
 }
 }
 if amax < 1.0e-4 {
 amax = 1.0e-4;
 }
 // scale = 2 ^ ceil(log2(amax / 448))
 let exp_arg = (amax / 448.0_f32).log2().ceil();
 let scale = exp_arg.exp2();
 for i in off..end {
 let mut v = x[i] / scale;
 if v > 448.0 {
 v = 448.0;
 }
 if v < -448.0 {
 v = -448.0;
 }
 x[i] = ds4_e4m3fn_round_trip(v) * scale;
 }
 off += 64;
 }
}

/// Round-trip `x` through IEEE-754 binary16 (half precision) precision —
/// mirrors antirez `kv_cache_push_raw` (ds4.c:6084-6097), which stores every
/// KV cache row as `f16_to_f32(f32_to_f16(value))`.
///
/// Implementation uses two well-tested branchless conversions:
/// 1. `f32_to_f16_bits(x)` — F::FloatX::to_bits-equivalent for half.
/// 2. `f16_bits_to_f32(h)` — half::f16::to_f32-equivalent.
#[inline]
pub fn f16_round_trip_f32(x: f32) -> f32 {
 f16_bits_to_f32(f32_to_f16_bits(x))
}

/// f32 → IEEE-754 binary16 bits (round-to-nearest-even).
/// Algorithm: ARM CMSIS / mini-mlx style, branched but correct.
fn f32_to_f16_bits(value: f32) -> u16 {
 let bits = value.to_bits();
 let sign: u16 = ((bits >> 16) & 0x8000) as u16;
 let exp: i32 = ((bits >> 23) & 0xff) as i32 - 127 + 15;
 let mant: u32 = bits & 0x7f_ffff;

 if exp == 143 - 15 + 15 {
 // f32 exp == 0xff: NaN or Inf (143 is wrong — recompute)
 // (Re-derive: exp32 == 0xff means exp == 0xff - 127 + 15 == 143.)
 }
 // f32 NaN/Inf when exp32 == 0xff, i.e. `exp == 143`.
 if (bits >> 23) & 0xff == 0xff {
 if mant != 0 {
 // NaN — set quiet bit
 return sign | 0x7e00;
 }
 // Inf
 return sign | 0x7c00;
 }

 if exp >= 0x1f {
 // Overflow → Inf
 return sign | 0x7c00;
 }

 if exp <= 0 {
 if exp < -10 {
 // Too small to represent even as subnormal → 0
 return sign;
 }
 // Subnormal half. Add implicit 1, shift to align.
 let mant_full: u32 = mant | 0x0080_0000;
 let shift: i32 = 14 - exp;
 let result: u32 = mant_full >> shift;
 // Round-to-nearest-even
 let round_bit_pos = shift - 1;
 let round_bit = (mant_full >> round_bit_pos) & 1;
 let sticky_mask = (1u32 << round_bit_pos) - 1;
 let sticky = (mant_full & sticky_mask) != 0;
 let mut h = result;
 if round_bit == 1 && (sticky || (h & 1) == 1) {
 h += 1;
 }
 return sign | (h as u16);
 }

 // Normal half.
 let h_mant: u32 = mant >> 13;
 let rem: u32 = mant & 0x1fff;
 let mut h_exp: u32 = exp as u32;
 let mut h_mant_rounded: u32 = h_mant;
 if rem > 0x1000 || (rem == 0x1000 && (h_mant & 1) == 1) {
 h_mant_rounded += 1;
 if h_mant_rounded == 0x400 {
 h_mant_rounded = 0;
 h_exp += 1;
 if h_exp >= 0x1f {
 // Mantissa overflow pushed exp to overflow → Inf
 return sign | 0x7c00;
 }
 }
 }
 sign | ((h_exp << 10) as u16) | (h_mant_rounded as u16)
}

/// IEEE-754 binary16 bits → f32.
fn f16_bits_to_f32(h: u16) -> f32 {
 let h = h as u32;
 let sign: u32 = (h & 0x8000) << 16;
 let exp: u32 = (h >> 10) & 0x1f;
 let mant: u32 = h & 0x3ff;

 let out_bits: u32 = if exp == 0 {
 if mant == 0 {
 sign
 } else {
 // Subnormal half → normalised f32
 let mut m: u32 = mant;
 let mut e: i32 = -1;
 while (m & 0x400) == 0 {
 m <<= 1;
 e -= 1;
 }
 m &= 0x3ff;
 let f_exp: u32 = (e + 1 + 127 - 15) as u32;
 sign | (f_exp << 23) | (m << 13)
 }
 } else if exp == 0x1f {
 // Inf or NaN
 sign | 0x7f80_0000 | (mant << 13)
 } else {
 sign | ((exp + 127 - 15) << 23) | (mant << 13)
 };
 f32::from_bits(out_bits)
}

/// Scalar Rust reference.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuAttentionDispatcher;

impl CpuAttentionDispatcher {
 fn rms_norm_inplace(eps: f32, x: &mut [f32], gamma: Option<&[f32]>) {
 // antirez `rms_norm_no_weight` / `rms_norm_weight` (ds4.c:2551-2566)
 // uses an f64 accumulator for the sum-of-squares, then casts to f32 BEFORE
 // adding eps and taking sqrtf. Match this exactly for byte-identical output
 // — f32 accumulator was losing ~10 bits of precision at d_embd=4096.
 let n = x.len();
 let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
 let scale = 1.0f32 / ((ss / n as f64) as f32 + eps).sqrt();
 match gamma {
 Some(g) => {
 debug_assert_eq!(g.len(), x.len());
 for (xi, gi) in x.iter_mut().zip(g.iter()) {
 *xi = *xi * scale * *gi;
 }
 }
 None => {
 for xi in x.iter_mut() {
 *xi *= scale;
 }
 }
 }
 }
}

impl AttentionDispatcher for CpuAttentionDispatcher {
 fn hc_collapse_norm(
 &self,
 params: &LayerParams,
 _kind: HcKind,
 hc_fn: &[f32],
 hc_scale: &[f32],
 hc_base: &[f32],
 prev_hc: &[f32],
 use_gamma: Option<&[f32]>,
 ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
 // CPU oracle mirrors `kernel_dsv4_hc_split_weighted_sum_norm4`
 // (`upstream/ds4/metal/dsv4_hc.metal:371-490`).
 //
 // Steps:
 // a. flat = rms_norm_plain(prev_hc) — uniform RMS, no gamma, over hc_dim.
 // b. mix = hc_fn @ flat → [mix_hc = 2*n_hc + n_hc²].
 // c. split[0..n_hc] = sigmoid(mix[0..n_hc] * scale[0] + base[0..n_hc]) + eps
 // d. split[n_hc..2n_hc] = 2·sigmoid(mix[..]*scale[1] + base[..])
 // e. split[2n_hc..] = sinkhorn-iter(softmax(mix[..]*scale[2] + base[..]))
 // f. cur = Σ_h split_pre[h] · prev_hc[h*d_embd .. (h+1)*d_embd]
 // g. normed = rms_norm(cur, use_gamma)
 //
 // The fused kernel writes the full split[mix_hc] buffer; we return it
 // unchanged so downstream HC-expand sites can index pre/post sections.
 let n_hc = params.n_hc as usize;
 let d_embd = params.d_embd as usize;
 let hc_dim = n_hc * d_embd;
 let mix_hc = 2 * n_hc + n_hc * n_hc;
 debug_assert_eq!(prev_hc.len(), hc_dim);
 debug_assert_eq!(hc_fn.len(), hc_dim * mix_hc);
 debug_assert_eq!(hc_scale.len(), 3);
 debug_assert_eq!(hc_base.len(), mix_hc);
 let eps = params.hc_eps;

 // a. flat = rms_norm_plain(prev_hc, hc_dim) — no gamma.
 let mut flat = prev_hc.to_vec();
 Self::rms_norm_inplace(params.rms_eps, &mut flat, None);

 // dump per-slot prev_hc and flat RMS to compare with antirez.
 // gate by pos when DS4_DUMP_HC_AT_POS is set, to avoid mixing
 // prefill positions into the decode-time comparison. `CURRENT_POS_HINT`
 // is set by `decode_step_with_attn` before each step.
 let current_pos = CURRENT_POS_HINT.with(|c| c.get());
 let pos_gate_ok = match std::env::var("DS4_DUMP_HC_AT_POS")
 .ok()
 .and_then(|s| s.parse::<u32>().ok())
 {
 Some(target) => current_pos == target,
 None => true,
 };
 if std::env::var("DS4_DUMP_HC_INTERNALS").is_ok() && params.layer_idx <= 1 && pos_gate_ok {
 let mut prev_rms = Vec::with_capacity(n_hc);
 let mut flat_rms_slot = Vec::with_capacity(n_hc);
 for h in 0..n_hc {
 let p_slot = &prev_hc[h * d_embd..(h + 1) * d_embd];
 let f_slot = &flat[h * d_embd..(h + 1) * d_embd];
 let pr = (p_slot.iter().map(|v| v * v).sum::<f32>() / d_embd as f32).sqrt();
 let fr = (f_slot.iter().map(|v| v * v).sum::<f32>() / d_embd as f32).sqrt();
 prev_rms.push(pr);
 flat_rms_slot.push(fr);
 }
 eprintln!(
 "HC_INPUT_RMS il={} pos={} kind={:?} prev_per_slot=[{:.6},{:.6},{:.6},{:.6}] flat_per_slot=[{:.6},{:.6},{:.6},{:.6}] prev0_first3=[{:.4},{:.4},{:.4}] flat0_first3=[{:.4},{:.4},{:.4}]",
 params.layer_idx,
 current_pos,
 _kind,
 prev_rms[0],
 prev_rms[1],
 prev_rms[2],
 prev_rms[3],
 flat_rms_slot[0],
 flat_rms_slot[1],
 flat_rms_slot[2],
 flat_rms_slot[3],
 prev_hc[0],
 prev_hc[1],
 prev_hc[2],
 flat[0],
 flat[1],
 flat[2],
 );
 }

 // b. mix[r] = Σ_i hc_fn[r*hc_dim + i] · flat[i] for r in 0..mix_hc.
 // antirez `matvec_f16` (ds4.c:2620) → `dot_f16_row`
 // (ds4.c:2587) uses 2×4-lane FMA partials + pairwise add via
 // `vaddvq_f32(vaddq_f32(...))`. Our left-fold sum is the same
 // reduction-tree class as gate behind the
 // existing `DS4_MATVEC_F32_FIDELITY=1` env (no new toggle).
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
 let mut mix = vec![0.0f32; mix_hc];
 for r in 0..mix_hc {
 let row = &hc_fn[r * hc_dim..(r + 1) * hc_dim];
 mix[r] = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(row, &flat)
 } else {
 row.iter().zip(flat.iter()).map(|(a, b)| a * b).sum()
 };
 }

 let pre_scale = hc_scale[0];
 let post_scale = hc_scale[1];
 let comb_scale = hc_scale[2];

 let mut split = vec![0.0f32; mix_hc];

 // c. pre = sigmoid(mix[0..n_hc] * pre_scale + base[0..n_hc]) + eps.
 for i in 0..n_hc {
 let z = mix[i] * pre_scale + hc_base[i];
 split[i] = 1.0 / (1.0 + (-z).exp()) + eps;
 }

 // d. post = 2 · sigmoid(mix[n_hc..2n_hc] * post_scale + base[..]).
 for i in 0..n_hc {
 let z = mix[n_hc + i] * post_scale + hc_base[n_hc + i];
 split[n_hc + i] = 2.0 / (1.0 + (-z).exp());
 }

 // dump per-slot mix/base/z for post when env-gated. Compares against
 // antirez sinkhorn input (ds4.c:4054-4058) to localise why post=[0,0,0,0].
 if std::env::var("DS4_DUMP_HC_INTERNALS").is_ok() && params.layer_idx <= 1 && pos_gate_ok {
 let flat_rms: f32 =
 (flat.iter().map(|v| (v as &f32) * v).sum::<f32>() / flat.len() as f32).sqrt();
 eprintln!(
 "HC_INTERNALS il={} pos={} kind={:?} flat_rms={:.4} pre_scale={:.6} post_scale={:.6} comb_scale={:.6} sinkhorn_iter={} mix_pre=[{:.4},{:.4},{:.4},{:.4}] base_pre=[{:.4},{:.4},{:.4},{:.4}] mix_post=[{:.4},{:.4},{:.4},{:.4}] base_post=[{:.4},{:.4},{:.4},{:.4}] split_post=[{:.6},{:.6},{:.6},{:.6}]",
 params.layer_idx,
 current_pos,
 _kind,
 flat_rms,
 pre_scale,
 post_scale,
 comb_scale,
 params.hc_sinkhorn_iter,
 mix[0],
 mix[1],
 mix[2],
 mix[3],
 hc_base[0],
 hc_base[1],
 hc_base[2],
 hc_base[3],
 mix[n_hc],
 mix[n_hc + 1],
 mix[n_hc + 2],
 mix[n_hc + 3],
 hc_base[n_hc],
 hc_base[n_hc + 1],
 hc_base[n_hc + 2],
 hc_base[n_hc + 3],
 split[n_hc],
 split[n_hc + 1],
 split[n_hc + 2],
 split[n_hc + 3],
 );
 // Dump comb-section of mix and hc_base (mix_hc=2*n_hc..2*n_hc+n_hc²).
 let mut mix_comb_strs = Vec::new();
 let mut base_comb_strs = Vec::new();
 for r in 0..n_hc {
 let mix_row: Vec<String> = (0..n_hc)
 .map(|c| format!("{:.4}", mix[2 * n_hc + r * n_hc + c]))
 .collect();
 let base_row: Vec<String> = (0..n_hc)
 .map(|c| format!("{:.4}", hc_base[2 * n_hc + r * n_hc + c]))
 .collect();
 mix_comb_strs.push(format!("[{}]", mix_row.join(",")));
 base_comb_strs.push(format!("[{}]", base_row.join(",")));
 }
 eprintln!(
 "HC_COMB_INPUT il={} mix_comb={{{}}} base_comb={{{}}}",
 params.layer_idx,
 mix_comb_strs.join(", "),
 base_comb_strs.join(", ")
 );
 }

 // e. comb = sinkhorn(softmax_rows( reshape(mix[2n_hc..], [n_hc, n_hc]) )).
 let comb_off = 2 * n_hc;
 let mut comb = vec![0.0f32; n_hc * n_hc];
 for i in 0..(n_hc * n_hc) {
 comb[i] = mix[comb_off + i] * comb_scale + hc_base[comb_off + i];
 }
 // Row softmax (numerically stable) + eps.
 for r in 0..n_hc {
 let row = &mut comb[r * n_hc..(r + 1) * n_hc];
 let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
 let mut sum = 0.0f32;
 for v in row.iter_mut() {
 *v = (*v - m).exp();
 sum += *v;
 }
 let inv = 1.0 / sum;
 for v in row.iter_mut() {
 *v = *v * inv + eps;
 }
 }
 // Column scale.
 let mut col_sum = vec![0.0f32; n_hc];
 for r in 0..n_hc {
 for c in 0..n_hc {
 col_sum[c] += comb[r * n_hc + c];
 }
 }
 for c in 0..n_hc {
 let inv = 1.0 / (col_sum[c] + eps);
 for r in 0..n_hc {
 comb[r * n_hc + c] *= inv;
 }
 }
 // Remaining sinkhorn iterations (row then col).
 for _ in 1..params.hc_sinkhorn_iter {
 for r in 0..n_hc {
 let row = &mut comb[r * n_hc..(r + 1) * n_hc];
 let row_sum: f32 = row.iter().sum();
 let inv = 1.0 / (row_sum + eps);
 for v in row.iter_mut() {
 *v *= inv;
 }
 }
 for c in 0..n_hc {
 col_sum[c] = 0.0;
 }
 for r in 0..n_hc {
 for c in 0..n_hc {
 col_sum[c] += comb[r * n_hc + c];
 }
 }
 for c in 0..n_hc {
 let inv = 1.0 / (col_sum[c] + eps);
 for r in 0..n_hc {
 comb[r * n_hc + c] *= inv;
 }
 }
 }
 for i in 0..(n_hc * n_hc) {
 split[comb_off + i] = comb[i];
 }

 // f. cur = Σ_h split_pre[h] · prev_hc[h * d_embd .. (h+1) * d_embd].
 let mut cur = vec![0.0f32; d_embd];
 for h in 0..n_hc {
 let w = split[h];
 let base = h * d_embd;
 for j in 0..d_embd {
 cur[j] += w * prev_hc[base + j];
 }
 }

 // g. RMS norm into `normed` with use_gamma.
 let mut normed = cur.clone();
 Self::rms_norm_inplace(params.rms_eps, &mut normed, use_gamma);

 (cur, normed, split)
 }

 fn qkv_rms_norm_rows(
 &self,
 params: &LayerParams,
 qr: &[f32],
 kv_raw: &[f32],
 gamma_q: &[f32],
 gamma_kv: &[f32],
 ) -> (Vec<f32>, Vec<f32>) {
 // qr: [n_lora_q], normalise full row with gamma_q.
 // kv_raw: [n_lora_kv], normalise the entire row with gamma_kv per
 // antirez `ds4.c:4494` (`rms_norm_weight(kv, raw, kv_norm, DS4_N_HEAD_DIM, ...)`
 // — the rope-tail at the end is normalised before the rope rotation
 // overwrites it).
 let n_lora_q = params.n_lora_q as usize;
 let n_lora_kv = params.n_lora_kv as usize;
 debug_assert_eq!(qr.len(), n_lora_q);
 debug_assert_eq!(kv_raw.len(), n_lora_kv);

 let mut qr_n = qr.to_vec();
 Self::rms_norm_inplace(params.rms_eps, &mut qr_n, Some(gamma_q));

 let mut kv_n = kv_raw.to_vec();
 Self::rms_norm_inplace(params.rms_eps, &mut kv_n, Some(gamma_kv));

 (qr_n, kv_n)
 }

 fn rope_tail(&self, params: &LayerParams, x: &mut [f32], pos: u32, backward: bool) {
 // Port of antirez `rope_tail_ext_inplace` (ds4.c:4545-4593).
 // Callers pre-slice down to `(n_head_iter * n_rot)` already — each
 // chunk of `n_rot` floats is the rope-tail of one head.
 let n_rot = params.n_rot as usize;
 if n_rot == 0 {
 return;
 }
 debug_assert_eq!(x.len() % n_rot, 0);
 let freq_base = params.rope_freq_base;
 let freq_scale = params.rope_freq_scale;
 let ext_factor = params.rope_ext_factor;
 let attn_factor = params.rope_attn_factor;
 let n_ctx_orig = params.rope_orig_ctx as u64;
 let beta_fast = 32.0f32;
 let beta_slow = 1.0f32;
 let theta_scale = freq_base.powf(-2.0 / (n_rot as f32));
 let sin_sign = if backward { -1.0 } else { 1.0 };

 let mut corr_lo = 0.0f32;
 let mut corr_hi = 0.0f32;
 if ext_factor != 0.0 {
 let corr_dim = |b: f32| -> f32 {
 (n_rot as f32) * ((n_ctx_orig as f32) / (b * 2.0 * PI)).ln()
 / (2.0 * freq_base.ln())
 };
 let start = corr_dim(beta_fast).floor();
 let end = corr_dim(beta_slow).ceil();
 corr_lo = start.max(0.0);
 corr_hi = end.min((n_rot as f32) - 1.0);
 }

 for tail in x.chunks_mut(n_rot) {
 let mut theta_extrap = pos as f32;
 let mut i = 0usize;
 while i + 1 < n_rot {
 let theta_interp = freq_scale * theta_extrap;
 let mut theta = theta_interp;
 let mut mscale = attn_factor;
 if ext_factor != 0.0 {
 let y = ((i as f32 / 2.0) - corr_lo) / (corr_hi - corr_lo).max(0.001);
 let ramp = 1.0 - y.clamp(0.0, 1.0);
 let ramp_mix = ramp * ext_factor;
 theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
 if freq_scale > 0.0 {
 mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
 }
 }
 let c = theta.cos() * mscale;
 let s = sin_sign * theta.sin() * mscale;
 let x0 = tail[i];
 let x1 = tail[i + 1];
 tail[i] = x0 * c - x1 * s;
 tail[i + 1] = x0 * s + x1 * c;
 theta_extrap *= theta_scale;
 i += 2;
 }
 }
 }

 fn kv_fp8_store(&self, params: &LayerParams, kv_row_f32: &[f32], view: &mut KvCacheView<'_>) {
 let row = params.n_lora_kv as usize;
 debug_assert_eq!(kv_row_f32.len(), row);
 let slot = (view.pos % view.raw_cap) as usize;
 let off = slot * row;
 debug_assert!(off + row <= view.raw.len());
 // Mirror antirez ds4.c:7608-7609 — the KV row goes through TWO
 // quantizations before reaching the cache:
 // 1. `dsv4_fp8_kv_quantize_row_inplace_cpu(kv, head_dim, n_rot)`
 // — per-64-block max-scaled FP8-E4M3FN on the NOPE prefix.
 // 2. `f16_round_inplace_cpu(kv, head_dim)` — IEEE 754 f16 round-trip
 // on the whole row (NOPE+ROPE).
 // The fp8 NOPE step was previously skipped because it didn't move
 // pos=0 argmax (only 1 KV row → softmax-identity). But at pos≥15 with
 // many accumulated KV rows, the per-block fp8 precision matters, so
 // we now apply both steps end-to-end.
 let dest = &mut view.raw[off..off + row];
 for (d, &v) in dest.iter_mut().zip(kv_row_f32.iter()) {
 *d = v;
 }
 let head_dim_u = params.head_dim as usize;
 let n_rot_u = params.n_rot as usize;
 // The NOPE prefix is `head_dim - n_rot` floats at the start of the row.
 // Only quantize when head_dim >= n_rot (else nothing to do).
 //
 // gated. The "silent at pos=0" argument assumed Q=K
 // in the dot product (so quant noise cancels). DS4 MLA builds Q from
 // a separate q_a/q_b LoRA path, so K-side FP8 noise perturbs the
 // single-row softmax weight at pos=0 and regresses the gate (saw
 // signature flip 1162/455 → 2581/943 on "What is 2+2?"). Opt-in via
 // DS4_FP8_KV_QUANT=1 once we can show it improves pos>=15 without
 // breaking pos=0.
 let want_fp8 = std::env::var("DS4_FP8_KV_QUANT").ok().as_deref() == Some("1");
 if head_dim_u > n_rot_u && want_fp8 {
 ds4_fp8_kv_quantize_row_inplace(dest, head_dim_u, n_rot_u);
 }
 // DS4_F32_KV=1 skips the f16 round-trip on the KV row. This
 // eliminates per-write half-precision rounding noise — used to test
 // whether pos=15 argmax drift is dominated by f16 KV quantization.
 let want_f32_kv = std::env::var("DS4_F32_KV").ok().as_deref() == Some("1");
 if !want_f32_kv {
 for d in dest.iter_mut() {
 *d = f16_round_trip_f32(*d);
 }
 }
 view.pos = view.pos.saturating_add(1);
 }

 fn flash_attn_decode(
 &self,
 params: &LayerParams,
 q: &[f32],
 kv_raw: &[f32],
 n_raw: u32,
 raw_cap: u32,
 raw_start: u32,
 kv_comp: Option<&[f32]>,
 n_comp: u32,
 comp_selected: Option<&[u32]>,
 n_selected: u32,
 attn_sinks: &[f32],
 ) -> AttnHeadsOut {
 // Reference scalar attention. For each head:
 // - compute scaled dot with each candidate K row (raw window +
 // optional compressed top-k),
 // - stable-softmax with sink prepended,
 // - weighted sum of V rows.
 // KV-cache row layout is [n_lora_kv] (== head_dim); we treat the
 // same row as both K and V (MLA cache stores latent). The rope
 // tail occupies the last `n_rot` slots inside the row.
 let n_head = params.n_head as usize;
 let head_dim = params.head_dim as usize;
 let row = params.n_lora_kv as usize;
 debug_assert_eq!(q.len(), n_head * head_dim);
 debug_assert_eq!(attn_sinks.len(), n_head);

 let mut out = vec![0.0f32; n_head * head_dim];
 let scale = (head_dim as f32).sqrt().recip();

 // DS4_DUMP_FLASH_Q: dump flash inputs (q head0, kv rows 0 + last) to
 // compare against the verifier's flash_attn_k_mla inputs.
 if std::env::var("DS4_DUMP_FLASH_Q").is_ok() && n_raw > 0 {
 let rd = |s: &[f32]| -> (f64, Vec<f32>) {
 let ss: f64 = s.iter().map(|&v| (v as f64) * (v as f64)).sum();
 ((ss / s.len() as f64).sqrt(), s[..4.min(s.len())].to_vec())
 };
 let (qr, qh) = rd(&q[..head_dim]);
 let r0 = ((raw_start) % raw_cap) as usize;
 let rl = ((raw_start + n_raw - 1) % raw_cap) as usize;
 let (k0r, k0h) = rd(&kv_raw[r0 * row..r0 * row + head_dim.min(row)]);
 let (klr, klh) = rd(&kv_raw[rl * row..rl * row + head_dim.min(row)]);
 eprintln!(" [FLASH-IN cpu] q_head0 rms={:.4} head={:?} | kv[{}] rms={:.4} head={:?} | kv[{}] rms={:.4} head={:?} (n_raw={})",
 qr, qh, r0, k0r, k0h, rl, klr, klh, n_raw);
 }

 // Build a tagged candidate list. Raw rows are read from `kv_raw`
 // (n_lora_kv-wide stride); comp rows are read from `kv_comp`
 // (head_dim-wide stride, no rope tail extension). The previous
 // shared-u32 form double-counted raw rows whenever a comp index
 // happened to fall in [0, raw_cap), which is always true.
 enum Src {
 Raw(u32),
 Comp(u32),
 }
 let mut idx: Vec<Src> = (0..n_raw)
 .map(|i| Src::Raw((raw_start + i) % raw_cap))
 .collect();
 if let (Some(sel), Some(_)) = (comp_selected, kv_comp) {
 for i in 0..n_selected as usize {
 idx.push(Src::Comp(sel[i]));
 }
 } else if kv_comp.is_some() {
 for i in 0..n_comp {
 idx.push(Src::Comp(i));
 }
 }

 // gate Q·K dot on DS4_MATVEC_F32_FIDELITY=1 to switch from
 // Iterator::sum (left-fold f32) to antirez `dot_f32` (NEON-style FMA
 // pair-reduce). Reduction-tree topology divergence — algebraically
 // equal, f32-ULP different. Reuses the env (same divergence
 // class, different callsite).
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");

 let scores_buf_len = idx.len() + 1;
 for h in 0..n_head {
 let q_h = &q[h * head_dim..(h + 1) * head_dim];
 let k_dim_raw = head_dim.min(row);
 let mut scores = Vec::with_capacity(scores_buf_len);
 scores.push(attn_sinks[h]);
 for s_src in &idx {
 let (src, k_dim): (&[f32], usize) = match s_src {
 Src::Raw(ki) => {
 let k_off = (*ki as usize) * row;
 (&kv_raw[k_off..k_off + row], k_dim_raw)
 }
 Src::Comp(ci) => {
 let comp = kv_comp.expect("Src::Comp requires kv_comp");
 let c_off = (*ci as usize) * head_dim;
 (&comp[c_off..c_off + head_dim], head_dim)
 }
 };
 let n_dot = k_dim.min(q_h.len());
 let dot: f32 = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(&q_h[..n_dot], &src[..n_dot])
 } else {
 q_h.iter()
 .zip(src.iter().take(k_dim))
 .map(|(a, b)| a * b)
 .sum()
 };
 scores.push(dot * scale);
 }
 let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
 let mut sum = 0.0f32;
 for s in scores.iter_mut() {
 *s = (*s - mx).exp();
 sum += *s;
 }

 // antirez `layer_attention_rows_one` (ds4.c:4748-4786)
 // accumulates UN-NORMALIZED `weight * kv` via axpy_f32, then
 // applies `scale_f32(oh, 1/denom)` ONCE at end. Pre-dividing
 // each weight by sum (normalize-EARLY) is algebraically
 // equivalent but f32-ULP-different.
 let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
 for (j, s_src) in idx.iter().enumerate() {
 let w = scores[j + 1];
 let (src, v_dim): (&[f32], usize) = match s_src {
 Src::Raw(ki) => {
 let k_off = (*ki as usize) * row;
 (&kv_raw[k_off..k_off + row], head_dim.min(row))
 }
 Src::Comp(ci) => {
 let comp = kv_comp.expect("Src::Comp requires kv_comp");
 let c_off = (*ci as usize) * head_dim;
 (&comp[c_off..c_off + head_dim], head_dim)
 }
 };
 for d in 0..v_dim {
 out_h[d] += w * src[d];
 }
 }
 let inv = sum.recip();
 for d in 0..head_dim {
 out_h[d] *= inv;
 }
 }
 out
 }

 fn attn_output_proj(
 &self,
 params: &LayerParams,
 heads: &[f32],
 w_o_a: &[f32],
 w_o_b: &[f32],
 cur_hc: &[f32],
 hc_split_post: &[f32],
 hc_split_comb: &[f32],
 ) -> Vec<f32> {
 // DS4 V4 Flash uses GROUPED output projection (antirez
 // `matvec_q8_0_grouped_rows`, ds4.c:3396):
 //
 // heads[q_dim] reshaped to [n_groups, group_dim]
 // w_o_a row-major [out_low_dim, group_dim]
 // where out_low_dim = n_groups * n_lora_o
 // row (g * n_lora_o + l) is mat-vec'd
 // against heads[g * group_dim ..]
 // attn_low[out_low_dim] n_lora_o per group
 // w_o_b row-major [d_embd, out_low_dim]
 // attn_out[d_embd] standard matvec
 //
 // Then HC expand-add into the residual.
 let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
 let heads_owned;
 let heads: &[f32] = if want_q80 {
 heads_owned = crate::forward::q8_0_round_trip(heads);
 &heads_owned
 } else {
 heads
 };
 let d_embd = params.d_embd as usize;
 let n_hc = params.n_hc as usize;
 let q_dim = heads.len();
 let n_groups = params.n_out_group as usize;
 debug_assert_eq!(
 q_dim % n_groups,
 0,
 "q_dim must be divisible by n_out_group"
 );
 let group_dim = q_dim / n_groups;
 debug_assert_eq!(w_o_a.len() % (n_groups * group_dim), 0);
 let n_lora_o = w_o_a.len() / (n_groups * group_dim);
 let out_low_dim = n_groups * n_lora_o;
 debug_assert_eq!(w_o_b.len(), d_embd * out_low_dim);

 // Stage 1: grouped matvec. For each (g, l), dot row of w_o_a
 // with the g-th chunk of heads.
 //
 // antirez `matvec_q8_0_grouped_rows` (ds4.c:3396) uses
 // 2×4-lane FMA pair-reduce. Same reduction-tree class as
 // Gate behind DS4_MATVEC_F32_FIDELITY=1.
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
 let mut attn_low = vec![0.0f32; out_low_dim];
 for g in 0..n_groups {
 let heads_g = &heads[g * group_dim..(g + 1) * group_dim];
 for l in 0..n_lora_o {
 let row =
 &w_o_a[(g * n_lora_o + l) * group_dim..(g * n_lora_o + l + 1) * group_dim];
 attn_low[g * n_lora_o + l] = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(row, heads_g)
 } else {
 row.iter().zip(heads_g.iter()).map(|(a, b)| a * b).sum()
 };
 }
 }

 // Stage 2: dense matvec w_o_b · attn_low ⇒ [d_embd]
 // antirez `matvec_q8_0(out, attn_output_b, low)` (ds4.c:4813)
 // internally `quantize_q8_0_activation`s `low` before the int8 dot.
 let attn_low_owned;
 let attn_low_in: &[f32] = if want_q80 {
 attn_low_owned = crate::forward::q8_0_round_trip(&attn_low);
 &attn_low_owned
 } else {
 &attn_low
 };
 let mut attn_out = vec![0.0f32; d_embd];
 for e in 0..d_embd {
 let row = &w_o_b[e * out_low_dim..(e + 1) * out_low_dim];
 attn_out[e] = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(row, attn_low_in)
 } else {
 row.iter().zip(attn_low_in.iter()).map(|(a, b)| a * b).sum()
 };
 }

 // Verifier-bisection output-proj tap: capture attn_out (pre-hc_expand).
 ATTN_OUT_CAPTURE.with(|c| {
 let mut cc = c.borrow_mut();
 if cc.0 != usize::MAX && cc.0 == CURRENT_LAYER_HINT.with(|h| h.get()) {
 cc.1 = attn_out.clone();
 }
 });

 // env-gated dump for `attn_out` (post-stage-2, pre-HC-expand)
 // to localise whether L0 slot-3 divergence comes from output_proj or
 // upstream. DS4_DUMP_ATTN_OUT=POS fires only at incoming state.pos == POS.
 {
 let current_pos = CURRENT_POS_HINT.with(|c| c.get());
 let target_pos = std::env::var("DS4_DUMP_ATTN_OUT")
 .ok()
 .and_then(|s| s.parse::<u32>().ok());
 if let Some(p) = target_pos {
 if current_pos == p {
 let ss: f64 = attn_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
 let rms = (ss / attn_out.len() as f64).sqrt();
 let mut ranked: Vec<(usize, f32)> =
 attn_out.iter().copied().enumerate().collect();
 ranked.sort_by(|a, b| {
 b.1.abs()
 .partial_cmp(&a.1.abs())
 .unwrap_or(std::cmp::Ordering::Equal)
 });
 let top3: Vec<(usize, f32)> = ranked.iter().take(3).copied().collect();
 let post: Vec<f32> = hc_split_post.iter().copied().collect();
 let attn_low_rms = {
 let ssl: f64 = attn_low.iter().map(|&v| (v as f64) * (v as f64)).sum();
 (ssl / attn_low.len() as f64).sqrt()
 };
 eprintln!(
 "ATTN_OUT il={} pos={} attn_low_rms={attn_low_rms:.4} attn_out_rms={rms:.4} attn_out_top3={top3:?} hc_split_post={post:?}",
 params.layer_idx, current_pos,
 );
 }
 }
 }

 // HC expand + add per antirez `hc_post_one` (ds4.c:4217-4237):
 // after[dst, e] = attn_out[e] * hc_split_post[dst]
 // + Σ_src hc_split_comb[dst + src*n_hc] * cur_hc[src, e]
 // Comb addressing is `comb[dst + src * n_hc]` (column-major in src).
 debug_assert_eq!(hc_split_post.len(), n_hc);
 debug_assert_eq!(hc_split_comb.len(), n_hc * n_hc);
 debug_assert_eq!(cur_hc.len(), n_hc * d_embd);
 let mut after = vec![0.0f32; n_hc * d_embd];
 for dst in 0..n_hc {
 let base = dst * d_embd;
 let w_post = hc_split_post[dst];
 for e in 0..d_embd {
 let mut acc = w_post * attn_out[e];
 for src in 0..n_hc {
 acc += hc_split_comb[dst + src * n_hc] * cur_hc[src * d_embd + e];
 }
 after[base + e] = acc;
 }
 }
 after
 }

 fn shared_expert(
 &self,
 params: &LayerParams,
 ffn_norm: &[f32],
 w_gate: &[f32],
 w_up: &[f32],
 shared_dim: u32,
 clamp: f32,
 ) -> Vec<f32> {
 // Antirez `layer_shared_ffn_one` (ds4.c:4880-4912) is pure
 // `silu(gate) * up` with NO clamping, and that stays the behaviour
 // whenever the model declares no clamp: `clamp <= 1e-6` degenerates to
 // exactly that expression, so pre-MXFP4 models are bit-identical.
 //
 // The DS4-Flash MXFP4 release DOES declare one, per layer, via
 // `deepseek4.swiglu_clamp_shexp` (= 10.0). Dropping the parameter — as
 // this impl used to — silently disagreed with `kernel_swiglu_f32` and
 // with antirez's own comment in glu.metal: "DS4 clamps the shared expert
 // with the same swiglu_limit used by routed experts".
 let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
 let ffn_owned;
 let ffn_norm: &[f32] = if want_q80 {
 ffn_owned = crate::forward::q8_0_round_trip(ffn_norm);
 &ffn_owned
 } else {
 ffn_norm
 };
 let d_embd = params.d_embd as usize;
 let sd = shared_dim as usize;
 debug_assert_eq!(w_gate.len(), sd * d_embd);
 debug_assert_eq!(w_up.len(), sd * d_embd);

 // gate left-fold dots (same reduction-tree class as
 // callsite — reuse DS4_MATVEC_F32_FIDELITY)
 // and replace inline silu with `crate::forward::silu` so the
 // DS4_SILU_FIDELITY=1 toggle reaches the shared expert.
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");

 let mut out = vec![0.0f32; sd];
 for i in 0..sd {
 let row_g = &w_gate[i * d_embd..(i + 1) * d_embd];
 let row_u = &w_up[i * d_embd..(i + 1) * d_embd];
 let (g, u): (f32, f32) = if antirez_dot_fidelity {
 (
 crate::forward::dot_f32_antirez(row_g, ffn_norm),
 crate::forward::dot_f32_antirez(row_u, ffn_norm),
 )
 } else {
 (
 row_g.iter().zip(ffn_norm.iter()).map(|(a, b)| a * b).sum(),
 row_u.iter().zip(ffn_norm.iter()).map(|(a, b)| a * b).sum(),
 )
 };
 // `clamp <= 1e-6` is the disabled case and reduces to `silu(g) * u`,
 // so unclamped models stay bit-identical to the previous code.
 let (g, u) = if clamp > 1e-6 {
 (g.min(clamp), u.clamp(-clamp, clamp))
 } else {
 (g, u)
 };
 out[i] = crate::forward::silu(g) * u;
 }
 out
 }

 fn shared_down_hc_expand_add(
 &self,
 params: &LayerParams,
 shared_mid: &[f32],
 w_down: &[f32],
 routed_out: &[f32],
 after_attn_hc: &[f32],
 hc_split_post: &[f32],
 hc_split_comb: &[f32],
 ) -> Vec<f32> {
 // shared_out = w_down · shared_mid ⇒ [d_embd]
 let d_embd = params.d_embd as usize;
 let n_hc = params.n_hc as usize;
 let sd = shared_mid.len();
 debug_assert_eq!(w_down.len(), d_embd * sd);
 debug_assert_eq!(routed_out.len(), d_embd);

 // antirez `layer_shared_ffn_one` (ds4.c:4905) calls
 // `matvec_q8_0(out, ffn_down_shexp, mid)`, which internally invokes
 // `matvec_q8_0_rows` (ds4.c:3337) that `quantize_q8_0_activation`s
 // `mid` BEFORE the int8 dot. We were dotting f32-direct, retaining
 // precision antirez throws away. Match antirez fidelity by
 // round-tripping `shared_mid` through q8_0 before the down matvec
 // — gated by DS4_Q8_0_ACT (same toggle as the gate/up matvecs).
 let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
 let mid_owned;
 let shared_mid_in: &[f32] = if want_q80 {
 mid_owned = crate::forward::q8_0_round_trip(shared_mid);
 &mid_owned
 } else {
 shared_mid
 };
 // antirez `matvec_q8_0` (ds4.c:3337) uses 2×4-lane FMA
 // pair-reduce. Same reduction-tree class as
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
 let mut shared_out = vec![0.0f32; d_embd];
 for e in 0..d_embd {
 let row = &w_down[e * sd..(e + 1) * sd];
 shared_out[e] = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(row, shared_mid_in)
 } else {
 row.iter()
 .zip(shared_mid_in.iter())
 .map(|(a, b)| a * b)
 .sum()
 };
 }

 // Antirez `hc_post_one` (ds4.c:4225-4236) on the FFN block:
 // ffn_out[e] = shared_out[e] + routed_out[e] (ds4.c:5492-5494)
 // after[dst, e] = ffn_out[e] * hc_split_post[dst]
 // + Σ_src hc_split_comb[dst+src*n_hc] * after_attn_hc[src, e]
 debug_assert_eq!(hc_split_post.len(), n_hc);
 debug_assert_eq!(hc_split_comb.len(), n_hc * n_hc);
 debug_assert_eq!(after_attn_hc.len(), n_hc * d_embd);
 let mut after = vec![0.0f32; n_hc * d_embd];
 for dst in 0..n_hc {
 let base = dst * d_embd;
 let w_post = hc_split_post[dst];
 for e in 0..d_embd {
 let mut acc = w_post * (shared_out[e] + routed_out[e]);
 for src in 0..n_hc {
 acc += hc_split_comb[dst + src * n_hc] * after_attn_hc[src * d_embd + e];
 }
 after[base + e] = acc;
 }
 }
 after
 }
}

// ---------------------------------------------------------------------------
// Phase A of #225 — per-layer attention-half driver.
//
// `decode_attn_layer_with<A>` runs the 8 AttentionDispatcher methods over one
// decode position in antirez per-layer order (`ds4.c:8684`). Inputs and
// outputs are f32 — quantized weight access is the caller's problem
// (LayerView/QuantizedExpertWeights, #213). This driver lives next to the
// trait so that swapping `CpuAttentionDispatcher` for `MetalDispatcher` is a
// type substitution; the call sequence is the contract.
//
// drive every method once, produce post-attention HC residual.
// Phase B will plumb it into `decode_step_with` alongside the MoE half via
// per-layer `LayerView`.
// ---------------------------------------------------------------------------

/// Per-layer weight bundle for the attention half. All f32 — the dispatcher
/// boundary stays f32, and concrete dequant happens in the backend before
/// these slices are built.
///
/// Field shapes follow antirez naming (`attn_q_b`, `attn_kv_a_norm`, etc.).
///
/// Hyper-connection (HC) weights come in two parallel triplets per layer:
/// the **pre-attention** triplet (`hc_attn_fn`/`hc_attn_scale`/`hc_attn_base`)
/// is consumed by `hc_collapse_norm(HcKind::Attn)` before flash attention;
/// the **pre-FFN** triplet (`hc_ffn_fn`/`hc_ffn_scale`/`hc_ffn_base`) is
/// consumed by `hc_collapse_norm(HcKind::Ffn)` before the shared expert.
/// See `ds4.c:8727` and `ds4.c:9234` for the two antirez call sites.
#[derive(Debug, Clone)]
pub struct AttnLayerWeights {
 /// Per-layer shape pack (also threaded into every trait call).
 pub params: LayerParams,
 /// Pre-attention HC projection F16 weight `[hc_dim, mix_hc=24]`.
 pub hc_attn_fn: Vec<f32>,
 /// Pre-attention HC mix-scale F32 `[3]` (pre/post/comb scalars).
 pub hc_attn_scale: Vec<f32>,
 /// Pre-attention HC mix-base F32 `[mix_hc=24]`.
 pub hc_attn_base: Vec<f32>,
 /// Pre-FFN HC projection F16 weight `[hc_dim, mix_hc=24]`.
 pub hc_ffn_fn: Vec<f32>,
 /// Pre-FFN HC mix-scale F32 `[3]`.
 pub hc_ffn_scale: Vec<f32>,
 /// Pre-FFN HC mix-base F32 `[mix_hc=24]`.
 pub hc_ffn_base: Vec<f32>,
 /// No-copy F16 mmap bytes for the HC projections `hc_attn_fn`/`hc_ffn_fn`
 /// (the `[mix_hc, hc_dim]` weight matvec'd in `hc_collapse_norm`). Present
 /// only in lean mode; empty otherwise (the f32 above runs). The GPU
 /// hc-collapse reads these via the f16 fused kernel / `matvec_f16`.
 pub hc_attn_fn_f16: Cow<'static, [u8]>,
 pub hc_ffn_fn_f16: Cow<'static, [u8]>,
 /// RMS-norm γ for the pre-attn HC-collapse step (`attn_norm`).
 pub hc_norm_gamma: Vec<f32>,
 /// RMS-norm γ for the pre-FFN HC-collapse step (`ffn_norm`).
 pub hc_ffn_norm_gamma: Vec<f32>,
 /// QKV joint-norm γ, two halves: γ_q[n_lora_q], γ_kv[n_lora_kv].
 pub qkv_gamma_q: Vec<f32>,
 pub qkv_gamma_kv: Vec<f32>,
 /// Q LoRA stage A: `[d_embd, n_lora_q]` row-major. Antirez `attn_q_a` (Q8_0).
 /// Used by `attn_qkv_proj_with` step 1: `qr = attn_q_a · normed`.
 pub attn_q_a: Vec<f32>,
 /// Q LoRA stage B: `[n_lora_q, q_dim]` row-major. Antirez `attn_q_b` (Q8_0).
 /// Used by step 3: `q = attn_q_b · rms_norm(qr, γ_q)`.
 pub attn_q_b: Vec<f32>,
 /// Combined KV down-projection: `[d_embd, n_lora_kv]` row-major
 /// (`n_lora_kv == head_dim == 512` per antirez `DS4_N_HEAD_DIM`).
 /// Antirez `attn_kv` (Q8_0). DS4 MLA collapses the kv-a/kv-b split into
 /// a single matvec — see `ds4.c:4493`.
 pub attn_kv: Vec<f32>,
 /// Raw GGUF `block_q8_0` bytes for `attn_q_a`/`attn_q_b`/`attn_kv`, copied
 /// once from the model at load (owned, so page-aligned for a no-copy
 /// MTLBuffer that shares warm resident pages — the f32 path's trick). Lets
 /// the Metal decode run the q/kv projections at 1 byte/weight instead of
 /// dequant→f32 at 4 bytes. Empty when the source tensor isn't Q8_0; the
 /// encoder falls back to the f32 matvec then. See `the accelerator host crate` weight_q8_0_raw.
 pub attn_q_a_q8: Cow<'static, [u8]>,
 pub attn_q_b_q8: Cow<'static, [u8]>,
 pub attn_kv_q8: Cow<'static, [u8]>,
 /// Raw GGUF block_q8_0 bytes for the output projection (`w_o_a`/`w_o_b`,
 /// antirez `attn_output_a/b`, Q8_0) and the shared expert (`gate/up/down`,
 /// Q8_0). Same role as the q/kv `*_q8` bytes — a resident no-copy buffer for
 /// the 1-byte/weight Metal matvec. Empty when the source isn't Q8_0.
 pub w_o_a_q8: Cow<'static, [u8]>,
 pub w_o_b_q8: Cow<'static, [u8]>,
 pub w_shared_gate_q8: Cow<'static, [u8]>,
 pub w_shared_up_q8: Cow<'static, [u8]>,
 pub w_shared_down_q8: Cow<'static, [u8]>,
 /// `attn_output_a` and `attn_output_b` (lora-up split for W_o).
 pub w_o_a: Vec<f32>,
 pub w_o_b: Vec<f32>,
 /// Per-head attention sinks (length `n_head`).
 pub attn_sinks: Vec<f32>,
 /// Shared-expert weights.
 pub w_shared_gate: Vec<f32>,
 pub w_shared_up: Vec<f32>,
 pub w_shared_down: Vec<f32>,
 /// Shared-expert intermediate width and silu/clamp value.
 pub shared_dim: u32,
 pub shared_clamp: f32,
 /// Compressor weights for ratio>1 layers. Empty for dense.
 /// Antirez `attn_compressor_{kv,gate,ape,norm}` (ds4.c:6208-6301).
 /// Shapes:
 /// `attn_compressor_kv`: `[d_embd, width]` where `width = coff * head_dim`
 /// and `coff = ratio==4 ? 2 : 1`
 /// `attn_compressor_gate`: `[d_embd, width]` (parallel score projection)
 /// `attn_compressor_ape`: `[width, compress_ratio]` positional table
 /// indexed `(j, pos_mod)`
 /// `attn_compressor_norm`: `[head_dim]` γ for the pooled RMS norm
 pub attn_compressor_kv: Vec<f32>,
 pub attn_compressor_gate: Vec<f32>,
 pub attn_compressor_ape: Vec<f32>,
 pub attn_compressor_norm: Vec<f32>,
 /// Raw GGUF F16 bytes (2 B/elem) for `attn_compressor_kv`/`gate`, borrowed
 /// no-copy straight from the model mmap (same invariant as `attn_q_a_q8`:
 /// MAP_SHARED keeps them clean file-backed, DecodeRunner drops `composed`
 /// before the mmap). Present only in `lean` mode (server/encoder), where the
 /// f32 dequants above are SKIPPED to save ~1 GB — the GPU compressor reads
 /// these via `matvec_f16`. Empty when the tensor isn't F16 or in non-lean
 /// mode (then the f32 path runs). See `compressor::matvec_f16`.
 pub attn_compressor_kv_f16: Cow<'static, [u8]>,
 pub attn_compressor_gate_f16: Cow<'static, [u8]>,
 /// Indexer compressor (ratio==4 only). Empty for ratio!=4.
 /// Same role names with `indexer_` prefix; `indexer_compressor_kv` has
 /// width `coff * DS4_N_INDEXER_HEAD_DIM` (= 256) instead of head_dim.
 pub indexer_compressor_kv: Vec<f32>,
 pub indexer_compressor_gate: Vec<f32>,
 pub indexer_compressor_ape: Vec<f32>,
 pub indexer_compressor_norm: Vec<f32>,
 /// No-copy F16 mmap bytes for `indexer_compressor_kv`/`gate` — the indexer
 /// twin of `attn_compressor_kv_f16`. Present only in `lean` mode; empty
 /// otherwise (f32 path runs).
 pub indexer_compressor_kv_f16: Cow<'static, [u8]>,
 pub indexer_compressor_gate_f16: Cow<'static, [u8]>,
 /// Indexer projection weights (ratio==4 only). Empty for ratio!=4.
 /// Antirez `tensor_expect_layout` (ds4.c:2177-2178):
 /// `indexer.attn_q_b`: F16 `[DS4_N_LORA_Q, n_indexer_head * indexer_head_dim]`
 /// `indexer.proj`: F16 `[DS4_N_EMBD, n_indexer_head]`
 pub indexer_attn_q_b: Vec<f32>,
 pub indexer_proj: Vec<f32>,
 /// No-copy F16 mmap bytes for the indexer SCORING projections
 /// `indexer_attn_q_b`/`indexer_proj` (the long-context `encode_indexer_scores`
 /// matvecs). Present only in lean mode AND when the GPU indexer is enabled
 /// (the CPU long-context path, `DS4_GPU_INDEXER=0`, still reads the f32 — so
 /// the f32 above is kept then). Empty otherwise. Distinct from the compressor
 /// kv/gate f16 (the indexer COMPRESSOR vs the indexer SCORING projections).
 pub indexer_attn_q_b_f16: Cow<'static, [u8]>,
 pub indexer_proj_f16: Cow<'static, [u8]>,
}

impl AttnLayerWeights {
 /// Build an `AttnLayerWeights` by dequantizing the per-layer tensors out
 /// of a `LayerViews` ( Phase B2). All DS4 attention weights are
 /// F16/F32 in the antirez Q2-imatrix layout (see `DS4_Q2_LAYOUT` in
 /// `gguf.rs`), so `LayerViews::dequant_f32_simple` is sufficient — no
 /// the host-side `cpu_via_dequant` helper callback needed for the attention half.
 ///
 /// `params` is supplied by the caller (built from the GGUF metadata in
 /// `ModelManifest`) and threaded through unchanged. `shared_dim` and
 /// `shared_clamp` come from caller-provided defaults (antirez uses
 /// `shared_dim` = `moe_intermediate_size` and `shared_clamp` = 7.0).
 pub fn from_layer_view<S: crate::w8a8_roles::WeightSource>(
 src: &S,
 view: &crate::layer_view::LayerView,
 params: LayerParams,
 shared_dim: u32,
 shared_clamp: f32,
 // When true, SKIP dequantizing the f32 duplicates of the q8-backed
 // projections (attn q_a/q_b/kv, output o_a/o_b, shared gate/up/down)
 // whose q8 path the encoder uses — so the ~20 GB is never allocated
 // (vs allocated-then-freed, which leaves it in malloc's free-cache,
 // inflating phys_footprint). The compressor/indexer/hc/norm f32 (no q8)
 // are always built. Only the encoder/DecodeSession path is `lean`; the
 // trait/run_argmax f32 path passes `false`.
 lean: bool,
 ) -> anyhow::Result<Self> {
 let deq = |role: &str| -> anyhow::Result<Vec<f32>> {
 let h = view.require(role)?;
 src.dequant_f32(h)
 };
 // DS4_COMPRESSOR_F16 (default on) gates ALL the no-copy f16 weight paths
 // (compressor kv/gate, indexer scoring, hc projections); =0 reverts to
 // the f32 dequant — the A/B baseline. `raw_f16` borrows the F16 bytes
 // straight from the retained model mmap (twin of `raw_q8`); empty unless
 // the tensor is genuinely F16. SAFETY: identical to `raw_q8` (views
 // outlive `composed`, MAP_SHARED keeps pages clean).
 // DS4_FUSED_COMP: the GPU-resident decode path for the ratio≠4 (no-indexer)
 // attn-compressor layers is f16-precision-sensitive — f16 weights accumulate
 // error across the ~20 such layers and corrupt/repeat the output (the staged
 // CPU path is not sensitive). Force f32 (ALL f16 paths: hc, compressor,
 // indexer) for ONLY those layers, keeping f16-no-copy everywhere else, so
 // DS4_FUSED_COMP=1 is coherent without the global F16=0 memory regression.
 let force_f32 = std::env::var("DS4_FUSED_COMP").ok().as_deref() != Some("0")
 && params.compress_ratio != 0
 && params.compress_ratio != 4;
 let comp_f16_on = !force_f32
 && std::env::var("DS4_COMPRESSOR_F16")
 .map(|v| v != "0")
 .unwrap_or(true);
 let raw_f16 = |role: &str| -> anyhow::Result<Cow<'static, [u8]>> {
 if comp_f16_on {
 if let Some(h) = view.handles.get(role) {
 if h.ttype == crate::gguf::GgmlType::F16 {
 let s_owned = src.bytes_for(h)?;
 let s: &[u8] = &s_owned;
 let s: &'static [u8] =
 unsafe { std::mem::transmute::<&[u8], &'static [u8]>(s) };
 return Ok(Cow::Borrowed(s));
 }
 }
 }
 Ok(Cow::Borrowed(&[]))
 };

 let n_hc = params.n_hc as usize;
 let mix_hc = 2 * n_hc + n_hc * n_hc;

 // HC projections (hc_attn_fn/hc_ffn_fn are F16): borrow no-copy + lean-skip
 // the f32 dequant (the GPU hc-collapse reads f16 via the fused f16 kernel
 // / matvec_f16). scale/base/norm_gamma stay f32 (tiny). For the DS4_FUSED_COMP
 // ratio≠4 layers, comp_f16_on is false (above) so these come back f32.
 let hc_attn_fn_f16 = raw_f16("hc_attn_fn")?;
 let hc_ffn_fn_f16 = raw_f16("hc_ffn_fn")?;
 let skip_hc = lean && !hc_attn_fn_f16.is_empty() && !hc_ffn_fn_f16.is_empty();
 let hc_attn_fn = if skip_hc {
 Vec::new()
 } else {
 deq("hc_attn_fn")?
 };
 let hc_attn_scale = deq("hc_attn_scale")?;
 let hc_attn_base = deq("hc_attn_base")?;
 let hc_ffn_fn = if skip_hc {
 Vec::new()
 } else {
 deq("hc_ffn_fn")?
 };
 let hc_ffn_scale = deq("hc_ffn_scale")?;
 let hc_ffn_base = deq("hc_ffn_base")?;
 debug_assert_eq!(hc_attn_scale.len(), 3);
 debug_assert_eq!(hc_attn_base.len(), mix_hc);
 debug_assert_eq!(hc_ffn_scale.len(), 3);
 debug_assert_eq!(hc_ffn_base.len(), mix_hc);

 let hc_norm_gamma = deq("attn_norm")?;
 let hc_ffn_norm_gamma = deq("ffn_norm")?;
 let qkv_gamma_q = deq("mla_q_a_norm")?;
 let qkv_gamma_kv = deq("mla_kv_a_norm")?;
 // (attn_q_a/q_b/kv f32 are dequantized below, after the q8 flags are
 // known, so `lean` can skip them when the encoder uses the q8 path.)
 // Raw GGUF Q8_0 bytes for the q/kv projections (owned copy, for a
 // resident no-copy MTLBuffer in the Metal decode path). Empty unless the
 // tensor is genuinely Q8_0, so non-Q8_0 fixtures fall back to f32.
 let raw_q8 = |role: &str| -> anyhow::Result<Cow<'static, [u8]>> {
 let h = view.require(role)?;
 if h.ttype == crate::gguf::GgmlType::Q8_0 {
 // No-copy: borrow the q8 bytes straight from the model mmap
 // (Phase 2). SAFETY: `views` is the model mmap retained for the
 // whole process by `DecodeRunner` (_views_keepalive), which drops
 // `composed` (holding these slices) BEFORE the mmap; moving the
 // Mmap struct doesn't remap the pages. MAP_SHARED + the no-copy
 // MTLBuffer keep the pages clean file-backed. Lifetime-extend to
 // 'static — a self-contained invariant, never escapes DecodeRunner.
 let s_owned = src.bytes_for(h)?;
 let s: &[u8] = &s_owned;
 let s: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(s) };
 // QUALITY PROBE (DS4_ATTN_Q4_PROBE): degrade the ATTENTION projection
 // weights (mla_*, attn_output_*) to q4 precision in a q8 container so the
 // existing q8 kernel runs them at q4 — needle-gates a q8→q4 attention cut
 // before building the K-batched q4 kernel. Excludes shared/MoE roles.
 let is_attn = role.starts_with("mla_") || role.starts_with("attn_output");
 if is_attn && std::env::var("DS4_ATTN_Q4_PROBE").ok().as_deref() == Some("1") {
 // Validated quality-safe: byte-identical greedy decode (32 tok) at
 // @600 and @3000 despite ~78% of attention bytes changing → q8→q4 on
 // the attention projections preserves argmax. The native K-batched q4
 // kernel (the speed win) is the follow-up; this proves it's worth it.
 return Ok(Cow::Owned(crate::layer_view::q4_precision_in_q8_container(
 s,
 )?));
 }
 Ok(Cow::Borrowed(s))
 } else {
 Ok(Cow::Borrowed(&[]))
 }
 };
 let attn_q_a_q8 = raw_q8("mla_q_a")?;
 let attn_q_b_q8 = raw_q8("mla_q_b")?;
 let attn_kv_q8 = raw_q8("mla_kv")?;
 let w_o_a_q8 = raw_q8("attn_output_a")?;
 let w_o_b_q8 = raw_q8("attn_output_b")?;
 let w_shared_gate_q8 = raw_q8("moe_gate_shared")?;
 let w_shared_up_q8 = raw_q8("moe_up_shared")?;
 let w_shared_down_q8 = raw_q8("moe_down_shared")?;
 // Lean: skip the f32 dequant of the q8-backed projections the encoder
 // reads via q8 — matches free_dead_f32_weights' conditions exactly, so
 // the f32 is never allocated (no malloc free-cache → real footprint).
 let q8_proj = std::env::var("DS4_Q8_PROJ")
 .map(|v| v != "0")
 .unwrap_or(true);
 let skip_qkv = lean
 && q8_proj
 && !attn_q_a_q8.is_empty()
 && !attn_q_b_q8.is_empty()
 && !attn_kv_q8.is_empty();
 let skip_o = lean && q8_proj && !w_o_a_q8.is_empty() && !w_o_b_q8.is_empty();
 // Hash-routed layers (V4 Flash 0/1/2) keep their f32 shared weights:
 // the long-context slow path (indexer/compressor engaged) routes hash
 // layers through `run_ffn_half_metal_scoped`'s f32 fallback (the generic
 // `run_ffn_half` → `moe_and_shared_chain_batched`), which reads the f32
 // shared gate/up/down directly. Emptying them crashed the daemon on
 // prompts past ~2048 tokens (2026-06-02 incident — the short-prompt
 // fast/chain path uses q8 inline, so it was never exercised). This
 // mirrors the f32-router carve-out for hash layers at decode_step.rs:475.
 let is_hash_layer = view.handles.get("moe_routing_table").is_some();
 let skip_shared = lean
 && !is_hash_layer
 && !w_shared_gate_q8.is_empty()
 && !w_shared_up_q8.is_empty()
 && !w_shared_down_q8.is_empty();
 let attn_q_a = if skip_qkv {
 Vec::new()
 } else {
 deq("mla_q_a")?
 };
 let attn_q_b = if skip_qkv {
 Vec::new()
 } else {
 deq("mla_q_b")?
 };
 let attn_kv = if skip_qkv { Vec::new() } else { deq("mla_kv")? };
 let w_o_a = if skip_o {
 Vec::new()
 } else {
 deq("attn_output_a")?
 };
 let w_o_b = if skip_o {
 Vec::new()
 } else {
 deq("attn_output_b")?
 };
 let attn_sinks = deq("attn_sinks")?;
 let w_shared_gate = if skip_shared {
 Vec::new()
 } else {
 deq("moe_gate_shared")?
 };
 let w_shared_up = if skip_shared {
 Vec::new()
 } else {
 deq("moe_up_shared")?
 };
 let w_shared_down = if skip_shared {
 Vec::new()
 } else {
 deq("moe_down_shared")?
 };

 // Compressor weights. Compressed layers (ratio > 1) have
 // four tensors per layer; dense layers (0, 1) have none. The classifier
 // emits `attn_compressor_{kv,gate,ape,norm}` role names; absent on dense.
 let try_deq = |role: &str| -> anyhow::Result<Vec<f32>> {
 if let Some(h) = view.handles.get(role) {
 src.dequant_f32(h)
 } else {
 Ok(Vec::new())
 }
 };
 // No-copy F16 mmap bytes for the compressor kv/gate projections — the
 // F16 twin of `raw_q8`. Borrow straight from the retained model mmap;
 // empty unless the tensor is genuinely F16. SAFETY: identical invariant
 // to `raw_q8` (views outlive `composed`, MAP_SHARED keeps pages clean).
 let attn_compressor_kv_f16 = raw_f16("attn_compressor_kv")?;
 let attn_compressor_gate_f16 = raw_f16("attn_compressor_gate")?;
 let indexer_compressor_kv_f16 = raw_f16("indexer_compressor_kv")?;
 let indexer_compressor_gate_f16 = raw_f16("indexer_compressor_gate")?;
 // Lean: skip the f32 dequant of the compressor kv/gate projections when
 // their F16 bytes are present (the GPU compressor reads them via
 // matvec_f16). `has_attn/indexer_compressor()` treats f16-OR-f32 as
 // "present", so every compressor gate still fires. The ape/norm f32
 // stay (tiny; ape used on CPU, norm is the pooled-rms gamma).
 let skip_attn_comp =
 lean && !attn_compressor_kv_f16.is_empty() && !attn_compressor_gate_f16.is_empty();
 let skip_idx_comp = lean
 && !indexer_compressor_kv_f16.is_empty()
 && !indexer_compressor_gate_f16.is_empty();
 let attn_compressor_kv = if skip_attn_comp {
 Vec::new()
 } else {
 try_deq("attn_compressor_kv")?
 };
 let attn_compressor_gate = if skip_attn_comp {
 Vec::new()
 } else {
 try_deq("attn_compressor_gate")?
 };
 let attn_compressor_ape = try_deq("attn_compressor_ape")?;
 let attn_compressor_norm = try_deq("attn_compressor_norm")?;
 let indexer_compressor_kv = if skip_idx_comp {
 Vec::new()
 } else {
 try_deq("indexer_compressor_kv")?
 };
 let indexer_compressor_gate = if skip_idx_comp {
 Vec::new()
 } else {
 try_deq("indexer_compressor_gate")?
 };
 let indexer_compressor_ape = try_deq("indexer_compressor_ape")?;
 let indexer_compressor_norm = try_deq("indexer_compressor_norm")?;
 // Indexer SCORING projections: borrow F16 no-copy for the GPU
 // `encode_indexer_scores` path. Skip the f32 dequant only when the GPU
 // indexer is on (default) — the CPU long-context path (DS4_GPU_INDEXER=0)
 // reads the f32, and short-context decode never reads either (the CPU
 // `indexer_allowed_decode_one` early-returns all-allowed at n_comp<=top_k).
 let indexer_attn_q_b_f16 = raw_f16("indexer_attn_q_b")?;
 let indexer_proj_f16 = raw_f16("indexer_proj")?;
 let gpu_indexer_on = std::env::var("DS4_GPU_INDEXER")
 .map(|v| v != "0")
 .unwrap_or(true);
 let skip_idx_score = lean
 && gpu_indexer_on
 && !indexer_attn_q_b_f16.is_empty()
 && !indexer_proj_f16.is_empty();
 let indexer_attn_q_b = if skip_idx_score {
 Vec::new()
 } else {
 try_deq("indexer_attn_q_b")?
 };
 let indexer_proj = if skip_idx_score {
 Vec::new()
 } else {
 try_deq("indexer_proj")?
 };

 Ok(Self {
 params,
 hc_attn_fn,
 hc_attn_scale,
 hc_attn_base,
 hc_ffn_fn,
 hc_ffn_scale,
 hc_ffn_base,
 hc_attn_fn_f16,
 hc_ffn_fn_f16,
 hc_norm_gamma,
 hc_ffn_norm_gamma,
 qkv_gamma_q,
 qkv_gamma_kv,
 attn_q_a,
 attn_q_b,
 attn_q_a_q8,
 attn_q_b_q8,
 attn_kv_q8,
 w_o_a_q8,
 w_o_b_q8,
 w_shared_gate_q8,
 w_shared_up_q8,
 w_shared_down_q8,
 attn_kv,
 w_o_a,
 w_o_b,
 attn_sinks,
 w_shared_gate,
 w_shared_up,
 w_shared_down,
 shared_dim,
 shared_clamp,
 attn_compressor_kv,
 attn_compressor_gate,
 attn_compressor_ape,
 attn_compressor_norm,
 attn_compressor_kv_f16,
 attn_compressor_gate_f16,
 indexer_compressor_kv,
 indexer_compressor_gate,
 indexer_compressor_ape,
 indexer_compressor_norm,
 indexer_compressor_kv_f16,
 indexer_compressor_gate_f16,
 indexer_attn_q_b,
 indexer_proj,
 indexer_attn_q_b_f16,
 indexer_proj_f16,
 })
 }

 /// True iff this layer has a main attention compressor — checking the F16
 /// no-copy bytes (lean) OR the f32 dequant (non-lean). Lean mode empties the
 /// f32 (the GPU reads f16), so the old `attn_compressor_kv.is_empty()` gate
 /// would wrongly read "no compressor"; every compressor gate must use this.
 #[inline]
 pub fn has_attn_compressor(&self) -> bool {
 !self.attn_compressor_kv.is_empty() || !self.attn_compressor_kv_f16.is_empty()
 }

 /// True iff this layer has an indexer compressor (f16-OR-f32). See
 /// [`Self::has_attn_compressor`].
 #[inline]
 pub fn has_indexer_compressor(&self) -> bool {
 !self.indexer_compressor_kv.is_empty() || !self.indexer_compressor_kv_f16.is_empty()
 }

 /// No-copy F16 bytes for the main compressor kv/gate, as the `Option<&[u8]>`
 /// the GPU `CompressorInputs.w_kv_f16`/`w_gate_f16` want — `Some` only when
 /// the f16 path applies (lean), else `None` (the f32 path runs).
 #[inline]
 pub fn attn_compressor_f16(&self) -> (Option<&[u8]>, Option<&[u8]>) {
 (
 (!self.attn_compressor_kv_f16.is_empty()).then_some(&*self.attn_compressor_kv_f16),
 (!self.attn_compressor_gate_f16.is_empty()).then_some(&*self.attn_compressor_gate_f16),
 )
 }

 /// Indexer twin of [`Self::attn_compressor_f16`].
 #[inline]
 pub fn indexer_compressor_f16(&self) -> (Option<&[u8]>, Option<&[u8]>) {
 (
 (!self.indexer_compressor_kv_f16.is_empty())
 .then_some(&*self.indexer_compressor_kv_f16),
 (!self.indexer_compressor_gate_f16.is_empty())
 .then_some(&*self.indexer_compressor_gate_f16),
 )
 }

 /// True iff this layer has the indexer SCORING projections (`indexer_attn_q_b`
 /// + `indexer_proj`) — f16-OR-f32. Lean mode empties the f32, so the
 /// `indexer_attn_q_b.is_empty()` "is the indexer active" gate must use this.
 #[inline]
 pub fn has_indexer_qb(&self) -> bool {
 !self.indexer_attn_q_b.is_empty() || !self.indexer_attn_q_b_f16.is_empty()
 }

 /// No-copy F16 bytes for the indexer scoring projections (`q_b`, `proj`) as
 /// `Option<&[u8]>` for `encode_indexer_scores` — `Some` only when the f16
 /// path applies (lean + GPU indexer), else `None` (the f32 path runs).
 #[inline]
 pub fn indexer_scoring_f16(&self) -> (Option<&[u8]>, Option<&[u8]>) {
 (
 (!self.indexer_attn_q_b_f16.is_empty()).then_some(&*self.indexer_attn_q_b_f16),
 (!self.indexer_proj_f16.is_empty()).then_some(&*self.indexer_proj_f16),
 )
 }

 /// No-copy F16 bytes for the pre-attn HC projection (`hc_attn_fn`) as
 /// `Option<&[u8]>` — `Some` only on the lean GPU path (f32 skipped), else
 /// `None` (f32 path). The encoder builds the hc_fn buffer accordingly.
 #[inline]
 pub fn hc_attn_fn_f16(&self) -> Option<&[u8]> {
 (!self.hc_attn_fn_f16.is_empty()).then_some(&*self.hc_attn_fn_f16)
 }

 /// Pre-FFN twin of [`Self::hc_attn_fn_f16`] (`hc_ffn_fn`).
 #[inline]
 pub fn hc_ffn_fn_f16(&self) -> Option<&[u8]> {
 (!self.hc_ffn_fn_f16.is_empty()).then_some(&*self.hc_ffn_fn_f16)
 }

 /// `hc_attn_fn` as f32: the dequant when present, else reconstructed from
 /// the no-copy F16 bytes (lean). Used by the K-position verifier
 /// (`base_run_bundle`), which has no f16 hc-collapse kernel and runs the
 /// fused `matvec_f32_k` path — so it needs f32 hc. Allocates only on the
 /// lean+verifier path (opt-in spec-decode), once per scope.
 pub fn hc_attn_fn_as_f32(&self) -> std::borrow::Cow<'_, [f32]> {
 if !self.hc_attn_fn.is_empty() || self.hc_attn_fn_f16.is_empty() {
 std::borrow::Cow::Borrowed(&self.hc_attn_fn)
 } else {
 std::borrow::Cow::Owned(
 self.hc_attn_fn_f16
 .chunks_exact(2)
 .map(|c| crate::layer_view::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
 .collect(),
 )
 }
 }

 /// Pre-FFN twin of [`Self::hc_attn_fn_as_f32`] (`hc_ffn_fn`).
 pub fn hc_ffn_fn_as_f32(&self) -> std::borrow::Cow<'_, [f32]> {
 if !self.hc_ffn_fn.is_empty() || self.hc_ffn_fn_f16.is_empty() {
 std::borrow::Cow::Borrowed(&self.hc_ffn_fn)
 } else {
 std::borrow::Cow::Owned(
 self.hc_ffn_fn_f16
 .chunks_exact(2)
 .map(|c| crate::layer_view::f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
 .collect(),
 )
 }
 }
}

/// Per-step working state (the slice of inputs that change every decode call).
///
/// Owned by the caller: `kv_cache` (mutable view), `cur_hc` carried across
/// layers, `pos`. The per-layer HC sinkhorn `hc_split` is RECOMPUTED inside
/// every `hc_collapse_norm` call from the projection weight (no carry).
pub struct AttnStepInputs<'a> {
 /// HC-residual coming in (`cur_hc` from prior layer or input embd).
 pub cur_hc: &'a [f32],
 /// Pre-projected KV-down vector — `[n_lora_kv]` (== head_dim).
 pub kv_raw_row: &'a [f32],
 /// QR → Q lora-up post product, shape `[n_head * head_dim]`.
 /// In real DS4 this is `attn_q_b · qr_normed`; the caller materializes it
 /// (CPU dispatcher uses an extra matvec; Metal fuses it into qkv_rms).
 pub q_heads: &'a [f32],
 /// Position index into the KV cache.
 pub pos: u32,
 /// KV-cache rolling state.
 pub kv_view: KvCacheView<'a>,
 /// Pre-attention raw window — number of raw rows visible to attention
 /// at this position (capped at `kv_view.raw_cap`).
 pub n_raw: u32,
 /// Start row inside `kv_view.raw` for the SWA window.
 pub raw_start: u32,
 /// Routed-expert output (from the MoE half — Phase B will produce it
 /// from `KernelDispatcher::moe_routed_step`). For Phase A it is supplied
 /// by the caller (zero vector is a valid attention-only test).
 pub routed_out: &'a [f32],
 /// Compressed KV rows for this layer. `None` or `Some(&[])`
 /// when there are no comp rows yet (e.g., pos < compress_ratio−1, or
 /// dense layer). Layout matches raw_kv: `[n_comp, n_lora_kv]` row-major.
 pub kv_comp_rows: Option<&'a [f32]>,
 /// Number of valid rows in `kv_comp_rows`.
 pub n_comp: u32,
 /// Indexer-selected subset ( =4 only). `None` means
 /// attend over all `n_comp` rows; `Some(sel)` means only rows in `sel`.
 pub comp_selected: Option<&'a [u32]>,
 /// Number of valid entries in `comp_selected`.
 pub n_selected: u32,
}

/// Outputs of one attention-half decode layer.
pub struct AttnLayerOut {
 /// Post-attention + shared-expert HC residual (`after_ffn_hc`).
 pub after_ffn_hc: Vec<f32>,
 /// Post-norm activation used downstream (e.g. as router input).
 pub normed: Vec<f32>,
}

/// Output of the attention prefix (steps 1–8 of `decode_attn_layer_with`).
///
/// Splits the layer at the natural fan-out point: `normed` feeds into the
/// MoE-router half (via `KernelDispatcher::matvec_f32` etc.), and the same
/// `normed` is later re-normed with `ffn_norm_gamma` for the shared expert.
/// `after_attn_hc` carries the post-W_o HC residual into the suffix.
///
/// `hc_split_attn` is the full mix_hc=24 buffer from the pre-attention
/// `hc_collapse_norm(HcKind::Attn)` call. Only `[0..n_hc]` was consumed in
/// the prefix; the suffix discards the rest because the pre-FFN call
/// recomputes a fresh split from the FFN weight triplet.
pub struct AttnPrefixOut {
 pub after_attn_hc: Vec<f32>,
 pub hc_split_attn: Vec<f32>,
 pub normed: Vec<f32>,
}

/// Inputs to `compressor_decode_one`.
///
/// `wkv` / `wgate` / `ape` are dequantized matrices already loaded into
/// `AttnLayerWeights`. `norm` is the per-`head_dim` gamma for the post-pool
/// rms norm. State buffers live on `AttnStepState`.
pub struct CompressorInputs<'a> {
 pub w_kv: &'a [f32], // [in_dim, width]
 pub w_gate: &'a [f32], // [in_dim, width]
 /// No-copy F16 mmap bytes for `w_kv`/`w_gate` (2 B/elem, `[in_dim, width]`).
 /// `Some` only on the lean GPU path, where the f32 above is empty and the
 /// matvec runs via `matvec_f16`; `None` → the f32 path (`matvec_f32` or the
 /// CPU loop) runs. The two are bit-identical (F16→f32 is exact). CPU-only
 /// callers (the `decode_step` trait path) always pass `None`.
 pub w_kv_f16: Option<&'a [u8]>,
 pub w_gate_f16: Option<&'a [u8]>,
 pub w_ape: &'a [f32], // [width, compress_ratio] (j-major × pos_mod)
 pub w_norm: &'a [f32], // [head_dim]
 pub head_dim: u32,
 pub compress_ratio: u32,
}

/// Pool the current compression window with a per-`head_dim` softmax over
/// `state_score`, weighted-summing `state_kv`. Port of antirez
/// `compressor_pool_decode_state` (ds4.c:6153).
///
/// `state_kv` / `state_score` are `[rows, width]` flat, where
/// `width = coff * head_dim`, `coff = ratio==4 ? 2 : 1`,
/// `rows = ratio==4 ? 2*ratio : ratio`. Writes `head_dim` floats to `out`.
fn compressor_pool_decode_state(
 out: &mut [f32],
 state_kv: &[f32],
 state_score: &[f32],
 head_dim: u32,
 compress_ratio: u32,
) {
 let coff = if compress_ratio == 4 { 2u32 } else { 1u32 } as usize;
 let width = coff * head_dim as usize;
 let hd = head_dim as usize;
 let r4 = compress_ratio == 4;
 let ratio = compress_ratio as usize;
 const NEG_INF: f32 = -1.0e9;

 for j in 0..hd {
 // 1. max over the active score rows
 let mut max_score = NEG_INF;
 if r4 {
 for r in 0..ratio {
 let sp = state_score[r * width + j];
 let sc = state_score[(ratio + r) * width + hd + j];
 if sp > max_score {
 max_score = sp;
 }
 if sc > max_score {
 max_score = sc;
 }
 }
 } else {
 for r in 0..ratio {
 let s = state_score[r * width + j];
 if s > max_score {
 max_score = s;
 }
 }
 }

 if max_score <= NEG_INF * 0.5 {
 out[j] = 0.0;
 continue;
 }

 let mut denom = 0.0f32;
 let mut sum = 0.0f32;
 if r4 {
 for r in 0..ratio {
 let wp = (state_score[r * width + j] - max_score).exp();
 let wc = (state_score[(ratio + r) * width + hd + j] - max_score).exp();
 denom += wp + wc;
 sum += wp * state_kv[r * width + j];
 sum += wc * state_kv[(ratio + r) * width + hd + j];
 }
 } else {
 for r in 0..ratio {
 let w = (state_score[r * width + j] - max_score).exp();
 denom += w;
 sum += w * state_kv[r * width + j];
 }
 }
 out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
 }
}

/// Streaming compressor update for one decode token. Port of antirez
/// `compressor_decode_one` (ds4.c:6208). Returns `Some(pooled)` (head_dim
/// floats) when a compressed row is emitted at this position, or `None`
/// otherwise. The caller is responsible for appending the returned row to
/// the per-layer ring + incrementing `n_comp`.
///
/// Side effects:
/// - writes one row into `state_kv` + `state_score` at `[row, ..]`;
/// - on emit (every `compress_ratio` positions), rotates the second half
/// of the state into the first half (ratio-4 only).
pub fn compressor_decode_one<A: AttentionDispatcher>(
 dispatch: &A,
 params: &LayerParams,
 comp: &CompressorInputs<'_>,
 x: &[f32],
 state_kv: &mut [f32],
 state_score: &mut [f32],
 pos: u32,
) -> Option<Vec<f32>> {
 let ratio = comp.compress_ratio;
 let head_dim = comp.head_dim;
 let coff = if ratio == 4 { 2u32 } else { 1u32 } as usize;
 let width = coff * head_dim as usize;
 let pos_mod = (pos % ratio) as usize;
 let row = if ratio == 4 {
 (ratio as usize) + pos_mod
 } else {
 pos_mod
 };
 let should_compress = ((pos + 1) % ratio) == 0;

 // 1. Project x onto kv_cur and sc_cur: kv_cur = x @ w_kv, sc_cur = x @ w_gate.
 // antirez `compressor_decode_one` (ds4.c:6240) calls
 // `quantize_q8_0_activation(x, ...)` then `matvec_q8_0_pair_prequant` —
 // i.e. x is q8_0-round-tripped before the int8 dot. Mirror with the same
 // DS4_Q8_0_ACT toggle that gates the attn LoRA / lm_head / shared_expert
 // q8_0 sites.
 let in_dim = x.len();
 debug_assert_eq!(comp.w_kv.len(), in_dim * width);
 debug_assert_eq!(comp.w_gate.len(), in_dim * width);
 let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");
 let x_owned;
 let x_in: &[f32] = if want_q80 {
 x_owned = crate::forward::q8_0_round_trip(x);
 &x_owned
 } else {
 x
 };
 // paired dot reduction tree. Antirez `matvec_q8_0_pair_prequant`
 // (ds4.c:6241) → `matvec_q8_0_pair_worker` (ds4.c:3035) → `dot_q8_0_row_pair`
 // (ds4.c:2876) uses 4× float32x4_t FMA accumulators reduced via NEON pair-add.
 // Our default scalar left-fold is f32-ULP-different. Gate the dot via
 // `dot_f32_antirez` under the existing DS4_MATVEC_F32_FIDELITY env (same
 // class as ); paired loop becomes two
 // independent dots — semantically identical, reduction tree matches antirez.
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
 let mut kv_cur = vec![0.0f32; width];
 let mut sc_cur = vec![0.0f32; width];
 for j in 0..width {
 // antirez weights are stored col-major [out=width, in=in_dim] (see
 // matvec_any in ds4.c); element at (in=k, out=j) is at w[j*in_dim + k].
 let base = j * in_dim;
 let row_kv = &comp.w_kv[base..base + in_dim];
 let row_sc = &comp.w_gate[base..base + in_dim];
 if antirez_dot_fidelity {
 kv_cur[j] = crate::forward::dot_f32_antirez(row_kv, x_in);
 sc_cur[j] = crate::forward::dot_f32_antirez(row_sc, x_in);
 } else {
 let mut acc_kv = 0.0f32;
 let mut acc_sc = 0.0f32;
 for k in 0..in_dim {
 acc_kv += row_kv[k] * x_in[k];
 acc_sc += row_sc[k] * x_in[k];
 }
 kv_cur[j] = acc_kv;
 sc_cur[j] = acc_sc;
 }
 }

 // 2. Add positional ape into sc_cur: sc_cur[j] += ape[j, pos_mod].
 debug_assert_eq!(comp.w_ape.len(), width * ratio as usize);
 for j in 0..width {
 // antirez tensor_2d_value(ape, j, pos_mod): ape is [dim0=width, dim1=ratio]
 // stored row-major so element (j, pos_mod) = w_ape[j*ratio + pos_mod].
 sc_cur[j] += comp.w_ape[j * ratio as usize + pos_mod];
 }

 // 3. Commit current row into state.
 state_kv[row * width..(row + 1) * width].copy_from_slice(&kv_cur);
 state_score[row * width..(row + 1) * width].copy_from_slice(&sc_cur);

 if !should_compress {
 return None;
 }

 // 4. Pool: per-head_dim softmax over state_score, weighted sum of state_kv.
 let hd = head_dim as usize;
 let mut pooled = vec![0.0f32; hd];
 compressor_pool_decode_state(&mut pooled, state_kv, state_score, head_dim, ratio);

 // 5. RMS-norm with comp.w_norm gamma.
 let mut ss: f64 = 0.0;
 for v in pooled.iter() {
 ss += (*v as f64) * (*v as f64);
 }
 let rms = 1.0_f32 / ((ss / hd as f64) as f32 + 1.0e-6).sqrt();
 debug_assert_eq!(comp.w_norm.len(), hd);
 let mut out_comp = vec![0.0f32; hd];
 for i in 0..hd {
 out_comp[i] = pooled[i] * rms * comp.w_norm[i];
 }

 // 6. rope_tail on the pooled row at comp_pos = pos + 1 - compress_ratio.
 let n_rot = params.n_rot as usize;
 if n_rot > 0 {
 let comp_pos = pos + 1 - ratio;
 let tail = &mut out_comp[hd - n_rot..hd];
 dispatch.rope_tail(params, tail, comp_pos, false);
 }
 // 6b. fp8-NOPE quantization when head_dim == DS4_N_HEAD_DIM (=512).
 // Antirez ds4.c:6276-6278: `dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp,
 // head_dim, DS4_N_ROT)`. gated together with kv_fp8_store so we
 // can A/B both KV-write quant sites under one flag (DS4_FP8_KV_QUANT=1).
 let want_fp8_comp = std::env::var("DS4_FP8_KV_QUANT").ok().as_deref() == Some("1");
 if hd == 512 && n_rot > 0 && want_fp8_comp {
 ds4_fp8_kv_quantize_row_inplace(&mut out_comp, hd, n_rot);
 }

 // 7. Ratio-4 rotation: second half → first half, then mirror back so
 // the next 4 positions overwrite the now-stale "previous window"
 // slots while the indexer lane keeps the just-finished window.
 if ratio == 4 {
 let r = ratio as usize;
 for k in 0..r {
 let src = ((r + k) * width)..((r + k + 1) * width);
 let dst = (k * width)..((k + 1) * width);
 state_kv.copy_within(src.clone(), dst.start);
 state_score.copy_within(src, dst.start);
 }
 for k in 0..r {
 let src = (k * width)..((k + 1) * width);
 let dst = ((r + k) * width)..((r + k + 1) * width);
 state_kv.copy_within(src.clone(), dst.start);
 state_score.copy_within(src, dst.start);
 }
 }

 Some(out_comp)
}

/// Antirez `DS4_N_INDEXER_HEAD` (ds4.h:100).
pub const DS4_N_INDEXER_HEAD: u32 = 64;
/// Antirez `DS4_N_INDEXER_HEAD_DIM` (ds4.h:101).
pub const DS4_N_INDEXER_HEAD_DIM: u32 = 128;
/// Antirez `DS4_N_INDEXER_TOP_K` (ds4.h:102).
pub const DS4_N_INDEXER_TOP_K: u32 = 512;

/// Effective indexer top-k cap, with a test/diagnostic env override
/// (`DS4_N_INDEXER_TOP_K_OVERRIDE`). Production default is the const (512).
/// The override lets the long-context validation harness engage the real
/// staged indexer path early — a small threshold means `n_comp` crosses it
/// after ~`override * ratio` tokens, avoiding the ~2048-token warmup (which is
/// memory-bound on the full model). Invalid/zero values fall back to the const.
pub fn ds4_n_indexer_top_k() -> u32 {
 std::env::var("DS4_N_INDEXER_TOP_K_OVERRIDE")
 .ok()
 .and_then(|s| s.parse::<u32>().ok())
 .filter(|&v| v >= 1)
 .unwrap_or(DS4_N_INDEXER_TOP_K)
}

/// Inputs to `indexer_allowed_decode_one`.
///
/// All slices are owned upstream; the function returns an `allowed` bitmap
/// of length `n_comp`. The 3 weight matrices below are the indexer-specific
/// projection tensors loaded from per-layer GGUF role names
/// `indexer.attn_q_b` and `indexer.proj` (F16 in `DS4_Q2_LAYOUT`).
///
/// Antirez ref: `indexer_allowed_decode_one` (ds4.c:6633-6692) and
/// `indexer_allowed_decode_one_decode_scratch` (ds4.c:6695-6754).
pub struct IndexerInputs<'a> {
 /// `[n_lora_q, n_head*head_dim]` flat (col-major as antirez stores it):
 /// element `(in=k, out=j)` at `w[j*n_lora_q + k]`. Maps `qr_norm` →
 /// `q[n_head*head_dim]`.
 pub w_q_b: &'a [f32],
 /// `[d_embd, n_head]` flat (col-major): element `(in=k, out=h)` at
 /// `w[h*d_embd + k]`. Maps `cur` (= `normed`) → `weights[n_head]`.
 pub w_proj: &'a [f32],
 /// Rolling cache of per-compressed-row indexer KV. Shape:
 /// `[n_comp, head_dim]` flat, `head_dim = DS4_N_INDEXER_HEAD_DIM = 128`.
 /// Produced by a second `compressor_decode_one` call on the same layer
 /// using `indexer_compressor_{kv,gate,ape,norm}` weights.
 pub index_comp_kv: &'a [f32],
 /// Number of valid rows in `index_comp_kv`.
 pub n_comp: u32,
 /// Indexer head count (= `DS4_N_INDEXER_HEAD`).
 pub n_head: u32,
 /// Indexer head dim (= `DS4_N_INDEXER_HEAD_DIM`).
 pub head_dim: u32,
 /// Top-k cap (= `DS4_N_INDEXER_TOP_K`).
 pub top_k: u32,
}

/// Streaming indexer-top-k selection for one decode token. Port of antirez
/// `indexer_allowed_decode_one` (ds4.c:6633-6692). Returns `(allowed, n_selected)`:
///
/// - When `n_comp <= top_k`, every row is allowed and the function returns
/// `(allowed[i]=true for i in 0..n_comp, n_comp)`. Antirez early-returns
/// here without computing any matvec.
/// - Otherwise computes scores per row and picks the top `top_k` indices.
///
/// `cur` is the post-attn-norm residual (== `normed` upstream), `qr_norm` is
/// the Q-LoRA mid-stage after `rms_norm(qr, gamma_q)`. Both come from
/// `decode_step_with_attn` for free — see #267 wiring there.
pub fn indexer_allowed_decode_one<A: AttentionDispatcher>(
 dispatch: &A,
 params: &LayerParams,
 inputs: &IndexerInputs<'_>,
 cur: &[f32],
 qr_norm: &[f32],
 pos: u32,
) -> (Vec<bool>, u32) {
 let n_comp = inputs.n_comp as usize;
 let mut allowed = vec![false; n_comp];
 if n_comp == 0 {
 return (allowed, 0);
 }

 let top_k = (inputs.top_k as usize).min(n_comp);
 if top_k == n_comp {
 for a in allowed.iter_mut() {
 *a = true;
 }
 return (allowed, n_comp as u32);
 }

 let head_dim = inputs.head_dim as usize;
 let n_head = inputs.n_head as usize;
 let n_lora_q = qr_norm.len();
 let d_embd = cur.len();

 debug_assert_eq!(inputs.w_q_b.len(), n_lora_q * n_head * head_dim);
 debug_assert_eq!(inputs.w_proj.len(), d_embd * n_head);
 debug_assert_eq!(inputs.index_comp_kv.len(), n_comp * head_dim);

 // 1. q = w_q_b · qr_norm — shape `[n_head*head_dim]` col-major matvec.
 // Antirez `matvec_any` (ds4.c:6657) routes through dot_f32 internally;
 // use the same reduction-tree fidelity env so the
 // indexer matvec switches in lockstep with the rest of attn-dispatch.
 let q_dim = n_head * head_dim;
 let antirez_dot_fidelity =
 std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
 let mut q = vec![0.0f32; q_dim];
 for j in 0..q_dim {
 let row = &inputs.w_q_b[j * n_lora_q..(j + 1) * n_lora_q];
 q[j] = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(row, qr_norm)
 } else {
 let mut acc = 0.0f32;
 for k in 0..n_lora_q {
 acc += row[k] * qr_norm[k];
 }
 acc
 };
 }

 // 2. rope_tail on q: per-head, rotate the trailing `n_rot` dims at `pos`.
 let n_rot = params.n_rot as usize;
 if n_rot > 0 {
 for head in q.chunks_mut(head_dim) {
 let tail = &mut head[head_dim - n_rot..head_dim];
 dispatch.rope_tail(params, tail, pos, false);
 }
 }

 // 3. weights[h] = w_proj · cur — shape `[n_head]`, scaled by
 // `1/sqrt(head_dim * n_head)` (antirez ds4.c:6661).
 let mut weights = vec![0.0f32; n_head];
 for h in 0..n_head {
 let row = &inputs.w_proj[h * d_embd..(h + 1) * d_embd];
 weights[h] = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(row, cur)
 } else {
 let mut acc = 0.0f32;
 for k in 0..d_embd {
 acc += row[k] * cur[k];
 }
 acc
 };
 }
 let scale = 1.0f32 / ((head_dim * n_head) as f32).sqrt();
 for w in weights.iter_mut() {
 *w *= scale;
 }

 // 4. Per-comp-row score: Σ_h max(0, dot(kv_c, q_h)) * weights[h].
 let mut scores = vec![0.0f32; n_comp];
 for c in 0..n_comp {
 let kv = &inputs.index_comp_kv[c * head_dim..(c + 1) * head_dim];
 let mut s = 0.0f32;
 for h in 0..n_head {
 let qh = &q[h * head_dim..(h + 1) * head_dim];
 let mut dot = if antirez_dot_fidelity {
 crate::forward::dot_f32_antirez(kv, qh)
 } else {
 let mut acc = 0.0f32;
 for k in 0..head_dim {
 acc += kv[k] * qh[k];
 }
 acc
 };
 if dot < 0.0 {
 dot = 0.0;
 }
 s += dot * weights[h];
 }
 scores[c] = s;
 }

 // 5. Top-k pass: K linear scans for argmax over `!allowed[c]`. Antirez
 // does the same (ds4.c:6676-6686) — O(K*N), fine because top_k ≤ 512
 // and n_comp grows by 1 per emit step (so K << N only when context is
 // very long).
 for _ in 0..top_k {
 let mut best = 0usize;
 let mut best_score = f32::NEG_INFINITY;
 for c in 0..n_comp {
 if !allowed[c] && scores[c] > best_score {
 best = c;
 best_score = scores[c];
 }
 }
 allowed[best] = true;
 }
 (allowed, top_k as u32)
}

/// Run step 1 of the attention half in isolation: `hc_collapse_norm(Attn)`.
/// Exposed so the composed driver can compute `normed` once and feed it to
/// the compressor before re-entering `decode_attn_prefix_with`,
/// which will independently recompute `normed`. The duplication is bit-exact
/// and cheap (one HC norm per compressed layer).
pub fn attn_collapse_norm_only<A: AttentionDispatcher>(
 dispatch: &A,
 layer: &AttnLayerWeights,
 cur_hc: &[f32],
) -> Vec<f32> {
 let p = &layer.params;
 let (_collapsed, normed, _split) = dispatch.hc_collapse_norm(
 p,
 HcKind::Attn,
 &layer.hc_attn_fn,
 &layer.hc_attn_scale,
 &layer.hc_attn_base,
 cur_hc,
 Some(&layer.hc_norm_gamma),
 );
 normed
}

/// Precomputed (c, s) interleaved table for `rope_tail`, indexed by rot-pair.
///
/// Layout: `table[2*i] = cos(θ_i) * mscale_i`, `table[2*i + 1] = sin(θ_i) * mscale_i`
/// for `i = 0 .. n_rot/2`. For `backward=true` the sign of `s` is flipped at
/// build time so the apply loop is direction-agnostic.
///
/// Lifecycle: build ONCE per decode token (cos/sin depend only on `pos` and
/// the rope params, both constant across the 43 layers in one step) and reuse
/// for the Q-side forward (step 5) and Q-side backward (step 7) rotations. The
/// per-call cost of `rope_tail` is 15ms/layer × 43 = 645ms/token of
/// pure trig work; this collapses it to ~16ms/token (one table build per
/// direction + 43 layers of f32 mul-add).
pub fn precompute_rope_tail_table(
 n_rot: usize,
 pos: u32,
 backward: bool,
 freq_base: f32,
 freq_scale: f32,
 ext_factor: f32,
 attn_factor: f32,
 n_ctx_orig: u32,
) -> Vec<f32> {
 if n_rot == 0 {
 return Vec::new();
 }
 let theta_scale = freq_base.powf(-2.0 / (n_rot as f32));
 let sin_sign = if backward { -1.0 } else { 1.0 };
 let beta_fast = 32.0f32;
 let beta_slow = 1.0f32;

 let mut corr_lo = 0.0f32;
 let mut corr_hi = 0.0f32;
 if ext_factor != 0.0 {
 let corr_dim = |b: f32| -> f32 {
 (n_rot as f32) * ((n_ctx_orig as f32) / (b * 2.0 * PI)).ln() / (2.0 * freq_base.ln())
 };
 let start = corr_dim(beta_fast).floor();
 let end = corr_dim(beta_slow).ceil();
 corr_lo = start.max(0.0);
 corr_hi = end.min((n_rot as f32) - 1.0);
 }

 let pairs = n_rot / 2;
 let mut table = Vec::with_capacity(pairs * 2);
 let mut theta_extrap = pos as f32;
 let mut i = 0usize;
 while i + 1 < n_rot {
 let theta_interp = freq_scale * theta_extrap;
 let mut theta = theta_interp;
 let mut mscale = attn_factor;
 if ext_factor != 0.0 {
 let y = ((i as f32 / 2.0) - corr_lo) / (corr_hi - corr_lo).max(0.001);
 let ramp = 1.0 - y.clamp(0.0, 1.0);
 let ramp_mix = ramp * ext_factor;
 theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
 if freq_scale > 0.0 {
 mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
 }
 }
 let c = theta.cos() * mscale;
 let s = sin_sign * theta.sin() * mscale;
 table.push(c);
 table.push(s);
 theta_extrap *= theta_scale;
 i += 2;
 }
 table
}

/// Apply a precomputed rope-tail table to a buffer of `(n_heads * n_rot)`
/// floats (or any multiple of `n_rot`). Each `n_rot`-wide chunk is one head's
/// rope-tail slice; the table is reused across all chunks. Mirrors the inner
/// loop of `AttentionDispatcher::rope_tail` exactly.
#[inline]
pub fn apply_rope_tail_with_table(x: &mut [f32], table: &[f32], n_rot: usize) {
 if n_rot == 0 || table.is_empty() {
 return;
 }
 debug_assert_eq!(table.len(), n_rot); // (c, s) pairs × (n_rot / 2) = n_rot floats
 debug_assert_eq!(x.len() % n_rot, 0);
 for tail in x.chunks_mut(n_rot) {
 let mut i = 0usize;
 let mut j = 0usize;
 while i + 1 < n_rot {
 let c = table[j];
 let s = table[j + 1];
 let x0 = tail[i];
 let x1 = tail[i + 1];
 tail[i] = x0 * c - x1 * s;
 tail[i + 1] = x0 * s + x1 * c;
 i += 2;
 j += 2;
 }
 }
}

/// Run steps 1–8 of the attention half (through `attn_output_proj`).
///
/// this is the seam between attention and MoE/router. After this
/// returns, the caller runs the `KernelDispatcher` MoE half on `normed`,
/// produces a `routed_out`, then calls `decode_attn_suffix_with` to fold
/// the shared expert and HC-expand into the next-layer residual.
pub fn decode_attn_prefix_with<A: AttentionDispatcher>(
 dispatch: &A,
 layer: &AttnLayerWeights,
 step: &mut AttnStepInputs<'_>,
) -> AttnPrefixOut {
 let p = &layer.params;
 let n_rot = p.n_rot as usize;
 let n_lora_kv = p.n_lora_kv as usize;
 let _ds4_attn_trace = std::env::var("DS4_ATTN_TRACE").is_ok();

 // env-gated batched-prefix path. Default OFF
 // preserves the existing 6-trait-call sequential body below; ON routes
 // through `attn_prefix_batched` so backends that override it can fuse
 // the GPU dispatches. Default impl is bit-identical to the sequential
 // body (same trait calls, same order).
 if std::env::var("DS4_ATTN_PREFIX_BATCHED").ok().as_deref() == Some("1") {
 let out = dispatch.attn_prefix_batched(
 p,
 &layer.hc_attn_fn,
 &layer.hc_attn_scale,
 &layer.hc_attn_base,
 step.cur_hc,
 &layer.hc_norm_gamma,
 step.kv_raw_row,
 &layer.qkv_gamma_kv,
 step.pos,
 &mut step.kv_view,
 step.q_heads,
 step.n_raw,
 step.raw_start,
 step.kv_comp_rows,
 step.n_comp,
 step.comp_selected,
 step.n_selected,
 &layer.attn_sinks,
 &layer.w_o_a,
 &layer.w_o_b,
 );
 return AttnPrefixOut {
 after_attn_hc: out.after_attn_hc,
 hc_split_attn: out.hc_split_attn,
 normed: out.normed,
 };
 }

 let t_hc = std::time::Instant::now();
 // 1. Pre-attention HC collapse + norm (HcKind::Attn).
 let (_cur_collapsed, normed, hc_split_attn) = dispatch.hc_collapse_norm(
 p,
 HcKind::Attn,
 &layer.hc_attn_fn,
 &layer.hc_attn_scale,
 &layer.hc_attn_base,
 step.cur_hc,
 Some(&layer.hc_norm_gamma),
 );
 let hc_us = t_hc.elapsed().as_micros();
 let n_hc = p.n_hc as usize;
 let hc_split_post_attn = &hc_split_attn[n_hc..2 * n_hc];
 let hc_split_comb_attn = &hc_split_attn[2 * n_hc..2 * n_hc + n_hc * n_hc];

 let t_kvrms = std::time::Instant::now();
 // 2. KV RMS norm (NOPE part only on KV). Phase C.3a.3:
 // qr arm is dead in the decode hot path (qr_normed is produced
 // upstream by `KernelDispatcher::layer_qa_rms_batched`), so we
 // call the kv-only variant.
 let mut kv_normed = dispatch.kv_rms_norm_row(p, step.kv_raw_row, &layer.qkv_gamma_kv);
 let kvrms_us = t_kvrms.elapsed().as_micros();

 let t_rope_kv = std::time::Instant::now();
 // 3. RoPE-tail on the KV row's rope slice (last `n_rot` of the
 // `n_lora_kv`-wide row, per antirez `ds4.c:6792`).
 dispatch.rope_tail(
 p,
 &mut kv_normed[n_lora_kv - n_rot..n_lora_kv],
 step.pos,
 false,
 );
 let rope_kv_us = t_rope_kv.elapsed().as_micros();

 let t_kvstore = std::time::Instant::now();
 // 4. Append KV row to cache.
 dispatch.kv_fp8_store(p, &kv_normed, &mut step.kv_view);
 let kvstore_us = t_kvstore.elapsed().as_micros();

 let t_rope_q = std::time::Instant::now();
 // 5. RoPE-tail on Q heads. precompute (c, s) ONCE — cos/sin
 // depend only on `pos` + rope params (constant across layers in a
 // decode step), so collapse 43 × (128 heads × 32 trig pairs) trig
 // calls into one table build + 43 × f32 mul-add layer-applies.
 let mut q_heads_rot = step.q_heads.to_vec();
 let rope_q_table = if n_rot > 0 {
 precompute_rope_tail_table(
 n_rot,
 step.pos,
 false,
 p.rope_freq_base,
 p.rope_freq_scale,
 p.rope_ext_factor,
 p.rope_attn_factor,
 p.rope_orig_ctx,
 )
 } else {
 Vec::new()
 };
 if n_rot > 0 {
 let head_dim = p.head_dim as usize;
 debug_assert_eq!(q_heads_rot.len() % head_dim, 0);
 for head in q_heads_rot.chunks_mut(head_dim) {
 let tail = &mut head[head_dim - n_rot..];
 apply_rope_tail_with_table(tail, &rope_q_table, n_rot);
 }
 }
 let rope_q_us = t_rope_q.elapsed().as_micros();

 let t_fa = std::time::Instant::now();
 // 6. Flash attention against the raw KV window (+ optional compressed rows).
 let heads = dispatch.flash_attn_decode(
 p,
 &q_heads_rot,
 step.kv_view.raw,
 step.n_raw,
 step.kv_view.raw_cap,
 step.raw_start,
 step.kv_comp_rows,
 step.n_comp,
 step.comp_selected,
 step.n_selected,
 &layer.attn_sinks,
 );
 // Verifier-bisection flash tap: capture this layer's flash heads output.
 FLASH_CAPTURE.with(|c| {
 let mut cc = c.borrow_mut();
 if cc.0 != usize::MAX && cc.0 == CURRENT_LAYER_HINT.with(|h| h.get()) {
 cc.1 = heads.clone();
 }
 });
 let fa_us = t_fa.elapsed().as_micros();

 if _ds4_attn_trace {
 eprintln!(
 "DS4_ATTN_TRACE,pos={},n_raw={},n_sel={},hc={},kvrms={},rope_kv={},kvstore={},rope_q={},fa={}",
 step.pos, step.n_raw, step.n_selected, hc_us, kvrms_us, rope_kv_us, kvstore_us, rope_q_us, fa_us
 );
 }

 // 7. RoPE-tail backward on output heads. reuse precompute
 // pattern with backward=true (sign of s flipped at table build time).
 let mut heads_back = heads;
 if n_rot > 0 {
 let head_dim = p.head_dim as usize;
 let rope_q_back_table = precompute_rope_tail_table(
 n_rot,
 step.pos,
 true,
 p.rope_freq_base,
 p.rope_freq_scale,
 p.rope_ext_factor,
 p.rope_attn_factor,
 p.rope_orig_ctx,
 );
 for head in heads_back.chunks_mut(head_dim) {
 let tail = &mut head[head_dim - n_rot..];
 apply_rope_tail_with_table(tail, &rope_q_back_table, n_rot);
 }
 }

 // 8. W_o low-rank + HC expand using post[n_hc..2*n_hc] from the pre-attn
 // split (matches antirez `hc_post_one` arg 4, ds4.c:6851 / 7614).
 let after_attn_hc = dispatch.attn_output_proj(
 p,
 &heads_back,
 &layer.w_o_a,
 &layer.w_o_b,
 step.cur_hc,
 hc_split_post_attn,
 hc_split_comb_attn,
 );

 // Per-step magnitude trace (env-gated). For bisecting #260 (post-#259
 // residual growth). Prints RMS of every intermediate at pos=0 layer 0.
 if step.pos == 0 && std::env::var("DS4_DUMP_ATTN_RMS").ok().as_deref() == Some("1") {
 fn rms(v: &[f32]) -> f32 {
 let n = v.len() as f64;
 let ss: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
 (ss / n).sqrt() as f32
 }
 eprintln!(
 "ATTN_RMS pos=0 cur_hc={:.4} normed={:.4} kv_raw={:.4} q_heads={:.4} kv_normed={:.4} heads={:.4} heads_back={:.4} post={:.4} comb={:.4} after_attn_hc={:.4}",
 rms(step.cur_hc),
 rms(&normed),
 rms(step.kv_raw_row),
 rms(step.q_heads),
 rms(&kv_normed),
 rms(&q_heads_rot),
 rms(&heads_back),
 rms(hc_split_post_attn),
 rms(hc_split_comb_attn),
 rms(&after_attn_hc),
 );
 }

 // dump per-slot values of hc_split_post_attn and per-slot RMS of
 // after_attn_hc, gated on DS4_DUMP_HC_SPLIT_AT_POS=<pos> (e.g. 27 for the
 // first decode step after a 27-token prompt). Layer index printed so a
 // single run produces an L0…L42 fingerprint at the target position.
 if let Ok(target) = std::env::var("DS4_DUMP_HC_SPLIT_AT_POS") {
 if let Ok(pos) = target.parse::<u32>() {
 if step.pos == pos {
 let n_hc = p.n_hc as usize;
 let d_embd = p.d_embd as usize;
 let mut comb_rowsum = vec![0.0f32; n_hc];
 for dst in 0..n_hc {
 for src in 0..n_hc {
 comb_rowsum[dst] += hc_split_comb_attn[dst + src * n_hc];
 }
 }
 let mut after_slot_rms = vec![0.0f32; n_hc];
 for h in 0..n_hc {
 let slot = &after_attn_hc[h * d_embd..(h + 1) * d_embd];
 let ss: f64 = slot.iter().map(|&x| (x as f64) * (x as f64)).sum();
 after_slot_rms[h] = (ss / d_embd as f64).sqrt() as f32;
 }
 eprintln!(
 "HC_SPLIT_DBG il={} pos={} post=[{:.4}, {:.4}, {:.4}, {:.4}] comb_rowsum=[{:.4}, {:.4}, {:.4}, {:.4}] after_slot_rms=[{:.4}, {:.4}, {:.4}, {:.4}]",
 p.layer_idx,
 pos,
 hc_split_post_attn[0],
 hc_split_post_attn[1],
 hc_split_post_attn[2],
 hc_split_post_attn[3],
 comb_rowsum[0],
 comb_rowsum[1],
 comb_rowsum[2],
 comb_rowsum[3],
 after_slot_rms[0],
 after_slot_rms[1],
 after_slot_rms[2],
 after_slot_rms[3],
 );
 if p.layer_idx <= 1 {
 let mut row_strs = Vec::new();
 for r in 0..n_hc {
 let row: Vec<String> = (0..n_hc)
 .map(|c| format!("{:.4}", hc_split_comb_attn[r + c * n_hc]))
 .collect();
 row_strs.push(format!("[{}]", row.join(",")));
 }
 eprintln!(
 "HC_COMB_DBG il={} pos={} comb_matrix={{{}}}",
 p.layer_idx,
 pos,
 row_strs.join(", ")
 );
 }
 }
 }
 }

 AttnPrefixOut {
 after_attn_hc,
 hc_split_attn,
 normed,
 }
}

/// Output of `decode_attn_ffn_pre_with`: the FFN-half `hc_collapse_norm` plus
/// the split tensors needed by the post-MoE step. `ffn_normed` is the
/// d_embd-vector that antirez calls `_ffn_norm` (ds4.c:9251) — it is the
/// MoE/router/shared-expert input.
pub struct FfnPreOut {
 pub ffn_normed: Vec<f32>,
 /// `hc_split_ffn` from `hc_collapse_norm(HcKind::Ffn)` — `[pre | post | comb]`.
 pub hc_split_ffn: Vec<f32>,
}

/// Run step 8b in isolation: pre-FFN `hc_collapse_norm` over `after_attn_hc`.
///
/// Produces `ffn_normed` (the MoE input) and `hc_split_ffn` (consumed by the
/// post-MoE step). Antirez `ds4.c:9233-9251`: this is the FFN-half HC mix-norm
/// fused block.
pub fn decode_attn_ffn_pre_with<A: AttentionDispatcher>(
 dispatch: &A,
 layer: &AttnLayerWeights,
 prefix: &AttnPrefixOut,
) -> FfnPreOut {
 let p = &layer.params;
 // hc_ffn_fn is F16-no-copy under lean (empty f32 Vec) — reconstruct it for
 // this CPU hc-collapse (the GPU encoder reads f16 directly). Cow::Borrowed
 // (no alloc) when the f32 is present (non-lean / dense layers).
 let hc_ffn_fn = layer.hc_ffn_fn_as_f32();
 let (_ffn_cur, ffn_normed, hc_split_ffn) = dispatch.hc_collapse_norm(
 p,
 HcKind::Ffn,
 &hc_ffn_fn,
 &layer.hc_ffn_scale,
 &layer.hc_ffn_base,
 &prefix.after_attn_hc,
 Some(&layer.hc_ffn_norm_gamma),
 );
 FfnPreOut {
 ffn_normed,
 hc_split_ffn,
 }
}

/// Run steps 9–10 of the attention half (shared expert + add routed).
///
/// `routed_out` is the MoE/router output produced between the prefix and
/// suffix calls. `prefix` is the output of `decode_attn_prefix_with`.
pub fn decode_attn_suffix_with<A: AttentionDispatcher>(
 dispatch: &A,
 layer: &AttnLayerWeights,
 prefix: &AttnPrefixOut,
 routed_out: &[f32],
 pos: u32,
) -> Vec<f32> {
 let p = &layer.params;
 let d_embd = p.d_embd as usize;
 debug_assert_eq!(prefix.normed.len(), d_embd);

 let FfnPreOut {
 ffn_normed,
 hc_split_ffn,
 } = decode_attn_ffn_pre_with(dispatch, layer, prefix);
 let n_hc = p.n_hc as usize;
 let hc_split_post = &hc_split_ffn[n_hc..2 * n_hc];
 let hc_split_comb_ffn = &hc_split_ffn[2 * n_hc..2 * n_hc + n_hc * n_hc];

 // dump ffn_normed + hc_split_ffn at pos=DS4_DUMP_FFN_NORMED_AT_POS for L0/L1.
 // Compare element-wise against antirez `_ffn_norm-<L>_pos<P>.bin` to localize
 // the FFN-half direction divergence (input prefix.after_attn_hc matches antirez,
 // output ffn_normed direction does not).
 if let Ok(target) = std::env::var("DS4_DUMP_FFN_NORMED_AT_POS") {
 if let Ok(target_pos) = target.parse::<u32>() {
 let layer_idx = p.layer_idx;
 if pos == target_pos && layer_idx <= 1 {
 let n_norm = ffn_normed.len() as f64;
 let ss: f64 = ffn_normed.iter().map(|&v| (v as f64) * (v as f64)).sum();
 let rms = (ss / n_norm).sqrt();
 eprintln!(
 "FFN_NORMED_DUMP il={} pos={} ffn_normed_len={} rms={:.6} first8=[{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}]",
 layer_idx,
 pos,
 ffn_normed.len(),
 rms,
 ffn_normed[0],
 ffn_normed[1],
 ffn_normed[2],
 ffn_normed[3],
 ffn_normed[4],
 ffn_normed[5],
 ffn_normed[6],
 ffn_normed[7],
 );
 // Dump hc_split_ffn structure: [pre | post | comb].
 let pre = &hc_split_ffn[0..n_hc];
 let post = &hc_split_ffn[n_hc..2 * n_hc];
 eprintln!(
 "FFN_NORMED_DUMP il={} pos={} hc_split_pre={:?} hc_split_post={:?}",
 layer_idx, pos, pre, post,
 );
 // Also dump prefix.after_attn_hc per-slot RMS to confirm input matches.
 let hc_dim = (p.n_hc as usize) * d_embd;
 debug_assert_eq!(prefix.after_attn_hc.len(), hc_dim);
 let mut slot_rms = vec![0.0f64; p.n_hc as usize];
 for s in 0..(p.n_hc as usize) {
 let off = s * d_embd;
 let ss_s: f64 = prefix.after_attn_hc[off..off + d_embd]
 .iter()
 .map(|&v| (v as f64) * (v as f64))
 .sum();
 slot_rms[s] = (ss_s / d_embd as f64).sqrt();
 }
 eprintln!(
 "FFN_NORMED_DUMP il={} pos={} after_attn_hc_slot_rms={:?}",
 layer_idx, pos, slot_rms,
 );
 }
 }
 }

 // 9. Shared expert on ffn_normed.
 let shared_mid = dispatch.shared_expert(
 p,
 &ffn_normed,
 &layer.w_shared_gate,
 &layer.w_shared_up,
 layer.shared_dim,
 layer.shared_clamp,
 );

 // 10. Shared-down + HC expand + add routed (uses post[n_hc..2n_hc] +
 // comb[2n_hc..2n_hc+n_hc²]).
 let after_ffn_hc = dispatch.shared_down_hc_expand_add(
 p,
 &shared_mid,
 &layer.w_shared_down,
 routed_out,
 &prefix.after_attn_hc,
 hc_split_post,
 hc_split_comb_ffn,
 );

 // FFN-half dump, gated on DS4_DUMP_HC_SPLIT_AT_POS=<pos>.
 // Prints layer index alongside slot fingerprint for cross-layer comparison.
 if let Ok(target) = std::env::var("DS4_DUMP_HC_SPLIT_AT_POS") {
 if let Ok(target_pos) = target.parse::<u32>() {
 if pos == target_pos {
 let n_hc = p.n_hc as usize;
 let mut comb_rowsum = vec![0.0f32; n_hc];
 for dst in 0..n_hc {
 for src in 0..n_hc {
 comb_rowsum[dst] += hc_split_comb_ffn[dst + src * n_hc];
 }
 }
 let mut after_slot_rms = vec![0.0f32; n_hc];
 for h in 0..n_hc {
 let slot = &after_ffn_hc[h * d_embd..(h + 1) * d_embd];
 let ss: f64 = slot.iter().map(|&x| (x as f64) * (x as f64)).sum();
 after_slot_rms[h] = (ss / d_embd as f64).sqrt() as f32;
 }
 eprintln!(
 "HC_FFN_DBG il={} pos={} ffn_post=[{:.4}, {:.4}, {:.4}, {:.4}] ffn_comb_rowsum=[{:.4}, {:.4}, {:.4}, {:.4}] after_ffn_slot_rms=[{:.4}, {:.4}, {:.4}, {:.4}]",
 p.layer_idx,
 target_pos,
 hc_split_post[0],
 hc_split_post[1],
 hc_split_post[2],
 hc_split_post[3],
 comb_rowsum[0],
 comb_rowsum[1],
 comb_rowsum[2],
 comb_rowsum[3],
 after_slot_rms[0],
 after_slot_rms[1],
 after_slot_rms[2],
 after_slot_rms[3],
 );
 }
 }
 }
 after_ffn_hc
}

/// Antirez `hc_post_one` (ds4.c:4225-4236) post-amble — the HC expand-add
/// fold from `shared_down_hc_expand_add` minus the down-matvec. Used by the
/// C.3b.3 fast path: when `shared_chain_batched` runs the whole gate+up+
/// SwiGLU+(q80)+down chain in one MTLCommandBuffer, the resulting
/// `shared_out` still needs to be expand-add-folded against `routed_out` and
/// `after_attn_hc` to produce `after_ffn_hc`. Mirrors lines 1470-1488 of
/// `CpuAttentionDispatcher::shared_down_hc_expand_add` verbatim.
pub fn hc_expand_add_only(
 params: &LayerParams,
 shared_out: &[f32],
 routed_out: &[f32],
 after_attn_hc: &[f32],
 hc_split_post: &[f32],
 hc_split_comb: &[f32],
) -> Vec<f32> {
 let d_embd = params.d_embd as usize;
 let n_hc = params.n_hc as usize;
 debug_assert_eq!(shared_out.len(), d_embd);
 debug_assert_eq!(routed_out.len(), d_embd);
 debug_assert_eq!(hc_split_post.len(), n_hc);
 debug_assert_eq!(hc_split_comb.len(), n_hc * n_hc);
 debug_assert_eq!(after_attn_hc.len(), n_hc * d_embd);
 let mut after = vec![0.0f32; n_hc * d_embd];
 for dst in 0..n_hc {
 let base = dst * d_embd;
 let w_post = hc_split_post[dst];
 for e in 0..d_embd {
 let mut acc = w_post * (shared_out[e] + routed_out[e]);
 for src in 0..n_hc {
 acc += hc_split_comb[dst + src * n_hc] * after_attn_hc[src * d_embd + e];
 }
 after[base + e] = acc;
 }
 }
 after
}

/// Run steps 9–10 of the attention half given a precomputed `FfnPreOut` and the
/// MoE/router `routed_out`. The driver uses this to feed the MoE the correct
/// `ffn_normed` (not the attention-half `prefix.normed`).
///
/// under `DS4_SILU_FIDELITY=0` (default-branch silu),
/// routes the shared-expert FFN body (gate matvec + up matvec + SwiGLU +
/// optional q8_0 + down matvec) through `K::shared_chain_batched`, which the
/// Metal backend overrides to pack into a single MTLCommandBuffer (saves up to
/// 3 cwr per layer when `DS4_Q8_0_ACT=1`). The HC-expand-add post-amble runs
/// separately via `hc_expand_add_only`. Under `DS4_SILU_FIDELITY=1`
/// the batched primitive is unsafe (Metal swiglu kernel uses default-branch
/// silu only), so we fall back to the sequential `A::shared_expert` +
/// `A::shared_down_hc_expand_add` trait calls — same behaviour as before
/// C.3b.3.
pub fn decode_attn_ffn_post_with<K: crate::dispatch::KernelDispatcher, A: AttentionDispatcher>(
 k: &K,
 dispatch: &A,
 layer: &AttnLayerWeights,
 prefix: &AttnPrefixOut,
 ffn_pre: &FfnPreOut,
 routed_out: &[f32],
 pos: u32,
 // B Slice 5-redo: if `Some(s)`, use `s` as `shared_out`
 // instead of computing it via `k.shared_chain_batched`. The fused
 // `moe_and_shared_chain_batched` produces this; callers under the
 // fidelity branch (`DS4_SILU_FIDELITY=1`) pass `None`. The caller
 // is responsible for the q8_0 round-trip on the ffn_in input (the
 // fused method takes care of that internally).
 precomputed_shared_out: Option<&[f32]>,
) -> Vec<f32> {
 let p = &layer.params;
 let d_embd = p.d_embd as usize;
 let n_hc = p.n_hc as usize;
 let hc_split_post = &ffn_pre.hc_split_ffn[n_hc..2 * n_hc];
 let hc_split_comb_ffn = &ffn_pre.hc_split_ffn[2 * n_hc..2 * n_hc + n_hc * n_hc];

 let silu_fidelity = std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
 let want_q80 = std::env::var("DS4_Q8_0_ACT").ok().as_deref() == Some("1");

 let after_ffn_hc = if silu_fidelity {
 let shared_mid = dispatch.shared_expert(
 p,
 &ffn_pre.ffn_normed,
 &layer.w_shared_gate,
 &layer.w_shared_up,
 layer.shared_dim,
 layer.shared_clamp,
 );

 dispatch.shared_down_hc_expand_add(
 p,
 &shared_mid,
 &layer.w_shared_down,
 routed_out,
 &prefix.after_attn_hc,
 hc_split_post,
 hc_split_comb_ffn,
 )
 } else {
 // Mirror `CpuAttentionDispatcher::shared_expert` q80(ffn_norm) when
 // DS4_Q8_0_ACT=1 — the C.3b primitive does NOT q80 the activation
 // input itself, so we round-trip here to preserve bit-identity with
 // the trait path.
 let shared_owned;
 let shared_out: &[f32] = if let Some(s) = precomputed_shared_out {
 s
 } else {
 let ffn_owned;
 let ffn_in: &[f32] = if want_q80 {
 ffn_owned = crate::forward::q8_0_round_trip(&ffn_pre.ffn_normed);
 &ffn_owned
 } else {
 &ffn_pre.ffn_normed
 };
 shared_owned = k.shared_chain_batched(
 ffn_in,
 &layer.w_shared_gate,
 &layer.w_shared_up,
 &layer.w_shared_down,
 layer.shared_dim,
 want_q80,
 );
 &shared_owned
 };
 hc_expand_add_only(
 p,
 shared_out,
 routed_out,
 &prefix.after_attn_hc,
 hc_split_post,
 hc_split_comb_ffn,
 )
 };

 if let Ok(target) = std::env::var("DS4_DUMP_HC_SPLIT_AT_POS") {
 if let Ok(target_pos) = target.parse::<u32>() {
 if pos == target_pos {
 let mut comb_rowsum = vec![0.0f32; n_hc];
 for dst in 0..n_hc {
 for src in 0..n_hc {
 comb_rowsum[dst] += hc_split_comb_ffn[dst + src * n_hc];
 }
 }
 let mut after_slot_rms = vec![0.0f32; n_hc];
 for h in 0..n_hc {
 let slot = &after_ffn_hc[h * d_embd..(h + 1) * d_embd];
 let ss: f64 = slot.iter().map(|&x| (x as f64) * (x as f64)).sum();
 after_slot_rms[h] = (ss / d_embd as f64).sqrt() as f32;
 }
 eprintln!(
 "HC_FFN_DBG il={} pos={} ffn_post=[{:.4}, {:.4}, {:.4}, {:.4}] ffn_comb_rowsum=[{:.4}, {:.4}, {:.4}, {:.4}] after_ffn_slot_rms=[{:.4}, {:.4}, {:.4}, {:.4}]",
 p.layer_idx,
 target_pos,
 hc_split_post[0],
 hc_split_post[1],
 hc_split_post[2],
 hc_split_post[3],
 comb_rowsum[0],
 comb_rowsum[1],
 comb_rowsum[2],
 comb_rowsum[3],
 after_slot_rms[0],
 after_slot_rms[1],
 after_slot_rms[2],
 after_slot_rms[3],
 );
 }
 }
 }
 after_ffn_hc
}

/// Drive one attention-half decode layer through any `AttentionDispatcher`.
///
/// Sequence (per `ds4.c:8684 metal_graph_encode_decode_layer`):
/// 1. `hc_collapse_norm` (Block A3 / G3 fused)
/// 2. `qkv_rms_norm_rows` (Block B3)
/// 3. `rope_tail(forward)` (Block C3 — on KV-raw rope-tail slice)
/// 4. `kv_fp8_store` (Block D1 — append normed+rotated KV row)
/// 5. `rope_tail(forward)` (Block C4 — on Q heads, rope-tail per head)
/// 6. `flash_attn_decode` (Block E1)
/// 7. `rope_tail(backward)` (Block E2 — undo on output heads)
/// 8. `attn_output_proj` (Block F1 — W_o + HC expand)
/// 9. `shared_expert` (Block J1)
/// 10. `shared_down_hc_expand_add`(Block K1)
///
/// This is now a thin wrapper over `decode_attn_prefix_with` +
/// `decode_attn_suffix_with`; the composed step driver uses the split
/// version so the MoE half can run between them.
pub fn decode_attn_layer_with<A: AttentionDispatcher>(
 dispatch: &A,
 layer: &AttnLayerWeights,
 mut step: AttnStepInputs<'_>,
) -> AttnLayerOut {
 let pos = step.pos;
 let prefix = decode_attn_prefix_with(dispatch, layer, &mut step);
 let after_ffn_hc = decode_attn_suffix_with(dispatch, layer, &prefix, step.routed_out, pos);
 AttnLayerOut {
 after_ffn_hc,
 normed: prefix.normed,
 }
}

#[cfg(test)]

mod tests {
 use super::*;
 use std::sync::Mutex;

 /// The ONE crate-wide env lock (see `crate::ENV_LOCK`).
 ///
 /// This was a second, file-local mutex. Two mutexes guarding one piece of process-global
 /// state is the same as no mutex: this file's tests excluded each other while a test in
 /// another module set the same variable freely -- which is exactly how the SiLU gate test
 /// in forward.rs failed intermittently. Aliased rather than deleted so all 14 existing
 /// call sites keep working and cannot drift back apart.
 use crate::ENV_LOCK;

 fn dummy_params() -> LayerParams {
 LayerParams {
 layer_idx: 0,
 d_embd: 4,
 n_hc: 2,
 n_head: 2,
 head_dim: 4,
 n_rot: 2,
 n_lora_q: 4,
 n_lora_kv: 4,
 hc_sinkhorn_iter: 1,
 hc_eps: 1e-6,
 rms_eps: 1e-6,
 rope_orig_ctx: 4096,
 rope_freq_base: 10000.0,
 rope_freq_scale: 1.0,
 rope_ext_factor: 0.0,
 rope_attn_factor: 1.0,
 compress_ratio: 1,
 n_out_group: 2,
 }
 }

 #[test]
 fn layer_params_helpers() {
 let p = dummy_params();
 assert!(!p.is_compressed());
 assert!(!p.uses_indexer());
 assert_eq!(p.q_dim(), 8);
 assert_eq!(p.hc_dim(), 8);
 }

 /// antirez `layer_shared_ffn_one` (ds4.c:4880-4912) calls
 /// `matvec_q8_0(out, ffn_down_shexp, mid)` which `quantize_q8_0_activation`s
 /// `mid` BEFORE the int8 dot (ds4.c:3337). Our `shared_down_hc_expand_add`
 /// previously dotted f32-direct against `shared_mid`, retaining precision
 /// antirez throws away. The fix round-trips `shared_mid` through q8_0
 /// (gated by DS4_Q8_0_ACT=1) before the down matvec. This test exercises
 /// an adversarial 64-element mid where the round-trip changes at least
 /// one element's bit pattern, then asserts that the gated path differs
 /// from the un-gated path — proving the gate is functional and the
 /// implementation actually performs the round-trip.
 #[test]
 fn shared_expert_down_q8_0_round_trip_changes_output_when_gate_on() {
 // 64-element mid: 2 q8_0 blocks of 32 each. Adversarial values that
 // straddle half-quanta so lrintf rounding effectively differs from
 // a no-op identity.
 let mut shared_mid: Vec<f32> = (0..64)
 .map(|i| {
 // block 0: amax≈10.0 with half-quanta at ~0.039
 // block 1: amax≈100.0 with half-quanta at ~0.394
 if i < 32 {
 let frac = (i as f32 - 16.0) * 0.625; // -10.0..+9.375
 frac + 0.0193 // sit between two integer quanta
 } else {
 let frac = (i as f32 - 48.0) * 6.25; // -100.0..+93.75
 frac - 0.1969
 }
 })
 .collect();
 shared_mid[0] = 10.0; // pin block-0 amax
 shared_mid[32] = 100.0; // pin block-1 amax

 // Discriminator: confirm q8_0 round-trip actually perturbs mid.
 let mid_q80 = crate::forward::q8_0_round_trip(&shared_mid);
 let any_diff = shared_mid
 .iter()
 .zip(mid_q80.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_diff,
 "q8_0 round-trip identity on adversarial mid — test is trivial"
 );

 // d_embd=8, sd=64. Hand-roll w_down with bit patterns that propagate
 // any change in `mid` to a different `shared_out` value.
 let d_embd = 8usize;
 let sd = 64usize;
 let mut p = dummy_params();
 p.d_embd = d_embd as u32;
 p.n_hc = 2;
 let n_hc = p.n_hc as usize;
 let mut seed = 0xDEADBEEF1234CAFEu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 let w_down: Vec<f32> = (0..d_embd * sd).map(|_| lcg(&mut seed)).collect();
 let routed_out = vec![0.0f32; d_embd];
 let after_attn_hc = vec![0.0f32; n_hc * d_embd];
 let hc_split_post = vec![1.0f32; n_hc];
 let hc_split_comb = vec![0.0f32; n_hc * n_hc];

 let d = CpuAttentionDispatcher;
 // Snapshot env, mutate carefully (under shared ENV_LOCK).
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let prev = std::env::var("DS4_Q8_0_ACT").ok();

 std::env::remove_var("DS4_Q8_0_ACT");
 let ungated = d.shared_down_hc_expand_add(
 &p,
 &shared_mid,
 &w_down,
 &routed_out,
 &after_attn_hc,
 &hc_split_post,
 &hc_split_comb,
 );

 std::env::set_var("DS4_Q8_0_ACT", "1");
 let gated = d.shared_down_hc_expand_add(
 &p,
 &shared_mid,
 &w_down,
 &routed_out,
 &after_attn_hc,
 &hc_split_post,
 &hc_split_comb,
 );

 // Restore env.
 match prev {
 Some(v) => std::env::set_var("DS4_Q8_0_ACT", v),
 None => std::env::remove_var("DS4_Q8_0_ACT"),
 }

 // The two paths MUST differ at bit level — otherwise the gate is
 // a no-op and we have not actually applied the antirez-fidelity fix.
 let any_bit_diff = ungated
 .iter()
 .zip(gated.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_bit_diff,
 "DS4_Q8_0_ACT gate did not change shared_down output — round-trip is a no-op"
 );

 // Sanity: gated output must match a hand-stamped antirez-spec
 // reference computed by manually round-tripping mid through q8_0.
 let want_shared_out: Vec<f32> = (0..d_embd)
 .map(|e| {
 let row = &w_down[e * sd..(e + 1) * sd];
 row.iter()
 .zip(mid_q80.iter())
 .map(|(a, b)| a * b)
 .sum::<f32>()
 })
 .collect();
 // hc_split_post = 1.0, hc_split_comb = 0 → after[dst, e] = shared_out[e]
 // (since routed_out=0, after_attn_hc=0). So `gated[dst*d_embd+e]` must
 // equal `want_shared_out[e]` for each dst.
 for dst in 0..n_hc {
 for e in 0..d_embd {
 let g = gated[dst * d_embd + e];
 let w = want_shared_out[e];
 assert_eq!(
 g.to_bits(),
 w.to_bits(),
 "dst={dst} e={e}: gated={g} antirez-ref={w}"
 );
 }
 }
 }

 /// antirez `hc_post_one` (ds4.c:4813) calls
 /// `matvec_q8_0(out, attn_output_b, low)` which `quantize_q8_0_activation`s
 /// `low` (the grouped-matvec output) BEFORE the int8 dot (ds4.c:3337).
 /// Our `attn_output_proj` stage-2 previously dotted `attn_low` f32-direct
 /// against `w_o_b`, retaining precision antirez throws away. The fix
 /// round-trips `attn_low` through q8_0 (gated by DS4_Q8_0_ACT=1) before
 /// stage-2. This test hand-stamps inputs where the round-trip changes
 /// at least one element's bit pattern, then asserts gated != ungated
 /// at bit level (proving the gate fires) AND that gated matches a
 /// hand-stamped antirez-spec reference.
 #[test]
 fn attn_output_proj_q8_0_round_trip_on_attn_low_changes_output_when_gate_on() {
 // Build inputs sized so attn_low has 64 elements (2 q8_0 blocks of 32).
 // dummy_params() has p.n_out_group small; we need attn_low len = n_groups * n_lora_o.
 // Use n_groups = 2, n_lora_o = 32 so out_low_dim = 64.
 let mut p = dummy_params();
 let d_embd = 8usize;
 let n_hc = 2u32;
 let n_groups = 2u32;
 let group_dim = 4u32;
 p.d_embd = d_embd as u32;
 p.n_hc = n_hc;
 p.n_out_group = n_groups;
 // q_dim must be n_groups * group_dim; set heads len accordingly below.
 // head_dim & n_head: choose so that p.q_dim() == n_groups*group_dim = 8.
 p.n_head = 4;
 p.head_dim = 2; // q_dim = n_head * head_dim = 8

 let q_dim = (n_groups * group_dim) as usize; // 8
 let n_lora_o = 32usize;
 let out_low_dim = n_groups as usize * n_lora_o; // 64
 let d = CpuAttentionDispatcher;

 let mut seed = 0xDECAFBADBADC0FFEu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };

 // Hand-craft heads + w_o_a so attn_low has known adversarial values
 // that straddle half-quanta. Easiest: heads = canonical basis-ish,
 // then attn_low = sum of row entries; just let LCG generate dense
 // inputs and verify post-hoc that q8_0 round-trip changes attn_low.
 let heads: Vec<f32> = (0..q_dim).map(|_| lcg(&mut seed)).collect();
 let w_o_a: Vec<f32> = (0..n_groups as usize * n_lora_o * group_dim as usize)
 .map(|_| lcg(&mut seed))
 .collect();
 let w_o_b: Vec<f32> = (0..d_embd * out_low_dim).map(|_| lcg(&mut seed)).collect();
 let cur_hc = vec![0.0f32; n_hc as usize * d_embd];
 let hc_split_post = vec![1.0f32; n_hc as usize];
 let hc_split_comb = vec![0.0f32; n_hc as usize * n_hc as usize];

 // Pre-compute attn_low ourselves (using identical math to the impl,
 // including the DS4_Q8_0_ACT round-trip on `heads` when gate is on)
 // and verify q8_0 round-trip on attn_low actually perturbs it.
 let compute_attn_low = |heads_in: &[f32]| -> Vec<f32> {
 let mut al = vec![0.0f32; out_low_dim];
 for g in 0..n_groups as usize {
 let heads_g = &heads_in[g * group_dim as usize..(g + 1) * group_dim as usize];
 for l in 0..n_lora_o {
 let row = &w_o_a[(g * n_lora_o + l) * group_dim as usize
 ..(g * n_lora_o + l + 1) * group_dim as usize];
 al[g * n_lora_o + l] = row.iter().zip(heads_g.iter()).map(|(a, b)| a * b).sum();
 }
 }
 al
 };
 let heads_q80 = crate::forward::q8_0_round_trip(&heads);
 let attn_low_gated_input = compute_attn_low(&heads_q80);
 let attn_low_q80 = crate::forward::q8_0_round_trip(&attn_low_gated_input);
 let any_diff = attn_low_gated_input
 .iter()
 .zip(attn_low_q80.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_diff,
 "q8_0 round-trip on attn_low produced identity — test is trivial"
 );

 // Snapshot env, mutate carefully (under shared ENV_LOCK).
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let prev = std::env::var("DS4_Q8_0_ACT").ok();

 std::env::remove_var("DS4_Q8_0_ACT");
 let ungated = d.attn_output_proj(
 &p,
 &heads,
 &w_o_a,
 &w_o_b,
 &cur_hc,
 &hc_split_post,
 &hc_split_comb,
 );

 std::env::set_var("DS4_Q8_0_ACT", "1");
 let gated = d.attn_output_proj(
 &p,
 &heads,
 &w_o_a,
 &w_o_b,
 &cur_hc,
 &hc_split_post,
 &hc_split_comb,
 );

 // Restore env.
 match prev {
 Some(v) => std::env::set_var("DS4_Q8_0_ACT", v),
 None => std::env::remove_var("DS4_Q8_0_ACT"),
 }

 // Gate must fire — outputs MUST differ.
 let any_bit_diff = ungated
 .iter()
 .zip(gated.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_bit_diff,
 "DS4_Q8_0_ACT gate did not change attn_output_proj output — round-trip is a no-op"
 );

 // Hand-stamp antirez-spec reference:
 // 1. q8_0 round-trip heads (already done in heads_q80)
 // 2. compute attn_low from q80 heads (already done in attn_low_gated_input)
 // 3. q8_0 round-trip attn_low (already done in attn_low_q80)
 // 4. stage-2 dot w_o_b · attn_low_q80
 // 5. HC expand with hc_split_post=1, hc_split_comb=0, routed_out=0
 // → after[dst*d_embd + e] = attn_out_ref[e] + cur_hc[dst*d_embd+e]
 // = attn_out_ref[e] (cur_hc is 0)
 let attn_out_ref: Vec<f32> = (0..d_embd)
 .map(|e| {
 let row = &w_o_b[e * out_low_dim..(e + 1) * out_low_dim];
 row.iter()
 .zip(attn_low_q80.iter())
 .map(|(a, b)| a * b)
 .sum::<f32>()
 })
 .collect();
 for dst in 0..n_hc as usize {
 for e in 0..d_embd {
 let g = gated[dst * d_embd + e];
 let w = attn_out_ref[e];
 assert_eq!(
 g.to_bits(),
 w.to_bits(),
 "dst={dst} e={e}: gated={g} antirez-ref={w}"
 );
 }
 }
 }

 #[test]
 fn qkv_rms_norm_full_row_is_normalised() {
 // Antirez `ds4.c:4494`: `rms_norm_weight(kv, raw, kv_norm, DS4_N_HEAD_DIM, ...)`
 // — the whole kv row (including the rope-tail slots) is rms-normed
 // before the rope rotation overwrites the tail. Our oracle must
 // match: every output should be (input/rms)·gamma.
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let qr = vec![1.0, 2.0, 3.0, 4.0];
 let kv = vec![5.0, 6.0, 7.0, 8.0]; // n_lora_kv=4 (no separate rope extension)
 let gq = vec![1.0; 4];
 let gkv = vec![1.0; 4];
 let (_qrn, kvn) = d.qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gkv);
 // Hand-computed normalization: sum_sq = 25+36+49+64 = 174; rms = sqrt(174/4) = 6.59545…
 let rms = (174.0f32 / 4.0).sqrt();
 for (i, &v) in kvn.iter().enumerate() {
 let expected = (i as f32 + 5.0) / rms;
 assert!(
 (v - expected).abs() < 1e-5,
 "kvn[{i}] = {v}, expected {expected}"
 );
 }
 }

 #[test]
 fn rope_tail_two_pair_is_pure_rotation() {
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let mut x = vec![1.0, 0.0]; // single adjacent (i0=0, i0+1=1) pair
 d.rope_tail(&p, &mut x, 0, false);
 // pos=0 ⇒ θ=0 ⇒ rotation is identity.
 assert!((x[0] - 1.0).abs() < 1e-6);
 assert!(x[1].abs() < 1e-6);
 }

 #[test]
 fn rope_tail_adjacent_pair_rotation_at_nonzero_pos() {
 // Adjacent-pair convention (j0, j0+1). With n_rot=2, base=10000,
 // scale=1: freq = 1/base^(0/2) = 1.0; θ = pos·1.0 = 5.0.
 let mut p = dummy_params();
 p.n_rot = 2;
 let d = CpuAttentionDispatcher;
 let mut x = vec![1.0f32, 0.0f32];
 d.rope_tail(&p, &mut x, 5, false);
 let theta = 5.0f32;
 let (s, c) = (theta.sin(), theta.cos());
 // (x0, x1) = (1,0) ⇒ (cos θ, sin θ).
 assert!((x[0] - c).abs() < 1e-6, "x0={}, expected {}", x[0], c);
 assert!((x[1] - s).abs() < 1e-6, "x1={}, expected {}", x[1], s);
 }

 #[test]
 fn rope_tail_yarn_attn_factor_can_cancel_mscale() {
 let mut p = dummy_params();
 p.n_rot = 4;
 p.rope_freq_scale = 0.25;
 p.rope_ext_factor = 1.0;
 let yarn_scale = 1.0 + 0.1 * (1.0f32 / p.rope_freq_scale).ln();
 p.rope_attn_factor = 1.0 / yarn_scale;

 let d = CpuAttentionDispatcher;
 let mut x = vec![1.0f32, 2.0, -3.0, 4.0];
 let before_norm = x.iter().map(|v| v * v).sum::<f32>();
 d.rope_tail(&p, &mut x, 37, false);
 let after_norm = x.iter().map(|v| v * v).sum::<f32>();

 assert!(
 (after_norm - before_norm).abs() < 1e-4,
 "before={before_norm}, after={after_norm}"
 );
 }

 /// assert `precompute_rope_tail_table` + `apply_rope_tail_with_table`
 /// is BIT-IDENTICAL to the trait `rope_tail` across n_heads × n_rot input
 /// shapes, both directions (forward + backward), with YaRN scaling enabled
 /// — the precompute path is the optimization replacing per-head trig with
 /// a once-per-token table, and any drift here will break the trace-equality
 /// oracle that gates acceptance.
 #[test]
 fn precompute_rope_tail_table_bit_identical_to_trait() {
 let mut p = dummy_params();
 p.n_rot = 8;
 p.rope_freq_base = 10000.0;
 p.rope_freq_scale = 0.25;
 p.rope_ext_factor = 1.0;
 p.rope_attn_factor = 1.13;
 p.rope_orig_ctx = 4096;

 let d = CpuAttentionDispatcher;
 let n_heads = 5usize;
 let n_rot = p.n_rot as usize;
 // Distinct per-head values so a table mismatch would be visible.
 let base: Vec<f32> = (0..(n_heads * n_rot))
 .map(|i| ((i as f32) * 0.13 - 0.4).sin() * 1.7)
 .collect();

 for &backward in &[false, true] {
 for &pos in &[0u32, 1, 7, 37, 1023] {
 // Reference: trait rope_tail on per-head tails.
 let mut ref_buf = base.clone();
 for head in ref_buf.chunks_mut(n_rot) {
 d.rope_tail(&p, head, pos, backward);
 }

 // Optimized: precompute table once, apply to whole buffer.
 let mut opt_buf = base.clone();
 let table = precompute_rope_tail_table(
 n_rot,
 pos,
 backward,
 p.rope_freq_base,
 p.rope_freq_scale,
 p.rope_ext_factor,
 p.rope_attn_factor,
 p.rope_orig_ctx,
 );
 apply_rope_tail_with_table(&mut opt_buf, &table, n_rot);

 for (i, (a, b)) in ref_buf.iter().zip(opt_buf.iter()).enumerate() {
 assert_eq!(
 a.to_bits(),
 b.to_bits(),
 "precompute mismatch pos={pos} backward={backward} idx={i}: {a} vs {b}"
 );
 }
 }
 }
 }

 /// `indexer_allowed_decode_one` aggressive top-k discriminator.
 ///
 /// Build a fixture where each comp row has a hand-chosen, distinct score
 /// rank, then assert that the returned `allowed[]` bitmap contains EXACTLY
 /// the top-k indices by score — not the bottom-k, not a permutation, and
 /// not all-true. This rules out the common bug classes:
 /// - selecting min instead of max (wrong sign on argmax)
 /// - selecting the same index k times (forgetting `!allowed[c]`)
 /// - returning all-true (top_k clamp bug)
 /// - off-by-one on n_comp (writing past the end)
 ///
 /// The fixture uses small n_head=1 and head_dim=2 so the weights/q/kv
 /// dimensions stay hand-computable. We zero the rope tail (n_rot=0) so
 /// it's a no-op — rope itself is exercised by the rope_tail_* tests.
 #[test]
 fn indexer_allowed_decode_one_picks_top_k_by_score() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 // Params: indexer ignores most p.* fields; we set n_rot=0 to disable
 // rope (tested separately), and pick n_head/head_dim small for the
 // fixture.
 let mut p = dummy_params();
 p.n_rot = 0;
 let n_head = 1u32;
 let head_dim = 2u32;
 let n_comp = 6u32;
 let top_k = 3u32;

 // qr_norm = [1, 0]; w_q_b layout is col-major over j: row[j, :] of length n_lora_q.
 // So q[j] = w_q_b[j*n_lora_q .. (j+1)*n_lora_q] · qr_norm = w_q_b[j*2+0]*1 + w_q_b[j*2+1]*0
 // = w_q_b[j*2].
 // With q_dim = n_head*head_dim = 2 → q = [w_q_b[0], w_q_b[2]] = [1.0, 0.0].
 let w_q_b: Vec<f32> = vec![
 1.0, 0.0, // j=0: q[0] = 1.0
 0.0, 0.0, // j=1: q[1] = 0.0
 ];
 let qr_norm = vec![1.0f32, 0.0];

 // cur = [1, 0]; w_proj layout is row-major over h: row[h, :] of length d_embd.
 // weights[0] = w_proj[0..2] · cur = w_proj[0]*1 + w_proj[1]*0 = w_proj[0] = 1.0
 // After scale = 1/sqrt(head_dim*n_head) = 1/sqrt(2):
 // weights[0] = 1/sqrt(2) ≈ 0.7071.
 let w_proj: Vec<f32> = vec![1.0, 0.0];
 let cur = vec![1.0f32, 0.0];

 // index_comp_kv: n_comp rows of head_dim each. With q=[1,0], dot(kv_c, q) = kv_c[0].
 // Score per row (single head, weight ≈ 0.7071) = max(0, kv_c[0]) * weights[0].
 // Pick distinct kv_c[0] values: rank by descending score.
 // c=0 → kv[0]=10.0 → score = 10*0.7071 ≈ 7.07 (rank 1)
 // c=1 → kv[0]= 1.0 → score = 1*0.7071 ≈ 0.71 (rank 4)
 // c=2 → kv[0]= 7.0 → score = 7*0.7071 ≈ 4.95 (rank 2)
 // c=3 → kv[0]=-3.0 → score = 0 (clamped to 0) (rank 6, tied with c=5)
 // c=4 → kv[0]= 4.0 → score = 4*0.7071 ≈ 2.83 (rank 3)
 // c=5 → kv[0]= 0.5 → score = 0.5*0.7071≈ 0.35 (rank 5)
 // Top-3 by score = {0, 2, 4}.
 let index_comp_kv: Vec<f32> = vec![
 10.0, 0.0, // c=0
 1.0, 0.0, // c=1
 7.0, 0.0, // c=2
 -3.0, 0.0, // c=3 — negative → ReLU clamps score to 0
 4.0, 0.0, // c=4
 0.5, 0.0, // c=5
 ];

 let inputs = IndexerInputs {
 w_q_b: &w_q_b,
 w_proj: &w_proj,
 index_comp_kv: &index_comp_kv,
 n_comp,
 n_head,
 head_dim,
 top_k,
 };
 let d = CpuAttentionDispatcher;
 let (allowed, n_sel) = indexer_allowed_decode_one(&d, &p, &inputs, &cur, &qr_norm, 0);

 assert_eq!(n_sel, 3, "n_selected must equal top_k");
 assert_eq!(allowed.len(), n_comp as usize);
 // Top-3 = {0, 2, 4}. All others must be unselected.
 let want_selected = [true, false, true, false, true, false];
 assert_eq!(
 allowed, want_selected,
 "top-k selection wrong: got {:?}, want {:?}",
 allowed, want_selected
 );
 // Aggressive cross-checks against common bugs:
 // - bottom-k bug would select {3, 5, 1} (lowest scores).
 let bottom_k = [false, true, false, true, false, true];
 assert_ne!(
 allowed, bottom_k,
 "indexer picked bottom-k instead of top-k"
 );
 // - all-true bug (e.g., top_k clamp missing).
 assert!(allowed.iter().any(|&a| !a), "indexer returned all-true");
 // - missing ReLU on negative dot: if dot<0 was kept, c=3 score=-2.12
 // would still be < c=5 score=0.35, BUT if the impl took
 // abs(dot)*w instead of max(0,dot)*w, then c=3 score = 2.12
 // would beat c=4 (2.83)? No — 2.12 < 2.83, still rank 4. But it
 // would beat c=5 (0.35) and c=1 (0.71). Top-3 would still be
 // {0, 2, 4}. So this test does NOT discriminate abs() vs max();
 // we'd need a fixture where |neg_dot| exceeds a positive dot in
 // the top-k band. Add a second test for that.
 }

 /// Companion to the previous test: forces a fixture where a negative dot
 /// would steal a top-k slot if the impl took `abs()` instead of `max(0, ·)`.
 /// Picks scores so c=3 has the LARGEST |dot| but a negative sign, and
 /// only the correct ReLU implementation excludes it from the top-3.
 #[test]
 fn indexer_allowed_decode_one_relu_clamps_negative_dot() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let mut p = dummy_params();
 p.n_rot = 0;
 let n_head = 1u32;
 let head_dim = 2u32;
 let n_comp = 5u32;
 let top_k = 3u32;

 // Same q construction: qr_norm=[1,0] → q=[1.0, 0.0]. n_lora_q=2, q_dim=2.
 let w_q_b: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
 let qr_norm = vec![1.0f32, 0.0];

 // cur=[1,0] → weights[0] = w_proj[0] * (1/sqrt(2)) > 0. d_embd=2.
 let w_proj: Vec<f32> = vec![1.0, 0.0];
 let cur = vec![1.0f32, 0.0];

 // kv[0] values:
 // c=0 → +3.0 (rank 2 by max(0,·): 3.0; rank 2 by abs: 3.0)
 // c=1 → +2.0 (rank 3 by max(0,·): 2.0; rank 4 by abs: 2.0)
 // c=2 → +1.0 (rank 4 by max(0,·): 1.0; rank 5 by abs: 1.0)
 // c=3 → -8.0 (rank 5 by max(0,·): 0; rank 1 by abs: 8.0) ← KEY
 // c=4 → +4.0 (rank 1 by max(0,·): 4.0; rank 1 by abs: 4.0 — but c=3 wins under abs)
 // Top-3 under correct max(0,·) ReLU: {4, 0, 1} (scores 4, 3, 2).
 // Top-3 under buggy abs(): {3, 4, 0} (scores 8, 4, 3).
 let index_comp_kv: Vec<f32> = vec![
 3.0, 0.0, // c=0
 2.0, 0.0, // c=1
 1.0, 0.0, // c=2
 -8.0, 0.0, // c=3 — large magnitude negative
 4.0, 0.0, // c=4
 ];

 let inputs = IndexerInputs {
 w_q_b: &w_q_b,
 w_proj: &w_proj,
 index_comp_kv: &index_comp_kv,
 n_comp,
 n_head,
 head_dim,
 top_k,
 };
 let d = CpuAttentionDispatcher;
 let (allowed, _) = indexer_allowed_decode_one(&d, &p, &inputs, &cur, &qr_norm, 0);

 let want_selected = [true, true, false, false, true];
 assert_eq!(
 allowed, want_selected,
 "ReLU on dot must clamp negatives to 0; got {:?}, want {:?}",
 allowed, want_selected
 );
 // Buggy-abs() selection would be {0, 3, 4} — explicitly assert NOT that.
 let buggy_abs = [true, false, false, true, true];
 assert_ne!(
 allowed, buggy_abs,
 "indexer used abs(dot) instead of max(0, dot) — would have picked the large negative"
 );
 }

 /// Early-return path: when `top_k >= n_comp`, ALL rows must be selected
 /// (returns all-true bitmap). Guards against off-by-one in the clamp.
 #[test]
 fn indexer_allowed_decode_one_returns_all_when_top_k_ge_n_comp() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let mut p = dummy_params();
 p.n_rot = 0;
 let n_comp = 4u32;
 let top_k = 4u32; // == n_comp triggers early return
 let inputs = IndexerInputs {
 w_q_b: &[0.0; 2 * 1 * 2], // n_lora_q * n_head * head_dim
 w_proj: &[0.0; 2 * 1], // d_embd * n_head
 index_comp_kv: &[0.0; 4 * 2], // n_comp * head_dim
 n_comp,
 n_head: 1,
 head_dim: 2,
 top_k,
 };
 let d = CpuAttentionDispatcher;
 let (allowed, n_sel) = indexer_allowed_decode_one(
 &d,
 &p,
 &inputs,
 &[0.0f32; 2], // cur
 &[0.0f32; 2], // qr_norm
 0,
 );
 assert_eq!(n_sel, n_comp);
 assert!(allowed.iter().all(|&a| a), "all comp rows must be selected");
 }

 /// Edge-case: `n_comp == 0` must return empty bitmap and n_selected=0.
 #[test]
 fn indexer_allowed_decode_one_handles_empty_n_comp() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let mut p = dummy_params();
 p.n_rot = 0;
 let inputs = IndexerInputs {
 w_q_b: &[0.0; 2 * 1 * 2],
 w_proj: &[0.0; 2 * 1],
 index_comp_kv: &[],
 n_comp: 0,
 n_head: 1,
 head_dim: 2,
 top_k: 5,
 };
 let d = CpuAttentionDispatcher;
 let (allowed, n_sel) =
 indexer_allowed_decode_one(&d, &p, &inputs, &[0.0f32; 2], &[0.0f32; 2], 0);
 assert_eq!(n_sel, 0);
 assert!(allowed.is_empty());
 }

 // failing-first: verbatim antirez `rope_tail_ext_inplace` (ds4.c:4545-4593)
 // exercised at a realistic compressed-layer config — freq_base=160000, scale=1/16,
 // ext_factor=1.0, attn_factor≈0.78 — across n_head=2, n_rot=64, pos=27.
 // If our port diverges at any pair index, this test pinpoints the bug.
 fn antirez_rope_tail_ext_inplace(
 x: &mut [f32],
 n_head: usize,
 head_dim: usize,
 n_rot: usize,
 pos: u32,
 n_ctx_orig: u64,
 freq_base: f32,
 freq_scale: f32,
 ext_factor: f32,
 attn_factor: f32,
 beta_fast: f32,
 beta_slow: f32,
 inverse: bool,
 ) {
 let n_nope = head_dim - n_rot;
 let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
 let sin_sign = if inverse { -1.0f32 } else { 1.0 };
 let mut corr_dims = [0.0f32, 0.0];
 if ext_factor != 0.0 {
 let corr_dim = |b: f32| -> f32 {
 (n_rot as f32) * ((n_ctx_orig as f32) / (b * 2.0 * std::f32::consts::PI)).ln()
 / (2.0 * freq_base.ln())
 };
 let start = corr_dim(beta_fast).floor();
 let end = corr_dim(beta_slow).ceil();
 corr_dims[0] = start.max(0.0);
 corr_dims[1] = end.min((n_rot as f32) - 1.0);
 }
 for h in 0..n_head {
 let tail_off = h * head_dim + n_nope;
 let mut theta_extrap = pos as f32;
 let mut i = 0usize;
 while i < n_rot {
 let theta_interp = freq_scale * theta_extrap;
 let mut theta = theta_interp;
 let mut mscale = attn_factor;
 if ext_factor != 0.0 {
 // Antirez uses (i0 / 2) INTEGER DIVISION, but i is always even.
 let y =
 ((i / 2) as f32 - corr_dims[0]) / 0.001f32.max(corr_dims[1] - corr_dims[0]);
 let ramp = 1.0 - y.clamp(0.0, 1.0);
 let ramp_mix = ramp * ext_factor;
 theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
 mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
 }
 let c = theta.cos() * mscale;
 let s = sin_sign * theta.sin() * mscale;
 let x0 = x[tail_off + i];
 let x1 = x[tail_off + i + 1];
 x[tail_off + i] = x0 * c - x1 * s;
 x[tail_off + i + 1] = x0 * s + x1 * c;
 theta_extrap *= theta_scale;
 i += 2;
 }
 }
 }

 // failing-first: verbatim antirez `rms_norm_weight` (ds4.c:2560-2566)
 // ported as reference. Our `rms_norm_inplace` MUST match bit-identically (after
 // the f64-accumulator fix) at d_embd=4096 — anything else means a regression.
 fn antirez_rms_norm_weight(x: &mut [f32], weight: &[f32], eps: f32) {
 let n = x.len();
 let mut ss: f64 = 0.0;
 for &v in x.iter() {
 ss += v as f64 * v as f64;
 }
 let scale: f32 = 1.0 / ((ss / n as f64) as f32 + eps).sqrt();
 for (xi, &gi) in x.iter_mut().zip(weight.iter()) {
 *xi = *xi * scale * gi;
 }
 }

 fn antirez_rms_norm_no_weight(x: &mut [f32], eps: f32) {
 let n = x.len();
 let mut ss: f64 = 0.0;
 for &v in x.iter() {
 ss += v as f64 * v as f64;
 }
 let scale: f32 = 1.0 / ((ss / n as f64) as f32 + eps).sqrt();
 for xi in x.iter_mut() {
 *xi *= scale;
 }
 }

 #[test]
 fn rms_norm_inplace_matches_antirez_at_d_embd_4096() {
 // Discriminating test: at d_embd=4096 with mixed-magnitude values, the f32
 // accumulator we previously used loses ~10 bits vs antirez's f64 accumulator.
 // After fix, both must match bit-identically.
 let mut seed = 0xC0FFEE_C0FFEE_C0u64;
 let lcg = |s: &mut u64| {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (*s >> 32) as u32
 };
 // d_embd=4096, mixed magnitudes from -3.0 to +3.0
 let x_orig: Vec<f32> = (0..4096)
 .map(|_| {
 let u = lcg(&mut seed) as f32 / u32::MAX as f32;
 (u - 0.5) * 6.0
 })
 .collect();
 let gamma: Vec<f32> = (0..4096)
 .map(|_| {
 let u = lcg(&mut seed) as f32 / u32::MAX as f32;
 0.5 + u
 })
 .collect();
 let eps = 1e-6f32;

 // With gamma
 let mut ours = x_orig.clone();
 let mut want = x_orig.clone();
 CpuAttentionDispatcher::rms_norm_inplace(eps, &mut ours, Some(&gamma));
 antirez_rms_norm_weight(&mut want, &gamma, eps);
 for (i, (&a, &b)) in ours.iter().zip(want.iter()).enumerate() {
 assert_eq!(
 a.to_bits(),
 b.to_bits(),
 "rms_norm_weight i={i}: ours={a} antirez={b} (diff={})",
 a - b
 );
 }

 // Without gamma
 let mut ours = x_orig.clone();
 let mut want = x_orig.clone();
 CpuAttentionDispatcher::rms_norm_inplace(eps, &mut ours, None);
 antirez_rms_norm_no_weight(&mut want, eps);
 for (i, (&a, &b)) in ours.iter().zip(want.iter()).enumerate() {
 assert_eq!(
 a.to_bits(),
 b.to_bits(),
 "rms_norm_no_weight i={i}: ours={a} antirez={b}"
 );
 }
 }

 #[test]
 fn rms_norm_inplace_f64_accumulator_distinguishes_from_f32_at_d_embd_4096() {
 // Adversarial input where f32-accumulator and f64-accumulator give
 // measurably different mean_sq. Many tiny + few large values force
 // f32 catastrophic cancellation.
 let mut x: Vec<f32> = (0..4096).map(|i| 1.0e-4 * (i as f32 + 1.0)).collect();
 // Add a small number of large values to spread magnitudes.
 x[0] = 100.0;
 x[2000] = -50.0;
 x[4095] = 75.0;

 // f32 accumulator (the wrong way)
 let n = x.len() as f32;
 let mean_sq_f32: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
 let scale_f32 = 1.0f32 / (mean_sq_f32 + 1e-6).sqrt();

 // f64 accumulator (antirez)
 let mean_sq_f64: f64 = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / x.len() as f64;
 let scale_f64 = 1.0f32 / (mean_sq_f64 as f32 + 1e-6).sqrt();

 // The two scales must differ at the float ULP level — that's what
 // confirms d_embd=4096 was being affected by the accumulator choice.
 assert_ne!(
 scale_f32.to_bits(),
 scale_f64.to_bits(),
 "if f32 and f64 accumulators agree on adversarial input, this test \
 can't discriminate the f64-accumulator fix"
 );

 // After the fix, our implementation must match the f64 path.
 let mut got = x.clone();
 CpuAttentionDispatcher::rms_norm_inplace(1e-6, &mut got, None);
 let want: Vec<f32> = x.iter().map(|&v| v * scale_f64).collect();
 for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
 assert_eq!(
 a.to_bits(),
 b.to_bits(),
 "i={i}: ours={a} f64-ref={b} (rms_norm_inplace must use f64 acc)"
 );
 }
 }

 #[test]
 fn rope_tail_matches_antirez_at_compressed_layer_config() {
 // DS4 compressed layer: freq_base=160000, freq_scale=1/16, ext_factor=1.0.
 let mut p = dummy_params();
 p.n_rot = 64;
 p.head_dim = 64; // n_nope = 0 — exercise entire head.
 p.n_head = 2;
 p.rope_orig_ctx = 4096;
 p.rope_freq_base = 160000.0;
 p.rope_freq_scale = 1.0 / 16.0;
 p.rope_ext_factor = 1.0;
 // YaRN attn factor matching antirez `layer_rope_attn_factor` for compressed.
 p.rope_attn_factor = 1.0 / (1.0 + 0.1 * (16.0f32).ln());

 // Two heads × 64 rot dims, distinct deterministic values.
 let mut x: Vec<f32> = (0..(2 * 64))
 .map(|i| ((i as f32 * 0.137).sin()) * 1.3 + 0.5)
 .collect();
 let mut want = x.clone();

 let d = CpuAttentionDispatcher;
 d.rope_tail(&p, &mut x, 27, false);

 antirez_rope_tail_ext_inplace(
 &mut want,
 2,
 64,
 64,
 27,
 4096,
 160000.0,
 1.0 / 16.0,
 1.0,
 p.rope_attn_factor,
 32.0,
 1.0,
 false,
 );

 let mut max_abs = 0.0f32;
 let mut max_rel = 0.0f32;
 for (i, (&got, &exp)) in x.iter().zip(want.iter()).enumerate() {
 let abs = (got - exp).abs();
 let rel = abs / exp.abs().max(1e-6);
 if abs > max_abs {
 max_abs = abs;
 }
 if rel > max_rel {
 max_rel = rel;
 }
 assert!(
 abs < 1e-5 || rel < 1e-5,
 "i={i}: got={got}, antirez={exp}, abs={abs}, rel={rel}"
 );
 }
 assert!(
 max_abs < 1e-5,
 "rope_tail max_abs={max_abs} max_rel={max_rel}"
 );
 }

 #[test]
 fn rope_tail_inverse_undoes_forward_at_compressed_layer() {
 // Forward then inverse must return the original input (modulo precision).
 let mut p = dummy_params();
 p.n_rot = 64;
 p.head_dim = 64;
 p.n_head = 1;
 p.rope_orig_ctx = 4096;
 p.rope_freq_base = 160000.0;
 p.rope_freq_scale = 1.0 / 16.0;
 p.rope_ext_factor = 1.0;
 p.rope_attn_factor = 1.0 / (1.0 + 0.1 * (16.0f32).ln());

 let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.211).cos()).collect();
 let mut x = original.clone();
 let d = CpuAttentionDispatcher;
 d.rope_tail(&p, &mut x, 27, false);
 d.rope_tail(&p, &mut x, 27, true);
 for (i, (&got, &exp)) in x.iter().zip(original.iter()).enumerate() {
 // attn_factor is applied twice (once per direction) so they double-scale.
 // Antirez's forward applies mscale on both forward and inverse, so the
 // round-trip is multiplied by mscale^2 ≈ 1 if attn_factor ≈ 1/(1+0.1*ln(1/scale)).
 let m = p.rope_attn_factor;
 let m2 = (m * (1.0 + 0.1 * (1.0f32 / p.rope_freq_scale).ln())).powi(2);
 let expected = exp * m2;
 let abs = (got - expected).abs();
 assert!(
 abs < 1e-3,
 "i={i}: forward+inverse round-trip drifts: got={got}, expected={expected}, mscale^2={m2}"
 );
 }
 }

 #[test]
 fn f16_round_trip_matches_reference_values() {
 // Exact in f16
 assert_eq!(f16_round_trip_f32(0.0), 0.0);
 assert_eq!(f16_round_trip_f32(1.0), 1.0);
 assert_eq!(f16_round_trip_f32(-1.0), -1.0);
 assert_eq!(f16_round_trip_f32(0.5), 0.5);

 // 1.0 + 2^-10 = 1.0009765625 is exactly representable in f16
 let exact = 1.0_f32 + 1.0 / 1024.0;
 assert_eq!(f16_round_trip_f32(exact), exact);

 // 1.0 + 2^-11 is NOT representable; rounds to 1.0 (ties-to-even)
 let half_ulp = 1.0_f32 + 1.0 / 2048.0;
 assert_eq!(f16_round_trip_f32(half_ulp), 1.0);

 // A value with f32-only precision must round
 let x = 0.1_f32;
 let y = f16_round_trip_f32(x);
 assert_ne!(y.to_bits(), x.to_bits());
 // f16 precision ~= 2^-10 of magnitude; |diff| < 1e-3 for x near 0.1
 assert!((y - x).abs() < 1e-3);

 // Large in-range: 1000.0 → f16 has limited mantissa, but still close
 let y = f16_round_trip_f32(1000.0);
 assert!((y - 1000.0).abs() < 1.0);

 // Over-range: f16 max is 65504; 70000.0 → +Inf
 assert!(f16_round_trip_f32(70_000.0).is_infinite());

 // Subnormal underflow
 assert_eq!(f16_round_trip_f32(1e-10), 0.0);
 }

 #[test]
 fn kv_fp8_store_handles_realistic_row_no_nan() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 // Realistic post-rms-norm KV row with head_dim=512 (DS4 real shape).
 let mut p = dummy_params();
 p.head_dim = 512;
 p.n_rot = 64;
 p.n_lora_kv = 512;
 let d = CpuAttentionDispatcher;
 let row = p.n_lora_kv as usize;
 let cap = 2;
 let mut storage = vec![0.0f32; row * cap];
 let kv: Vec<f32> = (0..row).map(|i| ((i as f32) * 0.13).sin() * 0.5).collect();

 let mut view = KvCacheView {
 raw: &mut storage,
 raw_cap: cap as u32,
 pos: 0,
 };
 d.kv_fp8_store(&p, &kv, &mut view);

 let stored = &storage[..row];
 assert!(
 stored.iter().all(|v| v.is_finite()),
 "found non-finite KV cache entry"
 );
 // Result should be close to input (FP8 + f16 round-trip preserves
 // magnitude). With NOPE FP8 at 448 max + per-64 max scaling, expect
 // |err| < 0.1 for inputs in [-0.5, 0.5].
 let max_err = kv
 .iter()
 .zip(stored.iter())
 .map(|(a, b)| (a - b).abs())
 .fold(0.0f32, f32::max);
 assert!(max_err < 0.1, "max_err = {max_err}, expected < 0.1");
 }

 #[test]
 fn kv_fp8_store_advances_position() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let row = p.n_lora_kv as usize;
 let cap = 4;
 let mut storage = vec![0.0f32; row * cap];
 let kv = vec![1.0f32; row];
 {
 let mut view = KvCacheView {
 raw: &mut storage,
 raw_cap: cap as u32,
 pos: 0,
 };
 d.kv_fp8_store(&p, &kv, &mut view);
 assert_eq!(view.pos, 1);
 }
 for &v in &storage[..row] {
 assert_eq!(v, 1.0);
 }
 // Wrap-around honours raw_cap.
 {
 let mut view = KvCacheView {
 raw: &mut storage,
 raw_cap: cap as u32,
 pos: cap as u32,
 };
 d.kv_fp8_store(&p, &kv, &mut view);
 }
 for &v in &storage[..row] {
 assert_eq!(v, 1.0);
 }
 }

 #[test]
 fn fp8_kv_quantize_only_changes_nope_prefix() {
 let mut row = vec![0.1f32, -0.2, 0.3, -0.4, 7.25, -8.5, 9.75, -10.0];
 let tail_before = row[4..].to_vec();

 ds4_fp8_kv_quantize_row_inplace(&mut row, 8, 4);

 assert_ne!(row[0].to_bits(), 0.1f32.to_bits());
 assert_eq!(&row[4..], tail_before.as_slice());
 }

 #[test]
 fn fp8_kv_quantize_uses_independent_64_value_scales() {
 let mut row = vec![0.0f32; 128];
 row[0] = 0.1;
 row[63] = -0.05;
 row[64] = 100.0;
 row[127] = -50.0;

 ds4_fp8_kv_quantize_row_inplace(&mut row, 128, 0);

 assert!(
 row[0] > 0.09 && row[0] < 0.11,
 "small block should keep its own scale, got {}",
 row[0]
 );
 assert!(
 row[64] > 95.0 && row[64] < 105.0,
 "large block should keep its own scale, got {}",
 row[64]
 );
 assert!(row[63] < -0.045, "block-0 negative value got {}", row[63]);
 assert!(row[127] < -47.0, "block-1 negative value got {}", row[127]);
 }

 // failing-first: verbatim antirez `dsv4_fp8_kv_quantize_row_inplace_cpu`
 // (ds4.c:1486-1504) as ground truth; assert bit-exact match across adversarial
 // inputs that target each branch (amax<1e-4 clamp, log2-ceil scale rounding,
 // ±448 clip, NOPE-only operation).
 fn antirez_fp8_kv_quantize_row_inplace(x: &mut [f32], head_dim: usize, n_rot: usize) {
 fn e4m3_value(i: i32) -> f32 {
 const EXP_SCALE: [f32; 16] = [
 0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
 64.0, 128.0, 256.0,
 ];
 let exp = (i >> 3) & 0x0f;
 let mant = i & 0x07;
 if exp == 0 {
 (mant as f32) * 0.001_953_125
 } else {
 (1.0 + (mant as f32) * 0.125) * EXP_SCALE[exp as usize]
 }
 }
 fn dequant(x: f32) -> f32 {
 let sign = if x < 0.0 { -1.0 } else { 1.0 };
 let ax = x.abs().min(448.0);
 let mut lo: i32 = 0;
 let mut hi: i32 = 126;
 while lo < hi {
 let mid = (lo + hi + 1) >> 1;
 if e4m3_value(mid) <= ax {
 lo = mid;
 } else {
 hi = mid - 1;
 }
 }
 let mut best = lo;
 if best < 126 {
 let bd = (ax - e4m3_value(best)).abs();
 let nd = (ax - e4m3_value(best + 1)).abs();
 if nd < bd || (nd == bd && ((best + 1) & 1) == 0 && (best & 1) != 0) {
 best += 1;
 }
 }
 sign * e4m3_value(best)
 }
 let n_nope = head_dim - n_rot;
 let mut off = 0usize;
 while off + 64 <= n_nope {
 let mut amax = 0.0f32;
 for i in 0..64 {
 let av = x[off + i].abs();
 if av > amax {
 amax = av;
 }
 }
 if amax < 1.0e-4 {
 amax = 1.0e-4;
 }
 let scale = ((amax / 448.0).log2().ceil()).exp2();
 for i in 0..64 {
 let mut v = x[off + i] / scale;
 if v > 448.0 {
 v = 448.0;
 }
 if v < -448.0 {
 v = -448.0;
 }
 x[off + i] = dequant(v) * scale;
 }
 off += 64;
 }
 }

 #[test]
 fn fp8_kv_quantize_matches_verbatim_antirez_port() {
 // Adversarial fixture: 4 blocks × 64 = 256 NOPE entries + 64 rope tail.
 // Block 0: tiny (amax < 1e-4) — exercises clamp.
 // Block 1: large (amax = 500 > 448) — exercises clip.
 // Block 2: amax exactly at 448 — exercises log2-ceil boundary.
 // Block 3: amax = 449 (just over 448) — log2-ceil flips to scale=2.
 let mut x = vec![0.0f32; 320];
 for i in 0..64 {
 x[i] = (1e-5) * (i as f32 - 31.0);
 }
 for i in 64..128 {
 x[i] = if i % 2 == 0 { 500.0 } else { -123.456 };
 }
 for i in 128..192 {
 x[i] = if i == 130 {
 448.0
 } else {
 (i as f32 * 0.073).sin() * 200.0
 };
 }
 for i in 192..256 {
 x[i] = if i == 200 {
 449.0
 } else {
 -(i as f32 * 0.111).cos() * 150.0
 };
 }
 // Rope tail should stay byte-exact.
 for i in 256..320 {
 x[i] = (i as f32 * 0.5) - 100.0;
 }

 let mut ours = x.clone();
 let mut want = x.clone();
 ds4_fp8_kv_quantize_row_inplace(&mut ours, 320, 64);
 antirez_fp8_kv_quantize_row_inplace(&mut want, 320, 64);

 for (i, (&a, &b)) in ours.iter().zip(want.iter()).enumerate() {
 assert_eq!(
 a.to_bits(),
 b.to_bits(),
 "i={i}: bit-exact required, got {a} vs antirez {b}"
 );
 }
 // Sanity: rope tail untouched.
 for i in 256..320 {
 assert_eq!(ours[i], x[i]);
 }
 }

 #[test]
 fn fp8_kv_quantize_properties_over_deterministic_random_rows() {
 fn lcg(seed: &mut u64) -> u32 {
 *seed = seed
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (*seed >> 32) as u32
 }
 fn lcg_f32(seed: &mut u64, lo: f32, hi: f32) -> f32 {
 let u = lcg(seed) as f32 / u32::MAX as f32;
 lo + u * (hi - lo)
 }

 let mut seed = 0x4650_385f_5052_4f50u64;
 for trial in 0..64 {
 let n_rot = ((lcg(&mut seed) as usize % 4) + 1) * 8;
 let head_dim = 64 + n_rot + (lcg(&mut seed) as usize % 4) * 64;
 let n_nope = head_dim - n_rot;
 let mut row: Vec<f32> = (0..head_dim)
 .map(|_| lcg_f32(&mut seed, -250.0, 250.0))
 .collect();
 let before = row.clone();

 ds4_fp8_kv_quantize_row_inplace(&mut row, head_dim, n_rot);

 assert_eq!(
 &row[n_nope..],
 &before[n_nope..],
 "trial {trial}: rope tail changed"
 );
 assert!(
 row[..n_nope].iter().all(|v| v.is_finite()),
 "trial {trial}: NOPE prefix contains non-finite values"
 );

 for off in (0..n_nope).step_by(64) {
 let end = (off + 64).min(n_nope);
 let amax = before[off..end]
 .iter()
 .map(|v| v.abs())
 .fold(0.0f32, f32::max)
 .max(1.0e-4);
 let scale = (amax / 448.0).log2().ceil().exp2();
 for i in off..end {
 let v = (before[i] / scale).clamp(-448.0, 448.0);
 let expected = ds4_e4m3fn_round_trip(v) * scale;
 assert_eq!(
 row[i].to_bits(),
 expected.to_bits(),
 "trial {trial}, i={i}: before={}, after={}, expected={expected}",
 before[i],
 row[i]
 );
 }
 }
 }
 }

 #[test]
 fn flash_attn_decode_attends_to_only_row_with_unit_q() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 // n_head=2, head_dim=4, n_lora_kv=4 (no separate rope extension)
 let row = p.n_lora_kv as usize;
 let cap = 2u32;
 // Two KV rows, V part (first head_dim floats) distinct per row.
 let kv = vec![
 // row 0: row width = n_lora_kv = 4 (== head_dim under new convention)
 1.0, 0.0, 0.0, 0.0, // row 1
 0.0, 0.0, 0.0, 0.0,
 ];
 assert_eq!(kv.len(), row * cap as usize);
 // q: head 0 strongly biased to row 0 K direction, head 1 zero.
 let q = vec![10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
 let sinks = vec![-1e9, -1e9]; // disable sink contribution
 let out = d.flash_attn_decode(
 &p, &q, &kv, /*n_raw*/ 1, /*raw_cap*/ cap, /*raw_start*/ 0, None, 0,
 None, 0, &sinks,
 );
 // head 0 should mostly recover V row 0 = [1,0,0,0].
 assert!(out[0] > 0.9, "head0 out[0]={}", out[0]);
 // head 1 attended uniformly to one row → should be near V[0]=[1,0,0,0] too,
 // but with no preference; with single row attention, w≈1.
 assert!(out[4] > 0.9, "head1 out[0]={}", out[4]);
 }

 /// antirez `layer_attention_rows_one` (ds4.c:4773-4782) does:
 /// 1. `denom = expf(sinks - max);`
 /// 2. `for r: weight = expf(score[r] - max); denom += weight;
 /// axpy_f32(oh, kv_r, weight)` // un-normalized!
 /// 3. `scale_f32(oh, 1/denom)` // single normalize at the end
 ///
 /// Our `flash_attn_decode` previously normalized each weight first
 /// (`w = scores[j+1] * inv`) then accumulated `out_h[d] += w * src[d]`.
 /// Mathematically equivalent in real arithmetic; at f32 ULP, the
 /// magnitudes of the intermediate accumulator differ → different
 /// rounding cascade → different bit-level output.
 ///
 /// This test hand-stamps BOTH orderings (normalize-late vs
 /// normalize-early) with an adversarial input where they diverge, and
 /// asserts our impl matches the antirez (normalize-late) order
 /// bit-exactly.
 #[test]
 fn flash_attn_decode_matches_antirez_normalize_late_order() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 // Build n_head=2, head_dim=8, n_kv=32 with kv values straddling the
 // f32 precision boundary so that summing un-normalized weights vs
 // pre-normalized weights produces distinct ULP rounding cascades.
 let mut p = dummy_params();
 let n_head = 2u32;
 let head_dim = 8u32;
 p.n_head = n_head;
 p.head_dim = head_dim;
 p.n_lora_kv = head_dim;
 let row = head_dim as usize;
 let n_kv = 32u32;
 let cap = n_kv;

 // q designed so that dot(q_h, kv_r) sweeps a 30-unit range across r
 // (one row matches q strongly, others weakly). exp(s - max) then
 // spans many orders of magnitude → denom dominated by 1 row +
 // long tail of tiny non-zero weights. Accumulator order matters at
 // f32 ULP under such "1 + ε + ε + …" sums.
 //
 // Build q = [3.0, 0, 0, 0, 0, 0, 0, 0] for head 0 (head 1 mirror).
 // Then dot(q, kv_r) = 3.0 * kv_r[0]. By varying kv_r[0] across r
 // from -5 to +5, dot ranges from -15 to +15 → score = dot/sqrt(8).
 let mut q = vec![0.0f32; n_head as usize * row];
 q[0] = 3.0;
 q[row] = 3.0;
 let mut kv = vec![0.0f32; cap as usize * row];
 let mut seed = 0xCAFEBABEDEADBEEFu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 for r in 0..cap as usize {
 // first slot drives the score; r==0 is the strongly-attended row.
 kv[r * row] = if r == 0 {
 5.0
 } else {
 -2.0 + (r as f32) * 0.07
 };
 // remaining slots are "V" — give them diverse magnitudes so
 // weighted sums collide between orderings.
 for d in 1..row {
 let mag = if (r + d) % 4 == 0 { 1e3 } else { 1e-3 };
 kv[r * row + d] = lcg(&mut seed) * mag;
 }
 }
 // sinks pulled below the active row's score so they only contribute
 // to denom (matches antirez behaviour where sinks are part of max
 // pool but not output sum).
 let attn_sinks: Vec<f32> = vec![-1.0, -1.0];

 let d = CpuAttentionDispatcher;
 let ours = d.flash_attn_decode(
 &p,
 &q,
 &kv,
 /*n_raw*/ n_kv,
 /*raw_cap*/ cap,
 /*raw_start*/ 0,
 None,
 0,
 None,
 0,
 &attn_sinks,
 );

 // Hand-stamp antirez-spec (normalize-LATE):
 // 1. score[r] = dot(q_h, kv_r) * (1/sqrt(head_dim))
 // 2. max = max(sinks, scores...)
 // 3. denom = exp(sinks - max); for r: w_r = exp(s_r - max);
 // denom += w_r; oh[d] += w_r * kv_r[d]
 // 4. oh /= denom
 let mut want = vec![0.0f32; n_head as usize * row];
 let scale = (head_dim as f32).sqrt().recip();
 for h in 0..n_head as usize {
 let q_h = &q[h * row..(h + 1) * row];
 let mut score = vec![0.0f32; n_kv as usize];
 let mut max_score = attn_sinks[h];
 for r in 0..n_kv as usize {
 let kv_r = &kv[r * row..(r + 1) * row];
 let dot: f32 = q_h.iter().zip(kv_r.iter()).map(|(a, b)| a * b).sum();
 score[r] = dot * scale;
 if score[r] > max_score {
 max_score = score[r];
 }
 }
 let oh = &mut want[h * row..(h + 1) * row];
 let mut denom = (attn_sinks[h] - max_score).exp();
 for r in 0..n_kv as usize {
 let kv_r = &kv[r * row..(r + 1) * row];
 let weight = (score[r] - max_score).exp();
 denom += weight;
 for d in 0..row {
 oh[d] += weight * kv_r[d];
 }
 }
 let inv = 1.0 / denom;
 for d in 0..row {
 oh[d] *= inv;
 }
 }

 // Hand-stamp old-port spec (normalize-EARLY):
 // for r: w_r = exp(s_r - max); oh[d] += (w_r / denom) * kv_r[d]
 // where denom is precomputed over ALL r (so each r uses the final
 // denom — matches the pre-fix `flash_attn_decode` exactly).
 let mut want_early = vec![0.0f32; n_head as usize * row];
 for h in 0..n_head as usize {
 let q_h = &q[h * row..(h + 1) * row];
 let mut score = vec![0.0f32; n_kv as usize];
 let mut max_score = attn_sinks[h];
 for r in 0..n_kv as usize {
 let kv_r = &kv[r * row..(r + 1) * row];
 let dot: f32 = q_h.iter().zip(kv_r.iter()).map(|(a, b)| a * b).sum();
 score[r] = dot * scale;
 if score[r] > max_score {
 max_score = score[r];
 }
 }
 let mut sum = (attn_sinks[h] - max_score).exp();
 let mut exp_scores = vec![0.0f32; n_kv as usize];
 for r in 0..n_kv as usize {
 exp_scores[r] = (score[r] - max_score).exp();
 sum += exp_scores[r];
 }
 let inv = 1.0 / sum;
 let oh = &mut want_early[h * row..(h + 1) * row];
 for r in 0..n_kv as usize {
 let w = exp_scores[r] * inv;
 let kv_r = &kv[r * row..(r + 1) * row];
 for d in 0..row {
 oh[d] += w * kv_r[d];
 }
 }
 }

 // Discriminator: the two orderings MUST differ at bit level on
 // this adversarial input — otherwise the test is trivial.
 let any_diff = want
 .iter()
 .zip(want_early.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_diff,
 "normalize-late and normalize-early gave bit-identical outputs — \
 test is trivial; pick a more adversarial input."
 );

 // Ours must match antirez-spec (normalize-LATE) bit-exactly.
 for i in 0..ours.len() {
 assert_eq!(
 ours[i].to_bits(),
 want[i].to_bits(),
 "head/dim={i}: ours={ours} antirez_late={want} (early would be {early})",
 ours = ours[i],
 want = want[i],
 early = want_early[i]
 );
 }
 }

 /// `flash_attn_decode` Q·K dot reduction-tree topology vs
 /// antirez `dot_f32` (ds4.c:4689). Default path uses
 /// `Iterator::sum::<f32>()` — a sequential left-fold f32 accumulator.
 /// Antirez `dot_f32` uses two 4-lane f32 FMA partials reduced via a
 /// final pairwise add (NEON `vaddvq_f32(vaddq_f32(acc0, acc1))`).
 /// Algebraically equal in real arithmetic; f32-ULP different.
 ///
 /// `DS4_MATVEC_F32_FIDELITY=1` switches the dot to the antirez form.
 /// Same env as (matvec_f32) — both are the same divergence
 /// class, different callsites.
 ///
 /// Test exercises 256-element vectors with mixed-magnitude entries so
 /// the two reduction trees pick up distinct ULP rounding cascades.
 #[test]
 fn flash_attn_decode_dot_antirez_fidelity_gate_switches_reducer() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());

 // n_head=1, head_dim=256, n_kv=2 (one strongly-attended row +
 // one decoy). head_dim must be ≥ 8 to exercise the 8-wide FMA
 // partial loop in `dot_f32_antirez`; we pick 256 to push many
 // partials through and amplify reduction-tree differences.
 let mut p = dummy_params();
 let n_head = 1u32;
 let head_dim = 256u32;
 p.n_head = n_head;
 p.head_dim = head_dim;
 p.n_lora_kv = head_dim;
 let row = head_dim as usize;
 let n_kv = 2u32;
 let cap = n_kv;

 // Adversarial input: q has mixed-magnitude entries (1.0 / -1e-3)
 // alternating to span 3 orders of magnitude per summand. KV rows
 // are scaled-down near-copies of q with magnitudes tuned so the
 // dots, after `* scale = 1/sqrt(256) = 1/16`, give *small* scores
 // (≈ 0.5 / 0.45) — softmax then keeps both rows with non-trivial
 // weights, so the f32-ULP dot difference propagates to the
 // output instead of getting absorbed by saturating exp().
 let mut q = vec![0.0f32; n_head as usize * row];
 for d in 0..row {
 q[d] = if d & 1 == 0 { 1.0 } else { -1e-3 };
 }

 let mut kv = vec![0.0f32; cap as usize * row];
 let mut seed = 0x1234567890ABCDEFu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 // row 0 + row 1 dots come out close (~64 and ~57.6 before scale),
 // after scale ≈ 4.0 / 3.6 — softmax keeps both with non-tiny
 // weights so the dot ULP difference cascades into the output.
 for d in 0..row {
 kv[d] = 0.5 * q[d] + lcg(&mut seed) * 1e-3;
 }
 for d in 0..row {
 kv[row + d] = 0.45 * q[d] + lcg(&mut seed) * 1e-3;
 }
 // Sinks well below the row scores so they only matter to denom.
 let attn_sinks: Vec<f32> = vec![-10.0];

 let d = CpuAttentionDispatcher;

 // Hand-stamp BOTH dot orderings explicitly.
 let scale_f = (head_dim as f32).sqrt().recip();
 let dot_leftfold =
 |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };
 let dot_antirez = |a: &[f32], b: &[f32]| -> f32 { crate::forward::dot_f32_antirez(a, b) };

 // Verify the two dots actually diverge on (q, kv_row0) — if not,
 // the test is trivial.
 let lf0 = dot_leftfold(&q[..row], &kv[..row]);
 let ar0 = dot_antirez(&q[..row], &kv[..row]);
 let lf1 = dot_leftfold(&q[..row], &kv[row..2 * row]);
 let ar1 = dot_antirez(&q[..row], &kv[row..2 * row]);
 let any_dot_diff = lf0.to_bits() != ar0.to_bits() || lf1.to_bits() != ar1.to_bits();
 assert!(
 any_dot_diff,
 "left-fold sum and antirez dot_f32 produced bit-identical results — \
 test is trivial; pick a more adversarial input. \
 lf0={lf0:?} ar0={ar0:?} lf1={lf1:?} ar1={ar1:?}"
 );

 // Hand-stamp full flash-attn output for each dot path. (Both use
 // normalize-LATE — that's and not the discriminator here.)
 let attn_with = |dot: &dyn Fn(&[f32], &[f32]) -> f32| -> Vec<f32> {
 let mut out = vec![0.0f32; n_head as usize * row];
 for h in 0..n_head as usize {
 let q_h = &q[h * row..(h + 1) * row];
 let mut score = [0.0f32; 2];
 let mut mx = attn_sinks[h];
 for r in 0..n_kv as usize {
 let kv_r = &kv[r * row..(r + 1) * row];
 score[r] = dot(q_h, kv_r) * scale_f;
 if score[r] > mx {
 mx = score[r];
 }
 }
 let oh = &mut out[h * row..(h + 1) * row];
 let mut denom = (attn_sinks[h] - mx).exp();
 for r in 0..n_kv as usize {
 let kv_r = &kv[r * row..(r + 1) * row];
 let w = (score[r] - mx).exp();
 denom += w;
 for di in 0..row {
 oh[di] += w * kv_r[di];
 }
 }
 let inv = 1.0 / denom;
 for di in 0..row {
 oh[di] *= inv;
 }
 }
 out
 };

 let want_leftfold = attn_with(&dot_leftfold);
 let want_antirez = attn_with(&dot_antirez);

 // The two paths must diverge somewhere at bit level.
 let any_out_diff = want_leftfold
 .iter()
 .zip(want_antirez.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_out_diff,
 "left-fold and antirez dot produced bit-identical full attn outputs — \
 test is trivial; pick a more adversarial input."
 );

 // Gate OFF → ours == left-fold.
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 let ours_off =
 d.flash_attn_decode(&p, &q, &kv, n_kv, cap, 0, None, 0, None, 0, &attn_sinks);
 for i in 0..ours_off.len() {
 assert_eq!(
 ours_off[i].to_bits(),
 want_leftfold[i].to_bits(),
 "gate OFF: head/dim={i}: ours={off} expected_leftfold={lf}",
 off = ours_off[i],
 lf = want_leftfold[i]
 );
 }

 // Gate ON → ours == antirez dot.
 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 let ours_on = d.flash_attn_decode(&p, &q, &kv, n_kv, cap, 0, None, 0, None, 0, &attn_sinks);
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 for i in 0..ours_on.len() {
 assert_eq!(
 ours_on[i].to_bits(),
 want_antirez[i].to_bits(),
 "gate ON: head/dim={i}: ours={on} expected_antirez={ar}",
 on = ours_on[i],
 ar = want_antirez[i]
 );
 }
 }

 /// `hc_collapse_norm` step (b) computes `mix = hc_fn @ flat`
 /// row-by-row using `Iterator::sum::<f32>()` (left-fold). Antirez
 /// `matvec_f16` (ds4.c:2620) → `dot_f16_row` (ds4.c:2587) uses two
 /// 4-lane FMA partials + pairwise add via `vaddvq_f32(vaddq_f32(...))`,
 /// same reduction-tree topology as
 ///
 /// Algebraically equal; f32-ULP different. `DS4_MATVEC_F32_FIDELITY=1`
 /// switches the dot to `dot_f32_antirez`. Reuses existing env (no new
 /// toggle). Called twice per layer per decode step (Attn + Ffn halves
 /// × 43 layers), so per-call ULP differences compound.
  /// THE TWO IMPLEMENTATIONS OF THE SAME FORMULA MUST AGREE.
 ///
 /// `hc_collapse_norm` (the DS4 path, f32) and `hyper_connection::ref_mixes` (transcribed from the reference, f64) both
 /// implement `kernel.py:426-459`. If either drifts, the DS4 path and the GLM/Kimi path stop matching the reference **in
 /// opposite directions** -- and no existing test would notice, because the cross-driver test compares two drivers that both
 /// call the same one.
 #[test]
 fn the_two_implementations_of_the_reference_hc_agree() {
     let mut p = dummy_params();
     p.d_embd = 16;
     p.hc_eps = 1e-6;
     let n_hc = p.n_hc as usize;
     let d_embd = p.d_embd as usize;
     let hc_dim = n_hc * d_embd;
     let mix_hc = 2 * n_hc + n_hc * n_hc;

     let mut seed = 0x5EED5EEDu64;
     let mut lcg = |s: &mut u64| -> f32 {
         *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
         (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
     };
     let prev_hc: Vec<f32> = (0..hc_dim).map(|_| lcg(&mut seed) * 0.5).collect();
     let hc_fn: Vec<f32> = (0..hc_dim * mix_hc).map(|_| lcg(&mut seed) * 0.05).collect();
     let hc_base: Vec<f32> = (0..mix_hc).map(|_| lcg(&mut seed) * 0.4).collect();
     let hc_scale = vec![0.9f32, 1.1, 0.7];

     let d = CpuAttentionDispatcher;
     let (_cur, _normed, split) = d.hc_collapse_norm(&p, HcKind::Attn, &hc_fn, &hc_scale, &hc_base, &prev_hc, None);

     let flat64: Vec<f64> = prev_hc.iter().map(|v| *v as f64).collect();
     let r = crate::hyper_connection::ref_mixes(&flat64, &hc_fn, &hc_scale, &hc_base, n_hc, 20, p.hc_eps as f64);

     let close = |a: f32, b: f64, what: &str, i: usize| {
         let (a, b) = (a as f64, b);
         let denom = b.abs().max(1e-6);
         // 1e-3 relative: f32 vs f64 through 20 sinkhorn iterations drifts ~2.3e-4 (measured), which is precision, not
         // formula. A DIFFERENT formula differs by orders of magnitude, so this tolerance still catches one.
         assert!((a - b).abs() / denom < 1e-3 || (a - b).abs() < 1e-5,
                 "{what}[{i}]: hc_collapse_norm {a:.7} vs ref_mixes {b:.7}");
     };
     for i in 0..n_hc {
         close(split[i], r.pre[i], "pre", i);
         close(split[n_hc + i], r.post[i], "post", i);
     }
     for i in 0..n_hc * n_hc {
         close(split[2 * n_hc + i], r.comb[i], "comb", i);
     }
 }

#[test]
 fn hc_collapse_norm_mix_dot_antirez_fidelity_gate_switches_reducer() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());

 // d_embd = 256 → hc_dim = 512 (n_hc=2); the per-row dot of length
 // 512 pushes 64 iterations through the 8-wide FMA partial loop.
 let mut p = dummy_params();
 p.d_embd = 256;
 let n_hc = p.n_hc as usize;
 let d_embd = p.d_embd as usize;
 let hc_dim = n_hc * d_embd;
 let mix_hc = 2 * n_hc + n_hc * n_hc;

 // Adversarial: prev_hc + hc_fn with mixed magnitudes so the two
 // reduction trees pick up distinct ULP cascades. We also need the
 // mix output to actually flow through to a downstream measurable
 // (the `cur` aggregate) so the propagation can be asserted.
 let mut seed = 0xFEEDFACEBADDEEDu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 // Inputs tuned so the dot reduction trees diverge cleanly AND
 // the downstream sigmoid/sinkhorn pipeline doesn't absorb it.
 // The pre_scale is large so the mix→sigmoid output is sensitive,
 // and prev_hc has near-equal-magnitude values across both slots
 // so split_pre weighting maps to measurable `cur` differences.
 let mut prev_hc = vec![0.0f32; hc_dim];
 for i in 0..hc_dim {
 prev_hc[i] = lcg(&mut seed) * 10.0;
 }
 let mut hc_fn = vec![0.0f32; hc_dim * mix_hc];
 // Mixed-magnitude entries (1.0 vs 1e-4) per row so left-fold
 // catastrophically rounds when summing big+small in a single
 // accumulator, but `dot_f32_antirez` keeps two parallel partial
 // accumulators (acc0/acc1) so the small terms aren't lost.
 for r in 0..mix_hc {
 for i in 0..hc_dim {
 hc_fn[r * hc_dim + i] = if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed);
 }
 }
 // pre_scale = 1e-2 keeps mix*scale small enough for sigmoid to
 // have non-zero derivative; near 0 the derivative is 0.25 max.
 let hc_scale = vec![1e-2f32, 1e-2, 1.0];
 let hc_base = vec![0.0f32; mix_hc];

 // Hand-stamp the FLAT vector (rms_norm_inplace, no gamma) that
 // hc_collapse_norm computes from prev_hc — both gate paths use
 // the same flat, so only the dot reduction differs downstream.
 let mut flat = prev_hc.clone();
 CpuAttentionDispatcher::rms_norm_inplace(p.rms_eps, &mut flat, None);

 // Probe the dot reductions across rows to confirm they actually
 // diverge at bit level — otherwise the test is trivial.
 let mut any_dot_diff = false;
 for r in 0..mix_hc {
 let row = &hc_fn[r * hc_dim..(r + 1) * hc_dim];
 let lf: f32 = row.iter().zip(flat.iter()).map(|(a, b)| a * b).sum();
 let ar: f32 = crate::forward::dot_f32_antirez(row, &flat);
 if lf.to_bits() != ar.to_bits() {
 any_dot_diff = true;
 break;
 }
 }
 assert!(
 any_dot_diff,
 "left-fold and antirez dot produced bit-identical mix \
 across all rows — pick a more adversarial input."
 );

 let d = CpuAttentionDispatcher;

 // Gate OFF: hc_collapse_norm uses left-fold sum.
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 let (cur_off, _normed_off, _split_off) = d.hc_collapse_norm(
 &p,
 HcKind::Attn,
 &hc_fn,
 &hc_scale,
 &hc_base,
 &prev_hc,
 None,
 );

 // Gate ON: hc_collapse_norm uses antirez FMA pair-reduce dot.
 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 let (cur_on, _normed_on, _split_on) = d.hc_collapse_norm(
 &p,
 HcKind::Attn,
 &hc_fn,
 &hc_scale,
 &hc_base,
 &prev_hc,
 None,
 );
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");

 // The gate must change the `split` output (which carries the
 // sigmoid+sinkhorn of mix) — `cur` is the residual-magnitude
 // weighted sum and may collapse on near-identical split values.
 let (_, _, split_off) = d.hc_collapse_norm(
 &p,
 HcKind::Attn,
 &hc_fn,
 &hc_scale,
 &hc_base,
 &prev_hc,
 None,
 );
 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 let (_, _, split_on) = d.hc_collapse_norm(
 &p,
 HcKind::Attn,
 &hc_fn,
 &hc_scale,
 &hc_base,
 &prev_hc,
 None,
 );
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");

 let _ = cur_off;
 let _ = cur_on;

 let any_split_diff = split_off
 .iter()
 .zip(split_on.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_split_diff,
 "DS4_MATVEC_F32_FIDELITY=1 did NOT change hc_collapse_norm \
 `split` output — gate is a no-op on this input."
 );
 }

 /// `attn_output_proj` performs two left-fold dots (stage-1
 /// grouped matvec `w_o_a · heads_g` and stage-2 dense `w_o_b · attn_low`)
 /// and `shared_down_hc_expand_add` performs a left-fold dot (`w_down ·
 /// shared_mid`). Antirez `matvec_q8_0_grouped_rows` (ds4.c:3396) and
 /// `matvec_q8_0` (ds4.c:3337) use 2×4-lane FMA partials + pairwise add
 /// — same reduction-tree class as Reuses
 /// `DS4_MATVEC_F32_FIDELITY=1`. Three callsites covered by this gate.
 ///
 /// The test exercises `attn_output_proj` (which compounds two gated
 /// dots: w_o_a · heads_g then w_o_b · attn_low) with adversarial
 /// alternating 1.0/1e-4 magnitudes per row.
 #[test]
 fn attn_output_proj_dots_antirez_fidelity_gate_switches_reducer() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());

 // d_embd=256, n_head=2, head_dim=128 → q_dim=256. n_out_group=2 →
 // group_dim=128. n_lora_o=4 → out_low_dim=8. Stage-1 row length
 // = group_dim = 128 (16 FMA-loop iters). Stage-2 row length = 8.
 let mut p = dummy_params();
 p.d_embd = 256;
 p.n_head = 2;
 p.head_dim = 128;
 p.n_out_group = 2;
 let q_dim = p.q_dim();
 let n_groups = p.n_out_group as usize;
 let group_dim = q_dim / n_groups;
 let n_lora_o = 4usize;
 let out_low_dim = n_groups * n_lora_o;
 let d_embd = p.d_embd as usize;
 let n_hc = p.n_hc as usize;

 // Adversarial heads: alternating 1.0/1e-4 per element. w_o_a rows
 // alternate magnitudes so the reduction tree picks up ULP cascades.
 let mut seed = 0xCAFEBABE_DEADBEEFu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 let heads: Vec<f32> = (0..q_dim)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 let w_a: Vec<f32> = (0..out_low_dim * group_dim)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 // w_o_b is small (out_low_dim=8), so just use uniform small spread.
 let w_b: Vec<f32> = (0..d_embd * out_low_dim).map(|_| lcg(&mut seed)).collect();
 // cur_hc, split_post zeroed so the test signal flows only through
 // the gated dots (no add of cur*comb terms).
 let cur = vec![0.0f32; n_hc * d_embd];
 let split_post: Vec<f32> = (0..n_hc).map(|i| (i as f32) + 1.0).collect();
 let comb = vec![0.0f32; n_hc * n_hc];

 let d = CpuAttentionDispatcher;

 // Probe stage-1 left-fold vs antirez dot across rows (sanity).
 let mut any_dot_diff = false;
 for g in 0..n_groups {
 let heads_g = &heads[g * group_dim..(g + 1) * group_dim];
 for l in 0..n_lora_o {
 let row = &w_a[(g * n_lora_o + l) * group_dim..(g * n_lora_o + l + 1) * group_dim];
 let lf: f32 = row.iter().zip(heads_g.iter()).map(|(a, b)| a * b).sum();
 let ar: f32 = crate::forward::dot_f32_antirez(row, heads_g);
 if lf.to_bits() != ar.to_bits() {
 any_dot_diff = true;
 }
 }
 }
 assert!(
 any_dot_diff,
 "stage-1 left-fold and antirez dot agree on all rows — \
 pick a more adversarial input."
 );

 // Make sure DS4_Q8_0_ACT is OFF — it would round-trip heads through
 // q8_0 and contaminate the reduction-tree signal we are isolating.
 std::env::remove_var("DS4_Q8_0_ACT");

 // Gate OFF → left-fold.
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 let out_off = d.attn_output_proj(&p, &heads, &w_a, &w_b, &cur, &split_post, &comb);

 // Gate ON → antirez FMA pair-reduce.
 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 let out_on = d.attn_output_proj(&p, &heads, &w_a, &w_b, &cur, &split_post, &comb);
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");

 let any_out_diff = out_off
 .iter()
 .zip(out_on.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_out_diff,
 "DS4_MATVEC_F32_FIDELITY=1 did NOT change attn_output_proj \
 output — gate is a no-op."
 );
 }

 /// companion — `shared_down_hc_expand_add` dot `w_down · mid`.
 #[test]
 fn shared_down_dot_antirez_fidelity_gate_switches_reducer() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());

 let mut p = dummy_params();
 p.d_embd = 256;
 let d_embd = p.d_embd as usize;
 let n_hc = p.n_hc as usize;
 let sd = 256usize; // 32 FMA-loop iters

 let mut seed = 0xBEEFFACE_F00DCAFEu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 // Adversarial mid: alternating 1.0/1e-4 magnitudes.
 let shared_mid: Vec<f32> = (0..sd)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 // w_down row alternates magnitudes too.
 let w_down: Vec<f32> = (0..d_embd * sd)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 let routed_out = vec![0.0f32; d_embd];
 let after_attn_hc = vec![0.0f32; n_hc * d_embd];
 let split_post: Vec<f32> = (0..n_hc).map(|i| (i as f32) + 1.0).collect();
 let comb = vec![0.0f32; n_hc * n_hc];

 // Probe: left-fold vs antirez dot bit-differ across rows.
 let mut any_dot_diff = false;
 for e in 0..d_embd {
 let row = &w_down[e * sd..(e + 1) * sd];
 let lf: f32 = row.iter().zip(shared_mid.iter()).map(|(a, b)| a * b).sum();
 let ar: f32 = crate::forward::dot_f32_antirez(row, &shared_mid);
 if lf.to_bits() != ar.to_bits() {
 any_dot_diff = true;
 break;
 }
 }
 assert!(
 any_dot_diff,
 "shared_down left-fold and antirez dot agree on all rows — \
 pick a more adversarial input."
 );

 // Keep DS4_Q8_0_ACT OFF so it doesn't perturb mid.
 std::env::remove_var("DS4_Q8_0_ACT");

 let d = CpuAttentionDispatcher;

 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 let out_off = d.shared_down_hc_expand_add(
 &p,
 &shared_mid,
 &w_down,
 &routed_out,
 &after_attn_hc,
 &split_post,
 &comb,
 );

 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 let out_on = d.shared_down_hc_expand_add(
 &p,
 &shared_mid,
 &w_down,
 &routed_out,
 &after_attn_hc,
 &split_post,
 &comb,
 );
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");

 let any_out_diff = out_off
 .iter()
 .zip(out_on.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_out_diff,
 "DS4_MATVEC_F32_FIDELITY=1 did NOT change shared_down output \
 — gate is a no-op."
 );
 }

 #[test]
 fn compressor_decode_one_q8_0_round_trip_on_x_changes_output_when_gate_on() {
 // antirez `compressor_decode_one` (ds4.c:6240) calls
 // `quantize_q8_0_activation(x, ...)` then `matvec_q8_0_pair_prequant` —
 // x is q8_0-round-tripped before the int8 dot. Our port did the dot
 // f32-direct; this test exercises the DS4_Q8_0_ACT=1 gate.
 //
 // Discrimination strategy:
 // 1. Build an adversarial x sized as 1 q8_0 block (32 elements)
 // where values straddle half-quanta so q8_0_round_trip(x) != x.
 // 2. Build weights and ape so the f32-direct projection differs
 // from the q8_0-quantized projection at bit level.
 // 3. Call compressor_decode_one with gate OFF → ungated.
 // 4. Call again with gate ON → gated.
 // 5. Assert ungated != gated (proves the gate FIRES — not no-op).
 // 6. Hand-stamp antirez-spec reference (q8_0(x) then col-major dot
 // + ape add) and assert gated state matches bit-exactly.
 let mut p = dummy_params();
 p.compress_ratio = 4;
 p.head_dim = 2;
 p.n_rot = 0;
 p.d_embd = 32; // 1 q8_0 block
 let ratio = p.compress_ratio as usize;
 let head_dim = p.head_dim as usize;
 let width = 2 * head_dim;
 let rows = 2 * ratio;
 let in_dim = p.d_embd as usize;

 // Adversarial x: values that round to non-zero deltas under q8_0
 // (amax-based scale, mix of close-to-half-quanta and full-range).
 let mut x = vec![0.0f32; in_dim];
 let mut seed = 0xFEEDFACECAFEBABEu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 for i in 0..in_dim {
 x[i] = lcg(&mut seed) * 1.5;
 }
 // pin amax so half-quanta straddling is sharp
 x[0] = 1.4999;
 x[1] = -1.4999;
 x[2] = 0.5 * (1.4999 / 127.0); // half-quanta straddler

 // Confirm q8_0 round-trip actually changes x.
 let x_q80 = crate::forward::q8_0_round_trip(&x);
 let perturbed = x
 .iter()
 .zip(x_q80.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 perturbed,
 "q8_0 round-trip did not change x — test is trivial; tune adversarial input"
 );

 // Weights: small but non-zero, distinct per (j, k) so the dot
 // picks up the q8_0 perturbation differently per output.
 let mut w_kv = vec![0.0f32; in_dim * width];
 let mut w_gate = vec![0.0f32; in_dim * width];
 for j in 0..width {
 for k in 0..in_dim {
 w_kv[j * in_dim + k] = 0.01 * (j as f32 + 1.0) + 0.003 * (k as f32);
 w_gate[j * in_dim + k] = 0.007 * (j as f32 + 1.0) - 0.002 * (k as f32);
 }
 }
 let w_ape = vec![0.0f32; width * ratio];
 let w_norm = vec![1.0f32; head_dim];
 let comp = CompressorInputs {
 w_kv: &w_kv,
 w_gate: &w_gate,
 // CPU trait path — always f32 (non-lean weights keep the dequant).
 w_kv_f16: None,
 w_gate_f16: None,
 w_ape: &w_ape,
 w_norm: &w_norm,
 head_dim: p.head_dim,
 compress_ratio: p.compress_ratio,
 };

 let d = CpuAttentionDispatcher;
 // Acquire ENV_LOCK before mutating DS4_Q8_0_ACT — other tests in
 // this module race on the same var.
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let prev = std::env::var("DS4_Q8_0_ACT").ok();

 // Run UNGATED (no env var).
 std::env::remove_var("DS4_Q8_0_ACT");
 let mut state_kv_un = vec![0.0f32; rows * width];
 let mut state_score_un = vec![0.0f32; rows * width];
 let _ = compressor_decode_one(&d, &p, &comp, &x, &mut state_kv_un, &mut state_score_un, 0);

 // Run GATED.
 std::env::set_var("DS4_Q8_0_ACT", "1");
 let mut state_kv_g = vec![0.0f32; rows * width];
 let mut state_score_g = vec![0.0f32; rows * width];
 let _ = compressor_decode_one(&d, &p, &comp, &x, &mut state_kv_g, &mut state_score_g, 0);
 match prev {
 Some(v) => std::env::set_var("DS4_Q8_0_ACT", v),
 None => std::env::remove_var("DS4_Q8_0_ACT"),
 }

 // Discriminator: gate must perturb at least one state entry.
 let differ = state_kv_un
 .iter()
 .zip(state_kv_g.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits())
 || state_score_un
 .iter()
 .zip(state_score_g.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 differ,
 "DS4_Q8_0_ACT gate had NO effect on compressor state — \
 either the round-trip is the identity on this x, or the gate is a no-op."
 );

 // Hand-stamp antirez-spec reference:
 // x_q80 = quantize_q8_0_activation+dequantize(x)
 // kv_cur[j] = sum_k w_kv[j*in+k] * x_q80[k]
 // sc_cur[j] = sum_k w_gate[j*in+k] * x_q80[k]
 // commit to state_kv/state_score at row=(ratio + pos_mod)=4
 // (ratio==4, pos_mod=0).
 let mut ref_state_kv = vec![0.0f32; rows * width];
 let mut ref_state_score = vec![0.0f32; rows * width];
 let row_idx = ratio + 0; // ratio==4, pos_mod=0
 for j in 0..width {
 let mut acc_kv = 0.0f32;
 let mut acc_sc = 0.0f32;
 let base = j * in_dim;
 for k in 0..in_dim {
 acc_kv += w_kv[base + k] * x_q80[k];
 acc_sc += w_gate[base + k] * x_q80[k];
 }
 ref_state_kv[row_idx * width + j] = acc_kv;
 ref_state_score[row_idx * width + j] = acc_sc; // w_ape is zero
 }

 // Gated state must match the antirez-spec reference bit-exactly
 // at the committed row.
 for j in 0..width {
 let idx = row_idx * width + j;
 assert_eq!(
 state_kv_g[idx].to_bits(),
 ref_state_kv[idx].to_bits(),
 "state_kv mismatch at j={j}: ours={ours} ref={r}",
 ours = state_kv_g[idx],
 r = ref_state_kv[idx]
 );
 assert_eq!(
 state_score_g[idx].to_bits(),
 ref_state_score[idx].to_bits(),
 "state_score mismatch at j={j}: ours={ours} ref={r}",
 ours = state_score_g[idx],
 r = ref_state_score[idx]
 );
 }
 }

 /// compressor paired matvec dot reduction tree.
 ///
 /// Antirez `matvec_q8_0_pair_prequant` (ds4.c:6241) → `dot_q8_0_row_pair`
 /// (ds4.c:2876) uses 4× float32x4_t FMA accumulators reduced via NEON
 /// pair-add. Our default scalar left-fold is f32-ULP-different. The gate
 /// `DS4_MATVEC_F32_FIDELITY=1` routes the per-row dot through
 /// `dot_f32_antirez` ( reduction tree) — bit-different from the
 /// scalar fold and bit-matching the antirez reduction shape.
 ///
 /// Discrimination strategy:
 /// 1. Build adversarial x and weights where the left-fold and antirez
 /// dot reductions produce bit-distinct rows.
 /// 2. Gate OFF → capture state.
 /// 3. Gate ON → capture state.
 /// 4. Assert ungated != gated (proves the gate FIRES).
 /// 5. Hand-stamp antirez-spec reference dot via `dot_f32_antirez` and
 /// assert gated state matches bit-exactly at the committed row.
 #[test]
 fn compressor_paired_matvec_antirez_fidelity_gate_switches_reducer() {
 let mut p = dummy_params();
 p.compress_ratio = 4;
 p.head_dim = 2;
 p.n_rot = 0;
 p.d_embd = 256; // 32 FMA-loop iters per dot
 let ratio = p.compress_ratio as usize;
 let head_dim = p.head_dim as usize;
 let width = 2 * head_dim;
 let rows = 2 * ratio;
 let in_dim = p.d_embd as usize;

 let mut seed = 0xC0DE_BEEF_F00D_FACEu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 // Adversarial x: alternating 1.0 / 1e-4 magnitudes so partial sums
 // accumulate distinct rounding under left-fold vs FMA pair-reduce.
 let x: Vec<f32> = (0..in_dim)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 // Weights alternate magnitudes too.
 let w_kv: Vec<f32> = (0..in_dim * width)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 let w_gate: Vec<f32> = (0..in_dim * width)
 .map(|i| if i & 1 == 0 { 1.0 } else { 1e-4 } * lcg(&mut seed))
 .collect();
 let w_ape = vec![0.0f32; width * ratio];
 let w_norm = vec![1.0f32; head_dim];
 let comp = CompressorInputs {
 w_kv: &w_kv,
 w_gate: &w_gate,
 // CPU trait path — always f32 (non-lean weights keep the dequant).
 w_kv_f16: None,
 w_gate_f16: None,
 w_ape: &w_ape,
 w_norm: &w_norm,
 head_dim: p.head_dim,
 compress_ratio: p.compress_ratio,
 };

 // Pre-flight probe: at least one (j, src) row pair must bit-differ
 // between left-fold and antirez dot. Otherwise the test is trivial.
 let mut any_dot_diff = false;
 for j in 0..width {
 let base = j * in_dim;
 let row_kv = &w_kv[base..base + in_dim];
 let row_sc = &w_gate[base..base + in_dim];
 let lf_kv: f32 = row_kv.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
 let ar_kv: f32 = crate::forward::dot_f32_antirez(row_kv, &x);
 let lf_sc: f32 = row_sc.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
 let ar_sc: f32 = crate::forward::dot_f32_antirez(row_sc, &x);
 if lf_kv.to_bits() != ar_kv.to_bits() || lf_sc.to_bits() != ar_sc.to_bits() {
 any_dot_diff = true;
 break;
 }
 }
 assert!(
 any_dot_diff,
 "left-fold and antirez dot agree on all rows — pick a more \
 adversarial input."
 );

 let d = CpuAttentionDispatcher;
 // Serialize against other env-mutating tests (DS4_MATVEC_F32_FIDELITY
 // and DS4_Q8_0_ACT both consumed by this codepath).
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 // Keep DS4_Q8_0_ACT OFF so x is not q8_0-perturbed (would mask the
 // reduction-tree discriminator).
 let prev_q80 = std::env::var("DS4_Q8_0_ACT").ok();
 let prev_fma = std::env::var("DS4_MATVEC_F32_FIDELITY").ok();
 std::env::remove_var("DS4_Q8_0_ACT");

 // Gate OFF.
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 let mut state_kv_off = vec![0.0f32; rows * width];
 let mut state_score_off = vec![0.0f32; rows * width];
 let _ = compressor_decode_one(
 &d,
 &p,
 &comp,
 &x,
 &mut state_kv_off,
 &mut state_score_off,
 0,
 );

 // Gate ON.
 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 let mut state_kv_on = vec![0.0f32; rows * width];
 let mut state_score_on = vec![0.0f32; rows * width];
 let _ = compressor_decode_one(&d, &p, &comp, &x, &mut state_kv_on, &mut state_score_on, 0);
 // Restore env.
 match prev_q80 {
 Some(v) => std::env::set_var("DS4_Q8_0_ACT", v),
 None => std::env::remove_var("DS4_Q8_0_ACT"),
 }
 match prev_fma {
 Some(v) => std::env::set_var("DS4_MATVEC_F32_FIDELITY", v),
 None => std::env::remove_var("DS4_MATVEC_F32_FIDELITY"),
 }

 let differ = state_kv_off
 .iter()
 .zip(state_kv_on.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits())
 || state_score_off
 .iter()
 .zip(state_score_on.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 differ,
 "DS4_MATVEC_F32_FIDELITY=1 did NOT change compressor paired \
 matvec output — gate is a no-op."
 );

 // Hand-stamp antirez-spec reference at the committed row
 // (row_idx = ratio + pos_mod = 4 + 0 = 4).
 let row_idx = ratio + 0;
 for j in 0..width {
 let base = j * in_dim;
 let row_kv = &w_kv[base..base + in_dim];
 let row_sc = &w_gate[base..base + in_dim];
 let ref_kv = crate::forward::dot_f32_antirez(row_kv, &x);
 let ref_sc = crate::forward::dot_f32_antirez(row_sc, &x);
 let idx = row_idx * width + j;
 assert_eq!(
 state_kv_on[idx].to_bits(),
 ref_kv.to_bits(),
 "gated state_kv[{j}] = {ours} != antirez-spec {r}",
 ours = state_kv_on[idx],
 r = ref_kv,
 );
 assert_eq!(
 state_score_on[idx].to_bits(),
 ref_sc.to_bits(),
 "gated state_score[{j}] = {ours} != antirez-spec {r}",
 ours = state_score_on[idx],
 r = ref_sc,
 );
 }
 }

 #[test]
 fn attn_output_proj_shape_matches_hc() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let q_dim = p.q_dim();
 let n_groups = p.n_out_group as usize;
 let group_dim = q_dim / n_groups;
 let n_lora_o = 3usize;
 let out_low_dim = n_groups * n_lora_o;
 let heads = vec![0.5f32; q_dim];
 let w_a = vec![0.1f32; n_groups * n_lora_o * group_dim];
 let w_b = vec![0.2f32; p.d_embd as usize * out_low_dim];
 let cur = vec![0.0f32; p.hc_dim()];
 let split_post = vec![0.5f32; p.n_hc as usize];
 let split_comb = vec![0.0f32; (p.n_hc as usize) * (p.n_hc as usize)];
 let after = d.attn_output_proj(&p, &heads, &w_a, &w_b, &cur, &split_post, &split_comb);
 assert_eq!(after.len(), p.hc_dim());
 }

 #[test]
 fn shared_expert_silu_clamp_smoke() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let sd = 2u32;
 let ffn = vec![0.5f32; p.d_embd as usize];
 let wg = vec![0.1f32; sd as usize * p.d_embd as usize];
 let wu = vec![0.2f32; sd as usize * p.d_embd as usize];
 let out = d.shared_expert(&p, &ffn, &wg, &wu, sd, 7.0);
 assert_eq!(out.len(), sd as usize);
 }

 /// `shared_expert` exercises BOTH (left-fold dot vs
 /// antirez `dot_f32`) AND (inline silu vs branched
 /// `sigmoid_stable`) in a single callsite. Previously:
 /// - dotted `row_g·ffn` / `row_u·ffn` with `Iterator::sum::<f32>()`
 /// - applied `silu` inline as `g / (1.0 + (-g).exp())`
 /// Now: both reductions and the activation route through
 /// #311/#312-gated paths (`crate::forward::dot_f32_antirez` and
 /// `crate::forward::silu`).
 ///
 /// `DS4_MATVEC_F32_FIDELITY=1` switches dot to antirez form (
 /// env reuse). `DS4_SILU_FIDELITY=1` switches silu to branched
 /// `sigmoid_stable` ( env). The test asserts the gates each
 /// flip the result, and the combined-ON path bit-matches a
 /// hand-stamped antirez-spec.
 #[test]
 fn shared_expert_dot_silu_antirez_fidelity_gates_switch_implementation() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());

 // d_embd = 256 → exercises the 8-wide FMA partial loop in
 // `dot_f32_antirez`; sd = 2 rows so the test is small but covers
 // both rows. Hand-stamped reference uses identical formulae.
 let mut p = dummy_params();
 p.d_embd = 256;
 let d_embd = p.d_embd as usize;
 let sd = 2u32;
 let sds = sd as usize;

 // Adversarial input: ffn_norm has mixed-magnitude entries so the
 // left-fold and FMA-pair reductions pick up distinct ULP cascades.
 // The g values per row must straddle 0 so silu's negative branch
 // fires and the stable-form rewrite differs from the naive
 // positive-branch expression.
 let mut ffn = vec![0.0f32; d_embd];
 for i in 0..d_embd {
 ffn[i] = if i & 1 == 0 { 1.0 } else { -1e-3 };
 }
 let mut seed = 0xCAFEF00DDEADBEEFu64;
 let lcg = |s: &mut u64| -> f32 {
 *s = s
 .wrapping_mul(6364136223846793005)
 .wrapping_add(1442695040888963407);
 (((*s >> 33) as u32) as f32) / (u32::MAX as f32) * 2.0 - 1.0
 };
 let mut wg = vec![0.0f32; sds * d_embd];
 let mut wu = vec![0.0f32; sds * d_embd];
 // row 0: gate ≈ -0.5·ffn (negative g → silu negative branch fires)
 // row 1: gate ≈ +0.5·ffn (positive g)
 // both rows: up similar magnitude
 for i in 0..d_embd {
 wg[i] = -0.5 * ffn[i] + lcg(&mut seed) * 1e-3;
 wg[d_embd + i] = 0.5 * ffn[i] + lcg(&mut seed) * 1e-3;
 wu[i] = 0.3 * ffn[i] + lcg(&mut seed) * 1e-3;
 wu[d_embd + i] = 0.4 * ffn[i] + lcg(&mut seed) * 1e-3;
 }

 // Hand-stamp left-fold + naive-silu (ungated path).
 let dot_lf =
 |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| x * y).sum() };
 let silu_naive = |x: f32| -> f32 { x / (1.0 + (-x).exp()) };
 let ref_default = |w_gate: &[f32], w_up: &[f32]| -> Vec<f32> {
 let mut out = vec![0.0f32; sds];
 for i in 0..sds {
 let rg = &w_gate[i * d_embd..(i + 1) * d_embd];
 let ru = &w_up[i * d_embd..(i + 1) * d_embd];
 let g = dot_lf(rg, &ffn);
 let u = dot_lf(ru, &ffn);
 out[i] = silu_naive(g) * u;
 }
 out
 };

 // Hand-stamp antirez-spec: FMA-pair dot + branched silu.
 let dot_ar = |a: &[f32], b: &[f32]| -> f32 { crate::forward::dot_f32_antirez(a, b) };
 let silu_ar = |x: f32| -> f32 {
 // sigmoid_stable: branch on sign so the intermediate exp() stays ≤1.
 let s = if x >= 0.0 {
 let e = (-x).exp();
 1.0 / (1.0 + e)
 } else {
 let e = x.exp();
 e / (1.0 + e)
 };
 x * s
 };
 let ref_antirez = |w_gate: &[f32], w_up: &[f32]| -> Vec<f32> {
 let mut out = vec![0.0f32; sds];
 for i in 0..sds {
 let rg = &w_gate[i * d_embd..(i + 1) * d_embd];
 let ru = &w_up[i * d_embd..(i + 1) * d_embd];
 let g = dot_ar(rg, &ffn);
 let u = dot_ar(ru, &ffn);
 out[i] = silu_ar(g) * u;
 }
 out
 };

 let out_default = ref_default(&wg, &wu);
 let out_antirez = ref_antirez(&wg, &wu);

 // Discrimination check: the two hand-stamped paths must produce
 // bit-different outputs on at least one slot — otherwise the test
 // is trivial.
 let any_diff = out_default
 .iter()
 .zip(out_antirez.iter())
 .any(|(a, b)| a.to_bits() != b.to_bits());
 assert!(
 any_diff,
 "left-fold+naive-silu and antirez dot+branched-silu produced \
 bit-identical outputs — pick a more adversarial input. \
 default={out_default:?} antirez={out_antirez:?}"
 );

 let disp = CpuAttentionDispatcher;

 // Gate OFF (both envs unset): shared_expert must bit-match default.
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 std::env::remove_var("DS4_SILU_FIDELITY");
 let out_off = disp.shared_expert(&p, &ffn, &wg, &wu, sd, 0.0);
 for i in 0..sds {
 assert_eq!(
 out_off[i].to_bits(),
 out_default[i].to_bits(),
 "gate OFF: slot {i} ours={off} ref={r}",
 off = out_off[i],
 r = out_default[i]
 );
 }

 // Gate ON (both envs set): shared_expert must bit-match antirez.
 std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
 std::env::set_var("DS4_SILU_FIDELITY", "1");
 let out_on = disp.shared_expert(&p, &ffn, &wg, &wu, sd, 0.0);
 for i in 0..sds {
 assert_eq!(
 out_on[i].to_bits(),
 out_antirez[i].to_bits(),
 "gate ON: slot {i} ours={on} ref={r}",
 on = out_on[i],
 r = out_antirez[i]
 );
 }

 // Cleanup.
 std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
 std::env::remove_var("DS4_SILU_FIDELITY");
 }

 #[test]
 fn shared_down_writes_full_hc_residual() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let sd = 3;
 let mid = vec![0.5f32; sd];
 let wd = vec![0.25f32; p.d_embd as usize * sd];
 let routed = vec![0.0f32; p.d_embd as usize];
 let after_attn = vec![1.0f32; p.hc_dim()];
 let split = vec![0.5f32; p.n_hc as usize];
 // Identity comb (dst==src → 1.0 else 0.0) so the residual carries
 // through verbatim and the legacy assertion (v > 1.0) holds.
 let n_hc = p.n_hc as usize;
 let mut comb = vec![0.0f32; n_hc * n_hc];
 for h in 0..n_hc {
 comb[h + h * n_hc] = 1.0;
 }
 let after = d.shared_down_hc_expand_add(&p, &mid, &wd, &routed, &after_attn, &split, &comb);
 assert_eq!(after.len(), p.hc_dim());
 for &v in &after {
 assert!(v > 1.0);
 }
 }

 /// Build a tiny GGUF (single layer) carrying all 10 DS4-attention roles
 /// at F32 (so `dequant_f32_simple` round-trips with a known bit pattern).
 /// Tensor shapes mirror `dummy_params()`.
 fn write_attn_gguf(path: &std::path::Path, _with_hc_aux: bool) {
 let p = dummy_params();
 let d = p.d_embd as usize;
 let n_hc = p.n_hc as usize;
 let mix_hc = 2 * n_hc + n_hc * n_hc;
 let hc_dim = n_hc * d;
 let n_head = p.n_head as usize;
 let head_dim = p.head_dim as usize;
 let n_lora_q = p.n_lora_q as usize;
 let n_lora_kv = p.n_lora_kv as usize;
 let n_rot = p.n_rot as usize;
 let _ = n_rot;
 let q_dim = n_head * head_dim;
 let rank = 3usize;
 let shared_dim = 5usize;

 // (role-name suffix, dims, fill-value)
 let tensors: Vec<(&str, Vec<u64>, f32)> = vec![
 ("attn_norm.weight", vec![d as u64], 1.0),
 ("ffn_norm.weight", vec![d as u64], 1.05),
 ("attn_q_a_norm.weight", vec![n_lora_q as u64], 1.1),
 ("attn_kv_a_norm.weight", vec![n_lora_kv as u64], 1.2),
 ("attn_q_a.weight", vec![d as u64, n_lora_q as u64], 0.11),
 ("attn_q_b.weight", vec![n_lora_q as u64, q_dim as u64], 0.12),
 ("attn_kv.weight", vec![d as u64, n_lora_kv as u64], 0.13),
 (
 "attn_output_a.weight",
 vec![q_dim as u64, rank as u64],
 0.10,
 ),
 ("attn_output_b.weight", vec![rank as u64, d as u64], 0.20),
 ("attn_sinks.weight", vec![n_head as u64], -1.0),
 (
 "ffn_gate_shexp.weight",
 vec![d as u64, shared_dim as u64],
 0.30,
 ),
 (
 "ffn_up_shexp.weight",
 vec![d as u64, shared_dim as u64],
 0.40,
 ),
 (
 "ffn_down_shexp.weight",
 vec![shared_dim as u64, d as u64],
 0.50,
 ),
 ("hc_attn_fn.weight", vec![hc_dim as u64, mix_hc as u64], 0.7),
 ("hc_attn_scale.weight", vec![3u64], 0.06),
 ("hc_attn_base.weight", vec![mix_hc as u64], 0.05),
 ("hc_ffn_fn.weight", vec![hc_dim as u64, mix_hc as u64], 0.71),
 ("hc_ffn_scale.weight", vec![3u64], 0.061),
 ("hc_ffn_base.weight", vec![mix_hc as u64], 0.051),
 ];

 let mut buf = Vec::new();
 buf.extend_from_slice(&0x46554747u32.to_le_bytes());
 buf.extend_from_slice(&3u32.to_le_bytes());
 buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
 buf.extend_from_slice(&3u64.to_le_bytes());

 let write_str = |buf: &mut Vec<u8>, s: &str| {
 buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
 buf.extend_from_slice(s.as_bytes());
 };
 write_str(&mut buf, "general.architecture");
 buf.extend_from_slice(&8u32.to_le_bytes());
 write_str(&mut buf, "deepseek4");
 write_str(&mut buf, "deepseek4.block_count");
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&1u32.to_le_bytes());
 write_str(&mut buf, "deepseek4.embedding_length");
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&(p.d_embd).to_le_bytes());

 let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
 let mut running = 0u64;
 for (_, dims, _) in &tensors {
 offsets.push(running);
 let n_elem: u64 = dims.iter().product();
 running += n_elem * 4;
 }

 for ((suffix, dims, _), &off) in tensors.iter().zip(offsets.iter()) {
 let name = format!("blk.0.{suffix}");
 write_str(&mut buf, &name);
 buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
 for d in dims {
 buf.extend_from_slice(&d.to_le_bytes());
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
 }

 #[test]
 fn from_layer_view_populates_required_roles() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let tmp = std::env::temp_dir().join("ds4_attn_from_layer_view.gguf");
 write_attn_gguf(&tmp, /*with_hc_aux=*/ true);
 let views = crate::layer_view::LayerViews::open(&tmp, 1).expect("open");
 let lv = views.layer(0);
 let p = dummy_params();
 let w = AttnLayerWeights::from_layer_view(&views, lv, p, 5, 7.0, false).expect("build");

 // Bit pattern from the fill values above (F32 round-trip is exact).
 assert!(w.hc_norm_gamma.iter().all(|&v| v == 1.0));
 assert!(w.hc_ffn_norm_gamma.iter().all(|&v| v == 1.05));
 assert!(w.qkv_gamma_q.iter().all(|&v| v == 1.1));
 assert!(w.qkv_gamma_kv.iter().all(|&v| v == 1.2));
 assert!(w.attn_sinks.iter().all(|&v| v == -1.0));
 assert!(w.w_o_a.iter().all(|&v| v == 0.10));
 assert!(w.w_o_b.iter().all(|&v| v == 0.20));
 assert!(w.w_shared_gate.iter().all(|&v| v == 0.30));
 assert!(w.w_shared_up.iter().all(|&v| v == 0.40));
 assert!(w.w_shared_down.iter().all(|&v| v == 0.50));
 assert!(w.hc_attn_fn.iter().all(|&v| v == 0.7));
 assert!(w.hc_attn_scale.iter().all(|&v| v == 0.06));
 assert!(w.hc_attn_base.iter().all(|&v| v == 0.05));
 assert!(w.hc_ffn_fn.iter().all(|&v| v == 0.71));
 assert!(w.hc_ffn_scale.iter().all(|&v| (v - 0.061).abs() < 1e-6));
 assert!(w.hc_ffn_base.iter().all(|&v| (v - 0.051).abs() < 1e-6));
 assert_eq!(w.shared_dim, 5);
 assert_eq!(w.shared_clamp, 7.0);
 assert_eq!(w.params.layer_idx, 0);

 std::fs::remove_file(&tmp).ok();
 }

 /// Write a tiny GGUF with the metadata keys `LayerParams::from_gguf`
 /// reads. Returns the d_model that was written.
 fn write_metadata_gguf(path: &std::path::Path, ds4: bool) -> u32 {
 let arch = if ds4 { "deepseek4" } else { "deepseek2" };
 let mut buf = Vec::new();
 buf.extend_from_slice(&0x46554747u32.to_le_bytes());
 buf.extend_from_slice(&3u32.to_le_bytes());
 buf.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
 // 7 keys: architecture, head_count, key_length, value_length,
 // rope.dimension_count, rope.freq_base, rope.scaling.factor.
 buf.extend_from_slice(&7u64.to_le_bytes());

 let write_str = |buf: &mut Vec<u8>, s: &str| {
 buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
 buf.extend_from_slice(s.as_bytes());
 };
 write_str(&mut buf, "general.architecture");
 buf.extend_from_slice(&8u32.to_le_bytes());
 write_str(&mut buf, arch);

 write_str(&mut buf, &format!("{arch}.attention.head_count"));
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&64u32.to_le_bytes());

 write_str(&mut buf, &format!("{arch}.attention.key_length"));
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&128u32.to_le_bytes());

 write_str(&mut buf, &format!("{arch}.attention.value_length"));
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&128u32.to_le_bytes());

 write_str(&mut buf, "deepseek4.rope.dimension_count");
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&64u32.to_le_bytes());

 write_str(&mut buf, "deepseek4.rope.freq_base");
 buf.extend_from_slice(&6u32.to_le_bytes()); // F32
 buf.extend_from_slice(&500000.0f32.to_le_bytes());

 write_str(&mut buf, "deepseek4.rope.scaling.factor");
 buf.extend_from_slice(&6u32.to_le_bytes()); // F32
 buf.extend_from_slice(&40.0f32.to_le_bytes()); // → freq_scale = 1/40

 while buf.len() % 32 != 0 {
 buf.push(0);
 }
 std::fs::write(path, &buf).unwrap();
 4096 // d_model
 }

 #[test]
 fn from_gguf_reads_ds4_keys_with_defaults() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let tmp = std::env::temp_dir().join("ds4_layer_params_from_gguf.gguf");
 let d_model = write_metadata_gguf(&tmp, /*ds4=*/ true);
 let g = crate::gguf::GgufFile::open(&tmp).expect("open gguf");
 let defaults = DefaultsDs4::ds4_v4_flash();
 let p = LayerParams::from_gguf(&g, d_model, 7, defaults).expect("from_gguf");
 assert_eq!(p.layer_idx, 7);
 assert_eq!(p.n_head, 64);
 assert_eq!(p.head_dim, 128);
 assert_eq!(p.n_rot, 64);
 assert_eq!(p.rope_freq_base, 500000.0);
 // scaling.factor = 40 → freq_scale = 1/40 = 0.025
 assert!((p.rope_freq_scale - 0.025).abs() < 1e-7);
 // In DS4 every HC slot is a full-width embedding copy, so
 // d_embd == d_model (not d_model / n_hc).
 assert_eq!(p.d_embd, d_model);
 assert_eq!(p.n_hc, defaults.n_hc);
 assert_eq!(p.n_lora_q, defaults.n_lora_q);
 assert_eq!(p.n_lora_kv, defaults.n_lora_kv);
 assert_eq!(p.compress_ratio, 1);
 std::fs::remove_file(&tmp).ok();
 }

 #[test]
 fn from_gguf_bails_on_missing_required_head_count() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let mut buf = Vec::new();
 buf.extend_from_slice(&0x46554747u32.to_le_bytes());
 buf.extend_from_slice(&3u32.to_le_bytes());
 buf.extend_from_slice(&0u64.to_le_bytes());
 buf.extend_from_slice(&0u64.to_le_bytes()); // 0 keys
 while buf.len() % 32 != 0 {
 buf.push(0);
 }
 let tmp = std::env::temp_dir().join("ds4_layer_params_empty.gguf");
 std::fs::write(&tmp, &buf).unwrap();
 let g = crate::gguf::GgufFile::open(&tmp).expect("open");
 let defaults = DefaultsDs4::ds4_v4_flash();
 let err = LayerParams::from_gguf(&g, 256, 0, defaults)
 .unwrap_err()
 .to_string();
 assert!(err.contains("head_count"), "err = {err}");
 std::fs::remove_file(&tmp).ok();
 }

 #[test]
 fn from_gguf_bails_on_k_v_length_mismatch() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let mut buf = Vec::new();
 buf.extend_from_slice(&0x46554747u32.to_le_bytes());
 buf.extend_from_slice(&3u32.to_le_bytes());
 buf.extend_from_slice(&0u64.to_le_bytes());
 buf.extend_from_slice(&4u64.to_le_bytes());

 let write_str = |buf: &mut Vec<u8>, s: &str| {
 buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
 buf.extend_from_slice(s.as_bytes());
 };
 write_str(&mut buf, "general.architecture");
 buf.extend_from_slice(&8u32.to_le_bytes());
 write_str(&mut buf, "deepseek4");
 write_str(&mut buf, "deepseek4.attention.head_count");
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&4u32.to_le_bytes());
 write_str(&mut buf, "deepseek4.attention.key_length");
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&128u32.to_le_bytes());
 write_str(&mut buf, "deepseek4.attention.value_length");
 buf.extend_from_slice(&4u32.to_le_bytes());
 buf.extend_from_slice(&256u32.to_le_bytes()); // mismatch with key_length

 while buf.len() % 32 != 0 {
 buf.push(0);
 }
 let tmp = std::env::temp_dir().join("ds4_layer_params_kv_mismatch.gguf");
 std::fs::write(&tmp, &buf).unwrap();
 let g = crate::gguf::GgufFile::open(&tmp).expect("open");
 let defaults = DefaultsDs4::ds4_v4_flash();
 let err = LayerParams::from_gguf(&g, 256, 0, defaults)
 .unwrap_err()
 .to_string();
 assert!(
 err.contains("key_length") && err.contains("value_length"),
 "err = {err}"
 );
 std::fs::remove_file(&tmp).ok();
 }

 #[test]
 fn from_layer_view_bails_on_missing_required_role() {
 // Build then drop a required role: just check that `require` propagates.
 let lv = crate::layer_view::LayerView::default();
 let views = crate::layer_view::LayerViews {
 data_offset: 0,
 bytes: crate::layer_view::ByteBuf::Owned(vec![]),
 per_layer: vec![lv.clone()],
 global: std::collections::BTreeMap::new(),
 };
 let p = dummy_params();
 let err = AttnLayerWeights::from_layer_view(&views, &lv, p, 5, 7.0, false)
 .unwrap_err()
 .to_string();
 // First required role checked is `hc_attn_fn`.
 assert!(err.contains("hc_attn_fn"), "err = {err}");
 }

 #[test]
 fn hc_collapse_norm_keeps_split_within_eps() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let n_hc = p.n_hc as usize;
 let mix_hc = 2 * n_hc + n_hc * n_hc;
 let hc_fn = vec![1.0f32; p.hc_dim() * mix_hc];
 let hc_scale = vec![1.0f32; 3];
 let hc_base = vec![0.0f32; mix_hc];
 let prev_hc = vec![1.0f32; p.hc_dim()];
 let (_cur, normed, split_out) = d.hc_collapse_norm(
 &p,
 HcKind::Attn,
 &hc_fn,
 &hc_scale,
 &hc_base,
 &prev_hc,
 None,
 );
 assert_eq!(normed.len(), p.d_embd as usize);
 assert_eq!(split_out.len(), mix_hc);
 // pre[0..n_hc] are sigmoid outputs (+ hc_eps), so strictly ≥ hc_eps.
 for &s in &split_out[..n_hc] {
 assert!(s >= p.hc_eps);
 }
 }

 #[test]
 fn hc_collapse_norm_weighted_sum_uses_pre_split_weights() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let mut p = dummy_params();
 p.hc_sinkhorn_iter = 0;
 let d = CpuAttentionDispatcher;
 let n_hc = p.n_hc as usize;
 let d_embd = p.d_embd as usize;
 let mix_hc = 2 * n_hc + n_hc * n_hc;
 let hc_fn = vec![0.0f32; p.hc_dim() * mix_hc];
 let hc_scale = vec![0.0f32; 3];
 let mut hc_base = vec![0.0f32; mix_hc];
 hc_base[0] = 0.0; // sigmoid -> 0.5
 hc_base[1] = (3.0f32).ln(); // sigmoid -> 0.75
 hc_base[n_hc] = -20.0; // post split must not affect collapse
 hc_base[n_hc + 1] = 20.0;

 let mut prev_hc = vec![0.0f32; p.hc_dim()];
 for e in 0..d_embd {
 prev_hc[e] = 10.0 + e as f32;
 prev_hc[d_embd + e] = 100.0 + 2.0 * e as f32;
 }

 let (cur, _normed, split_out) = d.hc_collapse_norm(
 &p,
 HcKind::Attn,
 &hc_fn,
 &hc_scale,
 &hc_base,
 &prev_hc,
 None,
 );

 let w0 = split_out[0];
 let w1 = split_out[1];
 assert!((w0 - (0.5 + p.hc_eps)).abs() < 1e-6, "w0={w0}");
 assert!((w1 - (0.75 + p.hc_eps)).abs() < 1e-6, "w1={w1}");
 for e in 0..d_embd {
 let want = w0 * prev_hc[e] + w1 * prev_hc[d_embd + e];
 assert!(
 (cur[e] - want).abs() < 1e-5,
 "e={e}: got {}, want {want}",
 cur[e]
 );
 }
 }

 /// Synthetic ratio-4 compressor: assert no emit at pos∈{0,1,2}, emit at
 /// pos=3 (the first `(pos+1) % 4 == 0` boundary). The pooled+normed+roped
 /// row must be finite and length-`head_dim`. This is the regression gate
 /// for without the compressor wired, our pipeline silently
 /// runs flash_attn over the wrong K-set starting at pos=3.
 #[test]
 fn compressor_ratio4_emits_first_row_at_pos3() {
 // Acquire ENV_LOCK because compressor_decode_one reads DS4_Q8_0_ACT;
 // a parallel test setting that var would perturb our f32-direct
 // expectations beyond tolerance.
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let p = LayerParams {
 layer_idx: 2, // even ≥2 → ratio-4 in real DS4
 d_embd: 4,
 n_hc: 2,
 n_head: 2,
 head_dim: 4,
 n_rot: 2,
 n_lora_q: 4,
 n_lora_kv: 4,
 hc_sinkhorn_iter: 1,
 hc_eps: 1e-6,
 rms_eps: 1e-6,
 rope_orig_ctx: 4096,
 rope_freq_base: 10000.0,
 rope_freq_scale: 1.0,
 rope_ext_factor: 0.0,
 rope_attn_factor: 1.0,
 compress_ratio: 4,
 n_out_group: 2,
 };
 let ratio = p.compress_ratio as usize;
 let head_dim = p.head_dim as usize;
 let coff = 2usize; // ratio == 4 → coff = 2
 let width = coff * head_dim; // = 8
 let rows = 2 * ratio; // ratio-4 state has 2*ratio rows
 let in_dim = p.d_embd as usize;

 // Tiny deterministic weights — all-0.1 / all-0.05 / all-0 so the
 // pooled row is non-degenerate but won't blow up.
 let w_kv = vec![0.1f32; in_dim * width];
 let w_gate = vec![0.05f32; in_dim * width];
 let w_ape = vec![0.0f32; width * ratio];
 let w_norm = vec![1.0f32; head_dim];
 let comp = CompressorInputs {
 w_kv: &w_kv,
 w_gate: &w_gate,
 // CPU trait path — always f32 (non-lean weights keep the dequant).
 w_kv_f16: None,
 w_gate_f16: None,
 w_ape: &w_ape,
 w_norm: &w_norm,
 head_dim: p.head_dim,
 compress_ratio: p.compress_ratio,
 };
 let mut state_kv = vec![0.0f32; rows * width];
 let mut state_score = vec![-1.0e9f32; rows * width];

 let d = CpuAttentionDispatcher;
 let x = vec![0.25f32; in_dim];

 for pos in 0..3u32 {
 let emit =
 compressor_decode_one(&d, &p, &comp, &x, &mut state_kv, &mut state_score, pos);
 assert!(emit.is_none(), "pos={pos} should not emit (ratio=4)");
 }
 let emit = compressor_decode_one(&d, &p, &comp, &x, &mut state_kv, &mut state_score, 3);
 let row = emit.expect("pos=3 must emit on ratio-4");
 assert_eq!(row.len(), head_dim);
 for (i, &v) in row.iter().enumerate() {
 assert!(v.is_finite(), "emitted comp row[{i}] = {v} is not finite");
 }
 }

 #[test]
 fn compressor_decode_one_uses_col_major_projection_and_j_major_ape() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let mut p = dummy_params();
 p.compress_ratio = 4;
 p.head_dim = 2;
 p.n_rot = 0;
 let ratio = p.compress_ratio as usize;
 let head_dim = p.head_dim as usize;
 let width = 2 * head_dim;
 let rows = 2 * ratio;
 let x = vec![2.0f32, -1.0, 0.5];
 let in_dim = x.len();

 let mut w_kv = vec![0.0f32; in_dim * width];
 for j in 0..width {
 for k in 0..in_dim {
 w_kv[j * in_dim + k] = 10.0 * j as f32 + (k as f32 + 1.0);
 }
 }
 let w_gate = vec![0.0f32; in_dim * width];
 let mut w_ape = vec![0.0f32; width * ratio];
 for j in 0..width {
 for r in 0..ratio {
 w_ape[j * ratio + r] = 100.0 * j as f32 + r as f32;
 }
 }
 let w_norm = vec![1.0f32; head_dim];
 let comp = CompressorInputs {
 w_kv: &w_kv,
 w_gate: &w_gate,
 // CPU trait path — always f32 (non-lean weights keep the dequant).
 w_kv_f16: None,
 w_gate_f16: None,
 w_ape: &w_ape,
 w_norm: &w_norm,
 head_dim: p.head_dim,
 compress_ratio: p.compress_ratio,
 };
 let mut state_kv = vec![0.0f32; rows * width];
 let mut state_score = vec![-1.0e9f32; rows * width];

 let d = CpuAttentionDispatcher;
 assert!(
 compressor_decode_one(&d, &p, &comp, &x, &mut state_kv, &mut state_score, 1).is_none()
 );

 let row = ratio + 1;
 for j in 0..width {
 let want_kv: f32 = (0..in_dim).map(|k| w_kv[j * in_dim + k] * x[k]).sum();
 let got_kv = state_kv[row * width + j];
 assert!(
 (got_kv - want_kv).abs() < 1e-6,
 "kv[{j}] got {got_kv}, want {want_kv}"
 );
 let want_score = w_ape[j * ratio + 1];
 let got_score = state_score[row * width + j];
 assert!(
 (got_score - want_score).abs() < 1e-6,
 "score[{j}] got {got_score}, want {want_score}"
 );
 }
 }

 #[test]
 fn compressor_ratio4_rotates_finished_window_into_first_half() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let mut p = dummy_params();
 p.compress_ratio = 4;
 p.head_dim = 2;
 p.n_rot = 0;
 let ratio = p.compress_ratio as usize;
 let head_dim = p.head_dim as usize;
 let width = 2 * head_dim;
 let rows = 2 * ratio;
 let in_dim = 1usize;

 let mut w_kv = vec![0.0f32; in_dim * width];
 for j in 0..width {
 w_kv[j] = (j + 1) as f32;
 }
 let w_gate = vec![0.0f32; in_dim * width];
 let w_ape = vec![0.0f32; width * ratio];
 let w_norm = vec![1.0f32; head_dim];
 let comp = CompressorInputs {
 w_kv: &w_kv,
 w_gate: &w_gate,
 // CPU trait path — always f32 (non-lean weights keep the dequant).
 w_kv_f16: None,
 w_gate_f16: None,
 w_ape: &w_ape,
 w_norm: &w_norm,
 head_dim: p.head_dim,
 compress_ratio: p.compress_ratio,
 };
 let mut state_kv = vec![0.0f32; rows * width];
 let mut state_score = vec![-1.0e9f32; rows * width];
 let d = CpuAttentionDispatcher;

 for pos in 0..4u32 {
 let x = vec![(pos + 1) as f32];
 let _ = compressor_decode_one(&d, &p, &comp, &x, &mut state_kv, &mut state_score, pos);
 }

 for r in 0..ratio {
 for j in 0..width {
 let want = (r + 1) as f32 * (j + 1) as f32;
 let got_first = state_kv[r * width + j];
 let got_second = state_kv[(ratio + r) * width + j];
 assert!(
 (got_first - want).abs() < 1e-6,
 "first row {r}, col {j}: got {got_first}, want {want}"
 );
 assert!(
 (got_second - want).abs() < 1e-6,
 "second row {r}, col {j}: got {got_second}, want {want}"
 );
 }
 }
 }

 #[test]
 fn compressor_pool_uses_independent_scores_per_dimension() {
 let head_dim = 2u32;
 let ratio = 2u32;
 let mut out = vec![0.0f32; head_dim as usize];
 let state_kv = vec![10.0f32, 100.0, 20.0, 200.0];
 let state_score = vec![20.0f32, -20.0, -20.0, 20.0];

 compressor_pool_decode_state(&mut out, &state_kv, &state_score, head_dim, ratio);

 assert!((out[0] - 10.0).abs() < 1e-4, "out0={}", out[0]);
 assert!((out[1] - 200.0).abs() < 1e-4, "out1={}", out[1]);
 }

 /// Dense ratio-0 path is a no-op: we never call the compressor for
 /// layers with `compress_ratio == 0`. Guard that the wiring in
 /// `decode_step_with_attn` respects the gate. (Pure call-site check.)
 #[test]
 fn compressor_ratio128_emits_only_every_128_steps() {
 let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
 let p = LayerParams {
 layer_idx: 3, // odd ≥3 → ratio-128 in real DS4
 d_embd: 4,
 n_hc: 2,
 n_head: 2,
 head_dim: 4,
 n_rot: 2,
 n_lora_q: 4,
 n_lora_kv: 4,
 hc_sinkhorn_iter: 1,
 hc_eps: 1e-6,
 rms_eps: 1e-6,
 rope_orig_ctx: 4096,
 rope_freq_base: 10000.0,
 rope_freq_scale: 1.0,
 rope_ext_factor: 0.0,
 rope_attn_factor: 1.0,
 compress_ratio: 128,
 n_out_group: 2,
 };
 let ratio = p.compress_ratio as usize;
 let head_dim = p.head_dim as usize;
 let coff = 1usize; // ratio != 4 → coff = 1
 let width = coff * head_dim;
 let rows = ratio;
 let in_dim = p.d_embd as usize;

 let w_kv = vec![0.01f32; in_dim * width];
 let w_gate = vec![0.005f32; in_dim * width];
 let w_ape = vec![0.0f32; width * ratio];
 let w_norm = vec![1.0f32; head_dim];
 let comp = CompressorInputs {
 w_kv: &w_kv,
 w_gate: &w_gate,
 // CPU trait path — always f32 (non-lean weights keep the dequant).
 w_kv_f16: None,
 w_gate_f16: None,
 w_ape: &w_ape,
 w_norm: &w_norm,
 head_dim: p.head_dim,
 compress_ratio: p.compress_ratio,
 };
 let mut state_kv = vec![0.0f32; rows * width];
 let mut state_score = vec![-1.0e9f32; rows * width];

 let d = CpuAttentionDispatcher;
 let x = vec![0.5f32; in_dim];

 // pos ∈ [0, 36] is the entire gate window (29 prefill + 8 decode).
 // No ratio-128 emit should fire — first emit is at pos=127.
 for pos in 0..=36u32 {
 let emit =
 compressor_decode_one(&d, &p, &comp, &x, &mut state_kv, &mut state_score, pos);
 assert!(
 emit.is_none(),
 "ratio-128 should not emit before pos=127 (got emit at pos={pos})"
 );
 }
 }

 // ---- AGGRESSIVE TESTS for hardening ----

 /// regression: `attn_output_proj` HC expand uses
 /// `hc_split_post` (slot 2 of split) NOT `hc_split_pre` (slot 0).
 /// Build inputs where post != pre and verify result depends on post.
 /// If buggy impl wires pre instead, the result will differ by the
 /// post/pre ratio.
 #[test]
 fn attn_output_proj_uses_post_weights_not_some_other_slot() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let q_dim = p.q_dim();
 let n_groups = p.n_out_group as usize;
 let group_dim = q_dim / n_groups;
 let n_lora_o = 1usize;
 let out_low_dim = n_groups * n_lora_o;
 let n_hc = p.n_hc as usize;
 let d_embd = p.d_embd as usize;

 // Make attn_out non-zero, identity comb.
 let heads = vec![1.0f32; q_dim];
 let w_a = vec![1.0f32; n_groups * n_lora_o * group_dim];
 let w_b = vec![1.0f32; d_embd * out_low_dim];
 let cur = vec![0.0f32; n_hc * d_embd]; // zero cur_hc → only post path matters
 // Distinct post values per slot.
 let split_post: Vec<f32> = (0..n_hc).map(|i| (i as f32) + 1.0).collect(); // [1.0, 2.0]
 let comb = vec![0.0f32; n_hc * n_hc]; // identity-zero, so cur_hc disappears (it's 0 anyway)
 let after = d.attn_output_proj(&p, &heads, &w_a, &w_b, &cur, &split_post, &comb);
 assert_eq!(after.len(), n_hc * d_embd);

 // attn_out[e] = sum of all w_a,w_b products. With all-1 weights:
 // attn_low[g*1+0] = group_dim (each entry is 1*1 summed group_dim times)
 // attn_out[e] = out_low_dim * group_dim = n_groups * 1 * group_dim = q_dim
 let attn_out_e = q_dim as f32;
 // after[dst, e] = post[dst] * attn_out[e].
 // dst=0 → post[0]=1 → 1*q_dim
 // dst=1 → post[1]=2 → 2*q_dim
 // Ratio after[1*d_embd..] / after[0..d_embd] MUST be 2.0 (not 1.0).
 for e in 0..d_embd {
 let v0 = after[0 * d_embd + e];
 let v1 = after[1 * d_embd + e];
 assert!((v0 - attn_out_e).abs() < 1e-3, "v0={v0}, want {attn_out_e}");
 assert!(
 (v1 - 2.0 * attn_out_e).abs() < 1e-3,
 "v1={v1}, want {}",
 2.0 * attn_out_e
 );
 }
 }

 /// regression: HC `comb` matrix is column-major in src
 /// (`comb[dst + src*n_hc]`). Asymmetric comb proves we don't accidentally
 /// transpose it.
 #[test]
 fn attn_output_proj_comb_is_column_major_in_src() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let q_dim = p.q_dim();
 let n_groups = p.n_out_group as usize;
 let group_dim = q_dim / n_groups;
 let n_lora_o = 1usize;
 let out_low_dim = n_groups * n_lora_o;
 let n_hc = p.n_hc as usize;
 let d_embd = p.d_embd as usize;

 // Zero attn_out path: heads=0 → attn_out=0 → only comb·cur_hc contributes.
 let heads = vec![0.0f32; q_dim];
 let w_a = vec![1.0f32; n_groups * n_lora_o * group_dim];
 let w_b = vec![1.0f32; d_embd * out_low_dim];
 let split_post = vec![0.0f32; n_hc]; // zero post path
 // cur_hc: slot 0 = 1.0, slot 1 = 10.0 (distinguishable)
 let mut cur = vec![0.0f32; n_hc * d_embd];
 for e in 0..d_embd {
 cur[0 * d_embd + e] = 1.0;
 cur[1 * d_embd + e] = 10.0;
 }
 // ASYMMETRIC comb: dst=0,src=0 = 1, dst=0,src=1 = 0
 // dst=1,src=0 = 100, dst=1,src=1 = 0
 // comb is indexed as comb[dst + src*n_hc]. So:
 // comb[0 + 0*2] = comb[0] = 1.0 → dst=0,src=0
 // comb[1 + 0*2] = comb[1] = 100.0 → dst=1,src=0
 // comb[0 + 1*2] = comb[2] = 0.0 → dst=0,src=1
 // comb[1 + 1*2] = comb[3] = 0.0 → dst=1,src=1
 let comb = vec![1.0f32, 100.0, 0.0, 0.0];

 let after = d.attn_output_proj(&p, &heads, &w_a, &w_b, &cur, &split_post, &comb);
 // after[dst=0, e] = comb[0]·cur[0,e] + comb[2]·cur[1,e] = 1·1 + 0·10 = 1.0
 // after[dst=1, e] = comb[1]·cur[0,e] + comb[3]·cur[1,e] = 100·1 + 0·10 = 100.0
 for e in 0..d_embd {
 let v0 = after[0 * d_embd + e];
 let v1 = after[1 * d_embd + e];
 assert!(
 (v0 - 1.0).abs() < 1e-5,
 "v0={v0}, want 1.0 (transposed comb would give 0)"
 );
 assert!(
 (v1 - 100.0).abs() < 1e-5,
 "v1={v1}, want 100.0 (transposed comb would give 0)"
 );
 }
 // If the indexer were buggy (`comb[src + dst*n_hc]` instead),
 // we'd get v0 = comb[0]·cur[0] + comb[1]·cur[1] = 1 + 1000 = 1001.0 ≠ 1.0
 // and v1 = comb[2]·cur[0] + comb[3]·cur[1] = 0 + 0 = 0.0 ≠ 100.0.
 // Either way the asserts would fire.
 }

 /// regression: comb identity (dst==src) MUST pass residual through verbatim.
 /// This is the "no-op" sanity check — a buggy comb wired as all-zero (the bug
 /// before #254) would zero the residual.
 #[test]
 fn attn_output_proj_identity_comb_passes_residual_through() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let q_dim = p.q_dim();
 let n_groups = p.n_out_group as usize;
 let group_dim = q_dim / n_groups;
 let n_lora_o = 1usize;
 let out_low_dim = n_groups * n_lora_o;
 let n_hc = p.n_hc as usize;
 let d_embd = p.d_embd as usize;

 let heads = vec![0.0f32; q_dim];
 let w_a = vec![1.0f32; n_groups * n_lora_o * group_dim];
 let w_b = vec![1.0f32; d_embd * out_low_dim];
 let split_post = vec![0.0f32; n_hc];

 let mut cur = vec![0.0f32; n_hc * d_embd];
 for s in 0..n_hc {
 for e in 0..d_embd {
 cur[s * d_embd + e] = (s as f32) * 10.0 + (e as f32);
 }
 }

 // Identity comb: comb[dst + src*n_hc] = 1 if dst==src else 0
 let mut comb = vec![0.0f32; n_hc * n_hc];
 for s in 0..n_hc {
 comb[s + s * n_hc] = 1.0;
 }

 let after = d.attn_output_proj(&p, &heads, &w_a, &w_b, &cur, &split_post, &comb);
 for s in 0..n_hc {
 for e in 0..d_embd {
 let want = cur[s * d_embd + e];
 let got = after[s * d_embd + e];
 assert!(
 (got - want).abs() < 1e-5,
 "s={s}, e={e}: got {got}, want {want}"
 );
 }
 }
 }

 /// regression: shared_expert must NOT clamp when the model does not
 /// ask for it (antirez `layer_shared_ffn_one` is plain silu*up). A buggy impl
 /// that clamps unconditionally would saturate large values.
 ///
 /// Updated for the model-declared clamp: the argument used to be an inert
 /// 7.0 that every consumer dropped, so the literal here proved nothing. It is
 /// now 0.0 — the value the production path passes for a model that declares
 /// no `deepseek4.swiglu_clamp_shexp` — which is what makes this a real guard
 /// against unconditional clamping. `shared_expert_applies_declared_clamp`
 /// covers the other direction.
 #[test]
 fn shared_expert_does_not_clamp_large_gate() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let sd = 1u32;
 let d_embd = p.d_embd as usize;
 // ffn=[1,1,1,1]; w_gate row = [50,0,0,0] → gate = 50
 // w_up row = [1,0,0,0] → up = 1
 // silu(50) ≈ 50 (saturates), out = 50*1 = 50
 // BUGGY clamp at ±7: silu(7) ≈ 6.993, out ≈ 6.993 (far from 50)
 let ffn = vec![1.0f32; d_embd];
 let mut wg = vec![0.0f32; d_embd];
 wg[0] = 50.0;
 let mut wu = vec![0.0f32; d_embd];
 wu[0] = 1.0;
 let out = d.shared_expert(&p, &ffn, &wg, &wu, sd, 0.0);
 assert!(
 out[0] > 49.0,
 "shared expert must not clamp; got {}",
 out[0]
 );
 }

 /// with no declared clamp, shared_expert MUST pass `up` through
 /// unchanged (no clamp on up either). See the note on
 /// `shared_expert_does_not_clamp_large_gate` about the 7.0 -> 0.0 change.
 #[test]
 fn shared_expert_no_clamp_on_up_either() {
 // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
 // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
 // so the gate cannot change underneath us, while staying parallel with peers.
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let sd = 1u32;
 let d_embd = p.d_embd as usize;
 // gate = 1, up = 100 (would be clamped by buggy impl)
 // silu(1) ≈ 0.731, out = 0.731 * 100 = 73.1
 // Buggy ±7 clamp on up: 0.731 * 7 = 5.12
 let ffn = vec![1.0f32; d_embd];
 let mut wg = vec![0.0f32; d_embd];
 wg[0] = 1.0;
 let mut wu = vec![0.0f32; d_embd];
 wu[0] = 100.0;
 let out = d.shared_expert(&p, &ffn, &wg, &wu, sd, 0.0);
 let want = crate::forward::silu(1.0) * 100.0;
 assert!(
 (out[0] - want).abs() < 0.1,
 "shared_expert up must not clamp; got {}, want ≈{}",
 out[0],
 want
 );
 }

 /// The other direction of the pair: when the model DOES declare a
 /// clamp (`deepseek4.swiglu_clamp_shexp`, = 10.0 in the DS4-Flash MXFP4
 /// release), `shared_expert` must apply it — `min(gate, c)` and
 /// `clamp(up, -c, c)`, matching `kernel_swiglu_f32` and antirez's own
 /// "DS4 clamps the shared expert with the same swiglu_limit used by routed
 /// experts". This is the assertion that would have failed for the entire
 /// period the parameter was accepted as `_clamp` and dropped.
 #[test]
 fn shared_expert_applies_declared_clamp() {
 let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
 let p = dummy_params();
 let d = CpuAttentionDispatcher;
 let sd = 1u32;
 let d_embd = p.d_embd as usize;
 // gate = 50, up = 100, clamp = 10 -> silu(min(50,10)) * clamp(100,-10,10)
 let ffn = vec![1.0f32; d_embd];
 let mut wg = vec![0.0f32; d_embd];
 wg[0] = 50.0;
 let mut wu = vec![0.0f32; d_embd];
 wu[0] = 100.0;
 let out = d.shared_expert(&p, &ffn, &wg, &wu, sd, 10.0);
 let want = crate::forward::silu(10.0) * 10.0;
 assert!(
 (out[0] - want).abs() < 0.01,
 "declared clamp must be applied; got {}, want ~{}",
 out[0],
 want
 );
 }

 /// f16 round-trip: ties-to-even at half-ULP boundary. The CRITICAL boundary
 /// at 1.0 + 2^-11 must round DOWN to 1.0 (NOT up to 1.0 + 2^-10).
 /// A buggy round-half-up would give 1.0 + 2^-10.
 #[test]
 fn f16_round_trip_ties_to_even_not_half_up() {
 // 1.0 + 2^-11 is EXACTLY between 1.0 (even mantissa) and 1.0+2^-10
 // (odd mantissa). Banker's rounding picks 1.0 (even). Round-half-up
 // would pick 1.0+2^-10. This test discriminates them.
 let half_ulp = 1.0_f32 + 1.0 / 2048.0;
 let y = f16_round_trip_f32(half_ulp);
 assert_eq!(y, 1.0, "ties-to-even: 1.0+2^-11 → 1.0 (not 1.0+2^-10)");
 // Symmetric case: 1.0 + 3·2^-11 lies between 1.0+2^-10 (odd) and
 // 1.0+2·2^-10 (even). Banker's rounding picks the even one (i.e.
 // 1.0+2·2^-10 = 1.001953125). Round-half-up would pick 1.0+2^-10.
 let three_half_ulp = 1.0_f32 + 3.0 / 2048.0;
 let y2 = f16_round_trip_f32(three_half_ulp);
 let want_up = 1.0_f32 + 2.0 / 1024.0;
 assert_eq!(y2, want_up, "ties-to-even (odd→even): expect {want_up}");
 // Just below the half-ulp: rounds down to 1.0.
 let below = 1.0_f32 + 0.4 / 2048.0;
 assert_eq!(f16_round_trip_f32(below), 1.0);
 }

 /// f16 round-trip: sign preservation under tiny magnitudes (subnormal).
 /// Round-trip of -1e-10 must return ±0 (subnormal underflow), not NaN.
 #[test]
 fn f16_round_trip_subnormal_underflow_no_nan() {
 let y = f16_round_trip_f32(-1e-10);
 assert!(y.is_finite(), "got {}", y);
 assert_eq!(y.abs(), 0.0);
 }

 /// rope_tail at θ=π/2 (a specific rotation amount) — distinguishes from
 /// identity / wrong-direction rotation.
 /// With n_rot=2, base=10000, scale=1, ext_factor=0, attn_factor=1:
 /// freq[0] = 1/10000^(0/2) = 1.0; θ = pos · 1.0
 /// For θ = π/2 ≈ 1.5708 we need pos ≈ 1.5708. Use pos that gives a
 /// known θ ≠ multiple of π (so cos and sin both differ from each other).
 #[test]
 fn rope_tail_rotates_correct_direction() {
 let mut p = dummy_params();
 p.n_rot = 2;
 let d = CpuAttentionDispatcher;
 // pos=1: θ = 1.0; (cos 1, sin 1) ≈ (0.5403, 0.8415).
 // (1, 0) → (cos θ, sin θ). The sin component must be POSITIVE (not negative).
 let mut x = vec![1.0f32, 0.0];
 d.rope_tail(&p, &mut x, 1, false);
 assert!(
 x[1] > 0.5 && x[1] < 0.9,
 "x[1]={}, expected positive ~0.84 (rotation direction)",
 x[1]
 );
 // Reverse direction (sin negative) would mean rotation is clockwise, which
 // is the wrong RoPE convention.
 assert!(x[0] > 0.4 && x[0] < 0.7, "x[0]={}, expected ~0.54", x[0]);
 }
}
