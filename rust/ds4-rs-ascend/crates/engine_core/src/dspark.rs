// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! DSpark speculative-decoding drafter for DeepSeek-V4-Flash.
//!
//! Port of DeepSeek's DSpark ("Confidence-Scheduled Speculative Decoding with
//! Semi-Autoregressive Generation", DeepSpec, MIT). DSpark is a
//! *drafter* for spec decoding; the target (the accelerator host crate) verifies and accepts — so
//! draft quality (and our future FP8/FP4→q8 quantization of these weights) affects
//! only the ACCEPTANCE RATE, never the output. The correctness gate is byte-identity
//! to non-spec greedy.
//!
//! ## Names + architecture — PINNED FROM THE REAL CHECKPOINT
//! (`deepseek-ai/DeepSeek-V4-Flash-DSpark`, 72,317-tensor safetensors index +
//! `inference/model.py`, see the design notes).
//!
//! The drafter is **3 MTP stages `mtp.{0,1,2}.*`, each a FULL DeepSeek-V4 block**:
//! - MLA attention `attn.{wq_a,wq_b,wkv,wo_a,wo_b}` (FP8 blockwise: `.weight`+`.scale`),
//!   `attn.q_norm`, `attn.kv_norm`, `attn.attn_sink`;
//! - 256-expert MoE `ffn.experts.{0..255}.{w1,w2,w3}` + `ffn.gate` + `ffn.shared_experts`;
//! - norms `attn_norm`, `ffn_norm`;
//! - Hyper-Connections (`hc_mult=4`) `hc_{attn,ffn}_{base,fn,scale}`.
//!
//! Stage-specific extras:
//! - `mtp.0`: `main_proj` (W_c, `Linear(3·dim → dim)` ctx projection) + `main_norm`.
//! - `mtp.2` (last stage): `norm`, `markov_head.markov_w{1,2}`, `confidence_head.proj`,
//!   `hc_head_{base,fn,scale}`.
//! - **Shared from the BASE model, NOT in the `mtp.*` namespace**: `embed.weight`,
//!   `head.weight`. The loader deliberately keeps these OUT — they come from the base
//!   GGUF (the target model already loads them).
//!
//! ### Status
//! The **loader (`DsparkWeights`/`DsparkLayer`/`from_gguf`) + consts + `validate()` are
//! now CORRECTED to the real scheme** (no longer superseded). The DSpark-specific MATH
//! — `context_inject` (main_proj/main_norm), `markov_sample_block`, `confidence_block`,
//! `confident_prefix_len` — is confirmed against the paper + `inference/model.py` and is
//! kept verbatim.
//!
//! **REMAINING TODO**: the full draft *block forward*. Each `mtp.N` is the SAME
//! DeepSeek-V4 block the accelerator host crate already runs for the main model (MLA + 256-expert MoE +
//! Hyper-Connections + sparse_attn + rope θ=10000 + attn_sink), so Phase 2 is REUSE of
//! the existing `the accelerator host crate`/`CpuDispatcher` block forward driven on the `mtp.*` weights
//! (+ HC-expand, `DSparkAttention` main_kv injection, `hc_head` collapse). The vanilla
//! q/k/v/o + SwiGLU CPU `backbone_forward` scaffold that used to live here was
//! architecturally WRONG and has been REMOVED — do not reintroduce it; wire the real
//! block instead. See `forward_spec` in the design notes.

#![allow(clippy::too_many_arguments)]

use crate::dispatch::{CpuDispatcher, KernelDispatcher};
use crate::gguf::{GgufFile, TensorInfo};
use anyhow::{anyhow, bail, Result};

/// V4-Flash DSpark constants (from `config.json` / `inference/config.json`).
pub mod consts {
    pub const BLOCK_SIZE: usize = 5; // γ — dspark_block_size
    pub const MARKOV_RANK: usize = 256; // r — dspark_markov_rank
    pub const NOISE_TOKEN_ID: u32 = 128799; // dspark_noise_token_id (the "mask")
    pub const TARGET_LAYER_IDS: [usize; 3] = [40, 41, 42]; // dspark_target_layer_ids
    pub const N_MTP_LAYERS: usize = 3; // n_mtp_layers — the 3 `mtp.{0,1,2}` stages
    pub const N_EXPERTS: usize = 256; // MoE experts per stage (ffn.experts.0..255)
    pub const HC_MULT: usize = 4; // hyper-connection residual multiplicity
    pub const HEAD_DIM: usize = 512; // MLA per-head dim
    pub const N_HEADS: usize = 64; // query heads
    pub const N_KV_HEADS: usize = 1; // kv heads (MLA latent)
    pub const ROPE_THETA: f32 = 10000.0; // rope_theta
    pub const N_EMBD: usize = 4096; // base hidden_size (dim)
    pub const N_VOCAB: usize = 129280; // base vocab_size
}

// ───────────────────────── weights (loader) ─────────────────────────

/// A located tensor in the DSpark GGUF (name + type + absolute mmap offset).
/// Mirrors [`crate::mtp::MtpTensor`]; dequantize on demand.
#[derive(Clone)]
pub struct DsparkTensor {
    pub info: TensorInfo,
    pub abs_offset: u64,
}

impl DsparkTensor {
    fn new(info: TensorInfo, tensor_data_offset: u64) -> Self {
        let abs_offset = tensor_data_offset + info.offset;
        Self { info, abs_offset }
    }
    pub fn n_elements(&self) -> u64 {
        self.info.dims.iter().product()
    }
}

/// An FP8-blockwise-quantized weight: the `.weight` tensor + its optional `.scale`
/// sibling (per-block dequant scales). `scale` is `None` for unquantized / synthetic
/// GGUFs that omit it.
#[derive(Clone)]
pub struct QTensor {
    pub weight: DsparkTensor,
    pub scale: Option<DsparkTensor>,
}

/// One MoE expert's three projections (`w1`/`w3` = gate/up, `w2` = down).
#[derive(Clone)]
pub struct ExpertTensors {
    pub w1: QTensor,
    pub w2: QTensor,
    pub w3: QTensor,
}

/// One DSpark MTP stage (`mtp.N`) — a full DeepSeek-V4 block.
#[derive(Clone)]
pub struct DsparkLayer {
    // ── MLA attention ──
    pub wq_a: QTensor,
    pub wq_b: QTensor,
    pub wkv: QTensor,
    pub wo_a: QTensor,
    pub wo_b: QTensor,
    pub q_norm: DsparkTensor,
    pub kv_norm: DsparkTensor,
    pub attn_sink: DsparkTensor,
    // ── norms ──
    pub attn_norm: DsparkTensor,
    pub ffn_norm: DsparkTensor,
    // ── MoE FFN ──
    pub gate_w: DsparkTensor,
    pub gate_b: DsparkTensor,
    pub experts: Vec<ExpertTensors>, // N_EXPERTS (256)
    pub shared_experts: ExpertTensors,
    // ── Hyper-Connections (hc_mult=4) ──
    pub hc_attn_base: DsparkTensor,
    pub hc_attn_fn: DsparkTensor,
    pub hc_attn_scale: DsparkTensor,
    pub hc_ffn_base: DsparkTensor,
    pub hc_ffn_fn: DsparkTensor,
    pub hc_ffn_scale: DsparkTensor,
}

/// All DSpark-module weights. `embed.weight`/`head.weight` are shared from the base
/// model and supplied separately, NOT duplicated here.
#[derive(Clone)]
pub struct DsparkWeights {
    pub layers: Vec<DsparkLayer>, // N_MTP_LAYERS (3 stages)
    // ── mtp.0 only: context projection ──
    pub main_proj: QTensor,      // W_c: Linear(3·dim → dim) [dim, 3·dim]
    pub main_norm: DsparkTensor, // RMSNorm on H_ctx [dim]
    // ── mtp.2 only (last stage) ──
    pub norm: DsparkTensor,            // final RMSNorm [dim]
    pub markov_w1: DsparkTensor,       // [V, r] embedding lookup
    pub markov_w2: DsparkTensor,       // [V, r] logit projection
    pub confidence_proj: DsparkTensor, // (dim+r) → 1 accept-rate predictor (fp32)
    pub hc_head_base: DsparkTensor,
    pub hc_head_fn: DsparkTensor,
    pub hc_head_scale: DsparkTensor,
}

/// Every tensor name the loader resolves, in load order (weights + hc params; the
/// optional `.scale` siblings are not listed). Source of truth shared with the loader
/// name test; matches the real `mtp.{0,1,2}.*` checkpoint scheme.
pub fn dspark_tensor_names() -> Vec<String> {
    let mut names = Vec::new();
    for i in 0..consts::N_MTP_LAYERS {
        let p = format!("mtp.{i}");
        names.push(format!("{p}.attn.wq_a.weight"));
        names.push(format!("{p}.attn.wq_b.weight"));
        names.push(format!("{p}.attn.wkv.weight"));
        names.push(format!("{p}.attn.wo_a.weight"));
        names.push(format!("{p}.attn.wo_b.weight"));
        names.push(format!("{p}.attn.q_norm.weight"));
        names.push(format!("{p}.attn.kv_norm.weight"));
        names.push(format!("{p}.attn.attn_sink"));
        names.push(format!("{p}.attn_norm.weight"));
        names.push(format!("{p}.ffn_norm.weight"));
        names.push(format!("{p}.ffn.gate.weight"));
        names.push(format!("{p}.ffn.gate.bias"));
        for e in 0..consts::N_EXPERTS {
            names.push(format!("{p}.ffn.experts.{e}.w1.weight"));
            names.push(format!("{p}.ffn.experts.{e}.w2.weight"));
            names.push(format!("{p}.ffn.experts.{e}.w3.weight"));
        }
        names.push(format!("{p}.ffn.shared_experts.w1.weight"));
        names.push(format!("{p}.ffn.shared_experts.w2.weight"));
        names.push(format!("{p}.ffn.shared_experts.w3.weight"));
        for hc in [
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
        ] {
            names.push(format!("{p}.{hc}"));
        }
    }
    // mtp.0 extras
    names.push("mtp.0.main_proj.weight".into());
    names.push("mtp.0.main_norm.weight".into());
    // mtp.2 extras (last stage)
    names.push("mtp.2.norm.weight".into());
    names.push("mtp.2.markov_head.markov_w1.weight".into());
    names.push("mtp.2.markov_head.markov_w2.weight".into());
    names.push("mtp.2.confidence_head.proj.weight".into());
    names.push("mtp.2.hc_head_base".into());
    names.push("mtp.2.hc_head_fn".into());
    names.push("mtp.2.hc_head_scale".into());
    names
}

impl DsparkWeights {
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path.as_ref())?)
    }

    /// Locate every DSpark tensor by its real `mtp.{0,1,2}.*` name. `find` tolerates an
    /// optional `dspark.` / `model.dspark.` GGUF-converter prefix on top of the name.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let off = gguf.tensor_data_offset;
        let find = |suffix: &str| -> Result<DsparkTensor> {
            gguf.tensors
                .iter()
                .find(|t| {
                    t.name == suffix
                        || t.name == format!("dspark.{suffix}")
                        || t.name == format!("model.dspark.{suffix}")
                })
                .cloned()
                .map(|info| DsparkTensor::new(info, off))
                .ok_or_else(|| anyhow!("DSpark GGUF: missing tensor `{suffix}`"))
        };
        // FP8-blockwise weight: required `.weight` + optional `.scale` sibling.
        let find_q = |prefix: &str| -> Result<QTensor> {
            Ok(QTensor {
                weight: find(&format!("{prefix}.weight"))?,
                scale: find(&format!("{prefix}.scale")).ok(),
            })
        };
        let find_expert = |p: &str| -> Result<ExpertTensors> {
            Ok(ExpertTensors {
                w1: find_q(&format!("{p}.w1"))?,
                w2: find_q(&format!("{p}.w2"))?,
                w3: find_q(&format!("{p}.w3"))?,
            })
        };

        let mut layers = Vec::with_capacity(consts::N_MTP_LAYERS);
        for i in 0..consts::N_MTP_LAYERS {
            let p = format!("mtp.{i}");
            let mut experts = Vec::with_capacity(consts::N_EXPERTS);
            for e in 0..consts::N_EXPERTS {
                experts.push(find_expert(&format!("{p}.ffn.experts.{e}"))?);
            }
            layers.push(DsparkLayer {
                wq_a: find_q(&format!("{p}.attn.wq_a"))?,
                wq_b: find_q(&format!("{p}.attn.wq_b"))?,
                wkv: find_q(&format!("{p}.attn.wkv"))?,
                wo_a: find_q(&format!("{p}.attn.wo_a"))?,
                wo_b: find_q(&format!("{p}.attn.wo_b"))?,
                q_norm: find(&format!("{p}.attn.q_norm.weight"))?,
                kv_norm: find(&format!("{p}.attn.kv_norm.weight"))?,
                attn_sink: find(&format!("{p}.attn.attn_sink"))?,
                attn_norm: find(&format!("{p}.attn_norm.weight"))?,
                ffn_norm: find(&format!("{p}.ffn_norm.weight"))?,
                gate_w: find(&format!("{p}.ffn.gate.weight"))?,
                gate_b: find(&format!("{p}.ffn.gate.bias"))?,
                experts,
                shared_experts: find_expert(&format!("{p}.ffn.shared_experts"))?,
                hc_attn_base: find(&format!("{p}.hc_attn_base"))?,
                hc_attn_fn: find(&format!("{p}.hc_attn_fn"))?,
                hc_attn_scale: find(&format!("{p}.hc_attn_scale"))?,
                hc_ffn_base: find(&format!("{p}.hc_ffn_base"))?,
                hc_ffn_fn: find(&format!("{p}.hc_ffn_fn"))?,
                hc_ffn_scale: find(&format!("{p}.hc_ffn_scale"))?,
            });
        }
        Ok(Self {
            // mtp.0 context projection
            main_proj: find_q("mtp.0.main_proj")?,
            main_norm: find("mtp.0.main_norm.weight")?,
            // mtp.2 head extras
            norm: find("mtp.2.norm.weight")?,
            markov_w1: find("mtp.2.markov_head.markov_w1.weight")?,
            markov_w2: find("mtp.2.markov_head.markov_w2.weight")?,
            confidence_proj: find("mtp.2.confidence_head.proj.weight")?,
            hc_head_base: find("mtp.2.hc_head_base")?,
            hc_head_fn: find("mtp.2.hc_head_fn")?,
            hc_head_scale: find("mtp.2.hc_head_scale")?,
            layers,
        })
    }

    /// Shape sanity-check against the V4-Flash constants (run after load on the REAL
    /// checkpoint; skip for synthetic/tiny converter GGUFs).
    pub fn validate(&self) -> Result<()> {
        let d = consts::N_EMBD as u64;
        let v = consts::N_VOCAB as u64;
        let r = consts::MARKOV_RANK as u64;
        if self.layers.len() != consts::N_MTP_LAYERS {
            bail!(
                "layers: {} != N_MTP_LAYERS ({})",
                self.layers.len(),
                consts::N_MTP_LAYERS
            );
        }
        for (i, l) in self.layers.iter().enumerate() {
            if l.experts.len() != consts::N_EXPERTS {
                bail!(
                    "mtp.{i}: {} experts != {}",
                    l.experts.len(),
                    consts::N_EXPERTS
                );
            }
        }
        if self.markov_w1.n_elements() != v * r {
            bail!(
                "markov_w1: {} != V*r ({})",
                self.markov_w1.n_elements(),
                v * r
            );
        }
        if self.markov_w2.n_elements() != v * r {
            bail!(
                "markov_w2: {} != V*r ({})",
                self.markov_w2.n_elements(),
                v * r
            );
        }
        // confidence_head.proj: Linear(dim+r → 1) → weight has (dim+r) elements.
        if self.confidence_proj.n_elements() != d + r {
            bail!(
                "confidence_proj: {} != d+r ({})",
                self.confidence_proj.n_elements(),
                d + r
            );
        }
        // main_proj = W_c: Linear(3·dim → dim) → weight [dim, 3·dim].
        if self.main_proj.weight.n_elements() != d * 3 * d {
            bail!(
                "main_proj(W_c): {} != d*3d ({})",
                self.main_proj.weight.n_elements(),
                d * 3 * d
            );
        }
        Ok(())
    }
}

// ───────────────── DSpark-specific CPU reference (the oracle core) ─────────────
//
// These stages are DSpark's novelty over a plain parallel drafter and are fully
// specified by the paper + `inference/model.py`, so they are implemented + tested
// here. They are CONFIRMED CORRECT against the real checkpoint and unchanged.

const RMS_EPS: f32 = 1e-6;

// The two heavy primitives (`matvec` row-major `[out,in]`·x, and `rms_norm` with
// `eps = RMS_EPS`) are now dispatched through `KernelDispatcher` so the SAME draft
// forward runs on the CPU oracle (`CpuDispatcher`) or on the GPU (`MetalDispatcher`).
// The trait's `matvec_f32(w, x, d_out)` infers `d_in = x.len()`; its `rms_norm(x,
// gamma, eps)` matches the old inline f32 form within f32 ULP (the CPU impl reduces
// in f64 — algebraically more precise, ~1e-6 different; every dspark assertion that
// touches it uses a tolerance, see DSpark Phase-2 port). Non-dispatchable ops
// (softmax, silu/sigmoid, the HC sinkhorn/row-softmax, rope, attn_sink, dot
// products, top-k) stay on the CPU as plain Rust below.

/// Context injection (`mtp.0.forward_embed`): `H_ctx = main_norm( main_proj · h_concat )`.
/// `h_concat` is the 3·d concatenation of the target's `mean(dim=2)` residual rows at
/// `TARGET_LAYER_IDS`; `main_proj` = W_c `[d, 3d]`, `main_norm` the RMSNorm gain `[d]`.
///
/// CPU-default entry point — delegates to [`context_inject_with`] on a `CpuDispatcher`
/// so existing call sites (incl. the H_ctx Metal smoke oracle) are unchanged.
pub fn context_inject(
    h_concat: &[f32],
    main_proj: &[f32],
    main_norm: &[f32],
    d: usize,
) -> Vec<f32> {
    context_inject_with(&CpuDispatcher, h_concat, main_proj, main_norm, d)
}

/// Dispatcher-generic context injection: `H_ctx = rms_norm( matvec(main_proj, h_concat) )`.
/// The `matvec` + `rms_norm` run on `disp` (CPU oracle or Metal GPU).
pub fn context_inject_with<D: KernelDispatcher>(
    disp: &D,
    h_concat: &[f32],
    main_proj: &[f32],
    main_norm: &[f32],
    d: usize,
) -> Vec<f32> {
    assert_eq!(h_concat.len(), 3 * d);
    let proj = disp.matvec_f32(main_proj, h_concat, d);
    disp.rms_norm(&proj, main_norm, RMS_EPS)
}

/// Sequential Markov stage (`markov_head`, paper eq 4-5). Greedy (argmax) left-to-right
/// sample of the γ-token block, each step adding the first-order transition bias
/// `B_k(v) = markov_w1[x_{k-1}] · markov_w2[v]`.
///
/// - `base_logits`: γ×V row-major (the head's `U_k`).
/// - `anchor`: `x0` (the target's last accepted token).
/// - `w1`: V×r row-major (markov_w1 embedding).
/// - `w2`: V×r ([`DraftWeight`], f32 or q8) — markov_w2 = `Linear(r→V)`, weight `[V,r]`.
///
/// The per-step transition bias `B_k = w2 · w1[x_{k-1}]` (a `[V]` vector) is a matvec of
/// `w2[V,r]` on the previous token's `w1` row, so markov_w2 can live as q8 (dequant→dot
/// via the q8 reference). It is computed on the **CPU** (`CpuDispatcher`) regardless of
/// the block's dispatcher: greedy Markov sampling is an inherently CPU argmax loop, and a
/// V×r bias matvec keeps it off the GPU (and clear of the GPU matvec's `d_in % 4` /
/// `d_out % 2` shape constraints, which `rank` need not satisfy). Returns the γ drafted
/// token ids. (temp>0 sampling is a later phase; greedy is the spec-decode default and
/// the byte-identity gate target.)
pub fn markov_sample_block(
    base_logits: &[f32],
    gamma: usize,
    vocab: usize,
    anchor: u32,
    w1: &[f32],
    w2: DraftWeight,
    rank: usize,
) -> Vec<u32> {
    assert_eq!(base_logits.len(), gamma * vocab);
    let mut out = Vec::with_capacity(gamma);
    let mut prev = anchor as usize;
    for k in 0..gamma {
        let w1_row = &w1[prev * rank..prev * rank + rank];
        let bias = w2.mv(&CpuDispatcher, w1_row, vocab); // B_k[v] = Σ_j w1[prev,j]·w2[v,j]
        let u_k = &base_logits[k * vocab..k * vocab + vocab];
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for v in 0..vocab {
            let logit = u_k[v] + bias[v];
            if logit > best_v {
                best_v = logit;
                best = v;
            }
        }
        out.push(best as u32);
        prev = best;
    }
    out
}

/// Confidence head (`confidence_head`, paper eq 7): `c_k = σ(w·[h_k ; markov_w1[x_{k-1}]] + b)`,
/// the estimated probability that draft token k survives verification given the prefix
/// was accepted. `prev_tokens[k] = x_{k-1}` (`x0` for k=0).
///
/// `conf_w` is the projection over the `[d + r]` feature; `conf_b` its bias (0 if none).
/// Returns γ probabilities in (0,1).
pub fn confidence_block(
    hidden: &[f32],
    gamma: usize,
    d: usize,
    prev_tokens: &[u32],
    w1: &[f32],
    rank: usize,
    conf_w: &[f32],
    conf_b: f32,
) -> Vec<f32> {
    assert_eq!(hidden.len(), gamma * d);
    assert_eq!(prev_tokens.len(), gamma);
    assert_eq!(conf_w.len(), d + rank);
    let mut out = Vec::with_capacity(gamma);
    for k in 0..gamma {
        let h = &hidden[k * d..k * d + d];
        let prev = prev_tokens[k] as usize;
        let w1_row = &w1[prev * rank..prev * rank + rank];
        let mut acc = conf_b;
        for i in 0..d {
            acc += conf_w[i] * h[i];
        }
        for j in 0..rank {
            acc += conf_w[d + j] * w1_row[j];
        }
        out.push(1.0 / (1.0 + (-acc).exp()));
    }
    out
}

/// Confident-prefix length ℓ (single-user static threshold; DeepSpec
/// `_confident_prefix_length`). The batched hardware-aware throughput scheduler
/// (paper Algorithm 1) is a later phase — on a one-request Mac this static rule is the
/// intended collapse. `threshold <= 0` ⇒ keep the whole block.
pub fn confident_prefix_len(confidence: &[f32], threshold: f32) -> usize {
    if threshold <= 0.0 {
        return confidence.len();
    }
    confidence
        .iter()
        .position(|&c| c < threshold)
        .unwrap_or(confidence.len())
}

// ───────────────────────── draft block forward (the oracle) ─────────────────────────
//
// CORRECTNESS STATUS — read first.
// This is a *faithful transcription* of `inference/model.py::Transformer.forward_spec`
// + `DSparkBlock.{forward_embed,forward,forward_head}` (DeepSpec/DSpark, MIT). It is the
// "byte-exact-by-CONSTRUCTION" CPU reference the later Metal phase gates against. It is
// **NOT** verified against the 284B torch model — the real FP8 weights are not downloaded
// here, so torch cannot be run. The genuine byte-identity gate is Phase 5 (draft vs
// non-spec greedy), not this module. Treat every number below as "transcribed, not
// torch-confirmed".
//
// The HC sinkhorn (`hc_split_sinkhorn`) and `sparse_attn` are *imported kernels* in
// model.py (no Python source). For those two we mirror the already-confirmed CPU oracle
// of the SAME ops that engine_core runs for the main DeepSeek-V4 model
// (`attn_dispatch::CpuAttentionDispatcher::hc_collapse_norm` + its HC-expand sites) —
// that reverse-engineering is the de-facto answer key, more reliable than guessing the
// kernel internals. The block math (MLA low-rank, per-head q-rmsnorm, rope, attn_sink,
// MoE sqrt-softplus routing, SwiGLU experts, Hyper-Connections pre/post/head) is
// transcribed directly from model.py.
//
// DOCUMENTED SIMPLIFICATIONS (each marked `TODO(byte-exact)` at its site):
//  S1. The DSpark attention context is the SINGLE current-step `main_kv` (1 key) — the
//      full sliding-window (win=128) history of prior main-model KV is NOT modelled. The
//      oracle is given only one `main_hidden`, matching the first-decode-step case. With
//      1 ctx key + γ block keys = 1+γ keys and `get_dspark_topk_idxs` selecting *all* of
//      them for every query (the topk window ≥ #keys), `sparse_attn` reduces to DENSE
//      bidirectional attention over those 1+γ keys — exact for this block size.
//  S2. FP8/FP4 weight dequant + the QAT `act_quant` round-trip on activations are NOT
//      applied: `draft_forward` runs in pure f32 over already-dequantized weight slices.
//      `fp8_e4m3_to_f32` / `dequant_fp8_blockwise` below give a faithful (untested)
//      weight dequant; the activation QAT round-trip is dropped (draft quality only,
//      never correctness — DSpark is a drafter).
//  S3. Greedy (temperature=0) Markov sampling — the spec-decode default and the
//      byte-identity target (see `markov_sample_block`).

/// A matvec weight as either dequantized f32 or raw GGUF `block_q8_0` bytes. The
/// `Q8` variant carries the 34-B/32-weight blocks straight from the GGUF mmap (1 B
/// per weight, no f32 materialized) and dispatches through `KernelDispatcher::matvec_q8`
/// (native GPU q8 kernel on Metal, dequant→matvec_f32 reference on CPU). The `F32`
/// variant keeps the existing fully-dequantized path byte-for-byte.
///
/// Both arms wrap a plain slice reference, so `DraftWeight` is `Copy` — the borrowed
/// weight views (`ExpertF32`, `DraftLayerF32`, …) stay trivially copyable.
///
/// ⚠ `matvec_q8` requires `d_out % 2 == 0 && d_in % 32 == 0`; only construct `Q8` for
/// tensors that satisfy both (the converter q8's exactly those — see `--q8`).
#[derive(Clone, Copy)]
pub enum DraftWeight<'a> {
    F32(&'a [f32]),
    Q8(&'a [u8]),
}

impl<'a> DraftWeight<'a> {
    /// Matvec dispatch: `[d_out, d_in] · x[d_in] → [d_out]` (`d_in = x.len()`), via
    /// `matvec_f32` (F32) or `matvec_q8` (Q8).
    #[inline]
    pub fn mv<D: KernelDispatcher>(&self, disp: &D, x: &[f32], d_out: usize) -> Vec<f32> {
        match self {
            DraftWeight::F32(w) => disp.matvec_f32(w, x, d_out),
            DraftWeight::Q8(w) => disp.matvec_q8(w, x, d_out),
        }
    }

    /// Sub-view of `n_rows` consecutive rows starting at row `row`, each of width
    /// `d_in`. For `Q8` a row is `d_in/32·34` bytes (so `d_in % 32 == 0` is required —
    /// satisfied by every q8-able operand). Used for the grouped o-proj `wo_a`, whose
    /// per-group block is a contiguous row range.
    #[inline]
    pub fn rows(&self, row: usize, n_rows: usize, d_in: usize) -> DraftWeight<'a> {
        match self {
            DraftWeight::F32(w) => DraftWeight::F32(&w[row * d_in..(row + n_rows) * d_in]),
            DraftWeight::Q8(w) => {
                debug_assert_eq!(d_in % 32, 0, "DraftWeight::rows q8 d_in {d_in} % 32 != 0");
                let rb = (d_in / 32) * 34;
                DraftWeight::Q8(&w[row * rb..(row + n_rows) * rb])
            }
        }
    }
}

/// An owned matvec weight (f32 or q8 bytes), produced by the lazy real-weight bridge
/// when it dequantizes / reads a routed expert on demand. Borrowed as a [`DraftWeight`]
/// for the duration of one `expert_ffn` call.
pub enum OwnedDraftWeight {
    F32(Vec<f32>),
    Q8(Vec<u8>),
}
impl OwnedDraftWeight {
    #[inline]
    pub fn view(&self) -> DraftWeight<'_> {
        match self {
            OwnedDraftWeight::F32(w) => DraftWeight::F32(w),
            OwnedDraftWeight::Q8(w) => DraftWeight::Q8(w),
        }
    }
}

/// View of one MoE expert's three projections (each f32 or q8).
/// `w1`/`w3` are `[inter, d]` (gate/up), `w2` is `[d, inter]` (down).
///
/// `Copy`: the fields are `DraftWeight` (a slice reference either way), so a borrowed
/// expert view is trivially copyable — the lazy [`ExpertSrc`] provider relies on this.
#[derive(Clone, Copy)]
pub struct ExpertF32<'a> {
    pub w1: DraftWeight<'a>,
    pub w2: DraftWeight<'a>,
    pub w3: DraftWeight<'a>,
}

/// A SELECTED MoE expert, obtained on demand through the [`moe`] provider closure.
/// Either a zero-cost borrow of an already-resident expert (the synthetic /
/// fully-dequantized path) or a freshly-dequantized owned expert that the caller
/// just read from the GGUF mmap (the lazy real-weight path) and that drops the
/// instant `expert_ffn` returns — so only the `n_activated` selected experts are
/// ever materialized at once, never all 256.
pub enum ExpertSrc<'a> {
    Borrowed(ExpertF32<'a>),
    Owned {
        w1: OwnedDraftWeight,
        w2: OwnedDraftWeight,
        w3: OwnedDraftWeight,
    },
}

impl<'a> ExpertSrc<'a> {
    /// Borrow this expert as an [`ExpertF32`] view, regardless of ownership / dtype.
    #[inline]
    pub fn as_view(&self) -> ExpertF32<'_> {
        match self {
            ExpertSrc::Borrowed(v) => *v,
            ExpertSrc::Owned { w1, w2, w3 } => ExpertF32 {
                w1: w1.view(),
                w2: w2.view(),
                w3: w3.view(),
            },
        }
    }
}

/// f32 view of one DSpark MTP stage (`mtp.N`) — a full DeepSeek-V4 block.
/// All matrices are row-major `[out, in]` (PyTorch `Linear.weight` layout).
pub struct DraftLayerF32<'a> {
    pub wq_a: DraftWeight<'a>, // [q_lora_rank, d]            (matvec, q8-able)
    pub wq_b: DraftWeight<'a>, // [n_heads*head_dim, q_lora_rank] (matvec, q8-able)
    pub wkv: DraftWeight<'a>,  // [head_dim, d]               (matvec, q8-able)
    pub wo_a: DraftWeight<'a>, // [n_groups*o_lora_rank, gd]  (grouped matvec, q8-able)
    pub wo_b: DraftWeight<'a>, // [d, n_groups*o_lora_rank]   (matvec, q8-able)
    pub q_norm: &'a [f32],     // [q_lora_rank]
    pub kv_norm: &'a [f32],    // [head_dim]
    pub attn_sink: &'a [f32],  // [n_heads]
    pub attn_norm: &'a [f32],  // [d]
    pub ffn_norm: &'a [f32],   // [d]
    pub gate_w: DraftWeight<'a>, // [n_experts, d]             (matvec, q8-able)
    pub gate_b: &'a [f32],     // [n_experts]
    pub experts: Vec<ExpertF32<'a>>,
    pub shared: ExpertF32<'a>,
    pub hc_attn_fn: &'a [f32],    // [mix_hc, hc*d]
    pub hc_attn_scale: &'a [f32], // [3]
    pub hc_attn_base: &'a [f32],  // [mix_hc]
    pub hc_ffn_fn: &'a [f32],
    pub hc_ffn_scale: &'a [f32],
    pub hc_ffn_base: &'a [f32],
}

/// f32 view of the shared / stage-specific aux weights (mtp.0 ctx projection + mtp.2 head).
/// `embed`/`head` come from the BASE model (`[vocab, d]`), not the `mtp.*` namespace.
pub struct DraftAuxF32<'a> {
    pub main_proj: DraftWeight<'a>, // [d, 3d]  (mtp.0, matvec, q8-able)
    pub main_norm: &'a [f32],       // [d]      (mtp.0)
    pub embed: &'a [f32],           // [vocab, d]  (base, shared) — embedding lookup, NOT matvec
    pub head: DraftWeight<'a>,      // [vocab, d]  (base, shared, matvec, q8-able)
    pub norm: &'a [f32],            // [d]      (mtp.2 final RMSNorm)
    pub hc_head_fn: &'a [f32],      // [hc_mult, hc*d]  (mtp.2)
    pub hc_head_scale: &'a [f32],   // [1]
    pub hc_head_base: &'a [f32],    // [hc_mult]
    pub markov_w1: &'a [f32],       // [vocab, rank] — embedding lookup, NOT matvec
    pub markov_w2: DraftWeight<'a>, // [vocab, rank]  (markov bias matvec, q8-able)
    pub confidence_proj: &'a [f32], // [d + rank]
}

/// Whole DSpark drafter as dequantized f32 slices.
pub struct DraftModelF32<'a> {
    pub layers: Vec<DraftLayerF32<'a>>, // N_MTP_LAYERS stages
    pub aux: DraftAuxF32<'a>,
}

/// All scalar dims for the draft forward. `DraftDims::v4_flash()` returns the real
/// checkpoint values; tests build tiny synthetic ones.
#[derive(Clone, Copy, Debug)]
pub struct DraftDims {
    pub d: usize, // hidden dim
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub n_experts: usize,
    pub n_activated: usize, // routed top-k (n_activated_experts)
    pub route_scale: f32,   // gate routing-weight scale (config route_scale)
    pub inter: usize,       // moe_inter_dim
    pub hc_mult: usize,
    pub gamma: usize,     // dspark_block_size
    pub noise_token: u32, // dspark_noise_token_id (the draft "mask" id)
    pub vocab: usize,
    pub markov_rank: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
}

impl DraftDims {
    /// Real DeepSeek-V4-Flash-DSpark dims (from `ModelArgs` / config.json).
    ///
    /// Confirmed against `inference/config.json`: n_activated_experts=6,
    /// moe_inter_dim=2048, route_scale=1.5, score_func=sqrtsoftplus, q_lora_rank=1024.
    pub fn v4_flash() -> Self {
        Self {
            d: consts::N_EMBD,
            n_heads: consts::N_HEADS,
            head_dim: consts::HEAD_DIM,
            rope_head_dim: 64,
            q_lora_rank: 1024,
            o_lora_rank: 1024,
            o_groups: 8,
            n_experts: consts::N_EXPERTS,
            n_activated: 6,
            route_scale: 1.5,
            inter: 2048,
            hc_mult: consts::HC_MULT,
            gamma: consts::BLOCK_SIZE,
            noise_token: consts::NOISE_TOKEN_ID,
            vocab: consts::N_VOCAB,
            markov_rank: consts::MARKOV_RANK,
            rope_theta: consts::ROPE_THETA,
            rms_eps: RMS_EPS,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 20,
        }
    }
    /// HC mix vector length `2·hc_mult + hc_mult²` (pre + post + comb sections).
    #[inline]
    pub fn mix_hc(&self) -> usize {
        2 * self.hc_mult + self.hc_mult * self.hc_mult
    }
}

#[inline]
fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}
#[inline]
fn silu(z: f32) -> f32 {
    z * sigmoid(z)
}
/// Numerically-stable softplus: `ln(1 + e^z) = max(z,0) + ln(1 + e^-|z|)`.
#[inline]
fn softplus(z: f32) -> f32 {
    z.max(0.0) + (-z.abs()).exp().ln_1p()
}

/// Interleaved (GPT-NeoX adjacent-pair) RoPE on the LAST `rd` dims of `v`, mirroring
/// `model.py::apply_rotary_emb` (view-as-complex over `(x[2j], x[2j+1])` pairs). Plain
/// rope (no YaRN — DSpark attention has `compress_ratio == 0`, `original_seq_len == 0`).
fn rope_inplace(v: &mut [f32], pos: usize, rd: usize, theta: f32, inverse: bool) {
    let start = v.len() - rd;
    for j in 0..rd / 2 {
        let inv_freq = (theta as f64).powf(-(2.0 * j as f64) / (rd as f64)) as f32;
        let ang = pos as f32 * inv_freq;
        let (mut sn, cs) = ang.sin_cos();
        if inverse {
            sn = -sn;
        }
        let i0 = start + 2 * j;
        let a = v[i0];
        let b = v[i0 + 1];
        v[i0] = a * cs - b * sn;
        v[i0 + 1] = a * sn + b * cs;
    }
}

/// Hyper-Connections `hc_pre` (model.py:680) collapse. Mirrors the confirmed main-model
/// oracle `attn_dispatch::CpuAttentionDispatcher::hc_collapse_norm` (the answer key for
/// the imported `hc_split_sinkhorn`). Returns `(collapsed[d], post[n_hc], comb[n_hc²])`;
/// the caller then applies the block's `attn_norm`/`ffn_norm` (kept separate, exactly as
/// model.py does `x = attn_norm(hc_pre(x))`).
fn hc_collapse(
    x_hc: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    n_hc: usize,
    d: usize,
    dims: &DraftDims,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hc_dim = n_hc * d;
    let mix_hc = 2 * n_hc + n_hc * n_hc;
    assert_eq!(x_hc.len(), hc_dim);
    assert_eq!(hc_fn.len(), mix_hc * hc_dim);
    assert_eq!(hc_scale.len(), 3);
    assert_eq!(hc_base.len(), mix_hc);
    let (eps, hc_eps) = (dims.rms_eps, dims.hc_eps);

    // flat = rms_norm_plain(x_hc) — uniform RMS over hc_dim, no gamma. Equivalent to
    // model.py `F.linear(x, hc_fn) * rsqrt(mean(x²)+eps)` since rsqrt is a per-row scalar.
    let ss: f64 = x_hc.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scl = 1.0f32 / ((ss / hc_dim as f64) as f32 + eps).sqrt();
    let flat: Vec<f32> = x_hc.iter().map(|&v| v * scl).collect();

    let mut mix = vec![0.0f32; mix_hc];
    for r in 0..mix_hc {
        mix[r] = hc_fn[r * hc_dim..(r + 1) * hc_dim]
            .iter()
            .zip(&flat)
            .map(|(a, b)| a * b)
            .sum();
    }
    let (pre_s, post_s, comb_s) = (hc_scale[0], hc_scale[1], hc_scale[2]);

    // pre = sigmoid(mix[..n_hc]*pre_s + base) + eps   → collapse weights.
    let mut pre = vec![0.0f32; n_hc];
    for i in 0..n_hc {
        pre[i] = sigmoid(mix[i] * pre_s + hc_base[i]) + hc_eps;
    }
    // post = 2·sigmoid(mix[n_hc..2n_hc]*post_s + base).
    let mut post = vec![0.0f32; n_hc];
    for i in 0..n_hc {
        post[i] = 2.0 * sigmoid(mix[n_hc + i] * post_s + hc_base[n_hc + i]);
    }
    // comb = sinkhorn(row-softmax(reshape(mix[2n_hc..], [n_hc,n_hc])*comb_s + base)).
    let off = 2 * n_hc;
    let mut comb = vec![0.0f32; n_hc * n_hc];
    for i in 0..n_hc * n_hc {
        comb[i] = mix[off + i] * comb_s + hc_base[off + i];
    }
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
            *v = *v * inv + hc_eps;
        }
    }
    let mut col_sum = vec![0.0f32; n_hc];
    for r in 0..n_hc {
        for c in 0..n_hc {
            col_sum[c] += comb[r * n_hc + c];
        }
    }
    for c in 0..n_hc {
        let inv = 1.0 / (col_sum[c] + hc_eps);
        for r in 0..n_hc {
            comb[r * n_hc + c] *= inv;
        }
    }
    for _ in 1..dims.hc_sinkhorn_iters {
        for r in 0..n_hc {
            let row = &mut comb[r * n_hc..(r + 1) * n_hc];
            let rs: f32 = row.iter().sum();
            let inv = 1.0 / (rs + hc_eps);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        col_sum.iter_mut().for_each(|c| *c = 0.0);
        for r in 0..n_hc {
            for c in 0..n_hc {
                col_sum[c] += comb[r * n_hc + c];
            }
        }
        for c in 0..n_hc {
            let inv = 1.0 / (col_sum[c] + hc_eps);
            for r in 0..n_hc {
                comb[r * n_hc + c] *= inv;
            }
        }
    }

    let mut collapsed = vec![0.0f32; d];
    for h in 0..n_hc {
        let w = pre[h];
        for j in 0..d {
            collapsed[j] += w * x_hc[h * d + j];
        }
    }
    (collapsed, post, comb)
}

/// Hyper-Connections `hc_post` (model.py:690) expansion:
/// `out[dst] = post[dst]·new_x + Σ_src comb[src,dst]·residual[src]`. Comb addressing
/// `comb[src*n_hc + dst]` matches the confirmed main-model HC-expand sites.
fn hc_expand(
    new_x: &[f32],
    residual_hc: &[f32],
    post: &[f32],
    comb: &[f32],
    n_hc: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_hc * d];
    for dst in 0..n_hc {
        let wp = post[dst];
        for e in 0..d {
            let mut acc = wp * new_x[e];
            for src in 0..n_hc {
                acc += comb[src * n_hc + dst] * residual_hc[src * d + e];
            }
            out[dst * d + e] = acc;
        }
    }
    out
}

/// `hc_head` collapse (model.py:709, mtp.2). Like `hc_collapse` but `hc_fn` is
/// `[n_hc, hc*d]` (one mix per copy), `hc_scale` is length-1, no post/comb, no sinkhorn:
/// `pre = sigmoid(mix*scale + base) + eps`, `y = Σ_h pre[h]·x[h]`.
fn hc_head_collapse(
    x_hc: &[f32],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    n_hc: usize,
    d: usize,
    dims: &DraftDims,
) -> Vec<f32> {
    let hc_dim = n_hc * d;
    assert_eq!(x_hc.len(), hc_dim);
    assert_eq!(hc_fn.len(), n_hc * hc_dim);
    assert_eq!(hc_base.len(), n_hc);
    let ss: f64 = x_hc.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let scl = 1.0f32 / ((ss / hc_dim as f64) as f32 + dims.rms_eps).sqrt();
    let flat: Vec<f32> = x_hc.iter().map(|&v| v * scl).collect();
    let mut y = vec![0.0f32; d];
    for h in 0..n_hc {
        let mix: f32 = hc_fn[h * hc_dim..(h + 1) * hc_dim]
            .iter()
            .zip(&flat)
            .map(|(a, b)| a * b)
            .sum();
        let pre = sigmoid(mix * hc_scale[0] + hc_base[h]) + dims.hc_eps;
        for j in 0..d {
            y[j] += pre * x_hc[h * d + j];
        }
    }
    y
}

/// One SwiGLU expert (model.py:592): `w2( silu(w1·x) * (w3·x) )`. `swiglu_limit==0` ⇒ no
/// clamp. Routing weight is applied by the caller (linear, so order-free).
fn expert_ffn<D: KernelDispatcher>(
    disp: &D,
    x: &[f32],
    e: &ExpertF32,
    dims: &DraftDims,
) -> Vec<f32> {
    let (d, inter) = (dims.d, dims.inter);
    let g = e.w1.mv(disp, x, inter);
    let u = e.w3.mv(disp, x, inter);
    let act: Vec<f32> = g.iter().zip(&u).map(|(&gi, &ui)| silu(gi) * ui).collect();
    e.w2.mv(disp, &act, d)
}

/// 256-expert MoE FFN (model.py:614). Gate (model.py:551): `sqrtsoftplus` scores,
/// bias-shifted top-k *selection*, but routing *weights* are the unbiased scores,
/// L1-renormalized (score_func != softmax). `+ shared_expert` (unweighted).
// Confirmed vs inference/config.json: score_func=sqrtsoftplus, route_scale=1.5,
// n_activated_experts=6, n_shared_experts=1.
// The SELECTED experts are fetched through `get_expert`, never indexed directly out
// of `layer.experts` — so the real-weight path can dequantize just the `n_activated`
// activated experts from the GGUF mmap on demand (see [`DraftRealWeights`]) while the
// synthetic path returns a zero-cost borrow of the resident expert. The shared expert
// stays dense on `layer.shared`.
fn moe<'a, D: KernelDispatcher>(
    disp: &D,
    x: &[f32],
    layer: &DraftLayerF32,
    dims: &DraftDims,
    get_expert: impl Fn(usize) -> ExpertSrc<'a>,
) -> Vec<f32> {
    let ne = dims.n_experts;
    let raw = layer.gate_w.mv(disp, x, ne); // gate_w [n_experts, d]
    let scores: Vec<f32> = raw.iter().map(|&z| softplus(z).sqrt()).collect();
    // selection on bias-shifted scores; weights from the unbiased scores.
    let mut order: Vec<usize> = (0..ne).collect();
    order.sort_by(|&a, &b| {
        let sa = scores[a] + layer.gate_b[a];
        let sb = scores[b] + layer.gate_b[b];
        sb.partial_cmp(&sa).unwrap().then(a.cmp(&b))
    });
    let sel = &order[..dims.n_activated];
    let wsum: f32 = sel.iter().map(|&e| scores[e]).sum();
    let mut y = vec![0.0f32; dims.d];
    for &e in sel {
        let w = dims.route_scale * scores[e] / wsum; // model.py Gate: weights *= route_scale
        let src = get_expert(e); // lazy: dequant-on-demand (real) or borrow (synthetic)
        let ey = expert_ffn(disp, x, &src.as_view(), dims);
        for (yi, eyi) in y.iter_mut().zip(&ey) {
            *yi += w * eyi;
        }
        // `src` (an `Owned` expert in the real path) drops here — at most `n_activated`
        // experts are resident at any instant.
    }
    let sy = expert_ffn(disp, x, &layer.shared, dims);
    for (yi, syi) in y.iter_mut().zip(&sy) {
        *yi += syi;
    }
    y
}

/// `DSparkAttention` (model.py:750) over the γ-position block, with `main_kv` injection.
/// Input `x_rows` = γ rows of `[d]` (already `attn_norm`'d). `main_x` = the single H_ctx
/// row `[d]`. Returns γ rows of `[d]`.
///
/// MLA: a single latent KV (head_dim) is BOTH key and value, shared across heads. Per
/// head: per-head q-rmsnorm (no gamma) → rope; keys/values rope'd at their positions;
/// `attn_sink[h]` is an extra softmax logit with value 0; output last-`rd` dims are
/// inverse-rope'd; then grouped low-rank `wo_a` (einsum bsgd,grd->bsgr) → `wo_b`.
// TODO(byte-exact) S1: context = the single current-step main_kv (1 key). The win=128
// sliding-window of prior main-model KV is not modelled — exact for the first decode step
// / single-context case; sparse_attn(topk≥#keys) ⇒ dense over the 1+γ keys.
fn attention<D: KernelDispatcher>(
    disp: &D,
    x_rows: &[f32],
    main_x: &[f32],
    layer: &DraftLayerF32,
    dims: &DraftDims,
    start_pos: usize,
) -> Vec<f32> {
    let (d, hd, nh, rd, g) = (
        dims.d,
        dims.head_dim,
        dims.n_heads,
        dims.rope_head_dim,
        dims.gamma,
    );
    let theta = dims.rope_theta;

    // Context key: main_kv = kv_norm(wkv·main_x), rope'd at the context position.
    let mut main_kv = disp.rms_norm(&layer.wkv.mv(disp, main_x, hd), layer.kv_norm, RMS_EPS);
    rope_inplace(&mut main_kv, start_pos, rd, theta, false);

    // Block keys (one per position) + queries (per head).
    let mut keys: Vec<Vec<f32>> = Vec::with_capacity(1 + g);
    keys.push(main_kv);
    let mut qheads: Vec<Vec<f32>> = Vec::with_capacity(g);
    for s in 0..g {
        let xs = &x_rows[s * d..(s + 1) * d];
        let pos = start_pos + 1 + s;
        let mut kv = disp.rms_norm(&layer.wkv.mv(disp, xs, hd), layer.kv_norm, RMS_EPS);
        rope_inplace(&mut kv, pos, rd, theta, false);
        keys.push(kv);

        let qr = disp.rms_norm(
            &layer.wq_a.mv(disp, xs, dims.q_lora_rank),
            layer.q_norm,
            RMS_EPS,
        );
        let mut q = layer.wq_b.mv(disp, &qr, nh * hd);
        for h in 0..nh {
            let hsl = &mut q[h * hd..(h + 1) * hd];
            let ms = hsl.iter().map(|v| v * v).sum::<f32>() / hd as f32;
            let inv = 1.0 / (ms + dims.rms_eps).sqrt();
            for v in hsl.iter_mut() {
                *v *= inv;
            }
            rope_inplace(hsl, pos, rd, theta, false);
        }
        qheads.push(q);
    }
    // NB keys ordering: [main_kv (pos start_pos), block_0..block_{γ-1}]. The block key for
    // position s is keys[1+s]; the corresponding query is qheads[s] — but every query
    // attends to ALL keys (bidirectional, per get_dspark_topk_idxs).

    let scale = (hd as f32).powf(-0.5);
    let (ng, olr) = (dims.o_groups, dims.o_lora_rank);
    let gd = nh * hd / ng;
    let mut out = vec![0.0f32; g * d];
    for s in 0..g {
        let q = &qheads[s];
        let pos = start_pos + 1 + s;
        let mut o_s = vec![0.0f32; nh * hd];
        for h in 0..nh {
            let qh = &q[h * hd..(h + 1) * hd];
            // logits: [sink_h, q·k_j * scale ...]; sink value is 0.
            let mut logits = Vec::with_capacity(keys.len() + 1);
            logits.push(layer.attn_sink[h]);
            for kj in &keys {
                let dotp: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
                logits.push(dotp * scale);
            }
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut den = 0.0f32;
            let ex: Vec<f32> = logits
                .iter()
                .map(|&l| {
                    let e = (l - m).exp();
                    den += e;
                    e
                })
                .collect();
            let oh = &mut o_s[h * hd..(h + 1) * hd];
            for (ki, kj) in keys.iter().enumerate() {
                let w = ex[ki + 1] / den; // skip ex[0] = sink (value 0)
                for t in 0..hd {
                    oh[t] += w * kj[t];
                }
            }
            rope_inplace(oh, pos, rd, theta, true); // de-rotate output rope dims
        }
        // grouped low-rank o-proj: wo_a [n_groups, o_lora_rank, gd] → wo_b [d, ng*olr].
        // Each group is a contiguous `[olr, gd]` row block of wo_a, so the per-group
        // einsum `oflat[grp,r] = Σ_t wo_a[grp,r,t]·og[grp,t]` is exactly a matvec of that
        // row block on the group's `og` — dispatched (q8 or f32) via `DraftWeight::mv`.
        let mut oflat = vec![0.0f32; ng * olr];
        for grp in 0..ng {
            let og = &o_s[grp * gd..(grp + 1) * gd];
            let wg = layer.wo_a.rows(grp * olr, olr, gd);
            let og_proj = wg.mv(disp, og, olr);
            oflat[grp * olr..(grp + 1) * olr].copy_from_slice(&og_proj);
        }
        let xout = layer.wo_b.mv(disp, &oflat, d);
        out[s * d..(s + 1) * d].copy_from_slice(&xout);
    }
    out
}

/// One DSpark MTP block (`DSparkBlock.forward` → `Block.forward`, model.py:695):
/// `hc_pre → attn_norm → DSparkAttention(main_x) → hc_post` then
/// `hc_pre → ffn_norm → MoE → hc_post`. Operates per γ-position independently for HC and
/// jointly for attention. `x` is γ rows of `[hc_mult*d]` (HC copies).
fn draft_block<'a, D: KernelDispatcher>(
    disp: &D,
    x: &mut Vec<f32>,
    layer: &DraftLayerF32,
    main_x: &[f32],
    dims: &DraftDims,
    start_pos: usize,
    get_expert: &impl Fn(usize) -> ExpertSrc<'a>,
) {
    let (d, g, nhc) = (dims.d, dims.gamma, dims.hc_mult);
    // ── attention half ──
    let mut attn_in = vec![0.0f32; g * d];
    let mut posts = Vec::with_capacity(g);
    let mut combs = Vec::with_capacity(g);
    for s in 0..g {
        let xhc = &x[s * nhc * d..(s + 1) * nhc * d];
        let (col, post, comb) = hc_collapse(
            xhc,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            nhc,
            d,
            dims,
        );
        attn_in[s * d..(s + 1) * d].copy_from_slice(&disp.rms_norm(&col, layer.attn_norm, RMS_EPS));
        posts.push(post);
        combs.push(comb);
    }
    let attn_out = attention(disp, &attn_in, main_x, layer, dims, start_pos);
    let mut x2 = vec![0.0f32; g * nhc * d];
    for s in 0..g {
        let resid = &x[s * nhc * d..(s + 1) * nhc * d];
        let exp = hc_expand(
            &attn_out[s * d..(s + 1) * d],
            resid,
            &posts[s],
            &combs[s],
            nhc,
            d,
        );
        x2[s * nhc * d..(s + 1) * nhc * d].copy_from_slice(&exp);
    }
    *x = x2;
    // ── FFN/MoE half ──
    let mut x3 = vec![0.0f32; g * nhc * d];
    for s in 0..g {
        let xhc = &x[s * nhc * d..(s + 1) * nhc * d];
        let (col, post, comb) = hc_collapse(
            xhc,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            nhc,
            d,
            dims,
        );
        let normed = disp.rms_norm(&col, layer.ffn_norm, RMS_EPS);
        let y = moe(disp, &normed, layer, dims, get_expert);
        let exp = hc_expand(&y, xhc, &post, &comb, nhc, d);
        x3[s * nhc * d..(s + 1) * nhc * d].copy_from_slice(&exp);
    }
    *x = x3;
}

/// Full DSpark draft round — the oracle entry point (`Transformer.forward_spec`
/// model.py:928 + `DSparkBlock.forward_head` model.py:860).
///
/// Inputs: dequantized f32 `weights`, the target tap `main_hidden` (`3·d`, the
/// `cat(mean over hc copies)` of the 3 target layers), the `anchor` token id (last
/// accepted token = `x0`), the absolute decode position `start_pos`, and `dims`.
///
/// Returns `(output_ids, base_logits, confidence)`:
///  - `output_ids`: γ greedily-Markov-sampled draft token ids;
///  - `base_logits`: γ×V head logits `U_k = head(norm(hc_head(x)))` (pre-Markov-bias);
///  - `confidence`: γ accept probabilities in (0,1).
pub fn draft_forward(
    weights: &DraftModelF32,
    main_hidden: &[f32],
    anchor: u32,
    start_pos: usize,
    dims: &DraftDims,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    // Default provider: borrow the resident expert. Used by the synthetic /
    // fully-dequantized path; behaviour is identical to indexing `layer.experts[e]`.
    // CPU oracle default — the heavy ops run on `CpuDispatcher`.
    draft_forward_with(
        &CpuDispatcher,
        weights,
        main_hidden,
        anchor,
        start_pos,
        dims,
        |li, e| ExpertSrc::Borrowed(weights.layers[li].experts[e]),
    )
}

/// Provider-driven draft round. Identical to [`draft_forward`] except the routed
/// experts are obtained through `get_expert(layer_idx, expert_idx) -> ExpertSrc`,
/// letting a caller dequantize only the activated experts from a GGUF mmap on demand
/// (see [`DraftRealWeights::run_draft`]). The dense weights — including the SHARED
/// expert (`layer.shared`) — come from `weights` as usual.
pub fn draft_forward_with<'a, D: KernelDispatcher>(
    disp: &D,
    weights: &'a DraftModelF32<'a>,
    main_hidden: &[f32],
    anchor: u32,
    start_pos: usize,
    dims: &DraftDims,
    get_expert: impl Fn(usize, usize) -> ExpertSrc<'a>,
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let (d, g, nhc, v, r) = (
        dims.d,
        dims.gamma,
        dims.hc_mult,
        dims.vocab,
        dims.markov_rank,
    );
    assert_eq!(main_hidden.len(), 3 * d);
    assert_eq!(weights.layers.len(), consts::N_MTP_LAYERS);

    // forward_embed: H_ctx + HC-expanded embeddings of [anchor, NOISE×(γ-1)].
    // H_ctx = rms_norm( main_proj · main_hidden ) — main_proj dispatched (q8 or f32).
    let aux = &weights.aux;
    let main_x = disp.rms_norm(
        &aux.main_proj.mv(disp, main_hidden, d),
        aux.main_norm,
        RMS_EPS,
    );
    let mut ids = vec![dims.noise_token; g];
    ids[0] = anchor;
    let mut x = vec![0.0f32; g * nhc * d];
    for s in 0..g {
        let tok = ids[s] as usize;
        let emb = &aux.embed[tok * d..tok * d + d];
        for c in 0..nhc {
            x[s * nhc * d + c * d..s * nhc * d + c * d + d].copy_from_slice(emb);
        }
    }

    // 3 DSpark MTP blocks (each injects main_x as the context KV).
    for (li, layer) in weights.layers.iter().enumerate() {
        draft_block(disp, &mut x, layer, &main_x, dims, start_pos, &|e| {
            get_expert(li, e)
        });
    }

    // forward_head (mtp.2): hc_head collapse → base logits + the (pre-norm) hidden.
    let mut hidden = vec![0.0f32; g * d];
    let mut base_logits = vec![0.0f32; g * v];
    for s in 0..g {
        let xhc = &x[s * nhc * d..(s + 1) * nhc * d];
        let col = hc_head_collapse(
            xhc,
            aux.hc_head_fn,
            aux.hc_head_scale,
            aux.hc_head_base,
            nhc,
            d,
            dims,
        );
        let logit = aux
            .head
            .mv(disp, &disp.rms_norm(&col, aux.norm, RMS_EPS), v);
        base_logits[s * v..(s + 1) * v].copy_from_slice(&logit);
        hidden[s * d..(s + 1) * d].copy_from_slice(&col); // confidence uses the pre-norm collapse
    }

    // Markov greedy sampling + confidence (the confirmed DSpark stages).
    let output_ids =
        markov_sample_block(&base_logits, g, v, anchor, aux.markov_w1, aux.markov_w2, r);
    // prev_tokens for confidence k = [anchor, out_0, …, out_{γ-2}] = output_ids[0..γ-1] prefixed by anchor.
    let mut prev = Vec::with_capacity(g);
    prev.push(anchor);
    prev.extend_from_slice(&output_ids[..g - 1]);
    let confidence = confidence_block(
        &hidden,
        g,
        d,
        &prev,
        aux.markov_w1,
        r,
        aux.confidence_proj,
        0.0,
    );

    (output_ids, base_logits, confidence)
}

// ───────────────────────── FP8 weight dequant (faithful, untested) ─────────────────────
//
// The checkpoint stores weights as FP8-E4M3 with per-block (block_size=128) UE8M0 scales
// (`.weight` + `.scale`). These helpers reproduce that scheme so a caller CAN build the
// f32 slices `draft_forward` expects. They are transcribed-not-torch-verified (S2) and
// the activation-side QAT (`act_quant`) round-trip is intentionally omitted (drafter
// quality only). The real-weight path is exercised only once weights are downloaded.

/// Decode one FP8-E4M3 byte (1 sign, 4 exp bias-7, 3 mantissa; no Inf; exp=15&mant=7=NaN).
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as i32;
    if exp == 0 {
        // subnormal: 2^(1-7) * (mant/8)
        sign * 2f32.powi(-6) * (mant as f32 / 8.0)
    } else if exp == 0x0f && mant == 0x07 {
        f32::NAN
    } else {
        sign * 2f32.powi(exp - 7) * (1.0 + mant as f32 / 8.0)
    }
}

/// UE8M0 (E8M0) scale byte → `2^(byte-127)` (byte 255 = NaN, treated as 0-scale here).
pub fn ue8m0_scale(b: u8) -> f32 {
    if b == 0xff {
        0.0
    } else {
        2f32.powi(b as i32 - 127)
    }
}

/// Dequantize an FP8-E4M3 blockwise weight `[out, in]` with a `[ceil(out/blk), ceil(in/blk)]`
/// UE8M0 scale grid into row-major f32. `block` is the per-block size (128 for V4-Flash).
// TODO(byte-exact): cross-check tiling/rounding against the checkpoint once weights land.
pub fn dequant_fp8_blockwise(
    weight: &[u8],
    scale: &[u8],
    out: usize,
    inp: usize,
    block: usize,
) -> Vec<f32> {
    assert_eq!(weight.len(), out * inp);
    let sc_cols = inp.div_ceil(block);
    let mut y = vec![0.0f32; out * inp];
    for o in 0..out {
        for i in 0..inp {
            let s = ue8m0_scale(scale[(o / block) * sc_cols + i / block]);
            y[o * inp + i] = fp8_e4m3_to_f32(weight[o * inp + i]) * s;
        }
    }
    y
}

// ───────────────────── lazy real-weight bridge (DraftRealWeights) ─────────────────────
//
// Runs `draft_forward` on the REAL 37 GB DSpark GGUF without ever materializing all
// ~19 B params as f32 (~77 GB). Strategy:
//   • DENSE weights (MLA, norms, gate, SHARED expert, HC params, aux head) are
//     dequantized EAGERLY into owned `Vec<f32>` — a few GB.
//   • the 256 ROUTED experts per stage are dequantized ON DEMAND via the
//     `draft_forward_with` provider — only the `n_activated` (≈6) selected per token
//     are ever read, and each drops the instant `expert_ffn` returns.
// The GGUF here is all F16 (the converter rewrote FP8→F16), so reads go straight
// through `f16_bits_to_f32`; the `.scale` siblings are absent (`QTensor::scale == None`).

use crate::gguf::GgmlType;
use crate::layer_view::{f16_bits_to_f32, MmapSource};
use std::cell::Cell;

/// Read tensor `t`'s raw bytes from the GGUF mmap `bytes` at its absolute offset and
/// dequantize to row-major f32. The real DSpark GGUF is F16; F32 is accepted too.
fn dequant_tensor(bytes: &[u8], t: &DsparkTensor) -> Result<Vec<f32>> {
    let n = t.n_elements() as usize;
    let start = t.abs_offset as usize;
    match t.info.ttype {
        GgmlType::F16 => {
            let need = n.checked_mul(2).ok_or_else(|| anyhow!("size overflow"))?;
            let end = start
                .checked_add(need)
                .ok_or_else(|| anyhow!("offset overflow"))?;
            let raw = bytes.get(start..end).ok_or_else(|| {
                anyhow!(
                    "tensor `{}` F16 bytes [{start}..{end}] out of file bounds",
                    t.info.name
                )
            })?;
            Ok((0..n)
                .map(|i| f16_bits_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])))
                .collect())
        }
        GgmlType::F32 => {
            let need = n.checked_mul(4).ok_or_else(|| anyhow!("size overflow"))?;
            let end = start
                .checked_add(need)
                .ok_or_else(|| anyhow!("offset overflow"))?;
            let raw = bytes.get(start..end).ok_or_else(|| {
                anyhow!(
                    "tensor `{}` F32 bytes [{start}..{end}] out of file bounds",
                    t.info.name
                )
            })?;
            Ok((0..n)
                .map(|i| {
                    f32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]])
                })
                .collect())
        }
        other => bail!(
            "dspark dequant: unsupported type {other:?} for `{}`",
            t.info.name
        ),
    }
}

/// Read a `Q8_0` tensor's raw `block_q8_0` bytes (34 B / 32 weights) straight from the
/// mmap — NO dequant. The byte layout is exactly what `KernelDispatcher::matvec_q8`
/// consumes (row-major `[d_out, d_in]` blocks).
fn raw_q8_bytes(bytes: &[u8], t: &DsparkTensor) -> Result<Vec<u8>> {
    let n = t.n_elements() as usize;
    if n % 32 != 0 {
        bail!("q8 tensor `{}` n_elements {n} % 32 != 0", t.info.name);
    }
    let nbytes = (n / 32) * 34;
    let start = t.abs_offset as usize;
    let end = start
        .checked_add(nbytes)
        .ok_or_else(|| anyhow!("q8 offset overflow"))?;
    let raw = bytes.get(start..end).ok_or_else(|| {
        anyhow!(
            "q8 tensor `{}` bytes [{start}..{end}] out of file bounds",
            t.info.name
        )
    })?;
    Ok(raw.to_vec())
}

/// Load one MATVEC operand as an [`OwnedDraftWeight`]: `Q8` (raw blocks, no dequant) if
/// the GGUF tensor is `Q8_0` (the `--q8` converter output), else `F32` (dequant F16/F32).
/// Lets the SAME `DraftRealWeights` load either the all-F16 or the q8 GGUF.
fn load_matvec(bytes: &[u8], t: &DsparkTensor) -> Result<OwnedDraftWeight> {
    match t.info.ttype {
        GgmlType::Q8_0 => Ok(OwnedDraftWeight::Q8(raw_q8_bytes(bytes, t)?)),
        _ => Ok(OwnedDraftWeight::F32(dequant_tensor(bytes, t)?)),
    }
}

/// One eagerly-dequantized MoE expert (the SHARED expert; routed experts stay lazy).
struct DenseExpert {
    w1: OwnedDraftWeight,
    w2: OwnedDraftWeight,
    w3: OwnedDraftWeight,
}
impl DenseExpert {
    fn load(bytes: &[u8], e: &ExpertTensors) -> Result<Self> {
        Ok(Self {
            w1: load_matvec(bytes, &e.w1.weight)?,
            w2: load_matvec(bytes, &e.w2.weight)?,
            w3: load_matvec(bytes, &e.w3.weight)?,
        })
    }
    fn view(&self) -> ExpertF32<'_> {
        ExpertF32 {
            w1: self.w1.view(),
            w2: self.w2.view(),
            w3: self.w3.view(),
        }
    }
}

/// All DENSE (non-routed-expert) f32 of one `mtp.N` stage, owned.
struct DenseLayer {
    wq_a: OwnedDraftWeight,
    wq_b: OwnedDraftWeight,
    wkv: OwnedDraftWeight,
    wo_a: OwnedDraftWeight,
    wo_b: OwnedDraftWeight,
    q_norm: Vec<f32>,
    kv_norm: Vec<f32>,
    attn_sink: Vec<f32>,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_w: OwnedDraftWeight,
    gate_b: Vec<f32>,
    shared: DenseExpert,
    hc_attn_fn: Vec<f32>,
    hc_attn_scale: Vec<f32>,
    hc_attn_base: Vec<f32>,
    hc_ffn_fn: Vec<f32>,
    hc_ffn_scale: Vec<f32>,
    hc_ffn_base: Vec<f32>,
}
impl DenseLayer {
    fn load(bytes: &[u8], l: &DsparkLayer) -> Result<Self> {
        Ok(Self {
            wq_a: load_matvec(bytes, &l.wq_a.weight)?,
            wq_b: load_matvec(bytes, &l.wq_b.weight)?,
            wkv: load_matvec(bytes, &l.wkv.weight)?,
            wo_a: load_matvec(bytes, &l.wo_a.weight)?,
            wo_b: load_matvec(bytes, &l.wo_b.weight)?,
            q_norm: dequant_tensor(bytes, &l.q_norm)?,
            kv_norm: dequant_tensor(bytes, &l.kv_norm)?,
            attn_sink: dequant_tensor(bytes, &l.attn_sink)?,
            attn_norm: dequant_tensor(bytes, &l.attn_norm)?,
            ffn_norm: dequant_tensor(bytes, &l.ffn_norm)?,
            gate_w: load_matvec(bytes, &l.gate_w)?,
            gate_b: dequant_tensor(bytes, &l.gate_b)?,
            shared: DenseExpert::load(bytes, &l.shared_experts)?,
            hc_attn_fn: dequant_tensor(bytes, &l.hc_attn_fn)?,
            hc_attn_scale: dequant_tensor(bytes, &l.hc_attn_scale)?,
            hc_attn_base: dequant_tensor(bytes, &l.hc_attn_base)?,
            hc_ffn_fn: dequant_tensor(bytes, &l.hc_ffn_fn)?,
            hc_ffn_scale: dequant_tensor(bytes, &l.hc_ffn_scale)?,
            hc_ffn_base: dequant_tensor(bytes, &l.hc_ffn_base)?,
        })
    }
    /// Borrow as a `DraftLayerF32` with an EMPTY routed-expert list — the provider
    /// supplies those lazily; only `shared` is dense.
    fn view(&self) -> DraftLayerF32<'_> {
        DraftLayerF32 {
            wq_a: self.wq_a.view(),
            wq_b: self.wq_b.view(),
            wkv: self.wkv.view(),
            wo_a: self.wo_a.view(),
            wo_b: self.wo_b.view(),
            q_norm: &self.q_norm,
            kv_norm: &self.kv_norm,
            attn_sink: &self.attn_sink,
            attn_norm: &self.attn_norm,
            ffn_norm: &self.ffn_norm,
            gate_w: self.gate_w.view(),
            gate_b: &self.gate_b,
            experts: Vec::new(), // routed experts are dequantized on demand
            shared: self.shared.view(),
            hc_attn_fn: &self.hc_attn_fn,
            hc_attn_scale: &self.hc_attn_scale,
            hc_attn_base: &self.hc_attn_base,
            hc_ffn_fn: &self.hc_ffn_fn,
            hc_ffn_scale: &self.hc_ffn_scale,
            hc_ffn_base: &self.hc_ffn_base,
        }
    }
}

/// DENSE aux f32, owned. `embed`/`head` come from the BASE model (not this GGUF) and
/// are supplied by the caller of `run_draft`, so they are NOT stored here.
struct DenseAux {
    main_proj: OwnedDraftWeight,
    main_norm: Vec<f32>,
    norm: Vec<f32>,
    hc_head_fn: Vec<f32>,
    hc_head_scale: Vec<f32>,
    hc_head_base: Vec<f32>,
    markov_w1: Vec<f32>,
    markov_w2: OwnedDraftWeight,
    confidence_proj: Vec<f32>,
}
impl DenseAux {
    fn load(bytes: &[u8], w: &DsparkWeights) -> Result<Self> {
        Ok(Self {
            main_proj: load_matvec(bytes, &w.main_proj.weight)?,
            main_norm: dequant_tensor(bytes, &w.main_norm)?,
            norm: dequant_tensor(bytes, &w.norm)?,
            hc_head_fn: dequant_tensor(bytes, &w.hc_head_fn)?,
            hc_head_scale: dequant_tensor(bytes, &w.hc_head_scale)?,
            hc_head_base: dequant_tensor(bytes, &w.hc_head_base)?,
            markov_w1: dequant_tensor(bytes, &w.markov_w1)?,
            markov_w2: load_matvec(bytes, &w.markov_w2)?,
            confidence_proj: dequant_tensor(bytes, &w.confidence_proj)?,
        })
    }
}

/// The SHARED base-model `embed.weight` + `head.weight` (DeepSeek-V4-Flash), which the
/// DSpark drafter reuses (`mtp[-1].embed = self.embed`, `.head = self.head`). They live in
/// a SEPARATE base GGUF (not the `mtp.*` module GGUF), so this is loaded optionally via
/// [`DraftRealWeights::with_base`].
///
/// `embed` stays LAZY — one row (`d` f16 values) is read from the mmap on demand, so the
/// full ~2 GB embedding table is never materialized. `head` is the LM-head matvec operand
/// `[vocab, d]`; it is dequantized eagerly to f32 (~2 GB resident) because the head matvec
/// touches every row each call.
struct BaseEmbedHead {
    mmap: MmapSource,    // keeps the base GGUF mapped for on-demand `embed` row reads
    embed: DsparkTensor, // embed.weight [vocab, d] (row-major) — F16, read one row at a time
    head: Vec<f32>,      // head.weight [vocab, d] dequantized to f32 (row-major)
    d: usize,
    vocab: usize,
}

impl BaseEmbedHead {
    /// Read row `tok` (the `d`-wide base embedding) from the mmap and dequant to f32. No
    /// table materialization — one bounds-checked row per call.
    fn embed_row(&self, tok: u32) -> Vec<f32> {
        let t = tok as usize;
        assert!(t < self.vocab, "embed token {t} >= vocab {}", self.vocab);
        let bytes = self.mmap.as_slice();
        let d = self.d;
        match self.embed.info.ttype {
            GgmlType::F16 => {
                let start = self.embed.abs_offset as usize + t * d * 2;
                let raw = &bytes[start..start + d * 2];
                (0..d)
                    .map(|i| f16_bits_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])))
                    .collect()
            }
            GgmlType::F32 => {
                let start = self.embed.abs_offset as usize + t * d * 4;
                let raw = &bytes[start..start + d * 4];
                (0..d)
                    .map(|i| {
                        f32::from_le_bytes([
                            raw[4 * i],
                            raw[4 * i + 1],
                            raw[4 * i + 2],
                            raw[4 * i + 3],
                        ])
                    })
                    .collect()
            }
            other => panic!("base embed: unsupported type {other:?}"),
        }
    }
}

/// Owner of the real DSpark GGUF for a memory-bounded `draft_forward`.
///
/// Holds the `MmapSource` (keeps the 37 GB file mapped, paged in lazily by the OS),
/// the located tensor table (`DsparkWeights`), and the EAGERLY-dequantized dense f32
/// (everything except the 256 routed experts per stage — a few GB total). The routed
/// experts are read+dequantized on demand inside [`run_draft`].
///
/// `base` is the OPTIONAL shared base-model embed/head ([`with_base`]); when present,
/// [`run_draft_real`] feeds the drafter the REAL embedding rows + LM head so the emitted
/// draft tokens are the actual model's (meaningful), not synthetic placeholders.
pub struct DraftRealWeights {
    _mmap: MmapSource, // keeps the GGUF pages valid for the on-demand expert reads
    weights: DsparkWeights,
    dims: DraftDims,
    dense_layers: Vec<DenseLayer>,
    aux: DenseAux,
    base: Option<BaseEmbedHead>,
}

impl DraftRealWeights {
    /// Load from a GGUF path: parse the tensor table, mmap the data, and eagerly
    /// dequantize all DENSE weights to f32. Routed experts stay on disk until selected.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let gguf = GgufFile::open(path)?;
        let weights = DsparkWeights::from_gguf(&gguf)?;
        // Our own mmap — `GgufFile` drops its mapping after parsing, so we keep an
        // independent `MmapSource` of the same file alive for the lazy expert reads.
        let mmap = MmapSource::open(path)?;
        let bytes = mmap.as_slice();
        let dense_layers = weights
            .layers
            .iter()
            .map(|l| DenseLayer::load(bytes, l))
            .collect::<Result<Vec<_>>>()?;
        let aux = DenseAux::load(bytes, &weights)?;
        Ok(Self {
            _mmap: mmap,
            weights,
            dims: DraftDims::v4_flash(),
            dense_layers,
            aux,
            base: None,
        })
    }

    /// Attach the SHARED base-model `embed.weight` + `head.weight` from a separate base
    /// GGUF (the DSpark module GGUF does not contain them — see [`DsparkWeights`]). Once
    /// attached, [`run_draft_real`] uses the REAL embedding rows + LM head, so the draft
    /// tokens are the actual model's. `embed` is kept lazy (one row read per token from the
    /// base mmap); `head` `[vocab, d]` is dequantized eagerly to f32 (~2 GB resident, the
    /// only large add — the head matvec touches every row).
    pub fn with_base<P: AsRef<std::path::Path>>(mut self, base_path: P) -> Result<Self> {
        let base_path = base_path.as_ref();
        let gguf = GgufFile::open(base_path)?;
        let off = gguf.tensor_data_offset;
        // Locate by name, tolerating GGUF-converter prefixes and the common llama-style
        // aliases (`token_embd`/`output`) in addition to the DSpark `embed`/`head` names.
        let find = |cands: &[&str]| -> Result<DsparkTensor> {
            gguf.tensors
                .iter()
                .find(|t| {
                    cands.iter().any(|c| {
                        t.name == *c
                            || t.name == format!("dspark.{c}")
                            || t.name == format!("model.{c}")
                            || t.name == format!("base.{c}")
                    })
                })
                .cloned()
                .map(|info| DsparkTensor::new(info, off))
                .ok_or_else(|| anyhow!("base GGUF: missing tensor (tried {cands:?})"))
        };
        let embed = find(&["embed.weight", "token_embd.weight", "tok_embeddings.weight"])?;
        let head = find(&["head.weight", "output.weight", "lm_head.weight"])?;

        let d = self.dims.d;
        let vocab = self.dims.vocab;
        let want = (vocab as u64) * (d as u64);
        if embed.n_elements() != want {
            bail!(
                "base embed.weight n_elements {} != vocab*d {want}",
                embed.n_elements()
            );
        }
        if head.n_elements() != want {
            bail!(
                "base head.weight n_elements {} != vocab*d {want}",
                head.n_elements()
            );
        }

        // Own mmap of the base file (GgufFile drops its mapping after parsing); kept alive
        // for the lazy `embed_row` reads. Head is dequantized to f32 here (~2 GB).
        let mmap = MmapSource::open(base_path)?;
        let head_f32 = dequant_tensor(mmap.as_slice(), &head)?;
        self.base = Some(BaseEmbedHead {
            mmap,
            embed,
            head: head_f32,
            d,
            vocab,
        });
        Ok(self)
    }

    /// Whether a base model (`embed`/`head`) is attached (see [`with_base`]).
    pub fn has_base(&self) -> bool {
        self.base.is_some()
    }

    /// Dequantize ONE routed expert `mtp.{stage}.ffn.experts.{e}.{w1,w2,w3}` from the
    /// mmap on demand. Counts toward the laziness budget.
    fn load_expert(&self, stage: usize, e: usize) -> ExpertSrc<'static> {
        let bytes = self._mmap.as_slice();
        let et = &self.weights.layers[stage].experts[e];
        // Reads are bounds-checked in `dequant_tensor`; the loader already validated
        // every expert tensor exists, so failure here is a corrupt file → panic in test.
        // Q8 GGUF → raw blocks (no dequant); F16/F32 GGUF → dequant to f32. Either way
        // only this one expert is resident; it drops the instant `expert_ffn` returns.
        let w1 = load_matvec(bytes, &et.w1.weight).expect("expert w1 load");
        let w2 = load_matvec(bytes, &et.w2.weight).expect("expert w2 load");
        let w3 = load_matvec(bytes, &et.w3.weight).expect("expert w3 load");
        ExpertSrc::Owned { w1, w2, w3 }
    }

    /// Run the full DSpark draft on the real `mtp.*` block for one `(main_hidden,
    /// anchor, start_pos)` anchor. `embed(tok) -> [d]` returns the BASE-model embedding
    /// row for a token id (not in this GGUF — pass a real base-model lookup for
    /// meaningful tokens, or a synthetic closure to validate the block's *finiteness*).
    /// `head` is the BASE `[vocab, d]` output projection (likewise base-model).
    ///
    /// Returns `(output_ids, base_logits, confidence, experts_dequanted)` — the last
    /// being the count of routed experts read from the mmap (the laziness proof: it must
    /// stay ≤ `n_activated · γ · N_MTP_LAYERS`, far below `256 · N_MTP_LAYERS`).
    pub fn run_draft(
        &self,
        main_hidden: &[f32],
        anchor: u32,
        start_pos: usize,
        embed: impl Fn(u32) -> Vec<f32>,
        head: &[f32],
    ) -> (Vec<u32>, Vec<f32>, Vec<f32>, usize) {
        let dims = self.dims;
        let d = dims.d;

        // Build the dense embed table for ONLY the tokens this round touches
        // ([anchor, NOISE×(γ-1)]) so we never allocate the 2 GB base embedding.
        // `draft_forward` indexes `embed[tok*d..]`, so we lay the needed rows into a
        // full-width table sized to the max touched id.
        let mut ids = vec![dims.noise_token; dims.gamma];
        ids[0] = anchor;
        let max_tok = ids.iter().copied().max().unwrap_or(0) as usize;
        let mut embed_tab = vec![0.0f32; (max_tok + 1) * d];
        for &tok in &ids {
            let row = embed(tok);
            assert_eq!(row.len(), d, "embed(tok) row must be d={d}");
            embed_tab[tok as usize * d..(tok as usize + 1) * d].copy_from_slice(&row);
        }

        let aux_view = DraftAuxF32 {
            main_proj: self.aux.main_proj.view(),
            main_norm: &self.aux.main_norm,
            embed: &embed_tab,
            head: DraftWeight::F32(head), // base-model head (not in the dspark GGUF)
            norm: &self.aux.norm,
            hc_head_fn: &self.aux.hc_head_fn,
            hc_head_scale: &self.aux.hc_head_scale,
            hc_head_base: &self.aux.hc_head_base,
            markov_w1: &self.aux.markov_w1,
            markov_w2: self.aux.markov_w2.view(),
            confidence_proj: &self.aux.confidence_proj,
        };
        let model = DraftModelF32 {
            layers: self.dense_layers.iter().map(|l| l.view()).collect(),
            aux: aux_view,
        };

        let counter = Cell::new(0usize);
        let (ids_out, logits, conf) = draft_forward_with(
            &CpuDispatcher,
            &model,
            main_hidden,
            anchor,
            start_pos,
            &dims,
            |stage, e| {
                counter.set(counter.get() + 1);
                self.load_expert(stage, e)
            },
        );
        (ids_out, logits, conf, counter.get())
    }

    /// Run the draft using the REAL base-model `embed`/`head` attached via [`with_base`].
    /// Same return as [`run_draft`], but the embedding rows are read on demand from the
    /// base mmap and the LM head is the real `head.weight` — so the emitted draft tokens
    /// are the actual model's (meaningful), not the synthetic-weight placeholders.
    ///
    /// Errors if no base GGUF is attached (caller should fall back to [`run_draft`] with a
    /// synthetic embed/head, which is what the existing finiteness tests do).
    pub fn run_draft_real(
        &self,
        main_hidden: &[f32],
        anchor: u32,
        start_pos: usize,
    ) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, usize)> {
        let base = self.base.as_ref().ok_or_else(|| {
            anyhow!("run_draft_real: no base GGUF attached — call with_base(...)")
        })?;
        // Real per-token embedding (lazy row read) + real f32 LM head.
        let embed = |tok: u32| -> Vec<f32> { base.embed_row(tok) };
        Ok(self.run_draft(main_hidden, anchor, start_pos, embed, &base.head))
    }

    /// The drafter dims (`DraftDims::v4_flash()`).
    pub fn dims(&self) -> &DraftDims {
        &self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny synthetic dims so the O(γ·V·r) reference is trivial to reason about.
    const V: usize = 6;
    const R: usize = 3;
    const G: usize = 4;

    #[test]
    fn markov_bias_steers_next_token() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // base logits: uniform 0 everywhere → without bias argmax = token 0.
        let base = vec![0.0f32; G * V];
        // w1: every token's latent = [1,0,0]; w2 row for token 4 = [10,0,0] → bias +10.
        let mut w1 = vec![0.0f32; V * R];
        let mut w2 = vec![0.0f32; V * R];
        for t in 0..V {
            w1[t * R] = 1.0;
        }
        w2[4 * R] = 10.0;
        let out = markov_sample_block(&base, G, V, 0, &w1, DraftWeight::F32(&w2), R);
        assert_eq!(out, vec![4, 4, 4, 4]);
    }

    #[test]
    fn markov_recurrence_uses_previous_sample() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // base favors token = step index; bias is zero → output is just the argmax
        // per position, proving the loop reads base_logits[k] independently.
        let mut base = vec![0.0f32; G * V];
        for k in 0..G {
            base[k * V + (k + 1)] = 5.0; // token k+1 best at step k
        }
        let w1 = vec![0.0f32; V * R];
        let w2 = vec![0.0f32; V * R];
        let out = markov_sample_block(&base, G, V, 0, &w1, DraftWeight::F32(&w2), R);
        assert_eq!(out, vec![1, 2, 3, 4]);
    }

    #[test]
    fn confidence_is_sigmoid_of_linear_feature() {
        let d = 2;
        let gamma = 1;
        let hidden = vec![1.0f32, 0.0]; // h = [1, 0]
        let prev = vec![0u32];
        let mut w1 = vec![0.0f32; V * R];
        w1[0] = 1.0; // markov_w1[0] = [1,0,0]
        let conf_w = vec![0.0f32; d + R]; // w·feat = 0 → c = 0.5
        let c = confidence_block(&hidden, gamma, d, &prev, &w1, R, &conf_w, 0.0);
        assert!((c[0] - 0.5).abs() < 1e-6);

        let mut conf_w2 = vec![0.0f32; d + R];
        conf_w2[0] = 20.0; // weight h0 by +large → c → ~1
        let c2 = confidence_block(&hidden, gamma, d, &prev, &w1, R, &conf_w2, 0.0);
        assert!(c2[0] > 0.999);
    }

    #[test]
    fn prefix_trims_low_confidence_tail() {
        let conf = vec![0.9, 0.8, 0.3, 0.95];
        assert_eq!(confident_prefix_len(&conf, 0.5), 2); // keep [0.9,0.8], drop at 0.3
        assert_eq!(confident_prefix_len(&conf, 0.0), 4); // threshold off ⇒ keep all
        assert_eq!(confident_prefix_len(&conf, 0.99), 0); // all below ⇒ keep none
    }

    #[test]
    fn context_inject_projects_then_rmsnorms() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // d=2, main_proj[2,6] picks the first two of the 3·d concat → proj=[5,7]; norm=1.
        let d = 2;
        let mut main_proj = vec![0.0f32; d * 3 * d];
        main_proj[0] = 1.0; // row0 = e0
        main_proj[3 * d + 1] = 1.0; // row1 = e1
        let h_concat = [5.0, 7.0, 0.0, 0.0, 0.0, 0.0];
        let main_norm = vec![1.0f32; d];
        let out = context_inject(&h_concat, &main_proj, &main_norm, d);
        let s = ((25.0 + 49.0) / 2.0_f32).sqrt(); // rms over [5,7]
        assert!((out[0] - 5.0 / s).abs() < 1e-5);
        assert!((out[1] - 7.0 / s).abs() < 1e-5);
    }

    // ── draft block forward: HC, attention (main_kv injection), full compose ──

    /// Deterministic small filler so synthetic weights are non-degenerate but bounded.
    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as u32 as f32 / (1u64 << 24) as f32 - 0.5) * 0.2
            })
            .collect()
    }

    #[test]
    fn hc_collapse_then_expand_is_hand_verifiable() {
        // n_hc=2, d=3, zero hc_fn + zero base + zero scale ⇒ mix=0 everywhere ⇒
        //   pre  = sigmoid(0)+eps = 0.5+eps      (both copies)
        //   post = 2·sigmoid(0)   = 1.0          (both copies)
        //   comb = sinkhorn(uniform) = 0.5 in every cell (doubly stochastic 2×2).
        let nhc = 2;
        let d = 3;
        let mut dims = tiny_dims();
        dims.hc_mult = nhc;
        dims.d = d;
        dims.hc_sinkhorn_iters = 5;
        let hc_fn = vec![0.0f32; dims.mix_hc() * nhc * d];
        let hc_scale = vec![0.0f32; 3];
        let hc_base = vec![0.0f32; dims.mix_hc()];
        // Two identical copies [1,2,3 | 1,2,3].
        let x_hc = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let (col, post, comb) = hc_collapse(&x_hc, &hc_fn, &hc_scale, &hc_base, nhc, d, &dims);
        // collapsed[j] = (0.5+eps)·x[j] + (0.5+eps)·x[j] = (1+2eps)·x[j] ≈ x[j].
        for (j, &want) in [1.0f32, 2.0, 3.0].iter().enumerate() {
            assert!(
                (col[j] - want).abs() < 1e-3,
                "col[{j}]={} != {want}",
                col[j]
            );
        }
        assert!(post.iter().all(|&p| (p - 1.0).abs() < 1e-4));
        assert!(comb.iter().all(|&c| (c - 0.5).abs() < 1e-4));

        // hc_expand: out[dst,e] = post[dst]·new + Σ_src comb·resid = new[e] + x[e].
        let new_x = vec![10.0, 20.0, 30.0];
        let out = hc_expand(&new_x, &x_hc, &post, &comb, nhc, d);
        for c in 0..nhc {
            for (e, (&n, &xe)) in new_x.iter().zip([1.0f32, 2.0, 3.0].iter()).enumerate() {
                let got = out[c * d + e];
                assert!((got - (n + xe)).abs() < 1e-3, "out[{c},{e}]={got}");
            }
        }
    }

    #[test]
    fn attention_injects_main_kv_and_respects_sink() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let dims = tiny_dims();
        let owned = OwnedLayer::new(&dims, 7);
        let layer = owned.view(&dims);
        let g = dims.gamma;
        let x_rows = rnd(g * dims.d, 11);

        // (a) main_kv injection: changing main_x must change the attention output.
        let main_a = rnd(dims.d, 101);
        let main_b = rnd(dims.d, 202);
        let oa = attention(&CpuDispatcher, &x_rows, &main_a, &layer, &dims, 4);
        let ob = attention(&CpuDispatcher, &x_rows, &main_b, &layer, &dims, 4);
        assert!(oa.iter().all(|v| v.is_finite()) && ob.iter().all(|v| v.is_finite()));
        let diff: f32 = oa.iter().zip(&ob).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-6,
            "main_kv not injected: outputs identical (diff={diff})"
        );

        // (b) a huge per-head attn_sink drives all softmax mass onto the (zero-value)
        // sink ⇒ attention output → ~0 (before the o-proj, which is linear in o).
        let mut sink_owned = OwnedLayer::new(&dims, 7);
        sink_owned.attn_sink = vec![1e9; dims.n_heads];
        let sink_layer = sink_owned.view(&dims);
        let o_sink = attention(&CpuDispatcher, &x_rows, &main_a, &sink_layer, &dims, 4);
        assert!(
            o_sink.iter().all(|v| v.abs() < 1e-3),
            "sink did not suppress attn"
        );
    }

    #[test]
    fn draft_forward_full_compose_shapes_and_finiteness() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let dims = tiny_dims();
        let model = OwnedModel::new(&dims, 3);
        let view = model.view(&dims);
        let main_hidden = rnd(3 * dims.d, 55);
        let anchor = 2u32;
        let (out_ids, base_logits, conf) = draft_forward(&view, &main_hidden, anchor, 6, &dims);

        assert_eq!(out_ids.len(), dims.gamma);
        assert!(out_ids.iter().all(|&t| (t as usize) < dims.vocab));
        assert_eq!(base_logits.len(), dims.gamma * dims.vocab);
        assert!(base_logits.iter().all(|v| v.is_finite()));
        assert_eq!(conf.len(), dims.gamma);
        assert!(conf.iter().all(|&c| c > 0.0 && c < 1.0 && c.is_finite()));
    }

    #[test]
    fn draft_forward_zero_confidence_proj_gives_half() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // With confidence_proj = 0 (and bias 0), every accept prob is σ(0) = 0.5 exactly,
        // independent of the hidden state — a hand-verifiable degenerate case.
        let dims = tiny_dims();
        let mut model = OwnedModel::new(&dims, 9);
        model.aux.confidence_proj = vec![0.0f32; dims.d + dims.markov_rank];
        let view = model.view(&dims);
        let main_hidden = rnd(3 * dims.d, 77);
        let (_ids, _logits, conf) = draft_forward(&view, &main_hidden, 1, 6, &dims);
        assert!(
            conf.iter().all(|&c| (c - 0.5).abs() < 1e-6),
            "conf={conf:?}"
        );
    }

    #[test]
    fn fp8_e4m3_dequant_basics() {
        // 0x00 = +0; 0x38 = exp 7 (bias 7 ⇒ 2^0), mant 0 ⇒ 1.0; sign bit ⇒ -1.0.
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
        assert!((fp8_e4m3_to_f32(0x38) - 1.0).abs() < 1e-6);
        assert!((fp8_e4m3_to_f32(0xb8) + 1.0).abs() < 1e-6);
        assert!(fp8_e4m3_to_f32(0x7f).is_nan()); // exp=15, mant=7
                                                 // ue8m0: byte 127 ⇒ 2^0 = 1.
        assert!((ue8m0_scale(127) - 1.0).abs() < 1e-6);
        assert!((ue8m0_scale(128) - 2.0).abs() < 1e-6);
        // blockwise: weight 0x38 (=1.0) × scale 2^1 ⇒ 2.0.
        let w = vec![0x38u8; 4];
        let s = vec![128u8; 1];
        let y = dequant_fp8_blockwise(&w, &s, 2, 2, 128);
        assert!(y.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    // ── f32-vs-q8 block gate: the QUANTIZED draft must match the f32 draft ──
    //
    // Quantize EVERY matvec operand of a tiny model to `block_q8_0`, run the draft block
    // once with all-`F32` weights and once with the SAME weights as `Q8`, and assert the
    // head logits agree within the q8 quantization granularity and the greedy ids agree.
    // This is the synthetic correctness gate for the quantized DSpark drafter (run here on
    // `CpuDispatcher`; the Metal variant lives in `dspark_q8_block_metal_smoke`).

    /// Round-to-nearest f32→f16 bits (normal range; synthetic test magnitudes).
    pub(super) fn f32_to_f16_bits(x: f32) -> u16 {
        let b = x.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
        let mant = b & 0x7f_ffff;
        if exp <= 0 {
            return sign;
        }
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        let mut h = sign | ((exp as u16) << 10) | ((mant >> 13) as u16);
        if (mant & 0x1000) != 0 {
            h += 1;
        }
        h
    }

    /// GGUF `block_q8_0` bytes for a row-major weight whose row length is a multiple of 32
    /// (34 B/block = f16 scale `amax/127` + 32×int8). Matches the converter + `matvec_q8`.
    pub(super) fn q8_quant(w: &[f32]) -> Vec<u8> {
        assert_eq!(w.len() % 32, 0, "q8_quant len {} % 32 != 0", w.len());
        let mut out = Vec::with_capacity(w.len() / 32 * 34);
        for blk in w.chunks_exact(32) {
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let db = f32_to_f16_bits(d);
            out.push((db & 0xff) as u8);
            out.push((db >> 8) as u8);
            for &v in blk {
                out.push((v * id).round().clamp(-128.0, 127.0) as i8 as u8);
            }
        }
        out
    }

    /// q8-compatible tiny dims: every matvec `[d_out, d_in]` has `d_out % 2 == 0` AND
    /// `d_in % 32 == 0` (the `matvec_q8` constraint), so all operands can be quantized.
    pub(super) fn q8_dims() -> DraftDims {
        DraftDims {
            d: 64,
            n_heads: 2,
            head_dim: 32,
            rope_head_dim: 32,
            q_lora_rank: 32,
            o_lora_rank: 32,
            o_groups: 2,
            n_experts: 4,
            n_activated: 2,
            route_scale: 1.0,
            inter: 64,
            hc_mult: 2,
            gamma: 3,
            noise_token: 5,
            vocab: 64,
            markov_rank: 32,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 4,
        }
    }

    struct Q8Expert {
        w1: Vec<u8>,
        w2: Vec<u8>,
        w3: Vec<u8>,
    }
    impl Q8Expert {
        fn from(e: &OwnedExpert) -> Self {
            Self {
                w1: q8_quant(&e.w1),
                w2: q8_quant(&e.w2),
                w3: q8_quant(&e.w3),
            }
        }
        fn view(&self) -> ExpertF32<'_> {
            ExpertF32 {
                w1: DraftWeight::Q8(&self.w1),
                w2: DraftWeight::Q8(&self.w2),
                w3: DraftWeight::Q8(&self.w3),
            }
        }
    }

    /// q8 mirror of `OwnedLayer`: matvec operands as `block_q8_0` bytes, norms cloned f32.
    struct Q8Layer {
        wq_a: Vec<u8>,
        wq_b: Vec<u8>,
        wkv: Vec<u8>,
        wo_a: Vec<u8>,
        wo_b: Vec<u8>,
        gate_w: Vec<u8>,
        q_norm: Vec<f32>,
        kv_norm: Vec<f32>,
        attn_sink: Vec<f32>,
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        gate_b: Vec<f32>,
        experts: Vec<Q8Expert>,
        shared: Q8Expert,
        hc_attn_fn: Vec<f32>,
        hc_attn_scale: Vec<f32>,
        hc_attn_base: Vec<f32>,
        hc_ffn_fn: Vec<f32>,
        hc_ffn_scale: Vec<f32>,
        hc_ffn_base: Vec<f32>,
    }
    impl Q8Layer {
        fn from(l: &OwnedLayer) -> Self {
            Self {
                wq_a: q8_quant(&l.wq_a),
                wq_b: q8_quant(&l.wq_b),
                wkv: q8_quant(&l.wkv),
                wo_a: q8_quant(&l.wo_a),
                wo_b: q8_quant(&l.wo_b),
                gate_w: q8_quant(&l.gate_w),
                q_norm: l.q_norm.clone(),
                kv_norm: l.kv_norm.clone(),
                attn_sink: l.attn_sink.clone(),
                attn_norm: l.attn_norm.clone(),
                ffn_norm: l.ffn_norm.clone(),
                gate_b: l.gate_b.clone(),
                experts: l.experts.iter().map(Q8Expert::from).collect(),
                shared: Q8Expert::from(&l.shared),
                hc_attn_fn: l.hc_attn_fn.clone(),
                hc_attn_scale: l.hc_attn_scale.clone(),
                hc_attn_base: l.hc_attn_base.clone(),
                hc_ffn_fn: l.hc_ffn_fn.clone(),
                hc_ffn_scale: l.hc_ffn_scale.clone(),
                hc_ffn_base: l.hc_ffn_base.clone(),
            }
        }
        fn view(&self) -> DraftLayerF32<'_> {
            DraftLayerF32 {
                wq_a: DraftWeight::Q8(&self.wq_a),
                wq_b: DraftWeight::Q8(&self.wq_b),
                wkv: DraftWeight::Q8(&self.wkv),
                wo_a: DraftWeight::Q8(&self.wo_a),
                wo_b: DraftWeight::Q8(&self.wo_b),
                q_norm: &self.q_norm,
                kv_norm: &self.kv_norm,
                attn_sink: &self.attn_sink,
                attn_norm: &self.attn_norm,
                ffn_norm: &self.ffn_norm,
                gate_w: DraftWeight::Q8(&self.gate_w),
                gate_b: &self.gate_b,
                experts: self.experts.iter().map(|e| e.view()).collect(),
                shared: self.shared.view(),
                hc_attn_fn: &self.hc_attn_fn,
                hc_attn_scale: &self.hc_attn_scale,
                hc_attn_base: &self.hc_attn_base,
                hc_ffn_fn: &self.hc_ffn_fn,
                hc_ffn_scale: &self.hc_ffn_scale,
                hc_ffn_base: &self.hc_ffn_base,
            }
        }
    }

    struct Q8Aux {
        main_proj: Vec<u8>,
        markov_w2: Vec<u8>,
        head: Vec<u8>,
        main_norm: Vec<f32>,
        embed: Vec<f32>,
        norm: Vec<f32>,
        hc_head_fn: Vec<f32>,
        hc_head_scale: Vec<f32>,
        hc_head_base: Vec<f32>,
        markov_w1: Vec<f32>,
        confidence_proj: Vec<f32>,
    }
    impl Q8Aux {
        fn from(a: &OwnedAux) -> Self {
            Self {
                main_proj: q8_quant(&a.main_proj),
                markov_w2: q8_quant(&a.markov_w2),
                head: q8_quant(&a.head),
                main_norm: a.main_norm.clone(),
                embed: a.embed.clone(),
                norm: a.norm.clone(),
                hc_head_fn: a.hc_head_fn.clone(),
                hc_head_scale: a.hc_head_scale.clone(),
                hc_head_base: a.hc_head_base.clone(),
                markov_w1: a.markov_w1.clone(),
                confidence_proj: a.confidence_proj.clone(),
            }
        }
        fn view(&self) -> DraftAuxF32<'_> {
            DraftAuxF32 {
                main_proj: DraftWeight::Q8(&self.main_proj),
                main_norm: &self.main_norm,
                embed: &self.embed,
                head: DraftWeight::Q8(&self.head),
                norm: &self.norm,
                hc_head_fn: &self.hc_head_fn,
                hc_head_scale: &self.hc_head_scale,
                hc_head_base: &self.hc_head_base,
                markov_w1: &self.markov_w1,
                markov_w2: DraftWeight::Q8(&self.markov_w2),
                confidence_proj: &self.confidence_proj,
            }
        }
    }

    struct Q8Model {
        layers: Vec<Q8Layer>,
        aux: Q8Aux,
    }
    impl Q8Model {
        fn from(m: &OwnedModel) -> Self {
            Self {
                layers: m.layers.iter().map(Q8Layer::from).collect(),
                aux: Q8Aux::from(&m.aux),
            }
        }
        fn view(&self) -> DraftModelF32<'_> {
            DraftModelF32 {
                layers: self.layers.iter().map(|l| l.view()).collect(),
                aux: self.aux.view(),
            }
        }
    }

    #[test]
    fn q8_block_matches_f32_within_quant_tolerance() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY).
        // Gate-flipping tests take write(); we take read() so the gate cannot change
        // mid-test, while staying parallel with other readers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let dims = q8_dims();
        let owned = OwnedModel::new(&dims, 3);
        let q8 = Q8Model::from(&owned);

        let main_hidden = rnd(3 * dims.d, 55);
        let anchor = 2u32;
        let start_pos = 6usize;

        let f32_view = owned.view(&dims);
        let (ids_f32, logits_f32, conf_f32) = draft_forward_with(
            &CpuDispatcher,
            &f32_view,
            &main_hidden,
            anchor,
            start_pos,
            &dims,
            |li, e| ExpertSrc::Borrowed(f32_view.layers[li].experts[e]),
        );

        let q8_view = q8.view();
        let (ids_q8, logits_q8, conf_q8) = draft_forward_with(
            &CpuDispatcher,
            &q8_view,
            &main_hidden,
            anchor,
            start_pos,
            &dims,
            |li, e| ExpertSrc::Borrowed(q8_view.layers[li].experts[e]),
        );

        let max_logit = logits_f32.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let logit_err = logits_f32
            .iter()
            .zip(&logits_q8)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let conf_err = conf_f32
            .iter()
            .zip(&conf_q8)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let id_agree = ids_f32.iter().zip(&ids_q8).filter(|(a, b)| a == b).count();
        eprintln!(
            "[q8_block] max|logit|={max_logit:.3e} logit max|Δ|={logit_err:.3e} \
             conf max|Δ|={conf_err:.3e} ids_f32={ids_f32:?} ids_q8={ids_q8:?} \
             ({id_agree}/{} agree)",
            ids_f32.len()
        );

        // q8 weights have ~1/256 relative granularity; over the multi-matvec block the
        // accumulated logit error stays a small fraction of the logit magnitude.
        assert!(logits_q8.iter().all(|v| v.is_finite()));
        assert!(conf_q8
            .iter()
            .all(|&c| c.is_finite() && (0.0..=1.0).contains(&c)));
        let rel = logit_err / max_logit.max(1e-6);
        assert!(
            rel < 0.05,
            "q8 logit drift {logit_err:.3e} ({rel:.2}× of |logit|) too large"
        );
        assert!(
            conf_err < 5e-2,
            "q8 confidence drift {conf_err:.3e} too large"
        );
        // Greedy ids: well-separated synthetic argmaxes ⇒ they agree. If a near-tie ever
        // flipped, the logits-close assertion above still pins correctness (documented).
        assert_eq!(
            ids_f32, ids_q8,
            "q8 greedy ids diverged (logits were close)"
        );
    }

    // ── tiny synthetic model plumbing for the draft-forward tests ──

    fn tiny_dims() -> DraftDims {
        DraftDims {
            d: 4,
            n_heads: 2,
            head_dim: 4,
            rope_head_dim: 2,
            q_lora_rank: 3,
            o_lora_rank: 2,
            o_groups: 1,
            n_experts: 3,
            n_activated: 2,
            route_scale: 1.0,
            inter: 4,
            hc_mult: 2,
            gamma: 3,
            noise_token: 4,
            vocab: 5,
            markov_rank: 2,
            rope_theta: 10000.0,
            rms_eps: 1e-6,
            hc_eps: 1e-6,
            hc_sinkhorn_iters: 4,
        }
    }

    struct OwnedExpert {
        w1: Vec<f32>,
        w2: Vec<f32>,
        w3: Vec<f32>,
    }
    impl OwnedExpert {
        fn new(d: usize, inter: usize, seed: u64) -> Self {
            Self {
                w1: rnd(inter * d, seed),
                w2: rnd(d * inter, seed ^ 1),
                w3: rnd(inter * d, seed ^ 2),
            }
        }
        fn view(&self) -> ExpertF32 {
            ExpertF32 {
                w1: DraftWeight::F32(&self.w1),
                w2: DraftWeight::F32(&self.w2),
                w3: DraftWeight::F32(&self.w3),
            }
        }
    }

    struct OwnedLayer {
        wq_a: Vec<f32>,
        wq_b: Vec<f32>,
        wkv: Vec<f32>,
        wo_a: Vec<f32>,
        wo_b: Vec<f32>,
        q_norm: Vec<f32>,
        kv_norm: Vec<f32>,
        attn_sink: Vec<f32>,
        attn_norm: Vec<f32>,
        ffn_norm: Vec<f32>,
        gate_w: Vec<f32>,
        gate_b: Vec<f32>,
        experts: Vec<OwnedExpert>,
        shared: OwnedExpert,
        hc_attn_fn: Vec<f32>,
        hc_attn_scale: Vec<f32>,
        hc_attn_base: Vec<f32>,
        hc_ffn_fn: Vec<f32>,
        hc_ffn_scale: Vec<f32>,
        hc_ffn_base: Vec<f32>,
    }
    impl OwnedLayer {
        fn new(dims: &DraftDims, seed: u64) -> Self {
            let (d, nh, hd) = (dims.d, dims.n_heads, dims.head_dim);
            let gd = nh * hd / dims.o_groups;
            let mix = dims.mix_hc();
            Self {
                wq_a: rnd(dims.q_lora_rank * d, seed),
                wq_b: rnd(nh * hd * dims.q_lora_rank, seed ^ 3),
                wkv: rnd(hd * d, seed ^ 5),
                wo_a: rnd(dims.o_groups * dims.o_lora_rank * gd, seed ^ 7),
                wo_b: rnd(d * dims.o_groups * dims.o_lora_rank, seed ^ 9),
                q_norm: vec![1.0; dims.q_lora_rank],
                kv_norm: vec![1.0; hd],
                attn_sink: rnd(nh, seed ^ 11),
                attn_norm: vec![1.0; d],
                ffn_norm: vec![1.0; d],
                gate_w: rnd(dims.n_experts * d, seed ^ 13),
                gate_b: rnd(dims.n_experts, seed ^ 15),
                experts: (0..dims.n_experts)
                    .map(|e| OwnedExpert::new(d, dims.inter, seed ^ (100 + e as u64)))
                    .collect(),
                shared: OwnedExpert::new(d, dims.inter, seed ^ 999),
                hc_attn_fn: rnd(mix * dims.hc_mult * d, seed ^ 17),
                hc_attn_scale: rnd(3, seed ^ 19),
                hc_attn_base: rnd(mix, seed ^ 21),
                hc_ffn_fn: rnd(mix * dims.hc_mult * d, seed ^ 23),
                hc_ffn_scale: rnd(3, seed ^ 25),
                hc_ffn_base: rnd(mix, seed ^ 27),
            }
        }
        fn view(&self, _dims: &DraftDims) -> DraftLayerF32 {
            DraftLayerF32 {
                wq_a: DraftWeight::F32(&self.wq_a),
                wq_b: DraftWeight::F32(&self.wq_b),
                wkv: DraftWeight::F32(&self.wkv),
                wo_a: DraftWeight::F32(&self.wo_a),
                wo_b: DraftWeight::F32(&self.wo_b),
                q_norm: &self.q_norm,
                kv_norm: &self.kv_norm,
                attn_sink: &self.attn_sink,
                attn_norm: &self.attn_norm,
                ffn_norm: &self.ffn_norm,
                gate_w: DraftWeight::F32(&self.gate_w),
                gate_b: &self.gate_b,
                experts: self.experts.iter().map(|e| e.view()).collect(),
                shared: self.shared.view(),
                hc_attn_fn: &self.hc_attn_fn,
                hc_attn_scale: &self.hc_attn_scale,
                hc_attn_base: &self.hc_attn_base,
                hc_ffn_fn: &self.hc_ffn_fn,
                hc_ffn_scale: &self.hc_ffn_scale,
                hc_ffn_base: &self.hc_ffn_base,
            }
        }
    }

    struct OwnedAux {
        main_proj: Vec<f32>,
        main_norm: Vec<f32>,
        embed: Vec<f32>,
        head: Vec<f32>,
        norm: Vec<f32>,
        hc_head_fn: Vec<f32>,
        hc_head_scale: Vec<f32>,
        hc_head_base: Vec<f32>,
        markov_w1: Vec<f32>,
        markov_w2: Vec<f32>,
        confidence_proj: Vec<f32>,
    }
    impl OwnedAux {
        fn new(dims: &DraftDims, seed: u64) -> Self {
            let d = dims.d;
            Self {
                main_proj: rnd(d * 3 * d, seed),
                main_norm: vec![1.0; d],
                embed: rnd(dims.vocab * d, seed ^ 31),
                head: rnd(dims.vocab * d, seed ^ 33),
                norm: vec![1.0; d],
                hc_head_fn: rnd(dims.hc_mult * dims.hc_mult * d, seed ^ 35),
                hc_head_scale: rnd(1, seed ^ 37),
                hc_head_base: rnd(dims.hc_mult, seed ^ 39),
                markov_w1: rnd(dims.vocab * dims.markov_rank, seed ^ 41),
                markov_w2: rnd(dims.vocab * dims.markov_rank, seed ^ 43),
                confidence_proj: rnd(d + dims.markov_rank, seed ^ 45),
            }
        }
        fn view(&self) -> DraftAuxF32 {
            DraftAuxF32 {
                main_proj: DraftWeight::F32(&self.main_proj),
                main_norm: &self.main_norm,
                embed: &self.embed,
                head: DraftWeight::F32(&self.head),
                norm: &self.norm,
                hc_head_fn: &self.hc_head_fn,
                hc_head_scale: &self.hc_head_scale,
                hc_head_base: &self.hc_head_base,
                markov_w1: &self.markov_w1,
                markov_w2: DraftWeight::F32(&self.markov_w2),
                confidence_proj: &self.confidence_proj,
            }
        }
    }

    struct OwnedModel {
        layers: Vec<OwnedLayer>,
        aux: OwnedAux,
    }
    impl OwnedModel {
        fn new(dims: &DraftDims, seed: u64) -> Self {
            Self {
                layers: (0..consts::N_MTP_LAYERS)
                    .map(|i| OwnedLayer::new(dims, seed ^ (1000 + i as u64)))
                    .collect(),
                aux: OwnedAux::new(dims, seed ^ 5000),
            }
        }
        fn view(&self, dims: &DraftDims) -> DraftModelF32 {
            DraftModelF32 {
                layers: self.layers.iter().map(|l| l.view(dims)).collect(),
                aux: self.aux.view(),
            }
        }
    }

    // ── loader: the targeted tensor names are the REAL mtp.{0,1,2}.* scheme ──

    #[test]
    fn loader_targets_real_tensor_names() {
        let names = dspark_tensor_names();
        let has = |s: &str| names.iter().any(|n| n == s);
        // mtp.0: MLA + ctx projection + experts + hc.
        assert!(has("mtp.0.attn.wq_a.weight"));
        assert!(has("mtp.0.attn.wq_b.weight"));
        assert!(has("mtp.0.attn.wkv.weight"));
        assert!(has("mtp.0.attn.wo_a.weight"));
        assert!(has("mtp.0.attn.wo_b.weight"));
        assert!(has("mtp.0.attn.q_norm.weight"));
        assert!(has("mtp.0.attn.kv_norm.weight"));
        assert!(has("mtp.0.attn.attn_sink"));
        assert!(has("mtp.0.attn_norm.weight"));
        assert!(has("mtp.0.ffn_norm.weight"));
        assert!(has("mtp.0.ffn.experts.0.w1.weight"));
        assert!(has("mtp.0.ffn.experts.255.w3.weight"));
        assert!(has("mtp.0.hc_attn_base") && has("mtp.0.hc_ffn_scale"));
        assert!(has("mtp.0.main_proj.weight") && has("mtp.0.main_norm.weight"));
        // mtp.1: a middle stage — block only, no main_*/head extras.
        assert!(has("mtp.1.attn.wkv.weight") && has("mtp.1.ffn.experts.0.w2.weight"));
        assert!(!has("mtp.1.main_proj.weight"));
        assert!(!has("mtp.1.markov_head.markov_w1.weight"));
        // mtp.2 (last stage): the head extras live here, NOT on mtp.0/1.
        assert!(has("mtp.2.norm.weight"));
        assert!(has("mtp.2.markov_head.markov_w1.weight"));
        assert!(has("mtp.2.markov_head.markov_w2.weight"));
        assert!(has("mtp.2.confidence_head.proj.weight"));
        assert!(has("mtp.2.hc_head_base") && has("mtp.2.hc_head_scale"));
        // embed/head are SHARED from the base model — never in the dspark loader.
        assert!(!has("embed.weight") && !has("head.weight"));
        // exactly N_MTP_LAYERS stages × 256 experts each are targeted.
        let expert_w1 = names
            .iter()
            .filter(|n| n.ends_with(".w1.weight") && n.contains(".experts."))
            .count();
        assert_eq!(expert_w1, consts::N_MTP_LAYERS * consts::N_EXPERTS);
    }

    /// OPT-IN real-weight run on the 37 GB DSpark GGUF (memory-bounded lazy path).
    /// `DS4_DSPARK_GGUF=/path/to/dspark.gguf cargo test -p engine_core dspark_real -- --nocapture`.
    ///
    /// Loads `DraftRealWeights` (dense eager, routed experts lazy), runs the real
    /// `mtp.*` block on a tiny anchor with SYNTHETIC base embed/head (the base model is
    /// NOT in this GGUF — so the *tokens* are not meaningful, but the block forward +
    /// main_proj/markov/confidence/hc must produce FINITE output). Asserts finiteness,
    /// id/conf ranges, AND the laziness budget (experts dequanted ≪ 256·N_MTP_LAYERS).
    #[test]
    fn dspark_real_lazy_draft_is_finite_and_bounded() {
        let Ok(path) = std::env::var("DS4_DSPARK_GGUF") else {
            eprintln!("DS4_DSPARK_GGUF unset — skipping dspark_real lazy-draft test");
            return;
        };
        let rw = DraftRealWeights::load(&path).expect("DraftRealWeights::load");
        let dims = *rw.dims();
        assert_eq!(dims.gamma, consts::BLOCK_SIZE);
        let d = dims.d;
        let vocab = dims.vocab;

        // Synthetic, small, deterministic base embed/head (the base model is absent).
        // Small magnitude keeps activations finite; correctness is NOT claimed.
        let synth = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed | 1;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 40) as u32 as f32 / (1u64 << 24) as f32 - 0.5) * 0.04
                })
                .collect()
        };
        // head [vocab, d] (~2 GB f32) — the only large synthetic alloc; needed to emit
        // γ×V logits. Far under the ~30 GB budget alongside the few-GB dense weights.
        let head = synth(vocab * d, 0xDEAD_BEEF_u64);
        // Per-token embed closure → no 2 GB embed table.
        let embed = |tok: u32| -> Vec<f32> { synth(d, 0x5151 ^ tok as u64) };

        let main_hidden = synth(3 * d, 0xABCD);
        let anchor = 1234u32;
        let (ids, logits, conf, n_experts) = rw.run_draft(&main_hidden, anchor, 7, embed, &head);

        // shape + finiteness
        assert_eq!(ids.len(), dims.gamma);
        assert_eq!(logits.len(), dims.gamma * vocab);
        assert_eq!(conf.len(), dims.gamma);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
        assert!(ids.iter().all(|&t| (t as usize) < vocab), "id out of vocab");
        // Confidences must be valid probabilities. With the REAL confidence_proj fed by
        // SYNTHETIC base embed/head, the sigmoid can saturate to exactly 0.0/1.0 in f32
        // (the activations are unrealistic) — so the gate is finite + [0,1], not the
        // open (0,1) that only holds for non-saturating (real-base-model) inputs.
        assert!(
            conf.iter()
                .all(|&c| c.is_finite() && (0.0..=1.0).contains(&c)),
            "conf not a valid probability: {conf:?}"
        );

        // laziness: ≤ n_activated · γ · N_MTP_LAYERS, and ≪ all 256·N_MTP_LAYERS.
        let budget = dims.n_activated * dims.gamma * consts::N_MTP_LAYERS;
        let all = consts::N_EXPERTS * consts::N_MTP_LAYERS;
        assert!(
            n_experts <= budget,
            "dequanted {n_experts} experts > budget {budget}"
        );
        assert!(n_experts < all, "not lazy: {n_experts} ≥ all {all}");
        eprintln!(
            "[dspark_real] FINITE ✓  ids={ids:?}  conf={conf:?}  experts_dequanted={n_experts} (budget {budget}, all {all})"
        );
    }

    /// OPT-IN real-weight run on the QUANTIZED (q8_0) DSpark GGUF produced by
    /// `convert_dspark_to_gguf.py --q8`. `DS4_DSPARK_Q8_GGUF=/path/to/dspark_q8.gguf
    /// cargo test -p engine_core dspark_real_q8 -- --nocapture`.
    ///
    /// Same memory-bounded lazy path as `dspark_real_lazy_draft_is_finite_and_bounded`,
    /// but the matvec operands are loaded as raw `block_q8_0` bytes (`OwnedDraftWeight::Q8`)
    /// and dispatched through `matvec_q8` — proving the q8 drafter runs end-to-end (FINITE
    /// output, valid id/conf ranges) without ever materializing the f32 weights, and with
    /// the routed experts read lazily (bounded count). Correctness is NOT claimed (the base
    /// embed/head are synthetic); the f32-vs-q8 numeric gate is `q8_block_matches_f32_*`.
    #[test]
    fn dspark_real_q8_lazy_draft_is_finite_and_bounded() {
        let Ok(path) = std::env::var("DS4_DSPARK_Q8_GGUF") else {
            eprintln!("DS4_DSPARK_Q8_GGUF unset — skipping dspark_real_q8 lazy-draft test");
            return;
        };
        let rw = DraftRealWeights::load(&path).expect("DraftRealWeights::load (q8)");
        let dims = *rw.dims();
        assert_eq!(dims.gamma, consts::BLOCK_SIZE);
        let (d, vocab) = (dims.d, dims.vocab);

        // Small deterministic synthetic base embed/head (base model absent — see the f16
        // sibling test). Keeps activations finite; correctness not claimed.
        let synth = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed | 1;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((s >> 40) as u32 as f32 / (1u64 << 24) as f32 - 0.5) * 0.04
                })
                .collect()
        };
        let head = synth(vocab * d, 0xC0FFEE_u64);
        let embed = |tok: u32| -> Vec<f32> { synth(d, 0x7777 ^ tok as u64) };

        let main_hidden = synth(3 * d, 0xBEEF);
        let anchor = 4321u32;
        let (ids, logits, conf, n_experts) = rw.run_draft(&main_hidden, anchor, 7, embed, &head);

        assert_eq!(ids.len(), dims.gamma);
        assert_eq!(logits.len(), dims.gamma * vocab);
        assert_eq!(conf.len(), dims.gamma);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite q8 logit");
        assert!(
            ids.iter().all(|&t| (t as usize) < vocab),
            "q8 id out of vocab"
        );
        assert!(
            conf.iter()
                .all(|&c| c.is_finite() && (0.0..=1.0).contains(&c)),
            "q8 conf not a valid probability: {conf:?}"
        );

        let budget = dims.n_activated * dims.gamma * consts::N_MTP_LAYERS;
        let all = consts::N_EXPERTS * consts::N_MTP_LAYERS;
        assert!(
            n_experts <= budget,
            "q8 dequanted {n_experts} experts > budget {budget}"
        );
        assert!(n_experts < all, "q8 not lazy: {n_experts} ≥ all {all}");
        eprintln!(
            "[dspark_real_q8] FINITE ✓  ids={ids:?}  conf={conf:?}  experts_read={n_experts} (budget {budget}, all {all})"
        );
    }

    /// OPT-IN end-to-end REAL draft: attach the base model's `embed.weight` +
    /// `head.weight` (via `with_base`) so the drafter runs with the ACTUAL model's
    /// embedding/LM-head — the emitted draft token ids are now meaningful, not synthetic.
    ///
    /// `DS4_DSPARK_BASE_GGUF=/path/to/dspark_base.gguf
    ///  DS4_DSPARK_Q8_GGUF=/path/to/dspark_q8.gguf  (or DS4_DSPARK_GGUF)
    ///  cargo test -p engine_core dspark_real_with_base -- --nocapture`.
    ///
    /// Memory-bounded: routed experts lazy, embed read one row at a time, head ~2 GB f32
    /// the only large add (peak well under ~12 GB). Asserts ids finite & `< vocab`, conf a
    /// valid probability, laziness budget; PRINTS the real-embed/head draft ids.
    #[test]
    fn dspark_real_with_base_emits_meaningful_ids() {
        let Ok(base_path) = std::env::var("DS4_DSPARK_BASE_GGUF") else {
            eprintln!("DS4_DSPARK_BASE_GGUF unset — skipping dspark_real_with_base test");
            return;
        };
        // Module GGUF: prefer the q8 module, else the f16 one.
        let module =
            std::env::var("DS4_DSPARK_Q8_GGUF").or_else(|_| std::env::var("DS4_DSPARK_GGUF"));
        let Ok(module_path) = module else {
            eprintln!(
                "DS4_DSPARK_Q8_GGUF/DS4_DSPARK_GGUF unset — skipping dspark_real_with_base test"
            );
            return;
        };

        let rw = DraftRealWeights::load(&module_path)
            .expect("DraftRealWeights::load (module)")
            .with_base(&base_path)
            .expect("DraftRealWeights::with_base");
        assert!(rw.has_base(), "base model should be attached");
        let dims = *rw.dims();
        let vocab = dims.vocab;

        // A real anchor token id (well inside vocab). main_hidden is the only synthetic
        // input here — the upstream V4-Flash context hidden state isn't available offline;
        // embed + head are the REAL model's, so the drafter is end-to-end real from the
        // anchor through the markov head.
        let mut s = 0x1234_5678u64;
        let main_hidden: Vec<f32> = (0..3 * dims.d)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as u32 as f32 / (1u64 << 24) as f32 - 0.5) * 0.04
            })
            .collect();
        let anchor = 100u32; // a real base-vocab token id

        let (ids, logits, conf, n_experts) = rw
            .run_draft_real(&main_hidden, anchor, 7)
            .expect("run_draft_real");

        assert_eq!(ids.len(), dims.gamma);
        assert_eq!(logits.len(), dims.gamma * vocab);
        assert_eq!(conf.len(), dims.gamma);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
        assert!(
            ids.iter().all(|&t| (t as usize) < vocab),
            "draft id out of vocab"
        );
        assert!(
            conf.iter()
                .all(|&c| c.is_finite() && (0.0..=1.0).contains(&c)),
            "conf not a valid probability: {conf:?}"
        );

        let budget = dims.n_activated * dims.gamma * consts::N_MTP_LAYERS;
        let all = consts::N_EXPERTS * consts::N_MTP_LAYERS;
        assert!(
            n_experts <= budget,
            "dequanted {n_experts} experts > budget {budget}"
        );
        assert!(n_experts < all, "not lazy: {n_experts} ≥ all {all}");

        eprintln!(
            "[dspark_real_with_base] REAL embed/head ✓  anchor={anchor}\n  \
             draft ids (MEANINGFUL — real base embed+head, NOT synthetic) = {ids:?}\n  \
             conf={conf:?}  experts_read={n_experts} (budget {budget}, all {all})\n  \
             (vs the synthetic-embed/head runs whose ids are placeholders)"
        );
    }

    /// Round-trip with the converter: load a GGUF produced by
    /// `scripts/convert_dspark_to_gguf.py` and confirm every loader tensor name
    /// resolves. Opt-in: `DS4_DSPARK_GGUF=/path/to/dspark.gguf` (the converter's
    /// `--selftest` prints a synthetic one). Skips dim `validate()` for synthetic
    /// GGUFs (tiny stand-in dims ≠ the V4-Flash consts).
    #[test]
    fn gguf_round_trip_loads_all_tensors() {
        let Ok(path) = std::env::var("DS4_DSPARK_GGUF") else {
            eprintln!("DS4_DSPARK_GGUF unset — skipping dspark gguf round-trip");
            return;
        };
        let w = DsparkWeights::from_path(&path).expect("DsparkWeights::from_path");
        assert_eq!(w.layers.len(), consts::N_MTP_LAYERS);
        assert_eq!(w.layers[0].experts.len(), consts::N_EXPERTS);
        assert!(w.markov_w1.n_elements() > 0 && w.main_proj.weight.n_elements() > 0);
        eprintln!(
            "[dspark] gguf round-trip OK: {} mtp stages + main_proj/markov/confidence resolved",
            w.layers.len()
        );
    }
}
