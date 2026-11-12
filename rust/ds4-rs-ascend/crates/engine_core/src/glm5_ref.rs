// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! GLM-5.2 (`glm_moe_dsa`) CPU reference forward — Phase 0 architecture port.
//!
//! GLM-5.2 is architecturally **DeepSeek-V3.2** (the mlx-lm PR #1410 model
//! literally subclasses `DeepseekV32Attention/DecoderLayer/Model`). This module
//! is a self-contained f32 reference that reproduces the mlx-lm forward
//! **byte/cos-exactly** on the tiny test config, so the four GLM-novel pieces
//! can be validated against mlx-lm as the oracle before any GPU work:
//!
//!   1. `RopeConvention::Interleaved`  — traditional RoPE (theta=8e6 in prod).
//!   2. `ScoringFunc::Sigmoid` + noaux_tc group routing (vs ds4's softplus+sqrt).
//!   3. The **V3.2 full-KV indexer** (scores top-`index_topk` over the FULL KV,
//!      not the V4-Flash compressed ring).
//!   4. Cross-layer indexer **sharing**: `LayerType::{Full,Shared}` — "shared"
//!      layers skip indexer compute and reuse the previous full layer's top-k.
//!
//! The module is gated behind [`ModelVariant::Glm5Dsa`] and does NOT touch the
//! DS4-V4-Flash decode path. It is a dense, allocation-heavy reference — the
//! goal is *correctness vs the oracle*, not speed.
//!
//! Layout matches the mlx-lm reference exactly. See
//! the design notes and `mlx_lm/models/{glm_moe_dsa,deepseek_v32}.py`.

#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

/// Which model architecture the reference forward should run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// The production DS4-V4-Flash path (compressor ring, softplus routing).
    /// Handled by the existing `decode_step`/`attn_dispatch` machinery — this
    /// reference module is not used for it.
    Ds4V4Flash,
    /// GLM-5.2 / DeepSeek-V3.2 sparse-attention MoE (this module).
    Glm5Dsa,
}

/// RoPE element layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeConvention {
    /// `mx.fast.rope(traditional=False)` — split-half: pairs `(x[i], x[i+d/2])`.
    /// This is the DS4-V4-Flash `rope_tail` convention.
    Neox,
    /// `mx.fast.rope(traditional=True)` — interleaved: pairs `(x[2i], x[2i+1])`.
    /// GLM-5.2 / DeepSeek-V3.2 use this for both main attn and the indexer.
    Interleaved,
}

/// MoE router scoring function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunc {
    /// DS4-V4-Flash: softplus then sqrt on the routing logits.
    SoftplusSqrt,
    /// GLM-5.2 / DeepSeek-V3.2: `sigmoid` + `noaux_tc` group select.
    Sigmoid,
}

/// Per-layer indexer role (cross-layer sharing schedule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Compute the indexer top-k for this layer.
    Full,
    /// Skip indexer compute; reuse the previous Full layer's top-k.
    Shared,
}

/// GLM-5.2 architecture constants. For Phase 0 these are fed directly from the
/// mlx tiny config; Phase 1 will populate them from GLM GGUF keys.
#[derive(Debug, Clone)]
pub struct DefaultsGlm5 {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    // MoE
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub first_k_dense_replace: usize,
    pub scoring_func: ScoringFunc,
    // Indexer (DSA)
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    // RoPE
    pub rope_theta: f32,
    pub rope_convention: RopeConvention,
    // Norm
    pub rms_norm_eps: f32,
    /// Per-layer indexer schedule (Full computes, Shared reuses).
    pub indexer_types: Vec<LayerType>,
}

impl DefaultsGlm5 {
    /// The mlx-lm `test_glm_moe_dsa` tiny config (PR #1410).
    pub fn tiny() -> Self {
        let indexer_types = "FSFSFS"
            .chars()
            .map(|c| {
                if c == 'F' {
                    LayerType::Full
                } else {
                    LayerType::Shared
                }
            })
            .collect();
        DefaultsGlm5 {
            vocab_size: 1024,
            hidden_size: 128,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            q_lora_rank: 24,
            kv_lora_rank: 16,
            qk_nope_head_dim: 16,
            qk_rope_head_dim: 16,
            v_head_dim: 32,
            intermediate_size: 256,
            moe_intermediate_size: 256,
            n_routed_experts: 4,
            n_shared_experts: 1,
            num_experts_per_tok: 2,
            n_group: 2,
            topk_group: 1,
            routed_scaling_factor: 2.5,
            norm_topk_prob: true,
            first_k_dense_replace: 1,
            scoring_func: ScoringFunc::Sigmoid,
            index_n_heads: 4,
            index_head_dim: 16,
            index_topk: 4,
            rope_theta: 10000.0,
            rope_convention: RopeConvention::Interleaved,
            rms_norm_eps: 1e-5,
            indexer_types,
        }
    }

    /// Build the real GLM-5.2 config from GGUF metadata (`glm-dsa.*` keys, with
    /// DeepSeek-V3.2 fallbacks). Validated against `unsloth/GLM-5.2-GGUF`
    /// (architecture `glm-dsa`, block_count=79). Every field is sourced from the
    /// header — nothing is hard-coded to the production model — so any
    /// GLM-DSA-family GGUF loads. Returns an error if a required key is absent.
    ///
    /// Derived geometry (from the real header):
    ///   - `qk_rope_head_dim` = `rope.dimension_count` (=64)
    ///   - `qk_nope_head_dim` = `attention.key_length_mla` − `qk_rope` … but the
    ///     header reports `key_length_mla=256` = qk_nope(192)+? — so we instead
    ///     read qk_nope from the `attn_k_b` tensor when present, else fall back
    ///     to `key_length_mla − qk_rope`. (The IQ1_S header gives `attn_k_b`
    ///     dims `[192,512,64]` → qk_nope=192.)
    ///   - `v_head_dim` = `attention.value_length_mla` (=256)
    ///   - `num_experts_per_tok`, `topk_group`, `n_group` from the expert keys.
    ///
    /// The indexer schedule is derived as all-`Full` by default: the GGUF stores
    /// indexer weights for **every** layer (`blk.N.indexer.*` present 0..n_layers
    /// in the real file), so cross-layer sharing is a *runtime* decode-time
    /// optimization (`index_topk_freq`/`index_skip_topk_offset`), not a
    /// weight-presence property. Callers that want the Full/Shared schedule pass
    /// `index_topk_freq` explicitly via [`Self::with_indexer_schedule`].
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> anyhow::Result<Self> {
        use crate::gguf::{meta_f32, meta_u32, MetaValue};
        let u = |keys: &[&str], what: &str| -> anyhow::Result<usize> {
            meta_u32(g, keys)
                .map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("GLM GGUF: missing {what} ({keys:?})"))
        };
        let u_opt = |keys: &[&str], dflt: usize| -> usize {
            meta_u32(g, keys).map(|v| v as usize).unwrap_or(dflt)
        };

        // THE FILE'S OWN ARCHITECTURE PREFIX. GLM 5.3 spells every key `glm5-next.*` -- its
        // `general.architecture` is `glm5-next` -- and the hardcoded `glm-dsa`/`deepseek2` lists did not
        // match, so the real 96.5 GB checkpoint was rejected for a missing key it actually had.
        // Reading the prefix from the file is the difference between "the reader knows the model" and
        // "the reader was told the model" -- and the legacy spellings stay as fallbacks.
        let arch = g
            .meta
            .iter()
            .find(|(k, _)| k == "general.architecture")
            .and_then(|(_, v)| match v {
                MetaValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "glm-dsa".to_string());
        let kf = |suffix: &str| -> Vec<String> {
            vec![
                format!("{arch}.{suffix}"),
                format!("glm-dsa.{suffix}"),
                format!("deepseek2.{suffix}"),
            ]
        };
        let ks = |suffix: &str| -> Vec<&str> {
            // leaked only for the duration of the lookup below, which is cheap and bounded
            kf(suffix).iter().map(|s| Box::leak(s.clone().into_boxed_str()) as &str).collect()
        };

        let num_hidden_layers = u(&ks("block_count"), "block_count")?;
        let hidden_size = u(&ks("embedding_length"), "embedding_length")?;
        let vocab_size = meta_u32(g, &ks("vocab_size"))
            .map(|v| v as usize)
            .or_else(|| {
                g.tensors
                    .iter()
                    .find(|t| t.name == "token_embd.weight")
                    .and_then(|t| t.dims.last().map(|d| *d as usize))
            })
            .ok_or_else(|| anyhow::anyhow!("GLM GGUF: missing vocab_size"))?;
        let num_attention_heads = u(
            &ks("attention.head_count"),
            "head_count",
        )?;
        let q_lora_rank = u(
            &ks("attention.q_lora_rank"),
            "q_lora_rank",
        )?;
        let kv_lora_rank = u(
            &ks("attention.kv_lora_rank"),
            "kv_lora_rank",
        )?;
        let qk_rope_head_dim = u(
            &{ let mut v = ks("rope.dimension_count"); v.extend(ks("attention.rope_dimension_count")); v },
            "rope.dimension_count",
        )?;
        let v_head_dim = u_opt(&ks("attention.value_length_mla"), 256);
        // qk_nope: prefer the authoritative `attn_k_b` tensor dim[0]; else derive
        // from key_length_mla. attn_k_b shape is [qk_nope, kv_lora, n_heads].
        let qk_nope_head_dim = g
            .tensors
            .iter()
            .find(|t| t.name.ends_with(".attn_k_b.weight"))
            .and_then(|t| t.dims.first().map(|d| *d as usize))
            .unwrap_or_else(|| u_opt(&ks("attention.key_length_mla"), 256));

        let intermediate_size = u(
            &ks("feed_forward_length"),
            "feed_forward_length",
        )?;
        let moe_intermediate_size = u_opt(
            &ks("expert_feed_forward_length"),
            intermediate_size,
        );
        let n_routed_experts = u(
            &ks("expert_count"),
            "expert_count",
        )?;
        let n_shared_experts = u_opt(
            &ks("expert_shared_count"),
            1,
        );
        let num_experts_per_tok = u(
            &ks("expert_used_count"),
            "expert_used_count",
        )?;
        let n_group = u_opt(
            &ks("expert_group_count"),
            1,
        );
        let topk_group = u_opt(
            &ks("expert_group_used_count"),
            1,
        );
        let first_k_dense_replace = u_opt(
            &ks("leading_dense_block_count"),
            0,
        );
        let routed_scaling_factor = meta_f32(g, &ks("expert_weights_scale")).unwrap_or(1.0);
        let norm_topk_prob = matches!(
            g.get_meta("glm-dsa.expert_weights_norm"),
            Some(MetaValue::Bool(true))
        ) || meta_u32(g, &ks("expert_weights_norm")) == Some(1);
        // expert_gating_func: 2 == sigmoid (noaux_tc). GLM-DSA is always sigmoid;
        // we surface the key for forward-compat but default to Sigmoid regardless.
        let _gating = meta_u32(g, &ks("expert_gating_func"));
        let scoring_func = ScoringFunc::Sigmoid;

        let index_n_heads = u(
            &ks("attention.indexer.head_count"),
            "indexer.head_count",
        )?;
        let index_head_dim = u(
            &ks("attention.indexer.key_length"),
            "indexer.key_length",
        )?;
        let index_topk = u(&ks("attention.indexer.top_k"), "indexer.top_k")?;

        // THE FILE DOES NOT STATE A ROPE FREQUENCY. `glm-meta` for GLM-5.3-Flash-Q2 carries
        // `attention.rope_dimension_count` and no `rope.freq_base` at all, so this is a DEFAULT, not a
        // reading -- and it is recorded as such rather than being silently absorbed into a passing run.
        let rope_theta = meta_f32(g, &ks("rope.freq_base")).unwrap_or(10_000.0);
        let rms_norm_eps = meta_f32(
            g,
            &ks("attention.layer_norm_rms_epsilon"),
        )
        .unwrap_or(1e-5);

        // All-Full schedule (weights present for every layer). See doc comment.
        let indexer_types = vec![LayerType::Full; num_hidden_layers];

        Ok(DefaultsGlm5 {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            intermediate_size,
            moe_intermediate_size,
            n_routed_experts,
            n_shared_experts,
            num_experts_per_tok,
            n_group,
            topk_group,
            routed_scaling_factor,
            norm_topk_prob,
            first_k_dense_replace,
            scoring_func,
            index_n_heads,
            index_head_dim,
            index_topk,
            rope_theta,
            rope_convention: RopeConvention::Interleaved,
            rms_norm_eps,
            indexer_types,
        })
    }

    /// Apply the cross-layer indexer-sharing schedule: layer 0 and every
    /// `freq`-th layer thereafter is `Full` (computes the indexer top-k); the
    /// rest are `Shared` (reuse the previous Full layer's top-k). `freq <= 1`
    /// (or 0) means every layer is Full (no sharing).
    pub fn with_indexer_schedule(mut self, index_topk_freq: usize) -> Self {
        let freq = index_topk_freq.max(1);
        self.indexer_types = (0..self.num_hidden_layers)
            .map(|l| {
                if l % freq == 0 {
                    LayerType::Full
                } else {
                    LayerType::Shared
                }
            })
            .collect();
        self
    }

    #[inline]
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

// ============================ tensor helpers ===============================
// All tensors are row-major `Vec<f32>` with explicit shapes. Small + simple.

/// `out[m,n] = sum_k x[m,k] * w[n,k]`  (nn.Linear with weight `[out,in]`).
pub fn linear(x: &[f32], rows: usize, in_dim: usize, w: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_dim];
    // Parallelize over output columns with scoped threads (no extra deps). The
    // streaming GLM-5.2 forward is dominated by these matvecs (hidden=6144,
    // out up to 16384); single-threaded is ~30+ min/forward. Threads each own a
    // disjoint column band, writing strided into `out` — no aliasing.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(out_dim.max(1));
    if threads <= 1 || out_dim < 64 {
        linear_band(x, rows, in_dim, w, out_dim, 0, out_dim, &mut out);
        return out;
    }
    let band = out_dim.div_ceil(threads);
    let out_ptr = out.as_mut_ptr() as usize;
    std::thread::scope(|s| {
        for t in 0..threads {
            let n0 = t * band;
            if n0 >= out_dim {
                break;
            }
            let n1 = (n0 + band).min(out_dim);
            let (x, w) = (x, w);
            s.spawn(move || {
                // SAFETY: each thread writes only out[r*out_dim + n] for
                // n in [n0,n1) — disjoint across threads, disjoint across rows.
                let out =
                    unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, rows * out_dim) };
                linear_band(x, rows, in_dim, w, out_dim, n0, n1, out);
            });
        }
    });
    out
}

#[inline]
fn linear_band(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    n0: usize,
    n1: usize,
    out: &mut [f32],
) {
    for r in 0..rows {
        let xr = &x[r * in_dim..r * in_dim + in_dim];
        for n in n0..n1 {
            let wn = &w[n * in_dim..n * in_dim + in_dim];
            let mut acc = 0.0f32;
            for k in 0..in_dim {
                acc += xr[k] * wn[k];
            }
            out[r * out_dim + n] = acc;
        }
    }
}

/// RMSNorm over the last dim: `x / sqrt(mean(x^2)+eps) * weight`.
pub fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..r * dim + dim];
        let mut ss = 0.0f32;
        for &v in xr {
            ss += v * v;
        }
        let inv = 1.0f32 / (ss / dim as f32 + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = xr[d] * inv * weight[d];
        }
    }
    out
}

/// LayerNorm over the last dim (mean-subtract + affine), eps=1e-5.
/// Matches `mlx.nn.LayerNorm` (used by the indexer `k_norm`).
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    dim: usize,
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * dim];
    for r in 0..rows {
        let xr = &x[r * dim..r * dim + dim];
        let mean = xr.iter().sum::<f32>() / dim as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0f32 / (var + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = (xr[d] - mean) * inv * weight[d] + bias[d];
        }
    }
    out
}

/// Apply RoPE in-place to a single `dim`-length vector at absolute position
/// `pos`, theta `base`, per `RopeConvention`.
pub fn rope_apply(v: &mut [f32], dim: usize, pos: usize, base: f32, conv: RopeConvention) {
    let half = dim / 2;
    for i in 0..half {
        let freq = (base as f64).powf(-2.0 * i as f64 / dim as f64) as f32;
        let ang = pos as f32 * freq;
        let (s, c) = ang.sin_cos();
        match conv {
            RopeConvention::Interleaved => {
                let a = v[2 * i];
                let b = v[2 * i + 1];
                v[2 * i] = a * c - b * s;
                v[2 * i + 1] = a * s + b * c;
            }
            RopeConvention::Neox => {
                let a = v[i];
                let b = v[i + half];
                v[i] = a * c - b * s;
                v[i + half] = a * s + b * c;
            }
        }
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `down( silu(gate) * up )` SwiGLU MLP. Weights are nn.Linear `[out,in]`.
pub fn swiglu_mlp(
    x: &[f32],
    rows: usize,
    hidden: usize,
    inter: usize,
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
) -> Vec<f32> {
    let g = linear(x, rows, hidden, gate_w, inter);
    let u = linear(x, rows, hidden, up_w, inter);
    let mut act = vec![0.0f32; rows * inter];
    for i in 0..rows * inter {
        act[i] = silu(g[i]) * u[i];
    }
    linear(&act, rows, inter, down_w, hidden)
}

// ============================ weight access ================================

/// Flat oracle weights keyed by mlx dotted names, with shapes.
pub struct Weights {
    pub data: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl Weights {
    fn get(&self, name: &str) -> &[f32] {
        self.data
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
    }
    fn lw(&self, layer: usize, suffix: &str) -> &[f32] {
        self.get(&format!("model.layers.{layer}.{suffix}"))
    }
    fn has(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }
}

// ============================ MoE routing ==================================

/// `group_expert_select` (deepseek_v32.py) for `ScoringFunc::Sigmoid` +
/// noaux_tc. Returns `(inds[top_k], scores[top_k])` for one token's gate
/// logits `gates[n_experts]`.
pub fn group_expert_select(
    gates: &[f32],
    e_bias: &[f32],
    top_k: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
) -> (Vec<usize>, Vec<f32>) {
    let n = gates.len();
    // sigmoid scores
    let orig: Vec<f32> = gates.iter().map(|&g| 1.0 / (1.0 + (-g).exp())).collect();
    let mut scores: Vec<f32> = orig.iter().zip(e_bias).map(|(s, b)| s + b).collect();

    if n_group > 1 {
        let per = n / n_group;
        // group_scores = sum of top-2 within each group
        let mut group_scores = vec![0.0f32; n_group];
        for g in 0..n_group {
            let mut grp: Vec<f32> = scores[g * per..g * per + per].to_vec();
            grp.sort_by(|a, b| b.partial_cmp(a).unwrap());
            group_scores[g] = grp[0] + grp.get(1).copied().unwrap_or(0.0);
        }
        // keep topk_group groups; zero the rest
        let mut gidx: Vec<usize> = (0..n_group).collect();
        gidx.sort_by(|&a, &b| group_scores[b].partial_cmp(&group_scores[a]).unwrap());
        let keep: std::collections::HashSet<usize> = gidx.into_iter().take(topk_group).collect();
        for g in 0..n_group {
            if !keep.contains(&g) {
                for e in g * per..g * per + per {
                    scores[e] = 0.0;
                }
            }
        }
    }

    // top_k experts by (biased) score
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
    let inds: Vec<usize> = order.into_iter().take(top_k).collect();
    // gather ORIGINAL (unbiased) scores at the selected experts
    let mut sel: Vec<f32> = inds.iter().map(|&i| orig[i]).collect();
    if top_k > 1 && norm_topk_prob {
        let denom: f32 = sel.iter().sum();
        for s in sel.iter_mut() {
            *s /= denom;
        }
    }
    for s in sel.iter_mut() {
        *s *= routed_scaling_factor;
    }
    (inds, sel)
}

// ============================ indexer (DSA) ================================

/// V3.2 full-KV indexer. Returns, per query row `s`, the `index_topk` KV
/// positions with the highest scores (UNORDERED, as `mx.argpartition`).
/// `None` if `seq_len <= index_topk` (mlx returns None → no sparsification).
///
/// `causal_mask[s*seq + t] == true` iff t <= s (valid attendable position).
fn indexer_topk(
    x: &[f32],  // [seq, hidden]
    qr: &[f32], // [seq, q_lora_rank]
    seq: usize,
    cfg: &DefaultsGlm5,
    w: &Weights,
    layer: usize,
    causal_mask: &[bool],
) -> Option<Vec<Vec<usize>>> {
    if seq <= cfg.index_topk {
        return None;
    }
    let n_heads = cfg.index_n_heads;
    let hd = cfg.index_head_dim;
    let hidden = cfg.hidden_size;

    // q = wq_b(qr) -> [seq, n_heads, hd]; rope per head (interleaved, rope_head_dim).
    let wq_b = w.lw(layer, "self_attn.indexer.wq_b.weight"); // [n_heads*hd, q_lora]
    let q_flat = linear(qr, seq, cfg.q_lora_rank, wq_b, n_heads * hd);
    // k = k_norm(wk(x)) -> [seq, hd]; rope (interleaved).
    let wk = w.lw(layer, "self_attn.indexer.wk.weight"); // [hd, hidden]
    let k_lin = linear(x, seq, hidden, wk, hd);
    let kn_w = w.lw(layer, "self_attn.indexer.k_norm.weight");
    let kn_b = w.lw(layer, "self_attn.indexer.k_norm.bias");
    let mut k = layer_norm(&k_lin, seq, hd, kn_w, kn_b, 1e-5);

    // weights = weights_proj(x) * (n_heads^-0.5 * head_dim^-0.5)  -> [seq, n_heads]
    let wp = w.lw(layer, "self_attn.indexer.weights_proj.weight"); // [n_heads, hidden]
    let mut weights = linear(x, seq, hidden, wp, n_heads);
    let softmax_scale = (hd as f32).powf(-0.5);
    let wscale = (n_heads as f32).powf(-0.5) * softmax_scale;
    for v in weights.iter_mut() {
        *v *= wscale;
    }

    // apply rope to q (per head) and k. The indexer rope covers the
    // qk_rope_head_dim (== index_head_dim in the tiny config). mlx ropes the
    // whole vector since dims == qk_rope_head_dim.
    let rope_dim = cfg.qk_rope_head_dim.min(hd);
    let mut q = q_flat;
    for s in 0..seq {
        for h in 0..n_heads {
            let base = (s * n_heads + h) * hd;
            rope_apply(
                &mut q[base..base + hd],
                rope_dim,
                s,
                cfg.rope_theta,
                cfg.rope_convention,
            );
        }
        let kb = s * hd;
        rope_apply(
            &mut k[kb..kb + hd],
            rope_dim,
            s,
            cfg.rope_theta,
            cfg.rope_convention,
        );
    }

    // scores[s, t] = sum_h weights[s,h] * relu( sum_d q[s,h,d]*k[t,d] )
    let scores = indexer_scores(&q, &k, &weights, seq, n_heads, hd, causal_mask);
    Some(indexer_topk_from_scores(&scores, seq, cfg.index_topk))
}

/// The DSA indexer SCORE assembly (shared by the oracle `indexer_topk` and the
/// GPU parity test). `scores[s,t] = Σ_h weights[s,h]·relu(Σ_d q[s,h,d]·k[t,d])`
/// for `causal_mask[s,t]`, else `f32::NEG_INFINITY`.
///
/// Shapes: q `[seq, n_heads, hd]` (roped), k `[seq, hd]` (roped+normed),
/// weights `[seq, n_heads]` (pre-scaled). Returns `[seq, seq]` row-major.
pub fn indexer_scores(
    q: &[f32],
    k: &[f32],
    weights: &[f32],
    seq: usize,
    n_heads: usize,
    hd: usize,
    causal_mask: &[bool],
) -> Vec<f32> {
    let mut scores = vec![f32::NEG_INFINITY; seq * seq];
    for s in 0..seq {
        for t in 0..seq {
            if !causal_mask[s * seq + t] {
                continue;
            }
            let mut acc = 0.0f32;
            for h in 0..n_heads {
                let qoff = (s * n_heads + h) * hd;
                let koff = t * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += q[qoff + d] * k[koff + d];
                }
                let relu = dot.max(0.0);
                acc += relu * weights[s * n_heads + h];
            }
            scores[s * seq + t] = acc;
        }
    }
    scores
}

/// Top-`top_k` select per query row of an indexer `[seq, seq]` score matrix:
/// descending by score, ties broken by lower index (deterministic; the SET is
/// what matters). The same selection the GPU argsort/`encode_indexer_topk`
/// performs. Returns one `Vec<usize>` of length `top_k` per query.
pub fn indexer_topk_from_scores(scores: &[f32], seq: usize, top_k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(seq);
    for s in 0..seq {
        let row = &scores[s * seq..s * seq + seq];
        let mut idx: Vec<usize> = (0..seq).collect();
        idx.sort_by(|&a, &b| {
            row[b]
                .partial_cmp(&row[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(top_k);
        out.push(idx);
    }
    out
}

// ============================ MLA attention ===============================

/// Captured intermediates for bisecting divergence vs the oracle.
#[derive(Default)]
pub struct Captures {
    pub map: HashMap<String, Vec<f32>>,
    /// Per-layer indexer top-k (as produced/reused), one Vec<usize> per query.
    pub topk: HashMap<usize, Vec<Vec<usize>>>,
    /// Per-layer router selections.
    pub router_inds: HashMap<usize, Vec<Vec<usize>>>,
    /// Per-layer router scores (aligned with `router_inds`).
    pub router_scores: HashMap<usize, Vec<Vec<f32>>>,
}

/// One decoder layer's attention sublayer (DSV3.2 MLA + DSA indexer), prefill
/// path (L>1). `prev_topk` carries the previous full layer's selection for
/// `LayerType::Shared` layers. Returns `(attn_out[seq,hidden], topk_used)`.
fn attention(
    x: &[f32], // input_layernorm(hidden), [seq, hidden]
    seq: usize,
    cfg: &DefaultsGlm5,
    w: &Weights,
    layer: usize,
    layer_type: LayerType,
    prev_topk: Option<&Vec<Vec<usize>>>,
    causal_mask: &[bool],
) -> (Vec<f32>, Option<Vec<Vec<usize>>>) {
    let hidden = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let qk_nope = cfg.qk_nope_head_dim;
    let qk_rope = cfg.qk_rope_head_dim;
    let qhd = cfg.q_head_dim();
    let kv_lora = cfg.kv_lora_rank;
    let vhd = cfg.v_head_dim;
    let scale = (qhd as f32).powf(-0.5);

    // qr = q_a_layernorm(q_a_proj(x))   (RMSNorm eps=1e-6)
    let qa = linear(
        x,
        seq,
        hidden,
        w.lw(layer, "self_attn.q_a_proj.weight"),
        cfg.q_lora_rank,
    );
    let qr = rms_norm(
        &qa,
        seq,
        cfg.q_lora_rank,
        w.lw(layer, "self_attn.q_a_layernorm.weight"),
        1e-6,
    );
    // q = q_b_proj(qr) -> [seq, nh, qhd]
    let q = linear(
        &qr,
        seq,
        cfg.q_lora_rank,
        w.lw(layer, "self_attn.q_b_proj.weight"),
        nh * qhd,
    );

    // compressed_kv = kv_a_proj_with_mqa(x) -> split [kv_lora | qk_rope]
    let ckv = linear(
        x,
        seq,
        hidden,
        w.lw(layer, "self_attn.kv_a_proj_with_mqa.weight"),
        kv_lora + qk_rope,
    );
    // kv_latent = kv_a_layernorm(compressed_kv[:kv_lora])  (RMSNorm eps=1e-6)
    let mut comp = vec![0.0f32; seq * kv_lora];
    let mut k_pe = vec![0.0f32; seq * qk_rope];
    for s in 0..seq {
        comp[s * kv_lora..s * kv_lora + kv_lora]
            .copy_from_slice(&ckv[s * (kv_lora + qk_rope)..s * (kv_lora + qk_rope) + kv_lora]);
        k_pe[s * qk_rope..s * qk_rope + qk_rope].copy_from_slice(
            &ckv[s * (kv_lora + qk_rope) + kv_lora..s * (kv_lora + qk_rope) + kv_lora + qk_rope],
        );
    }
    let kv_latent = rms_norm(
        &comp,
        seq,
        kv_lora,
        w.lw(layer, "self_attn.kv_a_layernorm.weight"),
        1e-6,
    );

    // Split q into q_nope[seq,nh,qk_nope] and q_pe[seq,nh,qk_rope]; rope q_pe & k_pe.
    let mut q_nope = vec![0.0f32; seq * nh * qk_nope];
    let mut q_pe = vec![0.0f32; seq * nh * qk_rope];
    for s in 0..seq {
        for h in 0..nh {
            let src = (s * nh + h) * qhd;
            q_nope[(s * nh + h) * qk_nope..(s * nh + h) * qk_nope + qk_nope]
                .copy_from_slice(&q[src..src + qk_nope]);
            let pe = &mut q_pe[(s * nh + h) * qk_rope..(s * nh + h) * qk_rope + qk_rope];
            pe.copy_from_slice(&q[src + qk_nope..src + qhd]);
            rope_apply(pe, qk_rope, s, cfg.rope_theta, cfg.rope_convention);
        }
        rope_apply(
            &mut k_pe[s * qk_rope..s * qk_rope + qk_rope],
            qk_rope,
            s,
            cfg.rope_theta,
            cfg.rope_convention,
        );
    }

    // ---- indexer (Full computes; Shared reuses prev_topk) ----
    let topk_used: Option<Vec<Vec<usize>>> = match layer_type {
        LayerType::Full => indexer_topk(x, &qr, seq, cfg, w, layer, causal_mask),
        LayerType::Shared => prev_topk.cloned(),
    };

    // ---- sparse attendance mask: causal AND (t in topk[s]). 1 = attendable.
    let attn_mask = mla_attn_mask(seq, causal_mask, topk_used.as_ref());

    // ---- absorbed K/V (prefill L>1 path) ----
    let (k_abs, v_abs) = mla_absorb_kv(&kv_latent, seq, cfg, w, layer);

    // ---- sdpa: softmax( scale*(q_nope·k_abs) + scale*(q_pe·k_pe) ) @ v_abs ----
    let out = mla_sdpa(
        &q_nope, &q_pe, &k_abs, &v_abs, &k_pe, &attn_mask, seq, nh, qk_nope, qk_rope, vhd, scale,
    );

    // o_proj: reshape [seq, nh*vhd] -> [seq, hidden]
    let attn_out = linear(
        &out,
        seq,
        nh * vhd,
        w.lw(layer, "self_attn.o_proj.weight"),
        hidden,
    );
    (attn_out, topk_used)
}

/// Build the MLA sparse attendance mask `[seq, seq]` (u8, 1 = attendable):
/// `causal_mask[s,t] AND (topk is None OR t ∈ topk[s])`. This is the exact mask
/// the oracle SDPA and the GPU `glm_mla_attn` kernel both consume.
pub fn mla_attn_mask(seq: usize, causal_mask: &[bool], topk: Option<&Vec<Vec<usize>>>) -> Vec<u8> {
    let mut m = vec![0u8; seq * seq];
    for s in 0..seq {
        for t in 0..seq {
            let allowed =
                causal_mask[s * seq + t] && topk.map(|tk| tk[s].contains(&t)).unwrap_or(true);
            m[s * seq + t] = allowed as u8;
        }
    }
    m
}

/// Absorbed K/V projection (`k_abs = kv_latent @ embed_q`,
/// `v_abs = kv_latent @ unembed_out^T`). Returns `(k_abs[nh,seq,qk_nope],
/// v_abs[nh,seq,vhd])`. Shared by the oracle and the GPU parity harness.
pub fn mla_absorb_kv(
    kv_latent: &[f32],
    seq: usize,
    cfg: &DefaultsGlm5,
    w: &Weights,
    layer: usize,
) -> (Vec<f32>, Vec<f32>) {
    let nh = cfg.num_attention_heads;
    let qk_nope = cfg.qk_nope_head_dim;
    let kv_lora = cfg.kv_lora_rank;
    let vhd = cfg.v_head_dim;
    let embed_q = w.lw(layer, "self_attn.embed_q.weight"); // [nh, kv_lora, qk_nope]
    let unembed = w.lw(layer, "self_attn.unembed_out.weight"); // [nh, vhd, kv_lora]
    let mut k_abs = vec![0.0f32; nh * seq * qk_nope];
    let mut v_abs = vec![0.0f32; nh * seq * vhd];
    for h in 0..nh {
        let eqh = &embed_q[h * kv_lora * qk_nope..(h + 1) * kv_lora * qk_nope]; // [kv_lora, qk_nope]
        let uoh = &unembed[h * vhd * kv_lora..(h + 1) * vhd * kv_lora]; // [vhd, kv_lora]
        for t in 0..seq {
            let kl = &kv_latent[t * kv_lora..t * kv_lora + kv_lora];
            for d in 0..qk_nope {
                let mut acc = 0.0f32;
                for l in 0..kv_lora {
                    acc += kl[l] * eqh[l * qk_nope + d];
                }
                k_abs[(h * seq + t) * qk_nope + d] = acc;
            }
            for d in 0..vhd {
                let mut acc = 0.0f32;
                for l in 0..kv_lora {
                    acc += kl[l] * uoh[d * kv_lora + l];
                }
                v_abs[(h * seq + t) * vhd + d] = acc;
            }
        }
    }
    (k_abs, v_abs)
}

/// The MLA SDPA softmax block: per (head, query), over attendable KV positions
/// `out[s,h,:] = softmax_t(scale*(q_nope[s,h]·k_abs[h,t]) +
/// scale*(q_pe[s,h]·k_pe[t])) @ v_abs[h,t]`. `mask[s,t]==0` → excluded. This is
/// the CPU oracle that the GPU `glm_mla_attn` kernel must match.
///
/// Shapes: q_nope `[seq,nh,qk_nope]`, q_pe `[seq,nh,qk_rope]` (roped),
/// k_abs `[nh,seq,qk_nope]`, v_abs `[nh,seq,vhd]`, k_pe `[seq,qk_rope]` (roped),
/// mask `[seq,seq]` (u8). Returns `out[seq,nh,vhd]`.
#[allow(clippy::too_many_arguments)]
pub fn mla_sdpa(
    q_nope: &[f32],
    q_pe: &[f32],
    k_abs: &[f32],
    v_abs: &[f32],
    k_pe: &[f32],
    mask: &[u8],
    seq: usize,
    nh: usize,
    qk_nope: usize,
    qk_rope: usize,
    vhd: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * nh * vhd];
    for h in 0..nh {
        for s in 0..seq {
            let qoff = (s * nh + h) * qk_nope;
            let qpoff = (s * nh + h) * qk_rope;
            let mut logits = vec![0.0f32; seq];
            let mut maxl = f32::NEG_INFINITY;
            for t in 0..seq {
                if mask[s * seq + t] == 0 {
                    logits[t] = f32::NEG_INFINITY;
                    continue;
                }
                let koff = (h * seq + t) * qk_nope;
                let mut dot = 0.0f32;
                for d in 0..qk_nope {
                    dot += q_nope[qoff + d] * k_abs[koff + d];
                }
                let kpoff = t * qk_rope;
                let mut pdot = 0.0f32;
                for d in 0..qk_rope {
                    pdot += q_pe[qpoff + d] * k_pe[kpoff + d];
                }
                logits[t] = scale * dot + scale * pdot;
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
            let ob = (s * nh + h) * vhd;
            for t in 0..seq {
                let p = logits[t] * inv;
                if p == 0.0 {
                    continue;
                }
                let vb = (h * seq + t) * vhd;
                for d in 0..vhd {
                    out[ob + d] += p * v_abs[vb + d];
                }
            }
        }
    }
    out
}

// ============================ MoE / MLP layer =============================

pub fn moe_layer(
    x: &[f32], // post_attention_layernorm(h), [seq, hidden]
    seq: usize,
    cfg: &DefaultsGlm5,
    w: &Weights,
    layer: usize,
    caps: &mut Captures,
) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let n_exp = cfg.n_routed_experts;

    let gate_w = w.lw(layer, "mlp.gate.weight"); // [n_exp, hidden]
    let e_bias = w.lw(layer, "mlp.gate.e_score_correction_bias");
    let sw_gate = w.lw(layer, "mlp.switch_mlp.gate_proj.weight"); // [n_exp, inter, hidden]
    let sw_up = w.lw(layer, "mlp.switch_mlp.up_proj.weight"); // [n_exp, inter, hidden]
    let sw_down = w.lw(layer, "mlp.switch_mlp.down_proj.weight"); // [n_exp, hidden, inter]

    let mut out = vec![0.0f32; seq * hidden];
    let mut router_inds = Vec::with_capacity(seq);
    let mut router_scores = Vec::with_capacity(seq);
    for s in 0..seq {
        let xs = &x[s * hidden..s * hidden + hidden];
        // gate logits = x @ weight.T
        let mut gates = vec![0.0f32; n_exp];
        for e in 0..n_exp {
            let we = &gate_w[e * hidden..e * hidden + hidden];
            gates[e] = xs.iter().zip(we).map(|(a, b)| a * b).sum();
        }
        let (inds, scores) = group_expert_select(
            &gates,
            e_bias,
            cfg.num_experts_per_tok,
            cfg.n_group,
            cfg.topk_group,
            cfg.routed_scaling_factor,
            cfg.norm_topk_prob,
        );
        router_inds.push(inds.clone());
        router_scores.push(scores.clone());
        // y = sum_k scores[k] * expert_k(xs)
        for (k, &e) in inds.iter().enumerate() {
            let g = &sw_gate[e * inter * hidden..(e + 1) * inter * hidden];
            let u = &sw_up[e * inter * hidden..(e + 1) * inter * hidden];
            let d = &sw_down[e * hidden * inter..(e + 1) * hidden * inter];
            let ye = swiglu_mlp(xs, 1, hidden, inter, g, u, d);
            let sc = scores[k];
            for i in 0..hidden {
                out[s * hidden + i] += sc * ye[i];
            }
        }
    }
    caps.router_inds.insert(layer, router_inds);
    caps.router_scores.insert(layer, router_scores);

    // + shared experts (a plain SwiGLU MLP over all tokens)
    if cfg.n_shared_experts > 0 {
        let inter_sh = cfg.moe_intermediate_size * cfg.n_shared_experts;
        let sg = w.lw(layer, "mlp.shared_experts.gate_proj.weight");
        let su = w.lw(layer, "mlp.shared_experts.up_proj.weight");
        let sd = w.lw(layer, "mlp.shared_experts.down_proj.weight");
        let sh = swiglu_mlp(x, seq, hidden, inter_sh, sg, su, sd);
        for i in 0..seq * hidden {
            out[i] += sh[i];
        }
    }
    out
}

// ============================ full forward ================================

/// Run the GLM-5.2 reference forward over `tokens` (a single sequence).
/// Returns `(logits[seq, vocab], captures)`. Captures every sub-step tensor
/// for bisecting vs the mlx oracle.
pub fn forward(tokens: &[usize], cfg: &DefaultsGlm5, w: &Weights) -> (Vec<f32>, Captures) {
    let seq = tokens.len();
    let hidden = cfg.hidden_size;
    let mut caps = Captures::default();

    // causal mask: t <= s
    let mut causal = vec![false; seq * seq];
    for s in 0..seq {
        for t in 0..=s {
            causal[s * seq + t] = true;
        }
    }

    // embed
    let embed = w.get("model.embed_tokens.weight"); // [vocab, hidden]
    let mut h = vec![0.0f32; seq * hidden];
    for (s, &tok) in tokens.iter().enumerate() {
        h[s * hidden..s * hidden + hidden]
            .copy_from_slice(&embed[tok * hidden..tok * hidden + hidden]);
    }
    caps.map.insert("embed".into(), h.clone());

    let mut prev_topk: Option<Vec<Vec<usize>>> = None;
    for layer in 0..cfg.num_hidden_layers {
        // input_layernorm
        let xin = rms_norm(
            &h,
            seq,
            hidden,
            w.lw(layer, "input_layernorm.weight"),
            cfg.rms_norm_eps,
        );
        caps.map.insert(format!("L{layer}.input_ln"), xin.clone());

        let (attn_out, topk) = attention(
            &xin,
            seq,
            cfg,
            w,
            layer,
            cfg.indexer_types[layer],
            prev_topk.as_ref(),
            &causal,
        );
        caps.map
            .insert(format!("L{layer}.attn_out"), attn_out.clone());
        if let Some(tk) = topk.clone() {
            caps.topk.insert(layer, tk.clone());
            prev_topk = Some(tk);
        }

        // h = h + attn_out
        for i in 0..seq * hidden {
            h[i] += attn_out[i];
        }
        caps.map.insert(format!("L{layer}.h_after_attn"), h.clone());

        // post_attention_layernorm
        let pln = rms_norm(
            &h,
            seq,
            hidden,
            w.lw(layer, "post_attention_layernorm.weight"),
            cfg.rms_norm_eps,
        );
        caps.map
            .insert(format!("L{layer}.post_attn_ln"), pln.clone());

        // mlp: dense for first_k_dense_replace layers, MoE after
        let mlp_out = if layer < cfg.first_k_dense_replace {
            swiglu_mlp(
                &pln,
                seq,
                hidden,
                cfg.intermediate_size,
                w.lw(layer, "mlp.gate_proj.weight"),
                w.lw(layer, "mlp.up_proj.weight"),
                w.lw(layer, "mlp.down_proj.weight"),
            )
        } else {
            moe_layer(&pln, seq, cfg, w, layer, &mut caps)
        };
        caps.map
            .insert(format!("L{layer}.mlp_out"), mlp_out.clone());

        for i in 0..seq * hidden {
            h[i] += mlp_out[i];
        }
        caps.map.insert(format!("L{layer}.hidden"), h.clone());
    }

    // final norm + lm_head
    let fnorm = rms_norm(
        &h,
        seq,
        hidden,
        w.get("model.norm.weight"),
        cfg.rms_norm_eps,
    );
    caps.map.insert("final_norm".into(), fnorm.clone());
    let logits = linear(&fnorm, seq, hidden, w.get("lm_head.weight"), cfg.vocab_size);
    caps.map.insert("logits".into(), logits.clone());

    (logits, caps)
}

/// Convenience: does this layer have indexer weights present?
pub fn layer_has_indexer(w: &Weights, layer: usize) -> bool {
    w.has(&format!("model.layers.{layer}.self_attn.indexer.wk.weight"))
}
