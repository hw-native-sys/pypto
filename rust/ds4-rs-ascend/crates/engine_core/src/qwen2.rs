// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Qwen2 **dense** (`qwen2` GGUF architecture) config — the small, runs-anywhere
//! model family (e.g. `DeepSeek-R1-Distill-Qwen-1.5B`). This is a plain dense
//! transformer: RMSNorm, Q/K/V projections **with bias**, split-half (NEOX) RoPE
//! over the full head_dim, GQA attention, a dense SwiGLU MLP (gate/up/silu/down),
//! final RMSNorm and an lm-head. No MoE, no MLA, no sparse indexer.
//!
//! Mirrors [`crate::glm5_ref::DefaultsGlm5`] in spirit: every field is read from
//! the GGUF header (`qwen2.*` keys) so any `qwen2`-architecture GGUF loads. The
//! Metal per-token forward that consumes this lives in
//! the host-side `qwen_session` helper (`qwen2_decode_step`); the exact forward math is the
//! dense-Qwen2 reference in `crates/deepseek_metal/src/engine.rs::forward_one`.

/// Qwen2 dense architecture constants, sourced from GGUF `qwen2.*` metadata.
#[derive(Debug, Clone)]
pub struct DefaultsQwen2 {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Per-head dim. GGUF may store `qwen2.attention.key_length`; else it is
    /// `hidden_size / num_attention_heads` (128 for the 1.5B model).
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl DefaultsQwen2 {
    /// Build the Qwen2 config from GGUF metadata (`qwen2.*` keys). Returns an
    /// error if a required key is absent. Validated against bartowski's
    /// `DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` (architecture `qwen2`,
    /// block_count=28, head_count=12, head_count_kv=2, embedding_length=1536,
    /// feed_forward_length=8960, rope.freq_base=10000).
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> anyhow::Result<Self> {
        use crate::gguf::meta_u32;
        let u = |keys: &[&str], what: &str| -> anyhow::Result<usize> {
            meta_u32(g, keys)
                .map(|v| v as usize)
                .ok_or_else(|| anyhow::anyhow!("Qwen2 GGUF: missing {what} ({keys:?})"))
        };

        // Keys are tried for both `qwen2.*` (R1-Distill / Qwen2.5) and `qwen3.*`
        // (Qwen3 dense — same fields, different arch prefix, adds QK-Norm + no bias).
        let num_hidden_layers = u(&["qwen2.block_count", "qwen3.block_count"], "block_count")?;
        let hidden_size = u(
            &["qwen2.embedding_length", "qwen3.embedding_length"],
            "embedding_length",
        )?;
        let num_attention_heads = u(
            &["qwen2.attention.head_count", "qwen3.attention.head_count"],
            "attention.head_count",
        )?;
        let num_key_value_heads = meta_u32(
            g,
            &[
                "qwen2.attention.head_count_kv",
                "qwen3.attention.head_count_kv",
            ],
        )
        .map(|v| v as usize)
        .unwrap_or(num_attention_heads);
        let head_dim = meta_u32(
            g,
            &["qwen2.attention.key_length", "qwen3.attention.key_length"],
        )
        .map(|v| v as usize)
        .unwrap_or_else(|| hidden_size / num_attention_heads.max(1));
        let intermediate_size = u(
            &["qwen2.feed_forward_length", "qwen3.feed_forward_length"],
            "feed_forward_length",
        )?;

        // vocab_size: prefer the explicit key, else the embedding table row count
        // (token_embd.weight last dim in the GGUF row-major [vocab, hidden]).
        let vocab_size = meta_u32(g, &["qwen2.vocab_size", "qwen3.vocab_size"])
            .map(|v| v as usize)
            .or_else(|| {
                g.tensors
                    .iter()
                    .find(|t| t.name == "token_embd.weight")
                    .and_then(|t| t.dims.last().map(|d| *d as usize))
            })
            .ok_or_else(|| anyhow::anyhow!("Qwen2 GGUF: missing vocab_size"))?;

        let rope_theta =
            crate::gguf::meta_f32(g, &["qwen2.rope.freq_base", "qwen3.rope.freq_base"])
                .unwrap_or(10000.0);
        let rms_norm_eps = crate::gguf::meta_f32(
            g,
            &[
                "qwen2.attention.layer_norm_rms_epsilon",
                "qwen3.attention.layer_norm_rms_epsilon",
            ],
        )
        .unwrap_or(1e-6);

        Ok(DefaultsQwen2 {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            rope_theta,
            rms_norm_eps,
        })
    }

    /// Total query-projection width (`num_attention_heads * head_dim`).
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// Total key/value-projection width (`num_key_value_heads * head_dim`).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}
