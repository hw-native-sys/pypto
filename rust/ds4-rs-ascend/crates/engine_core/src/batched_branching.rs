// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! per-token branching plan classifier.
//!
//! `BatchedBranchingPlan` records which layer indices need CPU readback
//! between the buffer-A and buffer-B halves of `SingleBufferEncoder`.
//! Three signal categories per antirez decode_step:
//!
//! - **compressor layers** (`params.compress_ratio != 0 &&
//!   !attn_compressor_kv.is_empty()`): need the compressed-KV row emitted
//!   from the compressor MLP back on CPU before the next layer's attention
//!   can read it (decode_step.rs:812).
//! - **indexer layers** (`compress_ratio == 4 && !indexer_compressor_kv.
//!   is_empty()`): a subset of compressor layers (DS4 ratio-4 layers) that
//!   additionally need the `indexer_allowed` bitmap on CPU
//!   (decode_step.rs:857).
//! - **hash router layers** (`moe.routing_table.is_some()`): the early
//!   layers (0/1/2 in DS4 V4 Flash) where MoE expert selection depends on
//!   `CURRENT_TOKEN_HINT` rather than the router top-k; the selection has
//!   to be computed on CPU (cheap, table lookup) and the MoE pipeline has
//!   to read it back (decode_step.rs:980).
//!
//! This module is target-independent — the classifier walks a
//! `ComposedModelWeights` and is callable on Linux. `the accelerator host crate`'s Phase D
//! encoder body consumes it to plan its host syncs.

use crate::decode_step::ComposedModelWeights;

/// Encoder-internal record of which state slots cross the buffer-A → buffer-B
/// boundary. Populated by `BatchedBranchingPlan::for_model`.
#[derive(Debug, Default, Clone)]
pub struct BatchedBranchingPlan {
    /// Layers that need their compressor selection / emitted compressed-KV
    /// row read back to host before the next layer encodes.
    pub compressor_layers: Vec<usize>,
    /// Layers (subset of `compressor_layers`) that additionally need the
    /// indexer (ratio==4) `indexer_allowed` bitmap on CPU.
    pub indexer_layers: Vec<usize>,
    /// Layers where MoE expert selection depends on the per-token hash
    /// routing table (CURRENT_TOKEN_HINT lookup, ds4.c:5155-5157).
    pub hash_router_layers: Vec<usize>,
}

impl BatchedBranchingPlan {
    /// Walk a `ComposedModelWeights` and classify each layer index.
    pub fn for_model(model: &ComposedModelWeights) -> Self {
        let n = model.layers.len();
        let mut plan = Self::default();
        for il in 0..n {
            let layer = &model.layers[il];
            let cr = layer.attn.params.compress_ratio;
            let has_compressor = cr != 0 && layer.attn.has_attn_compressor();
            if has_compressor {
                plan.compressor_layers.push(il);
                if cr == 4 && layer.attn.has_indexer_compressor() && layer.attn.has_indexer_qb() {
                    plan.indexer_layers.push(il);
                }
            }
            if layer.moe.routing_table.is_some() {
                plan.hash_router_layers.push(il);
            }
        }
        plan
    }

    /// True if no layer needs a CPU readback. A purely-pure model could
    /// run buffer A → buffer B with zero host sync between them.
    pub fn is_empty(&self) -> bool {
        self.compressor_layers.is_empty()
            && self.indexer_layers.is_empty()
            && self.hash_router_layers.is_empty()
    }

    /// All layer indices needing any kind of host sync between A and B,
    /// deduped and sorted. The encoder uses this to pick `L_split`: the
    /// first layer in this set is the earliest place buffer A could end.
    pub fn sync_layers(&self) -> Vec<usize> {
        let mut s: Vec<usize> = self
            .compressor_layers
            .iter()
            .chain(self.indexer_layers.iter())
            .chain(self.hash_router_layers.iter())
            .copied()
            .collect();
        s.sort_unstable();
        s.dedup();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn_dispatch::{AttnLayerWeights, LayerParams};
    use crate::decode_step::{ComposedLayerWeights, LayerWeights};

    fn tiny_params(d_embd: u32, compress_ratio: u32) -> LayerParams {
        LayerParams {
            layer_idx: 0,
            d_embd,
            n_hc: 1,
            n_head: 1,
            head_dim: 1,
            n_rot: 1,
            n_lora_q: 1,
            n_lora_kv: 1,
            hc_sinkhorn_iter: 1,
            hc_eps: 1e-6,
            rms_eps: 1e-6,
            rope_orig_ctx: 4096,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
            rope_ext_factor: 0.0,
            rope_attn_factor: 1.0,
            compress_ratio,
            n_out_group: 1,
        }
    }

    fn empty_attn(
        d_embd: u32,
        compress_ratio: u32,
        has_comp_kv: bool,
        has_idx_kv: bool,
    ) -> AttnLayerWeights {
        let comp_kv = if has_comp_kv {
            vec![0.0; 1]
        } else {
            Vec::new()
        };
        let (idx_kv, idx_qb) = if has_idx_kv {
            (vec![0.0; 1], vec![0.0; 1])
        } else {
            (Vec::new(), Vec::new())
        };
        AttnLayerWeights {
            params: tiny_params(d_embd, compress_ratio),
            hc_attn_fn: Vec::new(),
            hc_attn_scale: Vec::new(),
            hc_attn_base: Vec::new(),
            hc_ffn_fn: Vec::new(),
            hc_ffn_scale: Vec::new(),
            hc_ffn_base: Vec::new(),
            hc_attn_fn_f16: Vec::new().into(),
            hc_ffn_fn_f16: Vec::new().into(),
            hc_norm_gamma: Vec::new(),
            hc_ffn_norm_gamma: Vec::new(),
            qkv_gamma_q: Vec::new(),
            qkv_gamma_kv: Vec::new(),
            attn_q_a: Vec::new(),
            attn_q_b: Vec::new(),
            attn_kv: Vec::new(),
            attn_q_a_q8: Vec::new().into(),
            attn_q_b_q8: Vec::new().into(),
            attn_kv_q8: Vec::new().into(),
            w_o_a_q8: Vec::new().into(),
            w_o_b_q8: Vec::new().into(),
            w_shared_gate_q8: Vec::new().into(),
            w_shared_up_q8: Vec::new().into(),
            w_shared_down_q8: Vec::new().into(),
            w_o_a: Vec::new(),
            w_o_b: Vec::new(),
            attn_sinks: Vec::new(),
            w_shared_gate: Vec::new(),
            w_shared_up: Vec::new(),
            w_shared_down: Vec::new(),
            shared_dim: 0,
            shared_clamp: 0.0,
            attn_compressor_kv: comp_kv,
            attn_compressor_gate: Vec::new(),
            attn_compressor_ape: Vec::new(),
            attn_compressor_norm: Vec::new(),
            attn_compressor_kv_f16: Vec::new().into(),
            attn_compressor_gate_f16: Vec::new().into(),
            indexer_compressor_kv: idx_kv,
            indexer_compressor_gate: Vec::new(),
            indexer_compressor_ape: Vec::new(),
            indexer_compressor_norm: Vec::new(),
            indexer_compressor_kv_f16: Vec::new().into(),
            indexer_compressor_gate_f16: Vec::new().into(),
            indexer_attn_q_b: idx_qb,
            indexer_proj: Vec::new(),
            indexer_attn_q_b_f16: Vec::new().into(),
            indexer_proj_f16: Vec::new().into(),
        }
    }

    fn empty_moe(d_model: usize, with_table: bool) -> LayerWeights {
        LayerWeights {
            n_experts: 1,
            n_experts_used: 1,
            d_model,
            d_ffn: std::num::NonZeroUsize::new(1),
            attn_norm_gamma: vec![1.0; d_model],
            w_attn: Vec::new(),
            ffn_norm_gamma: vec![1.0; d_model],
            w_router: vec![0.0; d_model],
            w_router_f16: Vec::new().into(),
            router_bias: vec![0.0; 1],
            routing_table: if with_table {
                Some(vec![0i32; 1])
            } else {
                None
            },
            w_gate_exps: Vec::new(),
            w_up_exps: Vec::new(),
            w_down_exps: Vec::new(),
        }
    }

    fn model_from(layers: Vec<(AttnLayerWeights, LayerWeights)>) -> ComposedModelWeights {
        ComposedModelWeights {
            layers: layers
                .into_iter()
                .map(|(attn, moe)| ComposedLayerWeights { attn, moe })
                .collect(),
            final_norm_gamma: vec![1.0],
            lm_head: vec![0.0],
            lm_head_q8: vec![].into(),
            lm_head_q4: vec![],
            vocab_size: 1,
            d_model: 1,
            output_hc_base: None,
            output_hc_fn: None,
            output_hc_scale: None,
        }
    }

    #[test]
    fn for_model_classifies_dense_layer_as_empty() {
        let m = model_from(vec![(empty_attn(8, 0, false, false), empty_moe(8, false))]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert!(p.is_empty());
    }

    #[test]
    fn for_model_flags_compressor_layer() {
        let m = model_from(vec![(empty_attn(8, 128, true, false), empty_moe(8, false))]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert_eq!(p.compressor_layers, vec![0]);
        assert!(p.indexer_layers.is_empty());
        assert!(p.hash_router_layers.is_empty());
    }

    #[test]
    fn for_model_flags_indexer_layer_as_both_compressor_and_indexer() {
        let m = model_from(vec![(empty_attn(8, 4, true, true), empty_moe(8, false))]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert_eq!(p.compressor_layers, vec![0]);
        assert_eq!(p.indexer_layers, vec![0]);
    }

    #[test]
    fn for_model_flags_hash_router_layer() {
        let m = model_from(vec![(empty_attn(8, 0, false, false), empty_moe(8, true))]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert!(p.compressor_layers.is_empty());
        assert_eq!(p.hash_router_layers, vec![0]);
    }

    #[test]
    fn compress_ratio_nonzero_but_no_tensor_skips_classification() {
        let m = model_from(vec![(
            empty_attn(8, 128, false, false),
            empty_moe(8, false),
        )]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert!(p.is_empty(), "no kv tensor → not a real compressor layer");
    }

    #[test]
    fn ratio_4_without_indexer_tensors_is_compressor_only() {
        let m = model_from(vec![(empty_attn(8, 4, true, false), empty_moe(8, false))]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert_eq!(p.compressor_layers, vec![0]);
        assert!(p.indexer_layers.is_empty());
    }

    #[test]
    fn sync_layers_merges_and_dedups() {
        let m = model_from(vec![
            (empty_attn(8, 0, false, false), empty_moe(8, true)), // 0 hash
            (empty_attn(8, 4, true, true), empty_moe(8, true)),   // 1 compressor + idx + hash
            (empty_attn(8, 128, true, false), empty_moe(8, false)), // 2 compressor
            (empty_attn(8, 0, false, false), empty_moe(8, false)), // 3 nothing
        ]);
        let p = BatchedBranchingPlan::for_model(&m);
        assert_eq!(p.compressor_layers, vec![1, 2]);
        assert_eq!(p.indexer_layers, vec![1]);
        assert_eq!(p.hash_router_layers, vec![0, 1]);
        assert_eq!(p.sync_layers(), vec![0, 1, 2]);
    }
}
