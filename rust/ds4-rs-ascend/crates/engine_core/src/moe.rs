// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! MoE FFN reference path for DS4 decode.
//!
//! DS4 MoE routing (DeepSeek-V4 q2):
//!   - 256 experts in n_expert_groups groups (typically 8 groups × 32 experts).
//!   - Each token activates n_experts_used (=6 in the released checkpoint).
//!   - Router produces logits → softplus → sqrt → group-wise gating → top-6
//!     selection. Final per-expert weight is `prob[selected] / Σ_selected probs * 1.5`
//!     (the 1.5 is the DS4 router rescale; see antirez ds4.c router_finalize_one).
//!
//! Gate + up are projected per selected expert (paired matvec → `_pair_f32` kernels);
//! down projection is summed across the 6 selected experts (`_sum6_f32` kernels).
//! In addition there's a shared expert that's always active (no routing).
//!
//! This module carries both antirez-compatible helpers and math-reference
//! helpers. The distinction matters: hash lookup is an exact selection
//! shortcut, while antirez's tiny denominator floor is a compatibility policy
//! that intentionally differs from pure positive-sum normalization.

#![allow(dead_code)]

use crate::forward::{matvec_f32, silu};

/// Antirez clamps routed-MoE probability sums to this floor before computing
/// router weights. This is a parity guard, not the pure mathematical
/// normalization for a positive sum.
pub const ANTIREZ_ROUTER_SUM_FLOOR: f32 = 6.103515625e-5;

/// Softplus then sqrt — DS4 router-logit transform (registry:
/// `dsv4_softplus_sqrt_f32_4`).
///
/// Softplus(x) = log(1 + exp(x)); for large x falls back to x (fast path at x>20).
///
/// antirez `softplus_stable` (ds4.c:4867) is f32-only piecewise:
///   `if (x > 20) return x; if (x < -20) return expf(x); else return log1pf(expf(x));`
/// Our oracle uses f64 `(1.0 + (x as f64).exp()).ln()` plus f64 sqrt. These are
/// algebraically equivalent but f32-ULP different and they feed `weights_scaled`
/// (Σ=1.5) which multiplies INTO every expert's down output. `DS4_MOE_ROUTER_FIDELITY=1`
/// switches to the antirez f32 path bit-equal. Default OFF, opt-in (same
/// caution as the other antirez-fidelity gates).
pub fn softplus_sqrt(logits: &[f32]) -> Vec<f32> {
    let antirez_fidelity = std::env::var("DS4_MOE_ROUTER_FIDELITY").ok().as_deref() == Some("1");
    if antirez_fidelity {
        return logits
            .iter()
            .map(|&x| softplus_sqrt_antirez_one(x))
            .collect();
    }
    logits
        .iter()
        .map(|&x| {
            let sp = if x > 20.0 {
                x as f64
            } else {
                (1.0 + (x as f64).exp()).ln()
            };
            sp.sqrt() as f32
        })
        .collect()
}

/// Verbatim port of antirez `softplus_stable` (ds4.c:4867) composed with
/// `sqrtf`. f32 throughout — no f64 promotion. Each leg uses libm `expf` /
/// `log1pf` via the standard `f32` methods.
#[inline]
fn softplus_sqrt_antirez_one(x: f32) -> f32 {
    let sp: f32 = if x > 20.0_f32 {
        x
    } else if x < -20.0_f32 {
        x.exp()
    } else {
        x.exp().ln_1p()
    };
    sp.sqrt()
}

/// Top-k selection (descending). Returns (indices, values) of the k largest.
///
/// Reference for `argsort_f32_i32_desc_full` — though the kernel does
/// per-row bitonic sort and we just do a small partial sort here.
pub fn topk_desc(x: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(x.len());
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[b].partial_cmp(&x[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    let vals: Vec<f32> = idx.iter().map(|&i| x[i]).collect();
    (idx, vals)
}

/// DS4 router finalize: pick top-k_experts_used after adding per-expert bias,
/// then rescale selected probs by 1.5 / Σ_selected.
///
/// Reference for `dsv4_router_finalize_one` (registry entry).
pub fn router_finalize(probs: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    // antirez clamps `sum < 6.103515625e-5f` (f16 min subnormal) up
    // to that minimum before dividing (ds4.c:5119). Same bug class as
    // `hash_router_weights_from_probs`.
    assert_eq!(probs.len(), bias.len());
    let combined: Vec<f32> = probs
        .iter()
        .zip(bias.iter())
        .map(|(&p, &b)| p + b)
        .collect();
    let (selected, _) = topk_desc(&combined, k);
    let weights = router_weights_from_selected_antirez(probs, &selected);
    (selected, weights)
}

/// Math-reference router finalize: select by `probs + bias`, then normalize
/// selected probabilities by their actual positive sum. This intentionally
/// omits antirez's tiny-sum floor so tests can separate semantic correctness
/// from compatibility behavior.
pub fn router_finalize_math_reference(
    probs: &[f32],
    bias: &[f32],
    k: usize,
) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(probs.len(), bias.len());
    let combined: Vec<f32> = probs
        .iter()
        .zip(bias.iter())
        .map(|(&p, &b)| p + b)
        .collect();
    let (selected, _) = topk_desc(&combined, k);
    let weights = router_weights_from_selected_math_reference(probs, &selected);
    (selected, weights)
}

/// DS4 router expert-weight rescale, set at EVERY model load from GGUF
/// `deepseek4.expert_weights_scale` (Flash: 1.5, PRO: 2.5). Defaults to 1.5 (Flash) when the
/// metadata is absent — every pre-PRO call path keeps its old behavior.
///
/// Was a `OnceLock<f32>` with `set_router_scale` doing `let _ = ROUTER_SCALE.set(v)`.
/// `OnceLock::set` returns `Err` once initialised and that `let _` DISCARDED it, so the value
/// was pinned to whichever model loaded FIRST in the process. The app can load a second model
/// in one process (`ds4app_server_stop` then `ds4app_server_start` with a different path), so:
/// load Flash (1.5) → load PRO (2.5) → PRO silently routes with 1.5. And this is not merely a
/// wrong constant in the weight math — `router_scale() != 1.5` is a KERNEL SELECTOR
/// (macos.rs:4653, deferred.rs:10930), so the second model also takes the wrong specialised
/// path. Silent wrong output, no error. Mirror case: a model with no `expert_weights_scale`
/// inherited the previous model's value instead of the documented 1.5 default.
///
/// An atomic is settable per load and keeps the same free-function API. Relaxed is right: this
/// is written once per model load (long before any decode) and read on the compute path; there
/// is no ordering relationship to establish with other data.
static ROUTER_SCALE_BITS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(DEFAULT_ROUTER_SCALE.to_bits());

/// Flash default, used when the GGUF omits `deepseek4.expert_weights_scale`.
const DEFAULT_ROUTER_SCALE: f32 = 1.5;

pub fn set_router_scale(v: f32) {
    ROUTER_SCALE_BITS.store(v.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

pub fn router_scale() -> f32 {
    f32::from_bits(ROUTER_SCALE_BITS.load(std::sync::atomic::Ordering::Relaxed))
}

/// Restore the Flash default. Call when a model WITHOUT `expert_weights_scale` loads, so it
/// cannot inherit the previous model's scale.
pub fn reset_router_scale() {
    set_router_scale(DEFAULT_ROUTER_SCALE);
}

/// Pure selected-probability normalization for a positive selected sum.
pub fn router_weights_from_selected_math_reference(probs: &[f32], selected: &[usize]) -> Vec<f32> {
    let sum_p: f32 = selected.iter().map(|&i| probs[i]).sum();
    assert!(
        sum_p > 0.0,
        "math-reference router normalization requires positive selected prob sum"
    );
    let scale = router_scale() / sum_p;
    selected.iter().map(|&i| probs[i] * scale).collect()
}

/// Antirez-compatible selected-probability normalization. This preserves the
/// f16-min denominator floor used in ds4.c for both top-k and hash routing.
///
/// antirez (ds4.c:5046, 5121) evaluates `weights[i] / sum * 1.5`
/// left-to-right — two f32 ops with intermediate rounding *after* the divide.
/// Our default precomputes `scale = 1.5 / sum` then `probs[i] * scale` — two
/// f32 ops with intermediate rounding after a *different* divide.
/// Algebraically equal, f32-ULP different per-element. Same fidelity-gate
/// conceptual class as (softplus_sqrt), so reuse the existing
/// `DS4_MOE_ROUTER_FIDELITY=1` env. Default OFF — high-fanout flip needs Mac
/// evidence (same caution as ..#309).
pub fn router_weights_from_selected_antirez(probs: &[f32], selected: &[usize]) -> Vec<f32> {
    let sum_p: f32 = selected.iter().map(|&i| probs[i]).sum();
    let sum_clamped = if sum_p < ANTIREZ_ROUTER_SUM_FLOOR {
        ANTIREZ_ROUTER_SUM_FLOOR
    } else {
        sum_p
    };
    let antirez_fidelity = std::env::var("DS4_MOE_ROUTER_FIDELITY").ok().as_deref() == Some("1");
    if antirez_fidelity {
        // Antirez ds4.c:5046, 5121: `weights[i] / sum * 1.5f` left-to-right.
        selected
            .iter()
            .map(|&i| probs[i] / sum_clamped * router_scale())
            .collect()
    } else {
        let scale = router_scale() / sum_clamped;
        selected.iter().map(|&i| probs[i] * scale).collect()
    }
}

/// hash-mode router finalize (antirez ds4.c:5033-5048).
///
/// Layers 0/1/2 of DS4 V4 Flash use a deterministic token-id → expert lookup
/// table (`ffn_gate_tid2eid`, shape [n_experts_used, vocab]) instead of top-k
/// routing. `selected` comes from that table; weights are derived from probs
/// only (NO bias term) and rescaled to Σ = 1.5.
pub fn hash_router_weights_from_probs(probs: &[f32], selected: &[usize], _k: usize) -> Vec<f32> {
    // antirez clamps `sum < 6.103515625e-5f` (f16 min subnormal) up
    // to that minimum before dividing (ds4.c:5044). Tiny-but-positive sums
    // would otherwise blow up `scale = 1.5/sum_p` to ~1e10 and amplify input
    // prob noise into the residual stream.
    router_weights_from_selected_antirez(probs, selected)
}

/// Math-reference hash-router weights: the hash table supplies the selected
/// experts, then weights are normalized by the actual positive selected sum.
pub fn hash_router_weights_from_probs_math_reference(
    probs: &[f32],
    selected: &[usize],
) -> Vec<f32> {
    router_weights_from_selected_math_reference(probs, selected)
}

/// One expert FFN: gate_proj → up_proj → SwiGLU → down_proj.
///
/// Reference for the fused `mul_mv_id_*_pair_swiglu_f32` + sum6 path. In real
/// Metal we'd run the paired matvec to get `(gate, up)` in one kernel, then
/// SwiGLU, then sum6_down — but the math is the same.
pub fn expert_forward(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    d_ffn: usize,
) -> Vec<f32> {
    let d_in = x.len();
    let gate = matvec_f32(w_gate, x, d_ffn);
    let up = matvec_f32(w_up, x, d_ffn);
    let mid: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect();
    let out = matvec_f32(w_down, &mid, d_in);
    assert_eq!(out.len(), d_in);
    let _ = d_ffn;
    out
}

/// Routed MoE step: select 6 experts, run each, sum weighted outputs.
///
/// `experts_w_*` is the stacked weight tensor: row layout is
/// `[expert][d_ffn][d_in]` for gate/up, `[expert][d_in][d_ffn]` for down.
///
/// Returns the residual contribution `Σ_e weight_e * down_e(silu(gate_e) * up_e)`.
pub fn moe_routed_step(
    x: &[f32],
    selected: &[usize],
    weights: &[f32],
    experts_w_gate: &[f32],
    experts_w_up: &[f32],
    experts_w_down: &[f32],
    d_ffn: usize,
) -> Vec<f32> {
    assert_eq!(selected.len(), weights.len());
    let d_in = x.len();
    let mut acc = vec![0.0f32; d_in];
    let stride_gate_up = d_ffn * d_in;
    let stride_down = d_in * d_ffn;
    // Antirez `layer_routed_moe_one` clamp (ds4.c:5197-5202):
    //   gate: clamp upper only at +DS4_SWIGLU_CLAMP_EXP (=10.0)
    //   up:   clamp symmetrically at ±DS4_SWIGLU_CLAMP_EXP
    const CLAMP: f32 = 10.0;
    for (slot, (&e, &w_e)) in selected.iter().zip(weights.iter()).enumerate() {
        let w_gate = &experts_w_gate[e * stride_gate_up..(e + 1) * stride_gate_up];
        let w_up = &experts_w_up[e * stride_gate_up..(e + 1) * stride_gate_up];
        let w_down = &experts_w_down[e * stride_down..(e + 1) * stride_down];
        let gate = matvec_f32(w_gate, x, d_ffn);
        let up = matvec_f32(w_up, x, d_ffn);
        // antirez `matvec_iq2_xxs_mid_worker` (ds4.c:3690) folds
        // `expert_weight[slot]` INTO `mid[j]` before the down matvec. We must
        // match this — applying the weight after the down matvec is f32-ULP
        // off and breaks bit-equality with the cpu_via_dequant path.
        let mid: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| {
                let g_c = if g > CLAMP { CLAMP } else { g };
                let u_c = u.clamp(-CLAMP, CLAMP);
                silu(g_c) * u_c * w_e
            })
            .collect();
        let y_e = matvec_f32(w_down, &mid, d_in);
        for (av, &yv) in acc.iter_mut().zip(y_e.iter()) {
            *av += yv;
        }
        let _ = slot;
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate or read process-global env vars consumed by
    /// the antirez-fidelity gates (currently `DS4_MOE_ROUTER_FIDELITY`).
    /// Without this, cargo's parallel test runner can interleave env writes
    /// and produce flaky gate=ON/OFF reads. Same pattern as the #306
    /// ENV_LOCK in attn_dispatch.rs and the lock in forward.rs.
    /// The ONE crate-wide env lock (see `crate::ENV_LOCK`) -- a per-file mutex protects
    /// nothing when the state it guards is process-global.
    use crate::ENV_LOCK;

    #[test]
    fn softplus_sqrt_is_zero_at_minus_infinity() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // softplus(-large) → 0, sqrt(0) = 0
        let y = softplus_sqrt(&[-40.0]);
        assert!(y[0] < 1e-15, "got {}", y[0]);
    }

    #[test]
    fn softplus_sqrt_fast_path_matches_slow_path_far_from_threshold() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // At x = 30 (above the x>20 fast-path), output should still be ~sqrt(30).
        let y = softplus_sqrt(&[30.0]);
        let want = (30.0f64).sqrt() as f32;
        assert!((y[0] - want).abs() < 1e-3, "got {}, want {want}", y[0]);
    }

    #[test]
    fn topk_desc_picks_largest_three() {
        let x = vec![0.1f32, 0.9, 0.3, 0.5, 0.7];
        let (idx, vals) = topk_desc(&x, 3);
        assert_eq!(idx, vec![1, 4, 3]); // 0.9, 0.7, 0.5
        assert!((vals[0] - 0.9).abs() < 1e-6);
        assert!((vals[1] - 0.7).abs() < 1e-6);
        assert!((vals[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn topk_desc_k_larger_than_n_caps_at_n() {
        let x = vec![1.0f32, 2.0];
        let (idx, _) = topk_desc(&x, 10);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn router_finalize_weights_sum_to_1_5() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Per DS4 spec the scale produces Σ weights = 1.5.
        let probs = vec![0.1f32, 0.2, 0.3, 0.4];
        let bias = vec![0.0f32; 4];
        let (selected, weights) = router_finalize(&probs, &bias, 2);
        assert_eq!(selected, vec![3, 2]); // top-2 of probs
        let s: f32 = weights.iter().sum();
        assert!((s - 1.5).abs() < 1e-5, "Σ weights = {s}");
    }

    #[test]
    fn router_finalize_bias_can_reorder_selection() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // probs = uniform, but bias makes index 0 win.
        let probs = vec![0.25f32; 4];
        let bias = vec![10.0f32, 0.0, 0.0, 0.0];
        let (selected, _) = router_finalize(&probs, &bias, 1);
        assert_eq!(selected, vec![0]);
    }

    #[test]
    fn expert_forward_identity_weights_produces_silu_mul_input() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // d_in=2, d_ffn=2. W_gate = W_up = identity → gate=up=x.
        // mid = silu(x)*x. W_down = identity → out = silu(x)*x.
        let x = vec![1.0f32, -0.5];
        let w_id = vec![1.0f32, 0.0, 0.0, 1.0]; // 2x2 identity (row-major)
        let y = expert_forward(&x, &w_id, &w_id, &w_id, 2);
        let want: Vec<f32> = x.iter().map(|&v| silu(v) * v).collect();
        for (yv, wv) in y.iter().zip(want.iter()) {
            assert!((yv - wv).abs() < 1e-5, "got {yv}, want {wv}");
        }
    }

    #[test]
    fn moe_routed_step_single_expert_matches_expert_forward() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // With 1 selected expert at weight=1.0 the sum should equal the expert's output.
        let x = vec![0.7f32, -0.3];
        let w_gate = vec![1.0f32, 0.0, 0.0, 1.0];
        let w_up = vec![1.0f32, 0.0, 0.0, 1.0];
        let w_down = vec![1.0f32, 0.0, 0.0, 1.0];
        let y_single = expert_forward(&x, &w_gate, &w_up, &w_down, 2);
        let y_moe = moe_routed_step(&x, &[0], &[1.0], &w_gate, &w_up, &w_down, 2);
        for (a, b) in y_single.iter().zip(y_moe.iter()) {
            assert!((a - b).abs() < 1e-6, "single {a} vs moe {b}");
        }
    }

    #[test]
    fn moe_routed_step_zero_weight_excludes_expert() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let x = vec![0.5f32, 0.2];
        let w = vec![1.0f32, 0.0, 0.0, 1.0];
        // Two experts, weight=0 on both → output zero.
        let stacked_gu = vec![w.clone(), w.clone()].concat();
        let stacked_d = vec![w.clone(), w.clone()].concat();
        let y = moe_routed_step(
            &x,
            &[0, 1],
            &[0.0, 0.0],
            &stacked_gu,
            &stacked_gu,
            &stacked_d,
            2,
        );
        for v in y {
            assert!(v.abs() < 1e-7);
        }
    }

    // ---- AGGRESSIVE TESTS (M4 bisect hardening) ----

    /// hash router MUST IGNORE bias (antirez ds4.c:5048 hash path has no `+ bias`).
    /// A buggy port that adds bias would produce different weights when bias ≠ 0.
    #[test]
    fn hash_router_weights_ignore_bias_field() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.1f32, 0.4, 0.3, 0.2];
        let selected = vec![1, 2]; // not derived from bias
        let w = hash_router_weights_from_probs(&probs, &selected, 2);
        // sum_p = 0.7 → scale = 1.5/0.7 ≈ 2.1428
        let expected = vec![0.4 * 1.5 / 0.7, 0.3 * 1.5 / 0.7];
        for (a, b) in w.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
        }
    }

    /// hash router uses pre-selected indices, NOT top-k.
    /// If a buggy impl re-ran top-k internally, picking [1,2] (small probs)
    /// would be overridden to [0,3] (or whatever has largest probs).
    #[test]
    fn hash_router_respects_caller_selection_over_topk() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // probs say indices 0 and 3 are largest, but caller selected 1 and 2.
        let probs = vec![0.99f32, 0.01, 0.005, 0.98];
        let selected = vec![1, 2]; // hash table dictated this
        let w = hash_router_weights_from_probs(&probs, &selected, 2);
        // sum_p = probs[1] + probs[2] = 0.015 → scale = 1.5/0.015 = 100
        // weights[0] = 0.01*100 = 1.0; weights[1] = 0.005*100 = 0.5
        assert!((w[0] - 1.0).abs() < 1e-4, "w[0]={}", w[0]);
        assert!((w[1] - 0.5).abs() < 1e-4, "w[1]={}", w[1]);
        // BUGGY top-k impl would have picked [0,3] and produced weights summing to 1.5
        // off these much larger probs — quite different result.
    }

    /// hash router weights MUST sum to 1.5 (antirez router rescale).
    /// Trivial-shape test (sum == 1.5) is fine here because we already test
    /// independence of bias and respecting caller selection.
    #[test]
    fn hash_router_weights_sum_to_1_5() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.2f32, 0.3, 0.4, 0.1, 0.05];
        let selected = vec![0, 2, 4];
        let w = hash_router_weights_from_probs(&probs, &selected, 3);
        let s: f32 = w.iter().sum();
        assert!((s - 1.5).abs() < 1e-5, "Σ = {s}");
    }

    /// hash router with all-zero probs MUST NOT NaN (division guard).
    #[test]
    fn hash_router_zero_probs_returns_zeros_not_nan() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.0f32; 4];
        let selected = vec![0, 1];
        let w = hash_router_weights_from_probs(&probs, &selected, 2);
        for v in &w {
            assert!(v.is_finite(), "got NaN/Inf: {v}");
            assert_eq!(*v, 0.0);
        }
    }

    /// hash router MUST clamp sum at f16 min (6.103515625e-5) before
    /// dividing (antirez ds4.c:5044). With sum_p tiny but positive (e.g.
    /// 1e-10), unclamped `1.5/sum_p` blows up to ~1.5e10, and weights inherit
    /// that magnitude. Antirez clamps `sum := max(sum, 6.1e-5)` → bounded
    /// scale ≈ 24576 → bounded weights.
    ///
    /// Construct: 2 selected indices with probs 1e-10 each → sum_p = 2e-10.
    ///   - ours (no clamp): scale = 1.5/2e-10 = 7.5e9, weights = (7.5e-1, 7.5e-1)
    ///                      summing to 1.5 — actually FINITE but with extreme
    ///                      amplification of input prob noise.
    ///   - antirez clamp:  scale = 1.5 / 6.1e-5 ≈ 24576, weights = (2.46e-6, 2.46e-6)
    ///                      summing to 4.9e-6 (NOT 1.5; the clamp deliberately
    ///                      drops the rescale to 1.5 when the underlying probs
    ///                      are degenerate).
    ///
    /// This test will FAIL on the current implementation: it expects weights
    /// summing to ~4.9e-6 (antirez), but our code rescales to 1.5.
    #[test]
    fn hash_router_clamps_tiny_sum_at_f16_min_per_antirez() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let mut probs = vec![0.0f32; 4];
        probs[0] = 1e-10;
        probs[1] = 1e-10;
        let selected = vec![0, 1];
        let w = hash_router_weights_from_probs(&probs, &selected, 2);
        // Antirez: sum = 2e-10 → clamped to 6.103515625e-5
        //          scale = 1.5 / 6.103515625e-5 ≈ 24576
        //          w[i] = 1e-10 * 24576 ≈ 2.458e-6
        let want_each = 1e-10f32 / 6.103515625e-5f32 * 1.5f32;
        let want_sum = 2.0 * want_each;
        let got_sum: f32 = w.iter().sum();
        assert!(
            (got_sum - want_sum).abs() / want_sum.max(1e-12) < 1e-3,
            "tiny-sum block must use f16-min clamp; want sum≈{want_sum:e}, got {got_sum:e}"
        );
        assert!(
            got_sum < 1e-3,
            "tiny-sum weights must stay tiny; got Σ={got_sum:e}"
        );
    }

    /// routed MoE: gate uses UPPER-ONLY clamp at +10.0.
    /// A symmetric clamp would saturate large NEGATIVE gates to -10 → silu(-10) ≈ 0;
    /// asymmetric leaves negatives untouched → silu(-100) ≈ 0 (same magnitude here,
    /// but only because silu saturates). The discriminator: gate = +20 vs gate = -20.
    #[test]
    fn moe_routed_step_clamps_gate_upper_only_not_symmetric() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // 1 expert, d_in=1, d_ffn=1. Identity-ish weights so we can predict mid.
        // gate = +20.0 with clamp at +10 → silu(10) ≈ 10
        // up = +1.0 (symmetric clamp at ±10 leaves alone)
        // mid = silu(10) * 1.0 ≈ 10.0
        // y = mid * 1.0 (w_down) = 10.0
        // Weight = 1.0 → acc = 10.0
        // BUGGY symmetric gate clamp (-10..+10): same result (silu(+10)≈10) — NO!
        // We need gate that would CLAMP DIFFERENTLY upper vs symmetric.
        // gate = +20 → upper-only=10, symmetric=10 — same. Not discriminating.
        // Need gate = -20 → upper-only leaves -20, silu(-20)≈0; symmetric clamps to -10, silu(-10)≈-0.00045
        // Both basically 0. Still not discriminating numerically.
        //
        // Better discriminator: gate small POSITIVE just above clamp.
        // gate = +11 → upper-only=10, silu(10)≈9.9995
        //           → symmetric=10 (same — symmetric also clamps to upper)
        //
        // The true discriminator is when symmetric would clamp LOWER bound.
        // gate = -3, clamp magnitude = 2:
        //   upper-only at +2: gate stays -3, silu(-3) ≈ -0.142
        //   symmetric ±2:     gate becomes -2, silu(-2) ≈ -0.238
        // Use a custom CLAMP via a test on swiglu_clamped directly — done in forward.rs.
        // For routed_step, the CLAMP=10.0 is fixed; build inputs where the
        // distinction shows up:
        //   gate large negative, symmetric clamp would saturate silu away from 0.
        //   gate = -50, silu(-50) ≈ 0 exactly; symmetric clamps to -10, silu(-10) ≈ -0.000454
        // Up = 1, w_down = 1, weight = 1 → output:
        //   correct (upper-only) → 0 * 1 = 0
        //   buggy (symmetric)    → -0.000454 * 1 ≈ -0.000454
        // Tolerance 1e-6 should distinguish.
        let x = vec![1.0f32]; // d_in=1
                              // gate row: [-50.0]; up row: [+1.0]; both d_ffn=1
        let w_gate = vec![-50.0f32]; // matvec → gate = -50 * 1 = -50
        let w_up = vec![1.0f32]; // matvec → up = 1
        let w_down = vec![1.0f32]; // matvec → out[0] = mid[0] * 1
                                   // stack 1 expert
        let stacked_gu = w_gate.clone();
        let stacked_up = w_up.clone();
        let stacked_d = w_down.clone();
        let y = moe_routed_step(&x, &[0], &[1.0], &stacked_gu, &stacked_up, &stacked_d, 1);
        assert!(
            y[0].abs() < 1e-6,
            "gate=-50 must not be clamped; silu(-50)≈0 → y≈0; got {}",
            y[0]
        );
        // BUGGY symmetric clamp would produce y ≈ -0.000454, failing this assert.
    }

    /// routed MoE: up uses SYMMETRIC clamp at ±10.0.
    /// A buggy upper-only-up (or no-up-clamp) would let up=+50 through.
    #[test]
    fn moe_routed_step_clamps_up_symmetrically() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // gate = 1.0 (silu(1)≈0.731), up = +50.0
        // correct (up symmetric clamp ±10): mid = silu(1) * 10 ≈ 7.31
        // buggy (no up clamp): mid = silu(1) * 50 ≈ 36.57
        // w_down=1, weight=1 → distinguishes 7.31 vs 36.57
        let x = vec![1.0f32];
        let w_gate = vec![1.0f32];
        let w_up = vec![50.0f32];
        let w_down = vec![1.0f32];
        let y = moe_routed_step(&x, &[0], &[1.0], &w_gate, &w_up, &w_down, 1);
        let want = silu(1.0) * 10.0;
        assert!(
            (y[0] - want).abs() < 0.01,
            "up=+50 must clamp to +10; want≈{want}, got {}",
            y[0]
        );
        // BUGGY no-clamp would give ~36.57, failing.
    }

    /// routed MoE: up clamp catches large NEGATIVE up too.
    #[test]
    fn moe_routed_step_clamps_up_negative_too() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let x = vec![1.0f32];
        let w_gate = vec![1.0f32];
        let w_up = vec![-50.0f32]; // up = -50
        let w_down = vec![1.0f32];
        let y = moe_routed_step(&x, &[0], &[1.0], &w_gate, &w_up, &w_down, 1);
        let want = silu(1.0) * (-10.0);
        assert!(
            (y[0] - want).abs() < 0.01,
            "up=-50 must clamp to -10; want≈{want}, got {}",
            y[0]
        );
    }

    /// hash router differentiates from top-k router (regression
    /// guard against accidentally routing through router_finalize when
    /// tid2eid is present). Same probs+selected fed to both: top-k would
    /// add bias and re-select; hash uses selected as-is.
    #[test]
    fn router_finalize_and_hash_diverge_when_bias_reorders() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.1f32, 0.4, 0.3, 0.2];
        let bias = vec![10.0f32, 0.0, 0.0, 0.0];
        let hash_sel = vec![1, 2]; // hash table picked these (NOT influenced by bias)
        let hash_w = hash_router_weights_from_probs(&probs, &hash_sel, 2);
        let (topk_sel, topk_w) = router_finalize(&probs, &bias, 2);
        // top-k must have picked index 0 (bias dominates).
        assert!(
            topk_sel.contains(&0),
            "topk should pick bias-boosted index 0"
        );
        assert!(!hash_sel.contains(&0), "hash should NOT pick index 0");
        // Weights MUST be different (different selections → different normalizations).
        // sum_p_hash = 0.4+0.3 = 0.7 → w[0]=0.4*1.5/0.7≈0.857
        // sum_p_topk picks 0 and 1 → 0.1+0.4=0.5 → first w ≈ 0.1*1.5/0.5=0.3 (or 0.4*3=1.2)
        let h_sum: f32 = hash_w.iter().sum();
        let t_sum: f32 = topk_w.iter().sum();
        assert!((h_sum - 1.5).abs() < 1e-5);
        assert!((t_sum - 1.5).abs() < 1e-5);
        // Distinguishing top weight values:
        assert!((hash_w[0] - 0.4 * 1.5 / 0.7).abs() < 1e-5);
    }

    /// router_finalize MUST clamp sum at f16 min (antirez ds4.c:5119).
    /// Same bug class as hash router. Construct probs where the bias-driven
    /// top-k selects indices whose unbiased prob sum is tiny (< 6.1e-5).
    #[test]
    fn router_finalize_clamps_tiny_sum_at_f16_min_per_antirez() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // 4 experts. Bias forces top-k to pick indices [2, 3]. probs at those
        // indices are tiny → sum_p ≈ 2e-10. Without clamp, scale blows up.
        let probs = vec![0.5f32, 0.4, 1e-10, 1e-10];
        let bias = vec![0.0f32, 0.0, 100.0, 100.0];
        let (selected, weights) = router_finalize(&probs, &bias, 2);
        // Bias guarantees selection of [2, 3].
        assert!(
            selected.contains(&2) && selected.contains(&3),
            "bias-boosted indices [2,3] must win, got {:?}",
            selected
        );
        // Antirez: sum = 2e-10 → clamped to 6.1e-5 → scale ≈ 24576 → weights ≈ 2.46e-6 each
        let want_each = 1e-10f32 / 6.103515625e-5f32 * 1.5f32;
        let want_sum = 2.0 * want_each;
        let got_sum: f32 = weights.iter().sum();
        assert!(
            (got_sum - want_sum).abs() / want_sum.max(1e-12) < 1e-3,
            "tiny-sum top-k block must use f16-min clamp; want Σ≈{want_sum:e}, got {got_sum:e}"
        );
        assert!(
            got_sum < 1e-3,
            "tiny-sum weights must stay tiny; got Σ={got_sum:e}"
        );
    }

    /// router_finalize: when bias is zero, weights must be the same as
    /// hash_router_weights_from_probs called with the top-k selection.
    /// (Sanity-check: the two paths agree on the trivial common case.)
    #[test]
    fn router_finalize_zero_bias_agrees_with_hash_on_topk_selection() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.1f32, 0.4, 0.3, 0.2];
        let bias = vec![0.0f32; 4];
        let (selected, w_topk) = router_finalize(&probs, &bias, 2);
        let w_hash = hash_router_weights_from_probs(&probs, &selected, 2);
        for (a, b) in w_topk.iter().zip(w_hash.iter()) {
            assert!((a - b).abs() < 1e-6, "topk {a} vs hash {b}");
        }
    }

    /// Contract split: for ordinary positive sums, antirez parity and the
    /// math reference are the same function. Fast paths should satisfy both
    /// unless they intentionally enter a tiny-sum compatibility corner.
    #[test]
    fn router_math_reference_matches_antirez_above_sum_floor() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.1f32, 0.4, 0.3, 0.2];
        let bias = vec![0.0f32, 0.01, 0.0, 0.0];
        let (sel_a, w_a) = router_finalize(&probs, &bias, 3);
        let (sel_m, w_m) = router_finalize_math_reference(&probs, &bias, 3);
        assert_eq!(sel_a, sel_m);
        for (a, m) in w_a.iter().zip(w_m.iter()) {
            assert!((a - m).abs() < 1e-6, "antirez {a} vs math {m}");
        }
    }

    /// Contract split: below antirez's denominator floor, parity deliberately
    /// stops normalizing weights to sum 1.5. The math reference still does the
    /// pure positive-sum normalization.
    #[test]
    fn router_tiny_sum_exposes_antirez_vs_math_contract() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.5f32, 0.4, 1e-10, 1e-10];
        let bias = vec![0.0f32, 0.0, 100.0, 100.0];
        let (sel_a, w_a) = router_finalize(&probs, &bias, 2);
        let (sel_m, w_m) = router_finalize_math_reference(&probs, &bias, 2);
        assert_eq!(sel_a, sel_m);
        assert!(sel_a.contains(&2) && sel_a.contains(&3));

        let sum_a: f32 = w_a.iter().sum();
        let sum_m: f32 = w_m.iter().sum();
        assert!(
            sum_a < 1e-3,
            "antirez tiny-sum parity should stay tiny, got {sum_a:e}"
        );
        assert!(
            (sum_m - 1.5).abs() < 1e-6,
            "math reference should normalize positive tiny sums to 1.5, got {sum_m:e}"
        );
    }

    /// Hash lookup is an exact performance/dispatch trick for selection: once
    /// selected experts are supplied, its ordinary-sum weights match the math
    /// reference exactly. This distinguishes it from approximation tricks.
    #[test]
    fn hash_lookup_weights_match_math_reference_above_sum_floor() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let probs = vec![0.99f32, 0.01, 0.005, 0.98];
        let selected = vec![1, 2];
        let w_fast = hash_router_weights_from_probs(&probs, &selected, 2);
        let w_math = hash_router_weights_from_probs_math_reference(&probs, &selected);
        for (a, m) in w_fast.iter().zip(w_math.iter()) {
            assert!((a - m).abs() < 1e-6, "hash {a} vs math {m}");
        }
    }

    /// The same hash selection becomes a parity-vs-math tradeoff only when
    /// the selected probability sum falls under antirez's f16 floor.
    #[test]
    fn hash_tiny_sum_exposes_antirez_vs_math_contract() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let mut probs = vec![0.0f32; 4];
        probs[0] = 1e-10;
        probs[1] = 1e-10;
        let selected = vec![0, 1];

        let w_a = hash_router_weights_from_probs(&probs, &selected, 2);
        let w_m = hash_router_weights_from_probs_math_reference(&probs, &selected);
        let sum_a: f32 = w_a.iter().sum();
        let sum_m: f32 = w_m.iter().sum();
        assert!(
            sum_a < 1e-3,
            "antirez tiny-sum parity should stay tiny, got {sum_a:e}"
        );
        assert!(
            (sum_m - 1.5).abs() < 1e-6,
            "math reference should normalize positive tiny sums to 1.5, got {sum_m:e}"
        );
    }

    /// topk_desc: ties at the boundary must include all tied indices up to k.
    /// (Order may vary; just check the set.)
    #[test]
    fn topk_desc_handles_ties_at_boundary() {
        let x = vec![1.0f32, 1.0, 1.0, 0.0];
        let (idx, _) = topk_desc(&x, 2);
        for &i in &idx {
            assert!(i < 3, "tied 1.0 indices are 0,1,2; got {i}");
        }
        assert_eq!(idx.len(), 2);
    }

    /// softplus_sqrt: AT x=0, softplus(0) = log(2), sqrt(log(2)) ≈ 0.832
    /// Trivial smoke (x>20 fast path) won't catch a slow-path log/exp swap.
    #[test]
    fn softplus_sqrt_at_zero_is_sqrt_ln_2() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let y = softplus_sqrt(&[0.0]);
        let want = (2.0f64).ln().sqrt() as f32;
        assert!((y[0] - want).abs() < 1e-5, "got {}, want {want}", y[0]);
    }

    /// softplus_sqrt: monotone-increasing — a regression on the sqrt step
    /// (e.g., dropping it) would still pass simple value tests but break monotonicity.
    #[test]
    fn softplus_sqrt_is_monotone_increasing() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let xs: Vec<f32> = (-5..25).map(|i| i as f32 * 0.5).collect();
        let ys = softplus_sqrt(&xs);
        for i in 1..ys.len() {
            assert!(
                ys[i] >= ys[i - 1] - 1e-6,
                "non-monotone at i={i}: {} -> {}",
                ys[i - 1],
                ys[i]
            );
        }
    }

    /// failing-first discrimination test for the antirez f32 softplus
    /// fidelity gate. Antirez `softplus_stable` (ds4.c:4867) is f32 piecewise
    /// (`expf`/`log1pf`) plus an `x < -20` fast path; ours uses f64 `exp().ln()`
    /// without that fast path. The two paths produce algebraically-equivalent
    /// outputs that differ by 1+ f32 ULP on adversarial logits. Hand-stamp both
    /// paths and assert the gate switches between them bit-exactly.
    #[test]
    fn softplus_sqrt_antirez_fidelity_gate_switches_reducer() {
        let _g = ENV_LOCK.write().unwrap();
        // Adversarial logits exercising every leg of the antirez piecewise:
        //   - x in mid-range where log1pf(expf(x)) f32-rounds differently from
        //     f64 log1p of f64 exp;
        //   - x near the +20/-20 fast-path boundaries;
        //   - x > 20 (fast path: x itself; trivially identical);
        //   - x < -20 (antirez returns expf(x); ours computes (1+exp(x)).ln()
        //     in f64 then casts — algebraically equal as x → -∞, ULP-different
        //     in the transition window).
        let logits: Vec<f32> = vec![
            -19.999_f32,
            -19.5,
            -0.5,
            -0.25,
            0.123_456_7,
            0.5,
            0.999,
            1.000_001,
            5.5,
            15.25,
            19.5,
            19.999,
            20.000_001,
            30.0,
        ];

        // Hand-stamped f64 oracle (the current default path).
        let want_f64: Vec<f32> = logits
            .iter()
            .map(|&x| {
                let sp = if x > 20.0 {
                    x as f64
                } else {
                    (1.0 + (x as f64).exp()).ln()
                };
                sp.sqrt() as f32
            })
            .collect();

        // Hand-stamped antirez f32 path.
        let want_antirez: Vec<f32> = logits
            .iter()
            .map(|&x| {
                let sp: f32 = if x > 20.0_f32 {
                    x
                } else if x < -20.0_f32 {
                    x.exp()
                } else {
                    x.exp().ln_1p()
                };
                sp.sqrt()
            })
            .collect();

        // Discrimination guard: at least one index must differ at the bit level.
        // If the two paths happened to agree everywhere we'd be testing nothing.
        let any_diff = want_f64
            .iter()
            .zip(want_antirez.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            any_diff,
            "test inputs failed to discriminate f64 oracle vs antirez f32 path"
        );

        // Gate OFF (default): must match the f64 oracle bit-for-bit.
        std::env::remove_var("DS4_MOE_ROUTER_FIDELITY");
        let got_off = softplus_sqrt(&logits);
        for (i, (g, w)) in got_off.iter().zip(want_f64.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "gate=OFF should bit-match the f64 oracle at i={i}; got {} want {}",
                g,
                w
            );
        }

        // Gate ON: must bit-match the antirez f32 path.
        std::env::set_var("DS4_MOE_ROUTER_FIDELITY", "1");
        let got_on = softplus_sqrt(&logits);
        for (i, (g, w)) in got_on.iter().zip(want_antirez.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "gate=ON should bit-match the antirez f32 softplus_stable at i={i}; got {} want {}",
                g,
                w
            );
        }

        // Restore env so we don't perturb sibling tests.
        std::env::remove_var("DS4_MOE_ROUTER_FIDELITY");
    }

    /// router-weight normalize reassociation: antirez evaluates
    /// `weights[i] / sum * 1.5` left-to-right (ds4.c:5046, 5121); our default
    /// precomputes `scale = 1.5 / sum` then `probs[i] * scale`. Algebraically
    /// equal, f32-ULP different.
    #[test]
    fn router_weights_from_selected_antirez_reassoc_gate_switches() {
        let _g = ENV_LOCK.write().unwrap();
        std::env::remove_var("DS4_MOE_ROUTER_FIDELITY");

        // Construct probs/selected so the two evaluation orders pick up a
        // bit-distinct ULP cascade. Mix of magnitudes around the order of
        // sum_p ~ 0.3-0.5 (typical DS4 selected-sum range pre-clamp).
        let probs: Vec<f32> = vec![
            0.12345_f32,
            0.07_f32,
            0.04_f32,
            0.03_f32,
            0.02_f32,
            0.01_f32,
        ];
        let selected: Vec<usize> = (0..6).collect();

        // Pre-flight probe: assert the two orders differ bit-wise on at
        // least one element. If they don't, the test setup is trivial.
        let sum_p: f32 = probs.iter().copied().sum();
        let scale = 1.5_f32 / sum_p;
        let mut any_probe_diff = false;
        for &i in &selected {
            let our = probs[i] * scale;
            let anti = probs[i] / sum_p * 1.5_f32;
            if our.to_bits() != anti.to_bits() {
                any_probe_diff = true;
                break;
            }
        }
        assert!(
            any_probe_diff,
            "test setup is trivial: ours and antirez reassoc bit-match on all probes"
        );

        // Gate OFF — default `(1.5/sum)*p`.
        std::env::remove_var("DS4_MOE_ROUTER_FIDELITY");
        let off = router_weights_from_selected_antirez(&probs, &selected);

        // Gate ON — antirez `p/sum*1.5`.
        std::env::set_var("DS4_MOE_ROUTER_FIDELITY", "1");
        let on = router_weights_from_selected_antirez(&probs, &selected);
        std::env::remove_var("DS4_MOE_ROUTER_FIDELITY");

        let any_diff = off
            .iter()
            .zip(on.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            any_diff,
            "DS4_MOE_ROUTER_FIDELITY=1 did NOT change router weights — gate is a no-op"
        );

        // Cross-check: gate-on output must bit-match a hand-stamped antirez
        // reference computed exactly the same way (left-to-right divide-mul).
        for (i, &p_idx) in selected.iter().enumerate() {
            let want = probs[p_idx] / sum_p * 1.5_f32;
            assert_eq!(
                on[i].to_bits(),
                want.to_bits(),
                "gate=ON should bit-match antirez reassoc at i={i}; got {} want {}",
                on[i],
                want
            );
        }
    }
}

#[cfg(test)]
mod router_scale_tests {
    use super::{reset_router_scale, router_scale, set_router_scale, DEFAULT_ROUTER_SCALE};

    /// The bug: `set_router_scale` was `let _ = OnceLock.set(v)`, so the FIRST model to load
    /// pinned the value for the process and every later model silently inherited it. The app
    /// can load a second model in one process (ds4app_server_stop -> _start), so this shipped
    /// as: load Flash (1.5), load PRO (2.5), PRO routes with 1.5 AND takes the wrong
    /// specialised kernel (router_scale() != 1.5 is a kernel selector).
    ///
    /// These run in one process and share the global, so they are ordered inside one #[test].
    #[test]
    fn a_second_model_load_actually_changes_the_scale() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        set_router_scale(1.5);
        assert_eq!(router_scale(), 1.5, "first load (Flash)");

        // Second model, different scale. With the OnceLock this was silently dropped.
        set_router_scale(2.5);
        assert_eq!(
            router_scale(),
            2.5,
            "a second model load must take effect — with OnceLock::set the Err was discarded \
             and PRO silently routed with Flash's 1.5"
        );

        // Third model with NO expert_weights_scale metadata must fall back to the documented
        // default, NOT inherit 2.5 from the previous model.
        reset_router_scale();
        assert_eq!(
            router_scale(),
            DEFAULT_ROUTER_SCALE,
            "a model without expert_weights_scale must reset to the Flash default, not inherit"
        );

        set_router_scale(DEFAULT_ROUTER_SCALE); // leave the global as found
    }
}
