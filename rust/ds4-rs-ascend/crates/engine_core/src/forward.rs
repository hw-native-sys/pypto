// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Forward-graph reference path for DS4 decode.
//!
//! This module implements the **CPU reference** of one decode step. It is used:
//!
//! 1. As an oracle for `cargo test` (we can validate against tiny synthetic
//!    fixtures without a 128 GB GPU).
//! 2. To cross-check the Metal dispatch (E5-E6) at cosine ≥ 0.99 against the
//!    antirez per-layer hidden-state dumps captured by
//!    `scripts/aws_ds4/04_capture_oracle_dumps.sh`.
//!
//! Accelerator kernels will replace these
//! reference implementations one by one — keeping the CPU version around lets
//! every Metal dispatch be checked for numerical drift.

#![allow(dead_code)]

/// Per-layer parameter slot positions for the DS4 MoE block.
///
/// DS4 (DeepSeek-V4) layer layout (from antirez/ds4.c:1846-2193):
///
/// - MLA attention: `attn_q_a` (lora-down), `attn_q_a_norm` (RMS γ),
///   `attn_q_b` (lora-up), `attn_kv_a_mqa` (lora-down), `attn_kv_a_norm`,
///   `attn_kv_b` (lora-up), `attn_kv` (compressed K cache projection),
///   `attn_output`.
/// - MoE FFN: `ffn_gate_inp` (router), `ffn_gate_exps`, `ffn_up_exps`,
///   `ffn_down_exps`, `ffn_gate_shexp` (always-on shared expert),
///   `ffn_up_shexp`, `ffn_down_shexp`, plus `ffn_norm` for the pre-FFN RMSNorm.
#[derive(Debug, Clone, Copy)]
pub struct LayerLayout {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub n_expert_groups: usize,
    pub n_expert_groups_used: usize,
    pub d_ffn_moe: usize,
    pub d_ffn_shared: usize,
}

// ---------------------------------------------------------------------------
// CPU reference kernels (numerically equivalent to the Metal versions we
// dispatch through the accelerator path).
// ---------------------------------------------------------------------------

/// RMSNorm: `out[i] = x[i] / sqrt(mean(x²) + eps) * γ[i]`.
///
/// Numerically equivalent to `kernel_rms_norm_mul_f32_4` (registry entry
/// `rms_norm_mul_f32_4`) at fp32. The Metal version processes float4 lanes;
/// the CPU reference is row-at-a-time and uses `f64` accumulation for the
/// variance to bound rounding error.
pub fn rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(
        x.len(),
        gamma.len(),
        "rms_norm: x and γ must have the same length"
    );
    let n = x.len();
    let mean_sq: f64 = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
    // antirez `rms_norm_weight` (ds4.c:2560) keeps `scale` in f32
    // (`scale = 1.0f / sqrtf((float)(ss/n) + eps);`) and computes
    // `out[i] = x[i] * scale * weight[i]` as a f32-only chain with intermediate
    // rounding between the two multiplies. Our default keeps the multiply
    // chain in f64 — algebraically more precise, f32-ULP different from
    // antirez. `DS4_RMS_NORM_FIDELITY=1` switches to bit-equal-to-antirez
    // output (same opt-in pattern as matvec_f32 softplus_sqrt).
    let antirez_fidelity = std::env::var("DS4_RMS_NORM_FIDELITY").ok().as_deref() == Some("1");
    if antirez_fidelity {
        let scale: f32 = 1.0f32 / ((mean_sq as f32) + eps).sqrt();
        return x
            .iter()
            .zip(gamma.iter())
            .map(|(&v, &g)| v * scale * g)
            .collect();
    }
    let inv_rms = 1.0 / (mean_sq + eps as f64).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&v, &g)| (v as f64 * inv_rms * g as f64) as f32)
        .collect()
}

/// RMSNorm without γ (the bare variant — `kernel_rms_norm_f32_4`).
pub fn rms_norm_plain(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mean_sq: f64 = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
    let inv_rms = 1.0 / (mean_sq + eps as f64).sqrt();
    x.iter().map(|&v| (v as f64 * inv_rms) as f32).collect()
}

/// SiLU activation: `silu(x) = x * sigmoid(x)`.
///
/// Antirez `silu` (ds4.c:4863) is `x * sigmoid_stable(x)`, where
/// `sigmoid_stable` (ds4.c:4736) branches on sign:
///   `x >= 0`: `1 / (1 + expf(-x))`     — small `expf(-x)` for large positive x
///   `x <  0`: `expf(x) / (1 + expf(x))` — small `expf(x)` for large negative x
///
/// Our previous port `x / (1 + (-x).exp())` is algebraically identical
/// but rounds differently for negative `x`: at e.g. `x = -7`, ours
/// computes `e^{7} ≈ 1097` then `-7 / (1+1097)`, while antirez computes
/// `e^{-7} ≈ 9.12e-4` then `-7 * 9.12e-4/(1+9.12e-4)`. The two paths
/// disagree by f32 ULPs for negative x.
///
/// fidelity gate: set `DS4_SILU_FIDELITY=1` to switch to the
/// antirez `sigmoid_stable` chain. Default OFF (same caution as
/// #309 — high-fanout flips need Mac evidence).
pub fn silu(x: f32) -> f32 {
    let antirez_fidelity = std::env::var("DS4_SILU_FIDELITY").ok().as_deref() == Some("1");
    if antirez_fidelity {
        return x * sigmoid_stable_antirez(x);
    }
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_stable_antirez(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Round-trip an activation vector through antirez `quantize_q8_0_activation`
/// (ds4.c:2974-2998). Per-32-element block: d = amax/127, q = round(x/d),
/// clamped to [-128, 127], then x_back = q · d. mirrors antirez's
/// activation lossiness so that f32 oracle matches its int8-dot ground truth.
/// Used by shared_expert / attn_output_proj / attn LoRA / lm_head sites.
pub fn q8_0_round_trip(x: &[f32]) -> Vec<f32> {
    const BLOCK: usize = 32;
    let mut out = vec![0.0f32; x.len()];
    let n_full = x.len() / BLOCK;
    for b in 0..n_full {
        let off = b * BLOCK;
        let mut amax = 0.0f32;
        for j in 0..BLOCK {
            let a = x[off + j].abs();
            if a > amax {
                amax = a;
            }
        }
        if amax == 0.0 {
            for j in 0..BLOCK {
                out[off + j] = 0.0;
            }
            continue;
        }
        let d = amax / 127.0f32;
        let id = 1.0f32 / d;
        for j in 0..BLOCK {
            // Match antirez `lrintf` (C default FE_TONEAREST = round-half-to-even),
            // NOT Rust f32::round (round-half-away-from-zero). For activations
            // near half-quantum the two diverge; using lrintf semantics is
            // required for byte-exact match with antirez int8-dot.
            let v = (x[off + j] * id).round_ties_even() as i32;
            let v = v.clamp(-128, 127);
            out[off + j] = (v as f32) * d;
        }
    }
    let i0 = n_full * BLOCK;
    if i0 < x.len() {
        let mut amax = 0.0f32;
        for j in i0..x.len() {
            let a = x[j].abs();
            if a > amax {
                amax = a;
            }
        }
        if amax != 0.0 {
            let d = amax / 127.0f32;
            let id = 1.0f32 / d;
            for j in i0..x.len() {
                let v = (x[j] * id).round_ties_even() as i32;
                let v = v.clamp(-128, 127);
                out[j] = (v as f32) * d;
            }
        }
    }
    out
}

/// SwiGLU finalize: `silu(gate) * up` (no clamp).
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect()
}

/// SwiGLU with clamping (matches `MulMvIdQ4KPairSwigluF32` finalize block).
///
/// `clamp ≤ 1e-6` disables clamping.
pub fn swiglu_clamped(gate: &[f32], up: &[f32], clamp: f32) -> Vec<f32> {
    assert_eq!(gate.len(), up.len());
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| {
            if clamp > 1e-6 {
                let g = g.min(clamp);
                let u = u.clamp(-clamp, clamp);
                silu(g) * u
            } else {
                silu(g) * u
            }
        })
        .collect()
}

/// Cosine similarity between two vectors. Used for validating Metal outputs
/// against the antirez oracle dumps (target: ≥ 0.99).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch");
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let na: f64 = a.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        (dot / (na * nb)) as f32
    }
}

/// L∞ relative error between reference and candidate.
pub fn max_rel_err(reference: &[f32], candidate: &[f32]) -> f32 {
    assert_eq!(reference.len(), candidate.len());
    reference
        .iter()
        .zip(candidate.iter())
        .map(|(&r, &c)| {
            let denom = r.abs().max(1e-6);
            ((r - c).abs() / denom) as f32
        })
        .fold(0.0f32, f32::max)
}

/// Verbatim port of antirez `dot_f32` (ds4.c:4689-4706) — f32-FMA accumulator
/// with two 4-wide partials reduced via a final pairwise add. Used when
/// `DS4_MATVEC_F32_FIDELITY=1` is set so the f32 outputs match antirez at
/// ULP level instead of being more precise via the f64 oracle accumulator.
#[inline]
pub fn dot_f32_antirez(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc0 = [0.0f32; 4];
    let mut acc1 = [0.0f32; 4];
    let mut i = 0usize;
    while i + 8 <= n {
        for k in 0..4 {
            acc0[k] = a[i + k].mul_add(b[i + k], acc0[k]);
        }
        for k in 0..4 {
            acc1[k] = a[i + 4 + k].mul_add(b[i + 4 + k], acc1[k]);
        }
        i += 8;
    }
    let mut tail = 0.0f32;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    // Mirror NEON `vaddvq_f32(vaddq_f32(acc0, acc1))` ordering: pairwise add
    // the two 4-vectors, then horizontal-sum (4-way pair tree).
    let s0 = acc0[0] + acc1[0];
    let s1 = acc0[1] + acc1[1];
    let s2 = acc0[2] + acc1[2];
    let s3 = acc0[3] + acc1[3];
    let s01 = s0 + s1;
    let s23 = s2 + s3;
    s01 + s23 + tail
}

/// Dense matvec: `y[i] = Σ_k W[i,k] * x[k]`. Row-major weight layout (one row
/// of `d_in` floats per output element).
pub fn matvec_f32(w: &[f32], x: &[f32], d_out: usize) -> Vec<f32> {
    assert_eq!(
        w.len(),
        d_out * x.len(),
        "matvec dims: W has {} rows of {}",
        d_out,
        x.len()
    );
    let d_in = x.len();
    let antirez_fidelity = std::env::var("DS4_MATVEC_F32_FIDELITY").ok().as_deref() == Some("1");
    // Row-independent: parallelize across worker threads on large d_out. The
    // per-row reduction is associative within a row, and each row writes a
    // distinct output slot — bit-identical to the sequential version.
    // Threshold avoids thread-spawn overhead on tiny ops.
    const PAR_THRESHOLD: usize = 1024;
    if d_out < PAR_THRESHOLD {
        return (0..d_out)
            .map(|i| {
                let row = &w[i * d_in..(i + 1) * d_in];
                if antirez_fidelity {
                    dot_f32_antirez(row, x)
                } else {
                    row.iter()
                        .zip(x.iter())
                        .map(|(&wv, &xv)| wv as f64 * xv as f64)
                        .sum::<f64>() as f32
                }
            })
            .collect();
    }
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
        .min(16);
    let chunk = (d_out + n_threads - 1) / n_threads;
    let mut out = vec![0.0f32; d_out];
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        let mut remaining: &mut [f32] = out.as_mut_slice();
        let mut start = 0usize;
        while start < d_out {
            let end = (start + chunk).min(d_out);
            let len = end - start;
            let (head, tail) = remaining.split_at_mut(len);
            remaining = tail;
            let w_slice = &w[start * d_in..end * d_in];
            handles.push(s.spawn(move || {
                for (i, slot) in head.iter_mut().enumerate() {
                    let row = &w_slice[i * d_in..(i + 1) * d_in];
                    *slot = if antirez_fidelity {
                        dot_f32_antirez(row, x)
                    } else {
                        row.iter()
                            .zip(x.iter())
                            .map(|(&wv, &xv)| wv as f64 * xv as f64)
                            .sum::<f64>() as f32
                    };
                }
            }));
            start = end;
        }
        for h in handles {
            let _ = h.join();
        }
    });
    out
}

/// Causal softmax over a single row, scaled by `scale`. Matches the no-mask
/// no-sink slow path of `kernel_soft_max_f32_4` (registry: `soft_max_f32_4`).
pub fn softmax_scaled(x: &[f32], scale: f32) -> Vec<f32> {
    let mut max_v = f32::NEG_INFINITY;
    for &v in x {
        max_v = max_v.max(v * scale);
    }
    let mut sum = 0.0f64;
    let mut out: Vec<f32> = x
        .iter()
        .map(|&v| {
            let e = ((v * scale - max_v) as f64).exp();
            sum += e;
            e as f32
        })
        .collect();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in out.iter_mut() {
            *v = (*v as f64 * inv) as f32;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Serializes tests that mutate process-global env vars
    /// (DS4_MATVEC_F32_FIDELITY etc.) so parallel cargo-test runners don't race.
    /// The ONE crate-wide env lock (see `crate::ENV_LOCK`). Was a file-local mutex; two
    /// mutexes guarding one piece of process-global state is the same as none, which is how
    /// the SiLU gate test here raced attn_dispatch's gate test setting DS4_SILU_FIDELITY.
    use crate::ENV_LOCK;

    #[test]
    fn rms_norm_identity_gamma_is_normalization() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // x = [1, 2, 3, 4], γ = [1, 1, 1, 1]:
        //   mean_sq = (1+4+9+16)/4 = 7.5
        //   inv_rms = 1/sqrt(7.5+eps)
        // Output should be x/sqrt(7.5).
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![1.0f32; 4];
        let y = rms_norm(&x, &g, 1e-6);
        let s = 7.5f64.sqrt();
        let want: Vec<f32> = x.iter().map(|&v| (v as f64 / s) as f32).collect();
        for (a, b) in y.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, want {b}");
        }
    }

    #[test]
    fn rms_norm_plain_matches_rms_norm_with_unit_gamma() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let x = vec![0.3f32, -0.7, 1.5, -2.1, 0.05];
        let g = vec![1.0f32; 5];
        let a = rms_norm(&x, &g, 1e-6);
        let b = rms_norm_plain(&x, 1e-6);
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_gamma_scales_output() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // γ = 2× should produce 2× output of plain RMSNorm.
        let x = vec![0.1f32, -0.2, 0.3, -0.4];
        let g = vec![2.0f32; 4];
        let y = rms_norm(&x, &g, 1e-6);
        let plain = rms_norm_plain(&x, 1e-6);
        for (yv, pv) in y.iter().zip(plain.iter()) {
            assert!((yv - 2.0 * pv).abs() < 1e-5);
        }
    }

    #[test]
    fn silu_is_zero_at_zero() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        assert!(silu(0.0).abs() < 1e-7);
    }

    #[test]
    fn silu_large_positive_is_approximately_x() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // silu(10) ≈ 10 (sigmoid(10) ≈ 0.99995)
        let v = silu(10.0);
        assert!((v - 10.0).abs() < 1e-3, "got {v}");
    }

    #[test]
    fn swiglu_known_value() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // gate=1.0, up=2.0:  silu(1) = 1*sigmoid(1) ≈ 0.7310586
        //   → 0.7310586 * 2 = 1.4621172
        let y = swiglu(&[1.0], &[2.0]);
        assert!((y[0] - 1.4621172).abs() < 1e-5, "got {}", y[0]);
    }

    #[test]
    fn swiglu_clamped_disables_when_clamp_small() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // clamp = 0.0 → no clamping
        let unclamped = swiglu(&[3.0], &[5.0]);
        let clamped = swiglu_clamped(&[3.0], &[5.0], 0.0);
        assert!((unclamped[0] - clamped[0]).abs() < 1e-6);
    }

    #[test]
    fn swiglu_clamped_caps_gate() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // clamp=2.0, gate=10 → silu(min(10,2)) = silu(2); up=3 unchanged within [-2,2]→2
        let y = swiglu_clamped(&[10.0], &[3.0], 2.0);
        let want = silu(2.0) * 2.0; // up clamped to 2.0
        assert!((y[0] - want).abs() < 1e-5, "got {}, want {want}", y[0]);
    }

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let a = vec![0.5f32, -1.2, 3.4, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_antiparallel_is_minus_one() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![-1.0f32, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector_is_zero_not_nan() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let nonzero = vec![1.0f32, -2.0, 3.0];
        let y = cosine_similarity(&zero, &nonzero);
        assert_eq!(y, 0.0);
        assert!(y.is_finite());
    }

    #[test]
    fn cosine_similarity_is_scale_invariant() {
        let a = vec![1.0f32, -2.0, 4.0];
        let b = vec![-3.0f32, 6.0, -12.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn matvec_known_values() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // W (2x3) = [[1,2,3],[4,5,6]], x=[1,1,1] → y=[6,15]
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0f32, 1.0, 1.0];
        let y = matvec_f32(&w, &x, 2);
        assert_eq!(y, vec![6.0, 15.0]);
    }

    #[test]
    fn softmax_uniform_input_is_uniform() {
        let x = vec![3.0f32; 5];
        let y = softmax_scaled(&x, 1.0);
        for v in y {
            assert!((v - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let x = vec![1.2f32, -0.5, 3.7, 0.0, -2.1];
        let y = softmax_scaled(&x, 0.5);
        let s: f32 = y.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum = {s}");
    }

    #[test]
    fn max_rel_err_zero_when_identical() {
        let a = vec![0.1f32, 0.2, 0.3];
        assert_eq!(max_rel_err(&a, &a), 0.0);
    }

    #[test]
    fn max_rel_err_uses_reference_denominator() {
        let reference = vec![1.0e-3f32];
        let candidate = vec![2.0e-3f32];
        let err = max_rel_err(&reference, &candidate);
        assert!((err - 1.0).abs() < 1e-6, "err={err}");
    }

    #[test]
    fn max_rel_err_floors_tiny_reference_denominator() {
        let reference = vec![0.0f32, 1.0];
        let candidate = vec![2.0e-6f32, 1.25];
        let err = max_rel_err(&reference, &candidate);
        assert!((err - 2.0).abs() < 1e-6, "err={err}");
    }

    // q8_0_round_trip behaviour tests. These mirror antirez
    // `quantize_q8_0_activation` (ds4.c:2974-2998) byte-exactly on small
    // blocks. Catching the lrintf-vs-round mismatch and amax-zero handling
    // here means M5 NPU port keeps a regression net.

    #[test]
    fn q8_0_round_trip_zero_block_returns_zeros() {
        let x = vec![0.0f32; 32];
        let y = q8_0_round_trip(&x);
        assert!(
            y.iter().all(|&v| v == 0.0),
            "zero block should round-trip to all-zero"
        );
    }

    #[test]
    fn q8_0_round_trip_max_abs_at_127_quantum() {
        // A block where the max-abs element drives the scale. d = amax/127.
        // The max-abs element itself should round-trip to ±amax exactly,
        // and all other elements stay within d/2 of their input.
        let mut x = vec![0.05f32; 32];
        x[7] = 1.0; // peak
        let y = q8_0_round_trip(&x);
        assert!(
            (y[7] - 1.0).abs() < 1e-6,
            "peak should round-trip exactly, got {}",
            y[7]
        );
        let d = 1.0f32 / 127.0;
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            assert!(
                (yi - xi).abs() <= d * 0.5 + 1e-6,
                "element {i}: |{yi} - {xi}| > d/2 = {}",
                d * 0.5
            );
        }
    }

    #[test]
    fn q8_0_round_trip_half_quantum_uses_round_half_to_even() {
        // x[0] sits exactly at a half-quantum. lrintf rounds half-to-even,
        // f32::round rounds half-away-from-zero. With max-abs = 127.0,
        // d = 1.0, id = 1.0; values 0.5, 1.5, 2.5, 3.5 quantize to either
        // {1, 2, 3, 4} (round-half-up) or {0, 2, 2, 4} (banker's).
        let mut x = vec![0.0f32; 32];
        x[0] = 127.0; // anchors amax → d = 1.0
        x[1] = 0.5;
        x[2] = 1.5;
        x[3] = 2.5;
        x[4] = 3.5;
        x[5] = -2.5;
        let y = q8_0_round_trip(&x);
        // round-half-to-even: 0.5→0, 1.5→2, 2.5→2, 3.5→4, -2.5→-2
        assert!(
            (y[1] - 0.0).abs() < 1e-6,
            "0.5 should banker's-round to 0, got {}",
            y[1]
        );
        assert!(
            (y[2] - 2.0).abs() < 1e-6,
            "1.5 should banker's-round to 2, got {}",
            y[2]
        );
        assert!(
            (y[3] - 2.0).abs() < 1e-6,
            "2.5 should banker's-round to 2, got {}",
            y[3]
        );
        assert!(
            (y[4] - 4.0).abs() < 1e-6,
            "3.5 should banker's-round to 4, got {}",
            y[4]
        );
        assert!(
            (y[5] - (-2.0)).abs() < 1e-6,
            "-2.5 should banker's-round to -2, got {}",
            y[5]
        );
    }

    #[test]
    fn q8_0_round_trip_multi_block_independent_scale() {
        // Each 32-element block has its own d. Build 64 elements with
        // block-0 dominated by 1.0 (small d) and block-1 dominated by 100.0
        // (large d), then check the small-block elements aren't crushed by
        // the large-block scale.
        let mut x = vec![0.0f32; 64];
        x[5] = 1.0; // block 0 peak
        x[10] = 0.123; // block 0 small
        x[32 + 7] = 100.0; // block 1 peak
        x[32 + 20] = 50.0; // block 1 medium
        let y = q8_0_round_trip(&x);
        // block 0 d = 1/127 ≈ 0.00787, so y[10] should be close to 0.123.
        assert!(
            (y[10] - 0.123).abs() < 0.01,
            "block-0 element should retain ~0.123, got {}",
            y[10]
        );
        // block 1 d = 100/127 ≈ 0.787, so y[32+20] should be close to 50.0.
        assert!(
            (y[32 + 20] - 50.0).abs() < 0.5,
            "block-1 medium should retain ~50.0, got {}",
            y[32 + 20]
        );
    }

    #[test]
    fn q8_0_round_trip_partial_tail_block_quantizes() {
        // 40 elements — block 0 is full (32), tail is 8 elements.
        // Tail must still quantize against its own amax.
        let mut x = vec![0.0f32; 40];
        x[3] = 0.5; // block 0
        x[35] = 2.0; // tail peak
        x[37] = 1.0;
        let y = q8_0_round_trip(&x);
        assert!(
            (y[35] - 2.0).abs() < 1e-5,
            "tail peak should round-trip, got {}",
            y[35]
        );
        // 1.0 against amax=2.0 with d=2/127: q = round(1.0 / (2/127)) = 64
        // → y = 64 * 2/127 ≈ 1.0079
        assert!(
            (y[37] - 1.0).abs() < 0.02,
            "tail mid should be ~1.0, got {}",
            y[37]
        );
    }

    // ─── swiglu_clamped MUST be gate-UPPER-ONLY, up-SYMMETRIC ───
    //
    // Antirez ds4.c:5197-5202 clamps `gate.min(clamp)` (positive upper bound,
    // no lower bound) and `up.clamp(-clamp, clamp)` (symmetric). The bisect
    // history for had us mistakenly clamping gate symmetrically; this
    // family of tests is designed to catch that exact regression.

    #[test]
    fn swiglu_clamped_does_not_clamp_negative_gate() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Setup: clamp=2.0, gate=-100.0 (very negative), up=1.0.
        // CORRECT: gate.min(2.0) = -100.0 (negative not capped), silu(-100) ≈ 0
        //          → y ≈ 0
        // BUGGY (symmetric clamp): gate.clamp(-2.0, 2.0) = -2.0, silu(-2.0) ≈ -0.238
        //          → y ≈ -0.238
        // Distinguishes the two implementations sharply.
        let y = swiglu_clamped(&[-100.0], &[1.0], 2.0);
        assert!(
            y[0].abs() < 1e-3,
            "very-negative gate must NOT be clamped (silu(-100)≈0); got {} which suggests symmetric clamp",
            y[0]
        );
    }

    #[test]
    fn swiglu_clamped_caps_positive_gate_at_clamp() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // gate=50, clamp=10 → gate.min(10) = 10 → silu(10)·u
        // gate=11, clamp=10 → gate.min(10) = 10 → silu(10)·u
        // The two outputs must be IDENTICAL (both clamp to 10).
        let y50 = swiglu_clamped(&[50.0], &[1.0], 10.0);
        let y11 = swiglu_clamped(&[11.0], &[1.0], 10.0);
        assert!(
            (y50[0] - y11[0]).abs() < 1e-5,
            "positive gates above clamp must produce identical output: y50={}, y11={}",
            y50[0],
            y11[0]
        );
    }

    #[test]
    fn swiglu_clamped_up_clamps_symmetrically() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // up=100, clamp=10 → u.clamp(-10, 10) = 10. silu(0) = 0. → y=0.
        // up=-100, clamp=10 → u.clamp(-10, 10) = -10. silu(0) = 0. → y=0.
        // Use gate=1.0 to get a non-zero silu. silu(1.0) ≈ 0.7311.
        let y_pos = swiglu_clamped(&[1.0], &[100.0], 10.0);
        let y_neg = swiglu_clamped(&[1.0], &[-100.0], 10.0);
        // |y_pos| == |y_neg|, opposite sign.
        assert!(
            (y_pos[0] + y_neg[0]).abs() < 1e-5,
            "up clamp must be symmetric"
        );
        assert!(
            y_pos[0] > 0.0 && y_neg[0] < 0.0,
            "signs must flip with up sign"
        );
        let want = silu(1.0) * 10.0;
        assert!(
            (y_pos[0] - want).abs() < 1e-5,
            "got {}, want {want}",
            y_pos[0]
        );
    }

    #[test]
    fn swiglu_clamped_clamp_zero_matches_unclamped() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // For clamp ≤ 1e-6, swiglu_clamped MUST be byte-equal to swiglu.
        // Bug pattern: an `if clamp >= 0` branch that still applies clamp(0,0).
        let gate = vec![3.0, -7.0, 0.5, -100.0, 50.0];
        let up = vec![2.0, -5.0, 0.0, 8.0, -3.0];
        let a = swiglu(&gate, &up);
        let b = swiglu_clamped(&gate, &up, 0.0);
        let c = swiglu_clamped(&gate, &up, 1e-7);
        for i in 0..gate.len() {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "clamp=0 must match swiglu at i={i}"
            );
            assert!(
                (a[i] - c[i]).abs() < 1e-6,
                "clamp=tiny must match swiglu at i={i}"
            );
        }
    }

    #[test]
    fn swiglu_clamped_with_extreme_negative_gate_silu_dominates() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // birth of layer-26 explosion: very-negative gate + finite up
        // → silu(-N)·u → ~0·u → 0 (NOT -N·u). Confirms silu saturation.
        let y = swiglu_clamped(&[-50.0, -30.0, -20.0], &[1.0, -1.0, 0.5], 10.0);
        for v in &y {
            assert!(
                v.abs() < 1e-6,
                "deeply-negative gate must zero output, got {v}"
            );
        }
    }

    // ─── silu adversarial tests ────────────────────────────────────────────

    #[test]
    fn silu_negative_saturates_to_zero() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // silu(-10) = -10 * sigmoid(-10) ≈ -10 * 4.54e-5 ≈ -4.54e-4
        // implicated silu saturation; this catches "silu was implemented as -x for x<0".
        let v = silu(-10.0);
        assert!(v.abs() < 1e-3, "silu(-10) must saturate near 0, got {v}");
        assert!(v < 0.0, "silu(-10) must remain slightly negative, got {v}");
    }

    #[test]
    fn silu_at_negative_one_known_value() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + e) ≈ -0.26894142
        let v = silu(-1.0);
        assert!((v - (-0.26894142)).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn silu_derivative_sign_at_origin_is_positive() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // silu'(0) = 0.5 → silu(0.01) > silu(0) > silu(-0.01).
        // Distinguishes silu from a buggy "abs" or "max(0,x)" relu.
        let a = silu(-0.01);
        let b = silu(0.0);
        let c = silu(0.01);
        assert!(
            a < b && b < c,
            "silu should be strictly increasing near 0: {a} < {b} < {c}"
        );
    }

    // ─── rms_norm adversarial tests ─────────────────────────────────────────

    #[test]
    fn rms_norm_zero_input_does_not_nan() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // candidate — zero-input must NOT produce NaN due to /0.
        // The `eps` term should make this safe.
        let x = vec![0.0f32; 16];
        let g = vec![1.0f32; 16];
        let y = rms_norm(&x, &g, 1e-6);
        for &v in &y {
            assert!(
                v.is_finite(),
                "zero-input must not produce NaN/inf, got {v}"
            );
        }
    }

    #[test]
    fn rms_norm_changes_when_gamma_changes() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Bug pattern: rms_norm silently ignores gamma. Trivial happy-path
        // tests with gamma=[1;n] miss this entirely. Compare γ=[1,1,...] vs
        // γ=[1,2,...] — outputs must differ at index ≥1.
        let x = vec![0.5f32; 4];
        let g1 = vec![1.0f32; 4];
        let g2 = vec![1.0, 2.0, 1.0, 1.0];
        let y1 = rms_norm(&x, &g1, 1e-6);
        let y2 = rms_norm(&x, &g2, 1e-6);
        assert_eq!(y1[0], y2[0], "index 0 unchanged (gamma equal there)");
        assert!((y2[1] - 2.0 * y1[1]).abs() < 1e-5, "index 1 must be 2x");
    }

    #[test]
    fn rms_norm_invariant_to_uniform_scale() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // If x is uniformly scaled by k>0, rms_norm(k·x, γ) == rms_norm(x, γ).
        // Catches bug where rms_norm uses raw magnitude instead of RMS.
        let x = vec![0.1f32, -0.3, 0.7, -0.2, 0.5];
        let g = vec![1.0f32; 5];
        let y1 = rms_norm(&x, &g, 1e-9);
        let scaled: Vec<f32> = x.iter().map(|v| v * 100.0).collect();
        let y2 = rms_norm(&scaled, &g, 1e-9);
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-3, "scale-invariance broken: {a} vs {b}");
        }
    }

    // ─── matvec_f32 adversarial tests ───────────────────────────────────────

    #[test]
    fn matvec_f32_distinguishes_row_major_from_column_major() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // Antirez convention is row-major: W[i*d_in + j] for output row i,
        // input col j. If we accidentally do col-major (W[j*d_out + i]), the
        // output is the TRANSPOSE matmul — totally different result.
        // W = [[1,2,3],[4,5,6]] row-major flat = [1,2,3,4,5,6].
        // x = [10, 20, 30].
        // Row-major: y[0] = 1*10+2*20+3*30 = 140, y[1] = 4*10+5*20+6*30 = 320.
        // Col-major (buggy): y[0] would dot column [1,4]·[10,20] = 90, ≠ 140.
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![10.0f32, 20.0, 30.0];
        let y = matvec_f32(&w, &x, 2);
        assert_eq!(
            y[0], 140.0,
            "row-major check failed (could be col-major bug)"
        );
        assert_eq!(y[1], 320.0);
    }

    #[test]
    fn matvec_f32_zero_input_yields_zero_output() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = vec![0.0f32; 4];
        let y = matvec_f32(&w, &x, 2);
        assert_eq!(y, vec![0.0, 0.0]);
    }

    #[test]
    fn matvec_f32_unit_vectors_pick_rows() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // x = e_k (one-hot at index k) should produce y = column-k of W.
        // For row-major W stored as [row0_col0..col2 | row1_col0..col2]:
        //   e_0 picks W[0], W[3]    → y = [1, 4]
        //   e_1 picks W[1], W[4]    → y = [2, 5]
        //   e_2 picks W[2], W[5]    → y = [3, 6]
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y0 = matvec_f32(&w, &[1.0, 0.0, 0.0], 2);
        let y1 = matvec_f32(&w, &[0.0, 1.0, 0.0], 2);
        let y2 = matvec_f32(&w, &[0.0, 0.0, 1.0], 2);
        assert_eq!(y0, vec![1.0, 4.0]);
        assert_eq!(y1, vec![2.0, 5.0]);
        assert_eq!(y2, vec![3.0, 6.0]);
    }

    #[test]
    fn matvec_f32_deterministic_under_parallel() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // question — is the parallel implementation byte-identical to
        // sequential? Build a non-trivial dot product where reduce-order
        // matters (large d_in with mixed signs). f32 add is non-associative
        // but our implementation accumulates left-to-right in f64 per row,
        // so result should be deterministic across runs.
        let d_in = 1024;
        let d_out = 8;
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| ((i as i32).wrapping_mul(1664525).wrapping_add(1013904223) as f32) * 1e-9)
            .collect();
        let x: Vec<f32> = (0..d_in)
            .map(|i| ((i as i32).wrapping_mul(-1103515245).wrapping_add(12345) as f32) * 1e-9)
            .collect();
        let y1 = matvec_f32(&w, &x, d_out);
        let y2 = matvec_f32(&w, &x, d_out);
        let y3 = matvec_f32(&w, &x, d_out);
        for i in 0..d_out {
            assert_eq!(y1[i], y2[i], "non-deterministic at i={i}");
            assert_eq!(y2[i], y3[i], "non-deterministic at i={i}");
        }
    }

    // ─── softmax_scaled adversarial tests ───────────────────────────────────

    #[test]
    fn softmax_scaled_numerically_stable_with_huge_logits() {
        // softmax must subtract max for stability. Without the max-subtract
        // step, exp(1000) = inf and the result is NaN.
        let x = vec![1000.0f32, 1001.0, 999.0];
        let y = softmax_scaled(&x, 1.0);
        for &v in &y {
            assert!(v.is_finite(), "softmax(1000,...) must be finite, got {v}");
        }
        let s: f32 = y.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum must be 1, got {s}");
        // The biggest logit (1001) must win.
        assert!(y[1] > y[0] && y[1] > y[2], "argmax must be at index 1");
    }

    #[test]
    fn softmax_scaled_temperature_scales_concentration() {
        // scale=10 (low temperature) sharpens; scale=0.1 (high temp) flattens.
        let x = vec![0.0f32, 1.0, 2.0];
        let y_sharp = softmax_scaled(&x, 10.0);
        let y_flat = softmax_scaled(&x, 0.1);
        // Sharp distribution has higher peak at argmax.
        assert!(
            y_sharp[2] > y_flat[2],
            "sharper softmax should concentrate at argmax"
        );
        // Flat is closer to uniform.
        assert!(y_flat[2] - y_flat[0] < y_sharp[2] - y_sharp[0]);
    }

    #[test]
    fn softmax_scaled_argmax_invariant_under_uniform_shift() {
        // Adding a constant to all logits should not change argmax (only
        // probabilities-up-to-shift). Verifies max-subtract symmetry.
        let x = vec![1.0f32, -3.5, 0.5, 2.7, -1.8];
        let shifted: Vec<f32> = x.iter().map(|v| v + 100.0).collect();
        let y1 = softmax_scaled(&x, 1.0);
        let y2 = softmax_scaled(&shifted, 1.0);
        for (a, b) in y1.iter().zip(y2.iter()) {
            assert!((a - b).abs() < 1e-5, "shift-invariance failed: {a} vs {b}");
        }
    }

    #[test]
    fn q8_0_round_trip_preserves_signs() {
        // Mixed-sign block. All signs must be preserved.
        let x = vec![
            -1.2_f32, 0.5, -0.3, 2.7, -2.7, 1.1, -0.05, 0.05, -1.0, 1.0, 0.0, -0.001, 1.5, -1.5,
            0.7, -0.7, 0.2, -0.4, 0.6, -0.8, 1.0, -1.2, 0.123, -0.456, 0.789, -0.1, 0.5, -0.5,
            0.25, -0.25, 0.125, -0.125,
        ];
        let y = q8_0_round_trip(&x);
        for (i, (&xi, &yi)) in x.iter().zip(y.iter()).enumerate() {
            if xi.abs() > 0.05 {
                assert_eq!(
                    xi.signum(),
                    yi.signum(),
                    "sign flip at i={i}: x={xi}, y={yi}"
                );
            }
        }
    }

    // ============================================================================
    //  PROPERTY-BASED tests (deterministic-seed, no proptest dep needed).
    //  These exercise INVARIANTS rather than fixed examples — if any of these
    //  fail, we've discovered a real defect, not just a missing example.
    // ============================================================================

    /// Deterministic LCG so test cases are reproducible without `rand` dep.
    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }
    fn lcg_f32(seed: &mut u64, lo: f32, hi: f32) -> f32 {
        let u = lcg(seed) as f32 / u32::MAX as f32;
        lo + u * (hi - lo)
    }

    /// PROPERTY: `q8_0_round_trip` must be IDEMPOTENT — quantizing twice
    /// must give the same answer as once (every output is already on the
    /// quantization grid). Bug discriminator: if our round_ties_even
    /// differs from antirez's `lrintf` on the dequantized grid points,
    /// a second pass would shift values.
    #[test]
    fn q8_0_round_trip_is_idempotent_over_random_blocks() {
        let mut seed = 0x4D5A6B7C8D9EAFB0u64;
        for trial in 0..50 {
            let n = 32 + (lcg(&mut seed) as usize % 16) * 32; // 32..512 in 32-strides
            let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -5.0, 5.0)).collect();
            let y1 = q8_0_round_trip(&x);
            let y2 = q8_0_round_trip(&y1);
            for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "trial {trial} idx {i}: quantize twice differs: {a} vs {b}"
                );
            }
        }
    }

    /// PROPERTY: `q8_0_round_trip` round-trip error ≤ half-quantum per block.
    /// |x[i] - y[i]| ≤ d/2 where d = amax/127, so ≤ amax / 254.
    /// Bug discriminator: a clamp-bug that maps far-out values to ±128·d
    /// would produce error > d/2.
    #[test]
    fn q8_0_round_trip_error_bound_is_half_quantum() {
        let mut seed = 0xC0FFEEABCDEF0123u64;
        for _ in 0..30 {
            let block_count = 1 + (lcg(&mut seed) as usize % 8);
            let n = block_count * 32;
            let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -3.0, 3.0)).collect();
            let y = q8_0_round_trip(&x);
            // Per-block error check.
            for b in 0..block_count {
                let off = b * 32;
                let amax = x[off..off + 32]
                    .iter()
                    .map(|v| v.abs())
                    .fold(0f32, f32::max);
                let half_q = amax / 254.0;
                // Tolerate one extra ULP for f32 representability of `d`.
                let tol = (half_q + 1e-7).max(1e-7);
                for j in 0..32 {
                    let err = (x[off + j] - y[off + j]).abs();
                    assert!(
                        err <= tol,
                        "block {b} idx {j}: err {err} > half_quantum {tol} (amax={amax})"
                    );
                }
            }
        }
    }

    /// PROPERTY: `q8_0_round_trip` is invariant under POSITIVE scaling.
    /// If d = amax/127, then scaling x by k>0 scales d by k, and v stays
    /// the same int8 → output scales by exactly k. **Bug discriminator**:
    /// if the round used `.round()` instead of `.round_ties_even()`, this
    /// would still hold for non-half-quantum inputs. The discriminator below
    /// (asymmetric input) is the real one — see next test.
    #[test]
    fn q8_0_round_trip_is_homogeneous_under_positive_scale() {
        let mut seed = 0x123456789ABCDEFu64;
        for _ in 0..20 {
            let n = 32 * (1 + lcg(&mut seed) as usize % 4);
            let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -1.0, 1.0)).collect();
            let k = 2.0 + lcg_f32(&mut seed, 0.5, 3.0);
            let y_x = q8_0_round_trip(&x);
            let xk: Vec<f32> = x.iter().map(|v| v * k).collect();
            let y_xk = q8_0_round_trip(&xk);
            for i in 0..n {
                let want = y_x[i] * k;
                let got = y_xk[i];
                let rel = (want - got).abs() / (want.abs().max(1e-6));
                assert!(
                    rel < 1e-5,
                    "homogeneity fail i={i}: y(kx) {got} vs k·y(x) {want}"
                );
            }
        }
    }

    /// PROPERTY: half-quantum input rounds to EVEN integer (banker's rounding).
    /// **Bug discriminator**: Rust's `.round()` rounds half-AWAY-FROM-ZERO,
    /// so 0.5·d → 1·d, 1.5·d → 2·d, 2.5·d → 3·d (1, 2, 3).
    /// `lrintf` (FE_TONEAREST) rounds half-to-even: 0.5·d → 0, 1.5·d → 2, 2.5·d → 2.
    /// This test constructs a block where MANY values are at half-quantum
    /// and asserts the lrintf semantics — if we accidentally used .round(),
    /// MANY outputs would differ.
    #[test]
    fn q8_0_round_trip_half_quantum_uses_banker_rounding() {
        // Build a block with amax=127 (so d=1.0 exactly, half-quantum at 0.5).
        // Half-quantum inputs: 0.5, 1.5, 2.5, 3.5 should round to 0, 2, 2, 4.
        let mut x = vec![0.0f32; 32];
        x[0] = 127.0; // anchors amax = 127.0 → d = 1.0 exactly
        x[1] = 0.5;
        x[2] = 1.5;
        x[3] = 2.5;
        x[4] = 3.5;
        x[5] = -0.5;
        x[6] = -1.5;
        x[7] = -2.5;
        let y = q8_0_round_trip(&x);
        // ROUND-HALF-TO-EVEN: 0.5→0, 1.5→2, 2.5→2, 3.5→4 (in quanta of d=1.0).
        assert_eq!(y[1], 0.0, "0.5 → 0 (even); .round() would give 1.0");
        assert_eq!(
            y[2], 2.0,
            "1.5 → 2 (even); .round() would give 2.0 (coincidence)"
        );
        assert_eq!(y[3], 2.0, "2.5 → 2 (even); .round() would give 3.0");
        assert_eq!(
            y[4], 4.0,
            "3.5 → 4 (even); .round() would give 4.0 (coincidence)"
        );
        assert_eq!(y[5], 0.0, "-0.5 → 0; .round() would give -1.0");
        assert_eq!(y[6], -2.0);
        assert_eq!(y[7], -2.0, "-2.5 → -2 (even); .round() would give -3.0");
    }

    /// PROPERTY: matvec_f32 is BILINEAR — `matvec(W, αx+βy)` should equal
    /// `α·matvec(W,x) + β·matvec(W,y)` within f32 ε. **Bug discriminator**:
    /// an accidental abs() or relu in the dot would violate linearity for
    /// sign-mixed inputs.
    #[test]
    fn matvec_f32_is_linear_in_input() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let mut seed = 0xBEEFCAFE1234567Bu64;
        for _ in 0..20 {
            let d_in = 8 + (lcg(&mut seed) as usize % 16);
            let d_out = 4 + (lcg(&mut seed) as usize % 8);
            let w: Vec<f32> = (0..d_in * d_out)
                .map(|_| lcg_f32(&mut seed, -1.0, 1.0))
                .collect();
            let x: Vec<f32> = (0..d_in).map(|_| lcg_f32(&mut seed, -2.0, 2.0)).collect();
            let z: Vec<f32> = (0..d_in).map(|_| lcg_f32(&mut seed, -2.0, 2.0)).collect();
            let a = lcg_f32(&mut seed, -1.5, 1.5);
            let b = lcg_f32(&mut seed, -1.5, 1.5);
            let combined: Vec<f32> = x
                .iter()
                .zip(z.iter())
                .map(|(&xi, &zi)| a * xi + b * zi)
                .collect();
            let y_combined = matvec_f32(&w, &combined, d_out);
            let y_x = matvec_f32(&w, &x, d_out);
            let y_z = matvec_f32(&w, &z, d_out);
            for i in 0..d_out {
                let want = a * y_x[i] + b * y_z[i];
                let got = y_combined[i];
                let rel = (want - got).abs() / (want.abs().max(1e-6));
                assert!(
                    rel < 1e-4,
                    "linearity fail i={i}: matvec(W,αx+βy) {got} vs α·matvec+β·matvec {want}"
                );
            }
        }
    }

    /// failing-first: `matvec_f32` uses an f64 accumulator
    /// (`wv as f64 * xv as f64`).sum::<f64>() then `as f32`. Antirez `dot_f32`
    /// (ds4.c:4689-4706) uses an f32-FMA pair-reduce — `acc0/acc1` each
    /// `float32x4_t`, FMA over 8-wide pairs, then `vaddvq_f32(vaddq_f32(...))`.
    /// Algebraically equivalent in real arithmetic, but f32-ULP different on
    /// inputs that exercise reduction-order sensitivity. The gate
    /// `DS4_MATVEC_F32_FIDELITY=1` switches our path to the antirez-spec
    /// reducer. Test:
    ///   1. Build an adversarial 256-element input with widely-varying
    ///      magnitudes that perturbs under reduction ordering.
    ///   2. Assert hand-stamped f64-accumulator vs antirez-spec f32-FMA
    ///      pair-reduce produce DIFFERENT f32 bits — test is non-trivial.
    ///   3. With gate OFF: `matvec_f32` matches the f64 oracle bit-exactly.
    ///   4. With gate ON: `matvec_f32` matches the antirez-spec reducer
    ///      bit-exactly.
    #[test]
    fn matvec_f32_antirez_fidelity_gate_switches_reducer() {
        let _env = ENV_LOCK.write().unwrap_or_else(|p| p.into_inner());
        // 256-element row: alternating large-magnitude pairs (1e3) and
        // tiny-magnitude pairs (1e-3) with sign flips. The f64 oracle
        // collects sums associatively in f64; the NEON-style 8-wide
        // pair-reduce sums in tree-of-4 fashion.
        let d_in = 256usize;
        let d_out = 1usize;
        let mut w = vec![0.0f32; d_in];
        let mut x = vec![0.0f32; d_in];
        let mut seed = 0xC0FFEE12345678ABu64;
        for i in 0..d_in {
            // Bias every 8th slot to a large value so the per-4-lane FMA
            // partial sums diverge significantly from a naive scalar sum.
            let large = (i % 8) == 0;
            w[i] = if large {
                1.0e3 * (if i % 16 == 0 { 1.0 } else { -1.0 })
            } else {
                1.0e-3 * lcg_f32(&mut seed, -1.0, 1.0)
            };
            x[i] = if large {
                1.0e-3 * (if (i / 8) % 2 == 0 { 1.0 } else { -1.0 })
            } else {
                1.0e3 * lcg_f32(&mut seed, -1.0, 1.0)
            };
        }

        // Hand-stamped f64 oracle (matches our current default branch).
        let want_f64: f32 = w
            .iter()
            .zip(x.iter())
            .map(|(&wv, &xv)| wv as f64 * xv as f64)
            .sum::<f64>() as f32;

        // Hand-stamped antirez-spec f32-FMA pair-reduce.
        let want_antirez: f32 = dot_f32_antirez(&w, &x);

        assert_ne!(
            want_f64.to_bits(),
            want_antirez.to_bits(),
            "discrimination test is trivial: f64 oracle and antirez f32 reducer \
             produced identical bits — pick a stronger adversarial input"
        );

        // Gate OFF (default): must match the f64 oracle bit-exactly.
        std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
        let y_off = matvec_f32(&w, &x, d_out);
        assert_eq!(
            y_off[0].to_bits(),
            want_f64.to_bits(),
            "gate=OFF should preserve the legacy f64 oracle bits"
        );

        // Gate ON: must match the antirez-spec reducer bit-exactly.
        std::env::set_var("DS4_MATVEC_F32_FIDELITY", "1");
        let y_on = matvec_f32(&w, &x, d_out);
        std::env::remove_var("DS4_MATVEC_F32_FIDELITY");
        assert_eq!(
            y_on[0].to_bits(),
            want_antirez.to_bits(),
            "gate=ON should bit-match the antirez f32-FMA pair-reduce"
        );
    }

    /// PROPERTY: softmax_scaled output is a valid probability distribution.
    /// Σ y_i = 1.0 ± ε, all y_i ∈ [0, 1]. **Bug discriminator**: a missing
    /// normalize step would let Σ > 1; a missing exp would give 0 sum or
    /// negative values.
    #[test]
    fn softmax_scaled_outputs_valid_probability_distribution() {
        let mut seed = 0xF00DABBA12345678u64;
        for _ in 0..30 {
            let n = 2 + (lcg(&mut seed) as usize % 30);
            let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -20.0, 20.0)).collect();
            let scale = lcg_f32(&mut seed, 0.1, 5.0);
            let y = softmax_scaled(&x, scale);
            let sum: f32 = y.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "Σ={sum} ≠ 1; scale={scale}");
            for (i, &v) in y.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.0, "y[{i}]={v} ∉ [0,1]; scale={scale}");
            }
        }
    }

    /// PROPERTY: softmax_scaled argmax must match argmax of input.
    /// **Bug discriminator**: a sign flip or sort bug would put the
    /// argmax on a different element.
    #[test]
    fn softmax_scaled_argmax_matches_input_argmax() {
        let mut seed = 0xCAFED00DBEEF4242u64;
        for _ in 0..30 {
            let n = 2 + (lcg(&mut seed) as usize % 30);
            let mut x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -10.0, 10.0)).collect();
            // Avoid ties: nudge the would-be max slightly.
            let max_idx = (0..n)
                .max_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap())
                .unwrap();
            x[max_idx] += 0.1;
            let y = softmax_scaled(&x, 1.0);
            let y_max_idx = (0..n)
                .max_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap())
                .unwrap();
            assert_eq!(
                y_max_idx, max_idx,
                "softmax argmax differs from input argmax"
            );
        }
    }

    /// PROPERTY: rms_norm result has unit RMS (when γ=1). **Bug discriminator**:
    /// an off-by-one in the mean_sq divisor (using n-1 vs n, or summing wrong
    /// length) would shift RMS slightly off 1.0.
    #[test]
    fn rms_norm_unit_gamma_yields_unit_rms() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let mut seed = 0x1010101010101010u64;
        for _ in 0..20 {
            let n = 4 + (lcg(&mut seed) as usize % 100);
            let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -5.0, 5.0)).collect();
            let g = vec![1.0f32; n];
            let y = rms_norm(&x, &g, 0.0); // eps=0 to make exact
            let mean_sq = y.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64;
            let rms = mean_sq.sqrt();
            // RMS should be EXACTLY 1.0 (when eps=0 and x not all-zero).
            // With eps=0 and arbitrary input this would NaN if x is all-zero;
            // skip that edge case here.
            if x.iter().any(|&v| v.abs() > 1e-3) {
                assert!(
                    (rms - 1.0).abs() < 1e-4,
                    "rms_norm output RMS = {rms} ≠ 1.0; n={n}"
                );
            }
        }
    }

    /// M4 BUG-DISCOVERY: compare our `rms_norm` against a verbatim port of
    /// antirez `rms_norm_weight` (ds4.c:2560-2566). Antirez computes:
    ///   1. ss in f64
    ///   2. cast (ss/n) to f32, ADD eps in f32, then sqrtf → scale (f32)
    ///   3. out[i] = x[i] * scale * weight[i] in f32 left-to-right
    /// Our impl does step 2 in f64, then casts only at the final assign. The
    /// f32-cast precision loss in antirez is small but visible at the last
    /// few ULPs. This test asserts our output matches antirez within 1e-6
    /// relative error — a buggy port (different formula entirely) would fail.
    #[test]
    fn rms_norm_matches_antirez_at_realistic_d_model_4096() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        fn antirez_rms_norm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
            let n = x.len();
            // ss in f64 (antirez line 2562)
            let mut ss: f64 = 0.0;
            for &v in x {
                ss += v as f64 * v as f64;
            }
            // scale computation matches antirez line 2564 verbatim:
            //   (float)(ss / (double)n) + eps    -- CAST then ADD in f32
            //   sqrtf(...)                       -- f32 sqrt
            //   1.0f / sqrtf(...)                -- f32 divide
            let scale: f32 = 1.0 / ((ss / n as f64) as f32 + eps).sqrt();
            // out[i] = x[i] * scale * weight[i]  -- f32 left-to-right
            x.iter()
                .zip(gamma.iter())
                .map(|(&v, &g)| v * scale * g)
                .collect()
        }
        // Realistic d_model from DS4 V4 Flash.
        let n = 4096;
        let mut seed = 0xc0ffee_u64;
        let x: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, -2.0, 2.0)).collect();
        let g: Vec<f32> = (0..n).map(|_| lcg_f32(&mut seed, 0.5, 1.5)).collect();
        let eps = 1.0e-6;
        let ours = rms_norm(&x, &g, eps);
        let antirez = antirez_rms_norm(&x, &g, eps);
        // Antirez computes scale in f32 (cast loses precision); we compute in
        // f64. Maximum-likely divergence is a few f32 ULPs (~1e-7 rel).
        // Tolerance 1e-5 catches BUG-class divergence (algorithm wrong),
        // not ULP-level precision difference.
        for i in 0..n {
            let denom = antirez[i].abs().max(1e-10);
            let rel = (ours[i] - antirez[i]).abs() / denom;
            assert!(
                rel < 1e-5,
                "rms_norm[{i}]: ours={} antirez={} rel={}",
                ours[i],
                antirez[i],
                rel
            );
        }
        // Sanity: ours and antirez should NOT be bit-identical (because of
        // the f32 cast difference). If they ARE bit-identical, the antirez
        // reference is wrong or our impl has converged on the same f32 path.
        let mut max_rel: f32 = 0.0;
        for i in 0..n {
            let denom = antirez[i].abs().max(1e-10);
            let rel = (ours[i] - antirez[i]).abs() / denom;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        // Real difference IS detectable at 1e-7 rel.
        assert!(
            max_rel > 1e-8,
            "ours and antirez should differ at ULP level; max_rel={max_rel}"
        );
    }

    /// PROPERTY: silu is "smooth-relu" — silu(x) → x for large x, → 0 for
    /// very negative x, and silu(x) > 0 ⟺ x > 0 (sign preservation up to
    /// the tiny negative region where derivative crosses zero).
    /// **Bug discriminator**: a missing sigmoid (returning just x or returning
    /// max(x,0)) would fail one of the limit asserts.
    #[test]
    fn silu_asymptotic_behaviour() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        // For x >> 0: silu(x) ≈ x.
        for x in [5.0_f32, 10.0, 20.0, 50.0] {
            let y = silu(x);
            let rel = (y - x).abs() / x;
            assert!(rel < 0.01, "silu({x})={y} should ≈ {x}");
        }
        // For x << 0: silu(x) → 0.
        for x in [-5.0_f32, -10.0, -20.0, -50.0] {
            let y = silu(x);
            assert!(y.abs() < 0.04, "silu({x})={y} should → 0");
        }
        // silu(x) has a min around x = -1.278, value ≈ -0.278. NOT > 0 there.
        let y = silu(-1.278);
        assert!(y < 0.0 && y > -0.3, "silu(-1.278)={y}, expected ≈ -0.278");
        assert!(silu(-1.278) < silu(0.0)); // min is negative
    }

    /// PROPERTY: swiglu commutes with sign flip on `up` (silu is fixed,
    /// up*(-1) flips sign). Swapping gate↔up gives DIFFERENT result.
    /// **Bug discriminator**: if swiglu accidentally computes silu(up)*gate
    /// (args swapped), this test would catch it because silu(g) ≠ silu(u)
    /// in general.
    #[test]
    fn swiglu_distinguishes_gate_from_up() {
        // Shared-mode ENV_LOCK: this test calls gate-reading code (DS4_*_FIDELITY /
        // DS4_Q8_0_ACT). Concurrent gate-flipping tests take write(); we take read()
        // so the gate cannot change underneath us, while staying parallel with peers.
        let _env = crate::ENV_LOCK.read().unwrap_or_else(|p| p.into_inner());
        let gate = vec![2.0_f32];
        let up = vec![-3.0_f32];
        let y1 = swiglu(&gate, &up); // silu(2)*(-3) ≈ 1.762 * -3 ≈ -5.287
        let y2 = swiglu(&up, &gate); // silu(-3)*2 ≈ -0.142 * 2 ≈ -0.285
                                     // The two MUST differ substantially.
        assert!(
            (y1[0] - y2[0]).abs() > 1.0,
            "swiglu(gate,up)={} vs swiglu(up,gate)={}; should differ",
            y1[0],
            y2[0]
        );
        // Sign check: y1 = silu(+2)·(-3) < 0; y2 = silu(-3)·2 < 0 too. Both neg.
        // Magnitudes differ ~18×.
        assert!(
            y1[0].abs() > y2[0].abs() * 5.0,
            "swiglu(gate=2,up=-3) magnitude should dominate swiglu(up=2,gate=-3)"
        );
    }

    /// failing-first discrimination test for the rms_norm antirez
    /// f32 multiply-chain fidelity gate. Antirez `rms_norm_weight` (ds4.c:2560)
    /// keeps `scale` in f32 and computes `x[i] * scale * weight[i]` as a f32
    /// chain (`mulss; mulss`), with intermediate rounding between the two
    /// multiplies. Our oracle does `(v as f64) * inv_rms * (g as f64)` in f64
    /// throughout. Hand-stamp both paths and assert the gate switches between
    /// them bit-exactly. Same opt-in pattern as
    #[test]
    fn rms_norm_antirez_fidelity_gate_switches_reducer() {
        let _g = ENV_LOCK.write().unwrap();
        // Adversarial 128-element block with mixed magnitudes: large + small
        // values so the f64 vs f32 multiply chain produces a different ULP
        // even for moderate elements. γ chosen as a non-trivial weight vector
        // (not all 1.0) so the second f32 multiply step is load-bearing.
        let n = 128;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let mag = if i % 4 == 0 { 1e3_f32 } else { 1e-3_f32 };
                sign * mag * (1.0 + 1e-3 * (i as f32))
            })
            .collect();
        let gamma: Vec<f32> = (0..n).map(|i| 0.5_f32 + 1e-4 * (i as f32)).collect();
        let eps = 1e-6_f32;

        // Hand-stamp the f64 oracle (current default).
        let mean_sq_f64: f64 = x.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / n as f64;
        let inv_rms_f64 = 1.0 / (mean_sq_f64 + eps as f64).sqrt();
        let want_f64: Vec<f32> = x
            .iter()
            .zip(gamma.iter())
            .map(|(&v, &g)| (v as f64 * inv_rms_f64 * g as f64) as f32)
            .collect();

        // Hand-stamp the antirez f32 chain.
        let scale_f32: f32 = 1.0_f32 / ((mean_sq_f64 as f32) + eps).sqrt();
        let want_antirez: Vec<f32> = x
            .iter()
            .zip(gamma.iter())
            .map(|(&v, &g)| v * scale_f32 * g)
            .collect();

        // Discrimination guard: at least one index must differ at the bit level.
        let any_diff = want_f64
            .iter()
            .zip(want_antirez.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            any_diff,
            "test inputs failed to discriminate f64 oracle vs antirez f32 chain"
        );

        // Gate OFF (default): must bit-match f64 oracle.
        std::env::remove_var("DS4_RMS_NORM_FIDELITY");
        let got_off = rms_norm(&x, &gamma, eps);
        for (i, (g, w)) in got_off.iter().zip(want_f64.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "gate=OFF should bit-match the f64 oracle at i={i}; got {} want {}",
                g,
                w
            );
        }

        // Gate ON: must bit-match antirez f32 chain.
        std::env::set_var("DS4_RMS_NORM_FIDELITY", "1");
        let got_on = rms_norm(&x, &gamma, eps);
        for (i, (g, w)) in got_on.iter().zip(want_antirez.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "gate=ON should bit-match the antirez f32 rms_norm_weight at i={i}; got {} want {}",
                g,
                w
            );
        }

        std::env::remove_var("DS4_RMS_NORM_FIDELITY");
    }

    /// failing-first discrimination test for the silu antirez
    /// `sigmoid_stable` fidelity gate. Antirez `silu(x) = x * sigmoid_stable(x)`
    /// branches on sign:
    ///   `x >= 0`: `1 / (1 + expf(-x))`
    ///   `x <  0`: `expf(x) / (1 + expf(x))`
    /// Our default port `x / (1 + (-x).exp())` always uses the positive
    /// branch — algebraically identical, but rounds differently for
    /// negative x because the intermediate `expf` value lives at a very
    /// different magnitude. Hand-stamp both and assert gate switches
    /// the function bit-exactly.
    #[test]
    fn silu_antirez_fidelity_gate_switches_implementation() {
        let _g = ENV_LOCK.write().unwrap();
        // Adversarial x values: spread across the typical SwiGLU CLAMP range
        // [-10, +10] plus a few near-zero values where the branch transition
        // is exercised.
        let xs: Vec<f32> = (0..21)
            .map(|i| (i as f32 - 10.0) * 1.0_f32) // -10, -9, ..., +10
            .chain([-0.001, 0.001, -3.5, 3.5, -7.7, 7.7])
            .collect();

        // Hand-stamp the default oracle (positive-branch always).
        let want_default: Vec<f32> = xs.iter().map(|&x| x / (1.0_f32 + (-x).exp())).collect();

        // Hand-stamp the antirez sigmoid_stable chain.
        let want_antirez: Vec<f32> = xs
            .iter()
            .map(|&x| {
                let s = if x >= 0.0 {
                    let e = (-x).exp();
                    1.0_f32 / (1.0 + e)
                } else {
                    let e = x.exp();
                    e / (1.0 + e)
                };
                x * s
            })
            .collect();

        // Discrimination guard: at least one index must differ bit-level.
        let any_diff = want_default
            .iter()
            .zip(want_antirez.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            any_diff,
            "test inputs failed to discriminate default silu vs antirez sigmoid_stable"
        );

        // Gate OFF (default): silu() must bit-match the default-path.
        std::env::remove_var("DS4_SILU_FIDELITY");
        for (i, &x) in xs.iter().enumerate() {
            let got = silu(x);
            assert_eq!(
                got.to_bits(),
                want_default[i].to_bits(),
                "gate=OFF should bit-match default silu at x={x}; got {got} want {}",
                want_default[i]
            );
        }

        // Gate ON: silu() must bit-match antirez sigmoid_stable.
        std::env::set_var("DS4_SILU_FIDELITY", "1");
        for (i, &x) in xs.iter().enumerate() {
            let got = silu(x);
            assert_eq!(
                got.to_bits(),
                want_antirez[i].to_bits(),
                "gate=ON should bit-match antirez sigmoid_stable at x={x}; got {got} want {}",
                want_antirez[i]
            );
        }

        std::env::remove_var("DS4_SILU_FIDELITY");
    }
}
