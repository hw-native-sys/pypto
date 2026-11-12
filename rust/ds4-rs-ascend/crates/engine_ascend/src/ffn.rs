// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! DS4-Flash routed-expert FFN, computed on the CPU over the loader's packed planes.
//!
//! The finish line for this target is "NPU argmax == CPU argmax on the real model", so a CPU
//! reference is not optional scaffolding — it is half the acceptance test. This module is that
//! half for the routed FFN, built on the same `PackedWeight` planes the device kernels read, so
//! the two paths share an input format and differ only in who does the arithmetic.
//!
//! It is deliberately dense and allocation-heavy. Correctness against an oracle is the goal;
//! the device path is where speed lives.

use crate::mxfp4_repack::{matvec_ref, permute_activation, PackedWeight};
use anyhow::Result;

/// DS4's SwiGLU, including the clamp — and note that the clamp is ASYMMETRIC.
///
/// `silu(min(gate, c)) * clamp(up, -c, c)`: the gate is bounded ABOVE only, the up projection
/// on both sides. `engine_core` pins this against antirez's own "DS4 clamps the shared expert
/// with the same swiglu_limit used by routed experts", and its test records that the parameter
/// spent a long period being accepted and then dropped — so treating the clamp as symmetric,
/// or as applying to neither, both look plausible and are both wrong.
///
/// `clamp <= 0.0` means the model declared no limit, and nothing is bounded.
#[inline]
pub fn swiglu_ds4(gate: f32, up: f32, clamp: f32) -> f32 {
    let (g, u) = if clamp > 0.0 {
        (gate.min(clamp), up.clamp(-clamp, clamp))
    } else {
        (gate, up)
    };
    silu(g) * u
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// One routed expert: `down( swiglu(gate·x, up·x) )`.
///
/// `x` is in NATURAL order; the permutation into the planes' even|odd block order is applied
/// here rather than asked of the caller, because getting it wrong is silent — a dot product
/// does not care about the order of K, so a mismatched permutation between weights and
/// activation produces a plausible wrong number rather than an error.
pub fn expert_forward(
    x: &[f32],
    gate: &PackedWeight,
    up: &PackedWeight,
    down: &PackedWeight,
    n_embd: usize,
    ff: usize,
    clamp: f32,
) -> Result<Vec<f32>> {
    let xp = permute_activation(x, n_embd)?;
    let g = matvec_ref(gate, &xp, n_embd, ff);
    let u = matvec_ref(up, &xp, n_embd, ff);
    let mut act = vec![0f32; ff];
    for i in 0..ff {
        act[i] = swiglu_ds4(g[i], u[i], clamp);
    }
    // The down projection reduces over ff, so it needs its OWN permutation.
    let ap = permute_activation(&act, ff)?;
    Ok(matvec_ref(down, &ap, ff, n_embd))
}

/// Combine the routed experts a token selected.
///
/// `expert_weights_scale` is 1.5 for DS4-Flash and multiplies the whole routed contribution;
/// it is a model parameter, not a normalisation that can be folded into the router, so it is
/// taken explicitly rather than defaulted.
pub fn combine_routed(
    per_expert: &[(f32, Vec<f32>)],
    n_embd: usize,
    expert_weights_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; n_embd];
    for (w, v) in per_expert {
        debug_assert_eq!(v.len(), n_embd);
        for i in 0..n_embd {
            out[i] += w * v[i];
        }
    }
    for o in out.iter_mut() {
        *o *= expert_weights_scale;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mxfp4_repack::{repack_weight, BLK, QK};
    use engine_core::glm5_stream::dequant_mxfp4;

    /// Dense f32 weights in NATURAL element order, straight from the stored bytes via
    /// `engine_core`'s dequant. Shares no code and no layout with `repack_weight`, which is what
    /// makes it usable as the oracle for the packed path.
    fn dense_rows(raw: &[u8], reduction: usize, rows: usize) -> Vec<f32> {
        let bpr = reduction / QK * BLK;
        let mut w = vec![0f32; rows * reduction];
        for r in 0..rows {
            let d = dequant_mxfp4(&raw[r * bpr..(r + 1) * bpr]);
            w[r * reduction..(r + 1) * reduction].copy_from_slice(&d);
        }
        w
    }

    fn synth_rows(reduction: usize, rows: usize, seed: u32) -> Vec<u8> {
        let nblk = reduction / QK * rows;
        let mut v = vec![0u8; nblk * BLK];
        let mut s = seed | 1;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for b in 0..nblk {
            v[b * BLK] = 122 + (rnd() % 10) as u8;
            for j in 1..BLK {
                v[b * BLK + j] = (rnd() & 0xFF) as u8;
            }
        }
        v
    }

    /// The whole expert, packed path vs a dense path built from the same stored bytes through
    /// `engine_core`'s dequant and `glm5_ref`'s own `linear`. Two independent implementations of
    /// every step: dequantisation, layout, and the matvec itself.
    #[test]
    fn expert_forward_matches_a_dense_reference() {
        // 256/128, not 64/64: at n_embd=64 there is a single block PAIR, so the
        // even|odd permutation is the identity and the layout goes untested.
        let (n_embd, ff) = (256usize, 128usize);
        for clamp in [0.0f32, 10.0] {
            let (rg, ru, rd) = (
                synth_rows(n_embd, ff, 11),
                synth_rows(n_embd, ff, 22),
                synth_rows(ff, n_embd, 33),
            );
            let gate = repack_weight(&rg, n_embd, ff).unwrap();
            let up = repack_weight(&ru, n_embd, ff).unwrap();
            let down = repack_weight(&rd, ff, n_embd).unwrap();
            let x: Vec<f32> = (0..n_embd)
                .map(|i| (i as f32 * 0.37).sin() * 1.5)
                .collect();

            let got = expert_forward(&x, &gate, &up, &down, n_embd, ff, clamp).unwrap();

            // Dense oracle, natural order throughout.
            let (dg, du, dd) = (
                dense_rows(&rg, n_embd, ff),
                dense_rows(&ru, n_embd, ff),
                dense_rows(&rd, ff, n_embd),
            );
            let gl = engine_core::glm5_ref::linear(&x, 1, n_embd, &dg, ff);
            let ul = engine_core::glm5_ref::linear(&x, 1, n_embd, &du, ff);
            let act: Vec<f32> = (0..ff).map(|i| swiglu_ds4(gl[i], ul[i], clamp)).collect();
            let want = engine_core::glm5_ref::linear(&act, 1, ff, &dd, n_embd);

            let mut num = 0f64;
            let mut den = 0f64;
            for i in 0..n_embd {
                let e = (got[i] - want[i]) as f64;
                num += e * e;
                den += (want[i] as f64) * (want[i] as f64);
            }
            let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
            assert!(
                rel < 1e-5,
                "clamp {clamp}: packed expert vs dense reference, norm-relative {rel:e}"
            );
        }
    }

    /// The clamp's ASYMMETRY, which is the part that is easy to get wrong: the gate is bounded
    /// above only, the up projection on both sides. Mirrors `engine_core`'s own assertion.
    #[test]
    fn swiglu_clamp_is_asymmetric() {
        let c = 10.0f32;
        assert_eq!(swiglu_ds4(50.0, 100.0, c), silu(10.0) * 10.0, "both clamped");
        assert_eq!(swiglu_ds4(-50.0, 1.0, c), silu(-50.0) * 1.0, "gate NOT bounded below");
        assert_eq!(swiglu_ds4(1.0, -100.0, c), silu(1.0) * -10.0, "up bounded below");
        // Declared-no-clamp must bound nothing at all.
        assert_eq!(swiglu_ds4(50.0, 100.0, 0.0), silu(50.0) * 100.0);
    }

    /// `expert_weights_scale` multiplies the routed sum, and is 1.5 for DS4-Flash — a value
    /// that is easy to leave at 1.0 without any test noticing.
    #[test]
    fn combine_applies_the_expert_weight_scale() {
        let a = (0.25f32, vec![4.0f32, 8.0]);
        let b = (0.75f32, vec![2.0f32, 2.0]);
        let out = combine_routed(&[a, b], 2, 1.5);
        assert_eq!(out, vec![(0.25 * 4.0 + 0.75 * 2.0) * 1.5, (0.25 * 8.0 + 0.75 * 2.0) * 1.5]);
        // And with scale 1.0 it is a plain weighted sum, so the scale is visible in the test.
        let out1 = combine_routed(&[(0.5, vec![2.0]), (0.5, vec![4.0])], 1, 1.0);
        assert_eq!(out1, vec![3.0]);
    }

    /// A mismatched permutation between weights and activation is silent — the dot product does
    /// not care about K order — so this pins that `expert_forward` permutes for the caller and
    /// that feeding it pre-permuted input is WRONG.
    #[test]
    fn activation_permutation_is_owned_by_expert_forward() {
        // Must have >1 block pair or the permutation is the identity and this is vacuous.
        let (n_embd, ff) = (256usize, 128usize);
        let rg = synth_rows(n_embd, ff, 5);
        let ru = synth_rows(n_embd, ff, 6);
        let rd = synth_rows(ff, n_embd, 7);
        let gate = repack_weight(&rg, n_embd, ff).unwrap();
        let up = repack_weight(&ru, n_embd, ff).unwrap();
        let down = repack_weight(&rd, ff, n_embd).unwrap();
        let x: Vec<f32> = (0..n_embd).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let ok = expert_forward(&x, &gate, &up, &down, n_embd, ff, 10.0).unwrap();
        // Pre-permuting means the permutation is applied twice, which is a DIFFERENT order.
        let pre = permute_activation(&x, n_embd).unwrap();
        let bad = expert_forward(&pre, &gate, &up, &down, n_embd, ff, 10.0).unwrap();
        assert_ne!(ok, bad, "double-permuted input must not agree — it would hide the bug");
    }
}
