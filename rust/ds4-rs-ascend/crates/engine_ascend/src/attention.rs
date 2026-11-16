// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Attention with SINKS, for the CPU side of the acceptance test.
//!
//! This exists because of a defect the plan recorded and could not test: every DS4-Flash layer
//! carries `attn_sinks[n_head]`, a per-head logit that joins the softmax denominator with no
//! value vector. The blocked PTO lowering has no notion of it, so normalisation is wrong for
//! this model **despite the device matching the CPU oracle at 1e-5** — because the oracle
//! omitted sinks too. The note's own conclusion was that the ORACLE MUST LEARN SINKS FIRST, or
//! the test keeps passing for the wrong reason. This module is that step.
//!
//! Semantics are taken from `engine_core`'s CPU dispatcher rather than invented, and three
//! details there are load-bearing:
//!
//!   1. The sink is pushed **unscaled**. Key scores are `dot * scale`; the sink is the raw
//!      `attn_sinks[h]`. Scaling it too is the obvious mistake and would be nearly invisible.
//!   2. It participates in the running max AND the denominator, but contributes no value.
//!   3. Values accumulate UN-normalised and are divided once at the end. Pre-dividing each
//!      weight is algebraically identical and f32-ULP-different, and the reference follows
//!      antirez's order so the two can be compared bit-for-bit later.
//!
//! DS4-Flash has `n_kv_heads == 1` and K aliases V, so one latent row slice serves as both key
//! and value — hence a single `kv` argument rather than separate `k` and `v`.

/// One head, over `n_rows` KV latent rows.
///
/// `sink = f32::NEG_INFINITY` disables the sink exactly (its `exp` term is zero), which is what
/// makes the no-sink path a special case of this one rather than separate code.
pub fn attention_head_with_sink(
    q: &[f32],
    kv: &[f32],
    n_rows: usize,
    head_dim: usize,
    scale: f32,
    sink: f32,
) -> Vec<f32> {
    assert!(q.len() >= head_dim, "q shorter than head_dim");
    assert!(kv.len() >= n_rows * head_dim, "kv shorter than n_rows*head_dim");
    // Sink first, unscaled — the order the dispatcher uses, kept so the two are comparable.
    let mut scores = Vec::with_capacity(n_rows + 1);
    scores.push(sink);
    for r in 0..n_rows {
        let row = &kv[r * head_dim..(r + 1) * head_dim];
        let dot: f32 = q[..head_dim].iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        scores.push(dot * scale);
    }
    let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        // `exp(-inf - mx)` is 0, so a disabled sink contributes nothing and needs no branch.
        *s = (*s - mx).exp();
        sum += *s;
    }
    // Un-normalised accumulate, one division at the end.
    let mut out = vec![0f32; head_dim];
    for r in 0..n_rows {
        let w = scores[r + 1];
        let row = &kv[r * head_dim..(r + 1) * head_dim];
        for d in 0..head_dim {
            out[d] += w * row[d];
        }
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
    out
}

/// All heads. `sinks` must have one entry per head — a length mismatch is a wiring error, not
/// something to pad or ignore.
pub fn attention_with_sinks(
    q: &[f32],
    kv: &[f32],
    n_head: usize,
    n_rows: usize,
    head_dim: usize,
    scale: f32,
    sinks: &[f32],
) -> Vec<f32> {
    assert_eq!(sinks.len(), n_head, "one sink per head");
    let mut out = vec![0f32; n_head * head_dim];
    for h in 0..n_head {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let oh = attention_head_with_sink(qh, kv, n_rows, head_dim, scale, sinks[h]);
        out[h * head_dim..(h + 1) * head_dim].copy_from_slice(&oh);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                ((s & 0xFFFF) as f32 / 65536.0 - 0.5) * 2.0
            })
            .collect()
    }

    /// With the sink disabled this must equal plain scaled softmax attention — checked against
    /// `engine_core::forward::softmax_scaled`, which is independent code.
    #[test]
    fn disabled_sink_reduces_to_plain_softmax_attention() {
        let (rows, d) = (17usize, 8usize);
        let q = synth(d, 1);
        let kv = synth(rows * d, 2);
        let scale = 0.35f32;
        let got = attention_head_with_sink(&q, &kv, rows, d, scale, f32::NEG_INFINITY);

        let dots: Vec<f32> = (0..rows)
            .map(|r| q.iter().zip(kv[r * d..(r + 1) * d].iter()).map(|(a, b)| a * b).sum())
            .collect();
        let w = engine_core::forward::softmax_scaled(&dots, scale);
        let mut want = vec![0f32; d];
        for r in 0..rows {
            for i in 0..d {
                want[i] += w[r] * kv[r * d + i];
            }
        }
        for i in 0..d {
            assert!(
                (got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0),
                "elem {i}: {} vs {}",
                got[i],
                want[i]
            );
        }
    }

    /// The sink's ONLY effect is to attenuate the head by `D / (D + exp(sink - mx))`.
    ///
    /// This is the strong check: it predicts the exact scalar the sink introduces, so it fails
    /// if the sink is scaled, if it is added to the numerator, or if it is left out of the max.
    /// Valid only while the sink does not become the maximum, which the chosen value ensures.
    #[test]
    fn sink_attenuates_by_exactly_the_predicted_scalar() {
        let (rows, d) = (23usize, 6usize);
        let q = synth(d, 7);
        let kv = synth(rows * d, 8);
        let scale = 0.5f32;
        let plain = attention_head_with_sink(&q, &kv, rows, d, scale, f32::NEG_INFINITY);

        let dots: Vec<f32> = (0..rows)
            .map(|r| q.iter().zip(kv[r * d..(r + 1) * d].iter()).map(|(a, b)| a * b).sum())
            .collect();
        let scaled: Vec<f32> = dots.iter().map(|v| v * scale).collect();
        let mx = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let dsum: f32 = scaled.iter().map(|s| (s - mx).exp()).sum();

        for sink in [-4.0f32, -1.0, 0.0] {
            assert!(sink < mx, "test needs the sink below the max to keep mx fixed");
            let with = attention_head_with_sink(&q, &kv, rows, d, scale, sink);
            let factor = dsum / (dsum + (sink - mx).exp());
            for i in 0..d {
                let want = plain[i] * factor;
                assert!(
                    (with[i] - want).abs() <= 2e-5 * want.abs().max(1.0),
                    "sink {sink} elem {i}: {} vs predicted {want}",
                    with[i]
                );
            }
        }
    }

    /// The guard the plan asked for: a finite sink MUST change the answer. Without this an
    /// oracle that silently ignores sinks agrees with a kernel that ignores them too, and the
    /// test passes for the wrong reason — which is exactly what happened at 1e-5 for a while.
    #[test]
    fn ignoring_the_sink_is_detectable() {
        let (rows, d) = (11usize, 4usize);
        let q = synth(d, 3);
        let kv = synth(rows * d, 4);
        let plain = attention_head_with_sink(&q, &kv, rows, d, 0.4, f32::NEG_INFINITY);
        let with = attention_head_with_sink(&q, &kv, rows, d, 0.4, 0.0);
        let rel: f32 = (0..d)
            .map(|i| (with[i] - plain[i]).abs() / plain[i].abs().max(1e-6))
            .fold(0.0, f32::max);
        assert!(
            rel > 1e-3,
            "a sink of 0.0 changed the output by only {rel:e} — an oracle this insensitive \
             cannot distinguish a kernel that implements sinks from one that does not"
        );
    }

    /// The sink is UNSCALED. Scaling it too is the obvious mistake, so it gets its own test:
    /// with two different `scale` values the sink's contribution must be identical, which is
    /// only true if `scale` never touches it.
    #[test]
    fn sink_is_not_multiplied_by_scale() {
        let (rows, d) = (9usize, 4usize);
        let q = synth(d, 5);
        let kv = synth(rows * d, 6);
        // A very negative sink under one scale and another must attenuate identically only if
        // the sink itself is untouched; compare the implied denominators directly.
        let implied = |scale: f32| -> f32 {
            let with = attention_head_with_sink(&q, &kv, rows, d, scale, -2.0);
            let plain = attention_head_with_sink(&q, &kv, rows, d, scale, f32::NEG_INFINITY);
            // factor = D/(D+e); recover e/D from the ratio.
            let f = with[0] / plain[0];
            (1.0 - f) / f
        };
        let dots: Vec<f32> = (0..rows)
            .map(|r| q.iter().zip(kv[r * d..(r + 1) * d].iter()).map(|(a, b)| a * b).sum())
            .collect();
        for scale in [0.25f32, 1.0] {
            let scaled: Vec<f32> = dots.iter().map(|v| v * scale).collect();
            let mx = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let dsum: f32 = scaled.iter().map(|s| (s - mx).exp()).sum();
            let want = (-2.0f32 - mx).exp() / dsum;
            let got = implied(scale);
            assert!(
                (got - want).abs() <= 1e-3 * want.max(1e-6),
                "scale {scale}: implied sink term {got} vs {want} — sink must be unscaled"
            );
        }
    }

    /// Numerical stability: the max must include the sink, or a sink far above every score
    /// overflows. This is what "stable-softmax sink" means.
    #[test]
    fn a_dominant_sink_stays_finite_and_suppresses_the_head() {
        let (rows, d) = (5usize, 4usize);
        let q = synth(d, 9);
        let kv = synth(rows * d, 10);
        let out = attention_head_with_sink(&q, &kv, rows, d, 1.0, 200.0);
        assert!(out.iter().all(|v| v.is_finite()), "dominant sink overflowed: {out:?}");
        // With the sink dominating the denominator the head is almost entirely suppressed.
        assert!(
            out.iter().all(|v| v.abs() < 1e-30),
            "a sink 200 above the scores should suppress the head, got {out:?}"
        );
    }

    /// One sink per head, and heads must not bleed into each other.
    #[test]
    fn per_head_sinks_are_applied_independently() {
        let (heads, rows, d) = (3usize, 7usize, 4usize);
        let q = synth(heads * d, 11);
        let kv = synth(rows * d, 12);
        let sinks = vec![f32::NEG_INFINITY, 0.0, 200.0];
        let all = attention_with_sinks(&q, &kv, heads, rows, d, 0.5, &sinks);
        for h in 0..heads {
            let one = attention_head_with_sink(
                &q[h * d..(h + 1) * d],
                &kv,
                rows,
                d,
                0.5,
                sinks[h],
            );
            assert_eq!(&all[h * d..(h + 1) * d], &one[..], "head {h} differs when batched");
        }
        // Head 2's sink dominates, head 0's is disabled: they must not look alike.
        assert!(all[2 * d].abs() < 1e-30 && all[0].abs() > 1e-30);
    }
}
