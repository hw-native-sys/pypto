// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Sequence-sharded attention: each chip holds a slice of the POSITIONS, and the partials merge.
//!
//! This is the prerequisite for any `ep > 1` actually running rather than merely being selectable.
//! DS4-Flash has `n_kv_heads == 1` with K aliasing V, so the MLA latent cannot be split by KV head;
//! under expert parallelism the alternative to sharding the SEQUENCE is replicating KV on every
//! cooperating chip, which multiplies it by `tp*ep` and eats the expert pool the design exists to
//! fill. Measured: replication takes the customer's box from 19.6% resident to 4.2%.
//!
//! # What each shard computes, and what it must NOT
//!
//! A shard runs the same flash-style pass our attention kernel already does — a running max and sum
//! over its own rows — but stops before the final division and reports the three quantities that let
//! the pieces be merged exactly:
//!
//!   * `m` the max score it saw,
//!   * `d` the sum of `exp(score - m)` over its rows,
//!   * `o[head_dim]` the UN-NORMALISED weighted sum of its value rows.
//!
//! The merge is then exact, not approximate:
//!
//! ```text
//!   M = max_i m_i
//!   D = Σ_i d_i · exp(m_i − M)
//!   O = Σ_i o_i · exp(m_i − M)      out = O / D
//! ```
//!
//! # The trap: the sink must be counted ONCE
//!
//! Every DS4-Flash layer carries `attn_sinks[n_head]`, a per-head logit that joins the softmax
//! DENOMINATOR with no value vector. A shard that folded the sink into its own `d` would have it
//! counted once per shard — `n` times over — and the error would be a smooth attenuation of every
//! head, which looks like a slightly mistuned model rather than a bug. So shards report sink-free
//! partials and [`combine_with_sink`] adds it exactly once, in the merged frame.
//!
//! The sink is also UNSCALED: key scores are `dot·scale`, and the sink is the raw logit. Scaling it
//! too is the obvious mistake and is nearly invisible.

use anyhow::{bail, Result};

/// One shard's contribution for one head, before normalisation.
#[derive(Debug, Clone, PartialEq)]
pub struct Partial {
    /// Max score over this shard's rows. `NEG_INFINITY` for an empty shard.
    pub m: f32,
    /// `Σ exp(score − m)` over this shard's rows. Zero for an empty shard.
    pub d: f32,
    /// Un-normalised `Σ exp(score − m) · V[row]`, length `head_dim`.
    pub o: Vec<f32>,
}

impl Partial {
    /// The identity element: a shard holding no positions.
    ///
    /// Not a special case in the merge — `exp(-inf - M)` is 0, so an empty shard contributes nothing
    /// to either sum. Having it be the identity rather than a branch is what makes an uneven split
    /// (43 layers over 8 chips, or a context that does not divide) safe.
    pub fn empty(head_dim: usize) -> Self {
        Self { m: f32::NEG_INFINITY, d: 0.0, o: vec![0.0; head_dim] }
    }
}

/// Compute one shard's partial over the rows it owns.
///
/// `kv` is this shard's rows only. Deliberately takes no sink: see the module note — a sink folded in
/// here would be counted once per shard.
pub fn shard_partial(
    q: &[f32],
    kv: &[f32],
    n_rows: usize,
    head_dim: usize,
    scale: f32,
) -> Result<Partial> {
    if q.len() < head_dim {
        bail!("q has {} elements, need at least head_dim {head_dim}", q.len());
    }
    if kv.len() < n_rows * head_dim {
        bail!("kv has {} elements, need {n_rows}*{head_dim}", kv.len());
    }
    if n_rows == 0 {
        return Ok(Partial::empty(head_dim));
    }
    let mut scores = Vec::with_capacity(n_rows);
    for r in 0..n_rows {
        let row = &kv[r * head_dim..(r + 1) * head_dim];
        let dot: f32 = q[..head_dim].iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        scores.push(dot * scale);
    }
    let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut d = 0.0f32;
    let mut o = vec![0.0f32; head_dim];
    for (r, s) in scores.iter().enumerate() {
        let w = (s - m).exp();
        d += w;
        let row = &kv[r * head_dim..(r + 1) * head_dim];
        for k in 0..head_dim {
            o[k] += w * row[k];
        }
    }
    Ok(Partial { m, d, o })
}

/// Merge shard partials, with the head's sink counted exactly once.
///
/// `sink = f32::NEG_INFINITY` disables it exactly, so the no-sink case is this same code rather than
/// a second path.
pub fn combine_with_sink(partials: &[Partial], head_dim: usize, sink: f32) -> Result<Vec<f32>> {
    if partials.is_empty() {
        bail!("no partials to combine; a head needs at least one shard, even an empty one");
    }
    for (i, p) in partials.iter().enumerate() {
        if p.o.len() != head_dim {
            bail!("shard {i} reported {} values, expected head_dim {head_dim}", p.o.len());
        }
    }
    // The global frame includes the sink, because the sink participates in the max — a sink far above
    // every score would otherwise overflow the rescaling. That is what "stable-softmax sink" means.
    let big = partials.iter().map(|p| p.m).fold(f32::NEG_INFINITY, f32::max);
    let global_max = big.max(sink);
    if !global_max.is_finite() {
        // Every shard empty and no finite sink: no positions attended, so the output is zero rather
        // than a NaN from 0/0.
        return Ok(vec![0.0; head_dim]);
    }

    let mut d_total = 0.0f32;
    let mut o_total = vec![0.0f32; head_dim];
    for p in partials {
        // exp(-inf - finite) is 0, so an empty shard adds nothing and needs no branch.
        let r = (p.m - global_max).exp();
        if r == 0.0 {
            continue;
        }
        d_total += p.d * r;
        for k in 0..head_dim {
            o_total[k] += p.o[k] * r;
        }
    }
    // ONCE, here, in the merged frame — not per shard.
    d_total += (sink - global_max).exp();

    if d_total > 0.0 {
        let inv = 1.0 / d_total;
        for v in o_total.iter_mut() {
            *v *= inv;
        }
    }
    Ok(o_total)
}

/// Split `n_rows` positions across `shards` as evenly as possible, earlier shards taking the
/// remainder — the same convention `parallel::assign` uses for layers, so a reader does not have to
/// hold two rules.
pub fn shard_ranges(n_rows: usize, shards: usize) -> Vec<(usize, usize)> {
    if shards == 0 {
        return Vec::new();
    }
    let base = n_rows / shards;
    let rem = n_rows % shards;
    let mut out = Vec::with_capacity(shards);
    let mut at = 0usize;
    for s in 0..shards {
        let take = base + if s < rem { 1 } else { 0 };
        out.push((at, at + take));
        at += take;
    }
    out
}

/// End-to-end: shard `kv` across `shards`, compute partials, merge. The reference path a device
/// implementation must reproduce.
pub fn sharded_attention_head(
    q: &[f32],
    kv: &[f32],
    n_rows: usize,
    head_dim: usize,
    scale: f32,
    sink: f32,
    shards: usize,
) -> Result<Vec<f32>> {
    let ranges = shard_ranges(n_rows, shards);
    if ranges.is_empty() {
        bail!("shards must be at least 1");
    }
    let mut parts = Vec::with_capacity(ranges.len());
    for (a, b) in ranges {
        let rows = b - a;
        let slice = &kv[a * head_dim..b * head_dim];
        parts.push(shard_partial(q, slice, rows, head_dim, scale)?);
    }
    combine_with_sink(&parts, head_dim, sink)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::attention_head_with_sink;

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

    /// The whole point: sharding must be EXACT, not approximate. Oracle is
    /// `attention_head_with_sink`, whose semantics are already device-verified — including that the
    /// sink is unscaled and joins only the max and denominator.
    #[test]
    fn sharding_reproduces_unsharded_attention_at_every_shard_count() {
        let (rows, d) = (128usize, 512usize); // DS4-Flash's MLA latent
        let q = synth(d, 11);
        let kv = synth(rows * d, 12);
        let scale = 1.0 / (d as f32).sqrt();
        for sink in [f32::NEG_INFINITY, -3.0, 0.0, 2.5] {
            let want = attention_head_with_sink(&q, &kv, rows, d, scale, sink);
            for shards in [1usize, 2, 3, 4, 5, 8, 16, 128] {
                let got = sharded_attention_head(&q, &kv, rows, d, scale, sink, shards).unwrap();
                // NORM-relative, not max-relative with a tiny floor. Sharding changes the SUMMATION
                // ORDER — 128 terms in one pass versus 64+64 rescaled and merged — so f32 rounding
                // differs, and a max-rel metric with a 1e-6 denominator floor reports 7e-5 on the
                // near-zero components while the vectors agree to 1e-7 overall. That is the
                // cancellation trap this project has already been caught by; norm-relative is the
                // measure that prices the difference honestly.
                let num: f32 = (0..d).map(|i| (got[i] - want[i]).powi(2)).sum::<f32>().sqrt();
                let den: f32 = want.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-20);
                let rel = num / den;
                assert!(
                    rel < 1e-6,
                    "sink {sink} shards {shards}: norm-rel {rel:e} — sharding must be exact up to \
                     summation order"
                );
                // And a max-rel with a sane floor, to catch a single component going wrong in a way
                // the norm would average away.
                let worst = (0..d)
                    .map(|i| (got[i] - want[i]).abs() / want[i].abs().max(1e-3))
                    .fold(0.0f32, f32::max);
                assert!(worst < 1e-4, "sink {sink} shards {shards}: max-rel {worst:e}");
            }
        }
    }

    /// The trap the module note names. A shard that folded the sink into its own `d` would count it
    /// once per shard, and the result would be a smooth attenuation that looks like mistuning.
    /// This proves the merge is not doing that: adding the sink n times is DETECTABLY different.
    #[test]
    fn the_sink_is_counted_once_not_once_per_shard() {
        let (rows, d) = (64usize, 8usize);
        let q = synth(d, 21);
        let kv = synth(rows * d, 22);
        let (scale, sink) = (0.4f32, 0.0f32);
        let shards = 8usize;
        let correct = sharded_attention_head(&q, &kv, rows, d, scale, sink, shards).unwrap();

        // Reproduce the bug deliberately: fold the sink into every shard's denominator.
        let mut parts = Vec::new();
        for (a, b) in shard_ranges(rows, shards) {
            let mut p =
                shard_partial(&q, &kv[a * d..b * d], b - a, d, scale).unwrap();
            let r = (sink - p.m).exp();
            p.d += r; // <- the mistake
            parts.push(p);
        }
        let buggy = combine_with_sink(&parts, d, f32::NEG_INFINITY).unwrap();
        let diff = (0..d)
            .map(|i| (correct[i] - buggy[i]).abs() / correct[i].abs().max(1e-6))
            .fold(0.0f32, f32::max);
        assert!(
            diff > 1e-3,
            "counting the sink per shard differs by only {diff:e}; this test cannot tell the bug \
             from the fix and would pass on broken code"
        );
        // And the correct version must match the unsharded oracle, which the buggy one does not.
        let want = attention_head_with_sink(&q, &kv, rows, d, scale, sink);
        for i in 0..d {
            assert!((correct[i] - want[i]).abs() < 2e-5 * want[i].abs().max(1.0));
        }
    }

    /// Empty shards are the IDENTITY, not a special case — which is what makes an uneven split safe.
    #[test]
    fn empty_shards_contribute_nothing() {
        let (rows, d) = (7usize, 4usize);
        let q = synth(d, 31);
        let kv = synth(rows * d, 32);
        let scale = 0.5f32;
        // More shards than rows: the tail shards are empty.
        let got = sharded_attention_head(&q, &kv, rows, d, scale, -1.0, 16).unwrap();
        let want = attention_head_with_sink(&q, &kv, rows, d, scale, -1.0);
        for i in 0..d {
            assert!((got[i] - want[i]).abs() < 2e-5 * want[i].abs().max(1.0));
        }
        // Padding a real split with explicit empties must change nothing.
        let mut parts = vec![shard_partial(&q, &kv, rows, d, scale).unwrap()];
        parts.push(Partial::empty(d));
        parts.push(Partial::empty(d));
        let padded = combine_with_sink(&parts, d, -1.0).unwrap();
        for i in 0..d {
            assert!((padded[i] - want[i]).abs() < 2e-5 * want[i].abs().max(1.0));
        }
        // All shards empty and no sink: zero, not NaN.
        let none = combine_with_sink(&[Partial::empty(d)], d, f32::NEG_INFINITY).unwrap();
        assert!(none.iter().all(|v| *v == 0.0), "0/0 must not become NaN: {none:?}");
    }

    /// Numerical stability: one shard whose scores dwarf the others must not overflow the rescaling,
    /// and a sink above every score must not either — the sink joins the global max for that reason.
    #[test]
    fn a_dominant_shard_or_sink_stays_finite() {
        let (rows, d) = (32usize, 4usize);
        let q = synth(d, 41);
        let mut kv = synth(rows * d, 42);
        // Make the last shard's rows enormous.
        for v in kv[24 * d..].iter_mut() {
            *v *= 300.0;
        }
        let got = sharded_attention_head(&q, &kv, rows, d, 1.0, f32::NEG_INFINITY, 4).unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "dominant shard overflowed: {got:?}");
        let got = sharded_attention_head(&q, &kv, rows, d, 1.0, 5000.0, 4).unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "dominant sink overflowed: {got:?}");
        // A sink that dominates suppresses the head almost entirely.
        assert!(got.iter().all(|v| v.abs() < 1e-6), "a sink 5000 above the scores should suppress");
    }

    /// Ranges must partition the positions exactly — a gap drops KV rows and an overlap counts them
    /// twice, and both produce a plausible vector.
    #[test]
    fn shard_ranges_partition_exactly() {
        for (n, s) in [(128usize, 8usize), (43, 8), (7, 16), (0, 4), (1, 1), (1000, 3)] {
            let r = shard_ranges(n, s);
            assert_eq!(r.len(), s);
            assert_eq!(r[0].0, 0);
            assert_eq!(r.last().unwrap().1, n, "n={n} s={s} must cover every row");
            for w in r.windows(2) {
                assert_eq!(w[0].1, w[1].0, "n={n} s={s}: gap or overlap");
            }
            let total: usize = r.iter().map(|(a, b)| b - a).sum();
            assert_eq!(total, n);
            // Even to within one row, remainder to the earlier shards.
            let sizes: Vec<usize> = r.iter().map(|(a, b)| b - a).collect();
            let (lo, hi) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
            assert!(hi - lo <= 1, "n={n} s={s} uneven: {sizes:?}");
        }
        assert!(shard_ranges(10, 0).is_empty());
    }

    /// A mis-sized partial must be refused, not reshaped: it means a shard ran at the wrong head_dim.
    #[test]
    fn a_mis_sized_partial_is_refused() {
        let p = Partial { m: 0.0, d: 1.0, o: vec![1.0; 3] };
        let e = combine_with_sink(&[p], 4, 0.0).unwrap_err().to_string();
        assert!(e.contains("expected head_dim 4"), "got: {e}");
        let e = combine_with_sink(&[], 4, 0.0).unwrap_err().to_string();
        assert!(e.contains("at least one shard"), "got: {e}");
    }
}
