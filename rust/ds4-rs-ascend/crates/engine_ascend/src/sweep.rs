// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Ranking knob-sweep results against the roofline, so a kernel win is only adopted if the
//! system can actually deliver it.
//!
//! # Why this module exists rather than a spreadsheet
//!
//! Every kernel measurement in this project is a per-kernel time, and every DECISION is about
//! tokens per second. Those differ by more than a unit conversion once the two sides of the
//! roofline are close, which on this box they now are: decode sits at compute 138 ms against
//! transfer 123 ms, and prefill's sweep at 7190 against 5269. A kernel that gets 1.8x faster
//! moves the system by 1.12x and then stops, because the transfer term binds — and a sweep that
//! reports the 1.8x has told the reader something true and useless.
//!
//! So the harness reports both, and names the ceiling. This is rule 28 made mechanical: price a
//! lever against the `max()` of the terms, not against the term it improves.
//!
//! # What a point is
//!
//! One combination of tile knobs, with the kernels it produced timed on device. The knobs are read
//! from the environment by the emitter (`TILE_MXFP4_PAIRS_PER_CHUNK`, `TILE_E8M0_ROWS_PER_CHUNK`),
//! so a sweep needs no recompile — and, importantly, each point carries its own CORRECTNESS
//! verdict. A configuration that is fast and wrong must be unrankable, not merely ranked lower;
//! this project has already met a kernel that compiled, ran, and computed nothing.

use crate::placement::{measured, ExpertLayout, Geometry, Prefill, Throughput};

/// One sweep point: the knobs, what they measured, and whether the output was right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnobPoint {
    /// `TILE_MXFP4_PAIRS_PER_CHUNK`.
    pub pairs_per_chunk: u32,
    /// `TILE_E8M0_ROWS_PER_CHUNK`.
    pub rows_per_chunk: u32,
    /// blockDim the kernels were launched with.
    pub block_dim: u32,
    /// Packed 4-bit matvec, one projection at K=4096 N=2048, milliseconds.
    pub matvec_ms: f64,
    /// Prefill unpack, one projection at K=4096 N=512, milliseconds. Scaled to the real width by
    /// [`Ranked::of`], the same way `Prefill` scales its own reference measurement.
    pub unpack_ms_at_n512: f64,
    /// False if this point's output did not match its oracle. Fast and wrong is not a data point.
    pub correct: bool,
}

/// A point, converted into what the system would actually do.
#[derive(Debug, Clone, Copy)]
pub struct Ranked {
    pub point: KnobPoint,
    /// Decode, tokens per second, transfer overlapped.
    pub decode_tok_s: f64,
    /// Decode's compute/transfer ratio at this point.
    pub decode_balance: f64,
    /// Prefill at the reference prompt length, tokens per second.
    pub prefill_tok_s: f64,
    pub prefill_balance: f64,
    /// The most any FURTHER pure compute win could add to prefill, as a speedup.
    pub prefill_remaining_headroom: f64,
}

/// The prompt length prefill is ranked at. 8k is long enough that the fixed expert sweep dominates
/// and short enough to be a realistic request; the sweep's conclusion should not depend on it, and
/// a caller that finds otherwise has found something worth knowing.
pub const RANK_PROMPT_TOKENS: u32 = 8_000;

impl Ranked {
    /// Convert one point, under a given layout and residency.
    ///
    /// `matvec_ms` replaces the recorded `measured::MATVEC_MS` for the decode term, and
    /// `unpack_ms_at_n512` replaces `measured::UNPACK_MS_REF` for the prefill term. Everything
    /// else — the expert counts, the byte accounting, the transfer term — comes from the shipped
    /// model, so a point cannot flatter itself by also changing the denominator.
    pub fn of(point: KnobPoint, layout: ExpertLayout, resident_fraction: f64, h2d_gbps: f64) -> Self {
        let g = Geometry::ds4_flash(layout);

        // Decode: the model's compute term is projections x MATVEC_MS, so rescale it.
        let base = Throughput::estimate_with_layout(&g, layout, resident_fraction, h2d_gbps);
        let projections = g.layers as f64 * g.experts_used_per_token as f64 * 3.0;
        let expansion = base.compute_ms - projections * measured::MATVEC_MS;
        let mut d = base;
        d.compute_ms = projections * point.matvec_ms + expansion;

        // Prefill: rescale the unpack, keeping the expansion's measured FRACTION of it.
        let p = Prefill::estimate(&g, layout, RANK_PROMPT_TOKENS, resident_fraction, h2d_gbps, 1);
        let scale = if measured::UNPACK_MS_REF > 0.0 {
            point.unpack_ms_at_n512 / measured::UNPACK_MS_REF
        } else {
            1.0
        };
        let mut pp = p;
        pp.unpack_ms = p.unpack_ms * scale;
        pp.expand_ms = p.expand_ms * scale;

        Ranked {
            point,
            decode_tok_s: d.tokens_per_sec_overlapped(),
            decode_balance: if d.transfer_ms > 0.0 { d.compute_ms / d.transfer_ms } else { f64::INFINITY },
            prefill_tok_s: pp.tokens_per_sec_overlapped(),
            prefill_balance: pp.balance(),
            prefill_remaining_headroom: pp.compute_headroom(),
        }
    }

    /// Is this point's advantage one the SYSTEM gets, or only the kernel?
    ///
    /// A point can be the fastest kernel in the sweep and tie on tokens per second, which is the
    /// signal to stop optimising that kernel and go find the term that binds.
    pub fn kernel_only_win(&self, baseline: &Ranked) -> bool {
        let kernel_better = self.point.unpack_ms_at_n512 < baseline.point.unpack_ms_at_n512 * 0.99
            || self.point.matvec_ms < baseline.point.matvec_ms * 0.99;
        let system_better = self.prefill_tok_s > baseline.prefill_tok_s * 1.01
            || self.decode_tok_s > baseline.decode_tok_s * 1.01;
        kernel_better && !system_better
    }
}

/// Rank a sweep, best deliverable prefill first. Incorrect points are DROPPED, not ordered last.
///
/// Returns `(ranked, dropped)` so the caller can report what was excluded. Silently discarding a
/// point would make an incomplete sweep read as a clean one.
pub fn rank(
    points: &[KnobPoint],
    layout: ExpertLayout,
    resident_fraction: f64,
    h2d_gbps: f64,
) -> (Vec<Ranked>, Vec<KnobPoint>) {
    let (ok, dropped): (Vec<KnobPoint>, Vec<KnobPoint>) =
        points.iter().copied().partition(|p| p.correct);
    let mut ranked: Vec<Ranked> = ok
        .into_iter()
        .map(|p| Ranked::of(p, layout, resident_fraction, h2d_gbps))
        .collect();
    ranked.sort_by(|a, b| {
        b.prefill_tok_s
            .partial_cmp(&a.prefill_tok_s)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (ranked, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::Budget;

    const BOX: [(u32, u64); 8] = [
        (0, 4_800 * 1024 * 1024), (1, 4_800 * 1024 * 1024),
        (2, 4_800 * 1024 * 1024), (3, 4_800 * 1024 * 1024),
        (4, 4_800 * 1024 * 1024), (5, 4_800 * 1024 * 1024),
        (6, 4_800 * 1024 * 1024), (7, 4_800 * 1024 * 1024),
    ];

    fn rf(layout: ExpertLayout) -> f64 {
        Budget::of(&Geometry::ds4_flash(layout), &BOX, 0, 4096).resident_fraction()
    }

    fn baseline() -> KnobPoint {
        KnobPoint {
            pairs_per_chunk: 64,
            rows_per_chunk: 64,
            block_dim: 48,
            matvec_ms: measured::MATVEC_MS,
            unpack_ms_at_n512: measured::UNPACK_MS_REF,
            correct: true,
        }
    }


    /// The sweep as MEASURED on 910c, 2026-08-07. All 26 points passed their oracle.
    ///
    /// `unpack` at K=4096 N=512, byte-identity checked against the pre-expanded-f32 variant;
    /// `expand` on a 262,144-scale plane, checked against the host e8m0 reference over all 256
    /// exponent bytes. `matvec_ms` is NOT swept here and carries the shipped measurement at every
    /// point — the unpack was targeted first because it holds 96-97% of the prefill sweep and had
    /// 1.24-1.34x of headroom against decode's 1.12x.
    ///
    /// | pairs | bd=8 | 16 | 24 | 32 | 40 | 48 | 96 |
    /// |---|---|---|---|---|---|---|---|
    /// | 32 | .3617 | .1829 | .1268 | .0935 | .0774 | .0664 | .0669 |
    /// | 64 | .2783 | .1408 | .0981 | .0731 | .0610 | **.0526** | .0534 |
    ///
    /// expansion, ms per plane: rows 16 -> .0176/.0110/.0116, 32 -> .0105/.0087/.0096,
    /// 64 -> .0086/**.0073**/.0079, 128 -> .0074/.0072/.0084 at bd 24/48/96.
    fn measured_sweep() -> Vec<KnobPoint> {
        let d = baseline();
        let unpack = [
            (32u32, 8u32, 0.3617), (32, 16, 0.1829), (32, 24, 0.1268), (32, 32, 0.0935),
            (32, 40, 0.0774), (32, 48, 0.0664), (32, 96, 0.0669),
            (64, 8, 0.2783), (64, 16, 0.1408), (64, 24, 0.0981), (64, 32, 0.0731),
            (64, 40, 0.0610), (64, 48, 0.0526), (64, 96, 0.0534),
        ];
        unpack
            .iter()
            .map(|&(pairs, bd, ms)| KnobPoint {
                pairs_per_chunk: pairs,
                block_dim: bd,
                unpack_ms_at_n512: ms,
                ..d
            })
            .collect()
    }

    /// The sweep found NOTHING, and that is the result worth pinning.
    ///
    /// The shipped configuration — 64 pairs per chunk at blockDim 48 — is the best point in the
    /// accessible space, and the space is small for reasons that are structural rather than a lack
    /// of effort: at K=4096 a row holds 64 block pairs so `npc = min(pairs, 64)` caps the knob, the
    /// e8m0 variant needs npc a multiple of 32 for its i8 scale row, and blockDim is pinned by the
    /// 48 usable AIVs (96 measured WORSE at every single point, 14 of 14).
    ///
    /// If a future emitter change makes some other point better, this test fails and says so —
    /// which is the only reason to keep a measured table in the source.
    #[test]
    fn the_measured_sweep_confirms_the_shipped_configuration() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let (ranked, dropped) = rank(&measured_sweep(), layout, rf(layout), measured::H2D_PINNED_GBPS);
        assert!(dropped.is_empty(), "every measured point passed its oracle");
        let best = ranked.first().expect("non-empty sweep");
        eprintln!(
            "best of {} points: pairs={} bd={} unpack {:.4} ms => prefill {:.0} tok/s \
             (balance {:.2}, remaining headroom {:.2}x)",
            ranked.len(), best.point.pairs_per_chunk, best.point.block_dim,
            best.point.unpack_ms_at_n512, best.prefill_tok_s, best.prefill_balance,
            best.prefill_remaining_headroom
        );
        assert_eq!(best.point.pairs_per_chunk, 64, "the shipped chunk width is the best measured");
        assert_eq!(best.point.block_dim, 48, "the shipped blockDim is the best measured");

        // blockDim 96 must be worse than 48 at EVERY chunk width — the >48 rule, re-measured.
        for pairs in [32u32, 64] {
            let at = |bd: u32| {
                measured_sweep()
                    .into_iter()
                    .find(|p| p.pairs_per_chunk == pairs && p.block_dim == bd)
                    .map(|p| p.unpack_ms_at_n512)
                    .expect("point present")
            };
            assert!(
                at(96) > at(48),
                "pairs={pairs}: blockDim 96 ({:.4}) should be worse than 48 ({:.4})",
                at(96), at(48)
            );
        }

        // And the honest consequence: the best point is the shipped one, so the sweep buys 0%.
        let shipped = Ranked::of(baseline(), layout, rf(layout), measured::H2D_PINNED_GBPS);
        let gain = best.prefill_tok_s / shipped.prefill_tok_s;
        assert!(
            (gain - 1.0).abs() < 0.02,
            "the sweep should find no material gain, got {gain:.3}x — if it does, adopt it"
        );
    }

    /// A point carrying the shipped measurements must reproduce the shipped figures, or the
    /// rescaling is wrong and every other number in the sweep is off by the same factor.
    #[test]
    fn the_baseline_point_reproduces_the_shipped_estimate() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let r = Ranked::of(baseline(), layout, rf(layout), measured::H2D_PINNED_GBPS);
        let g = Geometry::ds4_flash(layout);
        let d = Throughput::estimate_with_layout(&g, layout, rf(layout), measured::H2D_PINNED_GBPS);
        let p = Prefill::estimate(
            &g, layout, RANK_PROMPT_TOKENS, rf(layout), measured::H2D_PINNED_GBPS, 1);
        assert!(
            (r.decode_tok_s - d.tokens_per_sec_overlapped()).abs() < 1e-9,
            "decode {} vs {}", r.decode_tok_s, d.tokens_per_sec_overlapped()
        );
        assert!(
            (r.prefill_tok_s - p.tokens_per_sec_overlapped()).abs() < 1e-6,
            "prefill {} vs {}", r.prefill_tok_s, p.tokens_per_sec_overlapped()
        );
    }

    /// The whole point of the module: a large KERNEL win must not be reported as a large SYSTEM
    /// win once the transfer term binds.
    #[test]
    fn a_big_kernel_win_is_capped_by_the_roofline() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let (rfx, h) = (rf(layout), measured::H2D_PINNED_GBPS);
        let base = Ranked::of(baseline(), layout, rfx, h);
        // Ten times faster on both kernels — far beyond anything a knob could give.
        let mut fast = baseline();
        fast.matvec_ms /= 10.0;
        fast.unpack_ms_at_n512 /= 10.0;
        let r = Ranked::of(fast, layout, rfx, h);

        let kernel_gain = 10.0;
        let prefill_gain = r.prefill_tok_s / base.prefill_tok_s;
        let decode_gain = r.decode_tok_s / base.decode_tok_s;
        eprintln!(
            "kernels {kernel_gain:.0}x faster => prefill {prefill_gain:.2}x, decode {decode_gain:.2}x \
             (base headroom was prefill {:.2}x)",
            base.prefill_remaining_headroom
        );
        assert!(
            prefill_gain < base.prefill_remaining_headroom * 1.001,
            "prefill gained {prefill_gain:.2}x but its ceiling was {:.2}x",
            base.prefill_remaining_headroom
        );
        assert!(
            prefill_gain < 2.0 && decode_gain < 1.5,
            "a 10x kernel win must not read as a 10x system win: got prefill {prefill_gain:.2}x, \
             decode {decode_gain:.2}x"
        );
        // And once compute is tiny, both sides must be transfer-bound.
        assert!(r.prefill_balance < 1.0 && r.decode_balance < 1.0);
    }

    /// `kernel_only_win` must fire exactly when the kernel improved and the system did not.
    #[test]
    fn a_kernel_only_win_is_identified_as_such() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let (rfx, h) = (rf(layout), measured::H2D_PINNED_GBPS);
        let base = Ranked::of(baseline(), layout, rfx, h);
        // Enormously faster kernels: past the ceiling, so tokens/s stops responding.
        let mut huge = baseline();
        huge.matvec_ms /= 1000.0;
        huge.unpack_ms_at_n512 /= 1000.0;
        let a = Ranked::of(huge, layout, rfx, h);
        let mut faster_still = huge;
        faster_still.matvec_ms /= 10.0;
        faster_still.unpack_ms_at_n512 /= 10.0;
        let b = Ranked::of(faster_still, layout, rfx, h);
        assert!(
            b.kernel_only_win(&a),
            "10x faster again past the ceiling must be reported as kernel-only"
        );
        // A modest win below the ceiling IS a system win.
        let mut modest = baseline();
        modest.unpack_ms_at_n512 *= 0.9;
        let m = Ranked::of(modest, layout, rfx, h);
        assert!(!m.kernel_only_win(&base), "a 10% win below the ceiling should reach the system");
    }

    /// Incorrect points must be excluded and REPORTED, never silently ordered last.
    #[test]
    fn incorrect_points_are_dropped_and_counted() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let mut wrong = baseline();
        wrong.pairs_per_chunk = 16;
        wrong.unpack_ms_at_n512 = 0.001; // would win on speed alone
        wrong.correct = false;
        let pts = [baseline(), wrong];
        let (ranked, dropped) = rank(&pts, layout, rf(layout), measured::H2D_PINNED_GBPS);
        assert_eq!(ranked.len(), 1, "the incorrect point must not be ranked");
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].pairs_per_chunk, 16);
    }

    /// Ranking must be by the SYSTEM figure, not by kernel time — otherwise the fastest kernel
    /// wins even when it buys nothing.
    #[test]
    fn ranking_orders_by_the_system_figure() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let mut slower_kernel_better_shape = baseline();
        slower_kernel_better_shape.unpack_ms_at_n512 *= 0.8;
        let pts = [baseline(), slower_kernel_better_shape];
        let (ranked, _) = rank(&pts, layout, rf(layout), measured::H2D_PINNED_GBPS);
        assert!(
            ranked[0].point.unpack_ms_at_n512 < ranked[1].point.unpack_ms_at_n512,
            "the faster unpack should rank first while below the ceiling"
        );
        assert!(ranked[0].prefill_tok_s >= ranked[1].prefill_tok_s);
    }
}
