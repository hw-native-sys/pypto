// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Speculative decoding (MTP / DSpark) priced BEFORE it is built.
//!
//! The directive is to port the DSpark/MTP drafter to this backend, and the folklore figure for
//! the lever is "~1.6-2x". This module prices that claim against the measured per-layer numbers
//! instead of assuming it, because the last audit of this program found the recurring failure mode
//! was pricing a lever against the term being improved rather than against what binds
//! (memory: tile-rs-optimization-rules, rules 28-29).
//!
//! # The round being priced
//!
//! One speculative round is: the DRAFTER proposes `k` tokens, the target VERIFIES all `k` in one
//! batched forward, and greedy longest-prefix acceptance emits `accept_len + 1` tokens (the +1 is
//! the target's own next token at the first disagreement — free, because the verify computed it).
//! Output is byte-identical to plain greedy by construction; the drafter only sets the RATE.
//!
//! # Where a batched verify can and cannot amortise
//!
//! A `k`-row verify launches the same kernels with `k` activation rows. What is measured on this
//! silicon (commit 6408345b2) is that a `tload` costs far more than a vector op — hoisting the
//! loop-invariant activation loads out of the matvec row loop was worth 2.2-2.7x. So the useful
//! split of any kernel's cost is LOADS (shareable across rows when the rows read the SAME bytes)
//! versus OPS (scale with `k`, always):
//!
//! * **Attention**: all `k` query rows read the SAME KV cache. Loads share; ops scale.
//! * **Dense projections** (q/kv down-up, output): same weights for every row. Loads share.
//! * **Shared expert**: same weights for every row. Loads share.
//! * **Routed experts**: each row routes to its OWN top-6 of 256. Cross-token expert overlap is
//!   the measured ~2.3% uniform floor (memory: ds4-expert-locality-closed), so the routed FFN
//!   does NOT share loads. It scales with `k` no matter how good the kernel is.
//!
//! That last line is the structural fact this module exists to state: **the routed FFN is
//! spec-decode-immune**, and on this model it is ~24% of the resident layer (1.914 of 7.878 ms,
//! six sevenths of it routed). The attainable speedup is therefore bounded by how much of the
//! OTHER 76% actually shares, which is `shared_fraction` below — a number that must come from a
//! device measurement of the k-row kernels, not from this file.
//!
//! # What the defaults say, and what would change them
//!
//! With `shared_fraction = 0.55` (the hoist measurement's lower bound for the load share of a
//! matvec) and the Metal-MEASURED emits per round, the tests below land at ~1.17x for MTP-1 at
//! k=2 and ~parity at k=4 — NOT the folklore 1.6-2x. The folklore number needs `shared_fraction`
//! near 0.9, which is plausible ONLY where attention is KV-traffic-bound: the measured 5.964 ms
//! attention sublayer is a short-KV figure, and KV grows linearly with context while the per-row
//! op work does not. So the honest statement is: **spec decode on this backend pays roughly in
//! proportion to context length, and the go/no-go is the measured k-row cost factor of the
//! batched attention kernel at a real cache length** — a one-kernel device experiment, far
//! cheaper than building the loop first. `break_even_shared_fraction` names the target that
//! experiment has to beat.
//!
//! Streaming changes nothing in spec decode's favour: the k rows' routed experts are ~disjoint,
//! so the streamed bytes per round scale with `k` exactly like the routed compute does
//! ([`crate::placement::Geometry::experts_touched_per_layer`] gives the expected union).

use crate::placement::{measured, Geometry};

/// Fraction of a load-shareable kernel's k=1 cost that is loads (shared across rows), as opposed
/// to ops (which scale with `k`).
///
/// A k-row launch of such a kernel is modelled as `shared + (1 - shared) * k` times the k=1 cost.
/// The default is derived from the one relevant measurement this program has — hoisting the
/// loop-invariant `x` loads gave 2.2x on the packed matvec, putting loads at >= 55% of its
/// runtime — and it is a DEFAULT, not a measurement of the batched kernels themselves. Both
/// bounding cases are provided for reporting ranges.
#[derive(Debug, Clone, Copy)]
pub struct RowScaling {
    pub shared_fraction: f64,
}

impl RowScaling {
    /// MEASURED on the 910c (2026-08-07): `__tile_attention_sink_batched_kq_f32` at kq=2
    /// versus two launches of the proven k=1 kernel, NH=64, S=1024, D=512, SB=8, blockDim 48,
    /// every head and query oracle-gated (bad 0/65536 both sides, and both kernels carry the
    /// IDENTICAL 1.72e-3 f32-accumulation divergence from the double oracle, so they compute
    /// the same thing). t_k1 39.44 ms, t_kq2 66.88 ms -> cost_factor(2) = 1.696 ->
    /// shared_fraction = 0.304.
    ///
    /// HALF the hoist-derived 0.55 default, and far under the 0.78 break-even: the kernel is
    /// OP-bound (consistent with this program's standing finding), and the per-query score
    /// chain is ~5 tile ops against ~1.5 shared loads per block, so sharing the loads buys
    /// only ~30%. Longer context does NOT improve it — the per-query ops scale with S exactly
    /// as the shared loads do, and the SAME A/B at S=128 (the SWA-window operating point,
    /// where t_k1 = 4.95 ms matches the measured 5.964 ms sublayer) gave 0.306. The number is
    /// S-independent, as the op-bound structure predicts.
    pub fn measured_kq2() -> Self {
        RowScaling { shared_fraction: 0.306 }
    }
    /// The hoist-derived default described above — kept for the record of what was ASSUMED
    /// before [`RowScaling::measured_kq2_s1024`] halved it.
    pub fn load_shared() -> Self {
        RowScaling { shared_fraction: 0.55 }
    }
    /// Nothing shares; a k-row verify costs exactly k single decodes. Spec decode cannot win
    /// here (the +1 bonus token is its only gain), so this is the floor of any report.
    pub fn worst_case() -> Self {
        RowScaling { shared_fraction: 0.0 }
    }
    /// Cost of a k-row launch as a multiple of the k=1 launch.
    pub fn cost_factor(&self, k: u32) -> f64 {
        let s = self.shared_fraction.clamp(0.0, 1.0);
        s + (1.0 - s) * k as f64
    }
}

/// Which support model drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drafter {
    /// Legacy 1-layer MTP (`DeepSeek-V4-Flash-MTP`, 3.5 GB). Cheap, depth-limited: Metal
    /// measured 87.5% per-position accept at k=2 collapsing to 43.8% at k=4.
    Mtp1,
    /// DSpark, 3 full DeepSeek-V4 stages (~19B params). Deeper accept on paper (60-80%); the
    /// Metal port measured only ~0.6-0.7 accepted drafts per round because of the documented
    /// single-main_kv simplification, so treat paper accept as unproven here.
    Dspark3,
}

impl Drafter {
    pub fn stages(&self) -> u32 {
        match self {
            Drafter::Mtp1 => 1,
            Drafter::Dspark3 => 3,
        }
    }
    /// Draft cost per round: one full decode layer per stage, plus the head.
    ///
    /// The head figure is SCALED, not measured: lm_head is a [129280 x 4096] projection and the
    /// measured matvec reference is [2048 x 4096] at 0.171 ms, so ~63 reference matvecs. The
    /// real head on this backend has never been timed; this is the placeholder the first device
    /// measurement should replace.
    pub fn draft_ms(&self) -> f64 {
        const LM_HEAD_SCALED_MS: f64 = (129_280.0 / 2_048.0) * measured::MATVEC_MS;
        self.stages() as f64 * measured::DECODE_LAYER_MS + LM_HEAD_SCALED_MS
    }
}

/// One speculative round, priced. All costs are RESIDENT-compute figures at the tuned launch
/// configuration — the same basis as the 338.8 ms/token baseline, so the ratio is meaningful.
/// (Both sides equally exclude lm_head-per-emitted-token, the host router matvec and KV growth.)
#[derive(Debug, Clone, Copy)]
pub struct SpecRound {
    pub k: u32,
    pub drafter: Drafter,
    pub draft_ms: f64,
    pub verify_ms: f64,
    /// Expected tokens emitted per round: E[accept_len] + 1.
    pub emit: f64,
    /// Expected distinct routed experts a k-row verify touches per layer (for streaming plans).
    pub experts_touched_per_layer: f64,
}

impl SpecRound {
    pub fn round_ms(&self) -> f64 {
        self.draft_ms + self.verify_ms
    }
    pub fn ms_per_token(&self) -> f64 {
        self.round_ms() / self.emit
    }
    /// Speedup over plain greedy decode of the SAME resident-compute model.
    pub fn speedup(&self) -> f64 {
        baseline_ms_per_token() / self.ms_per_token()
    }
}

/// Plain greedy decode on the same basis: 43 layers at the measured tuned layer.
pub fn baseline_ms_per_token() -> f64 {
    let g = Geometry::ds4_flash(crate::placement::ExpertLayout::PackedNibbleNativeE8m0);
    g.layers as f64 * measured::DECODE_LAYER_MS
}

/// Expected emitted tokens per round under greedy longest-prefix acceptance with a flat
/// per-position accept probability `p`: `E[accept_len] + 1`.
///
/// ⚠ Exploration helper only. Flat `p` is OPTIMISTIC against measurement: Metal's MTP-1 at k=2
/// showed ~90% per-position accept yet emitted only ~1.9 tok/round, because acceptance is
/// longest-PREFIX and the drafter's misses cluster early. Price with a MEASURED emit whenever
/// one exists (Metal anchors: MTP-1 k=2 ~1.9, k=4 ~2.65; DSpark ~1.6-1.7).
pub fn emit_per_round(p: f64, k: u32) -> f64 {
    let p = p.clamp(0.0, 1.0);
    // accept_len = j with probability p^j (1-p) for j < k, and p^k at j = k.
    let mut e_accept = 0.0;
    for j in 1..=k {
        e_accept += p.powi(j as i32); // P(accept_len >= j) summed = E[accept_len]
    }
    e_accept + 1.0
}

/// Price one round at a given emitted-tokens-per-round.
///
/// `emit` is taken directly (not derived from an accept probability) because it is the quantity
/// that has actually been measured elsewhere and the flat-p model overstates it — see
/// [`emit_per_round`].
///
/// The verify is 43 layers of: attention and dense projections at `scaling.cost_factor(k)`
/// (rows share KV and weights), the shared expert likewise, and the ROUTED experts at a flat
/// `k` (rows route to ~disjoint experts; the measured ~2.3% overlap is ignored as noise).
/// The measured FFN sublayer is 7 distinct expert weight sets, so 6/7 of it is routed.
pub fn price_round(
    g: &Geometry,
    drafter: Drafter,
    k: u32,
    emit: f64,
    scaling: RowScaling,
) -> SpecRound {
    let cf = scaling.cost_factor(k);
    let routed_share = 6.0 / 7.0;
    let ffn_ms = measured::FFN_SUBLAYER_MS
        * (routed_share * k as f64 + (1.0 - routed_share) * cf);
    let attn_ms = measured::ATTN_SUBLAYER_MS * cf;
    let verify_ms = g.layers as f64 * (attn_ms + ffn_ms);
    SpecRound {
        k,
        drafter,
        draft_ms: drafter.draft_ms(),
        verify_ms,
        emit,
        experts_touched_per_layer: g.experts_touched_per_layer(k),
    }
}

/// The `shared_fraction` at which a configuration reaches `target_speedup` — the number the
/// k-row kernel measurement has to beat before the loop is worth building.
///
/// Returns `None` if even perfect sharing (`shared_fraction = 1.0`, a k-row verify as cheap as
/// one decode outside the routed experts) cannot reach the target: in that case no kernel work
/// closes the gap and the lever is the ACCEPT RATE, not the verify.
pub fn break_even_shared_fraction(
    g: &Geometry,
    drafter: Drafter,
    k: u32,
    emit: f64,
    target_speedup: f64,
) -> Option<f64> {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let at = |s: f64| {
        price_round(g, drafter, k, emit, RowScaling { shared_fraction: s }).speedup()
    };
    if at(hi) < target_speedup {
        return None;
    }
    if at(lo) >= target_speedup {
        return Some(0.0);
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if at(mid) >= target_speedup {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::ExpertLayout;

    fn g() -> Geometry {
        Geometry::ds4_flash(ExpertLayout::PackedNibbleNativeE8m0)
    }

    #[test]
    fn baseline_is_the_measured_track() {
        // 43 x 7.878 = 338.8 ms/token, ~2.95 tok/s — the handoff's headline resident figure.
        let ms = baseline_ms_per_token();
        assert!((ms - 338.754).abs() < 0.1, "baseline {ms}");
    }

    #[test]
    fn emit_model_matches_the_metal_anchors() {
        // Flat p reproducing Metal's MTP-1 measurements: at k=4 the measured overall accept was
        // 43.8% and the run emitted ~2.65 tok/round.
        let e = emit_per_round(0.65, 4);
        assert!((e - 2.6).abs() < 0.15, "emit {e}");
        // DSpark on Metal emitted ~1.6-1.7 per round.
        let e = emit_per_round(0.45, 5);
        assert!(e > 1.55 && e < 1.95, "emit {e}");
    }

    /// Metal-measured emits per round — the honest anchors (flat-p overstates them).
    const MTP1_EMIT_K2: f64 = 1.9;
    const MTP1_EMIT_K4: f64 = 2.65;

    #[test]
    fn worst_case_always_loses() {
        // With nothing shared a k-row verify costs k decodes; the only gain is the bonus token,
        // and the drafter costs extra — so at the MEASURED emits spec decode prices BELOW 1.0.
        for (k, emit) in [(2, MTP1_EMIT_K2), (4, MTP1_EMIT_K4), (8, 3.0)] {
            let r = price_round(&g(), Drafter::Mtp1, k, emit, RowScaling::worst_case());
            assert!(r.speedup() < 1.0, "k={k} speedup {}", r.speedup());
        }
    }

    #[test]
    fn load_shared_default_is_parity_not_folklore() {
        // The point of the module: with the hoist-derived 0.55 sharing and measured emits,
        // MTP-1 is ~1.17x at k=2 and ~parity at k=4 — the folklore 1.6-2x is NOT priced in.
        let k2 = price_round(&g(), Drafter::Mtp1, 2, MTP1_EMIT_K2, RowScaling::load_shared());
        assert!(k2.speedup() > 1.05 && k2.speedup() < 1.3, "k2 {}", k2.speedup());
        let k4 = price_round(&g(), Drafter::Mtp1, 4, MTP1_EMIT_K4, RowScaling::load_shared());
        assert!(k4.speedup() > 0.85 && k4.speedup() < 1.1, "k4 {}", k4.speedup());
    }

    #[test]
    fn the_lever_is_attention_sharing_times_deep_accept() {
        // Near-perfect sharing (KV-traffic-bound attention at long context) TIMES a deep-accept
        // drafter (emit ~3 at k=4, which MTP-1 does not reach — DSpark's promise) is where the
        // folklore figure lives. This is the regime the k-row kernel measurement plus the accept
        // measurement have to confirm or kill.
        let r = price_round(&g(), Drafter::Mtp1, 4, 3.05, RowScaling { shared_fraction: 0.95 });
        assert!(r.speedup() > 1.6, "speedup {}", r.speedup());
        // Either factor alone is NOT enough: perfect sharing at the measured MTP-1 emit...
        let sharing_only =
            price_round(&g(), Drafter::Mtp1, 4, MTP1_EMIT_K4, RowScaling { shared_fraction: 1.0 });
        assert!(sharing_only.speedup() < 1.6, "sharing alone {}", sharing_only.speedup());
        // ...and deep accept at the default sharing.
        let accept_only = price_round(&g(), Drafter::Mtp1, 4, 3.05, RowScaling::load_shared());
        assert!(accept_only.speedup() < 1.6, "accept alone {}", accept_only.speedup());
    }

    #[test]
    fn measured_sharing_verdict() {
        // The 910c A/B (2026-08-07) measured shared_fraction 0.306 at the operating point.
        // At that sharing, MTP-1 spec decode is parity at k=2 and a LOSS at k=4 — the loop is
        // not worth building on this drafter with the current vector-core attention lowering.
        let k2 = price_round(&g(), Drafter::Mtp1, 2, MTP1_EMIT_K2, RowScaling::measured_kq2());
        assert!(k2.speedup() < 1.1, "k2 {}", k2.speedup());
        let k4 = price_round(&g(), Drafter::Mtp1, 4, MTP1_EMIT_K4, RowScaling::measured_kq2());
        assert!(k4.speedup() < 0.9, "k4 {}", k4.speedup());
        // What WOULD clear it at the measured sharing: a deep-accept drafter emitting ~2.6 of
        // a possible 3 at k=2 — i.e. the lever is ACCEPT RATE, exactly what
        // break_even_shared_fraction said for k=4. DSpark's 3 stages cost more to draft and
        // still price >= 1.35x IF that accept materialises (Metal measured it short: ~1.7).
        let deep =
            price_round(&g(), Drafter::Dspark3, 2, 2.6, RowScaling::measured_kq2());
        assert!(deep.speedup() > 1.35, "deep-accept k2 {}", deep.speedup());
    }

    #[test]
    fn break_even_reports_a_target_or_refuses() {
        // At k=4 with the Metal emit, reaching 1.2x needs shared_fraction ~0.78...
        let s = break_even_shared_fraction(&g(), Drafter::Mtp1, 4, MTP1_EMIT_K4, 1.2);
        assert!(s.is_some());
        let s = s.unwrap();
        assert!(s > 0.7 && s < 0.85, "break-even {s}");
        // ...but no sharing reaches 2x there: the answer must be a refusal, telling the caller
        // the lever is accept rate, not kernels.
        assert!(break_even_shared_fraction(&g(), Drafter::Mtp1, 4, MTP1_EMIT_K4, 2.0).is_none());
    }

    #[test]
    fn streamed_bytes_scale_with_k() {
        // The expected expert union for k rows: no meaningful sharing at small k, so a
        // streaming plan must budget ~k x the per-token expert traffic for a verify.
        let r = price_round(&g(), Drafter::Mtp1, 4, MTP1_EMIT_K4, RowScaling::load_shared());
        let per_token = g().experts_touched_per_layer(1);
        assert!(
            r.experts_touched_per_layer > 3.5 * per_token,
            "union {} vs per-token {per_token}",
            r.experts_touched_per_layer
        );
    }
}
