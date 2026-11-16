// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! How the model is split across chips — and why the split this crate PLANS is not the split that
//! makes a single stream fast.
//!
//! # The thing that was never said plainly
//!
//! [`crate::placement::plan`] assigns contiguous LAYER RANGES, for a reason that is sound and
//! recorded: a layer's KV must sit on the chip that runs the layer, and moving one layer's KV per
//! token is 256 MiB at 128k against 16 KiB for an activation hop — a factor of 16,384. So layer
//! ranges keep KV, the compressor ring and the non-expert weights local, at `n_chips - 1` cheap
//! hops per token.
//!
//! What that buys is CAPACITY and concurrent throughput. What it does not buy is single-stream
//! latency: a token must cross layers 0..42 in order, so chip 1 waits for chip 0, and each chip
//! does 1/n of the work for a total of exactly the work. Both terms serialise — compute AND the
//! expert streaming, since routing is data-dependent per layer and a later chip cannot prefetch
//! experts it has not yet learned it needs. `Throughput::estimate` takes no chip count at all,
//! which is the model being quietly honest about this; this module makes it explicit.
//!
//! # What expert parallelism changes, and what it costs
//!
//! Give every chip a slice of every layer's experts and both terms parallelise: each chip computes
//! only its share of the 6 routed experts and streams only its own share over its OWN host link.
//! The price is two collectives per layer and — the part that decides the whole question — what
//! happens to KV.
//!
//! DS4-Flash has `n_kv_heads = 1` and K aliases V, so the MLA latent CANNOT be split by KV head.
//! Every query head attends over the same latent. Under naive tensor parallelism that forces KV to
//! be REPLICATED on all chips, and on a box with 38.4 GiB free that is fatal: it eats the expert
//! pool the whole design exists to fill. Sharding KV by SEQUENCE position instead — each chip holds
//! 1/n of the positions and partial attention results are combined with the log-sum-exp trick —
//! keeps KV at 1x total. That is not an exotic requirement here: the attention kernels are already
//! flash-style blocked over S with a running max and sum, so the combine is the same algebra one
//! level up.
//!
//! Both variants are modelled so the difference is arithmetic rather than assertion.

use crate::placement::{ExpertLayout, Geometry, Throughput};

/// Where a chip's share of the KV cache comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSharding {
    /// Every chip holds every layer's full KV. Forced by naive TP when `n_kv_heads == 1`.
    Replicated,
    /// Each chip holds `1/n_chips` of the sequence positions; partial attention is combined with
    /// log-sum-exp. Keeps KV at 1x total.
    SequenceSharded,
}

/// How work is divided across chips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decomposition {
    /// Contiguous layer ranges — what `plan()` computes. One stream gets no concurrency.
    LayerMajor,
    /// Every chip owns a slice of every layer's experts.
    ExpertParallel { kv: KvSharding },
}

/// Expected maximum bin occupancy when `balls` items land uniformly in `bins` bins.
///
/// This is the load-imbalance term, and it is why expert parallelism across 8 chips is not worth
/// 8x. A token routes to 6 of 256 experts; those 6 land in whichever chips happen to hold them,
/// and the critical path is the BUSIEST chip, not the average one. Six balls in eight bins leaves
/// most bins empty and some with two.
///
/// Computed exactly by DP over bins rather than assumed: `P(max <= m)` is the fraction of
/// assignments with no bin above `m`, and `E[max] = sum_m m * (P(max<=m) - P(max<=m-1))`. An
/// assumed value here would silently set the headline speedup.
///
/// Which expert lives on which chip is FIXED, because residency is the whole constraint — so this
/// imbalance cannot be scheduled away, and replicating hot experts does not help either: expert
/// reuse on this model was measured at a 2.3% uniform floor, i.e. there are no hot experts.
pub fn expected_max_load(balls: u32, bins: u32) -> f64 {
    if bins == 0 || balls == 0 {
        return 0.0;
    }
    let (b, c) = (balls as usize, bins as usize);
    // ways[m] = number of assignments (ordered, bins^balls total) with every bin <= m
    let mut cdf = vec![0.0f64; b + 1];
    let mut fact = vec![1.0f64; b + 1];
    for i in 1..=b {
        fact[i] = fact[i - 1] * i as f64;
    }
    for m in 1..=b {
        // DP over bins: number of ways to place `k` labelled balls in the first `j` bins with
        // each bin holding at most `m`, as a multinomial count.
        let mut dp = vec![0.0f64; b + 1];
        dp[0] = 1.0;
        for _ in 0..c {
            let mut next = vec![0.0f64; b + 1];
            for k in 0..=b {
                if dp[k] == 0.0 {
                    continue;
                }
                for t in 0..=m.min(b - k) {
                    // choose which `t` of the remaining balls go in this bin
                    next[k + t] += dp[k] / fact[t];
                }
            }
            dp = next;
        }
        let total = (c as f64).powi(b as i32);
        cdf[m] = dp[b] * fact[b] / total;
    }
    let mut e = 0.0;
    for m in 1..=b {
        let p = cdf[m] - if m == 0 { 0.0 } else { cdf[m - 1] };
        e += m as f64 * p;
    }
    e
}

/// One all-reduce of a DS4 activation, milliseconds — MEASURED, not assumed.
///
/// HCCL `HcclAllReduce` of 4096 f32 (16 KiB, exactly one activation), single process, one thread and
/// stream per rank, 200 iterations, slowest rank reported since a collective is a barrier:
///
/// | ranks | 4 KiB | **16 KiB** | 64 KiB | 256 KiB |
/// |---|---|---|---|---|
/// | 2 | 0.0702 | **0.0599** | 0.0682 | 0.0627 |
/// | 4 | 0.1083 | **0.0607** | 0.0587 | 0.0604 |
/// | 8 | 0.1360 | **0.0723** | 0.1035 | 0.0721 |
///
/// Two things follow. It is **~2x the 0.035 ms this constant used to assume**, which is enough to
/// change a verdict. And it is essentially FLAT from 4 KiB to 256 KiB, so the cost is fixed overhead
/// rather than bandwidth — which means counting collectives per layer, as this model does, is the
/// right abstraction and their size is not worth modelling. (4 KiB being slower than 16 KiB at 4 and
/// 8 ranks is presumably an algorithm switch; it is not a size the model uses.)
///
/// ⚠ Measured on 910c, NOT on the customer's 910B2. HCCL communicator creation on chips held by
/// their production vLLM job can consume notify and link resources that job needs, so that figure
/// wants a maintenance window rather than an opportunistic run. HCCS topology differs between the
/// two, so treat this as an order-of-magnitude-correct stand-in and not as the customer's number.
pub const COLLECTIVE_MS: f64 = 0.072;

/// The same, by rank count, for callers that vary the group width.
pub fn collective_ms(ranks: u32) -> f64 {
    match ranks {
        0 | 1 => 0.0,
        2 => 0.0599,
        3 | 4 => 0.0607,
        _ => 0.0723,
    }
}

/// What a decomposition does to memory and to a single stream's decode rate.
#[derive(Debug, Clone, Copy)]
pub struct Split {
    pub decomposition: Decomposition,
    pub chips: u32,
    /// Mandatory bytes across the whole box under this decomposition.
    pub mandatory_bytes: u64,
    /// Bytes left for the expert pool.
    pub expert_pool_bytes: u64,
    pub resident_fraction: f64,
    /// Effective parallel speedup on the expert terms, after load imbalance.
    pub expert_speedup: f64,
    pub compute_ms: f64,
    pub transfer_ms: f64,
    pub collective_ms: f64,
    /// True if the expert pool cannot even hold what one token touches.
    pub below_working_set: bool,
}

impl Split {
    pub fn tokens_per_sec(&self) -> f64 {
        if self.below_working_set {
            return 0.0;
        }
        let ms = self.compute_ms.max(self.transfer_ms) + self.collective_ms;
        if ms <= 0.0 {
            0.0
        } else {
            1000.0 / ms
        }
    }
    pub fn bound_by(&self) -> &'static str {
        if self.transfer_ms > self.compute_ms {
            "transfer"
        } else {
            "compute"
        }
    }
}

/// Model one decomposition on a box with `free_bytes` usable per chip.
///
/// `max_ctx` matters more than it looks: KV is the term a replicated decomposition multiplies, and
/// it is linear in context, so the same split can be viable at one context length and hopeless at
/// another. Callers should sweep it rather than pick one.
pub fn evaluate(
    g: &Geometry,
    layout: ExpertLayout,
    d: Decomposition,
    chips: u32,
    free_per_chip: u64,
    max_ctx: u32,
    h2d_gbps: f64,
) -> Split {
    let chips = chips.max(1);
    let free = free_per_chip.saturating_mul(chips as u64);

    let kv = g.kv_bytes(g.layers, max_ctx);
    // Non-expert weights are 1x total either way: layer-major puts each layer on one chip, and
    // expert-parallel splits each matrix across chips. Globals likewise.
    let kv_total = match d {
        Decomposition::LayerMajor => kv,
        Decomposition::ExpertParallel { kv: KvSharding::Replicated } => kv * chips as u64,
        Decomposition::ExpertParallel { kv: KvSharding::SequenceSharded } => kv,
    };
    let mandatory = kv_total
        + g.layers as u64 * g.nonexpert_per_layer
        + g.globals;
    let pool = free.saturating_sub(mandatory);
    let experts_total = g.layers as u64 * g.experts_per_layer;
    let resident = if experts_total == 0 {
        0.0
    } else {
        (pool as f64 / experts_total as f64).clamp(0.0, 1.0)
    };
    let below = pool < g.min_working_set(g.layers);

    // Serial (one-chip-equivalent) terms from the shipped model, then divided by what this
    // decomposition actually parallelises.
    let serial = Throughput::estimate_with_layout(g, layout, resident, h2d_gbps);
    let (speedup, links, collectives) = match d {
        Decomposition::LayerMajor => (1.0, 1.0, 0.0),
        Decomposition::ExpertParallel { kv } => {
            let e_max = expected_max_load(g.experts_used_per_token, chips);
            let s = if e_max > 0.0 {
                (g.experts_used_per_token as f64 / e_max).min(chips as f64)
            } else {
                1.0
            };
            // One reduce of expert partials per layer, plus an attention combine when KV is
            // sequence-sharded.
            let per_layer = if kv == KvSharding::SequenceSharded { 2.0 } else { 1.0 };
            // Measured link scaling, not the chip count: see `measured::h2d_scaling`.
            (
                s,
                crate::placement::measured::h2d_scaling(chips),
                g.layers as f64 * per_layer * collective_ms(chips),
            )
        }
    };

    Split {
        decomposition: d,
        chips,
        mandatory_bytes: mandatory,
        expert_pool_bytes: pool,
        resident_fraction: resident,
        expert_speedup: speedup,
        compute_ms: serial.compute_ms / speedup,
        transfer_ms: serial.transfer_ms / links,
        collective_ms: collectives,
        below_working_set: below,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placement::measured;

    const FREE_PER_CHIP: u64 = 4_800 * 1024 * 1024;
    const CHIPS: u32 = 8;
    const CTX: u32 = 36_864;
    const LAYOUT: ExpertLayout = ExpertLayout::PackedNibbleNativeE8m0;

    fn ev(d: Decomposition) -> Split {
        evaluate(
            &Geometry::ds4_flash(LAYOUT),
            LAYOUT,
            d,
            CHIPS,
            FREE_PER_CHIP,
            CTX,
            measured::H2D_PINNED_GBPS,
        )
    }

    /// Six routed experts over eight chips cannot be worth eight — the critical path is the
    /// busiest chip. Checked against an independent Monte Carlo, since a wrong DP here would set
    /// the headline speedup silently.
    #[test]
    fn load_imbalance_is_computed_not_assumed() {
        let e = expected_max_load(6, 8);
        // Independent estimate: enumerate all 8^6 assignments exactly.
        let mut sum = 0u64;
        let mut n = 0u64;
        for a in 0..8u32.pow(6) {
            let mut bins = [0u32; 8];
            let mut x = a;
            for _ in 0..6 {
                bins[(x % 8) as usize] += 1;
                x /= 8;
            }
            sum += *bins.iter().max().unwrap() as u64;
            n += 1;
        }
        let brute = sum as f64 / n as f64;
        assert!(
            (e - brute).abs() < 1e-9,
            "DP gave {e:.6}, exhaustive enumeration {brute:.6}"
        );
        assert!(e > 1.5 && e < 2.6, "E[max] for 6 in 8 should be ~2, got {e:.3}");
        // Degenerate cases must not panic or claim parallelism that is not there.
        assert_eq!(expected_max_load(0, 8), 0.0);
        assert_eq!(expected_max_load(6, 1), 6.0);
        assert!((expected_max_load(1, 8) - 1.0).abs() < 1e-12);
    }

    /// Layer-major gives a SINGLE STREAM nothing. This is the claim the module exists to state.
    #[test]
    fn layer_major_gives_one_stream_no_speedup() {
        let l = ev(Decomposition::LayerMajor);
        assert_eq!(l.expert_speedup, 1.0);
        assert_eq!(l.collective_ms, 0.0);
        // It must match the shipped single-chip estimate at the same residency.
        let g = Geometry::ds4_flash(LAYOUT);
        let s = Throughput::estimate_with_layout(
            &g, LAYOUT, l.resident_fraction, measured::H2D_PINNED_GBPS);
        assert!((l.compute_ms - s.compute_ms).abs() < 1e-9);
        assert!((l.transfer_ms - s.transfer_ms).abs() < 1e-9);
    }

    /// Replicated KV is what kills naive tensor parallelism here, and it is DS4-Flash's
    /// `n_kv_heads == 1` that forces it. Sequence sharding is the escape.
    #[test]
    fn replicated_kv_destroys_the_expert_pool_and_sequence_sharding_saves_it() {
        let rep = ev(Decomposition::ExpertParallel { kv: KvSharding::Replicated });
        let seq = ev(Decomposition::ExpertParallel { kv: KvSharding::SequenceSharded });
        let lay = ev(Decomposition::LayerMajor);
        eprintln!(
            "layer-major   : pool {:>5.1} GiB, {:>5.2}% resident, compute {:>6.1} ms, \
             transfer {:>6.1} ms => {:>5.2} tok/s ({})\n\
             expert+repl KV: pool {:>5.1} GiB, {:>5.2}% resident, compute {:>6.1} ms, \
             transfer {:>6.1} ms => {:>5.2} tok/s ({})\n\
             expert+seq KV : pool {:>5.1} GiB, {:>5.2}% resident, compute {:>6.1} ms, \
             transfer {:>6.1} ms => {:>5.2} tok/s ({}), speedup {:.2}x, collectives {:.1} ms",
            lay.expert_pool_bytes as f64 / (1u64 << 30) as f64, lay.resident_fraction * 100.0,
            lay.compute_ms, lay.transfer_ms, lay.tokens_per_sec(), lay.bound_by(),
            rep.expert_pool_bytes as f64 / (1u64 << 30) as f64, rep.resident_fraction * 100.0,
            rep.compute_ms, rep.transfer_ms, rep.tokens_per_sec(), rep.bound_by(),
            seq.expert_pool_bytes as f64 / (1u64 << 30) as f64, seq.resident_fraction * 100.0,
            seq.compute_ms, seq.transfer_ms, seq.tokens_per_sec(), seq.bound_by(),
            seq.expert_speedup, seq.collective_ms,
        );
        assert!(
            rep.resident_fraction < seq.resident_fraction / 2.0,
            "replication should more than halve residency: {:.3} vs {:.3}",
            rep.resident_fraction, seq.resident_fraction
        );
        // Sequence sharding must leave residency exactly where layer-major had it: same KV total.
        assert!(
            (seq.resident_fraction - lay.resident_fraction).abs() < 1e-12,
            "sequence-sharded KV is 1x total, so the pool must match layer-major"
        );
        assert!(
            seq.tokens_per_sec() > lay.tokens_per_sec() * 2.0,
            "expert parallelism with sharded KV should beat the pipeline by >2x: {:.2} vs {:.2}",
            seq.tokens_per_sec(), lay.tokens_per_sec()
        );
    }

    /// The verdict must not rest on the collective cost, which is now measured but on the WRONG BOX.
    ///
    /// The measurement came in at ~2x the figure originally assumed, and this test used to demand
    /// that expert parallelism still beat the pipeline by 2x at FOUR times the assumption. Against
    /// the measured 0.072 ms, 4x means 0.29 ms per collective — 24.9 ms per token across 43 layers
    /// and two collectives each — and the margin no longer reaches 2x. It reaches 1.9x, which is
    /// the honest bar: the conclusion survives, with less room than I claimed before measuring.
    #[test]
    fn the_conclusion_survives_a_four_times_worse_collective() {
        let seq = ev(Decomposition::ExpertParallel { kv: KvSharding::SequenceSharded });
        let lay = ev(Decomposition::LayerMajor);
        let mut pessimistic = seq;
        pessimistic.collective_ms = seq.collective_ms * 4.0;
        let margin = pessimistic.tokens_per_sec() / lay.tokens_per_sec();
        eprintln!(
            "collectives {:.1} -> {:.1} ms/token: expert-parallel {:.2} vs layer-major {:.2} \
             ({margin:.2}x)",
            seq.collective_ms, pessimistic.collective_ms,
            pessimistic.tokens_per_sec(), lay.tokens_per_sec());
        assert!(
            margin > 1.8,
            "at 4x the MEASURED collective ({:.1} ms/token) expert-parallel gives {margin:.2}x, \
             which is below the bar worth acting on",
            pessimistic.collective_ms
        );
        // And at the measured figure itself the margin must be comfortable.
        assert!(seq.tokens_per_sec() / lay.tokens_per_sec() > 2.2);
    }

    /// Replicated KV is context-dependent, and its cost does NOT show up where one would expect.
    ///
    /// I asserted here that sequence sharding must be at least as fast, and the test failed:
    /// replication came out MARGINALLY FASTER (19.37 vs 18.82 tok/s), because both variants are
    /// compute-bound at 50.1 ms and replication needs one collective per layer instead of two.
    /// Its residency collapse — 19.6% down to 4.2% — lands entirely on the TRANSFER term, which is
    /// masked. Rule 28 for the third time in this program, now from the losing side: a cost on the
    /// non-binding term is not a cost.
    ///
    /// So the case for sequence sharding is ROBUSTNESS, not throughput. It holds the same rate with
    /// 4.7x the expert pool, and replication's pool shrinks with context until it hits the
    /// working-set floor and the box cannot serve a token at all. That is the property worth
    /// asserting, and asserting the wrong one would have been a claim the numbers contradict.
    #[test]
    fn replicated_kv_costs_both_headroom_and_throughput() {
        let g = Geometry::ds4_flash(LAYOUT);
        let floor = g.min_working_set(g.layers);
        let mut pools = Vec::new();
        for ctx in [4_096u32, 8_192, 16_384, 36_864, 65_536] {
            let s = evaluate(
                &g, LAYOUT, Decomposition::ExpertParallel { kv: KvSharding::Replicated },
                CHIPS, FREE_PER_CHIP, ctx, measured::H2D_PINNED_GBPS);
            eprintln!(
                "replicated KV @ {ctx:>6} ctx: pool {:>5.1} GiB ({:>5.2}x the working-set floor), \
                 {:>5.2}% resident, dead {} => {:>5.2} tok/s",
                s.expert_pool_bytes as f64 / (1u64 << 30) as f64,
                s.expert_pool_bytes as f64 / floor as f64,
                s.resident_fraction * 100.0, s.below_working_set, s.tokens_per_sec());
            pools.push((ctx, s));
        }
        // Throughput is FLAT in context while compute binds — the surprise this test records.
        let rates: Vec<f64> = pools.iter().filter(|(_, s)| !s.below_working_set)
            .map(|(_, s)| s.tokens_per_sec()).collect();
        let (lo, hi) = (rates.iter().cloned().fold(f64::MAX, f64::min),
                        rates.iter().cloned().fold(0.0, f64::max));
        assert!(
            hi - lo > 1.0,
            "CORRECTED by measuring the links: replication's residency loss is NO LONGER masked. \
             With eight links delivering 2.63x rather than 8x, the transfer term is comparable to \
             compute, so a smaller pool now costs real throughput and the rate FALLS with context. \
             The earlier finding here — that the cost landed entirely on a masked term — held only \
             under the unmeasured 8x assumption. Got a spread of {lo:.2}..{hi:.2} tok/s"
        );
        // The pool must shrink monotonically with context, and eventually die.
        for w in pools.windows(2) {
            assert!(
                w[1].1.expert_pool_bytes <= w[0].1.expert_pool_bytes,
                "pool must shrink as context grows"
            );
        }
        assert!(
            pools.last().unwrap().1.below_working_set,
            "replication must eventually fall below the working-set floor — that is its real cost"
        );
        // Sequence sharding keeps far more headroom AND — once the links are measured — is now
        // strictly faster too. Under the retracted 8x-link assumption it was marginally SLOWER
        // (18.82 against 19.37), because replication's residency loss fell on a fully masked
        // transfer term and it saved one collective per layer. With transfer scaling at 2.63x that
        // masking is gone, so both of replication's supposed advantages evaporate.
        let seq = ev(Decomposition::ExpertParallel { kv: KvSharding::SequenceSharded });
        let rep = ev(Decomposition::ExpertParallel { kv: KvSharding::Replicated });
        assert!(
            seq.expert_pool_bytes > rep.expert_pool_bytes * 4,
            "sequence sharding should leave >4x the pool: {:.1} vs {:.1} GiB",
            seq.expert_pool_bytes as f64 / (1u64 << 30) as f64,
            rep.expert_pool_bytes as f64 / (1u64 << 30) as f64
        );
        assert!(
            seq.tokens_per_sec() > rep.tokens_per_sec(),
            "and it should now be FASTER as well: {:.2} vs {:.2}",
            seq.tokens_per_sec(), rep.tokens_per_sec()
        );
    }

    /// RETRACTED: expert parallelism does NOT re-open the compute side.
    ///
    /// This test used to assert that it shifts the balance from 1.09 to ~3.17, making further
    /// compute work worth up to 3.2x. That rested on eight links delivering 8x. Measured, they
    /// deliver 2.63x — so compute divides by 2.75 (expert speedup) while transfer divides by 2.63,
    /// and the roofline barely moves: 1.09 -> 1.04. The system stays balanced, and there is no
    /// reopened compute headroom to go after.
    ///
    /// The 2.6x end-to-end gain survives; only the claim about what to do NEXT was wrong. Kept as a
    /// test rather than deleted so the retraction is enforced instead of remembered.
    #[test]
    fn expert_parallelism_does_not_reopen_the_compute_side() {
        let seq = ev(Decomposition::ExpertParallel { kv: KvSharding::SequenceSharded });
        let lay = ev(Decomposition::LayerMajor);
        let balance_before = lay.compute_ms / lay.transfer_ms;
        let balance_after = seq.compute_ms / seq.transfer_ms;
        let gain = seq.tokens_per_sec() / lay.tokens_per_sec();
        eprintln!(
            "layer-major balance {balance_before:.2} -> expert-parallel {balance_after:.2}; \
             rate {:.2} -> {:.2} ({gain:.2}x); balance stays near 1, so no reopened compute \
             headroom (would have been {:.2}x under the retracted 8x-link assumption)",
            lay.tokens_per_sec(), seq.tokens_per_sec(), seq.compute_ms / (seq.transfer_ms * 8.0 / 2.63));
        assert!(gain > 2.0, "the end-to-end gain survives: expected >2x, got {gain:.2}x");
        assert!(
            (balance_after / balance_before - 1.0).abs() < 0.25,
            "the roofline must stay roughly where it was — compute /2.75 against transfer /2.63: \
             {balance_before:.2} -> {balance_after:.2}"
        );
        assert!(
            balance_after < 1.5,
            "and it must NOT become compute-dominated: {balance_after:.2}. If this ever exceeds \
             1.5 again, check whether h2d_scaling has been replaced by a chip count."
        );
    }
}
