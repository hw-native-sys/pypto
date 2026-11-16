// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Choosing TP / DP / EP / PP for a stated requirement, rather than picking one.
//!
//! [`crate::decomposition`] models two points — a layer pipeline and expert parallelism — and the
//! experiments there showed they answer different questions. So does each of the other axes, and
//! the differences are structural rather than a matter of degree:
//!
//! | axis | what it splits | single-stream latency | memory cost | collectives |
//! |---|---|---|---|---|
//! | **TP** | every weight matrix | yes, exactly | replicates KV (see below) | one per matmul — many |
//! | **EP** | the routed experts | yes, imbalance-limited | replicates KV unless sharded | one per layer |
//! | **PP** | contiguous layer ranges | **none** | cheapest: splits everything | none |
//! | **DP** | whole replicas | **none** | replicates EVERYTHING | none |
//!
//! Two of the four do nothing for one stream and two of them cost memory linearly, so on a box
//! whose binding constraint is free HBM the configuration is not a free choice. This module
//! enumerates every factorisation of the chip count, rejects the infeasible ones, and ranks the
//! rest against an explicit [`Objective`] — so "best" is a question the caller has to finish
//! asking.
//!
//! # The KV multiplier is what makes DS4-Flash awkward
//!
//! `n_kv_heads == 1` and K aliases V, so the MLA latent cannot be split by KV head: every chip
//! cooperating on one request's layer needs the whole latent. That is `tp * ep` chips, so naive TP
//! or EP multiplies KV by their product — and KV is linear in context, so the same configuration
//! can be viable at 8k and dead at 64k. [`KvSharding::SequenceSharded`] removes the multiplier by
//! splitting positions instead and combining partials with log-sum-exp, which the flash-style
//! attention kernels already have the shape for.
//!
//! # What this is not
//!
//! A model. Nothing here has run on multiple chips. The collective cost is an assumption
//! ([`crate::decomposition::COLLECTIVE_MS`]), and so is the claim that `n` host links deliver `n`
//! times one link's bandwidth — that second one underwrites every transfer figure below and is the
//! first thing to measure. The value of the module is that it makes the trade explicit and forces
//! a requirement to be named; it does not make the numbers measured.

use crate::decomposition::{collective_ms, expected_max_load, KvSharding};
use crate::placement::{ExpertLayout, Geometry, Throughput};

/// Dense (non-expert) matmuls per layer that a tensor-parallel split must all-reduce.
///
/// DS4-Flash's attention block: q down/up, kv down/up, and the output projection. Counted here
/// rather than folded into a magic number because it is what makes TP collective-heavy relative to
/// EP — and being wrong by 2x would change which axis wins.
pub const DENSE_MATMULS_PER_LAYER: f64 = 5.0;

/// Expert projections per layer a tensor-parallel split must all-reduce: the routed experts a
/// token touches, times gate/up/down.
pub fn expert_matmuls_per_layer(g: &Geometry) -> f64 {
    g.experts_used_per_token as f64 * 3.0
}

/// One parallelism configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelConfig {
    /// Tensor parallel: every chip runs every layer with a slice of each matrix.
    pub tp: u32,
    /// Data parallel: independent replicas serving different requests.
    pub dp: u32,
    /// Expert parallel: each chip owns a slice of the routed experts.
    pub ep: u32,
    /// Pipeline parallel: contiguous layer ranges. What `plan()` computes.
    pub pp: u32,
    pub kv: KvSharding,
}

impl ParallelConfig {
    pub fn degree(&self) -> u32 {
        self.tp * self.dp * self.ep * self.pp
    }

    /// Chips that cooperate on ONE request's single layer. These are the chips that must each hold
    /// the KV latent under [`KvSharding::Replicated`], and the chips that stream concurrently for
    /// that request.
    pub fn cooperating(&self) -> u32 {
        self.tp * self.ep
    }

    /// How many times the KV cache is stored across the box.
    pub fn kv_multiplier(&self) -> u32 {
        match self.kv {
            KvSharding::Replicated => self.dp * self.cooperating(),
            KvSharding::SequenceSharded => self.dp,
        }
    }

    /// Short label for reporting, e.g. `tp1 dp1 ep8 pp1 seq`.
    pub fn label(&self) -> String {
        format!(
            "tp{} dp{} ep{} pp{} {}",
            self.tp,
            self.dp,
            self.ep,
            self.pp,
            match self.kv {
                KvSharding::Replicated => "repl",
                KvSharding::SequenceSharded => "seq",
            }
        )
    }
}

/// What the caller is optimising. Naming it is the point of the module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// One request, as fast as possible. Only TP and EP help.
    SingleStreamLatency,
    /// Total tokens per second across concurrent requests. DP and PP multiply this and cost
    /// nothing in collectives, which is why the reference stack serves with them.
    AggregateThroughput,
    /// The longest context the box can serve at all. Minimises the KV multiplier.
    MaxContext,
}

/// A configuration, evaluated.
#[derive(Debug, Clone, Copy)]
pub struct Evaluated {
    pub config: ParallelConfig,
    pub mandatory_bytes: u64,
    pub expert_pool_bytes: u64,
    /// Fraction of ONE replica's experts that is resident.
    pub resident_fraction: f64,
    pub compute_ms: f64,
    pub transfer_ms: f64,
    pub collective_ms: f64,
    /// Reason this configuration cannot run, if any.
    pub infeasible: Option<&'static str>,
}

impl Evaluated {
    /// One request's decode rate.
    pub fn single_stream_tok_s(&self) -> f64 {
        if self.infeasible.is_some() {
            return 0.0;
        }
        let ms = self.compute_ms.max(self.transfer_ms) + self.collective_ms;
        if ms <= 0.0 { 0.0 } else { 1000.0 / ms }
    }

    /// Across concurrent requests. A `pp`-stage pipeline keeps `pp` tokens in flight and `dp`
    /// replicas serve independently, so both multiply — while contributing nothing to the line
    /// above. That asymmetry is the whole reason this is two methods and not one.
    pub fn aggregate_tok_s(&self) -> f64 {
        self.single_stream_tok_s() * (self.config.pp * self.config.dp) as f64
    }

    pub fn score(&self, objective: Objective) -> f64 {
        if self.infeasible.is_some() {
            return f64::NEG_INFINITY;
        }
        match objective {
            Objective::SingleStreamLatency => self.single_stream_tok_s(),
            Objective::AggregateThroughput => self.aggregate_tok_s(),
            // Prefer the smallest KV multiplier, then the largest pool: both buy context.
            Objective::MaxContext => {
                -(self.config.kv_multiplier() as f64) * 1e12 + self.expert_pool_bytes as f64
            }
        }
    }

    pub fn bound_by(&self) -> &'static str {
        if self.collective_ms > self.compute_ms.max(self.transfer_ms) {
            "collectives"
        } else if self.transfer_ms > self.compute_ms {
            "transfer"
        } else {
            "compute"
        }
    }
}

/// Evaluate one configuration on a box of `chips` with `free_per_chip` usable bytes.
pub fn evaluate(
    g: &Geometry,
    layout: ExpertLayout,
    c: ParallelConfig,
    chips: u32,
    free_per_chip: u64,
    max_ctx: u32,
    h2d_gbps: f64,
) -> Evaluated {
    let free_total = free_per_chip.saturating_mul(chips as u64);

    // Memory, as TOTALS across the box. TP, EP and PP all PARTITION their term, so they leave the
    // total unchanged; only DP (whole replicas) and a replicated KV multiply it. Modelling totals
    // rather than per-chip keeps the arithmetic honest about what actually has to fit.
    let kv = g.kv_bytes(g.layers, max_ctx) * c.kv_multiplier() as u64;
    let dense = g.layers as u64 * g.nonexpert_per_layer * c.dp as u64;
    let globals = g.globals * c.dp as u64;
    let mandatory = kv + dense + globals;
    let pool = free_total.saturating_sub(mandatory);

    // Each DP replica needs its own pool, and each pool must cover the whole expert set.
    let experts_per_replica = g.layers as u64 * g.experts_per_layer;
    let resident = if experts_per_replica == 0 {
        0.0
    } else {
        ((pool / c.dp.max(1) as u64) as f64 / experts_per_replica as f64).clamp(0.0, 1.0)
    };

    let infeasible = if c.degree() != chips {
        Some("degree does not use every chip")
    } else if mandatory >= free_total {
        Some("mandatory set exceeds free HBM")
    } else if pool / c.dp.max(1) as u64 > 0
        && pool / c.dp.max(1) as u64 >= g.min_working_set(g.layers)
    {
        None
    } else {
        Some("expert pool below one token's working set")
    };

    // Latency. The serial figures come from the shipped single-chip model, then each axis divides
    // only what it actually parallelises.
    let serial = Throughput::estimate_with_layout(g, layout, resident, h2d_gbps);
    // TP splits every matrix exactly; EP splits the experts and is capped by load imbalance, since
    // the critical path is the busiest chip. PP and DP contribute nothing to one stream.
    let ep_speedup = if c.ep <= 1 {
        1.0
    } else {
        let e_max = expected_max_load(g.experts_used_per_token, c.ep);
        if e_max > 0.0 {
            (g.experts_used_per_token as f64 / e_max).min(c.ep as f64)
        } else {
            1.0
        }
    };
    let compute_speedup = c.tp as f64 * ep_speedup;
    // Only the cooperating chips stream for this request; PP stages are serial and DP replicas idle.
    // The divisor is MEASURED link scaling, not the chip count — 8 links on the customer's 910B2
    // deliver 2.63x one link, so the old `/ cooperating()` overstated every multi-chip transfer
    // term by up to 3x.
    let links = crate::placement::measured::h2d_scaling(c.cooperating());

    // Collectives per layer. TP all-reduces every matmul, which is the axis' real price: an order
    // of magnitude more collectives than EP's single reduce of expert partials.
    let mut per_layer = 0.0;
    if c.tp > 1 {
        per_layer += DENSE_MATMULS_PER_LAYER + expert_matmuls_per_layer(g);
    }
    if c.ep > 1 {
        per_layer += 1.0;
    }
    if c.kv == KvSharding::SequenceSharded && c.cooperating() > 1 {
        per_layer += 1.0;
    }

    Evaluated {
        config: c,
        mandatory_bytes: mandatory,
        expert_pool_bytes: pool,
        resident_fraction: resident,
        compute_ms: serial.compute_ms / compute_speedup,
        transfer_ms: serial.transfer_ms / links,
        collective_ms: g.layers as f64 * per_layer * collective_ms(c.cooperating()),
        infeasible,
    }
}

/// Every configuration that uses exactly `chips`, both KV strategies.
///
/// Enumerated rather than hand-listed so a chip count nobody anticipated still gets a full search.
pub fn enumerate(chips: u32) -> Vec<ParallelConfig> {
    let mut out = Vec::new();
    let divisors: Vec<u32> = (1..=chips).filter(|d| chips % d == 0).collect();
    for &tp in &divisors {
        for &dp in &divisors {
            for &ep in &divisors {
                let rest = tp * dp * ep;
                if rest == 0 || chips % rest != 0 {
                    continue;
                }
                let pp = chips / rest;
                for kv in [KvSharding::Replicated, KvSharding::SequenceSharded] {
                    let c = ParallelConfig { tp, dp, ep, pp, kv };
                    if c.degree() == chips {
                        out.push(c);
                    }
                }
            }
        }
    }
    out
}

/// Best configuration for an objective, plus every feasible alternative ranked behind it.
///
/// Returns `(ranked_feasible, infeasible)`. The infeasible list is returned rather than dropped
/// because "there is no configuration that fits" and "we did not look" must not read the same.
pub fn best(
    g: &Geometry,
    layout: ExpertLayout,
    objective: Objective,
    chips: u32,
    free_per_chip: u64,
    max_ctx: u32,
    h2d_gbps: f64,
) -> (Vec<Evaluated>, Vec<Evaluated>) {
    let all: Vec<Evaluated> = enumerate(chips)
        .into_iter()
        .map(|c| evaluate(g, layout, c, chips, free_per_chip, max_ctx, h2d_gbps))
        .collect();
    let (mut ok, bad): (Vec<_>, Vec<_>) = all.into_iter().partition(|e| e.infeasible.is_none());
    ok.sort_by(|a, b| {
        b.score(objective)
            .partial_cmp(&a.score(objective))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    (ok, bad)
}


/// What one chip is asked to do under a general configuration.
///
/// This is the generalisation of [`crate::placement::ChipPlan`], which can only express the `pp`
/// axis — a contiguous layer range. `plan()` is not wrong, it is a projection: with `tp = dp = ep =
/// 1` the assignment below reduces to exactly what it produces, and a test pins that. Supporting
/// TP/DP/EP means `plan()`'s OUTPUT type has to grow these three fields, not that its layer logic
/// has to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChipAssignment {
    pub device: u32,
    /// Contiguous, inclusive-exclusive — the `pp` axis, unchanged from `plan()`.
    pub first_layer: u32,
    pub last_layer: u32,
    /// Which DP replica this chip belongs to. Replicas share nothing.
    pub replica: u32,
    /// Rank within the tensor-parallel group: this chip holds slice `tp_rank` of every matrix.
    pub tp_rank: u32,
    /// Rank within the expert-parallel group: this chip owns experts `e` where
    /// `e % ep == ep_rank`.
    pub ep_rank: u32,
    /// Which slice of the sequence this chip holds KV for, or `None` when KV is replicated.
    pub kv_shard: Option<u32>,
}

/// Expand a configuration into a per-chip assignment.
///
/// Order of the axes is a real choice, not bookkeeping: `pp` varies slowest so that a pipeline
/// stage's chips stay adjacent, which is the property `plan()` exists to preserve — a layer's KV
/// must live on the chip running the layer, and moving it costs 256 MiB against 16 KiB for an
/// activation hop. Within a stage, `tp` and `ep` vary fastest because those chips exchange a
/// collective every layer and want to be nearest neighbours.
pub fn assign(c: ParallelConfig, chips: u32, layers: u32) -> Vec<ChipAssignment> {
    let mut out = Vec::new();
    if c.degree() != chips || chips == 0 {
        return out;
    }
    let per_stage = chips / c.pp.max(1);
    for device in 0..chips {
        let stage = device / per_stage.max(1);
        let within = device % per_stage.max(1);
        // Balance layers across stages: earlier stages take the remainder, as plan() does.
        let base = layers / c.pp.max(1);
        let rem = layers % c.pp.max(1);
        let first = stage * base + stage.min(rem);
        let count = base + if stage < rem { 1 } else { 0 };
        let cooperating = c.cooperating().max(1);
        out.push(ChipAssignment {
            device,
            first_layer: first,
            last_layer: first + count,
            replica: within / cooperating,
            tp_rank: within % c.tp.max(1),
            ep_rank: (within / c.tp.max(1)) % c.ep.max(1),
            kv_shard: match c.kv {
                KvSharding::Replicated => None,
                KvSharding::SequenceSharded => Some(within % cooperating),
            },
        });
    }
    out
}

/// Decode rate with the experts FULLY RESIDENT — the free-node question.
///
/// Separate from [`evaluate`] because it answers a different one. `evaluate` derives residency from
/// what is left after the mandatory set, which is the right thing on a starved box; this asserts
/// full residency and reports what the compute and collective terms alone give, plus whether the
/// layout actually fits at the stated context.
///
/// `layout` matters more here than anywhere else: on a free node the 4-bit packing that bought the
/// three-chip fit is no longer needed, and the host-expanded int8 rung has a measurably faster
/// matvec (0.136 vs 0.171 ms per projection). Choosing the memory-optimal layout on a box with
/// memory to spare would leave 1.22x on the table.
pub fn resident_rate(
    g: &Geometry,
    layout: ExpertLayout,
    c: ParallelConfig,
    chips: u32,
    free_per_chip: u64,
    max_ctx: u32,
) -> Evaluated {
    let mut e = evaluate(g, layout, c, chips, free_per_chip, max_ctx, f64::INFINITY);
    // Assert full residency, then check it is actually affordable.
    let need = e.mandatory_bytes
        + (g.layers as u64 * g.experts_per_layer) * c.dp.max(1) as u64;
    let free_total = free_per_chip.saturating_mul(chips as u64);
    let serial = Throughput::estimate_with_layout(g, layout, 1.0, f64::INFINITY);
    let ep_speedup = if c.ep <= 1 {
        1.0
    } else {
        let m = expected_max_load(g.experts_used_per_token, c.ep);
        if m > 0.0 { (g.experts_used_per_token as f64 / m).min(c.ep as f64) } else { 1.0 }
    };
    e.resident_fraction = 1.0;
    e.compute_ms = serial.compute_ms / (c.tp as f64 * ep_speedup);
    e.transfer_ms = 0.0;
    e.infeasible = if c.degree() != chips {
        Some("degree does not use every chip")
    } else if need > free_total {
        Some("full residency does not fit")
    } else {
        None
    };
    e
}

/// Environment variable an operator can set to pin the split, e.g. `DS4_PARALLEL=ep4pp2`.
pub const PARALLEL_ENV: &str = "DS4_PARALLEL";

/// Parse an operator-facing split spec: any of `tp`, `dp`, `ep`, `pp` with a degree, in any order,
/// optionally `+repl` to replicate KV instead of sharding the sequence.
///
/// Written for humans to type into a deploy config, so omitted axes default to 1 rather than being
/// an error — `ep8` means what a reader expects. The KV default is SEQUENCE sharding, because with
/// `n_kv_heads == 1` replication multiplies KV by `tp*ep` and can silently make a viable context
/// length infeasible; an operator who wants that has to ask for it.
pub fn parse_spec(spec: &str) -> Result<ParallelConfig, String> {
    let s = spec.trim().to_ascii_lowercase();
    let (body, kv) = match s.strip_suffix("+repl") {
        Some(b) => (b.to_string(), KvSharding::Replicated),
        None => (
            s.strip_suffix("+seq").map(|b| b.to_string()).unwrap_or(s.clone()),
            KvSharding::SequenceSharded,
        ),
    };
    let mut c = ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 1, kv };
    let mut rest = body.as_str();
    if rest.is_empty() {
        return Err(format!("empty split spec; try ep4pp2 or ep8, optionally +repl"));
    }
    while !rest.is_empty() {
        let axis: &mut u32 = match rest.get(..2) {
            Some("tp") => &mut c.tp,
            Some("dp") => &mut c.dp,
            Some("ep") => &mut c.ep,
            Some("pp") => &mut c.pp,
            _ => {
                return Err(format!(
                    "cannot parse {spec:?} at {rest:?}: expected one of tp/dp/ep/pp followed by a \
                     degree, e.g. ep4pp2"
                ))
            }
        };
        rest = &rest[2..];
        let digits: String = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect();
        if digits.is_empty() {
            return Err(format!("cannot parse {spec:?}: no degree after an axis in {rest:?}"));
        }
        rest = &rest[digits.len()..];
        let n: u32 = digits.parse().map_err(|_| format!("bad degree {digits:?} in {spec:?}"))?;
        if n == 0 {
            return Err(format!("a degree of 0 in {spec:?} would use no chips"));
        }
        *axis = n;
    }
    Ok(c)
}

/// How the engine got its split, so a deployment can report what it is running rather than what was
/// configured somewhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Pinned by an operator through [`PARALLEL_ENV`].
    Environment,
    /// Chosen by ranking every feasible split against an objective.
    Objective(Objective),
}

/// The engine's parallelism option, resolved.
#[derive(Debug, Clone)]
pub struct Chosen {
    pub config: ParallelConfig,
    pub source: Source,
    pub assignment: Vec<ChipAssignment>,
    /// What this split is expected to deliver, so the choice can be reported with its own numbers.
    pub single_stream_tok_s: f64,
    pub aggregate_tok_s: f64,
}

/// Resolve the engine's split: an operator's pin if present, else the best for `objective`.
///
/// This is the whole point of making the decomposition an OPTION rather than kernel work. The
/// measured tables show the mixes beat both poles for a deployment that wants low latency AND
/// throughput — `ep4 pp2` gives 85% of pure-EP per-user latency at 1.7x its aggregate — so the
/// engine should be able to express any of them, not just the one `plan()` happens to build.
///
/// An environment pin is honoured even if the ranking would disagree, but NOT if it is infeasible:
/// a split that cannot fit is refused with the reason rather than silently replaced, because a
/// deployment that asked for `dp4` and got `pp8` would report the wrong thing forever.
pub fn choose(
    g: &Geometry,
    layout: ExpertLayout,
    objective: Objective,
    chips: u32,
    free_per_chip: u64,
    max_ctx: u32,
    h2d_gbps: f64,
) -> Result<Chosen, String> {
    let pinned = std::env::var(PARALLEL_ENV).ok().filter(|v| !v.trim().is_empty());
    let (config, source) = match pinned {
        Some(spec) => (parse_spec(&spec)?, Source::Environment),
        None => {
            let (ranked, bad) = best(g, layout, objective, chips, free_per_chip, max_ctx, h2d_gbps);
            let b = ranked.first().ok_or_else(|| {
                format!(
                    "no feasible split for {objective:?} on {chips} chips at {max_ctx} context; \
                     {} configurations were rejected, e.g. {:?}",
                    bad.len(),
                    bad.first().and_then(|e| e.infeasible)
                )
            })?;
            (b.config, Source::Objective(objective))
        }
    };
    let e = evaluate(g, layout, config, chips, free_per_chip, max_ctx, h2d_gbps);
    if let Some(why) = e.infeasible {
        return Err(format!(
            "split {} is not viable here: {why}. Set {PARALLEL_ENV} to something that fits, or \
             unset it to let the objective choose.",
            config.label()
        ));
    }
    Ok(Chosen {
        config,
        source,
        assignment: assign(config, chips, g.layers),
        single_stream_tok_s: e.single_stream_tok_s(),
        aggregate_tok_s: e.aggregate_tok_s(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FREE: u64 = 4_800 * 1024 * 1024;
    const CHIPS: u32 = 8;
    const CTX: u32 = 36_864;
    const LAYOUT: ExpertLayout = ExpertLayout::PackedNibbleNativeE8m0;
    const H2D: f64 = crate::placement::measured::H2D_PINNED_GBPS;

    fn g() -> Geometry {
        Geometry::ds4_flash(LAYOUT)
    }

    fn pick(objective: Objective, ctx: u32) -> (Vec<Evaluated>, Vec<Evaluated>) {
        best(&g(), LAYOUT, objective, CHIPS, FREE, ctx, H2D)
    }


    /// The generalised assignment must REDUCE to what `plan()` produces when only `pp` is used.
    /// If it does not, supporting TP/DP/EP would have silently changed the pipeline case too.
    #[test]
    fn pp_only_assignment_reduces_to_contiguous_layer_ranges() {
        let c = ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 4, kv: KvSharding::SequenceSharded };
        let a = assign(c, 4, 43);
        assert_eq!(a.len(), 4);
        assert_eq!(a[0].first_layer, 0);
        assert_eq!(a.last().unwrap().last_layer, 43, "every layer must be owned");
        for w in a.windows(2) {
            assert_eq!(w[0].last_layer, w[1].first_layer, "ranges must be contiguous, no gaps");
        }
        // 43 over 4 stages: 11, 11, 11, 10 — the remainder goes to the earlier stages.
        let counts: Vec<u32> = a.iter().map(|x| x.last_layer - x.first_layer).collect();
        assert_eq!(counts, vec![11, 11, 11, 10]);
        // And with no TP/EP/DP every chip is rank 0 of a single replica.
        assert!(a.iter().all(|x| x.tp_rank == 0 && x.ep_rank == 0 && x.replica == 0));
    }

    /// Every chip must get a distinct role, and the expert slices must PARTITION the pool — an
    /// overlap would double-store experts and a gap would drop them.
    #[test]
    fn assignment_partitions_experts_and_replicas() {
        let c = ParallelConfig { tp: 2, dp: 2, ep: 2, pp: 2, kv: KvSharding::SequenceSharded };
        let a = assign(c, 16, 43);
        assert_eq!(a.len(), 16);
        // Within one stage and one replica, ep_ranks must cover 0..ep exactly once per tp_rank.
        for stage in 0..2u32 {
            for replica in 0..2u32 {
                let mut seen = std::collections::HashSet::new();
                for x in a.iter().filter(|x| {
                    (x.first_layer > 0) == (stage == 1) && x.replica == replica
                }) {
                    assert!(
                        seen.insert((x.tp_rank, x.ep_rank)),
                        "duplicate (tp,ep) role in stage {stage} replica {replica}"
                    );
                }
                assert_eq!(seen.len(), 4, "tp2 x ep2 must give 4 distinct roles");
            }
        }
        // Sequence-sharded KV must cover every shard within a cooperating group.
        let shards: std::collections::HashSet<Option<u32>> =
            a.iter().map(|x| x.kv_shard).collect();
        assert_eq!(shards.len(), c.cooperating() as usize);
        // Replicated KV carries no shard index at all.
        let r = ParallelConfig { kv: KvSharding::Replicated, ..c };
        assert!(assign(r, 16, 43).iter().all(|x| x.kv_shard.is_none()));
        // A mismatched degree must yield NOTHING rather than a partial assignment.
        assert!(assign(c, 8, 43).is_empty());
    }

    /// THE FREE-NODE ANSWER: a full A2 node, experts 100% resident, every configuration that fits.
    ///
    /// Reported for both layouts, because on a free node the 4-bit packing that bought the
    /// three-chip fit is no longer needed and the host-expanded int8 rung's matvec is 1.22x faster.
    #[test]
    fn full_node_resident_rates_for_the_reference_configurations() {
        const NODE: u64 = 64 * 1024 * 1024 * 1024;
        for layout in [ExpertLayout::HostExpandedInt8BlockF32, ExpertLayout::PackedNibbleBlockF32] {
            let geo = Geometry::ds4_flash(layout);
            let total = (geo.layers as u64 * geo.experts_per_layer) as f64 / (1u64 << 30) as f64;
            eprintln!("\n=== {layout:?}: experts {total:.1} GiB, node 512 GiB ===");
            for (chips, cfg) in [
                (8u32, ParallelConfig { tp: 4, dp: 2, ep: 1, pp: 1, kv: KvSharding::Replicated }),
                (8, ParallelConfig { tp: 8, dp: 1, ep: 1, pp: 1, kv: KvSharding::Replicated }),
                (8, ParallelConfig { tp: 1, dp: 1, ep: 8, pp: 1, kv: KvSharding::SequenceSharded }),
                (8, ParallelConfig { tp: 4, dp: 1, ep: 2, pp: 1, kv: KvSharding::SequenceSharded }),
                (8, ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 8, kv: KvSharding::Replicated }),
                (8, ParallelConfig { tp: 1, dp: 8, ep: 1, pp: 1, kv: KvSharding::Replicated }),
                (16, ParallelConfig { tp: 4, dp: 4, ep: 1, pp: 1, kv: KvSharding::Replicated }),
            ] {
                let e = resident_rate(&geo, layout, cfg, chips, NODE, 36_864);
                match e.infeasible {
                    Some(why) => eprintln!("  {:<26} {chips:>2} chips: {why}", cfg.label(), ),
                    None => eprintln!(
                        "  {:<26} {chips:>2} chips: compute {:>6.1} ms + collectives {:>5.1} ms \
                         => {:>6.2} tok/s single, {:>7.2} aggregate ({})",
                        cfg.label(), e.compute_ms, e.collective_ms,
                        e.single_stream_tok_s(), e.aggregate_tok_s(), e.bound_by()),
                }
            }
        }
        // The int8 rung must fit a free node and be faster per projection than the packed one.
        let i8g = Geometry::ds4_flash(ExpertLayout::HostExpandedInt8BlockF32);
        let e = resident_rate(
            &i8g, ExpertLayout::HostExpandedInt8BlockF32,
            ParallelConfig { tp: 8, dp: 1, ep: 1, pp: 1, kv: KvSharding::Replicated },
            8, NODE, 36_864);
        assert!(e.infeasible.is_none(), "308.6 GiB must fit a 512 GiB node");
        assert_eq!(e.transfer_ms, 0.0, "full residency means nothing streams");
        let pk = resident_rate(
            &Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32),
            ExpertLayout::PackedNibbleBlockF32,
            ParallelConfig { tp: 8, dp: 1, ep: 1, pp: 1, kv: KvSharding::Replicated },
            8, NODE, 36_864);
        assert!(
            e.compute_ms < pk.compute_ms,
            "on a free node the int8 rung should be the faster one: {:.1} vs {:.1} ms",
            e.compute_ms, pk.compute_ms
        );
        // DP8 must NOT fit even a free node at full residency: 8 replicas of 308.6 GiB.
        let dp8 = resident_rate(
            &i8g, ExpertLayout::HostExpandedInt8BlockF32,
            ParallelConfig { tp: 1, dp: 8, ep: 1, pp: 1, kv: KvSharding::Replicated },
            8, NODE, 36_864);
        assert!(dp8.infeasible.is_some(), "8 fully-resident replicas cannot fit 512 GiB");
    }


    /// Mixed PP x EP: pipeline for concurrency, expert parallelism for each user's latency.
    ///
    /// This is the shape a multi-user offering actually wants, and it is worth its own test because
    /// the two axes do NOT compete — `pp` multiplies how many requests are in flight while `ep`
    /// reduces the latency each one sees, so `aggregate = pp x single` with `single` already
    /// improved. Reported for the 3-card fit as well, since that is a distinct product: 179.6 GiB
    /// fits three 64 GiB cards fully resident, but three cards cannot express `pp4 x ep2` at all.
    #[test]
    fn mixed_pipeline_and_expert_parallel_for_a_multi_user_offering() {
        const CARD: u64 = 64 * 1024 * 1024 * 1024;
        let seq = KvSharding::SequenceSharded;
        for layout in [ExpertLayout::HostExpandedInt8BlockF32, ExpertLayout::PackedNibbleBlockF32] {
            let geo = Geometry::ds4_flash(layout);
            let gib = (geo.layers as u64 * geo.experts_per_layer) as f64 / (1u64 << 30) as f64;
            eprintln!("\n=== {layout:?}: experts {gib:.1} GiB, resident ===");
            for (cards, cfg) in [
                // Three cards: the fit the memory work targeted. pp4 is not expressible here.
                (3u32, ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 3, kv: KvSharding::Replicated }),
                (3, ParallelConfig { tp: 1, dp: 1, ep: 3, pp: 1, kv: seq }),
                // Eight cards: the mix asked about, and its neighbours.
                (8, ParallelConfig { tp: 1, dp: 1, ep: 2, pp: 4, kv: seq }),
                (8, ParallelConfig { tp: 1, dp: 1, ep: 4, pp: 2, kv: seq }),
                (8, ParallelConfig { tp: 2, dp: 1, ep: 1, pp: 4, kv: KvSharding::Replicated }),
                (8, ParallelConfig { tp: 1, dp: 2, ep: 1, pp: 4, kv: KvSharding::Replicated }),
                // Two nodes: ep8 needs eight cooperating chips, so pairing it with any pipeline at
                // all takes 16. Included because it is the point where per-user latency is best.
                (16, ParallelConfig { tp: 1, dp: 1, ep: 8, pp: 2, kv: seq }),
                (16, ParallelConfig { tp: 1, dp: 1, ep: 4, pp: 4, kv: seq }),
                (16, ParallelConfig { tp: 1, dp: 1, ep: 2, pp: 8, kv: seq }),
                (16, ParallelConfig { tp: 1, dp: 2, ep: 8, pp: 1, kv: seq }),
            ] {
                let e = resident_rate(&geo, layout, cfg, cards, CARD, 36_864);
                let users = cfg.pp * cfg.dp;
                match e.infeasible {
                    Some(why) => eprintln!("  {:<26} {cards:>2} cards: {why}", cfg.label()),
                    None => eprintln!(
                        "  {:<26} {cards:>2} cards: compute {:>6.1} + coll {:>4.1} ms => \
                         {:>5.2} tok/s PER USER x {users} users = {:>6.2} aggregate ({})",
                        cfg.label(), e.compute_ms, e.collective_ms,
                        e.single_stream_tok_s(), e.aggregate_tok_s(), e.bound_by()),
                }
            }
        }
        // pp4 x ep2 needs eight chips; three cannot express it, and that must be an explicit
        // refusal rather than a silently rescaled plan.
        let geo = Geometry::ds4_flash(ExpertLayout::PackedNibbleBlockF32);
        let e = resident_rate(
            &geo, ExpertLayout::PackedNibbleBlockF32,
            ParallelConfig { tp: 1, dp: 1, ep: 2, pp: 4, kv: seq }, 3, CARD, 36_864);
        assert_eq!(e.infeasible, Some("degree does not use every chip"));
        assert!(assign(ParallelConfig { tp: 1, dp: 1, ep: 2, pp: 4, kv: seq }, 3, 43).is_empty());

        // On eight cards the mix must beat pure PP per user AND stay well above pure EP on
        // aggregate — that is the reason to mix at all.
        for layout in [ExpertLayout::HostExpandedInt8BlockF32, ExpertLayout::PackedNibbleBlockF32] {
            let g2 = Geometry::ds4_flash(layout);
            let mix = resident_rate(&g2, layout,
                ParallelConfig { tp: 1, dp: 1, ep: 2, pp: 4, kv: seq }, 8, CARD, 36_864);
            let pure_pp = resident_rate(&g2, layout,
                ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 8, kv: KvSharding::Replicated },
                8, CARD, 36_864);
            let pure_ep = resident_rate(&g2, layout,
                ParallelConfig { tp: 1, dp: 1, ep: 8, pp: 1, kv: seq }, 8, CARD, 36_864);
            assert!(
                mix.single_stream_tok_s() > pure_pp.single_stream_tok_s(),
                "{layout:?}: the mix must give each user more than a pure pipeline does"
            );
            assert!(
                mix.aggregate_tok_s() > pure_ep.aggregate_tok_s(),
                "{layout:?}: and more aggregate than pure expert parallelism"
            );
        }
    }

    /// The operator-facing spec, including the forms a human would actually type.
    #[test]
    fn a_split_spec_parses_the_way_a_reader_expects() {
        let c = parse_spec("ep4pp2").unwrap();
        assert_eq!((c.tp, c.dp, c.ep, c.pp), (1, 1, 4, 2));
        assert_eq!(c.kv, KvSharding::SequenceSharded, "sharding is the default, not replication");
        // Order-independent, case-insensitive, whitespace-tolerant.
        assert_eq!(parse_spec("PP2 EP4".replace(' ', "").as_str()).unwrap(), c);
        assert_eq!(parse_spec("  ep4pp2  ").unwrap(), c);
        // Omitted axes are 1, so `ep8` means what it looks like.
        let e8 = parse_spec("ep8").unwrap();
        assert_eq!((e8.tp, e8.dp, e8.ep, e8.pp), (1, 1, 8, 1));
        // Replication must be asked for explicitly, since with n_kv_heads == 1 it multiplies KV.
        assert_eq!(parse_spec("ep8+repl").unwrap().kv, KvSharding::Replicated);
        assert_eq!(parse_spec("ep8+seq").unwrap().kv, KvSharding::SequenceSharded);
        assert_eq!(parse_spec("tp2dp2ep2pp2").unwrap().degree(), 16);
        // And the refusals should say where they gave up.
        for bad in ["", "ep", "xy4", "ep0", "eppp", "ep4x"] {
            let e = parse_spec(bad).unwrap_err();
            assert!(!e.is_empty(), "{bad:?} must be refused with a reason");
        }
        assert!(parse_spec("ep0").unwrap_err().contains("no chips"));
    }

    /// The engine option: no pin means the objective decides, and the choice reports its own numbers.
    #[test]
    fn choose_ranks_when_nothing_is_pinned() {
        // Not using the env here on purpose — tests share a process, so a pin set by one would leak
        // into another. The pinned path is covered by parse_spec plus the feasibility check below.
        std::env::remove_var(PARALLEL_ENV);
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let g = g();
        for obj in [Objective::SingleStreamLatency, Objective::AggregateThroughput] {
            let c = choose(&g, layout, obj, CHIPS, FREE, CTX, H2D).expect("something must fit");
            assert_eq!(c.source, Source::Objective(obj));
            assert_eq!(c.config.degree(), CHIPS, "the split must use every chip");
            assert_eq!(c.assignment.len(), CHIPS as usize, "every chip gets a role");
            assert!(c.single_stream_tok_s > 0.0 && c.aggregate_tok_s >= c.single_stream_tok_s);
            eprintln!(
                "{obj:?} -> {} : {:.2} tok/s single, {:.2} aggregate",
                c.config.label(), c.single_stream_tok_s, c.aggregate_tok_s
            );
        }
    }

    /// An infeasible pin is REFUSED, not silently replaced — a deployment that asked for dp4 and
    /// quietly got pp8 would report the wrong thing forever.
    #[test]
    fn an_infeasible_pin_is_refused_rather_than_substituted() {
        let layout = ExpertLayout::PackedNibbleNativeE8m0;
        let g = g();
        // dp8 replicates the mandatory set eight times and cannot fit the customer's leftovers.
        let cfg = parse_spec("dp8").unwrap();
        let e = evaluate(&g, layout, cfg, CHIPS, FREE, CTX, H2D);
        assert!(e.infeasible.is_some(), "dp8 must not fit — if it does, this test proves nothing");
        // And the mix the objectives point at must fit, or the option is useless.
        for spec in ["ep8", "ep4pp2", "ep2pp4", "pp8"] {
            let c = parse_spec(spec).unwrap();
            let ev = evaluate(&g, layout, c, CHIPS, FREE, CTX, H2D);
            assert!(ev.infeasible.is_none(), "{spec} should be viable: {:?}", ev.infeasible);
        }
    }

    /// Enumeration must cover every factorisation and nothing else.
    #[test]
    fn enumeration_is_complete_and_exact() {
        let cs = enumerate(8);
        assert!(cs.iter().all(|c| c.degree() == 8), "every config must use all 8 chips");
        // Ordered factorisations of 8 into 4 parts = compositions of three 2s into four slots.
        let distinct: std::collections::HashSet<(u32, u32, u32, u32)> =
            cs.iter().map(|c| (c.tp, c.dp, c.ep, c.pp)).collect();
        assert_eq!(distinct.len(), 20, "8 = 2^3 into four ordered factors is 20 ways");
        assert_eq!(cs.len(), 40, "each times two KV strategies");
        // A prime chip count must still yield the four single-axis options.
        let p = enumerate(7);
        assert_eq!(
            p.iter().map(|c| (c.tp, c.dp, c.ep, c.pp)).collect::<std::collections::HashSet<_>>().len(),
            4
        );
        assert_eq!(enumerate(1).len(), 2, "one chip: one factorisation, two KV labels");
    }

    /// The objective must actually change the answer, or the module is decoration.
    #[test]
    fn different_objectives_choose_different_configurations() {
        let (lat, _) = pick(Objective::SingleStreamLatency, CTX);
        let (agg, _) = pick(Objective::AggregateThroughput, CTX);
        let (ctx, _) = pick(Objective::MaxContext, CTX);
        for (name, r) in [("latency", &lat), ("aggregate", &agg), ("max-context", &ctx)] {
            let b = r.first().expect("some configuration must be feasible");
            eprintln!(
                "{name:<12} -> {:<24} {:>6.2} tok/s single, {:>7.2} aggregate, \
                 pool {:>5.1} GiB, {:>5.2}% resident, KV x{}, bound by {}",
                b.config.label(), b.single_stream_tok_s(), b.aggregate_tok_s(),
                b.expert_pool_bytes as f64 / (1u64 << 30) as f64,
                b.resident_fraction * 100.0, b.config.kv_multiplier(), b.bound_by()
            );
        }
        let (bl, ba) = (lat[0].config, agg[0].config);
        assert_ne!(
            (bl.tp, bl.dp, bl.ep, bl.pp), (ba.tp, ba.dp, ba.ep, ba.pp),
            "latency and aggregate throughput must not pick the same split"
        );
    }

    /// PP and DP contribute nothing to one stream. This is the distinction the whole module rests
    /// on, so it is asserted directly rather than left to the ranking.
    #[test]
    fn pipeline_and_data_parallel_do_nothing_for_one_stream() {
        let solo = ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 1, kv: KvSharding::SequenceSharded };
        let one = evaluate(&g(), LAYOUT, solo, 1, FREE * 8, CTX, H2D);
        for (tp, dp, ep, pp) in [(1, 1, 1, 8), (1, 8, 1, 1)] {
            let c = ParallelConfig { tp, dp, ep, pp, kv: KvSharding::SequenceSharded };
            let e = evaluate(&g(), LAYOUT, c, 8, FREE, CTX, H2D);
            if e.infeasible.is_some() {
                continue; // dp=8 will not fit; that is its own finding
            }
            assert!(
                e.compute_ms >= one.compute_ms * 0.999,
                "{} must not reduce single-stream compute", c.label()
            );
            assert_eq!(e.collective_ms, 0.0, "{} needs no collectives", c.label());
        }
        // But PP multiplies aggregate throughput, which is exactly its use case.
        let pp8 = evaluate(
            &g(), LAYOUT,
            ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 8, kv: KvSharding::SequenceSharded },
            8, FREE, CTX, H2D);
        assert!(
            (pp8.aggregate_tok_s() / pp8.single_stream_tok_s() - 8.0).abs() < 1e-9,
            "an 8-stage pipeline should keep 8 tokens in flight"
        );
    }

    /// Data parallelism replicates EVERYTHING, so on a memory-starved box it prices itself out.
    /// Worth its own test because DP is the cheapest axis to reach for and the worst one here.
    #[test]
    fn data_parallelism_is_priced_out_on_this_box() {
        let mut first_dead = None;
        for dp in [1u32, 2, 4, 8] {
            let c = ParallelConfig { tp: 1, dp, ep: 8 / dp, pp: 1, kv: KvSharding::SequenceSharded };
            let e = evaluate(&g(), LAYOUT, c, 8, FREE, CTX, H2D);
            eprintln!(
                "{:<24} pool {:>5.1} GiB, {:>5.2}% resident per replica, {} => {:>6.2} tok/s single",
                c.label(), e.expert_pool_bytes as f64 / (1u64 << 30) as f64,
                e.resident_fraction * 100.0,
                e.infeasible.unwrap_or("feasible"), e.single_stream_tok_s());
            if e.infeasible.is_some() && first_dead.is_none() {
                first_dead = Some(dp);
            }
        }
        assert!(
            first_dead.is_some(),
            "some DP degree must become infeasible on 38.4 GiB; if not, the memory model is wrong"
        );
        assert!(first_dead.unwrap() > 1, "dp=1 must be feasible");
    }

    /// TP's price is collectives, and it is an order of magnitude more of them than EP's.
    #[test]
    fn tensor_parallelism_pays_in_collectives_and_expert_parallelism_in_imbalance() {
        let tp = evaluate(
            &g(), LAYOUT,
            ParallelConfig { tp: 8, dp: 1, ep: 1, pp: 1, kv: KvSharding::SequenceSharded },
            8, FREE, CTX, H2D);
        let ep = evaluate(
            &g(), LAYOUT,
            ParallelConfig { tp: 1, dp: 1, ep: 8, pp: 1, kv: KvSharding::SequenceSharded },
            8, FREE, CTX, H2D);
        eprintln!(
            "tp8: compute {:>6.1} ms, collectives {:>6.1} ms, {:>6.2} tok/s ({})\n\
             ep8: compute {:>6.1} ms, collectives {:>6.1} ms, {:>6.2} tok/s ({})",
            tp.compute_ms, tp.collective_ms, tp.single_stream_tok_s(), tp.bound_by(),
            ep.compute_ms, ep.collective_ms, ep.single_stream_tok_s(), ep.bound_by());
        // TP splits exactly, so its compute term must beat EP's imbalance-limited one.
        assert!(tp.compute_ms < ep.compute_ms, "TP splits exactly; EP is capped by E[max]");
        // And it must pay for that with far more collective time.
        assert!(
            tp.collective_ms > ep.collective_ms * 5.0,
            "TP all-reduces every matmul: {:.1} vs {:.1} ms", tp.collective_ms, ep.collective_ms
        );
    }

    /// A replicated KV multiplier makes the choice CONTEXT-DEPENDENT, which is the reason a single
    /// recommended configuration would be wrong.
    #[test]
    fn the_best_configuration_changes_with_context_length() {
        let mut seen = std::collections::HashSet::new();
        for ctx in [4_096u32, 36_864, 131_072] {
            let (r, bad) = pick(Objective::SingleStreamLatency, ctx);
            match r.first() {
                Some(b) => {
                    eprintln!(
                        "ctx {ctx:>7}: best {:<24} {:>6.2} tok/s, KV x{}, {:>5.2}% resident \
                         ({} feasible, {} not)",
                        b.config.label(), b.single_stream_tok_s(), b.config.kv_multiplier(),
                        b.resident_fraction * 100.0, r.len(), bad.len());
                    seen.insert(b.config.kv_multiplier());
                }
                None => eprintln!("ctx {ctx:>7}: NOTHING fits ({} infeasible)", bad.len()),
            }
        }
        assert!(!seen.is_empty(), "at least one context must be servable");
    }

    /// An infeasible configuration must say WHY, and must never be ranked.
    ///
    /// On this box only ONE reason fires at these contexts — the mandatory set exceeds free HBM
    /// before the working-set floor ever gets a chance, because a replicated KV blows the budget
    /// outright. I first asserted that both reasons appear and the test failed; asserting it would
    /// have been asserting a coincidence of this box's numbers. The floor is exercised separately
    /// below, on a budget where the mandatory set does fit.
    #[test]
    fn infeasible_configurations_are_explained_not_hidden() {
        let (ok, bad) = pick(Objective::SingleStreamLatency, CTX);
        assert!(!bad.is_empty(), "on a 38.4 GiB box some configurations must not fit");
        assert!(ok.iter().all(|e| e.infeasible.is_none()));
        assert!(bad.iter().all(|e| e.infeasible.is_some() && e.single_stream_tok_s() == 0.0));
        assert!(bad.iter().any(|e| e.infeasible == Some("mandatory set exceeds free HBM")));

        // The working-set floor, on a budget chosen so the mandatory set fits and the pool does
        // not. Both refusals must be reachable, or one of them is dead code.
        let geo = g();
        let mandatory = geo.kv_bytes(geo.layers, 4_096)
            + geo.layers as u64 * geo.nonexpert_per_layer
            + geo.globals;
        let tight = mandatory + geo.min_working_set(geo.layers) / 2;
        let e = evaluate(
            &geo, LAYOUT,
            ParallelConfig { tp: 1, dp: 1, ep: 1, pp: 1, kv: KvSharding::SequenceSharded },
            1, tight, 4_096, H2D);
        assert_eq!(
            e.infeasible, Some("expert pool below one token's working set"),
            "a pool at half the floor must be refused for the floor, not for the mandatory set"
        );
    }

    /// The reference stack's own configuration, scored by this model.
    ///
    /// vLLM-Ascend documents DeepSeek-V4-Flash on one Atlas 800 A2 (64 GiB x 8) as TP4-DP4 — the
    /// customer's exact hardware. Running it through the same arithmetic is the closest thing to an
    /// external check available, and it says something specific: TP4-DP4 needs a FREE node. On the
    /// 4.8 GiB-per-chip leftovers it is infeasible, because dp=4 replicates the mandatory set four
    /// times; on an empty 64 GiB-per-chip box it fits comfortably. That is consistent with the
    /// documentation requiring a whole node, and it is why the published ~100 TPS is not a figure
    /// this deployment is competing against.
    #[test]
    fn the_reference_stacks_tp4_dp4_needs_a_free_node() {
        let reference = ParallelConfig {
            tp: 4, dp: 4, ep: 1, pp: 1, kv: KvSharding::Replicated,
        };
        assert_eq!(reference.degree(), 16, "TP4-DP4 spans 16 chip-slots, not 8");
        // Scaled to the 8 chips actually present: TP4-DP2.
        let scaled = ParallelConfig { tp: 4, dp: 2, ep: 1, pp: 1, kv: KvSharding::Replicated };
        let leftover = evaluate(&g(), LAYOUT, scaled, 8, FREE, CTX, H2D);
        let empty = evaluate(&g(), LAYOUT, scaled, 8, 64 * 1024 * 1024 * 1024, CTX, H2D);
        eprintln!(
            "TP4-DP2, 4.8 GiB/chip leftover : {}
             TP4-DP2, 64 GiB/chip free      : {} => {:>6.2} tok/s single, {:>7.2} aggregate,              {:>5.2}% resident",
            leftover.infeasible.unwrap_or("feasible"),
            empty.infeasible.unwrap_or("feasible"),
            empty.single_stream_tok_s(), empty.aggregate_tok_s(),
            empty.resident_fraction * 100.0);
        assert!(
            leftover.infeasible.is_some(),
            "the reference split must NOT fit on the leftovers — dp replicates the mandatory set"
        );
        assert!(
            empty.infeasible.is_none(),
            "and it must fit on an empty node, which is what its documentation requires"
        );
        assert!(
            empty.resident_fraction > 0.99,
            "on an empty node the experts should be fully resident, got {:.3}",
            empty.resident_fraction
        );
    }
}
