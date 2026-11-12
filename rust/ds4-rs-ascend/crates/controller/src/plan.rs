// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Turn a memory profile + a user mode into a deployment plan, using the SAME chooser the
//! engine consumes (`engine_ascend::parallel::choose`) — the CLI does not invent a second policy.

use crate::cli::{DeployOpts, Mode};
use crate::memprofile::ChipMem;
use anyhow::{bail, Result};
use engine_ascend::parallel::{self, Objective};
use engine_ascend::placement::{measured, ExpertLayout, Geometry};

/// A rendered plan, ready to print.
pub struct Plan {
    pub chips: usize,
    pub min_free_gib: f64,
    pub busiest_pct: f64,
    pub mode: Mode,
    pub layout: ExpertLayout,
    pub max_ctx: u32,
    pub split_label: String,
    pub split_source: String,
    pub single_stream_tok_s: f64,
    pub aggregate_tok_s: f64,
    pub per_stage: Vec<String>,
}

/// A user mode is an objective plus, for `balanced`, a nudge toward a mixed split. The chooser's
/// objective enum is where the real policy lives; this is only the surface mapping.
fn objective_of(mode: Mode) -> Objective {
    match mode {
        Mode::Latency => Objective::SingleStreamLatency,
        Mode::Throughput => Objective::AggregateThroughput,
        // Balanced wants low per-user latency AND several users — the measured mixes (ep4 pp2)
        // beat both poles for that, and they rank highest under the latency objective once PP
        // is allowed to multiply the aggregate. So balanced = latency objective; the ranking
        // itself surfaces the mix.
        Mode::Balanced => Objective::SingleStreamLatency,
        Mode::MaxContext => Objective::MaxContext,
    }
}

pub fn build_plan(opts: &DeployOpts, chips_mem: &[ChipMem]) -> Result<Plan> {
    if chips_mem.is_empty() {
        bail!("no chips in the memory profile — cannot plan");
    }
    let layout = if opts.four_bit {
        ExpertLayout::PackedNibbleBlockF32
    } else {
        ExpertLayout::HostExpandedInt8BlockF32
    };
    let g = Geometry::ds4_flash(layout);

    // Feasibility is decided at the TIGHTEST chip, minus a headroom reserve — splits assign
    // near-uniform work, so the fullest chip is the one that OOMs. Same rule as the engine's
    // DecodeRunner::choose.
    const RESERVE_BYTES: u64 = 1 << 30;
    let min_free = chips_mem
        .iter()
        .map(|c| c.free_bytes.saturating_sub(RESERVE_BYTES))
        .min()
        .unwrap_or(0);
    let busiest = chips_mem
        .iter()
        .map(|c| c.busy_fraction())
        .fold(0.0f64, f64::max);

    // An explicit pin routes through the same DS4_PARALLEL mechanism the chooser reads, so the
    // pin and the ranking cannot diverge. Set it for the duration of the call only.
    let restore = std::env::var(parallel::PARALLEL_ENV).ok();
    if let Some(pin) = &opts.parallel_pin {
        std::env::set_var(parallel::PARALLEL_ENV, pin);
    }
    let chosen = parallel::choose(
        &g,
        layout,
        objective_of(opts.mode),
        chips_mem.len() as u32,
        min_free,
        opts.max_ctx,
        measured::H2D_PINNED_GBPS,
    );
    // Restore the operator's environment exactly.
    match restore {
        Some(v) => std::env::set_var(parallel::PARALLEL_ENV, v),
        None => std::env::remove_var(parallel::PARALLEL_ENV),
    }
    let chosen = chosen.map_err(|e| anyhow::anyhow!("{e}"))?;

    // A compact per-stage view of the assignment: which chips run which layers.
    let mut per_stage = Vec::new();
    for a in &chosen.assignment {
        per_stage.push(format!(
            "chip {:>2}: layers {}..{}  (replica {}, tp{}, ep{}{})",
            a.device,
            a.first_layer,
            a.last_layer,
            a.replica,
            a.tp_rank,
            a.ep_rank,
            a.kv_shard.map(|s| format!(", kv-shard {s}")).unwrap_or_default(),
        ));
    }

    Ok(Plan {
        chips: chips_mem.len(),
        min_free_gib: min_free as f64 / (1u64 << 30) as f64,
        busiest_pct: busiest * 100.0,
        mode: opts.mode,
        layout,
        max_ctx: opts.max_ctx,
        split_label: chosen.config.label(),
        split_source: format!("{:?}", chosen.source),
        single_stream_tok_s: chosen.single_stream_tok_s,
        aggregate_tok_s: chosen.aggregate_tok_s,
        per_stage,
    })
}

pub fn render(p: &Plan) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "DeepSeek-V4-Flash deployment plan\n\
         =================================\n\
         Box:        {} chip(s), tightest free {:.1} GiB, busiest {:.0}% used\n\
         Mode:       {}  (expert layout: {:?})\n\
         Context:    {} tokens\n\
         Split:      {}   [{}]\n\
         Projected:  {:.2} tok/s single-stream, {:.1} tok/s aggregate\n\
         \n\
         Placement:\n",
        p.chips,
        p.min_free_gib,
        p.busiest_pct,
        p.mode.as_str(),
        p.layout,
        p.max_ctx,
        p.split_label,
        p.split_source,
        p.single_stream_tok_s,
        p.aggregate_tok_s,
    ));
    for line in &p.per_stage {
        s.push_str("  ");
        s.push_str(line);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memprofile::ChipMem;

    // build_plan reads/writes DS4_PARALLEL (process-global), so plan tests must not run
    // concurrently or one test's pin leaks into another. Same discipline the engine's
    // env-touching tests use.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn box_of(n: usize, free_gib: u64) -> Vec<ChipMem> {
        (0..n)
            .map(|i| ChipMem {
                id: i as u32,
                free_bytes: free_gib << 30,
                total_bytes: 64 << 30,
            })
            .collect()
    }

    #[test]
    fn a_full_node_plans_under_every_mode() {
        let _g = ENV_LOCK.lock().unwrap();
        // 8x ~60 GiB free — the reference node. Every mode must yield a feasible split.
        let mem = box_of(8, 60);
        for mode in [Mode::Latency, Mode::Throughput, Mode::Balanced, Mode::MaxContext] {
            let opts = DeployOpts { mode, max_ctx: 8192, ..Default::default() };
            let p = build_plan(&opts, &mem).expect("a full node must admit a split");
            assert_eq!(p.chips, 8);
            assert!(p.single_stream_tok_s > 0.0, "{mode:?}");
            assert!(!p.per_stage.is_empty());
        }
    }

    #[test]
    fn an_infeasible_pin_is_refused_with_a_reason() {
        let _g = ENV_LOCK.lock().unwrap();
        let mem = box_of(8, 60);
        let opts = DeployOpts {
            parallel_pin: Some("tp1dp1ep1pp1".to_string()), // 1 chip can't hold the model
            ..Default::default()
        };
        assert!(build_plan(&opts, &mem).is_err());
    }

    #[test]
    fn latency_and_throughput_can_differ() {
        let _g = ENV_LOCK.lock().unwrap();
        let mem = box_of(8, 60);
        let lat = build_plan(
            &DeployOpts { mode: Mode::Latency, ..Default::default() },
            &mem,
        )
        .unwrap();
        let thr = build_plan(
            &DeployOpts { mode: Mode::Throughput, ..Default::default() },
            &mem,
        )
        .unwrap();
        // Not asserting a specific split (that is the chooser's business and it is tested
        // there) — only that the mode reaches it and both render.
        assert!(!lat.split_label.is_empty() && !thr.split_label.is_empty());
    }
}
