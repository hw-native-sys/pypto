// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The kernel-tuning step.
//!
//! `profile` sweeps tile-rs launch/tiling knobs on THIS silicon and writes a tuning file the
//! deploy step then loads, so an end-user gets kernels tuned for their exact box rather than the
//! builds describe what it would do and stop, so the feature is discoverable without being free.
//!
//! What it tunes is the knob space the campaign already established is real and bounded — the
//! blockDim per kernel class (48 for row-partitioned matvec/attention, 1 for the reductions and
//! in-place writers) and the attention S-block. The sweep here records the shape of that search
//! and the calibrated defaults; wiring each candidate to an on-device timing run is the part
//! on-box run fills in.

use crate::cli::{Mode, ProfileOpts};
use anyhow::Result;

/// One knob and the values worth trying for it.
struct Knob {
    name: &'static str,
    candidates: &'static [u32],
    default: u32,
    why: &'static str,
}

/// The bounded, calibrated knob space — from the measured campaign, not guessed.
const KNOBS: &[Knob] = &[
    Knob {
        name: "matvec_block_dim",
        candidates: &[40, 48],
        default: 48,
        why: "row-partitioned; 48 = usable AIV count, 56 measured worse, must be <=48 or a \
              multiple of 48",
    },
    Knob {
        name: "attn_block_dim",
        candidates: &[40, 48],
        default: 48,
        why: "batched-head attention partitions cleanly across cores",
    },
    Knob {
        name: "reduction_block_dim",
        candidates: &[1],
        default: 1,
        why: "rms_norm/rope/hc are reductions or in-place writers — extra cores RACE, so 1 is \
              not a default to sweep but a correctness constraint",
    },
    Knob {
        name: "attn_s_block",
        candidates: &[8, 16, 32, 64],
        default: 8,
        why: "flash S-block; nothing in the working set scales with S once scores are \
              recomputed, so the largest that fits the 192 KiB UB wins — box-dependent at D=512",
    },
];

/// Describe the sweep this box would run — always available, so the capability is discoverable.
pub fn describe(opts: &ProfileOpts) -> String {
    let mut s = format!(
        "profile plan for {} (mode {}, budget {}s)\n\
         the sweep would explore, on THIS silicon:\n",
        opts.model,
        opts.mode.as_str(),
        opts.budget_s,
    );
    for k in KNOBS {
        s.push_str(&format!(
            "  {:<20} candidates {:?}  default {}\n      {}\n",
            k.name, k.candidates, k.default, k.why
        ));
    }
    let combos: usize = KNOBS
        .iter()
        .map(|k| k.candidates.len().max(1))
        .product();
    s.push_str(&format!(
        "  -> {} kernel configurations, each timed on-device best-of-N with a correctness \
         gate at every point (a knob that changes tiling can change what the kernel computes).\n",
        combos
    ));
    s
}

/// Run the sweep. Off-box or without the ascend feature, it writes the calibrated
/// DEFAULTS as the tuning file — a valid, correct config — and says the on-device timing pass
/// was skipped, rather than fabricating measurements.
pub fn run(opts: &ProfileOpts) -> Result<(String, String)> {
    let mode_bias = match opts.mode {
        // A latency box wants every kernel partitioned wide; a throughput box the same, since
        // the knobs are correctness-and-occupancy, not latency-vs-throughput trades. The mode
        // is recorded for the deploy step, not used to pick different knobs here.
        Mode::Latency | Mode::Balanced | Mode::Throughput | Mode::MaxContext => opts.mode.as_str(),
    };
    let mut json = String::from("{\n");
    json.push_str(&format!("  \"model\": \"{}\",\n", opts.model));
    json.push_str(&format!("  \"mode\": \"{}\",\n", mode_bias));
    json.push_str("  \"tuned_on_device\": false,\n");
    json.push_str("  \"knobs\": {\n");
    for (i, k) in KNOBS.iter().enumerate() {
        let comma = if i + 1 < KNOBS.len() { "," } else { "" };
        json.push_str(&format!("    \"{}\": {}{}\n", k.name, k.default, comma));
    }
    json.push_str("  }\n}\n");

    let out = opts.out.clone().unwrap_or_else(|| "ds4-tune.json".to_string());
    std::fs::write(&out, &json)
        .map_err(|e| anyhow::anyhow!("writing tuning file {out}: {e}"))?;
    let msg = format!(
        "wrote calibrated defaults to {out}. On-device timing sweep is available in the \
         ascend-feature build on the box — this run recorded the correct defaults without \
         fabricating measurements."
    );
    // Return the config JSON too, so the caller can register it as a `profiled` set in the
    // kernel registry (decision 1/5): the tune output IS the shareable config.
    Ok((msg, json))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sweep can be described without running it: `describe` is pure, so a
    /// user can see what `profile` would tune before spending a box on it.
    #[test]
    fn the_sweep_can_be_described_before_it_runs() {
        let opts = ProfileOpts {
            model: "ds4".into(),
            mode: Mode::Balanced,
            budget_s: 10,
            out: None,
        };
        assert!(!describe(&opts).is_empty());
    }

    #[test]
    fn describe_is_always_available_and_names_the_space() {
        let d = describe(&ProfileOpts {
            model: "ds4".into(),
            mode: Mode::Balanced,
            budget_s: 60,
            out: None,
        });
        assert!(d.contains("matvec_block_dim") && d.contains("attn_s_block"));
        assert!(d.contains("configurations"));
    }

    #[test]
    fn run_writes_defaults_and_is_honest_about_not_measuring() {
        let out = std::env::temp_dir().join(format!("ds4tune-{}.json", std::process::id()));
        let (msg, cfg) = run(
            &ProfileOpts {
                model: "ds4".into(),
                mode: Mode::Latency,
                budget_s: 10,
                out: Some(out.to_string_lossy().to_string()),
            },
        )
        .unwrap();
        assert!(msg.contains("without fabricating"));
        assert!(cfg.contains("matvec_block_dim"), "run returns the config json");
        let written = std::fs::read_to_string(&out).unwrap();
        assert!(written.contains("\"tuned_on_device\": false"));
        assert!(written.contains("matvec_block_dim"));
        std::fs::remove_file(&out).ok();
    }
}
