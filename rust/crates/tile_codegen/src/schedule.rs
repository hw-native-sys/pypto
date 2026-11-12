// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Portable schedule knobs as DATA — the KernelBand / P4 surface.
//!
//! A schedule is *separate from the algorithm*: it says how a loop is
//! structured (unroll factor, block sizes, layout), never what is computed.
//! Lowering must therefore be **sound for every arm in [`Schedule::legal_arms`]**:
//! same mathematical result (same sum order for float reductions), only a
//! different legal machine form.
//!
//! ## Safety contract
//!
//! 1. `Schedule::default()` is exactly today's behavior — no attribute means
//!    no change.
//! 2. An arm is only legal if every target that will honor it preserves
//!    semantics. Today that is `k_unroll`, because
//!    `emit_unrolled_k_accumulation` visits `k` in strictly increasing order
//!    with one accumulator for every factor ≥ 1 (proven in `mlir_parse`).
//! 3. Illegal schedules are **rejected**, never silently rewritten.
//! 4. The bandit may only *select among legal arms after a verified run* —
//!    never invent a cost, never cache an unmeasured answer.

/// One concrete schedule choice. `None` fields mean “backend default”.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Schedule {
    /// K-loop unroll factor for matmul-style reductions. `None` = the
    /// backend's receipted default (MSL 16, GPU `DEFAULT_K_UNROLL`).
    pub k_unroll: Option<u32>,
}

/// Upper bound on a declared `k_unroll`. Soundness allows any factor ≥ 1;
/// this is a code-size / I-cache bound so a typo (`k_unroll = 100000`) is
/// rejected rather than expanded into a million loads.
pub const MAX_K_UNROLL: u32 = 64;

/// Powers-of-two arms the bandit proposes for `k_unroll`. Non-pow2 factors
/// are *sound* (same sum order) but not worth arms until a measurement says
/// so; legal_arms is the pruning surface, not a soundness whitelist beyond
/// the `validate` bounds.
pub const K_UNROLL_ARMS: &[u32] = &[1, 2, 4, 8, 16, 32, 64];

impl Schedule {
    pub const DEFAULT: Schedule = Schedule { k_unroll: None };

    /// Reject a schedule that no lowering could honor soundly (or at all).
    ///
    /// Zero is rejected rather than clamped: a schedule that “means 1” after
    /// a silent max() is how a knob becomes a lie. Oversized factors are
    /// rejected for code size, not for numerics.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(f) = self.k_unroll {
            if f == 0 {
                return Err(
                    "schedule k_unroll = 0: a factor of 0 does not unroll; \
                     use 1 for the scalar loop (or omit k_unroll for the \
                     backend default)"
                        .into(),
                );
            }
            if f > MAX_K_UNROLL {
                return Err(format!(
                    "schedule k_unroll = {f} exceeds MAX_K_UNROLL = {MAX_K_UNROLL}; \
                     an unrolled-by-{f} K loop is a code-size hazard, not a \
                     tuning choice"
                ));
            }
        }
        Ok(())
    }

    /// Stable arm id for bandit state keys (`"k_unroll=16"`, `"default"`).
    pub fn arm_key(&self) -> String {
        match self.k_unroll {
            None => "default".into(),
            Some(f) => format!("k_unroll={f}"),
        }
    }

    /// Parse `k_unroll=16` / `k_unroll = 16` (and aliases used by attrs).
    pub fn parse_kv(s: &str) -> Result<Schedule, String> {
        let mut sch = Schedule::DEFAULT;
        for part in s.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            let (k, v) = part
                .split_once('=')
                .ok_or_else(|| format!("schedule field `{part}` needs key=value"))?;
            match k.trim() {
                "k_unroll" => {
                    let f: u32 = v.trim().parse().map_err(|_| {
                        format!("schedule k_unroll `{}` is not an integer", v.trim())
                    })?;
                    sch.k_unroll = Some(f);
                }
                other => {
                    return Err(format!(
                        "unknown schedule field `{other}`; this build only \
                         implements k_unroll (block/layout/vec are not yet \
                         honored by lowering — refusing rather than ignoring)"
                    ));
                }
            }
        }
        sch.validate()?;
        Ok(sch)
    }

    /// The arm set a bandit may propose for this base schedule.
    ///
    /// Proven-safe pruning: only arms that `validate` accepts, and only knobs
    /// at least one emitter already honors via `emit_unrolled_k_accumulation`
    /// (same sum order → same float result).
    pub fn legal_arms(&self) -> Vec<Schedule> {
        let mut arms = vec![self.clone()];
        if self.k_unroll.is_none() {
            for &f in K_UNROLL_ARMS {
                let a = Schedule { k_unroll: Some(f) };
                if a.validate().is_ok() {
                    arms.push(a);
                }
            }
        }
        arms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty_and_equals_no_override() {
        assert_eq!(Schedule::DEFAULT.k_unroll, None);
        assert_eq!(Schedule::default(), Schedule::DEFAULT);
        assert_eq!(Schedule::DEFAULT.arm_key(), "default");
    }

    #[test]
    fn zero_k_unroll_is_rejected_not_clamped() {
        let e = Schedule { k_unroll: Some(0) }.validate().unwrap_err();
        assert!(e.contains("k_unroll = 0"), "{e}");
        assert!(Schedule::parse_kv("k_unroll=0").is_err());
    }

    #[test]
    fn oversized_k_unroll_is_rejected() {
        let e = Schedule { k_unroll: Some(MAX_K_UNROLL + 1) }.validate().unwrap_err();
        assert!(e.contains("MAX_K_UNROLL"), "{e}");
    }

    #[test]
    fn unknown_schedule_fields_are_refused_rather_than_ignored() {
        let e = Schedule::parse_kv("tile_size=64").unwrap_err();
        assert!(e.contains("unknown schedule field"), "{e}");
        assert!(e.contains("k_unroll"), "{e}");
    }

    #[test]
    fn legal_arms_are_a_subset_of_the_default_plus_valid_k_unrolls() {
        let arms = Schedule::DEFAULT.legal_arms();
        // default first — the no-change arm is always present
        assert_eq!(arms[0], Schedule::DEFAULT);
        assert!(arms.iter().all(|a| a.validate().is_ok()));
        assert!(arms.iter().any(|a| a.k_unroll == Some(16)));
        // a schedule that already pins k_unroll does not re-expand
        let pinned = Schedule { k_unroll: Some(8) };
        let expanded = pinned.legal_arms();
        assert_eq!(expanded, vec![pinned]);
    }

    #[test]
    fn parse_kv_accepts_spaces_and_trailing_comma() {
        let s = Schedule::parse_kv(" k_unroll = 32 , ").unwrap();
        assert_eq!(s.k_unroll, Some(32));
        assert_eq!(s.arm_key(), "k_unroll=32");
    }
}
