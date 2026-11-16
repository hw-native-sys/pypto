// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Try every candidate, ask the emitter, rank what it accepted.

use crate::knob::Knob;
use tile_codegen::registry::TargetRegistry;
use tile_codegen::target::EmitOpts;

/// What the emitter said about one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// It emitted. `pto_lines` is the size of what came out -- the only
    /// observable this module has without running anything.
    Accepted { pto_lines: usize },
    /// It refused, with its own message. Kept verbatim: the emitter's wording
    /// names the bound that was hit, and a summary here would lose that.
    Refused { why: String },
}

/// One point in the space.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub ssa: String,
    pub value: i64,
    pub outcome: Outcome,
}

impl Candidate {
    pub fn accepted(&self) -> bool {
        matches!(self.outcome, Outcome::Accepted { .. })
    }
}

/// The rule used to order accepted candidates.
///
/// A DECLARED MODEL, not a measurement. Nothing here runs on hardware, so this
/// cannot say which accepted tiling is faster -- only which one a stated rule
/// prefers. It is a named type rather than a comparator buried in a sort so that
/// a reader can disagree with the rule itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cost {
    /// Prefer the largest accepted extent.
    ///
    /// The reasoning, stated so it can be argued with: a larger extent means
    /// fewer tiles for the same work, so fewer launches and fewer DMA
    /// descriptors, and descriptor cost is per-descriptor and almost
    /// independent of how much each one carries. The failure mode of this rule
    /// is that the largest accepted extent may leave too little of the buffer
    /// for double buffering -- which this module cannot see, because it does not
    /// run anything.
    LargestAccepted,
}

impl Cost {
    pub fn label(self) -> &'static str {
        match self {
            Cost::LargestAccepted => "largest-accepted-extent",
        }
    }
}

/// Everything the search found, including what it rejected.
#[derive(Debug)]
pub struct SearchReport {
    pub knobs: Vec<Knob>,
    pub candidates: Vec<Candidate>,
    pub cost: Cost,
    /// The winner: `(ssa, value)`. `None` when nothing was accepted.
    pub chosen: Option<(String, i64)>,
    /// The MLIR of the winner, ready to write out.
    pub chosen_mlir: Option<String>,
    /// True when the winner is the value the kernel arrived with.
    pub unchanged: bool,
}

impl SearchReport {
    pub fn accepted(&self) -> usize {
        self.candidates.iter().filter(|c| c.accepted()).count()
    }

    /// The console report. Every candidate is listed, accepted or not: a search
    /// that printed only its winner would hide both the size of the space it
    /// looked at and the reasons the rest were refused, and those reasons are
    /// the useful output when nothing is accepted.
    pub fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        for k in &self.knobs {
            let _ = writeln!(
                s,
                "knob {} captured={} candidates={} from={}",
                k.ssa,
                k.captured,
                k.values.len(),
                k.derived_from.label()
            );
        }
        for c in &self.candidates {
            match &c.outcome {
                Outcome::Accepted { pto_lines } => {
                    let _ = writeln!(s, "  {} = {:<8} accepted ({pto_lines} lines)", c.ssa, c.value);
                }
                Outcome::Refused { why } => {
                    let first = why.lines().next().unwrap_or(why);
                    let _ = writeln!(s, "  {} = {:<8} refused: {first}", c.ssa, c.value);
                }
            }
        }
        let _ = writeln!(
            s,
            "{} of {} candidate(s) accepted; ranked by {} (a declared model, not a measurement)",
            self.accepted(),
            self.candidates.len(),
            self.cost.label()
        );
        match &self.chosen {
            Some((ssa, v)) if self.unchanged => {
                let _ = writeln!(s, "chose {ssa} = {v}, which is the captured value: nothing to change");
            }
            Some((ssa, v)) => {
                let _ = writeln!(s, "chose {ssa} = {v}");
            }
            None => {
                let _ = writeln!(
                    s,
                    "nothing was accepted. The refusals above are the result: each names \
                     the bound it hit."
                );
            }
        }
        s
    }
}

/// Which emitter decides whether a candidate is admissible.
///
/// There are two in this workspace and they do not agree, which is the whole
/// reason this is a choice rather than a constant. A kernel can satisfy the PTO
/// emitter's bounds and overflow the AscendC emitter's per-buffer budget --
/// measured over the bench, five do -- and searching against the wrong one
/// reports "nothing to change" for a kernel that is refused downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Oracle {
    /// `rspto_emit`, the PTO emitter. The default, because it is the route
    /// every other representation is derived through.
    Pto,
    /// `ascendc_emit`, the AscendC emitter. Different bounds: it divides the
    /// Unified Buffer into fixed slots, so its limit is per BUFFER rather than
    /// over the live working set.
    AscendC,
}

impl Oracle {
    /// The spelling a user passes to `--oracle`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "pto" => Some(Oracle::Pto),
            "ascendc" | "cpp" => Some(Oracle::AscendC),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Oracle::Pto => "pto",
            Oracle::AscendC => "ascendc",
        }
    }
}

/// Emit `mlir` through `oracle` and report what it said.
///
/// ⚠ The AscendC target MUST be given hardware parameters.
/// `EmitOpts::default()` leaves it with a ZERO-byte Unified Buffer, and it then
/// refuses every candidate with a message that reads like a verdict:
///
/// ```text
/// buffer 0 needs 128 elements (512 bytes), which is larger than the ENTIRE
/// 0 byte unified buffer
/// ```
///
/// Measured: the first version of this function passed the default and got
/// `0 of 8 candidate(s) accepted` on a kernel whose real bound is 18816 bytes
/// per buffer. The only tell was the literal `0` in the message. The PTO target
/// does not need this -- it carries its own arch default, which is why the
/// default worked there and hid the asymmetry.
fn probe_with(mlir: &str, oracle: Oracle) -> Outcome {
    let mut r = TargetRegistry::new();
    let (key, opts) = match oracle {
        Oracle::Pto => {
            rspto_emit::register(&mut r);
            ("pto", EmitOpts::default())
        }
        Oracle::AscendC => {
            ascendc_emit::register(&mut r);
            ("cpp", ascendc_opts())
        }
    };
    let Some(target) = r.select(key) else {
        return Outcome::Refused {
            why: format!("the {key} target did not register"),
        };
    };
    match target.emit(mlir, &opts) {
        Ok(out) => Outcome::Accepted {
            pto_lines: out.source.lines().count(),
        },
        Err(why) => Outcome::Refused { why },
    }
}

/// The hardware the AscendC oracle searches against: 910b, the same part
/// `pto-to-ascendc` emits for by default. A search is only as meaningful as the
/// part it searched against, so this is named rather than defaulted.
fn ascendc_opts() -> EmitOpts {
    let mut o = EmitOpts::default();
    o.hw = tile_codegen::target::HardwareParams::ascend_910b();
    o.sem = tile_codegen::target::TargetSemantics::ascend_arch32();
    o
}

/// Emit `mlir` and report what the PTO target said. Kept so every existing
/// caller and test keeps its meaning.
fn probe(mlir: &str) -> Outcome {
    probe_with(mlir, Oracle::Pto)
}

/// Rewrite one `i32` constant definition, leaving every use alone.
///
/// Textual, and only the definition line. Rewriting uses would mean rewriting
/// the kernel, and the point of varying a named extent is that its uses already
/// refer to it by name.
pub fn substitute(mlir: &str, ssa: &str, value: i64) -> String {
    let needle_prefix = format!("{ssa} = llvm.mlir.constant(");
    mlir.lines()
        .map(|line| {
            let t = line.trim_start();
            if t.starts_with(&needle_prefix) && t.ends_with(": i32") {
                let indent = &line[..line.len() - t.len()];
                format!("{indent}{ssa} = llvm.mlir.constant({value} : i32) : i32")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// Search one knob at a time, deciding each candidate with the emitter.
///
/// One knob at a time rather than the cross product, and the limit is honesty
/// rather than compute: with several knobs the cross product is large and this
/// module has no way to rank within it except the declared rule, so a reported
/// "best" over thousands of unmeasured points would be a stronger claim than
/// anything here supports. Searching the widest knob is a smaller claim that
/// holds.
pub fn search(mlir: &str, cost: Cost) -> SearchReport {
    search_with(mlir, cost, Oracle::Pto)
}

/// Search, deciding each candidate with `oracle`.
///
/// `search` is this at `Oracle::Pto`, which is what every existing caller and
/// test means.
pub fn search_with(mlir: &str, cost: Cost, oracle: Oracle) -> SearchReport {
    let knobs = crate::knob::extract(mlir);

    // The widest extent: the one whose tiling decision has the most room, and
    // the one the capacity bounds actually bite on.
    let target_knob = knobs
        .iter()
        .filter(|k| k.values.len() > 1)
        .max_by_key(|k| k.captured)
        .cloned();

    let mut candidates = Vec::new();
    let mut best: Option<(String, i64, String)> = None;

    if let Some(k) = &target_knob {
        for v in &k.values {
            let text = substitute(mlir, &k.ssa, *v);
            let outcome = probe_with(&text, oracle);
            let accepted = matches!(outcome, Outcome::Accepted { .. });
            candidates.push(Candidate {
                ssa: k.ssa.clone(),
                value: *v,
                outcome,
            });
            if accepted {
                let better = match (&best, cost) {
                    (None, _) => true,
                    (Some((_, b, _)), Cost::LargestAccepted) => *v > *b,
                };
                if better {
                    best = Some((k.ssa.clone(), *v, text));
                }
            }
        }
    }

    let captured = target_knob.as_ref().map(|k| k.captured);
    let unchanged = match (&best, captured) {
        (Some((_, v, _)), Some(c)) => *v == c,
        _ => false,
    };

    SearchReport {
        knobs,
        candidates,
        cost,
        chosen: best.as_ref().map(|(s, v, _)| (s.clone(), *v)),
        chosen_mlir: best.map(|(_, _, t)| t),
        unchanged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bench kernel the crate docs' sweep was measured on.
    const GATE_UP_SILU: &str = r#"
module {
  llvm.func @gate_up_silu(%a: !llvm.ptr<1>, %wg: !llvm.ptr<1>, %wu: !llvm.ptr<1>, %out: !llvm.ptr<1>) attributes {hacc.entry} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(8960 : i32) : i32
    %in0 = llvm.call @__tile_load_f32(%a, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %in1 = llvm.call @__tile_load_f32(%wg, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %in2 = llvm.call @__tile_load_f32(%wu, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    %res = llvm.call @__tile_gate_up_silu_f16(%c0, %in0, %in1, %in2, %r, %c) : (i32, i32, i32, i32, i32, i32) -> i32
    // A ROW REDUCTION, and the fixture needed one the day the emitter learned to
    // chunk: an elementwise chain at this width is no longer refused, because
    // rspto splits it into equal chunks and accepts it. A reduction accumulates
    // across the width, so it cannot be split, which is what keeps this fixture a
    // kernel the target refuses AS CAPTURED -- the premise the search exists for.
    %red = llvm.call @__tile_reduce_sum_f32(%c0, %res, %r, %c) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%out, %red, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

    #[test]
    fn substitution_rewrites_the_definition_and_not_the_uses() {
        let out = substitute(GATE_UP_SILU, "%c", 4480);
        assert!(out.contains("%c = llvm.mlir.constant(4480 : i32)"));
        assert!(!out.contains("constant(8960"));
        // Uses still refer to the name, so the kernel's shape is unchanged.
        assert_eq!(
            out.matches("%r, %c)").count(),
            GATE_UP_SILU.matches("%r, %c)").count()
        );
    }

    /// THE CENTRAL CASE. The kernel as captured is refused; the search must find
    /// a tiling the emitter accepts.
    #[test]
    fn the_search_rescues_a_kernel_the_target_refuses_as_captured() {
        assert!(
            matches!(probe(GATE_UP_SILU), Outcome::Refused { .. }),
            "the premise is that this kernel is refused as captured"
        );
        let r = search(GATE_UP_SILU, Cost::LargestAccepted);
        assert!(r.accepted() > 0, "nothing accepted:\n{}", r.render());
        assert!(!r.unchanged, "the captured value is refused, so it cannot win");
        let (ssa, v) = r.chosen.clone().expect("a winner");
        assert_eq!(ssa, "%c");
        // And the winner must itself emit.
        let text = r.chosen_mlir.expect("the winner's MLIR");
        assert!(
            matches!(probe(&text), Outcome::Accepted { .. }),
            "the chosen tiling does not emit, which makes the choice meaningless"
        );
        assert_eq!(v, 4480, "largest-accepted should pick 4480, not {v}");
    }

    /// The ladder's alignment filter is LOAD-BEARING, not decorative.
    ///
    /// This is the corrected form of a claim this crate got wrong first time. A
    /// hand sweep by repeated halving showed 4480 accepted, 2240 and 1120
    /// refused, and 896 accepted again -- so the accepted set looked
    /// non-contiguous, and the crate documentation said so. It is not: every one
    /// of those refusals is a MISALIGNED extent, and once the ladder drops them
    /// the remaining candidates are all accepted. The set is contiguous in the
    /// filtered space and ragged only in the unfiltered one.
    ///
    /// That makes the filter the thing worth testing. If the emitter stopped
    /// refusing misaligned extents, the ladder would be throwing away feasible
    /// tilings for no reason; if it started refusing aligned ones, the ladder
    /// would be proposing work that cannot run. So both directions are asserted
    /// here, against the real emitter.
    #[test]
    fn the_alignment_filter_matches_what_the_emitter_actually_refuses() {
        use crate::knob::FRACTAL_BYTES;

        // Misaligned: the emitter must refuse, which is why the ladder omits it.
        for misaligned in [2240, 1120, 560] {
            assert_ne!(misaligned * 4 % FRACTAL_BYTES, 0, "{misaligned} is aligned");
            let text = substitute(GATE_UP_SILU, "%c", misaligned);
            match probe(&text) {
                Outcome::Refused { why } => assert!(
                    why.contains("tile bound") || why.contains("fractal"),
                    "{misaligned} refused for an unexpected reason: {why}"
                ),
                Outcome::Accepted { .. } => panic!(
                    "the emitter now accepts the misaligned extent {misaligned}; the \
                     ladder is discarding feasible tilings and its filter needs \
                     revisiting"
                ),
            }
        }

        // Aligned and on the ladder: the emitter must accept, or the ladder is
        // proposing tilings that cannot run.
        let r = search(GATE_UP_SILU, Cost::LargestAccepted);
        let refused_aligned: Vec<i64> = r
            .candidates
            .iter()
            .filter(|c| !c.accepted() && c.value != 8960)
            .map(|c| c.value)
            .collect();
        assert!(
            refused_aligned.is_empty(),
            "aligned ladder values were refused: {refused_aligned:?}\n{}",
            r.render()
        );
    }

    /// A kernel the target already accepts must not be "improved" into a
    /// different one by default: doing nothing has to be reachable.
    #[test]
    fn an_already_accepted_kernel_can_be_left_alone() {
        let ok = substitute(GATE_UP_SILU, "%c", 4480);
        assert!(matches!(probe(&ok), Outcome::Accepted { .. }));
        let r = search(&ok, Cost::LargestAccepted);
        assert_eq!(r.chosen.as_ref().map(|(_, v)| *v), Some(4480));
        assert!(r.unchanged, "the captured value is accepted and largest:\n{}", r.render());
    }

    /// Every refusal is reported with the emitter's own words, because that is
    /// the output when nothing is accepted.
    #[test]
    fn refusals_carry_the_emitters_message() {
        let r = search(GATE_UP_SILU, Cost::LargestAccepted);
        let refused: Vec<&Candidate> = r.candidates.iter().filter(|c| !c.accepted()).collect();
        assert!(!refused.is_empty());
        for c in refused {
            let Outcome::Refused { why } = &c.outcome else {
                unreachable!()
            };
            assert!(
                why.contains("NPU") || why.contains("tile") || why.contains("UB"),
                "a refusal with no bound named: {why}"
            );
        }
    }

    #[test]
    fn the_report_states_that_the_ranking_is_a_model() {
        let r = search(GATE_UP_SILU, Cost::LargestAccepted);
        let text = r.render();
        assert!(text.contains("declared model"), "{text}");
        assert!(text.contains("not a measurement"), "{text}");
    }

    #[test]
    fn a_kernel_with_no_varying_knob_reports_no_candidates() {
        let flat = r#"
module {
  llvm.func @k(%a: !llvm.ptr<1>, %out: !llvm.ptr<1>) attributes {hacc.entry} {
    %r = llvm.mlir.constant(1 : i32) : i32
    %t = llvm.call @__tile_load_f32(%a, %r, %r) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @__tile_store_f32(%out, %t, %r, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;
        let r = search(flat, Cost::LargestAccepted);
        assert!(r.candidates.is_empty());
        assert!(r.chosen.is_none());
        assert!(r.render().contains("nothing was accepted"));
    }
}
