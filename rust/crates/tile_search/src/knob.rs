// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! What a kernel lets you change, read out of the kernel itself.
//!
//! A knob is **extracted, not supplied**. The values a kernel admits follow from
//! its own shapes and from the part's bounds, so a user-supplied list is a list
//! of guesses -- and a search space acquires a value the hardware refuses the
//! moment one is typed in by hand.
//!
//! Every knob carries where its value set came from. A knob with no stated
//! provenance cannot be reviewed: a reader has no way to tell a bound that the
//! hardware enforces from a number somebody liked.

/// Why a knob's value set is what it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Read off the kernel's own extents. The captured value is in the set, so
    /// the search can always fall back to doing nothing.
    KernelShape,
    /// Bounded by a capacity the target enforces -- a buffer that must hold the
    /// live set. Exceeding it is a refusal, not a slowdown.
    Capacity(&'static str),
    /// Bounded by a correctness rule: an encoding field, an alignment, a repeat
    /// granularity. Violating it computes the wrong answer or refuses.
    Correctness(&'static str),
    /// A cost bound. Legal everywhere, slow somewhere, and invisible to any
    /// check -- which is why it ranks rather than filters.
    Cost(&'static str),
}

impl Provenance {
    pub fn label(self) -> String {
        match self {
            Provenance::KernelShape => "kernel-shape".to_string(),
            Provenance::Capacity(f) => format!("capacity:{f}"),
            Provenance::Correctness(f) => format!("correctness:{f}"),
            Provenance::Cost(f) => format!("cost:{f}"),
        }
    }
}

/// One dimension of the search space.
#[derive(Debug, Clone)]
pub struct Knob {
    /// The MLIR SSA name whose constant this knob varies, e.g. `%c`.
    pub ssa: String,
    /// The value the kernel was captured with. Always present in `values`.
    pub captured: i64,
    /// The candidate values, in the order they will be tried.
    pub values: Vec<i64>,
    pub derived_from: Provenance,
}

/// The fractal span a row-major data tile has to be a whole number of.
///
/// Not a style constant: a tile whose row is not a whole number of these spans
/// is refused, and that is what makes the feasible set non-contiguous. See the
/// crate docs for the measured sweep.
pub const FRACTAL_BYTES: i64 = 512;

/// Extract the knobs an MLIR kernel admits.
///
/// Conservative on purpose. Only an `i32` constant that is USED as a tile extent
/// in a `__tile_*` call is a knob: a constant used as an epsilon, an index or a
/// flag is not a tiling decision, and varying it would change what the kernel
/// computes rather than how it is laid out.
pub fn extract(mlir: &str) -> Vec<Knob> {
    let consts = integer_constants(mlir);
    let extents = extent_operands(mlir);

    let mut knobs: Vec<Knob> = Vec::new();
    for (ssa, value) in &consts {
        if !extents.contains(ssa) {
            continue;
        }
        // A unit extent is a row count, and reducing it is not available: there
        // is no tiling below one row. Growing it changes the work per launch
        // rather than the layout, which this module does not model, so a unit
        // extent is reported as a knob with one value -- visible, and not
        // silently dropped.
        let values = if *value <= 1 {
            vec![*value]
        } else {
            ladder(*value)
        };
        knobs.push(Knob {
            ssa: ssa.clone(),
            captured: *value,
            values,
            derived_from: Provenance::Correctness("fractal-span"),
        });
    }
    knobs.sort_by(|a, b| b.captured.cmp(&a.captured).then(a.ssa.cmp(&b.ssa)));
    knobs
}

/// The candidate values for an extent, largest first.
///
/// Divisors of the captured extent, so a tiling covers the original exactly and
/// no tail handling is invented. Filtered to those whose f32 row is a whole
/// number of fractals, because the rest are refused anyway and spending an
/// emitter run on them only slows the search -- the emitter still decides, this
/// only avoids asking a question with a known answer.
///
/// The captured value is always included even when it does not satisfy that
/// rule, because "as captured" has to stay in the space: a search that cannot
/// choose to do nothing is not a search.
pub fn ladder(captured: i64) -> Vec<i64> {
    if captured <= 1 {
        return vec![captured];
    }
    let mut out = vec![captured];
    // DIVISORS, not repeated halving. Halving cannot reach 896 from 8960
    // (8960/10), and 896 is one of the two extents the emitter accepts -- so a
    // halving ladder would report the kernel as having exactly one feasible
    // tiling when it has two. That was a real bug in the first version of this
    // function, caught by the test below.
    let mut d = 1;
    while d * d <= captured {
        if captured % d == 0 {
            for cand in [d, captured / d] {
                if cand != captured && cand * 4 % FRACTAL_BYTES == 0 {
                    out.push(cand);
                }
            }
        }
        d += 1;
    }
    out.sort_unstable_by(|a, b| b.cmp(a));
    out.dedup();
    out
}

/// `(%ssa, value)` for every `i32` constant the module defines.
fn integer_constants(mlir: &str) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    for line in mlir.lines() {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        let Some(eq) = t.find(" = llvm.mlir.constant(") else {
            continue;
        };
        if !t.ends_with(": i32") {
            continue;
        }
        let ssa = t[..eq].trim();
        if !ssa.starts_with('%') {
            continue;
        }
        let open = eq + " = llvm.mlir.constant(".len();
        let Some(close) = t[open..].find(" : i32)") else {
            continue;
        };
        if let Ok(v) = t[open..open + close].trim().parse::<i64>() {
            out.push((ssa.to_string(), v));
        }
    }
    out
}

/// Every SSA name that appears as an operand of a `__tile_*` call.
///
/// Deliberately not position-aware. The intrinsics do not agree on operand
/// order -- some carry a destination, some do not, and the emitters disambiguate
/// by operand KIND rather than by counting. A positional rule here would be
/// unsound for the same reason an arity guard is, so this over-approximates and
/// lets the emitter reject what it must.
fn extent_operands(mlir: &str) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    for line in mlir.lines() {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        let Some(at) = t.find("@__tile_") else { continue };
        // The argument list is the parenthesised group that FOLLOWS the callee
        // name, matched to its own close paren. Taking the last `)` on the line
        // instead swallows the type signature, and the final operand then keeps
        // a trailing bracket -- `%c)` rather than `%c` -- so the widest extent
        // on every kernel went unrecognised. That was a real bug here.
        let Some(open) = t[at..].find('(').map(|i| at + i) else {
            continue;
        };
        let mut depth = 0usize;
        let mut close = None;
        for (i, ch) in t[open..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { continue };
        for operand in t[open + 1..close].split(',') {
            let o = operand.trim().trim_end_matches(')');
            if o.starts_with('%') {
                out.insert(o.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = r#"
module {
  llvm.func @k(%a: !llvm.ptr<1>, %out: !llvm.ptr<1>) attributes {hacc.entry} {
    %r = llvm.mlir.constant(1 : i32) : i32
    %c = llvm.mlir.constant(8960 : i32) : i32
    %eps = llvm.mlir.constant(9.99999997E-7 : f32) : f32
    %t = llvm.call @__tile_load_f32(%a, %r, %c) : (!llvm.ptr<1>, i32, i32) -> i32
    llvm.call @__tile_store_f32(%out, %t, %r, %c) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

    #[test]
    fn only_constants_used_as_tile_extents_become_knobs() {
        let knobs = extract(KERNEL);
        let names: Vec<&str> = knobs.iter().map(|k| k.ssa.as_str()).collect();
        assert!(names.contains(&"%c"), "the column extent is a knob: {names:?}");
        assert!(names.contains(&"%r"), "the row extent is a knob: {names:?}");
        // An epsilon is not a tiling decision. Varying it would change the
        // answer, not the layout.
        assert!(!names.contains(&"%eps"), "eps must not be a knob: {names:?}");
    }

    #[test]
    fn a_unit_extent_is_reported_with_one_value_rather_than_dropped() {
        let knobs = extract(KERNEL);
        let r = knobs.iter().find(|k| k.ssa == "%r").expect("%r");
        assert_eq!(r.values, vec![1], "there is no tiling below one row");
    }

    /// The measured case, as a test. The ladder must contain BOTH accepted
    /// points from the sweep in the crate docs, and must not stop at the first.
    #[test]
    fn the_ladder_keeps_both_measured_feasible_points() {
        let l = ladder(8960);
        assert!(l.contains(&4480), "4480 emits and is missing: {l:?}");
        assert!(l.contains(&896), "896 emits and is missing: {l:?}");
        assert_eq!(l[0], 8960, "the captured value must stay in the space");
    }

    /// And the reason a threshold search fails: the ladder deliberately drops
    /// the misaligned extents, which sit BETWEEN the two feasible points.
    #[test]
    fn the_ladder_omits_extents_whose_row_is_not_whole_fractals() {
        let l = ladder(8960);
        for bad in [2240, 1120, 560] {
            assert!(
                !l.contains(&bad),
                "{bad} * 4 is not a whole number of {FRACTAL_BYTES}B fractals, \
                 and the emitter refuses it: {l:?}"
            );
        }
    }

    #[test]
    fn every_ladder_value_divides_the_captured_extent() {
        // So a tiling covers the original exactly; no invented tail handling.
        for captured in [8960, 1536, 4096, 512] {
            for v in ladder(captured) {
                assert_eq!(captured % v, 0, "{v} does not divide {captured}");
            }
        }
    }

    #[test]
    fn provenance_is_always_stated() {
        for k in extract(KERNEL) {
            let l = k.derived_from.label();
            assert!(!l.is_empty());
            assert!(l.contains(':') || l == "kernel-shape", "{l}");
        }
    }
}
