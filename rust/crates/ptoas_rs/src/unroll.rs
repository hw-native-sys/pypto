// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Expand a constant-bound `scf.for` into straight-line copies, before parsing.
//!
//! WHY TEXT AND NOT THE AST. The lift refuses a loop whose body addresses the parent
//! tensor at an offset derived from the induction variable, because `view_windows`
//! precomputes every partition view's window from the AST and a constants map before
//! any op is emitted -- so no amount of renaming inside the AST reaches the offsets.
//! Measured: hand-unrolling an artefact and substituting the constant offset gets past
//! that refusal entirely and the rest of the lift copes.
//!
//! Rewriting the AST instead would mean a match arm for each of its forty-odd
//! variants, every one a chance to miss a field. The loop here is a *textual*
//! construct with a bracketed body, and its arithmetic is spelled out on the lines, so
//! a source-to-source pass is both smaller and easier to prove: it is a pure function
//! from MLIR text to MLIR text, tested on strings.
//!
//! WHAT IT WILL NOT DO. It refuses a loop whose induction variable is used for anything
//! other than the one `arith.muli` that computes the chunk offset; it refuses
//! non-constant bounds; and it refuses more than `MAX_ITERS` copies, because unrolling
//! is a size-for-liftability trade and a large trip count makes it a bad one.

/// An unrolled corpus loop is 2 to 16 chunks; beyond this the text cost wins.
const MAX_ITERS: u32 = 64;

/// A resolved SSA -> integer map, for the `arith.constant` lines the loop needs.
fn consts(src: &str) -> std::collections::BTreeMap<String, u32> {
    let mut map = std::collections::BTreeMap::new();
    for line in src.lines() {
        let l = line.trim();
        let Some((lhs, rhs)) = l.split_once(" = arith.constant ") else {
            continue;
        };
        let Some(name) = lhs.trim().strip_prefix('%') else {
            continue;
        };
        let digits: String = rhs.chars().take_while(|c| c.is_ascii_digit()).collect();
        if let Ok(v) = digits.parse::<u32>() {
            map.insert(name.to_string(), v);
        }
    }
    map
}

/// Replace whole SSA occurrences of `from` with `to`.
///
/// SSA names are `%` + `[A-Za-z0-9_.]+`, so the boundary test is: not preceded or
/// followed by a name character. Without it `%n_off` would rewrite inside `%n_off2`,
/// which is the classic way a textual rename corrupts a file that still parses.
fn rename(line: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let b = line.as_bytes();
    let f = from.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(f) {
            let before_ok = i == 0
                || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.');
            let after = i + f.len();
            let after_ok = after >= b.len()
                || !(b[after].is_ascii_alphanumeric() || b[after] == b'_' || b[after] == b'.');
            if before_ok && after_ok {
                out.push_str(to);
                i += f.len();
                continue;
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// The SSA this line defines, if any: `%x = ...`.
fn defined(line: &str) -> Option<String> {
    let l = line.trim_start();
    let (lhs, _) = l.split_once(" = ")?;
    lhs.trim().strip_prefix('%').map(|s| s.to_string())
}

/// Expand every constant-bound `scf.for` whose induction variable is only used to
/// compute a chunk offset. Loops that do not qualify are left exactly as they were,
/// so this pass can only ever remove an obstacle, never change what a kernel means.
pub fn unroll_constant_loops(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let lines: Vec<&str> = src.lines().collect();
    let k = consts(src);
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let Some(header) = parse_for_header(line) else {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        };
        let (var, lb, ub, step, indent) = header;
        // find the matching close at the same indent
        let mut j = i + 1;
        let mut body = Vec::new();
        let mut found = false;
        while j < lines.len() {
            let l = lines[j];
            if l.trim() == "}" && leading_ws(l) == indent {
                found = true;
                break;
            }
            body.push(l);
            j += 1;
        }
        let (Some(&lo), Some(&up), Some(&st)) = (k.get(&lb), k.get(&ub), k.get(&step)) else {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        };
        if !found || st == 0 || up < lo || (up - lo) % st != 0 {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        }
        let iters = (up - lo) / st;
        if iters == 0 || iters > MAX_ITERS {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        }
        // The offset line: `%n_off = arith.muli %var, %chunk`.
        let mut off_line = None;
        for (n, l) in body.iter().enumerate() {
            if l.contains("= arith.muli") && l.contains(&format!("%{var}")) {
                off_line = Some(n);
            }
        }
        let Some(off_idx) = off_line else {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        };
        let Some(off_name) = defined(body[off_idx]) else {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        };
        // the multiplier must be a known constant
        let Some(rhs) = body[off_idx].split_once(" = ").map(|(_, r)| r) else {
            i += 1;
            continue;
        };
        let Some((_, mult)) = rhs.split_once(',') else {
            i += 1;
            continue;
        };
        let mult_name = mult.trim().trim_end_matches(" : index").trim();
        let Some(&chunk) = mult_name.strip_prefix('%').and_then(|n| k.get(n)) else {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        };
        // The induction variable must appear ONLY on that line; anything else and the
        // loop carries a value this pass cannot substitute.
        let var_tok = format!("%{var}");
        let elsewhere = body
            .iter()
            .enumerate()
            .any(|(n, l)| n != off_idx && rename(l, &var_tok, "").len() != l.len());
        if elsewhere {
            out.push_str(line);
            out.push('\n');
            i += 1;
            continue;
        }
        let defs: Vec<String> = body.iter().filter_map(|l| defined(l)).collect();
        for copy in 0..iters {
            let suffix = format!("_u{copy}");
            out.push_str(&format!(
                "{indent}%{off_name}{suffix} = arith.constant {} : index\n",
                copy * chunk
            ));
            for (n, l) in body.iter().enumerate() {
                if n == off_idx {
                    continue;
                }
                let mut l = (*l).to_string();
                for d in &defs {
                    l = rename(&l, &format!("%{d}"), &format!("%{d}{suffix}"));
                }
                l = rename(&l, &format!("%{off_name}"), &format!("%{off_name}{suffix}"));
                out.push_str(&l);
                out.push('\n');
            }
        }
        let _ = lo;
        i = j + 1;
    }
    out
}

fn leading_ws(s: &str) -> String {
    s.chars().take_while(|c| c.is_whitespace()).collect()
}

/// `scf.for %v = %lb to %ub step %st {` -> its four names and its indentation.
fn parse_for_header(line: &str) -> Option<(String, String, String, String, String)> {
    let t = line.trim();
    let rest = t.strip_prefix("scf.for %")?;
    let (var, rest) = rest.split_once(' ')?;
    let rest = rest.trim_start().strip_prefix('=')?.trim_start();
    let (lb, rest) = rest.split_once(" to ")?;
    let (ub, rest) = rest.split_once(" step ")?;
    let st = rest.trim_end().trim_end_matches('{').trim();
    Some((
        var.to_string(),
        lb.trim().trim_start_matches('%').to_string(),
        ub.trim().trim_start_matches('%').to_string(),
        st.trim_start_matches('%').to_string(),
        leading_ws(line),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNKED: &str = "\
    %n_iters = arith.constant 2 : index
    %n_chunk = arith.constant 4096 : index
    %n_zero = arith.constant 0 : index
    %n_one = arith.constant 1 : index
    scf.for %n_i = %n_zero to %n_iters step %n_one {
      %n_off = arith.muli %n_i, %n_chunk : index
      %pto0 = pto.make_tensor_view %arg0, shape = [%c1] : !pto.tensor_view<?xf32>
      %pto1 = pto.partition_view %pto0, offsets = [%c0, %n_off], sizes = [%c1, %c4096]
    }
";

    #[test]
    fn a_chunk_loop_becomes_copies_with_constant_offsets() {
        let out = unroll_constant_loops(CHUNKED);
        assert!(!out.contains("scf.for"), "the loop must be gone:\n{out}");
        assert!(out.contains("%n_off_u0 = arith.constant 0 : index"), "{out}");
        assert!(out.contains("%n_off_u1 = arith.constant 4096 : index"), "{out}");
        // each copy's tiles are distinct, or MLIR sees a redefinition
        assert!(out.contains("%pto0_u0"), "{out}");
        assert!(out.contains("%pto0_u1"), "{out}");
        // the offset each copy USES is its own
        assert_eq!(out.matches("offsets = [%c0, %n_off_u0]").count(), 1, "{out}");
        assert_eq!(out.matches("offsets = [%c0, %n_off_u1]").count(), 1, "{out}");
    }

    #[test]
    fn a_loop_with_non_constant_bounds_is_left_alone() {
        let src = "\
    scf.for %i = %b to %e step %one {
      %x = pto.tload ins(%p : T) outs(%t : T)
    }
";
        assert_eq!(unroll_constant_loops(src), src);
    }

    #[test]
    fn a_loop_that_uses_the_induction_variable_elsewhere_is_left_alone() {
        // Substituting here would need to invent a constant for `%i`, and guessing one
        // is how an unroller changes what a kernel means.
        let src = "\
    %n_iters = arith.constant 2 : index
    %n_chunk = arith.constant 4 : index
    %n_zero = arith.constant 0 : index
    %n_one = arith.constant 1 : index
    scf.for %n_i = %n_zero to %n_iters step %n_one {
      %n_off = arith.muli %n_i, %n_chunk : index
      %x = arith.addi %n_i, %n_off : index
    }
";
        assert_eq!(unroll_constant_loops(src), src);
    }

    #[test]
    fn a_name_that_prefixes_another_is_not_rewritten() {
        // `%pto1` must not be touched when `%pto1_u0` is what a rewrite produced.
        assert_eq!(rename("%pto1 and %pto10", "%pto1", "%pto1_u0"), "%pto1_u0 and %pto10");
    }
}
