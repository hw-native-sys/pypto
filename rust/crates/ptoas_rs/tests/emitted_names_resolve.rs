// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Every name the Rust lift emits must exist in `tile_std` as a SAFE WRAPPER.
//!
//! The lift writes `tile_<op>_<dtype>` by composing an op stem with a dtype
//! suffix. Nothing checked that the composition names something real, so an
//! arm could claim an op, return `Ok`, and hand back Rust that does not
//! compile. Three such names were in the emitter when this file was written
//! and the whole suite was green:
//!
//! | emitted                   | in `tile_std`                      |
//! |---------------------------|------------------------------------|
//! | `tile_row_expand_sub_f32` | absent -- no wrapper, no intrinsic |
//! | `tile_row_expand_div_f32` | absent -- no wrapper, no intrinsic |
//! | `tile_copy_f32`           | absent -- no wrapper, no intrinsic |
//! | `tile_abs_f32`            | `__tile_abs_f32` only, no wrapper  |
//!
//! That last row is the one to read twice. The intrinsic DOES exist, so a
//! search for `abs` in `tile_std` succeeds and the arm looks backed. But the
//! lift emits safe wrappers -- `tile_*`, taking and returning `Tile<R, C, T>`
//! -- and an `extern "C"` `__tile_*` taking `u32` handles is not one. The same
//! two-layer confusion put five casts in the `pto.tcvt` arm when only three
//! wrapper pairs exist; this gate is that lesson made mechanical, because the
//! fix there was a corrected table and a corrected table drifts again.
//!
//! The check reads the emitter's OUTPUT, not its source. A source scan would
//! have to model `"{fname}_{dt}::<{r}, {c}>"` and would go stale the first
//! time an arm formatted its call differently; running the emitter and
//! grepping what came out cannot.

use ptoas_rs::parser::MODELLED_OPS;

mod shared;
use shared::{module_with, op_case, Case};

/// The safe wrappers `tile_std` exports, read from its source.
///
/// `pub fn tile_...` only. An `extern "C"` block's `pub fn __tile_...` is
/// deliberately NOT collected: see the module docs.
fn tile_std_wrappers() -> std::collections::BTreeSet<String> {
    // Read at run time rather than through `include_str!`, so an absent
    // tile_std leaves these tests ignored instead of breaking the build.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tile_std/src/tile.rs");
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read the tile API at {path}: {e}"));
    let mut out = std::collections::BTreeSet::new();
    for line in src.lines() {
        let t = line.trim_start();
        // A wrapper is declared at module level; an intrinsic is declared
        // inside `extern "C" { ... }` and is indented. Requiring column 0
        // separates the two layers without tracking brace depth, and the
        // assertion on the set's size below fails if that ever stops holding.
        if !line.starts_with("pub fn tile_") {
            continue;
        }
        let rest = &t["pub fn ".len()..];
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        out.insert(rest[..end].to_string());
    }
    assert!(
        out.len() > 50,
        "only {} wrappers found in the tile API -- the scan is broken, not the \
         emitter; refusing to certify anything from it",
        out.len()
    );
    out
}

/// Pull every `tile_xxx::<...>(...)` CALL out of emitted Rust.
///
/// A plain `name(` is not enough to go on. The lift names each emitted kernel
/// after the PTO function, and those names start with `tile_` too, so
/// accepting `tile_softmax(` reported the generated
/// `pub unsafe extern "C" fn tile_softmax(...)` as a missing wrapper in 4
/// artifacts -- a gate failing on the thing it was built to certify. Only the
/// generic-argument form is a call here, and every tile wrapper the lift
/// reaches takes const generics.
fn calls_in(rust: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = rust;
    while let Some(i) = rest.find("tile_") {
        rest = &rest[i..];
        let end = rest
            .find(|c: char| !(c.is_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let (name, tail) = rest.split_at(end);
        if tail.starts_with("::<") {
            out.push(name.to_string());
        }
        rest = tail;
    }
    out
}

/// For each modelled op, lift it and check every name that came out.
///
/// An op the lift REFUSES is fine here -- a refusal emits no name, and this
/// gate is about names, not about coverage. `op_model_coverage` and the
/// corpus sweep are what keep refusals from quietly becoming the whole
/// answer.
/// DEFERRED, NOT RETIRED -- it reads `tile_std`'s exported wrappers, and that
/// crate is not in this tree yet. Restore it when the Rust input mode arrives:
/// the defect it guards is a name the lift emits that nothing exports, which is
/// invisible from the lift's side alone.
#[test]
fn every_name_the_lift_emits_exists_as_a_safe_wrapper() {
    let wrappers = tile_std_wrappers();
    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for op in MODELLED_OPS {
        let line = match op_case(op) {
            Case::Line(l) => l,
            // The preamble's own ops (views, allocs, load/store) are exercised
            // by every other case's module, so they are covered either way.
            Case::InPreamble => String::new(),
            Case::Missing => continue,
        };
        let src = module_with(&line);
        let Ok(rust) = ptoas_rs::convert_pto_to_rust(&src) else {
            continue;
        };
        for name in calls_in(&rust) {
            checked += 1;
            if !wrappers.contains(&name) {
                bad.push(format!("`{op}` lifts to `{name}`, which tile_std does not export"));
            }
        }
    }

    // Without this the test passes when every op refuses and nothing is
    // emitted at all -- green for the reason that makes it worthless.
    assert!(
        checked >= 10,
        "only {checked} emitted names were checked; the lift is refusing \
         nearly everything, so this gate proved nothing"
    );
    assert!(bad.is_empty(), "the lift emits names that do not exist:\n  {}", bad.join("\n  "));
}

/// A dtype the table does not claim is REFUSED, not composed anyway.
///
/// The two tests above check the table's contents; neither checks the guard
/// that consults it. Found by mutation: making `wrapper` return the composed
/// name for every known stem regardless of dtype left the whole suite green,
/// because every per-op fixture is f32 and f32 is claimed for every stem the
/// table has. The gap only shows at a dtype the table omits.
///
/// `pto.tsub` is f32-only in `tile_std` (`tile_sub_f16` does not exist), so an
/// f16 subtraction is the smallest case that separates a guard from a
/// formatter.
#[test]
fn a_dtype_the_table_does_not_claim_is_refused() {
    const TH: &str = "!pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>";
    // Three DISTINCT tiles. `ins(%h0, %h0)` would be the smaller fixture and
    // would stop testing this: a tile read twice is refused for being moved
    // twice (`Tile` is not `Copy`) before any dtype is looked up, so the
    // assertion below started reporting the linearity refusal instead.
    let line = format!("pto.tsub ins(%h0, %h1 : {TH}, {TH}) outs(%h2 : {TH})");
    // `%h1` and `%h2` have to exist as f16 tiles for the module to be
    // well-typed.
    let src = module_with(&line).replace(
        "    %h0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>",
        "    %h0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>\n    \
         %h1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>\n    \
         %h2 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>",
    );
    let err = ptoas_rs::convert_pto_to_rust(&src)
        .expect_err("an f16 `pto.tsub` has no wrapper and must be refused");
    assert!(
        err.contains("tile_sub_f16"),
        "the refusal does not name the wrapper that is missing: {err}"
    );
    // And it says which dtypes WOULD have worked, so the next reader does not
    // have to go and measure `tile_std` again.
    assert!(
        err.contains("f32"),
        "the refusal does not say what tile_std does have: {err}"
    );
}

/// The MIRROR of the test above: every row of the emitter's table must exist.
///
/// The test above samples the table through whatever the per-op fixtures
/// happen to reach, which is f32 for almost every stem. So it can certify
/// `tile_add_f32` and say nothing at all about the `f16` this table also
/// claims for `add`. A claim no fixture exercises is exactly where a table
/// drifts, and the drift is silent: the emitter would compose a name, return
/// `Ok`, and the gate above would never have looked at that dtype.
/// DEFERRED, NOT RETIRED -- it reads `tile_std`'s exported wrappers, and that
/// crate is not in this tree yet. Restore it when the Rust input mode arrives:
/// the defect it guards is a name the lift emits that nothing exports, which is
/// invisible from the lift's side alone.
#[test]
fn every_wrapper_the_emitter_claims_exists_in_tile_std() {
    let wrappers = tile_std_wrappers();
    let mut bad = Vec::new();
    let mut claimed = 0usize;
    for (stem, dtypes) in ptoas_rs::WRAPPERS {
        assert!(
            !dtypes.is_empty(),
            "`{stem}` claims no dtype; drop the row instead of leaving it empty"
        );
        for dt in *dtypes {
            claimed += 1;
            let name = format!("tile_{stem}_{dt}");
            if !wrappers.contains(&name) {
                bad.push(name);
            }
        }
    }
    assert!(
        claimed >= 20,
        "the table claims only {claimed} wrappers; it has been gutted rather than corrected"
    );
    assert!(
        bad.is_empty(),
        "the emitter's WRAPPERS table claims wrappers tile_std does not export: {bad:?}"
    );
}

/// The same check over the artifacts, which is where composition shows up:
/// a per-op case exercises one arm at a time, and the fusion pass can turn a
/// chain of arms into a different one.
/// DEFERRED, NOT RETIRED -- it reads `tile_std`'s exported wrappers, and that
/// crate is not in this tree yet. Restore it when the Rust input mode arrives:
/// the defect it guards is a name the lift emits that nothing exports, which is
/// invisible from the lift's side alone.
#[test]
fn every_name_the_corpus_lift_emits_exists_as_a_safe_wrapper() {
    let wrappers = tile_std_wrappers();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let mut files = Vec::new();
    shared::collect_pto_files(&shared::corpus_root(), &mut files);
    assert!(!files.is_empty(), "no .pto artifacts found under {}", root.display());

    let mut bad = Vec::new();
    let mut lifted = 0usize;
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        let Ok(rust) = ptoas_rs::convert_pto_to_rust(&src) else {
            continue;
        };
        lifted += 1;
        for name in calls_in(&rust) {
            if !wrappers.contains(&name) {
                bad.push(format!("{}: `{name}`", f.display()));
            }
        }
    }
    assert!(lifted > 0, "no artifact lifted, so no name was checked");
    assert!(
        bad.is_empty(),
        "{} of {} artifacts lift; these emit names tile_std does not export:\n  {}",
        lifted,
        files.len(),
        bad.join("\n  ")
    );
}
