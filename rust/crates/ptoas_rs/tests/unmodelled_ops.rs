// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `PtoOp::Unmodelled`: carried through PTO verbatim, refused by the lift.
//!
//! Both tests here exist because a mutant survived the rest of the suite.
//!
//! The corpus sweep asserts that a file holding an unmodelled op is refused by
//! the lift -- and that assertion turned out to be VACUOUS. Deleting the
//! `PtoOp::Unmodelled` arm from `emitter::emit_op` entirely, so an unmodelled op
//! emits nothing and the lift succeeds, left the whole suite green. Every one of
//! the 26 corpus files that holds an unmodelled op ALSO holds a non-origin
//! partition view, an `scf.for`, a cube `loc=` tile or a half-present dtype, and
//! one of those refuses first. The sweep was reading a refusal it had not
//! caused: the first gate a candidate hits is not the reason it fails, and that
//! applies to a gate's own evidence too.
//!
//! So the subject here is a module whose ONLY problem is the unmodelled op.

use ptoas_rs::{convert_pto_to_rust, convert_pto_to_tile_safe, op_model_coverage, reemit_pto};

/// A module that lifts cleanly, with `extra` spliced in before the store.
fn module(extra: &str) -> String {
    const T: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>";
    format!(
        r#"
module {{
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v0 = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p0 = pto.partition_view %v0, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t0 = pto.alloc_tile : {T}
    %t1 = pto.alloc_tile : {T}
    pto.tload ins(%p0 : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : {T})
    pto.texp ins(%t0 : {T}) outs(%t1 : {T})
{extra}
    %v1 = pto.make_tensor_view %b, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p1 = pto.partition_view %v1, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    pto.tstore ins(%t1 : {T}) outs(%p1 : !pto.partition_tensor_view<1x32xf32>)
    return
  }}
}}
"#
    )
}

/// `pto.trowargmax` -- 96 corpus instances, no arm here, and nothing in this module
/// is wrong except its presence.
const UNMODELLED: &str = "    pto.trowargmax ins(%t0, %t1 : \
    !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>, \
    !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%t1 : \
    !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)";

#[test]
fn the_control_module_lifts() {
    // Without this the test below proves nothing: a refusal is only attributable
    // to the unmodelled op if the same module without it succeeds.
    convert_pto_to_rust(&module("")).expect("the control module must lift");
}

#[test]
fn an_unmodelled_op_is_refused_by_the_rust_lift_and_named() {
    let err = convert_pto_to_rust(&module(UNMODELLED))
        .expect_err("an op with no AST meaning has no tile_std lowering");
    assert!(
        err.contains("`pto.trowargmax`"),
        "the refusal must name the op it could not lower, got: {err}"
    );
    assert!(
        err.contains("not modelled"),
        "the refusal must say WHY -- an unsupported dtype and an unmodelled op are \
         different gaps: {err}"
    );
}

/// The checked lift refuses it too -- at its TEXT PRE-SCREEN, not in its
/// emitter.
///
/// Worth stating precisely, because a mutant made the distinction matter:
/// deleting the `PtoOp::Unmodelled` arm from `emitter_tile` leaves the whole
/// suite green. That is not a hole, it is redundancy -- the pre-screen catches
/// the line first -- but a test that claimed to exercise the emitter arm while
/// actually exercising the pre-screen would be the same misattribution this
/// crate keeps paying for. So the refusal's own wording is asserted.
#[test]
fn an_unmodelled_op_is_refused_by_the_checked_lift_at_its_prescreen() {
    let err = convert_pto_to_tile_safe(&module(UNMODELLED))
        .expect_err("the checked lift refuses strictly more, not less");
    assert!(
        err.contains("`pto.trowargmax`"),
        "the refusal must name the op, got: {err}"
    );
    assert!(
        err.contains("outside the safe PTO lift subset"),
        "this is the pre-screen's refusal; if it has become the emitter's, the \
         subset test below needs rereading: {err}"
    );
}

/// The pre-screen's allowlist must stay a SUBSET of what the parser models.
///
/// While it is, `emitter_tile`'s `PtoOp::Unmodelled` arm is unreachable and the
/// pre-screen is the only gate that matters. Widen the allowlist by one row past
/// `MODELLED_OPS` and that inverts silently: the pre-screen would admit a line
/// the parser turns into `Unmodelled`, and the emitter arm becomes the only
/// thing standing between it and a lift that emits nothing for it. This repo has
/// already paid for that exact shape once -- an allowlist row that un-refused
/// spellings an exact-match dispatch did not handle.
///
/// The allowlist is READ from the source rather than restated here, for the same
/// reason the published-binary test reads the workflow: a copy drifts silently.
#[test]
fn the_checked_lifts_prescreen_is_a_subset_of_what_the_parser_models() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/emitter_tile.rs"),
    )
    .expect("read emitter_tile.rs");
    let marker = "if !matches!(";
    let at = src.find(marker).expect("the pre-screen's `matches!` allowlist");
    let close = src[at..]
        .find("\n            ) {")
        .expect("the allowlist's closing paren");
    let block = &src[at..at + close];

    let mut allowed: Vec<String> = Vec::new();
    let mut rest = block;
    while let Some(i) = rest.find('"') {
        rest = &rest[i + 1..];
        let Some(j) = rest.find('"') else { break };
        allowed.push(rest[..j].to_string());
        rest = &rest[j + 1..];
    }
    assert!(
        allowed.len() >= 18,
        "only {} mnemonic(s) read out of the pre-screen; this test has stopped \
         reading what it claims to read: {allowed:?}",
        allowed.len()
    );

    let modelled: Vec<&str> = ptoas_rs::MODELLED_OPS.to_vec();
    let extra: Vec<&String> = allowed
        .iter()
        .filter(|a| !modelled.contains(&a.as_str()))
        .collect();
    assert!(
        extra.is_empty(),
        "the checked lift's pre-screen admits {extra:?}, which parser::MODELLED_OPS \
         does not model; those lines now reach emitter_tile as PtoOp::Unmodelled, so \
         its refusal arm has become load-bearing and needs its own test"
    );
}

#[test]
fn an_unmodelled_op_round_trips_verbatim_and_is_counted() {
    let src = module(UNMODELLED);
    let out = reemit_pto(&src).expect("carrying the line keeps the round trip faithful");
    assert!(
        out.contains("pto.trowargmax ins(%t0, %t1 :"),
        "the unmodelled line was not re-emitted\n{out}"
    );
    // And it is counted as NOT understood, which is what stops a file passing
    // the sweep on byte passthrough.
    let (modelled, unmodelled) = op_model_coverage(&src).expect("coverage");
    assert_eq!(unmodelled, 1, "expected exactly one unmodelled op");
    assert!(modelled > 1, "the rest of the module should still be modelled");
    let (m0, u0) = op_model_coverage(&module("")).expect("coverage");
    assert_eq!(u0, 0, "the control module has nothing unmodelled");
    assert_eq!(
        modelled, m0,
        "carrying an unmodelled op must not change how many ops are MODELLED"
    );
}

/// A view whose declared rank disagrees with its operand count.
///
/// `ptoas` cannot emit this, which is why no corpus file exercises the check --
/// and why deleting it left the suite green. It is worth keeping because the
/// alternative to checking is reconciling, and reconciling would mean choosing
/// which of the three statements of the rank to believe.
#[test]
fn a_view_whose_rank_disagrees_with_its_operands_is_refused() {
    let bad = module("").replace(
        "shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>",
        "shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?x?xf32>",
    );
    let err = ptoas_rs::parser::parse_module(&bad)
        .expect_err("rank 3 with two shape operands is not a view this parser understood");
    assert!(
        err.contains("rank 3") && err.contains("2 shape"),
        "the refusal must name both counts it compared, got: {err}"
    );

    // And the other way round: three operands under a rank-2 declaration.
    let bad2 = module("").replace(
        "shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>",
        "shape = [%c1, %c1, %c32], strides = [%c32, %c32, %c1] : !pto.tensor_view<?x?xf32>",
    );
    let err2 = ptoas_rs::parser::parse_module(&bad2)
        .expect_err("rank 2 with three shape operands is not understood either");
    assert!(
        err2.contains("rank 2") && err2.contains("3 shape"),
        "the refusal must name both counts it compared, got: {err2}"
    );
}

/// The wrapper table is audited in BOTH directions, and the second one costs
/// refusals rather than wrong emissions.
///
/// `WRAPPERS` exists to stop the lift composing a `tile_<stem>_<dtype>` name
/// `tile_std` does not define — a wrong EMISSION. Its own note records being
/// measured for exactly that. Nothing checked the other direction: a stem
/// `tile_std` DOES export and the table omits, which is a wrong REFUSAL.
///
/// Measured 2026-11-09: `tile_std::tile` has 74 top-level
/// `pub fn tile_<stem>_<dtype>` wrappers against the table's 21, and the 53
/// missing include `topk`, `argmax`, `rope`, `sigmoid`, `silu`,
/// `attention_decode`, `quantize_f32`, `matmul_transposed`, `matvec`,
/// `mxfp4_unpack`, `concat`, `gather` and `embedding` — several of them the
/// names corpus artefacts refuse on.
///
/// This test asserts the gap is EXACTLY its known size, in both directions, so
/// that:
///   * the overclaim direction stays at zero, which is the safety property;
///   * the omission direction shrinks only by a deliberate edit with a verified
///     signature behind it, and GROWS loudly if `tile_std` gains a wrapper
///     nobody told the lift about.
///
/// A stem being a top-level `pub fn` is NOT sufficient to add it: the arm must
/// pass the operands that function takes, in its order, and several of the 53
/// take extra operands the composed `tile_<stem>_<dtype>(src)` call does not
/// supply. A name that exists with a mismatched signature is worse than an
/// absent one, because it compiles.
#[test]
fn the_wrapper_table_is_audited_in_both_directions() {
    use std::collections::{BTreeMap, BTreeSet};

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let src = std::fs::read_to_string(root.join("../tile_std/src/tile.rs"))
        .expect("tile_std/src/tile.rs is readable");

    // Top-level `pub fn tile_<stem>_<dtype><` only. Indentation matters: a `pub
    // fn` inside a module is not on the glob surface the lift imports.
    let mut exported: BTreeMap<String, BTreeSet<&str>> = BTreeMap::new();
    for line in src.lines() {
        let Some(rest) = line.strip_prefix("pub fn tile_") else { continue };
        let Some(name) = rest.split('<').next() else { continue };
        for dt in ["f32", "f16", "bf16", "i8"] {
            if let Some(stem) = name.strip_suffix(&format!("_{dt}")) {
                if !stem.is_empty() {
                    exported.entry(stem.to_string()).or_default().insert(dt);
                }
            }
        }
    }
    assert!(
        exported.len() > 40,
        "only {} wrappers parsed out of tile_std; the parse, not the table, is \
         what broke -- a shrinking denominator would make the gap below look closed",
        exported.len()
    );

    let listed: BTreeMap<&str, BTreeSet<&str>> = ptoas_rs::WRAPPERS
        .iter()
        .map(|(s, ds)| (*s, ds.iter().copied().collect()))
        .collect();

    // Direction 1, the safety property: never name something absent.
    let overclaimed: Vec<String> = listed
        .iter()
        .filter_map(|(s, ds)| {
            let have = exported.get(*s as &str).cloned().unwrap_or_default();
            let extra: Vec<&str> = ds.difference(&have).copied().collect();
            (!extra.is_empty()).then(|| format!("{s}: table claims {extra:?}"))
        })
        .collect();
    assert!(
        overclaimed.is_empty(),
        "the table names wrappers tile_std does not export top-level. This is the \
         direction that emits code which cannot compile: {overclaimed:?}"
    );

    // Direction 2, the cost: refusing a wrapper that exists.
    let omitted: Vec<&String> = exported.keys().filter(|s| !listed.contains_key(s.as_str())).collect();
    assert_eq!(
        omitted.len(),
        ptoas_rs::WRAPPER_TABLE_KNOWN_GAP,
        "the omission gap moved. If it SHRANK, lower WRAPPER_TABLE_KNOWN_GAP in the \
         same change that verified the signature. If it GREW, tile_std gained a \
         wrapper the lift was not told about, which is the case this test exists \
         for. Omitted now: {omitted:?}"
    );
}
