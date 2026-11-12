// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Regions, rank-N views, and the view operand's own type.
//!
//! Three more losses of the same kind as the rest of 2026-11-01, each measured
//! against the corpus rather than guessed:
//!
//! * `pto.tload`/`pto.tstore` re-emitted the VIEW's type by rebuilding it from
//!   the TILE's shape. **35 of the corpus's 487 such lines have a view whose
//!   dims differ from its tile's** — a `16x1` view came back as `16x8`, a `1x1`
//!   view as `8x1`.
//! * a rank-N (fractal/NZ) view was refused at parse, which cost five
//!   artifacts, and the refusal before that blamed the dtype.
//! * `scf.for` was dropped entirely. 46 instances, and the dominant blocker:
//!   of the 33 artifacts still refused before this, **29 held one** and 7 held
//!   nothing else unmodelled.
//!
//! The `scf.for` ranking is also a correction to my own: I first predicted the
//! rank-N fix alone would unlock five artifacts. It unlocked none — the rank-N
//! refusal was merely the FIRST gate those five hit, and each still needed four
//! or more ops behind it. Rank the full set, never the first failure.

use ptoas_rs::{ast::PtoOp, parser::parse_module, reemit_pto};

const T: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>";

/// A view whose dims differ from the tile it loads into — the 35-line case.
const MISMATCHED: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v = pto.make_tensor_view %a, shape = [%c1, %c1], strides = [%c1, %c1] : !pto.tensor_view<?x?xf32>
    %p = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c1, %c1] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1xf32>
    %t = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>
    pto.tstore ins(%t : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>) outs(%p : !pto.partition_tensor_view<1x1xf32>)
    return
  }
}
"#;

#[test]
fn a_view_keeps_its_own_dims_not_the_tiles() {
    let out = reemit_pto(MISMATCHED).expect("re-emit");
    assert!(
        out.contains("outs(%p : !pto.partition_tensor_view<1x1xf32>)"),
        "the 1x1 view was rebuilt from the 8x1 tile\n{out}"
    );
    assert!(
        !out.contains("partition_tensor_view<8x1xf32>"),
        "the view came back carrying the tile's shape\n{out}"
    );
    assert!(out.contains(&format!("ins(%t : {T})")), "the tile type moved\n{out}");
}

/// A rank-5 fractal view keeps all five dims, all five offsets and all five
/// sizes, and is refused only by the lift.
///
/// The earlier version of this test built its rank-5 fixture by swapping only
/// the RESULT TYPE into `MISMATCHED`, leaving `offsets = [%c0, %c0]` and
/// `sizes = [%c1, %c1]` behind it -- a rank-5 view with two of each, which is
/// not a program `ptoas` can emit. It passed, because the parser also kept only
/// two. Making the offsets rank-general made this fixture fail, which is the
/// right outcome: the test had been asserting that five dims survive while
/// supplying evidence for only two, and a check this loose is what let the
/// `gmm_*` artifacts sit behind a refusal attributed to the wrong line.
const RANK5: &str = r#"
module {
  func.func @k(%a: !pto.ptr<i8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %v = pto.make_tensor_view %a, shape = [%c1, %c4, %c16, %c16, %c32], strides = [%c512, %c512, %c512, %c32, %c1] {layout = #pto.layout<nz>} : !pto.tensor_view<?x?x?x?x?xi8>
    %p = pto.partition_view %v, offsets = [%c0, %c1, %c4, %c0, %c0], sizes = [%c1, %c4, %c16, %c16, %c32] : !pto.tensor_view<?x?x?x?x?xi8> -> !pto.partition_tensor_view<1x4x16x16x32xi8>
    %t = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=i8, rows=256, cols=128>
    pto.tload ins(%p : !pto.partition_tensor_view<1x4x16x16x32xi8>) outs(%t : !pto.tile_buf<loc=mat, dtype=i8, rows=256, cols=128>)
    return
  }
}
"#;

#[test]
fn a_rank_n_view_keeps_every_dim() {
    let out = reemit_pto(RANK5).expect("a rank-5 view must round-trip");
    assert!(
        out.contains("!pto.partition_tensor_view<1x4x16x16x32xi8>"),
        "the five dims did not survive\n{out}"
    );
    // All five offsets, including the three that are not the origin -- the
    // earlier fixed pair would have written the window back at [0, 0].
    assert!(
        out.contains("offsets = [%c0, %c1, %c4, %c0, %c0]"),
        "the five offsets did not survive\n{out}"
    );
    assert!(
        out.contains("sizes = [%c1, %c4, %c16, %c16, %c32]"),
        "the five sizes did not survive\n{out}"
    );
    // The source view too: rank-5 shape, rank-5 strides, and the NZ layout
    // attribute, without which an NZ view and an ND view compare equal.
    assert!(
        out.contains("!pto.tensor_view<?x?x?x?x?xi8>"),
        "the source view came back at the wrong rank\n{out}"
    );
    assert!(
        out.contains("strides = [%c512, %c512, %c512, %c32, %c1]"),
        "the five strides did not survive\n{out}"
    );
    assert!(
        out.contains("{layout = #pto.layout<nz>}"),
        "the NZ layout attribute was dropped\n{out}"
    );
    let err = ptoas_rs::convert_pto_to_rust(RANK5).expect_err("it cannot be lifted");
    // The tensor view is refused first, for the layout, which is the honest
    // order: an NZ parent makes the partition's dims moot.
    assert!(
        err.contains("layout") && err.contains("nz"),
        "the refusal must name the layout it refused: {err}"
    );
}

/// With the layout attribute gone, the refusal moves one op down to the rank.
#[test]
fn a_rank_n_view_without_a_layout_attr_is_refused_for_its_rank() {
    let src = RANK5.replace(" {layout = #pto.layout<nz>}", "");
    reemit_pto(&src).expect("still round-trips");
    let err = ptoas_rs::convert_pto_to_rust(&src).expect_err("it cannot be lifted");
    assert!(
        err.contains("rank-5"),
        "the refusal must name the rank: {err}"
    );
}

const LOOP: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %t0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    scf.for %r = %c0 to %c32 step %c1 {
      %p = pto.partition_view %v, offsets = [%r, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
      pto.tload ins(%p : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
      pto.texp ins(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
      pto.tstore ins(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%p : !pto.partition_tensor_view<1x32xf32>)
    }
    return
  }
}
"#;

#[test]
fn a_loop_body_is_parsed_not_flattened_or_dropped() {
    let m = parse_module(LOOP).expect("parse");
    let ops = &m.funcs[0].ops;
    let fors: Vec<&PtoOp> = ops
        .iter()
        .filter(|o| matches!(o, PtoOp::For { .. }))
        .collect();
    assert_eq!(fors.len(), 1, "expected exactly one region, got {ops:#?}");
    let PtoOp::For {
        var,
        lb,
        ub,
        step,
        body,
    } = fors[0]
    else {
        unreachable!()
    };
    assert_eq!((var.0.as_str(), lb.0.as_str(), ub.0.as_str(), step.0.as_str()),
               ("%r", "%c0", "%c32", "%c1"));
    assert_eq!(body.len(), 4, "the loop body lost ops: {body:#?}");
    // Not flattened into the function either.
    assert!(
        !ops.iter().any(|o| matches!(o, PtoOp::TLoad { .. })),
        "a body op leaked into the function's own op list"
    );
}

#[test]
fn a_loop_round_trips_with_its_body_and_bounds() {
    let out = reemit_pto(LOOP).expect("re-emit");
    assert!(
        out.contains("scf.for %r = %c0 to %c32 step %c1 {"),
        "the loop header did not come back\n{out}"
    );
    for want in ["pto.tload ins(%p", "pto.texp ins(%t0", "pto.tstore ins(%t1"] {
        assert!(out.contains(want), "the body lost `{want}`\n{out}");
    }
    // The body is indented deeper than the header, so the re-emission is a
    // readable program and not a flattened one.
    let body_line = out
        .lines()
        .find(|l| l.contains("pto.texp"))
        .expect("the body's texp");
    assert!(
        body_line.starts_with("      "),
        "the body was not re-indented inside the region: {body_line:?}"
    );
    // And the induction-variable offset survives.
    assert!(
        out.contains("offsets = [%r, %c0]"),
        "the loop-variant offset was lost or folded\n{out}"
    );
}

#[test]
fn a_loop_header_missing_a_bound_is_refused() {
    for (broken, must_say) in [
        ("scf.for %r = %c0 to %c32 {", "step"),
        ("scf.for %r = %c0 step %c1 {", "to"),
        ("scf.for %r to %c32 step %c1 {", "="),
        // A literal bound rather than an SSA name: every corpus bound is a
        // name, usually derived from `pto.get_block_idx`, and accepting a
        // literal here would quietly admit a loop whose extent this parser has
        // not actually read.
        ("scf.for %r = 0 to %c32 step %c1 {", "lower bound"),
    ] {
        let src = LOOP.replace("scf.for %r = %c0 to %c32 step %c1 {", broken);
        let err = reemit_pto(&src)
            .err()
            .unwrap_or_else(|| panic!("`{broken}` was accepted as a loop header"));
        assert!(
            err.contains(must_say),
            "the refusal for `{broken}` should mention `{must_say}`: {err}"
        );
    }
}

/// The loop is refused by the lift, and the refusal points at the loop rather
/// than at the first body op the reader would then have to trace back.
#[test]
fn the_lift_refuses_the_loop_and_says_how_big_it_is() {
    let err = ptoas_rs::convert_pto_to_rust(LOOP).expect_err("a loop cannot be lifted");
    assert!(
        err.contains("scf.for %r") && err.contains("4 body op(s)"),
        "the refusal should name the loop and its size: {err}"
    );
}

const CONDITIONAL: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %t0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %first = arith.cmpi eq, %c0, %c1 : index
    scf.if %first {
      pto.texp ins(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    } else {
      pto.tneg ins(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    }
    pto.tmov ins(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    return
  }
}
"#;

/// Both branches are parsed, and the ops AFTER the conditional survive.
///
/// The second half is the part that was broken: a line-based reader that does
/// not model `scf.if` reads `} else {` as the end of its ENCLOSING body, so
/// everything after the conditional vanishes. In `gmm_blk16_nb128.pto` that
/// silently dropped a `partition_view`, its `tload` and a `tmov` out of an
/// `scf.for` body -- and the only thing that noticed was the AST-vs-text
/// cross-check in `view_geometry`.
#[test]
fn both_branches_of_a_conditional_are_parsed_and_what_follows_survives() {
    let m = parse_module(CONDITIONAL).expect("parse");
    let ops = &m.funcs[0].ops;
    let ifs: Vec<&PtoOp> = ops
        .iter()
        .filter(|o| matches!(o, PtoOp::If { .. }))
        .collect();
    assert_eq!(ifs.len(), 1, "expected exactly one scf.if, got {ops:#?}");
    let PtoOp::If {
        cond,
        then_body,
        else_body,
    } = ifs[0]
    else {
        unreachable!()
    };
    // The condition decides WHICH BRANCH RUNS, so a dropped or substituted one
    // is a different program. Nothing else in the AST records it.
    assert_eq!(cond.0, "%first", "the condition operand was not carried");
    assert_eq!(then_body.len(), 1, "then-branch: {then_body:#?}");
    let els = else_body.as_ref().expect("the else-branch must be modelled");
    assert_eq!(els.len(), 1, "else-branch: {els:#?}");
    assert!(
        matches!(then_body[0], PtoOp::UnOp { .. }) && matches!(els[0], PtoOp::UnOp { .. }),
        "the two branches hold different ops and must not be conflated"
    );
    // The `tmov` after the conditional. This is the assertion that would have
    // caught the truncation.
    assert!(
        ops.iter().any(|o| matches!(o, PtoOp::TMov { .. })),
        "the op AFTER the conditional was dropped: {ops:#?}"
    );
}

#[test]
fn a_conditional_round_trips_with_its_condition_and_both_branches() {
    let out = reemit_pto(CONDITIONAL).expect("a conditional must round-trip");
    assert!(
        out.contains("scf.if %first {"),
        "the condition did not survive re-emission\n{out}"
    );
    assert!(
        out.contains("} else {"),
        "the else-branch did not survive re-emission\n{out}"
    );
    assert!(
        out.contains("pto.texp") && out.contains("pto.tneg"),
        "a branch body was lost\n{out}"
    );
    // And the lift refuses it. On THIS fixture the refusal names `arith.cmpi`,
    // not the conditional -- the cmpi that produces the condition is itself
    // unmodelled and is reached first. Asserting "the refusal names scf.if"
    // here would have been the first-gate mistake again, in a test written to
    // avoid it, so the fixture that proves the conditional's own refusal has to
    // give the condition a MODELLED producer.
    let err = ptoas_rs::convert_pto_to_rust(CONDITIONAL).expect_err("no conditional form");
    assert!(
        err.contains("`arith.cmpi`"),
        "expected the condition's producer to be the first gate: {err}"
    );

    let modelled_cond = CONDITIONAL.replace(
        "%first = arith.cmpi eq, %c0, %c1 : index",
        "%first = arith.constant 1 : i1",
    );
    let err = ptoas_rs::convert_pto_to_rust(&modelled_cond)
        .expect_err("the tile API has no conditional form");
    assert!(
        err.contains("scf.if %first"),
        "with a modelled condition, the refusal must name the conditional: {err}"
    );
}
