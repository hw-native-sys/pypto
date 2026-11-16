// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Text-level fidelity for the ten ops modelled on 2026-11-01.
//!
//! An AST comparison is only sensitive to what the AST carries, so a round-trip
//! cannot prove that an attribute is carried — if the parser drops it, both
//! passes produce the same node and agree. Every verbatim-carried field below
//! therefore gets a TEXT assertion, and each is paired with a mutant that
//! drops or normalises it.
//!
//! The ops, and what each one is worth:
//!
//! | op | corpus instances | artifacts it unlocked |
//! |---|---:|---:|
//! | `pto.barrier` | 814 | 4 |
//! | `pto.tcvt` + `pto.tshls` | 188 + 56 | 1 |
//! | `pto.get_block_idx`/`_num` + `arith.index_cast` | 44 + 40 + 84 | 1 |
//! | `pto.subview` | 1 | 1 |
//! | `pto.tgatherb` | 1 | 1 |
//! | `pto.tshrs`, `pto.tands` | 96 + 128 | 0 (their files need more) |

use ptoas_rs::reemit_pto;

const T32: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>";
const TH: &str = "!pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32>";
const TIX: &str = "!pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32>";

fn wrap(body: &str) -> String {
    format!(
        r#"module {{
  func.func @k(%a: !pto.ptr<f32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %si = arith.constant 3 : i32
    %v = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t0 = pto.alloc_tile : {T32}
    %t2 = pto.alloc_tile : {T32}
    %h0 = pto.alloc_tile : {TH}
    %ix = pto.alloc_tile : {TIX}
    %ix2 = pto.alloc_tile : {TIX}
    pto.tload ins(%p : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : {T32})
{body}
    return
  }}
}}
"#
    )
}

fn roundtrips_with(body: &str, wanted: &[&str]) {
    let out = reemit_pto(&wrap(body)).unwrap_or_else(|e| panic!("re-emit failed: {e}\n{body}"));
    for w in wanted {
        assert!(
            out.contains(w),
            "the re-emission lost `{w}`\n--- emitted ---\n{out}"
        );
    }
}

/// The pipe is carried, not normalised to the one spelling the corpus uses.
///
/// All 814 corpus barriers are `<PIPE_ALL>`, so a reverse emitter that
/// hardcoded that string would pass every corpus check. `PIPE_V` below is not
/// in the corpus, which is exactly why it is here.
#[test]
fn a_barrier_keeps_its_pipe() {
    roundtrips_with("    pto.barrier <PIPE_ALL>", &["pto.barrier <PIPE_ALL>"]);
    roundtrips_with("    pto.barrier <PIPE_V>", &["pto.barrier <PIPE_V>"]);
}

#[test]
fn a_barrier_with_no_pipe_is_refused() {
    let err = reemit_pto(&wrap("    pto.barrier"))
        .expect_err("a barrier with no pipe must not be given one");
    assert!(
        err.contains("pipe"),
        "the refusal must say what was missing: {err}"
    );
}

/// `sat_mode` decides whether an out-of-range conversion saturates or wraps.
///
/// It sits INSIDE the `ins` operand list, after the operand name — a position
/// no other op in this dialect uses, and one that defeated
/// `first_ssa_before_colon` until it was taught to stop at the first space.
#[test]
fn a_cast_keeps_its_saturation_mode() {
    roundtrips_with(
        &format!(
            "    pto.tcvt ins(%t0 {{sat_mode = #pto<saturation_mode ON>}} : {T32}) \
             outs(%h0 : {TH})"
        ),
        &["sat_mode = #pto<saturation_mode ON>", "pto.tcvt ins(%t0 "],
    );
    // And the 178 instances with no attribute must not grow one.
    let out = reemit_pto(&wrap(&format!(
        "    pto.tcvt ins(%t0 : {T32}) outs(%h0 : {TH})"
    )))
    .expect("re-emit");
    assert!(
        !out.contains("sat_mode"),
        "a cast with no saturation mode was given one\n{out}"
    );
}

/// The dtypes at both ends come from the shapes map, and they differ.
#[test]
fn a_cast_keeps_both_ends() {
    let out = reemit_pto(&wrap(&format!(
        "    pto.tcvt ins(%t0 : {T32}) outs(%h0 : {TH})"
    )))
    .expect("re-emit");
    let line = out
        .lines()
        .find(|l| l.contains("pto.tcvt"))
        .expect("the tcvt line");
    assert!(
        line.contains("dtype=f32") && line.contains("dtype=f16"),
        "a f32->f16 cast was re-emitted with one dtype at both ends: {line}"
    );
}

/// `operandSegmentSizes` and `sizes` are carried verbatim.
#[test]
fn a_subview_keeps_its_attributes() {
    roundtrips_with(
        &format!(
            "    %sv = \"pto.subview\"(%t0, %c0, %c1) {{operandSegmentSizes = \
             array<i32: 1, 2, 0, 0>, sizes = [1, 16]}} : ({T32}, index, index) -> \
             !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16>"
        ),
        &[
            "operandSegmentSizes = array<i32: 1, 2, 0, 0>",
            "sizes = [1, 16]",
            "(%t0, %c0, %c1)",
            "rows=1, cols=16",
        ],
    );
}

#[test]
fn a_subview_with_no_offsets_is_refused() {
    let err = reemit_pto(&wrap(&format!(
        "    %sv = \"pto.subview\"(%t0) {{operandSegmentSizes = array<i32: 1, 0, 0, 0>, \
         sizes = [1, 16]}} : ({T32}) -> !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16>"
    )))
    .expect_err("a subview with no offset operands must be refused");
    assert!(
        err.contains("offset"),
        "the refusal must name what was missing: {err}"
    );
}

/// The gather's index operand is an i32 tile; the two `ins` types differ.
#[test]
fn a_gather_keeps_its_index_tile_dtype() {
    let out = reemit_pto(&wrap(&format!(
        "    pto.tgatherb ins(%t0, %ix : {T32}, {TIX}) outs(%t2 : {T32})"
    )))
    .expect("re-emit");
    let line = out
        .lines()
        .find(|l| l.contains("pto.tgatherb"))
        .expect("the tgatherb line");
    assert!(
        line.contains("dtype=f32") && line.contains("dtype=i32"),
        "the gather's index tile lost its i32 dtype: {line}"
    );
}

/// The shift/mask family's scalar is `i32`, not `f32`.
#[test]
fn the_shift_family_keeps_its_i32_scalar() {
    for op in ["pto.tshls", "pto.tshrs", "pto.tands"] {
        roundtrips_with(
            &format!("    {op} ins(%ix, %si : {TIX}, i32) outs(%ix2 : {TIX})"),
            &[&format!("{op} ins(%ix, %si : "), ", i32) outs("],
        );
    }
}

/// `arith.index_cast` keeps its direction.
///
/// `i64 to index` and `index to i64` are different casts, and a reverse emitter
/// that printed its two type names in the wrong order would still round-trip
/// through a parser that read them in the same wrong order.
#[test]
fn an_index_cast_keeps_its_direction() {
    roundtrips_with(
        "    %bq = \"pto.get_block_idx\"() : () -> i64\n    \
         %bqi = arith.index_cast %bq : i64 to index",
        &[
            "\"pto.get_block_idx\"() : () -> i64",
            "arith.index_cast %bq : i64 to index",
        ],
    );
}

#[test]
fn both_block_queries_keep_their_own_name() {
    let out = reemit_pto(&wrap(
        "    %bq = \"pto.get_block_idx\"() : () -> i64\n    \
         %bn = \"pto.get_block_num\"() : () -> i64",
    ))
    .expect("re-emit");
    assert!(
        out.contains("%bq = \"pto.get_block_idx\"") && out.contains("%bn = \"pto.get_block_num\""),
        "one block query was re-emitted as the other\n{out}"
    );
}

/// Each op the lift cannot express is refused naming its own reason rather
/// than a generic "unsupported".
///
/// `pto.barrier` used to be on this list, refused because
/// `__tile_pipe_barrier` is outside the `tile_std::tile` glob. It is now
/// DROPPED instead, and `tests/barrier.rs` carries the evidence and the two
/// cases where dropping is still refused. The op moved from this test to
/// `a_barrier_alone_now_lifts` below rather than being deleted from both.
#[test]
fn each_new_op_refuses_the_lift_for_its_own_reason() {
    for (body, must_name) in [(
        "    %bq = \"pto.get_block_idx\"() : () -> i64",
        "tile_std::get_block_idx",
    )] {
        let err = ptoas_rs::convert_pto_to_rust(&wrap(body))
            .expect_err("these ops have no lift arm");
        assert!(
            err.contains(must_name),
            "the refusal for `{body}` should name `{must_name}`: {err}"
        );
    }

    // The two with hardware reasons name the hardware, not a missing function.
    let err = ptoas_rs::convert_pto_to_rust(&wrap(&format!(
        "    pto.tgatherb ins(%t0, %ix : {T32}, {TIX}) outs(%t2 : {T32})"
    )))
    .expect_err("tgatherb has no lift arm");
    assert!(
        err.contains("32-byte") && err.contains("a5"),
        "the tgatherb refusal should name the per-part offset unit: {err}"
    );

    let err = ptoas_rs::convert_pto_to_rust(&wrap(&format!(
        "    %sv = \"pto.subview\"(%t0, %c0, %c1) {{operandSegmentSizes = \
         array<i32: 1, 2, 0, 0>, sizes = [1, 16]}} : ({T32}, index, index) -> \
         !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16>"
    )))
    .expect_err("subview has no lift arm");
    assert!(
        err.contains("910B2"),
        "the subview refusal should name the sub-view hazard: {err}"
    );
}

/// A module whose only unliftable op was the barrier now lifts.
///
/// The counterpart to the barrier's removal from
/// `each_new_op_refuses_the_lift_for_its_own_reason`: without this, dropping
/// the op from that list would have reduced coverage silently.
#[test]
fn a_barrier_alone_now_lifts() {
    let rust = ptoas_rs::convert_pto_to_rust(&wrap("    pto.barrier <PIPE_ALL>"))
        .expect("a barrier is dropped, not refused");
    assert!(
        !rust.contains("barrier"),
        "the lift emitted a barrier call, which `mlir_to_pto` deletes:\n{rust}"
    );
}
