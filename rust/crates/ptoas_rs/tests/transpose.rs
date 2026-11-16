// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `pto.ttrans` -- transpose, with a workspace operand PTO threads through.
//!
//! Ported from `pto-rs-viewpoint-rebase`, which modelled this op first; the
//! refusal assertions are tightened to this branch's rule that a refusal must
//! name something the reader can find in the file.
//!
//! `tile_transpose_f32::<ROWS, COLS>` takes `Tile<ROWS, COLS>` and returns
//! `Tile<COLS, ROWS>`, so the const arguments are the SOURCE shape and the
//! declared output must be its swap. The workspace operand has no Rust
//! counterpart, but it must survive a PTO round-trip or the re-emitted line
//! stops being a `ttrans`.

use ptoas_rs::{convert_pto_to_rust, verify_roundtrip};

/// `src` is RxC; `out_r`/`out_c` let a test declare a WRONG output shape.
fn kernel(dt: &str, r: u32, c: u32, out_r: u32, out_c: u32) -> String {
    format!(
        r#"
module {{
  func.func @k(%p: !pto.ptr<f32>) {{
    %s = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype={dt}, rows={r}, cols={c}>
    %w = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype={dt}, rows={out_r}, cols={out_c}>
    %d = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype={dt}, rows={out_r}, cols={out_c}>
    pto.ttrans ins(%s, %w : !pto.tile_buf<loc=vec, dtype={dt}, rows={r}, cols={c}>, !pto.tile_buf<loc=vec, dtype={dt}, rows={out_r}, cols={out_c}>) outs(%d : !pto.tile_buf<loc=vec, dtype={dt}, rows={out_r}, cols={out_c}>)
    return
  }}
}}
"#
    )
}

#[test]
fn a_transpose_lowers_with_the_source_shape_as_const_args() {
    let rust = convert_pto_to_rust(&kernel("f32", 1, 64, 64, 1)).expect("1x64 -> 64x1 must lower");
    // ::<1, 64>, not ::<64, 1> -- the const args describe the INPUT.
    assert!(
        rust.contains("tile_transpose_f32::<1, 64>"),
        "expected tile_transpose_f32::<1, 64> (source shape), got:\n{rust}"
    );
    assert!(
        !rust.contains("tile_transpose_f32::<64, 1>"),
        "the const args are the OUTPUT shape, which inverts the call:\n{rust}"
    );
}

#[test]
fn an_output_that_is_not_the_swap_is_refused() {
    // Declared 64x2 for a 1x64 input: not a transpose of anything.
    let err = convert_pto_to_rust(&kernel("f32", 1, 64, 64, 2))
        .expect_err("a non-swapped output must not be accepted as a transpose");
    assert!(
        err.contains("64x2") && err.contains("1x64"),
        "the refusal should name both shapes, got: {err}"
    );
    assert!(
        err.contains("`%d`"),
        "the refusal must name the operand so it can be found in the file, got: {err}"
    );
}

#[test]
fn a_non_f32_transpose_is_refused_by_dtype() {
    let err = convert_pto_to_rust(&kernel("f16", 1, 64, 64, 1))
        .expect_err("tile_std has only tile_transpose_f32");
    assert!(
        err.contains("f16"),
        "the refusal should name the dtype, got: {err}"
    );
    // `__tile_transpose_f16` DOES exist -- as an intrinsic with no safe
    // wrapper. Saying so is the difference between "no such operation" and
    // "the operation exists one API layer down", which is the distinction this
    // crate has mis-stated more than once.
    assert!(
        err.contains("__tile_transpose_f16"),
        "the refusal should say the f16 intrinsic exists without a wrapper, got: {err}"
    );
}

#[test]
fn the_workspace_operand_survives_a_round_trip() {
    // The whole reason `scratch` is in the AST. If it were dropped, the
    // re-emitted line would carry one ins operand, `parse_ins2_outs1` would
    // decline it, and it would come back as `Unmodelled` -- a different AST
    // node, which is what this detects.
    let pto = kernel("f32", 1, 64, 64, 1);
    verify_roundtrip(&pto).unwrap_or_else(|e| panic!("ttrans must round-trip: {}", e.summary()));
    let out = ptoas_rs::reemit_pto(&pto).expect("re-emit");
    assert!(
        out.contains("pto.ttrans ins(%s, %w :"),
        "the workspace operand was dropped from the re-emitted line\n{out}"
    );
}
