// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Matmul: shape-agreement checks happen in the emitter before rustc sees
//! the output. A well-formed matmul lowers to a single `tile_matmul_f32::<M, K, N>`;
//! a malformed one (K_lhs != M_rhs) is rejected with a clear error.

use ptoas_rs::convert_pto_to_rust;

#[test]
fn good_matmul_emits_correct_shape() {
    let pto = r#"
module {
  func.func @mm(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>, %c: !pto.ptr<f32>) {
    %l = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=32, cols=128>
    %r = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=128, cols=64>
    %acc = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=64>
    pto.tmatmul ins(%l, %r : !pto.tile_buf<loc=left, dtype=f32, rows=32, cols=128>, !pto.tile_buf<loc=right, dtype=f32, rows=128, cols=64>) outs(%acc : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=64>)
    return
  }
}
"#;
    let rust = convert_pto_to_rust(pto).expect("well-formed matmul must succeed");
    assert!(
        rust.contains("tile_matmul_f32::<32, 128, 64>"),
        "expected M=32, K=128, N=64 const generics, got:\n{rust}"
    );
}

#[test]
fn bad_matmul_is_rejected_by_emitter() {
    let pto = include_str!("fixtures/matmul_bad_shape.pto");
    let err = convert_pto_to_rust(pto).expect_err("lhs K=128, rhs M=64 must be rejected");
    // The refusal must say WHICH operands disagree, not just that two shapes
    // did: the corpus artifacts hold several matmuls each, and "tmatmul shape
    // mismatch: lhs 32x128, rhs 64x32" -- what this asserted before -- names
    // nothing a reader can find in the file. The corpus sweep now rejects any
    // refusal that quotes no token present in its input, so this assertion and
    // that one are the same rule at two scales.
    assert!(err.contains("`pto.tmatmul` K disagrees"), "got: {err}");
    for operand in ["`%l`", "`%r`"] {
        assert!(
            err.contains(operand),
            "refusal must name the operand {operand}, got: {err}"
        );
    }
}
