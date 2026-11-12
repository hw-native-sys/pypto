// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The 1-query sink attention arm, pinned by its contract rather than by a count.
//!
//! `__tile_attention_sink_f32` is the intrinsic the engine's own CPU reference
//! defines: the sink is an EXTRA score with no V, every QK score is scaled by
//! 1/sqrt(D), and the weighted sum starts at scores[1]
//! (`ds4-rs-ascend/crates/engine_core/src/attn_dispatch.rs:1428-1490`).
//!
//! The S x S `__tile_attention_f32` arm cannot express it -- Q here is ONE row.
//! This test exists because the first version of the arm used `expf`, which the
//! part does not provide, and only the device compiler found it. A future
//! refactor that silently drops the sink would still emit, and would return a
//! subtly wrong attention; these assertions are the cheap local check.

use ascendc_emit::mlir_to_cpp::convert_mlir_to_cpp;

const TILE: &str = r#"module {
  llvm.func @v41_attn_decode(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>, %arg3: !llvm.ptr<1>) attributes {hacc.entry} {
    %z = llvm.mlir.constant(0 : i32) : i32
    %r = llvm.mlir.constant(1 : i32) : i32
    %s = llvm.mlir.constant(8 : i32) : i32
    %cl = llvm.mlir.constant(512 : i32) : i32
    %q = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @__tile_load_f32(%arg1, %s, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @__tile_load_f32(%arg1, %s, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_sink_f32(%z, %q, %k, %v, %arg2, %s, %cl) : (i32, i32, i32, i32, !llvm.ptr<1>, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg3, %o, %cl, %r) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

#[test]
fn one_query_sink_attention_emits_the_virtual_position_form() {
    let out = convert_mlir_to_cpp(TILE, 192 * 1024).expect("the sink arm must emit, not refuse");
    let cpp = &out.source;

    // The arm itself, and the launcher the harness names.
    assert!(
        cpp.contains("tile_attention_sink 1-query"),
        "the 1-query sink arm did not run; emitted:\n{cpp}"
    );
    assert!(
        cpp.contains("v41_attn_decode_do"),
        "the emitted kernel has no `_do` launcher; the harness runs the launcher, not the kernel"
    );

    // The sink is read from GM and enters the DENOMINATOR.
    assert!(
        cpp.contains("gm2Tensor.GetValue"),
        "the sink operand is not read from GM"
    );
    assert!(
        cpp.contains("__as_den"),
        "no denominator accumulation: the sink term was dropped"
    );
    // expf is not available on the part; the arm must use the vector Exp.
    assert!(
        cpp.contains("AscendC::Exp("),
        "the arm does not use AscendC::Exp; `expf` does not compile on the part"
    );
    assert!(
        !cpp.contains("expf("),
        "the arm still calls expf, which the device compiler rejects"
    );

    // The score scale is present; an unscaled arm would match ORACLE 23 and not
    // the engine's virtual-position reference.
    assert!(
        cpp.contains("__as_scale"),
        "the 1/sqrt(D) score scale is missing"
    );
}
