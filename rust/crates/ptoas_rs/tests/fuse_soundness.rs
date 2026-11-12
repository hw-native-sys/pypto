// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Fusion soundness: if an intermediate of the softmax chain is used
//! downstream, the fused op would drop a live value, so the fuse pass must
//! decline.
//!
//! What happens NEXT changed on 2026-11-01, and the change is the point.
//! These tests used to assert that the declined chain fell back to a
//! piecewise lowering, listing `tile_row_expand_sub_f32` and
//! `tile_row_expand_div_f32` among the names it should contain. Neither
//! exists in `tile_std` -- not as a wrapper, not as an intrinsic, at any
//! dtype. So the fallback these tests certified was Rust that cannot compile,
//! and they were the reason nobody noticed: the only two tests that reached
//! the un-fused path both asserted its output was correct.
//!
//! The row-expansions are liftable only inside a shape the fusion pass
//! recognises. That makes fusion load-bearing rather than an optimisation,
//! and makes a declined fusion a REFUSAL. The soundness property is
//! unchanged and is in fact easier to see now: a refusal that names
//! `pto.trowexpandsub` could only have been produced by an op that survived
//! the pass, so it is direct evidence the fusion did not fire.

use ptoas_rs::{convert_pto_to_rust, convert_pto_to_rust_raw};

/// Canonical softmax with all-dead intermediates -- fusion MUST fire.
const CLEAN: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %tv0 = pto.make_tensor_view %a, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>
    %pv0 = pto.partition_view %tv0, offsets = [%c0, %c0], sizes = [%c1, %c1024] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    %t_in = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>
    pto.tload ins(%pv0 : !pto.partition_tensor_view<1x1024xf32>) outs(%t_in : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>)
    %t_scr = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>
    %t_max = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>
    %t_sub = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>
    %t_exp = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>
    %t_sum = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>
    %t_out = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>
    pto.trowmax ins(%t_in, %t_scr : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>) outs(%t_max : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>)
    pto.trowexpandsub ins(%t_in, %t_max : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>) outs(%t_sub : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>)
    pto.texp ins(%t_sub : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>) outs(%t_exp : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>)
    pto.trowsum ins(%t_exp, %t_scr : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>) outs(%t_sum : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>)
    pto.trowexpanddiv ins(%t_exp, %t_sum : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>, !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1>) outs(%t_out : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>)
    %tv1 = pto.make_tensor_view %b, shape = [%c1, %c1024], strides = [%c1024, %c1] : !pto.tensor_view<?x?xf32>
    %pv1 = pto.partition_view %tv1, offsets = [%c0, %c0], sizes = [%c1, %c1024] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1024xf32>
    pto.tstore ins(%t_out : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>) outs(%pv1 : !pto.partition_tensor_view<1x1024xf32>)
    return
  }
}
"#;

#[test]
fn clean_chain_fuses() {
    let rust = convert_pto_to_rust(CLEAN).expect("clean softmax chain should lower");
    assert_eq!(
        rust.matches("tile_softmax_f32::<1, 1024>").count(),
        1,
        "expected fusion:\n{rust}"
    );
}

/// Same chain, but we also negate %t_exp after the chain. The fused op
/// would discard `%t_exp` -- dropping a value the kernel still needs --
/// so fusion must be suppressed and the un-fused lowering must survive.
#[test]
fn escaping_intermediate_blocks_fusion() {
    // Start from CLEAN and splice a `pto.tneg` that reads %t_exp AFTER
    // the chain, right before the final store. The extra alloc + op keeps
    // shapes legal.
    let patched = CLEAN.replace(
        "    %tv1 = pto.make_tensor_view",
        // Extra dead-use of %t_exp that escapes the fusion window:
        "    %t_leak = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>\n\
         \x20\x20\x20\x20pto.tneg ins(%t_exp : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>) outs(%t_leak : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1024>)\n\
         \x20\x20\x20\x20%tv1 = pto.make_tensor_view"
    );

    let err = convert_pto_to_rust(&patched)
        .expect_err("a chain that cannot fuse has no piecewise lowering, so it must refuse");

    // The refusal names the op that survived the pass. If fusion HAD fired
    // there would be no `RowExpandSub` left in the AST to refuse, so this
    // single assertion carries the soundness property: it can only pass when
    // the escaping `%t_exp` suppressed the fusion.
    assert!(
        err.contains("pto.trowexpandsub"),
        "expected a refusal naming the unfused row-expansion, got: {err}"
    );
    // And it points at the operands, so the message locates the line in an
    // artifact with several chains rather than merely naming the op.
    assert!(
        err.contains("%t_in") && err.contains("%t_max"),
        "refusal does not name its operands: {err}"
    );
}

/// Raw mode (fusion disabled) always emits the decomposed chain.
#[test]
fn raw_mode_never_fuses_even_on_clean() {
    // Raw mode disables the pass, so the chain stays decomposed -- and a
    // decomposed chain cannot be lifted, for the reason in the module docs.
    // The refusal is therefore what proves the flag took effect: the SAME
    // input lifts to one `tile_softmax_f32` with fusion on
    // (`clean_chain_fuses` above), so the two tests together show the flag
    // changes the result rather than being read and ignored.
    let err = convert_pto_to_rust_raw(CLEAN)
        .expect_err("raw mode leaves the chain decomposed, which has no lift");
    assert!(
        err.contains("pto.trowexpandsub"),
        "raw mode either fused anyway or refused for another reason: {err}"
    );
}
