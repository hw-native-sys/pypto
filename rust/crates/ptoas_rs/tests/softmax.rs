// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Softmax tests. S2 fuses the 5-op chain into a single `tile_softmax_fNN`;
//! the raw (un-fused) path stays available for debugging.

use ptoas_rs::{convert_pto_to_rust, convert_pto_to_rust_raw};

const SOFTMAX_PTO: &str = include_str!("fixtures/softmax.pto");

#[test]
fn fused_softmax_collapses_to_one_call() {
    let rust = convert_pto_to_rust(SOFTMAX_PTO).expect("PTO -> Rust conversion failed");

    assert!(
        rust.contains("pub unsafe extern \"C\" fn tile_softmax("),
        "missing function header\n--- output ---\n{rust}"
    );

    // After fusion, exactly one softmax call; no stray pieces of the chain.
    assert_eq!(
        rust.matches("tile_softmax_f32::<1, 1024>").count(),
        1,
        "expected fused softmax\n{rust}"
    );
    for needle in [
        "tile_reduce_max",
        "tile_reduce_sum",
        "tile_row_expand_sub",
        "tile_row_expand_div",
        "tile_exp",
    ] {
        assert!(
            !rust.contains(needle),
            "leaked decomposed op `{needle}`:\n{rust}"
        );
    }

    // Load + store still bracket the kernel.
    assert!(rust.contains("tile_load_f32::<1, 1024>"));
    assert!(rust.contains("tile_store_f32::<1, 1024>"));
}

/// With fusion off the chain stays decomposed, and it now LIFTS.
///
/// It used to refuse: the decomposed chain ends in `trowexpandsub` and
/// `trowexpanddiv`, and there was no tile-by-scalar subtract or divide to emit.
/// `tile_subs_f32` and (2026-11-10) `tile_divs_f32` close both, so raw mode
/// emits the decomposed body instead of refusing. The fused path still emits
/// ONE `tile_softmax_f32`, which is what makes the flag observable.
#[test]
fn raw_mode_lifts_the_decomposed_chain_now_that_the_scalar_ops_exist() {
    let rust = convert_pto_to_rust_raw(SOFTMAX_PTO)
        .expect("raw mode leaves the chain decomposed; both scalar ops now exist");
    assert!(
        rust.contains("tile_subs_f32") && rust.contains("tile_divs_f32"),
        "the decomposed chain did not emit its scalar subtract and divide:\n{rust}"
    );
    assert!(
        !rust.contains("tile_softmax_f32"),
        "fusion fired in raw mode, so this test no longer proves the flag works:\n{rust}"
    );
}

#[test]
fn empty_module_is_ok() {
    let rust = convert_pto_to_rust("module {\n}\n").expect("empty module should parse");
    assert!(rust.contains("use tile_std::tile"));
}

#[test]
fn unknown_type_surfaces_error() {
    let bad = r#"
module {
  func.func @bad(%a: !pto.ptr<f99>) {
    return
  }
}
"#;
    let err = convert_pto_to_rust(bad).unwrap_err();
    assert!(
        err.contains("f99"),
        "expected error to mention bad dtype, got: {err}"
    );
}
