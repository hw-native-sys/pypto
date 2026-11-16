// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The decomposed INT8 quantiser has no per-op lift, so it must fuse.
//!
//! `mlir_to_pto` expands the `.tile`'s one `__tile_quantize_f32_i8(x, x, %sc,
//! r, cl)` into a row broadcast divide and TWO saturating casts
//! (f32->f16, f16->s8). The pair is the part's only route to int8 -- there is
//! no f32->s8 convert -- and `tile_std` has no cast that takes a saturation
//! mode. Lowering the casts one at a time would either wrap where the artifact
//! clamps or drop the saturation silently.
//!
//! These tests pin the three things that make the fusion honest:
//!
//!   * the fused call carries the scale as a TILE (the artifact's own reading),
//!     not a scalar pulled out of it;
//!   * an intermediate that escapes the window suppresses the fusion, so it is
//!     a refusal rather than a dropped live value;
//!   * a `sat_mode` that is not ON is not fused into a clamping quantiser.

mod shared;

fn artifact() -> String {
    let path = shared::corpus_root().join("rust/bench/deepseek_v41/pto/quant_i8_v41_8x5120.pto");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The default path unrolls the 4-iteration chunk loop first; raw keeps it.
/// Both must reach the tile-scale wrapper.
#[test]
fn the_quantize_chain_lifts_to_the_tile_scale_wrapper() {
    let src = artifact();

    let fused = ptoas_rs::convert_pto_to_rust(&src).expect("the corpus artifact must lift");
    assert_eq!(
        fused
            .matches("tile_quantize_f32_i8_tile_scale::<8, 1280, 8>")
            .count(),
        4,
        "the unrolled default lift should carry one tile-scale quantise per \
         iteration:\n{fused}"
    );
    assert!(
        !fused.contains("tile_cast_"),
        "the saturating cast pair must be inside the quantiser, not emitted as \
         plain casts:\n{fused}"
    );

    let raw = ptoas_rs::convert_pto_to_rust_raw(&src).expect("raw must lift the quantiser too");
    assert_eq!(
        raw.matches("tile_quantize_f32_i8_tile_scale::<8, 1280, 8>")
            .count(),
        1,
        "raw keeps the loop, so the call appears once inside it:\n{raw}"
    );
    // The TILE reading, not the scalar one: `SC` is the padded 1x8 load's
    // storage width, and the scalar wrapper takes no such const generic.
    assert!(
        !raw.contains("tile_quantize_f32_i8::<"),
        "the scalar-scale wrapper was called; the artifact passes a tile:\n{raw}"
    );
}

/// Soundness: if the divide result is read after the chain, the fused op would
/// drop a live value, so fusion must decline and the lift must refuse at the
/// cast it cannot lower.
#[test]
fn an_escaping_intermediate_suppresses_the_fusion() {
    let src = artifact();
    let tensor = "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=1280, v_row=8, \
                  v_col=1280, blayout=row_major, slayout=none_box, fractal=512, pad=0>";
    let leak = format!(
        "      %leak = pto.alloc_tile : {tensor}\n\
         \x20     pto.tneg ins(%pto6 : {tensor}) outs(%leak : {tensor})\n"
    );
    let patched = src.replace(
        "      %pto10 = pto.make_tensor_view",
        &format!("{leak}      %pto10 = pto.make_tensor_view"),
    );
    assert_ne!(
        patched, src,
        "the splice point moved; the test patched nothing"
    );

    let err = ptoas_rs::convert_pto_to_rust_raw(&patched)
        .expect_err("a chain whose divide result escapes has no per-op lift");
    assert!(
        err.contains("sat_mode"),
        "the refusal should name the saturating cast it cannot lower, got: {err}"
    );
    assert!(
        !err.contains("tile_quantize"),
        "the fusion fired despite a live intermediate: {err}"
    );
}

/// A cast whose `sat_mode` is OFF is a wrap, not a clamp. Treating it as ON
/// would silently change the result, so it must not fuse.
#[test]
fn a_non_saturating_cast_is_not_fused_into_the_quantiser() {
    let src = artifact().replace("saturation_mode ON", "saturation_mode OFF");
    let err = ptoas_rs::convert_pto_to_rust_raw(&src)
        .expect_err("a wrapping cast is not the clamping quantiser");
    assert!(
        err.contains("sat_mode") && err.contains("OFF"),
        "the refusal should name the OFF mode, got: {err}"
    );
}
