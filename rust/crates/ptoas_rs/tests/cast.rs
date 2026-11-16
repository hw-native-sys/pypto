// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `pto.tcvt` -- dtype conversion between same-shaped tiles.
//!
//! Ported from `pto-rs-viewpoint-rebase`, whose `sat_mode` reasoning this branch
//! adopted, with the assertions tightened: there the saturating case was allowed
//! to be either refused or lowered-without-the-cast, because that branch did not
//! carry `sat_mode` in its AST and could not promise which. This branch carries
//! it, so the refusal is exact and is asserted exactly.
//!
//! Four things need to hold, each a separate failure the corpus can produce: a
//! supported pair lowers to the matching `tile_cast_*`; an unsupported pair is
//! refused by NAME rather than lowered to a nearby cast; a pair that exists only
//! as a `__tile_cast_*` INTRINSIC, with no safe wrapper, is refused and says so;
//! and an operand carrying a `sat_mode` attribute is not reinterpreted as a
//! plain cast, because saturation decides what an out-of-range value becomes.

use ptoas_rs::convert_pto_to_rust;

fn kernel(src_dt: &str, dst_dt: &str, attrs: &str) -> String {
    format!(
        r#"
module {{
  func.func @k(%p: !pto.ptr<f32>) {{
    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype={src_dt}, rows=16, cols=32>
    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype={dst_dt}, rows=16, cols=32>
    pto.tcvt ins(%a{attrs} : !pto.tile_buf<loc=vec, dtype={src_dt}, rows=16, cols=32>) outs(%b : !pto.tile_buf<loc=vec, dtype={dst_dt}, rows=16, cols=32>)
    return
  }}
}}
"#
    )
}

#[test]
fn a_supported_pair_lowers_to_the_matching_cast() {
    for (s, d, want) in [
        ("f16", "f32", "tile_cast_f16_f32::<16, 32>"),
        ("f32", "f16", "tile_cast_f32_f16::<16, 32>"),
        ("i8", "f32", "tile_cast_i8_f32::<16, 32>"),
    ] {
        let rust = convert_pto_to_rust(&kernel(s, d, ""))
            .unwrap_or_else(|e| panic!("{s} -> {d} must lower: {e}"));
        assert!(rust.contains(want), "expected {want}, got:\n{rust}");
    }
}

#[test]
fn an_unsupported_pair_is_refused_and_the_message_names_both_dtypes() {
    // i8 -> f16 is 22 occurrences in the repo corpus and has no tile_std
    // intrinsic at all; the nearest one is i8 -> f32, a DIFFERENT conversion.
    let err = convert_pto_to_rust(&kernel("i8", "f16", ""))
        .expect_err("i8 -> f16 has no tile_std cast and must be refused");
    assert!(
        err.contains("i8") && err.contains("f16"),
        "the refusal must name both dtypes, got: {err}"
    );
    assert!(
        !err.contains("not modelled"),
        "this is a known op with an unsupported dtype pair, not an unmodelled op: {err}"
    );
}

/// bf16 is the case that made a five-pair table wrong.
///
/// `__tile_cast_bf16_f32` and `__tile_cast_f32_bf16` both exist, so counting
/// casts in `tile.rs` gives five pairs -- and this branch's refusal message said
/// five. But the lift emits the SAFE WRAPPERS, and there is no
/// `tile_cast_bf16_f32`, so a five-entry table would have emitted a name that
/// does not resolve. Counting at the wrong API layer is this crate's recurring
/// trap, and it is why the refusal now names the layer.
#[test]
fn a_pair_with_an_intrinsic_but_no_wrapper_is_refused_and_says_which() {
    let err = convert_pto_to_rust(&kernel("bf16", "f32", ""))
        .expect_err("bf16 -> f32 has an intrinsic but no safe wrapper");
    assert!(
        err.contains("bf16") && err.contains("__tile_cast_"),
        "the refusal must distinguish the intrinsic from the wrapper, got: {err}"
    );
}

#[test]
fn a_saturating_cast_is_not_silently_treated_as_a_plain_one() {
    // The corpus form: `ins(%a {sat_mode = #pto<saturation_mode ON>} : ...)`.
    // f32 -> f16 WOULD lower if the attribute were ignored, which is exactly
    // the mistake this guards: the emitted cast would have different overflow
    // behaviour from the input.
    let attrs = " {sat_mode = #pto<saturation_mode ON>}";
    let plain = convert_pto_to_rust(&kernel("f32", "f16", "")).expect("control: must lower");
    assert!(plain.contains("tile_cast_f32_f16"));

    let err = convert_pto_to_rust(&kernel("f32", "f16", attrs))
        .expect_err("a saturating tcvt must not lower to the plain cast");
    assert!(
        err.contains("sat_mode"),
        "the refusal must name the attribute it refused on, got: {err}"
    );
    assert!(
        !err.contains("tile_std has no cast") && !err.contains("cannot be lifted: the tile API"),
        "f32 -> f16 IS a supported pair; the refusal is the saturation mode, \
         not the dtypes: {err}"
    );
}

/// The attribute has to survive the round trip, or the refusal above is a lie:
/// a dropped `sat_mode` would make the re-emitted kernel a plain cast.
#[test]
fn sat_mode_survives_a_round_trip() {
    let pto = kernel("f32", "f16", " {sat_mode = #pto<saturation_mode ON>}");
    ptoas_rs::verify_roundtrip(&pto)
        .unwrap_or_else(|e| panic!("a saturating tcvt must round-trip: {}", e.summary()));
    let out = ptoas_rs::reemit_pto(&pto).expect("re-emit");
    assert!(
        out.contains("sat_mode = #pto<saturation_mode ON>"),
        "the saturation mode was dropped from the re-emitted line\n{out}"
    );
}
