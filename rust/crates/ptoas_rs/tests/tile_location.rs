// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `loc=` says which unit's memory a tile is in, and it was not in the AST.
//!
//! On Ascend, `left`/`right`/`acc` are the cube unit's L0A/L0B/L0C buffers
//! (`ptoas` emits them as `__ca__`/`__cb__`/`__cc__`), `mat` is L1, `scaling`
//! holds quantization scales, and `vec` is the vector unit's UB. The reverse
//! emitter wrote a literal `loc=vec` for every tile until 2026-11-01.
//!
//! Corpus census that day: **233 tiles are not `vec`** — `mat` ×77, `acc` ×50,
//! `right` ×41, `left` ×38, `scaling` ×27. So `matmul_32x64x32.pto`, one of the
//! eight artifacts the corpus sweep reports as round-tripping, was re-emitted
//! with all three matmul operands relocated out of the cube and into UB, and
//! the verifier called it equal: the field was absent from both passes.

use ptoas_rs::{parser::parse_module, reemit_pto};

fn tile(loc: &str) -> String {
    format!("!pto.tile_buf<loc={loc}, dtype=f32, rows=32, cols=32>")
}

fn module(loc: &str) -> String {
    let t = tile(loc);
    format!(
        r#"module {{
  func.func @k(%a: !pto.ptr<f32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %v = pto.make_tensor_view %a, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p = pto.partition_view %v, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %t = pto.alloc_tile : {t}
    pto.tload ins(%p : !pto.partition_tensor_view<32x32xf32>) outs(%t : {t})
    return
  }}
}}
"#
    )
}

#[test]
fn every_known_location_survives_the_round_trip() {
    for loc in ["vec", "mat", "left", "right", "acc", "scaling"] {
        let out = reemit_pto(&module(loc)).unwrap_or_else(|e| panic!("loc={loc}: {e}"));
        assert!(
            out.contains(&format!("loc={loc},")),
            "a `loc={loc}` tile was re-emitted somewhere else\n{out}"
        );
        if loc != "vec" {
            assert!(
                !out.contains("loc=vec"),
                "a `loc={loc}` tile was relocated to the vector unit's UB\n{out}"
            );
        }
    }
}

#[test]
fn an_unknown_location_is_refused_by_name() {
    let err = reemit_pto(&module("l0c")).expect_err("an unknown `loc` must not default to vec");
    assert!(
        err.contains("l0c"),
        "the refusal must name the location it saw: {err}"
    );
}

#[test]
fn a_missing_location_is_refused() {
    let src = module("vec").replace("loc=vec, ", "");
    let err = reemit_pto(&src).expect_err("a tile_buf with no `loc` must not be assumed to be vec");
    assert!(
        err.contains("loc"),
        "the refusal must say what was missing: {err}"
    );
}

/// The cube matmul fixture keeps its operands in the cube.
#[test]
fn the_matmul_fixture_keeps_its_cube_operands() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/matmul_32x64x32.pto");
    let src = std::fs::read_to_string(&path).expect("the matmul fixture");
    let m = parse_module(&src).expect("parse");
    let locs: Vec<&str> = m.funcs[0]
        .shapes
        .values()
        .filter_map(|sh| sh.loc.map(|l| l.as_str()))
        .collect();
    for want in ["left", "right", "acc"] {
        assert!(
            locs.contains(&want),
            "the fixture's `loc={want}` operand is not in the AST; got {locs:?}"
        );
    }
    // Counted, not `contains`. Two separate places build a tile type --
    // `format_type` for `alloc_tile` and `tile_ty_str` for an operand -- and a
    // mutant that hardcoded `vec` in only one of them left every `contains`
    // satisfied by the other.
    let out = reemit_pto(&src).expect("re-emit");
    for want in ["loc=left", "loc=right", "loc=acc", "loc=vec"] {
        let before = src.matches(want).count();
        let after = out.matches(want).count();
        assert_eq!(
            after, before,
            "`{want}` appears {before}x in the fixture and {after}x in the \
             re-emission\n{out}"
        );
    }
}
