// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Regressions for the parser gaps the repo-wide round-trip sweep exposed.
//!
//! The sweep (`roundtrip::all_repo_pto_artifacts_roundtrip`) went from 28 of 49
//! tracked artifacts failing to 8. Each test here pins one of the causes, so a
//! later change cannot quietly reopen it while the sweep's total stays green
//! for some other reason.
//!
//! Measured 2026-10-31.

use ptoas_rs::{convert_pto_to_rust_raw, verify_roundtrip};

/// A kernel that scales a tile by a constant. The second `ins` operand is an
/// `f32`, not a tile.
const TMULS: &str = r#"
module {
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %s = arith.constant 0.000061035 : f32
    %v0 = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p0 = pto.partition_view %v0, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    %t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>
    pto.tload ins(%p0 : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    pto.tmuls ins(%t0, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>, f32) outs(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>)
    %v1 = pto.make_tensor_view %b, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p1 = pto.partition_view %v1, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    pto.tstore ins(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>) outs(%p1 : !pto.partition_tensor_view<1x32xf32>)
    return
  }
}
"#;

/// The defect: `line.starts_with("pto.tmul")` is also true of `pto.tmuls`, so
/// the scalar form was parsed as the two-tile `pto.tmul` and the constant `%s`
/// was bound into a tile operand slot. 277 op instances across the repo's
/// artifacts were affected (`pto.tmuls` x151, `pto.tadds` x80, `pto.tmaxs` x46).
///
/// Round-tripping is the sharp end of it: the reverse emitter must print the
/// mnemonic it parsed, so a mis-parse re-emits `pto.tmul` and loses the `s`.
#[test]
fn tmuls_round_trips_as_tmuls_not_tmul() {
    if let Err(m) = verify_roundtrip(TMULS) {
        panic!(
            "tmuls did not round-trip: {:?}\n--- re-emitted ---\n{}",
            m.failure, m.reemitted
        );
    }
}

/// The failure was NOT silent -- worth recording, because the opposite is easy
/// to assume. `ScalarConst` emits no Rust binding, so the old two-tile arm hit
/// `lookup` on the scalar's SSA and refused with `unresolved SSA %s`, blaming
/// the operand for being unresolved rather than naming the prefix match that
/// put it there. Fail-closed with a misleading diagnostic, not a wrong kernel.
///
/// It still refuses today -- the scalar's literal is not threaded into this
/// emitter -- but the message now names the op and says which wall it hit.
/// `pto.tmuls` lifts, and the literal the artifact carried is the one emitted.
///
/// This test used to assert the REFUSAL, on the reasoning that the scalar's
/// value was not threaded into the emitter (`ScalarConst` emits no Rust
/// binding, and the integer constant map loses a float). The map was then built
/// in `emit_func` from the constants the parser already keeps, so the arm has
/// the literal and refuses only when there is genuinely none. The property
/// worth pinning flipped: not "it refuses and says why", but "it emits the
/// value from the input".
#[test]
fn tmuls_lifts_to_the_scale_wrapper_with_its_own_literal() {
    let rust = convert_pto_to_rust_raw(TMULS).expect("tmuls has an arm and a literal");
    assert!(
        rust.contains("tile_scale_f32::<1, 32>"),
        "expected the f32 scale wrapper at the tile's extent, got:\n{rust}"
    );
    assert!(
        rust.contains("0.000061035"),
        "the emitted call must carry the scalar the input declared, got:\n{rust}"
    );
    // And the intrinsic question stays answered: the wrapper, not `__tile_scale`.
    assert!(
        !rust.contains("__tile_scale"),
        "the lift emits safe wrappers, not buffer-API intrinsics, got:\n{rust}"
    );
}

/// `pto.tadds` has no tile-API arm at all: the `__tile_adds_*` that exist are
/// buffer-API, over `UbBuf`. Reaching for one by name would cross the two APIs.
#[test]
fn tadds_lifts_to_the_tile_api_and_never_reaches_for_the_buffer_one() {
    // This test used to assert a REFUSAL, on the reasoning that the only
    // `__tile_adds_*` in the crate were buffer-API forms over `UbBuf`. That is
    // still true of those names, but `tile_adds_f32` now exists in the tile API
    // (2026-11-04), so the op lifts -- and the property worth pinning flipped
    // with it: the emitted call must be the tile wrapper, never `UbBuf`.
    let src = TMULS.replace("pto.tmuls", "pto.tadds");
    let rust = convert_pto_to_rust_raw(&src).expect("tadds has a tile-API arm now");
    assert!(
        rust.contains("tile_adds_f32"),
        "expected the safe scalar-add wrapper, got:\n{rust}"
    );
    assert!(
        !rust.contains("UbBuf") && !rust.contains("__tile_adds"),
        "the lift must not cross into the buffer API, got:\n{rust}"
    );
}

/// A trailing `// ...` note containing a COLON moved the `rfind(':')` that
/// locates an `alloc_tile`'s type, so `parse_type` was handed `broadcast Q`
/// and reported an "unrecognised type" that was never a type. `parse_type`
/// stripped comments itself, but only after the split had already gone wrong.
#[test]
fn a_comment_containing_a_colon_is_not_mistaken_for_the_type() {
    let src = TMULS.replace(
        "%t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>",
        "%t1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>  // pool: broadcast Q",
    );
    assert!(
        src.contains("// pool: broadcast Q"),
        "fixture did not apply"
    );
    if let Err(m) = verify_roundtrip(&src) {
        panic!("comment with a colon broke the parse: {:?}", m.failure);
    }
}

/// `i16`, `i32` and `ui64` parse so artifacts mentioning them round-trip, but
/// none can be lifted. `ui64` is the interesting one: it is a pointer dtype in
/// the repo's GMM artifacts, and a kernel may take `*mut u64` arguments while
/// computing on f32 tiles -- so accepting the dtype must not by itself refuse
/// the kernel.
#[test]
fn ui64_pointer_args_round_trip() {
    let src = TMULS
        .replace("%a: !pto.ptr<f32>", "%a: !pto.ptr<ui64>")
        .replace("pto.tmuls", "pto.tmul")
        .replace("%s : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>, f32", "%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>");
    if let Err(m) = verify_roundtrip(&src) {
        panic!("ui64 ptr did not round-trip: {:?}", m.failure);
    }
}

/// An i16 tile parses but cannot be lifted: `tile_std` has no `__tile_*_i16`
/// intrinsic of any kind, so there is no suffix to interpolate into a name.
#[test]
fn i16_tiles_refuse_to_lift_naming_the_dtype() {
    let src = TMULS
        .replace("dtype=f32", "dtype=i16")
        .replace("pto.tmuls", "pto.tmul")
        .replace("%s : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32>, f32", "%t0 : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32>, !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32>");
    let err = convert_pto_to_rust_raw(&src).expect_err("i16 has no tile-API arm");
    assert!(err.contains("i16"), "diagnostic must name the dtype, got: {err}");
}

/// A rank-5 `partition_tensor_view` is a fractal/NZ layout. The old message was
/// "bad dtype in `1x4x16x16x32xi8`", which blamed `16` for not being a dtype
/// while the real dtype, `i8`, sat at the end and parses fine. The refusal must
/// name the RANK, because what `rows`/`cols` mean for a 5-D view is a layout
/// decision this AST has not made.
/// A rank-5 fractal/NZ view PARSES and round-trips; the LIFT is what refuses.
///
/// It used to be refused at parse — which was better than the refusal before
/// it (that one said "bad dtype in `1x4x16x16x32xi8`", blaming `16` for not
/// being a dtype while the real dtype `i8` sat at the end and parses fine) but
/// still threw the dims away. They are now kept as written, because collapsing
/// five dims to rows/cols is a guess: `1x4x16x16x32` has 32768 elements and
/// the tile it feeds is 256x128, so neither "last dim is cols" nor "first is
/// rows" recovers the tile shape.
///
/// The refusal moved to the emitter, where there is something concrete to say.
///
/// The fixture declares five offsets and five sizes, because a rank-5 view with
/// two of each is not a program `ptoas` can emit. Swapping only the result type
/// -- which is what this test did until 2026-11-01 -- built exactly that, and it
/// passed because the parser also kept only two. The round-trip half of this
/// property is carried in full by
/// `regions_and_view_types::a_rank_n_view_keeps_every_dim`, which checks the
/// offsets, strides and layout attribute too; this test is about the REFUSAL.
#[test]
fn a_rank5_view_round_trips_and_the_lift_names_the_rank() {
    let src = TMULS
        .replace(
            "offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32>",
            "offsets = [%c0, %c0, %c0, %c0, %c0], \
             sizes = [%c1, %c1, %c32, %c32, %c32] : !pto.tensor_view<?x?x?x?x?xf32>",
        )
        .replace(
            "!pto.partition_tensor_view<1x32xf32>",
            "!pto.partition_tensor_view<1x4x16x16x32xi8>",
        );
    verify_roundtrip(&src).expect("a rank-5 view must round-trip verbatim");
    let out = ptoas_rs::reemit_pto(&src).expect("re-emit");
    assert!(
        out.contains("!pto.partition_tensor_view<1x4x16x16x32xi8>"),
        "the five dims were not carried through\n{out}"
    );

    let why = convert_pto_to_rust_raw(&src).expect_err("a rank-5 view cannot be lifted");
    assert!(why.contains("rank-5"), "must name the rank, got: {why}");
    assert!(
        why.contains("1x4x16x16x32"),
        "must quote the dims it saw, got: {why}"
    );
    assert!(
        !why.starts_with("bad dtype"),
        "must not blame the dtype, got: {why}"
    );
}

// ─── Widths the AST did not carry (2026-11-01) ──────────────────────────────

/// `module_scalar(mnemonic, tile_dtype, scalar_ty, literal)` -- one tile-by-scalar
/// op at an arbitrary width.
fn module_scalar(mn: &str, dt: &str, sty: &str, lit: &str) -> String {
    let t = format!("!pto.tile_buf<loc=vec, dtype={dt}, rows=1, cols=32>");
    format!(
        r#"module {{
  func.func @k(%a: !pto.ptr<{dt}>, %b: !pto.ptr<{dt}>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %s = arith.constant {lit} : {sty}
    %v0 = pto.make_tensor_view %a, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?x{dt}>
    %p0 = pto.partition_view %v0, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?x{dt}> -> !pto.partition_tensor_view<1x32x{dt}>
    %t0 = pto.alloc_tile : {t}
    %t1 = pto.alloc_tile : {t}
    pto.tload ins(%p0 : !pto.partition_tensor_view<1x32x{dt}>) outs(%t0 : {t})
    {mn} ins(%t0, %s : {t}, {sty}) outs(%t1 : {t})
    %v1 = pto.make_tensor_view %b, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?x{dt}>
    %p1 = pto.partition_view %v1, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?x{dt}> -> !pto.partition_tensor_view<1x32x{dt}>
    pto.tstore ins(%t1 : {t}) outs(%p1 : !pto.partition_tensor_view<1x32x{dt}>)
    return
  }}
}}
"#
    )
}

/// The scalar operand of a tile-by-scalar op is NOT always `f32`.
///
/// Corpus census 2026-11-01: `pto.tadds` carries an `f16` scalar in 30
/// instances and `pto.tmuls`/`pto.tmaxs` an `i16` one in 24; the unmodelled
/// shift and mask family uses `i32`. The first version of `PtoOp::BinOpScalar`
/// did not carry the field at all and the reverse emitter wrote a literal
/// `f32`, so each of those was re-emitted at the wrong width.
///
/// NEITHER gate saw it. `parse_ins2_outs1` discards the `ins` type list, so the
/// loss never reached the AST that the round-trip compares, and the corpus
/// sweep's 8 round-tripping artifacts use no scalar op at all. This test
/// asserts on the re-emitted TEXT, which is sensitive whether or not the AST
/// carries the field.
#[test]
fn a_scalar_operand_keeps_its_own_width() {
    for (mn, dt, sty, lit) in [
        ("pto.tmuls", "f32", "f32", "0.5"),
        ("pto.tadds", "f16", "f16", "1.5"),
        ("pto.tmaxs", "i16", "i16", "7"),
        ("pto.tmins", "f32", "f32", "-1.0"),
    ] {
        let src = module_scalar(mn, dt, sty, lit);
        let out = ptoas_rs::reemit_pto(&src)
            .unwrap_or_else(|e| panic!("{mn} at {sty} did not re-emit: {e}"));
        let want = format!(", {sty}) outs(");
        assert!(
            out.contains(&want),
            "{mn} re-emitted its scalar operand at the wrong width -- wanted \
             `{want}`\n{out}"
        );
        assert!(
            verify_roundtrip(&src).is_ok(),
            "{mn} at {sty} did not round-trip"
        );
    }
}

/// A width-typed integer constant is not an `index`.
///
/// `ty.starts_with('i')` is true of `i16` and `i32`, so they were folded into
/// the index arm and re-emitted as `: index`. Corpus census 2026-11-01: 144
/// `i16` and 28 `i32` constants, every one of them re-typed. Invisible to the
/// sweep because `ConstVal::Index` carries no type, so both passes produced the
/// same wrong node and the comparison was equal.
///
/// `index` itself must keep round-tripping as `index` -- that is the half that
/// index folding depends on.
#[test]
fn an_integer_constant_keeps_its_width() {
    // Literals chosen not to collide with the preamble's own `index`
    // constants (0, 1, 32) -- the negative assertion below is a substring
    // search, and `%c0 = arith.constant 0 : index` is legitimately there.
    for (sty, lit) in [("i16", "7"), ("i32", "9"), ("f16", "128.0")] {
        let src = module_scalar("pto.tadds", "f16", sty, lit);
        let out = ptoas_rs::reemit_pto(&src)
            .unwrap_or_else(|e| panic!("constant at {sty} did not re-emit: {e}"));
        assert!(
            out.contains(&format!("arith.constant {lit} : {sty}")),
            "a `{sty}` constant was re-emitted at another type\n{out}"
        );
        assert!(
            !out.contains(&format!("arith.constant {lit} : index")),
            "a `{sty}` constant was re-emitted as `index`\n{out}"
        );
    }

    // The index half, which folding needs, is unchanged.
    let out = ptoas_rs::reemit_pto(&module_scalar("pto.tmuls", "f32", "f32", "0.5")).unwrap();
    assert!(
        out.contains("arith.constant 32 : index"),
        "an `index` constant stopped round-tripping as `index`\n{out}"
    );
}
