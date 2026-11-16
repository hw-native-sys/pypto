// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The rms-norm fusion, tested against the artifacts that carry the chain.
//!
//! Every assertion here reads a real `mlir_to_pto` output file. That is the
//! whole point of the file. The sibling branch shipped this rule with three
//! tests, all of them on a hand-written fixture, and the rule matched a chain
//! that does not occur: it expected `pto.trsqrt` (absent from the corpus --
//! `mlir_to_pto` spells rsqrt as `tsqrt` then `trecip`) and ended on
//! `pto.trowexpandmul` (which occurs, but in the u8/u32 quantized kernels,
//! not in a normalisation). Three green tests, zero files matched.
//!
//! A fixture still earns its place for the NEGATIVE cases -- the corpus has
//! no chain with a wrong scalar, so the 1/N guard can only be exercised by
//! perturbing a real file, which is what `with_scale` below does.

use ptoas_rs::ast::{PtoOp, Ssa};

/// The six artifacts carrying the chain, as (path, source tile width).
///
/// Widths are from the kernel names and confirmed by the tile types:
/// `@ds4_rmsmul_4096` loads 1x4096, and its `tmuls` scalar is 0.000244141 =
/// 1/4096. The other scalars are 1/1024, 1/512 and 1/16384.
const CHAIN_FILES: &[&str] = &[
    "barriered_e.pto",
    "barriered_f.pto",
    "barriered_g.pto",
    "barriered_k.pto",
    "ffn.pto",
    "layer0.pto",
];

/// Fixtures live beside this test rather than at repository-root-relative
/// paths, which resolved into a tree this workspace does not carry.
fn read(rel: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/barrier")
        .join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn count_ops(ops: &[PtoOp], pred: &dyn Fn(&PtoOp) -> bool) -> usize {
    let mut n = 0;
    for op in ops {
        if pred(op) {
            n += 1;
        }
        match op {
            PtoOp::For { body, .. } => n += count_ops(body, pred),
            PtoOp::If {
                then_body,
                else_body,
                ..
            } => {
                n += count_ops(then_body, pred);
                n += count_ops(else_body.as_deref().unwrap_or(&[]), pred);
            }
            _ => {}
        }
    }
    n
}

fn fused_count(src: &str, pred: &dyn Fn(&PtoOp) -> bool) -> usize {
    let m = ptoas_rs::fused_module(src).expect("parse + fuse");
    m.funcs.iter().map(|f| count_ops(&f.ops, pred)).sum()
}

fn is_rms(op: &PtoOp) -> bool {
    matches!(op, PtoOp::RmsNorm { .. })
}

/// The rule fires on every artifact that carries the chain, and the total
/// matches the number of chains the text has.
///
/// `pto.tsqrt` occurs once per chain and nowhere else in the corpus, so
/// counting it in the TEXT gives an expected count derived independently of
/// the AST. Asserting "at least one fused" would pass while the rule matched
/// one chain of a file's three.
#[test]
fn the_rule_fires_once_per_chain_in_every_artifact_that_has_one() {
    let mut total_text = 0;
    let mut total_fused = 0;
    for rel in CHAIN_FILES {
        let src = read(rel);
        let in_text = src.matches("pto.tsqrt ").count();
        assert!(
            in_text > 0,
            "{rel} is listed as carrying the chain but has no `pto.tsqrt`; \
             the corpus moved and this list is stale"
        );
        let fused = fused_count(&src, &is_rms);
        assert_eq!(
            fused, in_text,
            "{rel}: {in_text} chain(s) in the text, {fused} fused"
        );
        total_text += in_text;
        total_fused += fused;
    }
    // 2026-11-01: 10 chains across the six files.
    assert_eq!(total_fused, total_text);
    assert!(
        total_fused >= 10,
        "only {total_fused} chains fused across the corpus (was 10 on 2026-11-01)"
    );
}

/// Fusing removes the chain's parts -- it does not shadow them.
///
/// Without this a rule that inserted `RmsNorm` and left the eight ops in
/// place would pass the count test above, and the lift would then emit the
/// normalisation twice.
#[test]
fn the_fused_chain_leaves_none_of_its_parts_behind() {
    for rel in CHAIN_FILES {
        let src = read(rel);
        // `tsqrt`/`trecip`/`trowexpand` appear ONLY in this chain, so after
        // fusion their count must be zero. `tmul`, `trowsum`, `tmuls` and
        // `tadds` also occur outside it, so they are not assertable this way.
        for (name, pred) in [
            (
                "tsqrt",
                &(|o: &PtoOp| {
                    matches!(
                        o,
                        PtoOp::UnOp {
                            kind: ptoas_rs::ast::UnKind::Sqrt,
                            ..
                        }
                    )
                }) as &dyn Fn(&PtoOp) -> bool,
            ),
            (
                "trecip",
                &|o: &PtoOp| {
                    matches!(
                        o,
                        PtoOp::UnOp {
                            kind: ptoas_rs::ast::UnKind::Recip,
                            ..
                        }
                    )
                },
            ),
            ("trowexpand", &|o: &PtoOp| {
                matches!(o, PtoOp::RowExpand { .. })
            }),
        ] {
            let left = fused_count(&src, pred);
            assert_eq!(left, 0, "{rel}: {left} `{name}` survived the fusion");
        }
    }
}

/// The fused op names the chain's own source tile and its own eps.
///
/// A rule can match the right eight ops and still wire the wrong operand --
/// `%src` has to be the tile that feeds BOTH the square and the final
/// multiply, which is what makes the result a norm of `%src`.
#[test]
fn the_fused_op_carries_the_chains_source_and_eps() {
    // `@ds4_rmsmul_4096`: `%pto2` is the loaded 1x4096 input, squared at the
    // top of the chain and multiplied by the broadcast reciprocal at the end.
    // eps is `%pto7 = arith.constant 0.000001 : f32`.
    let src = read("barriered_e.pto");
    let m = ptoas_rs::fused_module(&src).expect("parse + fuse");
    let fused: Vec<&PtoOp> = m
        .funcs
        .iter()
        .flat_map(|f| f.ops.iter())
        .filter(|o| is_rms(o))
        .collect();
    assert_eq!(fused.len(), 1, "expected one fused op, got {}", fused.len());
    match fused[0] {
        PtoOp::RmsNorm { src: s, eps, dst } => {
            assert_eq!(*s, Ssa("%pto2".into()), "wrong source tile");
            assert_eq!(eps, "0.000001", "eps was not carried verbatim");
            // The chain's output is the tile the final `tmul` wrote, and the
            // kernel's trailing gain multiply reads it -- so it must NOT be
            // the kernel's stored result.
            assert_eq!(*dst, Ssa("%pto16".into()), "wrong destination tile");
        }
        other => panic!("not an RmsNorm: {other:?}"),
    }
}

/// eps is READ from the artifact, not assumed.
///
/// Every chain in the corpus uses `0.000001`, so the assertion in the test
/// above -- `eps == "0.000001"` -- is equally satisfied by an emitter that
/// hardcodes it. Mutating the field to a literal `"0.000001"` survives that
/// test. Perturbing the artifact is the only way to tell the two apart.
///
/// The text comparison also matters: `1e-6` and `0.000001` are the same
/// number and different artifacts, and the generated Rust should carry what
/// the file said.
#[test]
fn eps_is_read_from_the_artifact_and_not_assumed() {
    let src = read("barriered_e.pto");
    for eps in ["0.0000152587890625", "1e-6", "0.00001"] {
        let patched = src.replace(
            "%pto7 = arith.constant 0.000001 : f32",
            &format!("%pto7 = arith.constant {eps} : f32"),
        );
        assert_ne!(patched, src, "the perturbation did not apply");
        let m = ptoas_rs::fused_module(&patched).expect("parse + fuse");
        let got: Vec<String> = m
            .funcs
            .iter()
            .flat_map(|f| f.ops.iter())
            .filter_map(|o| match o {
                PtoOp::RmsNorm { eps, .. } => Some(eps.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(got, vec![eps.to_string()], "eps was not carried from the file");
    }
}

/// Condition 3: the `tmuls` scalar must be 1/N for the tile's own width.
///
/// Perturbed from a real artifact rather than a fixture, so the only thing
/// that differs from a matching chain is the constant. Without this guard the
/// rule relabels any `x * k` chain of this shape a normalisation -- the
/// difference between a mean of squares and a scaled sum of squares is
/// entirely in the constant.
#[test]
fn a_scalar_that_is_not_one_over_n_does_not_fuse() {
    let src = read("barriered_e.pto");
    assert_eq!(fused_count(&src, &is_rms), 1, "baseline chain must fuse");

    // 1/4096 = 0.000244141, as the artifact has it.
    for wrong in ["0.5", "0.000488281", "0.000122070"] {
        let patched = src.replace("arith.constant 0.000244141 : f32", &format!("arith.constant {wrong} : f32"));
        assert_ne!(patched, src, "the perturbation did not apply");
        assert_eq!(
            fused_count(&patched, &is_rms),
            0,
            "fused a chain whose scalar is {wrong}, not 1/4096"
        );
    }
}

/// And the check follows the tile's width rather than a hardcoded number:
/// the same scalar that is right for 4096 is wrong for 1024.
///
/// `l0/f.pto` is the 1024-wide kernel with scalar 1/1024. Giving it e.pto's
/// 1/4096 must stop the fusion -- if the rule compared against a constant it
/// would accept both.
#[test]
fn the_one_over_n_check_follows_each_tiles_own_width() {
    let f = read("barriered_f.pto");
    assert_eq!(fused_count(&f, &is_rms), 1, "the 1024-wide chain must fuse");
    let crossed = f.replace("arith.constant 0.000976562 : f32", "arith.constant 0.000244141 : f32");
    assert_ne!(crossed, f, "the perturbation did not apply");
    assert_eq!(
        fused_count(&crossed, &is_rms),
        0,
        "1/4096 was accepted for a 1024-wide tile"
    );
}

/// Conditions 1 and 2, each perturbed on its own.
///
/// These two were found by mutation: deleting the `a == b` test on the opening
/// multiply, and deleting the operand test on the closing one, both left the
/// whole suite green. Neither changes anything on the corpus -- the real
/// chains satisfy both -- so only a perturbation can tell a rule that checks
/// them from one that does not. Without the first, any `tmul x, y` opens a
/// window; without the second, the chain's reciprocal gets applied to a tile
/// the norm was not computed from, which is a wrong kernel rather than a
/// missed fusion.
#[test]
fn the_opening_multiply_must_be_a_square() {
    let src = read("barriered_e.pto");
    assert_eq!(fused_count(&src, &is_rms), 1, "baseline chain must fuse");
    // `%pto5` is the gain tile, same 1x4096 f32 shape, so the module stays
    // well-typed and the ONLY thing that changed is that `%pto8` is no longer
    // the square of anything.
    let patched = src.replace(
        "pto.tmul ins(%pto2, %pto2 :",
        "pto.tmul ins(%pto2, %pto5 :",
    );
    assert_ne!(patched, src, "the perturbation did not apply");
    assert_eq!(
        fused_count(&patched, &is_rms),
        0,
        "fused a chain whose opening multiply is not a square"
    );
}

#[test]
fn the_closing_multiply_must_read_the_normalised_source() {
    let src = read("barriered_e.pto");
    // The closing multiply is `%pto2 * %pto15` (source times broadcast
    // reciprocal). Point it at `%pto5` instead: the norm is still computed
    // from `%pto2`, but applied to a different tile.
    let patched = src.replace(
        "pto.tmul ins(%pto2, %pto15 :",
        "pto.tmul ins(%pto5, %pto15 :",
    );
    assert_ne!(patched, src, "the perturbation did not apply");
    assert_eq!(
        fused_count(&patched, &is_rms),
        0,
        "fused a chain that scales a tile the norm was not computed from"
    );
}

/// The fused op is emitted AFTER the `alloc_tile` that declares its output.
///
/// The ported rule carried a comment saying that placing it at the window's
/// first position puts it ahead of its own allocation and the Rust emitter
/// fails with `unresolved SSA`. That comment was untestable here -- these
/// artifacts are refused on `pto.barrier` before name resolution runs -- so
/// moving the insert back to the window start left the suite green. The
/// ordering is a property of the AST, so it can be asserted there instead of
/// through the emitter.
#[test]
fn the_fused_op_follows_the_allocation_of_its_own_output() {
    let src = read("barriered_e.pto");
    let m = ptoas_rs::fused_module(&src).expect("parse + fuse");
    let mut checked = 0;
    for f in &m.funcs {
        for (i, op) in f.ops.iter().enumerate() {
            let PtoOp::RmsNorm { dst, .. } = op else {
                continue;
            };
            let alloc = f.ops[..i].iter().position(|o| {
                matches!(o, PtoOp::AllocTile { dst: d, .. } if d == dst)
            });
            assert!(
                alloc.is_some(),
                "the fused op at index {i} writes `{}`, whose `alloc_tile` is \
                 not before it -- the Rust emitter mints names from the allocs \
                 and would fail with `unresolved SSA`",
                dst.0
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 1, "expected one fused op to check, saw {checked}");
}

/// Soundness: an intermediate read after the chain blocks the fusion.
///
/// The same condition the softmax rule has, exercised on a real chain. `%pto14`
/// is the reciprocal, the last 1x8 intermediate; a `tneg` of it spliced in
/// after the chain means a fused node would drop a value the kernel still
/// uses.
#[test]
fn an_intermediate_read_after_the_chain_blocks_the_fusion() {
    let src = read("barriered_e.pto");
    assert_eq!(fused_count(&src, &is_rms), 1, "baseline chain must fuse");

    const T8: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=8>";
    let leak = format!(
        "    %leak = pto.alloc_tile : {T8}\n    \
         pto.tneg ins(%pto14 : {T8}) outs(%leak : {T8})\n    \
         %pto18 = pto.make_tensor_view"
    );
    let patched = src.replace("    %pto18 = pto.make_tensor_view", &leak);
    assert_ne!(patched, src, "the splice did not apply");
    assert_eq!(
        fused_count(&patched, &is_rms),
        0,
        "fused despite `%pto14` being read after the chain"
    );
}

/// The whole reason to fuse: none of the chain's own ops can be lifted.
///
/// If any of these four grows a `tile_std` wrapper, the piecewise path
/// becomes possible and this test should be revisited -- it is not asserting
/// that lifting them is wrong, only recording that today it is impossible, so
/// that the fusion is understood as the only route and not as an
/// optimisation.
#[test]
fn none_of_the_chains_parts_lift_on_their_own() {
    const T32: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>";
    const T1: &str = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=1>";
    for (mnem, line) in [
        ("pto.tsqrt", format!("pto.tsqrt ins(%t0 : {T32}) outs(%t2 : {T32})")),
        (
            "pto.trecip",
            format!("pto.trecip ins(%t0 : {T32}) outs(%t2 : {T32})"),
        ),
    ] {
        let src = shared::module_with(&line);
        let err = ptoas_rs::convert_pto_to_rust(&src)
            .expect_err(&format!("`{mnem}` lifted on its own; the fusion rationale changed"));
        assert!(
            err.contains(mnem),
            "`{mnem}` was refused for another reason first: {err}"
        );
    }

    // `pto.trowexpand` USED TO BE HERE too, and it lifts now:
    // `tile_row_expand_f32` was declared on 2026-11-04 for the delta-net's decode
    // step, so the broadcast no longer needs the rms-norm fusion to be liftable.
    let expand = format!("pto.trowexpand ins(%r : {T1}) outs(%t2 : {T32})");
    let lifted = ptoas_rs::convert_pto_to_rust(&shared::module_with(&expand))
        .expect("`pto.trowexpand` now has a tile-API arm and must lift");
    assert!(
        lifted.contains("tile_row_expand_f32"),
        "it must lift to the broadcast wrapper, got:\n{lifted}"
    );

    // `pto.tadds` USED TO BE HERE and is no longer refused: `tile_adds_f32`
    // exists as of 2026-11-04 (the PTO emitter grew the arm and `tile_std` grew
    // the declaration), so it lifts on its own and the fusion is not the only
    // route for it. Asserted in the positive direction here rather than deleted,
    // because "this part does not lift" was a fact about the API and it changed.
    let adds = format!("pto.tadds ins(%t0, %sc : {T32}, f32) outs(%t2 : {T32})");
    let lifted = ptoas_rs::convert_pto_to_rust(&shared::module_with(&adds))
        .expect("`pto.tadds` now has a tile-API arm and must lift");
    assert!(
        lifted.contains("tile_adds_f32"),
        "it must lift to the scalar-add wrapper, got:\n{lifted}"
    );
}

/// The generated call carries each tile's OWN shape and the file's own eps.
///
/// Reaching the emitter arm at all needs the barriers gone, since that is the
/// one gate still refusing these files. Stripping them from a real artifact is
/// a smaller fiction than writing a chain by hand, and it is the only way to
/// test the arm: with the gate in place, mutating `cols` to `rows` in the
/// emitted call left the whole suite green.
fn without_barriers(rel: &str) -> String {
    let src = read(rel);
    let out = src.replace("    pto.barrier <PIPE_ALL>\n", "");
    assert_ne!(out, src, "{rel} has no barriers to strip");
    out
}

#[test]
fn the_lifted_call_carries_the_tiles_own_shape_and_the_files_own_eps() {
    // Two widths, so a hardcoded shape fails on one of them.
    for (rel, want) in [
        (
            "barriered_e.pto",
            "tile_rms_norm_f32::<1, 4096>",
        ),
        (
            "barriered_f.pto",
            "tile_rms_norm_f32::<1, 1024>",
        ),
        (
            "barriered_g.pto",
            "tile_rms_norm_f32::<1, 512>",
        ),
    ] {
        let rust = ptoas_rs::convert_pto_to_rust(&without_barriers(rel))
            .unwrap_or_else(|e| panic!("{rel} without barriers should lift: {e}"));
        assert_eq!(
            rust.matches(want).count(),
            1,
            "{rel}: expected one `{want}`, got:\n{rust}"
        );
        // eps reaches the call as a runtime f32 argument, not a const generic.
        assert!(
            rust.contains("0.000001f32"),
            "{rel}: eps missing from the generated call:\n{rust}"
        );
        // And nothing from the decomposed chain survived into the output.
        for gone in ["tile_sqrt", "tile_recip", "tile_row_expand", "tile_reduce_sum"] {
            assert!(
                !rust.contains(gone),
                "{rel}: `{gone}` in the lifted output:\n{rust}"
            );
        }
    }
}

/// The four pure rms-norm kernels are fully modelled AND lift.
///
/// They were refused on `pto.barrier` when the fusion first landed; the
/// barrier question is now decided (see `tests/barrier.rs`), and these four
/// are the files it unblocked -- the lift went 7 of 49 to 11.
#[test]
fn the_rms_norm_kernels_are_fully_modelled_and_lift() {
    for (rel, want) in [
        ("barriered_e.pto", "tile_rms_norm_f32::<1, 4096>"),
        ("barriered_f.pto", "tile_rms_norm_f32::<1, 1024>"),
        ("barriered_g.pto", "tile_rms_norm_f32::<1, 512>"),
        ("barriered_k.pto", "tile_rms_norm_f32::<1, 16384>"),
    ] {
        let src = read(rel);
        // Nothing in the file is carried verbatim: the parser models it whole.
        let (modelled, unmodelled) = ptoas_rs::op_model_coverage(&src)
            .unwrap_or_else(|e| panic!("{rel} did not parse: {e}"));
        assert_eq!(unmodelled, 0, "{rel} has {unmodelled} unmodelled op(s)");
        assert!(modelled > 30, "{rel} modelled only {modelled} ops");

        let rust = ptoas_rs::convert_pto_to_rust(&src)
            .unwrap_or_else(|e| panic!("{rel} should lift: {e}"));
        assert!(
            rust.contains(want),
            "{rel}: expected `{want}` in:\n{rust}"
        );
    }
}

mod shared;
