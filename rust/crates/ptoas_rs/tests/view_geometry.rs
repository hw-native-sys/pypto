// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The geometry of a view — `shape`, `strides`, `offsets`, `sizes` — is part of
//! the kernel, and until 2026-11-01 none of it reached the S1 AST.
//!
//! Consequences measured that day:
//!
//!   * `pto.partition_view %v, offsets = [%c16, %c0], …` and the same line with
//!     `[%c0, %c0]` parsed to the SAME AST. 119 of the corpus's partition views
//!     are at the origin and **321 are not**, including a 16-way static row
//!     tiling (`[%c8, %c0]` … `[%c120, %c0]`, 9 instances each) and a
//!     loop-variant `[%r, %c0]` ×42.
//!   * `emitter::emit_op` lifts a partition view to `let p = v;`, so those
//!     sixteen row tiles lifted to sixteen kernels all reading row 0, exit 0.
//!   * the reverse emitter wrote `[?, ?]`, which is not a valid program, and
//!     the round-trip verifier called it equal to the input — because the drop
//!     happened on both passes.
//!
//! It was NOT reachable through the lift on 2026-11-01: all 8 fully-modelled
//! artifacts sit at the origin, and every file with a non-zero offset also
//! holds an op the parser refuses, so it never reached the emitter. Modelling
//! more ops is what would have armed it.

use ptoas_rs::{convert_pto_to_rust, parser::parse_module, reemit_pto, PtoOp};

fn module(shape: &str, strides: &str, offsets: &str, sizes: &str) -> String {
    let t = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>";
    format!(
        r#"module {{
  func.func @k(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %v0 = pto.make_tensor_view %a, shape = [{shape}], strides = [{strides}] : !pto.tensor_view<?x?xf32>
    %p0 = pto.partition_view %v0, offsets = [{offsets}], sizes = [{sizes}] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    %t0 = pto.alloc_tile : {t}
    %t1 = pto.alloc_tile : {t}
    pto.tload ins(%p0 : !pto.partition_tensor_view<1x32xf32>) outs(%t0 : {t})
    pto.texp ins(%t0 : {t}) outs(%t1 : {t})
    %v1 = pto.make_tensor_view %b, shape = [%c1, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %p1 = pto.partition_view %v1, offsets = [%c0, %c0], sizes = [%c1, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x32xf32>
    pto.tstore ins(%t1 : {t}) outs(%p1 : !pto.partition_tensor_view<1x32xf32>)
    return
  }}
}}
"#
    )
}

const ORIGIN: (&str, &str, &str, &str) = ("%c1, %c32", "%c32, %c1", "%c0, %c0", "%c1, %c32");

#[test]
fn the_origin_case_still_lifts() {
    let (sh, st, of, sz) = ORIGIN;
    convert_pto_to_rust(&module(sh, st, of, sz)).expect("a view at the origin must still lift");
}

/// An offset that LEAVES the parent is refused.
///
/// This fixture was written to prove a non-zero offset refuses rather than
/// aliasing the base pointer, and it asserted the word "origin". A constant
/// offset is now folded into the address instead, so what the fixture actually
/// shows is the other thing it always was: `[%c16, %c0]` of a 1x32 parent is
/// row 16 of one row. Folding it would emit an out-of-bounds address, which is
/// the same defect as aliasing, one step further on.
#[test]
fn an_offset_that_leaves_the_parent_is_refused() {
    let (sh, st, _, sz) = ORIGIN;
    let err = convert_pto_to_rust(&module(sh, st, "%c16, %c0", sz))
        .expect_err("a window outside the parent must not lift");
    assert!(
        err.contains("16") && err.contains("leaves the parent"),
        "the refusal must say the window is out of bounds: {err}"
    );
}

/// A constant offset that FITS is folded into the pointer.
///
/// `[%c0, %c16]` of a 1x32 parent, taking 16 columns: the second half of the
/// row. 316 of the corpus's 435 rank-2 partition views are non-origin, and
/// before this they were all refused -- the refusal's own text said the answer
/// was to fold the offset into the address.
#[test]
fn a_constant_offset_that_fits_is_folded_into_the_address() {
    let t = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16>";
    let src = module("%c1, %c32", "%c32, %c1", "%c0, %c16", "%c1, %c16")
        .replace("!pto.partition_tensor_view<1x32xf32>", "!pto.partition_tensor_view<1x16xf32>")
        .replace("!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>", t);
    let rust = convert_pto_to_rust(&src).unwrap_or_else(|e| panic!("must lift: {e}"));
    assert!(
        rust.contains(".wrapping_add(16)"),
        "the offset was not folded into the address:\n{rust}"
    );
    // `wrapping_add`, not `add`: the emitted file is `#![no_core]`, where the
    // only inherent raw-pointer impls are the ones `tile_std::core::ptr`
    // declares. `tests/lifted_kernels_compile.rs` is what proves that;
    // this pins the spelling so a "tidy-up" cannot quietly swap it.
    assert!(
        !rust.contains(".add("),
        "`.add` is a `core` method and does not resolve in a `no_core` crate:\n{rust}"
    );
}

/// The fold is an ADDRESS, scaled by the PARENT's row stride.
///
/// A row offset is `r * parent_cols` elements, not `r`. Both readings give
/// `wrapping_add` of something, and only one is the kernel.
#[test]
fn a_row_offset_is_scaled_by_the_parents_row_stride() {
    let t = "!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>";
    let src = module("%c64, %c32", "%c32, %c1", "%c16, %c0", "%c1, %c32")
        .replace("!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32>", t);
    let rust = convert_pto_to_rust(&src).unwrap_or_else(|e| panic!("must lift: {e}"));
    assert!(
        rust.contains(".wrapping_add(512)"),
        "row 16 of a 32-column parent is element 512, not 16:\n{rust}"
    );
}

#[test]
fn a_loop_variant_offset_is_refused_too() {
    let (sh, st, _, sz) = ORIGIN;
    // `%r` is in no constant map -- this is the `[%r, %c0]` ×42 shape, and the
    // one case a fold cannot resolve, so it must refuse rather than assume 0.
    let err = convert_pto_to_rust(&module(sh, st, "%r, %c0", sz))
        .expect_err("an unresolvable offset must not be assumed to be zero");
    assert!(err.contains("%r"), "the refusal must name the operand: {err}");
}

#[test]
fn a_strided_parent_is_refused_not_aliased() {
    let (sh, _, of, sz) = ORIGIN;
    // shape says 32 columns; the row stride says 64. The tile load walks 32
    // elements from the base, so every row after the first reads the wrong
    // elements.
    let err = convert_pto_to_rust(&module(sh, "%c64, %c1", of, sz))
        .expect_err("a non-contiguous parent must not lift to a base-pointer alias");
    assert!(
        err.contains("contiguous") || err.contains("strided"),
        "the refusal must say what it objected to: {err}"
    );
}

#[test]
fn view_geometry_round_trips_verbatim() {
    let (sh, st, _, sz) = ORIGIN;
    let src = module(sh, st, "%c16, %c0", sz);
    let out = reemit_pto(&src).expect("re-emit");
    for want in [
        "shape = [%c1, %c32]",
        "strides = [%c32, %c1]",
        "offsets = [%c16, %c0]",
        "sizes = [%c1, %c32]",
    ] {
        assert!(
            out.contains(want),
            "re-emission lost `{want}`; it used to write `[?, ?]`\n{out}"
        );
    }
    assert!(
        !out.contains("[?, ?]"),
        "re-emission still writes a `[?, ?]` placeholder\n{out}"
    );
}

/// Two kernels differing only in a partition offset must not share an AST.
///
/// This is the property the round-trip verifier needed and could not have: a
/// symmetric comparison cannot see a field that neither side carries.
#[test]
fn two_offsets_are_two_asts() {
    let (sh, st, _, sz) = ORIGIN;
    let a = parse_module(&module(sh, st, "%c0, %c0", sz)).unwrap();
    let b = parse_module(&module(sh, st, "%c16, %c0", sz)).unwrap();
    assert_ne!(
        format!("{:?}", a.funcs[0].ops),
        format!("{:?}", b.funcs[0].ops),
        "a row-offset view parsed to the same AST as one at the origin"
    );
}

/// The AST's geometry must equal what a plain text re-read finds.
///
/// `emitter_tile` validates the geometry by re-reading the raw source, which is
/// how it checked offsets while the AST still dropped them. Two readings of the
/// same text are now in the crate, so they have to agree — otherwise one of
/// them is wrong and the checked lift and the raw lift disagree about which
/// kernel the file describes.
#[test]
fn geometry_matches_the_text() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let mut files = Vec::new();
    collect(&root, &mut files);
    let mut checked = 0usize;
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap_or_default();
        let Ok(m) = parse_module(&src) else { continue };
        // Text reading: every `offsets = [..]` in source order.
        let from_text: Vec<String> = src
            .lines()
            .filter(|l| l.contains("pto.partition_view"))
            .filter_map(|l| {
                let i = l.find("offsets = [")? + "offsets = [".len();
                let j = l[i..].find(']')? + i;
                Some(l[i..j].replace(' ', ""))
            })
            .collect();
        // Flattened, because `scf.for` holds a region: a partition view inside
        // a loop body is not in `fu.ops`. The first version of this walk was
        // flat and the cross-check caught it the moment regions landed, which
        // is the whole point of having two readings.
        let mut flat = Vec::new();
        for fu in &m.funcs {
            push_flat(&fu.ops, &mut flat);
        }
        let from_ast: Vec<String> = flat
            .iter()
            // Rank-general on BOTH sides. The text reading always was (it
            // takes whatever sits between the brackets); the AST reading
            // printed `offsets[0], offsets[1]`, so a rank-5 view compared equal
            // to its own first two dims. That is the cross-check's first real
            // finding, and it only became visible once the rank-5 `gmm_*`
            // artifacts parsed at all.
            .filter_map(|op| match op {
                PtoOp::PartitionView { offsets, .. } => Some(offsets.join(",")),
                PtoOp::PartitionViewNd { offsets, .. } => Some(offsets.join(",")),
                _ => None,
            })
            .collect();
        assert_eq!(
            from_text,
            from_ast,
            "the AST and a text re-read disagree about the partition offsets in {}",
            f.display()
        );
        if !from_ast.is_empty() {
            checked += 1;
        }
    }
    assert!(
        checked >= 7,
        "only {checked} artifact(s) carried a partition view to compare; the \
         cross-check has nothing to do"
    );
}

fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(e) = std::fs::read_dir(dir) else { return };
    for en in e.flatten() {
        let p = en.path();
        if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
            if matches!(
                n,
                "target" | ".git" | ".claude" | "node_modules" | ".docker" | ".slicer-cache"
            ) {
                continue;
            }
        }
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("pto") {
            out.push(p);
        }
    }
}

/// Every op in source order, descending into regions.
fn push_flat<'a>(ops: &'a [PtoOp], out: &mut Vec<&'a PtoOp>) {
    for op in ops {
        out.push(op);
        match op {
            PtoOp::For { body, .. } => push_flat(body, out),
            // Both branches, in source order. `scf.if` landed after `scf.for`
            // and this walk did not follow it -- which is how the cross-check
            // reported its SECOND real finding: the AST was missing a
            // `partition_view` that the text had, because the parser read
            // `} else {` as the end of the enclosing `scf.for` body and
            // silently dropped the three ops after the conditional.
            PtoOp::If {
                then_body,
                else_body,
                ..
            } => {
                push_flat(then_body, out);
                if let Some(els) = else_body {
                    push_flat(els, out);
                }
            }
            _ => {}
        }
    }
}
