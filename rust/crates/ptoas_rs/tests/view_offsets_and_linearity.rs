// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Folding a constant partition offset into the address, and the two things
//! that had to move with it.
//!
//! The blanket refusal of a non-origin `pto.partition_view` named its own
//! answer -- "the offset has to be folded into the address, not the tile" --
//! and 316 of the corpus's 435 rank-2 partition views are non-origin. Folding
//! it, on its own, unlocked **nothing**: all six artifacts whose first blocker
//! was a constant offset then hit a second one, three on `pto.get_block_idx`
//! and three on the barrier rule's "a second `pto.tstore` follows an earlier
//! one". So this file covers one change in three parts, because no part of it
//! is worth anything alone:
//!
//!   1. the offset becomes `ptr.wrapping_add(flat)` (`view_geometry.rs`);
//!   2. two stores whose spans cannot overlap no longer block the barrier
//!      drop, because an ORDERING between accesses that never meet is not an
//!      ordering;
//!   3. a tile two ops read is re-loaded, since `Tile` is not `Copy`.
//!
//! Together: `rrope`, `rrope256` and `ds4_rope_halves` lift, 11 of 49 to 14.
//!
//! (3) was found by COMPILING the output for the first time, which is also how
//! the `#![no_core]` prologue defect in all eleven previously-lifting
//! artifacts was found. `tests/lifted_kernels_compile.rs` is that gate.

use ptoas_rs::{convert_pto_to_rust, fused_module, PtoOp};

/// Resolve a fixture or source path against this crate.
///
/// Three shapes arrive here, and the ported version resolved all of them
/// against the repository root of the tree they came from, which this workspace
/// is not: a bare fixture name, this crate's own spelled-out path, and the
/// forward emitter's source.
fn read(rel: &str) -> String {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = if rel.ends_with("mlir_to_pto.rs") {
        crate_dir.join("../rspto_emit/src/mlir_to_pto.rs")
    } else if let Some(inside) = rel.strip_prefix("crates/ptoas_rs/") {
        crate_dir.join(inside)
    } else if rel.starts_with("crates/tile_std") {
        // A sibling crate, resolved against the workspace root.
        crate_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("crate dir has a workspace root")
            .join(rel)
    } else {
        // Fixture: look in both fixture directories, since the corpora they
        // came from were separate trees.
        let name = rel.rsplit('/').next().unwrap_or(rel);
        let a = crate_dir.join("tests/fixtures/barrier").join(name);
        if a.is_file() {
            a
        } else {
            crate_dir.join("tests/fixtures/probes").join(name)
        }
    };
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// The three artifacts this change was for.
const ROPE: [&str; 3] = [
    "benchmarks/ds4flash_layer_runner/rrope.pto",
    "benchmarks/ds4flash_layer_runner/rrope256.pto",
    "crates/mlir_to_pto_tests/l0/h.pto",
];

// ─── The round-trip premise ─────────────────────────────────────────────────

/// PREMISE: the forward emitter still reads a constant GEP back as an offset.
///
/// This is what makes the fold sound rather than merely plausible. A lifted
/// `ptr.wrapping_add(n)` compiles to an `llvm.getelementptr` with a constant
/// index; `mlir_to_pto` records that as a flat element offset and `make_pv`
/// splits it into a `(row, col)` pair again. The pair need not be the one that
/// was read IN -- the same flat address split against a different tile width
/// gives a different pair -- and that is fine, because the address is what
/// both the lift and the tile calls work in.
///
/// Pinned by reading the forward emitter's source, for the same reason
/// `barrier.rs` pins its premises that way: `mlir_to_pto` is a rustc backend
/// in another workspace, so this crate cannot call it, and the fact can move
/// without this crate noticing.
///
/// The contrast that makes it load-bearing: `tile_offset_ptr_f32` is a SAFE
/// WRAPPER in `tile_std::tile` that does exactly this job, and emitting it
/// would be wrong. No `mlir_to_*` under `rustc_codegen_tile` matches
/// `__tile_offset_ptr_f32` -- only the Triton test binaries construct it --
/// so a recompile would fall through to "Unrecognized llvm calls: emit as
/// comment" and the kernel would read the parent's origin. Asserted below, so
/// that the day it IS matched, this test says so.
#[test]
fn the_forward_emitter_still_turns_a_constant_gep_into_a_view_offset() {
    let src = read("crates/rustc_codegen_tile/src/mlir_to_pto.rs");

    // 1. a constant GEP index is recorded as an element offset
    assert!(
        src.contains("ctx.gep_offsets.insert(result, off)"),
        "mlir_to_pto no longer records a constant `llvm.getelementptr` index as an \
         element offset, so a lifted `wrapping_add` would not come back as a view offset"
    );
    // 2. that offset drives the partition view's offsets
    assert!(
        src.contains("fn make_pv(") && src.contains("let row_off = if cols > 0 {"),
        "mlir_to_pto's `make_pv` no longer splits a flat element offset into \
         (row, col); the inverse of this crate's fold has moved"
    );
    assert!(
        src.contains("elem_offset / cols") && src.contains("elem_offset % cols"),
        "the flat-offset split is no longer `/ cols` and `% cols`"
    );
    // 3. and the load path still passes the GEP-derived offset to it
    assert!(
        src.contains("let elem_offset = ctx.resolve_offset(gm_arg);"),
        "mlir_to_pto's load path no longer resolves a GEP offset for the view"
    );
}

/// The MIRROR: the wrapper that looks right is the one that would be dropped.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL's declarations, which
/// are not in this tree yet. Restore it with the Rust input mode.
#[test]
fn the_forward_emitters_do_not_know_tile_offset_ptr() {
    // Both forward emitters in this workspace, not one directory in another
    // tree. Scanning both is the faithful translation and strictly wider: the
    // claim is that NO emitter reads this wrapper.
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut matched = Vec::new();
    for emitter in ["../rspto_emit/src", "../ascendc_emit/src"] {
        let dir = crate_dir.join(emitter);
        for e in std::fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("cannot read {}: {err}", dir.display()))
            .flatten()
        {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let Ok(txt) = std::fs::read_to_string(&p) else {
                continue;
            };
            if txt.contains("__tile_offset_ptr") {
                matched.push(p.file_name().unwrap().to_string_lossy().to_string());
            }
        }
    }
    assert!(
        matched.is_empty(),
        "an emitter now matches `__tile_offset_ptr` ({matched:?}). `tile_offset_ptr_f32` \
         is a safe wrapper that does exactly what this crate's address fold does, and the \
         only reason the fold is spelled as pointer arithmetic is that no emitter read the \
         call -- if one does now, emitting it is the clearer lift"
    );
    // And the wrapper really does exist, so this test is about ROUTING and not
    // about a name nobody declared.
    let tile = read("crates/tile_std/src/tile.rs");
    assert!(
        tile.contains("pub unsafe fn tile_offset_ptr_f32("),
        "tile_std no longer exports `tile_offset_ptr_f32`, so the trade-off this test \
         documents no longer exists"
    );
}

// ─── The three artifacts ────────────────────────────────────────────────────

#[test]
fn the_rope_kernels_lift() {
    for rel in ROPE {
        let rust = convert_pto_to_rust(&read(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        // The halves: one store at the origin, one at the half offset, and the
        // offset is in the ADDRESS.
        assert!(
            rust.contains(".wrapping_add("),
            "{rel}: lifted without folding either half's offset:\n{rust}"
        );
        assert_eq!(
            rust.matches("tile_store_f32").count(),
            2,
            "{rel}: a rope kernel writes both halves:\n{rust}"
        );
        assert!(
            !rust.contains("barrier"),
            "{rel}: emitted a barrier call, which `mlir_to_pto` would delete:\n{rust}"
        );
    }
}

/// The two halves are at DIFFERENT addresses, and the second is the first plus
/// the half width.
///
/// A fold that resolved every offset to 0 would pass `the_rope_kernels_lift`'s
/// store count and every name check, and would be two kernels writing the same
/// 16 elements.
#[test]
fn the_two_halves_go_to_different_addresses() {
    // (file, elements per half)
    for (rel, half) in [(ROPE[0], 16usize), (ROPE[1], 128), (ROPE[2], 32)] {
        let rust = convert_pto_to_rust(&read(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let want = format!(".wrapping_add({half})");
        assert!(
            rust.contains(&want),
            "{rel}: the second half should sit at +{half}:\n{rust}"
        );
        // Exactly two folds: the input's upper half and the output's.
        assert_eq!(
            rust.matches(&want).count(),
            2,
            "{rel}: expected the input and the output halves to be folded, no more:\n{rust}"
        );
    }
}

// ─── Part 2: disjoint stores no longer block the barrier drop ───────────────

/// Two stores that overlap DO still block it.
///
/// Built by perturbing a real artifact, because the corpus holds no kernel
/// that writes one window twice: the second store's offset is moved onto the
/// first, so the two spans coincide exactly.
#[test]
fn two_stores_to_the_same_window_still_block_the_drop() {
    let src = read(ROPE[2]);
    assert!(convert_pto_to_rust(&src).is_ok(), "baseline must lift");
    // `%pto19` is the output's upper half at `[%c0, %c32]`; aim it at the
    // origin so both stores write elements 0..32.
    let patched = src.replace(
        "%pto19 = pto.partition_view %pto17, offsets = [%c0, %c32]",
        "%pto19 = pto.partition_view %pto17, offsets = [%c0, %c0]",
    );
    assert_ne!(patched, src, "the perturbation did not apply");
    let err = convert_pto_to_rust(&patched).expect_err("two stores to one window must refuse");
    assert!(
        err.contains("second `pto.tstore`") && err.contains("not known to miss"),
        "expected the store-to-store refusal, got: {err}"
    );
}

/// Two stores that ABUT are disjoint; one element of overlap is not.
///
/// The boundary, because `off + len <= other.off` is exactly where an
/// off-by-one in the span would hide: at `[%c0, %c31]` the two stores share
/// element 31, and at `[%c0, %c32]` they do not share anything.
#[test]
fn the_disjointness_boundary_is_one_element() {
    let src = read(ROPE[2]);
    let overlap = src.replace(
        "%pto19 = pto.partition_view %pto17, offsets = [%c0, %c32]",
        "%pto19 = pto.partition_view %pto17, offsets = [%c0, %c31]",
    );
    assert_ne!(overlap, src, "the perturbation did not apply");
    // `%c31` is not a constant the artifact declares, so declare it beside the
    // others rather than letting the offset go unresolvable -- an unresolved
    // offset refuses for a different reason and would pass this test wrongly.
    let overlap = overlap.replace(
        "    %c32 = arith.constant 32 : index",
        "    %c32 = arith.constant 32 : index\n    %c31 = arith.constant 31 : index",
    );
    let err = convert_pto_to_rust(&overlap).expect_err("one element of overlap must refuse");
    assert!(
        err.contains("second `pto.tstore`"),
        "expected the store-to-store refusal at one element of overlap, got: {err}"
    );
}

/// Two stores through DIFFERENT pointers still block it.
///
/// Two distinct pointer arguments may alias at the call site and nothing in
/// the module says they do not, so "different base" is not "disjoint".
#[test]
fn two_stores_through_different_pointers_still_block_the_drop() {
    let src = read(ROPE[2]);
    // Send the second store to `%x` (`a0`, the input) instead of `%o`.
    let patched = src.replace(
        "%pto19 = pto.partition_view %pto17, offsets = [%c0, %c32]",
        "%pto19 = pto.partition_view %pto0, offsets = [%c0, %c32]",
    );
    assert_ne!(patched, src, "the perturbation did not apply");
    let err = convert_pto_to_rust(&patched)
        .expect_err("two stores through different pointers must refuse");
    assert!(
        err.contains("second `pto.tstore`"),
        "expected the store-to-store refusal, got: {err}"
    );
}

// ─── Part 3: re-loading a tile two ops read ─────────────────────────────────

/// The pass fires on the real artifacts, and the COUNT comes from the PTO.
///
/// `ds4_rope_halves` reads each of its four input tiles exactly twice -- the
/// two products of each half name all four between them -- so four re-loads
/// and no more. "At least one re-load" would pass while three tiles were
/// silently moved twice, and the emitter would then refuse the file rather
/// than lift it, so a bare count really is the assertion.
#[test]
fn each_doubly_read_input_is_reloaded_once() {
    for rel in ROPE {
        let src = read(rel);
        // Derived from the PTO text, not written down: a tile read twice by
        // ops the lift emits gets exactly one extra load.
        let m = fused_module(&src).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let f = &m.funcs[0];
        let reloads = f
            .ops
            .iter()
            .filter(|op| match op {
                PtoOp::TLoad { tile, .. } => tile.0.contains("_reload"),
                _ => false,
            })
            .count();
        assert_eq!(
            reloads, 4,
            "{rel}: a rope kernel reads x_lo, x_hi, cos and sin twice each, so four \
             re-loads; got {reloads}"
        );
        // Every re-load is allocated as well, or the emitter has no type for it.
        let allocs = f
            .ops
            .iter()
            .filter(|op| match op {
                PtoOp::AllocTile { dst, .. } => dst.0.contains("_reload"),
                _ => false,
            })
            .count();
        assert_eq!(allocs, reloads, "{rel}: a re-loaded tile was not allocated");
    }
}

/// A re-load reads the SAME view the original load did.
///
/// The failure this excludes is a re-load pointed at the wrong half: the lift
/// would compile, every count above would hold, and `o_hi` would be computed
/// from `x_lo` twice.
#[test]
fn a_reload_reads_the_view_its_original_read() {
    let m = fused_module(&read(ROPE[2])).expect("lift");
    let f = &m.funcs[0];
    let mut by_tile = std::collections::BTreeMap::new();
    for op in &f.ops {
        if let PtoOp::TLoad { view, tile, .. } = op {
            by_tile.insert(tile.0.clone(), view.0.clone());
        }
    }
    let mut checked = 0;
    for (tile, view) in &by_tile {
        let Some(orig) = tile.strip_suffix("_reload2") else {
            continue;
        };
        let want = by_tile
            .get(orig)
            .unwrap_or_else(|| panic!("`{tile}` re-loads `{orig}`, which has no load"));
        assert_eq!(view, want, "`{tile}` reads `{view}`, but `{orig}` read `{want}`");
        checked += 1;
    }
    assert_eq!(checked, 4, "expected four re-loads to check, saw {checked}");
}

/// A tile that is NOT loaded from memory cannot be re-loaded, and is refused.
///
/// The one case the pass must not guess at: there is nothing to repeat for a
/// compute result, and the tile API has no copy. Perturbed from a real
/// artifact by making the second half's product read a computed tile twice.
#[test]
fn a_doubly_read_compute_result_is_rematerialised() {
    // IT USED TO REFUSE, honestly: `Tile` is not `Copy` and the tile API has no copy, so a
    // compute result read twice could not be passed twice. The lift now REPRODUCES such a
    // value from its producer chain down to the loads, which is why this test asserts the
    // opposite of what it used to -- and the old assertion is the reason the change could
    // not be committed without touching this file.
    //
    // `%pto13` is `x_lo*cos - x_hi*sin`, stored once. Make the closing add read it instead
    // of `%pto15`, so a compute result is read by two ops.
    let src = read(ROPE[2]);
    let patched = src.replace("ins(%pto14, %pto15 :", "ins(%pto13, %pto13 :");
    assert_ne!(patched, src, "the perturbation did not apply");
    let out = convert_pto_to_rust(&patched).expect("a doubly-read result is reproduced");
    // The evidence that it was reproduced rather than shared: the subtraction that produces
    // %pto13 appears more than once in the emitted Rust.
    let subs = out.matches("tile_sub_f32").count();
    assert!(
        subs >= 2,
        "expected the producer to be recomputed for the second reader, got {subs} subtraction(s)"
    );
}

/// A tile read after a `pto.tstore` is not re-loaded either.
///
/// Past a store this crate will not say what memory holds: two distinct
/// pointer arguments may alias at the call site. Checked by moving the
/// kernel's first store to the front, ahead of every read.
#[test]
fn a_read_after_a_store_is_not_reloaded() {
    let src = read(ROPE[2]);
    let store = src
        .lines()
        .find(|l| l.contains("pto.tstore ins(%pto13"))
        .expect("the artifact stores the lower half")
        .to_string();
    // Hoist it above the first load. It stores a tile that does not exist yet,
    // which the parser does not mind -- the point is only the POSITION, which
    // is all the re-load rule reads.
    let first_load = src
        .lines()
        .find(|l| l.contains("pto.tload"))
        .expect("the artifact loads")
        .to_string();
    let patched = src
        .replace(&format!("{store}\n"), "")
        .replace(&first_load, &format!("{}\n{first_load}", store.trim_start()));
    assert_ne!(patched, src, "the perturbation did not apply");

    // Asserted on the PASS, not on the refusal text. With a store ahead of
    // every load the barrier rule refuses first, for its own load-after-store
    // reason, and that message would be satisfied whether or not the re-load
    // happened -- so reading it would test nothing about this rule.
    let m = fused_module(&patched).expect("the module still parses");
    let reloads = m.funcs[0]
        .ops
        .iter()
        .filter(|op| matches!(op, PtoOp::TLoad { tile, .. } if tile.0.contains("_reload")))
        .count();
    assert_eq!(
        reloads, 0,
        "a tile read after a `pto.tstore` was re-loaded anyway; past a store this crate \
         cannot say what memory holds"
    );
    // And the baseline re-loads four, so the zero above is the perturbation's
    // doing and not a pass that stopped running.
    let base = fused_module(&src).expect("baseline parses");
    assert_eq!(
        base.funcs[0]
            .ops
            .iter()
            .filter(|op| matches!(op, PtoOp::TLoad { tile, .. } if tile.0.contains("_reload")))
            .count(),
        4,
        "the baseline no longer re-loads, so this test's zero means nothing"
    );
    convert_pto_to_rust(&patched).expect_err("and the file is refused, not lifted");
}

/// A multi-row window narrower than its parent is refused -- but NOT for the
/// reason this test was written for.
///
/// RE-GROUNDED 2026-11-09, name and conclusion kept, mechanism corrected. The
/// old docstring said the rows are `parent_cols` apart while the tile call
/// reads `rows * cols` CONTIGUOUS ones, so the lift would read past every row
/// but the first. The second clause is no longer true: the lift emits
/// `tile_load_strided_f32<ROWS, COLS, ROW_STRIDE>`, which reads exactly this
/// geometry, and a matched window now lifts (see
/// `a_multi_row_window_lifts_when_the_tile_shares_its_geometry`).
///
/// What this perturbation actually builds is a window whose TILE disagrees with
/// it -- parent and window are widened and narrowed, the tile is left at
/// 1x4096 -- so the refusal that fires is the extent-agreement guard: two
/// candidate COLS and no ground to pick one. That is a real case worth keeping,
/// and it is the only coverage of it; only the expected TEXT moved.
///
/// The expectation was NOT weakened to make a failure pass: the assertion is
/// still `expect_err`, and the strided case it was thought to cover is now
/// asserted positively in the companion test above, including that ROW_STRIDE
/// is the parent's width and not the window's.
///
/// 26 corpus instances, all in the GMM cube files -- which are refused earlier
/// for other reasons, so this is perturbed from an artifact that does lift.
#[test]
fn a_multi_row_window_narrower_than_its_parent_is_refused() {
    let src = read("crates/mlir_to_pto_tests/l0/e.pto");
    assert!(convert_pto_to_rust(&src).is_ok(), "baseline must lift");
    // The artifact's view is 1x4096 with the partition taking all of it. Make
    // the parent 4 rows of 4096 and the window 4 rows of 1024: a window whose
    // rows are 4096 apart, read as 4096 contiguous elements.
    let patched = src
        .replace(
            "shape = [%c1, %c4096], strides = [%c4096, %c1]",
            "shape = [%c4, %c4096], strides = [%c4096, %c1]",
        )
        .replace("sizes = [%c1, %c4096]", "sizes = [%c4, %c1024]");
    assert_ne!(patched, src, "the perturbation did not apply");
    let patched = patched.replace(
        "    %c4096 = arith.constant 4096 : index",
        "    %c4096 = arith.constant 4096 : index\n    \
         %c4 = arith.constant 4 : index\n    %c1024 = arith.constant 1024 : index",
    );
    let err = convert_pto_to_rust(&patched)
        .expect_err("a window whose tile disagrees with its extents must not lift");
    assert!(
        err.contains("the strided wrapper takes one COLS for both"),
        "the refusal must say why the two extents cannot be reconciled: {err}"
    );
}

/// A multi-row window narrower than its parent LIFTS when the tile matches it.
///
/// The companion of the refusal above, and the case that one's perturbation does
/// NOT actually reach: it widens the parent and narrows the window but leaves
/// the TILE at 1x4096, so what it exercises is a tile/window disagreement, not a
/// stride. (Verified 2026-11-09: that perturbation is still refused, now by the
/// extent-agreement guard rather than by a contiguity claim.)
///
/// Uses a purpose-built fixture rather than a perturbation of `e.pto`. Narrowing
/// that artifact's tile breaks the rms-norm chain it belongs to, so the lift
/// refuses on an unfused `pto.tsqrt` and the window is never reached -- a test
/// that would have passed for the wrong reason had it only asserted `is_err`.
///
/// Added 2026-11-09 with strided-window support. Before it, `Window` discarded
/// the parent's row stride and this lift refused, calling the lift's own data
/// loss a hardware constraint.
#[test]
fn a_multi_row_window_lifts_when_the_tile_shares_its_geometry() {
    let src = read("strided_window_matched.pto");
    let rust = convert_pto_to_rust(&src)
        .unwrap_or_else(|e| panic!("a window whose tile shares its geometry must lift: {e}"));
    // The whole generic list, not just the function name: ROW_STRIDE must be the
    // PARENT's width (4096) and never the window's (1024). A lift that passed
    // 1024 would compile and silently read the wrong rows.
    assert!(
        rust.contains("tile_load_strided_f32::<4, 1024, 4096>"),
        "the parent's row stride did not reach the load: {rust}"
    );
    assert!(
        rust.contains("tile_store_strided_f32::<4, 1024, 4096>"),
        "the store did not take the stride either: {rust}"
    );
}

/// A fused row-expansion over a 1x1 operand lifts as a tile-by-scalar.
///
/// `pto.trowexpandmul` with a reduction operand of LOGICAL extent 1x1 is not a
/// broadcast at all -- there is nothing to broadcast -- it is
/// `tile_scale_f32(a, s)`. The scalar is read with `tile_get_value_f32`, the
/// public wrapper added 2026-11-09 over an intrinsic `tile_std` had declared
/// since 2026-11-08 and `mlir_to_cpp` had implemented for longer. Nothing could
/// reach it because `Tile::buf_id` is `pub(crate)`.
///
/// Three things are asserted, and the middle one is the one that bites.
#[test]
fn a_row_expansion_over_a_one_by_one_operand_lifts_as_a_scalar_op() {
    let src = read("scalar_row_expand.pto");
    let rust = convert_pto_to_rust(&src)
        .unwrap_or_else(|e| panic!("a 1x1 reduction operand must lift as a scalar op: {e}"));
    // 1. The op became a scalar multiply, not a broadcast.
    assert!(
        rust.contains("tile_scale_f32::<1, 512>(t0, tile_get_value_f32(s0, 0))"),
        "the expansion did not collapse to a tile-by-scalar: {rust}"
    );
    // 2. The scalar tile is loaded at its ALLOCATED extent (1x8 = 32 bytes),
    // never its logical 1x1. A 1x1 f32 load is 4 bytes, which this part copies
    // wrongly WITHOUT faulting (KNOWN_ISSUES, severity high) and which
    // `mlir_to_cpp` refuses outright as not a multiple of 32. A lift that used
    // the logical extent here would pass rustc and be wrong on device.
    assert!(
        rust.contains("tile_load_f32::<1, 8>"),
        "the scalar tile was not loaded at its 32-byte allocated extent: {rust}"
    );
    assert!(
        !rust.contains("tile_load_f32::<1, 1>"),
        "a 1x1 f32 load is 4 bytes and must never be emitted: {rust}"
    );
    // 3. The index is FLAT, so element 0 is the single valid element whichever
    // way the block was allocated (1x8 and 8x1 both occur in the corpus).
    assert!(
        rust.contains("tile_get_value_f32(s0, 0)"),
        "the scalar was not read at flat index 0: {rust}"
    );
}

/// `tile_reads` and `tile_reads_mut` list the same operands.
///
/// Rust cannot share one match arm between `&` and `&mut`, so the two tables
/// are written twice. If they disagree, `linearize` rewrites a read the
/// emitter's linearity check does not count, or counts one it cannot rewrite --
/// and either way the output is Rust that does not compile.
#[test]
fn tile_reads_and_tile_reads_mut_agree() {
    // `ast::Ssa` explicitly: the crate has TWO `Ssa` types and the bare
    // `ptoas_rs::Ssa` is the Stage-2 plan's, not this AST's.
    use ptoas_rs::ast::Ssa;
    use ptoas_rs::{BinKind, DType, PtoType, RowKind, UnKind};
    let t = || Ssa("%a".into());
    let u = || Ssa("%b".into());
    let v = || Ssa("%c".into());
    let mut ops = vec![
        PtoOp::BinOp { kind: BinKind::Add, dst: v(), a: t(), b: u() },
        PtoOp::UnOp { kind: UnKind::Exp, dst: v(), src: t() },
        PtoOp::RowReduce { kind: RowKind::Sum, dst: v(), src: t() },
        PtoOp::RowExpandSub { dst: v(), src: t(), red: u() },
        PtoOp::RowExpandDiv { dst: v(), src: t(), red: u() },
        PtoOp::RowExpandMul { dst: v(), src: t(), red: u() },
        PtoOp::RowExpand { dst: v(), src: t() },
        PtoOp::TMatmul { dst: v(), lhs: t(), rhs: u() },
        PtoOp::TMov { dst: v(), src: t() },
        PtoOp::Softmax { dst: v(), src: t() },
        PtoOp::RmsNorm { dst: v(), src: t(), eps: "1.0".into() },
        PtoOp::Cvt { dst: v(), src: t(), sat_mode: None },
        PtoOp::GatherB { dst: v(), src: t(), idx: u() },
        PtoOp::Transpose { dst: v(), src: t(), scratch: u() },
        PtoOp::TStore { tile: t(), view: u(), view_ty: PtoType::Ptr(DType::F32) },
        PtoOp::TLoad { view: u(), tile: t(), view_ty: PtoType::Ptr(DType::F32) },
        PtoOp::AllocTile { dst: v(), ty: PtoType::Ptr(DType::F32) },
        PtoOp::Barrier { pipe: "PIPE_ALL".into() },
    ];
    // Any op that reads a tile at all has to appear above, or this test is
    // comparing two tables over a sample that misses the disagreement.
    let reading = ops.iter().filter(|o| !o.tile_reads().is_empty()).count();
    assert!(
        reading >= 14,
        "only {reading} of the sampled ops read a tile; the sample is too thin to \
         compare the two tables"
    );
    for op in &mut ops {
        let want: Vec<String> = op.tile_reads().iter().map(|s| s.0.clone()).collect();
        let got: Vec<String> = op.tile_reads_mut().iter().map(|s| s.0.clone()).collect();
        assert_eq!(want, got, "the two read tables disagree for {op:?}");
    }
}

// ─── The mutation survivors ─────────────────────────────────────────────────
//
// Each of the three below killed nothing on the first mutation round, because
// each guards a condition no corpus artefact reaches TODAY -- a third store is
// in the GMM kernels, a region is `scf.for`, and a tile written twice is the
// matvec row loop, which now LIFTS (2026-11-09: the refusal that stood in its
// way was false; see the first test's body). A condition that stopped working
// would therefore look exactly like one that was never needed, which is what
// the perturbations are for. Each is perturbed from an artifact that lifts.

/// A tile WRITTEN by two ops is not re-loaded.
///
/// `loaded` keeps one view per tile, so a buffer that is re-`tload`ed from a
/// second window -- what every matvec row loop does -- would be re-loaded from
/// whichever window was recorded last. That is a wrong kernel at exit 0, and
/// the only thing standing between the pass and it is the write count.
#[test]
fn a_tile_written_by_two_ops_is_not_reloaded() {
    let src = read(ROPE[2]);
    assert!(convert_pto_to_rust(&src).is_ok(), "baseline must lift");
    // `%pto3` is the lower half of `x`. Load the UPPER half into it as well,
    // so two ops write it and the two windows differ.
    let lower = src
        .lines()
        .find(|l| l.contains("pto.tload ins(%pto1 ") && l.contains("outs(%pto3 "))
        .expect("the artifact loads the lower half into %pto3")
        .to_string();
    let second = lower.replace("ins(%pto1 ", "ins(%pto2 ");
    assert_ne!(second, lower, "the second load is identical to the first");
    let patched = src.replace(&lower, &format!("{lower}\n{second}"));
    assert_ne!(patched, src, "the perturbation did not apply");

    let m = fused_module(&patched).expect("parses");
    let reloads = m.funcs[0]
        .ops
        .iter()
        .filter(|op| matches!(op, PtoOp::TLoad { tile, .. } if tile.0.contains("_reload")))
        .count();
    assert!(
        !m.funcs[0]
            .ops
            .iter()
            .any(|op| matches!(op, PtoOp::TLoad { tile, .. } if tile.0.starts_with("%pto3_reload"))),
        "a tile two ops write was re-loaded; its recorded window is only one of the two \
         ({reloads} re-loads in all)"
    );
    // THE REFUSAL IS GONE AND THE CONTROL ABOVE IS WHAT THIS TEST IS FOR.
    //
    // Two reasons were asserted here in turn and both were retired. First "the
    // handle would be moved twice", until `Tile` became `Copy` on 2026-11-09.
    // Then "`%pto3` is written by two ops, so its name denotes two different
    // values" -- and that one was MINE and it was false. Rust `let` shadowing is
    // positional; this emitter walks the ops in PTO order into one flat statement
    // list (`PtoOp::For` and `PtoOp::If` both refuse without emitting a body), so
    // a read before the second `let` resolves to the first binding and a read
    // after it to the second. That is what overwriting a buffer means in PTO, and
    // 10 corpus artefacts were refused by the claim that it is not.
    //
    // So the perturbed kernel LIFTS, and what is asserted is that it lifts
    // FAITHFULLY: both `%pto3` windows appear, in op order, as two bindings of
    // one name. The first load is dead in the perturbed program -- the PTO writes
    // %pto3 twice before reading it -- and the lift reproducing that is correct,
    // not a kernel it repaired.
    let rs = convert_pto_to_rust(&patched).expect("the perturbed artifact lifts");
    let loads: Vec<&str> = rs
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("let t3 = tile_load_f32::<1, 32>("))
        .collect();
    assert_eq!(
        loads.len(),
        2,
        "the two writes to `%pto3` must appear as two bindings of one name, in op \
         order; got {loads:?} in:\n{rs}"
    );
    assert_ne!(
        loads[0], loads[1],
        "the two bindings must read the two DIFFERENT windows the perturbation \
         created -- identical lines would mean the second write was lifted from \
         the first write's recorded view, which is the `loaded` hazard this \
         test's first half rules out"
    );
    // And the reads resolve to the LAST write, because in the perturbed PTO both
    // `pto.tmul` reads come after it. A reload would have broken that.
    assert!(
        rs.contains("tile_mul_f32::<1, 32>(t3, t7)"),
        "the first mul must read the shadowed `t3`, not a reload: \n{rs}"
    );
}

/// A store is compared against EVERY earlier store, not just the last.
///
/// Three stores where the third overlaps the first and misses the second: a
/// rule that looks only at the most recent store calls that disjoint and drops
/// the barrier that orders two writes to the same 32 elements.
#[test]
fn a_store_is_compared_against_every_earlier_store() {
    let src = read(ROPE[2]);
    assert!(convert_pto_to_rust(&src).is_ok(), "baseline must lift");

    // A fourth loaded tile, cloned from the `sin` load so its type and view
    // type come from the artifact rather than from a literal here.
    let pv = src.lines().find(|l| l.contains("%pto9 = pto.partition_view")).unwrap();
    let alloc = src.lines().find(|l| l.contains("%pto10 = pto.alloc_tile")).unwrap();
    let load = src.lines().find(|l| l.contains("outs(%pto10 ")).unwrap();
    let extra = format!(
        "{}\n{}\n{}",
        pv.replace("%pto9 =", "%pto90 ="),
        alloc.replace("%pto10 =", "%pto91 ="),
        load.replace("ins(%pto9 ", "ins(%pto90 ").replace("outs(%pto10 ", "outs(%pto91 "),
    );
    // Third store, to the window the FIRST store already wrote (`%pto18`, the
    // output's lower half). The second store is at `%pto19`, which it misses.
    let last_store = src.lines().find(|l| l.contains("pto.tstore ins(%pto16")).unwrap();
    let third = last_store
        .replace("ins(%pto16 ", "ins(%pto91 ")
        .replace("outs(%pto19 ", "outs(%pto18 ");
    let patched = src
        .replace(load, &format!("{load}\n{extra}"))
        .replace(last_store, &format!("{last_store}\n{third}"));
    assert_ne!(patched, src, "the perturbation did not apply");

    let err = convert_pto_to_rust(&patched)
        .expect_err("a third store overlapping the first must block the barrier drop");
    assert!(
        err.contains("second `pto.tstore`") && err.contains("%pto18"),
        "expected the store-to-store refusal naming the window written twice, got: {err}"
    );
}

/// A function holding a region is left entirely alone by the pass.
///
/// `view_windows` does not look inside a region and neither do the read counts,
/// so every number the pass works from would be taken over part of the
/// function -- and a tile the region writes could be re-loaded from a window
/// the region has since replaced. Both region arms refuse, so the emitter never
/// shows this; the pass's own output is the only observable.
#[test]
fn a_function_with_a_region_is_not_linearized() {
    let src = read(ROPE[2]);
    let base = fused_module(&src).expect("baseline parses");
    let count = |m: &ptoas_rs::PtoModule| {
        m.funcs[0]
            .ops
            .iter()
            .filter(|op| matches!(op, PtoOp::TLoad { tile, .. } if tile.0.contains("_reload")))
            .count()
    };
    assert_eq!(count(&base), 4, "the baseline no longer re-loads, so this test is vacuous");

    // Wrap the closing `tadd` in a loop, keeping it in place by reusing
    // constants the artifact already declares for the bounds.
    let tail = src
        .lines()
        .find(|l| l.contains("pto.tadd ins(%pto14, %pto15"))
        .expect("the artifact adds the upper half's two products")
        .to_string();
    let patched = src.replace(
        &tail,
        &format!(
            "    scf.for %r = %c0 to %c1 step %c1 {{\n    {}\n    }}",
            tail.trim()
        ),
    );
    assert_ne!(patched, src, "the perturbation did not apply");

    let m = fused_module(&patched).expect("parses");
    assert_eq!(
        count(&m),
        0,
        "a function with a region was linearized from counts taken over its top level only"
    );
}

// ─── Closing the `pto.get_block_idx` tier ───────────────────────────────────
//
// 23 artifacts refuse there, the largest first-blocker count in the corpus,
// and the refusal used to read as an import fix. These pin the two facts that
// make it a closed tier rather than a pending one, both of which live in other
// crates' source and can move without this one noticing.

/// PREMISE: the forward emitter DELETES `get_block_idx` input lines.
///
/// The same wall `pto.barrier` hits, and for the same reason: a call the
/// forward emitter drops cannot survive a recompile, so emitting one is the
/// guess. If `mlir_to_pto` grows an arm that lowers the call, the lift can
/// express a per-core row index and should -- this is how that becomes visible.
#[test]
fn the_forward_emitter_still_deletes_block_index_queries() {
    let src = read("crates/rustc_codegen_tile/src/mlir_to_pto.rs");
    let i = src
        .find("if line.contains(\"get_block_idx\") {")
        .expect("mlir_to_pto no longer has a `get_block_idx` input guard at all");
    let arm = &src[i..(i + 400).min(src.len())];
    assert!(
        arm.contains("ops.push(") && arm.contains("// block index"),
        "the `get_block_idx` guard no longer replaces the line with a comment: {arm:?}"
    );
    assert!(
        arm.contains("continue"),
        "the `get_block_idx` line is matched but no longer `continue`d: {arm:?}"
    );
}

/// PREMISE: the grid-stride loop is synthesised from a SHAPE, not translated.
///
/// This is why lifting the loop would be re-encoding something the forward
/// emitter re-derives. The loop, the barriers, the allocs and the op chain of
/// a `ds4_q8mv_*` artifact all come out of one `llvm.call`.
#[test]
fn the_grid_stride_loop_is_synthesised_by_the_forward_emitter() {
    let src = read("crates/rustc_codegen_tile/src/mlir_to_pto.rs");
    assert!(
        src.contains("[AIV grid]"),
        "mlir_to_pto no longer writes the grid loop that these artifacts carry"
    );
    // Guarded by a shape test, with the bounds built from the block query it
    // emits itself -- nothing in the input decides either.
    assert!(
        src.contains("if ne0 > 1 {"),
        "the multi-row grid loop is no longer gated on the output-row count, so it may \
         no longer be derived purely from the kernel's shape"
    );
    assert!(
        // A raw string: the emitter's source carries the quotes ESCAPED,
        // because it is writing MLIR text from a Rust string literal.
        src.contains(r#"\"pto.get_block_idx\"() : () -> i64"#),
        "mlir_to_pto no longer emits `pto.get_block_idx` itself, which is where the \
         artifacts' copies come from"
    );
}

/// PREMISE: one name, two contracts.
///
/// `tile_std` declares the Metal/MSL shape (u32 handles + 11 dimensions);
/// `mlir_to_pto` matches four pointers and two `i32`. A lifted call would
/// either not compile or build a different kernel, which is the real reason
/// these artifacts have no lift. `codegen_tests/tests/fixture_operand_arity.rs`
/// records the narrow form as deliberate on the forward side; this records
/// what it costs the reverse one.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL's declarations, which
/// are not in this tree yet. Restore it with the Rust input mode.
#[test]
fn the_matvec_intrinsic_has_two_different_contracts() {
    let tile = read("crates/tile_std/src/tile.rs");
    let i = tile
        .find("pub fn __tile_mul_mv_id_q8_0_f32(")
        .expect("tile_std no longer declares the routed q8_0 matvec");
    let decl = &tile[i..i + tile[i..].find(';').expect("declaration is unterminated")];
    let params = decl.matches(": u32").count();
    assert_eq!(
        params, 15,
        "tile_std's `__tile_mul_mv_id_q8_0_f32` no longer takes 15 `u32`s; if it now takes \
         pointers it may match the PTO emitter's form, and these artifacts become liftable"
    );
    assert!(
        !decl.contains("*const") && !decl.contains("*mut"),
        "tile_std's declaration now takes pointers: {decl}"
    );

    let pto = read("crates/rustc_codegen_tile/src/mlir_to_pto.rs");
    assert!(
        pto.contains("__tile_mul_mv_id_q8_0_f32(%arg0, %arg1, %arg2, %arg3, %ne00, %ne0)"),
        "the PTO emitter's own golden input no longer uses the 6-argument form, so the \
         two contracts may have been reconciled"
    );
}
