// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `__tile_quantize_f32_i8`'s `scale` operand means three different things.
//!
//! `quant_i8_v41_8x5120` sits at `static-verified` and its only remaining
//! blocker reads like one missing arm: "mlir_to_cpp has no arm for
//! `__tile_quantize_f32_i8`". Writing that arm is a dozen lines against an
//! existing template -- `__tile_dequantize_i8_f32`, the exact inverse, is
//! implemented and device-verified in the same file. Measured 2026-11-09, the
//! arm cannot be written yet, because the layers do not agree on what they would
//! be implementing:
//!
//! ```text
//! tile_std declaration   scale: f32            -- a SCALAR
//! mlir_to_pto arm        ctx.get_tile(scale)   -- a TILE, divided elementwise
//!                                                 against a broadcast divisor
//! corpus .tile           __tile_load_f32(%arg1, 1, 1)      -- a 1x1 TILE
//! fixed/ .tile           __tile_get_value_f32(%sct, 0)     -- a SCALAR
//! mlir_to_cpp            (no arm)
//! ```
//!
//! A scalar `Muls` and a broadcast tile divide are **different programs**, not
//! two spellings of one. An AscendC arm written to the declaration would compute
//! something the stored `.pto` does not, and the two columns of this corpus exist
//! precisely to disagree out loud when that happens.
//!
//! # Which reading is load-bearing today
//!
//! The PTO one. `mlir_to_pto` accepts the corpus source and the stored artefact
//! carries the tile reading:
//!
//! ```text
//! %pto4 = pto.partition_view %pto3, offsets = [%c0, %c0], sizes = [%c1, %c1]
//!           : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x1xf32>
//! pto.tload ins(%pto4 : !pto.partition_tensor_view<1x1xf32>)
//!           outs(%pto5 : !pto.tile_buf<..., rows=1, cols=8, ...>)
//! ```
//!
//! A 1x1 f32 view -- four bytes -- read into a tile already padded to one block.
//! That is the SHAPE of the defect measured on device on 2026-11-09, where a
//! sub-block GM->UB copy neither faulted nor delivered its element. Whether this
//! `tload` actually moves four bytes or thirty-two is **UNKNOWN and not
//! determinable from the text**: ptoas emits `TLOAD(v31, v30)` with the extent
//! inside the descriptors, and `tload` on a2/a3 has no extent check at all, so
//! the assembler will not refuse it either way. It is a suspicion with a cheap
//! resolution -- run it -- and it is recorded as a suspicion.
//!
//! Updated 2026-11-09: the MECHANISM is now known and it is the TOOLCHAIN, not
//! the silicon. bisheng encodes the DMA length as the byte count rounded DOWN to
//! a multiple of 32, so a four-byte copy is a length-0 transfer -- the instruction
//! is emitted and the engine is never asked to move anything.
//!
//! An earlier version of this paragraph said the AscendC side was REPAIRED by a
//! `DataCopyPad` branch and that the doubt was removed from that column. **That is
//! retracted.** The pad does deliver element 0, but it fills the whole 32-byte
//! destination block with element 0's value (cells 1, 2, 4 and 7 all read it, 12
//! of 12 against a poison prologue; `isPad = true` does not change it), so it
//! trades a silent short read for a silent block-wide write. The pad was withdrawn
//! and `mlir_to_cpp` refuses a partial-block constant load again.
//!
//! So the doubt is NOT removed from either column. AscendC refuses the 1x1 scale
//! load outright, which is honest but leaves this kernel's scale unreadable on
//! that path; and the PTO column's `tload` carries its extent in descriptors that
//! no textual `% 32` rule can inspect. The `fixed/` sibling's whole-block load is
//! the only form that is sound on both, which is one more reason the owner
//! question below is the real blocker.
//!
//! The `fixed/` sibling loads one whole block and reads element 0, which matches
//! the declaration and clears that doubt. It is blocked on `mlir_to_pto` having
//! no `__tile_get_value_f32` arm, and iteration 56 measured why that arm is not
//! cheap: a tile-to-scalar read has no op in either the emitter's PTO vocabulary
//! or ptoas's.
//!
//! # The hazard any AscendC arm must honour
//!
//! `mlir_to_pto`'s arm carries a probe-verified fact that the AscendC side
//! records nowhere, and the naive arm violates it:
//!
//! > There is NO f32->s8 hardware convert on a2a3. The vconv table has f16->s8
//! > (`vconv_f162s8`) but no `vconv_f322s8`; ptoas nonetheless ACCEPTS a direct
//! > f32->si8 pto.tcvt and emits a TCVT for it, which on device produces garbage
//! > and can raise a vector core exception (507035).
//!
//! So `Cast(int8_dst, float_src, CAST_RINT, n)` is the obvious arm and it is
//! wrong. The route is f32 -> f16 -> s8, exact because `|y| <= 127` is inside
//! f16's integer-exact range. The inverse arm in `mlir_to_cpp` already takes the
//! mirror of this detour for its own direction ("dav-c220 Cast has no int8->float
//! member: go int8 -> half -> float"), so the file knows the shape of the
//! constraint and not this instance of it.
//!
//! # What this gate is for
//!
//! To stop the arm being written on a guess, and to make the owner's decision
//! visible: **what is `scale`?** Resolving it also decides whether the corpus
//! source or its `fixed/` sibling is the real one, and it is the same owner
//! question as `__tile_get_value_f32`'s. Until then `quant_i8_v41_8x5120` stays
//! at `static-verified`, which claims ptoas assembles it and claims nothing about
//! what it computes.
//!
//! Deliberately NOT asserted: that the stored `.pto` is wrong. It may be right.
//! The claim here is only that three layers disagree and that nobody has run it.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the rust tree must be reachable")
}

fn read(rel: &str) -> String {
    let p = root().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// The declaration says SCALAR.
#[test]
fn the_declaration_says_the_scale_is_a_scalar() {
    let t = read("crates/tile_std/src/tile.rs");
    let i = t
        .find("pub fn __tile_quantize_f32_i8(")
        .expect("__tile_quantize_f32_i8 must still be declared; if it was removed, this whole \
                 file's subject is gone");
    let decl = &t[i..i + t[i..].find(';').unwrap_or(0) + 1];
    assert!(
        decl.contains("scale: f32"),
        "the declared scale operand is no longer `f32`. If it became a `u32` buf id the \
         declaration now AGREES with the PTO arm, the disagreement this file pins is \
         resolved, and the AscendC arm can be written to it -- delete this file and write \
         it.\n{decl}"
    );
}

/// The PTO arm says TILE, and divides rather than multiplies.
#[test]
fn the_pto_arm_reads_the_scale_as_a_tile_and_divides_by_it() {
    let t = read("crates/rspto_emit/src/mlir_to_pto.rs");
    let i = t
        .find("fn translate_quantize_pto(")
        .expect("translate_quantize_pto must exist -- it is what makes the PTO column emit");
    let body = &t[i..i + 4000.min(t.len() - i)];
    assert!(
        body.contains("get_tile(scale_ssa)"),
        "the PTO arm no longer resolves `scale` as a tile. If it now takes a scalar it \
         agrees with the declaration and this file's premise is stale."
    );
    // The DIRECTION matters as much as the shape: the inverse arm in mlir_to_cpp
    // multiplies, so an arm copied from it without reading this would compute the
    // reciprocal of the intended result.
    assert!(
        t[i.saturating_sub(200)..i].contains("src / scale"),
        "the doc line above `translate_quantize_pto` no longer states the direction \
         (`round(src / scale)`). That direction is the one thing a reader copying the \
         `Muls`-based inverse arm would get backwards."
    );
}

/// The corpus and its `fixed/` sibling disagree with each other.
#[test]
fn the_two_corpus_sources_pass_different_kinds_of_scale() {
    let corpus = read("bench/deepseek_v41/tile/quant_i8_v41_8x5120.tile");
    let fixed = read("bench/deepseek_v41/tile/fixed/quant_i8_v41_8x5120.tile");
    assert!(
        corpus.contains("__tile_load_f32(%arg1, %one, %one)"),
        "the corpus source no longer loads its scale as a 1x1 tile. If it was widened or \
         moved to `__tile_get_value_f32`, re-measure: the stored .pto was emitted from the \
         1x1 form and would need regenerating."
    );
    assert!(
        !corpus.contains("__tile_get_value_f32"),
        "the corpus source now reads a scalar, so it has converged on the `fixed/` form \
         and the two no longer disagree."
    );
    assert!(
        fixed.contains("__tile_get_value_f32"),
        "the `fixed/` sibling no longer reads a scalar. It exists to show the \
         declaration-conforming form; without that this file has only one reading to point at."
    );
}

/// The stored artefact carries the tile reading, as a 1x1 view.
#[test]
fn the_stored_artefact_carries_a_one_by_one_scale_view() {
    let pto = read("bench/deepseek_v41/pto/quant_i8_v41_8x5120.pto");
    assert!(
        pto.contains("!pto.partition_tensor_view<1x1xf32>"),
        "the stored .pto no longer has a 1x1 scale view. Either the source changed or the \
         emitter widened it; either way the suspicion in this file's note was resolved and \
         the note should say how."
    );
}

/// The arm has landed. This test now checks it AGREES with the resolved reading of
/// `scale` and takes the route the part actually has, which is what the previous
/// version of this test asked for by name when it fired.
///
/// REWRITTEN 2026-11-09, from asserting the arm's ABSENCE to asserting its shape. The
/// absence was the premise, not the property: `mlir_to_cpp` now dispatches on
/// `__tile_quantize_f32_i8`, and a test pinning a premise that has been removed must
/// be rewritten rather than deleted, because the thing worth pinning survives the
/// premise.
#[test]
fn the_ascendc_arm_divides_by_a_broadcast_tile_and_reaches_int8_through_f16() {
    let cpp = read("crates/ascendc_emit/src/mlir_to_cpp.rs");
    assert!(
        cpp.contains("@__tile_quantize_f32_i8"),
        "the arm is gone. If it was withdrawn, restore this test to its previous form, \
         which asserted the absence and named what landing it would require."
    );

    // READING: an elementwise DIVIDE by a broadcast tile, not a scalar `Muls` by a
    // reciprocal. tile_std/src/tile.rs declares "round(x / scale)" and
    // `mlir_to_pto` emits a broadcast divide; the two readings compute different
    // things, and a reciprocal multiply is the looser of them.
    let arm = cpp
        .split("--- __tile_quantize_f32_i8 ---")
        .nth(1)
        .expect("the arm's section marker is gone; this test locates the arm by it");
    let arm = &arm[..arm.find("--- __tile_exp_f32").unwrap_or(arm.len())];
    // CODE ONLY. The first version of this test searched the arm INCLUDING its
    // comments, and the arm's own note explains the rounding mode before the
    // emission reaches it -- so `CAST_RINT` was found in prose, ahead of the
    // `CAST_NONE` hop, and the order assertion failed on a correct arm. A detector
    // that reads commentary is measuring the commentary. This tree has the same
    // failure on record from a grep that matched a header comment rather than a
    // call.
    let arm: String = arm
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let arm = arm.as_str();
    assert!(
        arm.contains("AscendC::Div(") && arm.contains("AscendC::Duplicate("),
        "the quantize arm no longer divides by a broadcast tile. tile_std declares \
         round(x / scale) and mlir_to_pto emits a broadcast divide; a scalar Muls by a \
         reciprocal computes something else."
    );

    // ROUTE: the part has no f32 -> s8 convert. An int8 destination must be reached
    // through f16 -- accepted statically either way, garbage on device, and able to
    // raise 507035. The arm's FIRST version emitted the direct cast and the previous
    // form of this test caught it before any device run.
    // CAST_ROUND, not CAST_RINT, since c9aeb0a77: the part offers seven rounding
    // modes and `tile_std`'s "round(x / scale)" means half AWAY FROM ZERO, which is
    // CAST_ROUND. Device-proven exact on the full 40960-element corpus shape
    // (CAST_RINT: wrong=10092; CAST_ROUND: wrong=0, max_abs 0.0). The test read
    // CAST_RINT because that is what the arm emitted before that commit.
    let rint = arm.find("CAST_ROUND").expect("the arm no longer rounds at all");
    let none = arm.find("CAST_NONE").expect(
        "the quantize arm has no CAST_NONE hop. An int8 destination must be reached \
         f32 -> f16 -> s8; a direct f32 -> s8 Cast is a convert this part does not have.",
    );
    assert!(
        none < rint,
        "the quantize arm rounds to int8 BEFORE its f16 hop. The order is \
         f32 -> f16 (CAST_NONE) -> s8 (CAST_RINT)."
    );
    assert!(
        arm.contains("buf_elem_types.insert(hbuf, \"half\".to_string())"),
        "the arm's intermediate is not recorded as `half`, so the store's dtype check \
         cannot see the hop."
    );
}
