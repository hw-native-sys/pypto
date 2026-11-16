// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The two topk kernels are not four missing arms, and `tile_gather_f32` is a trap.
//!
//! `topk_v41_1x384_k6` and `router_sigmoid_topk_k3_1x896_k16` are the last two
//! lift refusals a census called "closable by a session". Each refuses naming ONE
//! op, and each needs FOUR modelled before it lifts: `pto.tci`, `pto.textract`,
//! `pto.tgather` and `pto.tsort32`. A refusal reports the first blocker it
//! reaches, not the set, so a census built from refusal text understates this --
//! the same understatement that made an 8-member `v_row` class worth 4.
//!
//! Measured 2026-11-09, that work is not four wirings:
//!
//! ```text
//! pto.textract   tile_slice_f32              exists, correct shape        WIRING
//! pto.tci        __tile_arith_progression_i32  extern only, no wrapper    NEW WRAPPER
//! pto.tsort32    __tile_sort32_f32             extern only, no wrapper    NEW WRAPPER
//! pto.tgather    tile_gather_f32             EXISTS AND IS THE WRONG OP   CONTRACT
//! ```
//!
//! # The trap
//!
//! `tile_gather_f32` is an INDEXED ROW GATHER -- `<N, M, D>(src: Tile<M, D>,
//! indices: *const u32) -> Tile<N, D>`, "gather from (M, D) at positions". The
//! `pto.tgather` these kernels carry is the MASK-PATTERN form:
//!
//! ```text
//! pto.tgather ins(%pto5, {maskPattern = #pto.mask_pattern<P1010>} : ...)
//! ```
//!
//! which the PTO emitter documents as "extracts a sub-tile per the 4-bit mask
//! pattern attribute" -- lane selection by a static mask, with no index operand at
//! all. Lifting the mask form to `tile_gather_f32` would emit an indexed gather
//! where the artefact does a lane mask: a different program that compiles, runs,
//! and is wrong. The name matches and the operation does not.
//!
//! That is the third instance of this shape in one day -- the quantize `scale`
//! operand, where a scalar `Muls` and a broadcast divide are different programs;
//! the M=1 i8 gemm, where the padding the refusal invites yields a duplicate; and
//! this. In all three the cheap-looking fix is available, named plausibly, and
//! computes something else.
//!
//! # So this is an OWNER question, not session work
//!
//! The mask-pattern gather has no tile-API equivalent, and inventing one is a
//! change to the declared surface. Until it exists these two kernels are not
//! session-closable at any effort, which drops `C_b`'s session-reachable ceiling
//! from the 25/72 first computed to **23/72**. Three of the five refusals that
//! census called closable were closed (`pto.tsubs` and the two `kda_delta`
//! kernels); these two were never in reach.
//!
//! What this file does NOT claim: that a mask-pattern gather is unliftable in
//! principle. A `tile_mask_select`-shaped wrapper taking the pattern as a const
//! generic would express it. It claims only that the wrapper which exists today
//! is a different operation, so the work is a contract decision and not an arm.

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

const TOPK: [&str; 2] = [
    "bench/deepseek_v41/pto/topk_v41_1x384_k6.pto",
    "bench/kimi_k3/pto/router_sigmoid_topk_k3_1x896_k16.pto",
];

/// Both kernels carry the MASK form, which is what makes the existing wrapper wrong.
#[test]
fn both_topk_artefacts_use_the_mask_pattern_gather_and_not_an_indexed_one() {
    for rel in TOPK {
        let t = read(rel);
        assert!(
            t.contains("pto.tgather"),
            "{rel}: no `pto.tgather`. If the emitter stopped producing it, this \
             file's whole argument about the wrong wrapper is moot -- re-derive."
        );
        assert!(
            t.contains("maskPattern"),
            "{rel}: `pto.tgather` without a `maskPattern` attribute. If it now \
             takes an index operand, `tile_gather_f32` may be the RIGHT wrapper \
             after all and these kernels become session-closable."
        );
    }
}

/// The wrapper that exists takes indices, not a mask. That is the trap, asserted.
#[test]
fn the_existing_gather_wrapper_takes_indices_so_it_is_a_different_operation() {
    let ts = read("crates/tile_std/src/tile.rs");
    let i = ts
        .find("pub fn tile_gather_f32")
        .expect("tile_gather_f32 must still exist; if it was removed, the trap is \
                 gone and so is this test's subject");
    let sig = &ts[i..i + ts[i..].find('{').unwrap_or(400)];
    assert!(
        sig.contains("indices"),
        "`tile_gather_f32` no longer takes `indices`. If it became a mask-pattern \
         form, it is now the correct wrapper for what these artefacts carry -- \
         delete this file and wire it.\n{sig}"
    );
    assert!(
        !sig.contains("mask"),
        "`tile_gather_f32` now mentions a mask. Check whether it expresses the \
         pattern form; if so these two kernels are closable.\n{sig}"
    );
}

/// All four ops are needed, so closing any one of them banks nothing.
#[test]
fn each_topk_artefact_needs_all_four_ops_so_partial_work_banks_nothing() {
    let parser = read("crates/ptoas_rs/src/parser.rs");
    let needed = ["pto.tci", "pto.textract", "pto.tgather", "pto.tsort32"];
    for rel in TOPK {
        let t = read(rel);
        let present: Vec<&str> = needed.iter().copied().filter(|op| t.contains(op)).collect();
        assert_eq!(
            present.len(),
            needed.len(),
            "{rel} carries only {present:?} of {needed:?}. The set shrank, so the \
             'all four or nothing' claim needs re-deriving -- it may now be fewer."
        );
    }
    // None is modelled yet. When one is, this fires and names it: that is the
    // point at which "partial work banks nothing" should be re-checked rather
    // than assumed to still hold.
    let modelled: Vec<&str> = needed
        .iter()
        .copied()
        .filter(|op| parser.contains(&format!("\"{op}\"")))
        .collect();
    assert!(
        modelled.is_empty(),
        "the parser now models {modelled:?}. Re-run the two topk artefacts: if \
         they still refuse, the remaining ops are the blockers and this file's \
         table should say which are left."
    );
}
