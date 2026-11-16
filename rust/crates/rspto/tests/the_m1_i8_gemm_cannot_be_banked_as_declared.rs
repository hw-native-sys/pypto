// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `gemm_i8_v41_1x5120x2304` has no legal form that is not already in the corpus.
//!
//! It is one of the eleven kernels with no `verified` record, and its refusal
//! reads like a one-line source fix: "blocked matmul: M=1 must be a multiple of
//! 16 (910B2 cube fixedRowSize). Pad the M dim of your Rust tile_matmul kernel
//! source." The emitter means it -- `cube_m_size=16` in `Ascend910B2.ini` is the
//! fractal, and `mlir_to_pto`'s own note says the author is expected to pad M by
//! hand, with the host zeroing the unused rows and reading only row 0.
//!
//! Measured 2026-11-09: padding it produces a DUPLICATE. The two i8 gemms in this
//! corpus differ in exactly one constant:
//!
//! ```text
//! gemm_i8_v41_1x5120x2304    m=1    k=5120  n=2304
//! gemm_i8_v41_16x5120x2304   m=16   k=5120  n=2304   <- already emits, static-verified
//! ```
//!
//! So the only M this part accepts for that shape is 16, and the artefact at M=16
//! already exists and already banks. Padding the M=1 kernel does not produce a
//! new kernel; it produces the one beside it.
//!
//! # Why this is a gate and not a note
//!
//! Because the refusal invites the edit, the edit looks free, and the result would
//! be a second `.tile` that lowers to the same program -- which is the
//! "several corpus artefacts are the same call" defect the lift already refuses
//! nineteen kernels for, arriving here through the SOURCE instead. A reader who
//! pads this kernel gets a green corpus and one more banked row that measures
//! nothing new.
//!
//! # What it means for the primary KPI
//!
//! `C_m`'s denominator is pinned at 72 and does not move. But this kernel cannot
//! reach `verified` as declared, so the REACHABLE value is at most 71/72 until an
//! owner decides one of:
//!
//!   * accept that decode-shape (M=1) i8 gemm is not expressible on this part's
//!     cube and withdraw the declared shape, which lowers K -- an owner action,
//!     since K is a pinned benchmark input; or
//!   * accept the padded duplicate as its own row, which means two rows for one
//!     program and is the thing this file exists to flag; or
//!   * route M=1 i8 gemm through a non-cube path, which is a new arm and not a
//!     source edit.
//!
//! This file asserts the arithmetic and the duplication, not the refusal text.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// `cube_m_size` on Ascend910B2, i.e. `TileConfig::fixedRowSize`. Not a knob.
const FRACTAL_M: u32 = 16;

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench must be reachable")
}

fn constants(rel: &str) -> HashMap<String, u32> {
    let text = std::fs::read_to_string(bench().join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"));
    let mut out = HashMap::new();
    for line in text.lines() {
        let Some((lhs, rhs)) = line.trim().split_once(" = llvm.mlir.constant(") else {
            continue;
        };
        let name = lhs.trim_start_matches('%').trim().to_string();
        if let Some(v) = rhs.split(' ').next().and_then(|v| v.parse::<u32>().ok()) {
            out.insert(name, v);
        }
    }
    out
}

#[test]
fn the_m1_kernel_differs_from_its_banked_sibling_only_in_m() {
    // The `fixed/` sibling is the one with the well-formed 7-arg call; the corpus
    // copy carries a 6-arg call and refuses on arity first, which would mask this.
    let one = constants("deepseek_v41/tile/fixed/gemm_i8_v41_1x5120x2304.tile");
    let sixteen = constants("deepseek_v41/tile/gemm_i8_v41_16x5120x2304.tile");

    assert_eq!(one.get("m"), Some(&1), "the M=1 kernel no longer declares m=1");
    assert_eq!(
        sixteen.get("m"),
        Some(&FRACTAL_M),
        "the sibling no longer declares m=16, so it is no longer the padded form \
         of the other and this file's duplication claim needs re-deriving"
    );
    for dim in ["k", "n"] {
        assert_eq!(
            one.get(dim), sixteen.get(dim),
            "the two i8 gemms now differ in `{dim}` as well as M, so padding M \
             would NOT produce the sibling and this kernel may be bankable after \
             all -- re-price it"
        );
    }
    assert_ne!(
        one.get("m"), sixteen.get("m"),
        "the two kernels now declare the same M, which means the corpus holds two \
         copies of one program -- the outcome this file exists to prevent"
    );
}

#[test]
fn m_equals_one_is_below_the_cube_fractal_and_rounds_up_to_the_sibling() {
    let one = constants("deepseek_v41/tile/fixed/gemm_i8_v41_1x5120x2304.tile");
    let m = *one.get("m").expect("m");
    assert!(
        m % FRACTAL_M != 0,
        "M={m} is now a multiple of the {FRACTAL_M}-row cube fractal, so the \
         refusal this file is about cannot fire and the kernel should be banked \
         rather than flagged"
    );
    // The rounded-up M is the sibling's, which is the whole argument.
    let padded = m.div_ceil(FRACTAL_M) * FRACTAL_M;
    assert_eq!(
        padded, FRACTAL_M,
        "M={m} rounds up to {padded}, not {FRACTAL_M}. If it rounds to something \
         else there is no existing sibling at that shape and padding would \
         genuinely produce a new kernel."
    );
}
