// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `__tile_embedding_f32` declares its weight table as a TILE, and no vocabulary fits.
//!
//! Three of the fourteen kernels with no campaign record are the embeddings, and
//! they are the largest prize left. Their AscendC refusal reads like a tuning
//! problem -- "UB overflow ... gathers rows from a tile that holds 5120 element(s)
//! while the vocabulary needs 661913600" -- and it is not. It is a statement about
//! the DECLARATION, measured 2026-11-08 against `tile_std/src/tile.rs`:
//!
//! ```text
//! pub fn __tile_embedding_f32(
//!     dst: u32,
//!     weight: u32,              // <- a buf_id. A TILE.
//!     indices: *const u32,
//!     vocab_size: u32,
//!     embed_dim: u32,
//!     count: u32,
//! ) -> u32;
//! ```
//!
//! `weight` is a `u32` buffer id, and the doc line above it says the op "gathers
//! `count` rows from (V, D) weight table by index". So the contract requires the
//! WHOLE table resident as a tile. The numbers this gate asserts are the point:
//! the three model tables are 2.5 GB, 2.4 GB and 4.4 GB of f32 against a unified
//! buffer of 190 KB. The smallest overshoot is a factor of about 13,000.
//!
//! # Why a gate and not a note
//!
//! Because the cheap-looking fix is available and wrong. `mlir_to_pto` has no arm
//! for this intrinsic and refuses by name; adding one is a dozen lines, and with
//! the declaration unchanged any arm must read `weight` as a tile. The corpus
//! `.tile` loads ONE row (`__tile_load_f32(%arg0, %n, %d)` with `n = 1`), so such
//! an arm would emit a lookup into a one-row tile, assemble, and compute row 0 for
//! every index -- a wrong answer that looks like a win, which is the exact shape
//! this campaign keeps finding. Raising the UB bound cannot help either, and the
//! refusal's wording invites that reading.
//!
//! # What the fix actually is, and whose it is
//!
//! The table has to stay in GM and the gather has to index GM, which means
//! `weight` becomes a pointer rather than a buf_id. That is a change to a declared
//! intrinsic signature -- the contract every layer depends on -- so it is the
//! owner's under `.claude/rules/first-principles.md` sections 1 and 2, not a
//! harness edit. The DSL wrapper passes `weight.buf_id` too, so the surface moves
//! with it.
//!
//! At decode shape there is a second, smaller route worth pricing before anyone
//! builds the first: `count = 1` makes an embedding lookup a single row copy from
//! GM at offset `idx * D`, needing no gather intrinsic at all -- only a load at a
//! RUNTIME offset, where the offset is read from GM. That is a generator question
//! rather than a declaration change, and it is the cheaper experiment.
//!
//! This file asserts the arithmetic, not the refusal text, because a refusal is
//! wording and the bound is a fact.

use std::path::Path;

/// (model, vocab, dim) as the corpus kernel names declare them.
const TABLES: [(&str, u64, u64); 3] = [
    ("deepseek_v41", 129_280, 5_120),
    ("glm_5_3", 154_880, 4_096),
    ("kimi_k3", 163_840, 7_168),
];

/// The addressable unified buffer on a2/a3, in bytes. The emitter's own ceiling.
const UB_BYTES: u64 = 194_560;

#[test]
fn no_model_vocabulary_fits_the_unified_buffer_as_a_tile() {
    for (model, vocab, dim) in TABLES {
        let bytes = vocab * dim * 4;
        let factor = bytes / UB_BYTES;
        assert!(
            bytes > UB_BYTES,
            "{model}: a {vocab} x {dim} f32 table is {bytes} bytes, which now FITS in \
             the {UB_BYTES}-byte unified buffer. If that is real, the embedding \
             contract is implementable as declared and this file should go."
        );
        // Not just "does not fit" -- the margin is the argument. A factor of two
        // would be a tiling problem; four orders of magnitude is a contract problem.
        assert!(
            factor >= 1_000,
            "{model}: the table overshoots the unified buffer by only {factor}x \
             ({bytes} bytes vs {UB_BYTES}). This gate claims the overshoot is \
             structural rather than a tiling matter; at a small factor that claim is \
             wrong and the reasoning in the module note needs redoing."
        );
    }
}

/// The declaration this rests on, asserted rather than quoted.
#[test]
fn the_declared_weight_operand_is_still_a_tile_and_not_a_pointer() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tile_std/src/tile.rs")
        .canonicalize()
        .expect("tile_std's declarations must be reachable");
    let text = std::fs::read_to_string(&src).unwrap();
    let i = text
        .find("pub fn __tile_embedding_f32(")
        .expect("__tile_embedding_f32 must still be declared; if it was removed, the \
                 three embedding kernels need re-pricing from scratch");
    let decl = &text[i..text[i..].find(';').map(|e| i + e).unwrap_or(text.len())];
    assert!(
        decl.contains("weight: u32"),
        "the weight operand is no longer declared `u32`. If it became a pointer, the \
         table can stay in GM, the contract is implementable, and these three kernels \
         are unblocked -- delete this file and re-price them.\n{decl}"
    );
    assert!(
        decl.contains("indices: *const u32"),
        "the indices operand changed shape; this file's reading of the contract is \
         stale.\n{decl}"
    );
}

/// The corpus goes AROUND the contract: no embedding `.tile` calls the intrinsic.
///
/// Rewritten 2026-11-08, same day, when the generators started emitting the
/// operation as what it is at decode shape. The previous version asserted that
/// the three `.tile` sources loaded a single row through `__tile_embedding_f32` --
/// true then, and the reason an arm for that intrinsic would have been silently
/// wrong. It is no longer the state of the corpus, and the finding above is why:
/// the contract cannot fit a vocabulary, so the kernels went around it rather
/// than waiting for it.
///
/// What they emit instead is one row copy from GM -- read the index with
/// `__tile_get_value_f32`, multiply by the row width, `__tile_load_at_f32` at that
/// element offset -- which `pto-to-ascendc` lowers with the offset reaching the
/// generated `DataCopy`, so the table is never resident. This test holds the line
/// that matters now: if a kernel reaches for `__tile_embedding_f32` again, the
/// arithmetic above says it cannot work, and this says so at the source.
#[test]
fn no_embedding_tile_calls_the_intrinsic_that_cannot_fit() {
    let bench = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench must be reachable");
    let mut rowcopies = 0usize;
    for (model, _, _) in TABLES {
        let dir = bench.join(model).join("tile");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<std::path::PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths {
            if path.extension().and_then(|e| e.to_str()) != Some("tile") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(
                !text.contains("__tile_embedding_f32"),
                "{}: calls `__tile_embedding_f32`, whose declared contract needs the \
                 whole (V, D) table resident -- see the arithmetic in this file's \
                 module note. At decode shape the operation is a GM row copy; the \
                 generators emit that form and both emitters handle it (the PTO \
                 column refuses the scalar read-back, honestly, and the device run \
                 goes through AscendC).",
                path.display()
            );
            if !path.file_name().unwrap().to_string_lossy().starts_with("embedding") {
                continue;
            }
            rowcopies += 1;
            for needed in ["__tile_get_value_f32", "__tile_load_at_f32", "llvm.mul"] {
                assert!(
                    text.contains(needed),
                    "{}: an embedding kernel without `{needed}`. The row-copy form is \
                     read the index, multiply by the row width, load at that element \
                     offset; dropping a step silently changes which row is copied.",
                    path.display()
                );
            }
        }
    }
    assert_eq!(
        rowcopies, 3,
        "expected exactly three embedding kernels; found {rowcopies}. The count IS \
         the claim: one leaving means it was withdrawn."
    );
}
