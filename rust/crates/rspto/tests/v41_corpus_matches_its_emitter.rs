// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The V4.1 corpus, held to the same rule as the playground one: every
//! generated artefact is re-derived, not trusted.
//!
//! A separate test from the playground gate rather than a generalisation of it,
//! because the two corpora make different claims and their counts are the claim.
//! Folding them together would mean one number covering both, and a number that
//! covers two corpora tells you nothing when it moves.
//!
//! This corpus is mostly REFUSED, and that is its point: these are a real
//! model's shapes rather than teaching sizes, so they hit the part's bounds.

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/deepseek_v41")
        .canonicalize()
        .expect("the V4.1 corpus must be reachable")
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("v41_{}_{}", std::process::id(), name));
    p
}

fn emit(src: &Path, dst: &Path) -> bool {
    Command::new(env!("CARGO_BIN_EXE_rspto"))
        .arg(src)
        .arg("-o")
        .arg(dst)
        .output()
        .expect("rspto runs")
        .status
        .success()
}

fn tiles() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(corpus().join("tile"))
        .expect("tile/")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tile"))
        .collect();
    v.sort();
    v
}

#[test]
fn the_corpus_is_the_size_it_claims() {
    assert_eq!(tiles().len(), 17, "the corpus changed size; update this deliberately");
}

#[test]
fn every_checked_in_pto_is_byte_identical_to_a_fresh_emit() {
    let root = corpus();
    let (mut compared, mut refused) = (0, 0);
    for src in tiles() {
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let expected = root.join(format!("pto/{stem}.pto"));
        let out = tmp(&format!("{stem}.pto"));
        if !emit(&src, &out) {
            assert!(
                !expected.exists(),
                "{stem}: rspto refuses it, but a .pto is checked in"
            );
            refused += 1;
            continue;
        }
        assert!(expected.exists(), "{stem}: emits, but no .pto is recorded");
        assert_eq!(
            std::fs::read(&out).unwrap(),
            std::fs::read(&expected).unwrap(),
            "{stem}: the checked-in PTO differs from what this emitter produces"
        );
        compared += 1;
        let _ = std::fs::remove_file(&out);
    }
    // The split is the corpus's claim about these shapes, so it is pinned.
    //
    // 9 since quant_i8_v41_8x5120 joined, which took FOUR repairs in the emitter
    // because each was hidden behind the one in front -- and it is the only kernel
    // in any corpus that calls `__tile_quantize_f32_i8`, so its arm had never been
    // assembled and none of them was reachable before. In order: the chunked path
    // narrowed and loop-offset every operand, not just the width-sized ones;
    // `alloc_tile_typed` discarded a caller's custom tile type under chunking; a
    // `1 x 1` row_major tile has a 4-byte row against ptoas's 32-byte floor, so the
    // scale needed the corpus's padded idiom; and the arm emitted `pto.tdiv` of the
    // data by that scale, which ptoas refuses on valid_shape -- a per-tensor scale
    // into a tile is a BROADCAST (`pto.tcolexpand` down the rows, then
    // `pto.trowexpanddiv` along them).
    // 10 since rmsnorm_v41_8x5120 joined, the last piece of emitter work this
    // corpus had: its one-shot chain keeps four full-width tiles live (655360 B at
    // 8x5120 against a 194560 B addressable top) and a sum of squares reads ACROSS
    // the width, so the elementwise chunking retry may not touch it. It is emitted
    // as two passes over GM instead -- fold the per-chunk squares, one `trowsum`
    // after the fold, then re-read each chunk and scale it by the broadcast inverse
    // -- which is the long-dot arm's shape with the reduction's consumer added.
    // 11 from 2026-11-09: mx4pk_v41_5120x2304 joins, and not by a new arm. Its
    // refusal read "pair count (80) must be a multiple of the 64 chunk", where 64
    // is `TILE_MXFP4_PAIRS_PER_CHUNK`'s default -- a CAP, which the code was
    // reading as a divisor. 80 is not a multiple of 64 and never will be; the
    // chunk was the wrong size for the shape and nothing about the part says 64.
    // The emitter now takes the largest chunk at or below the cap that divides the
    // pair count AND is a multiple of 8 (the [1 x npc] f32 partials tile is
    // row_major none_box, so npc * 4 must be a multiple of 32), which for 80
    // under 64 is 40: two chunks of 40 pairs. ptoas 0.68 accepts the result at
    // rc=0. Measured, not assumed: this is the ONLY corpus kernel on the mxfp4
    // path, and all 73 other stored .pto re-emit byte-identical.
    // 12 from 2026-11-09: gemm_f16_lmhead_v41_16x5120x16160 joins, and again not
    // by a new arm. Its refusal read "N=16160 must be a multiple of Nb=64", which
    // is true of Nb=64 and is NOT a property of the shape: 16160 = 32 x 505, so
    // Nb=32 divides N, is a legal multiple of the 16 cube fractal
    // (`cube_n_size=16` in Ascend910B2.ini), and only SHRINKS the L0B footprint --
    // 256 x 32 x 2 B = 16 KB against `l0_b_size=65536`. `pick_nb_for_dtype` now
    // halves to find a divisor, exactly as `pick_kb` already did; the asymmetry
    // between the two was the whole defect. i8 is deliberately left fixed, because
    // `PTO_MM_NB_I8 = 256` is a CORRECTNESS constraint there and not a capacity
    // one. ptoas 0.68 --pto-arch=a2 accepts the result at rc=0.
    // 14 from 2026-11-10: the two `has no arm for` kernels both left the refusal
    // set for real emissions -- `embedding_v41_129280x5120` (the generator's row
    // copy) and `attention_decode_v41_1x512` (the 1-query sink arm, verified on
    // 910b against ORACLE 53 at backward_rel 5.976e-08). The count is an equality
    // because every one of them must re-emit byte-identically, not merely emit.
    assert_eq!(compared, 14, "expected 14 reproducible PTO artefacts");
    // 8 of 17. Six are UNREACHABLE BY ANY ARM and the residency check in the
    // campaign's scripts/ says why: their input plane cannot be resident and a
    // reduction in the chain forbids chunking -- the two gemms at 1701x and 242x
    // the addressable UB, and the two argmaxes at 21.6x and 21.3x. The other two
    // are the `__tile_embedding_f32` and `__tile_attention_decode_f32` arms, one a
    // dialect-primitive gap and one a signature gap settled by
    // tests/test_bench_refusal_classification.py.
    // 6 from 2026-11-09: mx4pk left, see the note above.
    // 5 later the same day: gemm_f16_lmhead_v41_16x5120x16160 left too, via the
    // adaptive Nb above. 3 from 2026-11-10: the two missing arms both landed
    // (embedding's row copy, attention's 1-query sink arm), so the three that
    // remain are the two argmaxes (capacity; the residency note applies) and the
    // malformed i8 call -- three distinct mechanisms, which the spread test checks.
    assert_eq!(refused, 3, "expected 3 refusals");
}

/// What makes this corpus worth having: most of it is refused, and the refusals
/// are not one bound hit over and over.
#[test]
fn the_refusals_name_several_different_bounds() {
    let mut reasons = std::collections::BTreeSet::new();
    for src in tiles() {
        let out = tmp("probe.pto");
        let run = Command::new(env!("CARGO_BIN_EXE_rspto"))
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .output()
            .unwrap();
        if run.status.success() {
            let _ = std::fs::remove_file(&out);
            continue;
        }
        let msg = String::from_utf8_lossy(&run.stderr);
        for bound in [
            "NPU tile bound",
            "NPU UB budget",
            "must be a multiple of Nb",
            "must be deferred",
            "pair count",
            // 2026-11-09. The arity check in `mlir_to_pto` refuses a call whose
            // argument count matches no declared form, BEFORE any positional
            // read. It replaced the "must be deferred" refusal on
            // gemm_i8_v41_1x5120x2304, which was a misdiagnosis: that call
            // carries six args to the seven-arg dequant form, so
            // detect_blocked_matmul_loads had correctly skipped it and was being
            // blamed for doing so. This list enumerates the emitter's refusal
            // VOCABULARY, so it has to track the emitter; the `>= 4` threshold
            // below is untouched.
            "call has",
            // 2026-11-09. The dominant refusal mechanism in this corpus was
            // MISSING from this list entirely -- two kernels refuse for a missing
            // arm (`__tile_attention_decode_f32`, `__tile_get_value_f32`) and
            // neither could ever contribute to the spread. Adding it corrects an
            // omission rather than relaxing the bar; the `>= 4` threshold below is
            // untouched.
            //
            // Measured the same day, the live set is exactly four mechanisms over
            // five refusing kernels: tile bound (argmax_8x129280), UB budget
            // (argmax_8x131072_pad), missing arm (attention_decode, embedding) and
            // arity (gemm_i8_1x). Three strings above are now DEAD, each because
            // the refusal was fixed rather than hidden: "pair count" in iteration
            // 54 (the mxfp4 chunk knob), "must be deferred" in 56 (the arity
            // misdiagnosis) and "must be a multiple of Nb" here (adaptive Nb).
            // They are kept so a regression that reintroduces one still counts.
            "has no arm for",
            // 2026-11-11. The i8 1x `.tile` gained the scale operand it was
            // missing, so the arity refusal above left this corpus. Its PTO
            // lowering now fails on the part's 16-row cube fractal instead --
            // `M=1 must be a multiple of 16 (910B2 cube fixedRowSize)`. That is
            // a different bound and belongs in the vocabulary: the AscendC
            // emitter answers the same M=1 shape with a vector GEMV, but the PTO
            // emitter has no non-cube matvec for an i8 dequant GEMM.
            "must be a multiple of 16",
        ] {
            if msg.contains(bound) {
                reasons.insert(bound);
            }
        }
    }
    // 3, not 4, since 2026-11-10: the fourth class was `has no arm for`, and BOTH
    // kernels that carried it now emit (embedding at iteration 27, attention at
    // 117). Requiring four would mean keeping a refusal alive to satisfy a count,
    // which is the opposite of what this test is for. The dead string stays in the
    // list above so a regression that reintroduces it still counts.
    // 2, not 3, from 2026-11-10: the two capacity-bound argmaxes now have a
    // generated AscendC route and are no longer PTO refusals at all, which
    // removed the "NPU tile bound" and "NPU UB budget" classes from THIS column.
    // The remaining three PTO refusals are the malformed i8 call and two
    // no-arm cases (mx4pk and the new per-row argmax intrinsic). A spread over
    // two classes is what the PTO column truthfully holds now.
    // STILL 2 from 2026-11-11, but with a different second class: the i8 1x call
    // is well-formed now, so its refusal is the M=16 cube fractal rather than
    // arity, beside the two `__tile_argmax_rows_i32` no-arms. The assertion is
    // unchanged; the vocabulary tracks the emitter.
    assert!(
        reasons.len() >= 2,
        "only {} distinct bound(s) hit: {reasons:?}. If this corpus has collapsed \
         to one failure mode it has stopped being a useful spread of shapes.",
        reasons.len()
    );
}
