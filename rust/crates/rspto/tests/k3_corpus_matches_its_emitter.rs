// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The Kimi K3 corpus, held to the same rule as the playground one: every
//! generated artefact is re-derived, not trusted.
//!
//! A separate test from the playground gate rather than a generalisation of it,
//! because the two corpora make different claims and their counts are the claim.
//! Folding them together would mean one number covering both, and a number that
//! covers two corpora tells you nothing when it moves.
//!
//! Two of this model's kernels are absent from the corpus rather than refused by
//! it -- the delta-net recurrence and its short convolution, which the tile API
//! has no vocabulary for. They are named in the corpus manifest. A gate over
//! generated artefacts cannot say anything about a kernel that was never
//! generated, so that claim lives in the manifest test instead.

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/kimi_k3")
        .canonicalize()
        .expect("the Kimi K3 corpus must be reachable")
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("k3_{}_{}", std::process::id(), name));
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
    // 20 of an estimated 21: the two MLA up-projections, the three situ
    // gate/up rows, the three FFN down-projections and the decode short-conv
    // were added, and all but the down-projections refuse on the UB budget.
    // 11 of an estimated 21 originally. This corpus had NO reproducibility gate until
    // now -- it was the only one of the four without, so nothing proved its
    // PTO column regenerates from its own `.tile`.
    assert_eq!(
        tiles().len(),
        28,
        "the corpus changed size; update this deliberately"
    );
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
    // 24 since the three long dots landed -- the q and kv up-projections and the
    // lm head, at 18432, 24576 and 163840 wide. They used to refuse on the UB
    // budget; the emitter now chunks a dot whose row does not fit and folds the
    // per-chunk products before one reduction, so the width stopped being a wall.
    // 21 before that, since the seven KDA projections landed. Three of those
    // (q/k/v at k=7168) go through the `_gm_f32` streaming spelling, chosen by the
    // generator because the resident form overflows the UB at this model's D; the
    // other four fit.
    // 25 since dense_gate_up_situ joined. Its chain was splittable all along and the
    // retry DID engage, but the chunked widths used to be fixed up by a text rewrite
    // that runs after the body is built, so an op constructed through a typed
    // `pto_ir` constructor was validated against the un-rewritten width and refused
    // first: "scale: `pto.tmuls` reads ... cols=2112 ... and writes ... cols=33792".
    // The extent is now narrowed where it enters the emitter, and it lands at 1536.
    // STILL 25 on 2026-11-08, and rope_k3_1x64 nearly left the count twice in one
    // change. Its generator stopped calling the `__tile_rope_f32` stub (which has
    // no cos/sin operands and lowered to an identity copy measured 32x wrong on
    // device) and now emits the halves formulation in primitives over four
    // pointers. That formulation loads `%arg0` at 0 AND at 32, and `get_or_make_tv`
    // declares a buffer's view with the shape of the FIRST tile taken from it, so
    // the second load partitioned outside a 1x32 parent -- an artefact `ptoas`
    // rejects ("offset+size at dim 0 (2) exceeds static source dim (1)") and
    // `ptoas-rs` refused to lift, while THIS gate passed, because it compares the
    // emitter against its own output. The fix is `contain_pv`: re-declare the
    // parent wide enough (here 2x32, stride 32) rather than refuse, since the
    // buffer really is that big and the emitter had simply never said so. The
    // count is unchanged because a kernel whose partitions already fit is
    // untouched -- which is also why no other artefact's bytes moved.
    // 26 from 2026-11-10: `embedding_k3_163840x7168` (banked at iteration 27)
    // had its .pto on disk while the count still said 25. The gate was red on a
    // count, not a byte.
    assert_eq!(compared, 26, "expected 26 reproducible PTO artefacts");
    // 3 of 28, down from 7. What remains, with the emitter's own message:
    //
    //   * argmax_k3_1x163840          C1, UB budget, and the ONLY remaining UB
    //                                 refusal in this corpus. Neither chunking path
    //                                 reaches it: argmax carries a running INDEX, so
    //                                 a chunk boundary restarts the scan, and a fold
    //                                 that carries an index alongside a value is the
    //                                 arm it would need.
    //   * embedding_k3_163840x7168 and attention_decode_k3_1x128: no arm for
    //                                 `__tile_embedding_f32` or
    //                                 `__tile_attention_decode_f32`.
    // 2 from 2026-11-10: `embedding_k3_163840x7168` left this list when its .pto
    // landed; the two remaining are the capacity refusals.
    assert_eq!(refused, 2, "expected 2 refusals");
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
            // A missing emitter arm is a DISTINCT failure mode from a capacity
            // bound, and the list inherited from the other corpora named only
            // hardware bounds. K3's four refusals are two UB overflows and two
            // operations `mlir_to_pto` has no arm for, so the spread read as
            // having collapsed to one mode when it has two. Not counting a real
            // category is the same error as collapsing two.
            "has no arm for",
            "must be a multiple of Nb",
            "must be deferred",
            "pair count",
        ] {
            if msg.contains(bound) {
                reasons.insert(bound);
            }
        }
    }
    assert!(
        reasons.len() >= 2,
        "only {} distinct bound(s) hit: {reasons:?}. If this corpus has collapsed \
         to one failure mode it has stopped being a useful spread of shapes.",
        reasons.len()
    );
}
