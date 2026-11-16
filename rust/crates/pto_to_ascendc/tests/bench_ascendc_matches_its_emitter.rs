// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Every `.cpp` in the bench must be reproducible from its `.tile` by the
//! emitter in this tree, byte for byte.
//!
//! A checked-in generated artefact has no owner unless something re-derives it.
//! Left alone it drifts: the emitter changes, the artefact does not, and both
//! stay green because nothing compares them. Then the corpus is documenting an
//! emitter that no longer exists. This test is the comparison.
//!
//! A sibling test does the same for the PTO column. Two emitters check
//! different bounds, so they are two gates rather than one.
//!
//! It is a cargo integration test rather than a script so that it cannot skip.
//! `CARGO_BIN_EXE_pto-to-ascendc` is always present here, so there is no "binary not
//! found, skipping" path -- and a skip that returns success is how a gate stops
//! being a gate.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bench_root() -> PathBuf {
    // crates/rspto -> rust -> repository root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/playground16")
        .canonicalize()
        .expect("the bench corpus must be reachable from this crate")
}

fn emit(src: &Path, dst: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_pto-to-ascendc"))
        .arg(src)
        .arg("-o")
        .arg(dst)
        .output()
        .expect("pto-to-ascendc runs")
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ascendc_bench_{}_{}", std::process::id(), name));
    p
}

/// Walk `tile/` and `tile/fixed/`, returning (source, expected-pto) pairs.
fn pairs() -> Vec<(PathBuf, PathBuf)> {
    let root = bench_root();
    let mut out = Vec::new();
    for sub in ["tile", "tile/fixed"] {
        let dir = root.join(sub);
        let mut entries: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "tile"))
            .collect();
        entries.sort();
        for src in entries {
            let stem = src.file_stem().unwrap().to_string_lossy().to_string();
            let expected = root
                .join(if sub == "tile" { "ascendc" } else { "ascendc/fixed" })
                .join(format!("{stem}.cpp"));
            out.push((src, expected));
        }
    }
    out
}

#[test]
fn the_corpus_is_not_empty() {
    // Guard against the whole suite passing because it found nothing to check.
    // Twenty sources: sixteen kernels plus four fixed variants. `matmul` joined
    // the fixed set when its capture was found to pass five operands where
    // `__tile_matmul_f16` declares six.
    let p = pairs();
    assert_eq!(p.len(), 20, "the corpus changed size; update this count deliberately");
}

#[test]
fn every_checked_in_ascendc_is_byte_identical_to_a_fresh_emit() {
    let mut compared = 0;
    let mut refused = Vec::new();

    for (src, expected) in pairs() {
        let name = src.file_name().unwrap().to_string_lossy().to_string();
        let out = tmp(&format!("{name}.cpp"));
        let run = emit(&src, &out);

        if !run.status.success() {
            assert!(
                !expected.exists(),
                "{name}: rspto refuses it, but a .pto is checked in. Either the \
                 artefact predates a tightened check, or the check regressed.",
            );
            refused.push(name);
            continue;
        }

        assert!(
            expected.exists(),
            "{name}: AscendC is emitted for it, but no .cpp is checked in. A kernel \
             that emits and is not recorded makes the corpus totals wrong.",
        );
        let fresh = std::fs::read(&out).unwrap();
        let stored = std::fs::read(&expected).unwrap();
        assert_eq!(
            fresh,
            stored,
            "{name}: the checked-in AscendC differs from what this emitter produces. \
             Regenerate the corpus in the same change that altered the emitter.",
        );
        compared += 1;
        let _ = std::fs::remove_file(&out);
    }

    // 19/1 since 2026-11-04: the buffer-aliasing pass made `gate_up_silu` fit
    // (its chain reuses dead slots), so only `argmax` is still refused -- it is a
    // multi-row reduction, not a budget case at all.
    assert_eq!(compared, 19, "expected 19 reproducible artefacts");
    assert_eq!(refused.len(), 1, "expected exactly 1 refusal, got {refused:?}");
}

#[test]
fn the_two_refusals_still_refuse_for_a_resource_reason() {
    // Not merely "still fails": a refusal that changed cause is a different
    // kernel being rejected for a different reason, and the one failure mode
    // that matters here is a resource check quietly becoming something else --
    // or worse, passing, which would emit a kernel the hardware cannot run.
    let root = bench_root();
    // `gate_up_silu` used to be the second entry. It emits now that dead buffers
    // can share slots, which is why this list is shorter rather than repointed:
    // the corpus has one resource refusal left, and pretending otherwise would
    // make this test assert something that is not true of the tree.
    for (kernel, needle) in [("argmax", "UB overflow")] {
        let src = root.join("tile").join(format!("{kernel}.tile"));
        let out = tmp(&format!("{kernel}_refusal.pto"));
        let run = emit(&src, &out);
        assert!(
            !run.status.success(),
            "{kernel} now emits AscendC. If that is intended, the manifest and the \
             corpus totals have to change in the same commit."
        );
        let msg = String::from_utf8_lossy(&run.stderr);
        assert!(
            msg.contains(needle),
            "{kernel} is refused, but not for the recorded reason.\nwanted: \
             {needle}\ngot: {msg}"
        );
        let _ = std::fs::remove_file(&out);
    }
}

#[test]
fn each_refused_kernel_has_a_reshaped_variant_that_emits() {
    // The reason a refusal is a result here rather than a dead end: the same
    // computation, at a width that fits, is in the corpus and does emit.
    let root = bench_root();
    for kernel in ["argmax", "gate_up_silu"] {
        let fixed = root.join("tile/fixed").join(format!("{kernel}.tile"));
        assert!(fixed.exists(), "{kernel} has no reshaped variant");
        let out = tmp(&format!("{kernel}_fixed.pto"));
        let run = emit(&fixed, &out);
        assert!(
            run.status.success(),
            "the reshaped {kernel} does not emit: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let _ = std::fs::remove_file(&out);
    }
}
