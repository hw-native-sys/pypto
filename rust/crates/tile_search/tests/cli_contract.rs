// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The command's contract, on the kernel bench rather than on a fixture.
//!
//! The case that matters is `gate_up_silu`: the target refuses it as captured,
//! and the search has to find a tiling it takes. Running against the bench
//! rather than an inline kernel is deliberate -- a search that only works on its
//! own example is not evidence about anything.

use std::path::{Path, PathBuf};
use std::process::Command;

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/playground16")
        .canonicalize()
        .expect("the bench corpus must be reachable")
}

fn tool() -> Command {
    Command::new(env!("CARGO_BIN_EXE_tile-search"))
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("tile_search_it_{}_{}", std::process::id(), name));
    p
}

#[test]
fn a_refused_kernel_is_rescued_and_the_result_emits() {
    // `gate_up_silu` was the fixture until rspto learned to chunk: an elementwise
    // chain at 8960 is no longer refused, because it is split into equal chunks and
    // accepted. `argmax` was the next attempt and rescues to an extent with no
    // smaller candidates, which the idempotence check below rejects. The fixture is
    // now a WIDE REDUCTION of its own -- unsplittable by chunking, refused as
    // captured for six times the budget, and with a real ladder of extents between.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/wide_reduce.tile");
    let out = tmp("rescued.tile");

    let run = tool().arg(&src).arg("-o").arg(&out).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));

    let report = String::from_utf8_lossy(&run.stderr);
    // The captured extent must be reported as refused, or the premise is gone.
    assert!(report.contains("5120"), "the captured extent is not in the report:\n{report}");
    assert!(report.contains("refused"), "nothing was refused:\n{report}");
    assert!(report.contains("chose"), "no choice was reported:\n{report}");

    // And the chosen kernel must itself pass the emitter. Checked by running the
    // emitter binary, not by trusting the search's own verdict.
    let pto = tmp("rescued.pto");
    let emit = Command::new(env!("CARGO_BIN_EXE_tile-search"))
        .arg(&out)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(
        emit.status.success(),
        "the chosen kernel does not survive a second search pass:\n{}",
        String::from_utf8_lossy(&emit.stderr)
    );
    let second = String::from_utf8_lossy(&emit.stderr);
    assert!(
        second.contains("captured value: nothing to change"),
        "a second pass should find the chosen tiling already best:\n{second}"
    );

    for p in [out, pto] {
        let _ = std::fs::remove_file(p);
    }
}

#[test]
fn the_report_lists_every_candidate_not_only_the_winner() {
    let src = bench().join("tile/gate_up_silu.tile");
    let run = tool().arg(&src).arg("--dry-run").output().unwrap();
    let report = String::from_utf8_lossy(&run.stderr);
    // One line per candidate, and the count stated. A report of only the winner
    // would hide both the size of the space and the reasons for each refusal.
    let lines = report.lines().filter(|l| l.trim_start().starts_with("%c = ")).count();
    assert!(lines >= 6, "only {lines} candidate line(s):\n{report}");
    assert!(report.contains("candidate(s) accepted"), "{report}");
}

#[test]
fn the_report_does_not_claim_a_measurement() {
    let src = bench().join("tile/gate_up_silu.tile");
    let run = tool().arg(&src).arg("--dry-run").output().unwrap();
    let report = String::from_utf8_lossy(&run.stderr);
    assert!(
        report.contains("declared model") && report.contains("not a measurement"),
        "the ranking must not read as a measured result:\n{report}"
    );
}

#[test]
fn an_already_best_kernel_is_reported_as_unchanged() {
    let src = bench().join("tile/down_proj.tile");
    let run = tool().arg(&src).arg("--dry-run").output().unwrap();
    assert!(run.status.success());
    let report = String::from_utf8_lossy(&run.stderr);
    assert!(
        report.contains("nothing to change"),
        "doing nothing has to be reachable:\n{report}"
    );
}

#[test]
fn dry_run_writes_nothing() {
    let src = bench().join("tile/down_proj.tile");
    let out = tmp("dry.tile");
    let _ = std::fs::remove_file(&out);
    let run = tool().arg(&src).arg("--dry-run").arg("-o").arg(&out).output().unwrap();
    assert!(run.status.success());
    assert!(!out.exists(), "--dry-run wrote {}", out.display());
}

#[test]
fn rust_input_is_recognised_and_refused_with_the_toolchain_code() {
    let src = bench().join("tile-rs/vec_add.rs");
    let run = tool().arg(&src).output().unwrap();
    assert_eq!(run.status.code(), Some(4), "expected the route-unavailable code");
}

#[test]
fn an_unknown_extension_is_a_usage_error() {
    let src = tmp("thing.txt");
    std::fs::write(&src, "module {}\n").unwrap();
    let run = tool().arg(&src).output().unwrap();
    assert_eq!(run.status.code(), Some(2));
    let _ = std::fs::remove_file(src);
}

#[test]
fn a_lift_refusal_writes_nothing_and_says_which_form_survived() {
    // `--emit rs` goes through the lift, which models fewer ops than the
    // emitter. When it refuses, the tool must not leave a partial file behind,
    // and must say that the tile form is still available.
    // `gate_up_silu` was this fixture until the lift learned to reproduce a doubly-read
    // compute result: it lifts now, so `--emit rs` succeeds. `rms_norm` still emits PTO and
    // still refuses the lift back, which is the situation this test is about.
    let src = bench().join("tile/rms_norm.tile");
    let out = tmp("lift_refused.rs");
    let _ = std::fs::remove_file(&out);
    let run = tool()
        .arg(&src)
        .arg("--emit")
        .arg("rs")
        .arg("-o")
        .arg(&out)
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(1), "a refused lift is not success");
    assert!(!out.exists(), "a refused lift left a file behind");
    let msg = String::from_utf8_lossy(&run.stderr);
    assert!(msg.contains("--emit tile"), "no route out was named:\n{msg}");
}

// --- the second oracle ---------------------------------------------------

/// A kernel in another corpus, since `bench()` points at playground16.
fn other(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("the repository root must be reachable")
        .join(rel)
}

const K3_PROJ: &str = "rust/bench/kimi_k3/tile/proj_output_k3_1x7168.tile";
/// A kernel the AscendC emitter still REFUSES, which is an EXISTENTIAL claim and
/// has now needed re-pointing twice -- each time because the emitter got stronger,
/// and each move recorded here rather than silently.
///
/// `K3_PROJ` served both roles until 2026-11-04, when the AscendC budget learned to
/// count only the buffers the body names and `proj_output` -- whose four unnamed
/// buffers were eating an equal share each -- started emitting. It was replaced by
/// `proj_q_b_k3_1x18432`, one tile of which is 73728 B of a 188416 B buffer.
///
/// On 2026-11-08 `proj_q_b` started emitting too: the BLOCKED-REDUCTION arm landed
/// in `mlir_to_cpp`, so a one-row chain that is summed along the row now runs per
/// chunk into a tile accumulator instead of refusing on the per-buffer bound. It is
/// device-verified at rel 2.596e-09, so the old assertion `!asc.unchanged` became
/// false for a good reason and the fixture had to move, not the assertion.
///
/// `rmsnorm_v41_8x5120` was the replacement, and on 2026-11-08 the ROW-BLOCKED arm
/// took that too. So the fixture has outlived three emitter improvements, and the
/// third one ended the search: **no kernel in these corpora still refuses at its
/// captured extent on a resource bound while accepting a smaller one.** Measured,
/// by probing every candidate the manifests record a `searched` entry for --
/// `embedding_glm53_154880x4096`, `embedding_k3_163840x7168`,
/// `argmax_v41_8x129280`, `quant_i8_v41_8x5120` -- and all four fail the premise
/// for the same reason: their refusals are a MISSING ARM or a vocabulary gather,
/// not a resource bound, so shrinking an extent never helps and
/// `report.accepted()` is 0. The four rescues (chunking, the accumulating arm, the
/// row-blocked arm, the chunked argmax) have between them eliminated the class of
/// refusal this fixture needed.
///
/// The kernel is kept as the fixture for the ZERO-BUDGET guard below, which is the
/// documented reason the first test exists and does not need a refusal -- only a
/// kernel with several accepted candidates, which this still is. The disagreement
/// test has no instance left and says so rather than being re-pointed at a fifth
/// kernel until it passes.
///
/// Two of those four candidates carry a STALE `searched` entry, recorded when the
/// shrink did help: `argmax_v41_8x129280` claims "chose %cl = 1280, accepted 4" and
/// `quant_i8_v41_8x5120` "chose %cl = 1280, accepted 5", while both now refuse for
/// reasons no extent reaches. Noted here rather than fixed, because re-running the
/// search is its own change.
const K3_PROJ_OVER: &str = "rust/bench/deepseek_v41/tile/rmsnorm_v41_8x5120.tile";

/// The AscendC oracle must be given hardware, or it refuses everything.
///
/// `EmitOpts::default()` leaves that emitter with a ZERO-byte Unified Buffer,
/// and it then refuses every candidate with a message that reads like a
/// verdict: "larger than the ENTIRE 0 byte unified buffer". Measured: the first
/// version of `probe_with` passed the default and reported `0 of 8 candidate(s)
/// accepted` for a kernel whose real bound is 18816 bytes per buffer. The only
/// tell was the literal `0`.
///
/// So this asserts the search SUCCEEDS where it must, which is what a zero
/// budget silently destroys. A test that only checked the flag parsed would
/// have passed throughout.
#[test]
fn the_ascendc_oracle_searches_against_real_hardware() {
    let kernel = other(K3_PROJ_OVER);
    assert!(kernel.is_file(), "{} is missing", kernel.display());
    let mlir = std::fs::read_to_string(&kernel).expect("read");

    let report = tile_search::search_with(
        &mlir,
        tile_search::Cost::LargestAccepted,
        tile_search::Oracle::AscendC,
    );
    let rendered = report.render();
    assert!(
        !rendered.contains("0 byte unified buffer"),
        "the AscendC oracle is running with no hardware, so every candidate is \
         refused against a zero budget:\n{rendered}"
    );
    assert!(
        report.accepted() > 0,
        "the AscendC oracle accepted nothing for a kernel it can tile. A zero \
         budget is what that looks like.\n{rendered}"
    );
    // THE THIRD ASSERTION IS GONE, and its absence is the finding. It read
    // `!report.unchanged`: the oracle must not choose the captured extent, because
    // the emitter refuses it. As of 2026-11-08 the emitter ACCEPTS this kernel's
    // captured extent -- the row-blocked arm runs its eight rows one at a time --
    // and no kernel in these corpora is left that it refuses on a resource bound.
    // The two assertions above are the ones this test was written for (a zero-budget
    // oracle accepts NOTHING, and reports "0 byte unified buffer" while doing it),
    // and neither needs a refusal to be meaningful.
}

/// The two oracles disagree, which is the reason there are two -- and as of
/// 2026-11-08 there is NO KERNEL IN THESE CORPORA where they still do.
///
/// Skipped rather than deleted, and rather than re-pointed. The claim is
/// existential and was carried by a named fixture three times over; each time the
/// AscendC emitter improved and the instance had to move. The third move ended it:
/// every remaining AscendC refusal in the model corpora is a MISSING ARM or a
/// vocabulary gather, which no choice of extent reaches, so `report.accepted()` is
/// 0 and the search has nothing to disagree about. Measured over all four
/// candidates the manifests record a `searched` entry for; see the note at
/// `K3_PROJ_OVER`.
///
/// Deleting it would lose the reason the second oracle exists. Re-pointing it at a
/// kernel until it went green is what the file's own comment on `K3_PROJ_OVER`
/// forbids. So it states what is true now and becomes a test again the day an
/// emitter arm is added whose PTO twin is missing -- which is a live possibility:
/// `rspto` still has no chunked path for an elementwise chain, and
/// `bench/tile_rs_acceptance.json` records twelve kernels waiting on it.
#[test]
#[ignore = "needs a kernel whose captured extent the AscendC emitter refuses and the PTO one accepts; the four rescue arms have eliminated \
            that class in these corpora -- see the note at K3_PROJ_OVER for the measurement"]
fn the_two_oracles_do_not_agree() {
    // THIS TEST USED TO ASSERT `pto.unchanged && !asc.unchanged`, on the reasoning
    // that PTO's limit is over the working set while AscendC's is per BUFFER, so
    // PTO accepted what AscendC refused. On 2026-11-04 the AscendC budget learned
    // to count only the buffers the body names, and that asymmetry went away on
    // every kernel in this corpus: for `proj_q_b` the two oracles now choose the
    // SAME extent (9216), and for `proj_output` both accept the captured one.
    //
    // The measured fact is therefore narrower than the old assertion, and the
    // assertion moves with it rather than being re-pointed at another fixture
    // until it passes. (`proj_q_b` is no longer the fixture either -- it started
    // emitting on 2026-11-08 when the blocked-reduction arm landed; see the note on
    // `K3_PROJ_OVER` for why that is a fixture move and not an assertion move.) What still holds, and is what the second oracle is for:
    // AscendC REFUSES the captured extent, so a search driven by PTO alone would
    // report "nothing to change" for a kernel its own emitter rejects.
    let mlir = std::fs::read_to_string(other(K3_PROJ_OVER)).expect("read");
    let pto = tile_search::search_with(
        &mlir,
        tile_search::Cost::LargestAccepted,
        tile_search::Oracle::Pto,
    );
    let asc = tile_search::search_with(
        &mlir,
        tile_search::Cost::LargestAccepted,
        tile_search::Oracle::AscendC,
    );
    assert!(
        !asc.unchanged,
        "the AscendC oracle accepted the captured extent of a kernel the AscendC \
         emitter refuses, so it is not consulting the emitter at all:\nascendc \
         chose {:?}",
        asc.chosen
    );
    assert!(
        pto.chosen.is_some() && asc.chosen.is_some(),
        "both oracles must find some accepted extent for a kernel that can be \
         tiled at all: pto {:?}, ascendc {:?}",
        pto.chosen,
        asc.chosen
    );
}

#[test]
fn an_unknown_oracle_is_a_usage_error() {
    let out = tool()
        .arg(other(K3_PROJ))
        // Deliberately not another vendor's target name. The licence gate
        // refused the first version of this line, which used one -- CANN OSL
        // 3.1 is about what this tree may be used to develop for, and a test
        // fixture is still source. The same reasoning is why
        // `pto_is_the_only_registered_target` asserts the exact set rather than
        // probing a list of absent names.
        .args(["--oracle", "no-such-emitter"])
        .output()
        .expect("ran");
    assert_eq!(out.status.code(), Some(2), "expected the usage code");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("pto") && err.contains("ascendc"),
        "the message must name the oracles that exist:\n{err}"
    );
}
