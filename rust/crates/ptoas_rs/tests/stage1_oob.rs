// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Integration tests for the stage-1 OOB safety oracle.
//!
//! Covers two layers:
//!   1. Library: `parse_stage1` + `check_all_stage1` on hand-written
//!      fixtures (positive = manually-OOB partition_view, negative =
//!      clean softmax slice).
//!   2. Binary: `pto-diff --stage1 <fixture>` end-to-end, asserting on
//!      the printed report.
//!
//! The fixtures live under `tests/fixtures/stage1_*.mlir`.

use ptoas_rs::{parse_stage1, safety::check_all_stage1, SafetyKind};
use std::path::PathBuf;
use std::process::Command;

fn fixture_path(name: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
        .iter()
        .collect()
}

fn fixture(name: &str) -> String {
    let p = fixture_path(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", p.display()))
}

// ── library layer ──

#[test]
fn stage1_softmax_clean_zero_findings() {
    let src = fixture("stage1_softmax_clean.mlir");
    let plan = parse_stage1(&src).expect("parse softmax");
    assert_eq!(plan.funcs.len(), 1);
    assert_eq!(plan.funcs[0].name, "tile_softmax");
    let rep = check_all_stage1(&plan);
    assert!(
        rep.violations.is_empty(),
        "expected zero stage-1 findings, got: {:#?}",
        rep.violations
    );
}

#[test]
fn stage1_oob_partition_flags_oob() {
    let src = fixture("stage1_oob_partition.mlir");
    let plan = parse_stage1(&src).expect("parse oob");
    let rep = check_all_stage1(&plan);
    let oob: Vec<_> = rep
        .violations
        .iter()
        .filter(|v| v.kind == SafetyKind::OobPartition)
        .collect();
    assert_eq!(
        oob.len(),
        1,
        "expected exactly 1 OobPartition finding (one per partition_view), got: {:#?}",
        rep.violations
    );
    let v = oob[0];
    assert_eq!(v.func, "bad");
    // The finding should mention both end coordinates (row=3, col=1190)
    // so a reader can see both axes are over-bound.
    assert!(
        v.message.contains("[3,1190]") || v.message.contains("[3, 1190]"),
        "expected end-coords [3,1190] in message, got: {}",
        v.message
    );
    assert!(
        v.message.contains("[1,1024]") || v.message.contains("[1, 1024]"),
        "expected parent-shape [1,1024] in message, got: {}",
        v.message
    );
}

// ── binary layer: `pto-diff --stage1 <file>` end-to-end ──

fn pto_diff_bin() -> PathBuf {
    // CARGO_BIN_EXE_<name> is set by cargo test for binary crates in the
    // same package. Underscore-substituted because the binary is named
    // `pto-diff` (with a hyphen).
    let raw = env!("CARGO_BIN_EXE_pto-diff");
    PathBuf::from(raw)
}

#[test]
fn pto_diff_stage1_clean_softmax_reports_zero() {
    let bin = pto_diff_bin();
    let fix = fixture_path("stage1_softmax_clean.mlir");
    let out = Command::new(&bin)
        .arg("--stage1")
        .arg(&fix)
        .output()
        .expect("spawn pto-diff");
    assert!(
        out.status.success(),
        "pto-diff failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("0 safety findings"),
        "expected `0 safety findings`, got: {stdout}"
    );
    assert!(
        !stdout.contains("oob-partition"),
        "expected no oob-partition mention on clean fixture, got: {stdout}"
    );
}

#[test]
fn pto_diff_stage1_oob_reports_oob_partition() {
    let bin = pto_diff_bin();
    let fix = fixture_path("stage1_oob_partition.mlir");
    let out = Command::new(&bin)
        .arg("--stage1")
        .arg(&fix)
        .output()
        .expect("spawn pto-diff");
    assert!(
        out.status.success(),
        "pto-diff failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("oob-partition"),
        "expected oob-partition finding on OOB fixture, got: {stdout}"
    );
    assert!(
        stdout.contains("bad"),
        "expected mention of @bad func, got: {stdout}"
    );
}

#[test]
fn pto_diff_stage1_rejects_combined_with_stage2_input() {
    let bin = pto_diff_bin();
    let fix = fixture_path("stage1_oob_partition.mlir");
    let out = Command::new(&bin)
        .arg("--stage1")
        .arg(&fix)
        .arg("some_stage2.mlir")
        .output()
        .expect("spawn pto-diff");
    assert!(
        !out.status.success(),
        "expected non-zero exit on conflicting args"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("mutually exclusive"),
        "expected mutual-exclusion error, got: {stderr}"
    );
}
