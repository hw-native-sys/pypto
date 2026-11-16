// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Integration tests for the stage-2 parser + safety oracle.
//!
//! Each test parses a fixture under `tests/fixtures/`, runs `check_all`
//! against the default a5_910b2_cann85 DeviceSpec, and asserts on the
//! set of `SafetyKind`s produced. Keeping the fixtures small and
//! hand-written lets a reader see exactly which rule fires without
//! reading through a 100-line ptoas dump.

use ptoas_rs::{
    check_all, default_a5_910b2_cann85, diff_plans, parse_stage2, AddressSpace, PtoArch,
    SafetyKind, Severity,
};
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", name]
        .iter()
        .collect();
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", p.display()))
}

fn has_kind(rep: &ptoas_rs::SafetyReport, k: SafetyKind) -> bool {
    rep.violations.iter().any(|v| v.kind == k)
}

// ── end-to-end: real ptoas output parses + lands with zero errors ──

#[test]
fn softmax_stage2_parses_and_is_clean() {
    let src = fixture("softmax_stage2.mlir");
    let plan = parse_stage2(&src).expect("softmax parse");
    assert_eq!(plan.arch, PtoArch::A5);
    assert_eq!(plan.funcs.len(), 1);
    let f = &plan.funcs[0];
    assert_eq!(f.name, "tile_softmax");
    // 7 tiles (see fixture): the main input tile + scratch + 4 intermediates + output.
    assert!(f.slots.len() >= 6, "want >= 6 slots, got {}", f.slots.len());
    assert!(f.slots.values().all(|s| s.space == AddressSpace::Ub));

    let spec = default_a5_910b2_cann85();
    let rep = check_all(&plan, &spec);
    // Real ptoas output for a working kernel — no errors, maybe dead-tile
    // warnings for scratch slots that the ops use as reusable scratch.
    let errors: Vec<_> = rep
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "softmax should have no safety errors, got {:?}",
        errors
    );
}

#[test]
fn i8_matmul_stage2_parses_and_is_clean() {
    let src = fixture("i8_matmul_stage2.mlir");
    let plan = parse_stage2(&src).expect("i8 matmul parse");
    assert_eq!(plan.arch, PtoArch::A5);
    assert_eq!(plan.funcs.len(), 2, "expected kv_proj + gate_up funcs");
    // Both funcs use all five on-chip spaces (mat, left, right, acc, scaling).
    for f in &plan.funcs {
        let spaces: std::collections::BTreeSet<_> = f.slots.values().map(|s| s.space).collect();
        for must_have in [
            AddressSpace::Left,
            AddressSpace::Right,
            AddressSpace::Acc,
            AddressSpace::Mat,
            AddressSpace::Scaling,
        ] {
            assert!(
                spaces.contains(&must_have),
                "func `{}` missing {:?}",
                f.name,
                must_have
            );
        }
    }

    let spec = default_a5_910b2_cann85();
    let rep = check_all(&plan, &spec);
    let errors: Vec<_> = rep
        .violations
        .iter()
        .filter(|v| v.severity == Severity::Error)
        .collect();
    assert!(
        errors.is_empty(),
        "i8 matmul should have no safety errors, got {:?}",
        errors
    );
}

// ── synthetic fixtures: each triggers exactly one rule ──

#[test]
fn synthetic_overlap_triggers_aliasing() {
    let src = fixture("synthetic_overlap.mlir");
    let plan = parse_stage2(&src).unwrap();
    let spec = default_a5_910b2_cann85();
    let rep = check_all(&plan, &spec);
    assert!(
        has_kind(&rep, SafetyKind::Aliasing),
        "expected Aliasing, got {:?}",
        rep.violations
    );
}

#[test]
fn synthetic_overflow_triggers_capacity() {
    let src = fixture("synthetic_overflow.mlir");
    let plan = parse_stage2(&src).unwrap();
    let spec = default_a5_910b2_cann85();
    let rep = check_all(&plan, &spec);
    assert!(
        has_kind(&rep, SafetyKind::Capacity),
        "expected Capacity, got {:?}",
        rep.violations
    );
}

#[test]
fn synthetic_bad_layout_triggers_op_constraint() {
    let src = fixture("synthetic_bad_layout.mlir");
    let plan = parse_stage2(&src).unwrap();
    let spec = default_a5_910b2_cann85();
    let rep = check_all(&plan, &spec);
    assert!(
        has_kind(&rep, SafetyKind::OpConstraint),
        "expected OpConstraint, got {:?}",
        rep.violations
    );
    // Specifically the "expected left" error.
    let msg = rep
        .violations
        .iter()
        .find(|v| v.kind == SafetyKind::OpConstraint)
        .map(|v| v.message.clone())
        .unwrap_or_default();
    assert!(msg.contains("expected left"), "got `{msg}`");
}

// ── plan-diff sanity ──

#[test]
fn plan_diff_same_plan_is_empty() {
    let src = fixture("softmax_stage2.mlir");
    let a = parse_stage2(&src).unwrap();
    let b = parse_stage2(&src).unwrap();
    let d = diff_plans(&a, &b);
    assert!(
        d.is_empty(),
        "same plan diffed against itself should be empty: {:?}",
        d.entries
    );
}

#[test]
fn plan_diff_different_funcs_reports_presence() {
    let a = parse_stage2(&fixture("softmax_stage2.mlir")).unwrap();
    let b = parse_stage2(&fixture("i8_matmul_stage2.mlir")).unwrap();
    let d = diff_plans(&a, &b);
    assert!(!d.is_empty());
    // softmax has only `tile_softmax`; i8 has `kv_proj…` + `gate_up…`.
    assert!(d
        .entries
        .iter()
        .any(|e| matches!(e.kind, ptoas_rs::DiffKind::FuncOnlyInA)));
    assert!(d
        .entries
        .iter()
        .any(|e| matches!(e.kind, ptoas_rs::DiffKind::FuncOnlyInB)));
}
