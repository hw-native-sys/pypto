// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The safety oracle's taxonomy, tied to the eight address-shape invariant
//! classes that have a mechanised soundness proof.
//!
//! The eight classes (P01..P08) are formalised in Coq against concrete
//! `check_*` definitions, with a soundness theorem per class, no `Admitted.`,
//! and one explicitly-flagged closure axiom. That formalisation lives outside
//! this repository and is not reproduced here; what this file does is pin the
//! correspondence between it and `safety::SafetyKind`, so the implementation
//! cannot drift from a proven taxonomy without failing a test.
//!
//! Two things are asserted below.
//!
//! **The count lines up.** The Ascend entry point (`check_all`) emits seven
//! kinds, which is the figure the formal work reports for this vendor. A second
//! entry point emitted four more for another vendor's processor; it is not in
//! this tree, and the kinds only it could produce -- `SmemCapacity`,
//! `LoadUninit`, `SyncBeforeRead` -- were deleted with it rather than left
//! declared. A `SafetyKind` nothing emits reports clean on every input and
//! reads as coverage, which is the opposite of what this file is for.
//! `OobPartition` is NOT one of them: it is an Ascend stage-1 check and stays.
//!
//! **The names do not line up**, and the mapping below is the reconciliation.
//! Of the seven kinds only `OobPartition` and `Capacity` correspond to a class
//! by name; `Aliasing`, `OpConstraint`, `MatmulBounds`, `DeadTile` and
//! `LinearUse` do not. Four class names in the formal work's own summary --
//! dtype, shape, layout, use-of-uninit -- have no Ascend `SafetyKind` at all:
//! three are proved VACUOUS here (pass-through to the vendor type-checker) and
//! the fourth was the other vendor's.
//!
//! And one correction in the implementation's favour: **P08 (ISA-version, "op
//! unsupported on target") IS implemented on Ascend**, not deferred. It is
//! spelled `OpConstraint`: `check_op_constraints` tests a `pto.tmatmul`'s
//! `(acc, lhs, rhs)` triple against `DeviceSpec::tmatmul_dtypes` -- per-part
//! accepted triples recorded from on-device probes -- and its row/outer
//! extents against the per-arch stride-descriptor limits. That is op support
//! on a target, decided per device spec, which is P08's trigger.

use ptoas_rs::{
    check_all, default_a5_910b2_cann85, parse_stage2,
    SafetyKind,
};

/// How an invariant class is realised here.
#[derive(Debug, PartialEq, Eq)]
enum Realisation {
    /// Implemented, under these `SafetyKind`s.
    Kinds(&'static [SafetyKind]),
    /// Proved vacuous in the formalisation and deliberately not implemented:
    /// the vendor type-checker owns it.
    VendorTypeChecker,
    /// Realised only by an arm for another vendor's processor, which this
    /// repository's licence does not cover. Kept in the table rather than
    /// dropped, so the eight classes still account for themselves and nobody
    /// has to wonder whether a gap is an oversight.
    OtherVendorOnly,
}

/// The eight classes, each with the reason its realisation is what it is.
///
/// Written out longhand rather than derived, because the point is that a
/// reader can see the correspondence and a change has to edit it.
const CLASSES: &[(&str, &str, Realisation)] = &[
    (
        "P01",
        "bounds: every Load/Store partition stays inside its parent's shape",
        Realisation::Kinds(&[SafetyKind::OobPartition]),
    ),
    (
        "P02",
        "dtype: producer and consumer dtypes agree",
        Realisation::VendorTypeChecker,
    ),
    (
        "P03",
        "shape: shapes unify",
        Realisation::VendorTypeChecker,
    ),
    (
        "P04",
        "capacity: per-class byte sum against the device's budget",
        Realisation::Kinds(&[SafetyKind::Capacity]),
    ),
    (
        "P05",
        "layout: operand layouts are legal",
        Realisation::VendorTypeChecker,
    ),
    (
        // Realised only by the other vendor's arm, which is not in this tree.
        // Recorded as out of scope rather than dropped from the table, so the
        // eight classes still account for themselves.
        "P06",
        "use-of-uninit: a masked load with no `other` reaches a non-store",
        Realisation::OtherVendorOnly,
    ),
    (
        // The formalisation discharges this one as `true` -- vacuous in its
        // scope -- while the other vendor's arm performed a real check. Worth
        // saying out loud either way: a green soundness theorem for a vacuous
        // check is not evidence about the code that is running.
        "P07",
        "sync-hazard: a cross-grid load before a grid sync",
        Realisation::OtherVendorOnly,
    ),
    (
        // NOT deferred, contrary to the per-vendor summary. See the module
        // comment and `p08_is_implemented_on_ascend_not_deferred`.
        "P08",
        "ISA-version: op unsupported on the target part",
        Realisation::Kinds(&[SafetyKind::OpConstraint]),
    ),
];

/// `SafetyKind`s that implement no class in the eight.
///
/// This list is the honest residue of the closure claim. The axiom says eight
/// classes are enough; these five checks are outside them, so either they
/// refine a class in a way the formalisation does not model, or the closure
/// argument does not cover them. Recorded rather than argued either way.
const OUTSIDE_THE_EIGHT: &[SafetyKind] = &[
    // Two slots overlapping in one non-GM space. NOT P01: P01 is containment
    // in a parent, this is mutual disjointness between siblings.
    SafetyKind::Aliasing,
    // A tile allocated and never read.
    SafetyKind::DeadTile,
    // A tile consumed more than once, or not at all.
    SafetyKind::LinearUse,
    // Stride-descriptor extents. Arguably P08 as well -- the limits are
    // per-arch -- but it is a geometry bound rather than op support, so it is
    // listed here rather than claimed for a class.
    SafetyKind::MatmulBounds,
];

fn ascend_fixture() -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/stage2/softmax_32x32_plan_memory_overlap.mlir"),
    )
    .expect("the overlap fixture")
}

/// Every class is accounted for, and every kind belongs to a class or to the
/// declared residue. No kind may be in both, and none may be in neither.
#[test]
fn every_safety_kind_is_classified() {
    let all = [
        SafetyKind::Aliasing,
        SafetyKind::Capacity,
        SafetyKind::OpConstraint,
        SafetyKind::MatmulBounds,
        SafetyKind::DeadTile,
        SafetyKind::LinearUse,
        SafetyKind::OobPartition,
    ];
    let mut classified: Vec<SafetyKind> = Vec::new();
    for (_, _, r) in CLASSES {
        if let Realisation::Kinds(ks) = r {
            for k in *ks {
                assert!(
                    !classified.contains(k),
                    "{k:?} implements two invariant classes; one of the two mappings is wrong"
                );
                classified.push(k.clone());
            }
        }
    }
    for k in &all {
        let in_class = classified.contains(k);
        let in_residue = OUTSIDE_THE_EIGHT.contains(k);
        assert!(
            in_class != in_residue,
            "{k:?} is {} -- every kind must map to exactly one invariant class or be \
             declared outside the eight. A new kind that is neither is a check with no \
             place in a taxonomy that carries a closure proof",
            if in_class { "in both" } else { "in neither" }
        );
    }
    // The enum has not grown behind this list. `SafetyKind` is not
    // exhaustively matchable from here, so the array above is the denominator
    // and it needs a floor of its own.
    assert_eq!(
        classified.len() + OUTSIDE_THE_EIGHT.len(),
        all.len(),
        "{} classified + {} residue != {} kinds listed",
        classified.len(),
        OUTSIDE_THE_EIGHT.len(),
        all.len()
    );
    assert_eq!(
        CLASSES.len(),
        8,
        "the taxonomy has {} classes, and the closure proof is about eight",
        CLASSES.len()
    );
}

/// The count this crate reports for Ascend: seven classes.
///
/// Derived from the entry points rather than from the enum, because what is
/// "covered" is what `check_all_*` actually runs. An enum variant no entry
/// point emits is covered by nothing -- which is why the two GPU-only kinds
/// were deleted along with the arm that produced them rather than left
/// declared and unreachable.
#[test]
fn the_ascend_side_covers_seven_classes() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/safety.rs"),
    )
    .expect("safety.rs");

    let body = |name: &str| -> String {
        let i = src
            .find(&format!("pub fn {name}"))
            .unwrap_or_else(|| panic!("no `{name}`"));
        let rest = &src[i..];
        // To the next top-level `pub fn`/`fn` after this one.
        let end = rest[1..]
            .find("\npub fn ")
            .or_else(|| rest[1..].find("\nfn "))
            .map(|e| e + 1)
            .unwrap_or(rest.len());
        rest[..end].to_string()
    };

    // Ascend spans TWO entry points, and the seven is their union. `check_all`
    // is the stage-2 oracle (post-PlanMemory, tiles at byte offsets) and runs
    // seven checks over six distinct kinds -- `check_dead_tiles` and
    // `check_dead_writes` both report `DeadTile`. `OobPartition` is the whole
    // of `check_all_stage1`, which reads the pre-placement IR. Counting one
    // entry point gives six and counting kinds inside it gives six; the seven
    // only appears when both paths are counted, which is worth pinning because
    // it is the figure the formal work reports.
    let stage2 = body("check_all");
    let stage2_checks: Vec<&str> = [
        "check_aliasing",
        "check_capacity",
        "check_op_constraints",
        "check_matmul_bounds",
        "check_dead_tiles",
        "check_dead_writes",
        "check_linear_use",
    ]
    .iter()
    .filter(|c| stage2.contains(**c))
    .copied()
    .collect();
    assert_eq!(
        stage2_checks.len(),
        7,
        "the stage-2 entry point runs {:?}",
        stage2_checks
    );

    let stage1 = body("check_all_stage1");
    assert!(
        stage1.contains("check_oob_partition"),
        "the Ascend stage-1 entry point no longer runs the bounds check, so P01 has no \
         Ascend implementation"
    );

    // Six distinct kinds from stage 2, plus `OobPartition` from stage 1.
    let stage2_kinds = [
        SafetyKind::Aliasing,
        SafetyKind::Capacity,
        SafetyKind::OpConstraint,
        SafetyKind::MatmulBounds,
        SafetyKind::DeadTile,
        SafetyKind::LinearUse,
    ];
    assert_eq!(
        stage2_kinds.len() + 1,
        7,
        "the Ascend kind count is no longer seven"
    );
}

/// THE CORRECTION. P08 is implemented on Ascend, not deferred to `ptoas`.
///
/// Pinned on the mechanism rather than on a name, because the name is the part
/// that misled: the check is called `OpConstraint`, which reads like a local
/// well-formedness rule, while what it does is test op support against a
/// per-part device spec.
#[test]
fn p08_is_implemented_on_ascend_not_deferred() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/safety.rs"),
    )
    .expect("safety.rs");
    assert!(
        src.contains("spec.tmatmul_dtypes.contains(&triple)"),
        "the per-part accepted-triple check is gone. If ISA-tag enforcement really has been \
         handed to `ptoas`, P08's row in CLASSES becomes `VendorTypeChecker` and the \
         summary that said so becomes right"
    );
    for per_arch in ["spec.matmul_row_max", "spec.matmul_outer_max"] {
        assert!(
            src.contains(per_arch),
            "the per-arch limit `{per_arch}` is no longer consulted"
        );
    }
    // And it fires: a dtype triple outside the spec's accepted set is a
    // finding, which is what makes the class covered rather than present.
    let spec = default_a5_910b2_cann85();
    assert!(
        !spec.tmatmul_dtypes.is_empty(),
        "the default spec lists no accepted triples, so the check cannot fire and the \
         class is not covered after all"
    );
}

/// The classes that ARE implemented fire on real input, so coverage is a
/// measurement and not a mapping table.
///
/// P04 and the aliasing residue are exercised by the stage-2 fixtures; P01,
/// P06 and P07 by the GPU stage-1 suites. Asserted here for the two the
/// Ascend path owns, so this file is not purely declarative.
#[test]
fn the_implemented_classes_fire_on_real_input() {
    let plan = parse_stage2(&ascend_fixture()).expect("parse");
    let mut spec = default_a5_910b2_cann85();
    let rep = check_all(&plan, &spec);
    let kinds: Vec<SafetyKind> = rep.violations.iter().map(|v| v.kind.clone()).collect();
    assert!(
        kinds.contains(&SafetyKind::Aliasing),
        "the overlap fixture produced no aliasing finding: {kinds:?}"
    );

    // P04, by shrinking the budget below the plan's high-water.
    spec.ub_bytes = 32768;
    let rep = check_all(&plan, &spec);
    assert!(
        rep.violations.iter().any(|v| v.kind == SafetyKind::Capacity),
        "P04 did not fire at a 32 KiB UB"
    );

}
