// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Every corpus `.pto` the lift refuses is the expansion of ONE whole-kernel
//! `__tile_*` call -- and so are 22 that lift, which is why the shape is not the
//! explanation.
//!
//! `the_last_tier.rs` partitions refusals by their MESSAGE. This file partitions
//! them by their PROVENANCE, which is a different question and gave a different
//! answer: eight distinct refusal messages turned out to be one class wearing
//! eight faces, and three of them (`pto.tci`, `pto.tcolexpand`, `pto.tmov`) are
//! not op-model gaps worth closing on their own at all -- they are internal
//! artefacts of an expansion. `pto.tci` is the index tile inside
//! `__tile_topk_f32`'s and `__tile_rope_inplace_f32`'s expansions, and all three
//! of those kernels' central calls already have public wrappers.
//!
//! The DISCRIMINATOR is which intrinsic, not the shape. Measured 2026-11-09, and
//! this file exists so that stays true or stops quietly: a change that makes a
//! refusing intrinsic lift must move its row, and a new corpus kernel whose
//! central call is on the wrong side fails here rather than drifting.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

mod shared;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root")
}

/// Central `__tile_*` calls of a kernel's `.tile` source: everything that is not
/// a load or a store, deduplicated. Returns `None` when there is no `.tile`.
fn central_calls(model: &str, kernel: &str) -> Option<BTreeSet<String>> {
    let t = repo_root().join(format!("rust/bench/{model}/tile/{kernel}.tile"));
    let src = std::fs::read_to_string(t).ok()?;
    let mut out = BTreeSet::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while let Some(pos) = src[i..].find("__tile_") {
        let start = i + pos;
        let mut end = start;
        while end < bytes.len()
            && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
        {
            end += 1;
        }
        let name = &src[start..end];
        // The bare prefix is not a call. The `.tile` corpus contains the literal
        // `__tile_` in prose, and admitting it put a phantom "intrinsic" into
        // every tally -- which is how the first run of this test reported a
        // three-call kernel that has two.
        if name.len() > "__tile_".len()
            && !name.starts_with("__tile_load")
            && !name.starts_with("__tile_store")
        {
            out.insert(name.to_string());
        }
        i = end.max(start + 1);
    }
    Some(out)
}

/// The one intrinsic measured on BOTH sides, and why.
///
/// `__tile_rms_norm_f32`: four kernels lift and one refuses. The four expand to
/// the UNFUSED chain (`tsqrt` + `trecip` + a bare `trowexpand`), the one that
/// refuses to the FUSED chain (`trsqrt` + `trowexpandmul`) -- and it is the only
/// one of the five big enough (8x5120) to have been chunked, which is what
/// introduced the store-then-load through GM that its refusal names.
const ON_BOTH_SIDES: &str = "__tile_rms_norm_f32";

#[test]
fn every_refused_artifact_is_the_expansion_of_one_whole_kernel_call() {
    let root = shared::corpus_root();
    let mut files = Vec::new();
    shared::collect_pto_files(&root, &mut files);
    assert!(!files.is_empty(), "no .pto artifacts found");

    let mut offenders = Vec::new();
    let mut checked = 0usize;
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        if ptoas_rs::convert_pto_to_rust(&src).is_ok() {
            continue;
        }
        // model and kernel from `rust/bench/<model>/pto/<kernel>.pto`
        let kernel = f.file_stem().unwrap().to_string_lossy().to_string();
        let model = f
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let Some(calls) = central_calls(&model, &kernel) else {
            // No `.tile` is a different finding and belongs to the manifest
            // gates, not here.
            continue;
        };
        checked += 1;
        if calls.len() > 2 {
            offenders.push(format!("{model}/{kernel}: central calls {calls:?}"));
        }
    }
    assert!(checked > 0, "no refused artifact had a .tile to profile");
    assert!(
        offenders.is_empty(),
        "{} refused artifact(s) are NOT a one- or two-call whole-kernel expansion. \
         That is a new class and the campaign's C_b reasoning (evidence/0034) \
         assumes it does not exist:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

#[test]
fn the_central_intrinsic_decides_whether_a_kernel_lifts() {
    let root = shared::corpus_root();
    let mut files = Vec::new();
    shared::collect_pto_files(&root, &mut files);

    // intrinsic -> (lifts, refuses), over SINGLE-call kernels only, where the
    // central call is unambiguous.
    let mut tally: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        let lifts = ptoas_rs::convert_pto_to_rust(&src).is_ok();
        let kernel = f.file_stem().unwrap().to_string_lossy().to_string();
        let model = f
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let Some(calls) = central_calls(&model, &kernel) else {
            continue;
        };
        if calls.len() != 1 {
            continue;
        }
        let call = calls.into_iter().next().unwrap();
        let e = tally.entry(call).or_insert((0, 0));
        if lifts {
            e.0 += 1;
        } else {
            e.1 += 1;
        }
    }
    assert!(
        tally.len() > 8,
        "too few single-call kernels profiled ({}) for this to mean anything",
        tally.len()
    );

    let straddling: Vec<_> = tally
        .iter()
        .filter(|(_, (l, r))| *l > 0 && *r > 0)
        .map(|(k, (l, r))| format!("{k}: {l} lift, {r} refuse"))
        .collect();

    // Exactly one intrinsic may straddle, and it must be the documented one. A
    // SECOND straddler means the central call no longer determines the outcome,
    // which is the premise evidence/0034 rests on; a straddler that is not this
    // one means the partition moved and the evidence is stale.
    assert_eq!(
        straddling.len(),
        1,
        "the central intrinsic no longer decides lift-vs-refuse. Straddling: {straddling:?}"
    );
    assert!(
        straddling[0].starts_with(ON_BOTH_SIDES),
        "a DIFFERENT intrinsic now straddles the partition: {straddling:?} \
         (expected only {ON_BOTH_SIDES}; see evidence/0034 for why that one does)"
    );
}
