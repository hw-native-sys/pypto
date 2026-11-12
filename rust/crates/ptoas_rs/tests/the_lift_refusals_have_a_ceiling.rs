// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Most of what the lift refuses, it refuses BY DESIGN -- so the tile-rs axis has
//! a ceiling, and it is far below the corpus.
//!
//! Measured 2026-11-09 over the three model corpora: 60 stored `.pto`, of which
//! the lift accepts 20 and refuses 40. Those 40 partition by MECHANISM, and the
//! partition is the point -- a bare refusal count reads like a backlog and most
//! of it is not one:
//!
//! ```text
//!  19  whole-kernel expansion, geometry from an emit-time knob   STRUCTURAL
//!   9  computed tile WRITTEN by several ops (one name, several values)  CONTRACT
//!   3  aic/aiv split pair                                        STRUCTURAL
//!   2  strided partition view                                    STRUCTURAL
//!   2  op not modelled                                           closable
//!   1  barrier over an in-place write                            CONTRACT
//!   1  computed tile read twice ACROSS an in-place write of it   CONTRACT
//! ```
//!
//! RE-MEASURED 2026-11-09, after `Tile` became `Copy` under an operator
//! authorisation to extend `tile_std`. The `Tile`-not-`Copy` class was 10 and is
//! now 7, re-tokened: "read twice" is no longer a refusal, because in PTO a
//! compute op reads its inputs and writes a NEW tile, so nothing was being
//! consumed. What survives is the half `Copy` cannot express -- a tile whose
//! CONTENTS CHANGE between its reads -- and that is a change to the KERNEL, not
//! to the lift.
//!
//! The lift went from 20 of 60 model artefacts to 23 and the ceiling is UNCHANGED
//! at 25/72 = 34.7%, because the kernels `Copy` unblocked were already going to be
//! counted by the two remaining closables. Worth recording that the first
//! relaxation looked much better: it reported 26 lifting and a 28/72 ceiling, and
//! three of those lifts were UNSAFE -- a tile written by several ops, which the
//! lift emits as several shadowed `let` bindings. The tighter condition took them
//! back. A ceiling that moves when a check is relaxed is a sign to re-derive the
//! check, not the ceiling.
//!
//! The arithmetic below is therefore re-derived rather than adjusted -- the `v_row` class is gone entirely (closed earlier the same day)
//! and `op not modelled` is down to the two topk kernels, which need four models
//! and a contract decision, not two arms.
//!
//! # Why the 19 are not a backlog
//!
//! Each is the expansion of ONE whole-kernel `__tile_*` call, and the call does
//! not determine the artefact: the tiling comes from an emit-time environment
//! knob, so several corpus artefacts are the same call at different geometries.
//! Lifting them would emit IDENTICAL Rust for files that differ -- a lift that is
//! not an inverse, which is the exact shape of defect this corpus exists to
//! catch. The refusal is the lift being right, and closing it would be a
//! regression wearing a KPI's clothes.
//!
//! The same argument already refused `mx4pk_v41_5120x2304` and was accepted
//! there, so this file records it once for the whole class rather than nineteen
//! times.
//!
//! # What that does to the ceiling
//!
//! `C_b` counts kernels carrying BOTH `ptoas-assembles` and `rustc-compiles`
//! against a fixed denominator of 72. Its denominator must not move -- that is
//! the campaign's own anti-gaming rule -- but the REACHABLE value is not 72:
//!
//! ```text
//! 72   declared kernels
//! -12  have no .pto at all, so there is nothing to lift
//! -24  structural: unliftable by design (19 + 3 + 2)
//! -11  declared-contract: needs an owner decision, not a session (10 + 1)
//! ----
//!  25  the ceiling if every closable refusal closes   = 34.7%
//!  36  the ceiling if the contracts open too          = 50.0%
//! ```
//!
//! So a session can move this axis by at most **five kernels**, and the honest
//! report of a 38.9% reading is that it is already near a 34.7% practical
//! ceiling, not a third of the way to 100%. This file exists so that arithmetic
//! is checked rather than remembered.
//!
//! # What it does NOT assert
//!
//! That the counts above are each correct in their classification -- the regexes
//! below key on wording the lift produces, and wording can drift. It asserts the
//! SHAPE: that the structural classes dominate, that the closable remainder is
//! small, and that the biggest single class is the whole-kernel expansion. If a
//! refusal's wording changes, this file goes red with an unclassified count and
//! the reader re-derives the partition, which is the intent.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `(regex-ish needle, class)`. First match wins, so order matters.
const CLASSES: [(&str, &str); 11] = [
    ("does exist, at the", "STRUCTURAL/expansion"),
    ("written in place by one of them", "CONTRACT/in-place"),
    ("different values", "CONTRACT/multi-write"),
    ("cube and vector halves", "STRUCTURAL/aic-aiv"),
    ("is not modelled", "closable/op-model"),
    ("elements apart in memory", "STRUCTURAL/strided-view"),
    ("window of a", "STRUCTURAL/strided-view"),
    ("partial or dynamic v_row", "closable/v_row"),
    ("cannot be dropped in", "CONTRACT/barrier"),
    // The two the 2026-11-10 attention work exposed, neither structural: a
    // `pto.tmov` refused because tile_std's only copy is a multiply by one
    // (a true move intrinsic would close it), and a fused row-broadcast over
    // an R x 1 column that `tile_row_expand_f32`'s 1x1 source cannot express.
    ("would lift to `tile_copy_f16`", "closable/tile-copy-f16"),
    ("genuine R x 1 column", "closable/row-broadcast"),
];

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench must be reachable")
}

#[test]
fn the_refusals_are_mostly_structural_and_the_closable_remainder_is_small() {
    let exe = env!("CARGO_BIN_EXE_ptoas-rs");
    let out_dir = std::env::temp_dir().join("lift_ceiling_probe");
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut total = 0usize;
    let mut lifted = 0usize;
    let mut by_class: BTreeMap<&str, usize> = BTreeMap::new();
    let mut unclassified: Vec<String> = Vec::new();

    for model in ["deepseek_v41", "glm_5_3", "kimi_k3"] {
        let dir = bench().join(model).join("pto");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for p in paths {
            if p.extension().and_then(|e| e.to_str()) != Some("pto") {
                continue;
            }
            total += 1;
            let out = out_dir.join("probe.rs");
            let run = Command::new(exe).arg(&p).arg("-o").arg(&out).output().unwrap();
            let msg = format!(
                "{}{}",
                String::from_utf8_lossy(&run.stdout),
                String::from_utf8_lossy(&run.stderr)
            );
            if !msg.contains("refused") {
                lifted += 1;
                continue;
            }
            match CLASSES.iter().find(|(needle, _)| msg.contains(needle)) {
                Some((_, class)) => *by_class.entry(class).or_default() += 1,
                None => unclassified.push(format!(
                    "{}: {}",
                    p.file_name().unwrap().to_string_lossy(),
                    msg.lines().last().unwrap_or("").chars().take(90).collect::<String>()
                )),
            }
        }
    }

    assert!(
        unclassified.is_empty(),
        "{} refusal(s) match no declared class. A refusal shape changed, or a new \
         one arrived; re-derive the partition in this file's note before trusting \
         the ceiling arithmetic in it.\n  {}",
        unclassified.len(),
        unclassified.join("\n  ")
    );
    // The corpus floor, so the gate cannot pass by finding nothing.
    assert!(
        total >= 55,
        "expected at least 55 stored model .pto, saw {total} -- a census over an \
         empty list satisfies every assertion below it"
    );

    let structural: usize = by_class
        .iter()
        .filter(|(k, _)| k.starts_with("STRUCTURAL"))
        .map(|(_, v)| *v)
        .sum();
    let contract: usize = by_class
        .iter()
        .filter(|(k, _)| k.starts_with("CONTRACT"))
        .map(|(_, v)| *v)
        .sum();
    let closable: usize = by_class
        .iter()
        .filter(|(k, _)| k.starts_with("closable"))
        .map(|(_, v)| *v)
        .sum();
    let refused = structural + contract + closable;

    // THE CLAIM. Not the exact counts -- those move as kernels land -- but the
    // shape: a session's reachable headroom on this axis is small, and most of
    // what looks like a backlog is the lift being correct.
    assert!(
        structural > closable * 2,
        "structural refusals ({structural}) no longer dominate the closable ones \
         ({closable}). Either the lift stopped refusing whole-kernel expansions -- \
         check that it has not started emitting identical Rust for artefacts that \
         differ -- or the closable set grew and this file's ceiling arithmetic is \
         stale.\n  {by_class:?}"
    );
    assert!(
        closable <= 8,
        "{closable} closable refusals, up from the 2 remaining on 2026-11-09. That \
         is good news and makes this file's ceiling wrong: re-measure it.\n  \
         {by_class:?}"
    );
    assert_eq!(
        lifted + refused,
        total,
        "every .pto must either lift or refuse; {lifted} + {refused} != {total}"
    );
}
