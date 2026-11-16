// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! No lifted kernel stores one tile twice — the guarantee `Copy` gave up.
//!
//! `Tile<R, C, T>` was deliberately not `Copy`, and its own doc said why: "each
//! tile must be consumed by exactly one store operation, preventing double-DMA
//! bugs." That is a real guarantee and the type system was enforcing it for free.
//!
//! It was also enforcing linearity on a handle whose compute uses are READS. In
//! PTO every compute op reads its inputs and writes a NEW tile, so nothing is
//! consumed, and ten corpus kernels could not be lifted because a computed tile
//! read by several ops would have been moved twice. `Tile` is `Copy` as of
//! 2026-11-09 and five of those kernels now lift and pass `--verify rustc`.
//!
//! So this file holds the line the type system used to. A tile passed to two
//! stores is two DMAs of the same buffer: at best redundant, and at worst two
//! writes to overlapping GM that the kernel's author did not intend.
//!
//! # What it checks, and the limit worth stating
//!
//! Every stored `.rs` under `bench/*/tile-rs/` plus a fresh lift of every stored
//! `.pto`, so a kernel cannot pass by not having been regenerated. It matches
//! `tile_store_*(dst, t)` and counts how many times each tile NAME appears as a
//! store's source.
//!
//! It is a TEXT check over generated code, which is sound here for a reason that
//! would not hold for hand-written Rust: the lift emits one `let tN = ...` per PTO
//! value and never shadows, so a name is a value. A hand-written kernel could
//! shadow `t` and defeat this, which is why the corpus `.rs` are in scope and
//! `examples/` are not -- those are authored, and for them the right check is a
//! compiler one. If authored kernels ever land in `bench/*/tile-rs/`, this needs
//! to become a real analysis rather than a grep.
//!
//! The in-place hazard is NOT this file's: a tile read twice whose contents change
//! between the reads is refused by the lift itself (`emitter.rs`), because
//! `Copy` makes multiple reads expressible but not reads of two different values
//! through one name.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn bench() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench")
        .canonicalize()
        .expect("the bench must be reachable")
}

/// `tile -> how many stores take it as their source`, from lifted Rust text.
fn store_sources(rust: &str) -> BTreeMap<String, usize> {
    let mut out: BTreeMap<String, usize> = BTreeMap::new();
    for line in rust.lines() {
        let Some(rest) = line.split_once("tile_store_") else {
            continue;
        };
        // `tile_store_f32::<R, C>(ptr as *mut f32, tN);`
        let Some(args) = rest.1.split_once('(') else { continue };
        let Some(inside) = args.1.rsplit_once(')') else { continue };
        let Some(last) = inside.0.rsplit(',').next() else {
            continue;
        };
        let name: String = last
            .trim()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        *out.entry(name).or_default() += 1;
    }
    out
}

fn check(label: &str, rust: &str, offenders: &mut Vec<String>) {
    for (tile, n) in store_sources(rust) {
        if n > 1 {
            offenders.push(format!("{label}: `{tile}` is stored {n} times"));
        }
    }
}

#[test]
fn neither_a_stored_nor_a_freshly_lifted_kernel_stores_a_tile_twice() {
    let exe = env!("CARGO_BIN_EXE_ptoas-rs");
    let tmp = std::env::temp_dir().join("double_store_probe");
    std::fs::create_dir_all(&tmp).unwrap();
    let mut offenders = Vec::new();
    let mut stored = 0usize;
    let mut lifted = 0usize;

    for model in ["deepseek_v41", "glm_5_3", "kimi_k3", "engine", "playground16"] {
        // (a) the .rs that are checked in
        if let Ok(entries) = std::fs::read_dir(bench().join(model).join("tile-rs")) {
            let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            for p in paths {
                if p.extension().and_then(|e| e.to_str()) != Some("rs") {
                    continue;
                }
                stored += 1;
                let text = std::fs::read_to_string(&p).unwrap();
                check(&p.file_name().unwrap().to_string_lossy(), &text, &mut offenders);
            }
        }
        // (b) a FRESH lift of every .pto, so a stale .rs cannot hide a regression
        if let Ok(entries) = std::fs::read_dir(bench().join(model).join("pto")) {
            let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            paths.sort();
            for p in paths {
                if p.extension().and_then(|e| e.to_str()) != Some("pto") {
                    continue;
                }
                let out = tmp.join("probe.rs");
                let run = Command::new(exe).arg(&p).arg("-o").arg(&out).output().unwrap();
                let msg = format!(
                    "{}{}",
                    String::from_utf8_lossy(&run.stdout),
                    String::from_utf8_lossy(&run.stderr)
                );
                if msg.contains("refused") {
                    continue;
                }
                lifted += 1;
                if let Ok(text) = std::fs::read_to_string(&out) {
                    check(
                        &format!("fresh lift of {}", p.file_name().unwrap().to_string_lossy()),
                        &text,
                        &mut offenders,
                    );
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} double-store(s). `Tile` is `Copy`, so the type system no longer refuses this -- \
         two stores of one tile are two DMAs of the same buffer, redundant at best and two \
         overlapping GM writes at worst. This file is the guarantee that `Copy` gave up.\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
    // Floors, so a green run cannot mean "found nothing to check".
    assert!(
        lifted >= 25,
        "only {lifted} .pto lifted (39 measured 2026-11-09). If the lift started refusing \
         broadly this gate stopped covering the thing it guards."
    );
    assert!(
        stored >= 5,
        "only {stored} stored .rs found; the checked-in half of this gate is not running"
    );
}
