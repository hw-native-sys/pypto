// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Every intrinsic the kernel-side DSL declares must be reachable from this
//! tree, and the measurement that made this test necessary.
//!
//! `tile_std` arrived here from a repository where it also described an Apple
//! GPU kernel surface. Measured on arrival: it declared **350** `__tile_*`
//! intrinsics, of which **118** were named by any emitter here; of the 182
//! carrying a `DS4 <kernel>` provenance comment only **8** were. So 208 of them
//! were declared, lowered by nothing, and referenced by no wrapper, test or
//! sibling module -- names with nothing behind them, which is the one failure
//! mode `DType::tile_suffix` returning `None` exists to prevent. **142 remain,
//! 122 of them named by an emitter directly**, and the file went from 6,419
//! lines to 3,182.
//!
//! They were removed. That also removed, by construction rather than by comment
//! surgery, 55 of the 64 foreign-target comment mentions and ALL 106 milestone
//! references -- a far better outcome than rewriting prose around declarations
//! nothing could lower.
//!
//! The removal had to run to a FIXPOINT, which is the part worth remembering.
//! A single pass decided the dead set against the original text, so a name
//! referenced only by another dead block's doc comment looked reachable and
//! survived; eight did, and this test is what found them. Three iterations
//! reached the fixpoint.
//!
//! This test is what stops them growing back. It is cheap, it runs on stable,
//! and it fails on the ADDITION of an unreachable declaration, which is when
//! the author is still present to say what they meant.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the crate sits two levels below the workspace root")
        .to_path_buf()
}

fn read(rel: &str) -> String {
    let p = workspace().join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

/// Source text of every crate that could lower an intrinsic.
fn emitter_text() -> String {
    let mut out = String::new();
    for crate_name in ["rspto_emit", "ascendc_emit", "tile_codegen", "ptoas_rs"] {
        let dir = workspace().join("crates").join(crate_name).join("src");
        let mut stack = vec![dir];
        while let Some(d) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
                }
            }
        }
    }
    assert!(
        out.len() > 100_000,
        "only {} bytes of emitter source found; the walk is broken, not the tree",
        out.len()
    );
    out
}

/// `(name, line index)` for every `pub fn __tile_*` in the DSL's tile module.
fn declarations(tile_rs: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for (i, line) in tile_rs.lines().enumerate() {
        let t = line.trim_start();
        for prefix in ["pub fn __tile_", "pub unsafe fn __tile_"] {
            if let Some(rest) = t.strip_prefix(prefix) {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                out.push((format!("__tile_{name}"), i));
            }
        }
    }
    out
}

#[test]
fn every_declared_intrinsic_is_reachable_from_this_tree() {
    let tile_rs = read("crates/tile_std/src/tile.rs");
    let decls = declarations(&tile_rs);
    assert!(
        decls.len() > 100,
        "only {} intrinsics parsed out of the DSL; the scan is broken",
        decls.len()
    );

    let emitters = emitter_text();

    // Everything in the DSL except the file being scanned, plus the macro
    // crate: a declaration a safe wrapper calls is reachable even when no
    // emitter names it directly.
    let mut dsl_rest = String::new();
    for f in [
        "crates/tile_std/src/lib.rs",
        "crates/tile_std/src/buf.rs",
        "crates/tile_std/src/core.rs",
        "crates/tile_std/src/kernel_ops.rs",
        "crates/tile_std/src/pipeline.rs",
        "crates/tile_std/src/emitted_kernels.rs",
        "crates/tile_std_macros/src/lib.rs",
        "examples/tile_softmax/kernels/src/lib.rs",
    ] {
        dsl_rest.push_str(&read(f));
    }

    // A declaration's own line does not count as a use of itself, so count
    // occurrences in `tile.rs` and subtract the declaration sites.
    let mut declared_times = std::collections::BTreeMap::new();
    for (name, _) in &decls {
        *declared_times.entry(name.clone()).or_insert(0usize) += 1;
    }

    let mut unreachable: Vec<String> = Vec::new();
    for (name, line) in &decls {
        if emitters.contains(name) || dsl_rest.contains(name) {
            continue;
        }
        let uses_in_tile_rs = tile_rs.matches(name.as_str()).count();
        let own_sites = declared_times[name];
        if uses_in_tile_rs > own_sites {
            continue;
        }
        unreachable.push(format!("  {}:{}  {name}", "crates/tile_std/src/tile.rs", line + 1));
    }

    assert!(
        unreachable.is_empty(),
        "{} intrinsic declaration(s) are lowered by no emitter in this tree and used by \
         nothing in the DSL, so a kernel calling them would compile and emit nothing:\n{}\n\
         Either wire the lowering, or remove the declaration. 208 such names were removed \
         when this crate was carried here; the point of this test is that the 209th has to \
         be argued for.",
        unreachable.len(),
        unreachable.join("\n")
    );

    eprintln!(
        "{} intrinsic declarations, all reachable ({} named by an emitter directly)",
        decls.len(),
        decls
            .iter()
            .filter(|(n, _)| emitters.contains(n.as_str()))
            .map(|(n, _)| n.clone())
            .collect::<BTreeSet<_>>()
            .len()
    );
}

#[test]
fn the_scan_can_actually_fail() {
    // A reachability check that cannot report an unreachable name would pass
    // on any tree. Feed it a declaration no emitter can name.
    let fake = "    pub fn __tile_a_name_no_emitter_will_ever_lower_f32(x: u32) -> u32;\n";
    let decls = declarations(fake);
    assert_eq!(decls.len(), 1, "the declaration parser missed a `pub fn`");
    assert_eq!(decls[0].0, "__tile_a_name_no_emitter_will_ever_lower_f32");
    assert!(
        !emitter_text().contains(decls[0].0.as_str()),
        "that name is supposed to be absent from every emitter"
    );
}

#[test]
fn the_dsl_carries_no_declaration_whose_only_purpose_was_provenance() {
    // The removed 200 were identifiable by a `DS4 <kernel>` doc comment with
    // no lowering behind it. 8 such intrinsics ARE lowered and stayed, so the
    // comment itself is not the problem and this does not ban it -- it bounds
    // it, because 182 of them is a map of another vendor's kernel surface
    // rather than documentation of this one.
    let tile_rs = read("crates/tile_std/src/tile.rs");
    let ds4 = tile_rs
        .lines()
        .filter(|l| l.trim_start().starts_with("///") && l.contains("DS4 "))
        .count();
    assert!(
        ds4 <= 20,
        "{ds4} `DS4 <kernel>` provenance comments in the DSL. It was 182 on arrival and 9 \
         after the unreachable declarations were removed; a number this high means the \
         vendor-kernel mirror is growing back",
    );
}
