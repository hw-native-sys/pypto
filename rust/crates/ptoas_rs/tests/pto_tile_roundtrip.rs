// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The gate on `--emit tile`: `.pto` -> `.tile` -> `rspto` -> `.pto`.
//!
//! An emitter that writes plausible MLIR is easy, and plausible MLIR is exactly
//! what a reviewer cannot distinguish from correct MLIR. This closes the loop
//! instead: `rspto` consumes the `.tile` form, so the two ends of the round trip
//! are both PTO and can be compared directly.
//!
//! It earned its place on the first run. The matmul arm emitted its extents as
//! `(M, N, K)` where the convention is `(M, K, N)` -- unguessable from the
//! argument names, well-formed MLIR, every type matching, and for a 32x64 @
//! 64x32 kernel it reads as a 32x64 result. `rspto` refused its own form with
//! "blocked matmul: store shape 32x32 != matmul result 32x64". Nothing else in
//! the crate would have noticed.
//!
//! # Three outcomes, not two
//!
//! Byte identity is the strongest and is NOT required of every artifact,
//! because two weaker outcomes are legitimate and collapsing them into "fail"
//! would make this gate something to switch off:
//!
//! Measured over the 60 distinct artifacts in the repository, deduplicated by
//! content: **10 accepted by `--emit tile`, of which 7 round-trip
//! byte-identically**, 2 semantically, and 1 structurally (the matmul).
//! The other 50 are refused, because this emitter validates through the
//! CHECKED lift, which refuses strictly more than the default one.
//!
//!  * **Semantically identical** -- every `pto.*` line matches and only
//!    COMMENTS differ. The input artifacts carry comments their own producer
//!    left behind (`// unhandled: %2216 = llvm.alloca ...`, `// block index:
//!    see block_idx intrinsic`), which record what that emitter could not
//!    handle. They are not kernel content and this emitter does not invent
//!    them.
//!  * **Structurally different, and correctly so** -- `rspto` lowers a matmul
//!    through the cube/vector split, emitting a `_aic`/`_aiv` function pair
//!    over a `c2v_fifo`. That is the canonical PTO for a matmul on this part,
//!    and it is richer than the plain single-function input. Demanding identity
//!    here would be demanding that `rspto` stop doing the right thing.
//!
//! What is NOT tolerated is `rspto` refusing the `.tile` at all: that means the
//! emitter wrote a form its own consumer cannot read.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root")
}

fn tool(name: &str) -> PathBuf {
    // The test binary lives in `target/<profile>/deps`; the CLIs are two up.
    let mut p = PathBuf::from(env!("CARGO_BIN_EXE_ptoas-rs"));
    p.pop();
    p.push(name);
    p
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pto_tile_rt_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch");
    d
}

fn semantic(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim_end)
        .filter(|l| {
            let t = l.trim_start();
            !t.is_empty() && !t.starts_with("//")
        })
        .map(str::to_string)
        .collect()
}

/// Every `.pto` in the repository, deduplicated by content.
fn corpus() -> Vec<(PathBuf, Vec<u8>)> {
    let root = repo_root();
    let mut files = Vec::new();
    collect(&root, &mut files);
    let mut by: std::collections::BTreeMap<Vec<u8>, PathBuf> = Default::default();
    for f in files {
        if let Ok(b) = std::fs::read(&f) {
            by.entry(b).or_insert(f);
        }
    }
    by.into_iter().map(|(b, p)| (p, b)).collect()
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if let Some(n) = p.file_name().and_then(|s| s.to_str()) {
            if matches!(n, "target" | ".git" | "node_modules") {
                continue;
            }
        }
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|x| x == "pto") {
            out.push(p);
        }
    }
}

struct Tally {
    byte_identical: Vec<String>,
    semantically_identical: Vec<String>,
    structurally_different: Vec<String>,
    rspto_refused: Vec<String>,
    emitter_refused: usize,
}

fn run() -> Tally {
    let dir = scratch("sweep");
    let root = repo_root();
    let (ptoas, rspto) = (tool("ptoas-rs"), tool("rspto"));
    assert!(ptoas.is_file(), "ptoas-rs not built at {}", ptoas.display());
    assert!(
        rspto.is_file(),
        "rspto not built at {} -- this gate needs BOTH tools, since the whole \
         point is that one consumes the other's output",
        rspto.display()
    );

    let mut t = Tally {
        byte_identical: Vec::new(),
        semantically_identical: Vec::new(),
        structurally_different: Vec::new(),
        rspto_refused: Vec::new(),
        emitter_refused: 0,
    };

    for (path, original) in corpus() {
        let name = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let tile = dir.join(format!("{stem}.tile"));
        let back = dir.join(format!("{stem}.back.pto"));

        let emitted = Command::new(&ptoas)
            .arg(&path)
            .arg("--emit")
            .arg("tile")
            .arg("-o")
            .arg(&tile)
            .output()
            .expect("ptoas-rs ran");
        if !emitted.status.success() || !tile.is_file() {
            t.emitter_refused += 1;
            continue;
        }

        let lowered = Command::new(&rspto)
            .arg(&tile)
            .arg("-o")
            .arg(&back)
            .output()
            .expect("rspto ran");
        if !lowered.status.success() || !back.is_file() {
            t.rspto_refused.push(format!(
                "{name}: {}",
                String::from_utf8_lossy(&lowered.stderr).trim()
            ));
            continue;
        }

        let round = std::fs::read(&back).expect("read back");
        if round == original {
            t.byte_identical.push(name);
        } else if semantic(&String::from_utf8_lossy(&round))
            == semantic(&String::from_utf8_lossy(&original))
        {
            t.semantically_identical.push(name);
        } else {
            t.structurally_different.push(name);
        }
    }
    let _ = std::fs::remove_dir_all(dir);
    t
}

#[test]
fn the_tile_form_round_trips_to_the_pto_it_came_from() {
    let t = run();
    let accepted = t.byte_identical.len()
        + t.semantically_identical.len()
        + t.structurally_different.len()
        + t.rspto_refused.len();

    eprintln!(
        "pto -> tile -> pto over {} distinct artifact(s): {} accepted by --emit tile, \
         {} refused",
        accepted + t.emitter_refused,
        accepted,
        t.emitter_refused
    );
    eprintln!("  {:>3}  byte-identical", t.byte_identical.len());
    eprintln!(
        "  {:>3}  semantically identical (comments only)",
        t.semantically_identical.len()
    );
    eprintln!(
        "  {:>3}  structurally different",
        t.structurally_different.len()
    );

    // The hard requirement: the emitter must not write a form its own consumer
    // cannot read.
    assert!(
        t.rspto_refused.is_empty(),
        "{} artifact(s): `rspto` refused the `.tile` this crate emitted, so the \
         emitter wrote a form its own consumer cannot read:\n{}",
        t.rspto_refused.len(),
        t.rspto_refused.join("\n")
    );

    // Byte identity on the majority. A floor rather than a count, so adding a
    // corpus artifact does not fail this, but losing the property does.
    assert!(
        t.byte_identical.len() >= 7,
        "only {} artifact(s) round-trip byte-identically; it was 7 when this gate was \
         written. A drop means the emitter changed what it writes:\n{:?}",
        t.byte_identical.len(),
        t.byte_identical
    );

    // And something must be accepted at all: an emitter that refuses everything
    // would satisfy every assertion above.
    // 10, deduplicated BY CONTENT. A shell sweep over file paths said 11 and
    // was counting two copies of one artifact -- the same denominator mistake
    // that has bitten every corpus count in this crate, which is why the
    // dedup is in the harness and not in the caller.
    assert!(
        accepted >= 10,
        "only {accepted} artifact(s) accepted by --emit tile; it was 10. A gate where \
         nothing is accepted passes every other check in this file"
    );
}

#[test]
fn the_only_structural_difference_is_the_matmul_cube_split() {
    // Named, not counted. A new structural difference is a question -- did the
    // emitter change, or did `rspto`'s lowering? -- and a bare count turns that
    // question into a number nobody reads.
    let t = run();
    let unexpected: Vec<_> = t
        .structurally_different
        .iter()
        .filter(|n| !n.contains("matmul"))
        .collect();
    assert!(
        unexpected.is_empty(),
        "artifact(s) round-trip to structurally different PTO for a reason not yet \
         examined:\n{unexpected:?}\n\nThe one known case is a matmul, where `rspto` \
         lowers through the cube/vector split and emits an `_aic`/`_aiv` pair over a \
         `c2v_fifo` -- richer than the plain input, and correct. Anything else needs \
         reading before it is added here."
    );
}

#[test]
fn a_matmul_round_trips_into_the_cube_vector_pair() {
    // The positive form of the exemption above: assert what the matmul DOES
    // become, so "structurally different" cannot quietly start meaning
    // "mangled".
    let root = repo_root();
    let src = root.join("rust/crates/ptoas_rs/tests/fixtures/probes/matmul_32x64x32.pto");
    if !src.is_file() {
        // Find it wherever it lives, rather than skipping on a moved fixture.
        let found = corpus()
            .into_iter()
            .map(|(p, _)| p)
            .find(|p| p.to_string_lossy().contains("matmul_32x64x32"));
        let Some(found) = found else {
            panic!("matmul_32x64x32.pto is not in the corpus; this test has no subject");
        };
        return check_matmul(&found);
    }
    check_matmul(&src);
}

fn check_matmul(src: &Path) {
    let dir = scratch("matmul");
    let tile = dir.join("mm.tile");
    let back = dir.join("mm.pto");
    assert!(
        Command::new(tool("ptoas-rs"))
            .arg(src)
            .args(["--emit", "tile", "-o"])
            .arg(&tile)
            .status()
            .expect("ran")
            .success(),
        "--emit tile refused the matmul"
    );
    let mlir = std::fs::read_to_string(&tile).expect("tile");
    // The extent order is the thing that was wrong, so assert it directly and
    // not only through the round trip: A is 32x64, B is 64x32, C is 32x32, so
    // (M, K, N) = (32, 64, 32) and the WRONG (M, N, K) would be (32, 32, 64).
    let call = mlir
        .lines()
        .find(|l| l.contains("__tile_matmul_f32"))
        .expect("no matmul call emitted");
    let args: Vec<&str> = call
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(a, _)| a.split(',').map(str::trim).collect())
        .expect("could not read the call's arguments");
    assert_eq!(args.len(), 6, "matmul arity changed: {call}");
    assert_eq!(
        (args[3], args[4], args[5]),
        ("%c32", "%c64", "%c32"),
        "the matmul extents are not (M, K, N) = (32, 64, 32). (M, N, K) would be \
         (32, 32, 64) and is the mistake this asserts against:\n{call}"
    );

    assert!(
        Command::new(tool("rspto"))
            .arg(&tile)
            .arg("-o")
            .arg(&back)
            .status()
            .expect("ran")
            .success(),
        "rspto refused the matmul .tile"
    );
    let pto = std::fs::read_to_string(&back).expect("back");
    for want in ["_aic(", "_aiv", "c2v_fifo", "kernel_kind<cube>"] {
        assert!(
            pto.contains(want),
            "the lowered matmul does not carry `{want}`, so it is not the cube/vector \
             pair this test exists to pin:\n{pto}"
        );
    }
    let _ = std::fs::remove_dir_all(dir);
}
