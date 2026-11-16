// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `.pto` input, which this tool used to refuse outright.
//!
//! The refusal said the direct route "needs the pinned ptoas assembler", which
//! was true and was not a reason to refuse: `ptoas` is the reference PTO ->
//! AscendC implementation and this repository already pins a version of it.
//! Reimplementing that translation here would produce a second answer that can
//! disagree with the one the device path uses, which is worse than shelling
//! out.
//!
//! What these tests pin is the behaviour around the assembler, because that is
//! where a shell-out goes wrong: the tool must say which version it used when
//! that is not the pinned one, must give an actionable message when the
//! assembler is absent rather than an errno, and must not report an absent
//! assembler as a failed translation.

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pto-to-ascendc"))
}

fn repo(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root")
        .join(rel)
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("p2a_it_{}_{name}", std::process::id()))
}

fn have_ptoas() -> bool {
    Command::new("sh")
        .arg("-c")
        .arg("command -v ptoas")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[test]
fn an_absent_assembler_is_unavailable_and_says_what_to_install() {
    // Exit 4, not 1: nothing was attempted, so nothing failed to translate.
    // And the message has to be actionable -- a bare `No such file or
    // directory` is what this printed when `$PTOAS` was taken on trust.
    let out = bin()
        .arg(repo("rust/bench/playground16/pto/vec_add.pto"))
        .arg("-o")
        .arg(scratch("absent.cpp"))
        .env("PTOAS", "/nonexistent/ptoas")
        .env("PATH", "/nonexistent-dir")
        .output()
        .expect("ran");
    assert_eq!(
        out.status.code(),
        Some(4),
        "an absent assembler must be exit 4 (unavailable), got {:?}",
        out.status.code()
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("toolchain/versions.env"),
        "the message must point at where the pin lives:\n{err}"
    );
    assert!(
        err.contains("$PTOAS"),
        "the message must mention the override:\n{err}"
    );
    assert!(
        err.contains("not a file"),
        "an explicitly set but missing $PTOAS should be called out, since the \
         user has already said where to look:\n{err}"
    );
}

#[test]
fn a_pto_kernel_assembles_to_ascendc() {
    if !have_ptoas() {
        // NOT a silent pass. A skip that returns Ok is how a gate comes to
        // certify a run that never happened, so this says so on stderr and
        // the absent-assembler test above still runs unconditionally.
        eprintln!(
            "SKIPPED a_pto_kernel_assembles_to_ascendc: no `ptoas` on PATH. \
             This test proves the .pto route end to end and it did NOT run."
        );
        return;
    }
    let dst = scratch("vec_add.cpp");
    let out = bin()
        .arg(repo("rust/bench/playground16/pto/vec_add.pto"))
        .arg("-o")
        .arg(&dst)
        .output()
        .expect("ran");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "the .pto route failed:\n{err}");
    assert!(
        err.contains("via ptoas"),
        "the output must say the assembler produced it, so a reader knows \
         which implementation emitted the C++:\n{err}"
    );
    let cpp = std::fs::read_to_string(&dst).expect("no .cpp written");
    assert!(
        cpp.contains("#include") && cpp.contains("pto"),
        "that does not look like AscendC C++:\n{}",
        &cpp[..cpp.len().min(200)]
    );
    let _ = std::fs::remove_file(dst);
}

#[test]
fn a_version_other_than_the_pin_is_reported() {
    if !have_ptoas() {
        eprintln!("SKIPPED a_version_other_than_the_pin_is_reported: no `ptoas` on PATH.");
        return;
    }
    // Either it matches the pin and says nothing, or it differs and says so.
    // What must never happen is a silent mismatch: the C++ this emits is what
    // reaches the device.
    let out = bin()
        .arg(repo("rust/bench/playground16/pto/vec_add.pto"))
        .arg("-o")
        .arg(scratch("ver.cpp"))
        .output()
        .expect("ran");
    let err = String::from_utf8_lossy(&out.stderr);
    let version = Command::new("sh")
        .arg("-c")
        .arg("ptoas --version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let matches_pin = version.contains("0.70");
    if !matches_pin {
        assert!(
            err.contains("this repository pins"),
            "ptoas reports `{version}`, which is not the pin, and the tool did \
             not say so:\n{err}"
        );
    }
    let _ = std::fs::remove_file(scratch("ver.cpp"));
}

#[test]
fn the_tile_route_still_needs_no_assembler() {
    // The point of the division: `.tile` input is handled in-tree, so the
    // common path does not acquire a new external dependency.
    let dst = scratch("from_tile.cpp");
    let out = bin()
        .arg(repo("rust/bench/playground16/tile/vec_add.tile"))
        .arg("-o")
        .arg(&dst)
        .env("PTOAS", "/nonexistent/ptoas")
        .env("PATH", "/nonexistent-dir")
        .output()
        .expect("ran");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "the .tile route broke:\n{err}");
    assert!(
        !err.contains("via ptoas"),
        "the .tile route must not reach the assembler:\n{err}"
    );
    let _ = std::fs::remove_file(dst);
}
