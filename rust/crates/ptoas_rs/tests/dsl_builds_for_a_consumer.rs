// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The DSL's build requirement, asserted from the OUTSIDE — as a crate that
//! depends on it, which is the only position that can see the problem.
//!
//! `tile_std` ICEs rustc when built with debuginfo: the panic is at
//! `rustc_middle/src/ty/layout.rs` (`assertion failed: i < this.fields.count()`)
//! inside `build_pointer_or_reference_di_node`, i.e. debuginfo generation over a
//! `no_core` pointer type. It is the toolchain, not the crate — measured on both
//! a pruned and an unpruned copy.
//!
//! The first fix for this was `[profile.dev] debug = 0` in the DSL's own
//! manifest, and it DOES NOT WORK: cargo ignores `[profile.*]` in a dependency's
//! manifest, so a kernel crate depending on tile_std gets the root crate's
//! debuginfo and ICEs anyway. The manifest comment said the profile "makes the
//! plain cargo build work too", which was true only of building the DSL
//! standalone — the case nobody actually has.
//!
//! These tests build a real consumer crate both ways, so the claim is checked
//! where it is used rather than where it is written.

use std::path::{Path, PathBuf};
use std::process::Command;

const TOOLCHAIN: &str = "nightly-2025-08-04";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn have_toolchain() -> bool {
    Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains(TOOLCHAIN))
        .unwrap_or(false)
}

/// Stage a minimal crate that depends on the DSL. `profile_debug_off` writes the
/// `[profile.dev]` the DSL's manifest cannot supply from a dependency position.
fn stage(name: &str, profile_debug_off: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tile_std_consumer_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).expect("staging");
    let dsl = workspace().join("crates/tile_std");
    assert!(dsl.join("src/tile.rs").exists(), "the DSL is not at {}", dsl.display());
    let mut manifest = format!(
        "[package]\nname = \"consumer_probe\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
         [dependencies]\ntile_std = {{ path = {:?} }}\n[workspace]\n",
        dsl.display().to_string()
    );
    if profile_debug_off {
        manifest.push_str("\n[profile.dev]\ndebug = 0\n");
    }
    std::fs::write(dir.join("Cargo.toml"), manifest).expect("manifest");
    std::fs::write(
        dir.join("src/lib.rs"),
        "#![feature(no_core)]\n#![no_std]\n#![no_core]\nuse tile_std::tile::*;\n",
    )
    .expect("lib");
    dir
}

fn build(dir: &Path, rustflags: Option<&str>) -> (bool, String) {
    let mut cmd = Command::new("cargo");
    cmd.arg(format!("+{TOOLCHAIN}"))
        .arg("build")
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", dir.join("t"))
        // `RUSTC_ICE=0` stops rustc writing an ICE dump, and this test exists
        // to make it panic -- so without this it litters. rustc writes the dump
        // into the CWD of the rustc invocation, which is the crate directory
        // being compiled, i.e. the DSL's own source tree and not this scratch
        // dir. 25 of them reached the branch before anyone noticed, swept in by
        // `git add -A`. `.gitignore` catches a stray one now; this stops them
        // being created.
        .env("RUSTC_ICE", "0")
        .env_remove("RUSTFLAGS");
    if let Some(f) = rustflags {
        cmd.env("RUSTFLAGS", f);
    }
    let out = cmd.output().expect("cargo ran");
    let text = String::from_utf8_lossy(&out.stderr).to_string();
    (out.status.success(), text)
}

#[test]
fn a_consumer_that_turns_debuginfo_off_builds_the_dsl() {
    if !have_toolchain() {
        eprintln!(
            "SKIPPED a_consumer_that_turns_debuginfo_off_builds_the_dsl: {TOOLCHAIN} is not \
             installed. This test did NOT run."
        );
        return;
    }
    let dir = stage("profile", true);
    let (ok, text) = build(&dir, None);
    assert!(
        ok,
        "a consumer with `[profile.dev] debug = 0` could not build the DSL:\n{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rustflags_is_the_other_way_and_also_works() {
    if !have_toolchain() {
        eprintln!("SKIPPED rustflags_is_the_other_way_and_also_works: {TOOLCHAIN} is not installed.");
        return;
    }
    let dir = stage("rustflags", false);
    let (ok, text) = build(&dir, Some("-C debuginfo=0"));
    assert!(ok, "`RUSTFLAGS=-C debuginfo=0` did not work:\n{text}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_requirement_is_real_and_the_dependency_profile_does_not_cover_it() {
    // The negative half. Without it, the two tests above would pass on a
    // toolchain where the ICE had been fixed and would stop testing anything,
    // while the manifest kept telling people to do something unnecessary.
    //
    // If this starts FAILING -- i.e. a consumer builds with debuginfo on -- the
    // rustc bug is fixed. Delete the warning from `crates/tile_std/Cargo.toml`
    // and these tests with it; do not just invert the assertion.
    if !have_toolchain() {
        eprintln!(
            "SKIPPED the_requirement_is_real_and_the_dependency_profile_does_not_cover_it: \
             {TOOLCHAIN} is not installed."
        );
        return;
    }
    let dir = stage("bare", false);
    let (ok, text) = build(&dir, None);
    assert!(
        !ok,
        "a consumer built the DSL with debuginfo ON. Either rustc fixed the ICE, or cargo \
         started honouring a dependency's [profile.dev] -- check which, then delete the \
         warning in crates/tile_std/Cargo.toml rather than inverting this test."
    );
    assert!(
        text.contains("unexpectedly panicked") || text.contains("fields.count()"),
        "the consumer failed for some OTHER reason than the debuginfo ICE, so this test is \
         no longer measuring what it claims:\n{text}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
