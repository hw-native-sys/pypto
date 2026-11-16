// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The verification arm, now that it can actually run, and the three ways it
//! was wrong when it could not.
//!
//! All three had the same shape: something about the HARNESS was reported as
//! something about the kernel.
//!
//!  1. The verifier shelled out to the ambient `cargo`, which is this
//!     workspace's stable toolchain. The DSL is `#![no_core]`, so stable
//!     rejects it at `#![feature]` with E0554 -- and the tool printed
//!     "rustc: rejected the lifted kernel". Every kernel in the corpus would
//!     have read as unsafe, on the strength of the wrong toolchain.
//!  2. `cargo miri run` needs a binary. The lifted kernel is a library with
//!     `extern "C"` entry points, so Miri exited non-zero with "a bin target
//!     must be available" and that, too, printed as a rejection. Separating it
//!     out made the tier honest; `tile_std_ref` then made it work, and this
//!     file's Miri test now asserts the real outcome.
//!  3. `--check` is `rspto`'s flag. Passed to this tool it fell through to the
//!     positional arm and became the input path, so the error was "could not
//!     read --check: No such file or directory".
//!
//! An availability failure reported as a rejection is the worst of the three
//! kinds of wrong answer here, because it is indistinguishable from a real
//! finding to anyone reading the output.

use std::process::Command;

mod shared;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_ptoas-rs"))
}

fn corpus(rel: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("repository root")
        .join(rel)
}

fn scratch(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("ptoas_verify_it_{}_{name}", std::process::id()))
}

#[test]
fn an_unknown_flag_is_a_usage_error_and_not_an_input_path() {
    let out = bin().arg("--check").output().expect("ran");
    assert_eq!(
        out.status.code(),
        Some(2),
        "an unknown flag must be exit 2 (usage), got {:?}",
        out.status.code()
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("unknown flag"),
        "the message must name the flag as a flag, not a missing file:\n{err}"
    );
    assert!(
        !err.contains("could not read"),
        "the flag was still treated as a path:\n{err}"
    );
}

#[test]
fn rustc_verification_accepts_a_kernel_the_lift_produced() {
    if shared::skip_unless(
        shared::have_kernel_toolchain(),
        "rustc_verification_accepts_a_kernel_the_lift_produced",
        "the kernel-side toolchain",
    ) {
        return;
    }
    // The claim deliverable (2) rests on: a lifted kernel is accepted by a
    // real compiler, not merely emitted as text. Asserted on an artifact the
    // lift handles end to end.
    let src = corpus("rust/bench/playground16/pto/vec_add.pto");
    assert!(src.exists(), "{} is missing", src.display());
    let dst = scratch("accept.rs");
    let out = bin()
        .arg(&src)
        .arg("--verify")
        .arg("rustc")
        .arg("-o")
        .arg(&dst)
        .output()
        .expect("ran");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("rustc: accepted the lifted kernel"),
        "rustc did not accept it. If this says `unavailable`, the pinned \
         nightly is missing -- install it with `rustup toolchain install \
         nightly-2025-08-04`. If it says `rejected`, read the diagnostics: the \
         lift emitted Rust that does not compile.\n{err}"
    );
    assert_eq!(out.status.code(), Some(0), "a proved kernel must exit 0:\n{err}");
    let _ = std::fs::remove_file(dst);
}

#[test]
fn miri_now_runs_the_kernel_rather_than_reporting_itself_absent() {
    // This test used to assert the OPPOSITE: that Miri reported itself
    // unavailable, because `cargo miri run` needs a binary and a lifted kernel
    // is a library of `extern "C"` functions. It said, in its own failure
    // message, "If Miri has gained a driver that can run one, say so here and
    // assert the real outcome instead" -- and that is what happened, so this
    // is that edit rather than a loosened assertion.
    //
    // The driver is `tile_std_ref`, a std crate mirroring the tile API's
    // signatures with ordinary Rust bodies, plus a `main` whose buffers are
    // sized from the PTO. `tests/miri_runs_the_lifted_kernel.rs` holds the
    // tier's own tests, including the one proving it can still REJECT.
    if shared::skip_unless(
        shared::have_miri(),
        "miri_now_runs_the_kernel_rather_than_reporting_itself_absent",
        "miri",
    ) {
        return;
    }
    let src = corpus("rust/bench/playground16/pto/vec_add.pto");
    let dst = scratch("miri.rs");
    let out = bin()
        .arg(&src)
        .arg("--verify")
        .arg("miri")
        .arg("-o")
        .arg(&dst)
        .output()
        .expect("ran");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !err.contains("miri: rejected"),
        "Miri rejected a kernel that should be sound. Read the diagnostics before          relaxing this: the tier now RUNS the kernel, so a rejection is a real          finding.\n{err}"
    );
    assert!(
        err.contains("miri: accepted"),
        "expected Miri to accept the kernel. `unavailable` here means the pinned          nightly or its miri component is missing -- install with \
         `rustup +nightly-2025-08-04 component add miri`.\n{err}"
    );
    assert_eq!(
        out.status.code(),
        Some(0),
        "a proved kernel must exit 0:\n{err}"
    );
    let _ = std::fs::remove_file(dst);
}

#[test]
fn verify_and_emit_tile_cannot_be_combined() {
    // `--verify` checks the lifted Rust; `--emit tile` writes MLIR. Verifying
    // one file while writing another would report a proof about a file the
    // user never received.
    let src = corpus("rust/bench/playground16/pto/vec_add.pto");
    let out = bin()
        .arg(&src)
        .args(["--verify", "rustc", "--emit", "tile"])
        .output()
        .expect("ran");
    assert_eq!(
        out.status.code(),
        Some(2),
        "the combination must be a usage error, got {:?}",
        out.status.code()
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--emit tile"),
        "the message does not name the conflicting flag:\n{err}"
    );
}

#[test]
fn the_verifier_names_the_toolchain_it_needs_in_one_place() {
    // The constant exists so the compile gate and the verifier cannot drift
    // onto different nightlies, which would make one of them lie.
    assert_eq!(ptoas_rs::verify_rustc::KERNEL_TOOLCHAIN, "nightly-2025-08-04");
}
