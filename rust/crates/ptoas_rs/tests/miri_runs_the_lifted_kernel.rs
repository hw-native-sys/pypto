// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The Miri tier, which now runs the kernel instead of reporting itself absent.
//!
//! It reported `Unavailable` for a structural reason: `cargo miri run` needs a
//! binary, a lifted kernel is a library of `extern "C"` functions, and the
//! kernel-side DSL is `#![no_core]` -- so it cannot host a `#[test]` and cannot
//! share a crate graph with `std`, because it declares its own lang items.
//! `tile_std_ref` mirrors the same signatures with ordinary Rust bodies in an
//! ordinary `std` crate, so the kernel repointed at it is executable.
//!
//! # The property that makes this a measurement
//!
//! The driver's buffers are sized from `kernel_signatures`, i.e. **from the
//! PTO**, never from the emitted Rust. Extents read back out of the lift's own
//! output would agree with it by construction: a kernel whose extents were
//! wrong would get buffers wrong in the same way and pass clean. Taking them
//! from the source keeps the two independent.
//!
//! `a_kernel_that_reads_past_its_buffer_is_rejected` is the test that proves it,
//! by mutating the LIFTED source while leaving the PTO alone -- exactly the
//! disagreement the tier exists to catch, and one that no amount of
//! self-consistency would find.

use std::path::PathBuf;
use ptoas_rs::verify_rustc::{Outcome, Tier};

const TOOLCHAIN: &str = "nightly-2025-08-04";

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn corpus(rel: &str) -> PathBuf {
    manifest_dir().join("../../..").canonicalize().unwrap().join(rel)
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("miri_tier_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// The toolchain the tier will actually use.
///
/// Read from the same place `verify_miri` reads it. The probe used to hardcode
/// `TOOLCHAIN` while the runner honoured `PTOAS_RS_KERNEL_TOOLCHAIN`, so with
/// the override pointed at an absent toolchain the probe said "miri is here"
/// about a DIFFERENT toolchain and the test then failed inside the runner. A
/// probe that checks something other than what the code under test will use
/// can only mislead -- in whichever direction happens to hurt.
fn toolchain() -> String {
    std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN").unwrap_or_else(|_| TOOLCHAIN.to_string())
}

fn have_miri() -> bool {
    let tc = toolchain();
    std::process::Command::new("cargo")
        .args(["miri", "--version"])
        .env("RUSTUP_TOOLCHAIN", &tc)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn skip(name: &str) -> bool {
    if have_miri() {
        return false;
    }
    // Said out loud, not silently. A skip that returns Ok is how a gate comes
    // to certify a run that never happened.
    let tc = toolchain();
    eprintln!(
        "SKIPPED {name}: miri is not available on `{tc}`. This test did NOT run. \
         Install it with `rustup +{tc} component add miri`."
    );
    true
}

fn lift(pto: &str) -> String {
    ptoas_rs::convert_pto_to_rust(pto).expect("the lift refused an artifact it should accept")
}

#[test]
fn miri_proves_a_lifted_kernel() {
    if skip("miri_proves_a_lifted_kernel") {
        return;
    }
    let pto = std::fs::read_to_string(corpus("rust/bench/playground16/pto/vec_add.pto")).unwrap();
    let rust = lift(&pto);
    let out = ptoas_rs::verify_rustc::verify(
        &rust,
        &pto,
        Tier::Miri,
        &scratch("good"),
        &manifest_dir(),
    );
    match out {
        Outcome::Proved { .. } => {}
        Outcome::Unavailable { why, .. } => panic!(
            "Miri could not run, which is the state this tier was in before the `std` \
             reference existed: {why}"
        ),
        Outcome::Rejected { diagnostics, .. } => {
            panic!("Miri rejected a kernel that should be sound:\n{diagnostics}")
        }
    }
}

#[test]
fn a_kernel_that_reads_past_its_buffer_is_rejected() {
    if skip("a_kernel_that_reads_past_its_buffer_is_rejected") {
        return;
    }
    let pto = std::fs::read_to_string(corpus("rust/bench/playground16/pto/vec_add.pto")).unwrap();
    let rust = lift(&pto);
    assert!(
        rust.contains("::<1, 256>"),
        "the lifted vec_add no longer carries 1x256 extents, so this mutation does not \
         apply. Re-derive it from the current output:\n{rust}"
    );
    // Widen the kernel's extents and leave the PTO alone. The driver still
    // allocates 256 elements, because it reads the PTO -- so the kernel reads
    // 512 from a 256-element buffer. This is the defect class the lift can
    // actually produce (it chooses extents), and it is invisible to any check
    // that takes the extents from the same place the kernel did.
    let mutated = rust.replace("::<1, 256>", "::<1, 512>");
    assert_ne!(mutated, rust, "the mutation did not apply");

    let out = ptoas_rs::verify_rustc::verify(
        &mutated,
        &pto,
        Tier::Miri,
        &scratch("bad"),
        &manifest_dir(),
    );
    match out {
        Outcome::Rejected { diagnostics, .. } => {
            assert!(
                diagnostics.contains("Undefined Behavior")
                    || diagnostics.contains("beyond the end of the allocation"),
                "Miri rejected it, but not for the out-of-bounds read this seeded. If the \
                 kernel now fails to COMPILE, the mutation is testing rustc and not \
                 Miri:\n{diagnostics}"
            );
        }
        Outcome::Proved { .. } => panic!(
            "Miri PROVED a kernel that reads 512 f32 from a 256-element buffer. The tier \
             cannot fire, so every `miri: accepted` it has ever printed means nothing. The \
             likely cause is the driver sizing its buffers from the emitted Rust instead of \
             from the PTO."
        ),
        Outcome::Unavailable { why, .. } => panic!("Miri could not run: {why}"),
    }
}

#[test]
fn the_driver_sizes_buffers_from_the_pto_and_not_from_the_rust() {
    // The same property as a direct claim about the signature API, so that it
    // is pinned even when Miri is unavailable: this test needs no toolchain.
    let pto = std::fs::read_to_string(corpus("rust/bench/playground16/pto/vec_add.pto")).unwrap();
    let sigs = ptoas_rs::emitter_tile::kernel_signatures(&pto).expect("signatures");
    assert_eq!(sigs.len(), 1);
    let s = &sigs[0];
    assert_eq!(s.name, "vec_add");
    assert_eq!(s.args.len(), 3, "vec_add takes three pointers");
    for a in &s.args {
        assert_eq!((a.rows, a.cols), (1, 256), "the PTO declares 1x256 views");
        assert_eq!(a.len(), 256);
    }
    // Two readable inputs, one writable output -- and the ORDER matters,
    // because the driver passes `as_mut_ptr` for exactly the writable ones.
    assert_eq!(
        s.args.iter().map(|a| a.writable).collect::<Vec<_>>(),
        vec![false, false, true],
        "the read/write roles or their order changed; a driver built from this would \
         hand a read-only buffer to a store"
    );
}

#[test]
fn a_matmul_signature_is_m_by_k_then_k_by_n_then_m_by_n() {
    // The shape relation the whole API rests on, read from the PTO. The matmul
    // extent order was emitted wrongly elsewhere in this crate, so it is worth
    // pinning from more than one direction.
    let path = corpus("rust/crates/ptoas_rs/tests/fixtures/matmul_32x64x32.pto");
    if !path.is_file() {
        panic!("{} is missing; this test has no subject", path.display());
    }
    let pto = std::fs::read_to_string(&path).unwrap();
    let sigs = ptoas_rs::emitter_tile::kernel_signatures(&pto).expect("signatures");
    let s = &sigs[0];
    let shapes: Vec<_> = s.args.iter().map(|a| (a.rows, a.cols, a.writable)).collect();
    assert_eq!(
        shapes,
        vec![(32, 64, false), (64, 32, false), (32, 32, true)],
        "a 32x64 @ 64x32 matmul must declare a 32x32 output"
    );
}

#[test]
fn the_reference_crate_is_reachable_from_here() {
    // The locator is the one piece that fails quietly: a missing reference
    // makes the tier report Unavailable, which reads like "not implemented".
    let found = ptoas_rs::verify_rustc::locate_tile_std_ref(&manifest_dir());
    let p = found.expect(
        "tile_std_ref was not found next to this crate, so the Miri tier will report \
         itself unavailable and the reason will look like the old structural one",
    );
    assert!(p.join("src/lib.rs").is_file(), "{} has no src/lib.rs", p.display());
    assert!(p.join("Cargo.toml").is_file(), "{} has no manifest", p.display());
}
