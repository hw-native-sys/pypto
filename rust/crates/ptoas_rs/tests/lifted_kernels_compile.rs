// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Every artifact the lift accepts must produce Rust that rustc accepts.
//!
//! Until 2026-11-01 nothing in this crate had ever compiled the lift's output.
//! The consequences were not subtle:
//!
//! | defect | effect |
//! |---|---|
//! | prologue said only `#![no_std]` | `E0152: duplicate lang item in crate tile_std: sized` -- **all 11** artifacts |
//! | a PTO tile read by two ops became two uses of one binding | `E0382: use of moved value` -- `Tile` is not `Copy` |
//! | `ptr.add(n)` for a folded view offset | `E0599: no method named add found for raw pointer` -- `add` is a `core` method and the file is `#![no_core]` |
//!
//! Every one of those is invisible to a gate that reads the emitted TEXT. The
//! first is a property of the prologue against `tile_std`'s lang items, the
//! second of Rust's move checker, the third of which inherent impls exist in a
//! `no_core` world. `emitted_names_resolve.rs` checks that every name the lift
//! writes EXISTS, which is as far as grepping output can go; this is the gate
//! that asks whether the file is a program.
//!
//! Cost: it shells out to `cargo check` once per artifact, into a scratch
//! crate under this crate's own `target/`. That is why it is one test and not
//! one per file -- `tile_std` is compiled once and reused.
//!
//! `cargo check` and not `cargo build` deliberately: `tile_std` ICEs this
//! toolchain (`nightly-2025-08-04`) during codegen, which is a separate,
//! pre-existing problem (there is an ICE dump checked in beside its source)
//! and not one the lift can do anything about. Type-checking is also the whole
//! of what this gate is for: const-generic shape agreement, move semantics,
//! and name resolution are all settled before codegen.

use std::path::{Path, PathBuf};
use std::process::Command;

mod shared;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A scratch crate that depends on `tile_std` by path, built once and then
/// rewritten per artifact.
///
/// Lives under `target/` so it inherits the repo's `rust-toolchain.toml` --
/// `tile_std` needs `#![feature(...)]` and so fails on stable with
/// `E0554`, which would make this gate fail for a reason that has nothing to
/// do with the lift.
///
/// One directory PER TEST, named by the caller. The two tests in this file ran
/// in parallel over one shared directory on the first attempt and each read
/// the other's `src/lib.rs`: the negative premise reported that a double move
/// compiles, and the corpus sweep blamed a matmul artifact for an error in a
/// hand-written fixture. Both lies pointed AWAY from the truth.
struct Scratch {
    dir: PathBuf,
}

/// The toolchain the kernel-side DSL needs.
///
/// Not the workspace's toolchain, and not a stabilisation wait: `tile_std` is
/// `#![no_core]` and declares its own lang items, so it needs nightly
/// permanently. This constant is the one place the version is written down for
/// the compile gates; `crates/tile_std/Cargo.toml` carries the same name in
/// prose for a human building it by hand.
const KERNEL_TOOLCHAIN: &str = "nightly-2025-08-04";

/// Is the toolchain the DSL needs actually installed?
///
/// Separated from `check` because "the toolchain is absent" and "the kernel
/// does not compile" are different answers and were being reported as one.
/// Measured: with no usable nightly, both gates in this file FAILED rather than
/// skipping, so the Rust CI job on a GitHub-hosted runner -- which has neither
/// the pinned nightly nor Miri nor ptoas -- would have gone red on a tree where
/// nothing is wrong. Every other gate in this crate already had a loud-skip
/// path; these two were written on a machine that happens to have everything.
fn toolchain_available() -> bool {
    let toolchain = std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN")
        .unwrap_or_else(|_| KERNEL_TOOLCHAIN.to_string());
    Command::new("rustup")
        .args(["run", &toolchain, "rustc", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Print why a gate did not run, and return true so the caller returns early.
///
/// NOT silent. A skip that returns `Ok` is how a suite comes to certify a run
/// that never happened, so the name of the gate and the reason both go to
/// stderr where a CI log keeps them.
fn skipped(name: &str) -> bool {
    let toolchain = std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN")
        .unwrap_or_else(|_| KERNEL_TOOLCHAIN.to_string());
    eprintln!(
        "SKIPPED {name}: toolchain `{toolchain}` is not installed, so the lifted kernels \
         were NOT compiled. This gate proves the lift's central claim; install the \
         toolchain with `rustup toolchain install {toolchain}` to run it."
    );
    true
}

impl Scratch {
    fn new(who: &str) -> Self {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/lifted_compile_gate")
            .join(who);
        std::fs::create_dir_all(dir.join("src")).expect("cannot create the scratch crate");
        let tile_std = repo_root().join("crates/tile_std");
        assert!(
            tile_std.join("src/tile.rs").exists(),
            "tile_std is not at {}; this gate has nothing to compile against",
            tile_std.display()
        );
        std::fs::write(
            dir.join("Cargo.toml"),
            format!(
                "[workspace]\n\
                 [package]\nname = \"lifted_compile_gate\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
                 [lib]\npath = \"src/lib.rs\"\ntest = false\ndoctest = false\n\
                 [dependencies]\ntile_std = {{ path = \"{}\" }}\n",
                tile_std.display()
            ),
        )
        .expect("cannot write the scratch manifest");
        Self { dir }
    }

    /// `Ok(())` if rustc accepts `src`, else its diagnostics.
    fn check(&self, src: &str) -> Result<(), String> {
        std::fs::write(self.dir.join("src/lib.rs"), src).expect("cannot write the scratch lib");
        // NOT `std::env::var("CARGO")`. That is the cargo running this test,
        // which is the workspace's STABLE toolchain -- and `tile_std` is a
        // `#![no_core]` crate, so stable rejects it at `#![feature]` with
        // E0554 before reading a line of the kernel. The gate then failed for
        // a reason that had nothing to do with the lifted code, which is why
        // it spent its life ignored.
        //
        // Plain `cargo` from PATH is the rustup proxy, which honours
        // `RUSTUP_TOOLCHAIN`; the real cargo binary that `CARGO` points at
        // does not. Override the toolchain with `PTOAS_RS_KERNEL_TOOLCHAIN`.
        let toolchain = std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN")
            .unwrap_or_else(|_| KERNEL_TOOLCHAIN.to_string());
        let out = Command::new("cargo")
            .arg("check")
            .arg("--message-format=short")
            .current_dir(&self.dir)
            // Its own target dir: a nested cargo sharing the parent's would
            // block on the package-cache lock the test runner already holds.
            .env("CARGO_TARGET_DIR", self.dir.join("t"))
            .env("RUSTUP_TOOLCHAIN", &toolchain)
            .env_remove("RUSTFLAGS")
            .output()
            .map_err(|e| {
                format!(
                    "could not run `cargo check` under toolchain `{toolchain}`: {e}\n\
                     Install it with `rustup toolchain install {toolchain}`."
                )
            })?;
        let text = String::from_utf8_lossy(&out.stderr);
        if text.contains("is not installed") || text.contains("toolchain") && text.contains("not installed")
        {
            return Err(format!(
                "toolchain `{toolchain}` is not installed, so this gate compiled nothing. \
                 It is REQUIRED, not optional: the kernel-side DSL only builds on it. \
                 Install it with `rustup toolchain install {toolchain}`, or point \
                 `PTOAS_RS_KERNEL_TOOLCHAIN` at another nightly.\n{text}"
            ));
        }
        if out.status.success() {
            return Ok(());
        }
        let err = String::from_utf8_lossy(&out.stderr);
        // Keep only the lines about the lifted file. A warning from inside
        // `tile_std` (it has one, for `generic_const_exprs`) is not this
        // gate's business, and neither is cargo's own progress output.
        let lines: Vec<&str> = err
            .lines()
            .filter(|l| l.contains("src/lib.rs") || l.starts_with("error"))
            .collect();
        Err(if lines.is_empty() {
            err.to_string()
        } else {
            lines.join("\n")
        })
    }
}

/// The gate's own premise: the scratch crate works at all.
///
/// Without this, a broken manifest or a missing toolchain would make the main
/// test report "0 artifacts failed" over 0 artifacts checked, which reads
/// exactly like success.
/// DEFERRED, NOT RETIRED. This is the rustc-verification arm: it compiles the
/// kernel the lift emitted, which is where const-generic shape agreement and
/// tile-handle linearity actually get proved. It needs `tile_std`, the
/// `#![no_core]` kernel-side crate that arrives with the Rust input mode --
/// measured on stable, it fails with E0554 inside that crate rather than
/// anywhere in the lift. Restore it the moment that crate lands: this is the
/// test that makes the lift's central claim checkable rather than asserted.
#[test]
fn the_scratch_crate_accepts_a_kernel_and_rejects_a_broken_one() {
    if !toolchain_available() && skipped("the_scratch_crate_accepts_a_kernel_and_rejects_a_broken_one") {
        return;
    }
    let s = Scratch::new("premise");
    let good = "#![feature(no_core)]\n#![no_std]\n#![no_core]\n\
                use tile_std::tile::*;\n\
                #[no_mangle]\npub unsafe extern \"C\" fn k(a: *mut f32, b: *mut f32) {\n\
                \x20   let t = tile_load_f32::<1, 32>(a as *const f32);\n\
                \x20   let u = tile_exp_f32::<1, 32>(t);\n\
                \x20   tile_store_f32::<1, 32>(b as *mut f32, u);\n}\n";
    s.check(good).expect("the scratch crate cannot compile a hand-written tile kernel");

    // ONE TILE, TWO CONSUMERS. This asserted the OPPOSITE until 2026-11-09: that
    // it must NOT compile, with E0382, because `Tile` was deliberately not
    // `Copy`. The contract changed under an explicit operator authorisation to
    // extend `tile_std`, and the reason is that the old guarantee was enforcing
    // linearity on a handle whose compute uses are READS -- in PTO every compute
    // op reads its inputs and writes a NEW tile, so nothing is consumed, and ten
    // corpus kernels could not be lifted because a computed tile read by several
    // ops would have been moved twice. Five of them now lift and pass rustc.
    //
    // So this now pins the NEW contract, and names where each half of the old
    // guarantee went rather than letting it lapse silently:
    //
    //   * double-STORE -- "each tile must be consumed by exactly one store
    //     operation, preventing double-DMA bugs", the reason `Copy` was withheld
    //     -- moved to `no_lifted_kernel_stores_a_tile_twice.rs`, which checks
    //     every stored .rs AND a fresh lift of every .pto.
    //   * a tile whose CONTENTS CHANGE between its reads, i.e. an in-place write,
    //     is refused by the lift itself in `emitter.rs`. `Copy` makes multiple
    //     reads expressible but not reads of two different values through one
    //     name.
    let shared = good.replace(
        "    tile_store_f32::<1, 32>(b as *mut f32, u);",
        "    let v = tile_exp_f32::<1, 32>(t);\n    tile_store_f32::<1, 32>(b as *mut f32, u);\n         \x20   let _ = v;",
    );
    s.check(&shared).expect(
        "one tile read by two ops must now COMPILE: `Tile` is `Copy`. If this fails with E0382 \
         the derive was reverted, and the ten kernels it unblocked are refused again -- in which \
         case restore the old assertion here rather than leaving both states half-applied",
    );

    // And the prologue this emitter shipped for its whole life.
    let no_core_missing = good.replace("#![feature(no_core)]\n#![no_std]\n#![no_core]\n", "#![no_std]\n");
    let err = s
        .check(&no_core_missing)
        .expect_err("`#![no_std]` alone must not compile against a `#![no_core]` tile_std");
    assert!(
        err.contains("E0152"),
        "expected a duplicate-lang-item error, got: {err}"
    );

    // And the SHAPE half of the headline claim, exhibited against code that is
    // actually wrong rather than against correct output.
    //
    // `sv.pto` is refused for `pto.subview`. Relaxing that refusal to the
    // obvious thing -- alias the sub-view to its parent tile -- makes the file
    // LIFT, and this is what it emits, with the row offset silently gone:
    // a 4x32 load stored through a 2x32 store. It is hand-written here because
    // the emitter will not produce it, which is the point; what is under test
    // is the second line of defence, that rustc rejects the shape even when a
    // refusal does not. Measured 2026-11-01: `E0308: expected 2, found 4`.
    //
    // Note what this does NOT catch: a lift that folded the offset into the
    // address and got the shape right would compile, and on 910B2 a non-zero
    // `LocalTensor` sub-view mis-computes vector ops -- a claim that is
    // DISPUTED (a device run on real 910B2 found the vectorised offset form
    // bit-identical for one shape) and is not what the refusal rests on. The
    // type system covers
    // the shape; only the refusal covers the silicon.
    let dropped_subview = "#![feature(no_core)]\n#![no_std]\n#![no_core]\n\
                use tile_std::tile::*;\n\
                #[no_mangle]\npub unsafe extern \"C\" fn svprobe(a0: *mut f32, a1: *mut f32) {\n\
                \x20   let f = tile_load_f32::<4, 32>(a0 as *const f32);\n\
                \x20   tile_store_f32::<2, 32>(a1 as *mut f32, f);\n}\n";
    let err = s
        .check(dropped_subview)
        .expect_err("a sub-view dropped to its parent must not compile -- the shapes differ");
    assert!(
        err.contains("E0308"),
        "expected a const-generic shape mismatch, got: {err}"
    );

    // WHY THIS GATE COVERS ONE OF THE TWO LIFT PATHS.
    //
    // `convert_pto_to_tile_safe` (`ptoas-rs --checked`) emits the
    // intrinsic-level form, and every kernel it emits carries
    // `#[tile_std::tile_kernel]`, whose expansion includes the codegen marker
    // `#[tile::kernel]`. Stock rustc has no `tile` module, so the form stops at
    // `E0433` before any of its contents are looked at -- and the repo's own
    // hand-written kernels stop in exactly the same place (measured
    // 2026-11-01: `cargo check` in `examples/tile_softmax/kernels` fails with
    // the identical error at its `#[tile_std::tile_kernel]`).
    //
    // So this is not a hole to plug in `emitter_tile`; the attribute is what
    // routes a kernel to the `rustc_codegen_tile` backend, and removing it to
    // get a green check would stop the output being a tile-rs kernel. The
    // control is here so that the day stock rustc CAN resolve the marker, this
    // test fails and the checked path becomes gateable. See
    // `tests/two_lift_paths.rs`.
    let marked = good.replace(
        "#[no_mangle]\npub unsafe extern \"C\" fn k(",
        "#[tile_std::tile_kernel]\n#[no_mangle]\npub unsafe extern \"C\" fn k(",
    );
    assert!(
        marked.contains("tile_kernel"),
        "the attribute was not inserted, so this control proves nothing"
    );
    let err = s.check(&marked).expect_err(
        "stock rustc now resolves the `#[tile::kernel]` codegen marker. The checked lift's          output has become type-checkable here -- extend this gate to cover it",
    );
    assert!(
        err.contains("E0433"),
        "expected the codegen marker to be unresolvable, got: {err}"
    );
}

/// DEFERRED, NOT RETIRED. This is the rustc-verification arm: it compiles the
/// kernel the lift emitted, which is where const-generic shape agreement and
/// tile-handle linearity actually get proved. It needs `tile_std`, the
/// `#![no_core]` kernel-side crate that arrives with the Rust input mode --
/// measured on stable, it fails with E0554 inside that crate rather than
/// anywhere in the lift. Restore it the moment that crate lands: this is the
/// test that makes the lift's central claim checkable rather than asserted.
#[test]
fn every_artifact_the_lift_accepts_compiles() {
    if !toolchain_available() && skipped("every_artifact_the_lift_accepts_compiles") {
        return;
    }
    let mut files = Vec::new();
    shared::collect_pto_files(&shared::corpus_root(), &mut files);
    assert!(!files.is_empty(), "no .pto artifacts found");

    let s = Scratch::new("corpus");
    let mut lifted = 0usize;
    let mut bad: Vec<String> = Vec::new();
    for f in &files {
        let Ok(src) = std::fs::read_to_string(f) else {
            continue;
        };
        let Ok(rust) = ptoas_rs::convert_pto_to_rust(&src) else {
            continue;
        };
        lifted += 1;
        if let Err(e) = s.check(&rust) {
            bad.push(format!("{}:\n    {}", f.display(), e.replace('\n', "\n    ")));
        }
    }
    // The corpus total moves as artifacts are added; what this pins is that
    // the set is not EMPTY and not a handful, because a gate over two files
    // would pass while the lift regressed to two.
    assert!(
        lifted >= 10,
        "only {lifted} of {} artifacts lifted, so this gate checked almost nothing",
        files.len()
    );
    assert!(
        bad.is_empty(),
        "{} of {lifted} lifted artifacts do not compile:\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}
