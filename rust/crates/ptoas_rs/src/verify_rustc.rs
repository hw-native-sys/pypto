// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Hand the lifted kernel to the Rust compiler, and report what it proved.
//!
//! The lift's whole claim is that compiling its output puts two invariants in
//! front of a checker that the assembler emits without independently proving:
//!
//! * **shape agreement** -- const-generic `Tile<R, C, T>` makes a `tmatmul`
//!   whose inner dimensions disagree a type error rather than a surprise on a
//!   card; and
//! * **tile-handle linearity** -- a tile is deliberately not `Copy`, so a
//!   double store or a use after scope is a borrow-check error.
//!
//! Neither claim is worth anything until something actually compiles the
//! output. That is what this module does, and it is the difference between
//! "rustc could check this" and "rustc did".
//!
//! # What it does NOT prove
//!
//! A clean report here is not kernel correctness. It says nothing about pipe
//! barriers, about the per-part sub-view hazard, about which core a kernel is
//! partitioned onto, or about DMA bounds when the shapes are runtime-dynamic.
//! Those belong to the emitters and to the device. Saying so is part of the
//! report rather than a footnote, because a verifier that is quiet about its
//! scope invites being read as a proof of more than it checked.
//!
//! # Why this can report "unavailable" rather than "clean"
//!
//! Compiling the lift's output needs the kernel-side DSL it targets, and that
//! crate is `#![no_core]` -- it declares its own lang items because the
//! accelerator target has no `core` to link, so it builds only on the pinned
//! nightly. It is not in this workspace yet. Miri is nightly-only for the same
//! practical purpose.
//!
//! So this module locates the DSL and refuses to guess. **A verifier that
//! cannot run must say so rather than return a clean report** -- an empty
//! finding list and an unavailable checker are indistinguishable to a caller
//! reading an exit code, and the second one has been mistaken for the first
//! often enough to be worth the extra state.

use std::path::{Path, PathBuf};
use std::process::Command;

/// How hard to look at the lifted kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// `cargo check`: shape agreement and handle linearity. Cheap, and it is
    /// where the lift's two claims actually live.
    Check,
    /// `cargo miri run` over a host harness: out-of-bounds slicing and
    /// undefined behaviour in the pointer arithmetic the lift emits.
    ///
    /// Applies to the DEFAULT (wrapper) lift only. The `--checked`
    /// intrinsic-level form carries a kernel attribute and needs the codegen
    /// backend to compile at all, so there is nothing for Miri to run.
    Miri,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Check => "rustc",
            Tier::Miri => "miri",
        }
    }
}

/// What the compiler said.
#[derive(Debug)]
pub enum Outcome {
    /// The checker ran and accepted the kernel.
    Proved {
        tier: Tier,
    },
    /// The checker ran and rejected it. `diagnostics` is its output verbatim --
    /// not summarised, because the compiler's own message is more precise than
    /// anything this module would write about it.
    Rejected {
        tier: Tier,
        diagnostics: String,
    },
    /// The checker could not run. Distinct from `Proved` on purpose.
    Unavailable {
        tier: Tier,
        why: String,
    },
}

impl Outcome {
    /// True only when a checker ran and accepted. `Unavailable` is not success.
    pub fn proved(&self) -> bool {
        matches!(self, Outcome::Proved { .. })
    }

    /// Console form. One line for a verdict, the compiler's own text below it
    /// for a rejection.
    pub fn render(&self) -> String {
        match self {
            Outcome::Proved { tier } => {
                format!("{}: accepted the lifted kernel", tier.name())
            }
            Outcome::Rejected { tier, diagnostics } => {
                format!("{}: rejected the lifted kernel\n{}", tier.name(), diagnostics)
            }
            Outcome::Unavailable { tier, why } => {
                format!("{}: unavailable -- {why}", tier.name())
            }
        }
    }
}

/// Where the kernel-side DSL is, if it is anywhere.
///
/// Searched rather than assumed, and the search is three fixed places rather
/// than an environment variable: a path a user can set is a path that can point
/// at a stale copy, and a verifier reporting on the wrong declarations is worse
/// than one reporting unavailable.
/// The toolchain the kernel-side DSL needs: nightly, permanently, because it
/// is `#![no_core]` and declares its own lang items. Not a stabilisation wait.
pub const KERNEL_TOOLCHAIN: &str = "nightly-2025-08-04";

pub fn locate_tile_std(manifest_dir: &Path) -> Option<PathBuf> {
    locate_sibling(manifest_dir, "tile_std")
}

/// Where the `std` CPU reference is. Same search, different crate.
pub fn locate_tile_std_ref(manifest_dir: &Path) -> Option<PathBuf> {
    locate_sibling(manifest_dir, "tile_std_ref")
}

fn locate_sibling(manifest_dir: &Path, name: &str) -> Option<PathBuf> {
    for rel in [
        format!("../{name}"),
        format!("../../{name}"),
        format!("../../../rust/crates/{name}"),
    ] {
        let p = manifest_dir.join(rel);
        if p.join("Cargo.toml").is_file() {
            return p.canonicalize().ok();
        }
    }
    None
}

/// Compile `lifted` and report what the compiler proved.
///
/// `scratch` is a directory this function may create a throwaway crate in. The
/// caller owns it, so a failed run leaves the evidence in place.
pub fn verify(
    lifted: &str,
    pto_src: &str,
    tier: Tier,
    scratch: &Path,
    manifest_dir: &Path,
) -> Outcome {
    // The Miri tier runs the kernel, which needs a driver and a reference, so
    // it takes a different path entirely from here. See `verify_miri`.
    if tier == Tier::Miri {
        return verify_miri(lifted, pto_src, scratch, manifest_dir);
    }
    let Some(dsl) = locate_tile_std(manifest_dir) else {
        return Outcome::Unavailable {
            tier,
            why: "the kernel-side DSL is not in this workspace, so there is \
                  nothing to compile the lifted kernel against. It arrives with \
                  the Rust input mode."
                .to_string(),
        };
    };

    if let Err(e) = write_scratch_crate(lifted, scratch, &dsl) {
        return Outcome::Unavailable {
            tier,
            why: format!("could not stage a crate in {}: {e}", scratch.display()),
        };
    }

    // Plain `cargo` from PATH is the rustup proxy, which honours
    // `RUSTUP_TOOLCHAIN`. The toolchain matters: `tile_std` is `#![no_core]`,
    // so the ambient stable cargo rejects it at `#![feature]` with E0554 --
    // and the verifier then reported "rustc: rejected the lifted kernel",
    // blaming the kernel for the toolchain. Miri is nightly-only for the same
    // reason. Override with `PTOAS_RS_KERNEL_TOOLCHAIN`.
    let toolchain = std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN")
        .unwrap_or_else(|_| KERNEL_TOOLCHAIN.to_string());
    let mut cmd = Command::new("cargo");
    cmd.env("RUSTUP_TOOLCHAIN", &toolchain);
    // The scratch crate is its own cargo root; sharing the caller's RUSTFLAGS
    // would carry a `-C debuginfo` that ICEs the no_core crate.
    cmd.env_remove("RUSTFLAGS");
    match tier {
        Tier::Check => {
            cmd.arg("check");
        }
        Tier::Miri => {
            cmd.arg("miri").arg("run");
        }
    }
    cmd.arg("--manifest-path").arg(scratch.join("Cargo.toml"));

    match cmd.output() {
        Err(e) => Outcome::Unavailable {
            tier,
            why: format!("cargo could not be run: {e}"),
        },
        Ok(out) if out.status.success() => Outcome::Proved { tier },
        Ok(out) => {
            let diagnostics = String::from_utf8_lossy(&out.stderr).to_string();
            // Miri absent is not the kernel's fault, and reporting it as a
            // rejection would blame the input for a missing component.
            if tier == Tier::Miri && diagnostics.contains("no such command") {
                return Outcome::Unavailable {
                    tier,
                    why: "miri is not installed on this toolchain (it is a \
                          nightly component)"
                        .to_string(),
                };
            }
            // Nor is the SHAPE of the scratch crate the kernel's fault, and
            // this one was reported as a rejection until the DSL arrived and
            // Miri could run at all. `cargo miri run` needs a binary; the
            // lifted kernel is a `#![no_core]` library whose entry points are
            // `#[no_mangle] extern "C"` functions. There is nothing to run,
            // and `cargo miri test` is no way out either: a `no_core` crate
            // cannot host `#[test]`.
            //
            // So Miri's tier is UNAVAILABLE, not failing: proving anything
            // with it needs a driver that calls a kernel with real buffers,
            // and that driver cannot itself be `no_core`. Saying "rejected"
            // here would mark every kernel in the corpus unsafe on the
            // strength of a missing `fn main`.
            if tier == Tier::Miri && diagnostics.contains("a bin target must be available") {
                return Outcome::Unavailable {
                    tier,
                    why: "the lifted kernel is a no_core library with C entry points, so \
                          there is no binary for `cargo miri run` to execute. Miri needs a \
                          driver that calls a kernel with real buffers; that driver does \
                          not exist in this tree yet"
                        .to_string(),
                };
            }
            Outcome::Rejected { tier, diagnostics }
        }
    }
}

fn write_scratch_crate(lifted: &str, dir: &Path, dsl: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"lifted_kernel\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n\
             [dependencies]\ntile_std = {{ path = {:?} }}\n\n[workspace]\n",
            dsl
        ),
    )?;
    std::fs::write(dir.join("src/lib.rs"), lifted)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The path that exists today, and the one that matters: an absent DSL is
    /// reported as unavailable, NOT as a clean verification.
    #[test]
    fn an_absent_dsl_is_unavailable_and_not_proved() {
        let nowhere = Path::new("/nonexistent-ptoas-rs-probe");
        let out = verify("pub fn k() {}", "", Tier::Check, nowhere, nowhere);
        assert!(!out.proved(), "an absent checker must not report success");
        match out {
            Outcome::Unavailable { why, .. } => {
                assert!(why.contains("kernel-side DSL"), "{why}");
            }
            other => panic!("expected Unavailable, got {other:?}"),
        }
    }

    #[test]
    fn locate_returns_none_when_there_is_nothing_to_find() {
        assert!(locate_tile_std(Path::new("/nonexistent-ptoas-rs-probe")).is_none());
    }

    /// Rendering is part of the contract: these strings are what a user reads.
    #[test]
    fn every_outcome_renders_distinguishably() {
        let p = Outcome::Proved { tier: Tier::Check }.render();
        let r = Outcome::Rejected {
            tier: Tier::Check,
            diagnostics: "error[E0308]: mismatched types".into(),
        }
        .render();
        let u = Outcome::Unavailable {
            tier: Tier::Miri,
            why: "no toolchain".into(),
        }
        .render();
        assert!(p.contains("accepted"));
        assert!(r.contains("rejected") && r.contains("E0308"));
        assert!(u.contains("unavailable"));
        // The one confusion that must not be possible.
        assert!(!u.contains("accepted"));
    }
}

// --- the Miri tier ---------------------------------------------------------
//
// Miri used to report itself unavailable here, and the reason was structural:
// `cargo miri run` needs a binary, a lifted kernel is a library whose entry
// points are `extern "C"` functions, and the kernel-side DSL is `#![no_core]`
// -- which cannot host a `#[test]` and cannot share a crate graph with `std`,
// because it declares its own lang items. There was nothing to execute and no
// way to write a driver that executed it.
//
// `tile_std_ref` is the way out: the same tile-API signatures with ordinary
// Rust bodies, in an ordinary `std` crate. A lifted kernel repointed at it
// compiles and runs as normal Rust, so Miri can see every load and store the
// kernel makes through the pointers it was handed.

/// Rewrite a lifted kernel so it compiles against the `std` reference.
///
/// Three edits, and each one is here because the lift's output targets a device:
///
///   * the `#![feature(no_core)]` / `#![no_std]` / `#![no_core]` attributes go,
///     because the reference is a `std` crate;
///   * `use tile_std::tile::*` becomes the reference's path;
///   * nothing else. In particular the kernel BODY is untouched, which is the
///     whole point -- Miri must see the pointer arithmetic the lift actually
///     emitted, not a cleaned-up version of it.
fn repoint_at_reference(lifted: &str) -> String {
    lifted
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("#![feature(") || t.starts_with("#![no_std]") || t.starts_with("#![no_core]"))
        })
        .map(|l| {
            l.replace("use tile_std::tile::*;", "use tile_std_ref::tile::*;")
                .replace("tile_std::tile::", "tile_std_ref::tile::")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `main` that allocates buffers and calls each kernel.
///
/// Buffer sizes come from `kernel_signatures`, i.e. from the PTO, NOT from the
/// emitted Rust. That is what makes the run a comparison rather than a
/// tautology: extents read back out of the lift's own output would agree with
/// it by construction, so a kernel whose extents were wrong would get buffers
/// wrong in the same way and pass clean.
///
/// Inputs are a deterministic ramp rather than zeros. Zeros make a
/// wrong-extent read land on an in-bounds zero often enough to matter, and they
/// make every softmax uniform, so an indexing error would produce the right
/// answer for the wrong reason.
fn driver(sigs: &[crate::emitter_tile::KernelSignature]) -> String {
    let mut m = String::from(
        "\nfn main() {\n    \
         // Deterministic, distinct, and sized EXACTLY as the PTO declares.\n",
    );
    for (k, sig) in sigs.iter().enumerate() {
        m.push_str(&format!("    // {}\n", sig.name));
        for (i, a) in sig.args.iter().enumerate() {
            let n = a.len();
            if a.writable {
                m.push_str(&format!(
                    "    let mut k{k}_a{i}: Vec<f32> = vec![0.0; {n}];\n"
                ));
            } else {
                m.push_str(&format!(
                    "    let k{k}_a{i}: Vec<f32> = (0..{n}).map(|x| \
                     (x % 17) as f32 * 0.25 - 2.0).collect();\n"
                ));
            }
        }
        let args: Vec<String> = sig
            .args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if a.writable {
                    format!("k{k}_a{i}.as_mut_ptr()")
                } else {
                    // `as *mut` on a read-only buffer, because the lift emits
                    // `*mut f32` for every parameter. Miri will report it if
                    // the kernel actually writes through one.
                    format!("k{k}_a{i}.as_ptr() as *mut f32")
                }
            })
            .collect();
        m.push_str(&format!(
            "    unsafe {{ {}({}) }};\n",
            sig.name,
            args.join(", ")
        ));
        // Read every output back, so a store Miri allowed but that wrote
        // nothing is still visible as a value.
        for (i, a) in sig.args.iter().enumerate() {
            if a.writable {
                m.push_str(&format!(
                    "    let s{k}_{i}: f32 = k{k}_a{i}.iter().sum();\n    \
                     assert!(s{k}_{i}.is_finite(), \"{}: output {i} is not finite\");\n",
                    sig.name
                ));
            }
        }
    }
    m.push_str("    println!(\"ok\");\n}\n");
    m
}

fn verify_miri(lifted: &str, pto_src: &str, scratch: &Path, manifest_dir: &Path) -> Outcome {
    let tier = Tier::Miri;
    let Some(reference) = locate_tile_std_ref(manifest_dir) else {
        return Outcome::Unavailable {
            tier,
            why: "the `std` CPU reference (tile_std_ref) is not in this workspace, so \
                  there is nothing to run the lifted kernel against"
                .to_string(),
        };
    };
    let sigs = match crate::emitter_tile::kernel_signatures(pto_src) {
        Ok(s) if !s.is_empty() => s,
        Ok(_) => {
            return Outcome::Unavailable {
                tier,
                why: "no kernel signatures could be read from the PTO, so a driver \
                      cannot size its buffers"
                    .to_string(),
            }
        }
        Err(e) => {
            // The CHECKED lift refuses strictly more than the default one, and
            // the signatures come from it. So a kernel the default lift
            // accepted can still have no signature -- that is an availability
            // failure, not a finding about the kernel.
            return Outcome::Unavailable {
                tier,
                why: format!(
                    "the checked lift cannot describe this kernel's parameters, so a \
                     driver cannot be built for it: {e}"
                ),
            };
        }
    };

    let dir = scratch.join("miri");
    if let Err(e) = std::fs::create_dir_all(dir.join("src")) {
        return Outcome::Unavailable {
            tier,
            why: format!("could not stage a crate in {}: {e}", dir.display()),
        };
    }
    let manifest = format!(
        "[package]\nname = \"lifted_kernel_miri\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
         [[bin]]\nname = \"lifted_kernel_miri\"\npath = \"src/main.rs\"\n\n\
         [dependencies]\ntile_std_ref = {{ path = {:?} }}\n\n[workspace]\n",
        reference.display().to_string()
    );
    let source = format!("{}{}", repoint_at_reference(lifted), driver(&sigs));
    if let Err(e) = std::fs::write(dir.join("Cargo.toml"), manifest)
        .and_then(|()| std::fs::write(dir.join("src/main.rs"), &source))
    {
        return Outcome::Unavailable {
            tier,
            why: format!("could not write the Miri driver: {e}"),
        };
    }

    let toolchain = std::env::var("PTOAS_RS_KERNEL_TOOLCHAIN")
        .unwrap_or_else(|_| KERNEL_TOOLCHAIN.to_string());
    let out = Command::new("cargo")
        .args(["miri", "run", "--quiet"])
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", dir.join("t"))
        .env("RUSTUP_TOOLCHAIN", &toolchain)
        .env_remove("RUSTFLAGS")
        .output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            return Outcome::Unavailable {
                tier,
                why: format!("could not run `cargo miri` under `{toolchain}`: {e}"),
            }
        }
    };
    let text = String::from_utf8_lossy(&out.stderr).to_string();
    if out.status.success() {
        return Outcome::Proved { tier };
    }
    // Miri absent, or the toolchain absent, is not the kernel's fault.
    for marker in ["no such command", "is not installed", "not supported by the following"] {
        if text.contains(marker) {
            return Outcome::Unavailable {
                tier,
                why: format!(
                    "miri could not run under `{toolchain}`: install it with \
                     `rustup +{toolchain} component add miri`.\n{}",
                    text.lines().take(4).collect::<Vec<_>>().join("\n")
                ),
            };
        }
    }
    // Anything else is a real finding: Miri executed the kernel and objected.
    Outcome::Rejected {
        tier,
        diagnostics: text,
    }
}
