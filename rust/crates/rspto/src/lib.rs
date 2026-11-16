// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `rspto` -- lower a tile-rs kernel to PTO assembly.
//!
//! ```text
//! kernel.tile ──→ rspto_emit ──→ kernel.pto
//! ```
//!
//! The emitter is the only code path, so there is exactly one answer for a given
//! input. It is pure text to text over the standard library: no filesystem, no
//! environment, no toolchain, and no LLVM.
//!
//! # The `.rs` input mode is not in this build
//!
//! Lowering a tile-rs `.rs` kernel to MLIR means loading a codegen backend into
//! `rustc`, which needs a nightly toolchain, a prebuilt backend fetched at first
//! use, and a device target specification. None of that is required to emit PTO
//! from MLIR, so none of it is a dependency here -- this crate builds on stable
//! Rust with no third-party dependencies at all.
//!
//! `.rs` input is recognised and refused with exit code 4, "toolchain
//! unavailable", which is the same code the tool uses whenever the Rust half is
//! not usable on a machine. When that half lands it will converge on this
//! emitter, which is what makes the two modes incapable of disagreeing.

pub mod cli;

use tile_codegen::registry::TargetRegistry;
use tile_codegen::target::EmitOpts;

#[derive(Debug)]
pub enum Error {
    /// The emitter rejected the MLIR.
    Emit(String),
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Emit(s) => write!(f, "{s}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

/// MLIR text in, PTO text out. The one place either input mode ends up.
///
/// Goes through the registry rather than calling the emitter directly, so the CLI
/// exercises the same `CodegenTarget` dispatch any other consumer would.
pub fn mlir_to_pto(mlir: &str) -> Result<String, Error> {
    if mlir.trim().is_empty() {
        return Err(Error::Emit("the MLIR input is empty".into()));
    }
    let mut r = TargetRegistry::new();
    rspto_emit::register(&mut r);
    let target = r
        .select("pto")
        .ok_or_else(|| Error::Emit("the pto target did not register".into()))?;
    let out = target
        .emit(mlir, &EmitOpts::default())
        .map_err(Error::Emit)?;
    Ok(out.source)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MLIR: &str = r#"
module {
  llvm.func @vec_add(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
    %t0 = llvm.call @__tile_load_f32(%arg0, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %c32, %c32) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_add_f32(%c0, %t0, %t1, %c32, %c32) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t2, %c32, %c32) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

    #[test]
    fn mlir_in_pto_out() {
        let pto = mlir_to_pto(MLIR).expect("emit");
        assert!(pto.contains("func.func @vec_add("), "not PTO-MLIR:\n{pto}");
        assert!(pto.contains("!pto.ptr<f32>"), "no pto types:\n{pto}");
    }

    /// Empty input is a usage-level mistake, not a parse error from deep inside the
    /// emitter. Reported here so the message names the real problem.
    #[test]
    fn empty_mlir_is_rejected_with_a_clear_message() {
        let e = mlir_to_pto("   \n ").unwrap_err();
        assert!(format!("{e}").contains("empty"), "{e}");
    }

    /// The registry route is what the CLI uses, so a target that fails to register
    /// must surface as an error rather than a panic.
    #[test]
    fn the_pto_target_is_reachable_through_the_registry() {
        let mut r = TargetRegistry::new();
        rspto_emit::register(&mut r);
        assert!(r.select("pto").is_some());
        assert!(r.select("cpp").is_none(), "cpp belongs to ascendc-rs");
    }
}
