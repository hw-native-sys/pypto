// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Built-in target registration -- the extension seam.
//!
//! * [`DebugTarget`] is self-contained and always present, so the trait, the
//!   registry and the emit path can be exercised with no external dependency.
//! * [`EmitterTarget`] is the generic adapter: any
//!   `fn(&str) -> Result<String, String>` becomes a target through it.
//! * A real target lives in its own crate and registers itself with
//!   [`TargetRegistry::register`]. `rspto_emit` does exactly that for PTO.
//!
//! That last point is the whole design. Adding a target is implementing one
//! trait and making one registration call from outside; there is no enum to
//! extend here and no dispatch arm to add, so nothing in this crate knows which
//! targets exist.

use crate::registry::TargetRegistry;
use crate::target::{CodegenTarget, EmitOpts, EmitOut, TargetMeta};

/// Register every target compiled into this build.
pub fn register_builtin(r: &mut TargetRegistry) {
    r.register(Box::new(DebugTarget));

}

/// Self-contained reference target. Proves the trait/registry/emit path end to
/// end with no LLVM, no parser, no toolchain — this is what makes the open
/// skeleton verifiable on any machine (including this macOS build). Echoes the
/// MLIR back inside a comment banner.
pub struct DebugTarget;

impl CodegenTarget for DebugTarget {
    fn name(&self) -> &'static str {
        "debug"
    }

    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        if mlir_text.trim().is_empty() {
            return Err("empty MLIR module".to_string());
        }
        let lines = mlir_text.lines().count();
        let funcs = mlir_text.matches("func.func").count();
        Ok(EmitOut {
            source: format!(
                "// tile-rs debug target\n// {lines} MLIR lines, {funcs} func.func op(s)\n{mlir_text}\n"
            ),
            ext: "mlir.txt",
            meta: TargetMeta::default(),
        })
    }
}

// `register_emitters` lived here: one `EmitterTarget` registration per
// non-Ascend backend. Those backends target processors this repository's licence
// does not cover, so neither they nor their registrations are part of this tree.
// `EmitterTarget` itself stays -- it is the generic adapter any
// `(&str) -> Result<String, String>` emitter plugs into, and `rspto_emit` uses it.

pub struct EmitterTarget {
    name: &'static str,
    ext: &'static str,
    f: fn(&str) -> Result<String, String>,
}

impl EmitterTarget {
    pub fn new(name: &'static str, ext: &'static str, f: fn(&str) -> Result<String, String>) -> Self {
        Self { name, ext, f }
    }
}

impl CodegenTarget for EmitterTarget {
    fn name(&self) -> &'static str {
        self.name
    }
    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        let source = (self.f)(mlir_text)?;
        Ok(EmitOut { source, ext: self.ext, meta: TargetMeta::default() })
    }
}
