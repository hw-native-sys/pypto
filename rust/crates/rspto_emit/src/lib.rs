// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `pto` target: tile-dialect MLIR to PTO assembly.
//!
//! One crate, one target. [`PtoTarget`] implements the same
//! [`tile_codegen::target::CodegenTarget`] trait as any other and joins through
//! the same [`tile_codegen::registry::TargetRegistry::register`] call, so PTO is
//! not a bespoke signature sitting outside the dispatch. The emitter itself is
//! pure text to text over the standard library, which is why this builds on
//! stable Rust in seconds and needs neither LLVM nor MLIR bindings.
//!
//! # Why both modules sit at the crate root
//!
//! `mlir_to_pto` imports `crate::mlir_parse`, which resolves only while
//! `mlir_parse` is a root module and the emitter is its sibling. Nesting them one
//! level down breaks the emitter's `crate::` paths, which is a compile error
//! rather than a style question -- so the flat layout here is load-bearing.

#[allow(warnings, clippy::all)]
#[rustfmt::skip]
mod mlir_parse;

#[allow(warnings, clippy::all)]
#[rustfmt::skip]
mod mlir_to_pto;
use mlir_to_pto as pto_impl;

use tile_codegen::registry::TargetRegistry;
use tile_codegen::target::{CodegenTarget, EmitOpts, EmitOut, TargetMeta};

/// Register the PTO target as a peer of the open ones.
pub fn register(r: &mut TargetRegistry) {
    r.register(Box::new(PtoTarget));
}

/// PTO-MLIR assembly (`mlir_to_pto`) → ptoas → bisheng.
pub struct PtoTarget;

impl CodegenTarget for PtoTarget {
    fn name(&self) -> &'static str {
        "pto"
    }

    fn emit(&self, mlir_text: &str, _opts: &EmitOpts) -> Result<EmitOut, String> {
        let source = pto_impl::convert_mlir_to_pto(mlir_text)?;
        Ok(EmitOut {
            source,
            ext: "pto",
            meta: TargetMeta::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `hacc.entry` kernel with real `__tile_*` intrinsics — the shape the
    /// emitter accepts.
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

    /// `pto` is the ONLY target this crate contributes. A second one appearing
    /// means a backport brought in a backend that belongs in another viewpoint.
    #[test]
    fn registers_exactly_pto() {
        let mut r = TargetRegistry::new();
        let before = r.names().len();
        register(&mut r);
        assert!(r.names().contains(&"pto"), "pto did not register");
        assert!(
            !r.names().contains(&"cpp"),
            "cpp belongs in ascendc_codegen"
        );
        assert_eq!(r.names().len(), before + 1, "registered more than pto");
    }

    #[test]
    fn emits_pto_through_the_uniform_shape() {
        let out = PtoTarget
            .emit(MLIR, &EmitOpts::default())
            .expect("pto emitter rejected the module");
        assert_eq!(out.ext, "pto");
        assert!(
            out.source.contains("func.func @vec_add("),
            "not PTO-MLIR:\n{}",
            out.source
        );
    }
}
