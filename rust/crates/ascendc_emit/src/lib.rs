// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The `cpp` target: tile-dialect MLIR to AscendC C++ for bisheng.
//!
//! One crate, one target. [`AscendCTarget`] implements the same
//! [`tile_codegen::target::CodegenTarget`] trait as any other and joins through
//! the same [`tile_codegen::registry::TargetRegistry::register`] call. The only
//! things that distinguish it are that it reads the `ub_size` hardware parameter
//! and returns cube-kernel metadata, and both are absorbed by the uniform
//! `EmitOpts` / `EmitOut` shape -- so AscendC is not a bespoke signature sitting
//! outside the dispatch.
//!
//! Emission is pure text to text over the standard library: no filesystem, no
//! environment, no toolchain. It did read an optimization-knowledge JSON file to
//! add an advisory comment, and `mlir_to_cpp.rs` records at the former call site
//! why that is gone -- the short version is that it made the same binary emit
//! different C++ on different machines.
//!
//! # Why both modules sit at the crate root
//!
//! `mlir_to_cpp` reaches the parser as a root-level sibling. Nesting either one
//! level down breaks those paths, which is a compile error rather than a style
//! question, so the flat layout is load-bearing.

#[allow(warnings, clippy::all)]
#[rustfmt::skip]
mod mlir_parse;

#[allow(warnings, clippy::all)]
#[rustfmt::skip]
pub mod mlir_to_cpp;
use mlir_to_cpp as cpp_impl;

use tile_codegen::registry::TargetRegistry;
use tile_codegen::target::{CodegenTarget, EmitOpts, EmitOut, TargetMeta};

/// Register the AscendC target as a peer of the open ones.
pub fn register(r: &mut TargetRegistry) {
    r.register(Box::new(AscendCTarget));
}

/// AscendC C++ (`mlir_to_cpp`) → bisheng.
pub struct AscendCTarget;

impl CodegenTarget for AscendCTarget {
    fn name(&self) -> &'static str {
        "cpp"
    }

    fn emit(&self, mlir_text: &str, opts: &EmitOpts) -> Result<EmitOut, String> {
        // The one emitter that needs a hardware param and returns metadata.
        let out = cpp_impl::convert_mlir_to_cpp(mlir_text, opts.hw.ub_size)?;
        Ok(EmitOut {
            source: out.source,
            ext: "cpp",
            meta: TargetMeta {
                has_cube_kernel: out.has_cube_kernel,
                kernel_names: out.kernel_names,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `hacc.entry` kernel with real `__tile_*` intrinsics. `mlir_to_cpp` refuses
    /// a module with no entry-point kernel carrying tile intrinsics, which is what
    /// makes a trivial `func.func` fixture useless here.
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

    /// `cpp` is the ONLY target this crate contributes. A second one appearing
    /// means a backport brought in a backend that belongs in another viewpoint.
    #[test]
    fn registers_exactly_cpp() {
        let mut r = TargetRegistry::new();
        let before = r.names().len();
        register(&mut r);
        assert!(r.names().contains(&"cpp"), "cpp did not register");
        assert!(!r.names().contains(&"pto"), "pto belongs in pto_codegen");
        assert_eq!(r.names().len(), before + 1, "registered more than cpp");
    }

    #[test]
    fn emits_ascendc_through_the_uniform_shape() {
        let out = AscendCTarget
            .emit(MLIR, &EmitOpts::default())
            .expect("cpp emitter rejected the module");
        assert_eq!(out.ext, "cpp");
        assert!(
            out.source.contains("kernel_operator.h"),
            "not AscendC:\n{}",
            out.source
        );
    }
}
