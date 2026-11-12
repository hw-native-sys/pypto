// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! `TargetRegistry` — name -> target lookup that replaces the hardcoded
//! `enum CodegenPath` + if/else chain in `compile_ascend.rs`.
//!
//! Today the host does:
//! ```ignore
//! if codegen_path == CodegenPath::Cuda      { compile_via_cuda(..) }
//! else if codegen_path == CodegenPath::Metal { compile_via_metal(..) }
//! else if ...  // one arm per target, edited by hand for every new backend
//! ```
//! With the registry it becomes:
//! ```ignore
//! let target = registry.select(&name).ok_or("unknown target")?;
//! let art = target.emit(&mlir_text, &opts)?;
//! // host drives the toolchain for `name` (compile-side stays host-side)
//! ```

use crate::target::CodegenTarget;

/// An ordered set of registered targets. Selection is by [`CodegenTarget::name`].
#[derive(Default)]
pub struct TargetRegistry {
    targets: Vec<Box<dyn CodegenTarget>>,
}

impl TargetRegistry {
    /// Empty registry. Use [`with_builtin`](Self::with_builtin) for the usual
    /// pre-populated one.
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    /// Registry pre-populated with every built-in target compiled into this
    /// build (the open reference targets always; the closed Ascend targets when
    /// the `ascend` feature is on). A closed/downstream crate can add more with
    /// [`register`](Self::register).
    pub fn with_builtin() -> Self {
        let mut r = Self::new();
        crate::targets::register_builtin(&mut r);
        r
    }

    /// Add a target. Later registrations of the same `name` shadow earlier ones
    /// in iteration but `select` returns the first match — so register overrides
    /// before built-ins if you want to replace one.
    pub fn register(&mut self, target: Box<dyn CodegenTarget>) {
        self.targets.push(target);
    }

    /// Resolve a target by its `name()` (the `TILERS_CODEGEN_PATH` value).
    pub fn select(&self, name: &str) -> Option<&dyn CodegenTarget> {
        self.targets
            .iter()
            .map(|b| b.as_ref())
            .find(|t| t.name() == name)
    }

    /// Names of all registered targets (for `--help` / diagnostics / "unknown
    /// target X; available: ...").
    pub fn names(&self) -> Vec<&'static str> {
        self.targets.iter().map(|t| t.name()).collect()
    }

    /// Number of registered targets.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}
