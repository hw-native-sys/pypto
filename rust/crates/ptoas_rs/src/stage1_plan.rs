// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Structural model of a stage-1 PTO plan — the IR `mlir_to_pto` writes and
//! `ptoas` consumes, *before* `PlanMemoryPass`.
//!
//! Where stage-2 (`plan.rs`) tracks UB byte offsets, stage-1 retains the
//! emitter-level invariants we lose by stage-2: the **parent tensor shape**
//! that backs each `pto.partition_view`, and the **literal sub-tile offsets**
//! the emitter chose into that parent.
//!
//! Rationale: a class of `mlir_to_pto` bugs (e.g. commit 06d8d30b — GEP
//! runtime-index misinterpreted as a literal integer) corrupts those offsets
//! while still producing syntactically valid PTO. By stage-2 the offset is
//! baked into a `pto.pointer_cast(%c<N>_i64)` constant in UB, with no
//! parent-tensor link to check against. Stage-1 still has both halves.
//!
//! Parser scope is intentionally narrow: only the address-shape skeleton
//! (`arith.constant`, `pto.make_tensor_view`, `pto.partition_view`,
//! `func.func`) — every other op (alloc_tile, tload, …) is skipped.

use std::collections::BTreeMap;

/// Static parent tensor created by `pto.make_tensor_view`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorView {
    /// SSA name (e.g. `%pto0`).
    pub ssa: String,
    /// GM pointer SSA (e.g. `%arg601`).
    pub gm_arg: String,
    /// Static parent rows resolved from the `shape = [%cR, %cC]` operands.
    pub rows: u64,
    /// Static parent cols.
    pub cols: u64,
    /// Element dtype as written (e.g. `f32`).
    pub dtype: String,
}

/// `pto.partition_view %tv, offsets = [%cRO, %cCO], sizes = [%cN, %cM]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionView {
    pub ssa: String,
    /// Backing tensor_view SSA.
    pub parent: String,
    /// `None` when the offset is a runtime SSA value (e.g. `%pto13` from a
    /// loop-induction `arith.muli`); only `Some` literal offsets are eligible
    /// for the OOB check.
    pub row_offset: Option<u64>,
    pub col_offset: Option<u64>,
    pub size_rows: u64,
    pub size_cols: u64,
    pub dtype: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stage1Func {
    pub name: String,
    /// `tv_ssa -> TensorView` so partition_view can be resolved against parent.
    pub tensor_views: BTreeMap<String, TensorView>,
    pub partitions: Vec<PartitionView>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Stage1Plan {
    pub funcs: Vec<Stage1Func>,
}
