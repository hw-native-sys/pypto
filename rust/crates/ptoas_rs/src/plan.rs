// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Structural model of a ptoas stage-2 plan.
//!
//! A `Plan` is what ptoas produces after `PlanMemoryPass` runs: every tile
//! has a concrete byte offset in a specific address space, with known
//! shape/dtype/layout. `stage2_parser` populates this struct;
//! `safety` walks it to detect errors decidable from the plan alone
//! (aliasing, capacity overflow, op constraints, dead tiles).

use std::collections::BTreeMap;
use std::ops::Range;

/// Target Ascend architecture — corresponds to ptoas's `--pto-arch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtoArch {
    A3,
    A5,
}

impl PtoArch {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "a3" => Some(PtoArch::A3),
            "a5" => Some(PtoArch::A5),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            PtoArch::A3 => "a3",
            PtoArch::A5 => "a5",
        }
    }
}

/// The five on-chip address spaces stage-2 uses, plus global memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AddressSpace {
    Ub,
    Mat,
    Left,
    Right,
    Acc,
    Scaling,
    Gm,
}

impl AddressSpace {
    /// Parse the string inside `#pto.address_space<…>`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "vec" => Some(AddressSpace::Ub),
            "mat" => Some(AddressSpace::Mat),
            "left" => Some(AddressSpace::Left),
            "right" => Some(AddressSpace::Right),
            "acc" => Some(AddressSpace::Acc),
            "scaling" => Some(AddressSpace::Scaling),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            AddressSpace::Ub => "vec",
            AddressSpace::Mat => "mat",
            AddressSpace::Left => "left",
            AddressSpace::Right => "right",
            AddressSpace::Acc => "acc",
            AddressSpace::Scaling => "scaling",
            AddressSpace::Gm => "gm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BLayout {
    RowMajor,
    ColMajor,
}

impl BLayout {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "row_major" => Some(BLayout::RowMajor),
            "col_major" => Some(BLayout::ColMajor),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SLayout {
    RowMajor,
    ColMajor,
    NoneBox,
}

impl SLayout {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "row_major" => Some(SLayout::RowMajor),
            "col_major" => Some(SLayout::ColMajor),
            "none_box" => Some(SLayout::NoneBox),
            _ => None,
        }
    }
}

/// Dtypes stage-2 uses. Broader than `ast::DType` because Plan must carry
/// i32 (cube accumulators) and ui64 (dequant scaling tiles).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExtDType {
    F32,
    F16,
    Bf16,
    I8,
    I32,
    UI64,
}

impl ExtDType {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "f32" => Some(ExtDType::F32),
            "f16" => Some(ExtDType::F16),
            "bf16" => Some(ExtDType::Bf16),
            "i8" => Some(ExtDType::I8),
            "i32" => Some(ExtDType::I32),
            "ui64" | "u64" => Some(ExtDType::UI64),
            _ => None,
        }
    }

    pub fn bytes(&self) -> u32 {
        match self {
            ExtDType::F32 | ExtDType::I32 => 4,
            ExtDType::F16 | ExtDType::Bf16 => 2,
            ExtDType::I8 => 1,
            ExtDType::UI64 => 8,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            ExtDType::F32 => "f32",
            ExtDType::F16 => "f16",
            ExtDType::Bf16 => "bf16",
            ExtDType::I8 => "i8",
            ExtDType::I32 => "i32",
            ExtDType::UI64 => "ui64",
        }
    }
}

/// An SSA name like `%0`, `%13`. Kept with leading `%` so the emitter's
/// output matches input textually.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ssa(pub String);

/// One on-chip tile, fully placed by `PlanMemoryPass`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileSlotX {
    pub ssa: Ssa,
    pub space: AddressSpace,
    pub offset: u64,
    pub rows: u32,
    pub cols: u32,
    pub dtype: ExtDType,
    /// The three layout fields are `Option` because one of the two stage-2
    /// spellings does not carry them.
    ///
    /// ptoas 0.26-0.29 printed placement as `pto.pointer_cast` +
    /// `pto.bind_tile` with a `#pto.tile_buf_config<…>` block holding all
    /// three. ptoas 0.68 prints `pto.alloc_tile addr = %cN_i64 :
    /// !pto.tile_buf<vec, 32x32xf32>`, with an optional `, blayout=col_major`
    /// suffix and NOTHING for `slayout` or the fractal size. Filling those in
    /// with a plausible default is the mistake this crate has already paid for
    /// five times over in its own AST -- `loc=vec`, `f32`, `: index` -- and
    /// here it would be worse than cosmetic: `check_op_constraints` warns when
    /// a `tstore_fp` scaling tile's `slayout` is not `NoneBox`, and a default
    /// of `NoneBox` turns that check into a guaranteed pass on every 0.68
    /// plan. `None` means "the dump did not say", and a consumer must decide
    /// what to do about that rather than be handed a guess.
    pub blayout: Option<BLayout>,
    pub slayout: Option<SLayout>,
    pub s_fractal: Option<u32>,
}

impl TileSlotX {
    pub fn size_bytes(&self) -> u64 {
        self.rows as u64 * self.cols as u64 * self.dtype.bytes() as u64
    }

    /// Size used when the tile is emitted as a standalone `TBuf<VECCALC>`
    /// by `mlir_to_cpp`'s vector path: `rows*cols*dtype` rounded up to
    /// 32 B (see `crates/rustc_codegen_tile/src/mlir_to_cpp.rs:4277`,
    /// `((elems * elem_size) + 31) & !31`).
    pub fn size_bytes_ub_aligned(&self) -> u64 {
        let raw = self.size_bytes();
        (raw + 31) & !31
    }

    pub fn byte_range(&self) -> Range<u64> {
        self.offset..self.offset + self.size_bytes()
    }
}

/// Ops a safety walker cares about. Control flow (`scf.for`/`scf.if`) is
/// flattened away — for aliasing/capacity/layout purposes, whether an op
/// runs once or in a loop doesn't matter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanOp {
    TLoad {
        tile: Ssa,
    },
    TStore {
        tile: Ssa,
    },
    TStoreFp {
        acc: Ssa,
        scaling: Ssa,
    },
    TMov {
        src: Ssa,
        dst: Ssa,
    },
    TMatmul {
        lhs: Ssa,
        rhs: Ssa,
        acc: Ssa,
    },
    TMatmulAcc {
        acc_in: Ssa,
        lhs: Ssa,
        rhs: Ssa,
        acc_out: Ssa,
    },
    TUnary {
        opname: String,
        src: Ssa,
        dst: Ssa,
    },
    TBinary {
        opname: String,
        a: Ssa,
        b: Ssa,
        dst: Ssa,
    },
    TRowReduce {
        opname: String,
        src: Ssa,
        scratch: Ssa,
        dst: Ssa,
    },
    TRowExpand {
        opname: String,
        src: Ssa,
        red: Ssa,
        /// The materialised scratch tile, third `ins` operand.
        ///
        /// Dropped until 2026-11-01, and it was the operand that mattered.
        /// `pto-materialize-implicit-tmp` creates one of these per
        /// `trowexpandsub`/`trowexpanddiv` -- on a 32x32 f32 softmax, two
        /// `1x2048xf32` tiles, 8192 B each, twice the size of the 4096 B data
        /// they serve. Reading only two operands left both of them with no
        /// reader, so `check_dead_tiles` reported them "allocated but never
        /// used": the only two tiles in the plan that a UB-pressure question
        /// is actually about.
        ///
        /// `Option` because the 0.26-0.29 spelling of these ops carries two
        /// operands, not three.
        scratch: Option<Ssa>,
        dst: Ssa,
    },
}

impl PlanOp {
    pub fn reads(&self) -> Vec<&Ssa> {
        match self {
            PlanOp::TLoad { .. } => vec![],
            PlanOp::TStore { tile } => vec![tile],
            PlanOp::TStoreFp { acc, scaling } => vec![acc, scaling],
            PlanOp::TMov { src, .. } => vec![src],
            PlanOp::TMatmul { lhs, rhs, .. } => vec![lhs, rhs],
            PlanOp::TMatmulAcc {
                acc_in, lhs, rhs, ..
            } => vec![acc_in, lhs, rhs],
            PlanOp::TUnary { src, .. } => vec![src],
            PlanOp::TBinary { a, b, .. } => vec![a, b],
            PlanOp::TRowReduce { src, scratch, .. } => vec![src, scratch],
            PlanOp::TRowExpand {
                src,
                red,
                scratch,
                ..
            } => match scratch {
                Some(t) => vec![src, red, t],
                None => vec![src, red],
            },
        }
    }

    pub fn writes(&self) -> Vec<&Ssa> {
        match self {
            PlanOp::TLoad { tile } => vec![tile],
            PlanOp::TStore { .. } => vec![],
            PlanOp::TStoreFp { .. } => vec![],
            PlanOp::TMov { dst, .. } => vec![dst],
            PlanOp::TMatmul { acc, .. } => vec![acc],
            PlanOp::TMatmulAcc { acc_out, .. } => vec![acc_out],
            PlanOp::TUnary { dst, .. } => vec![dst],
            PlanOp::TBinary { dst, .. } => vec![dst],
            PlanOp::TRowReduce { dst, .. } => vec![dst],
            PlanOp::TRowExpand { dst, .. } => vec![dst],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanFunc {
    pub name: String,
    pub slots: BTreeMap<Ssa, TileSlotX>,
    pub ops: Vec<PlanOp>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub arch: PtoArch,
    pub funcs: Vec<PlanFunc>,
}

impl Plan {
    pub fn func(&self, name: &str) -> Option<&PlanFunc> {
        self.funcs.iter().find(|f| f.name == name)
    }
}
