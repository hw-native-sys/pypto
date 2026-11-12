// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Schema-driven pointwise specialisation — the L2 lift.
//!
//! Two things live here, and they are the two halves of the O6 decision
//! (*monomorphise on the tile, not the tensor*):
//!
//! 1. **Type promotion.** [`elementwise_dtypes`] returns a *pair*: the dtype the
//!    kernel accumulates in, and the dtype it stores. Keeping those separate is
//!    what makes "accumulate f16 in f32" a property of the schema rather than a
//!    thing each operator remembers, and it is where a whole class of accuracy
//!    failures comes from when it is left implicit.
//!
//! 2. **Rank collapse.** [`collapse`] turns any N-D operation over an axis into
//!    `(OUTER, R, INNER)` plus an [`AxisClass`]. Rank then leaves the instance
//!    space entirely, and adversarial extents — `[1009, 1021]`,
//!    `[363, 367, 373]`, `[11, 13, 17, 67, 67]` — are handled by construction
//!    instead of by enumeration.
//!
//! Together they give [`PointwiseSchema::instances`]: the *bounded* set of
//! kernels a schema must emit, keyed by `(dtype, axis-class, TILE)` and never by
//! extent.
//!
//! Std-only and hardware-free, like the rest of this crate — every test here
//! runs on any box.
//!
//! ## Provenance
//!
//! The promotion lattice and the computation/result split follow PyTorch's
//! `torch._prims_common::elementwise_dtypes`, which is also what FlagGems'
//! `utils/type_utils.py` calls through to. This is a reimplementation from the
//! specified semantics, not a transcription of a table — see the note on
//! [`elementwise_dtypes`] about what is and is not covered.

use core::fmt;

/// A scalar element type. Covers the nine dtypes the CANNBench operator set
/// uses; complex is deliberately absent (see [`PromotionKind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarType {
    Bool,
    I8,
    U8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

/// Promotion proceeds category-first: a floating operand beats an integral one
/// whatever their widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeCategory {
    Bool,
    Integral,
    Floating,
}

impl ScalarType {
    /// Width in bytes.
    pub fn bytes(self) -> usize {
        match self {
            ScalarType::Bool | ScalarType::I8 | ScalarType::U8 => 1,
            ScalarType::I16 | ScalarType::F16 | ScalarType::BF16 => 2,
            ScalarType::I32 | ScalarType::F32 => 4,
            ScalarType::I64 | ScalarType::F64 => 8,
        }
    }

    pub fn category(self) -> TypeCategory {
        match self {
            ScalarType::Bool => TypeCategory::Bool,
            ScalarType::I8
            | ScalarType::U8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64 => TypeCategory::Integral,
            ScalarType::F16 | ScalarType::BF16 | ScalarType::F32 | ScalarType::F64 => {
                TypeCategory::Floating
            }
        }
    }

    pub fn is_floating(self) -> bool {
        self.category() == TypeCategory::Floating
    }

    /// The type this one accumulates in.
    ///
    /// The half formats accumulate in f32 — summing a long reduction in f16
    /// loses the tail of the sum outright. Everything else accumulates in
    /// itself.
    pub fn opmath(self) -> ScalarType {
        match self {
            ScalarType::F16 | ScalarType::BF16 => ScalarType::F32,
            other => other,
        }
    }

    /// Whether the vector unit has a compare path for this type.
    ///
    /// The A2/A3 vector units have no 64-bit compare and no 32-bit *integer*
    /// compare; a predicate on those drops to scalar at roughly an eighth of the
    /// vector rate. Cast the compared operand to f32, or split an i64 into hi/lo
    /// i32 planes.
    pub fn has_vector_compare(self) -> bool {
        !matches!(self, ScalarType::I64 | ScalarType::I32)
    }

    /// Canonical lowercase name, matching the `dtype` column of the CANNBench
    /// case matrix.
    pub fn name(self) -> &'static str {
        match self {
            ScalarType::Bool => "bool",
            ScalarType::I8 => "int8",
            ScalarType::U8 => "uint8",
            ScalarType::I16 => "int16",
            ScalarType::I32 => "int32",
            ScalarType::I64 => "int64",
            ScalarType::F16 => "float16",
            ScalarType::BF16 => "bfloat16",
            ScalarType::F32 => "float32",
            ScalarType::F64 => "float64",
        }
    }

    /// Parse a CANNBench `dtype` string.
    pub fn from_name(s: &str) -> Option<ScalarType> {
        Some(match s {
            "bool" => ScalarType::Bool,
            "int8" => ScalarType::I8,
            "uint8" => ScalarType::U8,
            "int16" => ScalarType::I16,
            "int32" => ScalarType::I32,
            "int64" => ScalarType::I64,
            "float16" | "half" => ScalarType::F16,
            "bfloat16" => ScalarType::BF16,
            "float32" | "float" => ScalarType::F32,
            "float64" | "double" => ScalarType::F64,
            _ => return None,
        })
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The binary promotion lattice.
///
/// Category first, then width. Two cases are not "take the wider one" and are
/// the ones worth remembering:
///
/// * `u8 + i8 -> i16` — neither holds the other's range.
/// * `f16 + bf16 -> f32` — same width, incomparable exponent/mantissa splits,
///   so neither can represent the other.
pub fn promote_types(a: ScalarType, b: ScalarType) -> ScalarType {
    use ScalarType::*;
    if a == b {
        return a;
    }
    if a == Bool {
        return b;
    }
    if b == Bool {
        return a;
    }

    match (a.category(), b.category()) {
        (TypeCategory::Floating, TypeCategory::Integral) => a,
        (TypeCategory::Integral, TypeCategory::Floating) => b,
        (TypeCategory::Integral, TypeCategory::Integral) => match (a, b) {
            (U8, I8) | (I8, U8) => I16,
            _ => {
                if a.bytes() >= b.bytes() {
                    signed_of(a)
                } else {
                    signed_of(b)
                }
            }
        },
        (TypeCategory::Floating, TypeCategory::Floating) => match (a, b) {
            (F16, BF16) | (BF16, F16) => F32,
            _ => {
                if a.bytes() >= b.bytes() {
                    a
                } else {
                    b
                }
            }
        },
        // Bool pairs were handled above.
        _ => a,
    }
}

/// The signed type of the same width — `u8` promoted against a wider signed
/// type yields a signed result.
fn signed_of(t: ScalarType) -> ScalarType {
    match t {
        ScalarType::U8 => ScalarType::I16,
        other => other,
    }
}

/// How an operand participates in promotion.
///
/// The distinction is not decoration: a Python scalar must not drag an f16
/// tensor up to f32, but it *does* raise the category of an integral tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandKind {
    /// A tensor with at least one dimension. Dominates promotion.
    Dimensioned,
    /// A 0-dim tensor. Considered only if no dimensioned operand shares the
    /// winning category.
    ZeroDim,
    /// A Python-level number. Sets the category floor and nothing else.
    Scalar,
}

/// One input to an elementwise operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    pub ty: ScalarType,
    pub kind: OperandKind,
}

impl Operand {
    pub fn tensor(ty: ScalarType) -> Operand {
        Operand {
            ty,
            kind: OperandKind::Dimensioned,
        }
    }
    pub fn zero_dim(ty: ScalarType) -> Operand {
        Operand {
            ty,
            kind: OperandKind::ZeroDim,
        }
    }
    pub fn scalar(ty: ScalarType) -> Operand {
        Operand {
            ty,
            kind: OperandKind::Scalar,
        }
    }
}

/// What the operator wants promotion to do beyond the default lattice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromotionKind {
    /// Promote, then accumulate in the opmath type. `add`, `mul`, `gelu`.
    Default,
    /// Promote, accumulate in the result type. For ops that must not widen —
    /// a pure data movement or a bit-exact copy.
    NoOpMath,
    /// Integral and bool inputs become f32. `sigmoid`, `mish`, `erf`.
    IntToFloat,
    /// The result is bool whatever the inputs. Comparisons.
    AlwaysBool,
    /// Bool becomes i64; anything else is unchanged. Counting reductions.
    BoolToLong,
}

/// The pair every elementwise kernel needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Promoted {
    /// What the kernel accumulates in.
    pub computation: ScalarType,
    /// What the kernel stores.
    pub result: ScalarType,
}

/// Resolve the computation and result dtypes for an elementwise operation.
///
/// Follows `torch._prims_common::elementwise_dtypes`:
///
/// 1. Take the highest [`TypeCategory`] across all operands.
/// 2. Within that category, promote over the dimensioned operands; if none
///    carry it, over the 0-dim ones; if still none — the category came from a
///    bare scalar — use that category's default type rather than the scalar's
///    own, which is why `int32_tensor + 1.0` is f32 and not f64.
/// 3. Apply the [`PromotionKind`].
///
/// Returns `None` for an empty operand list.
///
/// **Not covered**: complex types, and PyTorch's `COMPLEX_TO_FLOAT` kind. No
/// operator in the CANNBench set needs them. This is a reimplementation from the
/// specified semantics; a differential test against a real torch build is the
/// natural next check and is not possible on a box without torch.
pub fn elementwise_dtypes(operands: &[Operand], kind: PromotionKind) -> Option<Promoted> {
    if operands.is_empty() {
        return None;
    }

    let highest = operands
        .iter()
        .map(|o| o.ty.category())
        .max()
        .expect("non-empty");

    let fold = |want: OperandKind| -> Option<ScalarType> {
        operands
            .iter()
            .filter(|o| o.kind == want && o.ty.category() == highest)
            .map(|o| o.ty)
            .reduce(promote_types)
    };

    let mut result = fold(OperandKind::Dimensioned)
        .or_else(|| fold(OperandKind::ZeroDim))
        .unwrap_or(match highest {
            // The category came from a bare scalar: use the category default.
            TypeCategory::Bool => ScalarType::Bool,
            TypeCategory::Integral => ScalarType::I64,
            TypeCategory::Floating => ScalarType::F32,
        });

    let mut computation = result.opmath();

    match kind {
        PromotionKind::Default => {}
        PromotionKind::NoOpMath => computation = result,
        PromotionKind::IntToFloat => {
            if !result.is_floating() {
                result = ScalarType::F32;
                computation = ScalarType::F32;
            }
        }
        PromotionKind::AlwaysBool => result = ScalarType::Bool,
        PromotionKind::BoolToLong => {
            if result == ScalarType::Bool {
                result = ScalarType::I64;
                computation = ScalarType::I64;
            }
        }
    }

    Some(Promoted {
        computation,
        result,
    })
}

/// Where the operated-on axis sits once rank is collapsed away.
///
/// Two classes, not N: either the axis is the innermost one — so each slice is
/// a contiguous run and a DMA can take it whole — or it is not, and the walk is
/// strided. Everything else about rank is arithmetic the host already did.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AxisClass {
    /// `INNER == 1`: the axis is innermost, each run of `R` is contiguous.
    LastContiguous,
    /// `INNER > 1`: elements along the axis are `INNER` apart.
    StridedOuter,
}

impl AxisClass {
    pub fn name(self) -> &'static str {
        match self {
            AxisClass::LastContiguous => "last_contiguous",
            AxisClass::StridedOuter => "strided_outer",
        }
    }
}

/// An N-D operation over one axis, flattened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Collapsed {
    /// Product of the dimensions before the axis.
    pub outer: usize,
    /// Extent of the axis itself.
    pub r: usize,
    /// Product of the dimensions after the axis.
    pub inner: usize,
    pub axis: AxisClass,
}

impl Collapsed {
    /// Total elements — invariant under collapse.
    pub fn numel(&self) -> usize {
        self.outer * self.r * self.inner
    }
}

/// Collapse a shape and an axis to `(OUTER, R, INNER)`.
///
/// `dim` follows PyTorch: negative counts from the end. Returns `None` if `dim`
/// is out of range for the shape's rank, or if the shape is empty.
///
/// This is the whole of the rank-handling story. `[363, 367, 373]` over dim 1
/// and `[2, 7, 256, 251]` over dim 1 produce the same kernel; only the three
/// runtime numbers differ.
pub fn collapse(shape: &[usize], dim: isize) -> Option<Collapsed> {
    let ndim = shape.len();
    if ndim == 0 {
        return None;
    }
    let d = if dim < 0 {
        let from_end = dim.checked_neg()?;
        let from_end: usize = usize::try_from(from_end).ok()?;
        ndim.checked_sub(from_end)?
    } else {
        usize::try_from(dim).ok()?
    };
    if d >= ndim {
        return None;
    }

    let outer: usize = shape[..d].iter().product();
    let r = shape[d];
    let inner: usize = shape[d + 1..].iter().product();

    Some(Collapsed {
        outer,
        r,
        inner,
        axis: if inner == 1 {
            AxisClass::LastContiguous
        } else {
            AxisClass::StridedOuter
        },
    })
}

/// One monomorphised kernel: the key the dispatch table is indexed by.
///
/// Note what is *absent* — any extent. That absence is the decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstanceKey {
    pub dtype: ScalarType,
    pub axis: AxisClass,
    /// Elements processed per block per round. Compile-time, so the UB budget,
    /// the 32/512-byte alignment and the 8-bit repeat chunking are all
    /// derivable rather than hoped for.
    pub tile: usize,
}

impl InstanceKey {
    /// Symbol suffix for the emitted kernel, e.g. `float32_last_contiguous_8192`.
    pub fn symbol_suffix(&self) -> String {
        format!("{}_{}_{}", self.dtype.name(), self.axis.name(), self.tile)
    }

    /// Bytes of Unified Buffer one tile of this instance occupies, given how
    /// many live tile buffers the schema needs at once.
    ///
    /// Sized in the *computation* type, which is the point of keeping that
    /// separate: an f16 kernel accumulating in f32 needs 4 bytes a lane, not 2.
    pub fn ub_bytes(&self, computation: ScalarType, live_buffers: usize) -> usize {
        self.tile * computation.bytes() * live_buffers
    }
}

/// A pointwise (or reduce-then-pointwise) operator, described once.
///
/// The schema is what gets written per operator. Everything downstream — the
/// instance set, the promotion, the tail handling — is derived from it, which is
/// the "one compiler change, many kernels" lever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointwiseSchema {
    pub name: String,
    /// Element types the operator's inputs may take.
    pub inputs: Vec<OperandKind>,
    pub promotion: PromotionKind,
    /// Live tile buffers the kernel body needs at once — inputs, outputs and
    /// any scratch. Drives the UB budget.
    pub live_buffers: usize,
}

impl PointwiseSchema {
    pub fn new(
        name: impl Into<String>,
        inputs: Vec<OperandKind>,
        promotion: PromotionKind,
        live_buffers: usize,
    ) -> PointwiseSchema {
        PointwiseSchema {
            name: name.into(),
            inputs,
            promotion,
            live_buffers,
        }
    }

    /// Number of inputs.
    pub fn arity(&self) -> usize {
        self.inputs.len()
    }

    /// The bounded instance set: the cross product of dtypes, axis classes and
    /// tiles. Sorted, so the emitted symbol order is stable across builds.
    ///
    /// Shape is not an axis here, and that is the whole point: the CANNBench set
    /// carries 936 distinct input shapes, so a shape-keyed instance set could
    /// neither be enumerated nor cover the hidden half of the evaluation.
    pub fn instances(&self, dtypes: &[ScalarType], tiles: &[usize]) -> Vec<InstanceKey> {
        let mut out = Vec::with_capacity(dtypes.len() * tiles.len() * 2);
        for &dtype in dtypes {
            for &axis in &[AxisClass::LastContiguous, AxisClass::StridedOuter] {
                for &tile in tiles {
                    out.push(InstanceKey { dtype, axis, tile });
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Pick the instance for a concrete call. Returns `None` when the dtype was
    /// not among those the schema was instantiated for — a missing instance is a
    /// reported gap, never a silent fallback.
    pub fn select(
        &self,
        instances: &[InstanceKey],
        dtype: ScalarType,
        axis: AxisClass,
    ) -> Option<InstanceKey> {
        instances
            .iter()
            .filter(|i| i.dtype == dtype && i.axis == axis)
            .max_by_key(|i| i.tile)
            .copied()
    }
}

/// How a runtime extent is covered by a compile-time tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TilePlan {
    /// Whole tiles.
    pub full: usize,
    /// Leftover elements in the last, partial tile. `0` when the extent divides.
    pub tail: usize,
    /// Blocks to launch — never more than the core count, so each block loops.
    pub blocks: usize,
    /// Tiles a single block walks, worst case.
    pub rounds_per_block: usize,
}

/// Plan a runtime extent against a compile-time tile and a physical core count.
///
/// The grid is capped at `cores` and each block loops over its share, which is
/// A4 and A9 together: a 1-D grid equal to the physical core count, never one
/// block per tile. `[1048583]` at tile 8192 is 129 tiles over 48 cores — three
/// rounds, not a 129-block launch.
///
/// Returns `None` for a zero tile or zero core count.
pub fn plan_tiles(extent: usize, tile: usize, cores: usize) -> Option<TilePlan> {
    if tile == 0 || cores == 0 {
        return None;
    }
    let full = extent / tile;
    let tail = extent % tile;
    let total_tiles = full + usize::from(tail > 0);
    let blocks = total_tiles.min(cores).max(1);
    let rounds_per_block = total_tiles.div_ceil(blocks);
    Some(TilePlan {
        full,
        tail,
        blocks,
        rounds_per_block,
    })
}

/// One member of a tensor list, placed on the shared grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ListMember {
    /// Elements this member holds.
    pub extent: usize,
    /// First block of the contiguous run this member owns.
    pub first_block: usize,
    /// Blocks in that run. Never zero: a member with no block never runs.
    pub blocks: usize,
}

/// A tensor list planned onto ONE grid.
///
/// A `foreach` operator exists because the list is issued together, so issuing
/// it as `k` launches gives the cost back. Each member owns a CONTIGUOUS run of
/// blocks, so a block finds its member by scanning runs — no indirection, no
/// device-side prefix sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListPlan {
    pub members: Vec<ListMember>,
    /// Blocks the launch asks for: the sum of the runs.
    pub blocks: usize,
}

impl ListPlan {
    /// Is batching this list worth a descriptor at all?
    ///
    /// A list of one has nothing to batch and would pay the argument buffer for
    /// nothing. Measured on 910B2: routing `k == 1` back to the plain launch
    /// moved ForeachAddcdivScalar from 72.76 to 76.17.
    pub fn worth_batching(&self) -> bool {
        self.members.len() > 1
    }
}

/// Plan a whole tensor list onto one grid, giving each member cores in
/// proportion to its work.
///
/// `tile` caps a member's run at the number of tiles it actually has, so a
/// short member does not hold blocks it cannot fill. Every member still gets at
/// least one block — a member with none would silently not be computed, which
/// is the failure mode a proportional split invites.
///
/// Returns `None` for a zero tile or core count, or an empty list.
pub fn plan_list(extents: &[usize], tile: usize, cores: usize) -> Option<ListPlan> {
    if tile == 0 || cores == 0 || extents.is_empty() {
        return None;
    }
    let total: usize = extents.iter().sum();
    let mut members = Vec::with_capacity(extents.len());
    let mut used = 0usize;
    for &extent in extents {
        let tiles = extent.div_ceil(tile).max(1);
        // Proportional share, floored, then clamped into [1, tiles]. Flooring
        // leaves the grid at or under `cores` for a list that fits.
        let share = if total == 0 { 0 } else { cores * extent / total };
        let blocks = share.clamp(1, tiles);
        members.push(ListMember { extent, first_block: used, blocks });
        used += blocks;
    }
    Some(ListPlan { members, blocks: used })
}

/// Split a list into the batches one launch may each describe.
///
/// Returns the member counts, in order. `width` comes from
/// `HardwareParams::list_batch_width`, so the descriptor stays inside the
/// argument bytes that ride free with a launch.
pub fn batch_list(len: usize, width: usize) -> Vec<usize> {
    if len == 0 || width == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(len.div_ceil(width));
    let mut left = len;
    while left > 0 {
        let take = left.min(width);
        out.push(take);
        left -= take;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::ScalarType::*;
    use super::*;

    // ---- the promotion lattice ----

    #[test]
    fn bool_yields_to_everything() {
        for t in [I8, U8, I16, I32, I64, F16, BF16, F32, F64] {
            assert_eq!(promote_types(Bool, t), t);
            assert_eq!(promote_types(t, Bool), t);
        }
        assert_eq!(promote_types(Bool, Bool), Bool);
    }

    #[test]
    fn floating_beats_integral_at_any_width() {
        // An i64 and an f16 promote to f16, not to something 64-bit wide:
        // category dominates width.
        assert_eq!(promote_types(I64, F16), F16);
        assert_eq!(promote_types(F16, I64), F16);
        assert_eq!(promote_types(I32, BF16), BF16);
        assert_eq!(promote_types(I8, F32), F32);
    }

    #[test]
    fn the_two_lattice_special_cases() {
        // Neither u8 nor i8 holds the other's range.
        assert_eq!(promote_types(U8, I8), I16);
        assert_eq!(promote_types(I8, U8), I16);
        // Same width, incomparable exponent/mantissa split.
        assert_eq!(promote_types(F16, BF16), F32);
        assert_eq!(promote_types(BF16, F16), F32);
    }

    #[test]
    fn integral_widths_take_the_wider_signed_type() {
        assert_eq!(promote_types(U8, I16), I16);
        assert_eq!(promote_types(U8, I32), I32);
        assert_eq!(promote_types(I8, I64), I64);
        assert_eq!(promote_types(I16, I32), I32);
        assert_eq!(promote_types(I32, I64), I64);
    }

    #[test]
    fn floating_widths_take_the_wider() {
        assert_eq!(promote_types(F16, F32), F32);
        assert_eq!(promote_types(BF16, F32), F32);
        assert_eq!(promote_types(F32, F64), F64);
        assert_eq!(promote_types(F16, F64), F64);
    }

    #[test]
    fn promotion_is_commutative_over_the_whole_set() {
        let all = [Bool, I8, U8, I16, I32, I64, F16, BF16, F32, F64];
        for a in all {
            for b in all {
                assert_eq!(
                    promote_types(a, b),
                    promote_types(b, a),
                    "promote({a}, {b}) is not symmetric"
                );
            }
        }
    }

    // ---- computation vs result ----

    #[test]
    fn half_formats_accumulate_in_f32_but_store_narrow() {
        // The rule that stops a long f16 reduction losing the tail of its sum.
        let p = elementwise_dtypes(
            &[Operand::tensor(F16), Operand::tensor(F16)],
            PromotionKind::Default,
        )
        .unwrap();
        assert_eq!(p.computation, F32);
        assert_eq!(p.result, F16);

        let p = elementwise_dtypes(&[Operand::tensor(BF16)], PromotionKind::Default).unwrap();
        assert_eq!(p.computation, F32);
        assert_eq!(p.result, BF16);
    }

    #[test]
    fn no_opmath_keeps_the_narrow_type_for_accumulation() {
        let p = elementwise_dtypes(&[Operand::tensor(F16)], PromotionKind::NoOpMath).unwrap();
        assert_eq!(p.computation, F16);
        assert_eq!(p.result, F16);
    }

    #[test]
    fn a_python_scalar_does_not_widen_a_half_tensor() {
        // The classic accuracy bug: `f16_tensor * 2.0` must stay f16, because the
        // scalar shares the tensor's category and so does not participate.
        let p = elementwise_dtypes(
            &[Operand::tensor(F16), Operand::scalar(F64)],
            PromotionKind::Default,
        )
        .unwrap();
        assert_eq!(p.result, F16);
        assert_eq!(p.computation, F32);
    }

    #[test]
    fn a_python_float_does_raise_an_integer_tensor() {
        // ... but it *does* raise the category, and then the default float type
        // is used rather than the scalar's own width.
        let p = elementwise_dtypes(
            &[Operand::tensor(I32), Operand::scalar(F64)],
            PromotionKind::Default,
        )
        .unwrap();
        assert_eq!(
            p.result, F32,
            "category came from a scalar: use the default"
        );
    }

    #[test]
    fn a_dimensioned_tensor_beats_a_zero_dim_one() {
        let p = elementwise_dtypes(
            &[Operand::tensor(F16), Operand::zero_dim(F32)],
            PromotionKind::Default,
        )
        .unwrap();
        assert_eq!(p.result, F16);

        // With no dimensioned operand in the winning category, the 0-dim one wins.
        let p = elementwise_dtypes(
            &[Operand::tensor(I32), Operand::zero_dim(F32)],
            PromotionKind::Default,
        )
        .unwrap();
        assert_eq!(p.result, F32);
    }

    #[test]
    fn int_to_float_lifts_integral_inputs() {
        let p = elementwise_dtypes(&[Operand::tensor(I32)], PromotionKind::IntToFloat).unwrap();
        assert_eq!(p.result, F32);
        assert_eq!(p.computation, F32);
        // A floating input is left where it is, and still gets opmath.
        let p = elementwise_dtypes(&[Operand::tensor(F16)], PromotionKind::IntToFloat).unwrap();
        assert_eq!(p.result, F16);
        assert_eq!(p.computation, F32);
    }

    #[test]
    fn always_bool_and_bool_to_long() {
        let p = elementwise_dtypes(
            &[Operand::tensor(F32), Operand::tensor(F32)],
            PromotionKind::AlwaysBool,
        )
        .unwrap();
        assert_eq!(p.result, Bool);
        assert_eq!(p.computation, F32, "the compare itself still runs in f32");

        let p = elementwise_dtypes(&[Operand::tensor(Bool)], PromotionKind::BoolToLong).unwrap();
        assert_eq!(p.result, I64);
    }

    #[test]
    fn no_operands_is_none() {
        assert!(elementwise_dtypes(&[], PromotionKind::Default).is_none());
    }

    // ---- rank collapse ----

    #[test]
    fn collapse_last_axis_is_contiguous() {
        let c = collapse(&[2048, 2048], -1).unwrap();
        assert_eq!((c.outer, c.r, c.inner), (2048, 2048, 1));
        assert_eq!(c.axis, AxisClass::LastContiguous);
    }

    #[test]
    fn collapse_outer_axis_is_strided() {
        let c = collapse(&[8192, 8192], 0).unwrap();
        assert_eq!((c.outer, c.r, c.inner), (1, 8192, 8192));
        assert_eq!(c.axis, AxisClass::StridedOuter);
    }

    #[test]
    fn adversarial_shapes_collapse_to_the_same_two_kernels() {
        // Real CANNBench ArgMax cases 10 and 13: prime, unaligned, different
        // ranks, different axes — one kernel each, chosen only by axis class.
        let a = collapse(&[363, 367, 373], 1).unwrap();
        assert_eq!((a.outer, a.r, a.inner), (363, 367, 373));
        assert_eq!(a.axis, AxisClass::StridedOuter);

        let b = collapse(&[2, 7, 256, 251], 1).unwrap();
        assert_eq!((b.outer, b.r, b.inner), (2, 7, 256 * 251));
        assert_eq!(b.axis, AxisClass::StridedOuter);

        // Case 15: 5-D, middle axis. Same class again.
        let c = collapse(&[11, 13, 17, 67, 67], 2).unwrap();
        assert_eq!(c.axis, AxisClass::StridedOuter);
        assert_eq!(c.numel(), 11 * 13 * 17 * 67 * 67);

        // Case 14: 1-D prime extent, last axis.
        let d = collapse(&[1_048_583], -1).unwrap();
        assert_eq!((d.outer, d.r, d.inner), (1, 1_048_583, 1));
        assert_eq!(d.axis, AxisClass::LastContiguous);
    }

    #[test]
    fn collapse_preserves_numel() {
        let shape = [3, 7, 13, 4001];
        for dim in 0..shape.len() {
            let c = collapse(&shape, dim as isize).unwrap();
            assert_eq!(c.numel(), shape.iter().product::<usize>());
        }
    }

    #[test]
    fn collapse_rejects_an_out_of_range_axis() {
        assert!(collapse(&[4, 4], 2).is_none());
        assert!(collapse(&[4, 4], -3).is_none());
        assert!(collapse(&[], 0).is_none());
        // -1 and -2 are in range for rank 2.
        assert!(collapse(&[4, 4], -1).is_some());
        assert!(collapse(&[4, 4], -2).is_some());
    }

    // ---- the instance set ----

    #[test]
    fn the_instance_set_is_small_and_shape_free() {
        let schema = PointwiseSchema::new(
            "add",
            vec![OperandKind::Dimensioned, OperandKind::Dimensioned],
            PromotionKind::Default,
            3,
        );
        // The five dtypes ArgMax's twenty cases actually use.
        let dtypes = [F16, BF16, F32, I32, I64];
        let insts = schema.instances(&dtypes, &[8192]);
        assert_eq!(insts.len(), 10, "5 dtypes x 2 axis classes x 1 tile");

        // Two tiles, the worst case in the decision record.
        let insts2 = schema.instances(&[F16, BF16, F32, I8, I32, I64, Bool], &[4096, 8192]);
        assert_eq!(insts2.len(), 28);
    }

    #[test]
    fn instances_are_sorted_and_deduped() {
        let schema = PointwiseSchema::new(
            "exp",
            vec![OperandKind::Dimensioned],
            PromotionKind::IntToFloat,
            2,
        );
        let insts = schema.instances(&[F32, F32, F16], &[1024, 1024]);
        assert_eq!(insts.len(), 4, "2 dtypes x 2 axis classes x 1 tile");
        let mut sorted = insts.clone();
        sorted.sort();
        assert_eq!(insts, sorted);
    }

    #[test]
    fn select_takes_the_largest_tile_and_reports_a_gap() {
        let schema = PointwiseSchema::new(
            "exp",
            vec![OperandKind::Dimensioned],
            PromotionKind::Default,
            2,
        );
        let insts = schema.instances(&[F32, F16], &[2048, 8192]);

        let picked = schema
            .select(&insts, F32, AxisClass::LastContiguous)
            .unwrap();
        assert_eq!(picked.tile, 8192);
        assert_eq!(picked.symbol_suffix(), "float32_last_contiguous_8192");

        // The dtype gap the decision record calls out: int64 was never
        // instantiated, and that must surface rather than silently fall back.
        assert!(schema
            .select(&insts, I64, AxisClass::LastContiguous)
            .is_none());
    }

    #[test]
    fn ub_bytes_sizes_in_the_computation_type() {
        let k = InstanceKey {
            dtype: F16,
            axis: AxisClass::LastContiguous,
            tile: 8192,
        };
        const UB: usize = 192 * 1024;
        // Stored as f16 but accumulated in f32: four bytes a lane, not two.
        // Sizing this in the *result* type would under-count by half and hand
        // bisheng an `ub overflow` that the budget check had already blessed.
        assert_eq!(k.ub_bytes(F32, 3), 8192 * 4 * 3);
        assert_eq!(k.ub_bytes(F16, 3), 8192 * 2 * 3);

        // At tile 8192 in f32 the UB divides exactly six ways. That is the
        // schema's whole budget: six live buffers fit to the byte, seven do not.
        assert!(k.ub_bytes(F32, 3) < UB);
        assert_eq!(k.ub_bytes(F32, 6), UB);
        assert!(k.ub_bytes(F32, 7) > UB);
    }

    #[test]
    fn the_i64_compare_gap_is_visible_in_the_type() {
        assert!(!I64.has_vector_compare());
        assert!(!I32.has_vector_compare());
        assert!(F32.has_vector_compare());
        assert!(F16.has_vector_compare());
    }

    // ---- tiling ----

    #[test]
    fn a_prime_extent_gets_a_tail_not_a_new_kernel() {
        // ArgMax case 14: 1048583 is prime.
        let p = plan_tiles(1_048_583, 8192, 48).unwrap();
        assert_eq!(p.full, 128);
        assert_eq!(p.tail, 1_048_583 - 128 * 8192);
        assert!(p.tail > 0);
        assert_eq!(p.blocks, 48, "grid is capped at the core count");
        assert_eq!(p.rounds_per_block, 3, "129 tiles over 48 cores");
    }

    #[test]
    fn a_divisible_extent_has_no_tail() {
        let p = plan_tiles(8192 * 96, 8192, 48).unwrap();
        assert_eq!(p.tail, 0);
        assert_eq!(p.full, 96);
        assert_eq!(p.blocks, 48);
        assert_eq!(p.rounds_per_block, 2);
    }

    #[test]
    fn a_tiny_extent_does_not_launch_more_blocks_than_tiles() {
        let p = plan_tiles(100, 8192, 48).unwrap();
        assert_eq!(p.full, 0);
        assert_eq!(p.tail, 100);
        assert_eq!(p.blocks, 1, "one tile, one block — not 48 idle ones");
        assert_eq!(p.rounds_per_block, 1);
    }

    #[test]
    fn plan_tiles_rejects_degenerate_inputs() {
        assert!(plan_tiles(1024, 0, 48).is_none());
        assert!(plan_tiles(1024, 8192, 0).is_none());
        // A zero extent still plans one empty block rather than dividing by zero.
        let p = plan_tiles(0, 8192, 48).unwrap();
        assert_eq!(p.blocks, 1);
        assert_eq!(p.full, 0);
        assert_eq!(p.tail, 0);
    }

    /// The twenty real ArgMax cases from `cann-bench/tasks/level2/arg_max/cases.csv`,
    /// as `(shape, dim, dtype)`. Twenty distinct shapes, five dtypes, ranks 1
    /// through 5, and every extent chosen to be prime or unaligned.
    const ARGMAX_CASES: &[(&[usize], isize, ScalarType)] = &[
        (&[1_048_576], -1, F16),
        (&[2048, 2048], -1, F32),
        (&[4096, 4096], -1, BF16),
        (&[8192, 8192], 0, I32),
        (&[4096, 8192], -1, I64),
        (&[8192, 8192], 0, F32),
        (&[1023, 1023], -1, F16),
        (&[1009, 1021], 0, F32),
        (&[1537, 769], -1, BF16),
        (&[363, 367, 373], 1, I32),
        (&[2049, 513], -1, F16),
        (&[3, 7, 13, 4001], -1, F32),
        (&[2, 7, 256, 251], 1, BF16),
        (&[1_048_583], -1, F32),
        (&[11, 13, 17, 67, 67], 2, F16),
        (&[3, 7, 11, 13, 1013], -1, I64),
        (&[512, 2049], -1, F32),
        (&[255, 8193], 0, BF16),
        (&[4097, 511], -1, I32),
        (&[2, 511, 2049], 1, F16),
    ];

    #[test]
    fn twenty_argmax_cases_need_nine_instances() {
        // The claim the whole O6 decision rests on, checked against the real
        // case matrix: twenty distinct shapes reduce to nine kernels, because
        // shape is not part of the key.
        let mut keys: Vec<(ScalarType, AxisClass)> = ARGMAX_CASES
            .iter()
            .map(|(shape, dim, dt)| {
                let c = collapse(shape, *dim).expect("every real case must collapse");
                assert_eq!(
                    c.numel(),
                    shape.iter().product::<usize>(),
                    "collapse must preserve numel for {shape:?}"
                );
                (*dt, c.axis)
            })
            .collect();

        // Nineteen distinct shapes over twenty cases: `[8192, 8192]` is reused
        // once, by cases 4 and 6, and even then at different dtypes. There is
        // effectively no shape reuse to exploit.
        let shapes: std::collections::BTreeSet<_> =
            ARGMAX_CASES.iter().map(|(s, _, _)| *s).collect();
        assert_eq!(shapes.len(), 19);

        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 9, "twenty shapes, nine instances: {keys:?}");

        // Both axis classes carry real weight — this is not a lopsided split.
        let strided = keys
            .iter()
            .filter(|(_, a)| *a == AxisClass::StridedOuter)
            .count();
        assert_eq!(strided, 4);
        assert_eq!(keys.len() - strided, 5);
    }

    #[test]
    fn the_emitted_argmax_covers_three_of_the_five_dtypes() {
        // What ships today is float/half/bf16. int32 and int64 appear in five of
        // the twenty cases and fall through to hand-written arms; int64 also has
        // no vector compare path, so it needs the hi/lo i32 plane split.
        let emitted = [F32, F16, BF16];
        let needed: std::collections::BTreeSet<ScalarType> =
            ARGMAX_CASES.iter().map(|(_, _, d)| *d).collect();
        assert_eq!(needed.len(), 5);

        let missing: Vec<_> = needed
            .iter()
            .filter(|d| !emitted.contains(d))
            .copied()
            .collect();
        assert_eq!(missing, vec![I32, I64]);

        let uncovered = ARGMAX_CASES
            .iter()
            .filter(|(_, _, d)| missing.contains(d))
            .count();
        assert_eq!(uncovered, 5, "cases 4, 5, 10, 16 and 19");

        assert!(missing.iter().any(|d| !d.has_vector_compare()));
    }

    #[test]
    fn every_cannbench_dtype_name_round_trips() {
        for t in [Bool, I8, U8, I16, I32, I64, F16, BF16, F32, F64] {
            assert_eq!(ScalarType::from_name(t.name()), Some(t));
        }
        assert_eq!(ScalarType::from_name("half"), Some(F16));
        assert_eq!(ScalarType::from_name("complex64"), None);
    }

    // ---- planning a tensor list onto one grid ----

    #[test]
    fn every_member_gets_at_least_one_block() {
        // A member 10000x smaller than its neighbour still rounds to a
        // proportional share of zero. Zero blocks means it never runs, so the
        // clamp is correctness, not tidiness.
        let p = plan_list(&[1_000_000, 64], 8192, 48).unwrap();
        assert_eq!(p.members[1].blocks, 1);
        assert!(p.members.iter().all(|m| m.blocks >= 1));
    }

    #[test]
    fn members_own_contiguous_non_overlapping_block_runs() {
        // A block finds its member by scanning the runs, so they must tile the
        // grid exactly: any gap is a block that computes nothing, any overlap
        // is two members writing from one block.
        let p = plan_list(&[400_000, 200_000, 100_000], 8192, 48).unwrap();
        let mut next = 0;
        for m in &p.members {
            assert_eq!(m.first_block, next);
            next += m.blocks;
        }
        assert_eq!(next, p.blocks);
    }

    #[test]
    fn a_short_member_does_not_hold_blocks_it_cannot_fill() {
        // 64 elements at tile 8192 is one tile, so one block however many
        // cores the proportional share would hand it.
        let p = plan_list(&[64, 64], 8192, 48).unwrap();
        assert!(p.members.iter().all(|m| m.blocks == 1));
        assert_eq!(p.blocks, 2);
    }

    #[test]
    fn an_equal_split_stays_within_the_core_count() {
        let p = plan_list(&[1_000_000; 4], 8192, 48).unwrap();
        assert!(p.blocks <= 48, "grid {} exceeded 48 cores", p.blocks);
    }

    #[test]
    fn a_list_of_one_is_not_worth_a_descriptor() {
        assert!(!plan_list(&[1_000_000], 8192, 48).unwrap().worth_batching());
        assert!(plan_list(&[1_000_000, 1_000_000], 8192, 48).unwrap().worth_batching());
    }

    #[test]
    fn plan_list_refuses_the_degenerate_inputs() {
        assert!(plan_list(&[], 8192, 48).is_none());
        assert!(plan_list(&[1024], 0, 48).is_none());
        assert!(plan_list(&[1024], 8192, 0).is_none());
    }

    // ---- the descriptor budget ----

    #[test]
    fn batch_width_comes_from_the_free_argument_bytes() {
        let hw = crate::target::HardwareParams::ascend_910b();
        // 56 bytes a member -- four pointers, an extent, a tile and a run --
        // is four members inside the 256 that ride free.
        assert_eq!(hw.list_batch_width(56).unwrap(), 4);
        // Widening the descriptor is what made the batched launch lose.
        assert!(hw.list_batch_width(56).unwrap() < 16);
    }

    #[test]
    fn batch_width_refuses_an_unmeasured_architecture() {
        let hw = crate::target::HardwareParams::ascend_950_unmeasured();
        assert!(hw.list_batch_width(56).is_err());
    }

    #[test]
    fn a_member_too_large_to_describe_cannot_be_batched() {
        let hw = crate::target::HardwareParams::ascend_910b();
        let e = hw.list_batch_width(512).unwrap_err();
        assert!(e.contains("cannot be batched"), "{e}");
    }

    #[test]
    fn a_target_with_no_argument_bound_batches_the_whole_list() {
        let hw = crate::target::HardwareParams::default();
        assert_eq!(hw.list_batch_width(56).unwrap(), usize::MAX);
    }

    #[test]
    fn batch_list_covers_the_list_exactly() {
        assert_eq!(batch_list(9, 4), vec![4, 4, 1]);
        assert_eq!(batch_list(4, 4), vec![4]);
        assert_eq!(batch_list(1, 4), vec![1]);
        assert!(batch_list(0, 4).is_empty());
        assert_eq!(batch_list(9, 4).iter().sum::<usize>(), 9);
    }
}
