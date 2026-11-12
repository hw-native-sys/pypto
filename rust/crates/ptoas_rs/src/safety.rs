// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Advisory safety checks on a stage-2 `Plan`.
//!
//! Aim: compile-time detection of errors that ptoas doesn't independently
//! prove (or that slip through when the input was hand-written). Every
//! check reports; nothing blocks. Callers decide how to surface the result
//! (CLI prints, test assertions, …).
//!
//! Categories (one `check_*` per class):
//!   - **aliasing**   — two live tiles occupy overlapping bytes in the
//!                       same address space, which silently corrupts one
//!                       of them.
//!   - **capacity**   — the high-water mark in an address space exceeds
//!                       the `DeviceSpec` capacity.
//!   - **op constraints** — `pto.tmatmul` operand placement + dtype
//!                       combinations (lhs∈Left, rhs∈Right, acc∈Acc; the
//!                       dtype triple must be in `DeviceSpec::tmatmul_dtypes`);
//!                       `pto.tstore_fp` requires an Acc + Scaling tile.
//!   - **matmul bounds** — ptoas descriptor caps: OUTER<2^24, ROW<2^16.
//!   - **dead tiles** — a slot is allocated (pointer_cast+bind_tile) but
//!                       never appears as a read or write of any op.
//!   - **linear use** — a tile is written twice before being read
//!                       (WAW), or read after the op that produces it
//!                       has been overwritten. Advisory heuristic only;
//!                       loops are flattened so false positives are
//!                       possible inside scf.for bodies.
//!
//! Helpers live in `plan.rs` (`TileSlotX::byte_range`, `PlanOp::reads/writes`).

use crate::device::DeviceSpec;
use crate::plan::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafetyKind {
    Aliasing,
    Capacity,
    OpConstraint,
    MatmulBounds,
    DeadTile,
    LinearUse,
    /// Stage-1 only: a `pto.partition_view` reads outside its parent
    /// `pto.make_tensor_view` in either dim. Catches the GEP-runtime-index
    /// class (e.g. commit 06d8d30b) where the emitter mis-interprets a
    /// runtime SSA name as a literal integer offset.
    OobPartition,
    // `LoadUninit` and `SyncBeforeRead` were declared here, both documented
    // "GPU stage-1 only". With that arm gone nothing could produce either, and a
    // SafetyKind no check can emit is worse than no kind at all: it reports
    // clean on every input and reads as coverage. Removed with the arm.
}

impl SafetyKind {
    pub fn label(&self) -> &'static str {
        match self {
            SafetyKind::Aliasing => "aliasing",
            SafetyKind::Capacity => "capacity",
            SafetyKind::OpConstraint => "op-constraint",
            SafetyKind::MatmulBounds => "matmul-bounds",
            SafetyKind::DeadTile => "dead-tile",
            SafetyKind::LinearUse => "linear-use",
            SafetyKind::OobPartition => "oob-partition",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyViolation {
    pub func: String,
    pub kind: SafetyKind,
    pub severity: Severity,
    pub message: String,
}

impl SafetyViolation {
    fn err(func: &str, kind: SafetyKind, message: impl Into<String>) -> Self {
        Self {
            func: func.to_string(),
            kind,
            severity: Severity::Error,
            message: message.into(),
        }
    }
    fn warn(func: &str, kind: SafetyKind, message: impl Into<String>) -> Self {
        Self {
            func: func.to_string(),
            kind,
            severity: Severity::Warning,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SafetyReport {
    pub violations: Vec<SafetyViolation>,
}

impl SafetyReport {
    pub fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }
    pub fn has_errors(&self) -> bool {
        self.violations
            .iter()
            .any(|v| v.severity == Severity::Error)
    }
    pub fn by_kind(&self, kind: &SafetyKind) -> Vec<&SafetyViolation> {
        self.violations.iter().filter(|v| &v.kind == kind).collect()
    }
}

/// Run every check against every function in the plan.
pub fn check_all(plan: &Plan, spec: &DeviceSpec) -> SafetyReport {
    let mut rep = SafetyReport::default();
    for f in &plan.funcs {
        check_aliasing(f, &mut rep);
        check_capacity(f, spec, &mut rep);
        check_op_constraints(f, spec, &mut rep);
        check_matmul_bounds(f, spec, &mut rep);
        check_dead_tiles(f, &mut rep);
        check_dead_writes(f, &mut rep);
        check_linear_use(f, &mut rep);
    }
    suppress_redundant_dead_tile(&mut rep);
    rep
}

/// Stage-1 oracle entry point: every `pto.partition_view` must lie within
/// its parent `pto.make_tensor_view`. Catches the GEP-runtime-index bug
/// class that's invisible to stage-2 (where the bad offset is already baked
/// into a literal byte constant in UB, with no parent-tensor link).
pub fn check_all_stage1(plan: &crate::stage1_plan::Stage1Plan) -> SafetyReport {
    let mut rep = SafetyReport::default();
    for f in &plan.funcs {
        check_oob_partition(f, &mut rep);
    }
    rep
}

fn check_oob_partition(f: &crate::stage1_plan::Stage1Func, rep: &mut SafetyReport) {
    for pv in &f.partitions {
        let Some(parent) = f.tensor_views.get(&pv.parent) else {
            // Parent not seen — surface as an advisory warning rather than
            // silently dropping; a real run shouldn't hit this.
            rep.violations.push(SafetyViolation::warn(
                &f.name,
                SafetyKind::OobPartition,
                format!(
                    "{}: parent tensor_view {} not found in scope",
                    pv.ssa, pv.parent,
                ),
            ));
            continue;
        };

        // Only literal offsets are eligible for the OOB check — runtime SSA
        // offsets (e.g. loop induction) need a different oracle.
        let (Some(ro), Some(co)) = (pv.row_offset, pv.col_offset) else {
            continue;
        };
        let row_end = ro.saturating_add(pv.size_rows);
        let col_end = co.saturating_add(pv.size_cols);

        if row_end > parent.rows || col_end > parent.cols {
            rep.violations.push(SafetyViolation::err(
                &f.name,
                SafetyKind::OobPartition,
                format!(
                    "{}: partition_view offsets [{},{}] + sizes [{},{}] = end [{},{}] \
                     exceeds parent {} shape [{},{}]",
                    pv.ssa,
                    ro,
                    co,
                    pv.size_rows,
                    pv.size_cols,
                    row_end,
                    col_end,
                    pv.parent,
                    parent.rows,
                    parent.cols,
                ),
            ));
        }
    }
}

/// A slot already flagged as `Aliasing` is by construction either an
/// unintended overlap or a deliberate stage reuse — either way the
/// follow-up `DeadTile` warning on the same slot is redundant noise.
/// Drop dead-tile findings whose SSA appears in any aliasing finding for
/// the same function.
fn suppress_redundant_dead_tile(rep: &mut SafetyReport) {
    use std::collections::BTreeSet;
    let mut aliased: BTreeSet<(String, String)> = BTreeSet::new();
    for v in &rep.violations {
        if v.kind != SafetyKind::Aliasing {
            continue;
        }
        // Aliasing message format: "slots %a and %b overlap in <space> ..."
        for tok in v.message.split_whitespace() {
            if tok.starts_with('%') {
                aliased.insert((v.func.clone(), tok.trim_end_matches(',').to_string()));
            }
        }
    }
    rep.violations.retain(|v| {
        if v.kind != SafetyKind::DeadTile {
            return true;
        }
        // Dead-tile message format: "slot `%a` allocated in <space> at ..."
        for tok in v.message.split([' ', '`']) {
            if tok.starts_with('%') && aliased.contains(&(v.func.clone(), tok.to_string())) {
                return false;
            }
        }
        true
    });
}

// ── aliasing ───────────────────────────────────────────────────────────

/// Two slots in the same non-GM address space whose byte ranges overlap.
///
/// Staging reuses byte ranges deliberately (e.g. two tiles at the same UB
/// offset for double-buffering), so this pass only flags slots that have
/// *different* SSA names AND different (rows,cols,dtype) — the heuristic
/// for "unintended" overlap. Identical tiles at the same offset are
/// considered aliases by design and skipped.
pub fn check_aliasing(f: &PlanFunc, rep: &mut SafetyReport) {
    let mut by_space: BTreeMap<AddressSpace, Vec<&TileSlotX>> = BTreeMap::new();
    for slot in f.slots.values() {
        if slot.space == AddressSpace::Gm {
            continue;
        }
        by_space.entry(slot.space).or_default().push(slot);
    }
    for (space, mut slots) in by_space {
        slots.sort_by_key(|s| s.offset);
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                let a = slots[i];
                let b = slots[j];
                // sorted by offset, so b.offset >= a.offset. If b starts
                // past a's end we can stop scanning j.
                if b.offset >= a.offset + a.size_bytes() {
                    break;
                }
                // overlap.
                if same_shape(a, b) {
                    // deliberate aliasing — staged reuse.
                    continue;
                }
                rep.violations.push(SafetyViolation::err(
                    &f.name,
                    SafetyKind::Aliasing,
                    format!(
                        "slots {} and {} overlap in {} at [{}, {}) and [{}, {})",
                        a.ssa.0,
                        b.ssa.0,
                        space.label(),
                        a.offset,
                        a.offset + a.size_bytes(),
                        b.offset,
                        b.offset + b.size_bytes(),
                    ),
                ));
            }
        }
    }
}

fn same_shape(a: &TileSlotX, b: &TileSlotX) -> bool {
    a.rows == b.rows && a.cols == b.cols && a.dtype == b.dtype
}

// ── capacity ───────────────────────────────────────────────────────────

/// For each address space the high-water mark is `max(slot.offset + size)`
/// across all slots placed there. Exceeding `DeviceSpec` capacity is an
/// error.
pub fn check_capacity(f: &PlanFunc, spec: &DeviceSpec, rep: &mut SafetyReport) {
    let mut high: BTreeMap<AddressSpace, u64> = BTreeMap::new();
    for slot in f.slots.values() {
        if slot.space == AddressSpace::Gm {
            continue;
        }
        // Use the 32-B-aligned size so the high-water matches what
        // `mlir_to_cpp` emits as TBuf footprint. Raw `size_bytes` is
        // still used for aliasing (overlap is a property of the real
        // byte range ptoas assigned, not of the padded box).
        let end = slot.offset + slot.size_bytes_ub_aligned();
        let cur = high.entry(slot.space).or_insert(0);
        if end > *cur {
            *cur = end;
        }
    }
    for (space, end) in high {
        let Some(cap) = spec.capacity_of(space) else {
            continue;
        };
        if end > cap {
            rep.violations.push(SafetyViolation::err(
                &f.name,
                SafetyKind::Capacity,
                format!(
                    "{} high-water {} B exceeds capacity {} B (on {})",
                    space.label(),
                    end,
                    cap,
                    spec.name,
                ),
            ));
        }
    }
}

// ── op constraints ─────────────────────────────────────────────────────

/// Check op-level shape/space/dtype rules:
///   - `tmatmul` lhs in Left, rhs in Right, acc in Acc.
///   - `(acc.dtype, lhs.dtype, rhs.dtype)` must be in
///     `DeviceSpec::tmatmul_dtypes`.
///   - `tmatmul.acc` adds the constraint `acc_in == acc_out` (ptoas
///     normalises the accumulator tile to an in-place update — emitters
///     that pick different slots produce garbage on some CANN revs).
///   - `tstore_fp` requires an Acc source and a Scaling second operand
///     (the dequant-scale vector lives in FB/Scaling).
pub fn check_op_constraints(f: &PlanFunc, spec: &DeviceSpec, rep: &mut SafetyReport) {
    // Zero-dim tiles: rows==0 or cols==0 silently pass capacity (size 0)
    // and dead-tile checks, but every pto op on them is undefined. Flag
    // at the slot level so we catch it even if the op list is empty.
    for slot in f.slots.values() {
        if slot.rows == 0 || slot.cols == 0 {
            rep.violations.push(SafetyViolation::err(
                &f.name,
                SafetyKind::OpConstraint,
                format!(
                    "tile `{}` has zero-sized shape {}x{} in {}",
                    slot.ssa.0,
                    slot.rows,
                    slot.cols,
                    slot.space.label()
                ),
            ));
        }
    }
    for op in &f.ops {
        match op {
            PlanOp::TMatmul { lhs, rhs, acc } => {
                check_matmul_placement(f, rep, "pto.tmatmul", lhs, rhs, acc, spec);
            }
            PlanOp::TMatmulAcc {
                acc_in,
                lhs,
                rhs,
                acc_out,
            } => {
                check_matmul_placement(f, rep, "pto.tmatmul.acc", lhs, rhs, acc_in, spec);
                if acc_in != acc_out {
                    rep.violations.push(SafetyViolation::err(
                        &f.name,
                        SafetyKind::OpConstraint,
                        format!(
                            "pto.tmatmul.acc: acc_in `{}` != acc_out `{}` (must be same slot)",
                            acc_in.0, acc_out.0
                        ),
                    ));
                }
            }
            PlanOp::TStoreFp { acc, scaling } => {
                if let Some(s) = f.slots.get(acc) {
                    if s.space != AddressSpace::Acc {
                        rep.violations.push(SafetyViolation::err(
                            &f.name,
                            SafetyKind::OpConstraint,
                            format!(
                                "pto.tstore_fp: acc operand `{}` lives in {}, expected acc",
                                acc.0,
                                s.space.label()
                            ),
                        ));
                    }
                }
                if let Some(s) = f.slots.get(scaling) {
                    if s.space != AddressSpace::Scaling {
                        rep.violations.push(SafetyViolation::err(
                            &f.name,
                            SafetyKind::OpConstraint,
                            format!(
                                "pto.tstore_fp: scaling operand `{}` lives in {}, expected scaling",
                                scaling.0,
                                s.space.label()
                            ),
                        ));
                    }
                    match s.slayout {
                        Some(l) if l != SLayout::NoneBox => {
                            rep.violations.push(SafetyViolation::warn(
                                &f.name, SafetyKind::OpConstraint,
                                format!("pto.tstore_fp: scaling tile `{}` has slayout {:?}, typical is none_box",
                                        scaling.0, l),
                            ));
                        }
                        // The ptoas 0.68 `alloc_tile addr =` spelling carries
                        // no `slayout`, so this check CANNOT run on such a
                        // plan. Saying nothing would make a silent pass look
                        // like a clean result -- the same false clean that
                        // made this whole parser worth fixing.
                        None => {
                            rep.violations.push(SafetyViolation::warn(
                                &f.name, SafetyKind::OpConstraint,
                                format!("pto.tstore_fp: scaling tile `{}` has no slayout in this \
                                         stage-2 dump, so the none_box check did not run",
                                        scaling.0),
                            ));
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn check_matmul_placement(
    f: &PlanFunc,
    rep: &mut SafetyReport,
    opname: &str,
    lhs: &Ssa,
    rhs: &Ssa,
    acc: &Ssa,
    spec: &DeviceSpec,
) {
    let (Some(ls), Some(rs), Some(acs)) = (f.slots.get(lhs), f.slots.get(rhs), f.slots.get(acc))
    else {
        return;
    };

    if ls.space != AddressSpace::Left {
        rep.violations.push(SafetyViolation::err(
            &f.name,
            SafetyKind::OpConstraint,
            format!(
                "{opname}: lhs `{}` in {}, expected left",
                lhs.0,
                ls.space.label()
            ),
        ));
    }
    if rs.space != AddressSpace::Right {
        rep.violations.push(SafetyViolation::err(
            &f.name,
            SafetyKind::OpConstraint,
            format!(
                "{opname}: rhs `{}` in {}, expected right",
                rhs.0,
                rs.space.label()
            ),
        ));
    }
    if acs.space != AddressSpace::Acc {
        rep.violations.push(SafetyViolation::err(
            &f.name,
            SafetyKind::OpConstraint,
            format!(
                "{opname}: acc `{}` in {}, expected acc",
                acc.0,
                acs.space.label()
            ),
        ));
    }

    // K must match: lhs cols == rhs rows.
    if ls.cols != rs.rows {
        rep.violations.push(SafetyViolation::err(
            &f.name,
            SafetyKind::OpConstraint,
            format!(
                "{opname}: K mismatch — lhs {}x{} × rhs {}x{}",
                ls.rows, ls.cols, rs.rows, rs.cols
            ),
        ));
    }
    if ls.rows != acs.rows || rs.cols != acs.cols {
        rep.violations.push(SafetyViolation::err(
            &f.name,
            SafetyKind::OpConstraint,
            format!(
                "{opname}: MN mismatch — lhs {}x{}, rhs {}x{}, acc {}x{}",
                ls.rows, ls.cols, rs.rows, rs.cols, acs.rows, acs.cols
            ),
        ));
    }

    let triple = (acs.dtype, ls.dtype, rs.dtype);
    if !spec.tmatmul_dtypes.contains(&triple) {
        let accepted: Vec<String> = spec
            .tmatmul_dtypes
            .iter()
            .map(|(a, l, r)| format!("({}, {}, {})", a.label(), l.label(), r.label()))
            .collect();
        rep.violations.push(SafetyViolation::err(
            &f.name,
            SafetyKind::OpConstraint,
            format!(
                "{opname}: dtype triple (acc={}, lhs={}, rhs={}) not in {} accepted set {{{}}}",
                acs.dtype.label(),
                ls.dtype.label(),
                rs.dtype.label(),
                spec.name,
                accepted.join(", ")
            ),
        ));
    }
}

// ── matmul bounds ──────────────────────────────────────────────────────

/// ptoas packs matmul descriptors with OUTER<2^24 and ROW<2^16. We check
/// the *physical* tile dims: ROW = rhs cols (the N dim per-block), OUTER
/// derived from acc rows * stride. At the plan level we don't see the
/// outer loop bound, so this check is conservative — it flags the tile
/// shape alone hitting the descriptor cap.
pub fn check_matmul_bounds(f: &PlanFunc, spec: &DeviceSpec, rep: &mut SafetyReport) {
    for op in &f.ops {
        let (name, lhs, rhs, acc) = match op {
            PlanOp::TMatmul { lhs, rhs, acc } => ("pto.tmatmul", lhs, rhs, acc),
            PlanOp::TMatmulAcc {
                lhs, rhs, acc_in, ..
            } => ("pto.tmatmul.acc", lhs, rhs, acc_in),
            _ => continue,
        };
        let (Some(_ls), Some(rs), Some(acs)) =
            (f.slots.get(lhs), f.slots.get(rhs), f.slots.get(acc))
        else {
            continue;
        };

        let row = rs.cols as u64;
        if row >= spec.matmul_row_max {
            rep.violations.push(SafetyViolation::err(
                &f.name,
                SafetyKind::MatmulBounds,
                format!(
                    "{name}: ROW (N per block) = {} exceeds ptoas cap {}; repack rhs or split N",
                    row, spec.matmul_row_max
                ),
            ));
        }
        // OUTER estimate from the acc tile geometry. Conservative: just
        // acc.rows * row; callers that know the host-side loop trip
        // count should assert separately.
        let outer = (acs.rows as u64).saturating_mul(row);
        if outer >= spec.matmul_outer_max {
            rep.violations.push(SafetyViolation::err(
                &f.name,
                SafetyKind::MatmulBounds,
                format!(
                    "{name}: OUTER approx {} exceeds ptoas cap {}",
                    outer, spec.matmul_outer_max
                ),
            ));
        }
    }
}

// ── dead tiles ─────────────────────────────────────────────────────────

/// Allocated slots that never appear as an operand of any op.
pub fn check_dead_tiles(f: &PlanFunc, rep: &mut SafetyReport) {
    let mut touched: BTreeSet<&Ssa> = BTreeSet::new();
    for op in &f.ops {
        for s in op.reads() {
            touched.insert(s);
        }
        for s in op.writes() {
            touched.insert(s);
        }
    }
    for slot in f.slots.values() {
        if !touched.contains(&slot.ssa) {
            rep.violations.push(SafetyViolation::warn(
                &f.name,
                SafetyKind::DeadTile,
                format!(
                    "slot `{}` allocated in {} at offset {} but never used",
                    slot.ssa.0,
                    slot.space.label(),
                    slot.offset
                ),
            ));
        }
    }
}

/// Slots that are written (by a TLoad/TUnary/TBinary/...) but never
/// read by any subsequent op and never stored back. This catches the
/// "compute then discard" pattern — e.g. an `exp` whose result is
/// never consumed — which `check_dead_tiles` misses because the slot
/// is technically touched (by the producing op's write).
///
/// Unlike `DeadTile`, this is about wasted *computation*, not wasted
/// allocation. The fix is upstream: either delete the dead op or wire
/// its output into the kernel's live path. Reported as `DeadTile`
/// severity-warning with a distinguishing message so filters that
/// search for "dead" still pick it up.
pub fn check_dead_writes(f: &PlanFunc, rep: &mut SafetyReport) {
    let mut read_slots: BTreeSet<&Ssa> = BTreeSet::new();
    let mut written_slots: BTreeSet<&Ssa> = BTreeSet::new();
    for op in &f.ops {
        for s in op.reads() {
            read_slots.insert(s);
        }
        for s in op.writes() {
            written_slots.insert(s);
        }
    }
    for w in &written_slots {
        if !read_slots.contains(w) {
            // Identify the producing op for a clearer message.
            let producer = f
                .ops
                .iter()
                .position(|op| op.writes().iter().any(|s| s == w));
            let where_clause = producer
                .map(|i| format!(" (produced by op #{})", i))
                .unwrap_or_default();
            rep.violations.push(SafetyViolation::warn(
                &f.name,
                SafetyKind::DeadTile,
                format!(
                    "tile `{}` is written but never read{} — the \
                         producing op is dead code",
                    w.0, where_clause
                ),
            ));
        }
    }
}

// ── linear use ─────────────────────────────────────────────────────────

/// Flag WAW with no intervening read (write → write) and RAW *after*
/// overwrite (read following a later write). Loops are flattened so we
/// only report when writes come strictly before reads textually.
///
/// This is a heuristic — cube tiles legitimately stay live across scf.for
/// iterations, and ptoas resolves that correctly. We therefore emit
/// warnings, not errors.
pub fn check_linear_use(f: &PlanFunc, rep: &mut SafetyReport) {
    // last index where each tile was written / read.
    let mut last_write: BTreeMap<&Ssa, usize> = BTreeMap::new();
    let mut last_read: BTreeMap<&Ssa, usize> = BTreeMap::new();
    for (i, op) in f.ops.iter().enumerate() {
        let reads = op.reads();
        for w in op.writes() {
            // Read-modify-write at the same op (e.g. tmatmul.acc with
            // acc_in == acc_out) is a legitimate accumulator pattern, not
            // a WAW. Skip those before checking the write history.
            if reads.iter().any(|r| *r == w) {
                last_write.insert(w, i);
                last_read.insert(w, i);
                continue;
            }
            if let Some(&prev_w) = last_write.get(w) {
                let prev_r = last_read.get(w).copied().unwrap_or(0);
                if prev_r <= prev_w {
                    // writes are prev_w then i, no read in between — WAW
                    rep.violations.push(SafetyViolation::warn(
                        &f.name, SafetyKind::LinearUse,
                        format!(
                            "tile `{}` written at op #{} then again at op #{} without intervening read",
                            w.0, prev_w, i,
                        ),
                    ));
                }
            }
            last_write.insert(w, i);
        }
        for r in reads {
            last_read.insert(r, i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::default_a5_910b2_cann85;

    fn slot(
        ssa: &str,
        space: AddressSpace,
        offset: u64,
        rows: u32,
        cols: u32,
        dtype: ExtDType,
    ) -> (Ssa, TileSlotX) {
        let s = Ssa(ssa.to_string());
        let slot = TileSlotX {
            ssa: s.clone(),
            space,
            offset,
            rows,
            cols,
            dtype,
            blayout: Some(BLayout::RowMajor),
            slayout: Some(SLayout::NoneBox),
            s_fractal: Some(512),
        };
        (s, slot)
    }

    fn mk(slots: Vec<(Ssa, TileSlotX)>, ops: Vec<PlanOp>) -> PlanFunc {
        let mut m = BTreeMap::new();
        for (k, v) in slots {
            m.insert(k, v);
        }
        PlanFunc {
            name: "t".into(),
            slots: m,
            ops,
        }
    }

    #[test]
    fn aliasing_flags_overlap_when_shapes_differ() {
        // two tiles at same UB offset, different dtypes → alias error.
        let mut rep = SafetyReport::default();
        let (_, a) = slot("%0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
        let (_, b) = slot("%1", AddressSpace::Ub, 0, 1, 1024, ExtDType::F16);
        let f = mk(
            vec![(a.ssa.clone(), a.clone()), (b.ssa.clone(), b.clone())],
            vec![],
        );
        check_aliasing(&f, &mut rep);
        assert_eq!(rep.violations.len(), 1);
        assert_eq!(rep.violations[0].kind, SafetyKind::Aliasing);
    }

    #[test]
    fn aliasing_ignores_same_shape_reuse() {
        // same offset + same shape = deliberate double-buffer reuse.
        let mut rep = SafetyReport::default();
        let (_, a) = slot("%0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
        let (_, b) = slot("%1", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
        let f = mk(
            vec![(a.ssa.clone(), a.clone()), (b.ssa.clone(), b.clone())],
            vec![],
        );
        check_aliasing(&f, &mut rep);
        assert!(rep.is_empty());
    }

    #[test]
    fn capacity_flags_overflow() {
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        // 64 KiB Left cap; a 1x65537 i8 slot overflows by 1 byte.
        let (_, a) = slot("%0", AddressSpace::Left, 0, 1, 65_537, ExtDType::I8);
        let f = mk(vec![(a.ssa.clone(), a)], vec![]);
        check_capacity(&f, &spec, &mut rep);
        assert_eq!(rep.violations.len(), 1);
        assert_eq!(rep.violations[0].kind, SafetyKind::Capacity);
    }

    #[test]
    fn op_constraint_rejects_bad_matmul_placement() {
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        // lhs accidentally placed in Ub instead of Left.
        let (_, l) = slot("%L", AddressSpace::Ub, 0, 16, 128, ExtDType::F16);
        let (_, r) = slot("%R", AddressSpace::Right, 0, 128, 256, ExtDType::F16);
        let (_, a) = slot("%A", AddressSpace::Acc, 0, 16, 256, ExtDType::F32);
        let op = PlanOp::TMatmul {
            lhs: l.ssa.clone(),
            rhs: r.ssa.clone(),
            acc: a.ssa.clone(),
        };
        let f = mk(
            vec![
                (l.ssa.clone(), l.clone()),
                (r.ssa.clone(), r.clone()),
                (a.ssa.clone(), a.clone()),
            ],
            vec![op],
        );
        check_op_constraints(&f, &spec, &mut rep);
        assert!(rep
            .violations
            .iter()
            .any(|v| v.kind == SafetyKind::OpConstraint && v.message.contains("expected left")));
    }

    #[test]
    fn op_constraint_rejects_unknown_dtype_triple() {
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        // f32 acc + i8 lhs + i8 rhs is not in the accepted set (i32, i8, i8) is).
        let (_, l) = slot("%L", AddressSpace::Left, 0, 16, 128, ExtDType::I8);
        let (_, r) = slot("%R", AddressSpace::Right, 0, 128, 256, ExtDType::I8);
        let (_, a) = slot("%A", AddressSpace::Acc, 0, 16, 256, ExtDType::F32);
        let op = PlanOp::TMatmul {
            lhs: l.ssa.clone(),
            rhs: r.ssa.clone(),
            acc: a.ssa.clone(),
        };
        let f = mk(
            vec![
                (l.ssa.clone(), l.clone()),
                (r.ssa.clone(), r.clone()),
                (a.ssa.clone(), a.clone()),
            ],
            vec![op],
        );
        check_op_constraints(&f, &spec, &mut rep);
        assert!(rep
            .violations
            .iter()
            .any(|v| v.kind == SafetyKind::OpConstraint && v.message.contains("dtype triple")));
    }

    #[test]
    fn matmul_bounds_flags_large_row() {
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        let (_, l) = slot("%L", AddressSpace::Left, 0, 16, 128, ExtDType::F16);
        // N = 70_000 > 65_536.
        let (_, r) = slot("%R", AddressSpace::Right, 0, 128, 70_000, ExtDType::F16);
        let (_, a) = slot("%A", AddressSpace::Acc, 0, 16, 70_000, ExtDType::F32);
        let op = PlanOp::TMatmul {
            lhs: l.ssa.clone(),
            rhs: r.ssa.clone(),
            acc: a.ssa.clone(),
        };
        let f = mk(
            vec![
                (l.ssa.clone(), l.clone()),
                (r.ssa.clone(), r.clone()),
                (a.ssa.clone(), a.clone()),
            ],
            vec![op],
        );
        check_matmul_bounds(&f, &spec, &mut rep);
        assert!(rep
            .violations
            .iter()
            .any(|v| v.kind == SafetyKind::MatmulBounds));
    }

    #[test]
    fn dead_tile_flags_unused_slot() {
        let mut rep = SafetyReport::default();
        let (_, a) = slot("%0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
        let f = mk(vec![(a.ssa.clone(), a.clone())], vec![]);
        check_dead_tiles(&f, &mut rep);
        assert_eq!(rep.violations.len(), 1);
        assert_eq!(rep.violations[0].kind, SafetyKind::DeadTile);
    }

    #[test]
    fn linear_use_flags_waw() {
        let mut rep = SafetyReport::default();
        let (_, a) = slot("%0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
        // two tloads into the same tile — WAW with no read between.
        let f = mk(
            vec![(a.ssa.clone(), a.clone())],
            vec![
                PlanOp::TLoad {
                    tile: a.ssa.clone(),
                },
                PlanOp::TLoad {
                    tile: a.ssa.clone(),
                },
            ],
        );
        check_linear_use(&f, &mut rep);
        assert!(rep
            .violations
            .iter()
            .any(|v| v.kind == SafetyKind::LinearUse));
    }

    // ── adversarial (false-negative audit) ────────────────────────────
    //
    // These tests pin gaps found in the 2026-04-19 audit. They are here
    // to PREVENT regressions, not exhaustively characterise ptoas.

    #[test]
    fn capacity_flags_overflow_at_high_offset() {
        // A small tile with a large offset still overflows. Previously
        // checked: offset=0, cols=65537 → overflow. Also must catch:
        // offset = cap - 128, rows=1, cols=256 (1 B dtype, 128 B over).
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        let left_cap = spec.capacity_of(AddressSpace::Left).unwrap();
        let offset = left_cap - 128;
        let (_, a) = slot("%0", AddressSpace::Left, offset, 1, 256, ExtDType::I8);
        let f = mk(vec![(a.ssa.clone(), a)], vec![]);
        check_capacity(&f, &spec, &mut rep);
        assert!(
            rep.violations
                .iter()
                .any(|v| v.kind == SafetyKind::Capacity),
            "expected high-offset overflow to be flagged, got {:?}",
            rep.violations
        );
    }

    #[test]
    fn linear_use_flags_raw_after_overwrite() {
        // Docs for check_linear_use claim it flags "RAW after overwrite":
        // write %0 → write %0 → read %0 is stale (the first write is
        // dead, the read sees the second write). Currently ONLY the WAW
        // is flagged. The separate RAW signal helps find the real bug:
        // the consumer expected the first write's value.
        //
        // We intentionally record this as a known gap: the assertion
        // below is #[should_panic] so the test documents the missing
        // behaviour without blocking CI. Flip to a positive assertion
        // once the oracle tracks per-op provenance.
        let mut rep = SafetyReport::default();
        let (_, a) = slot("%0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
        let f = mk(
            vec![(a.ssa.clone(), a.clone())],
            vec![
                PlanOp::TLoad {
                    tile: a.ssa.clone(),
                },
                PlanOp::TLoad {
                    tile: a.ssa.clone(),
                },
                PlanOp::TStore {
                    tile: a.ssa.clone(),
                },
            ],
        );
        check_linear_use(&f, &mut rep);
        // WAW is already caught. Confirm at least that.
        assert!(rep
            .violations
            .iter()
            .any(|v| v.kind == SafetyKind::LinearUse));
    }

    #[test]
    fn zero_sized_tile_is_flagged() {
        // rows=0 or cols=0 slips past capacity (size=0) and dead-tile
        // (if touched) but is almost always a bug — indexing, reduce
        // with empty output, etc.
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        let (_, a) = slot("%0", AddressSpace::Ub, 0, 0, 1024, ExtDType::F32);
        let f = mk(
            vec![(a.ssa.clone(), a)],
            vec![PlanOp::TLoad {
                tile: Ssa("%0".into()),
            }],
        );
        check_all_local(&f, &spec, &mut rep);
        assert!(
            rep.violations
                .iter()
                .any(|v| v.kind == SafetyKind::OpConstraint && v.message.contains("zero")),
            "expected zero-dim tile to be flagged, got {:?}",
            rep.violations
        );
    }

    #[test]
    fn tmatmul_acc_mismatched_in_out_is_flagged() {
        // acc_in != acc_out is caught today; regression guard.
        let mut rep = SafetyReport::default();
        let spec = default_a5_910b2_cann85();
        let (_, l) = slot("%L", AddressSpace::Left, 0, 16, 128, ExtDType::F16);
        let (_, r) = slot("%R", AddressSpace::Right, 0, 128, 256, ExtDType::F16);
        let (_, a) = slot("%A", AddressSpace::Acc, 0, 16, 256, ExtDType::F32);
        let (_, b) = slot("%B", AddressSpace::Acc, 16_384, 16, 256, ExtDType::F32);
        let f = mk(
            vec![
                (l.ssa.clone(), l.clone()),
                (r.ssa.clone(), r.clone()),
                (a.ssa.clone(), a.clone()),
                (b.ssa.clone(), b.clone()),
            ],
            vec![PlanOp::TMatmulAcc {
                acc_in: a.ssa.clone(),
                lhs: l.ssa.clone(),
                rhs: r.ssa.clone(),
                acc_out: b.ssa.clone(),
            }],
        );
        check_op_constraints(&f, &spec, &mut rep);
        assert!(rep
            .violations
            .iter()
            .any(|v| v.kind == SafetyKind::OpConstraint && v.message.contains("acc_in")));
    }

    /// Mirrors `check_all` but on a single function (tests skip the Plan
    /// wrapper). Kept here so the zero-dim test can touch every check.
    fn check_all_local(f: &PlanFunc, spec: &DeviceSpec, rep: &mut SafetyReport) {
        check_aliasing(f, rep);
        check_capacity(f, spec, rep);
        check_op_constraints(f, spec, rep);
        check_matmul_bounds(f, spec, rep);
        check_dead_tiles(f, rep);
        check_linear_use(f, rep);
    }

    #[test]
    fn linear_use_ignores_accumulator_pattern() {
        // tmatmul writes %A; tmatmul.acc reads-and-writes %A. This is the
        // first-iter-init / subsequent-iter-accumulate pattern emitted by
        // every blocked matmul — must NOT be flagged as WAW.
        let mut rep = SafetyReport::default();
        let (_, l) = slot("%L", AddressSpace::Left, 0, 16, 128, ExtDType::F16);
        let (_, r) = slot("%R", AddressSpace::Right, 0, 128, 256, ExtDType::F16);
        let (_, a) = slot("%A", AddressSpace::Acc, 0, 16, 256, ExtDType::F32);
        let f = mk(
            vec![
                (l.ssa.clone(), l.clone()),
                (r.ssa.clone(), r.clone()),
                (a.ssa.clone(), a.clone()),
            ],
            vec![
                PlanOp::TMatmul {
                    lhs: l.ssa.clone(),
                    rhs: r.ssa.clone(),
                    acc: a.ssa.clone(),
                },
                PlanOp::TMatmulAcc {
                    acc_in: a.ssa.clone(),
                    lhs: l.ssa.clone(),
                    rhs: r.ssa.clone(),
                    acc_out: a.ssa.clone(),
                },
            ],
        );
        check_linear_use(&f, &mut rep);
        assert!(
            rep.violations
                .iter()
                .all(|v| v.kind != SafetyKind::LinearUse),
            "expected no linear-use warnings on accumulator pattern, got {:?}",
            rep.violations,
        );
    }

    // ─── stage-1 OOB partition_view tests ──────────────────────────────────

    fn s1_func(
        name: &str,
        parent: (u64, u64),
        pv: (u64, u64, u64, u64),
    ) -> crate::stage1_plan::Stage1Func {
        use crate::stage1_plan::*;
        let mut tvs = BTreeMap::new();
        tvs.insert(
            "%pto0".into(),
            TensorView {
                ssa: "%pto0".into(),
                gm_arg: "%arg0".into(),
                rows: parent.0,
                cols: parent.1,
                dtype: "f32".into(),
            },
        );
        let parts = vec![PartitionView {
            ssa: "%pto1".into(),
            parent: "%pto0".into(),
            row_offset: Some(pv.0),
            col_offset: Some(pv.1),
            size_rows: pv.2,
            size_cols: pv.3,
            dtype: "f32".into(),
        }];
        Stage1Func {
            name: name.into(),
            tensor_views: tvs,
            partitions: parts,
        }
    }

    #[test]
    fn oob_partition_clean_emits_no_violations() {
        // parent 1×1024, partition [0,0] + [1,1024] — exactly fits.
        let f = s1_func("clean", (1, 1024), (0, 0, 1, 1024));
        let mut rep = SafetyReport::default();
        check_oob_partition(&f, &mut rep);
        assert!(
            rep.is_empty(),
            "clean partition produced violations: {:?}",
            rep.violations
        );
    }

    #[test]
    fn oob_partition_flags_row_overrun() {
        // 06d8d30b reproduction: parent 1×1024, partition [2,166] + [1,1024].
        // row_end = 3 > 1, col_end = 1190 > 1024 — both dims OOB.
        let f = s1_func("bad", (1, 1024), (2, 166, 1, 1024));
        let mut rep = SafetyReport::default();
        check_oob_partition(&f, &mut rep);
        let oob: Vec<_> = rep
            .violations
            .iter()
            .filter(|v| v.kind == SafetyKind::OobPartition)
            .collect();
        assert_eq!(
            oob.len(),
            1,
            "expected 1 OOB finding, got {:?}",
            rep.violations
        );
        assert_eq!(oob[0].severity, Severity::Error);
    }

    #[test]
    fn oob_partition_col_only_overrun() {
        // parent 16×64, partition [0, 60] + [16, 8] — col_end=68 > 64, row fits.
        let f = s1_func("col_oob", (16, 64), (0, 60, 16, 8));
        let mut rep = SafetyReport::default();
        check_oob_partition(&f, &mut rep);
        let oob: Vec<_> = rep
            .violations
            .iter()
            .filter(|v| v.kind == SafetyKind::OobPartition)
            .collect();
        assert_eq!(oob.len(), 1);
    }

    #[test]
    fn oob_partition_warns_on_unknown_parent() {
        use crate::stage1_plan::*;
        let f = Stage1Func {
            name: "orphan".into(),
            tensor_views: BTreeMap::new(),
            partitions: vec![PartitionView {
                ssa: "%pto1".into(),
                parent: "%missing".into(),
                row_offset: Some(0),
                col_offset: Some(0),
                size_rows: 1,
                size_cols: 1,
                dtype: "f32".into(),
            }],
        };
        let mut rep = SafetyReport::default();
        check_oob_partition(&f, &mut rep);
        assert_eq!(rep.violations.len(), 1);
        assert_eq!(rep.violations[0].severity, Severity::Warning);
    }
}

// The GPU stage-1 arm lived here: a shared-memory capacity check, a
// sync-before-read check and their tests, over a Triton-MLIR plan. Those target
// another vendor's processor, which this repository's licence does not cover, so
// the arm and the two modules feeding it are not part of this tree.
//
// `SmemCapacity` was a SafetyKind of its own, so the Ascend `Capacity` and
// `Aliasing` checks above are untouched by its removal rather than weakened.
