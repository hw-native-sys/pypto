// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Which PTO ops a refinement to the λ_tile calculus could cover.
//!
//! `proofs/lambda_tile/coq/audit/CrossLower.v` proves safety preservation for
//! any emitter satisfying two hypotheses and leaves discharging them for a
//! concrete backend as an explicit obligation. `sp_step_pres` requires
//! OP-CONSERVATIVITY -- no target step without a source counterpart -- so the
//! first question is which emitted ops have one.
//!
//! This module answers it as a DECISION on an artifact, not as documentation.
//! The classification used to live in a test; a test can say the table is
//! consistent but cannot tell a caller whether the file in front of it is
//! inside the fragment. `pto-diff --fragment` now can.
//!
//! Failing closed is the point: an op this module does not know is
//! [`OpClass::Unknown`], which puts the artifact outside the fragment and names
//! the op. A new op in the emitter must therefore be classified deliberately.

use std::collections::BTreeSet;

/// What a refinement to λ_tile could do with an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpClass {
    /// Changes allocation, buffer typestate or view structure, and has a named
    /// counterpart among the calculus's 18 term constructors.
    Structural,
    /// Writes a tile back. **No counterpart exists**: nothing in the calculus
    /// consumes a `TTile` into a buffer or memory, and no hazard family covers
    /// a write. Admitting these needs a calculus extension, not a mapping.
    Store,
    /// Consumes tiles, produces a tile of declared shape, touches no buffer
    /// state.
    ///
    /// **No op is in this class any more.** The 24 that were became
    /// `Structural` on 2026-11-01, when `tm_cmp1`/`tm_cmp2` were added to the
    /// calculus. The variant is kept for the same reason [`OpClass::Store`] is:
    /// it is the class for "a counterpart is planned and not yet mechanised",
    /// and the next op in that position should land here rather than in
    /// [`OpClass::OutOfFragment`], which means something different -- needs its
    /// own model, not a bigger calculus. `Assume::WithStoreAndCompute` still
    /// admits it, so the three-assumption report keeps working for whatever
    /// lands here next.
    Compute,
    /// Outside any refinement to THIS calculus -- the cross-core AIC/AIV FIFO
    /// protocol, the pipe's quantisation epilogue, and a gather whose offset
    /// UNIT is part-dependent. These need their own model, not a bigger λ_tile.
    OutOfFragment,
    /// Not classified. Fails closed and names itself.
    Unknown,
}

/// `(op, class, the counterpart or the reason)`.
///
/// Written out longhand so the mapping is reviewable rather than inferred from
/// a name pattern -- `pto.tmov` is structural and `pto.tmul` is compute, and
/// nothing about the spelling says so.
pub const OP_MAP: &[(&str, OpClass, &str)] = &[
    // --- structural: a counterpart among the 18 constructors ---------------
    ("pto.alloc_tile", OpClass::Structural, "tm_alloc"),
    ("pto.tload", OpClass::Structural, "tm_load (DMA issue: Rdy -> Pnd)"),
    ("pto.barrier", OpClass::Structural, "tm_sync (Pnd -> Rdy)"),
    ("pto.make_tensor_view", OpClass::Structural, "the view a partition is taken of"),
    ("pto.partition_view", OpClass::Structural, "tm_partition / tm_part"),
    ("pto.tmatmul", OpClass::Structural, "tm_mma"),
    ("pto.tcvt", OpClass::Structural, "tm_dequant, for the quant-format cases"),
    ("pto.tmov", OpClass::Structural, "a tile-to-tile copy; structural, not arithmetic"),
    // --- store: a counterpart since 2026-11-01 ------------------------------
    //
    // This row read `OpClass::Store, "nothing consumes a TTile into memory"`
    // and that was true of the calculus as mechanised. `tm_store` was added to
    // it (Syntax.v), with `T_Store`, `S_Store`/`S_Store1`/`S_Store2`, and the
    // two store-side hazards `Hz_store_uaf` / `Hz_store_pnd` -- all five
    // soundness lemmas re-proved and the seven headline theorems still Closed
    // under the global context. So the store now HAS a counterpart and this op
    // is structural.
    //
    // `OpClass::Store` is kept in the enum rather than deleted: it is the class
    // for "the calculus has no construct for this at all", and the next op in
    // that position should land in it rather than in `OutOfFragment`, which
    // means something different (needs its own model, not a bigger calculus).
    ("pto.tstore", OpClass::Structural, "tm_store (Rdy buffer + shape-matched tile -> Pnd)"),
    // --- compute: tm_cmp1 / tm_cmp2 since 2026-11-01 ------------------------
    //
    // These 24 rows read `OpClass::Compute, "elementwise"` and the like, with
    // the class documented as "one generic shape-preserving rule covers the
    // whole class". Two things were wrong with that and both are fixed here.
    //
    // The rule is now MECHANISED, not planned: `tm_cmp1` and `tm_cmp2` are
    // constructors of the calculus (Syntax.v), with `T_Cmp1`/`T_Cmp2`,
    // `S_Cmp1`/`S_Cmp1Arg`/`S_Cmp2`/`S_Cmp2Arg1`/`S_Cmp2Arg2`, and the three
    // compute type-confusion hazards `Hz_cmp1_tc`/`Hz_cmp2_tc1`/`Hz_cmp2_tc2`.
    // All five soundness lemmas are re-proved and the seven headline theorems
    // remain "Closed under the global context". So these ops have counterparts
    // and are structural.
    //
    // And it is not ONE rule, nor shape-preserving. "Shape-preserving" is false
    // for `ttrans` (RxC -> CxR), the four reductions and the two broadcasts --
    // more than a third of the class. What covers all 24 is one rule PER
    // ARITY, with the result shape DECLARED rather than derived; see the design
    // note on `tm_cmp1` in Syntax.v for what that does and does not claim.
    ("pto.tadd", OpClass::Structural, "tm_cmp2: elementwise"),
    ("pto.tsub", OpClass::Structural, "tm_cmp2: elementwise"),
    ("pto.tmul", OpClass::Structural, "tm_cmp2: elementwise"),
    ("pto.tdiv", OpClass::Structural, "tm_cmp2: elementwise"),
    ("pto.tmax", OpClass::Structural, "tm_cmp2: elementwise"),
    ("pto.tmaxs", OpClass::Structural, "tm_cmp1: elementwise, scalar operand"),
    ("pto.tmuls", OpClass::Structural, "tm_cmp1: elementwise, scalar operand"),
    ("pto.tadds", OpClass::Structural, "tm_cmp1: elementwise, scalar operand"),
    ("pto.tands", OpClass::Structural, "tm_cmp1: elementwise, scalar operand"),
    ("pto.texp", OpClass::Structural, "tm_cmp1: elementwise"),
    ("pto.tsqrt", OpClass::Structural, "tm_cmp1: elementwise"),
    ("pto.trecip", OpClass::Structural, "tm_cmp1: elementwise"),
    ("pto.tabs", OpClass::Structural, "tm_cmp1: elementwise"),
    ("pto.tshls", OpClass::Structural, "tm_cmp1: elementwise, scalar operand"),
    ("pto.tshrs", OpClass::Structural, "tm_cmp1: elementwise, scalar operand"),
    ("pto.ttrans", OpClass::Structural, "tm_cmp1: layout; shape-permuting but value-preserving"),
    ("pto.tcolsum", OpClass::Structural, "tm_cmp1: reduction"),
    ("pto.tcolexpand", OpClass::Structural, "tm_cmp1: broadcast"),
    ("pto.trowsum", OpClass::Structural, "tm_cmp1: reduction"),
    ("pto.trowmax", OpClass::Structural, "tm_cmp1: reduction"),
    ("pto.trowexpand", OpClass::Structural, "tm_cmp1: broadcast"),
    ("pto.trowexpandsub", OpClass::Structural, "tm_cmp2: fused broadcast-subtract"),
    ("pto.trowexpandmul", OpClass::Structural, "tm_cmp2: fused broadcast-multiply"),
    ("pto.trowexpanddiv", OpClass::Structural, "tm_cmp2: fused broadcast-divide"),
    // --- out of fragment ----------------------------------------------------
    ("pto.reserve_buffer", OpClass::OutOfFragment, "cross-core FIFO"),
    ("pto.import_reserved_buffer", OpClass::OutOfFragment, "cross-core FIFO, names a peer function"),
    ("pto.aic_initialize_pipe", OpClass::OutOfFragment, "cross-core pipe, cube end"),
    ("pto.aiv_initialize_pipe", OpClass::OutOfFragment, "cross-core pipe, vector end"),
    ("pto.tpush_to_aiv", OpClass::OutOfFragment, "transfer out of the cube"),
    ("pto.tpop_from_aic", OpClass::OutOfFragment, "transfer into the vector unit"),
    ("pto.tfree_from_aic", OpClass::OutOfFragment, "releases a FIFO slot"),
    ("pto.syncall", OpClass::OutOfFragment, "grid-wide, not the buffer barrier tm_sync models"),
    ("pto.set_quant_vector", OpClass::OutOfFragment, "the pipe's dequantisation epilogue"),
    ("pto.tgatherb", OpClass::OutOfFragment, "offset UNIT is part-dependent: 32B blocks on a2/a3, elements on a5"),
];

/// How much of the calculus to assume, so the same artifact can be asked about
/// the calculus as it stands and as it would be after each planned extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Assume {
    /// λ_tile as mechanised: structural ops only.
    AsItStands,
    /// Plus a store constructor and its hazards.
    WithStore,
    /// Plus one generic shape-preserving compute rule.
    WithStoreAndCompute,
}

impl Assume {
    fn admits(self, c: OpClass) -> bool {
        matches!(
            (self, c),
            (_, OpClass::Structural)
                | (
                    Assume::WithStore | Assume::WithStoreAndCompute,
                    OpClass::Store
                )
                | (Assume::WithStoreAndCompute, OpClass::Compute)
        )
    }
}

/// `Some(class, why)` for a known op, else `None`.
pub fn classify(op: &str) -> Option<(OpClass, &'static str)> {
    OP_MAP.iter().find(|(o, ..)| *o == op).map(|(_, c, w)| (*c, *w))
}

/// Every `pto.*` op at statement position, in order of first appearance.
pub fn ops_of(pto_src: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for line in pto_src.lines() {
        let t = line.trim_start();
        // `%x = pto.op ...` or a bare `pto.op ...`
        let t = match t.strip_prefix('%') {
            Some(r) => r.split_once(" = ").map(|(_, b)| b).unwrap_or(t),
            None => t,
        };
        let Some(rest) = t.strip_prefix("pto.") else { continue };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
            .collect();
        if name.is_empty() {
            continue;
        }
        let full = format!("pto.{name}");
        if seen.insert(full.clone()) {
            out.push(full);
        }
    }
    out
}

/// Why an artifact is outside the fragment: the op, its class, and the note.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Excluder {
    pub op: String,
    pub class: OpClass,
    pub why: &'static str,
}

/// The verdict for one artifact under one assumption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    pub in_fragment: bool,
    /// Every op that puts it outside, in order of first appearance. Not just
    /// the first: a caller deciding whether to invest in an extension needs to
    /// know whether one op or twelve are in the way, and "the first gate is not
    /// the reason" is this crate's most expensive recurring lesson.
    pub excluders: Vec<Excluder>,
}

/// Decide whether `pto_src` is inside the fragment under `assume`.
pub fn verdict(pto_src: &str, assume: Assume) -> Verdict {
    let mut excluders = Vec::new();
    for op in ops_of(pto_src) {
        match classify(&op) {
            Some((c, _)) if assume.admits(c) => {}
            Some((c, why)) => excluders.push(Excluder { op, class: c, why }),
            None => excluders.push(Excluder {
                op,
                class: OpClass::Unknown,
                why: "not in OP_MAP -- classify it before claiming anything about this artifact",
            }),
        }
    }
    Verdict {
        in_fragment: excluders.is_empty(),
        excluders,
    }
}
