// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Data hazards between vector ops sharing Unified Buffer.
//!
//! This is the check that was missing when the AIV-only build turned five
//! CANNBench operators to garbage. Every constraint in `target.rs` is a
//! RESOURCE constraint -- does the tile fit UB, does the repeat count fit an
//! 8-bit field, does the stride fit 16 bits. All of them ask "is this op
//! legal in isolation". None of them asks whether two ops may run at once,
//! so a kernel could be emitted whose second op read a buffer the first had
//! not finished writing, and nothing in the compiler objected.
//!
//! The hardware did not object either, most of the time. On a MIX-scheduled
//! 910B the races happened to resolve; compiled AIV-only they did not, and
//! `exp` returned inf where it had returned the right answer for months. The
//! bug was always there. Only the schedule changed.
//!
//! ALIASING is the part an adjacency rule cannot see. `ReinterpretCast` gives
//! one piece of storage two names and two element types -- an int32 buffer
//! read as float to reach `Abs` -- so `f0` and `w0` are the same 32 bytes
//! under different spellings. A hazard between them is invisible to anything
//! comparing operand NAMES.

use std::collections::HashMap;

/// Storage a vector op touches: an index into the kernel's buffer table.
///
/// Two `LocalTensor`s that alias -- one a `ReinterpretCast` of the other --
/// MUST resolve to the same `BufId`, which is the whole reason this is an id
/// and not a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufId(pub usize);

/// One vector instruction: what it writes, what it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecOp {
    /// Source-level name, for the diagnostic.
    pub name: String,
    pub dst: BufId,
    pub srcs: Vec<BufId>,
}

impl VecOp {
    pub fn new(name: impl Into<String>, dst: BufId, srcs: &[BufId]) -> VecOp {
        VecOp { name: name.into(), dst, srcs: srcs.to_vec() }
    }
}

/// Why two ops may not overlap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hazard {
    /// Read-after-write: the consumer reads what the producer writes.
    Raw,
    /// Write-after-read: the writer clobbers a buffer still being read. This
    /// is the one that bites when a scratch buffer is REUSED, which is
    /// exactly what a tight UB budget forces a kernel to do.
    War,
    /// Write-after-write: two writers to one buffer, order not guaranteed.
    Waw,
}

impl Hazard {
    pub fn name(self) -> &'static str {
        match self {
            Hazard::Raw => "RAW",
            Hazard::War => "WAR",
            Hazard::Waw => "WAW",
        }
    }
}

/// A hazard between op `from` and a later op `to`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub kind: Hazard,
    pub buf: BufId,
}

/// Every hazard in a straight-line block of vector ops.
///
/// Only ADJACENT-in-time pairs matter for barrier placement -- a barrier
/// before `to` orders everything before it -- but the full set is what makes
/// the diagnostic readable, so it is returned whole.
pub fn hazards(ops: &[VecOp]) -> Vec<Edge> {
    let mut out = Vec::new();
    for (j, op) in ops.iter().enumerate() {
        for (i, prev) in ops.iter().enumerate().take(j) {
            // RAW: prev writes what op reads.
            if op.srcs.contains(&prev.dst) {
                out.push(Edge { from: i, to: j, kind: Hazard::Raw, buf: prev.dst });
            }
            // WAR: op writes what prev reads.
            if prev.srcs.contains(&op.dst) {
                out.push(Edge { from: i, to: j, kind: Hazard::War, buf: op.dst });
            }
            // WAW: both write the same buffer.
            if prev.dst == op.dst {
                out.push(Edge { from: i, to: j, kind: Hazard::Waw, buf: op.dst });
            }
        }
    }
    out
}

/// Indices that must be preceded by a pipe barrier.
///
/// An op needs one when it hazards against ANY op before it that is not
/// already separated by a barrier. Walking forward and clearing the frontier
/// at each barrier is what keeps this from emitting one before every
/// instruction: a barrier is a full ordering point, so what precedes it needs
/// no second sync.
pub fn barrier_points(ops: &[VecOp]) -> Vec<usize> {
    let mut points = Vec::new();
    // Ops issued since the last barrier -- the only ones that can still race.
    let mut open: Vec<usize> = Vec::new();
    for (j, op) in ops.iter().enumerate() {
        let races = open.iter().any(|&i| {
            let prev = &ops[i];
            op.srcs.contains(&prev.dst)      // RAW
                || prev.srcs.contains(&op.dst) // WAR
                || prev.dst == op.dst          // WAW
        });
        if races {
            points.push(j);
            open.clear();
        }
        open.push(j);
    }
    points
}

/// Check a block that ALREADY has barriers, and report what is still unsafe.
///
/// `barriers` holds the indices a barrier sits before, as the emitter placed
/// them. This is the form a lint wants: hand it what was written and get back
/// the races that survived.
pub fn unsynced(ops: &[VecOp], barriers: &[usize]) -> Vec<Edge> {
    let mut out = Vec::new();
    let mut open: Vec<usize> = Vec::new();
    for (j, op) in ops.iter().enumerate() {
        if barriers.contains(&j) {
            open.clear();
        }
        for &i in &open {
            let prev = &ops[i];
            if op.srcs.contains(&prev.dst) {
                out.push(Edge { from: i, to: j, kind: Hazard::Raw, buf: prev.dst });
            }
            if prev.srcs.contains(&op.dst) {
                out.push(Edge { from: i, to: j, kind: Hazard::War, buf: op.dst });
            }
            if prev.dst == op.dst {
                out.push(Edge { from: i, to: j, kind: Hazard::Waw, buf: op.dst });
            }
        }
        open.push(j);
    }
    out
}

/// Human-readable account of what is unsynchronised, for a compile error.
pub fn report(ops: &[VecOp], edges: &[Edge]) -> String {
    if edges.is_empty() {
        return "no unsynchronised hazards".to_string();
    }
    let mut s = format!("{} unsynchronised hazard(s):\n", edges.len());
    for e in edges {
        s.push_str(&format!(
            "  {} on buffer {} : op {} `{}` -> op {} `{}` (no barrier between)\n",
            e.kind.name(), e.buf.0, e.from, ops[e.from].name, e.to, ops[e.to].name));
    }
    s
}

/// Resolve aliased tensor names to shared storage.
///
/// `alias("f0", "w0")` records that a ReinterpretCast made them one buffer.
/// Without this the hazard between a write through `w0` and a read through
/// `f0` is invisible, which is precisely the shape of the bug in CANNBench's
/// int64 `maximum` helper.
#[derive(Default, Debug)]
pub struct Buffers {
    ids: HashMap<String, BufId>,
    next: usize,
}

impl Buffers {
    pub fn new() -> Buffers {
        Buffers::default()
    }

    /// Name a fresh buffer, or return the id a known name already has.
    pub fn get(&mut self, name: &str) -> BufId {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = BufId(self.next);
        self.next += 1;
        self.ids.insert(name.to_string(), id);
        id
    }

    /// Record that `name` is another view of `of` -- same storage, and so the
    /// same id.
    pub fn alias(&mut self, name: &str, of: &str) -> BufId {
        let id = self.get(of);
        self.ids.insert(name.to_string(), id);
        id
    }

    pub fn len(&self) -> usize {
        self.next
    }

    pub fn is_empty(&self) -> bool {
        self.next == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// exp's whole body, the bug of NOTES 117: three ops chained in place.
    fn exp_body() -> (Buffers, Vec<VecOp>) {
        let mut b = Buffers::new();
        let (o, i0) = (b.get("o"), b.get("i0"));
        let ops = vec![
            VecOp::new("Muls", o, &[i0]),
            VecOp::new("Adds", o, &[o]),
            VecOp::new("Exp", o, &[o]),
        ];
        (b, ops)
    }

    #[test]
    fn the_exp_bug_is_caught() {
        let (_, ops) = exp_body();
        // Emitted with no barriers -- which is what shipped.
        let bad = unsynced(&ops, &[]);
        assert!(!bad.is_empty(), "the exp race must be reported");
        assert!(bad.iter().any(|e| e.kind == Hazard::Raw));
        // And the fix is the two barriers that were added by hand.
        assert_eq!(barrier_points(&ops), vec![1, 2]);
        assert!(unsynced(&ops, &[1, 2]).is_empty());
    }

    #[test]
    fn a_single_op_body_needs_nothing() {
        // sigmoid: one op, nothing to race against, and it was the operator
        // that kept passing while the others broke.
        let mut b = Buffers::new();
        let (o, i0) = (b.get("o"), b.get("i0"));
        let ops = vec![VecOp::new("Sigmoid", o, &[i0])];
        assert!(barrier_points(&ops).is_empty());
        assert!(unsynced(&ops, &[]).is_empty());
    }

    #[test]
    fn independent_ops_do_not_need_a_barrier() {
        // The point of dependence over adjacency: these two touch nothing in
        // common and may overlap freely.
        let mut b = Buffers::new();
        let (x, y, i, j) = (b.get("x"), b.get("y"), b.get("i"), b.get("j"));
        let ops = vec![VecOp::new("Muls", x, &[i]), VecOp::new("Muls", y, &[j])];
        assert!(barrier_points(&ops).is_empty());
    }

    #[test]
    fn war_on_a_reused_scratch_buffer_is_caught() {
        // A tight UB budget forces scratch reuse, and the hazard is a WRITE
        // landing before an earlier READ finishes -- invisible to anyone
        // looking only for "reads what the last op wrote".
        let mut b = Buffers::new();
        let (s, f0, i0, i2) = (b.get("s"), b.get("f0"), b.get("i0"), b.get("i2"));
        let ops = vec![
            VecOp::new("Cast", f0, &[s]),      // reads s
            VecOp::new("Sub", s, &[i0, i2]),   // clobbers s -- WAR
        ];
        let bad = unsynced(&ops, &[]);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].kind, Hazard::War);
    }

    #[test]
    fn aliasing_through_reinterpret_cast_is_not_invisible() {
        // maximum_lo64 reads an int32 buffer as float to reach Abs/Mins, so
        // w0 and f0 are one piece of storage. Compare operand NAMES and the
        // hazard disappears; compare ids and it does not.
        let mut b = Buffers::new();
        let w0 = b.get("w0");
        let f0 = b.alias("f0", "w0");
        assert_eq!(w0, f0, "a ReinterpretCast must not mint a new buffer");
        let i0 = b.get("i0");
        let ops = vec![
            VecOp::new("Adds", w0, &[i0]),   // write through the int32 view
            VecOp::new("Abs", f0, &[f0]),    // read through the float view
        ];
        let bad = unsynced(&ops, &[]);
        assert!(bad.iter().any(|e| e.kind == Hazard::Raw),
                "the aliased RAW must be reported: {bad:?}");
    }

    #[test]
    fn a_barrier_clears_the_frontier_rather_than_one_edge() {
        // Three writes to one buffer: one barrier before each of the last two
        // is enough, and the count must not grow with the distance.
        let mut b = Buffers::new();
        let (o, i) = (b.get("o"), b.get("i"));
        let ops = vec![
            VecOp::new("Muls", o, &[i]),
            VecOp::new("Adds", o, &[o]),
            VecOp::new("Exp", o, &[o]),
        ];
        assert!(unsynced(&ops, &barrier_points(&ops)).is_empty());
    }

    #[test]
    fn the_report_names_the_ops_and_the_kind() {
        let (_, ops) = exp_body();
        let r = report(&ops, &unsynced(&ops, &[]));
        assert!(r.contains("RAW"), "{r}");
        assert!(r.contains("Muls"), "{r}");
        assert!(r.contains("Adds"), "{r}");
    }

    #[test]
    fn a_correctly_synced_block_reports_clean() {
        let (_, ops) = exp_body();
        assert_eq!(report(&ops, &unsynced(&ops, &[1, 2])), "no unsynchronised hazards");
    }
}
