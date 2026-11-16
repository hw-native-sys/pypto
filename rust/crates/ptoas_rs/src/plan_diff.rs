// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Advisory Plan-vs-Plan diff.
//!
//! When building alternative emitters ("would my rewrite produce the same
//! placement as ptoas?"), the goal isn't textual equality — identifiers
//! and emission order can differ and still be equivalent. What matters is
//! whether the *placed tiles* and the *op sequence's effect on them*
//! agree.
//!
//! This module diffs two `Plan`s along four axes:
//!   - per-function presence (same func names?),
//!   - slot placement (same (space, offset, rows, cols, dtype, layout) set?),
//!   - op count & kind histogram,
//!   - matmul operand triples (modulo SSA renaming).
//!
//! Every difference is reported; nothing is asserted. Callers decide
//! whether any field is load-bearing for their use case.

use crate::plan::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiffKind {
    FuncOnlyInA,
    FuncOnlyInB,
    ArchMismatch,
    SlotOnlyInA,
    SlotOnlyInB,
    OpCountMismatch,
    MatmulSetDiffers,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffEntry {
    pub kind: DiffKind,
    pub where_: String, // "<module>" or function name
    pub message: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanDiff {
    pub entries: Vec<DiffEntry>,
}

impl PlanDiff {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

pub fn diff_plans(a: &Plan, b: &Plan) -> PlanDiff {
    let mut d = PlanDiff::default();

    if a.arch != b.arch {
        d.entries.push(DiffEntry {
            kind: DiffKind::ArchMismatch,
            where_: "<module>".into(),
            message: format!("arch a={} vs b={}", a.arch.label(), b.arch.label()),
        });
    }

    let names_a: BTreeMap<&str, &PlanFunc> = a.funcs.iter().map(|f| (f.name.as_str(), f)).collect();
    let names_b: BTreeMap<&str, &PlanFunc> = b.funcs.iter().map(|f| (f.name.as_str(), f)).collect();

    for (n, fa) in &names_a {
        if !names_b.contains_key(n) {
            d.entries.push(DiffEntry {
                kind: DiffKind::FuncOnlyInA,
                where_: (*n).to_string(),
                message: format!("func `{n}` missing in B"),
            });
            continue;
        }
        let fb = names_b[n];
        diff_func(fa, fb, &mut d);
    }
    for n in names_b.keys() {
        if !names_a.contains_key(n) {
            d.entries.push(DiffEntry {
                kind: DiffKind::FuncOnlyInB,
                where_: (*n).to_string(),
                message: format!("func `{n}` missing in A"),
            });
        }
    }

    d
}

fn diff_func(a: &PlanFunc, b: &PlanFunc, out: &mut PlanDiff) {
    // Slot set: normalise each slot to a tuple minus the SSA name.
    #[derive(Eq, PartialEq, Hash, Ord, PartialOrd)]
    struct SlotKey(
        AddressSpace,
        u64,
        u32,
        u32,
        ExtDType,
        Option<BLayout>,
        Option<SLayout>,
        Option<u32>,
    );
    fn key(s: &TileSlotX) -> SlotKey {
        SlotKey(
            s.space,
            s.offset,
            s.rows,
            s.cols,
            s.dtype,
            s.blayout,
            s.slayout,
            s.s_fractal,
        )
    }
    let ka: std::collections::BTreeSet<_> = a.slots.values().map(key).collect();
    let kb: std::collections::BTreeSet<_> = b.slots.values().map(key).collect();
    for k in ka.difference(&kb) {
        out.entries.push(DiffEntry {
            kind: DiffKind::SlotOnlyInA,
            where_: a.name.clone(),
            message: format!(
                "slot {}@{} ({}x{} {}, bl={:?} sl={:?} sf={:?}) only in A",
                k.0.label(),
                k.1,
                k.2,
                k.3,
                k.4.label(),
                k.5,
                k.6,
                k.7
            ),
        });
    }
    for k in kb.difference(&ka) {
        out.entries.push(DiffEntry {
            kind: DiffKind::SlotOnlyInB,
            where_: a.name.clone(),
            message: format!(
                "slot {}@{} ({}x{} {}, bl={:?} sl={:?} sf={:?}) only in B",
                k.0.label(),
                k.1,
                k.2,
                k.3,
                k.4.label(),
                k.5,
                k.6,
                k.7
            ),
        });
    }

    // Op count histogram by opname.
    let hist_a = op_histogram(&a.ops);
    let hist_b = op_histogram(&b.ops);
    let mut all_keys: std::collections::BTreeSet<String> = hist_a.keys().cloned().collect();
    all_keys.extend(hist_b.keys().cloned());
    for kw in all_keys {
        let na = hist_a.get(&kw).copied().unwrap_or(0);
        let nb = hist_b.get(&kw).copied().unwrap_or(0);
        if na != nb {
            out.entries.push(DiffEntry {
                kind: DiffKind::OpCountMismatch,
                where_: a.name.clone(),
                message: format!("op `{kw}`: A={na}, B={nb}"),
            });
        }
    }

    // Matmul operand triples (space + shape + dtype, ignoring SSA names).
    let ma = matmul_triples(a);
    let mb = matmul_triples(b);
    if ma != mb {
        out.entries.push(DiffEntry {
            kind: DiffKind::MatmulSetDiffers,
            where_: a.name.clone(),
            message: format!(
                "matmul triples differ: A={} entries, B={} entries",
                ma.len(),
                mb.len()
            ),
        });
    }
}

fn op_histogram(ops: &[PlanOp]) -> BTreeMap<String, usize> {
    let mut h: BTreeMap<String, usize> = BTreeMap::new();
    for op in ops {
        let kw = match op {
            PlanOp::TLoad { .. } => "tload".to_string(),
            PlanOp::TStore { .. } => "tstore".to_string(),
            PlanOp::TStoreFp { .. } => "tstore_fp".to_string(),
            PlanOp::TMov { .. } => "tmov".to_string(),
            PlanOp::TMatmul { .. } => "tmatmul".to_string(),
            PlanOp::TMatmulAcc { .. } => "tmatmul.acc".to_string(),
            PlanOp::TUnary { opname, .. } => opname.clone(),
            PlanOp::TBinary { opname, .. } => opname.clone(),
            PlanOp::TRowReduce { opname, .. } => opname.clone(),
            PlanOp::TRowExpand { opname, .. } => opname.clone(),
        };
        *h.entry(kw).or_insert(0) += 1;
    }
    h
}

#[derive(Eq, PartialEq, Ord, PartialOrd)]
struct MatmulKey(u32, u32, u32, ExtDType, ExtDType, ExtDType); // (M, K, N, acc, lhs, rhs)

fn matmul_triples(f: &PlanFunc) -> std::collections::BTreeSet<MatmulKey> {
    let mut out = std::collections::BTreeSet::new();
    for op in &f.ops {
        let (lhs, rhs, acc) = match op {
            PlanOp::TMatmul { lhs, rhs, acc } => (lhs, rhs, acc),
            PlanOp::TMatmulAcc {
                lhs, rhs, acc_in, ..
            } => (lhs, rhs, acc_in),
            _ => continue,
        };
        let (Some(ls), Some(rs), Some(acs)) =
            (f.slots.get(lhs), f.slots.get(rhs), f.slots.get(acc))
        else {
            continue;
        };
        out.insert(MatmulKey(
            ls.rows, ls.cols, rs.cols, acs.dtype, ls.dtype, rs.dtype,
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn empty_plan(arch: PtoArch) -> Plan {
        Plan {
            arch,
            funcs: vec![],
        }
    }

    #[test]
    fn arch_mismatch_reported() {
        let a = empty_plan(PtoArch::A3);
        let b = empty_plan(PtoArch::A5);
        let d = diff_plans(&a, &b);
        assert!(d.entries.iter().any(|e| e.kind == DiffKind::ArchMismatch));
    }

    #[test]
    fn identical_plans_produce_empty_diff() {
        let a = empty_plan(PtoArch::A5);
        let b = empty_plan(PtoArch::A5);
        assert!(diff_plans(&a, &b).is_empty());
    }

    #[test]
    fn func_only_in_a_reported() {
        let mut a = empty_plan(PtoArch::A5);
        a.funcs.push(PlanFunc {
            name: "f".into(),
            slots: BTreeMap::new(),
            ops: vec![],
        });
        let b = empty_plan(PtoArch::A5);
        let d = diff_plans(&a, &b);
        assert!(d.entries.iter().any(|e| e.kind == DiffKind::FuncOnlyInA));
    }
}
