// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Oracle-side of the adversarial-ingress audit.
//!
//! Companion to `benchmarks/linalg/kernels_adversarial/` (which drives the
//! MLIR fixtures through the ingress → mlir_to_cpp pipeline). Here we
//! construct the equivalent stage-2 `Plan` directly — matching the buffer
//! allocation the ingress emitter actually produces for each adversarial
//! MLIR — and confirm that `check_all` flags each bug class.
//!
//! Per-fixture mapping (from `/tmp/adv_*.cce` observed on 2026-04-22):
//!
//! | Fixture           | Ingress emit                     | Oracle expectation   |
//! |-------------------|----------------------------------|----------------------|
//! | aliasing          | tbuf0,1,2 @50784 each; one unused | DeadTile (tbuf2)    |
//! | dead_tile         | tbuf2 = Exp(buf0), never read    | DeadTile (tbuf2)    |
//! | waw_double_write  | Exp(buf1), then unresolved call, return buf0 | DeadTile (buf1) + LinearUse would fire if log emitter hit the same slot |
//! | capacity_overflow | hard emitter error BEFORE oracle  | N/A (emitter wins)  |
//!
//! Because ingress emits ascend_tile, not PTO-MLIR, we can't feed its
//! output directly to `parse_stage2`. Instead each test below handcrafts
//! the Plan that mirrors the buffer structure we observed in the .cce.
//! The point: show the oracle *would* catch what the mlir_to_cpp path
//! emits silently today.
//!
//! If the ingress path were to grow an "emit-to-PTO" mode (or if the same
//! safety checks were lifted to the ascend_tile dialect), these tests
//! become the acceptance criteria.

use ptoas_rs::{
    check_all, check_ingress, default_a5_910b2_cann85, project, AddressSpace, BLayout, ExtDType,
    Plan, PlanFunc, PlanOp, PtoArch, SLayout, SafetyKind, Ssa, TileSlotX,
};
use std::collections::BTreeMap;

fn slot(
    ssa: &str,
    space: AddressSpace,
    offset: u64,
    rows: u32,
    cols: u32,
    dtype: ExtDType,
) -> (Ssa, TileSlotX) {
    let s = Ssa(ssa.to_string());
    let t = TileSlotX {
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
    (s, t)
}

fn mk_plan(name: &str, slots: Vec<(Ssa, TileSlotX)>, ops: Vec<PlanOp>) -> Plan {
    let mut m: BTreeMap<Ssa, TileSlotX> = BTreeMap::new();
    for (k, v) in slots {
        m.insert(k, v);
    }
    Plan {
        arch: PtoArch::A5,
        funcs: vec![PlanFunc {
            name: name.into(),
            slots: m,
            ops,
        }],
    }
}

fn has_kind(rep: &ptoas_rs::SafetyReport, k: SafetyKind) -> bool {
    rep.violations.iter().any(|v| v.kind == k)
}

// ── Fixture 1: aliasing_same_tensor_twice ────────────────────────────────
//
// Source MLIR: y = x + x with both linalg inputs bound to %arg0.
// Observed emitter behavior: DataCopy once to tbuf0, then Add(tbuf1,tbuf0,tbuf0).
// BUT tbuf2 was still allocated (50784 B) and never read/written.
// Oracle should flag the wasted tile as DeadTile.
#[test]
fn adv_aliasing_dead_tile_flagged() {
    let (s0, t0) = slot("%b0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
    let (s1, t1) = slot("%b1", AddressSpace::Ub, 50784, 1, 1024, ExtDType::F32);
    let (s2, t2) = slot("%b2", AddressSpace::Ub, 101568, 1, 1024, ExtDType::F32);
    // Ops mirror the emitted .cce: load b0, add b0+b0 → b1, store b1. b2 orphan.
    let ops = vec![
        PlanOp::TLoad { tile: s0.clone() },
        PlanOp::TBinary {
            opname: "add".into(),
            a: s0.clone(),
            b: s0.clone(),
            dst: s1.clone(),
        },
        PlanOp::TStore { tile: s1.clone() },
    ];
    let plan = mk_plan("adv_aliasing", vec![(s0, t0), (s1, t1), (s2, t2)], ops);
    let rep = check_all(&plan, &default_a5_910b2_cann85());
    assert!(
        has_kind(&rep, SafetyKind::DeadTile),
        "expected DeadTile for orphan buf2, got {:?}",
        rep.violations
    );
}

// ── Fixture 2: dead_tile_unused_intermediate ─────────────────────────────
//
// Source MLIR: compute Exp(%arg0) → %t, discard; return %arg0 + %arg1.
// Observed .cce: 5 tbufs. buf2 = Exp(buf0) is computed then never read.
// Oracle targets: DeadTile (buf4, entirely untouched) AND dead-write
// (buf2, written by Exp then never read).
//
// check_dead_tiles flags slots *never touched*. check_dead_writes (added
// 2026-04-22 as part of the ingress-oracle wiring) flags slots written
// but never subsequently read. Together they catch both wasted-allocation
// and wasted-computation patterns.
#[test]
fn adv_dead_intermediate_and_dead_write_flagged() {
    // Shrink slots so 5-way layout fits 192 KiB UB cap; we only need
    // distinct non-overlapping ranges, not the real 1×1024 footprint.
    let (s0, t0) = slot("%b0", AddressSpace::Ub, 0, 1, 512, ExtDType::F32);
    let (s1, t1) = slot("%b1", AddressSpace::Ub, 4096, 1, 512, ExtDType::F32);
    let (s2, t2) = slot("%b2", AddressSpace::Ub, 8192, 1, 512, ExtDType::F32); // Exp dst, never read
    let (s3, t3) = slot("%b3", AddressSpace::Ub, 12288, 1, 512, ExtDType::F32);
    let (s4, t4) = slot("%b4", AddressSpace::Ub, 16384, 1, 512, ExtDType::F32); // unused

    // Ops from the observed .cce:
    //   load b0, load b1; Exp b0→b2 (b2 never read); Add b0+b1→b3; store b3.
    let ops = vec![
        PlanOp::TLoad { tile: s0.clone() },
        PlanOp::TLoad { tile: s1.clone() },
        PlanOp::TUnary {
            opname: "exp".into(),
            src: s0.clone(),
            dst: s2.clone(),
        },
        PlanOp::TBinary {
            opname: "add".into(),
            a: s0.clone(),
            b: s1.clone(),
            dst: s3.clone(),
        },
        PlanOp::TStore { tile: s3.clone() },
    ];
    let plan = mk_plan(
        "adv_dead_tile",
        vec![(s0, t0), (s1, t1), (s2, t2), (s3, t3), (s4, t4)],
        ops,
    );
    let rep = check_all(&plan, &default_a5_910b2_cann85());

    // Messages look like: "slot `%b4` allocated in vec at offset 16384 but never used".
    // Pull out ssas inside backticks.
    fn ssa_in_msg(msg: &str) -> Vec<String> {
        msg.split('`')
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, s)| s.to_string())
            .collect()
    }
    let dead_ssas: Vec<String> = rep
        .violations
        .iter()
        .filter(|v| v.kind == SafetyKind::DeadTile)
        .flat_map(|v| ssa_in_msg(&v.message))
        .collect();
    assert!(
        dead_ssas.iter().any(|s| s == "%b4"),
        "oracle should flag %b4 (never touched), got {:?}",
        rep.violations
    );

    // Previously a documented false-negative (write-never-read). Now
    // closed by check_dead_writes — %b2 IS flagged as a dead write.
    assert!(
        dead_ssas.iter().any(|s| s == "%b2"),
        "oracle should flag %b2 (written by Exp, never read), got {:?}",
        rep.violations,
    );
    // The message from check_dead_writes should explain *why* it's dead.
    let dead_write_msgs: Vec<_> = rep
        .violations
        .iter()
        .filter(|v| v.kind == SafetyKind::DeadTile && v.message.contains("never read"))
        .collect();
    assert!(
        !dead_write_msgs.is_empty(),
        "expected a 'never read' dead-write finding, got {:?}",
        rep.violations,
    );
}

// ── Fixture 3: waw_double_write ──────────────────────────────────────────
//
// Source MLIR: two back-to-back linalg.generics, second's body writes to
// the first's output slot with no intervening read.
// Observed .cce: ingress EMITTED ONE Exp and then called a nonexistent
// `__tile_log_f32` as a fallback. Under a correct emitter, both
// writes would land on the same UB slot → WAW the oracle is designed
// to catch.
//
// Here we model the intended (but incorrectly emitted) Plan: two
// unary writes to the same tile with no read between. Oracle should
// flag LinearUse.
#[test]
fn adv_waw_double_write_flagged() {
    let (s_in, t_in) = slot("%b0", AddressSpace::Ub, 0, 1, 1024, ExtDType::F32);
    let (s_out, t_out) = slot("%b1", AddressSpace::Ub, 50784, 1, 1024, ExtDType::F32);
    let ops = vec![
        PlanOp::TLoad { tile: s_in.clone() },
        PlanOp::TUnary {
            opname: "exp".into(),
            src: s_in.clone(),
            dst: s_out.clone(),
        },
        PlanOp::TUnary {
            opname: "log".into(),
            src: s_in.clone(),
            dst: s_out.clone(),
        }, // WAW
        PlanOp::TStore {
            tile: s_out.clone(),
        },
    ];
    let plan = mk_plan("adv_waw", vec![(s_in, t_in), (s_out, t_out)], ops);
    let rep = check_all(&plan, &default_a5_910b2_cann85());
    assert!(
        has_kind(&rep, SafetyKind::LinearUse),
        "expected LinearUse (WAW) on back-to-back writes to %b1, got {:?}",
        rep.violations
    );
}

// ── Fixture 4: capacity_overflow_1x131072 ────────────────────────────────
//
// Source MLIR: exp on 1×131072 f32 = 512 KiB tile. UB cap is 192 KiB on
// 910B2 (actually ~256 KiB; mlir_to_cpp uses a 253952 B / N-slot budget).
// Observed: the mlir_to_cpp emitter ALREADY catches this with its own
// UB overflow check (`UB overflow in kernel 'adv_capacity_overflow'`).
// Oracle would also catch it; the test below confirms the oracle path
// is redundant-but-correct here — useful as a second line of defence
// if the emitter's static cap check is ever bypassed.
#[test]
fn adv_capacity_overflow_flagged() {
    // 1×131072 f32 = 524288 B at offset 0 in Ub. A5 Ub cap is way smaller.
    let (s0, t0) = slot("%b0", AddressSpace::Ub, 0, 1, 131072, ExtDType::F32);
    let ops = vec![PlanOp::TLoad { tile: s0.clone() }];
    let plan = mk_plan("adv_capacity_overflow", vec![(s0, t0)], ops);
    let rep = check_all(&plan, &default_a5_910b2_cann85());
    assert!(
        has_kind(&rep, SafetyKind::Capacity),
        "expected Capacity on 512 KiB tile in Ub, got {:?}",
        rep.violations
    );
}

// ── End-to-end: project actual ingress MLIR and check ingressed plan ─────
//
// Above tests handcraft the Plan. These tests drive the projector over
// real ingress-emitted ascend_tile MLIR (taken verbatim from dumping
// the 4 adversarial fixtures through the linalg bridge) and confirm the
// projector + check_ingress pipeline catches what the emitter silently
// passes on.

const INGRESS_ALIASING: &str = r#"
module {
  llvm.func @adv_aliasing(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
  ^bb0:
    %cR0 = llvm.mlir.constant(1 : i32) : i32
    %cC1 = llvm.mlir.constant(1024 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_add_f32(%t0, %t0, %t0, %cR0, %cC1) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const INGRESS_DEAD_TILE: &str = r#"
module {
  llvm.func @adv_dead_tile(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>, %arg2: !llvm.ptr<1>) attributes {hacc.entry} {
  ^bb0:
    %cR0 = llvm.mlir.constant(1 : i32) : i32
    %cC1 = llvm.mlir.constant(1024 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_load_f32(%arg1, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t2 = llvm.call @__tile_exp_f32(%t0, %t0, %cR0, %cC1) : (i32, i32, i32, i32) -> i32
    %t3 = llvm.call @__tile_add_f32(%t0, %t0, %t1, %cR0, %cC1) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %t3, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const INGRESS_WAW: &str = r#"
module {
  llvm.func @adv_waw(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
  ^bb0:
    %cR0 = llvm.mlir.constant(1 : i32) : i32
    %cC1 = llvm.mlir.constant(1024 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_exp_f32(%t0, %t0, %cR0, %cC1) : (i32, i32, i32, i32) -> i32
    %t2 = llvm.call @__tile_log_f32(%t0, %t0, %cR0, %cC1) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t2, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

const INGRESS_CAPACITY: &str = r#"
module {
  llvm.func @adv_capacity(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
  ^bb0:
    %cR0 = llvm.mlir.constant(1 : i32) : i32
    %cC1 = llvm.mlir.constant(131072 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_exp_f32(%t0, %t0, %cR0, %cC1) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

#[test]
fn ingress_aliasing_projects_cleanly() {
    // SSA-form linalg can't alias at the buffer level — `y = x + x`
    // collapses to one load + one add referring to the same SSA. The
    // projection should be clean; no oracle finding.
    let r = project(INGRESS_ALIASING);
    assert!(
        r.warnings.is_empty(),
        "projector warnings: {:?}",
        r.warnings
    );
    let rep = check_ingress(&r.plan, &default_a5_910b2_cann85());
    assert!(
        rep.violations.is_empty(),
        "expected clean projection of y=x+x aliasing, got {:?}",
        rep.violations,
    );
}

#[test]
fn ingress_dead_intermediate_caught_by_dead_write() {
    // exp(arg0) is computed into %t2 but %t2 is never consumed. The
    // add(arg0, arg1) path produces the stored output. check_dead_writes
    // should flag %t2 as a written-never-read slot.
    let r = project(INGRESS_DEAD_TILE);
    let rep = check_ingress(&r.plan, &default_a5_910b2_cann85());
    let dead_write = rep
        .violations
        .iter()
        .find(|v| v.kind == SafetyKind::DeadTile && v.message.contains("never read"));
    assert!(
        dead_write.is_some(),
        "expected dead-write finding on ingressed exp-then-discard, got {:?}",
        rep.violations,
    );
    assert!(
        dead_write.unwrap().message.contains("%t2"),
        "dead-write message should name %t2, got: {}",
        dead_write.unwrap().message,
    );
}

#[test]
fn ingress_waw_caught_as_dead_write() {
    // Two unaries whose outputs are %t1, %t2 respectively. Only %t2 is
    // stored; %t1 is a dead write. (The linalg-level WAW the source MLIR
    // describes was eliminated by torch-mlir's SSA conversion — what
    // survives to ingress is a write-never-read, which the oracle now
    // catches.)
    let r = project(INGRESS_WAW);
    let rep = check_ingress(&r.plan, &default_a5_910b2_cann85());
    let dead_write = rep.violations.iter().find(|v| {
        v.kind == SafetyKind::DeadTile
            && v.message.contains("never read")
            && v.message.contains("%t1")
    });
    assert!(
        dead_write.is_some(),
        "expected dead-write on %t1 (exp result never read), got {:?}",
        rep.violations,
    );
}

#[test]
fn ingress_capacity_1x131072_flagged() {
    // 1 × 131072 × 4B = 524288 B — > 192 KiB Ub cap.
    let r = project(INGRESS_CAPACITY);
    let rep = check_ingress(&r.plan, &default_a5_910b2_cann85());
    assert!(
        rep.violations
            .iter()
            .any(|v| v.kind == SafetyKind::Capacity),
        "expected Capacity finding on 512 KiB projected plan, got {:?}",
        rep.violations,
    );
}

// ── Alignment-parity tests (2026-04-22) ──────────────────────────────
//
// Property: for a projected plan on the ingress path, the UB high-water
// must equal the sum of 32-B-aligned tile sizes. This mirrors what
// mlir_to_cpp emits when each SSA becomes a separate TBuf<VECCALC>
// (see `rustc_codegen_tile/src/mlir_to_cpp.rs:4277`, rounding rule
// `((elems * elem_size) + 31) & !31`).
//
// These tests nail the property in CI so future projector changes that
// silently drop the 32-B round are caught.

/// Sum of `size_bytes_ub_aligned()` over all UB slots in a Plan.
fn ub_aligned_footprint(plan: &Plan) -> u64 {
    plan.funcs
        .iter()
        .flat_map(|f| f.slots.values())
        .filter(|s| s.space == AddressSpace::Ub)
        .map(|s| s.size_bytes_ub_aligned())
        .sum()
}

/// UB high-water as `check_capacity` sees it (max over slots of
/// `offset + size_bytes_ub_aligned()`).
fn ub_high_water(plan: &Plan) -> u64 {
    plan.funcs
        .iter()
        .flat_map(|f| f.slots.values())
        .filter(|s| s.space == AddressSpace::Ub)
        .map(|s| s.offset + s.size_bytes_ub_aligned())
        .max()
        .unwrap_or(0)
}

#[test]
fn alignment_parity_naturally_aligned_tiles() {
    // 1×1024 f32 = 4096 B — already 32-B-aligned. Aligned sum must
    // equal raw sum, and high-water must equal the sum (sequential
    // packing).
    let r = project(INGRESS_ALIASING);
    let raw: u64 = r
        .plan
        .funcs
        .iter()
        .flat_map(|f| f.slots.values())
        .filter(|s| s.space == AddressSpace::Ub)
        .map(|s| s.size_bytes())
        .sum();
    let aligned = ub_aligned_footprint(&r.plan);
    let high_water = ub_high_water(&r.plan);
    assert_eq!(
        raw, aligned,
        "1×1024 f32 tiles are 32-B-aligned; raw and aligned sums must match"
    );
    assert_eq!(
        aligned, high_water,
        "sequential allocator: high-water must equal sum of aligned sizes"
    );
}

// Sub-32-B tile: 1 × 4 × f32 = 16 B raw → 32 B aligned. Forces the
// aligned sum to exceed the raw sum, catching any regression that
// drops the round.
const INGRESS_SMALL_TILE: &str = r#"
module {
  llvm.func @small_tile(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) attributes {hacc.entry} {
  ^bb0:
    %cR0 = llvm.mlir.constant(1 : i32) : i32
    %cC1 = llvm.mlir.constant(4 : i32) : i32
    %t0 = llvm.call @__tile_load_f32(%arg0, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32) -> i32
    %t1 = llvm.call @__tile_exp_f32(%t0, %t0, %cR0, %cC1) : (i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg1, %t1, %cR0, %cC1) : (!llvm.ptr<1>, i32, i32, i32) -> ()
    llvm.return
  }
}
"#;

#[test]
fn alignment_parity_sub_32b_tiles_round_up() {
    // Two 1×4 f32 tiles (t0 loaded, t1 from exp). Raw = 16 B each = 32 B total.
    // Aligned = 32 B each = 64 B total — the aligned sum strictly exceeds raw.
    let r = project(INGRESS_SMALL_TILE);
    let raw: u64 = r
        .plan
        .funcs
        .iter()
        .flat_map(|f| f.slots.values())
        .filter(|s| s.space == AddressSpace::Ub)
        .map(|s| s.size_bytes())
        .sum();
    let aligned = ub_aligned_footprint(&r.plan);
    let high_water = ub_high_water(&r.plan);
    assert_eq!(raw, 32, "sanity: 2 × (1×4×f32) = 32 raw bytes");
    assert_eq!(
        aligned, 64,
        "2 tiles × 32-B-aligned slot = 64 B; alignment round must fire"
    );
    assert!(
        aligned > raw,
        "sub-32-B tiles must project larger than their raw product"
    );
    assert_eq!(
        aligned, high_water,
        "sequential allocator: high-water must equal sum of aligned sizes"
    );
}
