// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Project ingress-emitted `ascend_tile` MLIR into a stage-2 [`Plan`] so
//! the same [`safety::check_all`](crate::safety::check_all) that guards
//! hand-written PTO kernels can also run over machine-ingested linalg.
//!
//! Background: the linalg bridge (`rustc_codegen_tile/linalg_to_ascend_tile.rs`)
//! converts upstream linalg dialect into an `ascend_tile` form — basically
//! `llvm.func` containing `llvm.call @__tile_<op>_<dt>(…)` intrinsics.
//! That form then flows through `mlir_to_cpp` to AscendC. It never touches
//! `mlir_to_pto`, so the PTO safety oracle never sees it.
//!
//! This module closes the gap with a **naive structural projection**:
//! every tile SSA value becomes a distinct UB slot, sequential offsets,
//! dtype and shape recovered from the intrinsic name and the `(%cR, %cC)`
//! shape constants. Matmul-related space placement (Left / Right / Acc) is
//! *not* inferred — ingress output has no explicit placement, so the
//! `OpConstraint` and `MatmulBounds` checks would false-positive if run.
//! Callers should feed the projected `Plan` to [`check_ingress`] in this
//! module, which runs the applicable subset (aliasing / capacity /
//! dead-tile / linear-use).
//!
//! The naive allocator means the `Capacity` check here is coarse: it sums
//! all slot sizes as if they were live simultaneously, which is a strict
//! *over*-approximation. That's fine — it still fires for the adversarial
//! `1×131072 f32` fixture and for anything obviously too big; it just may
//! cry wolf on kernels that mlir_to_cpp would have kept within budget by
//! reusing buffers. Use the existing mlir_to_cpp-side UB check as the
//! authoritative capacity gate; this one is advisory.

use crate::device::DeviceSpec;
use crate::plan::{
    AddressSpace, BLayout, ExtDType, Plan, PlanFunc, PlanOp, PtoArch, SLayout, Ssa, TileSlotX,
};
use crate::safety::{self, SafetyKind, SafetyReport};
use std::collections::BTreeMap;

/// Project an ingress-emitted ascend_tile MLIR blob into a stage-2 `Plan`.
///
/// The projection is best-effort: malformed or unrecognized intrinsic
/// calls are skipped silently with a note in `warnings`. Shape constants
/// that can't be resolved default to (1, 1) — these kernels will likely
/// fail later emit anyway, but the oracle can still walk what it sees.
pub fn project(ingress_mlir: &str) -> ProjectResult {
    let mut plan = Plan {
        arch: PtoArch::A5,
        funcs: Vec::new(),
    };
    let mut warnings: Vec<String> = Vec::new();

    // Strip comments and split into logical tokens per line.
    let lines: Vec<&str> = ingress_mlir.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if let Some(func_name) = parse_func_header(line) {
            let (func, end) = project_func(&func_name, &lines, i, &mut warnings);
            plan.funcs.push(func);
            i = end;
            continue;
        }
        i += 1;
    }

    ProjectResult { plan, warnings }
}

/// Result of projecting ingress MLIR into a Plan.
pub struct ProjectResult {
    pub plan: Plan,
    /// Non-fatal notes about intrinsics we didn't recognize or shape
    /// constants we couldn't resolve. Useful for diagnosing oracle gaps
    /// but not for blocking compilation.
    pub warnings: Vec<String>,
}

/// Run the subset of oracle checks that apply to ingress-projected Plans.
///
/// Skips `OpConstraint` and `MatmulBounds` because the projector places
/// every slot in Ub — those two checks would false-positive on matmul
/// inputs without real Left/Right/Acc placement from ptoas.
pub fn check_ingress(plan: &Plan, spec: &DeviceSpec) -> SafetyReport {
    let mut rep = SafetyReport::default();
    for f in &plan.funcs {
        safety::check_aliasing(f, &mut rep);
        safety::check_capacity(f, spec, &mut rep);
        safety::check_dead_tiles(f, &mut rep);
        safety::check_dead_writes(f, &mut rep);
        safety::check_linear_use(f, &mut rep);
    }
    // No suppress_redundant_dead_tile call — that function is pub(crate)
    // inside safety and is only noise-reducing. The projected plans we
    // feed here don't produce Aliasing findings for legitimate reuse
    // (every SSA gets a fresh slot), so no redundancy to suppress.
    rep.violations
        .retain(|v| v.kind != SafetyKind::OpConstraint && v.kind != SafetyKind::MatmulBounds);
    rep
}

// ── parser internals ──────────────────────────────────────────────────────

fn parse_func_header(line: &str) -> Option<String> {
    // Format: `llvm.func @name(%arg0: !llvm.ptr<1>, ...) attributes {hacc.entry} {`
    let idx = line.find("llvm.func @")?;
    let tail = &line[idx + "llvm.func @".len()..];
    let end = tail.find('(')?;
    Some(tail[..end].to_string())
}

fn project_func(
    name: &str,
    lines: &[&str],
    start: usize,
    warnings: &mut Vec<String>,
) -> (PlanFunc, usize) {
    let mut slots: BTreeMap<Ssa, TileSlotX> = BTreeMap::new();
    let mut ops: Vec<PlanOp> = Vec::new();
    let mut shapes: BTreeMap<String, u32> = BTreeMap::new(); // %cN -> value
    let mut offset: u64 = 0;

    let mut i = start + 1;
    while i < lines.len() {
        let l = lines[i].trim();
        if l.starts_with('}') || l == "llvm.return" {
            i += 1;
            if l.starts_with('}') {
                break;
            }
            continue;
        }

        // Shape constants: `%cR0 = llvm.mlir.constant(1024 : i32) : i32`
        if let Some((ssa, v)) = parse_const_i32(l) {
            shapes.insert(ssa, v);
            i += 1;
            continue;
        }

        // llvm.call @__tile_*(...)
        if l.contains("llvm.call @__tile_") {
            if let Some(stmt) = parse_tile_call(l) {
                project_op(&stmt, &shapes, &mut slots, &mut ops, &mut offset, warnings);
            } else {
                warnings.push(format!("unparsed tile call: {}", l));
            }
        }

        i += 1;
    }

    (
        PlanFunc {
            name: name.to_string(),
            slots,
            ops,
        },
        i,
    )
}

fn parse_const_i32(line: &str) -> Option<(String, u32)> {
    // `%cX = llvm.mlir.constant(N : i32) : i32`
    let eq = line.find(" = ")?;
    let ssa = line[..eq].trim().to_string();
    if !ssa.starts_with('%') {
        return None;
    }
    let after = &line[eq + 3..];
    if !after.starts_with("llvm.mlir.constant(") {
        return None;
    }
    let open = after.find('(')? + 1;
    let close = after.find(" : i32)")?;
    let v: u32 = after[open..close].trim().parse().ok()?;
    Some((ssa, v))
}

#[derive(Debug)]
struct TileCall {
    /// SSA result of this call, if any. Store calls have no result.
    result: Option<String>,
    /// Intrinsic base name, e.g. `add`, `load`, `matmul`, `exp`.
    op: String,
    /// Element dtype suffix: `f32`, `f16`, `bf16`, `i8`, `i32`, ...
    dtype: String,
    /// SSA names of arguments in source order.
    args: Vec<String>,
}

fn parse_tile_call(line: &str) -> Option<TileCall> {
    // Either form:
    //   %r = llvm.call @__tile_<op>_<dt>(%a, %b, ...) : (...) -> i32
    //   llvm.call @__tile_<op>_<dt>(%a, %b, ...) : (...) -> ()
    let (result, rest) = match line.find(" = ") {
        Some(eq) => (Some(line[..eq].trim().to_string()), &line[eq + 3..]),
        None => (None, line),
    };
    let fn_at = rest.find("llvm.call @__tile_")? + "llvm.call @__tile_".len();
    let rest = &rest[fn_at..];
    let paren = rest.find('(')?;
    let name = &rest[..paren];
    // Split the intrinsic name into op + dtype suffix. The suffix is the
    // last `_`-delimited segment that looks like a dtype token.
    let (op, dtype) = split_op_dtype(name);

    let args_open = paren + 1;
    let args_close = rest.find(')')?;
    let arg_list = &rest[args_open..args_close];
    let args: Vec<String> = arg_list
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| s.starts_with('%'))
        .collect();
    Some(TileCall {
        result,
        op,
        dtype,
        args,
    })
}

fn split_op_dtype(s: &str) -> (String, String) {
    // Dtype tokens we recognize. Anything else gets treated as part of
    // the op name and dtype defaults to `f32`.
    const DTYPES: &[&str] = &["f32", "f16", "bf16", "i8", "i32", "i64", "u64", "ui64"];
    for dt in DTYPES {
        let suffix = format!("_{}", dt);
        if let Some(stripped) = s.strip_suffix(&suffix) {
            return (stripped.to_string(), (*dt).to_string());
        }
    }
    (s.to_string(), "f32".to_string())
}

fn dtype_to_ext(dt: &str) -> ExtDType {
    match dt {
        "f32" => ExtDType::F32,
        "f16" => ExtDType::F16,
        "bf16" => ExtDType::Bf16,
        "i8" => ExtDType::I8,
        "i32" => ExtDType::I32,
        "u64" | "ui64" => ExtDType::UI64,
        _ => ExtDType::F32,
    }
}

fn alloc_slot(
    ssa: &str,
    rows: u32,
    cols: u32,
    dtype: ExtDType,
    slots: &mut BTreeMap<Ssa, TileSlotX>,
    offset: &mut u64,
) -> Ssa {
    let s = Ssa(ssa.to_string());
    let t = TileSlotX {
        ssa: s.clone(),
        space: AddressSpace::Ub,
        offset: *offset,
        rows,
        cols,
        dtype,
        // `Some`, not `None`: these are not a dump's silence being papered
        // over, they are the layout `mlir_to_cpp`'s vector path actually
        // emits, which this function is modelling. `None` on a `TileSlotX`
        // means specifically "the stage-2 text did not state it".
        blayout: Some(BLayout::RowMajor),
        slayout: Some(SLayout::NoneBox),
        s_fractal: Some(512),
    };
    // Match mlir_to_cpp's vector path: each tile becomes a TBuf whose
    // byte count is rounded up to 32 B. Packing is still sequential
    // (mlir_to_cpp has no buffer reuse — see mlir_to_cpp.rs:4773), so
    // the high-water is a sum of 32-B-aligned tile sizes.
    *offset += t.size_bytes_ub_aligned();
    slots.insert(s.clone(), t);
    s
}

fn project_op(
    call: &TileCall,
    shapes: &BTreeMap<String, u32>,
    slots: &mut BTreeMap<Ssa, TileSlotX>,
    ops: &mut Vec<PlanOp>,
    offset: &mut u64,
    warnings: &mut Vec<String>,
) {
    let dtype = dtype_to_ext(&call.dtype);
    let last_two_shape = |args: &[String]| -> (u32, u32) {
        let n = args.len();
        if n < 2 {
            return (1, 1);
        }
        let r = shapes.get(&args[n - 2]).copied().unwrap_or(1);
        let c = shapes.get(&args[n - 1]).copied().unwrap_or(1);
        (r, c)
    };

    match call.op.as_str() {
        // Load: result is a freshly-allocated tile; args[0] is GM ptr.
        "load" => {
            if let Some(res) = &call.result {
                let (r, c) = last_two_shape(&call.args);
                let s = alloc_slot(res, r, c, dtype, slots, offset);
                ops.push(PlanOp::TLoad { tile: s });
            }
        }
        // Store: args[0] is GM ptr, args[1] is the src tile SSA.
        "store" => {
            if call.args.len() >= 2 {
                ops.push(PlanOp::TStore {
                    tile: Ssa(call.args[1].clone()),
                });
            } else {
                warnings.push(format!("store with <2 args: {:?}", call.args));
            }
        }
        // Fill: writes a constant into a tile. Args depend on variant; first
        // tile-typed arg (not a shape const) is the dst. We treat as a write
        // (TLoad) on the result slot.
        "fill" => {
            if let Some(res) = &call.result {
                let (r, c) = last_two_shape(&call.args);
                let s = alloc_slot(res, r, c, dtype, slots, offset);
                ops.push(PlanOp::TLoad { tile: s });
            }
        }
        // Unary: args[0] is the dst hint (unused), args[1] is the src tile.
        "exp" | "log" | "sqrt" | "rsqrt" | "tanh" | "abs" | "neg" | "sigmoid" | "silu" | "relu"
        | "softmax" | "rms_norm" => {
            if let Some(res) = &call.result {
                let (r, c) = last_two_shape(&call.args);
                let dst = alloc_slot(res, r, c, dtype, slots, offset);
                // Source tile is args[1] per the intrinsic convention used
                // across the unary family.
                let src = call.args.get(1).cloned().unwrap_or_default();
                ops.push(PlanOp::TUnary {
                    opname: call.op.clone(),
                    src: Ssa(src),
                    dst,
                });
            }
        }
        // Binary: args[0] dst hint, args[1] a, args[2] b.
        "add" | "sub" | "mul" | "div" | "max" | "min" => {
            if let Some(res) = &call.result {
                let (r, c) = last_two_shape(&call.args);
                let dst = alloc_slot(res, r, c, dtype, slots, offset);
                let a = call.args.get(1).cloned().unwrap_or_default();
                let b = call.args.get(2).cloned().unwrap_or_default();
                ops.push(PlanOp::TBinary {
                    opname: call.op.clone(),
                    a: Ssa(a),
                    b: Ssa(b),
                    dst,
                });
            }
        }
        // Matmul: args[0] dst, args[1] lhs, args[2] rhs. Projected as
        // TMatmul with all three slots in Ub (no Left/Right/Acc).
        // OpConstraint + MatmulBounds are suppressed by check_ingress so
        // these bogus placements don't false-positive.
        "matmul" => {
            if let Some(res) = &call.result {
                // Result shape = (rows of A) × (cols of B). We don't know
                // those from the last two shape consts (those are the
                // inner/outer dims); use them as-is for sizing purposes.
                let (r, c) = last_two_shape(&call.args);
                let acc = alloc_slot(res, r, c, dtype, slots, offset);
                let lhs = call.args.get(1).cloned().unwrap_or_default();
                let rhs = call.args.get(2).cloned().unwrap_or_default();
                ops.push(PlanOp::TMatmul {
                    lhs: Ssa(lhs),
                    rhs: Ssa(rhs),
                    acc,
                });
            }
        }
        // Attention and other multi-input fused ops: treat as unary on the
        // first tile operand so we at least record a read of it. Good
        // enough for dead-tile + aliasing detection; not for full dataflow.
        _ => {
            if let Some(res) = &call.result {
                let (r, c) = last_two_shape(&call.args);
                let dst = alloc_slot(res, r, c, dtype, slots, offset);
                let src = call.args.get(1).cloned().unwrap_or_default();
                ops.push(PlanOp::TUnary {
                    opname: call.op.clone(),
                    src: Ssa(src),
                    dst,
                });
                warnings.push(format!(
                    "unrecognized tile op `{}` projected as unary",
                    call.op
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALIAS: &str = r#"
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

    #[test]
    fn projects_aliasing_fixture_to_plan() {
        let r = project(ALIAS);
        assert_eq!(r.plan.funcs.len(), 1);
        let f = &r.plan.funcs[0];
        assert_eq!(f.name, "adv_aliasing");
        assert_eq!(
            f.slots.len(),
            2,
            "expected 2 slots (load, add), got {:?}",
            f.slots.keys().collect::<Vec<_>>()
        );
        assert_eq!(f.ops.len(), 3, "load + add + store");
        let t0 = f.slots.get(&Ssa("%t0".into())).unwrap();
        assert_eq!((t0.rows, t0.cols), (1, 1024));
        assert_eq!(t0.dtype, ExtDType::F32);
    }
}
