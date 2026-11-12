// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! PTO -> tile-dialect MLIR, the `.tile` output.
//!
//! # Why this exists, and why it is not a lowering of the lifted Rust
//!
//! `--emit tile` used to refuse with exit 4, saying the tile-dialect MLIR
//! "needs the Rust -> MLIR half". That described one route -- lift the PTO to
//! Rust, then compile the Rust with the codegen backend -- and took it for the
//! only one. It is not. The `.tile` form is `llvm.func` + `hacc.entry` +
//! `llvm.call @__tile_*`, which is the same intrinsic-level information the
//! checked lift already computes. Going via Rust would mean requiring a
//! prebuilt rustc backend and an exact nightly to produce text this module
//! writes directly from the AST.
//!
//! # What makes it trustworthy, which is not this file
//!
//! An emitter that writes plausible MLIR is the easy half. What settles it is
//! that `rspto` consumes this exact form: so for any artifact, `.pto` ->
//! `.tile` -> `rspto` -> `.pto` is a closed loop whose two ends can be
//! compared. `tests/pto_tile_roundtrip.rs` runs it over the whole corpus. That
//! gate does not care whether this file is pretty; it fails if the MLIR means
//! something different from the PTO it came from.
//!
//! # Validation is NOT duplicated
//!
//! Every shape, dtype, aliasing and liftability check lives in
//! `emitter_tile::emit_source`, and this module calls it first and throws the
//! Rust away. That costs one extra emit per conversion and buys the property
//! that matters: an artifact the checked lift refuses is refused here with the
//! identical message, and the two cannot drift into disagreeing about which
//! kernels are liftable. A second copy of the checks would have been faster and
//! would be wrong within a month.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::ast::{BinKind, PtoFunc, PtoModule, PtoOp, PtoType, Ssa, UnKind};

/// Convert PTO-dialect text to tile-dialect MLIR.
pub fn emit_source(source: &str) -> Result<String, String> {
    // Validate through the checked lift, discarding its Rust. See the module
    // header: this is deliberate, not laziness.
    crate::emitter_tile::emit_source(source)?;

    let mut out = String::from(
        "// Emitted from PTO-MLIR by ptoas-rs --emit tile.\n\
         // Form: llvm.func + hacc.entry + __tile_* -- the shape the emitters consume.\n\
         module {\n",
    );
    let bodies = crate::emitter_tile::split_functions(source)?;
    for body in bodies {
        let mut m = crate::parser::parse_module(&body)?;
        crate::fuse::fuse_patterns(&mut m);
        let [f] = &m.funcs[..] else {
            return Err("expected exactly one parsed function per body".into());
        };
        emit_func(&mut out, f, &m)?;
    }
    out.push_str("}\n");
    Ok(out)
}

/// `%pto6` -> `%pto6`; an arg keeps its own spelling.
///
/// MLIR SSA names may carry the `%` and the digits as written, so unlike the
/// Rust emitter this needs no identifier mangling -- which also means a reader
/// can line the output up against the input PTO by eye.
fn ssa(s: &Ssa) -> String {
    let t = s.0.trim();
    if t.starts_with('%') {
        t.to_string()
    } else {
        format!("%{t}")
    }
}

struct Const {
    name: String,
    value: i64,
}

/// Hoisted `llvm.mlir.constant`s, deduplicated by value.
///
/// Deduplicated because the extents repeat on every call: a 14-op kernel over
/// one tile shape would otherwise declare the same two constants 14 times. The
/// names are `%c<value>`, so they also read as their value.
struct Consts(Vec<Const>);

impl Consts {
    fn new() -> Self {
        Consts(Vec::new())
    }
    fn get(&mut self, value: i64) -> String {
        if let Some(c) = self.0.iter().find(|c| c.value == value) {
            return c.name.clone();
        }
        let name = format!("%c{value}");
        self.0.push(Const {
            name: name.clone(),
            value,
        });
        name
    }
}

fn emit_func(out: &mut String, f: &PtoFunc, m: &PtoModule) -> Result<(), String> {
    // Pointer args only, which `emitter_tile` has already enforced.
    let mut params = Vec::new();
    let mut names: HashMap<Ssa, String> = HashMap::new();
    for (id, ty) in &f.args {
        if !matches!(ty, PtoType::Ptr(_)) {
            return Err(format!(
                "{}: only pointer parameters are supported by the tile-dialect emitter",
                f.name
            ));
        }
        let n = ssa(id);
        names.insert(id.clone(), n.clone());
        params.push(format!("{n}: !llvm.ptr<1>"));
    }

    // The view maps: which pointer a view reads, and at what extent. Built the
    // same way the checked lift builds it, because a `pto.tload` names a VIEW
    // and the intrinsic takes a POINTER.
    let mut view_ptr: HashMap<Ssa, Ssa> = HashMap::new();
    let mut view_shape: HashMap<Ssa, (u32, u32)> = HashMap::new();
    for op in &f.ops {
        match op {
            PtoOp::MakeTensorView { dst, ptr, .. } => {
                view_ptr.insert(dst.clone(), ptr.clone());
            }
            PtoOp::PartitionView {
                dst,
                src,
                rows,
                cols,
                ..
            } => {
                let base = view_ptr
                    .get(src)
                    .cloned()
                    .ok_or_else(|| format!("{}: unresolved view {}", f.name, src.0))?;
                view_ptr.insert(dst.clone(), base);
                view_shape.insert(dst.clone(), (*rows, *cols));
            }
            _ => {}
        }
    }

    let mut consts = Consts::new();
    let mut body = String::new();

    for op in &f.ops {
        match op {
            // Structural: consumed by the maps above, nothing to emit.
            PtoOp::MakeTensorView { .. }
            | PtoOp::PartitionView { .. }
            | PtoOp::AllocTile { .. }
            | PtoOp::ScalarConst { .. } => {}

            PtoOp::TLoad { view, tile, .. } => {
                let ptr = view_ptr
                    .get(view)
                    .and_then(|p| names.get(p))
                    .cloned()
                    .ok_or_else(|| format!("{}: unresolved pointer for {}", f.name, view.0))?;
                let (r, c) = extent(m, tile, view, &view_shape)?;
                let rn = consts.get(i64::from(r));
                let cn = consts.get(i64::from(c));
                let dst = ssa(tile);
                names.insert(tile.clone(), dst.clone());
                writeln!(
                    body,
                    "    {dst} = llvm.call @__tile_load_f32({ptr}, {rn}, {cn}) : \
                     (!llvm.ptr<1>, i32, i32) -> i32"
                )
                .unwrap();
            }

            PtoOp::TStore { tile, view, .. } => {
                let ptr = view_ptr
                    .get(view)
                    .and_then(|p| names.get(p))
                    .cloned()
                    .ok_or_else(|| format!("{}: unresolved pointer for {}", f.name, view.0))?;
                let (r, c) = extent(m, tile, view, &view_shape)?;
                let rn = consts.get(i64::from(r));
                let cn = consts.get(i64::from(c));
                let src = use_of(&names, tile, &f.name)?;
                writeln!(
                    body,
                    "    llvm.call @__tile_store_f32({ptr}, {src}, {rn}, {cn}) : \
                     (!llvm.ptr<1>, i32, i32, i32) -> ()"
                )
                .unwrap();
            }

            PtoOp::BinOp { kind, dst, a, b } => {
                let name = match kind {
                    BinKind::Add => "__tile_add_f32",
                    BinKind::Sub => "__tile_sub_f32",
                    BinKind::Mul => "__tile_mul_f32",
                    BinKind::Div => "__tile_div_f32",
                    // `emitter_tile` refuses these before we get here; the arm
                    // exists so that adding one there without adding it here
                    // is a compile error rather than a wrong kernel.
                    BinKind::Max | BinKind::Min => {
                        return Err(format!(
                            "{}: `{kind:?}` has no tile-API intrinsic; the tile-dialect \
                             emitter will not invent a name for it",
                            f.name
                        ))
                    }
                };
                let (r, c) = tile_extent(m, dst)?;
                emit_binary(&mut body, &mut consts, &mut names, name, dst, a, b, r, c, &f.name)?;
            }

            PtoOp::UnOp { kind, dst, src } => {
                let name = match kind {
                    UnKind::Exp => "__tile_exp_f32",
                    UnKind::Neg => "__tile_neg_f32",
                    // `__tile_rsqrt_f32` is declared in `tile_std` AND has a
                    // safe wrapper, so unlike its three neighbours it can be
                    // named here without guessing. Kept beside them rather than
                    // folded in, because the reason they refuse -- a bare
                    // intrinsic with no wrapper, or nothing in the API at all --
                    // does not apply to it.
                    UnKind::Rsqrt => "__tile_rsqrt_f32",
                    UnKind::Abs | UnKind::Sqrt | UnKind::Recip => {
                        return Err(format!(
                            "{}: `{kind:?}` is not in the tile API's unary family here; \
                             refusing rather than naming an intrinsic that may not exist",
                            f.name
                        ))
                    }
                };
                let (r, c) = tile_extent(m, dst)?;
                emit_unary(&mut body, &mut consts, &mut names, name, dst, src, r, c, &f.name)?;
            }

            PtoOp::Softmax { dst, src } => {
                let (r, c) = tile_extent(m, dst)?;
                emit_unary(
                    &mut body,
                    &mut consts,
                    &mut names,
                    "__tile_softmax_f32",
                    dst,
                    src,
                    r,
                    c,
                    &f.name,
                )?;
            }

            PtoOp::Quantize { .. } => {
                // Synthetic node. `--emit tile` parses the artifact WITHOUT
                // fusing, so the pre-fusion file still carries the composed
                // `__tile_quantize_f32_i8` call and this arm is unreachable.
                // Refusing rather than re-deriving the chain from a node that
                // does not carry its four intermediate SSAs.
                return Err(format!(
                    "{}: cannot serialise synthetic Quantize; run this emitter on \
                     the pre-fusion AST",
                    f.name
                ));
            }

            PtoOp::TMatmul { dst, lhs, rhs } => {
                // The extent order is (M, K, N), NOT (M, N, K).
                //
                // Worth a comment because it is unguessable from the names and
                // because getting it wrong produces a kernel that looks right.
                // The first version of this arm emitted (M, N, K), which for a
                // 32x64 @ 64x32 matmul reads as a 32x64 result -- and the
                // `.pto -> .tile -> rspto -> .pto` round trip caught it
                // immediately, with `rspto` refusing its own form: "blocked
                // matmul: store shape 32x32 != matmul result 32x64". Nothing
                // else in this file would have noticed; the MLIR was
                // well-formed and every type matched.
                //
                // The convention is `emitter_tile`'s, which passes
                // `(output.rows, left.cols, output.cols)`.
                let (m_rows, n_cols) = tile_extent(m, dst)?;
                let (_, k) = tile_extent(m, lhs)?;
                let zero = consts.get(0);
                let mn = consts.get(i64::from(m_rows));
                let kn = consts.get(i64::from(k));
                let nn = consts.get(i64::from(n_cols));
                let a = use_of(&names, lhs, &f.name)?;
                let b = use_of(&names, rhs, &f.name)?;
                let d = ssa(dst);
                names.insert(dst.clone(), d.clone());
                writeln!(
                    body,
                    "    {d} = llvm.call @__tile_matmul_f32({zero}, {a}, {b}, {mn}, {kn}, {nn}) : \
                     (i32, i32, i32, i32, i32, i32) -> i32"
                )
                .unwrap();
            }

            // Everything else: refuse by NAME. The checked lift refuses these
            // first, so reaching one here means the two have drifted -- which
            // is worth a loud message rather than a silently dropped op.
            other => {
                return Err(format!(
                    "{}: `{}` reached the tile-dialect emitter, which has no arm for it. \
                     The checked lift accepted it, so the two emitters disagree about what \
                     is liftable -- that is the bug, not this op",
                    f.name,
                    op_name(other)
                ))
            }
        }
    }

    writeln!(
        out,
        "  llvm.func @{}({}) attributes {{hacc.entry}} {{",
        f.name,
        params.join(", ")
    )
    .unwrap();
    out.push_str("    ^bb0:\n");
    for c in &consts.0 {
        writeln!(
            out,
            "    {} = llvm.mlir.constant({} : i32) : i32",
            c.name, c.value
        )
        .unwrap();
    }
    out.push_str(&body);
    out.push_str("    llvm.return\n  }\n");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_binary(
    body: &mut String,
    consts: &mut Consts,
    names: &mut HashMap<Ssa, String>,
    intrinsic: &str,
    dst: &Ssa,
    a: &Ssa,
    b: &Ssa,
    rows: u32,
    cols: u32,
    func: &str,
) -> Result<(), String> {
    let zero = consts.get(0);
    let rn = consts.get(i64::from(rows));
    let cn = consts.get(i64::from(cols));
    let av = use_of(names, a, func)?;
    let bv = use_of(names, b, func)?;
    let d = ssa(dst);
    names.insert(dst.clone(), d.clone());
    writeln!(
        body,
        "    {d} = llvm.call @{intrinsic}({zero}, {av}, {bv}, {rn}, {cn}) : \
         (i32, i32, i32, i32, i32) -> i32"
    )
    .unwrap();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_unary(
    body: &mut String,
    consts: &mut Consts,
    names: &mut HashMap<Ssa, String>,
    intrinsic: &str,
    dst: &Ssa,
    src: &Ssa,
    rows: u32,
    cols: u32,
    func: &str,
) -> Result<(), String> {
    let zero = consts.get(0);
    let rn = consts.get(i64::from(rows));
    let cn = consts.get(i64::from(cols));
    let sv = use_of(names, src, func)?;
    let d = ssa(dst);
    names.insert(dst.clone(), d.clone());
    writeln!(
        body,
        "    {d} = llvm.call @{intrinsic}({zero}, {sv}, {rn}, {cn}) : \
         (i32, i32, i32, i32) -> i32"
    )
    .unwrap();
    Ok(())
}

/// A tile's logical extent.
fn tile_extent(m: &PtoModule, tile: &Ssa) -> Result<(u32, u32), String> {
    let s = crate::emitter_tile::shape_of(m, tile)?;
    Ok((s.rows, s.cols))
}

/// The extent a load/store should carry: the view's when it is known, since
/// that is what the checked lift uses for a store, and the tile's otherwise.
fn extent(
    m: &PtoModule,
    tile: &Ssa,
    view: &Ssa,
    view_shape: &HashMap<Ssa, (u32, u32)>,
) -> Result<(u32, u32), String> {
    if let Some((r, c)) = view_shape.get(view) {
        return Ok((*r, *c));
    }
    tile_extent(m, tile)
}

fn use_of(names: &HashMap<Ssa, String>, ssa_id: &Ssa, func: &str) -> Result<String, String> {
    names
        .get(ssa_id)
        .cloned()
        .ok_or_else(|| format!("{func}: `{}` is used before it is defined", ssa_id.0))
}

fn op_name(op: &PtoOp) -> &'static str {
    match op {
        PtoOp::ScalarConst { .. } => "pto.scalar_const",
        PtoOp::Barrier { .. } => "pto.barrier",
        PtoOp::Subview { .. } => "pto.subview",
        PtoOp::GatherB { .. } => "pto.tgatherb",
        PtoOp::Cvt { .. } => "pto.tcvt",
        PtoOp::BlockQuery { .. } => "pto.get_block_idx",
        PtoOp::IndexCast { .. } => "arith.index_cast",
        PtoOp::MulI { .. } | PtoOp::AddI { .. }
            | PtoOp::GetVal { .. }
            | PtoOp::FpToSi { .. }
            | PtoOp::Tci { .. }
            | PtoOp::TSort32 { .. }
            | PtoOp::TGather { .. }
            | PtoOp::Textract { .. }
        | PtoOp::FpToSi { .. }
            | PtoOp::Tci { .. }
            | PtoOp::TSort32 { .. }
            | PtoOp::TGather { .. }
            | PtoOp::Textract { .. } => "arith.muli",
        PtoOp::AddI { .. } => "arith.addi",
        PtoOp::GetVal { .. } => "pto.tgetval",
        PtoOp::FpToSi { .. } => "arith.fptosi",
        PtoOp::Tci { .. } => "pto.tci",
        PtoOp::TSort32 { .. } => "pto.tsort32",
        PtoOp::TGather { .. } => "pto.tgather",
        PtoOp::Textract { .. } => "pto.textract",
        PtoOp::PartitionViewNd { .. } => "pto.partition_view (rank != 2)",
        PtoOp::For { .. } => "scf.for",
        PtoOp::MakeTensorView { .. } => "pto.make_tensor_view",
        PtoOp::PartitionView { .. } => "pto.partition_view",
        PtoOp::AllocTile { .. } => "pto.alloc_tile",
        PtoOp::TLoad { .. } => "pto.tload",
        PtoOp::ColExpand { .. } => "pto.tcolexpand",
        PtoOp::ColSum { .. } => "pto.tcolsum",
        PtoOp::TStore { .. } => "pto.tstore",
        PtoOp::BinOp { .. } => "pto.t<binop>",
        PtoOp::BinOpScalar { .. } => "pto.t<binop>s",
        PtoOp::UnOp { .. } => "pto.t<unop>",
        PtoOp::RowReduce { .. } => "pto.trow<reduce>",
        PtoOp::RowExpandSub { .. } => "pto.trowexpandsub",
        PtoOp::RowExpandDiv { .. } => "pto.trowexpanddiv",
        PtoOp::RowExpandMul { .. } => "pto.trowexpandmul",
        PtoOp::RowExpand { .. } => "pto.trowexpand",
        PtoOp::TMatmul { .. } => "pto.tmatmul",
        PtoOp::TMov { .. } => "pto.tmov",
        PtoOp::Softmax { .. } => "pto.softmax",
        PtoOp::RmsNorm { .. } => "pto.rms_norm",
        PtoOp::Quantize { .. } => "pto.tquantize",
        PtoOp::If { .. } => "scf.if",
        PtoOp::Transpose { .. } => "pto.ttranspose",
        PtoOp::Unmodelled { .. } => "an unmodelled op",
    }
}
