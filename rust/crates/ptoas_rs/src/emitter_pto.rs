// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! AST → canonical PTO-MLIR text.
//!
//! Used by the S3 round-trip verifier: parse PTO, emit PTO, reparse, and
//! compare AST-to-AST. This is NOT byte-identical with `ptoas` output --
//! the parser deliberately drops non-load-bearing attributes (v_row,
//! blayout, slayout, fractal, pad, and the scratch-tile operand on
//! trowmax/trowsum). The reverse emitter therefore writes a canonical form
//! that omits those fields. Round-trip equality is defined modulo them.
//!
//! Synthetic `PtoOp::Softmax` (introduced by the S2 fuse pass) has no
//! direct PTO spelling, so the reverse emitter expands it back into the
//! five-op chain. This lets the verifier run on fused-then-reemitted
//! programs and still get a meaningful diff.

use crate::ast::*;
use std::collections::BTreeMap;
use std::fmt::Write;

pub fn emit_module_pto(m: &PtoModule) -> Result<String, String> {
    let mut out = String::new();
    writeln!(out, "module {{").unwrap();
    for f in &m.funcs {
        emit_func(&mut out, f, &f.shapes)?;
    }
    writeln!(out, "}}").unwrap();
    Ok(out)
}

fn emit_func(
    out: &mut String,
    f: &PtoFunc,
    shapes: &BTreeMap<Ssa, TileShape>,
) -> Result<(), String> {
    let mut arg_list = String::new();
    for (i, (ssa, ty)) in f.args.iter().enumerate() {
        if i > 0 {
            arg_list.push_str(", ");
        }
        write!(arg_list, "{}: {}", ssa.0, format_type(ty)).unwrap();
    }
    // The core assignment, re-emitted. Dropped here until 2026-11-01: all
    // seven cube/vector artifacts came back with both halves unqualified, and
    // `verify_roundtrip` reported them faithful because the AST it compares
    // did not carry the attribute on either pass. An AST-equality gate is only
    // ever as sensitive as the AST.
    let attrs = match f.kind {
        None => String::new(),
        Some(k) => format!(
            " attributes {{pto.kernel_kind = #pto.kernel_kind<{}>}}",
            k.pto_spelling()
        ),
    };
    writeln!(out, "  func.func @{}({}){} {{", f.name, arg_list, attrs).unwrap();
    for op in &f.ops {
        emit_op(out, op, shapes)?;
    }
    writeln!(out, "    return").unwrap();
    writeln!(out, "  }}").unwrap();
    Ok(())
}

fn emit_op(out: &mut String, op: &PtoOp, shapes: &BTreeMap<Ssa, TileShape>) -> Result<(), String> {
    match op {
        PtoOp::ScalarConst { dst, value } => match value {
            ConstVal::Index(v) => {
                writeln!(out, "    {} = arith.constant {} : index", dst.0, v).unwrap();
            }
            // Literal and type verbatim: reprinting a parsed float is what the
            // round-trip is there to catch.
            ConstVal::Typed { lit, ty } => {
                writeln!(out, "    {} = arith.constant {} : {}", dst.0, lit, ty).unwrap();
            }
        },
        PtoOp::BinOpScalar {
            kind,
            dst,
            a,
            scalar,
            scalar_ty,
        } => {
            // The second operand's type is the scalar's OWN, not a tile type --
            // which is what the prefix dispatch got wrong on the way in -- and
            // not a literal `f32` either, which is what this arm got wrong on
            // the way out until 2026-11-01. See `ast::PtoOp::BinOpScalar`.
            writeln!(
                out,
                "    {mn} ins({a}, {s} : {ty}, {sty}) outs({dst} : {dty})",
                mn = kind.pto_mnemonic(),
                a = a.0,
                s = scalar.0,
                sty = scalar_ty,
                ty = tile_ty_str(shapes, a)?,
                dst = dst.0,
                dty = tile_ty_str(shapes, dst)?,
            )
            .unwrap();
        }
        PtoOp::For {
            var,
            lb,
            ub,
            step,
            body,
        } => {
            writeln!(
                out,
                "    scf.for {} = {} to {} step {} {{",
                var.0, lb.0, ub.0, step.0
            )
            .unwrap();
            // Every arm writes a fixed four-space prefix, so the body is
            // emitted on its own and re-indented rather than threading a depth
            // through all twenty-odd arms.
            let mut inner = String::new();
            for op in body {
                emit_op(&mut inner, op, shapes)?;
            }
            for l in inner.lines() {
                writeln!(out, "  {l}").unwrap();
            }
            writeln!(out, "    }}").unwrap();
        }
        PtoOp::If {
            cond,
            then_body,
            else_body,
        } => {
            // Same re-indentation trick as `For`, twice. `} else {` is written
            // as one line because that is what it is: the then-region's closer
            // and the else-region's opener.
            let render = |ops: &[PtoOp], out: &mut String| -> Result<(), String> {
                let mut inner = String::new();
                for op in ops {
                    emit_op(&mut inner, op, shapes)?;
                }
                for l in inner.lines() {
                    writeln!(out, "  {l}").unwrap();
                }
                Ok(())
            };
            writeln!(out, "    scf.if {} {{", cond.0).unwrap();
            render(then_body, out)?;
            match else_body {
                Some(els) => {
                    writeln!(out, "    }} else {{").unwrap();
                    render(els, out)?;
                    writeln!(out, "    }}").unwrap();
                }
                None => writeln!(out, "    }}").unwrap(),
            }
        }
        PtoOp::Barrier { pipe } => {
            writeln!(out, "    pto.barrier <{pipe}>").unwrap();
        }
        PtoOp::BlockQuery { dst, which, ty } => {
            writeln!(
                out,
                "    {} = \"{}\"() : () -> {ty}",
                dst.0,
                which.pto_mnemonic(),
            )
            .unwrap();
        }
        PtoOp::MulI { dst, lhs, rhs } => {
            writeln!(
                out,
                "    {} = arith.muli {}, {} : index",
                dst.0, lhs.0, rhs.0,
            )
            .unwrap();
        }
        PtoOp::AddI { dst, lhs, rhs } => {
            writeln!(
                out,
                "    {} = arith.addi {}, {} : index",
                dst.0, lhs.0, rhs.0,
            )
            .unwrap();
        }
        PtoOp::GetVal { dst, src, idx } => {
            writeln!(
                out,
                "    {} = \"pto.tgetval\"({}, {}) : (...) -> f32",
                dst.0, src.0, idx.0,
            )
            .unwrap();
        }
        PtoOp::FpToSi { dst, src } => {
            writeln!(out, "    {} = arith.fptosi {} : f32 to i32", dst.0, src.0).unwrap();
        }
        PtoOp::Tci { dst, start } => {
            writeln!(out, "    {} = pto.tci ins({} : i32) outs({} : !pto.tile_buf)", dst.0, start.0, dst.0)
                .unwrap();
        }
        PtoOp::TSort32 { dst, values, indices } => {
            writeln!(out, "    {} = pto.tsort32 ins({}, {}) outs({})", dst.0, values.0, indices.0, dst.0)
                .unwrap();
        }
        PtoOp::TGather { dst, src, mask, axis } => {
            writeln!(out, "    {} = pto.tgather ins({}, {{maskPattern = P{mask}}} : tile) outs({}) {{axis = \"{axis}\"}}", dst.0, src.0, dst.0)
                .unwrap();
        }
        PtoOp::Textract { dst, src, row, col } => {
            writeln!(out, "    {} = pto.textract ins({}, {}, {}) outs({})", dst.0, src.0, row.0, col.0, dst.0)
                .unwrap();
        }
        PtoOp::IndexCast {
            dst,
            src,
            from,
            to,
        } => {
            writeln!(
                out,
                "    {} = arith.index_cast {} : {from} to {to}",
                dst.0, src.0,
            )
            .unwrap();
        }
        PtoOp::Cvt { dst, src, sat_mode } => {
            let attr = match sat_mode {
                Some(m) => format!(" {{sat_mode = {m}}}"),
                None => String::new(),
            };
            writeln!(
                out,
                "    pto.tcvt ins({}{} : {}) outs({} : {})",
                src.0,
                attr,
                tile_ty_str(shapes, src)?,
                dst.0,
                tile_ty_str(shapes, dst)?,
            )
            .unwrap();
        }
        PtoOp::Subview {
            dst,
            src,
            offsets,
            sizes,
            segment_sizes,
            ty,
        } => {
            let ops: Vec<&str> = offsets.iter().map(|o| o.0.as_str()).collect();
            // One `index` per dynamic offset, so the type list matches the
            // operand list however many there are.
            let idx_tys = vec!["index"; offsets.len()].join(", ");
            writeln!(
                out,
                "    {} = \"pto.subview\"({}, {}) {{operandSegmentSizes = {}, sizes = [{}, {}]}} : ({}, {}) -> {}",
                dst.0,
                src.0,
                ops.join(", "),
                segment_sizes,
                sizes[0],
                sizes[1],
                tile_ty_str(shapes, src)?,
                idx_tys,
                format_type(ty),
            )
            .unwrap();
        }
        PtoOp::GatherB { dst, src, idx } => {
            writeln!(
                out,
                "    pto.tgatherb ins({}, {} : {}, {}) outs({} : {})",
                src.0,
                idx.0,
                tile_ty_str(shapes, src)?,
                tile_ty_str(shapes, idx)?,
                dst.0,
                tile_ty_str(shapes, dst)?,
            )
            .unwrap();
        }
        PtoOp::MakeTensorView {
            dst,
            ptr,
            dtype,
            shape,
            strides,
            layout,
        } => {
            let attr = match layout {
                Some(l) => format!(" {{layout = {l}}}"),
                None => String::new(),
            };
            writeln!(
                out,
                "    {} = pto.make_tensor_view {}, shape = [{}], strides = [{}]{attr} : \
                 !pto.tensor_view<{}x{dt}>",
                dst.0,
                ptr.0,
                shape.join(", "),
                strides.join(", "),
                vec!["?"; shape.len()].join("x"),
                dt = dtype.pto_spelling(),
            )
            .unwrap();
        }
        PtoOp::PartitionViewNd {
            dst,
            src,
            dims,
            dtype,
            offsets,
            sizes,
        } => {
            let d: Vec<String> = dims.iter().map(|d| d.to_string()).collect();
            let shape = format!("{}x{}", d.join("x"), dtype.pto_spelling());
            // The SOURCE view's rank is not recorded -- the parser reads only
            // the type after `->` -- so it is printed as the result's rank.
            // That is a derivation, not an echo, and it is sound only because
            // `pto.partition_view` cannot change rank; a `?x?` literal, which
            // is what stood here, made the re-emitted text invalid for every
            // rank but 2.
            writeln!(
                out,
                "    {} = pto.partition_view {}, offsets = [{}], sizes = [{}] : \
                 !pto.tensor_view<{}x{dt}> -> !pto.partition_tensor_view<{shape}>",
                dst.0,
                src.0,
                offsets.join(", "),
                sizes.join(", "),
                vec!["?"; dims.len()].join("x"),
                dt = dtype.pto_spelling(),
            )
            .unwrap();
        }
        PtoOp::PartitionView {
            dst,
            src,
            rows,
            cols,
            dtype,
            offsets,
            sizes,
        } => {
            writeln!(
                out,
                "    {} = pto.partition_view {}, offsets = [{}, {}], sizes = [{}, {}] : !pto.tensor_view<?x?x{dt}> -> !pto.partition_tensor_view<{r}x{c}x{dt}>",
                dst.0, src.0,
                offsets[0], offsets[1], sizes[0], sizes[1],
                r = rows, c = cols,
                dt = dtype.pto_spelling(),
            ).unwrap();
        }
        PtoOp::AllocTile { dst, ty } => {
            let ty_str = format_type(ty);
            writeln!(out, "    {} = pto.alloc_tile : {ty_str}", dst.0).unwrap();
        }
        PtoOp::TLoad {
            view,
            tile,
            view_ty,
        } => {
            // The view's OWN type, not one rebuilt from the tile's shape --
            // 35 of the corpus's 487 tload/tstore lines disagree, and a rank-N
            // view cannot be rebuilt at all.
            let tile_ty = tile_ty_str(shapes, tile)?;
            writeln!(
                out,
                "    pto.tload ins({} : {}) outs({} : {tile_ty})",
                view.0,
                format_type(view_ty),
                tile.0,
            )
            .unwrap();
        }
        PtoOp::TStore {
            tile,
            view,
            view_ty,
        } => {
            let tile_ty = tile_ty_str(shapes, tile)?;
            writeln!(
                out,
                "    pto.tstore ins({} : {tile_ty}) outs({} : {})",
                tile.0,
                view.0,
                format_type(view_ty),
            )
            .unwrap();
        }
        PtoOp::BinOp { kind, dst, a, b } => {
            let a_ty = tile_ty_str(shapes, a)?;
            let b_ty = tile_ty_str(shapes, b)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            let kw = match kind {
                BinKind::Add => "pto.tadd",
                BinKind::Sub => "pto.tsub",
                BinKind::Mul => "pto.tmul",
                BinKind::Div => "pto.tdiv",
                BinKind::Max => "pto.tmax",
                BinKind::Min => "pto.tmin",
            };
            writeln!(
                out,
                "    {kw} ins({}, {} : {a_ty}, {b_ty}) outs({} : {dst_ty})",
                a.0, b.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::UnOp { kind, dst, src } => {
            let src_ty = tile_ty_str(shapes, src)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            let kw = match kind {
                UnKind::Exp => "pto.texp",
                UnKind::Neg => "pto.tneg",
                UnKind::Abs => "pto.tabs",
                UnKind::Sqrt => "pto.tsqrt",
                UnKind::Rsqrt => "pto.trsqrt",
                UnKind::Recip => "pto.trecip",
            };
            writeln!(
                out,
                "    {kw} ins({} : {src_ty}) outs({} : {dst_ty})",
                src.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::RowReduce { kind, dst, src } => {
            // The scratch operand isn't tracked in the AST; reuse `src` in
            // that slot so the text is well-formed and parses back to an
            // equivalent RowReduce (the parser also discards scratch).
            let src_ty = tile_ty_str(shapes, src)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            let kw = match kind {
                RowKind::Max => "pto.trowmax",
                RowKind::Sum => "pto.trowsum",
            };
            writeln!(
                out,
                "    {kw} ins({}, {} : {src_ty}, {src_ty}) outs({} : {dst_ty})",
                src.0, src.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::RowExpandSub { dst, src, red } => {
            emit_row_expand(out, "pto.trowexpandsub", dst, src, red, shapes)?;
        }
        PtoOp::RowExpandDiv { dst, src, red } => {
            emit_row_expand(out, "pto.trowexpanddiv", dst, src, red, shapes)?;
        }
        PtoOp::RowExpandMul { dst, src, red } => {
            emit_row_expand(out, "pto.trowexpandmul", dst, src, red, shapes)?;
        }
        PtoOp::RowExpand { dst, src } => {
            // Both ends come from the shapes map: `src` is the 1xK reduction
            // and `dst` the 1xN row, so neither can be derived from the other.
            let src_ty = tile_ty_str(shapes, src)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            writeln!(
                out,
                "    pto.trowexpand ins({} : {src_ty}) outs({} : {dst_ty})",
                src.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::ColExpand { dst, src } => {
            let src_ty = tile_ty_str(shapes, src)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            writeln!(
                out,
                "    pto.tcolexpand ins({} : {src_ty}) outs({} : {dst_ty})",
                src.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::ColSum { dst, src } => {
            let src_ty = tile_ty_str(shapes, src)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            writeln!(
                out,
                "    pto.tcolsum ins({} : {src_ty}) outs({} : {dst_ty})",
                src.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::TMatmul { dst, lhs, rhs } => {
            let l_ty = tile_ty_str(shapes, lhs)?;
            let r_ty = tile_ty_str(shapes, rhs)?;
            let d_ty = tile_ty_str(shapes, dst)?;
            writeln!(
                out,
                "    pto.tmatmul ins({}, {} : {l_ty}, {r_ty}) outs({} : {d_ty})",
                lhs.0, rhs.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::Transpose { dst, src, scratch } => {
            let src_ty = tile_ty_str(shapes, src)?;
            let scr_ty = tile_ty_str(shapes, scratch)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            writeln!(
                out,
                "    pto.ttrans ins({}, {} : {src_ty}, {scr_ty}) outs({} : {dst_ty})",
                src.0, scratch.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::Unmodelled { text, .. } => {
            // Byte-for-byte, which is the whole point of the variant: an op with
            // no AST meaning has no canonical form to re-derive it from, so the
            // only faithful re-emission is the one the parser read.
            //
            // Re-indented to this emitter's two spaces rather than echoed with
            // the original leading whitespace, because the parser trims and the
            // round trip compares ASTs, not bytes -- carrying indentation into
            // `text` would make `before == after` depend on the input's layout.
            writeln!(out, "    {text}").unwrap();
        }
        PtoOp::TMov { dst, src } => {
            let src_ty = tile_ty_str(shapes, src)?;
            let dst_ty = tile_ty_str(shapes, dst)?;
            writeln!(
                out,
                "    pto.tmov ins({} : {src_ty}) outs({} : {dst_ty})",
                src.0, dst.0,
            )
            .unwrap();
        }
        PtoOp::Softmax { .. } => {
            // Synthetic node: expanding it back into the 5-op chain would
            // require materialising four fresh intermediate SSAs with shapes
            // we didn't preserve. The verifier runs on the PRE-FUSION AST,
            // so a round-trip should never see this variant in practice --
            // flag it if we do.
            return Err("emitter_pto: cannot serialize synthetic Softmax; run the \
                 verifier on the pre-fusion AST"
                .into());
        }
        PtoOp::RmsNorm { .. } => {
            // Same reasoning as `Softmax` above, and more so: putting the
            // eight-op chain back would need six fresh SSAs, two of them at
            // shapes (the 8x1 reduce output and the 1x8 scaled form) that the
            // fused node does not carry.
            return Err("emitter_pto: cannot serialize synthetic RmsNorm; run the \
                 verifier on the pre-fusion AST"
                .into());
        }
        PtoOp::Quantize { .. } => {
            // Same reasoning: rebuilding the expanded quantise chain would
            // need the expanded scale tile, the divide result and the f16
            // intermediate, none of which the fused node carries. The
            // verifier runs on the pre-fusion AST.
            return Err("emitter_pto: cannot serialize synthetic Quantize; run the \
                 verifier on the pre-fusion AST"
                .into());
        }
    }
    Ok(())
}

fn emit_row_expand(
    out: &mut String,
    kw: &str,
    dst: &Ssa,
    src: &Ssa,
    red: &Ssa,
    shapes: &BTreeMap<Ssa, TileShape>,
) -> Result<(), String> {
    let src_ty = tile_ty_str(shapes, src)?;
    let red_ty = tile_ty_str(shapes, red)?;
    let dst_ty = tile_ty_str(shapes, dst)?;
    writeln!(
        out,
        "    {kw} ins({}, {} : {src_ty}, {red_ty}) outs({} : {dst_ty})",
        src.0, red.0, dst.0,
    )
    .unwrap();
    Ok(())
}

fn format_type(ty: &PtoType) -> String {
    match ty {
        PtoType::Ptr(dt) => format!("!pto.ptr<{}>", dt.pto_spelling()),
        PtoType::TensorView { rank, dtype } => format!(
            "!pto.tensor_view<{}x{}>",
            vec!["?"; *rank as usize].join("x"),
            dtype.pto_spelling()
        ),
        PtoType::PartitionView { rows, cols, dtype } => {
            format!(
                "!pto.partition_tensor_view<{rows}x{cols}x{}>",
                dtype.pto_spelling()
            )
        }
        PtoType::PartitionViewNd { dims, dtype } => {
            let d: Vec<String> = dims.iter().map(|x| x.to_string()).collect();
            format!(
                "!pto.partition_tensor_view<{}x{}>",
                d.join("x"),
                dtype.pto_spelling()
            )
        }
        PtoType::TileBuf {
            rows,
            cols,
            v_rows,
            v_cols,
            dtype,
            loc,
        } => {
            // Canonical form: only the fields the parser inspects. `loc` is one
            // of them now -- it used to be a hardcoded `vec`, and `v_row`/`v_col`
            // joined for the same reason on 2026-11-09.
            //
            // They are emitted ONLY when they differ from `rows`/`cols`, so every
            // unpadded tile keeps exactly the text it had and no stored artefact
            // moves. When they do differ the omission was LOSSY and the round-trip
            // could not see it: `rows=8, cols=1, v_row=1, v_col=1` re-emitted as
            // `rows=8, cols=1` loses the only record of which extent is logical,
            // and that extent is what the lifted Rust type must name.
            if v_rows == rows && v_cols == cols {
                format!(
                    "!pto.tile_buf<loc={lc}, dtype={dt}, rows={rows}, cols={cols}>",
                    lc = loc.as_str(),
                    dt = dtype.pto_spelling(),
                )
            } else {
                format!(
                    "!pto.tile_buf<loc={lc}, dtype={dt}, rows={rows}, cols={cols}, \
                     v_row={v_rows}, v_col={v_cols}>",
                    lc = loc.as_str(),
                    dt = dtype.pto_spelling(),
                )
            }
        }
    }
}

fn tile_ty_str(shapes: &BTreeMap<Ssa, TileShape>, s: &Ssa) -> Result<String, String> {
    let sh = shape_of(shapes, s)?;
    // No default: a tile whose `loc` is unknown is a partition view used where
    // a tile belongs, and writing `vec` would hide that AND relocate the tile.
    let loc = sh.loc.ok_or_else(|| {
        format!(
            "`{}` has a shape but no tile location; it is a view, not a tile_buf",
            s.0
        )
    })?;
    Ok(format!(
        "!pto.tile_buf<loc={lc}, dtype={dt}, rows={r}, cols={c}>",
        lc = loc.as_str(),
        dt = sh.dtype.pto_spelling(),
        r = sh.rows,
        c = sh.cols,
    ))
}

fn shape_of(shapes: &BTreeMap<Ssa, TileShape>, s: &Ssa) -> Result<TileShape, String> {
    shapes
        .get(s)
        .copied()
        .ok_or_else(|| format!("no shape for SSA `{}`", s.0))
}
