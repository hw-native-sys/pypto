// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Conservative PTO-MLIR -> tile-rs DSL lift, at the INTRINSIC level.
//!
//! This path deliberately accepts a small static subset. PTO can address tiles
//! through offsets and reuse an SSA value several times. This checked lift
//! refuses those cases rather than reconstructing a different kernel. An
//! explicit unsafe pointer boundary preserves the source's zero-offset ABI.
//!
//! Two things it is NOT, both measured 2026-11-01 and both read the other way
//! in this crate's own docs until then:
//!
//! * It emits no `Tile<R, C, T>` at all -- `extern "C"` intrinsics over `u32`
//!   buffer handles, extents as runtime `u32` arguments. The strictness is
//!   THIS module's, applied while lifting; nothing in the output is checked by
//!   rustc. `emitter.rs` is the path whose shapes live in the type system.
//! * Its output cannot be type-checked here. Every kernel it emits carries
//!   `#[tile_std::tile_kernel]`, whose expansion includes the codegen marker
//!   `#[tile::kernel]` (`tile_std_macros/src/lib.rs`), so stock rustc fails
//!   with `E0433: unresolved module `tile``. The repo's own hand-written
//!   kernels fail identically, which is the control that makes this the form's
//!   property rather than a bug here. See `tests/two_lift_paths.rs`.

use crate::ast::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write;

/// The licence header every emitted `.rs` carries.
///
/// Emitted here, not prepended afterwards: these files are generated, and a
/// header added by hand is stripped by the next regeneration. That is the drift
/// the corpus's byte-identity gates exist to catch, and `check_headers` requires
/// this exact block on every `.rs` in the tree.
pub const LICENSE_HEADER: &str = "\
// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the \"License\").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN \"AS IS\" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
";

pub fn emit_source(source: &str) -> Result<String, String> {
    let functions = split_functions(source)?;
    let mut out = String::from(LICENSE_HEADER);
    out.push_str(
        "// Synthesised from PTO-MLIR. Review before compiling or running.\n\
         #![feature(no_core)]\n#![no_std]\n#![no_core]\n\n\
         extern crate tile_std;\n\
         use tile_std::tile::{__tile_add_f32, __tile_div_f32, __tile_exp_f32, __tile_load_f32, __tile_matmul_f32, __tile_mul_f32, __tile_neg_f32, __tile_softmax_f32, __tile_store_f32, __tile_sub_f32};\n\n",
    );
    let mut function_names = BTreeSet::new();
    for source in functions {
        // The permissive corpus parser has module-wide SSA maps. Parse each
        // function independently so reused local names cannot overwrite another
        // function's offset constants or tile shapes.
        let mut m = crate::parser::parse_module(&source)?;
        if m.funcs.len() != 1 {
            return Err("safe PTO lift expected exactly one parsed function per body".into());
        }
        reject_unpreserved_source(&m, &source)?;
        crate::fuse::fuse_patterns(&mut m);
        validate_active_tile_extents(&m, &source)?;
        let f = &m.funcs[0];
        if !function_names.insert(rust_ident(&f.name)) {
            return Err(format!(
                "{}: function name collides after Rust identifier conversion",
                f.name
            ));
        }
        emit_func(&mut out, f, &m)?;
        out.push('\n');
    }
    Ok(out)
}

/// Accept only the straight-line, one-statement-per-line source grammar that
/// this lift can preserve. Inspect the opcode, never a PTO type mentioned by an
/// unrelated operation (a func.call with !pto.ptr operands still has effects).
pub(crate) fn split_functions(source: &str) -> Result<Vec<String>, String> {
    let mut functions = Vec::new();
    let mut current: Option<String> = None;
    let mut module_open = false;
    let mut module_closed = false;
    let mut returned = false;
    let mut definitions = BTreeSet::new();
    for (idx, original) in source.lines().enumerate() {
        let line = original.split("//").next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fail = |why: &str| format!("line {}: {why}", idx + 1);
        if current.is_none() {
            if line == "module {" && !module_open && !module_closed && functions.is_empty() {
                module_open = true;
            } else if line == "}" && module_open && !module_closed {
                module_closed = true;
            } else if line.starts_with("func.func @") && !module_closed {
                let tail = line.rsplit_once(')').map(|(_, tail)| tail.trim());
                if tail != Some("{") || line.contains("->") || line.contains("attributes") {
                    return Err(fail("safe PTO lift requires a one-line void function signature without attributes"));
                }
                current = Some(format!("{line}\n"));
                returned = false;
                definitions.clear();
            } else {
                return Err(fail(
                    "statement outside the supported PTO function/module grammar",
                ));
            }
            continue;
        }
        if line == "}" {
            if !returned {
                return Err(fail("safe PTO lift requires a final void return"));
            }
            let mut body = current.take().unwrap();
            body.push_str("}\n");
            functions.push(body);
            continue;
        }
        if returned {
            return Err(fail(
                "statements after return cannot be preserved by the safe PTO lift",
            ));
        }
        if line == "return" {
            returned = true;
        } else {
            let op = opcode(line).ok_or_else(|| fail("malformed PTO statement"))?;
            if operation_tokens(line) != [op] {
                return Err(fail(
                    "safe PTO lift requires exactly one operation per line",
                ));
            }
            if !matches!(
                op,
                "arith.constant"
                    | "pto.make_tensor_view"
                    | "pto.partition_view"
                    | "pto.alloc_tile"
                    | "pto.tload"
                    | "pto.tstore"
                    | "pto.tadd"
                    | "pto.tsub"
                    | "pto.tmul"
                    | "pto.tdiv"
                    | "pto.texp"
                    | "pto.tneg"
                    | "pto.trowmax"
                    | "pto.trowsum"
                    | "pto.trowexpandsub"
                    | "pto.trowexpanddiv"
                    | "pto.tmatmul"
                    | "pto.tmov"
            ) {
                return Err(fail(&format!(
                    "operation `{op}` is outside the safe PTO lift subset"
                )));
            }
            if line.contains(['{', '}', ';', '"']) {
                return Err(fail(
                    "nested regions, attributes, and multiple statements are unsupported",
                ));
            }
            if let Some((dst, _)) = line.split_once(" = ") {
                if !definitions.insert(dst.to_string()) {
                    return Err(fail("duplicate SSA definition in function"));
                }
            }
        }
        let body = current.as_mut().unwrap();
        body.push_str(line);
        body.push('\n');
    }
    if current.is_some() || (module_open && !module_closed) {
        return Err("safe PTO lift requires terminated function and module bodies".into());
    }
    if functions.is_empty() {
        return Err(
            "safe PTO lift found no kernel functions; empty implementations are refused".into(),
        );
    }
    Ok(functions)
}

fn opcode(line: &str) -> Option<&str> {
    let rest = if line.starts_with('%') {
        line.split_once(" = ")?.1
    } else {
        line
    };
    rest.split_whitespace().next()
}

fn operation_tokens(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_alphabetic() && bytes[i] != b'_' {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len()
            && (bytes[i].is_ascii_alphanumeric() || matches!(bytes[i], b'_' | b'.' | b'$' | b'-'))
        {
            i += 1;
        }
        // Type names and SSA/symbol identifiers are not executable opcodes.
        if (start == 0 || !matches!(bytes[start - 1], b'!' | b'%' | b'@'))
            && line[start..i].contains('.')
        {
            tokens.push(&line[start..i]);
        }
    }
    tokens
}

fn reject_unpreserved_source(m: &PtoModule, source: &str) -> Result<(), String> {
    let mut tensor_shapes = HashMap::new();
    for (idx, line) in source.lines().enumerate() {
        let trimmed = line.trim();
        if let Some(op) = opcode(trimmed) {
            if op == "pto.partition_view" {
                let offsets = bracket_values(trimmed, "offsets = [").ok_or_else(|| {
                    format!("line {}: partition_view has no readable offsets", idx + 1)
                })?;
                if offsets.len() != 2
                    || offsets
                        .iter()
                        .any(|value| parse_index(value, &m.consts) != Some(0))
                {
                    return Err(format!(
                        "line {}: partition_view has non-literal-zero or unrecognized offsets; the safe lift cannot represent subviews",
                        idx + 1
                    ));
                }
                let sizes = bracket_values(trimmed, "sizes = [").ok_or_else(|| {
                    format!("line {}: partition_view has no readable sizes", idx + 1)
                })?;
                let sizes: Vec<_> = sizes.iter().map(|v| parse_index(v, &m.consts)).collect();
                let dst = Ssa(trimmed.split_once(" = ").unwrap().0.to_string());
                let partition = m.funcs[0]
                    .ops
                    .iter()
                    .find_map(|op| match op {
                        PtoOp::PartitionView {
                            dst: id,
                            src,
                            rows,
                            cols,
                            ..
                        } if id == &dst => Some((src, *rows, *cols)),
                        _ => None,
                    })
                    .ok_or_else(|| format!("line {}: partition_view was not parsed", idx + 1))?;
                let (parent, rows, cols) = partition;
                if rows == 0
                    || cols == 0
                    || sizes != [Some(i64::from(rows)), Some(i64::from(cols))]
                    || tensor_shapes.get(parent) != Some(&(i64::from(rows), i64::from(cols)))
                {
                    return Err(format!("line {}: partition sizes/type must equal the complete parent tensor shape; partial or out-of-bounds views cannot be lifted contiguously", idx + 1));
                }
            }
            if op == "pto.make_tensor_view" {
                let shape = bracket_values(trimmed, "shape = [").ok_or_else(|| {
                    format!("line {}: tensor_view has no readable static shape", idx + 1)
                })?;
                let strides = bracket_values(trimmed, "strides = [").ok_or_else(|| {
                    format!("line {}: tensor_view has no readable strides", idx + 1)
                })?;
                let shape: Vec<_> = shape
                    .iter()
                    .map(|value| parse_index(value, &m.consts))
                    .collect();
                let strides: Vec<_> = strides
                    .iter()
                    .map(|value| parse_index(value, &m.consts))
                    .collect();
                if shape.len() != 2
                    || strides.len() != 2
                    || shape.iter().any(Option::is_none)
                    || shape
                        .iter()
                        .any(|value| value.is_some_and(|v| v <= 0 || v > i64::from(u32::MAX)))
                    || strides[1] != Some(1)
                    || strides[0] != shape[1]
                {
                    return Err(format!(
                        "line {}: tensor_view is not a statically sized row-major 2D view; the safe lift cannot preserve its addressing",
                        idx + 1
                    ));
                }
                let dst = Ssa(trimmed.split_once(" = ").unwrap().0.to_string());
                tensor_shapes.insert(dst, (shape[0].unwrap(), shape[1].unwrap()));
            }
        }
    }
    Ok(())
}

/// Refuse a tile whose VALID extent differs from its allocated extent.
///
/// Shared with the default lift path, which is the whole point. `rows`/`cols`
/// on a `pto.tile_buf` is the ALLOCATED extent -- padded for the 512-byte
/// fractal, and for a row-reduce output it can be `rows=8, cols=1` where the
/// logical result is one value, `v_row=1, v_col=1`. The tile API has no notion
/// of a padded allocation: `Tile<R, C, T>` is the logical shape. So an emitter
/// that reads `rows`/`cols` for its const generics emits
/// `tile_store_f32::<8, 1>` for a `Tile<1, 1>` value.
///
/// That is not a theory. Until this check was shared, the default path emitted
/// exactly that for 9 of the 32 artifacts it claimed to lift, every one a
/// matvec or projection kernel where a reduce feeds a store, and all 9 failed
/// with `E0308: expected 8, found 1` the first time anything compiled the
/// output. The checked path had refused them all along, so the two paths
/// disagreed about the same file and only the stricter one was right.
///
/// It refuses rather than substituting `v_row`/`v_col` into the const
/// generics. Emitting the logical extent would probably be correct and would
/// make these lift, but "probably" is doing real work in that sentence: a
/// store's extent decides how many elements reach memory, and nothing in this
/// tree can yet check the result on hardware. A named refusal is worth more
/// than a kernel that compiles and writes the wrong count.
///
/// THE "PROBABLY" IS NOW MEASURED, 2026-11-05, and it has two halves.
///
/// Counted: 14 of the 34 scored artefacts refuse HERE and nowhere else, and all
/// 14 carry exactly one such tile, always `rows=8, cols=1, v_row=1, v_col=1` --
/// a row-reduce result padded to one fractal. In 12 of them every consumer's
/// partition view is 1x1, so the element count the store moves and the logical
/// extent agree.
///
/// And the padding cannot simply be dropped: rewriting that tile to `rows=1` and
/// re-assembling fails with
///
/// `'pto.alloc_tile' op expects result col-major none_box tile column byte size
/// (rows * sizeof(dtype)) to be 32-byte aligned`
///
/// so the PHYSICAL extent is mandatory on re-emit while the LOGICAL one is what
/// the Rust type must say. That is a two-part change, not a substitution: the
/// const generics take `v_row`/`v_col`, and the re-emitted `alloc_tile` keeps the
/// padded type the artefact already carried. The second half is why this is still
/// a refusal -- doing only the first would emit a kernel that no longer
/// assembles.
pub(crate) fn validate_active_tile_extents(m: &PtoModule, source: &str) -> Result<(), String> {
    // The FIRST function only, and no indexing.
    //
    // `m.funcs[0]` panicked with "len is 0" the first time this ran on the
    // default path's inputs, which reach it before the checked path's own
    // non-empty precondition -- hence the guard.
    //
    // Widening it to every function looks strictly better and is not. The
    // default path parses a module with MODULE-WIDE SSA maps, so a local name
    // reused in a second function overwrites the first's shape entry; walking
    // all functions then asks for the shape of a clobbered `%pto2` and refuses
    // a six-kernel fixture with "has no static shape", which is a parser
    // limitation reported as a tile defect. The checked path splits the source
    // per function before calling this, which is where a multi-function file
    // gets every tile checked. Narrowing is the honest option until the
    // default path's parser is function-scoped too.
    // ONE function, or nothing. A module-wide parse cannot attribute a tile
    // shape in a multi-function file at all: `%pto2` is a local name that six
    // of the six_kernels bodies each reuse, so the last one parsed wins and
    // `shape_of` reports "no static shape" for a tile that is perfectly well
    // shaped in the function being checked. Narrowing to the first function
    // was not enough -- the TEXT walk below still reads every body's
    // `alloc_tile` lines. The checked path splits per function before calling
    // this and so validates all of them; here the precondition is a single
    // function, stated rather than worked around.
    let [first] = &m.funcs[..] else {
        return Ok(());
    };
    let active: BTreeSet<_> = first
        .ops
        .iter()
        .flat_map(|op| {
            let mut values = op_inputs(op);
            values.extend(op_output(op));
            values.into_iter().cloned()
        })
        .collect();
    // What each binding's `Tile<R, C, T>` TYPE is, which is not always the
    // allocation: a row reduction returns `ROWS x 1` from an `8x8, v_col=1`
    // buffer. Consumers that name this shape are exact, so the padding is
    // outside the value and there is nothing for this check to refuse.
    let value_shapes = crate::emitter::value_shapes(&first.ops, &m.shapes);
    for line in source.lines() {
        if opcode(line.trim()) != Some("pto.alloc_tile") {
            continue;
        }
        let (dst, _) = line.trim().split_once(" = ").unwrap();
        let id = Ssa(dst.to_string());
        if !active.contains(&id) {
            continue;
        }
        let shape = shape_of(m, &id)?;
        if shape.rows == 0 || shape.cols == 0 {
            return Err(format!("tile `{dst}` has an empty extent"));
        }
        // A padded tile is SAFE when every consumer of it is a store, because
        // the store now names the logical extent (`emitter.rs`, `PtoOp::TStore`)
        // and the PTO view it writes through is the logical shape too. Every
        // other op still takes its const generics from the allocated extent, so
        // the refusal stands for those -- narrowed, not removed.
        //
        // Measured 2026-11-09: this is the whole of the largest refusal class.
        // All of its members are matvec or projection kernels where a row-reduce
        // feeds a store and nothing else reads the reduce output.
        // The condition is NOT merely "consumed only by a store". A first
        // version of this narrowing said that and was wrong, caught by
        // `checked_lift::partial_active_tiles_are_refused`: its fixture is a
        // load feeding a store, and the `--emit tile` path writes the store's
        // extent from the VIEW as runtime `u32` args, so a 2x4 view over a
        // logically smaller tile still moved eight elements. Narrowing on the
        // consumer alone would have let exactly the defect through on one of the
        // two paths while fixing the other.
        //
        // What makes it safe is that the store's view AGREES with the tile's
        // logical extent -- then the element count the store moves and the type
        // the Rust names are the same number, whichever path emits it. That is
        // the condition the measurement actually found: all members of this
        // class carry a `rows=8, cols=1, v_row=1, v_col=1` reduce output whose
        // consumer view is 1x1.
        let stored_at_its_logical_extent = {
            let mut seen_any = false;
            let ok = first.ops.iter().all(|op| {
                if !op_inputs(op).into_iter().any(|v| *v == id) {
                    return true;
                }
                seen_any = true;
                match op {
                    PtoOp::TStore { view, .. } => match shape_of(m, view) {
                        Ok(v) => v.rows == shape.v_rows && v.cols == shape.v_cols,
                        Err(_) => false,
                    },
                    // `pto.trowexpand` is the OTHER consumer that cannot be hurt by
                    // the padding, and for a stronger reason than the store's.
                    // Its emission takes a SINGLE const generic from the
                    // DESTINATION (`tile_row_expand_f32::<{dst.cols}>`), so the
                    // source's allocated extent never reaches the generics, and
                    // `tile_std` declares the parameter as `src: Tile<1, 1, f32>`
                    // -- which IS the padded tile's logical extent. So the
                    // exemption is not merely safe here, it is required for the
                    // types to line up: the `trowsum` ahead of it returns
                    // `Tile<1, 1>` and that is exactly what row-expand consumes.
                    // The arm independently checks the source has one column.
                    //
                    // Added 2026-11-09 for the `kda_delta` pair, whose `%pto14` is
                    // a `rows=8, cols=1, v_row=1, v_col=1` reduce output feeding a
                    // row-expand rather than a store.
                    PtoOp::RowExpand { .. } => shape.v_rows == 1 && shape.v_cols == 1,
                    // The FUSED row-expansions, exempt on the same ground as
                    // the bare one but ONLY as the reduction operand. Added
                    // 2026-11-09.
                    //
                    // When `red` is logical 1x1 the Rust lift emits
                    // `tile_get_value_f32(red, 0)` -- a FLAT index 0 -- so
                    // neither the allocated extent nor any padded element is
                    // read, and none of it reaches a const generic.
                    //
                    // The `*v == id` test is load-bearing and is why this is not
                    // folded into the arm above. The same tile may appear as
                    // `src`, where the whole logical extent IS multiplied and
                    // the padding is NOT harmless; exempting it there would
                    // reintroduce the wrong-pair defect this check exists for.
                    // `op_inputs` has already told us `id` is one of the two.
                    PtoOp::RowExpandSub { red, .. }
                    | PtoOp::RowExpandDiv { red, .. }
                    | PtoOp::RowExpandMul { red, .. } => {
                        *red == id && shape.v_rows == 1 && shape.v_cols == 1
                    }
                    // NOT EXEMPT, and the reason is worth keeping, because the
                    // obvious extension from here is wrong. The elementwise
                    // links of a scalar chain (`playground16/rms_norm.pto`:
                    // trowsum -> tmuls -> tadds -> trsqrt -> trowexpandmul) all
                    // carry a logical-1x1 tile allocated 32 bytes, and
                    // exempting them looks safe -- each op is per-lane, so flat
                    // index 0 cannot be changed by what the other 7 lanes hold.
                    //
                    // It was tried on 2026-11-09 and REVERTED, because the
                    // padding is not the problem. The two provenances of a
                    // logical-1x1 tile emit DIFFERENT Rust types:
                    //
                    //   a `pto.tload` of a 1x1 view  -> `tile_load_f32::<1, 8>`
                    //        -> Tile<1, 8>, because a 1x1 f32 load is 4 bytes
                    //           and must be widened to the 32-byte block
                    //   a `pto.trowsum`              -> `tile_reduce_sum_f32`
                    //        -> Tile<1, 1>, the logical extent
                    //
                    // Consumer arms take their const generics from `shape_of`,
                    // which reports the ALLOCATED extent, so the reduce-produced
                    // chain lifts to `tile_scale_f32::<1, 8>` applied to a
                    // `Tile<1, 1>`: rustc says "expected `8`, found `1`". Using
                    // the logical extent instead just moves the mismatch onto
                    // the tload-produced chain, which is the majority case.
                    //
                    // The real fix is for the emitter to track each name's
                    // EMITTED type rather than re-derive it per arm. Until then
                    // this refuses, which is the conservative direction.
                    _ => false,
                }
            });
            seen_any && ok
        };
        // A partial tile is also safe when its VALUE shape IS its logical
        // extent -- the reduction-produced scalar chain (`trowsum` 8x1 ->
        // `tmuls` -> `tadds` -> `trsqrt`, all `Tile<8, 1>` over an `8x8`
        // allocation with `v_col=1`) -- provided every consumer emits against
        // that value shape. The consumers below are exactly the arms that now
        // take their const generics from `value_shapes`; a store is safe when
        // the view it writes through is the logical extent, the same condition
        // the store exemption above already uses. Any other consumer keeps the
        // refusal, because it would re-derive the ALLOCATION and hand rustc a
        // `Tile<8, 8>` where the value is `Tile<8, 1>`.
        let value_is_logical = value_shapes.get(&id).copied()
            == Some((shape.v_rows as usize, shape.v_cols as usize));
        let consumers_name_the_value = value_is_logical
            && first.ops.iter().all(|op| {
                if !op_inputs(op).into_iter().any(|v| *v == id) {
                    return true;
                }
                match op {
                    PtoOp::BinOpScalar { a, .. } => *a == id,
                    PtoOp::UnOp { src, .. } => *src == id,
                    PtoOp::RowExpandSub { red, .. }
                    | PtoOp::RowExpandDiv { red, .. }
                    | PtoOp::RowExpandMul { red, .. } => *red == id,
                    PtoOp::TStore { view, .. } => shape_of(m, view)
                        .map(|v| v.rows == shape.v_rows && v.cols == shape.v_cols)
                        .unwrap_or(false),
                    _ => false,
                }
            });
        if stored_at_its_logical_extent || consumers_name_the_value {
            continue;
        }
        for (key, expected) in [("v_row", shape.rows), ("v_col", shape.cols)] {
            for attr in line.split([',', '<', '>']) {
                if let Some((name, value)) = attr.trim().split_once('=') {
                    if name.trim() == key && value.trim().parse::<u32>().ok() != Some(expected) {
                        return Err(format!("tile `{dst}` has a partial or dynamic {key}; the lift requires the full logical tile"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn bracket_values<'a>(line: &'a str, label: &str) -> Option<Vec<&'a str>> {
    let start = line.find(label)? + label.len();
    let end = line[start..].find(']')? + start;
    Some(line[start..end].split(',').map(str::trim).collect())
}

fn parse_index(value: &str, consts: &BTreeMap<Ssa, i64>) -> Option<i64> {
    if let Ok(value) = value.parse::<i64>() {
        return Some(value);
    }
    consts.get(&Ssa(value.to_string())).copied()
}

#[derive(Clone)]
struct ViewUse {
    ptr: Ssa,
    rows: u32,
    cols: u32,
    dtype: DType,
    reads: bool,
    writes: bool,
}

fn emit_func(out: &mut String, f: &PtoFunc, m: &PtoModule) -> Result<(), String> {
    // Looks inside regions. A kernel whose only `tstore` sits in an `scf.for`
    // body is not an empty implementation, and reporting it as one would send
    // the reader to the wrong place -- the loop is refused a few lines below,
    // which is the diagnostic they can act on.
    if !has_store(&f.ops) {
        return Err(format!(
            "{}: kernel has no output store; empty implementations are refused",
            f.name
        ));
    }
    let ptr_args: Vec<_> = f
        .args
        .iter()
        .filter_map(|(ssa, ty)| matches!(ty, PtoType::Ptr(_)).then_some(ssa.clone()))
        .collect();
    if ptr_args.len() != f.args.len() {
        return Err(format!(
            "{}: only pointer parameters are supported by the safe PTO lift",
            f.name
        ));
    }
    let mut definitions = BTreeSet::new();
    for (id, _) in &f.args {
        if !definitions.insert(id.clone()) {
            return Err(format!(
                "{}: duplicate function argument `{}`",
                f.name, id.0
            ));
        }
    }
    let mut producers = BTreeSet::new();
    for op in &f.ops {
        let defined = match op {
            PtoOp::ScalarConst { dst, .. }
            | PtoOp::MakeTensorView { dst, .. }
            | PtoOp::PartitionView { dst, .. }
            | PtoOp::AllocTile { dst, .. } => Some(dst),
            _ => None,
        };
        if let Some(id) = defined {
            if !definitions.insert(id.clone()) {
                return Err(format!("{}: duplicate SSA definition `{}`", f.name, id.0));
            }
        }
        if let Some(id) = op_output(op) {
            if !producers.insert(id.clone()) {
                return Err(format!(
                    "{}: tile `{}` has multiple producers; mutable tile reuse is outside this lift",
                    f.name, id.0
                ));
            }
        }
    }

    let partition = partition_views(f)?;

    let mut arg_info = BTreeMap::<Ssa, (u32, u32, DType, bool)>::new();
    let mut used_views = BTreeSet::new();
    for (view_id, info) in &partition {
        if !info.reads && !info.writes {
            continue;
        }
        used_views.insert(view_id.clone());
        if info.reads && info.writes {
            return Err(format!("{}: pointer `{}` is both read and written; in-place pointer access is outside the checked lift subset", f.name, info.ptr.0));
        }
        if info.dtype != DType::F32 {
            return Err(format!(
                "{}: safe PTO lift currently supports f32 GM views, got {:?}",
                f.name, info.dtype
            ));
        }
        let item = (info.rows, info.cols, info.dtype, info.writes);
        if let Some(old) = arg_info.insert(info.ptr.clone(), item) {
            if old != item {
                return Err(format!(
                    "{}: pointer `{}` has multiple shapes or access roles",
                    f.name, info.ptr.0
                ));
            }
        }
    }
    if arg_info.len() != ptr_args.len() {
        return Err(format!(
            "{}: one or more pointer arguments have no uniquely shaped zero-offset partition view",
            f.name
        ));
    }

    let mut use_counts = HashMap::<Ssa, usize>::new();
    for op in &f.ops {
        for input in op_inputs(op) {
            *use_counts.entry(input.clone()).or_default() += 1;
        }
    }
    for op in &f.ops {
        if let Some(dst) = op_output(op) {
            let n = use_counts.get(dst).copied().unwrap_or(0);
            if n != 1 {
                return Err(format!("{}: tile `{}` has {n} consumers; this lift requires exactly one consumer because tile handles are linear", f.name, dst.0));
            }
        }
    }

    let mut names = HashMap::<Ssa, String>::new();
    let mut params = String::new();
    for (i, ptr) in ptr_args.iter().enumerate() {
        let (rows, cols, _, mutable) = arg_info
            .get(ptr)
            .copied()
            .ok_or_else(|| format!("{}: pointer `{}` has no safe view type", f.name, ptr.0))?;
        let arg = format!("arg{i}");
        names.insert(ptr.clone(), arg.clone());
        let ptr_ty = if mutable { "*mut" } else { "*const" };
        write!(params, "{arg}: {ptr_ty} f32, ").unwrap();
        writeln!(
            out,
            "// {arg}: contiguous {rows}x{cols} f32, {}",
            if mutable { "writable" } else { "readable" }
        )
        .unwrap();
    }
    writeln!(out, "/// # Safety\n/// Pointers must be valid for the extents listed above for the duration of the kernel.").unwrap();
    writeln!(out, "#[tile_std::tile_kernel]").unwrap();
    writeln!(
        out,
        "pub unsafe fn {}({}) {{",
        rust_ident(&f.name),
        params.trim_end_matches(", ")
    )
    .unwrap();

    for op in &f.ops {
        match op {
            PtoOp::ColExpand { .. } | PtoOp::ColSum { .. } => {
                return Err(
                    "the column expand/sum pair is lifted by the primary tile-rs emitter; \
                     this emitter refuses rather than approximating"
                        .to_string(),
                );
            }
            PtoOp::ScalarConst { .. }
            | PtoOp::MulI { .. } | PtoOp::AddI { .. }
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
            | PtoOp::Textract { .. }
            | PtoOp::AddI { .. }
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
            | PtoOp::Textract { .. }
            | PtoOp::MakeTensorView { .. }
            | PtoOp::PartitionView { .. }
            | PtoOp::AllocTile { .. } => {}
            // This emitter's contract is to refuse rather than approximate.
            // `pto.tmuls` is liftable in principle (`__tile_scale_f32`), but
            // it is not wired here yet, and saying so is the point.
            PtoOp::BlockQuery { dst, which, .. } => {
                return Err(format!(
                    "{}: `{}` for `{}` needs `{}`, which is at the tile_std crate \
                     root and not in `tile_std::tile`",
                    f.name,
                    which.pto_mnemonic(),
                    dst.0,
                    which.tile_std_path(),
                ));
            }
            PtoOp::IndexCast { dst, .. } => {
                return Err(format!(
                    "{}: `arith.index_cast` for `{}` carries a core-identity query \
                     into an offset, which the checked lift refuses",
                    f.name, dst.0
                ));
            }
            PtoOp::PartitionViewNd { dst, dims, .. } => {
                return Err(format!(
                    "{}: partition view `{}` is a rank-{} fractal/NZ layout, which \
                     has no rows/cols for the checked lift to verify",
                    f.name,
                    dst.0,
                    dims.len()
                ));
            }
            PtoOp::Cvt { dst, .. } => {
                // Wired in `emitter` (the Rust lift) as of the reconciliation
                // with `pto-rs-viewpoint-rebase`, but deliberately NOT here.
                // The checked lift's job is to hand rustc a program whose
                // shapes and tile-handle linearity it can prove; a cast changes
                // the dtype, and this emitter's `TileShape` bookkeeping tracks
                // one dtype per SSA with no provision for a value whose type
                // differs from its source's. Refusing is honest; threading a
                // dtype change through it is a change to this emitter's model,
                // not an arm.
                return Err(format!(
                    "{}: `pto.tcvt` for `{}` is not wired in the checked lift \
                     (the plain Rust lift does handle the unsaturated f16/f32/i8 pairs)",
                    f.name, dst.0
                ));
            }
            PtoOp::Transpose { dst, .. } => {
                // Same reason, one axis over: a transpose swaps rows and cols,
                // and the checked lift records one shape per SSA.
                return Err(format!(
                    "{}: `pto.ttrans` for `{}` is not wired in the checked lift \
                     (the plain Rust lift does handle the f32 case)",
                    f.name, dst.0
                ));
            }
            // UNREACHABLE BY CONSTRUCTION, and kept anyway.
            //
            // `convert_pto_to_tile_safe` text-screens every line against an
            // exact-match allowlist of 18 mnemonics before this emitter runs,
            // and that allowlist is a strict SUBSET of
            // `parser::MODELLED_OPS` -- so anything that would arrive here as
            // `Unmodelled` has already been refused with "operation `X` is
            // outside the safe PTO lift subset". Deleting this arm leaves the
            // whole suite green, which is a true statement about redundancy,
            // not a gap.
            //
            // It stays because the subset relation is the kind of thing that
            // drifts: widen the pre-screen by one row and this becomes the only
            // thing standing between an uninterpreted op and a lift that
            // silently emits nothing for it.
            // `tests/unmodelled_ops.rs::the_checked_lifts_prescreen_is_a_subset_of_what_the_parser_models`
            // asserts the relation, so a widening fails loudly instead of
            // making this arm quietly load-bearing.
            PtoOp::Unmodelled { op, .. } => {
                return Err(format!(
                    "{}: `{op}` is not modelled, so the checked lift has nothing to \
                     emit for it",
                    f.name
                ));
            }
            PtoOp::Subview { dst, .. } => {
                return Err(format!(
                    "{}: `pto.subview` for `{}` has no tile-API form, which is \
                     the whole reason and is not disputed. (This message used to \
                     add that 910B2 mis-computes vector ops on a non-zero \
                     `LocalTensor` sub-view; that claim is DISPUTED -- a device \
                     run on real 910B2 found the vectorised offset form \
                     bit-identical for one shape -- so it is no longer offered.)",
                    f.name, dst.0
                ));
            }
            PtoOp::GatherB { dst, .. } => {
                return Err(format!(
                    "{}: `pto.tgatherb` for `{}` counts offsets in 32-byte blocks \
                     on a2/a3 and in elements on a5; the part is not recorded here",
                    f.name, dst.0
                ));
            }
            PtoOp::If { cond, then_body, .. } => {
                return Err(format!(
                    "{}: `scf.if {}` has a region of {} op(s); the checked lift has \
                     no conditional form",
                    f.name,
                    cond.0,
                    then_body.len()
                ));
            }
            PtoOp::For { var, body, .. } => {
                return Err(format!(
                    "{}: `scf.for {}` has a region of {} op(s); the checked lift has \
                     no loop form",
                    f.name,
                    var.0,
                    body.len()
                ));
            }
            PtoOp::Barrier { pipe } => {
                return Err(format!(
                    "{}: `pto.barrier <{pipe}>` has no tile-API form reachable from \
                     `tile_std::tile` (`__tile_pipe_barrier` sits at the crate \
                     root), and a barrier cannot be dropped",
                    f.name
                ));
            }
            PtoOp::BinOpScalar { kind, dst, .. } => {
                let shape = shape_of(m, dst)?;
                return Err(format!(
                    "{}: {}",
                    f.name,
                    kind.unsupported_for_lift(shape.dtype)
                ));
            }
            PtoOp::TLoad { view, tile, .. } => {
                let info = partition
                    .get(view)
                    .ok_or_else(|| format!("{}: unresolved view {}", f.name, view.0))?;
                let tile_shape = shape_of(m, tile)?;
                if tile_shape.rows != info.rows
                    || tile_shape.cols != info.cols
                    || tile_shape.dtype != info.dtype
                {
                    return Err(format!(
                        "{}: load tile `{}` shape/dtype does not match view `{}`",
                        f.name, tile.0, view.0
                    ));
                }
                let arg = names
                    .get(&info.ptr)
                    .cloned()
                    .ok_or_else(|| format!("{}: unresolved pointer {}", f.name, info.ptr.0))?;
                let n = ssa_ident(tile);
                names.insert(tile.clone(), n.clone());
                let shape = shape_of(m, tile)?;
                writeln!(
                    out,
                    "    let {n} = unsafe {{ __tile_load_f32({arg}, {}u32, {}u32) }};",
                    shape.rows, shape.cols
                )
                .unwrap();
            }
            PtoOp::TStore { tile, view, .. } => {
                let info = partition
                    .get(view)
                    .ok_or_else(|| format!("{}: unresolved view {}", f.name, view.0))?;
                let tile_shape = shape_of(m, tile)?;
                if tile_shape.rows != info.rows
                    || tile_shape.cols != info.cols
                    || tile_shape.dtype != info.dtype
                {
                    return Err(format!(
                        "{}: stored tile `{}` shape/dtype does not match view `{}`",
                        f.name, tile.0, view.0
                    ));
                }
                let arg = names
                    .get(&info.ptr)
                    .ok_or_else(|| format!("{}: unresolved pointer {}", f.name, info.ptr.0))?;
                let tile = name_of(&names, tile)?;
                writeln!(
                    out,
                    "    unsafe {{ __tile_store_f32({arg}, {tile}, {}u32, {}u32) }};",
                    info.rows, info.cols
                )
                .unwrap();
            }
            PtoOp::BinOp { kind, dst, a, b } => {
                let shape = shape_of(m, dst)?;
                if shape.dtype != DType::F32
                    || !matches!(
                        kind,
                        BinKind::Add | BinKind::Sub | BinKind::Mul | BinKind::Div
                    )
                {
                    return Err(format!(
                        "{}: binary operation {:?} with dtype {:?} is not in the safe lift subset",
                        f.name, kind, shape.dtype
                    ));
                }
                if shape_of(m, a)? != shape || shape_of(m, b)? != shape {
                    return Err(format!(
                        "{}: binary op `{}` has mismatched operand/result shapes",
                        f.name, dst.0
                    ));
                }
                let func = match kind {
                    BinKind::Add => "__tile_add_f32",
                    BinKind::Sub => "__tile_sub_f32",
                    BinKind::Mul => "__tile_mul_f32",
                    BinKind::Div => "__tile_div_f32",
                    _ => unreachable!(),
                };
                let a = name_of(&names, a)?;
                let b = name_of(&names, b)?;
                let expr = format!(
                    "unsafe {{ {func}(0, {a}, {b}, {}u32, {}u32) }}",
                    shape.rows, shape.cols
                );
                emit_value_op(out, &mut names, dst, &expr);
            }
            PtoOp::UnOp { kind, dst, src } => {
                let shape = shape_of(m, dst)?;
                if shape.dtype != DType::F32 || !matches!(kind, UnKind::Exp | UnKind::Neg) {
                    return Err(format!(
                        "{}: unary operation {:?} with dtype {:?} is not in the safe lift subset",
                        f.name, kind, shape.dtype
                    ));
                }
                if shape_of(m, src)? != shape {
                    return Err(format!(
                        "{}: unary op `{}` has mismatched input/result shapes",
                        f.name, dst.0
                    ));
                }
                let func = if *kind == UnKind::Exp {
                    "__tile_exp_f32"
                } else {
                    "__tile_neg_f32"
                };
                let src = name_of(&names, src)?;
                let expr = format!(
                    "unsafe {{ {func}(0, {src}, {}u32, {}u32) }}",
                    shape.rows, shape.cols
                );
                emit_value_op(out, &mut names, dst, &expr);
            }
            PtoOp::Softmax { dst, src } => {
                let shape = shape_of(m, dst)?;
                let src_shape = shape_of(m, src)?;
                if shape.dtype != DType::F32 || shape != src_shape {
                    return Err(format!(
                        "{}: only shape-preserving f32 softmax can be lifted",
                        f.name
                    ));
                }
                let src = name_of(&names, src)?;
                let expr = format!(
                    "unsafe {{ __tile_softmax_f32(0, {src}, {}u32, {}u32) }}",
                    shape.rows, shape.cols
                );
                emit_value_op(out, &mut names, dst, &expr);
            }
            PtoOp::RowReduce { kind, .. } => {
                return Err(format!(
                    "{}: standalone row reduction {:?} has no safe lifted form yet",
                    f.name, kind
                ));
            }
            PtoOp::TMatmul { dst, lhs, rhs } => {
                let output = shape_of(m, dst)?;
                let left = shape_of(m, lhs)?;
                let right = shape_of(m, rhs)?;
                if left.dtype != DType::F32
                    || right.dtype != DType::F32
                    || output.dtype != DType::F32
                    || left.cols != right.rows
                    || output.rows != left.rows
                    || output.cols != right.cols
                {
                    return Err(format!(
                        "{}: matmul `{}` is not a compatible f32 MxK @ KxN -> MxN operation",
                        f.name, dst.0
                    ));
                }
                let a = name_of(&names, lhs)?;
                let b = name_of(&names, rhs)?;
                let expr = format!(
                    "unsafe {{ __tile_matmul_f32(0, {a}, {b}, {}u32, {}u32, {}u32) }}",
                    output.rows, left.cols, output.cols
                );
                emit_value_op(out, &mut names, dst, &expr);
            }
            // `RmsNorm` is here rather than lifted because the CHECKED lift
            // reaches `__tile_rms_norm_f32`, the raw intrinsic, and the three
            // things that makes it responsible for -- the handle discipline,
            // the eps argument's width, and the dispatch -- are unverified for
            // this op. The plain lift emits the safe wrapper, which carries
            // all three itself. Refusing a form the other emitter accepts is
            // the intended asymmetry: this one only emits what it can check.
            PtoOp::RowExpandSub { .. }
            | PtoOp::RowExpandDiv { .. }
            | PtoOp::RowExpandMul { .. }
            | PtoOp::RowExpand { .. }
            | PtoOp::RmsNorm { .. }
            // Synthetic: the checked lift parses the artifact without fusing,
            // so this node cannot reach it. Refused rather than silently
            // lowered through the plain wrapper if it ever does.
            | PtoOp::Quantize { .. }
            | PtoOp::TMov { .. } => {
                return Err(format!(
                    "{}: PTO operation {:?} has no safe lifted form yet",
                    f.name, op
                ));
            }
        }
    }
    writeln!(out, "}}").map_err(|e| format!("{}: failed to format lifted kernel: {e}", f.name))?;
    Ok(())
}

fn op_inputs(op: &PtoOp) -> Vec<&Ssa> {
    match op {
        PtoOp::TStore { tile, .. } => vec![tile],
        PtoOp::BinOp { a, b, .. } => vec![a, b],
        PtoOp::UnOp { src, .. } | PtoOp::Softmax { src, .. } | PtoOp::TMov { src, .. } => vec![src],
        PtoOp::RowReduce { src, .. } => vec![src],
        PtoOp::RowExpandSub { src, red, .. }
        | PtoOp::RowExpandDiv { src, red, .. }
        | PtoOp::RowExpandMul { src, red, .. } => {
            vec![src, red]
        }
        PtoOp::RowExpand { src, .. } | PtoOp::RmsNorm { src, .. } => vec![src],
        PtoOp::Quantize { src, scale, .. } => vec![src, scale],
        PtoOp::TMatmul { lhs, rhs, .. } => vec![lhs, rhs],
        // Added 2026-11-09. The `_` arm below had been swallowing this one, so a
        // tile read ONLY by a `pto.tmuls`/`tadds`/... counted as read by
        // nothing. That was conservative rather than unsafe --
        // `validate_active_tile_extents` needs `seen_any`, so a padded tile with
        // no visible consumer was refused rather than wrongly exempted -- but it
        // also made the rms-norm scalar chain unexemptable, since every link in
        // it is a tile-by-scalar. `ast::PtoOp::tile_reads` has always listed it.
        PtoOp::BinOpScalar { a, .. } => vec![a],
        _ => Vec::new(),
    }
}

pub(crate) fn op_output(op: &PtoOp) -> Option<&Ssa> {
    match op {
        PtoOp::TLoad { tile, .. } => Some(tile),
        PtoOp::BinOp { dst, .. }
        | PtoOp::UnOp { dst, .. }
        | PtoOp::RowReduce { dst, .. }
        | PtoOp::RowExpandSub { dst, .. }
        | PtoOp::RowExpandDiv { dst, .. }
        | PtoOp::RowExpandMul { dst, .. }
        | PtoOp::RowExpand { dst, .. }
        | PtoOp::RmsNorm { dst, .. }
        | PtoOp::Quantize { dst, .. }
        | PtoOp::TMatmul { dst, .. }
        | PtoOp::TMov { dst, .. }
        | PtoOp::Softmax { dst, .. } => Some(dst),
        _ => None,
    }
}

pub(crate) fn shape_of(m: &PtoModule, ssa: &Ssa) -> Result<TileShape, String> {
    m.shapes
        .get(ssa)
        .copied()
        .ok_or_else(|| format!("PTO value `{}` has no static shape", ssa.0))
}

fn name_of(names: &HashMap<Ssa, String>, ssa: &Ssa) -> Result<String, String> {
    names
        .get(ssa)
        .cloned()
        .ok_or_else(|| format!("unresolved PTO value `{}`", ssa.0))
}

fn emit_value_op(out: &mut String, names: &mut HashMap<Ssa, String>, dst: &Ssa, expr: &str) {
    let name = ssa_ident(dst);
    let _ = writeln!(out, "    let {name} = {expr};");
    names.insert(dst.clone(), name);
}

fn ssa_ident(ssa: &Ssa) -> String {
    let mut name = String::from("v_");
    for byte in ssa.0.trim_start_matches('%').bytes() {
        if byte.is_ascii_alphanumeric() {
            name.push(char::from(byte));
        } else {
            // Escape underscores as well, so a literal `_2e` cannot collide
            // with the encoded form of a dot in another MLIR identifier.
            write!(name, "_{byte:02x}").unwrap();
        }
    }
    name
}

fn rust_ident(name: &str) -> String {
    let id = name.replace(|c: char| !c.is_ascii_alphanumeric() && c != '_', "_");
    if id.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("k_{id}")
    } else {
        id
    }
}

/// Does this op list store anything, counting region bodies?
fn has_store(ops: &[PtoOp]) -> bool {
    ops.iter().any(|op| match op {
        PtoOp::TStore { .. } => true,
        PtoOp::For { body, .. } => has_store(body),
        // Either branch storing counts: a kernel whose only `tstore` sits in
        // one arm of a conditional is not an empty implementation. The flat
        // version of this check would have called it one.
        PtoOp::If {
            then_body,
            else_body,
            ..
        } => {
            has_store(then_body)
                || else_body.as_ref().map_or(false, |b| has_store(b))
        }
        _ => false,
    })
}


/// Which pointer each partition view reads or writes, and at what extent.
///
/// Extracted from `emit_func` so that `kernel_signatures` cannot compute this a
/// second, slightly different way. A driver that allocates buffers from one
/// reading while the kernel dereferences according to another would be a
/// measurement of nothing, and two readings drifting apart is exactly the kind of
/// thing that stays invisible while both sides are green.
fn partition_views(f: &PtoFunc) -> Result<HashMap<Ssa, ViewUse>, String> {
    let mut tensor_ptr = HashMap::<Ssa, Ssa>::new();
    let mut partition = HashMap::<Ssa, ViewUse>::new();
    for op in &f.ops {
        match op {
            // The geometry fields are ignored HERE on purpose: this emitter
            // validates `shape`/`strides`/`offsets`/`sizes` by re-reading the
            // raw source text above, which is how it checked them while the
            // AST still dropped them. Now that the AST carries them, the two
            // readings should agree, and `geometry_matches_the_text` asserts it.
            PtoOp::MakeTensorView { dst, ptr, dtype, .. } => {
                let Some((_, PtoType::Ptr(ptr_dtype))) = f.args.iter().find(|(arg, _)| arg == ptr)
                else {
                    return Err(format!(
                        "{}: tensor view `{}` does not originate at a function pointer",
                        f.name, dst.0
                    ));
                };
                if ptr_dtype != dtype {
                    return Err(format!(
                        "{}: tensor view dtype does not match pointer `{}`",
                        f.name, ptr.0
                    ));
                }
                tensor_ptr.insert(dst.clone(), ptr.clone());
            }
            PtoOp::PartitionView {
                dst,
                src,
                rows,
                cols,
                dtype,
                ..
            } => {
                let ptr = tensor_ptr.get(src).cloned().ok_or_else(|| {
                    format!(
                        "{}: partition view `{}` has no recognized tensor-view source",
                        f.name, dst.0
                    )
                })?;
                partition.insert(
                    dst.clone(),
                    ViewUse {
                        ptr,
                        rows: *rows,
                        cols: *cols,
                        dtype: *dtype,
                        reads: false,
                        writes: false,
                    },
                );
            }
            _ => {}
        }
    }

    for op in &f.ops {
        let view = match op {
            PtoOp::TLoad { view, .. } => Some((view, true)),
            PtoOp::TStore { view, .. } => Some((view, false)),
            _ => None,
        };
        if let Some((view, read)) = view {
            let info = partition.get_mut(view).ok_or_else(|| {
                format!(
                    "{}: load/store uses an unresolved partition view `{}`",
                    f.name, view.0
                )
            })?;
            if read {
                info.reads = true;
            } else {
                info.writes = true;
            }
        }
    }
    Ok(partition)
}
/// One pointer parameter of a lifted kernel, as the PTO declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelArg {
    pub rows: u32,
    pub cols: u32,
    /// `true` when the kernel writes through this pointer.
    pub writable: bool,
}

impl KernelArg {
    /// Elements a driver must allocate. EXACTLY this, never more: a generous
    /// buffer would hide the out-of-bounds access a Miri run exists to find.
    pub fn len(&self) -> usize {
        self.rows as usize * self.cols as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A lifted kernel's name and its pointer parameters, in declaration order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelSignature {
    pub name: String,
    pub args: Vec<KernelArg>,
}

/// The signatures a lift would emit, read straight from the PTO.
///
/// Exposed so a driver can allocate the right buffers WITHOUT reparsing the
/// emitted Rust, and that distinction decides what a Miri run means. Extents
/// read back out of this crate's own output would agree with it by
/// construction: a kernel whose extents were wrong would get buffers wrong in
/// the same way and pass. Taking them from the PTO keeps the two independent,
/// so the run compares what the kernel does against what the source said.
pub fn kernel_signatures(source: &str) -> Result<Vec<KernelSignature>, String> {
    // Validated through the same path that emits, so a signature is never
    // returned for a kernel the lift would refuse.
    emit_source(source)?;
    let mut out = Vec::new();
    for body in split_functions(source)? {
        let mut m = crate::parser::parse_module(&body)?;
        crate::fuse::fuse_patterns(&mut m);
        let [f] = &m.funcs[..] else {
            return Err("expected exactly one parsed function per body".into());
        };
        let partition = partition_views(f)?;
        let mut by_ptr: BTreeMap<Ssa, KernelArg> = BTreeMap::new();
        for info in partition.values() {
            if !info.reads && !info.writes {
                continue;
            }
            by_ptr.insert(
                info.ptr.clone(),
                KernelArg {
                    rows: info.rows,
                    cols: info.cols,
                    writable: info.writes,
                },
            );
        }
        // Declaration order, matching the order `emit_func` writes parameters.
        let mut args = Vec::new();
        for (ptr, _) in &f.args {
            if let Some(a) = by_ptr.get(ptr) {
                args.push(*a);
            }
        }
        if args.len() != by_ptr.len() {
            return Err(format!(
                "{}: {} view-carrying pointer(s) but {} parameter(s) matched; a driver \
                 built from this would allocate the wrong buffers",
                f.name,
                by_ptr.len(),
                args.len()
            ));
        }
        out.push(KernelSignature {
            name: rust_ident(&f.name),
            args,
        });
    }
    Ok(out)
}
