// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Parse ptoas stage-2 MLIR text (output of `PlanMemoryPass`) into a `Plan`.
//!
//! Stage-2 uses standard MLIR types (`memref<RxCxTY, strided<…>, #pto.address_space<…>>`)
//! plus `pto.pointer_cast` + `pto.bind_tile` with concrete byte offsets. The
//! grammar recognised here is a strict subset tailored to ptoas's pretty
//! printer; any unfamiliar line is skipped, which keeps the parser tolerant
//! of control-flow ops (`scf.for`, `scf.if`, `arith.muli`, …) that don't
//! reference tiles.
//!
//! Invoke like:
//! ```no_run
//! # use ptoas_rs::{parse_stage2, Plan};
//! let text = std::fs::read_to_string("stage2.mlir").unwrap();
//! let plan: Plan = parse_stage2(&text).unwrap();
//! ```

use crate::plan::*;
use std::collections::BTreeMap;

pub fn parse_stage2(src: &str) -> Result<Plan, String> {
    let mut lines = src.lines().peekable();
    let mut arch: Option<PtoArch> = None;
    let mut funcs: Vec<PlanFunc> = Vec::new();

    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if arch.is_none() {
            if let Some(a) = try_parse_module_header(line) {
                arch = Some(a);
                continue;
            }
        }
        if let Some(name) = try_parse_func_header(line) {
            let f = parse_func_body(name, &mut lines)?;
            funcs.push(f);
        }
    }

    let arch =
        arch.ok_or_else(|| "missing `module attributes { pto.target_arch = \"…\" }`".to_string())?;

    // A DUMP THAT PLACES TILES AND YIELDS NO SLOTS IS AN ERROR, NOT A PLAN.
    //
    // This is the half of the 0.68 skew that matters more than the missing
    // grammar arm. Every check in `safety` iterates `f.slots`; over an empty
    // map each one finds nothing and the report is clean. `pto-diff`'s exit
    // status is 0 by design -- "the point is to inform" -- so a caller reading
    // the status learns nothing either, and `--compare` called two plans
    // differing by 4352 bytes in one tile's address "identical".
    //
    // "Nothing is wrong" and "nothing was read" are not the same answer, and
    // only the input can tell them apart: a dump holding `alloc_tile` or
    // `bind_tile` lines has tiles to place. A genuinely tile-free kernel still
    // parses to a slotless plan, which is why the test is on the TEXT and not
    // on the slot count alone.
    let placements = src
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.starts_with("//")
                && (t.contains("pto.alloc_tile") || t.contains("pto.bind_tile"))
        })
        .count();
    let parsed: usize = funcs.iter().map(|f| f.slots.len()).sum();
    if placements > 0 && parsed == 0 {
        return Err(format!(
            "stage-2 dump holds {placements} tile placement line(s) and this parser read \
             0 slots from them, so every safety check would run over an empty plan and \
             report clean. The two spellings it knows are `pto.pointer_cast` + \
             `pto.bind_tile` (ptoas 0.26-0.29) and `pto.alloc_tile addr = %cN_i64` \
             (0.68); this dump is neither"
        ));
    }

    Ok(Plan { arch, funcs })
}

fn try_parse_module_header(line: &str) -> Option<PtoArch> {
    // `module attributes {pto.target_arch = "a5"} {`
    let tag = line.find("pto.target_arch")?;
    let rest = &line[tag..];
    let eq = rest.find('=')?;
    let after = rest[eq + 1..].trim();
    let q1 = after.find('"')?;
    let q2 = after[q1 + 1..].find('"')?;
    PtoArch::parse(&after[q1 + 1..q1 + 1 + q2])
}

fn try_parse_func_header(line: &str) -> Option<String> {
    // `func.func @name(%arg0: …) {`
    let rest = line.strip_prefix("func.func ")?;
    let at = rest.find('@')?;
    let open = rest[at..].find('(')?;
    Some(rest[at + 1..at + open].trim().to_string())
}

fn parse_func_body<'a, I: Iterator<Item = &'a str>>(
    name: String,
    lines: &mut std::iter::Peekable<I>,
) -> Result<PlanFunc, String> {
    // State shared across the body.
    let mut consts_i64: BTreeMap<Ssa, u64> = BTreeMap::new();
    // ssa -> the pto.pointer_cast info it carries (to be promoted to a slot
    // when a later `pto.bind_tile` supplies rows/cols).
    let mut pending_ptr: BTreeMap<Ssa, PointerCastInfo> = BTreeMap::new();
    let mut slots: BTreeMap<Ssa, TileSlotX> = BTreeMap::new();
    let mut ops: Vec<PlanOp> = Vec::new();
    let mut depth: i32 = 1; // we're one `{` into the module-level func body

    while let Some(raw) = lines.next() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }

        // Brace tracking — scf blocks open/close with their own braces; the
        // function ends when depth reaches 0. We don't track op nesting
        // otherwise (flattened into the func's op list).
        depth += count_char(line, '{') as i32;
        depth -= count_char(line, '}') as i32;
        if depth <= 0 {
            return Ok(PlanFunc { name, slots, ops });
        }

        // `%c…_i64 = arith.constant N : i64`
        if let Some((ssa, val)) = try_parse_i64_constant(line) {
            consts_i64.insert(ssa, val);
            continue;
        }

        // `%X = pto.pointer_cast(%cOff_i64) {config = …} : memref<RxCxTY, …, #pto.address_space<SP>>`
        if let Some((dst, info)) = try_parse_pointer_cast(line, &consts_i64)? {
            pending_ptr.insert(dst, info);
            continue;
        }

        // `%Y = pto.bind_tile %X, %cR, %cC {config = …} : memref<RxCxTY, …, #pto.address_space<SP>> -> …`
        if let Some((dst, slot)) = try_parse_bind_tile(line, &pending_ptr)? {
            slots.insert(dst, slot);
            continue;
        }

        // `%2 = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<vec, 32x1xf32, blayout=col_major>`
        // -- the ptoas 0.68 spelling. See `try_parse_alloc_tile_addr`.
        if let Some((dst, slot)) = try_parse_alloc_tile_addr(line, &consts_i64)? {
            slots.insert(dst, slot);
            continue;
        }

        // Op lines — try each keyword.
        if let Some(op) = try_parse_op_line(line)? {
            ops.push(op);
            continue;
        }

        // Everything else (reinterpret_cast, subview, scf.for, arith.muli,
        // etc.) is silently skipped: it doesn't affect slot placement.
    }

    Err(format!("function `{name}` body not terminated with `}}`"))
}

fn count_char(s: &str, c: char) -> usize {
    s.chars().filter(|&ch| ch == c).count()
}

fn try_parse_i64_constant(line: &str) -> Option<(Ssa, u64)> {
    // `%c4096_i64 = arith.constant 4096 : i64`
    let eq = line.find(" = arith.constant ")?;
    if !line.ends_with(": i64") {
        return None;
    }
    let ssa = Ssa(line[..eq].trim().to_string());
    let tail = &line[eq + " = arith.constant ".len()..];
    let val_end = tail.find(':')?;
    let val: i64 = tail[..val_end].trim().parse().ok()?;
    if val < 0 {
        return None;
    }
    Some((ssa, val as u64))
}

struct PointerCastInfo {
    offset: u64,
    space: AddressSpace,
    blayout: BLayout,
    slayout: SLayout,
    s_fractal: u32,
    // dtype captured from the memref type; rows/cols are re-captured at
    // bind_tile time (they're the same in practice but we don't rely on it)
    dtype: ExtDType,
}

fn try_parse_pointer_cast(
    line: &str,
    consts: &BTreeMap<Ssa, u64>,
) -> Result<Option<(Ssa, PointerCastInfo)>, String> {
    let eq = match line.find(" = pto.pointer_cast(") {
        Some(i) => i,
        None => return Ok(None),
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let after = &line[eq + " = pto.pointer_cast(".len()..];
    let close = after.find(')').ok_or("pointer_cast: missing `)`")?;
    let off_ssa = Ssa(after[..close].trim().to_string());
    let offset = *consts
        .get(&off_ssa)
        .ok_or_else(|| format!("pointer_cast references unknown constant `{}`", off_ssa.0))?;

    // Grab the config block (between `{config = ` and the matching `}`).
    let cfg = extract_config(line)
        .ok_or_else(|| format!("pointer_cast: missing config block: `{line}`"))?;
    let (blayout, slayout, s_fractal) = parse_tile_buf_config(&cfg)?;

    // The memref type follows `:`. We need address_space and dtype.
    let (dtype, space) = parse_memref_space_dtype(line)?;
    Ok(Some((
        dst,
        PointerCastInfo {
            offset,
            space,
            blayout,
            slayout,
            s_fractal,
            dtype,
        },
    )))
}

fn try_parse_bind_tile(
    line: &str,
    pending: &BTreeMap<Ssa, PointerCastInfo>,
) -> Result<Option<(Ssa, TileSlotX)>, String> {
    let eq = match line.find(" = pto.bind_tile ") {
        Some(i) => i,
        None => return Ok(None),
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let after = &line[eq + " = pto.bind_tile ".len()..];
    // Grab the first three comma-separated tokens before the `{` — src, rows, cols.
    let brace = after.find('{').ok_or("bind_tile: missing config block")?;
    let head = &after[..brace];
    let parts: Vec<&str> = head.split(',').map(str::trim).collect();
    if parts.len() < 3 {
        return Err(format!("bind_tile: expected `%src, %r, %c`, got `{head}`"));
    }
    let src = Ssa(parts[0].trim().to_string());
    let rows = extract_rc_from_memref(line, /*want_rows*/ true)?;
    let cols = extract_rc_from_memref(line, /*want_rows*/ false)?;

    let info = pending.get(&src).ok_or_else(|| {
        format!(
            "bind_tile `{}` references pointer_cast `{}` not seen yet",
            dst.0, src.0
        )
    })?;

    let ssa = dst.clone();
    Ok(Some((
        dst,
        TileSlotX {
            ssa,
            space: info.space,
            offset: info.offset,
            rows,
            cols,
            dtype: info.dtype,
            // Read from the `#pto.tile_buf_config<…>` block, so `Some`.
            blayout: Some(info.blayout),
            slayout: Some(info.slayout),
            s_fractal: Some(info.s_fractal),
        },
    )))
}

/// `%2 = pto.alloc_tile addr = %c8192_i64 : !pto.tile_buf<vec, 32x1xf32, blayout=col_major>`
///
/// The ptoas 0.68 spelling of a placed tile, and the reason this parser
/// reported clean results having read nothing. 0.26-0.29 printed placement as
/// `pto.pointer_cast` + `pto.bind_tile`; 0.68 prints this, with **no
/// `bind_tile` and no `pointer_cast` anywhere in the dump**. Counted on a real
/// `4_pto-plan-memory.mlir` from `ptoas 0.68`: `alloc_tile` 10, `bind_tile` 0,
/// `pointer_cast` 0 -- so `try_parse_bind_tile` returned `None` on every line
/// of every function, every check iterated an empty slot map, and `pto-diff`
/// printed `0 safety findings`. Moving one tile to force a 4 KiB overlap
/// changed nothing, and `--compare` called two plans differing by 4352 bytes
/// "identical". That is the control that proved zero slots were parsed.
///
/// One line carries everything `TileSlotX` needs except the layout fields:
/// space, offset (via the constant), rows, cols and dtype, with no
/// `pointer_cast` to correlate against. `blayout` appears as a suffix only
/// when it is not the default, and `slayout` and the fractal size do not
/// appear at all -- hence the `Option`s, rather than a guess that would make
/// the `tstore_fp` layout check pass by construction.
fn try_parse_alloc_tile_addr(
    line: &str,
    consts: &BTreeMap<Ssa, u64>,
) -> Result<Option<(Ssa, TileSlotX)>, String> {
    let eq = match line.find(" = pto.alloc_tile addr = ") {
        Some(i) => i,
        None => return Ok(None),
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let after = &line[eq + " = pto.alloc_tile addr = ".len()..];
    let colon = after
        .find(" : ")
        .ok_or_else(|| format!("alloc_tile: missing ` : ` before the type: `{line}`"))?;
    let off_ssa = Ssa(after[..colon].trim().to_string());
    // Deliberately an ERROR, not a skip. An address this parser cannot resolve
    // is a tile whose placement it does not know, and skipping it silently is
    // how a capacity or aliasing check comes to run over a subset of the plan
    // and report clean.
    let offset = *consts.get(&off_ssa).ok_or_else(|| {
        format!(
            "alloc_tile `{}` takes its address from `{}`, which is not an i64 constant \
             seen in this function",
            dst.0, off_ssa.0
        )
    })?;

    let ty = after[colon + 3..].trim();
    let inner = ty
        .strip_prefix("!pto.tile_buf<")
        .and_then(|r| r.strip_suffix('>'))
        .ok_or_else(|| format!("alloc_tile: unrecognised type `{ty}`"))?;

    // `vec, 32x1xf32` or `vec, 32x1xf32, blayout=col_major`
    let mut parts = inner.split(',').map(str::trim);
    let space_s = parts
        .next()
        .ok_or_else(|| format!("alloc_tile: no address space in `{inner}`"))?;
    let space = AddressSpace::parse(space_s)
        .ok_or_else(|| format!("alloc_tile: unknown address space `{space_s}`"))?;
    let shape = parts
        .next()
        .ok_or_else(|| format!("alloc_tile: no `RxCxTY` shape in `{inner}`"))?;
    let (rows, cols, dtype) = parse_rows_cols_dtype(shape)?;

    let mut blayout = None;
    for extra in parts {
        match extra.strip_prefix("blayout=") {
            Some(v) => {
                blayout = Some(match v {
                    "row_major" => BLayout::RowMajor,
                    "col_major" => BLayout::ColMajor,
                    _ => return Err(format!("alloc_tile: unknown blayout `{v}`")),
                })
            }
            // Refused rather than ignored: an attribute this parser does not
            // understand may be the one that decides the tile's extent, and
            // the whole point of this arm is that an unread field was being
            // treated as absent.
            None => {
                return Err(format!(
                    "alloc_tile `{}`: unrecognised type attribute `{extra}`. If 0.68+ has \
                     started printing a field here, read it -- treating it as absent is \
                     the defect this arm exists to fix",
                    dst.0
                ))
            }
        }
    }

    let ssa = dst.clone();
    Ok(Some((
        dst,
        TileSlotX {
            ssa,
            space,
            offset,
            rows,
            cols,
            dtype,
            blayout,
            slayout: None,
            s_fractal: None,
        },
    )))
}

/// `32x1xf32` -> `(32, 1, F32)`.
fn parse_rows_cols_dtype(s: &str) -> Result<(u32, u32, ExtDType), String> {
    let mut it = s.splitn(3, 'x');
    let r = it.next().ok_or_else(|| format!("bad shape `{s}`"))?;
    let c = it.next().ok_or_else(|| format!("bad shape `{s}`"))?;
    let ty = it.next().ok_or_else(|| format!("bad shape `{s}`"))?;
    let rows: u32 = r
        .trim()
        .parse()
        .map_err(|_| format!("bad row count `{r}` in `{s}`"))?;
    let cols: u32 = c
        .trim()
        .parse()
        .map_err(|_| format!("bad col count `{c}` in `{s}`"))?;
    let dtype = ExtDType::parse(ty.trim())
        .ok_or_else(|| format!("unknown dtype `{ty}` in shape `{s}`"))?;
    Ok((rows, cols, dtype))
}

fn try_parse_op_line(line: &str) -> Result<Option<PlanOp>, String> {
    // tload / tstore / tstore_fp / tmov first so they don't get eaten by
    // the generic unary/binary match below.
    if line.starts_with("pto.tload ") {
        let outs = single_ssa_from_outs(line)?;
        return Ok(Some(PlanOp::TLoad { tile: outs }));
    }
    if line.starts_with("pto.tstore_fp ") {
        let (acc, scaling) = two_ssa_from_ins(line)?;
        return Ok(Some(PlanOp::TStoreFp { acc, scaling }));
    }
    if line.starts_with("pto.tstore ") {
        let tile = single_ssa_from_ins(line)?;
        return Ok(Some(PlanOp::TStore { tile }));
    }
    if line.starts_with("pto.tmov ") {
        let src = single_ssa_from_ins(line)?;
        let dst = single_ssa_from_outs(line)?;
        return Ok(Some(PlanOp::TMov { src, dst }));
    }
    if line.starts_with("pto.tmatmul.acc") {
        let (acc_in, lhs, rhs) = three_ssa_from_ins(line)?;
        let acc_out = single_ssa_from_outs(line)?;
        return Ok(Some(PlanOp::TMatmulAcc {
            acc_in,
            lhs,
            rhs,
            acc_out,
        }));
    }
    if line.starts_with("pto.tmatmul") {
        let (lhs, rhs) = two_ssa_from_ins(line)?;
        let acc = single_ssa_from_outs(line)?;
        return Ok(Some(PlanOp::TMatmul { lhs, rhs, acc }));
    }

    // Row reductions & expansions.
    for kw in ["pto.trowmax", "pto.trowsum"] {
        if line.starts_with(kw) {
            let (src, scratch) = two_ssa_from_ins(line)?;
            let dst = single_ssa_from_outs(line)?;
            return Ok(Some(PlanOp::TRowReduce {
                opname: kw.to_string(),
                src,
                scratch,
                dst,
            }));
        }
    }
    for kw in ["pto.trowexpandsub", "pto.trowexpanddiv"] {
        if line.starts_with(kw) {
            // THREE operands under ptoas 0.68, and the third is the
            // materialised scratch tile. `two_ssa_from_ins` used to be called
            // here and took the first two, so the scratch tile had no reader
            // and `check_dead_tiles` called it unused.
            let ops = ssa_list_from_ins(line)?;
            let (src, red, scratch) = match ops.len() {
                2 => (ops[0].clone(), ops[1].clone(), None),
                3 => (ops[0].clone(), ops[1].clone(), Some(ops[2].clone())),
                n => {
                    return Err(format!(
                        "{kw}: expected 2 or 3 `ins` operands, got {n} in `{line}`"
                    ))
                }
            };
            let dst = single_ssa_from_outs(line)?;
            return Ok(Some(PlanOp::TRowExpand {
                opname: kw.to_string(),
                src,
                red,
                scratch,
                dst,
            }));
        }
    }

    // Pointwise binary.
    for kw in [
        "pto.tadd", "pto.tsub", "pto.tmul", "pto.tdiv", "pto.tmax", "pto.tmin",
    ] {
        if line.starts_with(kw) {
            let (a, b) = two_ssa_from_ins(line)?;
            let dst = single_ssa_from_outs(line)?;
            return Ok(Some(PlanOp::TBinary {
                opname: kw.to_string(),
                a,
                b,
                dst,
            }));
        }
    }

    // Pointwise unary. `pto.trowexpand` (bare) must be matched here only —
    // the suffixed `trowexpandsub`/`trowexpanddiv` forms are caught by the
    // row-expand block above and return early, so a starts_with check on
    // the bare keyword is unambiguous at this point.
    for kw in [
        "pto.texp",
        "pto.tneg",
        "pto.tabs",
        "pto.tsqrt",
        "pto.trecip",
        "pto.trowexpand",
    ] {
        if line.starts_with(kw) {
            let src = single_ssa_from_ins(line)?;
            let dst = single_ssa_from_outs(line)?;
            return Ok(Some(PlanOp::TUnary {
                opname: kw.to_string(),
                src,
                dst,
            }));
        }
    }

    Ok(None)
}

// ── attribute / type extraction helpers ────────────────────────────────

fn extract_config(line: &str) -> Option<String> {
    // Return the substring inside the outermost `{config = #pto.tile_buf_config<…>}`.
    let anchor = line.find("{config = ")?;
    let rest = &line[anchor + "{config = ".len()..];
    // Walk until matching `>}` with balanced `<>`.
    let bytes = rest.as_bytes();
    let mut depth = 0i32;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    if depth != 0 {
        return None;
    }
    Some(rest[..i + 1].to_string())
}

fn parse_tile_buf_config(cfg: &str) -> Result<(BLayout, SLayout, u32), String> {
    // Example: `#pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=…, compact=…>`
    let blayout = extract_inside(cfg, "blayout=#pto.blayout<")
        .and_then(|s| BLayout::parse(&s))
        .ok_or_else(|| format!("config missing/invalid blayout: `{cfg}`"))?;
    let slayout = extract_inside(cfg, "slayout=#pto.slayout<")
        .and_then(|s| SLayout::parse(&s))
        .ok_or_else(|| format!("config missing/invalid slayout: `{cfg}`"))?;
    let s_fractal = extract_kv_number(cfg, "s_fractal_size=")
        .ok_or_else(|| format!("config missing s_fractal_size: `{cfg}`"))?;
    Ok((blayout, slayout, s_fractal))
}

/// Return the substring inside a `<…>` that begins right after `prefix`.
fn extract_inside(s: &str, prefix: &str) -> Option<String> {
    let start = s.find(prefix)? + prefix.len();
    let close = s[start..].find('>')? + start;
    Some(s[start..close].to_string())
}

fn extract_kv_number(s: &str, prefix: &str) -> Option<u32> {
    let start = s.find(prefix)? + prefix.len();
    let rest = &s[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Extract (dtype, address-space) from the *first* `memref<…>` type on a
/// pointer_cast line. Stage-2 places the exact shape in the memref too
/// (e.g. `memref<1x1024xf32, strided<…>, #pto.address_space<vec>>`).
fn parse_memref_space_dtype(line: &str) -> Result<(ExtDType, AddressSpace), String> {
    let memref = first_memref(line).ok_or_else(|| format!("no memref<…> in `{line}`"))?;
    let dtype =
        parse_memref_dtype(&memref).ok_or_else(|| format!("cannot infer dtype from `{memref}`"))?;
    let space = extract_inside(&memref, "#pto.address_space<")
        .and_then(|s| AddressSpace::parse(&s))
        .ok_or_else(|| format!("missing address_space in `{memref}`"))?;
    Ok((dtype, space))
}

fn first_memref(line: &str) -> Option<String> {
    let anchor = line.find("memref<")?;
    let rest = &line[anchor..];
    let bytes = rest.as_bytes();
    // Match opening `<` at position 6 (after `memref`).
    let mut depth = 0i32;
    let mut i = 6;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..i + 1].to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn parse_memref_dtype(memref: &str) -> Option<ExtDType> {
    // `memref<1x1024xf32, strided<[1024,1]>, …>` — dtype sits before the first `,`.
    let lt = memref.find('<')? + 1;
    let first_comma = memref[lt..]
        .find(',')
        .map(|i| i + lt)
        .unwrap_or(memref.len() - 1);
    let head = &memref[lt..first_comma];
    // `RxCxTY` — the dtype is the last `x`-separated token.
    let tok = head.rsplit('x').next()?;
    ExtDType::parse(tok)
}

fn extract_rc_from_memref(line: &str, want_rows: bool) -> Result<u32, String> {
    let mr = first_memref(line).ok_or_else(|| format!("no memref<…> in `{line}`"))?;
    // `memref<1x1024xf32, …>`  -> head = "1x1024xf32"
    let lt = mr.find('<').unwrap() + 1;
    let first_comma = mr[lt..].find(',').map(|i| i + lt).unwrap_or(mr.len() - 1);
    let head = &mr[lt..first_comma];
    let parts: Vec<&str> = head.split('x').collect();
    if parts.len() < 3 {
        return Err(format!("memref head `{head}` not in RxCxTY form"));
    }
    let idx = if want_rows { 0 } else { 1 };
    parts[idx].trim().parse::<u32>().map_err(|_| {
        format!(
            "bad {} in memref head `{head}`",
            if want_rows { "rows" } else { "cols" }
        )
    })
}

// ── ins/outs SSA extractors ────────────────────────────────────────────

fn single_ssa_from_ins(line: &str) -> Result<Ssa, String> {
    let ins = extract_paren(line, "ins(").ok_or_else(|| format!("missing `ins(` in `{line}`"))?;
    first_ssa_before_colon(&ins).ok_or_else(|| format!("cannot extract ssa from `ins({ins})`"))
}

fn single_ssa_from_outs(line: &str) -> Result<Ssa, String> {
    let outs =
        extract_paren(line, "outs(").ok_or_else(|| format!("missing `outs(` in `{line}`"))?;
    first_ssa_before_colon(&outs).ok_or_else(|| format!("cannot extract ssa from `outs({outs})`"))
}

/// Every `ins(...)` operand, in order.
///
/// Exists because `two_ssa_from_ins` accepted `pieces.len() >= 2` and returned
/// the first two, so an op that gained a third operand lost it with no
/// diagnostic -- which is exactly what happened to the row-expansion scratch
/// tile.
fn ssa_list_from_ins(line: &str) -> Result<Vec<Ssa>, String> {
    let ins = extract_paren(line, "ins(").ok_or_else(|| format!("missing `ins(` in `{line}`"))?;
    let head = ins.split(':').next().unwrap_or("");
    Ok(split_top_commas(head)
        .into_iter()
        .map(|p| Ssa(p.trim().to_string()))
        .collect())
}

/// Exactly two `ins` operands.
///
/// `== 2`, not `>= 2`: a third operand here is a field this parser is not
/// reading, and the whole reason the stage-2 plan was wrong about the hidden
/// scratch tiles is that this function used to take the first two and discard
/// the rest.
fn two_ssa_from_ins(line: &str) -> Result<(Ssa, Ssa), String> {
    let pieces = ssa_list_from_ins(line)?;
    if pieces.len() != 2 {
        return Err(format!(
            "expected exactly 2 `ins` operands, got {} in `{line}`",
            pieces.len()
        ));
    }
    Ok((pieces[0].clone(), pieces[1].clone()))
}

fn three_ssa_from_ins(line: &str) -> Result<(Ssa, Ssa, Ssa), String> {
    let ins = extract_paren(line, "ins(").ok_or_else(|| format!("missing `ins(` in `{line}`"))?;
    let head = ins.split(':').next().unwrap_or("");
    let pieces = split_top_commas(head);
    if pieces.len() < 3 {
        return Err(format!("expected 3 ins operands, got `{head}`"));
    }
    Ok((
        Ssa(pieces[0].trim().to_string()),
        Ssa(pieces[1].trim().to_string()),
        Ssa(pieces[2].trim().to_string()),
    ))
}

fn extract_paren(line: &str, kw: &str) -> Option<String> {
    let start = line.find(kw)? + kw.len();
    let bytes = line.as_bytes();
    let mut depth = 1i32;
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'<' => depth += 1,
            b')' | b']' | b'>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(line[start..i].to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn first_ssa_before_colon(s: &str) -> Option<Ssa> {
    let head = s.split(':').next()?.trim();
    let tok = head.split(',').next()?.trim();
    if tok.starts_with('%') {
        Some(Ssa(tok.to_string()))
    } else {
        None
    }
}

fn split_top_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' | b'[' | b'<' => depth += 1,
            b')' | b']' | b'>' => depth -= 1,
            b',' if depth == 0 => {
                out.push(s[start..i].to_string());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].to_string());
    out
}
