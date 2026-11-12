// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Line-based PTO-MLIR text parser.
//!
//! Matches the text shapes emitted by `crates/rustc_codegen_tile/src/mlir_to_pto.rs`
//! and by `ptoas` (which share a pretty-printer style). Not a general MLIR parser;
//! extending it to new ops is explicitly a matter of adding a match arm here.

use crate::ast::*;
use std::collections::{BTreeMap, BTreeSet};

/// Join the physical lines of `src` into LOGICAL ones.
///
/// This parser is line-based, and MLIR wraps. Seven of the repo's 49 artifacts
/// -- every `gmm_*` cube/vector pair -- print an `aic_initialize_pipe` with its
/// attribute dictionary on one line and its operand list, in parentheses, on
/// the next:
///
/// ```text
/// pto.aic_initialize_pipe {id = 0, dir_mask = 1, ...}
///   (gm_slot_buffer = %slot : !pto.ptr<f32>, c2v_consumer_buf = %c2v : i32)
/// ```
///
/// Read line by line, the second line has no op name at all, and the first is a
/// pipe initialisation whose operands have vanished. The parser used to refuse
/// the file on the second line with a diagnostic that named the WRAPPING but
/// could not do anything about it.
///
/// Two joining rules, and no more: a line is a continuation if it opens with
/// `(`, or if the line before it has an unclosed `(`. Braces are deliberately
/// NOT balanced here -- `scf.for ... {` opens a region that spans lines on
/// purpose, and joining those would collapse a loop body onto its header.
fn logical_lines(src: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in src.lines() {
        let trimmed = raw.trim_start();
        let continues = match out.last() {
            None => false,
            Some(prev) => trimmed.starts_with('(') || unclosed_paren(prev),
        };
        if continues {
            let prev = out.last_mut().unwrap();
            prev.push(' ');
            prev.push_str(raw.trim());
        } else {
            out.push(raw.to_string());
        }
    }
    out
}

/// Whether `line` leaves a `(` open. Parens inside a quoted string would break
/// this, and PTO has them -- `{name = "c2v_fifo"}` -- so quotes are tracked.
fn unclosed_paren(line: &str) -> bool {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut prev_backslash = false;
    for c in line.chars() {
        if in_str {
            if c == '\\' && !prev_backslash {
                prev_backslash = true;
                continue;
            }
            if c == '"' && !prev_backslash {
                in_str = false;
            }
            prev_backslash = false;
            continue;
        }
        match c {
            '"' => in_str = true,
            '(' => depth += 1,
            ')' => depth -= 1,
            _ => {}
        }
    }
    depth > 0
}

pub fn parse_module(src: &str) -> Result<PtoModule, String> {
    let mut funcs = Vec::new();

    let joined = logical_lines(src);
    let lines: Vec<&str> = joined.iter().map(|s| s.as_str()).collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if let Some(func_hdr) = strip_func_header(line) {
            let FuncSignature { name, args, kind } = parse_func_signature(&func_hdr)?;
            // FRESH maps per function. SSA names are function-scoped in MLIR
            // and emitted PTO reuses them across functions in one module; a
            // shared map let the last binding win. See `ast::PtoFunc`.
            let mut consts = BTreeMap::new();
            let mut shapes = BTreeMap::new();
            let (body_end, body_ops) =
                parse_func_body(&lines, i + 1, &mut consts, &mut shapes, &args)?;
            funcs.push(PtoFunc {
                name,
                args,
                ops: body_ops,
                consts,
                shapes,
                kind,
            });
            i = body_end + 1; // skip the '}' line
        } else {
            i += 1;
        }
    }

    // The module-wide unions, with every ambiguous name withheld so a consumer
    // that reaches for one fails closed instead of getting the wrong function's
    // binding.
    let mut consts: BTreeMap<Ssa, i64> = BTreeMap::new();
    let mut shapes: BTreeMap<Ssa, TileShape> = BTreeMap::new();
    let mut ambiguous: BTreeSet<Ssa> = BTreeSet::new();
    for f in &funcs {
        for (k, v) in &f.consts {
            match consts.get(k) {
                Some(prev) if prev != v => {
                    ambiguous.insert(k.clone());
                }
                _ => {
                    consts.insert(k.clone(), *v);
                }
            }
        }
        for (k, v) in &f.shapes {
            match shapes.get(k) {
                Some(prev) if prev != v => {
                    ambiguous.insert(k.clone());
                }
                _ => {
                    shapes.insert(k.clone(), *v);
                }
            }
        }
    }
    for k in &ambiguous {
        consts.remove(k);
        shapes.remove(k);
    }

    Ok(PtoModule {
        funcs,
        consts,
        shapes,
        ambiguous,
    })
}

/// Recognise `func.func @name(...) {` (possibly with return types after `->`).
fn strip_func_header(line: &str) -> Option<String> {
    let rest = line.strip_prefix("func.func ")?;
    if !rest.ends_with('{') {
        return None;
    }
    Some(rest.trim_end_matches('{').trim().to_string())
}

/// Parse `@name(%a: !pto.ptr<f32>, %b: !pto.ptr<f32>) attributes {...}`.
///
/// Everything after the argument list used to be DISCARDED, which is how the
/// reverse emitter came to drop `pto.kernel_kind`. The tail is now parsed, and
/// an attribute dictionary holding anything else is an error: dropping one
/// attribute silently is what this function is being fixed for, and a
/// recogniser that keeps the one spelling it knows and ignores the rest is the
/// same bug with a smaller blast radius.
///
/// No corpus signature carries a `->` return type (0 of 49 artifacts), so one
/// is refused rather than guessed at.
struct FuncSignature {
    name: String,
    args: Vec<(Ssa, PtoType)>,
    kind: Option<KernelKind>,
}

fn parse_func_signature(hdr: &str) -> Result<FuncSignature, String> {
    // hdr example: `@tile_softmax(%arg601: !pto.ptr<f32>, %arg602: !pto.ptr<f32>)`
    let open = hdr.find('(').ok_or("missing `(` in func signature")?;
    let close = hdr.rfind(')').ok_or("missing `)` in func signature")?;
    let name = hdr[..open].trim_start_matches('@').trim().to_string();
    let arg_src = &hdr[open + 1..close];
    let kind = parse_func_attributes(hdr[close + 1..].trim(), &name)?;

    let mut args = Vec::new();
    if arg_src.trim().is_empty() {
        return Ok(FuncSignature { name, args, kind });
    }
    for piece in split_top_commas(arg_src) {
        let piece = piece.trim();
        let colon = piece.find(':').ok_or("missing `:` in arg")?;
        let ssa = Ssa(piece[..colon].trim().to_string());
        let ty = parse_type(piece[colon + 1..].trim())?;
        args.push((ssa, ty));
    }
    Ok(FuncSignature { name, args, kind })
}

/// The tail of a `func.func` signature: nothing, or one attribute dictionary.
///
/// Fails closed. The corpus has exactly two spellings here -- 7 `<cube>` and 7
/// `<vector>`, one pair per cube/vector artifact -- and anything else is
/// refused by text, because the alternative is the drop this exists to stop.
fn parse_func_attributes(tail: &str, func: &str) -> Result<Option<KernelKind>, String> {
    if tail.is_empty() {
        return Ok(None);
    }
    let dict = tail
        .strip_prefix("attributes")
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix('{'))
        .and_then(|r| r.strip_suffix('}'))
        .ok_or_else(|| {
            format!(
                "`{func}`: trailing `{tail}` in the signature is not an `attributes {{...}}`                  dictionary. It is not ignored, because the attribute that was ignored here                  until 2026-11-01 was `pto.kernel_kind`, which decides which core the                  function runs on"
            )
        })?;
    let kind = dict
        .trim()
        .strip_prefix("pto.kernel_kind")
        .map(str::trim_start)
        .and_then(|r| r.strip_prefix('='))
        .map(str::trim)
        .and_then(|r| r.strip_prefix("#pto.kernel_kind<"))
        .and_then(|r| r.strip_suffix('>'))
        .ok_or_else(|| {
            format!(
                "`{func}`: unrecognised function attribute `{dict}`. The only one this crate                  models is `pto.kernel_kind`; carrying this file on without the attribute                  would make a re-emission that drops it look lossless"
            )
        })?;
    KernelKind::parse(kind)
        .map(Some)
        .ok_or_else(|| format!("`{func}`: unknown `pto.kernel_kind<{kind}>`"))
}

/// Drop a trailing `// ...` note from an op line.
///
/// Must run before any split, not after. See the call site for the colon that
/// made this structural rather than cosmetic.
fn strip_trailing_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => line[..i].trim_end(),
        None => line,
    }
}

/// Parse `!pto.ptr<f32>`, `!pto.tensor_view<?x?xf32>`, `!pto.partition_tensor_view<1x1024xf32>`,
/// or `!pto.tile_buf<...>`.
fn parse_type(s: &str) -> Result<PtoType, String> {
    // A trailing `// ...` comment is not part of the type. Without this the
    // suffix match on `>` fails and the whole artifact is rejected with
    // "unrecognised type", naming a string that looks perfectly well formed
    // right up to the comment -- 9 of the repo's artifacts carry a
    // `// chunk fold` note on the tile_buf line.
    let s = match s.find("//") {
        Some(i) => s[..i].trim(),
        None => s.trim(),
    };
    if let Some(inner) = s
        .strip_prefix("!pto.ptr<")
        .and_then(|x| x.strip_suffix('>'))
    {
        let dt = DType::parse(inner).ok_or_else(|| format!("unknown ptr dtype `{inner}`"))?;
        return Ok(PtoType::Ptr(dt));
    }
    if let Some(inner) = s
        .strip_prefix("!pto.tensor_view<")
        .and_then(|x| x.strip_suffix('>'))
    {
        // `?x?xTy`, or any other rank of `?`.
        let dt_str = inner.rsplit('x').next().ok_or("malformed tensor_view")?;
        let dt =
            DType::parse(dt_str).ok_or_else(|| format!("unknown tensor_view dtype `{dt_str}`"))?;
        let rank = inner.matches('?').count() as u32;
        if rank == 0 {
            return Err(format!("tensor_view `{s}` declares no dynamic dims"));
        }
        return Ok(PtoType::TensorView { rank, dtype: dt });
    }
    if let Some(inner) = s
        .strip_prefix("!pto.partition_tensor_view<")
        .and_then(|x| x.strip_suffix('>'))
    {
        // `RxCxTy`, or any other rank.
        let (dims, dt) = parse_view_dims(inner)?;
        return Ok(if dims.len() == 2 {
            PtoType::PartitionView {
                rows: dims[0],
                cols: dims[1],
                dtype: dt,
            }
        } else {
            PtoType::PartitionViewNd { dims, dtype: dt }
        });
    }
    if let Some(inner) = s
        .strip_prefix("!pto.tile_buf<")
        .and_then(|x| x.strip_suffix('>'))
    {
        // `loc=vec, dtype=f32, rows=1, cols=1024, v_row=..., blayout=..., slayout=..., fractal=..., pad=...`
        let mut rows = None;
        let mut cols = None;
        let mut v_rows = None;
        let mut v_cols = None;
        let mut dtype = None;
        let mut loc = None;
        let mut loc_text = "";
        for kv in inner.split(',') {
            let kv = kv.trim();
            let (k, v) = match kv.split_once('=') {
                Some(x) => x,
                None => continue,
            };
            match k.trim() {
                "rows" => rows = v.trim().parse().ok(),
                "cols" => cols = v.trim().parse().ok(),
                // Absent or non-numeric (a dynamic extent) leaves these None
                // and they fall back to rows/cols below, which keeps every
                // unpadded tile behaving exactly as before.
                "v_row" => v_rows = v.trim().parse().ok(),
                "v_col" => v_cols = v.trim().parse().ok(),
                "dtype" => dtype = DType::parse(v.trim()),
                "loc" => {
                    loc_text = v.trim();
                    loc = TileLoc::parse(loc_text);
                }
                _ => {}
            }
        }
        let rows = rows.ok_or("tile_buf missing rows")?;
        let cols = cols.ok_or("tile_buf missing cols")?;
        let dtype = dtype.ok_or("tile_buf missing/unknown dtype")?;
        // Refuse an unknown location by name rather than defaulting to `vec`:
        // defaulting is what silently relocated 233 cube-side tiles into UB.
        let loc = loc.ok_or_else(|| {
            if loc_text.is_empty() {
                "tile_buf has no `loc=`; it decides which unit's memory the \
                 tile is in, so it cannot be assumed"
                    .to_string()
            } else {
                format!(
                    "tile_buf has unknown `loc={loc_text}`; known locations are \
                     vec, mat, left, right, acc, scaling"
                )
            }
        })?;
        return Ok(PtoType::TileBuf {
            rows,
            cols,
            v_rows: v_rows.unwrap_or(rows),
            v_cols: v_cols.unwrap_or(cols),
            dtype,
            loc,
        });
    }
    Err(format!("unrecognised type `{s}`"))
}

/// `1x1024xf32` -> (vec![1, 1024], F32); `1x4x16x16x32xi8` -> five dims.
///
/// Returns EVERY dim. A rank > 2 view is a fractal / NZ layout and used to be
/// refused here, which cost five artifacts; the caller now builds a
/// `PartitionViewNd` from the dims instead of picking two of them, which would
/// be a silent substitution. (The refusal before that blamed the dtype: it
/// said "bad dtype in `1x4x16x16x32xi8`" because `16` is not a dtype, while
/// the real dtype `i8` sat at the end and parses fine.)
fn parse_view_dims(s: &str) -> Result<(Vec<u32>, DType), String> {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() < 3 {
        return Err(format!("expected at least RxCxTy, got `{s}`"));
    }
    let (dim_parts, dt_part) = parts.split_at(parts.len() - 1);
    let dt = DType::parse(dt_part[0]).ok_or_else(|| {
        format!(
            "bad dtype `{}` in `{s}` (the dtype is the LAST `x`-separated field)",
            dt_part[0]
        )
    })?;
    let mut dims = Vec::with_capacity(dim_parts.len());
    for (i, d) in dim_parts.iter().enumerate() {
        dims.push(
            d.parse()
                .map_err(|_| format!("bad dim {i} `{d}` in `{s}`"))?,
        );
    }
    Ok((dims, dt))
}

/// Walks from `start` to the matching closing `}`. Returns (end_idx, ops).
fn parse_func_body(
    lines: &[&str],
    start: usize,
    consts: &mut BTreeMap<Ssa, i64>,
    shapes: &mut BTreeMap<Ssa, TileShape>,
    _args: &[(Ssa, PtoType)],
) -> Result<(usize, Vec<PtoOp>), String> {
    let mut ops = Vec::new();
    let mut i = start;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with('}') {
            return Ok((i, ops));
        }
        if line.is_empty() || line.starts_with("//") {
            i += 1;
            continue;
        }

        // Strip a trailing `// ...` note BEFORE anything splits the line.
        // `parse_type` strips one too, but by then it is too late: both
        // `alloc_tile` and `make_tensor_view` locate their type with
        // `rfind(':')`, so a comment that CONTAINS a colon moves the split.
        // `pto.alloc_tile : !pto.tile_buf<...>  // pool: broadcast Q` handed
        // `parse_type` the text `broadcast Q` and was rejected as an
        // unrecognised type -- naming a string that is not a type and never
        // claimed to be one. Three of the repo's artifacts failed exactly
        // this way, and the diagnostic pointed at the wrong half of the line.
        let line = strip_trailing_comment(line);
        if line.is_empty() {
            i += 1;
            continue;
        }

        // A REGION: parse the header, then recurse for the body, which this
        // same function terminates on the closing `}`. Body ops share `consts`
        // and `shapes` because MLIR scopes SSA names to the enclosing region
        // and the corpus reuses tile names across a loop boundary.
        // `scf.if %cond { .. } else { .. }`. The `} else {` line both CLOSES
        // the then-region and OPENS the else-region, which is why the recursion
        // has to inspect the line it stopped on rather than assume it is a bare
        // `}`.
        if mnemonic(line) == "scf.if" {
            let cond = line["scf.if".len()..]
                .trim()
                .strip_suffix('{')
                .map(str::trim)
                .filter(|c| c.starts_with('%') && !c.contains(char::is_whitespace))
                .ok_or_else(|| {
                    format!(
                        "`scf.if` has no single `%cond` operand in `{line}`; every corpus \
                         instance is `scf.if %c {{`, and guessing the condition would \
                         pick which branch runs"
                    )
                })?;
            let cond = Ssa(cond.to_string());
            let (then_end, then_body) = parse_func_body(lines, i + 1, consts, shapes, _args)?;
            let closer = lines
                .get(then_end)
                .map(|l| l.trim())
                .unwrap_or_default()
                .to_string();
            if closer.starts_with("} else") {
                let (else_end, else_body) =
                    parse_func_body(lines, then_end + 1, consts, shapes, _args)?;
                ops.push(PtoOp::If {
                    cond,
                    then_body,
                    else_body: Some(else_body),
                });
                i = else_end + 1;
            } else {
                ops.push(PtoOp::If {
                    cond,
                    then_body,
                    else_body: None,
                });
                i = then_end + 1;
            }
            continue;
        }

        if mnemonic(line) == "scf.for" {
            let (var, lb, ub, step) = parse_scf_for_header(line)?;
            let (body_end, body) = parse_func_body(lines, i + 1, consts, shapes, _args)?;
            ops.push(PtoOp::For {
                var,
                lb,
                ub,
                step,
                body,
            });
            i = body_end + 1; // skip the `}`
            continue;
        }

        match parse_line(line, consts, shapes)? {
            Some(op) => ops.push(op),
            None => {} // constant already recorded, or ignored line
        }
        i += 1;
    }
    Err("function body not terminated with `}`".into())
}

/// `scf.for %r = %lb to %ub step %st {` -> the four SSA names.
///
/// The bounds stay as names: they are usually derived from
/// `pto.get_block_idx`, so the extent is per-core and no constant map resolves
/// it. Refusing a header that does not have all four is deliberate -- a loop
/// with an assumed bound is a different kernel.
fn parse_scf_for_header(line: &str) -> Result<(Ssa, Ssa, Ssa, Ssa), String> {
    let rest = line
        .trim()
        .strip_prefix("scf.for")
        .ok_or_else(|| format!("not an `scf.for` header: `{line}`"))?
        .trim()
        .trim_end_matches('{')
        .trim();
    let (var, rest) = rest
        .split_once(" = ")
        .ok_or_else(|| format!("`scf.for` has no `= <lower>` in `{line}`"))?;
    let (lb, rest) = rest
        .split_once(" to ")
        .ok_or_else(|| format!("`scf.for` has no ` to <upper>` in `{line}`"))?;
    let (ub, step) = rest
        .split_once(" step ")
        .ok_or_else(|| format!("`scf.for` has no ` step <stride>` in `{line}`"))?;
    for (what, v) in [("induction variable", var), ("lower bound", lb), ("upper bound", ub), ("step", step)] {
        if !v.trim().starts_with('%') {
            return Err(format!(
                "`scf.for`'s {what} is `{}`, not an SSA name, in `{line}`",
                v.trim()
            ));
        }
    }
    Ok((
        Ssa(var.trim().to_string()),
        Ssa(lb.trim().to_string()),
        Ssa(ub.trim().to_string()),
        Ssa(step.trim().to_string()),
    ))
}

/// The op mnemonic of a statement line: the text up to the first whitespace.
///
/// Every dispatch below compares against this EXACTLY, because
/// `line.starts_with("pto.tmul")` is also true of `pto.tmuls`, the
/// tile-by-SCALAR form, whose second operand is an `f32` and not a tile.
/// Census of the repo's `.pto` artifacts, 2026-10-31: `pto.tmuls` x151,
/// `pto.tadds` x80, `pto.tmaxs` x46 -- 277 op instances that the prefix
/// dispatch accepted as their two-tile namesakes and bound a scalar constant
/// into a tile operand slot.
///
/// The round-trip sweep only caught it because the reverse emitter asks for a
/// shape it cannot have; the Rust emitter asks for no such thing and would
/// have emitted `tile_mul_f32(a, c)` against a scalar -- a wrong kernel, exit
/// 0. `pto.tload ` and `pto.tstore ` already defended against this with a
/// trailing space in their literal, so the hazard was known in two places and
/// not in the other eleven.
/// Every op mnemonic this parser MODELS, and the single source of truth for
/// that question.
///
/// The dispatch in `parse_line` is a chain, and its fallthrough used to be
/// `Ok(None)`: a line matching no arm was DROPPED, silently, and the round-trip
/// verifier then compared two ASTs that were both missing it, so the artifact
/// passed. Census of the sweep's own 49-file corpus, 2026-10-31: **1895 of
/// 6158 op lines (31%) were dropped, and 33 of the 41 artifacts the sweep
/// reported as round-tripping contained at least one of them.** Only 8 files
/// were fully modelled. The dropped set is not decoration -- `pto.tcvt` x188,
/// `pto.ttrans` x160, `pto.tcolsum` x96 and `scf.for` x46 are data flow and
/// control flow, so a re-emitted program would have been a different kernel.
///
/// The three artifacts that DID fail with `no shape for SSA` were the only
/// three where the silence became audible, and only incidentally: the dropped
/// op (`pto.subview`, `pto.tpop_from_aic`) happened to define an SSA that a
/// later modelled op consumed. The gate's sensitivity to dropping was an
/// accident of operand order, not a property of the gate.
///
/// This list was a GATE from 2026-10-31 to 2026-11-01: checked before the chain
/// ran, with anything outside it refused by name. Reconciling with
/// `pto-rs-viewpoint-rebase`, which had modelled the same drop independently and
/// chosen the other remedy, retired the gate. Refusing in the parser also
/// refuses the ops the round trip could have carried losslessly, and the round
/// trip is the instrument that proves the reader; so the fallthrough now builds
/// [`PtoOp::Unmodelled`], which re-emits the line byte-for-byte, and the two
/// LIFTS refuse it by name instead.
///
/// What the list still does, and what makes it worth keeping as a `const`:
///
/// * It is the declared denominator. [`crate::verify::op_model_coverage`]
///   counts modelled against unmodelled ops per file, so a file cannot pass the
///   round-trip sweep on byte passthrough alone -- "nothing was interpreted" and
///   "everything round-tripped" are no longer the same green.
/// * It is checked AGAINST the dispatch, both ways. `every_modelled_op_round_trips`
///   feeds one synthetic module per entry and asserts the parse is not
///   `Unmodelled`; the chain's own fallthrough-after-match is still an error. A
///   mnemonic listed here whose arm stopped claiming the line now fails loudly
///   instead of being certified by the list that names it.
pub const MODELLED_OPS: &[&str] = &[
    "arith.constant",
    "pto.make_tensor_view",
    "pto.partition_view",
    "pto.alloc_tile",
    "pto.tload",
    "pto.tstore",
    "pto.tadd",
    "pto.tsub",
    "pto.tmul",
    "pto.tdiv",
    "pto.tmax",
    "pto.tmin",
    "pto.tmuls",
    "pto.tadds",
    "pto.tsubs",
    "pto.tmaxs",
    "pto.tmins",
    "pto.texp",
    "pto.tneg",
    "pto.tabs",
    "pto.trowmax",
    "pto.trowsum",
    "pto.trowexpandsub",
    "pto.trowexpanddiv",
    "pto.trowexpandmul",
    "pto.trowexpand",
    "pto.tcolexpand",
    "pto.tcolsum",
    "pto.tsqrt",
    "pto.trsqrt",
    "pto.trecip",
    "pto.tmatmul",
    "pto.tmov",
    "pto.ttrans",
    "pto.barrier",
    "pto.subview",
    "pto.tgatherb",
    "pto.tcvt",
    "pto.tshls",
    "pto.tshrs",
    "pto.tands",
    "pto.get_block_idx",
    "pto.get_block_num",
    "arith.index_cast",
    "arith.muli",
    "arith.addi",
    "arith.fptosi",
    "pto.tgetval",
    "pto.tci",
    "pto.tsort32",
    "pto.tgather",
    "pto.textract",
    "scf.for",
    "scf.if",
];

/// The op name of a statement line, whether or not it binds a result:
/// `%d = pto.tmul ins(...)` -> `pto.tmul`, `pto.tstore ins(...)` ->
/// `pto.tstore`, and the quoted generic form `%s = "pto.subview"(%f, ...)` ->
/// `pto.subview`.
///
/// `mnemonic` below cannot serve: it returns the text up to the first space,
/// which for a result-binding line is the SSA name. That is why the four
/// result-binding ops are dispatched by `find(" = pto.xxx ")` instead -- and
/// why the coverage gate needed its own accessor.
pub fn op_name(line: &str) -> &str {
    // Only split on ` = ` when the line actually binds a result; an attribute
    // dictionary carries its own (`pto.tpop_from_aic {id = 0, split = 0}`).
    let rest = if line.starts_with('%') {
        match line.find(" = ") {
            Some(i) => line[i + 3..].trim_start(),
            None => line,
        }
    } else {
        line
    };
    let rest = rest.strip_prefix('"').unwrap_or(rest);
    let end = rest
        .find(|c: char| c.is_whitespace() || c == '(' || c == '"' || c == '{' || c == ':')
        .unwrap_or(rest.len());
    &rest[..end]
}

/// Build [`PtoOp::Unmodelled`] for a line no arm models, recording the result's
/// tile shape so ops downstream of it still resolve.
///
/// That missing shape is what used to surface, three artifacts out of 49, as
/// `no shape for SSA %s` -- blaming the `tstore` that consumed the vanished
/// result rather than the `pto.subview` that defined it.
fn unmodelled_op(
    line: &str,
    op: &str,
    shapes: &mut BTreeMap<Ssa, TileShape>,
) -> Result<Option<PtoOp>, String> {
    // A line-based parser also meets lines that are not ops at all, and those
    // are NOT unmodelled ops -- carrying one verbatim would hide a real parser
    // gap, because the arguments on a wrapped signature's second line are being
    // missed by `parse_func_signature` as well. Still an error.
    if op.is_empty() || !op.contains('.') || op.starts_with('%') {
        return Err(not_an_op_line(line));
    }
    // The result type is the LAST tile_buf on the line: for
    // `%s = "pto.subview"(%f, ...) : (!pto.tile_buf<rows=4..>, ..) ->
    // !pto.tile_buf<rows=2..>` the operand types come first.
    if let Some((dst, _)) = line.split_once('=') {
        let dst = dst.trim();
        if dst.starts_with('%') && !dst.contains(char::is_whitespace) {
            if let Some(start) = line.rfind("!pto.tile_buf<") {
                if let Some(end) = line[start..].find('>') {
                    if let Ok(PtoType::TileBuf {
                        rows,
                        cols,
                        v_rows,
                        v_cols,
                        dtype,
                        loc,
                    }) = parse_type(&line[start..start + end + 1])
                    {
                        shapes.insert(
                            Ssa(dst.to_string()),
                            TileShape {
                                rows,
                                cols,
                                v_rows,
                                v_cols,
                                dtype,
                                loc: Some(loc),
                            },
                        );
                    }
                }
            }
        }
    }
    Ok(Some(PtoOp::Unmodelled {
        op: op.to_string(),
        text: line.trim().to_string(),
    }))
}

fn not_an_op_line(line: &str) -> String {
    // A line-based parser also meets lines that are not ops at all: a wrapped
    // `func.func` signature's second half, or the continuation of an
    // `aic_initialize_pipe` operand list. Saying "unmodelled op `%arg0`" would
    // name the wrong thing -- the gap is the WRAPPING, and the arguments on
    // that line are being missed by `parse_func_signature` as well. That is
    // also why such a line is NOT carried as `PtoOp::Unmodelled`: round-tripping
    // the fragment would make a truncated argument list look lossless.
    let shown: String = line.chars().take(56).collect();
    // Deliberately does NOT name a cause. An earlier version asserted
    // "this looks like a wrapped signature or operand list", which is true
    // of the two corpus cases and false of anything else that reaches here
    // -- a mis-escaped quote in a test fixture landed on it and was told
    // its line was wrapped. Naming a cause you have not established is the
    // misattribution this crate keeps paying for.
    format!(
        "no op name found in `{shown}` -- this parser is line-based, so a \
         construct wrapped across lines reaches here with only its first \
         line (the corpus has two: a wrapped `func.func` signature and a \
         wrapped `aic_initialize_pipe` operand list)"
    )
}

fn mnemonic(line: &str) -> &str {
    line.split_whitespace().next().unwrap_or("")
}

/// Parse one op line. Returns `Ok(None)` if the line was handled by side-effect
/// (e.g. a constant binding stashed into `consts`).
/// Record the shape of the `outs(... : !pto.tile_buf<...>)` tile, for the ops
/// whose result is not a `%dst = ` binding. Without it the emitter has no shape
/// for the tile they define.
fn record_outs_shape(
    line: &str,
    dst: &Ssa,
    shapes: &mut BTreeMap<Ssa, TileShape>,
) {
    let Some(i) = line.rfind("!pto.tile_buf<") else { return };
    let Some(end) = line[i..].find('>') else { return };
    if let Ok(PtoType::TileBuf { rows, cols, v_rows, v_cols, dtype, .. }) =
        parse_type(&line[i..i + end + 1])
    {
        shapes.insert(
            dst.clone(),
            TileShape { rows, cols, v_rows, v_cols, dtype, loc: None },
        );
    }
}

/// The tile bound by `outs(%name ... )`; the topk-suite ops spell their result
/// there instead of with a `%dst = ` prefix.
fn outs_ssa(line: &str) -> Option<Ssa> {
    let after = line.split_once("outs(")?.1.trim_start();
    let name: String = after
        .chars()
        .take_while(|c| *c == '%' || c.is_alphanumeric() || *c == '_')
        .collect();
    if name.len() > 1 { Some(Ssa(name)) } else { None }
}

fn parse_line(
    line: &str,
    consts: &mut BTreeMap<Ssa, i64>,
    shapes: &mut BTreeMap<Ssa, TileShape>,
) -> Result<Option<PtoOp>, String> {
    let mnem = mnemonic(line);

    if line.starts_with("return") {
        return Ok(None);
    }

    // An op this crate does not model. Carried verbatim rather than dropped
    // (see MODELLED_OPS for the 1895-of-6158 census) and rather than refused
    // here, so the PTO round trip stays faithful; both lifts refuse it by name.
    let op = op_name(line);
    if !MODELLED_OPS.contains(&op) {
        return unmodelled_op(line, op, shapes);
    }

    // `%c1024 = arith.constant 1024 : index`
    if let Some(rest) = line.strip_prefix("%") {
        if let Some(eq) = rest.find(" = arith.constant ") {
            let ssa = Ssa(format!("%{}", &rest[..eq]));
            let tail = &rest[eq + " = arith.constant ".len()..];
            let val_str = tail.split(':').next().unwrap_or("").trim();
            let ty = tail.split(':').nth(1).unwrap_or("").trim();
            // An index constant folds into const-generic parameters at its use
            // sites, so it goes in `consts`. A typed constant -- 128.0 : f16
            // for the u8/e8 dequant scale, 0.000001 : f32 for an rms epsilon --
            // is a VALUE, never an index, so it must not enter that map: a
            // float silently truncated to i64 would be folded into a shape.
            let value = match val_str.parse::<i64>() {
                Ok(v) if ty.is_empty() || ty == "index" => {
                    consts.insert(ssa.clone(), v);
                    ConstVal::Index(v)
                }
                // An integer constant with a WIDTH is still foldable where a
                // size operand names it, so it stays in `consts` -- but its
                // spelling is load-bearing and has to survive the round-trip.
                // This arm used to be folded into the one above by
                // `ty.starts_with('i')`, which is true of `i16` and `i32` as
                // well as nothing. Corpus census 2026-11-01: **144 `i16` and
                // 28 `i32` constants were re-emitted as `: index`**, and
                // neither gate could see it, because `ConstVal::Index` carries
                // no type and so both passes agreed on the same wrong thing.
                // The doc comment on `ConstVal::Index` said "the only kind that
                // participates in index folding" and was describing the intent,
                // not the code.
                Ok(v) if ty.starts_with('i') => {
                    consts.insert(ssa.clone(), v);
                    ConstVal::Typed {
                        lit: val_str.to_string(),
                        ty: ty.to_string(),
                    }
                }
                _ if !val_str.is_empty() && !ty.is_empty() => ConstVal::Typed {
                    lit: val_str.to_string(),
                    ty: ty.to_string(),
                },
                _ => return Err(format!("bad constant `{val_str}`")),
            };
            return Ok(Some(PtoOp::ScalarConst { dst: ssa, value }));
        }
    }

    // `%dst = pto.make_tensor_view %ptr, shape = [...], strides = [...] : !pto.tensor_view<?x?xf32>`
    if let Some(op) = try_parse_make_tensor_view(line)? {
        return Ok(Some(op));
    }

    // `%dst = pto.partition_view %src, offsets=[...], sizes=[%cR, %cC] : <in> -> <out>`
    if let Some(op) = try_parse_partition_view(line, consts, shapes)? {
        return Ok(Some(op));
    }

    // `%dst = pto.alloc_tile : !pto.tile_buf<...>`
    if let Some(op) = try_parse_alloc_tile(line, shapes)? {
        return Ok(Some(op));
    }

    // `%dst = "pto.subview"(%src, %o0, %o1) {...} : (...) -> !pto.tile_buf<...>`
    if let Some(op) = try_parse_subview(line, shapes)? {
        return Ok(Some(op));
    }

    // `pto.tload ins(%view : ...) outs(%tile : ...)`
    if let Some(op) = try_parse_tload(line)? {
        return Ok(Some(op));
    }
    // `pto.tstore ins(%tile : ...) outs(%view : ...)`
    if let Some(op) = try_parse_tstore(line)? {
        return Ok(Some(op));
    }

    // Pointwise binary: `pto.tadd ins(%a, %b : t1, t2) outs(%dst : t3)`
    for (kw, kind) in [
        ("pto.tadd", BinKind::Add),
        ("pto.tsub", BinKind::Sub),
        ("pto.tmul", BinKind::Mul),
        ("pto.tdiv", BinKind::Div),
        ("pto.tmax", BinKind::Max),
        ("pto.tmin", BinKind::Min),
    ] {
        if mnem == kw {
            let (a, b, dst) = parse_ins2_outs1(line)?;
            return Ok(Some(PtoOp::BinOp { kind, dst, a, b }));
        }
    }

    // Pointwise tile-by-scalar: `pto.tmuls ins(%a, %s : <tile>, f32) outs(%dst)`.
    // Listed AFTER the two-tile table and matched exactly, so neither table can
    // reach the other's spellings.
    for (kw, kind) in [
        ("pto.tmuls", ScalarBinKind::Muls),
        ("pto.tadds", ScalarBinKind::Adds),
        // `Subs` existed in `ScalarBinKind` and `subs` in `WRAPPERS` from the day
        // `Adds` was added "for symmetry", but this table never gained the row, so
        // the lift refused `pto.tsubs` as "not modelled" while the model was
        // already there. Added 2026-11-09 after the refusal census named it as one
        // of only five a session could close.
        ("pto.tsubs", ScalarBinKind::Subs),
        ("pto.tmaxs", ScalarBinKind::Maxs),
        ("pto.tmins", ScalarBinKind::Mins),
        ("pto.tshls", ScalarBinKind::Shls),
        ("pto.tshrs", ScalarBinKind::Shrs),
        ("pto.tands", ScalarBinKind::Ands),
    ] {
        if mnem == kw {
            let (a, scalar, dst) = parse_ins2_outs1(line)?;
            let scalar_ty = ins_operand_type(line, 1).ok_or_else(|| {
                format!(
                    "`{kw}` has no type for its scalar operand in `{line}`; it is                      not safe to assume f32 -- f16, i16 and i32 all occur"
                )
            })?;
            return Ok(Some(PtoOp::BinOpScalar {
                kind,
                dst,
                a,
                scalar,
                scalar_ty,
            }));
        }
    }

    // Pointwise unary
    for (kw, kind) in [
        ("pto.texp", UnKind::Exp),
        ("pto.tneg", UnKind::Neg),
        ("pto.tabs", UnKind::Abs),
        ("pto.tsqrt", UnKind::Sqrt),
        ("pto.trsqrt", UnKind::Rsqrt),
        ("pto.trecip", UnKind::Recip),
    ] {
        if mnem == kw {
            // Unary with single ins: `pto.texp ins(%src : t) outs(%dst : t)`
            if let Some((src, dst)) = try_parse_ins1_outs1(line) {
                return Ok(Some(PtoOp::UnOp { kind, src, dst }));
            }
        }
    }

    // Row-reductions: `pto.trowmax ins(%src, %scratch : ...) outs(%dst : ...)`
    for (kw, kind) in [("pto.trowmax", RowKind::Max), ("pto.trowsum", RowKind::Sum)] {
        if mnem == kw {
            let (src, _scratch, dst) = parse_ins2_outs1(line)?;
            return Ok(Some(PtoOp::RowReduce { kind, src, dst }));
        }
    }

    // Row-expansions
    if mnem == "pto.trowexpandsub" {
        let (src, red, dst) = parse_ins2_outs1(line)?;
        return Ok(Some(PtoOp::RowExpandSub { src, red, dst }));
    }
    if mnem == "pto.trowexpanddiv" {
        let (src, red, dst) = parse_ins2_outs1(line)?;
        return Ok(Some(PtoOp::RowExpandDiv { src, red, dst }));
    }
    if mnem == "pto.trowexpandmul" {
        let (src, red, dst) = parse_ins2_outs1(line)?;
        return Ok(Some(PtoOp::RowExpandMul { src, red, dst }));
    }
    // One operand, and the two shapes DIFFER (1xK -> 1xN). Tested after the
    // three two-operand forms because `pto.trowexpand` is a prefix of all of
    // them; `mnem` is an exact match so the order is not load-bearing, but a
    // future `starts_with` dispatch here would silently swallow the other
    // three -- which is how `pto.tmuls` was once read as `pto.tmul`.
    if mnem == "pto.trowexpand" {
        if let Some((src, dst)) = try_parse_ins1_outs1(line) {
            return Ok(Some(PtoOp::RowExpand { src, dst }));
        }
    }

    // The column-direction pair, added 2026-11-10 with the 1-query sink
    // attention PTO. Tested after `trowexpand` because the mnemonics are
    // exact matches; the names share no prefix with each other.
    if mnem == "pto.tcolexpand" {
        if let Some((src, dst)) = try_parse_ins1_outs1(line) {
            return Ok(Some(PtoOp::ColExpand { src, dst }));
        }
    }
    if mnem == "pto.tcolsum" {
        if let Some((src, dst)) = try_parse_ins1_outs1(line) {
            return Ok(Some(PtoOp::ColSum { src, dst }));
        }
    }

    // `%bidx = "pto.get_block_idx"() : () -> i64`
    for (kw, which) in [
        ("pto.get_block_idx", BlockQueryKind::Idx),
        ("pto.get_block_num", BlockQueryKind::Num),
    ] {
        if op == kw {
            let eq = line.find(" = ").ok_or_else(|| {
                format!("`{kw}` binds no result in `{line}`")
            })?;
            let dst = Ssa(line[..eq].trim().to_string());
            let arrow = line.rfind("->").ok_or_else(|| {
                format!("`{kw}` has no result type in `{line}`")
            })?;
            let ty = line[arrow + 2..].trim().to_string();
            return Ok(Some(PtoOp::BlockQuery { dst, which, ty }));
        }
    }

    // `%i = arith.index_cast %bidx : i64 to index`
    if op == "arith.index_cast" {
        let eq = line
            .find(" = ")
            .ok_or_else(|| format!("`arith.index_cast` binds no result in `{line}`"))?;
        let dst = Ssa(line[..eq].trim().to_string());
        let rest = line[eq + 3..]
            .trim()
            .strip_prefix("arith.index_cast")
            .ok_or_else(|| format!("malformed index_cast in `{line}`"))?
            .trim();
        let (operand, tys) = rest
            .split_once(':')
            .ok_or_else(|| format!("`arith.index_cast` has no `:` in `{line}`"))?;
        let (from, to) = tys
            .trim()
            .split_once(" to ")
            .ok_or_else(|| format!("`arith.index_cast` has no `X to Y` types in `{line}`"))?;
        return Ok(Some(PtoOp::IndexCast {
            dst,
            src: Ssa(operand.trim().to_string()),
            from: from.trim().to_string(),
            to: to.trim().to_string(),
        }));
    }

    // `%dst = arith.muli %lhs, %rhs : index`
    if op == "arith.muli" {
        let eq = line
            .find(" = ")
            .ok_or_else(|| format!("`arith.muli` binds no result in `{line}`"))?;
        let dst = Ssa(line[..eq].trim().to_string());
        let rest = line[eq + 3..]
            .trim()
            .strip_prefix("arith.muli")
            .ok_or_else(|| format!("malformed muli in `{line}`"))?
            .trim();
        let operands = rest.split_once(':').map(|x| x.0).unwrap_or(rest);
        let (lhs, rhs) = operands
            .split_once(',')
            .ok_or_else(|| format!("`arith.muli` has no two operands in `{line}`"))?;
        if let (Some(l), Some(r)) = (
            crate::emitter::resolve(consts, lhs),
            crate::emitter::resolve(consts, rhs),
        ) {
            consts.insert(dst.clone(), l.saturating_mul(r));
        }
        return Ok(Some(PtoOp::MulI {
            dst,
            lhs: Ssa(lhs.trim().to_string()),
            rhs: Ssa(rhs.trim().to_string()),
        }));
    }
    // `%dst = arith.addi %lhs, %rhs : index`
    if op == "arith.addi" {
        let eq = line
            .find(" = ")
            .ok_or_else(|| format!("`arith.addi` binds no result in `{line}`"))?;
        let dst = Ssa(line[..eq].trim().to_string());
        let rest = line[eq + 3..]
            .trim()
            .strip_prefix("arith.addi")
            .ok_or_else(|| format!("malformed addi in `{line}`"))?
            .trim();
        let operands = rest.split_once(':').map(|x| x.0).unwrap_or(rest);
        let (lhs, rhs) = operands
            .split_once(',')
            .ok_or_else(|| format!("`arith.addi` has no two operands in `{line}`"))?;
        if let (Some(l), Some(r)) = (
            crate::emitter::resolve(consts, lhs),
            crate::emitter::resolve(consts, rhs),
        ) {
            consts.insert(dst.clone(), l.saturating_add(r));
        }
        return Ok(Some(PtoOp::MulI {
            dst,
            lhs: Ssa(lhs.trim().to_string()),
            rhs: Ssa(rhs.trim().to_string()),
        }));
    }


    // `%dst = arith.fptosi %src : f32 to i32`
    if op == "arith.fptosi" {
        let eq = line.find(" = ").ok_or_else(|| format!("`arith.fptosi` binds no result in `{line}`"))?;
        let dst = Ssa(line[..eq].trim().to_string());
        let rest = line[eq + 3..].trim().strip_prefix("arith.fptosi")
            .ok_or_else(|| format!("malformed fptosi in `{line}`"))?.trim();
        let operand = rest.split_once(':').map(|x| x.0).unwrap_or(rest);
        return Ok(Some(PtoOp::FpToSi { dst, src: Ssa(operand.trim().to_string()) }));
    }

    // `%dst = "pto.tgetval"(%tile, %idx) : (...) -> f32`
    if op == "pto.tgetval" {
        let eq = line.find(" = ").ok_or_else(|| format!("`pto.tgetval` binds no result in `{line}`"))?;
        let dst = Ssa(line[..eq].trim().to_string());
        let rest = line[eq + 3..].trim().strip_prefix("\"pto.tgetval\"")
            .ok_or_else(|| format!("malformed tgetval in `{line}`"))?.trim();
        let inner = rest.strip_prefix('(').ok_or_else(|| format!("no `(` in `{line}`"))?;
        let close = inner.find(')').ok_or_else(|| format!("no `)` in `{line}`"))?;
        let (src, idx) = inner[..close].split_once(',').ok_or_else(|| format!("no two operands in `{line}`"))?;
        return Ok(Some(PtoOp::GetVal {
            dst,
            src: Ssa(src.trim().to_string()),
            idx: Ssa(idx.trim().to_string()),
        }));
    }


    // `%dst = pto.tci ins(%start : i32) outs(%dst : tile)`
    if op == "pto.tci" {
        // The result is the `outs(...)` tile. The line's own `" = "` is NOT a
        // binding: these ops carry attributes like `{descending = false}`, and
        // splitting on ` = ` there made the whole line the SSA.
        let dst = outs_ssa(line)
            .ok_or_else(|| format!("`pto.tci` has no outs in `{line}`"))?;
        let ins = line
            .split_once("ins(")
            .and_then(|x| x.1.split_once(')'))
            .map(|x| x.0.trim())
            .ok_or_else(|| format!("`pto.tci` has no ins in `{line}`"))?;
        let start = Ssa(ins.split(':').next().unwrap_or(ins).trim().to_string());
        record_outs_shape(line, &dst, shapes);
        return Ok(Some(PtoOp::Tci { dst, start }));
    }


    // `%dst = pto.tsort32 ins(%values, %indices : tiles) outs(%dst : tile)`
    if op == "pto.tsort32" {
        // The result is the `outs(...)` tile. The line's own `" = "` is NOT a
        // binding: these ops carry attributes like `{descending = false}`, and
        // splitting on ` = ` there made the whole line the SSA.
        let dst = outs_ssa(line)
            .ok_or_else(|| format!("`pto.tsort32` has no outs in `{line}`"))?;
        let ins = line
            .split_once("ins(")
            .and_then(|x| x.1.split_once(')'))
            .map(|x| x.0.trim())
            .ok_or_else(|| format!("`pto.tsort32` has no ins in `{line}`"))?;
        let ops = ins.split(':').next().unwrap_or(ins);
        let mut it = ops.split(',').map(|s| s.trim().to_string());
        let values = Ssa(it.next().ok_or_else(|| format!("no values in `{line}`"))?);
        let indices = Ssa(it.next().ok_or_else(|| format!("no indices in `{line}`"))?);
        record_outs_shape(line, &dst, shapes);
        return Ok(Some(PtoOp::TSort32 { dst, values, indices }));
    }


    // `%dst = pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P1010>}) outs(%dst) {axis = "row"}`
    if op == "pto.tgather" {
        // The result is the `outs(...)` tile. The line's own `" = "` is NOT a
        // binding: these ops carry attributes like `{descending = false}`, and
        // splitting on ` = ` there made the whole line the SSA.
        let dst = outs_ssa(line)
            .ok_or_else(|| format!("`pto.tgather` has no outs in `{line}`"))?;
        let ins = line
            .split_once("ins(")
            .and_then(|x| x.1.split_once(','))
            .map(|x| x.0.trim())
            .ok_or_else(|| format!("`pto.tgather` has no src in `{line}`"))?;
        let src = Ssa(ins.to_string());
        let mask = line
            .split_once("mask_pattern<P")
            .and_then(|x| x.1.split_once('>'))
            .map(|x| u32::from_str_radix(x.0.trim(), 2).unwrap_or(0))
            .ok_or_else(|| format!("`pto.tgather` has no mask pattern in `{line}`"))?;
        let axis = line
            .split_once("axis = \"")
            .and_then(|x| x.1.split_once('\"'))
            .map(|x| x.0.to_string())
            .unwrap_or_default();
        record_outs_shape(line, &dst, shapes);
        return Ok(Some(PtoOp::TGather { dst, src, mask, axis }));
    }


    // `%dst = pto.textract ins(%src, %row, %col : tile, index, index) outs(%dst : tile)`
    if op == "pto.textract" {
        // The result is the `outs(...)` tile. The line's own `" = "` is NOT a
        // binding: these ops carry attributes like `{descending = false}`, and
        // splitting on ` = ` there made the whole line the SSA.
        let dst = outs_ssa(line)
            .ok_or_else(|| format!("`pto.textract` has no outs in `{line}`"))?;
        let ins = line
            .split_once("ins(")
            .and_then(|x| x.1.split_once(')'))
            .map(|x| x.0.trim())
            .ok_or_else(|| format!("`pto.textract` has no ins in `{line}`"))?;
        let ops = ins.split(':').next().unwrap_or(ins);
        let mut it = ops.split(',').map(|s| s.trim().to_string());
        let src = Ssa(it.next().ok_or_else(|| format!("no src in `{line}`"))?);
        let row = Ssa(it.next().ok_or_else(|| format!("no row in `{line}`"))?);
        let col = Ssa(it.next().ok_or_else(|| format!("no col in `{line}`"))?);
        record_outs_shape(line, &dst, shapes);
        return Ok(Some(PtoOp::Textract { dst, src, row, col }));
    }

    // `pto.tcvt ins(%src [{sat_mode = ...}] : <tile>) outs(%dst : <tile>)`
    if mnem == "pto.tcvt" {
        let (src, dst) = try_parse_ins1_outs1(line)
            .ok_or_else(|| format!("`pto.tcvt` has no single ins/outs pair in `{line}`"))?;
        // The attribute sits inside `ins(...)`, after the operand name -- a
        // position no other op here uses, and the reason this cannot reuse the
        // plain unary table.
        let ins = extract_parenthesised(line, "ins(")
            .ok_or_else(|| format!("missing `ins(` in `{line}`"))?;
        let sat_mode = ins.find("sat_mode = ").map(|i| {
            let t = &ins[i + "sat_mode = ".len()..];
            let end = t.find('}').unwrap_or(t.len());
            t[..end].trim().to_string()
        });
        return Ok(Some(PtoOp::Cvt { dst, src, sat_mode }));
    }

    // `pto.tgatherb ins(%src, %idx : ...) outs(%dst : ...)`
    if mnem == "pto.tgatherb" {
        let (src, idx, dst) = parse_ins2_outs1(line)?;
        return Ok(Some(PtoOp::GatherB { dst, src, idx }));
    }

    // `pto.barrier <PIPE_ALL>`
    if mnem == "pto.barrier" {
        let rest = line["pto.barrier".len()..].trim();
        let pipe = rest
            .strip_prefix('<')
            .and_then(|r| r.strip_suffix('>'))
            .ok_or_else(|| {
                format!("`pto.barrier` has no `<PIPE>` operand in `{line}`; the pipe \
                         decides what the barrier orders and cannot be assumed")
            })?
            .trim()
            .to_string();
        return Ok(Some(PtoOp::Barrier { pipe }));
    }

    // Matmul: `pto.tmatmul ins(%l, %r : ...) outs(%acc : ...)`
    if mnem == "pto.tmatmul" {
        let (lhs, rhs, dst) = parse_ins2_outs1(line)?;
        return Ok(Some(PtoOp::TMatmul { dst, lhs, rhs }));
    }

    // Transpose. Two ins operands: the source and a workspace tile.
    if mnem == "pto.ttrans" {
        let (src, scratch, dst) = parse_ins2_outs1(line)?;
        return Ok(Some(PtoOp::Transpose { dst, src, scratch }));
    }

    // Tile move
    if mnem == "pto.tmov" {
        if let Some((src, dst)) = try_parse_ins1_outs1(line) {
            return Ok(Some(PtoOp::TMov { src, dst }));
        }
    }

    // Listed in MODELLED_OPS but no arm claimed the line -- the two have
    // drifted apart, or an arm's inner parse declined (several are written as
    // `if let Some(..) = try_parse_...`, which used to fall through to a silent
    // drop). Either way it is a parser bug, not an unsupported input.
    Err(format!(
        "op `{op}` is listed in parser::MODELLED_OPS but no parse arm claimed          this line: `{line}`"
    ))
}

// ─── Per-op line parsers ────────────────────────────────────────────────────

/// The two operands of a `key = [a, b]` list, verbatim.
///
/// Tolerates the spaced and unspaced spellings. Returns `None` when the key is
/// absent or does not hold exactly two operands -- the caller refuses rather
/// than guessing, because a guessed geometry is how 321 non-zero partition
/// offsets came to be lifted as zero.
/// Every operand of `key = [...]`, in order. The rank-general form of
/// [`bracket_pair`], which returns `None` for anything but exactly two.
fn bracket_list(line: &str, key: &str) -> Option<Vec<String>> {
    let at = line
        .find(&format!("{key} = ["))
        .map(|i| i + key.len() + 4)
        .or_else(|| line.find(&format!("{key}=[")).map(|i| i + key.len() + 2))?;
    let end = line[at..].find(']')? + at;
    let parts = split_top_commas(&line[at..end]);
    if parts.is_empty() {
        return None;
    }
    Some(parts.iter().map(|p| p.trim().to_string()).collect())
}

fn bracket_pair(line: &str, key: &str) -> Option<[String; 2]> {
    let at = line
        .find(&format!("{key} = ["))
        .map(|i| i + key.len() + 4)
        .or_else(|| line.find(&format!("{key}=[")).map(|i| i + key.len() + 2))?;
    let end = line[at..].find(']')? + at;
    let parts = split_top_commas(&line[at..end]);
    if parts.len() != 2 {
        return None;
    }
    Some([parts[0].trim().to_string(), parts[1].trim().to_string()])
}

fn try_parse_subview(
    line: &str,
    shapes: &mut BTreeMap<Ssa, TileShape>,
) -> Result<Option<PtoOp>, String> {
    let Some(eq) = line.find(" = \"pto.subview\"(") else {
        return Ok(None);
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let rest = &line[eq + " = \"pto.subview\"(".len()..];
    let close = rest
        .find(')')
        .ok_or("subview operand list is not closed")?;
    let operands = split_top_commas(&rest[..close]);
    let mut it = operands.iter().map(|o| Ssa(o.trim().to_string()));
    let src = it.next().ok_or("subview has no source operand")?;
    let offsets: Vec<Ssa> = it.collect();
    if offsets.is_empty() {
        return Err(format!(
            "subview `{}` has no offset operands; a subview at no offset is not \
             a subview, and assuming the origin is what made 321 non-zero \
             partition offsets lift as zero",
            dst.0
        ));
    }

    let tail = &rest[close + 1..];
    let segment_sizes = tail
        .find("operandSegmentSizes = ")
        .map(|i| &tail[i + "operandSegmentSizes = ".len()..])
        .and_then(|t| {
            // `array<i32: 1, 2, 0, 0>` -- balanced to the matching `>`.
            let end = t.find('>')? + 1;
            Some(t[..end].to_string())
        })
        .ok_or_else(|| {
            format!("subview `{}` has no `operandSegmentSizes` attribute", dst.0)
        })?;
    let sizes = bracket_pair(tail, "sizes")
        .ok_or_else(|| format!("subview `{}` has no readable 2-D `sizes`", dst.0))?;

    // The result type is after the final `->`, like partition_view.
    let arrow = tail.rfind("->").ok_or("subview missing `->`")?;
    let ty = parse_type(tail[arrow + 2..].trim())?;
    let PtoType::TileBuf {
        rows,
        cols,
        v_rows,
        v_cols,
        dtype,
        loc,
    } = ty
    else {
        return Err(format!("subview `{}` does not produce a tile_buf", dst.0));
    };
    shapes.insert(
        dst.clone(),
        TileShape {
            rows,
            cols,
            v_rows,
            v_cols,
            dtype,
            loc: Some(loc),
        },
    );
    Ok(Some(PtoOp::Subview {
        dst,
        src,
        offsets,
        sizes,
        segment_sizes,
        ty,
    }))
}

fn try_parse_make_tensor_view(line: &str) -> Result<Option<PtoOp>, String> {
    // `%pto0 = pto.make_tensor_view %arg601, shape = [...], strides = [...] : !pto.tensor_view<?x?xf32>`
    let Some(eq) = line.find(" = pto.make_tensor_view ") else {
        return Ok(None);
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let rest = &line[eq + " = pto.make_tensor_view ".len()..];
    let ptr_end = rest
        .find(',')
        .ok_or("make_tensor_view missing comma after ptr")?;
    let ptr = Ssa(rest[..ptr_end].trim().to_string());
    let ty_start = rest.rfind(':').ok_or("make_tensor_view missing `:`")?;
    let ty = parse_type(rest[ty_start + 1..].trim())?;
    let (rank, dtype) = match ty {
        PtoType::TensorView { rank, dtype } => (rank, dtype),
        _ => return Err("make_tensor_view produced non-tensor_view type".into()),
    };
    let shape = bracket_list(line, "shape")
        .ok_or_else(|| format!("make_tensor_view has no readable `shape` in `{line}`"))?;
    let strides = bracket_list(line, "strides")
        .ok_or_else(|| format!("make_tensor_view has no readable `strides` in `{line}`"))?;
    // The declared rank, the shape operands and the stride operands are three
    // statements of the same number, and a view where they disagree is not a
    // view this parser has understood. Checked rather than reconciled.
    if shape.len() != strides.len() || shape.len() as u32 != rank {
        return Err(format!(
            "make_tensor_view declares rank {rank} but has {} shape and {} stride \
             operand(s) in `{line}`",
            shape.len(),
            strides.len()
        ));
    }
    let layout = line
        .split_once("{layout = ")
        .and_then(|(_, r)| r.split_once('}'))
        .map(|(v, _)| v.trim().to_string());
    Ok(Some(PtoOp::MakeTensorView {
        dst,
        ptr,
        dtype,
        shape,
        strides,
        layout,
    }))
}

fn try_parse_partition_view(
    line: &str,
    consts: &BTreeMap<Ssa, i64>,
    shapes: &mut BTreeMap<Ssa, TileShape>,
) -> Result<Option<PtoOp>, String> {
    let Some(eq) = line.find(" = pto.partition_view ") else {
        return Ok(None);
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let rest = &line[eq + " = pto.partition_view ".len()..];
    // Output type is after the final `->`; that's where we grab R/C/dtype
    // (the source type gives us no concrete shape).
    let arrow = rest.rfind("->").ok_or("partition_view missing `->`")?;
    let src_end = rest.find(',').ok_or("partition_view missing operands")?;
    let src = Ssa(rest[..src_end].trim().to_string());
    let out_ty = parse_type(rest[arrow + 2..].trim())?;
    // Read rank-general, then narrow for the rank-2 case. Reading as a pair
    // first is what refused the five rank-5 `gmm_*` artifacts on this line.
    let offsets = bracket_list(line, "offsets")
        .ok_or_else(|| format!("partition_view has no readable `offsets` in `{line}`"))?;
    let sizes = bracket_list(line, "sizes")
        .ok_or_else(|| format!("partition_view has no readable `sizes` in `{line}`"))?;
    let (rows, cols, dtype) = match out_ty {
        PtoType::PartitionView { rows, cols, dtype } => {
            if offsets.len() != 2 || sizes.len() != 2 {
                return Err(format!(
                    "partition_view produces a rank-2 view but has {} offset(s) and {} \
                     size(s) in `{line}`",
                    offsets.len(),
                    sizes.len()
                ));
            }
            (rows, cols, dtype)
        }
        // A rank-N view IS a partition type -- it just has no rows/cols to
        // record, so it binds no TileShape and anything needing one from it
        // fails closed. The old `_` arm said "non-partition type", which was
        // untrue, and before that the type parser refused the rank outright.
        PtoType::PartitionViewNd { dims, dtype } => {
            if offsets.len() != dims.len() || sizes.len() != dims.len() {
                return Err(format!(
                    "partition_view produces a rank-{} view but has {} offset(s) and {} \
                     size(s) in `{line}`",
                    dims.len(),
                    offsets.len(),
                    sizes.len()
                ));
            }
            return Ok(Some(PtoOp::PartitionViewNd {
                dst,
                src,
                dims,
                dtype,
                offsets,
                sizes,
            }));
        }
        _ => return Err("partition_view produced non-partition type".into()),
    };
    shapes.insert(
        dst.clone(),
        TileShape {
            rows,
            cols,
            v_rows: rows,
            v_cols: cols,
            dtype,
            loc: None,
        },
    );
    // `consts` stays a parameter: the emitter resolves these operands through
    // it. S1 still does not VALIDATE the geometry here -- it records it, and
    // `emitter::emit_op` refuses what it cannot honour.
    let _ = consts.len();
    Ok(Some(PtoOp::PartitionView {
        dst,
        src,
        rows,
        cols,
        dtype,
        offsets: [offsets[0].clone(), offsets[1].clone()],
        sizes: [sizes[0].clone(), sizes[1].clone()],
    }))
}

fn try_parse_alloc_tile(
    line: &str,
    shapes: &mut BTreeMap<Ssa, TileShape>,
) -> Result<Option<PtoOp>, String> {
    let Some(eq) = line.find(" = pto.alloc_tile") else {
        return Ok(None);
    };
    let dst = Ssa(line[..eq].trim().to_string());
    let ty_start = line.rfind(':').ok_or("alloc_tile missing `:`")?;
    let ty = parse_type(line[ty_start + 1..].trim())?;
    if let PtoType::TileBuf {
        rows,
        cols,
        v_rows,
        v_cols,
        dtype,
        loc,
    } = ty
    {
        shapes.insert(
            dst.clone(),
            TileShape {
                rows,
                cols,
                v_rows,
                v_cols,
                dtype,
                loc: Some(loc),
            },
        );
        Ok(Some(PtoOp::AllocTile {
            dst,
            ty: PtoType::TileBuf {
                rows,
                cols,
                v_rows,
                v_cols,
                dtype,
                loc,
            },
        }))
    } else {
        Err("alloc_tile produced non-tile_buf type".into())
    }
}

/// The type annotation of the single operand inside `ins(...)` or `outs(...)`.
///
/// `tload`/`tstore` need the VIEW's own type rather than one rebuilt from the
/// tile's shape: 35 of the corpus's 487 such lines have a view whose dims
/// differ from its tile's, and a rank-N view has no rows/cols to rebuild from.
fn view_operand_type(line: &str, key: &str) -> Result<PtoType, String> {
    let inner = extract_parenthesised(line, key)
        .ok_or_else(|| format!("missing `{key}` in `{line}`"))?;
    let colon = inner
        .find(':')
        .ok_or_else(|| format!("operand in `{key}` of `{line}` has no type annotation"))?;
    parse_type(inner[colon + 1..].trim())
}

fn try_parse_tload(line: &str) -> Result<Option<PtoOp>, String> {
    if !line.starts_with("pto.tload ") {
        return Ok(None);
    }
    let (view, tile) = parse_ins1_outs1_exact(line, "pto.tload")?;
    let view_ty = view_operand_type(line, "ins(")?;
    Ok(Some(PtoOp::TLoad {
        view,
        tile,
        view_ty,
    }))
}

fn try_parse_tstore(line: &str) -> Result<Option<PtoOp>, String> {
    if !line.starts_with("pto.tstore ") {
        return Ok(None);
    }
    let (tile, view) = parse_ins1_outs1_exact(line, "pto.tstore")?;
    let view_ty = view_operand_type(line, "outs(")?;
    Ok(Some(PtoOp::TStore {
        tile,
        view,
        view_ty,
    }))
}

// ─── ins(...) outs(...) helpers ────────────────────────────────────────────

/// Pull the single SSA from inside `ins(%x : ...)` and the single SSA from
/// `outs(%y : ...)`.
fn try_parse_ins1_outs1(line: &str) -> Option<(Ssa, Ssa)> {
    let ins = extract_parenthesised(line, "ins(")?;
    let outs = extract_parenthesised(line, "outs(")?;
    let a = first_ssa_before_colon(&ins)?;
    let b = first_ssa_before_colon(&outs)?;
    Some((a, b))
}

/// Same, but require the line to start with `expect` — used for ops whose
/// keyword disambiguates what the two sides mean (tload vs tstore).
fn parse_ins1_outs1_exact(line: &str, expect: &str) -> Result<(Ssa, Ssa), String> {
    if !line.starts_with(expect) {
        return Err(format!("expected `{expect}` at start of `{line}`"));
    }
    try_parse_ins1_outs1(line).ok_or_else(|| format!("malformed ins/outs in `{line}`"))
}

/// Pull two SSAs from `ins(%a, %b : ...)` and the single SSA from `outs(%c : ...)`.
/// The `n`th operand type in an `ins(...)` list: the text after the separating
/// colon, split on top-level commas.
///
/// `parse_ins2_outs1` deliberately throws these away -- it wants SSA names --
/// and for two-tile ops that is harmless, because both types are recoverable
/// from the shape map. For a tile-by-scalar op it is not: the scalar has no
/// shape, so its width exists ONLY here.
fn ins_operand_type(line: &str, n: usize) -> Option<String> {
    let ins = extract_parenthesised(line, "ins(")?;
    let colon = ins.find(':')?;
    let tys = split_top_commas(&ins[colon + 1..]);
    tys.get(n).map(|t| t.trim().to_string())
}

fn parse_ins2_outs1(line: &str) -> Result<(Ssa, Ssa, Ssa), String> {
    let ins =
        extract_parenthesised(line, "ins(").ok_or_else(|| format!("missing `ins(` in `{line}`"))?;
    let outs = extract_parenthesised(line, "outs(")
        .ok_or_else(|| format!("missing `outs(` in `{line}`"))?;
    let ins_head = ins.split(':').next().unwrap_or("");
    let pieces = split_top_commas(ins_head);
    if pieces.len() < 2 {
        return Err(format!("expected 2 ins operands, got `{ins_head}`"));
    }
    let a = Ssa(pieces[0].trim().to_string());
    let b = Ssa(pieces[1].trim().to_string());
    let c = first_ssa_before_colon(&outs).ok_or_else(|| format!("malformed outs in `{line}`"))?;
    Ok((a, b, c))
}

/// Return the substring inside `kw(...)` handling nested parens/brackets.
/// Returns None if not found.
fn extract_parenthesised(line: &str, kw: &str) -> Option<String> {
    let start = line.find(kw)? + kw.len();
    let bytes = line.as_bytes();
    let mut depth = 1usize;
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
    // An SSA name has no space in it. Stopping at the first one matters because
    // an operand may carry an inline attribute dictionary AFTER the name:
    // `ins(%div {sat_mode = #pto<saturation_mode ON>} : ...)` is how `pto.tcvt`
    // spells saturation, and 10 of the corpus's 188 `tcvt` instances do. Taking
    // the whole token bound an SSA literally named
    // "%div {sat_mode = #pto<saturation_mode ON>}", which then has no shape --
    // a fail-closed error, but one that names a string nothing ever defined.
    let tok = tok.split_whitespace().next()?;
    if tok.starts_with('%') {
        Some(Ssa(tok.to_string()))
    } else {
        None
    }
}

fn split_top_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0isize;
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
