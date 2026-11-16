// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Post-parse pattern fusion.
//!
//! Two patterns: softmax and rms-norm. The canonical softmax chain emitted by
//! `mlir_to_pto.rs` is
//!
//! ```text
//!   %max = pto.trowmax       %src
//!   %sub = pto.trowexpandsub %src, %max
//!   %exp = pto.texp          %sub
//!   %sum = pto.trowsum       %exp
//!   %out = pto.trowexpanddiv %exp, %sum
//! ```
//!
//! When those five ops are contiguous, consume one another in the above
//! order, and none of `%max`, `%sub`, `%exp`, `%sum` is read downstream,
//! we collapse them into a single `PtoOp::Softmax { dst: %out, src: %src }`.
//!
//! The dead-intermediate check is the soundness condition. If any of the
//! temporaries is used later in the function, fusion is unsafe -- we'd be
//! dropping a live value -- so we skip the match.

use crate::ast::*;
use std::collections::{BTreeMap, BTreeSet};

/// Rewrite each function's op list in-place, fusing recognised patterns.
pub fn fuse_patterns(m: &mut PtoModule) {
    for f in &mut m.funcs {
        f.ops = fuse_softmax(&f.ops);
        let shapes = f.shapes.clone();
        f.ops = fuse_rms_norm(&f.ops, &shapes);
    }
    fuse_quantize_module(m);
}

/// Run [`fuse_quantize`] at every nesting level.
///
/// The other two rules only ever see top level because `convert_pto_to_rust`
/// unrolls a constant-bound loop before parsing; raw mode does not unroll, so
/// the quantise chain is still inside the `scf.for` body there. Fusion inside
/// a region is safe for the same reason it is safe at top level -- the
/// dead-intermediate check runs on the body's own op list, and a body cannot
/// export an SSA to the enclosing function except through a loop-carried value,
/// which this corpus does not use.
pub(crate) fn fuse_quantize_module(m: &mut PtoModule) {
    for f in &mut m.funcs {
        let shapes = f.shapes.clone();
        fuse_quantize_ops(&mut f.ops, &shapes);
    }
}

fn fuse_quantize_ops(ops: &mut Vec<PtoOp>, shapes: &BTreeMap<Ssa, TileShape>) {
    *ops = fuse_quantize(ops, shapes);
    for op in ops.iter_mut() {
        match op {
            PtoOp::For { body, .. } => fuse_quantize_ops(body, shapes),
            PtoOp::If {
                then_body,
                else_body,
                ..
            } => {
                fuse_quantize_ops(then_body, shapes);
                if let Some(else_body) = else_body {
                    fuse_quantize_ops(else_body, shapes);
                }
            }
            _ => {}
        }
    }
}

/// Collapse the decomposed INT8 quantiser into one `PtoOp::Quantize`.
///
/// `mlir_to_pto` emits, for `__tile_quantize_f32_i8` with a TILE scale:
///
/// ```text
///   %ratio = pto.tcolexpand       %scale        (only when rows > 1)
///   %div   = pto.trowexpanddiv    %src, %ratio
///   %f16   = pto.tcvt sat=ON      %div
///   %out   = pto.tcvt sat=ON      %f16
/// ```
///
/// The two sat casts are not two independent conversions that could each be
/// lowered: together they are the part's f32 -> f16 -> s8 quantise route (there
/// is no f32 -> s8 convert), and the saturation is what makes an out-of-range
/// value clamp instead of wrap. No `tile_std` cast takes a saturation mode, so
/// the chain lifts whole or not at all -- the same load-bearing fusion as
/// softmax and rms-norm.
///
/// Conditions, all required:
///
///   1. the ops appear in this order, each consuming the previous;
///   2. the scale is a per-tensor `1 x SC` f32 tile whose valid extent is one
///      column (`v_col=1` for the padded 1x1 load the corpus stores);
///   3. BOTH casts carry `sat_mode = ... ON`. An absent or OFF mode is a
///      different overflow behaviour and is left to the per-op refusal;
///   4. the dtypes are exactly f32 -> f32 -> f16 -> i8 and every step keeps the
///      source shape, so the fused call is shape-preserving;
///   5. no intermediate is read past the window, the same soundness condition
///      the other two rules use.
pub(crate) fn fuse_quantize(ops: &[PtoOp], shapes: &BTreeMap<Ssa, TileShape>) -> Vec<PtoOp> {
    let inert = |op: &PtoOp| {
        matches!(
            op,
            PtoOp::AllocTile { .. } | PtoOp::ScalarConst { .. } | PtoOp::Barrier { .. }
        )
    };
    let idx: Vec<usize> = (0..ops.len()).filter(|&i| !inert(&ops[i])).collect();
    let compute: Vec<PtoOp> = idx.iter().map(|&i| ops[i].clone()).collect();
    let reads_after = compute_reads_after(&compute);

    let mut fused: BTreeMap<usize, PtoOp> = BTreeMap::new();
    let mut drop_at: BTreeSet<usize> = BTreeSet::new();
    let mut c = 0;
    while c < compute.len() {
        if let Some((op, skip)) = try_quantize(&compute, c, &reads_after, shapes) {
            fused.insert(idx[c + skip - 1], op);
            for k in c..c + skip {
                drop_at.insert(idx[k]);
            }
            c += skip;
            continue;
        }
        c += 1;
    }
    if fused.is_empty() {
        return ops.to_vec();
    }

    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if let Some(f) = fused.get(&i) {
            out.push(f.clone());
        } else if !drop_at.contains(&i) {
            out.push(op.clone());
        }
    }
    out
}

/// Only a `tcvt` that says it saturates can be folded into the quantiser: the
/// quantiser clamps, so an un-attributed (wrap) cast is a different program.
fn saturates_on(mode: &Option<String>) -> bool {
    mode.as_deref()
        .map(|m| m.contains("saturation_mode ON"))
        .unwrap_or(false)
}

fn try_quantize(
    ops: &[PtoOp],
    i: usize,
    reads_after: &[BTreeSet<Ssa>],
    shapes: &BTreeMap<Ssa, TileShape>,
) -> Option<(PtoOp, usize)> {
    // Two forms of the same chain. With more than one row the scale has to be
    // replicated down the rows before the broadcast divide; at one row there is
    // nothing to replicate and `mlir_to_pto` emits no `tcolexpand`.
    let (scale, ratio, col_expanded, div_i, data_hint) = match ops.get(i)? {
        PtoOp::ColExpand { dst, src } => {
            let s_sh = shapes.get(src)?;
            let r_sh = shapes.get(dst)?;
            if s_sh.dtype != DType::F32
                || s_sh.rows != 1
                || (s_sh.cols != 1 && s_sh.v_cols != 1)
                || r_sh.dtype != DType::F32
                || r_sh.v_cols != 1
            {
                return None;
            }
            (src.clone(), dst.clone(), true, i + 1, None)
        }
        PtoOp::RowExpandDiv { src, red, .. } => {
            let src_sh = shapes.get(src)?;
            let s_sh = shapes.get(red)?;
            if src_sh.rows != 1
                || s_sh.dtype != DType::F32
                || s_sh.rows != 1
                || (s_sh.cols != 1 && s_sh.v_cols != 1)
            {
                return None;
            }
            (red.clone(), red.clone(), false, i, Some(src.clone()))
        }
        _ => return None,
    };

    let (div_src, div_dst) = match ops.get(div_i)? {
        PtoOp::RowExpandDiv { dst, src, red } if *red == ratio => (src.clone(), dst.clone()),
        _ => return None,
    };
    let data_src = match data_hint {
        Some(hint) if hint != div_src => return None,
        Some(hint) => hint,
        None => div_src,
    };

    let (f16_dst, f16_src) = match ops.get(div_i + 1)? {
        PtoOp::Cvt { dst, src, sat_mode } if *src == div_dst && saturates_on(sat_mode) => {
            (dst.clone(), src.clone())
        }
        _ => return None,
    };
    let out_dst = match ops.get(div_i + 2)? {
        PtoOp::Cvt { dst, src, sat_mode } if *src == f16_dst && saturates_on(sat_mode) => {
            dst.clone()
        }
        _ => return None,
    };

    let src_sh = shapes.get(&data_src)?;
    let div_sh = shapes.get(&div_dst)?;
    let f16_sh = shapes.get(&f16_dst)?;
    let out_sh = shapes.get(&out_dst)?;
    let same_shape = |sh: &TileShape| (sh.rows, sh.cols) == (src_sh.rows, src_sh.cols);
    if src_sh.dtype != DType::F32
        || div_sh.dtype != DType::F32
        || !same_shape(div_sh)
        || f16_sh.dtype != DType::F16
        || !same_shape(f16_sh)
        || out_sh.dtype != DType::I8
        || !same_shape(out_sh)
    {
        return None;
    }
    if col_expanded {
        let r_sh = shapes.get(&ratio)?;
        if r_sh.rows != src_sh.rows || r_sh.v_cols != 1 {
            return None;
        }
    }

    // Dead intermediates. `scale` itself is read by the fused op, so it is not
    // in this set; the expanded copy is.
    let skip = (div_i + 3) - i;
    let past = i + skip;
    let mut dead = vec![&div_dst, &f16_dst];
    if col_expanded {
        dead.push(&ratio);
    }
    if past < reads_after.len() {
        for t in dead {
            if reads_after[past].contains(t) {
                return None;
            }
        }
    }
    // `f16_src` is named only to make the chain's topology explicit; the guard
    // above already established it is `div_dst`.
    debug_assert_eq!(f16_src, div_dst);

    Some((
        PtoOp::Quantize {
            dst: out_dst,
            src: data_src,
            scale,
        },
        skip,
    ))
}

/// Collapse the rms-norm chain `mlir_to_pto` emits into one `RmsNorm`.
///
/// The chain, read off `crates/mlir_to_pto_tests/l0/e.pto` (`@ds4_rmsmul_4096`)
/// and identical in the other five artifacts that carry it:
///
/// ```text
///   %sq  = pto.tmul       %src, %src        1xN -> 1xN
///   %sum = pto.trowsum    %sq, %scratch     1xN -> 8x1
///   %mn  = pto.tmuls      %sum, 1/N         8x1 -> 1x8
///   %pe  = pto.tadds      %mn, eps          1x8 -> 1x8
///   %rt  = pto.tsqrt      %pe               1x8 -> 1x8
///   %rs  = pto.trecip     %rt               1x8 -> 1x8
///   %bc  = pto.trowexpand %rs               1x8 -> 1xN
///   %out = pto.tmul       %src, %bc         1xN -> 1xN
/// ```
///
/// Eight ops, with a `pto.barrier <PIPE_ALL>` between every adjacent pair.
///
/// **This is not the chain the sibling branch's rule matched.** That rule
/// expected six ops ending `pto.trsqrt` then `pto.trowexpandmul`, and it was
/// right about the algebra and wrong about the text: there is no `pto.trsqrt`
/// anywhere in the corpus (`mlir_to_pto` spells rsqrt as sqrt-then-reciprocal),
/// and the 16 `pto.trowexpandmul` instances belong to the u8/u32 quantized
/// kernels, not to a normalisation. Its three tests all ran on a hand-written
/// fixture of its own chain, so they passed while the rule could not fire on
/// any file. A pattern rule's fixture has to be the emitter's own output, or
/// the test measures the author's model of the emitter.
///
/// Why fuse rather than lower op by op: four of the eight have no tile-API
/// counterpart. `pto.tadds` has no wrapper, `pto.tsqrt` has `__tile_sqrt_f32`
/// with no wrapper, `pto.trecip` has nothing in the tile API at all
/// (`__tile_reciprocal_f32` is the buffer API over `UbBuf`), and
/// `pto.trowexpand` has nothing at any layer. But the whole of it is
/// `tile_rms_norm_f32(src, eps)`, which exists as a safe wrapper. So fusion is
/// the only lift this computation has.
///
/// Conditions, all required:
///
///   1. the eight ops appear in this order, each consuming the previous;
///   2. `%src` feeds BOTH the square and the final multiply -- that is what
///      makes it a norm OF `%src` rather than of something else;
///   3. the `tmuls` scalar is 1/cols for the source tile's own width. Without
///      this the rule fires on any `x * k` chain of the same shape and
///      relabels a scaled sum of squares a normalisation. Corpus values are
///      1/4096, 1/1024, 1/512 and 1/16384, each matching its tile;
///   4. no intermediate is read after the window, the same soundness
///      condition the softmax rule uses.
fn fuse_rms_norm(ops: &[PtoOp], shapes: &BTreeMap<Ssa, TileShape>) -> Vec<PtoOp> {
    // Match over the LOAD-BEARING ops only. Three kinds of line are
    // interleaved through the chain and carry no dataflow: `pto.alloc_tile`
    // (every tile is allocated on the line before its writer),
    // `arith.constant`, and -- the one that matters here -- `pto.barrier`,
    // which sits between EVERY adjacent pair of the eight. A window over the
    // raw op list cannot match two consecutive chain ops, let alone eight.
    //
    // They are stepped over for matching and left in place afterwards: an
    // alloc nothing reads is harmless, and a barrier orders whatever else is
    // in the function, so dropping one would be a semantic change this pass
    // has no grounds for.
    let inert = |op: &PtoOp| {
        matches!(
            op,
            PtoOp::AllocTile { .. } | PtoOp::ScalarConst { .. } | PtoOp::Barrier { .. }
        )
    };
    let idx: Vec<usize> = (0..ops.len()).filter(|&i| !inert(&ops[i])).collect();
    let compute: Vec<PtoOp> = idx.iter().map(|&i| ops[i].clone()).collect();
    let reads_after = compute_reads_after(&compute);
    // The scalar literals, which `PtoFunc::consts` cannot hold: it is
    // `BTreeMap<Ssa, i64>` and both of these are floats. Condition 3 needs the
    // value, so it is read from the `arith.constant` ops themselves.
    let lits = float_literals(ops);

    let mut fused: BTreeMap<usize, PtoOp> = BTreeMap::new();
    let mut drop_at: BTreeSet<usize> = BTreeSet::new();
    let mut c = 0;
    while c < compute.len() {
        if let Some((op, skip)) = try_rms_norm(&compute, c, &reads_after, shapes, &lits) {
            // Placed at the window's LAST position, which is where the
            // value the fused op produces becomes available.
            //
            // The ported rule justified this by saying the first position puts
            // the op ahead of its own `alloc_tile` and the Rust emitter, which
            // mints names from the allocs, fails with `unresolved SSA`. That is
            // true of the layout its fixture had -- each tile allocated on the
            // line before its writer -- and NOT of this corpus: `mlir_to_pto`
            // hoists every alloc above the chain (`l0/e.pto` allocates at lines
            // 18-26 and the chain runs 27-41), so both positions resolve.
            // Mutation-checked: moving the insert to `idx[c]` kills no test.
            //
            // Kept at the last position anyway, because it is correct for both
            // layouts and the interleaved one is the harder case. Recorded as
            // a choice with a known-equivalent alternative rather than as a
            // constraint the corpus imposes.
            fused.insert(idx[c + skip - 1], op);
            for k in c..c + skip {
                drop_at.insert(idx[k]);
            }
            c += skip;
            continue;
        }
        c += 1;
    }
    if fused.is_empty() {
        return ops.to_vec();
    }

    let mut out = Vec::with_capacity(ops.len());
    for (i, op) in ops.iter().enumerate() {
        if let Some(f) = fused.get(&i) {
            out.push(f.clone());
        } else if !drop_at.contains(&i) {
            out.push(op.clone());
        }
    }
    out
}

/// Every `arith.constant` with a float literal, as text.
///
/// The dtype test is defensive rather than load-bearing: admitting integer
/// constants too is behaviourally equivalent on this corpus (mutation-checked
/// -- it kills no test), because the two literals the rule reads are always
/// `f32` and an integer would fail the 1/N comparison anyway. Kept because it
/// states which constants this map is for.
fn float_literals(ops: &[PtoOp]) -> BTreeMap<Ssa, String> {
    let mut out = BTreeMap::new();
    for op in ops {
        if let PtoOp::ScalarConst {
            dst,
            value: ConstVal::Typed { lit, ty },
        } = op
        {
            if ty.starts_with('f') || ty.starts_with("bf") {
                out.insert(dst.clone(), lit.clone());
            }
        }
    }
    out
}

fn try_rms_norm(
    ops: &[PtoOp],
    i: usize,
    reads_after: &[BTreeSet<Ssa>],
    shapes: &BTreeMap<Ssa, TileShape>,
    lits: &BTreeMap<Ssa, String>,
) -> Option<(PtoOp, usize)> {
    const WINDOW: usize = 8;
    if i + WINDOW > ops.len() {
        return None;
    }
    let (sq_dst, src) = match &ops[i] {
        PtoOp::BinOp {
            kind: BinKind::Mul,
            dst,
            a,
            b,
        } if a == b => (dst.clone(), a.clone()),
        _ => return None,
    };
    let sum_dst = match &ops[i + 1] {
        PtoOp::RowReduce {
            kind: RowKind::Sum,
            dst,
            src: s,
        } if *s == sq_dst => dst.clone(),
        _ => return None,
    };
    let (mean_dst, scale_ssa) = match &ops[i + 2] {
        PtoOp::BinOpScalar {
            kind: ScalarBinKind::Muls,
            dst,
            a,
            scalar,
            ..
        } if *a == sum_dst => (dst.clone(), scalar.clone()),
        _ => return None,
    };
    let (eps_dst, eps_ssa) = match &ops[i + 3] {
        PtoOp::BinOpScalar {
            kind: ScalarBinKind::Adds,
            dst,
            a,
            scalar,
            ..
        } if *a == mean_dst => (dst.clone(), scalar.clone()),
        _ => return None,
    };
    let sqrt_dst = match &ops[i + 4] {
        PtoOp::UnOp {
            kind: UnKind::Sqrt,
            dst,
            src: s,
        } if *s == eps_dst => dst.clone(),
        _ => return None,
    };
    let recip_dst = match &ops[i + 5] {
        PtoOp::UnOp {
            kind: UnKind::Recip,
            dst,
            src: s,
        } if *s == sqrt_dst => dst.clone(),
        _ => return None,
    };
    let bcast_dst = match &ops[i + 6] {
        PtoOp::RowExpand { dst, src: s } if *s == recip_dst => dst.clone(),
        _ => return None,
    };
    // The final multiply takes `%src` and the broadcast reciprocal, in either
    // operand order -- multiplication commutes and nothing guarantees which
    // side `mlir_to_pto` puts the broadcast on.
    let out_dst = match &ops[i + 7] {
        PtoOp::BinOp {
            kind: BinKind::Mul,
            dst,
            a,
            b,
        } if (*a == src && *b == bcast_dst) || (*a == bcast_dst && *b == src) => dst.clone(),
        _ => return None,
    };

    // Condition 3: the scalar really is 1/N for this tile's own width.
    let cols = shapes.get(&src).map(|s| s.cols)?;
    let want = 1.0f64 / cols as f64;
    let got: f64 = lits.get(&scale_ssa)?.parse().ok()?;
    if (got - want).abs() > want * 1e-3 {
        return None;
    }
    // The eps literal has to be resolvable too -- the fused op carries it into
    // the generated call, and a missing one would mean inventing a value.
    let eps = lits.get(&eps_ssa)?.clone();

    // Dead intermediates. `reads_after[j]` includes `ops[j]`'s own reads, so
    // the first index PAST an eight-op window at `i..=i+7` is `i + 8`. Using
    // `i + 7` would count the window's own last op as an outside reader --
    // it reads `%bc` -- and the rule would never fire on anything.
    let past = i + WINDOW;
    if past < reads_after.len() {
        for t in [
            &sq_dst,
            &sum_dst,
            &mean_dst,
            &eps_dst,
            &sqrt_dst,
            &recip_dst,
            &bcast_dst,
        ] {
            if reads_after[past].contains(t) {
                return None;
            }
        }
    }

    Some((
        PtoOp::RmsNorm {
            dst: out_dst,
            src,
            eps,
        },
        WINDOW,
    ))
}

fn fuse_softmax(ops: &[PtoOp]) -> Vec<PtoOp> {
    // Build the live-after-this-op set for each op, walking right-to-left.
    // An SSA is "escaping" at position i if it's read at any j > i+4 (i.e.
    // past the five-op window).
    let reads_after = compute_reads_after(ops);

    let mut out: Vec<PtoOp> = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if let Some((softmax_op, skip)) = try_match_softmax(ops, i, &reads_after) {
            out.push(softmax_op);
            i += skip;
        } else {
            out.push(ops[i].clone());
            i += 1;
        }
    }
    out
}

/// Try to match the softmax chain starting at `ops[i]`. On success, return
/// the synthetic op and the number of source ops consumed (5).
fn try_match_softmax(
    ops: &[PtoOp],
    i: usize,
    reads_after: &[BTreeSet<Ssa>],
) -> Option<(PtoOp, usize)> {
    if i + 5 > ops.len() {
        return None;
    }

    let (src, max_dst) = match &ops[i] {
        PtoOp::RowReduce {
            kind: RowKind::Max,
            dst,
            src,
        } => (src.clone(), dst.clone()),
        _ => return None,
    };
    let (sub_src, sub_red, sub_dst) = match &ops[i + 1] {
        PtoOp::RowExpandSub { src, red, dst } => (src.clone(), red.clone(), dst.clone()),
        _ => return None,
    };
    let (exp_src, exp_dst) = match &ops[i + 2] {
        PtoOp::UnOp {
            kind: UnKind::Exp,
            src,
            dst,
        } => (src.clone(), dst.clone()),
        _ => return None,
    };
    let (sum_src, sum_dst) = match &ops[i + 3] {
        PtoOp::RowReduce {
            kind: RowKind::Sum,
            dst,
            src,
        } => (src.clone(), dst.clone()),
        _ => return None,
    };
    let (div_src, div_red, div_dst) = match &ops[i + 4] {
        PtoOp::RowExpandDiv { src, red, dst } => (src.clone(), red.clone(), dst.clone()),
        _ => return None,
    };

    // Topology: src flows through the chain, each op consumes the previous.
    if sub_src != src {
        return None;
    }
    if sub_red != max_dst {
        return None;
    }
    if exp_src != sub_dst {
        return None;
    }
    if sum_src != exp_dst {
        return None;
    }
    if div_src != exp_dst {
        return None;
    }
    if div_red != sum_dst {
        return None;
    }

    // Soundness: intermediates must be dead after the chain. "After" means
    // at position i+5 onward. If any intermediate is read later, the fused
    // op would drop a live value.
    let after_idx = i + 5;
    let escaping = &reads_after[after_idx.min(reads_after.len().saturating_sub(1))];
    for intermediate in [&max_dst, &sub_dst, &exp_dst, &sum_dst] {
        if escaping.contains(intermediate) {
            return None;
        }
    }
    // `src` may legitimately escape (e.g. softmax input is reused); that's fine --
    // the fused op reads it, doesn't consume the temporaries.

    Some((PtoOp::Softmax { dst: div_dst, src }, 5))
}

/// For each op-index i, compute the set of SSAs read by ops[i..]. Used to
/// decide whether an intermediate escapes a would-be fusion window.
fn compute_reads_after(ops: &[PtoOp]) -> Vec<BTreeSet<Ssa>> {
    // reads_after[i] = union of reads(ops[j]) for all j >= i
    let n = ops.len();
    let mut out: Vec<BTreeSet<Ssa>> = vec![BTreeSet::new(); n + 1];
    for i in (0..n).rev() {
        let mut set = out[i + 1].clone();
        for r in reads_of(&ops[i]) {
            set.insert(r);
        }
        out[i] = set;
    }
    out
}

fn reads_of(op: &PtoOp) -> Vec<Ssa> {
    match op {
        // We don't know what an unmodelled op reads, and this set gates fusion
        // soundness, so assume it reads every SSA its text mentions. Erring the
        // other way would let a fusion delete an intermediate that an op we
        // cannot interpret still consumes. (Ported from
        // `pto-rs-viewpoint-rebase` together with the variant.)
        PtoOp::MulI { lhs, rhs, .. } | PtoOp::AddI { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        PtoOp::GetVal { src, .. } | PtoOp::FpToSi { src, .. } => vec![src.clone()],
        PtoOp::Tci { dst, .. } => vec![dst.clone()],
        PtoOp::TSort32 { values, indices, .. } => vec![values.clone(), indices.clone()],
        PtoOp::TGather { src, .. } => vec![src.clone()],
        PtoOp::Textract { src, .. } => vec![src.clone()],
        PtoOp::Unmodelled { text, .. } => text
            .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '%'))
            .filter(|t| t.len() > 1 && t.starts_with('%'))
            .map(|t| Ssa(t.to_string()))
            .collect(),
        // The scratch operand counts as a read: an op we hand a workspace tile
        // to may write it, so a fusion must not treat it as dead.
        PtoOp::Transpose { src, scratch, .. } => vec![src.clone(), scratch.clone()],
        PtoOp::ColExpand { src, .. } | PtoOp::ColSum { src, .. } => vec![src.clone()],
        PtoOp::ScalarConst { .. } | PtoOp::AllocTile { .. } => vec![],
        PtoOp::MakeTensorView { ptr, .. } => vec![ptr.clone()],
        PtoOp::PartitionView { src, .. } => vec![src.clone()],
        PtoOp::TLoad { view, tile, .. } => vec![view.clone(), tile.clone()],
        PtoOp::TStore { tile, view, .. } => vec![tile.clone(), view.clone()],
        PtoOp::BinOp { a, b, .. } => vec![a.clone(), b.clone()],
        // Both are reads even though only `a` is a tile: the fusion pass
        // uses this to decide whether an intermediate escapes, and a
        // scalar constant that escapes is as load-bearing as a tile.
        PtoOp::BinOpScalar { a, scalar, .. } => vec![a.clone(), scalar.clone()],
        // A barrier reads no SSA value, but it ORDERS the ones around it, so it
        // must never be treated as dead code that a fusion can step over.
        PtoOp::Barrier { .. } => vec![],
        // Bounds plus every read inside the region: a fusion that moved an op
        // across the loop header while only seeing the bounds would reorder it
        // against the whole body.
        // The condition plus every read in both regions: a fusion that moved
        // an op past a conditional while seeing only the condition would
        // reorder it against both branches.
        PtoOp::If {
            cond,
            then_body,
            else_body,
        } => {
            let mut v = vec![cond.clone()];
            for op in then_body.iter().chain(else_body.iter().flatten()) {
                v.extend(reads_of(op));
            }
            v
        }
        PtoOp::For {
            lb, ub, step, body, ..
        } => {
            let mut v = vec![lb.clone(), ub.clone(), step.clone()];
            for op in body {
                v.extend(reads_of(op));
            }
            v
        }
        PtoOp::Subview { src, offsets, .. } => {
            let mut v = vec![src.clone()];
            v.extend(offsets.iter().cloned());
            v
        }
        PtoOp::GatherB { src, idx, .. } => vec![src.clone(), idx.clone()],
        PtoOp::Cvt { src, .. } => vec![src.clone()],
        PtoOp::PartitionViewNd { src, .. } => vec![src.clone()],
        PtoOp::BlockQuery { .. } => vec![],
        PtoOp::IndexCast { src, .. } => vec![src.clone()],
        PtoOp::UnOp { src, .. } => vec![src.clone()],
        PtoOp::RowReduce { src, .. } => vec![src.clone()],
        PtoOp::RowExpandSub { src, red, .. }
        | PtoOp::RowExpandDiv { src, red, .. }
        | PtoOp::RowExpandMul { src, red, .. } => {
            vec![src.clone(), red.clone()]
        }
        PtoOp::RowExpand { src, .. } => vec![src.clone()],
        PtoOp::RmsNorm { src, .. } => vec![src.clone()],
        PtoOp::Quantize { src, scale, .. } => vec![src.clone(), scale.clone()],
        PtoOp::TMatmul { lhs, rhs, .. } => vec![lhs.clone(), rhs.clone()],
        PtoOp::TMov { src, .. } => vec![src.clone()],
        PtoOp::Softmax { src, .. } => vec![src.clone()],
    }
}
