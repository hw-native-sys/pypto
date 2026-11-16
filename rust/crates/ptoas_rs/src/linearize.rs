// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Make each tile value read exactly once, by re-loading the ones that are not.
//!
//! `tile_std::tile::Tile<R, C, T>` is deliberately not `Copy` -- "each tile
//! must be consumed by exactly one store operation, preventing double-DMA
//! bugs" -- so a tile-API call MOVES its operands. PTO tiles are mutable named
//! buffers with no such rule, and reading one twice is ordinary: the rope
//! kernels read all four of their input tiles twice, because
//! `o_lo = x_lo*cos - x_hi*sin` and `o_hi = x_hi*cos + x_lo*sin` between them
//! name each input in two products.
//!
//! The emitter used to write those out as two uses of one Rust binding, which
//! is `E0382: use of moved value`. Nothing caught it, because nothing had ever
//! COMPILED the lift's output -- the emitted header has claimed "every tile
//! handle is used exactly once (move semantics)" since the file was written,
//! and that was a claim about the input, never checked against it.
//!
//! Where the tile came from a `pto.tload`, the fix is to load it again rather
//! than refuse: a second load of the same window returns the same elements, so
//! the two forms compute the same kernel. That equivalence has a precondition
//! -- nothing may have written that window in between -- and it is checked
//! against addresses, not assumed. Where the tile came from anything else
//! (a compute op, a cast, a matmul), there is nothing to repeat and
//! `emitter::emit_func` refuses, naming the tile.

use crate::ast::*;
use crate::windows::view_windows;
use std::collections::BTreeMap;

/// Rewrite every function so no tile is read twice, where that is possible.
pub fn linearize(m: &mut PtoModule) {
    for f in &mut m.funcs {
        // ALTERNATE the reload pass and the rematerialisation pass: reproducing a value
        // re-reads its inputs, and those may be loads the reload pass reproduces more
        // cheaply. The previous attempt at this pass was written, compiled, exported and
        // NEVER CALLED -- `linearize` kept its one-line body -- which measures exactly like
        // a pass that does not work. Hence the debug line, kept.
        for _ in 0..32 {
            linearize_func(f);
            let changed = rematerialise(f);
            if std::env::var_os("PTOLIFT_REMAT_DEBUG").is_some() {
                let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
                for op in &f.ops {
                    for t in op.tile_reads() {
                        *counts.entry(t.0.as_str()).or_default() += 1;
                    }
                }
                let shared: Vec<(&str, usize)> =
                    counts.into_iter().filter(|(_, n)| *n > 1).collect();
                eprintln!("[remat] changed={changed} shared={shared:?}");
            }
            if !changed {
                break;
            }
        }
    }
}


/// Unique across the process: a second pass must not mint a name a first pass used,
/// because two `let t3__rc2` in one function shadow each other and hide a dead value.
static COPY_TAG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Re-point an op's written tile. The mirror of `tile_write`, beside it so the two cannot
/// drift: every arm that returns `Some` there must be assignable here. Returns whether it
/// could, because "did nothing" and "declined" are different answers.
fn set_tile_write(op: &mut PtoOp, new: &Ssa) -> bool {
    match op {
        PtoOp::BinOp { dst, .. }
        | PtoOp::BinOpScalar { dst, .. }
        | PtoOp::UnOp { dst, .. }
        | PtoOp::RowReduce { dst, .. }
        | PtoOp::RowExpandSub { dst, .. }
        | PtoOp::RowExpandDiv { dst, .. }
        | PtoOp::RowExpandMul { dst, .. }
        | PtoOp::RowExpand { dst, .. }
        | PtoOp::ColExpand { dst, .. }
        | PtoOp::ColSum { dst, .. }
        | PtoOp::RmsNorm { dst, .. }
        | PtoOp::TMatmul { dst, .. }
        | PtoOp::TMov { dst, .. }
        | PtoOp::Softmax { dst, .. }
        | PtoOp::Quantize { dst, .. }
        | PtoOp::Cvt { dst, .. }
        | PtoOp::GatherB { dst, .. }
        | PtoOp::Subview { dst, .. }
        | PtoOp::Transpose { dst, .. } => {
            *dst = new.clone();
            true
        }
        _ => false,
    }
}

/// Where a tile's value comes from, and whether it can be reproduced.
struct Origin {
    producer: Option<usize>,
    load: Option<(Ssa, PtoType)>,
    alloc: Option<PtoType>,
}

/// Reproduce `want` into a fresh tile, appending the ops that do it, in dependency order.
///
/// WHY IT RECURSES. Duplicating a producer is not enough: the duplicate re-reads its own
/// inputs, so a value read twice becomes its input read three times -- the same obstacle
/// one level down, measured. The chain is followed until it reaches something reproducible
/// for free, which is a LOAD of a known view.
fn reproduce(
    want: &Ssa,
    origins: &std::collections::BTreeMap<Ssa, Origin>,
    ops: &[PtoOp],
    shapes: &std::collections::BTreeMap<Ssa, TileShape>,
    pre: &mut Vec<PtoOp>,
    fresh_shapes: &mut Vec<(Ssa, TileShape)>,
    budget: &mut usize,
) -> Option<Ssa> {
    if *budget == 0 {
        return None;
    }
    let origin = origins.get(want)?;
    let fresh = Ssa(format!(
        "{}__rc{}",
        want.0,
        COPY_TAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    *budget -= 1;
    if let Some(sh) = shapes.get(want) {
        fresh_shapes.push((fresh.clone(), *sh));
    }
    // EVERY COPY NEEDS AN ALLOC. The Rust emitter mints its names from the alloc rows, so a
    // copy without one refuses as `unresolved SSA` -- which is the error this path produced
    // until it was seen to be about names rather than about values.
    let Some(ty) = origin.alloc.clone() else {
        return None;
    };
    pre.push(PtoOp::AllocTile { dst: fresh.clone(), ty });
    if let Some((view, view_ty)) = &origin.load {
        pre.push(PtoOp::TLoad {
            view: view.clone(),
            tile: fresh.clone(),
            view_ty: view_ty.clone(),
        });
        return Some(fresh);
    }
    let j = origin.producer?;
    let mut clone = ops.get(j)?.clone();
    let inputs: Vec<Ssa> = clone.tile_reads().into_iter().cloned().collect();
    for inp in inputs {
        let copy = reproduce(&inp, origins, ops, shapes, pre, fresh_shapes, budget)?;
        for r in clone.tile_reads_mut().iter_mut() {
            if **r == inp {
                **r = copy.clone();
                break;
            }
        }
    }
    if !set_tile_write(&mut clone, &fresh) {
        return None;
    }
    pre.push(clone);
    Some(fresh)
}

/// Give every multiply-read value its own copy, recursively. Returns whether anything moved.
fn rematerialise(f: &mut PtoFunc) -> bool {
    use std::collections::BTreeMap;
    if f.ops
        .iter()
        .any(|op| matches!(op, PtoOp::For { .. } | PtoOp::If { .. }))
    {
        return false;
    }
    let mut origins: BTreeMap<Ssa, Origin> = BTreeMap::new();
    let mut writers: BTreeMap<Ssa, usize> = BTreeMap::new();
    for (i, op) in f.ops.iter().enumerate() {
        if let Some(w) = tile_write(op) {
            *writers.entry(w.clone()).or_default() += 1;
            // `get_mut`, not `or_insert`: an alloc row for the same tile is seen FIRST and
            // creates the entry with `producer: None`, so `or_insert` kept that and every
            // tile looked producer-less.
            let e = origins.entry(w.clone()).or_insert(Origin {
                producer: None,
                load: None,
                alloc: None,
            });
            e.producer = Some(i);
        }
        match op {
            PtoOp::TLoad { tile, view, view_ty } => {
                // A LOAD is a writer. `tile_write` excludes it by design, and without this
                // a buffer written by two loads looked single-writer and was reproduced --
                // the one thing the reload pass refuses to do.
                *writers.entry(tile.clone()).or_default() += 1;
                let e = origins.entry(tile.clone()).or_insert(Origin {
                    producer: None,
                    load: None,
                    alloc: None,
                });
                e.load = Some((view.clone(), view_ty.clone()));
            }
            PtoOp::AllocTile { dst, ty } => {
                let e = origins.entry(dst.clone()).or_insert(Origin {
                    producer: None,
                    load: None,
                    alloc: None,
                });
                e.alloc = Some(ty.clone());
            }
            _ => {}
        }
    }
    let base_ops = f.ops.clone();
    let base_shapes = f.shapes.clone();
    let mut used: BTreeMap<Ssa, usize> = BTreeMap::new();
    let mut out: Vec<PtoOp> = Vec::with_capacity(base_ops.len());
    let mut fresh_shapes: Vec<(Ssa, TileShape)> = Vec::new();
    let mut budget: usize = 256;
    let mut changed = false;
    for i in 0..base_ops.len() {
        let mut op = base_ops[i].clone();
        let reads: Vec<Ssa> = op.tile_reads().into_iter().cloned().collect();
        let mut pre: Vec<PtoOp> = Vec::new();
        for r in reads {
            let seen = used.entry(r.clone()).or_default();
            *seen += 1;
            if *seen == 1 {
                continue;
            }
            if writers.get(&r).copied().unwrap_or(0) > 1 {
                continue; // a reused buffer: refused downstream, never reproduced
            }
            let mut local: Vec<PtoOp> = Vec::new();
            if let Some(copy) = reproduce(
                &r,
                &origins,
                &base_ops,
                &base_shapes,
                &mut local,
                &mut fresh_shapes,
                &mut budget,
            ) {
                pre.extend(local);
                for r2 in op.tile_reads_mut().iter_mut() {
                    if **r2 == r {
                        **r2 = copy.clone();
                        break;
                    }
                }
                changed = true;
            }
        }
        out.extend(pre);
        out.push(op);
    }
    if changed {
        for (ssa, sh) in fresh_shapes {
            f.shapes.insert(ssa, sh);
        }
    }
    f.ops = out;
    changed
}


fn linearize_func(f: &mut PtoFunc) {
    // A region's body can read and write anything, and `view_windows` does not
    // look inside one, so every count below would be taken over part of the
    // function. Both region arms refuse before emitting a body, so a function
    // with one is never lifted anyway; bailing out keeps this pass from being
    // the thing that decides otherwise if that changes.
    if f.ops
        .iter()
        .any(|op| matches!(op, PtoOp::For { .. } | PtoOp::If { .. }))
    {
        return;
    }
    let windows = view_windows(&f.ops, &f.consts);

    // How many times each tile is read, and how many ops write it. A tile
    // written more than once is a REUSED buffer -- the matvec kernels re-tload
    // one tile per row -- and re-loading from it would pick a window that the
    // other writer may since have replaced, so those are left alone for
    // `emit_func` to refuse.
    let mut reads: BTreeMap<&Ssa, usize> = BTreeMap::new();
    let mut writes: BTreeMap<&Ssa, usize> = BTreeMap::new();
    // tile -> (the view it was loaded from, that view's own PTO type).
    let mut loaded: BTreeMap<&Ssa, (&Ssa, &PtoType)> = BTreeMap::new();
    let mut alloc_ty: BTreeMap<&Ssa, &PtoType> = BTreeMap::new();
    // The first `pto.tstore`. A re-load placed after one could read elements
    // that store replaced -- and whether it did cannot be settled here, since
    // two distinct pointer ARGUMENTS may alias at the call site and nothing in
    // the module says they do not. So the cut is positional: a tile is only
    // re-loadable when every read of it comes before the first store, which
    // needs no aliasing claim at all. Every corpus case is of that shape (the
    // rope kernels load, compute, then store twice at the end).
    let first_store = f
        .ops
        .iter()
        .position(|op| matches!(op, PtoOp::TStore { .. }))
        .unwrap_or(f.ops.len());
    let mut after_store: BTreeMap<&Ssa, bool> = BTreeMap::new();
    for (i, op) in f.ops.iter().enumerate() {
        for t in op.tile_reads() {
            *reads.entry(t).or_default() += 1;
            if i > first_store {
                after_store.insert(t, true);
            }
        }
        match op {
            PtoOp::TLoad {
                view,
                tile,
                view_ty,
            } => {
                *writes.entry(tile).or_default() += 1;
                // The view's own type comes from the load that is being
                // repeated, not rebuilt from the tile shape, so a view type
                // this crate carries but does not model comes back unchanged.
                loaded.insert(tile, (view, view_ty));
            }
            PtoOp::AllocTile { dst, ty } => {
                alloc_ty.insert(dst, ty);
            }
            other => {
                if let Some(w) = tile_write(other) {
                    *writes.entry(w).or_default() += 1;
                }
            }
        }
    }

    // The tiles a second load would reproduce exactly, with the view to load
    // from and the type to allocate.
    let mut reloadable: BTreeMap<Ssa, (Ssa, PtoType, PtoType)> = BTreeMap::new();
    for (tile, n) in &reads {
        if *n < 2 || writes.get(*tile).copied().unwrap_or(0) != 1 {
            continue;
        }
        if after_store.contains_key(*tile) {
            continue;
        }
        let Some((view, view_ty)) = loaded.get(*tile).copied() else {
            continue;
        };
        // The view has to resolve to a known span, or "load it again" names no
        // definite address. `view_windows` answers `Err` for a loop-variant
        // offset and nothing at all for a parent this lift did not model.
        if !matches!(windows.get(view), Some(Ok(_))) {
            continue;
        }
        let Some(ty) = alloc_ty.get(*tile) else {
            continue;
        };
        reloadable.insert(
            (*tile).clone(),
            (view.clone(), (*ty).clone(), view_ty.clone()),
        );
    }
    if reloadable.is_empty() {
        return;
    }

    let shapes: BTreeMap<Ssa, TileShape> = f.shapes.clone();

    let mut out: Vec<PtoOp> = Vec::with_capacity(f.ops.len());
    // Reads already accounted for, per tile. The FIRST read keeps the original
    // binding; every later one gets its own load.
    let mut used: BTreeMap<Ssa, usize> = BTreeMap::new();
    let mut extra: Vec<(Ssa, TileShape)> = Vec::new();
    for mut op in f.ops.drain(..) {
        let mut pre: Vec<PtoOp> = Vec::new();
        for r in op.tile_reads_mut() {
            let Some((view, ty, view_ty)) = reloadable.get(r) else {
                continue;
            };
            let n = used.entry(r.clone()).or_default();
            *n += 1;
            if *n == 1 {
                continue;
            }
            // `%t3` read a third time becomes `%t3_reload3`. Derived from the
            // original name on purpose: a refusal or a diff downstream names
            // the tile the reader can find in the PTO.
            let fresh = Ssa(format!("{}_reload{}", r.0, *n));
            if let Some(sh) = shapes.get(r) {
                extra.push((fresh.clone(), *sh));
            }
            pre.push(PtoOp::AllocTile {
                dst: fresh.clone(),
                ty: ty.clone(),
            });
            pre.push(PtoOp::TLoad {
                view: view.clone(),
                tile: fresh.clone(),
                view_ty: view_ty.clone(),
            });
            *r = fresh;
        }
        out.extend(pre);
        out.push(op);
    }
    for (ssa, sh) in extra {
        f.shapes.insert(ssa, sh);
    }
    f.ops = out;
}

/// The tile an op writes, for the "written exactly once" test above.
///
/// Separate from [`PtoOp::tile_reads`] and narrower than
/// `emitter_tile::op_output`: a `PtoOp::Transpose` writes its `scratch` as
/// well as its `dst`, and missing that would let a scratch tile look
/// single-writer and be re-loaded from a window the transpose has since
/// overwritten.
fn tile_write(op: &PtoOp) -> Option<&Ssa> {
    match op {
        PtoOp::BinOp { dst, .. }
        | PtoOp::BinOpScalar { dst, .. }
        | PtoOp::UnOp { dst, .. }
        | PtoOp::RowReduce { dst, .. }
        | PtoOp::RowExpandSub { dst, .. }
        | PtoOp::RowExpandDiv { dst, .. }
        | PtoOp::RowExpandMul { dst, .. }
        | PtoOp::RowExpand { dst, .. }
        | PtoOp::ColExpand { dst, .. }
        | PtoOp::ColSum { dst, .. }
        | PtoOp::RmsNorm { dst, .. }
        | PtoOp::TMatmul { dst, .. }
        | PtoOp::TMov { dst, .. }
        | PtoOp::Softmax { dst, .. }
        | PtoOp::Quantize { dst, .. }
        | PtoOp::Cvt { dst, .. }
        | PtoOp::GatherB { dst, .. }
        | PtoOp::Subview { dst, .. }
        | PtoOp::Transpose { dst, .. } => Some(dst),
        // A region's body can write anything, so `None` here would be the
        // wrong answer -- but `linearize_func` returns before building this
        // map at all when the function holds a region, which is the only
        // reason the wrong answer is unreachable.
        PtoOp::For { .. } | PtoOp::If { .. } => None,
        PtoOp::TLoad { .. }
        | PtoOp::TStore { .. }
        | PtoOp::ScalarConst { .. }
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
        | PtoOp::Barrier { .. }
        | PtoOp::BlockQuery { .. }
        | PtoOp::IndexCast { .. }
        | PtoOp::PartitionViewNd { .. }
        | PtoOp::MakeTensorView { .. }
        | PtoOp::PartitionView { .. }
        | PtoOp::AllocTile { .. }
        | PtoOp::Unmodelled { .. } => None,
    }
}
