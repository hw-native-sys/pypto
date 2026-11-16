// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Where each view lands in the buffer the lift aliases.
//!
//! Its own module because three passes need the SAME answer: the
//! `PartitionView` emitter arm, which folds a constant offset into the
//! address; `emitter::barrier_blocker`, which can only tell two stores apart
//! by their addresses; and `linearize`, which may only re-load a tile when no
//! store can have touched the window it came from. Three copies of this
//! arithmetic would be three chances to disagree about whether two accesses
//! collide, and the one that disagreed would be the one that emitted a wrong
//! kernel at exit 0.

use crate::ast::*;
use std::collections::BTreeMap;

/// How a refused value should be NAMED in a message, so the reader can find it.
///
/// `unroll` renames every value of an `scf.for` body copy with a `_u<copy>`
/// suffix, so by the time a window is checked the value is called `%pto1_u0`
/// and the source file says `%pto1`. A refusal that quotes only the synthesised
/// name points at IR the reader does not have -- `ptoas_rs/tests/roundtrip.rs`
/// asserts exactly that property ("refusal names nothing in the file") and
/// caught two corpus artefacts on it, routed_gate_up_glm53_8x2048 and
/// quant_i8_v41_8x5120. The defect was the message, not the check: the value IS
/// per-copy, so naming the copy is right, and the fix is to name the source
/// value as well rather than to stop naming the copy.
///
/// A source value genuinely called `%foo_u0` is unharmed: both spellings are
/// quoted, and the file contains that one.
fn refusal_name(dst: &Ssa) -> String {
    let full = &dst.0;
    if let Some((stem, copy)) = full.rsplit_once("_u") {
        if !copy.is_empty() && copy.chars().all(|c| c.is_ascii_digit()) {
            return format!("`{stem}` (unrolled copy {copy}, `{full}`)");
        }
    }
    format!("`{full}`")
}

/// The span of the aliased buffer a `pto.partition_view` addresses.
///
/// The tile API has no view abstraction: the lift aliases the base pointer and
/// every `tile_load_*`/`tile_store_*` walks `rows * cols` CONTIGUOUS elements
/// from it. So a view is fully described, for this lift's purposes, by which
/// pointer it is rooted at and which elements of it the call will touch --
/// which is also the only thing that can say whether two stores collide.
/// A window's element offset: a constant, or -- for a `scf.for` body whose
/// partition offsets are `[%r, %c0]` -- a Rust expression in the induction
/// variable. `Expr` is rendered into the `wrapping_add` the load/store arms
/// emit; the pre-pass grows it when it starts resolving loop-variant windows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Off {
    Const(i64),
    Expr(String),
}

impl Off {
    pub fn is_zero(&self) -> bool {
        matches!(self, Off::Const(0))
    }
    pub fn render(&self) -> String {
        match self {
            Off::Const(v) => v.to_string(),
            Off::Expr(e) => e.clone(),
        }
    }
    pub fn as_const(&self) -> Option<i64> {
        match self {
            Off::Const(v) => Some(*v),
            Off::Expr(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    /// The PTO SSA of the pointer the view chain is rooted at, which is the
    /// `%ptr` of the `pto.make_tensor_view`. Two windows are only comparable
    /// when this matches: two distinct pointer ARGUMENTS may alias at the call
    /// site, and nothing in the module says they do not.
    pub base: Ssa,
    /// Elements from `base` to the first element this view addresses.
    pub off: Off,
    /// Elements this view spans.
    pub len: i64,
    /// `Some((cols, row_stride))` when this view's rows are NOT adjacent: a
    /// `sr x sc` window of a `prows x pcols` parent has rows `pcols` elements
    /// apart while spanning `sc`. `None` when the view is contiguous, which is
    /// every window as wide as its parent and every single-row window.
    ///
    /// Carried because `tile_std` has the wrapper this needs and the refusal
    /// that used to stand here predated knowing it:
    /// `tile_load_strided_f32<ROWS, COLS, ROW_STRIDE>(gm: *const f32)`
    /// (tile.rs:1592) takes exactly `(sr, sc, pcols)`, and
    /// `tile_store_strided_f32` is its mirror. The old refusal said "the tile
    /// call reads {sr*sc} contiguous ones", which is true of
    /// `tile_load_f32` and not of the strided form.
    pub strided: Option<(i64, i64)>,
}

/// Resolve every rank-2 view in `ops` to a [`Window`], or to the reason it
/// cannot be one.
///
/// `Err` is carried rather than dropped so the `PartitionView` arm can report
/// the specific reason instead of a generic "not resolvable": a loop-variant
/// offset, a window that leaves the parent, and a multi-row window narrower
/// than the parent are three different problems with three different answers.
///
/// A view missing from the map entirely is one whose PARENT this lift never
/// modelled -- `make_tensor_view` refuses a non-contiguous or rank-N parent
/// before any partition of it is reached, so in practice the absent case is a
/// partition of something that is not a tensor view at all.
///
/// Deliberately does NOT recurse into `scf.for` / `scf.if` bodies. A view
/// inside a loop would be recorded once while the loop writes through it on
/// every iteration, so a store at a constant offset in a loop body would look
/// like a single store that conflicts with nothing. Both region arms refuse
/// before emitting a body today, so the omission is unreachable; leaving it out
/// means the first change that starts emitting regions gets a refusal (every
/// lookup misses, and [`provably_disjoint`] answers `false`) rather than a
/// quietly wrong answer.
/// Symbolic scalar expressions over `arith.muli` / `arith.addi`, keyed by SSA.
/// The partition offsets of an expanded dequant loop are a product of the
/// induction variable and a constant; `Off::Expr` needs that whole expression,
/// not a bare loop variable.
fn build_sym(ops: &[PtoOp], consts: &BTreeMap<Ssa, i64>) -> BTreeMap<Ssa, String> {
    fn atom(x: &Ssa, consts: &BTreeMap<Ssa, i64>, sym: &BTreeMap<Ssa, String>) -> String {
        if let Some(v) = crate::emitter::resolve(consts, &x.0) {
            return v.to_string();
        }
        match sym.get(x) {
            Some(e) => e.clone(),
            // Not a constant and not yet an expression: it may be a loop
            // variable (bound by the `For` arm) or a runtime scalar (bound by
            // `tgetval`/`fptosi`). Both are resolved by `render_off` at emit
            // time; writing a guessed `lv_` name here is what referenced an
            // unbound identifier.
            None => format!("@{}", x.0),
        }
    }
    fn walk(ops: &[PtoOp], consts: &BTreeMap<Ssa, i64>, sym: &mut BTreeMap<Ssa, String>) {
        for op in ops {
            match op {
                PtoOp::MulI { dst, lhs, rhs } => {
                    let e = format!(
                        "(({}) * ({}))",
                        atom(lhs, consts, sym),
                        atom(rhs, consts, sym)
                    );
                    sym.insert(dst.clone(), e);
                }
                PtoOp::IndexCast { dst, src, .. } => {
                    // The loop variable is usually carried through an
                    // `arith.index_cast` before it offsets a partition; the
                    // expression must name the VARIABLE, not the cast result,
                    // or the generated Rust references a name the `For` arm
                    // never binds.
                    sym.insert(dst.clone(), atom(src, consts, sym));
                }
                PtoOp::AddI { dst, lhs, rhs } => {
                    let e = format!(
                        "(({}) + ({}))",
                        atom(lhs, consts, sym),
                        atom(rhs, consts, sym)
                    );
                    sym.insert(dst.clone(), e);
                }
                PtoOp::For { body, .. } => walk(body, consts, sym),
                _ => {}
            }
        }
    }
    let mut sym = BTreeMap::new();
    walk(ops, consts, &mut sym);
    sym
}

pub fn view_windows(
    ops: &[PtoOp],
    consts: &BTreeMap<Ssa, i64>,
) -> BTreeMap<Ssa, Result<Window, String>> {
    let mut parents = BTreeMap::new();
    let sym = build_sym(ops, consts);
    view_windows_scoped(ops, consts, &BTreeMap::new(), &mut parents, &sym)
}

/// The induction variable's Rust name, shared verbatim by this pre-pass (which
/// writes it into `Off::Expr`) and the `For` emitter (which binds it), so the
/// two cannot disagree about the identifier.
pub fn loop_var_name(v: &Ssa) -> String {
    let raw = v.0.trim_start_matches('%');
    let mut n = String::from("lv_");
    for ch in raw.chars() {
        n.push(if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' });
    }
    n
}

fn off_expr(
    op: &str,
    consts: &BTreeMap<Ssa, i64>,
    sym: &BTreeMap<Ssa, String>,
    loop_vars: &BTreeMap<Ssa, String>,
    mul: i64,
) -> Option<String> {
    if let Some(v) = crate::emitter::resolve(consts, op) {
        return Some((v.saturating_mul(mul)).to_string());
    }
    if let Some(e) = sym.get(&Ssa(op.to_string())) {
        return Some(format!("({e}) * {mul}"));
    }
    if let Some(n) = loop_vars.get(&Ssa(op.to_string())) {
        return Some(format!("({n} * {mul})"));
    }
    // A runtime scalar (a `tgetval`/`fptosi` chain) has no name yet: the
    // pre-pass runs before the emitter binds `names`. `@ssa` is substituted at
    // emit time, where the binding exists.
    Some(format!("(@{op}) * {mul}"))
}

fn view_windows_scoped(
    ops: &[PtoOp],
    consts: &BTreeMap<Ssa, i64>,
    loop_vars: &BTreeMap<Ssa, String>,
    parents: &mut BTreeMap<Ssa, (Ssa, i64, i64)>,
    sym: &BTreeMap<Ssa, String>,
) -> BTreeMap<Ssa, Result<Window, String>> {
    // `parents` is FUNCTION-scoped and shared with every region recursion: a
    // `make_tensor_view` at the top level is the parent of a `partition_view`
    // inside the `scf.for` body. A per-call map made the body's partitions look
    // like they addressed an unmodelled view.
    let mut out: BTreeMap<Ssa, Result<Window, String>> = BTreeMap::new();
    for op in ops {
        match op {
            PtoOp::MakeTensorView {
                dst,
                ptr,
                shape,
                strides,
                layout,
                ..
            } => {
                // Mirror of the `MakeTensorView` arm's own guards: a rank-N or
                // fractal view has no rows/cols, and a non-contiguous one makes
                // the flat offset below meaningless. Both are refused there, so
                // recording nothing here is enough.
                if layout.is_some() || shape.len() != 2 || strides.len() != 2 {
                    continue;
                }
                let (Some(r), Some(c), Some(rs), Some(cs)) = (
                    crate::emitter::resolve(consts, &shape[0]),
                    crate::emitter::resolve(consts, &shape[1]),
                    crate::emitter::resolve(consts, &strides[0]),
                    crate::emitter::resolve(consts, &strides[1]),
                ) else {
                    continue;
                };
                if rs != c || cs != 1 {
                    continue;
                }
                parents.insert(dst.clone(), (ptr.clone(), r, c));
                out.insert(
                    dst.clone(),
                    Ok(Window {
                        base: ptr.clone(),
                        off: Off::Const(0),
                        len: r.saturating_mul(c),
                        // A tensor view reaching here is contiguous row-major:
                        // the arm above refuses any other stride pair.
                        strided: None,
                    }),
                );
            }
            PtoOp::PartitionView {
                dst,
                src,
                offsets,
                sizes,
                ..
            } => {
                let Some((base, prows, pcols)) = parents.get(src).cloned() else {
                    continue;
                };
                let resolved = [
                    crate::emitter::resolve(consts, &offsets[0]),
                    crate::emitter::resolve(consts, &offsets[1]),
                    crate::emitter::resolve(consts, &sizes[0]),
                    crate::emitter::resolve(consts, &sizes[1]),
                ];
                let (Some(sr), Some(sc)) = (resolved[2], resolved[3]) else {
                    out.insert(
                        dst.clone(),
                        Err(format!(
                            "partition view {} has a size this lift cannot resolve to a \
                             constant (sizes [{}, {}])",
                            refusal_name(dst), sizes[0], sizes[1]
                        )),
                    );
                    continue;
                };
                let (r, c) = (resolved[0], resolved[1]);
                let off = match (r, c) {
                    (Some(r), Some(c)) => Off::Const(r.saturating_mul(pcols).saturating_add(c)),
                    _ => {
                        // A loop-variant offset is the `scf.for` form:
                        // `offsets = [%r, %c0]`, `%r` the induction variable.
                        // The window is a different span each iteration, so it
                        // is carried as an expression in that variable rather
                        // than collapsed to one address.
                        match (
                            off_expr(&offsets[0], consts, sym, loop_vars, pcols),
                            off_expr(&offsets[1], consts, sym, loop_vars, 1),
                        ) {
                            (Some(a), Some(b)) => Off::Expr(format!("({a}) + ({b})")),
                            _ => {
                                out.insert(
                                    dst.clone(),
                                    Err(format!(
                                        "partition view {} has an operand this lift cannot \
                                         resolve to a constant or a loop variable (offsets [{}, {}])",
                                        refusal_name(dst), offsets[0], offsets[1]
                                    )),
                                );
                                continue;
                            }
                        }
                    }
                };
                // A window narrower than its parent and taller than one row is
                // STRIDED in the parent, and the tile call reads contiguously.
                // That is wrong at the origin too -- this is not a consequence
                // of admitting an offset -- but admitting offsets is what makes
                // it reachable, so it is checked here rather than left for the
                // next reader. 26 corpus instances, all in the GMM cube files.
                // A window narrower than its parent and taller than one row is
                // STRIDED in the parent. This used to REFUSE, on the grounds
                // that "the tile call reads contiguously" -- true of
                // `tile_load_f32` and FALSE of `tile_load_strided_f32`, which
                // `tile_std` exports at tile.rs:1592 taking exactly
                // `<ROWS, COLS, ROW_STRIDE>` with a mirror store at :1603. The
                // refusal predated knowing the wrapper existed; it was one of 54
                // omitted from `WRAPPERS` (see that table's note).
                //
                // So the geometry is CARRIED instead of refused, and the load
                // and store arms pick the strided wrapper. What still refuses is
                // a window whose column stride is not 1 -- the wrapper strides
                // ROWS, not elements within a row -- and that is checked where
                // the strides are read, in the `MakeTensorView` arm.
                if let (Some(r), Some(c)) = (r, c) {
                if r < 0 || c < 0 || r + sr > prows || c + sc > pcols {
                    out.insert(
                        dst.clone(),
                        Err(format!(
                            "partition view {} is a {sr}x{sc} window at [{r}, {c}] of a \
                             {prows}x{pcols} parent, which leaves the parent; the lift would \
                             emit an out-of-bounds address",
                            refusal_name(dst)
                        )),
                    );
                    continue;
                }
                }
                out.insert(
                    dst.clone(),
                    Ok(Window {
                        base,
                        off,
                        // SPAN, not footprint. A strided window's rows are
                        // `pcols` apart, so it REACHES `(sr-1)*pcols + sc`
                        // elements while addressing `sr*sc`. `len` feeds the
                        // overlap and capacity checks, which ask what the view
                        // can touch, so it has to be the reach.
                        len: if sr > 1 && sc != pcols {
                            (sr - 1).saturating_mul(pcols).saturating_add(sc)
                        } else {
                            sr.saturating_mul(sc)
                        },
                        strided: if sr > 1 && sc != pcols {
                            Some((sc, pcols))
                        } else {
                            None
                        },
                    }),
                );
            }
            PtoOp::For { var, body, .. } => {
                let mut inner = loop_vars.clone();
                inner.insert(var.clone(), loop_var_name(var));
                out.extend(view_windows_scoped(body, consts, &inner, parents, sym));
            }
            _ => {}
        }
    }
    out
}

/// Whether two windows are known not to touch the same element.
///
/// `false` whenever anything is unknown -- a missing window, an unresolved one,
/// or two different base pointers, which may alias at the call site.
pub fn provably_disjoint(
    windows: &BTreeMap<Ssa, Result<Window, String>>,
    a: &Ssa,
    b: &Ssa,
) -> bool {
    let (Some(Ok(x)), Some(Ok(y))) = (windows.get(a), windows.get(b)) else {
        return false;
    };
    let (Some(xo), Some(yo)) = (x.off.as_const(), y.off.as_const()) else {
        // A symbolic offset can move between iterations; two such windows are
        // not provably disjoint from constants alone.
        return false;
    };
    x.base == y.base && (xo + x.len <= yo || yo + y.len <= xo)
}
