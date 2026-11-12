// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! PTO-MLIR text -> Rust source (targeting `tile_std::tile`).
//!
//! S1 scope (this file):
//!   - Line-based parser for the subset of PTO emitted by `mlir_to_pto.rs`
//!     and by `ptoas`.
//!   - Literal emitter: one PTO op -> one tile-API call. No pattern fusion.
//!   - Handles: make_tensor_view, partition_view, alloc_tile, tload, tstore,
//!     pointwise binary/unary, row-reductions, row-expansions, tmatmul, tmov.
//!   - Dtype: f32 / f16 / bf16 (emitted via the appropriate `tile_*_fNN` fn).
//!
//! Out of scope for S1 (documented in crate README):
//!   - Pattern fusion (softmax collapse) -- S2.
//!   - Round-trip verifier (Rust -> rustc_codegen_tile -> PTO -> diff) -- S3.
//!   - melior / mlir-sys integration (we stay text-based to match mlir_to_pto.rs).

// Public so a test can name `ast::Ssa` unambiguously. The crate has TWO
// `Ssa` types -- `ast::Ssa` for the S1 text AST and `plan::Ssa` for the
// Stage-2 plan -- and the explicit `pub use plan::{...}` wins over
// `pub use ast::*`, so a bare `ptoas_rs::Ssa` is the plan's, not this one's.
pub mod ast;
mod emitter;
mod emitter_pto;
pub mod emitter_tile;
pub mod emitter_mlir;
pub mod verify_rustc;
mod fuse;
pub mod fragment;
mod linearize;
pub mod parser;
pub mod unroll;
mod verify;
mod windows;

// Stage-2 plan-diff path (advisory safety oracle on ptoas `--print-after-all`
// output, post-PlanMemoryPass). Independent of the S1/S2 text path above.
pub mod ascend_tile_ingress;
pub mod device;
pub mod plan;
pub mod plan_diff;
pub mod ptoas_stderr;
pub mod safety;
pub mod stage1_parser;
pub mod stage1_plan;
pub mod stage2_parser;

pub use ascend_tile_ingress::{check_ingress, project, ProjectResult};
/// Public only so `tests/emitted_names_resolve.rs` can check every row
/// against `tile_std`'s source. The table is the Rust lift's record of which
/// safe wrappers exist, and it has to be checkable from outside the emitter
/// or it is just another hand-kept list.
// The table, plus its MEASURED omission gap, for
// `tests/unmodelled_ops.rs::the_wrapper_table_is_audited_in_both_directions`.
// The constant travels with the table because the audit has to compare them.
pub use emitter::WRAPPERS;
pub use ast::*;
pub use device::{default_a5_910b2_cann85, DeviceSpec};
pub use plan::{
    AddressSpace, BLayout, ExtDType, Plan, PlanFunc, PlanOp, PtoArch, SLayout, Ssa, TileSlotX,
};
pub use plan_diff::{diff_plans, DiffEntry, DiffKind, PlanDiff};
pub use ptoas_stderr::extract_last_module;
pub use safety::{
    check_all, SafetyKind, SafetyReport, SafetyViolation, Severity,
};
pub use stage1_parser::parse_stage1;
pub use stage1_plan::{PartitionView, Stage1Func, Stage1Plan, TensorView};
pub use stage2_parser::parse_stage2;
pub use fragment::{classify, ops_of, verdict as verdict_for, Assume, Excluder, OpClass, Verdict};
pub use parser::{op_name, MODELLED_OPS};
pub use verify::{op_model_coverage, verify_roundtrip, RoundTripMismatch};

/// Convert a PTO-MLIR text blob to Rust source.
///
/// S2: runs the fusion pass between parse and emit so recognised patterns
/// (softmax today) collapse to single tile-API calls. Callers who want to
/// see the un-fused output for debugging can use [`convert_pto_to_rust_raw`].
pub fn convert_pto_to_rust(pto_src: &str) -> Result<String, String> {
    // Expand a constant-bound chunk loop BEFORE parsing: the lift's offsets are
    // precomputed from the AST and a constants map, so a loop-carried offset cannot be
    // reached from inside the AST -- only a straight-line body with constant offsets
    // lifts, which is what this pass produces. Loops that do not qualify are left
    // untouched, so this can only remove an obstacle.
    let unrolled = unroll::unroll_constant_loops(pto_src);
    let mut module = parser::parse_module(&unrolled)?;
    fuse::fuse_patterns(&mut module);
    linearize::linearize(&mut module);
    // Emit FIRST, then check extents. Deliberate ordering, and the first
    // attempt had it the other way round: run ahead of emission, this check
    // preempted four refusals that name a more specific cause -- an in-place
    // write, a load after a store, a decomposed chain with no lift -- because
    // those fixtures also carry a padded tile. The tests asserting on those
    // messages then failed saying "the refusal does not name the in-place
    // tile", which is true and is the wrong complaint to surface.
    //
    // A file with no lift at all should be reported as having no lift. The
    // extent disagreement only matters for a file that would otherwise emit,
    // because that is exactly the case where the output compiles or does not.
    let out = emitter::emit_module(&module)?;
    emitter_tile::validate_active_tile_extents(&module, pto_src)?;
    Ok(out)
}

/// Parse PTO and run the fusion pass, returning the AST.
///
/// Exposed so a test can assert that a fusion rule FIRED on a real artifact.
/// Without it the only observable is the emitted Rust, and both fusions sit
/// behind other gates in every artifact that carries them -- the rms-norm
/// files are all refused on `pto.barrier` before the lift reaches the fused
/// node -- so a rule could stop matching and every test stay green.
///
/// That is not hypothetical. The sibling branch's rms-norm rule matched a
/// chain this corpus does not contain, and its three tests passed because all
/// three ran on a fixture the rule's author wrote. A pattern rule has to be
/// tested against the emitter's own output.
pub fn fused_module(pto_src: &str) -> Result<ast::PtoModule, String> {
    let mut module = parser::parse_module(pto_src)?;
    fuse::fuse_patterns(&mut module);
    linearize::linearize(&mut module);
    Ok(module)
}

/// Same as [`convert_pto_to_rust`] but without the pattern fusions that are
/// OPTIMISATIONS -- softmax and rms-norm are left decomposed. The quantise
/// recovery still runs; see below for why it is not one of those.
///
/// Useful when debugging the emitter or inspecting what `ptoas` produced
/// before any rewriting.
pub fn convert_pto_to_rust_raw(pto_src: &str) -> Result<String, String> {
    let mut module = parser::parse_module(pto_src)?;
    // Raw means no pattern FUSION, not "skip the legality checks".
    // `linearize` still runs. "Raw" means no pattern FUSION -- each PTO op
    // maps to one tile-API call -- not "emit Rust that cannot compile".
    // Re-loading a tile that two ops read is a legality normalisation against
    // the target API (`Tile` is not `Copy`), not a rewrite of the kernel, and
    // leaving it out made raw mode refuse at the first innocuous reader of a
    // doubly-read tile instead of at the op that actually has no lift.
    //
    // The quantise chain is the one fusion raw DOES run, and the distinction is
    // not a convenience: its four ops are the DECOMPOSITION of one intrinsic
    // (`__tile_quantize_f32_i8` with a tile scale), the two saturating casts
    // have no per-op tile-API form at all, and a decomposed chain has no lift.
    // Leaving it unfused would make raw mode refuse a kernel that the artifact
    // expresses in one call -- the same "no per-op lowering" asymmetry as
    // softmax, except that here the fused call is the ONLY sound lowering. The
    // softmax and rms-norm rules stay off, so `--raw` remains observably
    // different from the default on the artifacts that carry them.
    fuse::fuse_quantize_module(&mut module);
    linearize::linearize(&mut module);
    let out = emitter::emit_module(&module)?;
    emitter_tile::validate_active_tile_extents(&module, pto_src)?;
    Ok(out)
}

/// Parse PTO and write it back out in canonical form -- the two legs the
/// round-trip verifier runs, exposed so a test can assert on the TEXT.
///
/// An AST comparison is only sensitive to what the AST carries, which is how
/// two separate losses survived the sweep: the scalar operand's width
/// (re-emitted as a literal `f32`) and a width-typed integer constant
/// (re-emitted as `: index`). Both are now fields, so the AST sees them -- but
/// a text assertion would have caught them while they were not, which is the
/// reason to have this.
pub fn reemit_pto(pto_src: &str) -> Result<String, String> {
    let module = parser::parse_module(pto_src)?;
    emitter_pto::emit_module_pto(&module)
}

/// Conservatively lift the supported PTO subset into the INTRINSIC-level
/// tile-rs kernel form, with an explicit unsafe pointer boundary.
///
/// Unsupported operations and memory layouts return a diagnostic rather than
/// generating a partial kernel. The `_safe` suffix is retained for
/// compatibility; callers must uphold the generated kernel's pointer validity
/// requirements.
///
/// **"Statically checked shapes" is what this doc used to claim, and it is not
/// what this path emits.** Measured 2026-11-01: across the corpus it produces
/// **zero** `Tile<R, C, T>` values. It writes `extern "C"` intrinsics over
/// `u32` buffer handles with the extents as RUNTIME arguments --
/// `__tile_load_f32(arg0, 32u32, 32u32)` -- so rustc proves nothing about any
/// shape in it. [`convert_pto_to_rust`] is the path whose shapes are in the
/// type system, through 148 const-generic call sites over the same corpus.
///
/// What this path really is: a stricter lift-time check. It accepts **5** of
/// the 45 distinct corpus artifacts against [`convert_pto_to_rust`]'s 13, and
/// the 5 are a strict subset, so "refuses strictly more" holds. The checking
/// happens here, in `emitter_tile`, not in the emitted code.
///
/// Its output also carries `#[tile_std::tile_kernel]`, which expands to the
/// codegen marker `#[tile::kernel]` and therefore needs the `rustc_codegen_tile`
/// backend: stock rustc rejects it with `E0433: unresolved module `tile``. That
/// is a property of the FORM and not a defect in this emitter -- the repo's own
/// hand-written kernels under `examples/tile_softmax/kernels` fail stock
/// `cargo check` with the identical error. It is why
/// `tests/lifted_kernels_compile.rs` can gate [`convert_pto_to_rust`]'s output
/// and cannot gate this one.
pub fn convert_pto_to_tile_safe(pto_src: &str) -> Result<String, String> {
    emitter_tile::emit_source(pto_src)
}
