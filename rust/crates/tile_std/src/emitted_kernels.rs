// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Kernels in the exact shape `mlir_to_tile_rs` emits them, so that rustc
//! checks what review had been checking by eye.
//!
//! That backend's whole purpose is Rust a person can read, and the playground
//! reports a green cell as soon as text comes out — it never compiles it. Every
//! claim that a generated kernel "type-checks" was therefore an inspection of
//! the call shapes, and inspection missed a whole class of defect: the emitter
//! targets the `safe` module, and several wrappers existed only at the top level
//! of `tile::`, so the generated code named functions `safe` did not have. The
//! shapes were right and the code still would not have compiled.
//!
//! These functions are compiled by `cargo check -p tile_std` (the crate is
//! `#![no_core]`, so it cannot host a normal `#[test]`; `#![allow(dead_code)]`
//! is already set crate-wide, which is what lets them sit here unused). If a
//! wrapper is removed, renamed, or has its const generics changed so a real
//! kernel no longer type-checks, this stops building.
//!
//! What this does NOT catch, checked rather than assumed: widening one operand
//! of `rms_norm` from 1536 to 1024 fails to compile, as it should, but changing
//! `attn_gqa`'s destination from 16 to 17 rows compiles fine. `SEQ` there is a
//! free parameter inferred from the destination, and the relation that would
//! pin it — `QROWS == nh * SEQ` — cannot be written, because `nh` is a runtime
//! argument. So this checks that shapes are *consistent*, not that they are the
//! intended ones; an attention kernel with a wrong sequence length still type-
//! checks. That is a limit of the API's typing, not of this module.
//!
//! Bodies are copied from emitter output verbatim apart from the `use` path.
//! Where the playground's bundled shapes are degenerate — its uniform template
//! gives a matvec weight `(1, dim)`, making N collapse to 1 — the *correct*
//! shapes are used instead, since the point is to check the API contract rather
//! than to enshrine a bundle bug.

use crate::tile::{safe, GmView, GmViewMut, tile_load_view_f32, tile_load_view_mut_f32,
                  tile_store_view_f32, tile_store_view_u32};

/// Fused weighted RMS-norm. Emitted for the `rms_norm` kernel.
pub fn rms_norm(
    p0: GmView<'_, 1, 1536, f32>,
    p1: GmView<'_, 1, 1536, f32>,
    p2: GmViewMut<'_, 1, 1536, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = safe::tile_rms_norm_mul_f32_4(v0, v1, 9.99999997E-7);
    tile_store_view_f32(&p2, v2);
}

/// Fused grouped-query attention. Emitted for `attn_gqa`: q is (nh*seq, dim)
/// with nh=12, seq=16, dim=64; k and v are (nkv*seq, dim) with nkv=2; the
/// result is (seq, dim), and SEQ is inferable only from the destination view.
pub fn attn_gqa(
    p0: GmView<'_, 192, 64, f32>,
    p1: GmView<'_, 32, 64, f32>,
    p2: GmView<'_, 32, 64, f32>,
    p3: GmViewMut<'_, 16, 64, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = tile_load_view_f32(&p2);
    let v3 = safe::tile_attention_gqa_f32(v0, v1, v2, 12, 2);
    tile_store_view_f32(&p3, v3);
}

/// In-place RoPE. Emitted for `rope`, which loads and stores the same view, so
/// its pointer is a `GmViewMut` and the load must go through the mutable-view
/// loader. This is the case that was emitting uncompilable Rust under a green
/// matrix cell.
pub fn rope(p0: GmViewMut<'_, 12, 128, f32>) {
    let v0 = tile_load_view_mut_f32(&p0);
    let v1 = safe::tile_rope_inplace_f32(v0);
    tile_store_view_f32(&p0, v1);
}

/// Local-attention window mask, shape-preserving over a rectangular score tile.
pub fn window_mask(p0: GmView<'_, 96, 96, f32>, p1: GmViewMut<'_, 96, 96, f32>) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = safe::tile_window_mask_f32(v0, 32);
    tile_store_view_f32(&p1, v1);
}

/// Strided-attention dilation mask.
pub fn strided_mask(p0: GmView<'_, 128, 128, f32>, p1: GmViewMut<'_, 128, 128, f32>) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = safe::tile_strided_mask_f32(v0, 4);
    tile_store_view_f32(&p1, v1);
}

/// Greedy decode. The result is `(R, 1)` of u32 INDICES, so it stores through
/// `tile_store_view_u32` — the f32 store would not accept it, and the emitter
/// picks the store from the tile's dtype for exactly this reason.
pub fn argmin(p0: GmView<'_, 1, 151936, f32>, p1: GmViewMut<'_, 1, 1, u32>) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = safe::tile_argmin_f32(v0);
    tile_store_view_u32(&p1, v1);
}

/// A projection at its real shapes: activation (1, K), weight (N, K), result
/// (1, N). The bundle's template gives the weight (1, K), which collapses N to
/// 1 and is why that kernel is refused rather than emitted.
pub fn q_proj(
    p0: GmView<'_, 1, 1536, f32>,
    p1: GmView<'_, 1536, 1536, f32>,
    p2: GmViewMut<'_, 1, 1536, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = safe::tile_matvec_f16(v0, v1);
    tile_store_view_f32(&p2, v2);
}

/// Fused gate/up SiLU, which returns `(N, 1)` — note the transposed result
/// shape relative to matvec.
pub fn gate_up_silu(
    p0: GmView<'_, 1, 1536, f32>,
    p1: GmView<'_, 8960, 1536, f32>,
    p2: GmView<'_, 8960, 1536, f32>,
    p3: GmViewMut<'_, 8960, 1, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = tile_load_view_f32(&p2);
    let v3 = safe::tile_gate_up_silu_f16(v0, v1, v2);
    tile_store_view_f32(&p3, v3);
}

/// The pre-existing arms, so a change to them fails here too.
pub fn vec_add(
    p0: GmView<'_, 1, 2048, f32>,
    p1: GmView<'_, 1, 2048, f32>,
    p2: GmViewMut<'_, 1, 2048, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = safe::tile_add_f32(v0, v1);
    tile_store_view_f32(&p2, v2);
}

pub fn matmul(
    p0: GmView<'_, 1, 64, f32>,
    p1: GmView<'_, 64, 64, f32>,
    p2: GmViewMut<'_, 1, 64, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = safe::tile_matmul_f32(v0, v1);
    tile_store_view_f32(&p2, v2);
}

// ── The composed form: one kernel per stage, joined through GM buffers ──────
//
// Splitting an operator into atomic kernels is what makes every backend reach
// all five attention operators (Compose.v T6), and the tile-rs column only
// reached Local and Strided once `safe` gained the mask wrappers. These are the
// four stages the composer emits for Local at its default shapes, verbatim.
//
// What compiling them does NOT check is the chaining, and the first draft of
// this comment claimed the opposite. Measured: widening stage 1's destination
// from (96, 96) to (96, 97) is 2 errors, because the matmul's result type and
// the store must agree WITHIN a function. But narrowing stage 2 consistently to
// (96, 95) while stage 1 still writes (96, 96) compiles clean — the stages are
// separate functions joined through GM buffers at runtime, and no type links
// one's destination to the next one's source.
//
// So this pins each stage's internal consistency and the wrappers it names, and
// the composer remains the only thing that knows the stages fit together. Worth
// stating plainly, because "the composed chain compiles" would otherwise be
// read as a stronger claim than it is.

/// Stage 1/4: scores = Q @ Kᵀ, (96, 32) × (96, 32)ᵀ → (96, 96).
pub fn local_attention_s0(
    p0: GmView<'_, 96, 32, f32>,
    p1: GmView<'_, 96, 32, f32>,
    p2: GmViewMut<'_, 96, 96, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = safe::tile_matmul_transposed_f32(v0, v1);
    tile_store_view_f32(&p2, v2);
}

/// Stage 2/4: the local window mask, keeping |i - j| < 3.
pub fn local_attention_s1(p0: GmView<'_, 96, 96, f32>, p1: GmViewMut<'_, 96, 96, f32>) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = safe::tile_window_mask_f32(v0, 3);
    tile_store_view_f32(&p1, v1);
}

/// Stage 3/4: row-wise softmax, shape-preserving.
pub fn local_attention_s2(p0: GmView<'_, 96, 96, f32>, p1: GmViewMut<'_, 96, 96, f32>) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = safe::tile_softmax_f32(v0);
    tile_store_view_f32(&p1, v1);
}

/// Stage 4/4: O = P @ V, (96, 96) × (96, 32) → (96, 32).
pub fn local_attention_s3(
    p0: GmView<'_, 96, 96, f32>,
    p1: GmView<'_, 96, 32, f32>,
    p2: GmViewMut<'_, 96, 32, f32>,
) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = tile_load_view_f32(&p1);
    let v2 = safe::tile_matmul_f32(v0, v1);
    tile_store_view_f32(&p2, v2);
}

/// The Strided operator differs from Local only in stage 2's predicate, so just
/// that stage is mirrored rather than the whole chain again.
pub fn strided_attention_s1(p0: GmView<'_, 128, 128, f32>, p1: GmViewMut<'_, 128, 128, f32>) {
    let v0 = tile_load_view_f32(&p0);
    let v1 = safe::tile_strided_mask_f32(v0, 4);
    tile_store_view_f32(&p1, v1);
}
