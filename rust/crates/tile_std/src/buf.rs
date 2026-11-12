// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Safe, const-generic Buffer API for Ascend NPU kernels.
//!
//! This is the Buffer-API analogue of `tile.rs`. It wraps the raw UB buffer
//! intrinsics (`__tile_buf_alloc` / `__tile_buf_load_*` / `__tile_buf_store_*`
//! / `__tile_v_add_f16` / …) in shape-annotated types so that mismatches
//! between operands are compile errors instead of silent UB overflow.
//!
//! # Design
//!
//! * `UbView<'a, CAP, T>` — typed handle to a UB allocation of compile-time
//!   capacity `CAP` elements of `T`. The `'a` lifetime is branded by the
//!   `UbCtx` that minted it, so views cannot outlive the kernel scope.
//!
//! * `UbCtx<'a>` — zero-sized context token, one per kernel. Its sole
//!   purpose is to hand out `UbView`s with matched lifetimes.
//!
//! * `DmaPending<'a, CAP, T>` — type-state marker: a freshly loaded view
//!   that has not yet been `.sync()`'d. Forgetting to sync is a compile
//!   error because vector ops take `UbView`, not `DmaPending`.
//!
//! * Ops (`ub_load_f16`, `ub_add_f16`, …) take `&UbView<CAP, _>` operands
//!   and a runtime `len: u32 ≤ CAP`. All operands must share the *same*
//!   `CAP` — enforced at compile time.
//!
//! # Why const generics on a runtime-length API
//!
//! Most Buffer-API kernels allocate with a runtime length (e.g. `n = *len_buf;
//! __tile_buf_alloc(n)`). That runtime length is still bounded: the host
//! launcher knows an upper limit, and every kernel picks a constant (e.g.
//! 4096, 65536) sized for the worst case. `CAP` captures that compile-time
//! upper bound; `len` captures the active range within it. This is identical
//! to the relationship between `[T; N]` and `&[T]` in safe Rust.
//!
//! The const generic catches the bugs that actually happen in practice:
//!   * `buf_alloc(256)` then `buf_load(…, 512)` → overflow UB (caught).
//!   * `ub_add(dst<N=256>, a<N=256>, b<N=128>)` → wrong N (compile error).
//!   * Feeding an `UbView<_, f32>` into an `f16` op → type error.
//!
//! # Pipe-barrier safety
//!
//! `DmaPending → .sync() → UbView` mirrors `pipeline::DmaPending → VecBuf`
//! from `pipeline.rs`. The `.sync()` call is the one place a `pipe_barrier`
//! is inserted for DMA→VEC; `ub_store_*` inserts the VEC→DMA barrier
//! automatically. Manual `__tile_pipe_barrier()` is not needed in kernels
//! written against this API.

use crate::core::marker::PhantomData;
use crate::UbBuf;

// =============================================================================
// UbView + UbCtx
// =============================================================================

/// Typed UB allocation of compile-time capacity `CAP` elements of `T`.
///
/// `CAP` is a worst-case upper bound chosen by the kernel author; the
/// active length within it is a runtime `len: u32 ≤ CAP` passed to ops.
///
/// Branded by `'a` so the view can't outlive the `UbCtx` that minted it.
#[repr(transparent)]
pub struct UbView<'a, const CAP: usize, T> {
    buf: UbBuf,
    _brand: PhantomData<&'a mut T>,
}

impl<'a, const CAP: usize, T> UbView<'a, CAP, T> {
    /// Drop back to a raw `UbBuf` for interop with the existing unsafe
    /// intrinsics (e.g. `kernel_ops::*` composites). End users typically
    /// do not need this.
    #[inline(always)]
    pub fn raw(&self) -> UbBuf {
        self.buf
    }
}

/// Fresh UB allocation that has been populated by DMA but not yet
/// synchronized with the vector pipe. Not usable in vector ops — call
/// `.sync()` to obtain an `UbView`.
#[repr(transparent)]
pub struct DmaPending<'a, const CAP: usize, T> {
    buf: UbBuf,
    _brand: PhantomData<&'a mut T>,
}

impl<'a, const CAP: usize, T> DmaPending<'a, CAP, T> {
    /// Insert the DMA→VEC pipe barrier and convert to a usable `UbView`.
    #[inline(always)]
    pub fn sync(self) -> UbView<'a, CAP, T> {
        unsafe { crate::__tile_pipe_barrier() };
        UbView {
            buf: self.buf,
            _brand: PhantomData,
        }
    }
}

/// Zero-sized UB context. All `UbView`s minted from it inherit its lifetime.
pub struct UbCtx<'a> {
    _brand: PhantomData<&'a mut ()>,
}

impl<'a> UbCtx<'a> {
    /// Create a fresh context. Call once per kernel.
    ///
    /// # Safety
    /// Must be called from a kernel body (so the UB allocator is valid).
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Self {
            _brand: PhantomData,
        }
    }

    /// Allocate a fresh UB view of compile-time capacity `CAP`.
    ///
    /// # Panics
    /// UB has a finite capacity shared across all allocations in the
    /// kernel; the sum of all `CAP`s minted from this ctx must fit.
    #[inline(always)]
    pub fn alloc<const CAP: usize, T>(&'a self) -> UbView<'a, CAP, T> {
        let buf = unsafe { crate::__tile_buf_alloc(CAP as u32) };
        UbView {
            buf,
            _brand: PhantomData,
        }
    }
}

// =============================================================================
// DMA load — GM → UB
// =============================================================================
//
// Each loader allocates a fresh UB slot of `CAP` elements, issues the DMA,
// and returns a `DmaPending` that the caller must `.sync()` before using
// the data in any vector op.

/// DMA-load `len` f32 elements from `gm` into a fresh `UbView<CAP, f32>`.
///
#[inline(always)]
pub fn ub_load_f32<'a, const CAP: usize>(
    ctx: &'a UbCtx<'a>,
    gm: *const f32,
    len: u32,
) -> DmaPending<'a, CAP, f32> {
    let buf = unsafe { crate::__tile_buf_alloc(CAP as u32) };
    unsafe { crate::__tile_buf_load_f32(buf, gm, len) };
    DmaPending {
        buf,
        _brand: PhantomData,
    }
}

/// DMA-load `len` f16 elements (as `u16`) from `gm`.
#[inline(always)]
pub fn ub_load_f16<'a, const CAP: usize>(
    ctx: &'a UbCtx<'a>,
    gm: *const u16,
    len: u32,
) -> DmaPending<'a, CAP, u16> {
    let buf = unsafe { crate::__tile_buf_alloc(CAP as u32) };
    unsafe { crate::__tile_buf_load_f16(buf, gm, len) };
    DmaPending {
        buf,
        _brand: PhantomData,
    }
}

/// DMA-load `len` bf16 elements (as `u16`) from `gm`.
#[inline(always)]
pub fn ub_load_bf16<'a, const CAP: usize>(
    ctx: &'a UbCtx<'a>,
    gm: *const u16,
    len: u32,
) -> DmaPending<'a, CAP, u16> {
    let buf = unsafe { crate::__tile_buf_alloc(CAP as u32) };
    unsafe { crate::__tile_buf_load_bf16(buf, gm, len) };
    DmaPending {
        buf,
        _brand: PhantomData,
    }
}

// =============================================================================
// DMA store — UB → GM (auto-inserts VEC→DMA barrier)
// =============================================================================

/// DMA-store the first `len` f32 elements of `src` to `gm`.
#[inline(always)]
pub fn ub_store_f32<const CAP: usize>(gm: *mut f32, src: &UbView<'_, CAP, f32>, len: u32) {
    unsafe {
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_f32(gm, src.buf, len);
    }
}

/// DMA-store the first `len` f16 elements of `src` to `gm`.
#[inline(always)]
pub fn ub_store_f16<const CAP: usize>(gm: *mut u16, src: &UbView<'_, CAP, u16>, len: u32) {
    unsafe {
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_f16(gm, src.buf, len);
    }
}

/// DMA-store the first `len` bf16 elements of `src` to `gm`.
#[inline(always)]
pub fn ub_store_bf16<const CAP: usize>(gm: *mut u16, src: &UbView<'_, CAP, u16>, len: u32) {
    unsafe {
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_bf16(gm, src.buf, len);
    }
}

// =============================================================================
// Fill (broadcast scalar)
// =============================================================================

#[inline(always)]
pub fn ub_fill_f32<const CAP: usize>(dst: &UbView<'_, CAP, f32>, val: f32, len: u32) {
    unsafe { crate::__tile_buf_fill_f32(dst.buf, val, len) }
}

#[inline(always)]
pub fn ub_fill_f16<const CAP: usize>(dst: &UbView<'_, CAP, u16>, val: f32, len: u32) {
    unsafe { crate::__tile_buf_fill_f16(dst.buf, val, len) }
}

#[inline(always)]
pub fn ub_fill_bf16<const CAP: usize>(dst: &UbView<'_, CAP, u16>, val: f32, len: u32) {
    unsafe { crate::__tile_buf_fill_bf16(dst.buf, val, len) }
}

// =============================================================================
// Reductions — all operands share CAP
// =============================================================================

#[inline(always)]
pub fn ub_reduce_max_f32<const CAP: usize>(
    work: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    rwork: &UbView<'_, CAP, f32>,
    len: u32,
) -> f32 {
    unsafe { crate::__tile_v_reduce_max_f32(work.buf, src.buf, rwork.buf, len) }
}

#[inline(always)]
pub fn ub_reduce_min_f32<const CAP: usize>(
    work: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    rwork: &UbView<'_, CAP, f32>,
    len: u32,
) -> f32 {
    unsafe { crate::__tile_reduce_min_f32(work.buf, src.buf, rwork.buf, len) }
}

#[inline(always)]
pub fn ub_reduce_sum_f32<const CAP: usize>(
    work: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    rwork: &UbView<'_, CAP, f32>,
    len: u32,
) -> f32 {
    unsafe { crate::__tile_v_reduce_sum_f32(work.buf, src.buf, rwork.buf, len) }
}

#[inline(always)]
pub fn ub_reduce_max_f16<const CAP: usize>(
    work: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    rwork: &UbView<'_, CAP, u16>,
    len: u32,
) -> f32 {
    unsafe { crate::__tile_reduce_max_f16(work.buf, src.buf, rwork.buf, len) }
}

#[inline(always)]
pub fn ub_reduce_sum_f16<const CAP: usize>(
    work: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    rwork: &UbView<'_, CAP, u16>,
    len: u32,
) -> f32 {
    unsafe { crate::__tile_reduce_sum_f16(work.buf, src.buf, rwork.buf, len) }
}

// =============================================================================
// Unary vector ops (f32)
// =============================================================================

#[inline(always)]
pub fn ub_exp_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_exp_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_abs_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_abs_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_ln_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_ln_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sqrt_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_sqrt_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_rsqrt_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_rsqrt_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sigmoid_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_sigmoid_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_tanh_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_tanh_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_neg_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_neg_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_expm1_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_expm1_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_reciprocal_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_reciprocal_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sign_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_sign_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sin_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_sin_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_cos_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_cos_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_atan_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_atan_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_erf_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_erf_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_relu_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_relu_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_gelu_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_gelu_f32(dst.buf, src.buf, len) }
}

// =============================================================================
// Binary vector ops (f32)
// =============================================================================

#[inline(always)]
pub fn ub_add_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    a: &UbView<'_, CAP, f32>,
    b: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_add_f32(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_sub_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    a: &UbView<'_, CAP, f32>,
    b: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_sub_f32(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_mul_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    a: &UbView<'_, CAP, f32>,
    b: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_mul_f32(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_div_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    a: &UbView<'_, CAP, f32>,
    b: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_div_f32(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_max_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    a: &UbView<'_, CAP, f32>,
    b: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_max_f32(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_min_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    a: &UbView<'_, CAP, f32>,
    b: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_v_min_f32(dst.buf, a.buf, b.buf, len) }
}

// =============================================================================
// Scalar–vector ops (f32)
// =============================================================================

#[inline(always)]
pub fn ub_adds_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    s: f32,
    len: u32,
) {
    unsafe { crate::__tile_adds_f32(dst.buf, src.buf, s, len) }
}

#[inline(always)]
pub fn ub_muls_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    s: f32,
    len: u32,
) {
    unsafe { crate::__tile_muls_f32(dst.buf, src.buf, s, len) }
}

#[inline(always)]
pub fn ub_maxs_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    s: f32,
    len: u32,
) {
    unsafe { crate::__tile_maxs_f32(dst.buf, src.buf, s, len) }
}

#[inline(always)]
pub fn ub_mins_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, f32>,
    s: f32,
    len: u32,
) {
    unsafe { crate::__tile_mins_f32(dst.buf, src.buf, s, len) }
}

// =============================================================================
// f16 ops — same CAP, u16 element type
// =============================================================================

#[inline(always)]
pub fn ub_add_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    a: &UbView<'_, CAP, u16>,
    b: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_add_f16(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_sub_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    a: &UbView<'_, CAP, u16>,
    b: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_sub_f16(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_mul_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    a: &UbView<'_, CAP, u16>,
    b: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_mul_f16(dst.buf, a.buf, b.buf, len) }
}

#[inline(always)]
pub fn ub_adds_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    s: f32,
    len: u32,
) {
    unsafe { crate::__tile_adds_f16(dst.buf, src.buf, s, len) }
}

#[inline(always)]
pub fn ub_muls_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    s: f32,
    len: u32,
) {
    unsafe { crate::__tile_muls_f16(dst.buf, src.buf, s, len) }
}

#[inline(always)]
pub fn ub_exp_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_exp_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sigmoid_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_sigmoid_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_tanh_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_tanh_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_neg_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_neg_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_expm1_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_expm1_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_reciprocal_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_reciprocal_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sign_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_sign_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_sin_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_sin_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_cos_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_cos_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_relu_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_relu_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_gelu_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_v_gelu_f16(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_min_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    a: &UbView<'_, CAP, u16>,
    b: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_min_f16(dst.buf, a.buf, b.buf, len) }
}

// =============================================================================
// Cross-dtype cast ops (f16 ↔ f32)
// =============================================================================

#[inline(always)]
pub fn ub_cast_f16_to_f32<const CAP: usize>(
    dst: &UbView<'_, CAP, f32>,
    src: &UbView<'_, CAP, u16>,
    len: u32,
) {
    unsafe { crate::__tile_cast_f16_to_f32(dst.buf, src.buf, len) }
}

#[inline(always)]
pub fn ub_cast_f32_to_f16<const CAP: usize>(
    dst: &UbView<'_, CAP, u16>,
    src: &UbView<'_, CAP, f32>,
    len: u32,
) {
    unsafe { crate::__tile_cast_f32_to_f16(dst.buf, src.buf, len) }
}

// =============================================================================
// Duplicate (broadcast a scalar into a buffer)
// =============================================================================

#[inline(always)]
pub fn ub_duplicate_f32<const CAP: usize>(dst: &UbView<'_, CAP, f32>, scalar: f32, len: u32) {
    unsafe { crate::__tile_duplicate_f32(dst.buf, scalar, len) }
}

#[inline(always)]
pub fn ub_duplicate_f16<const CAP: usize>(dst: &UbView<'_, CAP, u16>, scalar: f32, len: u32) {
    unsafe { crate::__tile_duplicate_f16(dst.buf, scalar, len) }
}

// =============================================================================
// Scalar element accessors (GetValue / SetValue)
// =============================================================================

#[inline(always)]
pub fn ub_get_value_f32<const CAP: usize>(src: &UbView<'_, CAP, f32>, idx: u32) -> f32 {
    unsafe { crate::__tile_get_value_f32(src.buf, idx) }
}

#[inline(always)]
pub fn ub_set_value_f32<const CAP: usize>(dst: &UbView<'_, CAP, f32>, idx: u32, val: f32) {
    unsafe { crate::__tile_set_value_f32(dst.buf, idx, val) }
}
