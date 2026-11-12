// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// =============================================================================
// pipeline.rs — Type-state pipeline abstraction for automatic barrier insertion
// =============================================================================
//
// Zero-cost abstraction over DMA and Vector operations that uses Rust's type
// system to enforce correct pipe_barrier placement at compile time.
//
// # Phase 1: Synchronous type-state
//   load_f32() → DmaPending ──.sync()──→ VecBuf ──(compute)──→ store_f32()
//
// # Phase 2: Future-based async
//   load_f32_async() → DmaFuture ──block_on()──→ VecBuf
//
// # Safety: DmaPending has no vector methods → forgetting .sync() = compile error.

use crate::core::future::Future;
use crate::core::pin::Pin;
use crate::core::task::{Context, Poll};

// =============================================================================
// Core types
// =============================================================================

/// Data submitted to DMA engine, pending synchronization. Move-only.
#[repr(transparent)]
pub struct DmaPending(crate::UbBuf);

/// Data ready in Unified Buffer for vector computation.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct VecBuf(crate::UbBuf);

/// Allocate a fresh VecBuf workspace (no DMA needed).
#[inline(always)]
pub unsafe fn alloc(n: u32) -> VecBuf {
    VecBuf(unsafe { crate::__tile_buf_alloc(n) })
}

impl DmaPending {
    /// Synchronize DMA pipeline → VecBuf. Inserts pipe_barrier automatically.
    #[inline(always)]
    pub fn sync(self) -> VecBuf {
        unsafe { crate::__tile_pipe_barrier() };
        VecBuf(self.0)
    }

    /// Convert to Future (for block_on or future .await).
    #[inline(always)]
    pub fn into_future(self) -> DmaFuture {
        DmaFuture(self.0)
    }
}

impl VecBuf {
    /// Get raw UbBuf for interop with existing intrinsics.
    #[inline(always)]
    pub fn raw(self) -> crate::UbBuf {
        self.0
    }
}

// =============================================================================
// Phase 1: Synchronous DMA — f32, f16, bf16
// =============================================================================

// -- f32 --
#[inline(always)]
pub unsafe fn load_f32(gm: *const f32, n: u32) -> DmaPending {
    let buf = unsafe { crate::__tile_buf_alloc(n) };
    unsafe { crate::__tile_buf_load_f32(buf, gm, n) };
    DmaPending(buf)
}
#[inline(always)]
pub unsafe fn store_f32(gm: *mut f32, data: VecBuf, n: u32) {
    unsafe {
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_f32(gm, data.0, n);
    }
}

// -- f16 (pointer is *const u16 / *mut u16 per AscendC convention) --
#[inline(always)]
pub unsafe fn load_f16(gm: *const u16, n: u32) -> DmaPending {
    let buf = unsafe { crate::__tile_buf_alloc(n) };
    unsafe { crate::__tile_buf_load_f16(buf, gm, n) };
    DmaPending(buf)
}
#[inline(always)]
pub unsafe fn store_f16(gm: *mut u16, data: VecBuf, n: u32) {
    unsafe {
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_f16(gm, data.0, n);
    }
}

// -- bf16 (also *const u16 / *mut u16) --
#[inline(always)]
pub unsafe fn load_bf16(gm: *const u16, n: u32) -> DmaPending {
    let buf = unsafe { crate::__tile_buf_alloc(n) };
    unsafe { crate::__tile_buf_load_bf16(buf, gm, n) };
    DmaPending(buf)
}
#[inline(always)]
pub unsafe fn store_bf16(gm: *mut u16, data: VecBuf, n: u32) {
    unsafe {
        crate::__tile_pipe_barrier();
        crate::__tile_buf_store_bf16(gm, data.0, n);
    }
}

// =============================================================================
// Phase 2: Future-based async DMA
// =============================================================================

/// Future that resolves when DMA load completes. One-shot.
pub struct DmaFuture(crate::UbBuf);

impl Future for DmaFuture {
    type Output = VecBuf;
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<VecBuf> {
        unsafe { crate::__tile_pipe_barrier() };
        Poll::Ready(VecBuf(self.0))
    }
}

/// Future for f32 store.
pub struct StoreFuture {
    gm: *mut f32,
    data: VecBuf,
    n: u32,
}
impl Future for StoreFuture {
    type Output = ();
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            crate::__tile_pipe_barrier();
            crate::__tile_buf_store_f32(self.gm, self.data.0, self.n);
        }
        Poll::Ready(())
    }
}

/// Future for f16 store.
pub struct StoreFutureF16 {
    gm: *mut u16,
    data: VecBuf,
    n: u32,
}
impl Future for StoreFutureF16 {
    type Output = ();
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            crate::__tile_pipe_barrier();
            crate::__tile_buf_store_f16(self.gm, self.data.0, self.n);
        }
        Poll::Ready(())
    }
}

/// Future for bf16 store.
pub struct StoreFutureBf16 {
    gm: *mut u16,
    data: VecBuf,
    n: u32,
}
impl Future for StoreFutureBf16 {
    type Output = ();
    #[inline(always)]
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
        unsafe {
            crate::__tile_pipe_barrier();
            crate::__tile_buf_store_bf16(self.gm, self.data.0, self.n);
        }
        Poll::Ready(())
    }
}

// -- Async factory functions --

#[inline(always)]
pub unsafe fn load_f32_async(gm: *const f32, n: u32) -> DmaFuture {
    let buf = unsafe { crate::__tile_buf_alloc(n) };
    unsafe { crate::__tile_buf_load_f32(buf, gm, n) };
    DmaFuture(buf)
}
#[inline(always)]
pub unsafe fn store_f32_async(gm: *mut f32, data: VecBuf, n: u32) -> StoreFuture {
    StoreFuture { gm, data, n }
}
#[inline(always)]
pub unsafe fn load_f16_async(gm: *const u16, n: u32) -> DmaFuture {
    let buf = unsafe { crate::__tile_buf_alloc(n) };
    unsafe { crate::__tile_buf_load_f16(buf, gm, n) };
    DmaFuture(buf)
}
#[inline(always)]
pub unsafe fn store_f16_async(gm: *mut u16, data: VecBuf, n: u32) -> StoreFutureF16 {
    StoreFutureF16 { gm, data, n }
}
#[inline(always)]
pub unsafe fn load_bf16_async(gm: *const u16, n: u32) -> DmaFuture {
    let buf = unsafe { crate::__tile_buf_alloc(n) };
    unsafe { crate::__tile_buf_load_bf16(buf, gm, n) };
    DmaFuture(buf)
}
#[inline(always)]
pub unsafe fn store_bf16_async(gm: *mut u16, data: VecBuf, n: u32) -> StoreFutureBf16 {
    StoreFutureBf16 { gm, data, n }
}

// -- Minimal executor --

/// Poll a Future to completion. Moral equivalent of .await for no_core.
#[inline(always)]
pub fn block_on<F: Future>(mut f: F) -> F::Output {
    let waker = ();
    let mut cx = Context { waker: &waker };
    loop {
        match unsafe { Pin::new_unchecked(&mut f) }.poll(&mut cx) {
            Poll::Ready(val) => return val,
            Poll::Pending => {}
        }
    }
}

// =============================================================================
// Phase 3: join_sync! for overlapping DMA and compute (double buffering)
// =============================================================================

/// Execute a DMA load and a compute closure concurrently, then synchronize.
///
/// On Ascend hardware, the DMA engine and Vector engine run in parallel.
/// `join_sync!` issues the DMA load first, then runs the compute closure
/// (which uses the Vector engine), then inserts a single `pipe_barrier(PIPE_ALL)`
/// to synchronize both engines.
///
/// This is the pipeline equivalent of CANN's TQue double-buffering pattern.
///
/// # Example
/// ```ignore
/// // Process tiles with DMA/compute overlap:
/// let data_a = pipeline::load_f32(ptr_a, tile).sync();
/// let (data_b, _) = pipeline::join_dma_vec(
///     || pipeline::load_f32(ptr_b, tile),    // DMA engine
///     || { out.adds(data_a, -max_val, n); }, // VEC engine (concurrent)
/// );
/// let data_b = data_b.sync();
/// ```
#[inline(always)]
pub unsafe fn join_dma_vec<F>(
    dma_load: impl FnOnce() -> DmaPending,
    vec_compute: F,
) -> (DmaPending, ())
where
    F: FnOnce(),
{
    // Issue DMA first (hardware starts transfer immediately)
    let pending = dma_load();
    // Run compute on Vector engine (runs concurrently with DMA)
    vec_compute();
    // Return the pending DMA — caller must .sync() when they need the data
    (pending, ())
}

// =============================================================================
// Phase 4 bridge: IntoFuture for DmaPending
// =============================================================================

// When the MLIR codegen supports async fn desugaring (Phase 4),
// this enables: `let data = load_f32(input, n).await;`
// The .await desugars to IntoFuture::into_future() → DmaFuture → poll().
impl crate::core::future::IntoFuture for DmaPending {
    type Output = VecBuf;
    type IntoFuture = DmaFuture;
    #[inline(always)]
    fn into_future(self) -> DmaFuture {
        DmaFuture(self.0)
    }
}

// =============================================================================
// Phase 2.5b: Complete VecBuf operations (f32)
// =============================================================================

impl VecBuf {
    // -- Reductions (f32) --
    #[inline(always)]
    pub unsafe fn reduce_max(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_v_reduce_max_f32(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn reduce_min(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_reduce_min_f32(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn reduce_sum(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_v_reduce_sum_f32(work.0, self.0, rwork.0, n) }
    }

    // -- Unary (f32) --
    #[inline(always)]
    pub unsafe fn exp(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_exp_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn abs(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_abs_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn ln(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_ln_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sqrt(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_sqrt_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn rsqrt(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_rsqrt_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn reciprocal(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_reciprocal_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sign(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_sign_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn round(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_round_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sq(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_sq_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn ceil(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_ceil_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn floor(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_floor_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn trunc(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_trunc_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sin(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_sin_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn cos(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_cos_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn atan(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_atan_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn erf(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_erf_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn erfinv(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_erfinv_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn fast_gelu(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_fast_gelu_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn not(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_not_f32(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn softplus(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_softplus_f32(self.0, src.0, n) }
    }

    // -- Scalar (f32) --
    #[inline(always)]
    pub unsafe fn adds(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_adds_f32(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn muls(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_muls_f32(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn maxs(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_maxs_f32(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn mins(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_mins_f32(self.0, src.0, s, n) }
    }

    // -- Binary (f32) --
    #[inline(always)]
    pub unsafe fn add(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_add_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sub(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_sub_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn mul(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_mul_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn div(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_div_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn max(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_max_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn min(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_min_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn and(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_and_f32(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn or(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_or_f32(self.0, a.0, b.0, n) }
    }

    // -- f16 operations (same VecBuf, different intrinsics) --
    #[inline(always)]
    pub unsafe fn reduce_max_f16(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_reduce_max_f16(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn reduce_min_f16(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_reduce_min_f16(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn reduce_sum_f16(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_reduce_sum_f16(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn exp_f16(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_exp_f16(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn abs_f16(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_v_abs_f16(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn ln_f16(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_ln_f16(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn adds_f16(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_adds_f16(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn muls_f16(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_muls_f16(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn add_f16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_add_f16(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sub_f16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_sub_f16(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn mul_f16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_v_mul_f16(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn div_f16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_div_f16(self.0, a.0, b.0, n) }
    }

    // -- bf16 operations --
    #[inline(always)]
    pub unsafe fn reduce_max_bf16(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_reduce_max_bf16(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn reduce_sum_bf16(self, work: VecBuf, rwork: VecBuf, n: u32) -> f32 {
        unsafe { crate::__tile_reduce_sum_bf16(work.0, self.0, rwork.0, n) }
    }
    #[inline(always)]
    pub unsafe fn exp_bf16(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_exp_bf16(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn abs_bf16(self, src: VecBuf, n: u32) {
        unsafe { crate::__tile_abs_bf16(self.0, src.0, n) }
    }
    #[inline(always)]
    pub unsafe fn adds_bf16(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_adds_bf16(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn muls_bf16(self, src: VecBuf, s: f32, n: u32) {
        unsafe { crate::__tile_muls_bf16(self.0, src.0, s, n) }
    }
    #[inline(always)]
    pub unsafe fn add_bf16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_add_bf16(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn sub_bf16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_sub_bf16(self.0, a.0, b.0, n) }
    }
    #[inline(always)]
    pub unsafe fn mul_bf16(self, a: VecBuf, b: VecBuf, n: u32) {
        unsafe { crate::__tile_mul_bf16(self.0, a.0, b.0, n) }
    }

    // -- Fill (broadcast scalar) --
    #[inline(always)]
    pub unsafe fn fill(self, val: f32, n: u32) {
        unsafe { crate::__tile_buf_fill_f32(self.0, val, n) }
    }
    #[inline(always)]
    pub unsafe fn fill_f16(self, val: f32, n: u32) {
        unsafe { crate::__tile_buf_fill_f16(self.0, val, n) }
    }
    #[inline(always)]
    pub unsafe fn fill_bf16(self, val: f32, n: u32) {
        unsafe { crate::__tile_buf_fill_bf16(self.0, val, n) }
    }
}

// =============================================================================
// Phase 2.5c: Composite kernel_ops via pipeline
// =============================================================================

/// Composite operations that chain multiple vector intrinsics.
/// These wrap kernel_ops functions, providing the pipeline's type safety
/// while reusing the proven composite implementations.
pub mod ops {
    use super::VecBuf;

    // Simple composites: take UbBuf by value
    #[inline(always)]
    pub unsafe fn relu(dst: VecBuf, src: VecBuf, n: u32) {
        unsafe { crate::kernel_ops::relu_f32(dst.0, src.0, n) }
    }

    #[inline(always)]
    pub unsafe fn sigmoid(dst: VecBuf, src: VecBuf, n: u32) {
        unsafe { crate::kernel_ops::sigmoid_f32(dst.0, src.0, n) }
    }

    #[inline(always)]
    pub unsafe fn tanh(dst: VecBuf, src: VecBuf, n: u32) {
        unsafe { crate::kernel_ops::tanh_f32(dst.0, src.0, n) }
    }

    // Complex composites: take &mut UbBuf — extract raw buf and pass mutably.
    #[inline(always)]
    pub unsafe fn gelu(dst: VecBuf, src: VecBuf, tmp: VecBuf, n: u32) {
        let (mut d, mut t) = (dst.0, tmp.0);
        unsafe { crate::kernel_ops::gelu_f32(&mut d, &src.0, &mut t, n) }
    }

    #[inline(always)]
    pub unsafe fn softmax(dst: VecBuf, src: VecBuf, work: VecBuf, n: u32) {
        let (mut d, mut s, mut w) = (dst.0, src.0, work.0);
        unsafe { crate::kernel_ops::softmax_f32(&mut d, &mut s, &mut w, n) }
    }

    #[inline(always)]
    pub unsafe fn layernorm(dst: VecBuf, src: VecBuf, work: VecBuf, n: u32, eps: f32) {
        let (mut d, mut w) = (dst.0, work.0);
        unsafe { crate::kernel_ops::layernorm_f32(&mut d, &src.0, &mut w, n, eps) }
    }

    #[inline(always)]
    pub unsafe fn log_softmax(dst: VecBuf, src: VecBuf, work: VecBuf, rwork: VecBuf, n: u32) {
        let (mut d, mut s, mut w, mut r) = (dst.0, src.0, work.0, rwork.0);
        unsafe { crate::kernel_ops::log_softmax_f32(&mut d, &mut s, &mut w, &mut r, n) }
    }

    #[inline(always)]
    pub unsafe fn rms_norm(dst: VecBuf, src: VecBuf, work: VecBuf, n: u32, eps: f32) {
        let (mut d, mut w) = (dst.0, work.0);
        unsafe { crate::kernel_ops::rms_norm_f32(&mut d, &src.0, &mut w, n, eps) }
    }

    #[inline(always)]
    pub unsafe fn leaky_relu(dst: VecBuf, src: VecBuf, neg: VecBuf, alpha: f32, n: u32) {
        let (mut d, mut s, mut ne) = (dst.0, src.0, neg.0);
        unsafe { crate::kernel_ops::leaky_relu_f32(&mut d, &mut s, &mut ne, alpha, n) }
    }

    #[inline(always)]
    pub unsafe fn mish(dst: VecBuf, src: VecBuf, tmp: VecBuf, n: u32) {
        let (mut d, mut t) = (dst.0, tmp.0);
        unsafe { crate::kernel_ops::mish_f32(&mut d, &src.0, &mut t, n) }
    }

    #[inline(always)]
    pub unsafe fn fast_gelu(dst: VecBuf, src: VecBuf, n: u32) {
        unsafe { crate::__tile_fast_gelu_f32(dst.0, src.0, n) }
    }
}
