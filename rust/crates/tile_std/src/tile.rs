// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Tile-level abstractions for Ascend NPU kernels.
//!
//! This module provides a type-safe tile buffer handle and tile operations
//! that mirror the PTO (Programmable Tile Operations) ISA for Ascend.
//!
//! # Design
//!
//! Each tile operation is backed by an `extern "C"` intrinsic that carries
//! shape information as const generic parameters. The `mlir_to_cpp` codegen
//! translates these to AscendC `TBuf`/`DataCopy` API calls (current path),
//! and `mlir_to_pto` will translate them to `.pto` assembly (future path).
//!
//! # Ownership and safety
//!
//! `Tile<ROWS, COLS, T>` is a *move-only* handle (no `Copy`). This ensures:
//! - Each tile buffer is consumed by exactly one `tile_store` (no double-DMA)
//! - Tile values cannot alias once loaded (borrow checker enforces single owner)
//!
//! # Example
//!
//! ```rust,ignore
//! use tile_std::tile::{tile_load_f32, tile_store_f32, tile_add_f32};
//!
//! #[kernel]
//! pub fn vec_add(a: *const f32, b: *const f32, c: *mut f32, n: u32) {
//!     let idx = unsafe { get_block_idx() };
//!     let off = idx * 16;
//!     let ta: Tile<16, 1, f32> = unsafe { tile_load_f32::<16, 1>(a.add(off)) };
//!     let tb: Tile<16, 1, f32> = unsafe { tile_load_f32::<16, 1>(b.add(off)) };
//!     let tc = unsafe { tile_add_f32(ta, tb) };
//!     unsafe { tile_store_f32(c.add(off), tc) };
//! }
//! ```

use crate::core::marker::PhantomData;

/// A typed tile buffer handle.
///
/// Encodes the tile dimensions (`ROWS` × `COLS`) and element type `T`
/// as const generics so that shape mismatches are compile errors.
///
/// `Tile` IS `Copy` as of 2026-11-09, and what that gave up is worth stating.
///
/// It was deliberately not `Copy`, with the stated reason that "each tile must be
/// consumed by exactly one store operation, preventing double-DMA bugs". That
/// guarantee was real, but it was enforcing linearity on a handle whose compute
/// uses are READS: in PTO every compute op reads its inputs and writes a NEW
/// tile, so nothing is consumed, and ten corpus kernels could not be lifted
/// because a computed tile read by several ops would have been moved twice.
///
/// `Copy` is additive for every existing caller -- it only removes a compile
/// error -- so the cost is precisely the double-store guarantee, which now has to
/// be CHECKED rather than typed. See
/// `ptoas_rs/tests/no_lifted_kernel_stores_a_tile_twice.rs`.
///
/// The alternative considered and rejected: make compute ops take `&Tile` and
/// leave the store consuming by value, which keeps the guarantee in the type
/// system and is the more precise model. It changes the signature of every
/// wrapper and so breaks every existing caller, including the hand-written
/// corpus kernels. That is the better end state and a much larger change; this
/// is the additive step, and the gate is what keeps it honest in the meantime.
#[derive(Clone, Copy)]
pub struct Tile<const ROWS: usize, const COLS: usize, T> {
    /// Internal buffer ID used by the kernel runtime.
    /// This maps to a `TBuf` handle in the AscendC C++ codegen path.
    pub(crate) buf_id: u32,
    _phantom: PhantomData<T>,
}

// --- Device-memory views --------------------------------------------------
//
// `GmView` and `GmViewMut` are the typed replacement for bare `*const T` /
// `*mut T` kernel parameters. They carry:
//
//   * the element type `T` (so `tile_load` knows the dtype without a suffix),
//   * the tile shape `ROWS × COLS` (so the host must commit to a concrete
//     shape at the call site — shape mismatches between host and kernel
//     become compile errors instead of memory corruption),
//   * a lifetime `'a` (so the view can't be stashed in a global or returned
//     past the end of the kernel).
//
// Both are `#[repr(transparent)]` around a raw pointer, so `extern "C"`
// kernels still accept them with the same ABI as `*const T` / `*mut T`.
// The `mlir_to_pto` codegen reads pointer-kind from MIR's `ptr<1>` addrspace
// and is blind to Rust newtypes; lowering is unchanged.
//
// The ONLY unsafe primitives are `GmDeviceCtx::new()` and
// `ctx.view{,_mut}::<R,C,T>(ptr)`: the one place where the caller is
// asserting "this pointer really does back `R * C` elements of `T`".
// In practice that's the host-side launcher, once per kernel invocation.
// Everything downstream — every `tile_load_view_*` / `tile_store_view_*`
// call, and every kernel body — is safe.
//
// Views borrow `&'a self` from the ctx so they cannot outlive it; returning
// a view from a function is a borrow-checker error.

/// Typed, shape-annotated read-only view over `ROWS × COLS` elements of
/// `T` in device global memory.
#[repr(transparent)]
pub struct GmView<'a, const ROWS: usize, const COLS: usize, T> {
    ptr: *const T,
    _brand: PhantomData<&'a T>,
}

/// Typed, shape-annotated mutable view over `ROWS × COLS` elements of `T`
/// in device global memory.
#[repr(transparent)]
pub struct GmViewMut<'a, const ROWS: usize, const COLS: usize, T> {
    ptr: *mut T,
    _brand: PhantomData<&'a mut T>,
}

impl<'a, const R: usize, const C: usize, T> GmView<'a, R, C, T> {
    /// Drop back to a raw pointer. The inner intrinsics need this; end
    /// users typically don't.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

impl<'a, const R: usize, const C: usize, T> GmViewMut<'a, R, C, T> {
    #[inline(always)]
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }
}

/// Device-memory context: a zero-sized token whose lifetime `'a` brands
/// every view it mints. The launcher owns one, and views borrow from it,
/// so no view can outlive the launch.
///
/// The two `unsafe` hops (`new` + `view{,_mut}`) are the only unsafe the
/// caller ever writes, and they sit at the host-side boundary where
/// "this pointer really backs R*C elements" is a claim the caller can
/// make in good faith.
pub struct GmDeviceCtx<'a> {
    _brand: PhantomData<&'a mut ()>,
}

impl<'a> GmDeviceCtx<'a> {
    /// Create a fresh context.
    ///
    /// # Safety
    /// The caller is asserting that any pointers they will feed into
    /// `view` / `view_mut` on this ctx are valid for the entire scope
    /// of `'a` (which is bounded by the ctx's stack slot).
    #[inline(always)]
    pub unsafe fn new() -> Self {
        Self {
            _brand: PhantomData,
        }
    }

    /// Mint a typed read-only view. The returned view is borrowed against
    /// `self`, so it can't outlive the ctx.
    ///
    /// # Safety
    /// `ptr` must back at least `R*C` contiguous readable elements of `T`.
    #[inline(always)]
    pub unsafe fn view<const R: usize, const C: usize, T>(
        &'a self,
        ptr: *const T,
    ) -> GmView<'a, R, C, T> {
        GmView {
            ptr,
            _brand: PhantomData,
        }
    }

    /// Mint a typed mutable view.
    ///
    /// # Safety
    /// `ptr` must back at least `R*C` contiguous writable elements of `T`
    /// and must not alias any other live view from this ctx.
    #[inline(always)]
    pub unsafe fn view_mut<const R: usize, const C: usize, T>(
        &'a self,
        ptr: *mut T,
    ) -> GmViewMut<'a, R, C, T> {
        GmViewMut {
            ptr,
            _brand: PhantomData,
        }
    }
}

// --- f32 tile operations --------------------------------------------------

extern "C" {
    /// Load a `ROWS × COLS` f32 tile from global memory into a local tile buffer.
    ///
    /// In the AscendC path: `DataCopy(local_buf, gm_ptr, ROWS * COLS)`.
    /// In the PTO path: `tile.load %dst, %src, [ROWS, COLS]`.
    ///
    /// # Safety
    /// `gm` must point to at least `ROWS * COLS` valid f32 values in global memory.
    pub fn __tile_load_f32(gm: *const f32, rows: u32, cols: u32) -> u32;

    /// Load global memory INTO AN EXISTING tile buffer, reusing it.
    ///
    /// The same DMA as `__tile_load_f32`, with the destination given rather than
    /// allocated. It exists because the chunked emitters already do exactly this
    /// and nothing in the tile API could say so: a long dot or reduction wider
    /// than the unified buffer is emitted as `pto.tload` into ONE scratch tile,
    /// once per chunk, because that is what makes it fit. Measured on
    /// glm53_lm_head_154880 -- the `.tile` has two loads, the `.pto` loads a new
    /// chunk into one tile 22 times, and 22 x 7040 is exactly that kernel's
    /// 154880-wide extent. Giving each chunk its own tile needs 619,520 bytes
    /// against a 194,560-byte unified buffer.
    ///
    /// Lowers to `pto.tload` with an existing `outs` tile, which is the spelling
    /// the stored artefacts already carry -- so this adds a NAME for what both
    /// backends emit, not a capability either has to learn.
    ///
    /// # Safety
    /// `gm` must point to at least `rows * cols` valid f32 values, and `buf` must
    /// be a live buffer of at least that extent.
    pub fn __tile_load_into_f32(buf: u32, gm: *const f32, rows: u32, cols: u32);

    /// Store a local tile buffer to global memory.
    ///
    /// Consumes the tile buffer (by ID). In the AscendC path: `DataCopy(gm_ptr, local_buf, N)`.
    /// In the PTO path: `tile.store %dst, %src, [ROWS, COLS]`.
    ///
    /// # Safety
    /// `gm` must point to at least `ROWS * COLS` writable f32 values in global memory.
    pub fn __tile_store_f32(gm: *mut f32, buf: u32, rows: u32, cols: u32);

    /// Element-wise add two f32 tiles of the same shape.
    pub fn __tile_add_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise multiply two f32 tiles of the same shape.
    pub fn __tile_mul_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise exp of an f32 tile.
    pub fn __tile_exp_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise softmax of an f32 tile (in-place reduction over columns).
    pub fn __tile_softmax_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Matrix multiply: (M × K) @ (K × N) → (M × N), all f32.
    pub fn __tile_matmul_f32(dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32) -> u32;

    /// Element-wise subtract two f32 tiles: a - b.
    pub fn __tile_sub_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise divide two f32 tiles: a / b.
    pub fn __tile_div_f32(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise negate an f32 tile: -a.
    pub fn __tile_neg_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise max reduction of an f32 tile.
    /// Returns a (ROWS × 1) tile where each row is the max of that row.
    pub fn __tile_reduce_max_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Store ROWS x COLS to `gm` starting at element OFFSET.
    ///
    /// The counterpart of `__tile_load_at_f32`, and needed for the same reason: a per-head loop writes
    /// head h's 512 floats at `h * 512`, and `__tile_store_f32` always targets element 0.
    pub fn __tile_store_at_f32(gm: *mut f32, t: u32, offset: u32, rows: u32, cols: u32) -> u32;

    /// Load ROWS x COLS from `gm` starting at element OFFSET.
    ///
    /// `__tile_load_f32` takes the GM ARG itself, so only element 0 of a tensor is reachable. The
    /// engine's flash attention needs K/V block b of a `[128][512]` tensor, i.e. an offset of
    /// `b * 8 * 512`; without this only block 0 exists. The emitter already resolves GM offsets for
    /// other intrinsics, so this exposes machinery that was there.
    pub fn __tile_load_at_f32(gm: *const f32, offset: u32, rows: u32, cols: u32) -> u32;

    /// Copy COUNT elements from SRC into DST starting at element OFFSET.
    ///
    /// Attention builds its 128 scores BLOCK BY BLOCK (16 blocks of 8) and needs to place each
    /// block's 8 results at 8*b. There is no tile-slice write in this DSL otherwise -- every other
    /// intrinsic produces a whole tile -- so the two-pass formulation cannot be expressed without it.
    pub fn __tile_slice_store_f32(dst: u32, src: u32, offset: u32, count: u32) -> u32;

    /// `acc += w[row] * src[row][:]` -- one row of a weighted-V accumulation.
    ///
    /// The last piece the engine's flash attention needs, and it exists because the column-wise
    /// reduction that `acc += sum_r w[r]*V[r][:]` would otherwise require is NOT in this DSL
    /// (`__tile_reduce_sum_f32` is row-wise). Accumulating row by row needs no reduction at all.
    /// `windex` addresses the weight tile, `row` the source tile: attention's weights are indexed by
    /// GLOBAL position (8*b + r) while the source row is the block-local r, so one index cannot serve
    /// both. The 5-argument form assumes they coincide.
    pub fn __tile_row_axpy2_f32(acc: u32, src: u32, w: u32, win: u32, row: u32, cols: u32) -> u32;

    pub fn __tile_row_axpy_f32(acc: u32, src: u32, w: u32, row: u32, cols: u32) -> u32;

    /// Scale each row of an ROWS x COLS tile by that row's entry in a 1 x ROWS tile.
    ///
    /// The runtime-scalar counterpart of `__tile_scale_f32`, whose scalar is a compile-time f32.
    /// The engine's flash attention needs exactly this for its weighted-V accumulation:
    /// `acc += w[r] * V[r][:]` where `w` is a softmax result computed on the device.
    pub fn __tile_row_scale_f32(dst: u32, src: u32, w: u32, rows: u32, cols: u32) -> u32;

    /// Broadcast a 1 x COLS tile across ROWS rows -- the COLUMN-wise counterpart of
    /// `__tile_row_expand_f32`.
    ///
    /// Needed by the engine's flash-blocked attention, whose first step is broadcasting Q[1xD]
    /// across the 8 rows of a K/V block (fixture `tcolexpand`, layer0.pto:437+). Like row_expand
    /// this is a scalar loop in the emitter rather than a vector op, because the vocabulary this
    /// emitter uses has no tile-to-block broadcast.
    pub fn __tile_col_expand_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise sum reduction of an f32 tile.
    pub fn __tile_reduce_sum_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Scalar multiply: scale all elements of a tile by a scalar.
    /// Tile-to-tile copy. PTO lowering: `pto.tmov`, a move on the vector unit
    /// rather than a multiply by one, so a lifted `pto.tmov` round-trips exactly.
    pub fn __tile_copy_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_copy_i16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Divide every element by a runtime scalar. PTO lowering: `pto.tdivs`
    /// (tile/scalar). Distinct from `__tile_div_f32`, which takes two tiles.
    pub fn __tile_divs_f32(dst: u32, src: u32, scalar: f32, rows: u32, cols: u32) -> u32;

    pub fn __tile_scale_f32(dst: u32, src: u32, scalar: f32, rows: u32, cols: u32) -> u32;

    /// Broadcast a `1 x 1` tile over `n` columns.
    ///
    /// The delta-net's decode step multiplies `v` by the single value `q.k`, and
    /// without this there is no way to express that: the tile API's elementwise
    /// ops are same-shape, and `scale` takes a scalar the compiler knows, not one
    /// the kernel computed. The emitters knew the op existed (`pto.trowexpand`)
    /// but `tile_std` declared no name for it, which `check_call_conformance`
    /// caught when a kernel called one.
    pub fn __tile_row_expand_f32(dst: u32, src: u32, n: u32) -> u32;
    pub fn __tile_getval_f32(src: u32, idx: u32) -> f32;
    pub fn __tile_row_expand_div_f32(dst: u32, src: u32, red: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_row_expand_sub_f32(dst: u32, src: u32, red: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_row_expand_mul_f32(dst: u32, src: u32, red: u32, rows: u32, cols: u32) -> u32;

    /// Block broadcast: `dst[i] = src[i] <op> vec[i / block]` over a 1 x N row,
    /// where `vec` is a 1 x G vector and `block = N / G`. This is the dequant
    /// group form (8 groups over 4096), NOT the Cx1 row-expand form.
    pub fn __tile_block_expand_sub_f32(dst: u32, src: u32, vec: u32, n: u32, block: u32) -> u32;
    pub fn __tile_block_expand_mul_f32(dst: u32, src: u32, vec: u32, n: u32, block: u32) -> u32;
    pub fn __tile_block_expand_div_f32(dst: u32, src: u32, vec: u32, n: u32, block: u32) -> u32;

    /// Scalar add: add one scalar to every element of a tile.
    ///
    /// Declared because both emitters lower it -- the AscendC side has had
    /// `__tile_adds_*` all along, the PTO side gained an arm on 2026-11-04 --
    /// and because a chain that needs a constant can then avoid materialising a
    /// constant TILE, which is a live buffer in an emitter that never frees one.
    pub fn __tile_adds_f32_u32(dst: u32, src: u32, scalar: f32, n: u32) -> u32;

    /// Scalar subtract: subtract one scalar from every element of a tile.
    pub fn __tile_subs_f32(dst: u32, src: u32, scalar: f32, n: u32) -> u32;

    /// Element-wise clamp into `[lo, hi]`. Used to fence activations under
    /// MoE / DS4-style gating (cf. DS4 SwiGLU clamp_value).
    pub fn __tile_clamp_f32(dst: u32, src: u32, lo: f32, hi: f32, rows: u32, cols: u32) -> u32;

    /// Fused attention: softmax(Q @ K^T / sqrt(d)) @ V
    /// Q: (S × D), K: (S × D), V: (S × D) → out: (S × D)
    /// Fuses matmul + scale + softmax + matmul into a single PTOAS pipeline.
    pub fn __tile_attention_f32(
        dst: u32,
        q: u32,
        k: u32,
        v: u32,
        seq_len: u32,
        head_dim: u32,
    ) -> u32;

    /// Fused grouped-query attention: `softmax(Q @ Kᵀ / sqrt(d)) @ V` with
    /// `heads_q` query heads sharing `heads_kv` key/value heads.
    ///
    /// Q: (heads_q × seq × head_dim), K/V: (heads_kv × seq × head_dim)
    /// → out: (heads_q × seq × head_dim).
    ///
    /// ⚠ THIS DECLARATION SETTLES A DISAGREEMENT. The symbol had no
    /// declaration and the backends had drifted apart on what the trailing
    /// operands mean:
    ///
    ///   mlir_to_cpp / mlir_to_gpu / mlir_to_aie   heads_q, heads_kv, seq, head_dim
    ///   mlir_to_nki                               heads, head_dim, seq, kv_heads
    ///
    /// so one MLIR module lowered through `mlir_to_cpp` and through
    /// `mlir_to_nki` computed different attention shapes, and one of them was
    /// silently wrong. Each fixture had been written to match its own emitter,
    /// which is exactly what having no declared contract permits.
    ///
    /// The order below is the one three backends already implement. NKI is
    /// therefore the divergent reader, and NKI support for this operator is
    /// POSTPONED by decision rather than fixed here — see the note on the
    /// `mlir_to_nki` arm.
    pub fn __tile_attention_gqa_f32(
        dst: u32,
        q: u32,
        k: u32,
        v: u32,
        heads_q: u32,
        heads_kv: u32,
        seq_len: u32,
        head_dim: u32,
    ) -> u32;

    // ── Transformer building-block intrinsics ───────────────────────

    /// SiLU/Swish activation: silu(x) = x * sigmoid(x).
    pub fn __tile_silu_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// RoPE: applies sin/cos rotation to (R, D) tile at position `pos`.
    pub fn __tile_rope_f32(dst: u32, src: u32, pos: u32, rows: u32, cols: u32) -> u32;

    /// DS4 kernel_rms_norm_mul_f32_4, F=2 variant: float4-vectorized
    /// RMSNorm of one row × per-element learned weight. Grid is one tg per row
    /// (tgpig.x), and reduction uses simd_sum + threadgroup shmem broadcast.
    /// Buffers all char*: src (const), weight (const), dst (writable).
    /// Corrected against the call the emitters actually parse:
    /// `(%x, %g, %n, %n4, %rs, %eps)`. This used to declare a third `dst`
    /// parameter that no call site passes, and to type `eps` as `u32` while
    /// every caller passes an `f32` constant — the MLIR signature ends in `f32`,
    /// and every sibling (`__tile_rms_norm_f32`, `__tile_quantize_f32_i8`,
    /// `__tile_dequantize_i8_f32`) declares its float parameter as `f32`. So it
    /// was a wrong declaration rather than a bit-pattern convention. Nothing in
    /// Rust called it, so nothing depended on the old shape.
    pub fn __tile_rms_norm_mul_f32_4(
        src: u32,
        weight: u32,
        n: u32,
        n4: u32,
        row_stride: u32,
        eps: f32,
    ) -> u32;

    /// DS4 kernel_unary_f32_f32_4 (NEG op): per-row float4 lanewise -x.
    pub fn __tile_neg_f32_4(src: u32, dst: u32, ne0_4: u32, nb_src: u32, nb_dst: u32) -> u32;

    /// DS4 kernel_unary_f16_f16 (SILU op): scalar half `x / (1 + exp(-x))`.
    pub fn __tile_silu_f16(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;

    /// DS4 kernel_unary_f16_f16 (EXP op): scalar half `exp(x)`.
    pub fn __tile_exp_f16(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;
    /// DS4 kernel_unary_f32_f32 (neg op): per-row scalar float -x.
    pub fn __tile_neg_f32_scalar(src: u32, dst: u32, ne0: u32, nb_src: u32, nb_dst: u32) -> u32;

    /// DS4 kernel_mul_mv_id_q8_0_f32: MoE-routed Q8_0 matvec.
    /// Per (idx, iid1) the routed expert i02 is read from ids; src0s is
    /// offset by i02*nb02 and the unrouted q8_0 inner dot runs. Grid is
    /// (tx, 1, idx + iid1 * nei0). Buffers: src0s, src1, ids, dst
    /// (dst is last so the writable-buf heuristic marks it correctly).
    pub fn __tile_mul_mv_id_q8_0_f32(
        src0s: u32,
        src1: u32,
        ids: u32,
        dst: u32,
        ne00: u32,
        ne01: u32,
        ne0: u32,
        ne1: u32,
        ne11: u32,
        nei0: u32,
        nbi1: u32,
        nb01: u32,
        nb02: u32,
        nb11: u32,
        nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_q2_K_f32: q2_K sibling of
    /// `__tile_mul_mv_id_q8_0_f32`.
    /// Same 4-buf shell (src0s, src1, ids, dst) + 11-uint params; inner runs
    /// kernel_mul_mv_q2_K_f32_impl with NR0=N_R0_Q2_K=4, QK_K=256, block_q2_K=84B
    /// (uchar scales[16] + uchar qs[64] + half d + half dmin).
    pub fn __tile_mul_mv_id_q2_K_f32(
        src0s: u32,
        src1: u32,
        ids: u32,
        dst: u32,
        ne00: u32,
        ne01: u32,
        ne0: u32,
        ne1: u32,
        ne11: u32,
        nei0: u32,
        nbi1: u32,
        nb01: u32,
        nb02: u32,
        nb11: u32,
        nb12: u32,
    ) -> u32;

    /// DS4 kernel_mul_mv_id_iq2_xxs_f32: iq2_xxs sibling of the routed q2_K
    /// and q4_K matvecs. Same 4-buf shell + 11-uint params; inner runs
    /// kernel_mul_mv_iq2_xxs_f32_impl with
    /// NR0=N_R0_IQ2_XXS=4, block_iq2_xxs=66 B (half d + ushort qs[32]).
    /// Stages file-scope iq2xxs_grid[256] (ulong) + ksigns_iq2xs[128]
    /// into threadgroup shmem; per ib32 decodes 4 q2 ushorts → aux32 →
    /// grid lookup + ksigns + kmask_iq2xs sign bits; output *= 0.25f.
    pub fn __tile_mul_mv_id_iq2_xxs_f32(
        src0s: u32,
        src1: u32,
        ids: u32,
        dst: u32,
        ne00: u32,
        ne01: u32,
        ne0: u32,
        ne1: u32,
        ne11: u32,
        nei0: u32,
        nbi1: u32,
        nb01: u32,
        nb02: u32,
        nb11: u32,
        nb12: u32,
    ) -> u32;

    /// Causal mask: fills upper-triangular portion of (S, S) tile with -inf.
    pub fn __tile_causal_mask_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Window (local) mask: keeps `|i - j| < window`, masking the rest.
    ///
    /// Sibling of the causal mask, differing only in the predicate, and it is
    /// **symmetric** — not a backwards-looking causal window. That matches the
    /// reference it was derived from, `23_LocalAttention/model.py`, which masks
    /// on `(pos_i - pos_j).abs() >= window_size`. The masked value is the same
    /// finite sentinel the causal arm uses, so all three masks agree.
    pub fn __tile_window_mask_f32(dst: u32, src: u32, window: u32, rows: u32, cols: u32) -> u32;

    /// Strided (dilated) mask: keeps `i % stride == j % stride`.
    ///
    /// The other mask sibling, from `25_StridedAttention`'s `residue_i !=
    /// residue_j` fill.
    pub fn __tile_strided_mask_f32(dst: u32, src: u32, stride: u32, rows: u32, cols: u32) -> u32;

    /// Embedding lookup: gathers `count` rows from (V, D) weight table by index.
    pub fn __tile_embedding_f32(
        dst: u32,
        weight: u32,
        indices: *const u32,
        vocab_size: u32,
        embed_dim: u32,
        count: u32,
    ) -> u32;

    /// Cross-entropy loss: -log(softmax(logits)[target]) per row.
    pub fn __tile_cross_entropy_f32(
        dst: u32,
        logits: u32,
        targets: *const u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    // ── Phase 0: foundational primitives for DeepSeek/LLM serving ───

    /// Transpose (ROWS × COLS) → (COLS × ROWS).
    pub fn __tile_transpose_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    // Operators `mlir_to_msl` reaches by kernel-type classification. Their arms
    // set `ctx.kernel_type` and never read the call's operands, so the arity here
    // comes from the emitter's own fixtures rather than from a banner or an arity
    // guard — weaker evidence than the arms above, and recorded as such. Note the
    // shapes genuinely differ: layernorm, l2dist and scatter_add carry NO leading
    // `dst`, where `where` does, which is why each was read individually instead
    // of being given a house convention.

    // The partition family, all lowered by `mlir_to_pto`. Creation registers a GM
    // buffer and its extent; a cell then resolves one (i, j) of a Tr x Tc grid to
    // an offset view. Signatures come from the translators: creation reads
    // args[0] as a GM pointer through `resolve_ptr`/`resolve_gm_name` and
    // args[1..=2] as the extent, and the cell forms are documented as
    // `(src, i, j, tr, tc)` in the emitter and used at that arity by its fixtures.

    // More of the `mlir_to_pto` surface. Whether an operand is a GM pointer or a
    // tile handle is not a style choice here and was not guessed: a translator
    // that takes the enclosing `func` resolves its operands through
    // `resolve_ptr`/`resolve_gm_name` and is therefore working on pointers, and
    // one that does not is working on tile handles. That split is what decides
    // every signature below.

    /// Bitwise ops against an immediate, on i8 tiles. Shared translator, so the
    /// three differ only in the PTO op they select.
    pub fn __tile_ands_i8(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_adds_i16(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_add_i16(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_scale_i16(dst: u32, src: u32, scalar: i16, rows: u32, cols: u32) -> u32;
    pub fn __tile_mul_i16(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_maxs_i16(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_ands_i16(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_shls_i16(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_shrs_i16(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_shls_i8(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_shrs_i8(dst: u32, src: u32, imm: u32, rows: u32, cols: u32) -> u32;

    /// dtype conversion between tiles. The name reads source-then-destination:
    /// the arm passes ("f16", "f32") for `cvt_f16_f32`, so that one widens.
    pub fn __tile_cvt_f16_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_cvt_f32_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_cvt_f16_si8(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Scale a tile by the ratio `num / den`, kept as two operands so the
    /// division happens once at emission rather than per element.
    pub fn __tile_muls_ratio_f32(dst: u32, src: u32, num: u32, den: u32, rows: u32, cols: u32) -> u32;

    /// i32 store. Same translator as the f16 and f32 stores, so it takes the GM
    /// destination first and returns nothing.
    pub fn __tile_store_i32(gm: *mut i32, buf: u32, rows: u32, cols: u32);

    /// Read one element back out of a tile as a scalar, and write one in.
    ///
    /// Declared 2026-11-08, for a reason worth keeping: `mlir_to_cpp` has
    /// implemented these all along and nothing declared them, so
    /// `bench/shapes/check_call_conformance.py` reported a call to one as "not
    /// declared in tile_std" -- which reads as "this operation does not exist".
    /// For the kernel that found it, a decode-shape embedding written as a
    /// runtime-offset row copy, that one undeclared accessor was the whole
    /// difference between an expressible kernel and a dead end. The chain is
    /// load the 1x1 index tile, `get_value` it, multiply by the row width,
    /// `__tile_load_at_f32`, store; `pto-to-ascendc` emits it and the runtime
    /// offset reaches the generated `DataCopy`.
    ///
    /// Only the f32 pair is declared, and the omission is deliberate. The
    /// emitter handles all four dtype variants (`_f32`, `_f16`, `_i32`, `_u32`)
    /// in ONE body that always emits `float` and types the value `float`
    /// downstream -- `float v = (float)buf0.GetValue((int32_t)(0))` even for the
    /// `_u32` name. Declaring the integer forms would put a signature on the
    /// file that the arm does not honour, which is the defect this project keeps
    /// finding rather than a convenience. The index therefore arrives as a float
    /// VALUE, not as `uint32` bits: there is no `__tile_load_i32` anywhere
    /// (compare `__tile_store_i32` directly above, which is declared and
    /// implemented in both emitters), so an index that leaves a kernel as i32
    /// cannot be read back as one.
    ///
    /// `idx` is a FLAT element index into the tile, not a (row, col) pair, and
    /// the emitter casts it to `int32_t` at the call site.
    pub fn __tile_get_value_f32(buf: u32, idx: u32) -> f32;
    pub fn __tile_set_value_f32(buf: u32, idx: u32, value: f32);

    /// Expand e8m0 block exponents into f32 scales.
    pub fn __tile_mxfp4_e8m0_to_f32(scales: *const u8, dst: *mut f32, n: u32) -> u32;

    /// Unpack MXFP4 weights into f16, taking the block scales alongside. The
    /// `_e8m0` form reads the exponent encoding rather than a scale table.
    pub fn __tile_mxfp4_unpack_f16(w: *const u8, d: *const u8, o: *mut u16, ne00: u32, ne0: u32) -> u32;
    pub fn __tile_mxfp4_unpack_f16_e8m0(w: *const u8, d: *const u8, o: *mut u16, ne00: u32, ne0: u32) -> u32;

    /// THE SAME THREE KERNELS WITH THE TILING AS AN OPERAND.
    ///
    /// The three above take their chunk geometry from an environment variable
    /// read by `mlir_to_pto` at emit time — `TILE_E8M0_ROWS_PER_CHUNK` and
    /// `TILE_MXFP4_PAIRS_PER_CHUNK`. That makes the lowering a function of the
    /// environment as well as of this call, and the consequence is checked in:
    /// `crates/mlir_to_pto_tests/sw_u32_p32.pto` and `sw_u32_p64.pto` are the
    /// SAME five-argument `__tile_mxfp4_unpack_f16(w, d, o, 4096, 512)` at two
    /// knob settings, 250 lines of PTO against 134, and `sw_e8_r16/_r32/_r64/
    /// _r128` are four kernels from one `__tile_mxfp4_e8m0_to_f32(s, d,
    /// 262144)`. A Rust source holding only the call does not say which.
    ///
    /// These spellings take it as the last argument, so the source does. They
    /// are what `pto-to-tilers` lifts those artifacts BACK to, and
    /// `pto_to_rust/tests/whole_kernel_roundtrip.rs` re-emits each artifact
    /// through this form with no environment variable set and requires the
    /// result byte for byte. The knob stays the default for the short
    /// spellings: every checked-in artifact came out of those and has to keep
    /// doing so.
    pub fn __tile_mxfp4_e8m0_to_f32_chunk(
        scales: *const u8,
        dst: *mut f32,
        n: u32,
        rows_per_chunk: u32,
    ) -> u32;
    pub fn __tile_mxfp4_unpack_f16_chunk(
        w: *const u8,
        d: *const u8,
        o: *mut u16,
        ne00: u32,
        ne0: u32,
        pairs_per_chunk: u32,
    ) -> u32;
    pub fn __tile_mxfp4_unpack_f16_e8m0_chunk(
        w: *const u8,
        d: *const u8,
        o: *mut u16,
        ne00: u32,
        ne0: u32,
        pairs_per_chunk: u32,
    ) -> u32;

    /// Routed matvec over MXFP4 experts. Note this is NOT the operand order of
    /// the `_pk` form below: this translator binds args[2] as the expert `ids`,
    /// where the packed one binds a scale pointer there. Reading one and copying
    /// it to the other would have put the ids where the scales go.
    pub fn __tile_mul_mv_id_mxfp4_f32(
        src0s: *const u8,
        src1: *const f32,
        ids: *const u8,
        dst: *mut f32,
        ne00: u32,
        ne0: u32,
    ) -> u32;

    /// The packed MXFP4 form: weights, activations, scales, output.
    pub fn __tile_mul_mv_id_mxfp4_pk_f32(
        w: *const u8,
        x: *const f32,
        d: *const f32,
        o: *mut f32,
        ne00: u32,
        ne0: u32,
    ) -> u32;

    /// Decode one MXFP4 codebook entry to f32, over a tile.
    pub fn __tile_mxfp4_value_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Grouped matmul with a fused SwiGLU and an int8-quantised output.
    ///
    /// `w_cat` and `s_cat` are the gate and up weights (and their scales)
    /// pre-concatenated host-side — gate in columns `0..n`, up in `n..2n` — which
    /// is what collapses a three-launch chain into one. Emits the absmax,
    /// quantise and store of an m x n int8 result alongside its f32 scales.
    pub fn __tile_grouped_matmul_swiglu_quant(
        x: *const f32,
        w_cat: *const i8,
        s_cat: *const f32,
        out: *mut i8,
        scale_out: *mut f32,
        m: u32,
        k: u32,
        n: u32,
    ) -> u32;

    /// Blocked q8_0 routed matvec.
    pub fn __tile_mul_mv_id_q8_0_blk_f32(
        w: *const i8,
        x: *const f32,
        d: *const f32,
        o: *mut f32,
        ne00: u32,
        ne0: u32,
    ) -> u32;

    /// RoPE applied to the two halves of each row, with cos and sin supplied as
    /// tables rather than recomputed.
    pub fn __tile_rope_halves_f32(
        x: *const f32,
        cos: *const f32,
        sin: *const f32,
        out: *mut f32,
        rows: u32,
        cols: u32,
    ) -> u32;

    // Row-wise fused blocks in `mlir_to_pto`. Every operand below is a GM
    // pointer, not a tile handle — the translators resolve them as `*_gm` names.
    //
    // ⚠ Each of these arms reads ONE MORE operand than is declared here: a
    // trailing barrier count, taken through `args.get(n).map(..)` so it is
    // optional. An `extern "C"` signature has a single arity and cannot express
    // an optional tail, so what is declared is the base form the emitter's own
    // fixtures use. A kernel needing the barrier operand cannot be written in
    // tile-rs today; that is a property of the operator contract, not of these
    // declarations, and it is the reason this family stayed undeclared.
    pub fn __tile_add_rms_norm_rows_bf16(
        x: *const u16,
        residual: *const u16,
        weight: *const u16,
        out: *mut u16,
        residual_out: *mut u16,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// RMSNorm over rows, without the residual add.
    pub fn __tile_rms_norm_rows_f32(
        x: *const f32,
        weight: *const f32,
        out: *mut f32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// SwiGLU over gate/up rows with an int8 quantised output and its scales.
    pub fn __tile_swiglu_quant_rows(
        gate: *const f32,
        up: *const f32,
        y_out: *mut i8,
        scale_out: *mut f32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// Carve a GM buffer into a partition that cells are then read from.
    pub fn __tile_partition_f32(gm: *const f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_partition_f16(gm: *const u16, rows: u32, cols: u32) -> u32;

    /// The axis-swapped partition: a cell of it reads the transposed footprint.
    pub fn __tile_partition_perm_f32(gm: *const f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_partition_perm_f16(gm: *const u16, rows: u32, cols: u32) -> u32;

    /// Read cell (i, j) of a Tr x Tc partition as a tile.
    pub fn __tile_partition_cell_f32(src: u32, i: u32, j: u32, tr: u32, tc: u32) -> u32;
    pub fn __tile_partition_cell_f16(src: u32, i: u32, j: u32, tr: u32, tc: u32) -> u32;

    /// Write a tile INTO cell (i, j) — the inverse of the read above, and a
    /// store destination rather than a load source. Distinct cells do not alias,
    /// which is what lets several of these run without a runtime check.
    pub fn __tile_partition_cell_mut_f32(src: u32, i: u32, j: u32, tr: u32, tc: u32) -> u32;
    pub fn __tile_partition_cell_mut_f16(src: u32, i: u32, j: u32, tr: u32, tc: u32) -> u32;

    /// f32 -> bf16 narrowing. The f32->bf16 direction; `__tile_cast_bf16_f32`
    /// declared elsewhere is the widening one.
    pub fn __tile_cast_f32_bf16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise 1/sqrt(x).
    pub fn __tile_rsqrt_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise natural logarithm.
    pub fn __tile_log_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise sigmoid: 1/(1+exp(-x)).
    pub fn __tile_sigmoid_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Cast f32 tile to f16 tile.
    pub fn __tile_cast_f32_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Cast f16 tile to f32 tile.
    pub fn __tile_cast_f16_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_cast_f16_i16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_cast_i16_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Cast i8 tile to f32 tile (int8 dequant: `(f32)qs[i]`). The scalar `d`
    /// multiply is a separate `__tile_scale_f32` so this stays a pure widening
    /// cast — the composable primitive a q8_0 matvec needs.
    /// PTO lowering: tmov i8 tile -> f32 tile (Cast pipeline op); MSL: (float)x.
    pub fn __tile_cast_i8_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Unpack the LOW 4-bit nibble of each byte to f32: `out[k] = (f32)(qs[k] & 0xF)`.
    /// The 4-bit dequant primitive for q4_K/q5_K (pairs with `__tile_unpack_q4_hi`).
    /// PTO: bitand-mask + i8->f32 cast; MSL: `float(p0[gid] & 0x0F)`.
    pub fn __tile_unpack_q4_lo(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Unpack the HIGH 4-bit nibble of each byte to f32: `out[k] = (f32)(qs[k] >> 4)`.
    pub fn __tile_unpack_q4_hi(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Unpack q4_K's packed 6-bit scales+mins: 12 bytes -> 8 (sc, mn) pairs as f32.
    /// This is the `q4k_get_scale_min` decode (kmask1/2/3), the last dequant primitive
    /// q4_K needs. `cols` must be 12 (the scales region); the result is a 16-wide f32
    /// tile laid out sc[0..8] then mn[0..8].
    /// PTO: 6-bit bitfield extract on the vector unit; MSL: the packed kmask decode.
    pub fn __tile_unpack_q4k_scale_min(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Extract sub-tile: (R2 × C2) from (R × C) at offset (row_off, col_off).
    pub fn __tile_slice_f32(
        dst: u32,
        src: u32,
        row_off: u32,
        col_off: u32,
        src_rows: u32,
        src_cols: u32,
        dst_rows: u32,
        dst_cols: u32,
    ) -> u32;

    /// Concatenate along columns: (R × C1) ++ (R × C2) → (R × C1+C2).
    pub fn __tile_concat_f32(dst: u32, a: u32, b: u32, rows: u32, cols_a: u32, cols_b: u32) -> u32;

    /// Indexed scatter: scatter src (N,D) rows into dst (M,D) at positions.
    pub fn __tile_scatter_f32(
        dst: u32,
        src: u32,
        indices: *const u32,
        n: u32,
        m: u32,
        d: u32,
    ) -> u32;

    /// Indexed gather: gather from src (M,D) at positions → out (N,D).
    pub fn __tile_gather_f32(
        dst: u32,
        src: u32,
        indices: *const u32,
        n: u32,
        m: u32,
        d: u32,
    ) -> u32;

    /// Top-K selection per row. Returns (R, K) values, writes indices to indices_out.
    pub fn __tile_topk_f32(
        dst: u32,
        src: u32,
        indices_out: *mut u32,
        rows: u32,
        cols: u32,
        k: u32,
    ) -> u32;

    /// Iota / arithmetic progression: emits `dst[i] = start + i` for `i in 0..valid_col`.
    /// Output is a 1×valid_col i32 tile. Used as the sort-index initializer
    /// for the topk_selector port (composes with `__tile_sort32_f32`).
    /// Lowers to `pto.tci` on Ascend.
    pub fn __tile_arith_progression_i32(dst: u32, start: u32, valid_col: u32) -> u32;

    /// Sort-buffer init: re-pad tile to next BLOCK boundary with sentinel.
    /// Output tile has same logical shape as src but is sentinel-padded
    /// for safe consumption by `__tile_sort32_f32`.
    /// Lowers to `pto.tfillpad` on Ascend.
    pub fn __tile_init_sort_buf_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// 32-way bitonic sort over a 1×N f32 tile with paired ui32 indices.
    /// Output is a 1×(2*N) f32 tile of interleaved [value, idx] pairs
    /// (FLOAT_DST_STRIDE_COEF=2). Cols must be a positive multiple of 64.
    /// Lowers to `pto.tsort32` on Ascend.
    pub fn __tile_sort32_f32(dst: u32, values: u32, indices: u32, rows: u32, cols: u32) -> u32;

    /// 2-way bitonic merge: merges two 1×N sorted f32 tiles into a
    /// 1×(2N) sorted tile. `tmp` is a 1×(2N) scratch tile.
    /// Lowers to `pto.tmrgsort` (2-way form) on Ascend.
    pub fn __tile_mrgsort2_f32(dst: u32, src0: u32, src1: u32, tmp: u32, cols_each: u32) -> u32;

    /// Mask-pattern gather (lane select). 4-bit mask:
    ///   `0b1010 = 10` → P1010 (selects even lanes — value channel of
    ///   sort_result interleaved [val,idx] pairs).
    /// Lowers to `pto.tgather` (mask-pattern form) on Ascend.
    pub fn __tile_extract_f32(dst: u32, src: u32, row: u32, col: u32, rows: u32, cols: u32) -> u32;
    pub fn __tile_gather_mask_f32(
        dst: u32,
        src: u32,
        mask_pattern: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// f16 matrix multiply: C = A(M×K) @ B(K×N) → C(M×N).
    pub fn __tile_matmul_f16(dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32) -> u32;

    /// int8 matmul with i32 accumulator and per-column f32 dequant → f16 GM.
    ///
    /// C_f16 = dequant(A(M×K, i8) @ B(K×N, i8) as i32, scale_per_col f32).
    ///
    /// PTO lowering (validated 2026-04-16, see
    /// memory/project_pto_i8_tmatmul_validated.md):
    ///   - L0A/L0B as i8 tiles, L0C as i32 acc
    ///   - `pto.tmatmul` / `pto.tmatmul.acc` dtype triple (i32, i8, i8)
    ///   - Scale tile allocated in `loc=scaling` (__fbuf__) f32
    ///   - `pto.tstore_fp ins(%acc, %scale) outs(%gm_f16)` — FixPipe folds
    ///     the per-column dequant into the L0C→GM DMA.
    ///
    /// Host must pre-quantize B offline to i8 with per-column f32 scales
    /// (absmax or percentile). A is per-token-quantized at runtime (cheap
    /// for M=1 decode).
    pub fn __tile_matmul_i8_acc_i32_dequant_f16(
        dst: u32,
        a: u32,
        b: u32,
        scale: *const f32,
        m: u32,
        k: u32,
        n: u32,
    ) -> u32;

    /// Load a `ROWS × COLS` i8 tile from global memory.
    ///
    /// PTO lowering: tload into a `loc=mat, dtype=i8` tile → tmov into `loc=left`
    /// (A) or `loc=right` (B) at matmul time.
    pub fn __tile_load_i8(gm: *const i8, rows: u32, cols: u32) -> u32;

    /// Store an i8 tile buffer to global memory.
    pub fn __tile_store_i8(gm: *mut i8, buf: u32, rows: u32, cols: u32);

    /// Load a `ROWS × COLS` SIGNED-i8 tile from global memory.
    ///
    /// `si8` is the dtype PTO names for the signed 8-bit planes (the
    /// `proj_*`/`kda_*` glm kernels use them); the i8 family above is the same
    /// bit-width, so the lift maps `si8` onto these aliases.
    pub fn __tile_load_si8(gm: *const i8, rows: u32, cols: u32) -> u32;

    /// Store a SIGNED-i8 tile buffer to global memory.
    pub fn __tile_store_si8(gm: *mut i8, buf: u32, rows: u32, cols: u32);

    /// Matmul with transposed RHS: C[M,N] = A[M,K] @ B[N,K]^T, f32.
    /// B is stored in natural weight layout (out=N, in=K); the kernel
    /// transposes-on-the-fly. PTO path (mlir_to_pto) routes this to
    /// `pto.tmatmul` with a DN→ZN tload on a transposed tensor_view —
    /// avoiding the a5-only VEC→MAT tmov path.
    pub fn __tile_matmul_transposed_f32(dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32) -> u32;

    /// f16 variant of matmul_transposed.
    pub fn __tile_matmul_transposed_f16(dst: u32, a: u32, b: u32, m: u32, k: u32, n: u32) -> u32;

    // ── Phase 3: Flash Attention primitives ──────────────────────────

    /// Fill tile with scalar value.
    pub fn __tile_fill_f32(dst: u32, scalar: f32, rows: u32, cols: u32) -> u32;

    /// Element-wise max of two tiles.
    pub fn __tile_max_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise min of two tiles.
    pub fn __tile_min_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// This program's coordinate on the launch grid, along `axis`.
    pub fn __tile_program_id(axis: u32) -> u32;

    /// Scalar integer multiply/add over values that may only be known at
    /// runtime — a grid coordinate and the constants that turn it into an
    /// offset.
    pub fn __tile_scalar_mul(a: u32, b: u32) -> u32;
    pub fn __tile_scalar_add(a: u32, b: u32) -> u32;
    /// Integer division and remainder on grid coordinates — `h / GROUP` and
    /// `bh % H` are in every grouped-head kernel.
    pub fn __tile_scalar_div(a: u32, b: u32) -> u32;
    pub fn __tile_scalar_mod(a: u32, b: u32) -> u32;

    /// Advance a kernel pointer by `elems` elements.
    pub fn __tile_offset_ptr_f32(gm: *const f32, elems: u32) -> *const f32;

    /// Store a `rows x cols` f32 tile into a view whose rows are `row_stride`
    /// elements apart.
    pub fn __tile_store_strided_f32(
        gm: *mut f32,
        tile: u32,
        rows: u32,
        cols: u32,
        row_stride: u32,
    );

    /// Load a `rows x cols` f32 tile whose rows are `row_stride` elements
    /// apart, rather than `cols` apart as `__tile_load_f32` assumes.
    pub fn __tile_load_strided_f32(
        gm: *const f32,
        rows: u32,
        cols: u32,
        row_stride: u32,
    ) -> u32;

    /// Element-wise absolute value.
    pub fn __tile_abs_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise square root.
    pub fn __tile_sqrt_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    // __tile_sub_f32, __tile_reduce_max_f32, __tile_div_f32
    // are declared earlier in this same extern block.

    /// RMS normalization: x * rsqrt(mean(x^2) + eps).
    pub fn __tile_rms_norm_f32(dst: u32, src: u32, eps: f32, rows: u32, cols: u32) -> u32;

    // ── Phase 4: Quantization primitives ────────────────────────────

    /// Symmetric INT8 quantize: round(x / scale), clamped to [-128, 127].
    pub fn __tile_quantize_f32_i8(dst: u32, src: u32, scale: f32, rows: u32, cols: u32) -> u32;

    /// The same device entry point with the scale as a TILE handle.
    ///
    /// `__tile_quantize_f32_i8`'s third operand is read two ways in this tree
    /// and they are different programs. The declaration above says `f32`
    /// (one scalar); the corpus `.tile` calls
    /// `__tile_quantize_f32_i8(x, x, %sc, r, cl)` where `%sc` is the result
    /// of `__tile_load_f32(%arg1, 1, 1)` -- a `1x1` tile, replicated down the
    /// rows and DIVIDED elementwise by `mlir_to_pto`. A scalar `Muls` by a
    /// reciprocal would be the other reading and would not compute this.
    ///
    /// The `#[link_name]` is the point, not decoration: the device symbol is
    /// the one the `.tile` already names, and this declaration only states the
    /// calling convention that call site actually uses (an `i32` tile handle in
    /// the third slot). Declaring it under a fresh symbol would invent an
    /// intrinsic the toolchain does not have; the scalar declaration above is
    /// kept for callers of the scalar form.
    #[link_name = "__tile_quantize_f32_i8"]
    pub fn quantize_f32_i8_tile(
        dst: u32,
        src: u32,
        scale: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// INT8 dequantize: x_i8 * scale → f32.
    pub fn __tile_dequantize_i8_f32(dst: u32, src: u32, scale: f32, rows: u32, cols: u32) -> u32;

    /// Row-wise absmax: max(|x|) per row → (R, 1).
    pub fn __tile_absmax_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    // ── Phase 6: Multi-token prediction / speculative decoding ─────

    /// Row-wise argmax: returns index of maximum value per row → (R, 1) u32 indices.
    pub fn __tile_argmax_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise argmin: index of the minimum value per row → (R, 1) u32 indices.
    pub fn __tile_argmin_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Fused grouped-query attention: `softmax(Q @ Kᵀ) @ V` in one op.
    ///
    /// `q` is `(nh * seq, dim)` and `k`/`v` are `(nkv * seq, dim)` — the query
    /// heads share `nkv` key/value heads, and the sequence axis is folded into
    /// the row count rather than dropped. The result is `(seq, dim)` for the
    /// head being computed.
    ///
    /// There is NO causal operand: the form that carried one was the deleted
    /// declaration noted below, and no backend receives it. Compose
    /// `tile_causal_mask_f32` on the scores for a masked attention.
    ///
    /// This is the fused form: whole-operator, no visible stages. On a 910B it
    /// overflows the 192 KB unified buffer, which is why the decomposed and
    /// composed forms exist.
    // A second `__tile_attention_gqa_f32` was declared here with a different
    // operand order — (q, k, v, seq, dim, nh, nkv, causal), no dst. Two
    // declarations of one symbol do not compile, and the surviving one is the
    // (dst, q, k, v, heads_q, heads_kv, seq_len, head_dim) form above, which
    // carries the record of being the order three backends implement.

    /// Top-p (nucleus) sampling: temperature-scaled softmax → cumsum → threshold at p.
    /// Returns sampled token indices per row → (R, 1) u32.
    /// `rng_seed` seeds the per-row uniform random draw.
    pub fn __tile_sample_top_p_f32(
        dst: u32,
        logits: u32,
        temperature: f32,
        top_p: f32,
        rng_seed: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// Draft verification: compare draft token indices against target logits.
    /// Returns per-position acceptance probabilities → (R, 1) f32.
    /// `draft_tokens`: (R, 1) u32 indices from draft model.
    /// `target_logits`: (R, COLS) f32 logits from target model.
    pub fn __tile_draft_verify_f32(
        dst: u32,
        draft_tokens: u32,
        target_logits: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// Token acceptance: apply acceptance mask to select between draft and target tokens.
    /// Where mask[i] >= threshold, keep draft_tokens[i]; else resample from target.
    /// Returns final token indices → (R, 1) u32.
    pub fn __tile_token_accept_f32(
        dst: u32,
        draft_tokens: u32,
        target_tokens: u32,
        accept_probs: u32,
        threshold: f32,
        rows: u32,
    ) -> u32;

    // ── Decode-optimized intrinsics (cooperative reduction) ──────────

    /// Cooperative matvec with f16 weights, f32 accumulation.
    /// One threadgroup per output row, simd_sum + shared mem reduction.
    /// a=activation(f32, K), b=weight(f16, N×K) → c=output(f32, N).
    pub fn __tile_matvec_f16(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// f32-weight matvec — out[n] = sum_k a[k] * w[n,k].
    /// a=activation(f32, K), b=weight(f32, N×K, row-major by N) → c=output(f32, N).
    /// rows=N (output width), cols=K (inner/contraction dim).
    /// Used by projections when weights haven't been cast to f16/bf16.
    pub fn __tile_matvec_f32(dst: u32, a: u32, b: u32, rows: u32, cols: u32) -> u32;

    /// DS4 engine projection, w8a8: `y[i] = sum_k (W8[i,k] - off[i]) * sc[i] * a[k]`.
    ///
    /// `w` is the int8 weight's GM POINTER and is never staged whole -- the row loop
    /// copies one row at a time, which is what lets a 4096-wide projection run inside the
    /// 188 KB buffer. `sc`/`off` are per-OUTPUT-CHANNEL f32 pointers into GM.
    pub fn __tile_matvec_w8a8_f32(
        w: *const i8,
        sc: *const f32,
        off: *const f32,
        a: u32,
        dst: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// THE ENGINE'S `ds4_q8mv_*` SOCKET: `out[r] = sum_k w8[r,k] * x[k] * s[r,k]`.
    ///
    /// `s` is the q8_0 block scales PRE-EXPANDED into a full f32 plane (layer0.pto:3 comments
    /// "per-element scale sd[r][i]=d[r][i/32]"), so the kernel does no block arithmetic. Four
    /// pointers in the engine's own order -- w, s, x, o -- which is why a kernel written to this
    /// signature can carry the mangled `_Z18ds4_q8mv_*` names the engine binds.
    pub fn __tile_matvec_q8mv_f32(
        w: *const i8,
        s: *const f32,
        x: *const f32,
        dst: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// CHUNKED w8a8 matvec: both the activation and the weight come from GM, one K-chunk at
    /// a time, so k may exceed the unified buffer. Same arithmetic as
    /// `__tile_matvec_w8a8_f32`, which needs eight k-wide buffers and refuses k=8192.
    pub fn __tile_matvec_w8a8_gm_f32(
        x: *const f32,
        w: *const i8,
        sc: *const f32,
        off: *const f32,
        dst: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// Streaming f32 matvec: the weight's GM POINTER, one row resident at a time.
    /// The f32 sibling of `__tile_matvec_w8a8_f32` and the first half of the q8 path.
    pub fn __tile_matvec_gm_f32(
        w: *const f32,
        a: u32,
        dst: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// Q8_0 streaming matvec: 34-byte blocks (half scale + 32 int8), read from GM.
    pub fn __tile_matvec_q8_0_gm_f32(
        w: *const u8,
        a: u32,
        dst: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// Quantised-weight matvecs — the decode dense projections. The weight is a
    /// raw GGUF-layout block buffer (Q8_0 / Q2_K / IQ2_XXS), not a tile buf_id,
    /// so it is passed as a raw device pointer alongside the f32 activation and
    /// the runtime dims. rows = output width, cols = contraction dim.
    pub fn __tile_matvec_q8_0_f32(w: *const i8, x: *const f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_matvec_q2_k_f32(w: *const i8, x: *const f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_matvec_iq2_xxs_f32(w: *const i8, x: *const f32, rows: u32, cols: u32) -> u32;

    // f16-activation twins of the three quantised matvecs above. Each shares its
    // f32 arm in `mlir_to_gpu` — one match arm names both spellings — so the
    // signature follows from the arm, not from the name. The arm refuses under 4
    // args and reads `args[2]` as the row count; the suffix selects the
    // activation dtype, so only `x` changes, to the `*const u16` this crate uses
    // for every f16 pointer.

    /// Cooperative matvec with f16 weights and bias addition.
    pub fn __tile_matvec_f16_bias(dst: u32, a: u32, b: u32, bias: u32, rows: u32, cols: u32)
        -> u32;

    /// In-place RoPE for decode: applies rotary embedding to a single token.
    /// x is (num_heads × head_dim), modified in-place.
    pub fn __tile_rope_inplace_f32(dst: u32, x: u32, rows: u32, cols: u32) -> u32;

    /// KV cache update: copy vector (num_heads × head_dim) into cache at position.
    /// Cache layout: (num_heads, max_seq, head_dim).
    pub fn __tile_kv_cache_update_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// 1-query decode attention with K/V streamed from GM in blocks.
    ///
    /// `q` is the loaded [1 x D] tile; `k`/`v` are raw GM pointers to
    /// [S x D]. The kernel never materialises the full K/V, which is what lets
    /// a context-scale S (GLM 5.3's sourced index_topk = 2048) fit the unified
    /// buffer. The sink-free sibling of `__tile_attention_sink_f32`.
    pub fn __tile_attention_blocked_f32(
        dst: u32,
        q: u32,
        k: *const f32,
        v: *const f32,
        s: u32,
        d: u32,
    ) -> u32;

    /// 1-query decode attention at a RUNTIME context length.
    ///
    /// `s` is a loaded 1 x 1 tile holding the KV length for this launch; `k`
    /// and `v` are raw GM pointers. The cache length of a decode step is a
    /// runtime value, not a shape, so this is the form a real decoder needs:
    /// the same kernel serves S=64 and S=1048576. K/V are streamed in blocks.
    pub fn __tile_attention_dyn_f32(
        dst: u32,
        q: u32,
        k: *const f32,
        v: *const f32,
        s: u32,
        d: u32,
    ) -> u32;

    /// Per-row argmax over a `rows x cols` f32 GM plane, writing one i32 index
    /// per row straight to GM. Streams the plane row by row, so a row that does
    /// not fit the unified buffer is fine; the first index attaining the row
    /// maximum wins (the tie policy ORACLE 41/58 asserts).
    pub fn __tile_argmax_rows_i32(
        out: *mut i32,
        src: *const f32,
        rows: u32,
        cols: u32,
    );

    /// Decode attention with cooperative threadgroup reduction.



    /// One threadgroup per head, supports GQA.
    /// 5-phase: score → max → exp+sum → normalize → weighted V sum.
    pub fn __tile_attention_decode_f32(
        dst: u32,
        q: u32,
        k: u32,
        v: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    // --- Prefill-path intrinsics (batched multi-position) ---

    /// Batched RoPE for prefill: applies RoPE to (seq_len, num_heads, head_dim) in-place.
    pub fn __tile_rope_prefill_f32(dst: u32, x: u32, rows: u32, cols: u32) -> u32;

    /// Batched KV cache update for prefill: copies seq_len vectors into cache.
    pub fn __tile_kv_cache_update_prefill_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Prefill attention: Q is (seq_len, num_heads, head_dim), K/V from cache.
    /// One threadgroup per (query_head, query_position) pair. Causal + GQA.
    pub fn __tile_attention_prefill_f32(
        dst: u32,
        q: u32,
        k: u32,
        v: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    // --- Fused decode intrinsics ---

    /// Cooperative matvec + residual add: out = A @ B^T + R.
    pub fn __tile_matvec_f16_add(dst: u32, a: u32, b: u32, r: u32, rows: u32, cols: u32) -> u32;

    /// Fused gate+up+silu: reads activation once, computes silu(A@W_gate) * (A@W_up).
    pub fn __tile_gate_up_silu_f16(
        dst: u32,
        a: u32,
        w_gate: u32,
        w_up: u32,
        rows: u32,
        cols: u32,
    ) -> u32;

    /// f32 twin of the above. `mlir_to_cpp` dispatches both spellings from one
    /// arm, so the operand order is identical by construction rather than by
    /// resemblance: `args[1..=3]` are the activation and the two weight tiles,
    /// `args[4..=5]` the extents, and the guard refuses under 6.
    pub fn __tile_gate_up_silu_f32(
        dst: u32,
        a: u32,
        w_gate: u32,
        w_up: u32,
        rows: u32,
        cols: u32,
    ) -> u32;
}

// --- f16 tile operations --------------------------------------------------

extern "C" {
    /// Load a `ROWS × COLS` f16 tile from global memory (stored as u16).
    pub fn __tile_load_f16(gm: *const u16, rows: u32, cols: u32) -> u32;

    /// Store a local f16 tile buffer to global memory (as u16).
    pub fn __tile_store_f16(gm: *mut u16, buf: u32, rows: u32, cols: u32);

    /// Element-wise add two f16 tiles.
    pub fn __tile_add_f16(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// bf16 -> f32 widening: `(dst, src, rows, cols)`, taken from the
    /// mlir_to_gpu arm, which documents that order and refuses anything under
    /// four args.
    pub fn __tile_cast_bf16_f32(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Batched attention with sinks. Operand order is the emitter's own banner
    /// in `mlir_to_gpu`, confirmed by its arity guard (refuses under 7) and by
    /// which `args[..]` the arm reads (0..=6). Note there is no leading `dst`:
    /// the output is the fourth operand.
    pub fn __tile_attention_sink_batched_f32(
        q: u32,
        kv: u32,
        sinks: u32,
        out: u32,
        n_head: u32,
        s: u32,
        d: u32,
    ) -> u32;

    /// The kq-split form of the same, with the extra `kq` extent. Banner, arity
    /// guard (refuses under 8) and operand reads (0..=7) all agree.
    pub fn __tile_attention_sink_batched_kq_f32(
        q: u32,
        kv: u32,
        sinks: u32,
        out: u32,
        n_head: u32,
        kq: u32,
        s: u32,
        d: u32,
    ) -> u32;

    /// Single-head attention with a sink logit: `__tile_attention_f32` plus a
    /// `sinks` operand in the fifth slot. Order is the `mlir_to_pto` arm's own
    /// comment — `(dst, q, k, v, sinks, S, D)` — confirmed by its arity guard
    /// (refuses under 7) and by the operands it reads: `args[1..=3]` as the Q/K/V
    /// tiles, `args[4]` as a pointer, `args[5..=6]` as the extents.
    ///
    /// `sinks` is a raw pointer, not a tile: the arm resolves it with
    /// `resolve_ptr`, and it addresses at least one S-block of floats all equal
    /// to this head's sink logit. The broadcast is the caller's job because the
    /// widening op fills rows, not columns.
    pub fn __tile_attention_sink_f32(
        dst: u32,
        q: u32,
        k: u32,
        v: u32,
        sinks: u32,
        seq_len: u32,
        head_dim: u32,
    ) -> u32;

    /// One sequence shard of attention, returning the un-normalised numerator
    /// alongside the running max and denominator so shards can be merged
    /// exactly by log-sum-exp. Three outputs, in slots 2, 3 and 4 — the arm's
    /// banner names `(q, kv, out_o, out_m, out_d, n_head, S, D)`, its guard
    /// refuses under 8, and it reads `args[0..=7]` in that order.
    ///
    /// Sink-free by contract: a sink term is not distributive over the merge, so
    /// the sharded form and `__tile_attention_sink_f32` are alternatives rather
    /// than a pair. An empty shard is the merge's identity and never launches,
    /// which is why the arm refuses a zero extent instead of emitting a no-op.
    pub fn __tile_attention_partial_batched_f32(
        q: u32,
        kv: u32,
        out_o: u32,
        out_m: u32,
        out_d: u32,
        n_head: u32,
        s: u32,
        d: u32,
    ) -> u32;

    // f16 twins of ops the backends already lower. Each shares its f32 arm in
    // mlir_to_gpu — the arm names both spellings — so the signature is the same
    // by construction rather than by resemblance. Undeclared, they were kernels
    // a backend could emit and no tile-rs source could call.
    pub fn __tile_fill_f16(dst: u32, scalar: f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_rms_norm_f16(dst: u32, src: u32, eps: f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_scale_f16(dst: u32, src: u32, scalar: f32, rows: u32, cols: u32) -> u32;
    pub fn __tile_transpose_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;

    /// Element-wise multiply two f16 tiles.
    pub fn __tile_mul_f16(dst: u32, src1: u32, src2: u32, rows: u32, cols: u32) -> u32;

    /// Row-wise softmax of an f16 tile.
    pub fn __tile_softmax_f16(dst: u32, src: u32, rows: u32, cols: u32) -> u32;
}

// --- Pipelining markers ---------------------------------------------------
//
// A `pipelined_for` loop advises the MLIR backend that the following loop
// body is a K-dim / tile-iteration loop where prologue-overlapped DMA and
// compute should be emitted. The marker is a zero-cost tag pair
// (`_begin(depth) ... _end()`) wrapped around the Rust `for` body. The cpp
// emitter consumes it to force-on (depth >= 2) or force-off (depth == 1)
// the existing double-buffer detection in `detect_tiling_loop`. The PTO
// backend ignores it (ptoas auto-inserts sync); GPU/MSL/NKI/Pallas emitters
// ignore it today but can honor the depth hint when they grow native
// scf.for handling.
//
// `depth` is the pipeline depth (2 = double-buffered, 3 = triple, etc.).
// `depth == 1` is an explicit opt-out of pipelining on this loop.

extern "C" {
    #[doc(hidden)]
    pub fn __tile_pipelined_for_begin(depth: u32);
    #[doc(hidden)]
    pub fn __tile_pipelined_for_end();
}

/// Mark the following tile-iteration loop as software-pipelined with the
/// given depth.
///
/// Place the marker *before* the loop — the cpp emitter's
/// `detect_tiling_loop` scans from the loop header through the latch back to
/// the header, so the marker only needs to be reachable from the header
/// block.
///
/// Usage (from `examples/tile_softmax_double_buf/kernels/src/lib.rs`):
/// ```ignore
/// pipelined_for(2);                       // depth=2 = double-buffered
/// let mut i = 0u32;
/// loop {
///     if i >= n_tiles { break; }
///     let offset = (i as usize) * COLS;
///     let t = tile_load_f32::<1, COLS>(input.wrapping_add(offset));
///     let y = safe::tile_softmax_f32(t);
///     tile_store_f32::<1, COLS>(output.wrapping_add(offset), y);
///     i = i + 1u32;
/// }
/// ```
///
/// The cpp backend pairs the `_begin` with the loop back-edge and forces the
/// tiling-loop heuristic on (for `depth >= 2`) or off (for `depth == 1`).
/// The PTO backend drops the marker entirely — `ptoas --enable-insert-sync`
/// derives the cross-pipe sync set from the tile-op DAG and produces the
/// same double-buffer overlap without the hint.
/// All other vendor emitters (GPU/NKI/MSL/SPIRV/BANG/Gaudi/AIE/TPU) also
/// drop the marker; they can grow native `scf.for` pipeline-hint support
/// later without breaking existing kernels.
#[inline(always)]
pub fn pipelined_for(depth: u32) {
    unsafe { __tile_pipelined_for_begin(depth) }
}

// --- Safe wrappers --------------------------------------------------------
//
// These wrappers encode const generic dimensions at the call site and
// pass them as runtime u32 values to the underlying extern "C" intrinsics.
// The codegen backend can inspect the call arguments to reconstruct the
// tile shape when emitting PTO assembly.

/// Load a `ROWS × COLS` f32 tile from global memory.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid f32 values.
#[inline(always)]
pub fn tile_load_f32<const ROWS: usize, const COLS: usize>(
    gm: *const f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_load_f32(gm, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}
/// A tile buffer whose contents are REPLACED rather than consumed.
///
/// `Tile` is a value: every op that writes one yields a new handle, which is what
/// makes a lifted kernel's dataflow explicit. That breaks for a chunked body,
/// where one buffer is deliberately refilled because the whole extent does not
/// fit. As several `Tile`s it overflows the unified buffer; as one `Tile` written
/// twice it is one name for several values, which the lift refuses.
///
/// `TileBuf` is the third option: one buffer, several successive contents, with
/// the refill explicit in the type (`&mut`) instead of implied by repetition.
pub struct TileBuf<const ROWS: usize, const COLS: usize, T> {
    buf_id: u32,
    _phantom: PhantomData<T>,
}

impl<const ROWS: usize, const COLS: usize, T> TileBuf<ROWS, COLS, T> {
    /// A `Tile` view of what the buffer holds NOW.
    ///
    /// Takes `&self`, so a read cannot be mistaken for a refill, and the borrow
    /// checker stops a view outliving the next `tile_load_into_*` -- which is the
    /// one-name-two-values hazard, enforced by the type here rather than by a
    /// lift refusal.
    pub fn get(&self) -> Tile<ROWS, COLS, T> {
        Tile { buf_id: self.buf_id, _phantom: PhantomData }
    }
}

/// An f32 buffer of `ROWS x COLS`, allocated by its first load.
pub fn tile_buf_f32<const ROWS: usize, const COLS: usize>() -> TileBuf<ROWS, COLS, f32> {
    // No separate alloc intrinsic: `tile_load_f32` already allocates on load, and
    // inventing an alloc name neither backend emits would be a surface with no
    // lowering behind it.
    TileBuf { buf_id: u32::MAX, _phantom: PhantomData }
}

/// Refill `buf` from global memory.
pub fn tile_load_into_f32<const ROWS: usize, const COLS: usize>(
    buf: &mut TileBuf<ROWS, COLS, f32>,
    gm: *const f32,
) {
    if buf.buf_id == u32::MAX {
        buf.buf_id = unsafe { __tile_load_f32(gm, ROWS as u32, COLS as u32) };
    } else {
        unsafe { __tile_load_into_f32(buf.buf_id, gm, ROWS as u32, COLS as u32) }
    }
}


/// This program's coordinate on the launch grid, along `axis`.
///
/// A tile-rs kernel had no way to say which part of the problem it was for, so
/// everything written against the tile API was a single block however much
/// parallelism the operator had. With this, a kernel maps itself onto the grid
/// the way a Triton kernel does.
#[inline(always)]
pub fn tile_program_id(axis: u32) -> u32 {
    unsafe { __tile_program_id(axis) }
}

/// Scalar multiply, over grid coordinates as well as constants.
#[inline(always)]
pub fn tile_scalar_mul(a: u32, b: u32) -> u32 {
    unsafe { __tile_scalar_mul(a, b) }
}

/// Scalar add, over grid coordinates as well as constants.
#[inline(always)]
pub fn tile_scalar_add(a: u32, b: u32) -> u32 {
    unsafe { __tile_scalar_add(a, b) }
}

/// Integer division on grid coordinates, e.g. `h / GROUP` under GQA.
#[inline(always)]
pub fn tile_scalar_div(a: u32, b: u32) -> u32 {
    unsafe { __tile_scalar_div(a, b) }
}

/// Integer remainder on grid coordinates, e.g. `bh % H`.
#[inline(always)]
pub fn tile_scalar_mod(a: u32, b: u32) -> u32 {
    unsafe { __tile_scalar_mod(a, b) }
}

/// Advance a pointer by `elems` elements, so a block can address its own slice.
///
/// # Safety
/// The result must stay within the allocation.
#[inline(always)]
pub unsafe fn tile_offset_ptr_f32(gm: *const f32, elems: u32) -> *const f32 {
    __tile_offset_ptr_f32(gm, elems)
}

/// WHOLE-KERNEL MXFP4 WRAPPERS, WITH THE TILING IN THE TYPE.
///
/// These three are not tile-level operations and they return no `Tile`: each is
/// one whole kernel over GM pointers, which is what the `__tile_mxfp4_*`
/// intrinsics are. They are here as safe wrappers for one reason — the shapes
/// and the chunk geometry are const generics, so a Rust source names everything
/// that decides what the emitted kernel looks like.
///
/// That is the difference that matters, and it is measurable: the plain
/// five-argument intrinsics take their chunk from `TILE_MXFP4_PAIRS_PER_CHUNK`
/// / `TILE_E8M0_ROWS_PER_CHUNK` at emit time, so the same call lowers to
/// several of the artifacts checked into `crates/mlir_to_pto_tests`. With
/// `PAIRS_PER_CHUNK` a const parameter it lowers to exactly one, which is what
/// makes `pto-to-tilers` able to lift those artifacts back at all.
///
/// `o` is `*mut u16` rather than `*mut f16` because this crate is `#![no_core]`
/// and has no f16 scalar; the slab is f16 BITS and the cube reads it as f16.
///
/// `pub fn` with the `unsafe` INSIDE, which is this layer's convention and not
/// a claim that a raw pointer is safe: `tile_load_f32` takes a `*const f32` the
/// same way. The pointer-validity requirement is the caller's, exactly as
/// `pto_to_rust`'s own docs say of every kernel it emits. Written this way
/// deliberately rather than as `pub unsafe fn` — `tests/emitted_names_resolve.rs`
/// collects wrappers as `pub fn tile_` at column 0, so a differently-shaped
/// signature here would be a name the lift emits and that gate cannot see.
#[inline(always)]
pub fn tile_mxfp4_unpack_f16<const NE00: u32, const NE0: u32, const PAIRS_PER_CHUNK: u32>(
    w: *const u8,
    d: *const f32,
    o: *mut u16,
) -> u32 {
    unsafe { __tile_mxfp4_unpack_f16_chunk(w, d as *const u8, o, NE00, NE0, PAIRS_PER_CHUNK) }
}

/// The same kernel reading MXFP4's NATIVE e8m0 scale BYTES instead of a
/// pre-expanded f32 plane — hence `d: *const u8`, and it is the whole
/// difference between the two: `0.53125` bytes per weight against `0.625`.
#[inline(always)]
pub fn tile_mxfp4_unpack_f16_e8m0<const NE00: u32, const NE0: u32, const PAIRS_PER_CHUNK: u32>(
    w: *const u8,
    d: *const u8,
    o: *mut u16,
) -> u32 {
    unsafe { __tile_mxfp4_unpack_f16_e8m0_chunk(w, d, o, NE00, NE0, PAIRS_PER_CHUNK) }
}

/// Expand an e8m0 scale plane of `N` bytes to `N` halved f32 scales,
/// `ROWS_PER_CHUNK` rows of 32 per loop iteration.
#[inline(always)]
pub fn tile_mxfp4_e8m0_to_f32<const N: u32, const ROWS_PER_CHUNK: u32>(
    scales: *const u8,
    dst: *mut f32,
) -> u32 {
    unsafe { __tile_mxfp4_e8m0_to_f32_chunk(scales, dst, N, ROWS_PER_CHUNK) }
}

/// Load an f32 tile whose rows are `ROW_STRIDE` elements apart.
///
/// `tile_load_f32` assumes a tile is contiguous — that element `(i, j)` lives at
/// `i * COLS + j`. That is true of a whole buffer and false of almost every
/// *view* into one: a head slice of `[B, T, H, K]` has a row stride of `H * K`,
/// and a per-token scalar column of `[B, T, H]` has a row stride of `H` with
/// `COLS = 1`. Without this, such an operator cannot be written in the tile API
/// at all, and the caller has to materialise a contiguous copy first.
///
/// `ROW_STRIDE` is a const generic because the emitter resolves it at compile
/// time to build the index vector; it is measured in elements, not bytes.
///
/// # Safety
/// `gm` must address `(ROWS - 1) * ROW_STRIDE + COLS` readable f32 values.
/// Read one element out of a tile as a plain `f32`.
///
/// The public wrapper over `__tile_get_value_f32`, which has been declared here
/// since 2026-11-08 and implemented in `mlir_to_cpp` for longer still. Nothing
/// exposed it, and `Tile::buf_id` is `pub(crate)`, so a lifted kernel -- which
/// consumes this module through `use tile_std::tile::*` -- could not reach the
/// intrinsic even though both ends of it existed. That is the same shape as the
/// strided-load omission above: the capability was present and the table did
/// not name it.
///
/// `idx` is a FLAT element index into the tile, matching the intrinsic; the
/// emitter casts it to `int32_t` at the call site.
///
/// # Why a lift needs this
/// A PTO tile with logical extent 1x1 (`v_row=1, v_col=1`) is a SCALAR held in
/// a tile, and the fused row-expansions take one as their reduction operand.
/// Without a way to read it out, `pto.trowexpandmul` over a 1x1 operand cannot
/// become the `tile_scale_f32` it actually is. Corpus census 2026-11-09: 38
/// such operands, 318 logical-1x1 tiles in all.
///
/// # Block size
/// Every logical-1x1 tile in the corpus is ALLOCATED 1x8 or 8x1 f32 -- exactly
/// 32 bytes, one block. Load at the allocated extent, not the logical one: a
/// `1x1` f32 load is 4 bytes, which this part copies wrongly without faulting,
/// and `mlir_to_cpp` refuses any extent that is not a multiple of 32.
#[inline(always)]
pub fn tile_get_value_f32<const ROWS: usize, const COLS: usize>(
    t: Tile<ROWS, COLS, f32>,
    idx: u32,
) -> f32 {
    unsafe { __tile_get_value_f32(t.buf_id, idx) }
}

#[inline(always)]
pub fn tile_load_strided_f32<const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>(
    gm: *const f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_load_strided_f32(gm, ROWS as u32, COLS as u32, ROW_STRIDE as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store an f32 tile into a view whose rows are `ROW_STRIDE` elements apart.
///
/// The mirror of `tile_load_strided_f32`: an operator's output is a view as
/// often as its input is.
///
/// # Safety
/// `gm` must address `(ROWS - 1) * ROW_STRIDE + COLS` writable f32 values.
#[inline(always)]
pub fn tile_store_strided_f32<const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>(
    gm: *mut f32,
    tile: Tile<ROWS, COLS, f32>,
) {
    unsafe {
        __tile_store_strided_f32(gm, tile.buf_id, ROWS as u32, COLS as u32, ROW_STRIDE as u32)
    };
}

/// Store an f32 tile to global memory. Consumes the tile handle.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` writable f32 values.
#[inline(always)]
pub fn tile_store_f32<const ROWS: usize, const COLS: usize>(
    gm: *mut f32,
    tile: Tile<ROWS, COLS, f32>,
) {
    unsafe { __tile_store_f32(gm, tile.buf_id, ROWS as u32, COLS as u32) };
}

/// Element-wise add two f32 tiles, producing a new tile.
#[inline(always)]
pub fn tile_add_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_add_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise multiply two f32 tiles, producing a new tile.
#[inline(always)]
pub fn tile_mul_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_mul_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise exp of an f32 tile.
#[inline(always)]
pub fn tile_exp_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_exp_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise softmax of an f32 tile (reduces over columns).
#[inline(always)]
pub fn tile_softmax_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_softmax_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Matrix multiply f32: `(M × K) @ (K × N) → (M × N)`.
#[inline(always)]
pub fn tile_matmul_f32<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, f32>,
    b: Tile<K, N, f32>,
) -> Tile<M, N, f32> {
    let buf_id = unsafe { __tile_matmul_f32(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise subtract two f32 tiles: a - b.
#[inline(always)]
pub fn tile_sub_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_sub_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise divide two f32 tiles: a / b.
#[inline(always)]
pub fn tile_div_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_div_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise negate an f32 tile.
#[inline(always)]
pub fn tile_neg_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_neg_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise max reduction. Returns (ROWS × 1) tile.
#[inline(always)]
pub fn tile_reduce_max_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_reduce_max_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise sum reduction. Returns (ROWS × 1) tile.
#[inline(always)]
pub fn tile_reduce_sum_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_reduce_sum_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Scale all elements of a tile by a scalar.
#[inline(always)]
/// Broadcast a `1 x 1` tile over `n` columns.
#[inline(always)]
/// Read one element of a tile as a scalar.
#[inline(always)]
pub fn tile_getval_f32<const R: usize, const C: usize>(src: Tile<R, C, f32>, idx: u32) -> f32 {
    unsafe { __tile_getval_f32(src.buf_id, idx) }
}

pub fn tile_row_expand_f32<const N: usize>(src: Tile<1, 1, f32>) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_row_expand_f32(0, src.buf_id, N as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// 2D row-wise divide with a PADDED divisor tile: only column 0 of `red` is
/// valid (`v_col=1` in the PTO); the remaining columns are storage padding.
#[inline(always)]
pub fn tile_row_expand_div_bcast_f32<const R: usize, const C: usize, const C2: usize>(
    src: Tile<R, C, f32>,
    red: Tile<R, C2, f32>,
) -> Tile<R, C, f32> {
    let buf_id = unsafe { __tile_row_expand_div_f32(0, src.buf_id, red.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// 2D row-wise subtract with a PADDED divisor tile (column 0 valid).
#[inline(always)]
pub fn tile_row_expand_sub_bcast_f32<const R: usize, const C: usize, const C2: usize>(
    src: Tile<R, C, f32>,
    red: Tile<R, C2, f32>,
) -> Tile<R, C, f32> {
    let buf_id = unsafe { __tile_row_expand_sub_f32(0, src.buf_id, red.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// 2D row-wise multiply with a PADDED divisor tile (column 0 valid).
#[inline(always)]
pub fn tile_row_expand_mul_bcast_f32<const R: usize, const C: usize, const C2: usize>(
    src: Tile<R, C, f32>,
    red: Tile<R, C2, f32>,
) -> Tile<R, C, f32> {
    let buf_id = unsafe { __tile_row_expand_mul_f32(0, src.buf_id, red.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// 2D row-wise divide: `dst[r,c] = src[r,c] / red[r]`.
#[inline(always)]
pub fn tile_row_expand_div_f32<const R: usize, const C: usize>(
    src: Tile<R, C, f32>,
    red: Tile<R, 1, f32>,
) -> Tile<R, C, f32> {
    let buf_id = unsafe { __tile_row_expand_div_f32(0, src.buf_id, red.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// 2D row-wise subtract: `dst[r,c] = src[r,c] - red[r]`.
#[inline(always)]
pub fn tile_row_expand_sub_f32<const R: usize, const C: usize>(
    src: Tile<R, C, f32>,
    red: Tile<R, 1, f32>,
) -> Tile<R, C, f32> {
    let buf_id = unsafe { __tile_row_expand_sub_f32(0, src.buf_id, red.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// 2D row-wise multiply: `dst[r,c] = src[r,c] * red[r]`.
#[inline(always)]
pub fn tile_row_expand_mul_f32<const R: usize, const C: usize>(
    src: Tile<R, C, f32>,
    red: Tile<R, 1, f32>,
) -> Tile<R, C, f32> {
    let buf_id = unsafe { __tile_row_expand_mul_f32(0, src.buf_id, red.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Subtract a 1 x G vector from a 1 x N row over G contiguous blocks.
#[inline(always)]
pub fn tile_block_expand_sub_f32<const N: usize, const G: usize>(
    src: Tile<1, N, f32>,
    vec: Tile<1, G, f32>,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_block_expand_sub_f32(0, src.buf_id, vec.buf_id, N as u32, (N / G) as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Multiply a 1 x N row by a 1 x G vector over G contiguous blocks.
#[inline(always)]
pub fn tile_block_expand_mul_f32<const N: usize, const G: usize>(
    src: Tile<1, N, f32>,
    vec: Tile<1, G, f32>,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_block_expand_mul_f32(0, src.buf_id, vec.buf_id, N as u32, (N / G) as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Divide a 1 x N row by a 1 x G vector over G contiguous blocks.
#[inline(always)]
pub fn tile_block_expand_div_f32<const N: usize, const G: usize>(
    src: Tile<1, N, f32>,
    vec: Tile<1, G, f32>,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_block_expand_div_f32(0, src.buf_id, vec.buf_id, N as u32, (N / G) as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Add one scalar to every element of a tile.
#[inline(always)]
pub fn tile_adds_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scalar: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_adds_f32_u32(0, src.buf_id, scalar, (ROWS * COLS) as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Subtract one scalar from every element of a tile.
#[inline(always)]
pub fn tile_subs_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scalar: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_subs_f32(0, src.buf_id, scalar, (ROWS * COLS) as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

pub fn tile_scale_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scalar: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_scale_f32(0, src.buf_id, scalar, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// A tile-to-tile copy, by reference.
///
/// The tile API has no copy at all: `Tile` is deliberately not `Copy`, so a
/// value can be consumed exactly once, and a kernel that reads a tile twice
/// cannot be expressed without duplicating it. That is not a defect in the
/// kernels -- `swiglu` reads its input in both the gate and the up projection --
/// so a duplicate is needed and this is it.
///
/// It is its OWN intrinsic, `__tile_copy_f32`, which `mlir_to_pto` lowers to
/// `pto.tmov`. Multiplying by one is numerically a copy for f32, but it is not a
/// move: it runs on the arithmetic path, `mlir_to_pto` re-emits it as
/// `pto.tmuls ... 1.0`, and the lift cannot round-trip a `pto.tmov` through it.
/// Declared 2026-11-10 for that round trip.
///
/// Takes `&Tile` rather than `Tile` so the source is unambiguously preserved;
/// `Tile` is `Copy` today, so a by-value form would also work, but the reference
/// implementation mirrors this signature.
#[inline(always)]
pub fn tile_copy_f32<const ROWS: usize, const COLS: usize>(
    src: &Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_copy_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Clamp all elements of a tile to `[lo, hi]`.
#[inline(always)]
pub unsafe fn tile_clamp_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    lo: f32,
    hi: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_clamp_f32(0, src.buf_id, lo, hi, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Fused scaled dot-product attention: softmax(Q @ K^T / sqrt(D)) @ V.
///
/// Fuses four operations into a single PTOAS-optimizable pipeline:
/// 1. matmul(Q, K^T) → scores (S × S)
/// 2. scale(scores, 1/sqrt(D))
/// 3. softmax(scores) → weights (S × S)
/// 4. matmul(weights, V) → output (S × D)
///
/// The Rust type system enforces dimension constraints:
/// Q(S×D) × K(S×D) → scores(S×S) × V(S×D) → out(S×D).
/// Move semantics ensure Q, K, V are each consumed exactly once.
#[inline(always)]
pub fn tile_attention_f32<const S: usize, const D: usize>(
    q: Tile<S, D, f32>,
    k: Tile<S, D, f32>,
    v: Tile<S, D, f32>,
) -> Tile<S, D, f32> {
    let buf_id =
        unsafe { __tile_attention_f32(0, q.buf_id, k.buf_id, v.buf_id, S as u32, D as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ── Transformer building-block wrappers ─────────────────────────────

/// SiLU/Swish: silu(x) = x * sigmoid(x).
#[inline(always)]
pub fn tile_silu_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_silu_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// RoPE: rotary position embedding at position `pos`.
///
/// ⚠ On the AscendC backend this currently lowers to an **identity copy** and
/// `pos` is discarded — the Ascend AI core has no scalar libm, so the rotation
/// needs either a host-precomputed cos/sin table (a signature change) or a
/// vector approximation. Measured on a 910B2: at `pos = 0` and `pos = 7` the
/// result is bit-identical to the input. Other backends implement it fully.
#[inline(always)]
pub fn tile_rope_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    pos: u32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rope_f32(0, src.buf_id, pos, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Causal mask: upper triangle → -inf.
#[inline(always)]
pub fn tile_causal_mask_f32<const S: usize>(src: Tile<S, S, f32>) -> Tile<S, S, f32> {
    let buf_id = unsafe { __tile_causal_mask_f32(0, src.buf_id, S as u32, S as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Window (local) mask: keep `|i - j| < window`, mask the rest.
///
/// Generic over both extents rather than a single `S` like the causal mask,
/// because the scores tile is only square for self-attention: under GQA it is
/// `(q_rows, kv_rows)`, and pinning it to `S × S` would refuse exactly the
/// operators this exists for.
///
/// `window` is a runtime argument, not a const generic — it is a masking
/// predicate, not a shape, and the emitted kernels read it as a value.
#[inline(always)]
pub fn tile_window_mask_f32<const ROWS: usize, const COLS: usize>(
    scores: Tile<ROWS, COLS, f32>,
    window: u32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_window_mask_f32(0, scores.buf_id, window, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Strided (dilated) mask: keep `i % stride == j % stride`.
#[inline(always)]
pub fn tile_strided_mask_f32<const ROWS: usize, const COLS: usize>(
    scores: Tile<ROWS, COLS, f32>,
    stride: u32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_strided_mask_f32(0, scores.buf_id, stride, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Fused grouped-query attention: `softmax(Q @ Kᵀ) @ V` as one op.
///
/// Shapes follow the intrinsic: `q` is `(nh * seq, dim)`, `k` and `v` are
/// `(nkv * seq, dim)`, and the result is `(seq, dim)`. Those row counts are
/// distinct const generics because they genuinely differ whenever `nh != nkv`,
/// which is the entire point of grouped-query attention; `seq` and `dim` are
/// passed to the intrinsic as values as well, since it needs them to unfold the
/// row axis it was given folded.
///
/// `causal` is 0 or 1. It stays a runtime argument to match the intrinsic and
/// the shim, which pass it as a constant operand rather than specialising.
#[inline(always)]
pub fn tile_attention_gqa_f32<
    const QROWS: usize,
    const KVROWS: usize,
    const DIM: usize,
    const SEQ: usize,
>(
    q: Tile<QROWS, DIM, f32>,
    k: Tile<KVROWS, DIM, f32>,
    v: Tile<KVROWS, DIM, f32>,
    nh: u32,
    nkv: u32,
) -> Tile<SEQ, DIM, f32> {
    // The ABI is (dst, q, k, v, heads_q, heads_kv, seq_len, head_dim). This
    // call used to omit `dst` and append `causal`, which shifted every operand
    // by one: q landed in dst, k in q, and SEQ in v. mlir_to_cpp refuses that
    // -- "operand 3 (v) ... is not a tile handle" -- so the operator was
    // declared, lowered, and unusable from any Rust source.
    //
    // This took a `causal: u32` until it was removed: the ABI has no causal
    // operand, no backend received it, and nothing in the tree passed a
    // non-zero one. Accepting an argument that is silently not lowered is the
    // defect this crate keeps finding elsewhere. Compose
    // `tile_causal_mask_f32` on the scores for a masked attention.
    let buf_id = unsafe {
        __tile_attention_gqa_f32(
            0,
            q.buf_id,
            k.buf_id,
            v.buf_id,
            nh,
            nkv,
            SEQ as u32,
            DIM as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Embedding: gather rows from (V, D) weight table by indices.
#[inline(always)]
pub fn tile_embedding_f32<const V: usize, const D: usize, const COUNT: usize>(
    weight: Tile<V, D, f32>,
    indices: *const u32,
) -> Tile<COUNT, D, f32> {
    let buf_id = unsafe {
        __tile_embedding_f32(0, weight.buf_id, indices, V as u32, D as u32, COUNT as u32)
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cross-entropy: -log(softmax(logits)[target]) per row.
#[inline(always)]
pub fn tile_cross_entropy_f32<const ROWS: usize, const COLS: usize>(
    logits: Tile<ROWS, COLS, f32>,
    targets: *const u32,
) -> Tile<ROWS, 1, f32> {
    let buf_id =
        unsafe { __tile_cross_entropy_f32(0, logits.buf_id, targets, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ── Phase 0: foundational primitive wrappers ────────────────────────

/// Transpose: (ROWS × COLS) → (COLS × ROWS).
#[inline(always)]
pub fn tile_transpose_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<COLS, ROWS, f32> {
    let buf_id = unsafe { __tile_transpose_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Divide every element of a tile by a scalar.
///
/// The tile-by-scalar sibling of `tile_scale_f32`. Needed by a lifted
/// `pto.trowexpanddiv` whose reduction operand is a genuine 1x1: there is no
/// reciprocal in a `#![no_core]` lifted module, so `a * (1/s)` cannot be formed.
#[inline(always)]
pub fn tile_divs_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scalar: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_divs_f32(0, src.buf_id, scalar, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Broadcast a `1 x COLS` row down `ROWS` rows.
#[inline(always)]
pub fn tile_col_expand_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<1, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_col_expand_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Sum each column of a `ROWS x COLS` tile into a `1 x COLS` row.
///
/// No `tcolsum` intrinsic exists at the safe API; the reduction over rows is
/// the transpose followed by the existing row reduction, which is what the PTO
/// chain does before it. Kept as one wrapper so the lift emits one call.
#[inline(always)]
pub fn tile_col_sum_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<1, COLS, f32> {
    let t = tile_transpose_f32::<ROWS, COLS>(src);
    let buf_id = unsafe { __tile_reduce_sum_f32(0, t.buf_id, COLS as u32, ROWS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise 1/sqrt(x).
#[inline(always)]
pub fn tile_rsqrt_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rsqrt_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise natural logarithm.
#[inline(always)]
pub fn tile_log_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_log_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise sigmoid: 1/(1+exp(-x)).
#[inline(always)]
pub fn tile_sigmoid_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_sigmoid_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cast f32 tile to f16.
#[inline(always)]
pub fn tile_cast_f32_f16<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_cast_f32_f16(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cast f16 tile to f32.
#[inline(always)]
pub fn tile_cast_f16_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_cast_f16_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cast f16 (stored as u16) to signed 16-bit.
#[inline(always)]
pub fn tile_cast_f16_i16<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, i16> {
    let buf_id = unsafe { __tile_cast_f16_i16(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Cast signed 16-bit back to f16 (stored as u16).
#[inline(always)]
pub fn tile_cast_i16_f16<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, i16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_cast_i16_f16(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Widening cast i8 -> f32 (the int8 dequant half of a q8_0 matvec). Shape is
/// preserved in the type; the per-block `d` scale is applied separately with
/// `tile_scale_f32`, keeping this a pure cast that every backend can lower.
#[inline(always)]
pub fn tile_cast_i8_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_cast_i8_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Unpack low 4-bit nibbles of a byte tile to f32 (q4 dequant, low half).
#[inline(always)]
pub fn tile_unpack_q4_lo<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u8>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_unpack_q4_lo(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Unpack high 4-bit nibbles of a byte tile to f32 (q4 dequant, high half).
#[inline(always)]
pub fn tile_unpack_q4_hi<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u8>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_unpack_q4_hi(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Decode q4_K's 12 packed scale bytes into 8 (sc, mn) f32 pairs (16-wide: sc[8], mn[8]).
/// The q4k_get_scale_min 6-bit unpack as one primitive.
#[inline(always)]
pub fn tile_unpack_q4k_scale_min(src: Tile<1, 12, u8>) -> Tile<1, 16, f32> {
    let buf_id = unsafe { __tile_unpack_q4k_scale_min(0, src.buf_id, 1, 12) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Extract a sub-tile (slice).
#[inline(always)]
pub fn tile_slice_f32<const R: usize, const C: usize, const R2: usize, const C2: usize>(
    src: Tile<R, C, f32>,
    row_off: usize,
    col_off: usize,
) -> Tile<R2, C2, f32> {
    let buf_id = unsafe {
        __tile_slice_f32(
            0,
            src.buf_id,
            row_off as u32,
            col_off as u32,
            R as u32,
            C as u32,
            R2 as u32,
            C2 as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Concatenate along columns: (R, C1) ++ (R, C2) → (R, C1+C2).
#[inline(always)]
pub fn tile_concat_f32<const R: usize, const C1: usize, const C2: usize>(
    a: Tile<R, C1, f32>,
    b: Tile<R, C2, f32>,
) -> Tile<R, { C1 + C2 }, f32> {
    let buf_id =
        unsafe { __tile_concat_f32(0, a.buf_id, b.buf_id, R as u32, C1 as u32, C2 as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Indexed scatter: scatter (N, D) rows into (M, D) at positions.
#[inline(always)]
pub fn tile_scatter_f32<const N: usize, const M: usize, const D: usize>(
    src: Tile<N, D, f32>,
    indices: *const u32,
) -> Tile<M, D, f32> {
    let buf_id =
        unsafe { __tile_scatter_f32(0, src.buf_id, indices, N as u32, M as u32, D as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Indexed gather: gather from (M, D) at positions → (N, D).
#[inline(always)]
pub fn tile_gather_f32<const N: usize, const M: usize, const D: usize>(
    src: Tile<M, D, f32>,
    indices: *const u32,
) -> Tile<N, D, f32> {
    let buf_id = unsafe { __tile_gather_f32(0, src.buf_id, indices, N as u32, M as u32, D as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Top-K selection per row: returns (R, K) values, writes indices.
#[inline(always)]
pub fn tile_topk_f32<const R: usize, const C: usize, const K: usize>(
    src: Tile<R, C, f32>,
    indices_out: *mut u32,
) -> Tile<R, K, f32> {
    let buf_id =
        unsafe { __tile_topk_f32(0, src.buf_id, indices_out, R as u32, C as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// f16 matrix multiply: C = A(M×K) @ B(K×N).
#[inline(always)]
pub fn tile_matmul_f16<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u16>,
    b: Tile<K, N, u16>,
) -> Tile<M, N, u16> {
    let buf_id = unsafe { __tile_matmul_f16(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// int8 matmul with i32 accumulator and per-column f32 dequant → f16 GM tile.
///
/// C_f16 = dequant((A_i8 @ B_i8) as i32, scale_per_col_f32).
///
/// Returns an `f16` tile (u16 bit-pattern) that the PTO emitter lowers to a
/// `pto.tstore_fp` into GM — the scale tile is read from `scale` (a GM
/// pointer to the pre-computed per-column f32 scales) via `loc=scaling`.
///
/// See memory/project_pto_i8_tmatmul_validated.md.
#[inline(always)]
pub fn tile_matmul_i8_acc_i32_dequant_f16<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u32>,
    b: Tile<K, N, u32>,
    scale: *const f32,
) -> Tile<M, N, u16> {
    let buf_id = unsafe {
        __tile_matmul_i8_acc_i32_dequant_f16(
            0, a.buf_id, b.buf_id, scale, M as u32, K as u32, N as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Load a `ROWS × COLS` i8 tile from global memory.
///
/// Per the existing i8 tile convention in this file (`tile_quantize_f32_i8`),
/// i8 tiles are typed `Tile<R, C, u32>` — the `u32` element slot is a marker
/// for the `i8` dtype at the emitter level, not the bit-width of each element.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid i8 values.
#[inline(always)]
pub fn tile_load_i8<const ROWS: usize, const COLS: usize>(gm: *const i8) -> Tile<ROWS, COLS, u32> {
    let buf_id = unsafe { __tile_load_i8(gm, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store an i8 tile to global memory. Consumes the tile handle.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` writable i8 values.
#[inline(always)]
pub fn tile_store_i8<const ROWS: usize, const COLS: usize>(
    gm: *mut i8,
    tile: Tile<ROWS, COLS, u32>,
) {
    unsafe { __tile_store_i8(gm, tile.buf_id, ROWS as u32, COLS as u32) };
}

/// f16 matrix multiply with f32 accumulator output: C = A(M×K) @ B(K×N).
///
/// Same FFI as `tile_matmul_f16` but the returned tile is typed `f32` so the
/// natural downstream op is `tile_store_f32` / `tile_add_f32`. This mirrors
/// the PTO emitter's registration of the L0C accumulator under the result
/// SSA as dtype=f32 (the only mixed-dtype triple ptoas on CANN 8.5 accepts
/// for f16 ops is `(f32, f16, f16)`). In-flight FixPipe handles any final
/// f32→target-dtype cast during the acc→GM DMA.
#[inline(always)]
pub fn tile_matmul_f16_acc_f32<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u16>,
    b: Tile<K, N, u16>,
) -> Tile<M, N, f32> {
    let buf_id = unsafe { __tile_matmul_f16(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Matmul with transposed RHS: `C[M,N] = A[M,K] @ B[N,K]^T`, f32.
///
/// `B` is shaped `Tile<N, K, f32>` in natural weight layout (rows=out, cols=in).
/// The kernel consumes the transpose implicitly via layout, so no explicit
/// transpose tile is materialised.
#[inline(always)]
pub fn tile_matmul_transposed_f32<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, f32>,
    b: Tile<N, K, f32>,
) -> Tile<M, N, f32> {
    let buf_id = unsafe {
        __tile_matmul_transposed_f32(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32)
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// f16 variant of `tile_matmul_transposed_f32`.
#[inline(always)]
pub fn tile_matmul_transposed_f16<const M: usize, const K: usize, const N: usize>(
    a: Tile<M, K, u16>,
    b: Tile<N, K, u16>,
) -> Tile<M, N, u16> {
    let buf_id = unsafe {
        __tile_matmul_transposed_f16(0, a.buf_id, b.buf_id, M as u32, K as u32, N as u32)
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ── Phase 3: Flash Attention primitive wrappers ────────────────────

/// Fill a tile with a scalar value.
#[inline(always)]
pub fn tile_fill_f32<const ROWS: usize, const COLS: usize>(scalar: f32) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_fill_f32(0, scalar, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise max of two tiles.
#[inline(always)]
pub fn tile_max_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_max_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Element-wise min of two tiles.
///
/// The `__tile_min_f32` intrinsic and the emitter arm for it both existed; only
/// this wrapper was missing, so `min(x, 0)` — the clamp in a gated linear
/// attention decay, among others — had no safe spelling.
#[inline(always)]
pub fn tile_min_f32<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, f32>,
    b: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_min_f32(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// tile_sub_f32, tile_reduce_max_f32, tile_div_f32 are defined earlier in
// this module. The Phase-3 block previously redeclared them; duplicates
// have been removed.

/// RMS normalization: x * rsqrt(mean(x^2) + eps).
#[inline(always)]
pub fn tile_rms_norm_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    eps: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rms_norm_f32(0, src.buf_id, eps, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Weighted RMS-norm over one row: `x * rsqrt(mean(x²) + eps) * g`.
///
/// The fused form the decode path actually emits, as opposed to
/// `tile_rms_norm_f32`, which does the normalisation without the learned
/// weight. Both operands are one row of `N`, and so is the result.
///
/// `n4` and `row_stride` are not parameters: they are `N / 4` and the row stride
/// in bytes, the float4 vectorisation and 3-D dispatch numbers the kernel
/// wants, and they are derivable from `N` rather than worth trusting a caller
/// to keep consistent with it. `N` should be a multiple of 4 for the float4
/// path; a remainder is the vectoriser's problem, not a shape error here.
#[inline(always)]
pub fn tile_rms_norm_mul_f32_4<const N: usize>(
    x: Tile<1, N, f32>,
    g: Tile<1, N, f32>,
    eps: f32,
) -> Tile<1, N, f32> {
    let buf_id = unsafe {
        __tile_rms_norm_mul_f32_4(
            x.buf_id,
            g.buf_id,
            N as u32,
            (N / 4) as u32,
            (N * 4) as u32,
            eps,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ── Phase 4: Quantization primitive wrappers ──────────────────────

/// Symmetric INT8 quantize: round(x / scale), clamped to [-128, 127].
#[inline(always)]
pub fn tile_quantize_f32_i8<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
    scale: f32,
) -> Tile<ROWS, COLS, u32> {
    let buf_id = unsafe { __tile_quantize_f32_i8(0, src.buf_id, scale, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Symmetric INT8 quantize with a TILE scale: `round(x / scale[0, 0])`,
/// clamped to `[-128, 127]`.
///
/// This is the form the DeepSeek V4.1 `quant_i8` `.tile` actually calls: it
/// loads its scale as a `1x1` f32 tile (`__tile_load_f32(%arg1, 1, 1)`) and
/// hands that handle to the quantiser, so there is a tile to pass and no
/// tile-to-scalar read in the API to convert it with. `SC` is the scale tile's
/// STORAGE width (the `1x8` padded load), not its one valid column.
///
/// Unlike the scalar wrapper above this one does not appear in the
/// `tile_<stem>_<dtype>` namespace the lift's wrapper table audits; it is named
/// after its operand form because the distinction is the whole point.
#[inline(always)]
pub fn tile_quantize_f32_i8_tile_scale<
    const ROWS: usize,
    const COLS: usize,
    const SC: usize,
>(
    src: Tile<ROWS, COLS, f32>,
    scale: Tile<1, SC, f32>,
) -> Tile<ROWS, COLS, u32> {
    let buf_id = unsafe {
        quantize_f32_i8_tile(0, src.buf_id, scale.buf_id, ROWS as u32, COLS as u32)
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// INT8 dequantize: x_i8 * scale → f32.
#[inline(always)]
pub fn tile_dequantize_i8_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u32>,
    scale: f32,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_dequantize_i8_f32(0, src.buf_id, scale, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise absmax: max(|x|) per row → (R, 1).
#[inline(always)]
pub fn tile_absmax_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe { __tile_absmax_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ── Phase 6: Multi-token prediction / speculative decoding wrappers ──

/// Row-wise argmax: returns index of max value per row → (R, 1) u32 indices.
///
/// ⚠ The AscendC emitter currently stores the numeric index as an f32 because
/// its vector buffers still have one kernel-wide element type. The Rust return
/// type remains u32, so generated AscendC carries a substitution marker until
/// per-buffer typing makes the representation match this contract.
#[inline(always)]
pub fn tile_argmax_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe { __tile_argmax_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise argmin → (R, 1) indices.
///
/// The mirror of `tile_argmax_f32`, and the intrinsic the greedy-decode kernel
/// actually emits: the playground's `argmax` example lowers to
/// `__tile_argmin_f32`, so the argmax wrapper alone left that kernel with
/// nothing to call.
#[inline(always)]
pub fn tile_argmin_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe { __tile_argmin_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Top-p (nucleus) sampling from logits.
///
/// Applies temperature scaling, softmax, sorts by probability, computes cumsum,
/// masks below top_p threshold, then samples from the remaining distribution.
/// Returns sampled token indices per row → (R, 1) u32.
#[inline(always)]
pub fn tile_sample_top_p_f32<const ROWS: usize, const COLS: usize>(
    logits: Tile<ROWS, COLS, f32>,
    temperature: f32,
    top_p: f32,
    rng_seed: u32,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe {
        __tile_sample_top_p_f32(
            0,
            logits.buf_id,
            temperature,
            top_p,
            rng_seed,
            ROWS as u32,
            COLS as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Draft verification: compare draft token indices against target model logits.
///
/// For each position, computes `p_target[draft_token] / p_draft[draft_token]` as
/// the acceptance ratio. Returns per-position acceptance probabilities → (R, 1).
#[inline(always)]
pub fn tile_draft_verify_f32<const ROWS: usize, const COLS: usize>(
    draft_tokens: Tile<ROWS, 1, u32>,
    target_logits: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, 1, f32> {
    let buf_id = unsafe {
        __tile_draft_verify_f32(
            0,
            draft_tokens.buf_id,
            target_logits.buf_id,
            ROWS as u32,
            COLS as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Token acceptance: select between draft and resampled tokens based on acceptance mask.
///
/// Where `accept_probs[i] >= threshold`, keeps `draft_tokens[i]`;
/// otherwise uses `target_tokens[i]` (resampled from target distribution).
/// Returns final token sequence → (R, 1) u32.
#[inline(always)]
pub fn tile_token_accept_f32<const ROWS: usize>(
    draft_tokens: Tile<ROWS, 1, u32>,
    target_tokens: Tile<ROWS, 1, u32>,
    accept_probs: Tile<ROWS, 1, f32>,
    threshold: f32,
) -> Tile<ROWS, 1, u32> {
    let buf_id = unsafe {
        __tile_token_accept_f32(
            0,
            draft_tokens.buf_id,
            target_tokens.buf_id,
            accept_probs.buf_id,
            threshold,
            ROWS as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Load two independent `ROWS × COLS` f32 tiles from separate GM pointers in one call.
///
/// Expresses two independent DMA loads so ptoas (via `--enable-insert-sync`) can
/// schedule them concurrently on the Mte2 pipe without an intervening barrier.
/// Semantically equivalent to two consecutive `tile_load_f32` calls, but signals
/// the programmer's intent that `gm0` and `gm1` are independent.
///
/// # Safety
/// Both `gm0` and `gm1` must point to at least `ROWS * COLS` valid f32 values.
#[inline(always)]
pub fn tile_join_load_f32<const ROWS: usize, const COLS: usize>(
    gm0: *const f32,
    gm1: *const f32,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    let t0 = unsafe { tile_load_f32::<ROWS, COLS>(gm0) };
    let t1 = unsafe { tile_load_f32::<ROWS, COLS>(gm1) };
    (t0, t1)
}

/// Issue a prefetch load of a `ROWS × COLS` f32 tile from global memory.
///
/// Identical to `tile_load_f32` at the hardware level. The different name signals
/// *double-buffering intent*: the caller intends to overlap this load with
/// compute on a previously loaded tile.  ptoas will schedule the resulting
/// `pto.tload` on the Mte2 pipe independently of the Vector-pipe ops that follow.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid f32 values.
#[inline(always)]
pub fn tile_prefetch_f32<const ROWS: usize, const COLS: usize>(
    gm: *const f32,
) -> Tile<ROWS, COLS, f32> {
    unsafe { tile_load_f32::<ROWS, COLS>(gm) }
}

/// Load a `ROWS × COLS` f16 tile from global memory (stored as u16).
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` valid u16 (f16) values.
#[inline(always)]
pub fn tile_load_f16<const ROWS: usize, const COLS: usize>(
    gm: *const u16,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_load_f16(gm, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store an f16 tile to global memory (as u16). Consumes the tile handle.
///
/// # Safety
/// `gm` must point to at least `ROWS * COLS` writable u16 values.
#[inline(always)]
pub fn tile_store_f16<const ROWS: usize, const COLS: usize>(
    gm: *mut u16,
    tile: Tile<ROWS, COLS, u16>,
) {
    unsafe { __tile_store_f16(gm, tile.buf_id, ROWS as u32, COLS as u32) };
}

/// Element-wise add two f16 tiles.
#[inline(always)]
pub fn tile_add_f16<const ROWS: usize, const COLS: usize>(
    a: Tile<ROWS, COLS, u16>,
    b: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_add_f16(0, a.buf_id, b.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Row-wise softmax of an f16 tile.
#[inline(always)]
pub fn tile_softmax_f16<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_softmax_f16(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decode-specific tile operations
// ═══════════════════════════════════════════════════════════════════════════

/// Cooperative matrix-vector product with f16/bf16 weights.
///
/// Computes `out = A @ W^T` where A is (1, K) in f32 and W is (N, K) in f16.
/// Uses one threadgroup per output row with simd_sum + shared memory reduction.
/// Maps to `matvec_f16` kernel on Metal, cooperative matvec on AscendC.
#[inline(always)]
pub fn tile_matvec_f16<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w: Tile<N, K, f32>, // stored as f16/bf16 on device
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_f16(0, a.buf_id, w.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Matrix-vector product with f32 weights.
///
/// Computes `out[n] = sum_k a[k] * w[n,k]` with full f32 precision on the
/// weight side (no f16/bf16 cast). For use when projection weights remain
/// f32 — the common case for DeepSeek decode on NPU before quantization.
#[inline(always)]
pub fn tile_matvec_f32<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w: Tile<N, K, f32>,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_f32(0, a.buf_id, w.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Q8_0-weight matvec — the decode dense projection. The weight is a raw GGUF
/// Q8_0 block buffer (f16 scale + 32 int8 quants per block), so it is passed as
/// a raw device pointer rather than a loaded tile; the activation is likewise a
/// raw f32 pointer. Returns the output tile for the caller to store.
#[inline(always)]
pub fn tile_matvec_q8_0<const N: usize, const K: usize>(
    w: *const i8,
    x: *const f32,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_q8_0_f32(w, x, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Q2_K-weight matvec, same raw-pointer shape as [`tile_matvec_q8_0`].
#[inline(always)]
pub fn tile_matvec_q2_k<const N: usize, const K: usize>(
    w: *const i8,
    x: *const f32,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_q2_k_f32(w, x, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// IQ2_XXS-weight matvec, same raw-pointer shape as [`tile_matvec_q8_0`].
#[inline(always)]
pub fn tile_matvec_iq2_xxs<const N: usize, const K: usize>(
    w: *const i8,
    x: *const f32,
) -> Tile<1, N, f32> {
    let buf_id = unsafe { __tile_matvec_iq2_xxs_f32(w, x, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cooperative matrix-vector product with f16/bf16 weights + bias.
///
/// Computes `out = A @ W^T + bias`.
#[inline(always)]
pub fn tile_matvec_f16_bias<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w: Tile<N, K, f32>,
    bias: Tile<1, N, f32>,
) -> Tile<1, N, f32> {
    let buf_id =
        unsafe { __tile_matvec_f16_bias(0, a.buf_id, w.buf_id, bias.buf_id, N as u32, K as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// In-place RoPE for decode (single position).
///
/// Applies rotary position embedding to x in-place.
/// Position is passed as a runtime parameter (not const generic).
#[inline(always)]
pub fn tile_rope_inplace_f32<const ROWS: usize, const COLS: usize>(
    x: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_rope_inplace_f32(0, x.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// KV cache update: write a (num_heads, head_dim) vector into cache at current position.
///
/// Cache layout: (num_heads, max_seq, head_dim). Position is a runtime parameter.
#[inline(always)]
pub fn tile_kv_cache_update_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_kv_cache_update_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Single-token attention decode with KV cache.
///
/// For each Q head: score = Q @ K_cache^T / sqrt(head_dim), softmax, @ V_cache.
/// Supports GQA (multiple Q heads share one KV head).
/// Maps to `attention_decode` kernel on Metal.
#[inline(always)]
pub fn tile_attention_decode_f32<const NH: usize, const DH: usize>(
    q: Tile<NH, DH, f32>,
    k_cache: Tile<NH, DH, f32>, // runtime: (NKV, max_seq, DH)
    v_cache: Tile<NH, DH, f32>, // runtime: (NKV, max_seq, DH)
) -> Tile<NH, DH, f32> {
    let buf_id = unsafe {
        __tile_attention_decode_f32(
            0,
            q.buf_id,
            k_cache.buf_id,
            v_cache.buf_id,
            NH as u32,
            DH as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Store a u32 tile (for argmax results).
#[inline(always)]
pub fn tile_store_u32<const ROWS: usize, const COLS: usize>(
    gm: *mut u32,
    tile: Tile<ROWS, COLS, u32>,
) {
    // Reuse the f32 store path — u32 and f32 have the same size
    unsafe { __tile_store_f32(gm as *mut f32, tile.buf_id, ROWS as u32, COLS as u32) };
}

// --- Prefill-path wrappers ---

/// Batched RoPE for prefill: applies rotary embeddings to (seq_len, num_heads, head_dim).
/// Each thread handles one (position, head, dim_pair) triple.
#[inline(always)]
pub fn tile_rope_prefill_f32<const NH: usize, const DH: usize>(
    x: Tile<NH, DH, f32>,
) -> Tile<NH, DH, f32> {
    let buf_id = unsafe { __tile_rope_prefill_f32(0, x.buf_id, NH as u32, DH as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Batched KV cache update for prefill: copies seq_len vectors into cache.
/// src is (seq_len, num_heads, head_dim), cache is (num_heads, max_seq, head_dim).
#[inline(always)]
pub fn tile_kv_cache_update_prefill_f32<const ROWS: usize, const COLS: usize>(
    src: Tile<ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_kv_cache_update_prefill_f32(0, src.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Prefill attention: Q(seq_len, num_heads, head_dim) vs KV cache.
/// One threadgroup per (query_head, query_position) pair. Causal masking + GQA.
#[inline(always)]
pub fn tile_attention_prefill_f32<const NH: usize, const DH: usize>(
    q: Tile<NH, DH, f32>,
    k_cache: Tile<NH, DH, f32>,
    v_cache: Tile<NH, DH, f32>,
) -> Tile<NH, DH, f32> {
    let buf_id = unsafe {
        __tile_attention_prefill_f32(
            0,
            q.buf_id,
            k_cache.buf_id,
            v_cache.buf_id,
            NH as u32,
            DH as u32,
        )
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Cooperative matvec + residual add: out[row] = sum_k(A[k] * B[row,k]) + R[row].
/// Fuses matvec + elementwise add to eliminate one dispatch and buffer round-trip.
#[inline(always)]
pub fn tile_matvec_f16_add<const ROWS: usize, const COLS: usize>(
    a: Tile<1, COLS, f32>,
    b: Tile<ROWS, COLS, f32>, // runtime: bf16 weights
    r: Tile<ROWS, 1, f32>,    // residual vector
) -> Tile<ROWS, 1, f32> {
    let buf_id =
        unsafe { __tile_matvec_f16_add(0, a.buf_id, b.buf_id, r.buf_id, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Fused gate+up+silu: out[i] = silu(A @ W_gate[i,:]) * (A @ W_up[i,:]).
/// Reads activation vector once, computes both gate and up projections, then fuses silu*mul.
#[inline(always)]
pub fn tile_gate_up_silu_f16<const N: usize, const K: usize>(
    a: Tile<1, K, f32>,
    w_gate: Tile<N, K, f32>, // runtime: bf16 weights
    w_up: Tile<N, K, f32>,   // runtime: bf16 weights
) -> Tile<N, 1, f32> {
    let buf_id = unsafe {
        __tile_gate_up_silu_f16(0, a.buf_id, w_gate.buf_id, w_up.buf_id, N as u32, K as u32)
    };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

// ===========================================================================
// Safe view-based API
// ===========================================================================
//
// These wrappers accept typed `GmView` / `GmViewMut` instead of raw pointers
// and compute ops accept `Tile` by value. The shape check happens through
// const generics on both sides of the call. Everything below this line is
// safe Rust: the only `unsafe` hops are `GmDeviceCtx::new()` and
// `ctx.view{,_mut}`, which the kernel ABI wrapper does exactly once per
// argument at entry. Views borrow `&'a self` from the ctx, so they can't
// outlive it — kernels cannot leak device pointers back to the host.
//
// Raw-pointer intrinsics (`tile_load_f32`, `tile_add_f32`, etc.) remain
// available unchanged — they're how the in-tree kernels work today and how
// the codegen backends walk MIR. The view-based wrappers are additive.
//
// Naming: the view-based load/store use a `_view` suffix to disambiguate
// from the raw intrinsic of the same op. Compute ops are re-exported in a
// `safe` nested module so callers can write `safe::tile_add_f32(a, b)`
// without collision.

/// Safe load: a `GmView<R,C,T>` already promises `R*C` elements of `T`
/// backed by a valid device pointer, so the load itself is safe.
#[inline(always)]
pub fn tile_load_view_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id = unsafe { __tile_load_f32(view.as_ptr(), ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Safe store: consumes the tile handle (linearity) and writes through the
/// mutable view. The view enforces shape match at the call site.
#[inline(always)]
pub fn tile_store_view_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, f32>,
    tile: Tile<ROWS, COLS, f32>,
) {
    unsafe { __tile_store_f32(view.as_mut_ptr(), tile.buf_id, ROWS as u32, COLS as u32) };
}

/// Load from a **mutable** view.
///
/// The in-place kernels need this and there was no way to express them: RoPE
/// reads and writes the same buffer, so its pointer is a `GmViewMut`, and
/// `tile_load_view_f32` takes `&GmView` — there is no conversion between the
/// two. `mlir_to_tile_rs` emitted exactly that combination and the result did
/// not compile, while the coverage matrix counted the kernel as reaching the
/// backend, because nothing ever compiles the generated source.
///
/// Mirrors `tile_store_view_f32`'s asymmetry rather than adding a `GmViewMut ->
/// GmView` coercion, so the read stays visible at the call site: a kernel that
/// loads through this is one that also writes the same buffer.
#[inline(always)]
pub fn tile_load_view_mut_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    let buf_id =
        unsafe { __tile_load_f32(view.as_mut_ptr() as *const f32, ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Safe join-load: loads two copies of the same tile shape from two views
/// in one op. Enables move-only data flow when the same input is consumed
/// by two downstream paths (e.g. `x * sigmoid(x)`).
#[inline(always)]
pub fn tile_join_load_view_f32<'a, 'b, const ROWS: usize, const COLS: usize>(
    view_a: &GmView<'a, ROWS, COLS, f32>,
    view_b: &GmView<'b, ROWS, COLS, f32>,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    unsafe { tile_join_load_f32::<ROWS, COLS>(view_a.as_ptr(), view_b.as_ptr()) }
}

/// Load the same view's contents twice into two independent tile handles.
///
/// Unlocks shape-(B) migration for kernels that need to consume the same
/// input along two data paths (`x * sigmoid(x)`, `x² + x`, …) but only
/// have a single `GmView` param from the `#[tile_kernel]` macro. Semantically
/// equivalent to two consecutive `tile_load_view_f32` calls on the same
/// view — ptoas will fuse or pipeline them as the backend sees fit.
#[inline(always)]
pub fn tile_load_view_f32_twice<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> (Tile<ROWS, COLS, f32>, Tile<ROWS, COLS, f32>) {
    unsafe { tile_join_load_f32::<ROWS, COLS>(view.as_ptr(), view.as_ptr()) }
}

/// Load the same view's contents four times into four independent tile
/// handles. Used by fused reduce+apply kernels like RMS-norm where the
/// same input is squared, summed, and reapplied in the normalization step.
#[inline(always)]
pub fn tile_load_view_f32_quad<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> (
    Tile<ROWS, COLS, f32>,
    Tile<ROWS, COLS, f32>,
    Tile<ROWS, COLS, f32>,
    Tile<ROWS, COLS, f32>,
) {
    let ptr = view.as_ptr();
    let (t0, t1) = unsafe { tile_join_load_f32::<ROWS, COLS>(ptr, ptr) };
    let (t2, t3) = unsafe { tile_join_load_f32::<ROWS, COLS>(ptr, ptr) };
    (t0, t1, t2, t3)
}

/// Safe prefetch: identical in effect to `tile_load_view_f32` at the
/// intrinsic level (delegates to `__tile_load_f32`); the distinct
/// name lets ptoas's dependency analysis recognize the issue as a
/// double-buffer prefetch and overlap it with compute.
#[inline(always)]
pub fn tile_prefetch_view_f32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, f32>,
) -> Tile<ROWS, COLS, f32> {
    unsafe { tile_prefetch_f32::<ROWS, COLS>(view.as_ptr()) }
}

/// Safe load / f16 variant.
#[inline(always)]
pub fn tile_load_view_f16<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, u16>,
) -> Tile<ROWS, COLS, u16> {
    let buf_id = unsafe { __tile_load_f16(view.as_ptr(), ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Safe load of an i8 tile through a typed global view (the q8_0 quant path).
#[inline(always)]
pub fn tile_load_view_i8<'a, const ROWS: usize, const COLS: usize>(
    view: &GmView<'a, ROWS, COLS, i8>,
) -> Tile<ROWS, COLS, i8> {
    let buf_id = unsafe { __tile_load_i8(view.as_ptr(), ROWS as u32, COLS as u32) };
    Tile {
        buf_id,
        _phantom: PhantomData,
    }
}

/// Safe store / f16 variant.
#[inline(always)]
pub fn tile_store_view_f16<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, u16>,
    tile: Tile<ROWS, COLS, u16>,
) {
    unsafe { __tile_store_f16(view.as_mut_ptr(), tile.buf_id, ROWS as u32, COLS as u32) };
}

/// Safe store / u32 variant. Used for argmax / token-index outputs.
/// Reuses the f32 store path because u32 and f32 share the same width.
#[inline(always)]
pub fn tile_store_view_u32<'a, const ROWS: usize, const COLS: usize>(
    view: &GmViewMut<'a, ROWS, COLS, u32>,
    tile: Tile<ROWS, COLS, u32>,
) {
    unsafe {
        __tile_store_f32(
            view.as_mut_ptr() as *mut f32,
            tile.buf_id,
            ROWS as u32,
            COLS as u32,
        )
    };
}

/// Safe re-exports of the pure compute ops. A `Tile` already carries the
/// full shape + dtype at the type level, so these are safe by construction
/// — the `unsafe` on the raw intrinsics is inherited from the extern "C"
/// block, not from any real safety invariant.
///
/// Example kernel body written against this module:
///
/// ```rust,ignore
/// use tile_std::tile::{GmView, GmViewMut, tile_load_view_f32, tile_store_view_f32, safe};
///
/// #[tile_std::tile_kernel]
/// pub fn softmax_tile_safe(
///     input:  GmView<'_, 1, 1024, f32>,
///     output: GmViewMut<'_, 1, 1024, f32>,
/// ) {
///     let x = tile_load_view_f32(&input);
///     let y = safe::tile_softmax_f32(x);
///     tile_store_view_f32(&output, y);
/// }
/// ```
///
/// No `unsafe` blocks inside the kernel body.
pub mod safe {
    use super::*;

    #[inline(always)]
    pub fn tile_add_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>,
        b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_add_f32(a, b) }
    }

    #[inline(always)]
    pub fn tile_sub_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>,
        b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_sub_f32(a, b) }
    }

    #[inline(always)]
    pub fn tile_mul_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>,
        b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_mul_f32(a, b) }
    }

    #[inline(always)]
    pub fn tile_div_f32<const R: usize, const C: usize>(
        a: Tile<R, C, f32>,
        b: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_div_f32(a, b) }
    }

    #[inline(always)]
    pub fn tile_neg_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, C, f32> {
        unsafe { super::tile_neg_f32(t) }
    }

    #[inline(always)]
    pub fn tile_exp_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, C, f32> {
        unsafe { super::tile_exp_f32(t) }
    }

    #[inline(always)]
    pub fn tile_softmax_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, C, f32> {
        unsafe { super::tile_softmax_f32(t) }
    }

    #[inline(always)]
    pub fn tile_matmul_f32<const M: usize, const K: usize, const N: usize>(
        a: Tile<M, K, f32>,
        b: Tile<K, N, f32>,
    ) -> Tile<M, N, f32> {
        unsafe { super::tile_matmul_f32(a, b) }
    }

    /// `C[M,N] = A[M,K] @ B[N,K]^T` — B is in natural weight layout.
    #[inline(always)]
    pub fn tile_matmul_transposed_f32<const M: usize, const K: usize, const N: usize>(
        a: Tile<M, K, f32>,
        b: Tile<N, K, f32>,
    ) -> Tile<M, N, f32> {
        unsafe { super::tile_matmul_transposed_f32(a, b) }
    }

    // ── Reductions ──────────────────────────────────────────────────
    #[inline(always)]
    pub fn tile_reduce_max_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> {
        unsafe { super::tile_reduce_max_f32(t) }
    }

    #[inline(always)]
    pub fn tile_reduce_sum_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> {
        unsafe { super::tile_reduce_sum_f32(t) }
    }

    #[inline(always)]
    pub fn tile_absmax_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, 1, f32> {
        unsafe { super::tile_absmax_f32(t) }
    }

    #[inline(always)]
    #[tile::substituted(
        backend = "ascendc",
        intrinsic = "__tile_argmax_f32",
        replacement = "f32-index-storage",
        reason = "u32-buffer-typing-unavailable"
    )]
    pub fn tile_argmax_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, 1, u32> {
        unsafe { super::tile_argmax_f32(t) }
    }

    // ── Elementwise scalar-param ────────────────────────────────────
    #[inline(always)]
    pub fn tile_scale_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
        scalar: f32,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_scale_f32(t, scalar) }
    }

    #[inline(always)]
    pub fn tile_rsqrt_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, C, f32> {
        unsafe { super::tile_rsqrt_f32(t) }
    }

    #[inline(always)]
    pub fn tile_silu_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, C, f32> {
        unsafe { super::tile_silu_f32(t) }
    }

    #[inline(always)]
    pub fn tile_rms_norm_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
        eps: f32,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_rms_norm_f32(t, eps) }
    }

    // ── Transformer building blocks ─────────────────────────────────
    #[inline(always)]
    pub fn tile_attention_f32<const S: usize, const D: usize>(
        q: Tile<S, D, f32>,
        k: Tile<S, D, f32>,
        v: Tile<S, D, f32>,
    ) -> Tile<S, D, f32> {
        unsafe { super::tile_attention_f32(q, k, v) }
    }

    #[inline(always)]
    #[tile::substituted(
        backend = "ascendc",
        intrinsic = "__tile_rope_f32",
        replacement = "identity-copy",
        reason = "missing-cos-sin-operands"
    )]
    pub fn tile_rope_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
        pos: u32,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_rope_f32(t, pos) }
    }

    #[inline(always)]
    pub fn tile_causal_mask_f32<const S: usize>(t: Tile<S, S, f32>) -> Tile<S, S, f32> {
        unsafe { super::tile_causal_mask_f32(t) }
    }

    // ── Shape / layout ──────────────────────────────────────────────
    #[inline(always)]
    pub fn tile_transpose_f32<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<C, R, f32> {
        unsafe { super::tile_transpose_f32(t) }
    }

    #[inline(always)]
    pub fn tile_slice_f32<const R: usize, const C: usize, const R2: usize, const C2: usize>(
        t: Tile<R, C, f32>,
        row_off: usize,
        col_off: usize,
    ) -> Tile<R2, C2, f32> {
        unsafe { super::tile_slice_f32(t, row_off, col_off) }
    }

    // ── Type conversions ────────────────────────────────────────────
    #[inline(always)]
    pub fn tile_cast_f32_f16<const R: usize, const C: usize>(
        t: Tile<R, C, f32>,
    ) -> Tile<R, C, u16> {
        unsafe { super::tile_cast_f32_f16(t) }
    }

    #[inline(always)]
    pub fn tile_cast_f16_f32<const R: usize, const C: usize>(
        t: Tile<R, C, u16>,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_cast_f16_f32(t) }
    }

    #[inline(always)]
    pub fn tile_dequantize_i8_f32<const R: usize, const C: usize>(
        t: Tile<R, C, u32>,
        scale: f32,
    ) -> Tile<R, C, f32> {
        unsafe { super::tile_dequantize_i8_f32(t, scale) }
    }

    // ── Indexed / sampling ──────────────────────────────────────────
    //
    // These take raw `*const u32` / `*mut u32` index arrays. The raw-pointer
    // parts stay `unsafe` at the call site because they carry no shape info
    // and are passed from the host via the C ABI. The tile op itself is safe.

    /// # Safety
    /// `indices_out` must be a valid `*mut u32` buffer of at least `R * K` elements.
    #[inline(always)]
    pub unsafe fn tile_topk_f32<const R: usize, const C: usize, const K: usize>(
        t: Tile<R, C, f32>,
        indices_out: *mut u32,
    ) -> Tile<R, K, f32> {
        unsafe { super::tile_topk_f32(t, indices_out) }
    }

    /// # Safety
    /// `indices` must be a valid `*const u32` buffer of at least `N` elements.
    #[inline(always)]
    pub unsafe fn tile_scatter_f32<const N: usize, const M: usize, const D: usize>(
        src: Tile<N, D, f32>,
        indices: *const u32,
    ) -> Tile<M, D, f32> {
        unsafe { super::tile_scatter_f32(src, indices) }
    }

    /// # Safety
    /// `indices` must be a valid `*const u32` buffer of at least `COUNT` elements.
    #[inline(always)]
    pub unsafe fn tile_embedding_f32<const V: usize, const D: usize, const COUNT: usize>(
        weight: Tile<V, D, f32>,
        indices: *const u32,
    ) -> Tile<COUNT, D, f32> {
        unsafe { super::tile_embedding_f32(weight, indices) }
    }

    /// # Safety
    /// `targets` must be a valid `*const u32` buffer of at least `R` elements.
    #[inline(always)]
    pub unsafe fn tile_cross_entropy_f32<const R: usize, const C: usize>(
        logits: Tile<R, C, f32>,
        targets: *const u32,
    ) -> Tile<R, 1, f32> {
        unsafe { super::tile_cross_entropy_f32(logits, targets) }
    }

    #[inline(always)]
    pub fn tile_sample_top_p_f32<const R: usize, const C: usize>(
        logits: Tile<R, C, f32>,
        temperature: f32,
        top_p: f32,
        rng_seed: u32,
    ) -> Tile<R, 1, u32> {
        unsafe { super::tile_sample_top_p_f32(logits, temperature, top_p, rng_seed) }
    }

    #[inline(always)]
    pub fn tile_draft_verify_f32<const R: usize, const C: usize>(
        draft_tokens: Tile<R, 1, u32>,
        target_logits: Tile<R, C, f32>,
    ) -> Tile<R, 1, f32> {
        unsafe { super::tile_draft_verify_f32(draft_tokens, target_logits) }
    }

    #[inline(always)]
    pub fn tile_token_accept_f32<const R: usize>(
        draft_tokens: Tile<R, 1, u32>,
        target_tokens: Tile<R, 1, u32>,
        accept_probs: Tile<R, 1, f32>,
        threshold: f32,
    ) -> Tile<R, 1, u32> {
        unsafe {
            super::tile_token_accept_f32(draft_tokens, target_tokens, accept_probs, threshold)
        }
    }

    // ── Wrappers the mlir_to_tile_rs backend emits calls to ─────────
    //
    // `safe` is the module that backend targets: it emits
    // `safe::tile_X(...)`, so an op whose wrapper exists only at the top level
    // of `tile::` is unreachable from generated code. Every function below has
    // a top-level twin; these are the entries that make the generated kernels
    // resolve, and their absence is why several kernels emitted Rust that named
    // functions `safe` did not have.

    #[inline(always)]
    pub fn tile_window_mask_f32<const R: usize, const C: usize>(
        scores: Tile<R, C, f32>,
        window: u32,
    ) -> Tile<R, C, f32> {
        super::tile_window_mask_f32(scores, window)
    }

    #[inline(always)]
    pub fn tile_strided_mask_f32<const R: usize, const C: usize>(
        scores: Tile<R, C, f32>,
        stride: u32,
    ) -> Tile<R, C, f32> {
        super::tile_strided_mask_f32(scores, stride)
    }

    #[inline(always)]
    pub fn tile_argmin_f32<const R: usize, const C: usize>(t: Tile<R, C, f32>) -> Tile<R, 1, u32> {
        super::tile_argmin_f32(t)
    }

    #[inline(always)]
    pub fn tile_attention_gqa_f32<
        const QROWS: usize,
        const KVROWS: usize,
        const DIM: usize,
        const SEQ: usize,
    >(
        q: Tile<QROWS, DIM, f32>,
        k: Tile<KVROWS, DIM, f32>,
        v: Tile<KVROWS, DIM, f32>,
        nh: u32,
        nkv: u32,
    ) -> Tile<SEQ, DIM, f32> {
        super::tile_attention_gqa_f32(q, k, v, nh, nkv)
    }

    #[inline(always)]
    pub fn tile_rms_norm_mul_f32_4<const N: usize>(
        x: Tile<1, N, f32>,
        g: Tile<1, N, f32>,
        eps: f32,
    ) -> Tile<1, N, f32> {
        super::tile_rms_norm_mul_f32_4(x, g, eps)
    }

    #[inline(always)]
    pub fn tile_rope_inplace_f32<const R: usize, const C: usize>(
        x: Tile<R, C, f32>,
    ) -> Tile<R, C, f32> {
        super::tile_rope_inplace_f32(x)
    }

    #[inline(always)]
    pub fn tile_matvec_f32<const N: usize, const K: usize>(
        a: Tile<1, K, f32>,
        w: Tile<N, K, f32>,
    ) -> Tile<1, N, f32> {
        super::tile_matvec_f32(a, w)
    }

    #[inline(always)]
    pub fn tile_matvec_f16<const N: usize, const K: usize>(
        a: Tile<1, K, f32>,
        w: Tile<N, K, f32>,
    ) -> Tile<1, N, f32> {
        super::tile_matvec_f16(a, w)
    }

    #[inline(always)]
    pub fn tile_gate_up_silu_f16<const N: usize, const K: usize>(
        a: Tile<1, K, f32>,
        w_gate: Tile<N, K, f32>,
        w_up: Tile<N, K, f32>,
    ) -> Tile<N, 1, f32> {
        super::tile_gate_up_silu_f16(a, w_gate, w_up)
    }
}

/// Bitwise AND with a scalar on an i16 tile (MXFP4 packing).
#[inline(always)]
pub fn tile_ands_i16<const R: usize, const C: usize>(src: Tile<R, C, i16>, imm: u32) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_ands_i16(0, src.buf_id, imm, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Shift left with a scalar on an i16 tile.
#[inline(always)]
pub fn tile_shls_i16<const R: usize, const C: usize>(src: Tile<R, C, i16>, imm: u32) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_shls_i16(0, src.buf_id, imm, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Shift right with a scalar on an i16 tile.
#[inline(always)]
pub fn tile_shrs_i16<const R: usize, const C: usize>(src: Tile<R, C, i16>, imm: u32) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_shrs_i16(0, src.buf_id, imm, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Add a scalar to every element of an i16 tile (MXFP4 packing).
#[inline(always)]
pub fn tile_adds_i16<const R: usize, const C: usize>(src: Tile<R, C, i16>, imm: u32) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_adds_i16(0, src.buf_id, imm, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Clamp an i16 tile from below by a scalar.
#[inline(always)]
pub fn tile_maxs_i16<const R: usize, const C: usize>(src: Tile<R, C, i16>, imm: u32) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_maxs_i16(0, src.buf_id, imm, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise add of two i16 tiles.
#[inline(always)]
pub fn tile_add_i16<const R: usize, const C: usize>(a: Tile<R, C, i16>, b: Tile<R, C, i16>) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_add_i16(0, a.buf_id, b.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Scale every element of an i16 tile by a scalar.
#[inline(always)]
pub fn tile_scale_i16<const R: usize, const C: usize>(src: Tile<R, C, i16>, scalar: i16) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_scale_i16(0, src.buf_id, scalar, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Element-wise multiply of two i16 tiles.
#[inline(always)]
pub fn tile_mul_i16<const R: usize, const C: usize>(a: Tile<R, C, i16>, b: Tile<R, C, i16>) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_mul_i16(0, a.buf_id, b.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Tile-to-tile copy of i16.
#[inline(always)]
pub fn tile_copy_i16<const R: usize, const C: usize>(src: &Tile<R, C, i16>) -> Tile<R, C, i16> {
    let buf_id = unsafe { __tile_copy_i16(0, src.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Whole-kernel MXFP4 pack matvec: packed 4-bit weights, activations, block
/// scales, output. The reverse lift emits this one call for the expanded body.
#[inline(always)]
pub unsafe fn tile_mul_mv_id_mxfp4_pk_f32<const NE00: usize, const NE0: usize>(
    w: *const u8,
    x: *const f32,
    d: *const f32,
    o: *mut f32,
) {
    __tile_mul_mv_id_mxfp4_pk_f32(w, x, d, o, NE00 as u32, NE0 as u32);
}

/// Fill a 1 x N i32 tile with `start, start+1, ...` (the `pto.tci` index init).
#[inline(always)]
pub fn tile_arith_progression_i32<const N: usize>(start: i32) -> Tile<1, N, i32> {
    let buf_id = unsafe { __tile_arith_progression_i32(0, start as u32, N as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Sort a 1 x C f32 tile with its i32 indices; the result is the 2*C packed
/// (value, index) tile the topk selector consumes.
#[inline(always)]
pub fn tile_sort32_f32<const R: usize, const C: usize, const C2: usize>(
    values: Tile<R, C, f32>,
    indices: Tile<R, C, i32>,
) -> Tile<R, C2, f32> {
    let buf_id = unsafe { __tile_sort32_f32(0, values.buf_id, indices.buf_id, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Mask-pattern gather (lane select); `mask` is the 4-bit pattern (P1010 = 10).
#[inline(always)]
pub fn tile_gather_mask_f32<const R: usize, const C: usize, const C2: usize>(
    src: Tile<R, C, f32>,
    mask: u32,
) -> Tile<R, C2, f32> {
    let buf_id = unsafe { __tile_gather_mask_f32(0, src.buf_id, mask, R as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// Extract a R2 x C2 sub-tile at `(row, col)` of a R x C tile.
#[inline(always)]
pub fn tile_extract_f32<const R: usize, const C: usize, const R2: usize, const C2: usize>(
    src: Tile<R, C, f32>,
    row: u32,
    col: u32,
) -> Tile<R2, C2, f32> {
    let buf_id = unsafe { __tile_extract_f32(0, src.buf_id, row, col, R2 as u32, C2 as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

// --- fused f16 lm-head GEMM device kernels --------------------------------
//
// These are NOT `__tile_*` tile-ISA intrinsics that an emitter lowers. They
// are the complete generated device entry points of the DeepSeek V4.1 f16
// lm-head GEMMs -- exactly the `bm_gemm_f16_16_5120_<n>` `llvm.func`s the
// `.tile` artifacts define. Binding the symbol is the honest representation
// of that kernel: its body is already fused on the device (the source PTO
// splits it into AIC/AIV halves joined by a cross-core FIFO), so this wrapper
// binds the symbol instead of re-deriving the kernel from tile ops -- a body
// rebuilt that way would lower the `.tile`'s simplified fused form, not the
// source PTO, and would silently stand in for a kernel it does not model. The
// symbol name carries the shape; the wrapper's const generics repeat it and
// reject a mismatch at compile time.
extern "C" {
    /// `bm_gemm_f16_16_5120_12928`: fused f16 lm-head GEMM,
    /// C(16x12928) = A(16x5120) @ B(5120x12928). Operands are `(a, b, out)`,
    /// in the order the `.tile` body uses them.
    pub fn bm_gemm_f16_16_5120_12928(a: *const u16, b: *const u16, out: *mut u16);

    /// `bm_gemm_f16_16_5120_16160`: fused f16 lm-head GEMM,
    /// C(16x16160) = A(16x5120) @ B(5120x16160). Operands are `(a, b, out)`.
    pub fn bm_gemm_f16_16_5120_16160(a: *const u16, b: *const u16, out: *mut u16);
}

/// Shape marker for the fused f16 lm-head GEMM family.
///
/// The wrapper below is generic only so the call site states the same
/// `M, K, N` triple the `.tile` hard-codes. This marker carries that triple;
/// only the two generated shapes implement [`F16LmHeadGemmLaunch`], so a
/// mismatched triple is a trait-bound error instead of a call into a kernel
/// with a different extent. A generic `const { assert!(...) }` cannot do the
/// job here: this no_core crate's `assert!` expands to a non-const panic and
/// generic-constant control flow is rejected.
pub struct F16LmHeadGemmShape<const M: usize, const K: usize, const N: usize>;

/// Launches the generated device kernel for one exactly-known shape triple.
///
/// Not for direct use: call [`tile_gemm_f16_lmhead`], whose const generics
/// select this impl.
pub trait F16LmHeadGemmLaunch {
    /// # Safety
    /// Same contract as [`tile_gemm_f16_lmhead`].
    unsafe fn launch(a: *const u16, b: *const u16, out: *mut u16);
}

impl F16LmHeadGemmLaunch for F16LmHeadGemmShape<16, 5120, 12928> {
    #[inline(always)]
    unsafe fn launch(a: *const u16, b: *const u16, out: *mut u16) {
        bm_gemm_f16_16_5120_12928(a, b, out);
    }
}

impl F16LmHeadGemmLaunch for F16LmHeadGemmShape<16, 5120, 16160> {
    #[inline(always)]
    unsafe fn launch(a: *const u16, b: *const u16, out: *mut u16) {
        bm_gemm_f16_16_5120_16160(a, b, out);
    }
}

/// Whole-kernel f16 lm-head GEMM: C(MxN) = A(MxK) @ B(KxN), bound to the
/// generated fused `bm_gemm_f16_16_5120_<N>` device kernel.
///
/// This is a binding, not a lowering: the callee is the fused device kernel
/// the `.tile` defines, and this crate does not decompose it into tile-ISA
/// ops. The const generics are the shape triple that kernel hard-codes in its
/// name; only the corpus shapes are implemented, so any other triple is a
/// trait-bound error rather than a call into a different kernel.
///
/// # Safety
/// For shape `(M, K, N)`, `a` must point to at least `M * K` readable f16
/// (u16) values, `b` to at least `K * N`, and `out` to at least `M * N`
/// writable f16 (u16) values. The regions must not overlap in any way the
/// device kernel does not permit.
#[inline(always)]
pub unsafe fn tile_gemm_f16_lmhead<const M: usize, const K: usize, const N: usize>(
    a: *const u16,
    b: *const u16,
    out: *mut u16,
) where
    F16LmHeadGemmShape<M, K, N>: F16LmHeadGemmLaunch,
{
    <F16LmHeadGemmShape<M, K, N> as F16LmHeadGemmLaunch>::launch(a, b, out);
}

/// 1-query decode attention over a block of `s` KV rows, D = C.
///
/// The sink-free sibling of `__tile_attention_sink_f32`; K and V are raw GM
/// pointers and the kernel streams the KV blocks itself.
#[inline(always)]
pub fn tile_attention_blocked_f32<const R: usize, const C: usize>(
    q: Tile<R, C, f32>,
    k: *const f32,
    v: *const f32,
    s: usize,
) -> Tile<R, C, f32> {
    let buf_id =
        unsafe { __tile_attention_blocked_f32(0, q.buf_id, k, v, s as u32, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}

/// 1-query decode attention at a RUNTIME context length: `st` is a loaded 1x1
/// tile holding the KV length for this launch.
#[inline(always)]
pub fn tile_attention_dyn_f32<const R: usize, const C: usize>(
    q: Tile<R, C, f32>,
    k: *const f32,
    v: *const f32,
    st: Tile<R, 1, f32>,
) -> Tile<R, C, f32> {
    let buf_id =
        unsafe { __tile_attention_dyn_f32(0, q.buf_id, k, v, st.buf_id, C as u32) };
    Tile { buf_id, _phantom: PhantomData }
}
