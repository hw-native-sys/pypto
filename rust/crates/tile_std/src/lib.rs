// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#![feature(
    no_core,
    lang_items,
    intrinsics,
    unboxed_closures,
    extern_types,
    decl_macro,
    rustc_attrs,
    transparent_unions,
    auto_traits,
    freeze_impls,
    thread_local,
    f16,
    f128,
    const_trait_impl,
    fundamental,
    macro_metavar_expr,
    negative_impls,
    allow_internal_unstable,
    doc_notable_trait,
    prelude_import,
    never_type,
    min_specialization,
    generic_const_exprs,
    register_tool
)]
#![allow(
    dead_code,
    internal_features,
    ambiguous_wide_pointer_comparisons,
    unused,
    unused_variables
)]
#![no_std]
#![no_core]
#![rustc_coherence_is_core]
#![register_tool(tile)]

pub mod buf;
pub mod core;
pub mod kernel_ops;
pub mod pipeline;
// Host-only. These exist to be TYPE-CHECKED by `cargo check` (see the module header);
// they are dead code crate-wide. Code-generating them is a different matter: the tile
// backend emits MLIR for every function in the crate, so a `pub` module of kernels lands
// in EVERY kernel's merged module — and `emitted_kernels::window_mask` calls
// `tile_window_mask_f32`, which `mlir_to_pto` has no arm for. Since
// `reject_unknown_intrinsics` scans the whole module text, that one dead kernel made
// Rust -> PTO impossible for every kernel, including one that only loads, adds and
// stores. The gate keeps the type-checking and drops the codegen.
//
// The gate keys on a CARGO FEATURE, and the history matters because the comment here
// used to say something else. It read "`davinci-huawei-none.json` sets "os": "cuda", so
// this gate keys on the device spec's os -- THAT FIELD IS LOAD-BEARING". That was true
// of an earlier `#[cfg(not(target_os = "cuda"))]`, and it stopped being true when the
// cfg moved to this feature: a lowering driver now enables `kernel-codegen` on its
// tile_std dependency and builds for the HOST triple, with no device target spec
// involved at all.
//
// Worth stating plainly because the stale version was load-bearing in a different
// sense: it was the stated reason a custom rustc target specification -- one whose
// llvm-target, arch, os and linker-flavor all name another vendor's architecture --
// could not simply be replaced. It can. Nothing in this crate reads it.
#[cfg(not(feature = "kernel-codegen"))]
pub mod emitted_kernels;
pub mod tile;

pub use core::prelude::v1::*;
pub use core::*;
pub use kernel_ops::*;
pub use tile_std_macros::*;

/// Unified Buffer (UB) buffer ID — returned by __tile_buf_alloc.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct UbBuf(u32);

/// L1/CBUF buffer ID — returned by __tile_buf_alloc_l1.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct L1Buf(u32);

/// L0A buffer ID — returned by __tile_buf_alloc_l0a.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct L0aBuf(u32);

/// L0B buffer ID — returned by __tile_buf_alloc_l0b.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct L0bBuf(u32);

/// L0C buffer ID — returned by __tile_buf_alloc_l0c.
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct L0cBuf(u32);

extern "C" {
    // Block/sub-block index queries
    pub fn get_block_idx() -> usize;
    pub fn get_block_num() -> usize;
    pub fn get_sub_block_num() -> usize;
    pub fn get_sub_block_idx() -> usize;

    // Buffer management — mapped to TBuf/LocalTensor by mlir_to_cpp
    pub fn __tile_buf_alloc(elem_count: u32) -> UbBuf;
    pub fn __tile_buf_load_f32(buf: UbBuf, gm: *const f32, count: u32);
    pub fn __tile_buf_store_f32(gm: *mut f32, buf: UbBuf, count: u32);
    pub fn __tile_buf_load_f16(buf: UbBuf, gm: *const u16, count: u32);
    pub fn __tile_buf_store_f16(gm: *mut u16, buf: UbBuf, count: u32);
    pub fn __tile_buf_load_bf16(buf: UbBuf, gm: *const u16, count: u32);
    pub fn __tile_buf_store_bf16(gm: *mut u16, buf: UbBuf, count: u32);
    // Fill a buffer with a scalar constant (AscendC::Duplicate)
    pub fn __tile_buf_fill_f32(buf: UbBuf, val: f32, count: u32);
    pub fn __tile_buf_fill_f16(buf: UbBuf, val: f32, count: u32);
    pub fn __tile_buf_fill_bf16(buf: UbBuf, val: f32, count: u32);
    pub fn __tile_pipe_barrier();

    // Reduce operations (f32)
    pub fn __tile_v_reduce_max_f32(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;
    pub fn __tile_reduce_min_f32(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;
    pub fn __tile_v_reduce_sum_f32(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;

    // Unary vector operations (f32)
    pub fn __tile_v_exp_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_v_abs_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_ln_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_v_sqrt_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_v_rsqrt_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_reciprocal_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sign_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_round_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sq_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_ceil_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_floor_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_trunc_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sin_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cos_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_atan_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_erf_f32(dst: UbBuf, src: UbBuf, n: u32);

    // Elementwise math `mlir_to_msl` lowers from its own op table, which gives
    // both the Metal expression and the operand count — `acos` and friends at 1,
    // `pow`/`fmod` at 2. Declared in the buffer convention because that is what
    // their table-neighbours atan and erf already use, so the shape follows the
    // siblings the same emitter handles rather than a guess.
    pub fn __tile_acos_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_asin_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cbrt_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cosh_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sinh_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_square_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_pow_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_fmod_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_erfinv_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_fast_gelu_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_not_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_v_softplus_f32(dst: UbBuf, src: UbBuf, n: u32);

    // Scalar-vector operations (f32)
    pub fn __tile_adds_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_muls_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_maxs_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_mins_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);

    // Binary vector operations (f32)
    pub fn __tile_v_add_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_v_sub_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_v_mul_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_v_div_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_v_min_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_v_max_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_and_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_or_f32(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);

    // Binary vector operations (f16)
    pub fn __tile_v_add_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_sub_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_v_mul_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_div_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_min_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_max_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_and_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_or_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);

    // Unary vector operations (f16)
    pub fn __tile_v_exp_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_v_abs_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_ln_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sqrt_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_rsqrt_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_reciprocal_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sign_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_round_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sq_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_ceil_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_floor_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_trunc_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sin_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cos_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_atan_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_erf_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_acos_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_asin_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cbrt_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cosh_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sinh_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_square_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_pow_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_fmod_f16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_erfinv_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_fast_gelu_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_not_f16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_softplus_f16(dst: UbBuf, src: UbBuf, n: u32);

    // Scalar-vector operations (f16)
    // Note: scalar parameter is f32 because AscendC Adds/Muls accept float scalars
    // even for half-precision tensors.
    pub fn __tile_adds_f16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_muls_f16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_maxs_f16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_mins_f16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);

    // Reduce operations (f16)
    pub fn __tile_reduce_max_f16(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;
    pub fn __tile_reduce_min_f16(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;
    pub fn __tile_reduce_sum_f16(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;

    // Binary vector operations (bf16) — same physical layout as f16 (u16)
    pub fn __tile_add_bf16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_sub_bf16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_mul_bf16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_div_bf16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_min_bf16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);
    pub fn __tile_max_bf16(dst: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);

    // Unary vector operations (bf16)
    pub fn __tile_exp_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_abs_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_ln_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sqrt_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_rsqrt_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_reciprocal_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sign_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_round_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sq_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_ceil_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_floor_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_trunc_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_sin_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cos_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_neg_bf16(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_not_bf16(dst: UbBuf, src: UbBuf, n: u32);

    // Scalar-vector operations (bf16)
    pub fn __tile_adds_bf16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_muls_bf16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_maxs_bf16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_mins_bf16(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);

    // Reduce operations (bf16)
    pub fn __tile_reduce_max_bf16(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;
    pub fn __tile_reduce_sum_bf16(dst: UbBuf, src: UbBuf, work: UbBuf, n: u32) -> f32;

    // Duplicate (bf16)
    pub fn __tile_duplicate_bf16(dst: UbBuf, scalar: f32, n: u32);

    // =========================================================================
    // Cube engine (matrix multiply) operations
    // =========================================================================
    //
    // The cube engine performs matrix multiply-accumulate (Mmad) using dedicated
    // L0A/L0B/L0C memory. Data flows through: GM → L1 → L0A/L0B → Mmad → L0C.
    //
    // Buffer positions:
    //   L1 (CBUF)  - Shared staging area between GM and L0
    //   L0A        - Matrix A input buffer (left operand)
    //   L0B        - Matrix B input buffer (right operand)
    //   L0C        - Accumulator output buffer (FP32)

    // Cube buffer allocation — returns buffer ID for the specified memory level
    pub fn __tile_buf_alloc_l1(elem_count: u32) -> L1Buf;
    pub fn __tile_buf_alloc_l0a(elem_count: u32) -> L0aBuf;
    pub fn __tile_buf_alloc_l0b(elem_count: u32) -> L0bBuf;
    pub fn __tile_buf_alloc_l0c(elem_count: u32) -> L0cBuf;

    // Data movement for cube pipeline
    pub fn __tile_load_gm_to_l1_f16(l1_buf: L1Buf, gm: *const u16, count: u32);
    pub fn __tile_copy_l1_to_l0a_f16(l0a_buf: L0aBuf, l1_buf: L1Buf, count: u32);
    pub fn __tile_copy_l1_to_l0b_f16(l0b_buf: L0bBuf, l1_buf: L1Buf, count: u32);
    pub fn __tile_copy_l0c_to_ub_f32(ub_buf: UbBuf, l0c_buf: L0cBuf, count: u32);

    // Matrix multiply-accumulate: C[m,n] += A[m,k] * B[k,n]
    // A, B are f16 in L0A/L0B; C is f32 in L0C.
    // init: 1 = zero-initialize C before multiply, 0 = accumulate onto existing C
    pub fn __tile_mmad_f16(
        c_l0c: L0cBuf,
        a_l0a: L0aBuf,
        b_l0b: L0bBuf,
        m: u32,
        k: u32,
        n: u32,
        init: u32,
    );

    // L1 → L0A/L0B copy with LoadData2dParams (supports hardware transpose)
    // repeat_times: number of 16x16 fractal repeats
    // src_stride: stride between fractal blocks in source
    // transpose: 1 = transpose during copy, 0 = normal
    pub fn __tile_copy_l1_to_l0a_f16_2d(
        l0a_buf: L0aBuf,
        l1_buf: L1Buf,
        repeat_times: u32,
        src_stride: u32,
        transpose: u32,
    );
    pub fn __tile_copy_l1_to_l0b_f16_2d(
        l0b_buf: L0bBuf,
        l1_buf: L1Buf,
        repeat_times: u32,
        src_stride: u32,
        transpose: u32,
    );

    // Extended Mmad: exposes enBias, enATranspose, enBTranspose from MmadParams
    pub fn __tile_mmad_f16_ex(
        c_l0c: L0cBuf,
        a_l0a: L0aBuf,
        b_l0b: L0bBuf,
        m: u32,
        k: u32,
        n: u32,
        init: u32,
        en_bias: u32,
        transpose_a: u32,
        transpose_b: u32,
    );

    // GM→L1 with ND→NZ format conversion (for multi-fractal matrices)
    // height/width are matrix dimensions (in elements). Rearranges from row-major to
    // column-major fractal layout. For 16×16 matrices, equivalent to flat DataCopy.
    pub fn __tile_load_gm_to_l1_nd2nz_f16(l1_buf: L1Buf, gm: *const u16, height: u32, width: u32);

    // L0C→UB with BLOCK_MODE_MATRIX (takes m, n separately for proper blockCount/blockLen)
    pub fn __tile_copy_l0c_to_ub_f32_matrix(ub_buf: UbBuf, l0c_buf: L0cBuf, m: u32, n: u32);

    // UB→GM with NZ→ND format conversion (for multi-fractal f32 matrices)
    // m/n are output matrix dimensions. Rearranges from fractal to row-major layout.
    pub fn __tile_store_ub_to_gm_nz2nd_f32(gm: *mut f32, ub_buf: UbBuf, m: u32, n: u32);

    // Complete cube matmul pipeline: C[m,n] = A[m,k] × B[k,n] (f16→f32)
    // Generates the full reference-style code including:
    //   CopyND2NZ for GM→L1, SplitA loop, B-split loop with per-nBlock
    //   SplitB/Compute/Aggregate, and NZ→ND CopyOut.
    // This is a "macro intrinsic" — expanded by codegen into complete C++ code.
    pub fn __tile_matmul_cube(c: *mut f32, a: *const u16, b: *const u16, m: u32, k: u32, n: u32);

    // Type casting (f16 <-> f32)
    pub fn __tile_cast_f16_to_f32(dst: UbBuf, src: UbBuf, n: u32);
    pub fn __tile_cast_f32_to_f16(dst: UbBuf, src: UbBuf, n: u32);

    // Vector fill/duplicate
    pub fn __tile_duplicate_f32(dst: UbBuf, scalar: f32, n: u32);
    pub fn __tile_duplicate_f16(dst: UbBuf, scalar: f32, n: u32);

    // Scalar element access (GetValue / SetValue)
    pub fn __tile_get_value_f32(buf: UbBuf, idx: u32) -> f32;
    pub fn __tile_get_value_f16(buf: UbBuf, idx: u32) -> u16;
    pub fn __tile_get_value_i32(buf: UbBuf, idx: u32) -> i32;
    pub fn __tile_get_value_u32(buf: UbBuf, idx: u32) -> u32;
    pub fn __tile_set_value_f32(buf: UbBuf, idx: u32, val: f32);
    pub fn __tile_set_value_f16(buf: UbBuf, idx: u32, val: u16);
    pub fn __tile_set_value_i32(buf: UbBuf, idx: u32, val: i32);
    pub fn __tile_set_value_u32(buf: UbBuf, idx: u32, val: u32);

    // Compare / Select (conditional vector ops)
    pub fn __tile_compare_scalar_eq_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_compare_scalar_lt_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_compare_scalar_gt_f32(dst: UbBuf, src: UbBuf, scalar: f32, n: u32);
    pub fn __tile_select_f32(dst: UbBuf, sel: UbBuf, src1: UbBuf, src2: UbBuf, n: u32);

    // Buffer load/store for integer types
    pub fn __tile_buf_load_i32(buf: UbBuf, gm: *const i32, count: u32);
    pub fn __tile_buf_store_i32(gm: *mut i32, buf: UbBuf, count: u32);
    pub fn __tile_buf_load_u32(buf: UbBuf, gm: *const u32, count: u32);
    pub fn __tile_buf_store_u32(gm: *mut u32, buf: UbBuf, count: u32);
}

// ---------------------------------------------------------------------------
// ascend_-prefixed wrappers for kernel_ops composite functions
// (These delegate to kernel_ops:: implementations so that generated code
//  can call `tile_std::__tile_v_relu_f32(...)` uniformly.)
// ---------------------------------------------------------------------------

#[inline(always)]
pub unsafe fn __tile_v_relu_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::relu_f32(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_relu_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::relu_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_sigmoid_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::sigmoid_f32(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_sigmoid_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::sigmoid_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_tanh_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::tanh_f32(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_tanh_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::tanh_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_neg_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::neg_f32(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_neg_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::neg_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_expm1_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::expm1_f32(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_expm1_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::expm1_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_gelu_f32(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        let mut d = dst;
        let mut tmp = __tile_buf_alloc(n);
        kernel_ops::gelu_f32(&mut d, &src, &mut tmp, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_v_gelu_f16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        let mut d = dst;
        let mut tmp = __tile_buf_alloc(n);
        kernel_ops::gelu_f16(&mut d, &src, &mut tmp, n);
    }
}

// bf16 composite wrappers — delegate to f16 implementations
// (bf16 uses the same u16 physical representation)
#[inline(always)]
pub unsafe fn __tile_relu_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::relu_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_sigmoid_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::sigmoid_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_tanh_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::tanh_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_expm1_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        kernel_ops::expm1_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_gelu_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        let mut d = dst;
        let mut tmp = __tile_buf_alloc(n);
        kernel_ops::gelu_f16(&mut d, &src, &mut tmp, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_fast_gelu_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    // fast_gelu delegates to gelu for bf16
    unsafe {
        __tile_gelu_bf16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_softplus_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    // softplus(x) = ln(1 + exp(x)) — delegate to f32 version
    unsafe {
        kernel_ops::softplus_f32(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_atan_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        __tile_atan_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_erf_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        __tile_erf_f16(dst, src, n);
    }
}
#[inline(always)]
pub unsafe fn __tile_erfinv_bf16(dst: UbBuf, src: UbBuf, n: u32) {
    unsafe {
        __tile_erfinv_f16(dst, src, n);
    }
}

// ---------------------------------------------------------------------------
// Compile-time capacity guard (F1), no_core-safe.
//
// A trait bound that resolves only when the budget condition holds. When a tile
// footprint exceeds the declared budget the bound BudgetCheck<false>: BudgetOk
// fails to resolve — a TYPE error at compile time, with no panic/unwind path
// (kernels are no_std/no_core, where a message-assert would need the panic
// lang item). The #[tile_kernel(budget=..)] macro emits one bound per view.
// ---------------------------------------------------------------------------
pub struct BudgetCheck<const OK: bool>;
pub trait BudgetOk {}
impl BudgetOk for BudgetCheck<true> {}
