// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Transcribed from tile/gemm_f16_lmhead_v41_16x5120x16160.tile.
// The fused device symbol this kernel defines is bound by
// tile_std::tile::tile_gemm_f16_lmhead::<16, 5120, 16160>, whose extern
// declaration uses this exact ABI; the wrapper does not re-derive the body.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn bm_gemm_f16_16_5120_16160(
    a0: *const u16,
    a1: *const u16,
    a2: *mut u16,
) {
    let a = tile_load_f16::<16, 5120>(a0);
    let b = tile_load_f16::<5120, 16160>(a1);
    let o = tile_matmul_f16::<16, 5120, 16160>(a, b);
    tile_store_f16::<16, 16160>(a2, o);
}
