// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// RMS-norm over the 1536-d hidden state: out = x * rsqrt(mean(x^2)+eps) * gamma.
// NOT LayerNorm -- no mean subtraction and no beta, which is the norm Qwen/LLaMA use.
//
// n4 = 1536/4 and row_stride = 1536*4 bytes are the float4 vectorisation parameters the
// intrinsic declares; the captured corpus passes 384 and 6144, which is the same thing.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;use tile_std::tile::{__tile_load_f32, __tile_rms_norm_mul_f32_4, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn rms_norm(x: GmView<1, 1536, f32>, g: GmView<1, 1536, f32>, out: GmViewMut<1, 1536, f32>) {
    unsafe {
        let tx = __tile_load_f32(x.as_ptr(), 1, 1536);
        let tg = __tile_load_f32(g.as_ptr(), 1, 1536);
        let r = __tile_rms_norm_mul_f32_4(tx, tg, 1536, 384, 6144, 0.000001);
        __tile_store_f32(out.as_mut_ptr(), r, 1, 1536);
    }
}
