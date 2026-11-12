// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// f16 matmul, C(1x64) = A(1x64) @ B(64x64). One of the five teaching primitives.
//
// WORTH READING TWICE. tile_std declares
//
//     __tile_matmul_f16(dst, a, b, m, k, n)
//
// six operands, and the captured corpus passes FIVE -- (ta, tb, r, c, c) -- so every
// operand is shifted by one: a receives the b tile, b receives m, and m receives k. Same
// defect class as attn_gqa, and the same point: writing the kernel in Rust makes the shift
// impossible, because the declaration is the call's type.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;use tile_std::tile::{__tile_load_f16, __tile_matmul_f16, __tile_store_f16, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn matmul(a: GmView<1, 64, u16>, b: GmView<64, 64, u16>, out: GmViewMut<1, 64, u16>) {
    unsafe {
        let ta = __tile_load_f16(a.as_ptr(), 1, 64);
        let tb = __tile_load_f16(b.as_ptr(), 64, 64);
        let c = __tile_matmul_f16(0, ta, tb, 1, 64, 64);
        __tile_store_f16(out.as_mut_ptr(), c, 1, 64);
    }
}
