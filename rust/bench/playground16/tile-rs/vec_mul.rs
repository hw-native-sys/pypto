// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Element-wise multiply over 256 f32, 1x256 as the captured corpus has it.
// One of the five teaching primitives.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;use tile_std::tile::{__tile_mul_f32, __tile_load_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn vec_mul(a: GmView<1, 256, f32>, b: GmView<1, 256, f32>, out: GmViewMut<1, 256, f32>) {
    unsafe {
        let at = __tile_load_f32(a.as_ptr(), 1, 256);
        let bt = __tile_load_f32(b.as_ptr(), 1, 256);
        let y = __tile_mul_f32(0, at, bt, 1, 256);
        __tile_store_f32(out.as_mut_ptr(), y, 1, 256);
    }
}
