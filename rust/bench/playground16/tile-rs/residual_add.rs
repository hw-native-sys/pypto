// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Element-wise add (the decode residual, 1536-d hidden state) over 1536 f32, 1x1536 as the captured corpus has it.
// One of the five teaching primitives.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;use tile_std::tile::{__tile_add_f32, __tile_load_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn residual_add(a: GmView<1, 1536, f32>, b: GmView<1, 1536, f32>, out: GmViewMut<1, 1536, f32>) {
    unsafe {
        let at = __tile_load_f32(a.as_ptr(), 1, 1536);
        let bt = __tile_load_f32(b.as_ptr(), 1, 1536);
        let y = __tile_add_f32(0, at, bt, 1, 1536);
        __tile_store_f32(out.as_mut_ptr(), y, 1, 1536);
    }
}
