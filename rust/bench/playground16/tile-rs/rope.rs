// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Rotary position embedding, applied in place over 12 heads x 128 dims.
//
// The intrinsic is declared __tile_rope_inplace_f32(dst, x, rows, cols) and the capture
// passes the same tile as both dst and x, which is what "in place" means here. The store
// goes back to x rather than to a separate output for the same reason.
//
// One GmViewMut and as_mut_ptr for BOTH the load and the store: GmView is the read-only
// view and has as_ptr, GmViewMut has only as_mut_ptr, and an in-place kernel needs the
// single mutable view rather than one of each aliasing the same memory.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;use tile_std::tile::{__tile_load_f32, __tile_rope_inplace_f32, __tile_store_f32, GmViewMut};

#[tile_std::tile_kernel]
pub fn rope(x: GmViewMut<12, 128, f32>) {
    unsafe {
        let t = __tile_load_f32(x.as_mut_ptr(), 12, 128);
        let r = __tile_rope_inplace_f32(t, t, 12, 128);
        __tile_store_f32(x.as_mut_ptr(), r, 12, 128);
    }
}
