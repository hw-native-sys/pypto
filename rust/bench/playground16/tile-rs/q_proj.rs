// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// q_proj: the decode-time matvec, 1x1536. Loads are f32 and the matvec is the f16 kernel,
// exactly as the captured corpus has it.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;use tile_std::tile::{__tile_load_f32, __tile_matvec_f16, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn q_proj(a: GmView<1, 1536, f32>, w: GmView<1, 1536, f32>, out: GmViewMut<1, 1536, f32>) {
    unsafe {
        let at = __tile_load_f32(a.as_ptr(), 1, 1536);
        let wt = __tile_load_f32(w.as_ptr(), 1, 1536);
        let y = __tile_matvec_f16(0, at, wt, 1, 1536);
        __tile_store_f32(out.as_mut_ptr(), y, 1, 1536);
    }
}
