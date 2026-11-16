// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
#![feature(no_core)]
#![no_std]
#![no_core]

use tile_std::tile::{
    GmView, GmViewMut, safe, tile_load_view_f32, tile_store_view_f32,
};

// Row-wise softmax written against the safe GmView API.
//
// The `#[tile_kernel]` attribute now understands `GmView`/`GmViewMut` param
// types and injects the boundary prelude (`get_block_idx`, `GmDeviceCtx`,
// per-operand `view{,_mut}::<R,C,T>`) automatically. The kernel body
// therefore contains zero `unsafe` blocks.
//
// The FFI ABI is unchanged: `GmView` is `#[repr(transparent)]` over a
// raw pointer, and the macro rewrites the emitted signature back to
// `*const T` / `*mut T` before handing off to the codegen backend.

macro_rules! tile_softmax_kernel {
    ($name:ident, $rows:literal, $cols:literal) => {
        /// Row-wise softmax using the safe tile view API.
        ///
        /// Each block processes one tile of `ROWS × COLS` f32 values.
        #[tile_std::tile_kernel]
        pub fn $name(
            input:  GmView<'_, $rows, $cols, f32>,
            output: GmViewMut<'_, $rows, $cols, f32>,
        ) {
            let x = tile_load_view_f32(&input);
            let y = safe::tile_softmax_f32(x);
            tile_store_view_f32(&output, y);
        }
    };
}

// 1D softmax: 1 row × 1024 cols
tile_softmax_kernel!(tile_softmax, 1, 1024);
tile_softmax_kernel!(tile_softmax_safe, 1, 1024);

// Direct shape (B) instance kept as an explicit reference; identical
// expansion to the macro-generated `tile_softmax` / `_safe` above.
#[tile_std::tile_kernel]
pub fn tile_softmax_view(
    input:  GmView<'_, 1, 1024, f32>,
    output: GmViewMut<'_, 1, 1024, f32>,
) {
    let x = tile_load_view_f32(&input);
    let y = safe::tile_softmax_f32(x);
    tile_store_view_f32(&output, y);
}
