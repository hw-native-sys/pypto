// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// argmax over one UB-fitting block of the vocabulary.
//
// The captured corpus takes the whole 151936-wide row as one tile — 607744B against a
// 196608B UB — and stores 151936 values for a single result. Both emitters refuse it.
//
// 16384 columns is the largest power-of-two block that fits: the reduction allocates a
// second tile on top of the input, and 32768 missed by 512B (PTO reported a 197120B
// working set against 196608B). 16384/8 = 2048 elems = 8192B per fractal, and
// 8192 % 512 == 0, which is PTO's C2 stride rule.
//
// The full vocabulary is covered by invoking this per block and reducing the per-block
// results on the host: a running cross-block max needs scalar carry between tiles, which
// the tile intrinsics do not express.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::tile::{__tile_argmax_f32, __tile_load_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn argmax_block(x: GmView<1, 16384, f32>, out: GmViewMut<1, 1, f32>) {
    unsafe {
        let t = __tile_load_f32(x.as_ptr(), 1, 16384);
        let r = __tile_argmax_f32(0, t, 1, 16384);
        // One value out, not one per column — that is the other half of the fix.
        __tile_store_f32(out.as_mut_ptr(), r, 1, 1);
    }
}
