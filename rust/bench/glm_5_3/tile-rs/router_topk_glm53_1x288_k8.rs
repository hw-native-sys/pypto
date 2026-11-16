// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Hand-transcribed from router_topk_glm53_1x288_k8.tile (the tile-rs IR the
// generator writes); it mirrors the .tile's three calls exactly.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn glm53_router_288_8(a0: *mut f32, a1: *mut f32, a2: *mut u32) {
    let v = tile_load_f32::<1, 288>(a0);
    let o = tile_topk_f32::<1, 288, 8>(v, a2);
    tile_store_f32::<1, 8>(a1, o);
}
