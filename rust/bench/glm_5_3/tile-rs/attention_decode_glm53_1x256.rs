// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Hand-transcribed from attention_decode_glm53_1x256.tile: load Q 1x256, the
// blocked decode attention over S=2048 KV rows, store 1x256.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn glm53_attn_decode(a0: *mut f32, a1: *mut f32, a2: *mut f32, a3: *mut f32) {
    let q = tile_load_f32::<1, 256>(a0);
    let o = tile_attention_blocked_f32::<1, 256>(q, a1, a2, 2048);
    tile_store_f32::<1, 256>(a3, o);
}
