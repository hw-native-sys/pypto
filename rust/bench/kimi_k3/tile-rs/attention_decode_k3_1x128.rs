// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Hand-transcribed from attention_decode_k3_1x128.tile: load Q 1x128 and the
// runtime KV length tile, the dynamic decode attention, store 1x128.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn k3_attn_decode(a0: *mut f32, a1: *mut f32, a2: *mut f32, a3: *mut f32, a4: *mut f32) {
    let q = tile_load_f32::<1, 128>(a0);
    let st = tile_load_f32::<1, 1>(a3);
    let o = tile_attention_dyn_f32::<1, 128>(q, a1, a2, st);
    tile_store_f32::<1, 128>(a4, o);
}
