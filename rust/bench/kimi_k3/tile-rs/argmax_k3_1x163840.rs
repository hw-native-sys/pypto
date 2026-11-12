// Copyright (c) PyPTO Contributors.
// Hand-transcribed from argmax_k3_1x163840.tile: load 1x163840, the row argmax, store the 1x1 i32
// index. The raw externs are used because __tile_argmax_f32 and
// __tile_store_i32 disagree on u32-vs-i32 at the safe-wrapper type.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn k3_argmax_163840(a0: *mut f32, a1: *mut i32) {
    let v = __tile_load_f32(a0 as *const f32, 1, 163840);
    let o = __tile_argmax_f32(0, v, 1, 163840);
    __tile_store_i32(a1, o, 1, 1);
}
