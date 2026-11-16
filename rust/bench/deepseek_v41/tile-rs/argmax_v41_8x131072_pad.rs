// Copyright (c) PyPTO Contributors.
// Hand-transcribed from argmax_v41_8x131072_pad.tile: the streaming GM-plane row argmax over 8 rows
// of 131072 f32, writing 8 i32 indices. No tiles: the kernel streams rows itself.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn v41_argmax_8_131072(a0: *mut f32, a1: *mut i32) {
    __tile_argmax_rows_i32(a1, a0 as *const f32, 8, 131072);
}
