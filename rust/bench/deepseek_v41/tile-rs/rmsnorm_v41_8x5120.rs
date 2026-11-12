// Copyright (c) PyPTO Contributors.
// Hand-transcribed from rmsnorm_v41_8x5120.tile: load 8x5120, the fused
// RMSNorm (eps = 1e-20), store 8x5120.
#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn v41_rmsnorm_8_5120(a0: *mut f32, a1: *mut f32) {
    let x = tile_load_f32::<8, 5120>(a0);
    let o = tile_rms_norm_f32::<8, 5120>(x, 1e-20f32);
    tile_store_f32::<8, 5120>(a1, o);
}
