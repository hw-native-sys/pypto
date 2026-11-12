#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(unused_unsafe, unused_variables, non_snake_case)]
use tile_std::tile::*;

#[no_mangle]
pub unsafe extern "C" fn bm_gemm_i8_16_5120_2304(
    a0: *const i8,
    a1: *const i8,
    a2: *mut u16,
    a3: *const f32,
) {
    let a = tile_load_i8::<16, 5120>(a0);
    let b = tile_load_i8::<5120, 2304>(a1);
    let o = tile_matmul_i8_acc_i32_dequant_f16::<16, 5120, 2304>(a, b, a3);
    tile_store_f16::<16, 2304>(a2, o);
}
