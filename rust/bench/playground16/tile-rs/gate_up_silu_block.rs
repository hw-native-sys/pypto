// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// gate_up_silu over one block of the intermediate dimension.
//
// The captured corpus tiles all 8960 intermediate columns at once; six live tiles of
// 8960 f32 is 215040B against a 196608B UB.
//
// 2048 columns satisfies BOTH emitters, which budget the UB differently and were
// measured rather than guessed:
//   * PTO tracks a live working set — six tiles of 2048 f32 is 49152B.
//   * AscendC splits the UB into equal slots: "13440 bytes available per buffer
//     (188416 UB / 14 slots)", so a tile is at most 3360 f32 there. 4096 passed PTO and
//     was refused by AscendC.
//   * 2048/8 = 256 elems = 1024B per fractal, 1024 % 512 == 0, so PTO's C2 rule holds.
//
// The op is pointwise in the intermediate dimension, so blocking it is exact.
//
// The f16 SPELLING, not f32, and not by preference: mlir_to_pto has an arm for
// `__tile_gate_up_silu_f32` and none for `__tile_gate_up_silu_f32`, so the f32 twin is
// refused outright ("no arm for __tile_gate_up_silu_f32"). tile_std declares them as
// twins and the AscendC emitter "dispatches both spellings from one arm" —
// so the gap is on the PTO side and is a bug there, not a choice here.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::tile::{__tile_gate_up_silu_f32, __tile_load_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn gate_up_silu_block(
    a: GmView<1, 2048, f32>,
    w_gate: GmView<1, 2048, f32>,
    w_up: GmView<1, 2048, f32>,
    out: GmViewMut<1, 2048, f32>,
) {
    unsafe {
        let ta = __tile_load_f32(a.as_ptr(), 1, 2048);
        let tg = __tile_load_f32(w_gate.as_ptr(), 1, 2048);
        let tu = __tile_load_f32(w_up.as_ptr(), 1, 2048);
        let r = __tile_gate_up_silu_f32(0, ta, tg, tu, 1, 2048);
        __tile_store_f32(out.as_mut_ptr(), r, 1, 2048);
    }
}
