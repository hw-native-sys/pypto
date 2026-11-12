// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Grouped-query attention, with the operands in the order the intrinsic declares.
//
// The captured corpus calls it with the operands SHIFTED BY ONE. tile_std declares
// (crates/tile_std/src/tile.rs:262):
//
//     __tile_attention_gqa_f32(dst, q, k, v, heads_q, heads_kv, seq_len, head_dim)
//
// and the capture passes (tq, tk, tv, seq, dim, nh, nkv, causal): no dst, so q receives
// the k tile, k receives v, and v receives a constant 16. the AscendC emitter refuses it —
// "operand 3 (v) is %seq, which is not a tile" — while mlir_to_pto accepts it and emits
// attention over the wrong operands. Writing it in Rust makes the shift impossible: the
// declaration is the call's type.
//
// 192 = 12 query heads * 16 seq, 32 = 2 kv heads * 16 seq, head_dim 64.
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate tile_std;
use tile_std::tile::{__tile_attention_gqa_f32, __tile_load_f32, __tile_store_f32, GmView, GmViewMut};

#[tile_std::tile_kernel]
pub fn attn_gqa(
    q: GmView<192, 64, f32>,
    k: GmView<32, 64, f32>,
    v: GmView<32, 64, f32>,
    out: GmViewMut<192, 64, f32>,
) {
    unsafe {
        let tq = __tile_load_f32(q.as_ptr(), 192, 64);
        let tk = __tile_load_f32(k.as_ptr(), 32, 64);
        let tv = __tile_load_f32(v.as_ptr(), 32, 64);
        let r = __tile_attention_gqa_f32(0, tq, tk, tv, 12, 2, 16, 64);
        __tile_store_f32(out.as_mut_ptr(), r, 192, 64);
    }
}
