/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensormap_and_ringbuffer/common/intrinsic.h"
#include "tensormap_and_ringbuffer/runtime/tensor.h"

using pto::BLayout;
using pto::GlobalTensor;
using pto::Layout;
using pto::RoundMode;
using pto::Shape;
using pto::SLayout;
using pto::TASSIGN;
using pto::TCVT;
using pto::Tile;
using pto::TileType;
using pto::TLOAD;
using pto::TSTORE;
constexpr int K = 32;
constexpr int D = 512;
template <typename T, int C>
using GM = GlobalTensor<T, Shape<1, 1, 1, -1, C>, pto::Stride<-1, -1, -1, C, 1>, Layout::ND>;
template <typename T, int C>
__aicore__ inline GM<T, C> gm(__gm__ T* p) {
  return GM<T, C>(p, Shape<1, 1, 1, -1, C>(1, 1, 1, K, C),
                  pto::Stride<-1, -1, -1, C, 1>(K * C, K * C, K * C, C, 1));
}
template <typename T>
__aicore__ inline __gm__ T* data(__gm__ int64_t* args, int i) {
  auto t = reinterpret_cast<__gm__ Tensor*>(args[i]);
  return reinterpret_cast<__gm__ T*>(t->buffer.addr) + t->start_offset;
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
  set_atomic_none();
  auto out = data<bfloat16_t>(args, 0);
  auto input = data<uint8_t>(args, 1);
  int64_t rows = args[2];
  Tile<TileType::Vec, uint8_t, K, D / 2, BLayout::RowMajor, K, D / 2, SLayout::NoneBox> bytes;
  Tile<TileType::Vec, float4_e2m1x2_t, K, D, BLayout::RowMajor, K, D, SLayout::NoneBox> fp4;
  Tile<TileType::Vec, bfloat16_t, K, D, BLayout::RowMajor, K, D, SLayout::NoneBox> decoded;
  TASSIGN(decoded, 0);
  TASSIGN(bytes, K * D * 2);
  TASSIGN(fp4, K * D * 2);
  for (int64_t row = get_block_idx(args) * K; row < rows; row += 32 * K) {
    auto input_view = gm<uint8_t, D / 2>(input + row * (D / 2));
    auto output_view = gm<bfloat16_t, D>(out + row * D);
    TLOAD(bytes, input_view);
    pipe_barrier(PIPE_ALL);
    TCVT(decoded, fp4, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_ALL);
    TSTORE(output_view, decoded);
    pipe_barrier(PIPE_ALL);
  }
}
