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
#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

extern "C" __global__ AICORE void tile_softmax(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1024;
  unsigned v4 = 1;
  unsigned v5 = 0;
  int32_t v6 = 1024;
  int32_t v7 = 1;
  int64_t v8 = 0;
  int64_t v9 = 4096;
  int64_t v10 = 16384;
  int64_t v11 = 4128;
  int64_t v12 = 8224;
  int64_t v13 = 12320;
  int64_t v14 = 12352;
  using T = float;
  pto::Shape<1, 1, 1, 1, 1024> v15 = pto::Shape<1, 1, 1, 1, 1024>();
  pto::Stride<1024, 1024, 1024, 1024, 1> v16 = pto::Stride<1024, 1024, 1024, 1024, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v17 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v6 + v5 * (unsigned) v7), v15, v16);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v8);
  TLOAD(v18, v17);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 8, 1, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v9);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v20;
  __ubuf__ float* v21 = v19.data();
  uint64_t v22 = reinterpret_cast<uint64_t>(v21);
  TASSIGN(v20, v22);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v23;
  TASSIGN(v23, v10);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v24;
  TASSIGN(v24, v11);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v12);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 8, 1, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v13);
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1, SLayout::NoneBox, 512, PadValue::Null> v27;
  __ubuf__ float* v28 = v26.data();
  uint64_t v29 = reinterpret_cast<uint64_t>(v28);
  TASSIGN(v27, v29);
  Tile<TileType::Vec, float, 1, 1024, BLayout::RowMajor, 1, 1024, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v14);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWMAX(v20, v18, v23);
  pipe_barrier(PIPE_V);
  TROWEXPANDSUB(v24, v18, v20);
  pipe_barrier(PIPE_V);
  TEXP(v25, v24);
  pipe_barrier(PIPE_V);
  TROWSUM(v27, v25, v23);
  pipe_barrier(PIPE_V);
  TROWEXPANDDIV(v30, v25, v27);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 1, 1024> v31 = pto::Shape<1, 1, 1, 1, 1024>();
  pto::Stride<1024, 1024, 1024, 1024, 1> v32 = pto::Stride<1024, 1024, 1024, 1024, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v33 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 1024>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v6 + v5 * (unsigned) v7), v31, v32);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v33, v30);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  return;
}
