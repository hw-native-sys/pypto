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

__global__ AICORE void qwen3_prefill_layer_incore_3(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 5120;
  unsigned v4 = 1;
  unsigned v5 = 0;
  float v6 = 0.0f;
  int32_t v7 = 5120;
  int32_t v8 = 1;
  int64_t v9 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 5120, BLayout::RowMajor, 1, 5120, SLayout::NoneBox, 512, PadValue::Null> v10;
  TASSIGN(v10, v9);
  pto::Shape<1, 1, 1, 1, 5120> v11 = pto::Shape<1, 1, 1, 1, 5120>();
  pto::Stride<5120, 5120, 5120, 5120, 1> v12 = pto::Stride<5120, 5120, 5120, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 5120>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v13 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 5120>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v1 + (v5 + v5 * (unsigned) v7 + v5 * (unsigned) v8), v11, v12);
  TLOAD(v10, v13);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 1, 5120, BLayout::RowMajor, 1, 5120, SLayout::NoneBox, 512, PadValue::Null> v14;
  TASSIGN(v14, v9);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMULS(v14, v10, v6);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 1, 5120> v15 = pto::Shape<1, 1, 1, 1, 5120>();
  pto::Stride<5120, 5120, 5120, 5120, 1> v16 = pto::Stride<5120, 5120, 5120, 5120, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 5120>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v17 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 5120>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v2 + (v5 + v5 * (unsigned) v7 + v5 * (unsigned) v8), v15, v16);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v17, v14);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  #endif // __DAV_VEC__

  return;
}

