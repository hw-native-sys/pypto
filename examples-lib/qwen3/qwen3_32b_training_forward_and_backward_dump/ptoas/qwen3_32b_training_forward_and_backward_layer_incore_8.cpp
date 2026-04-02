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

__global__ AICORE void qwen3_32b_training_forward_and_backward_layer_incore_8(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2) {
  RoundMode v3 = RoundMode::CAST_ROUND;
  unsigned v4 = 8;
  unsigned v5 = 4;
  unsigned v6 = 2;
  unsigned v7 = 1;
  unsigned v8 = 0;
  int32_t v9 = 8;
  int32_t v10 = 4;
  int32_t v11 = 2;
  int32_t v12 = 1;
  int64_t v13 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, bfloat16_t, 2, 4, BLayout::RowMajor, 2, 4, SLayout::NoneBox, 512, PadValue::Null> v14;
  TASSIGN(v14, v13);
  pto::Shape<1, 1, 1, 2, 4> v15 = pto::Shape<1, 1, 1, 2, 4>();
  pto::Stride<8, 8, 8, 4, 1> v16 = pto::Stride<8, 8, 8, 4, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 4>, pto::Stride<8, 8, 8, 4, 1>, pto::Layout::ND> v17 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 4>, pto::Stride<8, 8, 8, 4, 1>, pto::Layout::ND>(v1 + ((v8 + v8 * (unsigned) v9) + v8 * (unsigned) v10 + v8 * (unsigned) v12), v15, v16);
  TLOAD(v14, v17);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, bfloat16_t, 2, 4, BLayout::RowMajor, 2, 4, SLayout::NoneBox, 512, PadValue::Null> v18;
  TASSIGN(v18, v13);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(v18, v14, v3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 2, 4> v19 = pto::Shape<1, 1, 1, 2, 4>();
  pto::Stride<8, 8, 8, 4, 1> v20 = pto::Stride<8, 8, 8, 4, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 4>, pto::Stride<8, 8, 8, 4, 1>, pto::Layout::ND> v21 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 4>, pto::Stride<8, 8, 8, 4, 1>, pto::Layout::ND>(v2 + ((v8 + v8 * (unsigned) v9) + v8 * (unsigned) v10 + v8 * (unsigned) v12), v19, v20);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v21, v18);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  #endif // __DAV_VEC__

  return;
}

