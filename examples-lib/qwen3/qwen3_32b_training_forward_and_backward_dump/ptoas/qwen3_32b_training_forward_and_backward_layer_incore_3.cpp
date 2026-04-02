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

__global__ AICORE void qwen3_32b_training_forward_and_backward_layer_incore_3(__gm__ bfloat16_t* v1, __gm__ bfloat16_t* v2) {
  RoundMode v3 = RoundMode::CAST_ROUND;
  unsigned v4 = 32;
  unsigned v5 = 16;
  unsigned v6 = 2;
  unsigned v7 = 1;
  unsigned v8 = 0;
  int32_t v9 = 1;
  int32_t v10 = 16;
  int32_t v11 = 2;
  int64_t v12 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, bfloat16_t, 2, 16, BLayout::RowMajor, 2, 16, SLayout::NoneBox, 512, PadValue::Null> v13;
  TASSIGN(v13, v12);
  pto::Shape<1, 1, 1, 2, 16> v14 = pto::Shape<1, 1, 1, 2, 16>();
  pto::Stride<32, 32, 32, 16, 1> v15 = pto::Stride<32, 32, 32, 16, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>, pto::Layout::ND> v16 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>, pto::Layout::ND>(v1 + (v8 + v8 * (unsigned) v10 + v8 * (unsigned) v9), v14, v15);
  TLOAD(v13, v16);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, bfloat16_t, 2, 16, BLayout::RowMajor, 2, 16, SLayout::NoneBox, 512, PadValue::Null> v17;
  TASSIGN(v17, v12);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(v17, v13, v3);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 2, 16> v18 = pto::Shape<1, 1, 1, 2, 16>();
  pto::Stride<32, 32, 32, 16, 1> v19 = pto::Stride<32, 32, 32, 16, 1>();
  GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>, pto::Layout::ND> v20 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 2, 16>, pto::Stride<32, 32, 32, 16, 1>, pto::Layout::ND>(v2 + (v8 + v8 * (unsigned) v10 + v8 * (unsigned) v9), v18, v19);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v20, v17);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  #endif // __DAV_VEC__

  return;
}

