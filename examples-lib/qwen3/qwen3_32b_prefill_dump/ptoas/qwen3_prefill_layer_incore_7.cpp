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

__global__ AICORE void qwen3_prefill_layer_incore_7(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  unsigned v4 = 20480;
  unsigned v5 = 5120;
  unsigned v6 = 256;
  unsigned v7 = 4;
  unsigned v8 = 1;
  unsigned v9 = 0;
  float v10 = 9.99999997E-7f;
  float v11 = 1.95312503E-4f;
  int32_t v12 = 256;
  int32_t v13 = 20;
  float v14 = 0.0f;
  int32_t v15 = 0;
  int32_t v16 = 1;
  int32_t v17 = 5120;
  int32_t v18 = 4;
  int64_t v19 = 8288;
  int64_t v20 = 8256;
  int64_t v21 = 4160;
  int64_t v22 = 64;
  int64_t v23 = 32;
  int64_t v24 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v24);
  pto::Shape<1, 1, 1, 4, 1> v26 = pto::Shape<1, 1, 1, 4, 1>();
  pto::Stride<4, 4, 4, 1, 4> v27 = pto::Stride<4, 4, 4, 1, 4>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v28 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>(v2 + (v9 + v9 * (unsigned) v16 + v9 * (unsigned) v18), v26, v27);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  TLOAD(v25, v28);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v29;
  TASSIGN(v29, v24);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v30;
  TASSIGN(v30, v23);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMULS(v30, v29, v14);
  for (size_t v31 = (size_t) v15; v31 < ((size_t) v13); v31 += (size_t) v16) {
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v32;
    TASSIGN(v32, v22);
    pto::Shape<1, 1, 1, 4, 256> v33 = pto::Shape<1, 1, 1, 4, 256>();
    pto::Stride<20480, 20480, 20480, 5120, 1> v34 = pto::Stride<20480, 20480, 20480, 5120, 1>();
    GlobalTensor<float, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<20480, 20480, 20480, 5120, 1>, pto::Layout::ND> v35 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<20480, 20480, 20480, 5120, 1>, pto::Layout::ND>(v1 + (v9 + v9 * (unsigned) v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) v31) * (uint32_t) v12) * (unsigned) v16), v33, v34);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v32, v35);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v36;
    TASSIGN(v36, v22);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TMUL(v36, v32, v32);
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v37;
    TASSIGN(v37, v21);
    Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v38;
    TASSIGN(v38, v24);
    TROWSUM(v38, v36, v37);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v39;
    TASSIGN(v39, v23);
    Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v40;
    TASSIGN(v40, v24);
    Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v41;
    TASSIGN(v41, v20);
    TADD(v41, v39, v40);
    Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v42;
    TASSIGN(v42, v20);
    Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v43;
    TASSIGN(v43, v19);
    TMOV(v43, v42);
  }
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v44;
  TASSIGN(v44, v20);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v45;
  TASSIGN(v45, v23);
  TMULS(v45, v44, v11);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v46;
  TASSIGN(v46, v23);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v47;
  TASSIGN(v47, v23);
  TADDS(v47, v46, v10);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v48;
  TASSIGN(v48, v23);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v49;
  TASSIGN(v49, v23);
  TRSQRT(v49, v48);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v50;
  TASSIGN(v50, v23);
  pto::Shape<1, 1, 1, 4, 1> v51 = pto::Shape<1, 1, 1, 4, 1>();
  pto::Stride<4, 4, 4, 1, 4> v52 = pto::Stride<4, 4, 4, 1, 4>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v53 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>(v3 + (v9 + v9 * (unsigned) v16 + v9 * (unsigned) v18), v51, v52);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v53, v50);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

