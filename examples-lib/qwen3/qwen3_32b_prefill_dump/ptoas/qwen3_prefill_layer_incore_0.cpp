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

__global__ AICORE void qwen3_prefill_layer_incore_0(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, int32_t v4, int32_t v5, int32_t v6) {
  RoundMode v7 = RoundMode::CAST_ROUND;
  unsigned v8 = 5120;
  unsigned v9 = 20971520;
  unsigned v10 = 256;
  unsigned v11 = 4;
  unsigned v12 = 1;
  unsigned v13 = 0;
  int32_t v14 = 20971520;
  float v15 = 9.99999997E-7f;
  float v16 = 1.95312503E-4f;
  int32_t v17 = 256;
  int32_t v18 = 20;
  float v19 = 0.0f;
  int32_t v20 = 0;
  int32_t v21 = 4;
  int32_t v22 = 1;
  int32_t v23 = 5120;
  int32_t v24 = 4096;
  int32_t v25 = 16;
  int64_t v26 = 10336;
  int64_t v27 = 10304;
  int64_t v28 = 6208;
  int64_t v29 = 2112;
  int64_t v30 = 64;
  int64_t v31 = 32;
  int64_t v32 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v33;
  TASSIGN(v33, v32);
  pto::Shape<1, 1, 1, 4, 1> v34 = pto::Shape<1, 1, 1, 4, 1>();
  pto::Stride<4, 4, 4, 1, 4> v35 = pto::Stride<4, 4, 4, 1, 4>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v36 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>(v2 + (v13 + v13 * (unsigned) v22 + v13 * (unsigned) v21), v34, v35);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  TLOAD(v33, v36);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v37;
  TASSIGN(v37, v32);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v38;
  TASSIGN(v38, v31);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TMULS(v38, v37, v19);
  for (size_t v39 = (size_t) v20; v39 < ((size_t) v18); v39 += (size_t) v22) {
    Tile<TileType::Vec, bfloat16_t, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v40;
    TASSIGN(v40, v30);
    pto::Shape<1, 1, 1, 4, 256> v41 = pto::Shape<1, 1, 1, 4, 256>();
    pto::Stride<20971520, 20971520, 20971520, 5120, 1> v42 = pto::Stride<20971520, 20971520, 20971520, 5120, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<20971520, 20971520, 20971520, 5120, 1>, pto::Layout::ND> v43 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<20971520, 20971520, 20971520, 5120, 1>, pto::Layout::ND>(v1 + ((v13 + (unsigned) v4 * (unsigned) v14) + (unsigned) v5 * (unsigned) v23 + (unsigned) ((int32_t) (uint32_t) ((int32_t) v39) * (uint32_t) v17) * (unsigned) v22), v41, v42);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    TLOAD(v40, v43);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v44;
    TASSIGN(v44, v29);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    TCVT(v44, v40, v7);
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v45;
    TASSIGN(v45, v29);
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v46;
    TASSIGN(v46, v29);
    TMUL(v46, v45, v45);
    Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v47;
    TASSIGN(v47, v28);
    Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v48;
    TASSIGN(v48, v32);
    TROWSUM(v48, v46, v47);
    Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v49;
    TASSIGN(v49, v31);
    Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v50;
    TASSIGN(v50, v32);
    Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v51;
    TASSIGN(v51, v27);
    TADD(v51, v49, v50);
    Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v52;
    TASSIGN(v52, v27);
    Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v53;
    TASSIGN(v53, v26);
    TMOV(v53, v52);
  }
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v54;
  TASSIGN(v54, v27);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v55;
  TASSIGN(v55, v31);
  TMULS(v55, v54, v16);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v56;
  TASSIGN(v56, v31);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v57;
  TASSIGN(v57, v31);
  TADDS(v57, v56, v15);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v58;
  TASSIGN(v58, v31);
  Tile<TileType::Vec, float, 1, 4, BLayout::RowMajor, 1, 4, SLayout::NoneBox, 512, PadValue::Null> v59;
  TASSIGN(v59, v31);
  TRSQRT(v59, v58);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v60;
  TASSIGN(v60, v31);
  pto::Shape<1, 1, 1, 4, 1> v61 = pto::Shape<1, 1, 1, 4, 1>();
  pto::Stride<4, 4, 4, 1, 4> v62 = pto::Stride<4, 4, 4, 1, 4>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v63 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>(v3 + (v13 + v13 * (unsigned) v22 + v13 * (unsigned) v21), v61, v62);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v63, v60);
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

