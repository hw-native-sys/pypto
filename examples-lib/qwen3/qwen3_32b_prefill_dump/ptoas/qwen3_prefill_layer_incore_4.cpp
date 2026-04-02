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

__global__ AICORE void qwen3_prefill_layer_incore_4(__gm__ float* v1, __gm__ float* v2, __gm__ bfloat16_t* v3, __gm__ bfloat16_t* v4, __gm__ float* v5, __gm__ float* v6, __gm__ bfloat16_t* v7, __gm__ bfloat16_t* v8, int32_t v9, int32_t v10, int32_t v11, int32_t v12) {
  RoundMode v13 = RoundMode::CAST_ROUND;
  unsigned v14 = 1024;
  unsigned v15 = 128;
  unsigned v16 = 64;
  unsigned v17 = 1;
  unsigned v18 = 0;
  int32_t v19 = 4096;
  int32_t v20 = 8;
  int32_t v21 = 0;
  int32_t v22 = 1024;
  int32_t v23 = 4;
  int32_t v24 = 128;
  int32_t v25 = 524288;
  int32_t v26 = 64;
  int32_t v27 = 1;
  int64_t v28 = 2560;
  int64_t v29 = 2304;
  int64_t v30 = 1792;
  int64_t v31 = 1280;
  int64_t v32 = 1024;
  int64_t v33 = 768;
  int64_t v34 = 512;
  int64_t v35 = 256;
  int64_t v36 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v37;
  TASSIGN(v37, v36);
  pto::Shape<1, 1, 1, 1, 64> v38 = pto::Shape<1, 1, 1, 1, 64>();
  pto::Stride<64, 64, 64, 64, 1> v39 = pto::Stride<64, 64, 64, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v40 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v1 + (v18 + v18 * (unsigned) v26 + v18 * (unsigned) v27), v38, v39);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  TLOAD(v37, v40);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v41;
  TASSIGN(v41, v35);
  pto::Shape<1, 1, 1, 1, 64> v42 = pto::Shape<1, 1, 1, 1, 64>();
  pto::Stride<64, 64, 64, 64, 1> v43 = pto::Stride<64, 64, 64, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v44 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v2 + (v18 + v18 * (unsigned) v26 + v18 * (unsigned) v27), v42, v43);
  TLOAD(v41, v44);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v45;
  TASSIGN(v45, v34);
  pto::Shape<1, 1, 1, 1, 64> v46 = pto::Shape<1, 1, 1, 1, 64>();
  pto::Stride<64, 64, 64, 64, 1> v47 = pto::Stride<64, 64, 64, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v48 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v5 + (v18 + v18 * (unsigned) v26 + v18 * (unsigned) v27), v46, v47);
  TLOAD(v45, v48);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v49;
  TASSIGN(v49, v33);
  pto::Shape<1, 1, 1, 1, 64> v50 = pto::Shape<1, 1, 1, 1, 64>();
  pto::Stride<64, 64, 64, 64, 1> v51 = pto::Stride<64, 64, 64, 64, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND> v52 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 64>, pto::Stride<64, 64, 64, 64, 1>, pto::Layout::ND>(v6 + (v18 + v18 * (unsigned) v26 + v18 * (unsigned) v27), v50, v51);
  TLOAD(v49, v52);
  for (size_t v53 = (size_t) v21; v53 < ((size_t) v23); v53 += (size_t) v27) {
    int32_t v54 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) v10 * (uint32_t) v23) + (uint32_t) ((int32_t) v53));
    int32_t v55 = (int32_t) ((uint32_t) v54 * (uint32_t) v24);
    Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v56;
    TASSIGN(v56, v32);
    pto::Shape<1, 1, 1, 1, 128> v57 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<1024, 1024, 1024, 1024, 1> v58 = pto::Stride<1024, 1024, 1024, 1024, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v59 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v4 + (v18 + (unsigned) v12 * (unsigned) v22 + (unsigned) v55 * (unsigned) v27), v57, v58);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(v56, v59);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v60;
    TASSIGN(v60, v31);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(v60, v56, v13);
    set_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v61;
    TASSIGN(v61, v31);
    wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
    TEXTRACT(v61, v60, v21, v21);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v62;
    TASSIGN(v62, v31);
    pipe_barrier(PIPE_MTE1);
    TEXTRACT(v62, v60, v21, v26);
    set_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);
    Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v63;
    TASSIGN(v63, v30);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v64;
    TASSIGN(v64, v29);
    wait_flag(PIPE_MTE1, PIPE_V, EVENT_ID0);
    TCOLEXPANDMUL(v64, v61, v41);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v65;
    TASSIGN(v65, v28);
    TCOLEXPANDMUL(v65, v62, v49);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v66;
    TASSIGN(v66, v29);
    TSUB(v66, v64, v65);
    Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v67;
    TASSIGN(v67, v30);
    TMOV(v67, v63);
    TINSERT(v67, v66, v21, v21);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v68;
    TASSIGN(v68, v29);
    TCOLEXPANDMUL(v68, v62, v37);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v69;
    TASSIGN(v69, v28);
    TCOLEXPANDMUL(v69, v61, v45);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v70;
    TASSIGN(v70, v29);
    TADD(v70, v68, v69);
    Tile<TileType::Vec, float, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v71;
    TASSIGN(v71, v30);
    TMOV(v71, v67);
    TINSERT(v71, v70, v21, v26);
    int32_t v72 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v9 * (uint32_t) v20) * (uint32_t) v19) + (uint32_t) ((int32_t) (uint32_t) v54 * (uint32_t) v19)) + (uint32_t) v11);
    Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v73;
    TASSIGN(v73, v32);
    TCVT(v73, v71, v13);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::Shape<1, 1, 1, 1, 128> v74 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v75 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v76 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v3 + (v18 + (unsigned) v72 * (unsigned) v24 + v18 * (unsigned) v27), v74, v75);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(v76, v73);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    Tile<TileType::Vec, bfloat16_t, 1, 128, BLayout::RowMajor, 1, 128, SLayout::NoneBox, 512, PadValue::Null> v77;
    TASSIGN(v77, v32);
    pto::Shape<1, 1, 1, 1, 128> v78 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<1024, 1024, 1024, 1024, 1> v79 = pto::Stride<1024, 1024, 1024, 1024, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND> v80 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<1024, 1024, 1024, 1024, 1>, pto::Layout::ND>(v8 + (v18 + (unsigned) v12 * (unsigned) v22 + (unsigned) v55 * (unsigned) v27), v78, v79);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    TLOAD(v77, v80);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    pto::Shape<1, 1, 1, 1, 128> v81 = pto::Shape<1, 1, 1, 1, 128>();
    pto::Stride<128, 128, 128, 128, 1> v82 = pto::Stride<128, 128, 128, 128, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND> v83 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 1, 128>, pto::Stride<128, 128, 128, 128, 1>, pto::Layout::ND>(v7 + (v18 + (unsigned) v72 * (unsigned) v24 + v18 * (unsigned) v27), v81, v82);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(v83, v77);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  }
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

