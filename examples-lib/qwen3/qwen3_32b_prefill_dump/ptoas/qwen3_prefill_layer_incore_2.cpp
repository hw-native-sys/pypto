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

AICORE void qwen3_prefill_layer_incore_2_aic(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ bfloat16_t* v5, __gm__ bfloat16_t* v6, __gm__ bfloat16_t* v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11) {
  unsigned v12 = 262144;
  unsigned v13 = 1024;
  unsigned v14 = 32;
  unsigned v15 = 256;
  unsigned v16 = 1;
  unsigned v17 = 0;
  __gm__ void * v18 = nullptr;
  int32_t v19 = 256;
  int32_t v20 = 20;
  int32_t v21 = 32;
  int32_t v22 = 8;
  int32_t v23 = 1024;
  int32_t v24 = 4;
  int32_t v25 = 1;
  int32_t v26 = 5120;
  int64_t v27 = 16384;
  int64_t v28 = 10240;
  int64_t v29 = 0;
  int32_t v30 = 0;
  using T = float;

  #if defined(__DAV_CUBE__)
  size_t v31 = (size_t) v25;
  size_t v32 = (size_t) v30;
  auto v33 = TPipe<0, Direction::DIR_C2V, 2048, 4>(v18, v30, v30);
  auto v34 = TPipe<2, Direction::DIR_V2C, 2048, 4>(v18, v30, v30);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  for (size_t v35 = v32; v35 < ((size_t) v22); v35 += v31) {
    int32_t v36 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v9 * (uint32_t) v22) + (uint32_t) ((int32_t) v35)) * (uint32_t) v21);
    for (size_t v37 = v32; v37 < ((size_t) v20); v37 += v31) {
      int32_t v38 = (int32_t) ((uint32_t) ((int32_t) v37) * (uint32_t) v19);
      Tile<TileType::Mat, bfloat16_t, 4, 256, BLayout::ColMajor, 4, 256, SLayout::RowMajor, 512, PadValue::Null> v39;
      TPOP<TPipe<2, Direction::DIR_V2C, 2048, 4>, Tile<TileType::Mat, bfloat16_t, 4, 256, BLayout::ColMajor, 4, 256, SLayout::RowMajor, 512, PadValue::Null>, TileSplitAxis::TILE_NO_SPLIT>(v34, v39);
      Tile<TileType::Left, bfloat16_t, 4, 256, BLayout::ColMajor, 4, 256, SLayout::RowMajor, 512, PadValue::Null> v40;
      TASSIGN(v40, v29);
      wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      TMOV(v40, v39);
      TFREE<TPipe<2, Direction::DIR_V2C, 2048, 4>, TileSplitAxis::TILE_NO_SPLIT>(v34);
      Tile<TileType::Mat, bfloat16_t, 256, 32, BLayout::ColMajor, 256, 32, SLayout::RowMajor, 512, PadValue::Null> v41;
      TASSIGN(v41, v28);
      pto::Shape<1, 1, 1, 256, 32> v42 = pto::Shape<1, 1, 1, 256, 32>();
      pto::Stride<262144, 262144, 262144, 1024, 1> v43 = pto::Stride<262144, 262144, 262144, 1024, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND> v44 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND>(v6 + (v17 + (unsigned) v38 * (unsigned) v23 + (unsigned) v36 * (unsigned) v25), v42, v43);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      TLOAD(v41, v44);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      Tile<TileType::Right, bfloat16_t, 256, 32, BLayout::RowMajor, 256, 32, SLayout::ColMajor, 512, PadValue::Null> v45;
      TASSIGN(v45, v29);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      TMOV(v45, v41);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      Tile<TileType::Mat, bfloat16_t, 256, 32, BLayout::ColMajor, 256, 32, SLayout::RowMajor, 512, PadValue::Null> v46;
      TASSIGN(v46, v28);
      pto::Shape<1, 1, 1, 256, 32> v47 = pto::Shape<1, 1, 1, 256, 32>();
      pto::Stride<262144, 262144, 262144, 1024, 1> v48 = pto::Stride<262144, 262144, 262144, 1024, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND> v49 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 256, 32>, pto::Stride<262144, 262144, 262144, 1024, 1>, pto::Layout::ND>(v7 + (v17 + (unsigned) v38 * (unsigned) v23 + (unsigned) v36 * (unsigned) v25), v47, v48);
      wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
      TLOAD(v46, v49);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      Tile<TileType::Right, bfloat16_t, 256, 32, BLayout::RowMajor, 256, 32, SLayout::ColMajor, 512, PadValue::Null> v50;
      TASSIGN(v50, v27);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
      TMOV(v50, v46);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
      Tile<TileType::Acc, float, 4, 32, BLayout::ColMajor, 4, 32, SLayout::RowMajor, 1024, PadValue::Null> v51;
      TASSIGN(v51, v29);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      TMATMUL(v51, v40, v45);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      TPUSH<TPipe<0, Direction::DIR_C2V, 2048, 4>, Tile<TileType::Acc, float, 4, 32, BLayout::ColMajor, 4, 32, SLayout::RowMajor, 1024, PadValue::Null>, TileSplitAxis::TILE_NO_SPLIT>(v33, v51);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      Tile<TileType::Acc, float, 4, 32, BLayout::ColMajor, 4, 32, SLayout::RowMajor, 1024, PadValue::Null> v52;
      TASSIGN(v52, v29);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
      TMATMUL(v52, v40, v50);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
      TPUSH<TPipe<0, Direction::DIR_C2V, 2048, 4>, Tile<TileType::Acc, float, 4, 32, BLayout::ColMajor, 4, 32, SLayout::RowMajor, 1024, PadValue::Null>, TileSplitAxis::TILE_NO_SPLIT>(v33, v52);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
    };
  }
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  #endif // __DAV_CUBE__

  return;
}

AICORE void qwen3_prefill_layer_incore_2_aiv(__gm__ bfloat16_t* v1, __gm__ float* v2, __gm__ float* v3, __gm__ bfloat16_t* v4, __gm__ bfloat16_t* v5, __gm__ bfloat16_t* v6, __gm__ bfloat16_t* v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11) {
  unsigned v12 = 4096;
  unsigned v13 = 1024;
  unsigned v14 = 32;
  RoundMode v15 = RoundMode::CAST_ROUND;
  unsigned v16 = 5120;
  unsigned v17 = 20971520;
  unsigned v18 = 256;
  unsigned v19 = 4;
  unsigned v20 = 1;
  unsigned v21 = 0;
  __gm__ void * v22 = nullptr;
  int32_t v23 = 20971520;
  int32_t v24 = 256;
  int32_t v25 = 20;
  float v26 = 0.0f;
  int32_t v27 = 32;
  int32_t v28 = 8;
  int32_t v29 = 1024;
  int32_t v30 = 4;
  int32_t v31 = 1;
  int32_t v32 = 5120;
  int32_t v33 = 4096;
  int32_t v34 = 16;
  int64_t v35 = 17952;
  int64_t v36 = 17440;
  int64_t v37 = 16928;
  int64_t v38 = 15392;
  int64_t v39 = 11296;
  int64_t v40 = 9248;
  int64_t v41 = 8736;
  int64_t v42 = 8224;
  int64_t v43 = 8192;
  int32_t v44 = 0;
  using T = float;

  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  size_t v45 = (size_t) v31;
  size_t v46 = (size_t) v44;
  auto v47 = TPipe<0, Direction::DIR_C2V, 2048, 4>(v22, v44, v44);
  auto v48 = TPipe<2, Direction::DIR_V2C, 2048, 4>(v22, v44, v44);
  Tile<TileType::Vec, float, 4, 1, BLayout::ColMajor, 4, 1, SLayout::NoneBox, 512, PadValue::Null> v49;
  TASSIGN(v49, v43);
  pto::Shape<1, 1, 1, 4, 1> v50 = pto::Shape<1, 1, 1, 4, 1>();
  pto::Stride<4, 4, 4, 1, 4> v51 = pto::Stride<4, 4, 4, 1, 4>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN> v52 = GlobalTensor<float, pto::Shape<1, 1, 1, 4, 1>, pto::Stride<4, 4, 4, 1, 4>, pto::Layout::DN>(v3 + (v21 + v21 * (unsigned) v31 + v21 * (unsigned) v30), v50, v51);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  TLOAD(v49, v52);
  for (size_t v53 = v46; v53 < ((size_t) v28); v53 += v45) {
    int32_t v54 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v9 * (uint32_t) v28) + (uint32_t) ((int32_t) v53)) * (uint32_t) v27);
    Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v55;
    TASSIGN(v55, v42);
    Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v56;
    TASSIGN(v56, v41);
    Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v57;
    TASSIGN(v57, v42);
    TMULS(v57, v55, v26);
    Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v58;
    TASSIGN(v58, v41);
    TMULS(v58, v56, v26);
    for (size_t v59 = v46; v59 < ((size_t) v25); v59 += v45) {
      int32_t v60 = (int32_t) ((uint32_t) ((int32_t) v59) * (uint32_t) v24);
      Tile<TileType::Vec, bfloat16_t, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v61;
      TASSIGN(v61, v40);
      pto::Shape<1, 1, 1, 4, 256> v62 = pto::Shape<1, 1, 1, 4, 256>();
      pto::Stride<20971520, 20971520, 20971520, 5120, 1> v63 = pto::Stride<20971520, 20971520, 20971520, 5120, 1>();
      GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<20971520, 20971520, 20971520, 5120, 1>, pto::Layout::ND> v64 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 256>, pto::Stride<20971520, 20971520, 20971520, 5120, 1>, pto::Layout::ND>(v1 + ((v21 + (unsigned) v8 * (unsigned) v23) + (unsigned) v10 * (unsigned) v32 + (unsigned) v60 * (unsigned) v31), v62, v63);
      wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      TLOAD(v61, v64);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v65;
      TASSIGN(v65, v39);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(v65, v61, v15);
      Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v66;
      TASSIGN(v66, v39);
      Tile<TileType::Vec, float, 1, 256, BLayout::RowMajor, 1, 256, SLayout::NoneBox, 512, PadValue::Null> v67;
      TASSIGN(v67, v38);
      pto::Shape<1, 1, 1, 1, 256> v68 = pto::Shape<1, 1, 1, 1, 256>();
      pto::Stride<5120, 5120, 5120, 5120, 1> v69 = pto::Stride<5120, 5120, 5120, 5120, 1>();
      GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND> v70 = GlobalTensor<float, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<5120, 5120, 5120, 5120, 1>, pto::Layout::ND>(v2 + (v21 + v21 * (unsigned) v32 + (unsigned) v60 * (unsigned) v31), v68, v69);
      TLOAD(v67, v70);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v71;
      TASSIGN(v71, v39);
      TROWEXPANDMUL(v71, v66, v49);
      Tile<TileType::Vec, float, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v72;
      TASSIGN(v72, v39);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCOLEXPANDMUL(v72, v71, v67);
      Tile<TileType::Vec, bfloat16_t, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null> v73;
      TASSIGN(v73, v40);
      TCVT(v73, v72, v15);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TPUSH<TPipe<2, Direction::DIR_V2C, 2048, 4>, Tile<TileType::Vec, bfloat16_t, 4, 256, BLayout::RowMajor, 4, 256, SLayout::NoneBox, 512, PadValue::Null>, TileSplitAxis::TILE_NO_SPLIT>(v48, v73);
      set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
      Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v74;
      TPOP<TPipe<0, Direction::DIR_C2V, 2048, 4>, Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null>, TileSplitAxis::TILE_NO_SPLIT>(v47, v74);
      Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v75;
      TASSIGN(v75, v37);
      TADD(v75, v57, v74);
      TFREE<TPipe<0, Direction::DIR_C2V, 2048, 4>, TileSplitAxis::TILE_NO_SPLIT>(v47);
      TPOP<TPipe<0, Direction::DIR_C2V, 2048, 4>, Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null>, TileSplitAxis::TILE_NO_SPLIT>(v47, v74);
      Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v76;
      TASSIGN(v76, v36);
      TADD(v76, v58, v74);
      TFREE<TPipe<0, Direction::DIR_C2V, 2048, 4>, TileSplitAxis::TILE_NO_SPLIT>(v47);
      Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v77;
      TASSIGN(v77, v42);
      TMOV(v77, v75);
      Tile<TileType::Vec, float, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v78;
      TASSIGN(v78, v41);
      TMOV(v78, v76);
    };
    Tile<TileType::Vec, bfloat16_t, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v79;
    TASSIGN(v79, v35);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    TCVT(v79, v57, v15);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    pto::Shape<1, 1, 1, 4, 32> v80 = pto::Shape<1, 1, 1, 4, 32>();
    pto::Stride<4096, 4096, 4096, 1024, 1> v81 = pto::Stride<4096, 4096, 4096, 1024, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 32>, pto::Stride<4096, 4096, 4096, 1024, 1>, pto::Layout::ND> v82 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 32>, pto::Stride<4096, 4096, 4096, 1024, 1>, pto::Layout::ND>(v4 + (v21 + v21 * (unsigned) v29 + (unsigned) v54 * (unsigned) v31), v80, v81);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
    TSTORE(v82, v79);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    Tile<TileType::Vec, bfloat16_t, 4, 32, BLayout::RowMajor, 4, 32, SLayout::NoneBox, 512, PadValue::Null> v83;
    TASSIGN(v83, v35);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    TCVT(v83, v58, v15);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    pto::Shape<1, 1, 1, 4, 32> v84 = pto::Shape<1, 1, 1, 4, 32>();
    pto::Stride<4096, 4096, 4096, 1024, 1> v85 = pto::Stride<4096, 4096, 4096, 1024, 1>();
    GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 32>, pto::Stride<4096, 4096, 4096, 1024, 1>, pto::Layout::ND> v86 = GlobalTensor<bfloat16_t, pto::Shape<1, 1, 1, 4, 32>, pto::Stride<4096, 4096, 4096, 1024, 1>, pto::Layout::ND>(v5 + (v21 + v21 * (unsigned) v29 + (unsigned) v54 * (unsigned) v31), v84, v85);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    TSTORE(v86, v83);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
  ptoas_auto_sync_tail(PTOAutoSyncTailMode::kBarrierAll);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  #endif // __DAV_VEC__

  return;
}

