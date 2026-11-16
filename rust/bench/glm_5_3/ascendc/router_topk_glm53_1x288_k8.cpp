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

#include "kernel_operator.h"

#ifndef TILE_OUT_FILL
#define TILE_OUT_FILL 0
#endif

class Kernel_glm53_router_288_8 {
public:
    __aicore__ inline Kernel_glm53_router_288_8() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ int32_t*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 1152);
        pipe.InitBuffer(tbuf1, 1152);
        pipe.InitBuffer(tbuf2, 1152);
        pipe.InitBuffer(tbuf3, 32);
        pipe.InitBuffer(tbuf4, 32);
        pipe.InitBuffer(tbuf5, 32);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        AscendC::LocalTensor<int32_t> buf5 = tbuf5.Get<int32_t>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        __v0 = 1;
        __v1 = 288;
        __v2 = 8;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (288));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_topk: k passes of ReduceMax + first-hit index, masking each taken position
          int __tk_c = (288);
          int __tk_k = (8);
          AscendC::Muls(buf1, buf0, (float)(1.0f), __tk_c);
          AscendC::PipeBarrier<PIPE_ALL>();
          for (int __tk_j = 0; __tk_j < __tk_k; ++__tk_j) {
            AscendC::ReduceMax(buf3, buf1, buf2, __tk_c);
            AscendC::PipeBarrier<PIPE_ALL>();
            float __tk_best = buf3.GetValue(0);
            int __tk_idx = 0;
            for (int __tk_i = 0; __tk_i < __tk_c; ++__tk_i) {
              if (buf1.GetValue(__tk_i) == __tk_best) { __tk_idx = __tk_i; break; }
            }
            buf4.SetValue(__tk_j, __tk_best);
            buf5.SetValue(__tk_j, (int32_t)__tk_idx);
            // -FLT_MAX rather than a sentinel the data could contain: the next pass must
            // not be able to re-select this position whatever the inputs are.
            buf1.SetValue(__tk_idx, (float)(-3.402823466e+38f));
            AscendC::PipeBarrier<PIPE_ALL>();
          }
        }
        { const uint32_t __tk_sb = (uint32_t)((8) * sizeof(int32_t));
          if (__tk_sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf5, (8)); }
          else { AscendC::DataCopyExtParams __tk_sp{1, __tk_sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm2Tensor, buf5, __tk_sp); } }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm1Tensor, buf4, ((__v0) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm1Tensor, buf4, __sp); } }
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf3;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf4;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf5;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<int32_t> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void glm53_router_288_8(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_glm53_router_288_8 op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void glm53_router_288_8_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    glm53_router_288_8<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
