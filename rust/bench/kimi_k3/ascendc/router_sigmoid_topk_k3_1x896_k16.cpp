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

class Kernel_k3_router_896_16 {
public:
    __aicore__ inline Kernel_k3_router_896_16() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ int32_t*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 3584);
        pipe.InitBuffer(tbuf1, 3584);
        pipe.InitBuffer(tbuf2, 3584);
        pipe.InitBuffer(tbuf3, 3584);
        pipe.InitBuffer(tbuf4, 3584);
        pipe.InitBuffer(tbuf5, 3584);
        pipe.InitBuffer(tbuf6, 3584);
        pipe.InitBuffer(tbuf7, 32);
        pipe.InitBuffer(tbuf8, 64);
        pipe.InitBuffer(tbuf9, 64);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        AscendC::LocalTensor<float> buf5 = tbuf5.Get<float>();
        AscendC::LocalTensor<float> buf6 = tbuf6.Get<float>();
        AscendC::LocalTensor<float> buf7 = tbuf7.Get<float>();
        AscendC::LocalTensor<float> buf8 = tbuf8.Get<float>();
        AscendC::LocalTensor<int32_t> buf9 = tbuf9.Get<int32_t>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 0;
        __v1 = 1;
        __v2 = 896;
        __v3 = 16;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (896));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_sigmoid: 1 / (1 + exp(-g)), the form that saturates instead of overflowing
          AscendC::Muls(buf1, buf0, (float)(-1.0f), (1) * (896));
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(buf1, buf1, (1) * (896));
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf2, buf1, (float)(1.0f), (1) * (896));
          AscendC::Duplicate(buf3, (float)(1.0f), (1) * (896));
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Div(buf4, buf3, buf2, (1) * (896));
        }
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_topk: k passes of ReduceMax + first-hit index, masking each taken position
          int __tk_c = (896);
          int __tk_k = (16);
          AscendC::Muls(buf5, buf4, (float)(1.0f), __tk_c);
          AscendC::PipeBarrier<PIPE_ALL>();
          for (int __tk_j = 0; __tk_j < __tk_k; ++__tk_j) {
            AscendC::ReduceMax(buf7, buf5, buf6, __tk_c);
            AscendC::PipeBarrier<PIPE_ALL>();
            float __tk_best = buf7.GetValue(0);
            int __tk_idx = 0;
            for (int __tk_i = 0; __tk_i < __tk_c; ++__tk_i) {
              if (buf5.GetValue(__tk_i) == __tk_best) { __tk_idx = __tk_i; break; }
            }
            buf8.SetValue(__tk_j, __tk_best);
            buf9.SetValue(__tk_j, (int32_t)__tk_idx);
            // -FLT_MAX rather than a sentinel the data could contain: the next pass must
            // not be able to re-select this position whatever the inputs are.
            buf5.SetValue(__tk_idx, (float)(-3.402823466e+38f));
            AscendC::PipeBarrier<PIPE_ALL>();
          }
        }
        { const uint32_t __tk_sb = (uint32_t)((16) * sizeof(int32_t));
          if (__tk_sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf9, (16)); }
          else { AscendC::DataCopyExtParams __tk_sp{1, __tk_sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm2Tensor, buf9, __tk_sp); } }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v3)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm1Tensor, buf8, ((__v1) * (__v3))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm1Tensor, buf8, __sp); } }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf6;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf7;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf8;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf9;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<int32_t> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void k3_router_896_16(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_k3_router_896_16 op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void k3_router_896_16_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    k3_router_896_16<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
