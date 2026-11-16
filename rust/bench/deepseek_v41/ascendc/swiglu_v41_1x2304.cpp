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

class Kernel_bm_swiglu_1_2304 {
public:
    __aicore__ inline Kernel_bm_swiglu_1_2304() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 9216);
        pipe.InitBuffer(tbuf1, 9216);
        pipe.InitBuffer(tbuf2, 9216);
        pipe.InitBuffer(tbuf3, 9216);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        int32_t __v0;
        int32_t __v1;
        __v0 = 1;
        __v1 = 2304;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (2304));
        AscendC::DataCopy(buf1, gm1Tensor, (1) * (2304));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_silu: x / (1 + exp(-x))
          AscendC::Muls(buf2, buf0, (float)(-1), (__v0) * (__v1));
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Exp(buf2, buf2, (__v0) * (__v1));
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Adds(buf2, buf2, (float)(1), (__v0) * (__v1));
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Div(buf2, buf0, buf2, (__v0) * (__v1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(buf3, buf2, buf1, (__v0) * (__v1));
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v1)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf3, ((__v0) * (__v1))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm2Tensor, buf3, __sp); } }
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
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void bm_swiglu_1_2304(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_bm_swiglu_1_2304 op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void bm_swiglu_1_2304_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    bm_swiglu_1_2304<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
