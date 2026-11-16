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

class Kernel_bm_quant_i8_8_5120 {
public:
    __aicore__ inline Kernel_bm_quant_i8_8_5120() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ int8_t*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 20480);
        pipe.InitBuffer(tbuf1, 20480);
        pipe.InitBuffer(tbuf2, 20480);
        pipe.InitBuffer(tbuf3, 20480);
        pipe.InitBuffer(tbuf4, 20480);
        pipe.InitBuffer(tbuf5, 20480);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<half> buf4 = tbuf4.Get<half>();
        AscendC::LocalTensor<int8_t> buf5 = tbuf5.Get<int8_t>();
        // chunked: 40960 elements as 8 x 5120, tail 5120
        constexpr int __CH = 5120;
        constexpr int __NW = 40960;
        constexpr int __NC = 8;
        for (int __c = 0; __c < __NC; ++__c) {
          const int __n = (__c == __NC - 1) ? (__NW - (__NC - 1) * __CH) : __CH;
          const int __off = __c * __CH;
          gm0Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm0_raw + __off, __NW - __off);
          gm1Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm1_raw, 1);
          gm2Tensor.SetGlobalBuffer((__gm__ int8_t*)(uintptr_t)gm2_raw + __off, __NW - __off);
          int32_t __v0;
          int32_t __v1;
          int32_t __v2;
          __v0 = 8;
          __v1 = 5120;
          __v2 = 1;
          AscendC::DataCopy(buf0, gm0Tensor, __n);
          { AscendC::DataCopyExtParams __ep{1, (uint32_t)(((1) * (1)) * sizeof(float)), 0, 0, 0}; AscendC::DataCopyPadExtParams<float> __pp{false, 0, 0, 0}; AscendC::DataCopyPad(buf1, gm1Tensor, __ep, __pp); }
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Duplicate(buf2, buf1.GetValue(0), __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Div(buf3, buf0, buf2, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Cast(buf4, buf3, AscendC::RoundMode::CAST_NONE, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Cast(buf5, buf4, AscendC::RoundMode::CAST_ROUND, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          { const uint32_t __sb = (uint32_t)(__n * sizeof(int8_t));
            if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf5, __n); }
            else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                   AscendC::DataCopyPad(gm2Tensor, buf5, __sp); } }
        }
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
    AscendC::GlobalTensor<int8_t> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void bm_quant_i8_8_5120(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_bm_quant_i8_8_5120 op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void bm_quant_i8_8_5120_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    bm_quant_i8_8_5120<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
