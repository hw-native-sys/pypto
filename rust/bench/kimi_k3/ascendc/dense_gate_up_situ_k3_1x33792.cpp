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

class Kernel_k3_dense_situ {
public:
    __aicore__ inline Kernel_k3_dense_situ() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 8448);
        pipe.InitBuffer(tbuf1, 8448);
        pipe.InitBuffer(tbuf2, 8448);
        pipe.InitBuffer(tbuf3, 8448);
        pipe.InitBuffer(tbuf4, 8448);
        pipe.InitBuffer(tbuf5, 8448);
        pipe.InitBuffer(tbuf6, 8448);
        pipe.InitBuffer(tbuf7, 8448);
        pipe.InitBuffer(tbuf8, 8448);
        pipe.InitBuffer(tbuf9, 8448);
        pipe.InitBuffer(tbuf10, 8448);
        pipe.InitBuffer(tbuf11, 8448);
        pipe.InitBuffer(tbuf12, 8448);
        pipe.InitBuffer(tbuf13, 8448);
        pipe.InitBuffer(tbuf14, 8448);
        pipe.InitBuffer(tbuf15, 8448);
        pipe.InitBuffer(tbuf16, 8448);
        pipe.InitBuffer(tbuf17, 8448);
        pipe.InitBuffer(tbuf18, 8448);
        pipe.InitBuffer(tbuf19, 8448);
        pipe.InitBuffer(tbuf20, 8448);
        pipe.InitBuffer(tbuf21, 8448);
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
        AscendC::LocalTensor<float> buf9 = tbuf9.Get<float>();
        AscendC::LocalTensor<float> buf10 = tbuf10.Get<float>();
        AscendC::LocalTensor<float> buf11 = tbuf11.Get<float>();
        AscendC::LocalTensor<float> buf12 = tbuf12.Get<float>();
        AscendC::LocalTensor<float> buf13 = tbuf13.Get<float>();
        AscendC::LocalTensor<float> buf14 = tbuf14.Get<float>();
        AscendC::LocalTensor<float> buf15 = tbuf15.Get<float>();
        AscendC::LocalTensor<float> buf16 = tbuf16.Get<float>();
        AscendC::LocalTensor<float> buf17 = tbuf17.Get<float>();
        AscendC::LocalTensor<float> buf18 = tbuf18.Get<float>();
        AscendC::LocalTensor<float> buf19 = tbuf19.Get<float>();
        AscendC::LocalTensor<float> buf20 = tbuf20.Get<float>();
        AscendC::LocalTensor<float> buf21 = tbuf21.Get<float>();
        // chunked: 33792 elements as 16 x 2112, tail 2112
        constexpr int __CH = 2112;
        constexpr int __NW = 33792;
        constexpr int __NC = 16;
        for (int __c = 0; __c < __NC; ++__c) {
          const int __n = (__c == __NC - 1) ? (__NW - (__NC - 1) * __CH) : __CH;
          const int __off = __c * __CH;
          gm0Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm0_raw + __off, __NW - __off);
          gm1Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm1_raw + __off, __NW - __off);
          gm2Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm2_raw + __off, __NW - __off);
          int32_t __v0;
          int32_t __v1;
          int32_t __v2;
          float __v3;
          float __v4;
          float __v5;
          float __v6;
          float __v7;
          float __v8;
          __v0 = 0;
          __v1 = 1;
          __v2 = __n;
          __v3 = 1.0f;
          __v4 = -1.0f;
          __v5 = 0.0f;
          __v6 = -0.5f;
          __v7 = -0.08f;
          __v8 = 100.0f;
          AscendC::DataCopy(buf0, gm0Tensor, __n);
          AscendC::DataCopy(buf1, gm1Tensor, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Muls(buf2, buf0, (0e0f), __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(buf3, buf2, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf4, buf3, (float)(__v3), __v2);
          AscendC::Muls(buf5, buf0, (-1e0f), __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(buf6, buf5, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf7, buf6, (float)(__v3), __v2);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Div(buf8, buf3, buf7, __n);
          AscendC::Muls(buf9, buf0, (-5e-1f), __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(buf10, buf9, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf11, buf10, (float)(__v3), __v2);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Div(buf12, buf4, buf11, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf13, buf12, (float)(-(__v3)), __v2);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Mul(buf14, buf13, buf8, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Muls(buf15, buf1, (-8e-2f), __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Exp(buf16, buf15, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf17, buf16, (float)(__v3), __v2);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Div(buf18, buf4, buf17, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Adds(buf19, buf18, (float)(-(__v3)), __v2);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Mul(buf20, buf14, buf19, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Muls(buf21, buf20, (1e2f), __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          { const uint32_t __sb = (uint32_t)(__n * sizeof(float));
            if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf21, __n); }
            else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                   AscendC::DataCopyPad(gm2Tensor, buf21, __sp); } }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf6;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf7;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf8;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf9;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf10;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf11;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf12;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf13;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf14;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf15;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf17;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf18;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf19;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf20;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf21;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf22;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf23;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf24;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf25;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void k3_dense_situ(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_k3_dense_situ op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void k3_dense_situ_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    k3_dense_situ<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
