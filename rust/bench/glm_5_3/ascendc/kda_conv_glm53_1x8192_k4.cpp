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

class Kernel_glm53_kda_conv_8192 {
public:
    __aicore__ inline Kernel_glm53_kda_conv_8192() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4, GM_ADDR gm5, GM_ADDR gm6, GM_ADDR gm7, GM_ADDR gm8) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm4_raw = (uint64_t)(uintptr_t)gm4;
        gm5_raw = (uint64_t)(uintptr_t)gm5;
        gm6_raw = (uint64_t)(uintptr_t)gm6;
        gm7_raw = (uint64_t)(uintptr_t)gm7;
        gm8_raw = (uint64_t)(uintptr_t)gm8;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        gm4Tensor.SetGlobalBuffer((__gm__ float*)gm4, 16777216);
        gm5Tensor.SetGlobalBuffer((__gm__ float*)gm5, 16777216);
        gm6Tensor.SetGlobalBuffer((__gm__ float*)gm6, 16777216);
        gm7Tensor.SetGlobalBuffer((__gm__ float*)gm7, 16777216);
        gm8Tensor.SetGlobalBuffer((__gm__ float*)gm8, 16777216);
        pipe.InitBuffer(tbuf0, 8192);
        pipe.InitBuffer(tbuf1, 8192);
        pipe.InitBuffer(tbuf2, 8192);
        pipe.InitBuffer(tbuf3, 8192);
        pipe.InitBuffer(tbuf4, 8192);
        pipe.InitBuffer(tbuf5, 8192);
        pipe.InitBuffer(tbuf6, 8192);
        pipe.InitBuffer(tbuf7, 8192);
        pipe.InitBuffer(tbuf8, 8192);
        pipe.InitBuffer(tbuf9, 8192);
        pipe.InitBuffer(tbuf10, 8192);
        pipe.InitBuffer(tbuf11, 8192);
        pipe.InitBuffer(tbuf12, 8192);
        pipe.InitBuffer(tbuf13, 8192);
        pipe.InitBuffer(tbuf14, 8192);
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
        // chunked: 8192 elements as 4 x 2048, tail 2048
        constexpr int __CH = 2048;
        constexpr int __NW = 8192;
        constexpr int __NC = 4;
        for (int __c = 0; __c < __NC; ++__c) {
          const int __n = (__c == __NC - 1) ? (__NW - (__NC - 1) * __CH) : __CH;
          const int __off = __c * __CH;
          gm0Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm0_raw + __off, __NW - __off);
          gm1Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm1_raw + __off, __NW - __off);
          gm2Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm2_raw + __off, __NW - __off);
          gm3Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm3_raw + __off, __NW - __off);
          gm4Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm4_raw + __off, __NW - __off);
          gm5Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm5_raw + __off, __NW - __off);
          gm6Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm6_raw + __off, __NW - __off);
          gm7Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm7_raw + __off, __NW - __off);
          gm8Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm8_raw + __off, __NW - __off);
          int32_t __v0;
          int32_t __v1;
          int32_t __v2;
          __v0 = 0;
          __v1 = 1;
          __v2 = __n;
          AscendC::DataCopy(buf0, gm0Tensor, __n);
          AscendC::DataCopy(buf1, gm1Tensor, __n);
          AscendC::DataCopy(buf2, gm2Tensor, __n);
          AscendC::DataCopy(buf3, gm3Tensor, __n);
          AscendC::DataCopy(buf4, gm4Tensor, __n);
          AscendC::DataCopy(buf5, gm5Tensor, __n);
          AscendC::DataCopy(buf6, gm6Tensor, __n);
          AscendC::DataCopy(buf7, gm7Tensor, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Mul(buf8, buf0, buf4, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Mul(buf9, buf1, buf5, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(buf10, buf8, buf9, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Mul(buf11, buf2, buf6, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(buf12, buf10, buf11, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Mul(buf13, buf3, buf7, __n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(buf14, buf12, buf13, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          { const uint32_t __sb = (uint32_t)(__n * sizeof(float));
            if (__sb % 32 == 0) { AscendC::DataCopy(gm8Tensor, buf14, __n); }
            else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                   AscendC::DataCopyPad(gm8Tensor, buf14, __sp); } }
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
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
    AscendC::GlobalTensor<float> gm4Tensor;
    uint64_t gm4_raw;
    AscendC::GlobalTensor<float> gm5Tensor;
    uint64_t gm5_raw;
    AscendC::GlobalTensor<float> gm6Tensor;
    uint64_t gm6_raw;
    AscendC::GlobalTensor<float> gm7Tensor;
    uint64_t gm7_raw;
    AscendC::GlobalTensor<float> gm8Tensor;
    uint64_t gm8_raw;
};

extern "C" __global__ __aicore__ void glm53_kda_conv_8192(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4, GM_ADDR gm5, GM_ADDR gm6, GM_ADDR gm7, GM_ADDR gm8) {
    Kernel_glm53_kda_conv_8192 op;
    op.Init(gm0, gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8);
    op.Process();
}


extern "C" void glm53_kda_conv_8192_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3, uint8_t *gm4, uint8_t *gm5, uint8_t *gm6, uint8_t *gm7, uint8_t *gm8) {
    glm53_kda_conv_8192<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8);
}
