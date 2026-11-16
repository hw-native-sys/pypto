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

class Kernel_k3_kda_conv_7168 {
public:
    __aicore__ inline Kernel_k3_kda_conv_7168() {}
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
        pipe.InitBuffer(tbuf0, 28672);
        pipe.InitBuffer(tbuf1, 28672);
        pipe.InitBuffer(tbuf2, 28672);
        pipe.InitBuffer(tbuf3, 28672);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf5 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf6 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf7 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf8 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf9 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf10 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf11 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf12 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf13 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf14 = tbuf0.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        __v0 = 0;
        __v1 = 1;
        __v2 = 7168;
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (7168));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf1, gm4Tensor, (1) * (7168));
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe_barrier(PIPE_ALL);
        AscendC::Mul(buf2, buf0, buf1, (__v1) * (__v2));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf3, gm1Tensor, (1) * (7168));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf4, gm5Tensor, (1) * (7168));
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe_barrier(PIPE_ALL);
        AscendC::Mul(buf5, buf3, buf4, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        pipe_barrier(PIPE_ALL);
        AscendC::Add(buf6, buf2, buf5, (__v1) * (__v2));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf7, gm2Tensor, (1) * (7168));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf8, gm6Tensor, (1) * (7168));
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe_barrier(PIPE_ALL);
        AscendC::Mul(buf9, buf7, buf8, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        pipe_barrier(PIPE_ALL);
        AscendC::Add(buf10, buf6, buf9, (__v1) * (__v2));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf11, gm3Tensor, (1) * (7168));
        pipe_barrier(PIPE_ALL);
        AscendC::DataCopy(buf12, gm7Tensor, (1) * (7168));
        AscendC::PipeBarrier<PIPE_ALL>();
        pipe_barrier(PIPE_ALL);
        AscendC::Mul(buf13, buf11, buf12, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        pipe_barrier(PIPE_ALL);
        AscendC::Add(buf14, buf10, buf13, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v2)) * sizeof(float));
        pipe_barrier(PIPE_ALL);
          if (__sb % 32 == 0) { AscendC::DataCopy(gm8Tensor, buf14, ((__v1) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
        pipe_barrier(PIPE_ALL);
                 AscendC::DataCopyPad(gm8Tensor, buf14, __sp); } }
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

extern "C" __global__ __aicore__ void k3_kda_conv_7168(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4, GM_ADDR gm5, GM_ADDR gm6, GM_ADDR gm7, GM_ADDR gm8) {
    Kernel_k3_kda_conv_7168 op;
    op.Init(gm0, gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8);
    op.Process();
}


extern "C" void k3_kda_conv_7168_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3, uint8_t *gm4, uint8_t *gm5, uint8_t *gm6, uint8_t *gm7, uint8_t *gm8) {
    k3_kda_conv_7168<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3, gm4, gm5, gm6, gm7, gm8);
}
