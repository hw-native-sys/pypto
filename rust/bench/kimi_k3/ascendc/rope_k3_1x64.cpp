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

class Kernel_k3_rope_64 {
public:
    __aicore__ inline Kernel_k3_rope_64() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 128);
        pipe.InitBuffer(tbuf1, 128);
        pipe.InitBuffer(tbuf2, 128);
        pipe.InitBuffer(tbuf3, 128);
        pipe.InitBuffer(tbuf4, 128);
        pipe.InitBuffer(tbuf5, 128);
        pipe.InitBuffer(tbuf6, 128);
        pipe.InitBuffer(tbuf7, 128);
        pipe.InitBuffer(tbuf8, 128);
        pipe.InitBuffer(tbuf9, 128);
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
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        __v0 = 1;
        __v1 = 32;
        __v2 = 0;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (32));
        AscendC::DataCopy(buf1, gm0Tensor[__v1], (1) * (32));
        AscendC::DataCopy(buf2, gm1Tensor, (1) * (32));
        AscendC::DataCopy(buf3, gm2Tensor, (1) * (32));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(buf4, buf0, buf2, (__v0) * (__v1));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(buf5, buf1, buf3, (__v0) * (__v1));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(buf6, buf4, buf5, (__v0) * (__v1));
        AscendC::Mul(buf7, buf1, buf2, (__v0) * (__v1));
        AscendC::Mul(buf8, buf0, buf3, (__v0) * (__v1));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(buf9, buf7, buf8, (__v0) * (__v1));
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v1)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf6, ((__v0) * (__v1))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor, buf6, __sp); } }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v1)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor[__v1], buf9, ((__v0) * (__v1))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor[__v1], buf9, __sp); } }
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
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void k3_rope_64(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_k3_rope_64 op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void k3_rope_64_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    k3_rope_64<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
