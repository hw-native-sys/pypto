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

class Kernel_rms_norm {
public:
    __aicore__ inline Kernel_rms_norm() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 6144);
        pipe.InitBuffer(tbuf1, 6144);
        pipe.InitBuffer(tbuf2, 6144);
        pipe.InitBuffer(tbuf3, 32);
        pipe.InitBuffer(tbuf4, 6144);
        pipe.InitBuffer(tbuf5, 6144);
        pipe.InitBuffer(tbuf6, 6144);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        AscendC::LocalTensor<float> buf5 = tbuf5.Get<float>();
        AscendC::LocalTensor<float> buf6 = tbuf6.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        float __v3;
        int32_t __v4;
        __bb_0:
        __v0 = 1536;
        __v1 = 384;
        __v2 = 6144;
        __v3 = 0.000001f;
        __v4 = 1;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (1536));
        AscendC::DataCopy(buf1, gm1Tensor, (1) * (1536));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_rms_norm_mul: out = x * rsqrt(mean(x^2)+eps) * g
          int __rnm_n = (__v0);
          AscendC::Mul(buf2, buf0, buf0, __rnm_n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::ReduceSum(buf3, buf2, buf4, __rnm_n);
          AscendC::PipeBarrier<PIPE_ALL>();
          float __rnm_ssq = buf3.GetValue(0);
          float __rnm_rms = 1.0f / (float)sqrt((float)(__rnm_ssq / (float)__rnm_n + (__v3)));
          AscendC::Muls(buf5, buf0, __rnm_rms, __rnm_n);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Mul(buf6, buf5, buf1, __rnm_n);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v4) * (__v0)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf6, ((__v4) * (__v0))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm2Tensor, buf6, __sp); } }
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

extern "C" __global__ __aicore__ void rms_norm(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_rms_norm op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void rms_norm_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    rms_norm<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
