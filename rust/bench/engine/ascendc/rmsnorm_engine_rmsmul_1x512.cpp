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

class Kernel_engine_rmsmul_1_512 {
public:
    __aicore__ inline Kernel_engine_rmsmul_1_512() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 2048);
        pipe.InitBuffer(tbuf1, 2048);
        pipe.InitBuffer(tbuf2, 2048);
        pipe.InitBuffer(tbuf3, 32);
        pipe.InitBuffer(tbuf4, 2048);
        pipe.InitBuffer(tbuf5, 2048);
        pipe.InitBuffer(tbuf6, 2048);
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
        float __v2;
        int32_t __v3;
        float __v;
        __v0 = 1;
        __v1 = 512;
        __v2 = 0.000001f;
        __v3 = 0;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (512));
        AscendC::DataCopy(buf1, gm1Tensor, (1) * (512));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_rms_norm vectorized (eps=1e-6f)
          int __rn_r = (__v0); int __rn_c = (__v1);
          if (__rn_r == 1) {
            // Fast path: whole tile is the single row. No subview offsets.
            AscendC::Mul(buf2, buf0, buf0, __rn_c);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(buf3, buf2, buf4, __rn_c);
            AscendC::PipeBarrier<PIPE_ALL>();
            float __ssq = buf3.GetValue(0);
            float __rms = 1.0f / (float)sqrt((float)(__ssq / (float)__rn_c + 1e-6f));
            AscendC::Muls(buf5, buf0, __rms, __rn_c);
          } else {
            // Fallback: per-row scalar loop (910B2 LocalTensor[i>0] sub-view has
            // a known vector-op miscompile). Each row is 1 pass so R is small.
            for (int __i = 0; __i < __rn_r; ++__i) {
              float __sum_sq = 0.0f;
              for (int __j = 0; __j < __rn_c; ++__j) {
        __v = buf0.GetValue(__i * __rn_c + __j);
                __sum_sq += __v * __v;
              }
              float __rms = 1.0f / (float)sqrt((float)(__sum_sq / (float)__rn_c + 1e-6f));
              for (int __j = 0; __j < __rn_c; ++__j)
                buf5.SetValue(__i * __rn_c + __j,
                  buf0.GetValue(__i * __rn_c + __j) * __rms);
            }
          }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(buf6, buf5, buf1, (__v0) * (__v1));
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v1)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf6, ((__v0) * (__v1))); }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf7;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void engine_rmsmul_1_512(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_engine_rmsmul_1_512 op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void engine_rmsmul_1_512_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    engine_rmsmul_1_512<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
