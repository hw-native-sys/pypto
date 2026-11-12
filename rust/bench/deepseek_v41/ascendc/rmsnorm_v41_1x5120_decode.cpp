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

class Kernel_bm_rmsnorm_1_5120 {
public:
    __aicore__ inline Kernel_bm_rmsnorm_1_5120() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        pipe.InitBuffer(tbuf0, 20480);
        pipe.InitBuffer(tbuf1, 20480);
        pipe.InitBuffer(tbuf2, 32);
        pipe.InitBuffer(tbuf3, 20480);
        pipe.InitBuffer(tbuf4, 20480);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        int32_t __v0;
        int32_t __v1;
        float __v2;
        float __v;
        __v0 = 1;
        __v1 = 5120;
        __v2 = 0.00000000000000000001f;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (5120));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_rms_norm vectorized (eps=1e-20f)
          int __rn_r = (__v0); int __rn_c = (__v1);
          if (__rn_r == 1) {
            // Fast path: whole tile is the single row. No subview offsets.
            AscendC::Mul(buf1, buf0, buf0, __rn_c);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(buf2, buf1, buf3, __rn_c);
            AscendC::PipeBarrier<PIPE_ALL>();
            float __ssq = buf2.GetValue(0);
            float __rms = 1.0f / (float)sqrt((float)(__ssq / (float)__rn_c + 1e-20f));
            AscendC::Muls(buf4, buf0, __rms, __rn_c);
          } else {
            // Fallback: per-row scalar loop (910B2 LocalTensor[i>0] sub-view has
            // a known vector-op miscompile). Each row is 1 pass so R is small.
            for (int __i = 0; __i < __rn_r; ++__i) {
              float __sum_sq = 0.0f;
              for (int __j = 0; __j < __rn_c; ++__j) {
        __v = buf0.GetValue(__i * __rn_c + __j);
                __sum_sq += __v * __v;
              }
              float __rms = 1.0f / (float)sqrt((float)(__sum_sq / (float)__rn_c + 1e-20f));
              for (int __j = 0; __j < __rn_c; ++__j)
                buf4.SetValue(__i * __rn_c + __j,
                  buf0.GetValue(__i * __rn_c + __j) * __rms);
            }
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v1)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm1Tensor, buf4, ((__v0) * (__v1))); }
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
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
};

extern "C" __global__ __aicore__ void bm_rmsnorm_1_5120(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_bm_rmsnorm_1_5120 op;
    op.Init(gm0, gm1);
    op.Process();
}


extern "C" void bm_rmsnorm_1_5120_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    bm_rmsnorm_1_5120<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
