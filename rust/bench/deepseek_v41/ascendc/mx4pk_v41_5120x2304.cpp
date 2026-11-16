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

class Kernel_bm_mx4pk_5120_2304 {
public:
    __aicore__ inline Kernel_bm_mx4pk_5120_2304() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
    }
    __aicore__ inline void Process() {
        int32_t __v0;
        int32_t __v1;
        __v0 = 5120;
        __v1 = 2304;
        { // packed MXFP4 matvec: matvec_ref's contract, streamed
          const int __C = (5120); const int __R = (2304);
          const int __nb = __C / 32; const int __np = __nb / 2;
          static const float __tab[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
                                          -0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
          AscendC::GlobalTensor<int8_t> __w; __w.SetGlobalBuffer((__gm__ int8_t*)(uintptr_t)gm0_raw, (int64_t)__R * __np * 32);
          AscendC::GlobalTensor<float> __x, __d, __o;
          __x.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm2_raw, __C);
          __d.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm1_raw, (int64_t)__R * __nb);
          __o.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm3_raw, __R);
          for (int __r = 0; __r < __R; ++__r) {
            float __acc = 0.0f;
            for (int __g = 0; __g < __np; ++__g) {
              float __se = 0.0f, __so = 0.0f;
              for (int __i = 0; __i < 32; ++__i) {
                const uint8_t __byte = (uint8_t)__w.GetValue((int64_t)(__r * __np + __g) * 32 + __i);
                __se += __tab[__byte & 0x0F] * __x.GetValue(__g * 32 + __i);
                __so += __tab[__byte >> 4] * __x.GetValue((__np + __g) * 32 + __i);
              }
              __acc += __se * __d.GetValue((int64_t)__r * __nb + __g)
                     + __so * __d.GetValue((int64_t)__r * __nb + __np + __g);
            }
            __o.SetValue(__r, __acc);
          }
        }
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void bm_mx4pk_5120_2304(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_bm_mx4pk_5120_2304 op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void bm_mx4pk_5120_2304_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    bm_mx4pk_5120_2304<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
