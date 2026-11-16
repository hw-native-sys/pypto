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

class Kernel_bm_argmax_8_129280 {
public:
    __aicore__ inline Kernel_bm_argmax_8_129280() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
    }
    __aicore__ inline void Process() {
        int32_t __v0;
        int32_t __v1;
        __v0 = 8;
        __v1 = 129280;
        { // per-row argmax: i32 indices written straight to GM
          int __R = (8); int __C = (129280);
          AscendC::GlobalTensor<int32_t> __idx;
          __idx.SetGlobalBuffer((__gm__ int32_t*)(uintptr_t)gm1_raw, __R);
          for (int __rr = 0; __rr < __R; ++__rr) {
            float __m = -1e38f; int __best = 0;
            for (int __cc = 0; __cc < __C; ++__cc) {
              float __x = (float)gm0Tensor.GetValue((0) + __rr * __C + __cc);
              if (__x > __m) { __m = __x; __best = __cc; }
            }
            __idx.SetValue(__rr, __best);
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
};

extern "C" __global__ __aicore__ void bm_argmax_8_129280(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_bm_argmax_8_129280 op;
    op.Init(gm0, gm1);
    op.Process();
}


extern "C" void bm_argmax_8_129280_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    bm_argmax_8_129280<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
