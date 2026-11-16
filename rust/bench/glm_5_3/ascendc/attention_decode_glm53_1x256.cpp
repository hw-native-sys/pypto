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

class Kernel_glm53_attn_decode_256 {
public:
    __aicore__ inline Kernel_glm53_attn_decode_256() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 1024);
        pipe.InitBuffer(tbuf1, 1024);
        pipe.InitBuffer(tbuf2, 256);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 0;
        __v1 = 1;
        __v2 = 2048;
        __v3 = 256;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (256));
        { // 1-query blocked attention: K/V streamed from GM
          int __S = (2048); int __D = (256); int __SB = 64;
          float __scale = 1.0f / (float)sqrt((float)__D);
          float __m = -1e30f;
          for (int __b = 0; __b < __S; __b += __SB) {
            int __bw = (__S - __b < __SB) ? (__S - __b) : __SB;
            for (int __p = 0; __p < __bw; ++__p) {
              float __dot = 0.0f;
              for (int __x = 0; __x < __D; ++__x)
                __dot += buf0.GetValue(__x) * (float)gm1Tensor.GetValue((0) + (__b + __p) * __D + __x);
              __dot *= __scale;
              if (__dot > __m) __m = __dot;
            }
          }
          for (int __x = 0; __x < __D; ++__x) buf1.SetValue(__x, 0.0f);
          float __den = 0.0f;
          for (int __b = 0; __b < __S; __b += __SB) {
            int __bw = (__S - __b < __SB) ? (__S - __b) : __SB;
            for (int __p = 0; __p < __bw; ++__p) {
              float __dot = 0.0f;
              for (int __x = 0; __x < __D; ++__x)
                __dot += buf0.GetValue(__x) * (float)gm1Tensor.GetValue((0) + (__b + __p) * __D + __x);
              buf2.SetValue(__p, __dot * __scale - __m);
            }
            for (int __p = __bw; __p < __SB; ++__p) buf2.SetValue(__p, -1e30f);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Exp(buf2, buf2, __SB);
            AscendC::PipeBarrier<PIPE_ALL>();
            for (int __p = 0; __p < __bw; ++__p) {
              float __e = buf2.GetValue(__p);
              __den += __e;
              for (int __x = 0; __x < __D; ++__x)
                buf1.SetValue(__x, buf1.GetValue(__x) + __e * (float)gm2Tensor.GetValue((0) + (__b + __p) * __D + __x));
            }
          }
          for (int __x = 0; __x < __D; ++__x) buf1.SetValue(__x, buf1.GetValue(__x) / __den);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v3)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf1, ((__v1) * (__v3))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor, buf1, __sp); } }
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf2;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void glm53_attn_decode_256(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_glm53_attn_decode_256 op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void glm53_attn_decode_256_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    glm53_attn_decode_256<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
