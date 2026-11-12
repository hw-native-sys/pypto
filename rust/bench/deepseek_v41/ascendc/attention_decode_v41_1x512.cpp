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

class Kernel_v41_attn_decode {
public:
    __aicore__ inline Kernel_v41_attn_decode() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 2048);
        pipe.InitBuffer(tbuf1, 16384);
        pipe.InitBuffer(tbuf2, 16384);
        pipe.InitBuffer(tbuf3, 2048);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 0;
        __v1 = 1;
        __v2 = 8;
        __v3 = 512;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (512));
        AscendC::DataCopy(buf1, gm1Tensor, (8) * (512));
        AscendC::DataCopy(buf2, gm1Tensor, (8) * (512));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_attention_sink 1-query: softmax([sink, QK^T/sqrt(D)]) . V
          int __as_s = (8); int __as_d = (512);
          float __as_scale = 1.0f / (float)sqrt((float)__as_d);
          float __as_sc[256];
          float __as_m = -1e30f;
          for (int __p = 0; __p < __as_s; ++__p) {
            float __dot = 0.0f;
            for (int __x = 0; __x < __as_d; ++__x)
              __dot += buf0.GetValue(__x) * buf1.GetValue(__p * __as_d + __x);
            __dot *= __as_scale;
            __as_sc[__p] = __dot;
            if (__dot > __as_m) __as_m = __dot;
          }
          for (int __j = 0; __j < __as_s; ++__j) {
            float __sk = (float)gm2Tensor.GetValue((0) + __j);
            if (__sk > __as_m) __as_m = __sk;
          }
          float __as_e[256];
          float __as_den = 0.0f;
          for (int __j = 0; __j < __as_s; ++__j)
            buf3.SetValue(__j, (float)gm2Tensor.GetValue((0) + __j) - __as_m);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Exp(buf3, buf3, __as_s);
          AscendC::PipeBarrier<PIPE_ALL>();
          for (int __j = 0; __j < __as_s; ++__j)
            __as_den += buf3.GetValue(__j) / (float)__as_s;
          for (int __p = 0; __p < __as_s; ++__p)
            buf3.SetValue(__p, __as_sc[__p] - __as_m);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Exp(buf3, buf3, __as_s);
          AscendC::PipeBarrier<PIPE_ALL>();
          for (int __p = 0; __p < __as_s; ++__p) {
            __as_e[__p] = buf3.GetValue(__p);
            __as_den += __as_e[__p];
          }
          for (int __x = 0; __x < __as_d; ++__x) {
            float __acc = 0.0f;
            for (int __p = 0; __p < __as_s; ++__p)
              __acc += __as_e[__p] * buf2.GetValue(__p * __as_d + __x);
            buf3.SetValue(__x, __acc / __as_den);
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v3) * (__v1)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf3, ((__v3) * (__v1))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor, buf3, __sp); } }
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf3;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void v41_attn_decode(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_v41_attn_decode op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void v41_attn_decode_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    v41_attn_decode<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
