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

class Kernel_attn_gqa {
public:
    __aicore__ inline Kernel_attn_gqa() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 49152);
        pipe.InitBuffer(tbuf1, 8192);
        pipe.InitBuffer(tbuf2, 8192);
        pipe.InitBuffer(tbuf3, 49152);
        pipe.InitBuffer(tbuf4, 1024);
        pipe.InitBuffer(tbuf5, 64);
        pipe.InitBuffer(tbuf6, 256);
        pipe.InitBuffer(tbuf7, 32);
        pipe.InitBuffer(tbuf8, 32);
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
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        int32_t __v4;
        int32_t __v5;
        int32_t __v6;
        __bb_0:
        __v0 = 0;
        __v1 = 12;
        __v2 = 2;
        __v3 = 16;
        __v4 = 64;
        __v5 = 192;
        __v6 = 32;
        AscendC::DataCopy(buf0, gm0Tensor, (192) * (64));
        AscendC::DataCopy(buf1, gm1Tensor, (32) * (64));
        AscendC::DataCopy(buf2, gm2Tensor, (32) * (64));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_attention_gqa vectorized: NH=12, NKV=2, S=16, D=64
          int __gqa_nh = (12); int __gqa_nkv = (2);
          int __gqa_s = (16); int __gqa_d = (64);
          int __gqa_gs = (__gqa_nkv > 0) ? (__gqa_nh / __gqa_nkv) : 1;
          float __gqa_scale = 1.0f / (float)sqrt((float)__gqa_d);
          int __gqa_sd = __gqa_s * __gqa_d;
          for (int __h = 0; __h < __gqa_nh; ++__h) {
            int __kvh = __h / __gqa_gs;
            int __q_base = __h * __gqa_sd;
            int __kv_base = __kvh * __gqa_sd;
            // Step 1: scores[i,j] = dot(Q_h[i,:], K_kvh[j,:]) * scale
            for (int __qi = 0; __qi < __gqa_s; ++__qi) {
              for (int __ki = 0; __ki < __gqa_s; ++__ki) {
                float __dot = 0.0f;
                for (int __dd = 0; __dd < __gqa_d; ++__dd)
                  __dot += buf0.GetValue(__q_base + __qi * __gqa_d + __dd)
                         * buf1.GetValue(__kv_base + __ki * __gqa_d + __dd);
                buf4.SetValue(__qi * __gqa_s + __ki, __dot * __gqa_scale);
              }
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            // Step 2: softmax(scores) row-wise → scores in-place
            for (int __qi = 0; __qi < __gqa_s; ++__qi) {
              for (int __j = 0; __j < __gqa_s; ++__j)
                buf5.SetValue(__j, buf4.GetValue(__qi * __gqa_s + __j));
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::ReduceMax(buf7, buf5, buf6, __gqa_s);
              AscendC::PipeBarrier<PIPE_ALL>();
              float __m = buf7.GetValue(0);
              AscendC::Adds(buf5, buf5, (float)(-__m), __gqa_s);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(buf5, buf5, __gqa_s);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::ReduceSum(buf8, buf5, buf6, __gqa_s);
              AscendC::PipeBarrier<PIPE_ALL>();
              float __inv = 1.0f / buf8.GetValue(0);
              AscendC::Muls(buf5, buf5, __inv, __gqa_s);
              // PIPE_ALL, not PIPE_V: the copy-out below reads this vector result with
              // SCALAR GetValue. Same defect as the softmax R>1 path, which was proven
              // on a 910B2 -- with PIPE_V the first element read raced the Muls and came
              // back un-scaled. Device-proven for softmax; fixed here by analogy.
              AscendC::PipeBarrier<PIPE_ALL>();
              for (int __j = 0; __j < __gqa_s; ++__j)
                buf4.SetValue(__qi * __gqa_s + __j, buf5.GetValue(__j));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            // Step 3: out_h[i,d] = sum_j weights[i,j] * V_kvh[j,d]
            for (int __qi = 0; __qi < __gqa_s; ++__qi) {
              for (int __di = 0; __di < __gqa_d; ++__di) {
                float __acc = 0.0f;
                for (int __ki = 0; __ki < __gqa_s; ++__ki)
                  __acc += buf4.GetValue(__qi * __gqa_s + __ki)
                         * buf2.GetValue(__kv_base + __ki * __gqa_d + __di);
                buf3.SetValue(__q_base + __qi * __gqa_d + __di, __acc);
              }
            }
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v5) * (__v4)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf3, ((__v5) * (__v4))); }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf4;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf5;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf6;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf7;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf8;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf9;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void attn_gqa(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_attn_gqa op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void attn_gqa_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    attn_gqa<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
