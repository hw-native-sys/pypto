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

class Kernel_glm53_proj_kv_b {
public:
    __aicore__ inline Kernel_glm53_proj_kv_b() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 16384);
        pipe.InitBuffer(tbuf1, 16384);
        pipe.InitBuffer(tbuf2, 16384);
        pipe.InitBuffer(tbuf3, 16384);
        pipe.InitBuffer(tbuf4, 16384);
        pipe.InitBuffer(tbuf5, 16384);
        pipe.InitBuffer(tbuf6, 16384);
        pipe.InitBuffer(tbuf8, 16384);
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
        int32_t __v3;
        __v0 = 0;
        __v1 = 1;
        __v2 = 1;
        __v3 = 28672;
        AscendC::LocalTensor<float> buf8 = tbuf8.Get<float>();
        // blocked reduction: 28672 elements as 7 x 4096, tail 4096
        constexpr int __CH = 4096;
        constexpr int __NW = 28672;
        constexpr int __NC = 7;
        AscendC::Duplicate(buf8, (float)0, __CH);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int __c = 0; __c < __NC; ++__c) {
          const int __n = (__c == __NC - 1) ? (__NW - (__NC - 1) * __CH) : __CH;
          const int __off = __c * __CH;
          gm0Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm0_raw + __off, __NW - __off);
          gm1Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm1_raw + __off, __NW - __off);
          __v3 = __n;
          AscendC::DataCopy(buf0, gm0Tensor, __n);
          AscendC::DataCopy(buf1, gm1Tensor, __n);
          AscendC::PipeBarrier<PIPE_ALL>();
          AscendC::Mul(buf2, buf0, buf1, (__v1) * (__v3));
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(buf8, buf8, buf2, __n);
        }
        __v3 = __CH;
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_reduce_sum: (r, c) -> (r, 1)
          int __rd_r = (__v1); int __rd_c = (__v3);
          if (__rd_r == 1) {
            AscendC::ReduceSum(buf6, buf8, buf3, __rd_c);
          } else {
            for (int __i = 0; __i < __rd_r; ++__i) {
              for (int __j = 0; __j < __rd_c; ++__j)
                buf3.SetValue(__j, buf8.GetValue(__i * __rd_c + __j));
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::ReduceSum(buf5, buf3, buf4, __rd_c);
              AscendC::PipeBarrier<PIPE_ALL>();
              buf6.SetValue(__i, buf5.GetValue(0));
            }
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v2) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf6, ((__v2) * (__v2))); }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf8;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void glm53_proj_kv_b(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_glm53_proj_kv_b op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void glm53_proj_kv_b_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    glm53_proj_kv_b<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
