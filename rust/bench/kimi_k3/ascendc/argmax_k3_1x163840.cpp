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

class Kernel_k3_argmax_163840 {
public:
    __aicore__ inline Kernel_k3_argmax_163840() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ int32_t*)gm1, 16777216);
        pipe.InitBuffer(tbuf0, 40960);
        pipe.InitBuffer(tbuf1, 40960);
        pipe.InitBuffer(tbuf2, 40960);
        pipe.InitBuffer(tbuf3, 40960);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<int32_t> buf1 = tbuf1.Get<int32_t>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        __v0 = 1;
        __v1 = 163840;
        __v2 = 1;
        // chunked argmax: 163840 elements as 16 x 10240, tail 10240; the (max, index) pair is carried
        constexpr int __AM_CH = 10240;
        constexpr int __AM_W = 163840;
        constexpr int __AM_NC = 16;
        float __am_gbest = 0;
        int __am_gidx = 0;
        for (int __ac = 0; __ac < __AM_NC; ++__ac) {
          const int __an = (__ac == __AM_NC - 1) ? (__AM_W - (__AM_NC - 1) * __AM_CH) : __AM_CH;
          const int __aoff = __ac * __AM_CH;
          gm0Tensor.SetGlobalBuffer((__gm__ float*)(uintptr_t)gm0_raw + __aoff, __AM_W - __aoff);
          AscendC::DataCopy(buf0, gm0Tensor, __an);
          AscendC::PipeBarrier<PIPE_ALL>();
          { // tile_argmax: ReduceMax for the value, one scalar pass for the index
            int __am_c = (__an);
            AscendC::ReduceMax(buf3, buf0, buf2, __am_c);
            AscendC::PipeBarrier<PIPE_ALL>();
            float __am_best = buf3.GetValue(0);
            int __am_idx = 0;
            for (int __i = 0; __i < __am_c; ++__i) {
              if (buf0.GetValue(__i) == __am_best) { __am_idx = __i; break; }
            }
            buf1.SetValue(0, (int32_t)__am_idx);
          }
          {
            const float __am_cbest = buf3.GetValue(0);
            if (__ac == 0 || __am_cbest > __am_gbest) {
              __am_gbest = __am_cbest;
              __am_gidx = __aoff + (int)buf1.GetValue(0);
            }
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        buf1.SetValue(0, (int32_t)__am_gidx);
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v2)) * sizeof(int32_t));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm1Tensor, buf1, ((__v0) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm1Tensor, buf1, __sp); } }
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
    AscendC::GlobalTensor<int32_t> gm1Tensor;
    uint64_t gm1_raw;
};

extern "C" __global__ __aicore__ void k3_argmax_163840(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_k3_argmax_163840 op;
    op.Init(gm0, gm1);
    op.Process();
}


extern "C" void k3_argmax_163840_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    k3_argmax_163840<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
