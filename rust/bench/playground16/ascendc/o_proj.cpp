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

class Kernel_o_proj {
public:
    __aicore__ inline Kernel_o_proj() {}
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
        pipe.InitBuffer(tbuf3, 6144);
        pipe.InitBuffer(tbuf4, 32);
        pipe.InitBuffer(tbuf5, 32);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        AscendC::LocalTensor<float> buf5 = tbuf5.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        __bb_0:
        __v0 = 0;
        __v1 = 1;
        __v2 = 1536;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (1536));
        AscendC::DataCopy(buf1, gm1Tensor, (1) * (1536));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // tile_matvec: y[i] = sum_j a[j] * w[i, j]
          int __mv_n = (__v1); int __mv_k = (__v2);
          for (int __i = 0; __i < __mv_n; ++__i) {
            for (int __j = 0; __j < __mv_k; ++__j)
              buf2.SetValue(__j, buf0.GetValue(__j) * buf1.GetValue(__i * __mv_k + __j));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::ReduceSum(buf4, buf2, buf3, __mv_k);
            AscendC::PipeBarrier<PIPE_ALL>();
            buf5.SetValue(__i, buf4.GetValue(0));
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf5, ((__v1) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm2Tensor, buf5, __sp); } }
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
};

extern "C" __global__ __aicore__ void o_proj(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_o_proj op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void o_proj_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    o_proj<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
