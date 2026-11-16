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

class Kernel_matmul {
public:
    __aicore__ inline Kernel_matmul() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm0Tensor.SetGlobalBuffer((__gm__ half*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ half*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ half*)gm2, 16777216);
        pipe.InitBuffer(tbuf0, 128);
        pipe.InitBuffer(tbuf1, 8192);
        pipe.InitBuffer(tbuf2, 128);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<half> buf0 = tbuf0.Get<half>();
        AscendC::LocalTensor<half> buf1 = tbuf1.Get<half>();
        AscendC::LocalTensor<half> buf2 = tbuf2.Get<half>();
        int32_t __v0;
        int32_t __v1;
        __bb_0:
        __v0 = 1;
        __v1 = 64;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (64));
        AscendC::DataCopy(buf1, gm1Tensor, (64) * (64));
        AscendC::PipeBarrier<PIPE_ALL>();
        { // TILE_DEGENERATE_MATMUL: scalar GetValue loop on the vector core -- the cube unit is idle here
          int __mm_m = (__v0); int __mm_k = (__v1); int __mm_n = (__v1);
          for (int __i = 0; __i < __mm_m; ++__i)
            for (int __j = 0; __j < __mm_n; ++__j) {
              float __acc = 0.0f;
              for (int __p = 0; __p < __mm_k; ++__p)
                __acc += (float)buf0.GetValue(__i * __mm_k + __p) *
                         (float)buf1.GetValue(__p * __mm_n + __j);
              buf2.SetValue(__i * __mm_n + __j, __acc);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v1)) * sizeof(half));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm2Tensor, buf2, ((__v0) * (__v1))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm2Tensor, buf2, __sp); } }
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf2;
    AscendC::GlobalTensor<half> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<half> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<half> gm2Tensor;
    uint64_t gm2_raw;
};

extern "C" __global__ __aicore__ void matmul(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_matmul op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void matmul_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    matmul<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
