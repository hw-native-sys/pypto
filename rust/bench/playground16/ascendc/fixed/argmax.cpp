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

class Kernel_argmax {
public:
    __aicore__ inline Kernel_argmax() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        pipe.InitBuffer(tbuf0, 65536);
        pipe.InitBuffer(tbuf1, 32);
        pipe.InitBuffer(tbuf2, 65536);
        pipe.InitBuffer(tbuf3, 32);
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
        __bb_0:
        __v0 = 0;
        __v1 = 1;
        __v2 = 16384;
        __v3 = 1;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (16384));
        AscendC::PipeBarrier<PIPE_ALL>();
        // TILE_SUBSTITUTION: intrinsic=__tile_argmax_f32; replacement=f32-index-storage; reason=the kernel's own __tile_store_f32 narrows an integer index; declare __tile_store_i32 to remove this
        { // tile_argmax: ReduceMax for the value, one scalar pass for the index
          int __am_c = (16384);
          AscendC::ReduceMax(buf3, buf0, buf2, __am_c);
          AscendC::PipeBarrier<PIPE_ALL>();
          float __am_best = buf3.GetValue(0);
          int __am_idx = 0;
          for (int __i = 0; __i < __am_c; ++__i) {
            if (buf0.GetValue(__i) == __am_best) { __am_idx = __i; break; }
          }
          buf1.SetValue(0, (float)__am_idx);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v3)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm1Tensor, buf1, ((__v1) * (__v3))); }
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
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
};

extern "C" __global__ __aicore__ void argmax(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_argmax op;
    op.Init(gm0, gm1);
    op.Process();
}


extern "C" void argmax_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    argmax<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
