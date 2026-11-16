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

class Kernel_glm53_proj_output {
public:
    __aicore__ inline Kernel_glm53_proj_output() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm4_raw = (uint64_t)(uintptr_t)gm4;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ int8_t*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        gm4Tensor.SetGlobalBuffer((__gm__ float*)gm4, 16777216);
        pipe.InitBuffer(tbuf0, 16384);
        pipe.InitBuffer(tbuf1, 4096);
        pipe.InitBuffer(tbuf2, 8192);
        pipe.InitBuffer(tbuf3, 16384);
        pipe.InitBuffer(tbuf4, 16384);
        pipe.InitBuffer(tbuf5, 16384);
        pipe.InitBuffer(tbuf6, 16384);
        pipe.InitBuffer(tbuf7, 32);
        pipe.InitBuffer(tbuf8, 16384);
        pipe.InitBuffer(tbuf9, 8192);
        pipe.InitBuffer(tbuf10, 8192);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        AscendC::LocalTensor<int8_t> buf1 = tbuf1.Get<int8_t>();
        AscendC::LocalTensor<half> buf2 = tbuf2.Get<half>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        AscendC::LocalTensor<float> buf5 = tbuf5.Get<float>();
        AscendC::LocalTensor<float> buf6 = tbuf6.Get<float>();
        AscendC::LocalTensor<float> buf7 = tbuf7.Get<float>();
        AscendC::LocalTensor<float> buf8 = tbuf8.Get<float>();
        AscendC::LocalTensor<float> buf9 = tbuf9.Get<float>();
        AscendC::LocalTensor<float> buf10 = tbuf10.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 1;
        __v1 = 4096;
        __v2 = 4096;
        __v3 = 0;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (4096));
        { // tile_matvec_w8a8: y[i] = sum_k a[k] * ((q - off[i]) * sc[i])
          int __n = (__v2); int __k = (__v1);
          // The V->MTE2 flag is PRIMED before the loop. A WaitFlag with no preceding SetFlag blocks
          // on the first iteration and the kernel never starts -- which is exactly what the first
          // attempt at this fix did.
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
          const int __band = 2048;
          for (int __b = 0; __b < __n; __b += __band) {
          const int __nb = (__n - __b < __band) ? (__n - __b) : __band;
          AscendC::DataCopy(buf9, gm2Tensor[__b], __nb);
          AscendC::DataCopy(buf10, gm3Tensor[__b], __nb);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
          for (int __i = __b; __i < __b + __nb; ++__i) {
            // THREE dependencies, not two. MTE2->V orders the copy-in before the cast, V->S orders
            // the reduction before the scalar readback -- and V->MTE2 orders the NEXT row's copy
            // against THIS row's cast, which is a write-after-read hazard on buf_w8. Without it the
            // DMA of row i+1 may land while row i's cast is still reading, which corrupts a subset
            // of the outputs; it went unnoticed at k=1024/4096 and produced 25 +inf of 4096 at
            // k=8192, where the cast is long enough for the race to be hit.
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::DataCopy(buf1, gm1Tensor[__i * __k], __k);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(buf2, buf1, AscendC::RoundMode::CAST_NONE, __k);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(buf3, buf2, AscendC::RoundMode::CAST_NONE, __k);
            AscendC::PipeBarrier<PIPE_V>();
            const float __sc = buf9.GetValue(__i - __b);
            const float __off = buf10.GetValue(__i - __b);
            AscendC::Adds(buf4, buf3, -__off, __k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(buf3, buf4, __sc, __k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(buf5, buf0, buf3, __k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(buf7, buf5, buf6, __k);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            buf8.SetValue(__i, buf7.GetValue(0));
          }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm4Tensor, buf8, ((__v0) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm4Tensor, buf8, __sp); } }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf10;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<int8_t> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
    AscendC::GlobalTensor<float> gm4Tensor;
    uint64_t gm4_raw;
};

extern "C" __global__ __aicore__ void glm53_proj_output(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4) {
    Kernel_glm53_proj_output op;
    op.Init(gm0, gm1, gm2, gm3, gm4);
    op.Process();
}


extern "C" void glm53_proj_output_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3, uint8_t *gm4) {
    glm53_proj_output<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3, gm4);
}
