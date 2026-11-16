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

class Kernel_k3_kda_delta_128 {
public:
    __aicore__ inline Kernel_k3_kda_delta_128() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm4_raw = (uint64_t)(uintptr_t)gm4;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        gm4Tensor.SetGlobalBuffer((__gm__ float*)gm4, 16777216);
        pipe.InitBuffer(tbuf0, 512);
        pipe.InitBuffer(tbuf1, 512);
        pipe.InitBuffer(tbuf2, 512);
        pipe.InitBuffer(tbuf3, 512);
        pipe.InitBuffer(tbuf4, 512);
        pipe.InitBuffer(tbuf5, 1024);
        pipe.InitBuffer(tbuf6, 1024);
        pipe.InitBuffer(tbuf7, 32);
        pipe.InitBuffer(tbuf8, 32);
        pipe.InitBuffer(tbuf9, 512);
        pipe.InitBuffer(tbuf10, 512);
        pipe.InitBuffer(tbuf11, 512);
        pipe.InitBuffer(tbuf12, 512);
        pipe.InitBuffer(tbuf13, 512);
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
        AscendC::LocalTensor<float> buf9 = tbuf9.Get<float>();
        AscendC::LocalTensor<float> buf10 = tbuf10.Get<float>();
        AscendC::LocalTensor<float> buf11 = tbuf11.Get<float>();
        AscendC::LocalTensor<float> buf12 = tbuf12.Get<float>();
        AscendC::LocalTensor<float> buf13 = tbuf13.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        float __v3;
        float __v4;
        __v0 = 0;
        __v1 = 1;
        __v2 = 128;
        __v3 = 0.5f;
        __v4 = 1.0f;
        AscendC::DataCopy(buf0, gm0Tensor, (1) * (128));
        AscendC::DataCopy(buf1, gm1Tensor, (1) * (128));
        AscendC::DataCopy(buf2, gm2Tensor, (1) * (128));
        AscendC::DataCopy(buf3, gm3Tensor, (1) * (128));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(buf4, buf0, buf1, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_reduce_sum: (r, c) -> (r, 1)
          int __rd_r = (__v1); int __rd_c = (__v2);
          if (__rd_r == 1) {
            AscendC::ReduceSum(buf8, buf4, buf5, __rd_c);
          } else {
            for (int __i = 0; __i < __rd_r; ++__i) {
              for (int __j = 0; __j < __rd_c; ++__j)
                buf5.SetValue(__j, buf4.GetValue(__i * __rd_c + __j));
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::ReduceSum(buf7, buf5, buf6, __rd_c);
              AscendC::PipeBarrier<PIPE_ALL>();
              buf8.SetValue(__i, buf7.GetValue(0));
            }
          }
        }
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_row_expand: one value across the row
          int __re_n = (__v2);
          for (int __i = 0; __i < __re_n; ++__i)
            buf9.SetValue(__i, buf8.GetValue(0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(buf10, buf9, buf2, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(buf11, buf10, (1e0f), (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Muls(buf12, buf3, (5e-1f), (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(buf13, buf12, buf11, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm4Tensor, buf13, ((__v1) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm4Tensor, buf13, __sp); } }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf11;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf12;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf13;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf14;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
    AscendC::GlobalTensor<float> gm4Tensor;
    uint64_t gm4_raw;
};

extern "C" __global__ __aicore__ void k3_kda_delta_128(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4) {
    Kernel_k3_kda_delta_128 op;
    op.Init(gm0, gm1, gm2, gm3, gm4);
    op.Process();
}


extern "C" void k3_kda_delta_128_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3, uint8_t *gm4) {
    k3_kda_delta_128<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3, gm4);
}
