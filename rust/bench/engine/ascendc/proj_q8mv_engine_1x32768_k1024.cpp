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

class Kernel_proj_q8mv_engine_1x32768_k1024 {
public:
    __aicore__ inline Kernel_proj_q8mv_engine_1x32768_k1024() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ int8_t*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 1024);
        pipe.InitBuffer(tbuf1, 4096);
        pipe.InitBuffer(tbuf2, 4096);
        pipe.InitBuffer(tbuf3, 4096);
        pipe.InitBuffer(tbuf4, 4096);
        pipe.InitBuffer(tbuf5, 32);
        pipe.InitBuffer(tbuf6, 2048);
        pipe.InitBuffer(tbuf7, 131072);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<int8_t> buf0 = tbuf0.Get<int8_t>();
        AscendC::LocalTensor<float> buf1 = tbuf1.Get<float>();
        AscendC::LocalTensor<float> buf2 = tbuf2.Get<float>();
        AscendC::LocalTensor<float> buf3 = tbuf3.Get<float>();
        AscendC::LocalTensor<float> buf4 = tbuf4.Get<float>();
        AscendC::LocalTensor<float> buf5 = tbuf5.Get<float>();
        AscendC::LocalTensor<half> buf6 = tbuf6.Get<half>();
        AscendC::LocalTensor<float> buf7 = tbuf7.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 1;
        __v1 = 1024;
        __v2 = 32768;
        __v3 = 0;
        { // tile_matvec_q8mv: out[r] = sum_k w8[r,k] * x[k] * s[r,k]
          int __n = (__v2); int __k = (__v1);
          AscendC::DataCopy(buf1, gm2Tensor, __k);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
          for (int __i = 0; __i < __n; ++__i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::DataCopy(buf0, gm0Tensor[__i * __k], __k);
            AscendC::DataCopy(buf2, gm1Tensor[__i * __k], __k);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(buf6, buf0, AscendC::RoundMode::CAST_NONE, __k);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(buf3, buf6, AscendC::RoundMode::CAST_NONE, __k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(buf3, buf3, buf1, __k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(buf3, buf3, buf2, __k);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::ReduceSum(buf5, buf3, buf4, __k);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            buf7.SetValue(__i, buf5.GetValue(0));
          }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v0) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf7, ((__v0) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor, buf7, __sp); } }
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
    AscendC::GlobalTensor<int8_t> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void proj_q8mv_engine_1x32768_k1024(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_proj_q8mv_engine_1x32768_k1024 op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void proj_q8mv_engine_1x32768_k1024_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    proj_q8mv_engine_1x32768_k1024<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
