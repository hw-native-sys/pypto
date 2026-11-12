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

class Kernel_attn_s2d {
public:
    __aicore__ inline Kernel_attn_s2d() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 256);
        pipe.InitBuffer(tbuf1, 2048);
        pipe.InitBuffer(tbuf2, 32);
        pipe.InitBuffer(tbuf3, 2048);
        pipe.InitBuffer(tbuf4, 2048);
        pipe.InitBuffer(tbuf5, 1024);
        pipe.InitBuffer(tbuf6, 1024);
        pipe.InitBuffer(tbuf7, 32);
        pipe.InitBuffer(tbuf8, 32);
        pipe.InitBuffer(tbuf9, 32);
        pipe.InitBuffer(tbuf10, 32);
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
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        __v0 = 0;
        __v1 = 1;
        __v2 = 8;
        __v3 = 64;
        AscendC::DataCopy(buf0, gm0Tensor[__v0], (__v1) * (__v3));
                                       AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_col_expand: one ROW copied down CE_ROWS rows
          // VECTORISED, one `Adds` per row into that row's slice. The first attempt was a scalar
          // SetValue/GetValue loop and the device returned zeros; it also gets no automatic ordering
          // against the DMA that produced the source, so the barrier is explicit either way.
          // `Adds(dst, src, 0.0f, n)` is this vocabulary's copy, and a row slice is `buf[i * cols]`.
          AscendC::PipeBarrier<PIPE_ALL>();
          int __ce_rows = (__v2); int __ce_cols = (__v3);
          if (__ce_cols == 1) {
            // SCALAR broadcast, at RUNTIME. `Adds` needs 32 B (8 f32) minimum, so a one-element copy is
            // illegal; `Duplicate` is the intrinsic for filling with one value. Deciding this at EMIT time
            // from a resolved string silently failed to fire and left the illegal Adds in place, which
            // faulted the device -- so the branch is in the generated code, where it cannot be missed.
            // V->S: the scalar is read on the SCALAR pipe from a buffer a VECTOR op just wrote.
            // `PipeBarrier<PIPE_ALL>` does not order that pair; the flag/wait does.
            // EVENT_ID1, not ID0: the reduce arms use ID0 for their own V_S pairs, and a second pair on the
            // same id can consume the wrong event -- which reads a stale (in this case inf) scalar.
            // SCALAR-form `Duplicate`. The TENSOR form was tried and has no matching overload in this
            // toolkit ("no matching function for call to 'Duplicate'"), and the scalar READ returns 0 for a
            // tile a vector op just produced. So a scalar broadcast has no working path here yet; this form
            // is kept because it COMPILES, and the failure it produces is a zero rather than a crash.
            float __ce_val = buf0.GetValue(0);
            AscendC::Duplicate(buf1, __ce_val, __ce_rows);
          } else {
            for (int __r = 0; __r < __ce_rows; ++__r)
              AscendC::Adds(buf1[__r * __ce_cols], buf0, 0.0f, __ce_cols);
          }
          AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::DataCopy(buf2, gm2Tensor[__v0], (__v1) * (__v2));
                                       AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(buf3, gm1Tensor[__v0], (__v2) * (__v3));
                                       AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(buf4, buf3, buf1, (__v2) * (__v3));
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_reduce_sum: (r, c) -> (r, 1)
          int __rd_r = (__v2); int __rd_c = (__v3);
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
        AscendC::Add(buf9, buf8, buf2, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(buf10, buf9, (__v1) * (__v2));
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v1) * (__v2)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf10, ((__v1) * (__v2))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor, buf10, __sp); } }
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
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void attn_s2d(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_attn_s2d op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void attn_s2d_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    attn_s2d<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
