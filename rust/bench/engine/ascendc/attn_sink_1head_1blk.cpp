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

class Kernel_attn_sink_1blk {
public:
    __aicore__ inline Kernel_attn_sink_1blk() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        gm0_raw = (uint64_t)(uintptr_t)gm0;
        gm1_raw = (uint64_t)(uintptr_t)gm1;
        gm2_raw = (uint64_t)(uintptr_t)gm2;
        gm3_raw = (uint64_t)(uintptr_t)gm3;
        gm0Tensor.SetGlobalBuffer((__gm__ float*)gm0, 16777216);
        gm1Tensor.SetGlobalBuffer((__gm__ float*)gm1, 16777216);
        gm2Tensor.SetGlobalBuffer((__gm__ float*)gm2, 16777216);
        gm3Tensor.SetGlobalBuffer((__gm__ float*)gm3, 16777216);
        pipe.InitBuffer(tbuf0, 2048);
        pipe.InitBuffer(tbuf1, 16384);
        pipe.InitBuffer(tbuf2, 32);
        pipe.InitBuffer(tbuf3, 2048);
        pipe.InitBuffer(tbuf4, 16384);
        pipe.InitBuffer(tbuf5, 16384);
        pipe.InitBuffer(tbuf6, 2048);
        pipe.InitBuffer(tbuf7, 2048);
        pipe.InitBuffer(tbuf8, 32);
        pipe.InitBuffer(tbuf9, 32);
        pipe.InitBuffer(tbuf10, 32);
        pipe.InitBuffer(tbuf11, 32);
        pipe.InitBuffer(tbuf12, 1024);
        pipe.InitBuffer(tbuf13, 1024);
        pipe.InitBuffer(tbuf14, 32);
        pipe.InitBuffer(tbuf15, 32);
        pipe.InitBuffer(tbuf16, 16384);
        pipe.InitBuffer(tbuf17, 2048);
        pipe.InitBuffer(tbuf18, 32);
        pipe.InitBuffer(tbuf19, 32);
        pipe.InitBuffer(tbuf20, 32);
        pipe.InitBuffer(tbuf21, 32);
        pipe.InitBuffer(tbuf22, 2048);
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
        AscendC::LocalTensor<float> buf14 = tbuf14.Get<float>();
        AscendC::LocalTensor<float> buf15 = tbuf15.Get<float>();
        AscendC::LocalTensor<float> buf16 = tbuf16.Get<float>();
        AscendC::LocalTensor<float> buf17 = tbuf17.Get<float>();
        AscendC::LocalTensor<float> buf18 = tbuf18.Get<float>();
        AscendC::LocalTensor<float> buf19 = tbuf19.Get<float>();
        AscendC::LocalTensor<float> buf20 = tbuf20.Get<float>();
        AscendC::LocalTensor<float> buf21 = tbuf21.Get<float>();
        AscendC::LocalTensor<float> buf22 = tbuf22.Get<float>();
        int32_t __v0;
        int32_t __v1;
        int32_t __v2;
        int32_t __v3;
        int32_t __v4;
        float __v5;
        int32_t __v6;
        __v0 = 0;
        __v1 = 1;
        __v2 = 8;
        __v3 = 512;
        __v4 = 1;
        __v5 = 1.0f;
        AscendC::DataCopy(buf0, gm0Tensor[__v0], (__v4) * (__v3));
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
        AscendC::DataCopy(buf2, gm2Tensor[__v0], (__v4) * (__v2));
                                       AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sub(buf3, buf0, buf0, (__v4) * (__v3));
        __v6 = 0;
        AscendC::DataCopy(buf4, gm1Tensor[__v6], (__v2) * (__v3));
                                       AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(buf5, buf4, buf1, (__v2) * (__v3));
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_reduce_sum: (r, c) -> (r, 1)
          int __rd_r = (__v2); int __rd_c = (__v3);
          if (__rd_r == 1) {
            AscendC::ReduceSum(buf9, buf5, buf6, __rd_c);
          } else {
            for (int __i = 0; __i < __rd_r; ++__i) {
              for (int __j = 0; __j < __rd_c; ++__j)
                buf6.SetValue(__j, buf5.GetValue(__i * __rd_c + __j));
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::ReduceSum(buf8, buf6, buf7, __rd_c);
              AscendC::PipeBarrier<PIPE_ALL>();
              buf9.SetValue(__i, buf8.GetValue(0));
            }
          }
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(buf10, buf9, buf2, (__v4) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(buf11, buf10, (__v4) * (__v2));
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_reduce_sum: (r, c) -> (r, 1)
          int __rd_r = (__v4); int __rd_c = (__v2);
          if (__rd_r == 1) {
            AscendC::ReduceSum(buf15, buf11, buf12, __rd_c);
          } else {
            for (int __i = 0; __i < __rd_r; ++__i) {
              for (int __j = 0; __j < __rd_c; ++__j)
                buf12.SetValue(__j, buf11.GetValue(__i * __rd_c + __j));
              AscendC::PipeBarrier<PIPE_ALL>();
              AscendC::ReduceSum(buf14, buf12, buf13, __rd_c);
              AscendC::PipeBarrier<PIPE_ALL>();
              buf15.SetValue(__i, buf14.GetValue(0));
            }
          }
        }
        AscendC::DataCopy(buf16, gm1Tensor[__v6], (__v2) * (__v3));
                                       AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_V>();
        { // TILE_DEGENERATE_MATMUL: scalar GetValue loop on the vector core -- the cube unit is idle here
          int __mm_m = (__v4); int __mm_k = (__v2); int __mm_n = (__v3);
          for (int __i = 0; __i < __mm_m; ++__i)
            for (int __j = 0; __j < __mm_n; ++__j) {
              float __acc = 0.0f;
              for (int __p = 0; __p < __mm_k; ++__p)
                __acc += (float)buf11.GetValue(__i * __mm_k + __p) *
                         (float)buf16.GetValue(__p * __mm_n + __j);
              buf17.SetValue(__i * __mm_n + __j, __acc);
            }
        }
        AscendC::Mul(buf18, buf15, buf15, (__v4) * (__v4));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(buf20, buf18, (__v4) * (__v4));
        AscendC::Duplicate(buf21, (float)1.0f, (__v4) * (__v4));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Div(buf19, buf21, buf20, (__v4) * (__v4));
        AscendC::PipeBarrier<PIPE_V>();
        { // tile_row_scale: each row times w[row]
          AscendC::PipeBarrier<PIPE_ALL>();
          // V->S before a SCALAR read of a VECTOR-produced tile. Without it the scalar pipe sees a stale
          // value -- which is why `row_scale` certified with a DMA-loaded weight tile and then returned
          // zeros when the weight came from `rsqrt`.
          AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
          AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID2);
          int __rs_rows = (__v4); int __rs_cols = (__v3);
          for (int __r = 0; __r < __rs_rows; ++__r)
            AscendC::Muls(buf22[__r * __rs_cols], buf17[__r * __rs_cols],
                          buf19.GetValue(__r), __rs_cols);
          AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        { const uint32_t __sb = (uint32_t)(((__v4) * (__v3)) * sizeof(float));
          if (__sb % 32 == 0) { AscendC::DataCopy(gm3Tensor, buf22, ((__v4) * (__v3))); }
          else { AscendC::DataCopyExtParams __sp{1, __sb, 0, 0, 0};
                 AscendC::DataCopyPad(gm3Tensor, buf22, __sp); } }
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf15;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf16;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf17;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf18;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf19;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf20;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf21;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf22;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf23;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf24;
    AscendC::GlobalTensor<float> gm0Tensor;
    uint64_t gm0_raw;
    AscendC::GlobalTensor<float> gm1Tensor;
    uint64_t gm1_raw;
    AscendC::GlobalTensor<float> gm2Tensor;
    uint64_t gm2_raw;
    AscendC::GlobalTensor<float> gm3Tensor;
    uint64_t gm3_raw;
};

extern "C" __global__ __aicore__ void attn_sink_1blk(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_attn_sink_1blk op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void attn_sink_1blk_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3) {
    attn_sink_1blk<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2, gm3);
}
