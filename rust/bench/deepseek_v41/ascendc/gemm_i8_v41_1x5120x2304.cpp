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
#include "acl/acl_rt.h"

#ifndef TILE_OUT_FILL
#define TILE_OUT_FILL 0
#endif

class Kernel_bm_gemm_i8_1_5120_2304_dequant {
public:
    __aicore__ inline Kernel_bm_gemm_i8_1_5120_2304_dequant() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4, int64_t nb, int64_t mode) {
        constexpr int CH = 2048;
        nb_ = (int)nb;
        mode_ = (int)mode;
        aIn.SetGlobalBuffer((__gm__ int8_t*)gm0, (int64_t)1 * 5120);
        bIn.SetGlobalBuffer((__gm__ int8_t*)gm1, (int64_t)5120 * 2304);
        scGm.SetGlobalBuffer((__gm__ float*)gm2, 2304);
        aOut.SetGlobalBuffer((__gm__ half*)gm3, 5120);
        bOut.SetGlobalBuffer((__gm__ half*)gm4, (int64_t)5120 * 2304);
        pipe.InitBuffer(bI, CH * sizeof(int8_t));
        pipe.InitBuffer(bH, CH * sizeof(half));
        pipe.InitBuffer(bF, CH * sizeof(float));
        pipe.InitBuffer(bS, CH * sizeof(float));
    }
    __aicore__ inline void Process() {
        constexpr int M = 1, K = 5120, N = 2304, CH = 2048;
        const int blk = (int)AscendC::GetBlockIdx();
        AscendC::LocalTensor<int8_t> in = bI.Get<int8_t>();
        AscendC::LocalTensor<half> h = bH.Get<half>();
        AscendC::LocalTensor<float> f = bF.Get<float>();
        AscendC::LocalTensor<float> sc = bS.Get<float>();
        AscendC::DataCopyPadExtParams<int8_t> p8{true, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pf{true, 0, 0, 0.0f};
        AscendC::DataCopyPadExtParams<half> ph{true, 0, 0, (half)0};
        if (mode_ == 0) {
            if (blk < M) {
                for (int k0 = 0; k0 < K; k0 += CH) {
                    int nn = K - k0;
                    if (nn > CH) nn = CH;
                    const uint32_t na = (uint32_t)((nn + 63) / 64 * 64);
                    AscendC::DataCopyExtParams ci{1, (uint32_t)(nn), 0, 0, 0};
                    AscendC::DataCopyPad(in, aIn[(int64_t)blk * K + k0], ci, p8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Cast(h, in, AscendC::RoundMode::CAST_NONE, na);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyExtParams co{1, (uint32_t)(nn * 2), 0, 0, 0};
                    AscendC::DataCopyPad(aOut[(int64_t)blk * K + k0], h, co);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
            for (int r = blk; r < K; r += nb_) {
                for (int n0 = 0; n0 < N; n0 += CH) {
                    int nn = N - n0;
                    if (nn > CH) nn = CH;
                    const uint32_t na = (uint32_t)((nn + 63) / 64 * 64);
                    AscendC::DataCopyExtParams ci{1, (uint32_t)(nn), 0, 0, 0};
                    AscendC::DataCopyPad(in, bIn[(int64_t)r * N + n0], ci, p8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Cast(h, in, AscendC::RoundMode::CAST_NONE, na);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyExtParams co{1, (uint32_t)(nn * 2), 0, 0, 0};
                    AscendC::DataCopyPad(bOut[(int64_t)r * N + n0], h, co);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
        } else {
            for (int r = blk; r < M; r += nb_) {
                for (int n0 = 0; n0 < N; n0 += CH) {
                    int nn = N - n0;
                    if (nn > CH) nn = CH;
                    const uint32_t na = (uint32_t)((nn + 63) / 64 * 64);
                    AscendC::DataCopyExtParams cf{1, (uint32_t)(nn * 4), 0, 0, 0};
                    AscendC::DataCopyPad(sc, scGm[n0], cf, pf);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyExtParams ch{1, (uint32_t)(nn * 2), 0, 0, 0};
                    AscendC::DataCopyPad(h, aOut[(int64_t)r * N + n0], ch, ph);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::Cast(f, h, AscendC::RoundMode::CAST_NONE, na);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(f, f, sc, na);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(h, f, AscendC::RoundMode::CAST_RINT, na);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyPad(bOut[(int64_t)r * N + n0], h, ch);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bI, bH, bF, bS;
    AscendC::GlobalTensor<int8_t> aIn, bIn;
    AscendC::GlobalTensor<float> scGm;
    AscendC::GlobalTensor<half> aOut, bOut;
    int nb_ = 1;
    int mode_ = 0;
};

extern "C" __global__ __aicore__ void bm_gemm_i8_1_5120_2304_dequant(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3, GM_ADDR gm4, int64_t nb, int64_t mode) {
    Kernel_bm_gemm_i8_1_5120_2304_dequant op;
    op.Init(gm0, gm1, gm2, gm3, gm4, nb, mode);
    op.Process();
}

class Kernel_bm_gemm_i8_1_5120_2304_mv {
public:
    __aicore__ inline Kernel_bm_gemm_i8_1_5120_2304_mv() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
        aGm.SetGlobalBuffer((__gm__ half*)gm0, (int64_t)1 * 5120);
        bGm.SetGlobalBuffer((__gm__ half*)gm1, (int64_t)5120 * 2304);
        scGm.SetGlobalBuffer((__gm__ float*)gm2, 2304);
        cGm.SetGlobalBuffer((__gm__ half*)gm3, (int64_t)1 * 2304);
        pipe.InitBuffer(bA, 5120 * sizeof(half));
        pipe.InitBuffer(bAF, 5120 * sizeof(float));
        pipe.InitBuffer(bBH, 48 * sizeof(half));
        pipe.InitBuffer(bBF, 48 * sizeof(float));
        pipe.InitBuffer(bProd, 48 * sizeof(float));
        pipe.InitBuffer(bAcc, 48 * sizeof(float));
        pipe.InitBuffer(bSc, 48 * sizeof(float));
        pipe.InitBuffer(bOut, 48 * sizeof(half));
    }
    __aicore__ inline void Process() {
        constexpr int M = 1, K = 5120, N = 2304, NC = 48;
        const int blk = (int)AscendC::GetBlockIdx();
        const int n0 = blk * NC;
        if (n0 >= N) return;
        const int nw = (N - n0 < NC) ? (N - n0) : NC;
        const int na = (nw + 7) / 8 * 8;
        AscendC::LocalTensor<half> aH = bA.Get<half>();
        AscendC::LocalTensor<float> aF = bAF.Get<float>();
        AscendC::LocalTensor<half> bH = bBH.Get<half>();
        AscendC::LocalTensor<float> bF = bBF.Get<float>();
        AscendC::LocalTensor<float> prod = bProd.Get<float>();
        AscendC::LocalTensor<float> acc = bAcc.Get<float>();
        AscendC::LocalTensor<float> sc = bSc.Get<float>();
        AscendC::LocalTensor<half> out = bOut.Get<half>();
        AscendC::DataCopyPadExtParams<half> ph{true, 0, 0, (half)0};
        AscendC::DataCopyPadExtParams<float> pf{true, 0, 0, 0.0f};
        AscendC::DataCopyExtParams ca{1, (uint32_t)(K * 2), 0, 0, 0};
        AscendC::DataCopyPad(aH, aGm[0], ca, ph);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(aF, aH, AscendC::RoundMode::CAST_NONE, K);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::Duplicate(acc, 0.0f, NC);
        AscendC::PipeBarrier<PIPE_V>();
        for (int r = 0; r < K; ++r) {
            AscendC::DataCopyExtParams cb{1, (uint32_t)(nw * 2), 0, 0, 0};
            AscendC::DataCopyPad(bH, bGm[(int64_t)r * N + n0], cb, ph);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(bF, bH, AscendC::RoundMode::CAST_NONE, na);
            AscendC::PipeBarrier<PIPE_V>();
            const float av = aF.GetValue(r);
            AscendC::Muls(prod, bF, av, na);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(acc, acc, prod, na);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::DataCopyExtParams cs{1, (uint32_t)(nw * 4), 0, 0, 0};
        AscendC::DataCopyPad(sc, scGm[n0], cs, pf);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Mul(acc, acc, sc, na);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, na);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams co{1, (uint32_t)(nw * 2), 0, 0, 0};
        AscendC::DataCopyPad(cGm[n0], out, co);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bA, bAF, bBH, bBF, bProd, bAcc, bSc, bOut;
    AscendC::GlobalTensor<half> aGm, bGm, cGm;
    AscendC::GlobalTensor<float> scGm;
};

extern "C" __global__ __aicore__ void bm_gemm_i8_1_5120_2304_mv(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2, GM_ADDR gm3) {
    Kernel_bm_gemm_i8_1_5120_2304_mv op;
    op.Init(gm0, gm1, gm2, gm3);
    op.Process();
}


extern "C" void bm_gemm_i8_1_5120_2304_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2, uint8_t *gm3, uint8_t *gm4) {
    // i8 GEMM at M=1: the AIV kernel materialises the f16 operands, then a
    // vector f16 dot per output column applies the per-column scale (QmPost).
    void *a_half = nullptr;
    void *b_half = nullptr;
    if (aclrtMalloc(&a_half, (size_t)1 * 5120 * 2, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return;
    if (aclrtMalloc(&b_half, (size_t)5120 * 2304 * 2, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { aclrtFree(a_half); return; }
    bm_gemm_i8_1_5120_2304_dequant<<<48, nullptr, stream>>>(gm0, gm1, gm3, (uint8_t *)a_half, (uint8_t *)b_half, 48, 0);
    aclrtSynchronizeStream(stream);
    bm_gemm_i8_1_5120_2304_mv<<<48, nullptr, stream>>>((uint8_t *)a_half, (uint8_t *)b_half, gm3, gm2);
    aclrtSynchronizeStream(stream);
    aclrtFree(a_half);
    aclrtFree(b_half);
    (void)gm4;
}
