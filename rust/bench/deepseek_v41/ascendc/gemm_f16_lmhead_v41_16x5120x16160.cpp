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

class Kernel_bm_gemm_f16_16_5120_16160 {
public:
    __aicore__ inline Kernel_bm_gemm_f16_16_5120_16160() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
        aGM.SetGlobalBuffer((__gm__ half*)gm0, 81920);
        bGM.SetGlobalBuffer((__gm__ half*)gm1, 82739200);
        cGM.SetGlobalBuffer((__gm__ half*)gm2, 258560);
        pipe.InitBuffer(qA1, 1, 4096 * sizeof(half));
        pipe.InitBuffer(qB1, 1, 4096 * sizeof(half));
        pipe.InitBuffer(qA2, 1, 4096 * sizeof(half));
        pipe.InitBuffer(qB2, 1, 4096 * sizeof(half));
        pipe.InitBuffer(qCO1, 1, 512 * sizeof(float));
        pipe.InitBuffer(qCO2, 1, 512 * sizeof(float));
        pipe.InitBuffer(qOut, 1, 512 * sizeof(half));
    }
    __aicore__ inline void Process() {
#ifdef __DAV_C220_CUBE__
        constexpr int M = 16, K = 5120, N = 16160;
        constexpr int MPAD = 16, KB = 256, NB = 32, F = 16;
        for (int n0 = 0; n0 < N; n0 += NB) {
          const int nw = (N - n0 < NB) ? (N - n0) : NB;
          AscendC::LocalTensor<float> c1 = qCO1.AllocTensor<float>();
          for (int k0 = 0; k0 < K; k0 += KB) {
            const int kw = (K - k0 < KB) ? (K - k0) : KB;
            AscendC::LocalTensor<half> l1a = qA1.AllocTensor<half>();
            AscendC::Nd2NzParams pa;
            pa.ndNum = 1; pa.nValue = M; pa.dValue = kw;
            pa.srcNdMatrixStride = 0; pa.srcDValue = K;
            pa.dstNzC0Stride = MPAD; pa.dstNzNStride = 1; pa.dstNzMatrixStride = 0;
            AscendC::DataCopy(l1a, aGM[k0], pa);
            qA1.EnQue(l1a);
            AscendC::LocalTensor<half> a1 = qA1.DeQue<half>();
            AscendC::LocalTensor<half> a2 = qA2.AllocTensor<half>();
            AscendC::LoadData2dParams la;
            la.repeatTimes = (uint8_t)(kw / F);
            la.srcStride = 1; la.ifTranspose = false;
            AscendC::LoadData(a2, a1, la);
            qA2.EnQue(a2); qA1.FreeTensor(a1);
            AscendC::LocalTensor<half> a2r = qA2.DeQue<half>();
            for (int g = 0; g < nw; g += F) {
              const int gw = (nw - g < F) ? (nw - g) : F;
              AscendC::LocalTensor<half> l1b = qB1.AllocTensor<half>();
              AscendC::Nd2NzParams pb;
              pb.ndNum = 1; pb.nValue = kw; pb.dValue = gw;
              pb.srcNdMatrixStride = 0; pb.srcDValue = N;
              pb.dstNzC0Stride = kw; pb.dstNzNStride = 1; pb.dstNzMatrixStride = 0;
              AscendC::DataCopy(l1b, bGM[(int64_t)k0 * N + n0 + g], pb);
              qB1.EnQue(l1b);
              AscendC::LocalTensor<half> b1 = qB1.DeQue<half>();
              AscendC::LocalTensor<half> b2 = qB2.AllocTensor<half>();
              AscendC::LoadData2dParams lb;
              lb.repeatTimes = (uint8_t)(kw / F);
              lb.srcStride = 1; lb.ifTranspose = true;
              AscendC::LoadData(b2, b1, lb);
              qB2.EnQue(b2); qB1.FreeTensor(b1);
              AscendC::LocalTensor<half> b2r = qB2.DeQue<half>();
              AscendC::MmadParams mp;
              mp.m = MPAD; mp.n = gw; mp.k = kw;
              mp.isBias = (k0 != 0);
              mp.cmatrixSource = false;
              mp.cmatrixInitVal = (k0 == 0);
              AscendC::Mmad(c1[g * MPAD * F], a2r, b2r, mp);
              AscendC::PipeBarrier<PIPE_ALL>();
              qB2.FreeTensor(b2r);
            }
            qA2.FreeTensor(a2r);
          }
          qCO1.EnQue(c1);
          AscendC::LocalTensor<float> c1r = qCO1.DeQue<float>();
          for (int g = 0; g < nw; g += F) {
            const int gw = (nw - g < F) ? (nw - g) : F;
            AscendC::FixpipeParamsV220 fp;
            fp.nSize = (uint16_t)gw;
            fp.mSize = (uint16_t)M;
            fp.srcStride = (uint16_t)MPAD;
            fp.dstStride = (uint32_t)N;
            fp.reluEn = false;
            fp.quantPre = QuantMode_t::F322F16;
            AscendC::Fixpipe<half, float, AscendC::CFG_ROW_MAJOR>(
                cGM[(int64_t)n0 + g], c1r[g * MPAD * F], fp);
          }
          qCO1.FreeTensor(c1r);
        }
#endif  // __DAV_C220_CUBE__
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::A1, 1> qA1;
    AscendC::TQue<AscendC::QuePosition::B1, 1> qB1;
    AscendC::TQue<AscendC::QuePosition::A2, 1> qA2;
    AscendC::TQue<AscendC::QuePosition::B2, 1> qB2;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> qCO1;
    AscendC::TQue<AscendC::QuePosition::CO2, 1> qCO2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qOut;
    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<half> cGM;
};

extern "C" __global__ __aicore__ void bm_gemm_f16_16_5120_16160(GM_ADDR gm0, GM_ADDR gm1, GM_ADDR gm2) {
    Kernel_bm_gemm_f16_16_5120_16160 op;
    op.Init(gm0, gm1, gm2);
    op.Process();
}


extern "C" void bm_gemm_f16_16_5120_16160_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1, uint8_t *gm2) {
    bm_gemm_f16_16_5120_16160<<<blockDim, nullptr, stream>>>(gm0, gm1, gm2);
}
