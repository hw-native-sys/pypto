#include "kernel_operator.h"

class Kernel_cosq {
public:
    __aicore__ inline Kernel_cosq() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0Tensor.SetGlobalBuffer((__gm__ float *)gm0, 1048576);
        gm1Tensor.SetGlobalBuffer((__gm__ float *)gm1, 1048576);
        pipe.InitBuffer(tbuf0, 2048);
        pipe.InitBuffer(tbuf1, 2048);
        pipe.InitBuffer(tbufw, 8192);   // shared tmp for the adv_api
        pipe.InitBuffer(tbufs, 2048);   // the shifted argument, kept SEPARATE
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> src = tbuf0.Get<float>();
        AscendC::LocalTensor<float> dst = tbuf1.Get<float>();
        AscendC::LocalTensor<uint8_t> work = tbufw.Get<uint8_t>();
        AscendC::DataCopy(src, gm0Tensor, 512);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<float> shifted = tbufs.Get<float>();
        AscendC::Adds(shifted, src, -1.5707963267948966f, 512);
        AscendC::PipeBarrier<PIPE_V>();
        // CONFORMING: src, dst and the shared tmp are three DISTINCT tiles. The
        // header states overlap is not allowed and the previous probe aliased src
        // into dst, so it measured a form the vendor does not sanction.
        AscendC::Cos(dst, shifted, work, 512);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gm1Tensor, dst, 512);
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0, tbuf1, tbufw, tbufs;
    AscendC::GlobalTensor<float> gm0Tensor, gm1Tensor;
};

extern "C" __global__ __aicore__ void cosq(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_cosq op; op.Init(gm0, gm1); op.Process();
}
extern "C" void cosq_do(uint32_t bd, void *st, uint8_t *gm0, uint8_t *gm1) {
    cosq<<<bd, nullptr, st>>>(gm0, gm1);
}
