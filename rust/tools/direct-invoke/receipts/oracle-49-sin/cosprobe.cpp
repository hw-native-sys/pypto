#include "kernel_operator.h"

class Kernel_cosp {
public:
    __aicore__ inline Kernel_cosp() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0Tensor.SetGlobalBuffer((__gm__ float *)gm0, 1048576);
        gm1Tensor.SetGlobalBuffer((__gm__ float *)gm1, 1048576);
        pipe.InitBuffer(tbuf0, 2048);
        pipe.InitBuffer(tbuf1, 2048);
        pipe.InitBuffer(tbufw, 8192);   // shared tmp for the adv_api
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> src = tbuf0.Get<float>();
        AscendC::LocalTensor<float> dst = tbuf1.Get<float>();
        AscendC::LocalTensor<uint8_t> work = tbufw.Get<uint8_t>();
        AscendC::DataCopy(src, gm0Tensor, 512);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Adds(dst, src, -1.5707963267948966f, 512);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cos(dst, dst, work, 512);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(gm1Tensor, dst, 512);
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0, tbuf1, tbufw;
    AscendC::GlobalTensor<float> gm0Tensor, gm1Tensor;
};

extern "C" __global__ __aicore__ void cosp(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_cosp op; op.Init(gm0, gm1); op.Process();
}
extern "C" void cosp_do(uint32_t bd, void *st, uint8_t *gm0, uint8_t *gm1) {
    cosp<<<bd, nullptr, st>>>(gm0, gm1);
}
