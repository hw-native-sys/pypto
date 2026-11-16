TPL = r'''#include "kernel_operator.h"

class Kernel_stprobe {
public:
    __aicore__ inline Kernel_stprobe() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0Tensor.SetGlobalBuffer((__gm__ float *)gm0, 1048576);
        gm1Tensor.SetGlobalBuffer((__gm__ float *)gm1, 1048576);
        pipe.InitBuffer(tbuf0, 64);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<float> buf0 = tbuf0.Get<float>();
        // Load 64 B -- two WHOLE blocks -- so the load is never the variable here.
        AscendC::DataCopy(buf0, gm0Tensor, 16);
        AscendC::PipeBarrier<PIPE_ALL>();
        // EXACTLY the form mlir_to_cpp emits at its three store sites when the byte
        // count is not a multiple of 32: DataCopyExtParams{1, bytes, 0, 0, 0} then
        // DataCopyPad(gm, buf, params). Nothing here is a probe-only construction.
        AscendC::DataCopyExtParams __sp{1, %d, 0, 0, 0};
        AscendC::DataCopyPad(gm1Tensor, buf0, __sp);
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::GlobalTensor<float> gm0Tensor;
    AscendC::GlobalTensor<float> gm1Tensor;
};

extern "C" __global__ __aicore__ void stprobe(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_stprobe op;
    op.Init(gm0, gm1);
    op.Process();
}

extern "C" void stprobe_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    stprobe<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
'''
for n in (4, 24, 32):
    open("probe-b30/st_%d.cpp" % n, "w").write(TPL % n)
print("generated st_4 st_24 st_32")
