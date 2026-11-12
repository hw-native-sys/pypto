TPL = r'''#include "kernel_operator.h"

class Kernel_i8probe {
public:
    __aicore__ inline Kernel_i8probe() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0Tensor.SetGlobalBuffer((__gm__ int8_t *)gm0, 1048576);
        gm1Tensor.SetGlobalBuffer((__gm__ int8_t *)gm1, 1048576);
        pipe.InitBuffer(tbuf0, 64);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<int8_t> buf0 = tbuf0.Get<int8_t>();
        // POISON every byte to 0xFF (-1 as int8) through an int32 view, because
        // Duplicate over int8 is not relied on here. 16 * 4 B = the whole 64 B buffer.
        AscendC::LocalTensor<int32_t> poison = buf0.ReinterpretCast<int32_t>();
        AscendC::Duplicate(poison, (int32_t)-1, 16);
        AscendC::PipeBarrier<PIPE_ALL>();
        // THE EXTENT UNDER TEST, in int8 elements = bytes.
        AscendC::DataCopy(buf0, gm0Tensor, %d);
        AscendC::PipeBarrier<PIPE_ALL>();
        // The store is ALWAYS 64 B -- two whole blocks -- so this measures the LOAD
        // floor and not the store's.
        AscendC::DataCopy(gm1Tensor, buf0, 64);
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::GlobalTensor<int8_t> gm0Tensor;
    AscendC::GlobalTensor<int8_t> gm1Tensor;
};

extern "C" __global__ __aicore__ void i8probe(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_i8probe op;
    op.Init(gm0, gm1);
    op.Process();
}

extern "C" void i8probe_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    i8probe<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
'''
for n in (8, 16, 31, 32, 64):
    open("probe-b30/i8_%d.cpp" % n, "w").write(TPL % n)
print("generated i8_8 i8_16 i8_31 i8_32 i8_64")
