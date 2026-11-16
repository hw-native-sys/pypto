set -u
cd /tmp/di-i27
python3 - <<'PY'
# int8 pad STORE. ORACLE 47's comparison is already byte-based, so the only change
# needed is the kernel's dtype. The load is held at 64 B of whole blocks so it is
# never the variable, and the store goes through the emitter's own pad form.
TPL = r'''#include "kernel_operator.h"

class Kernel_stprobe {
public:
    __aicore__ inline Kernel_stprobe() {}
    __aicore__ inline void Init(GM_ADDR gm0, GM_ADDR gm1) {
        gm0Tensor.SetGlobalBuffer((__gm__ int8_t *)gm0, 1048576);
        gm1Tensor.SetGlobalBuffer((__gm__ int8_t *)gm1, 1048576);
        pipe.InitBuffer(tbuf0, 64);
    }
    __aicore__ inline void Process() {
        AscendC::LocalTensor<int8_t> buf0 = tbuf0.Get<int8_t>();
        AscendC::DataCopy(buf0, gm0Tensor, 64);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams __sp{1, %d, 0, 0, 0};
        AscendC::DataCopyPad(gm1Tensor, buf0, __sp);
        return;
    }
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tbuf0;
    AscendC::GlobalTensor<int8_t> gm0Tensor;
    AscendC::GlobalTensor<int8_t> gm1Tensor;
};

extern "C" __global__ __aicore__ void stprobe(GM_ADDR gm0, GM_ADDR gm1) {
    Kernel_stprobe op; op.Init(gm0, gm1); op.Process();
}
extern "C" void stprobe_do(uint32_t blockDim, void *stream, uint8_t *gm0, uint8_t *gm1) {
    stprobe<<<blockDim, nullptr, stream>>>(gm0, gm1);
}
'''
for n in (4, 24, 32):
    open("probe-b30/i8st_%d.cpp" % n, "w").write(TPL % n)
print("generated i8st_4 i8st_24 i8st_32")
PY
# PREDICTION: if the store is dtype-invariant, these match the f32 arms exactly --
# delivered = the extent, guard_breach = 0. Any other shape means the store's
# behaviour depends on the dtype, which is what the LOAD side needed int8 to reveal.
echo "=== int8 pad STORE, guard 128 B, 3 reps ==="
for n in 32 24 4; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/i8st_$n.cpp" dav-2201 stprobe_do 5120 2 47 1e-5f 1e-6 1 $n 256 \
      > "logs/i8st_n${n}_rep${rep}.log" 2>&1
    rc=$?
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/i8st_n${n}_rep${rep}.log")
    d=$(grep -m1 -oE "store_bytes=.*guard_end=[0-9]+" "logs/i8st_n${n}_rep${rep}.log")
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    printf '  %-2s B rep%s | %-5s | %s\n' "$n" "$rep" "${v:-?(rc=$rc)}" "${d:-no readout}"
  done
done
