set -u
cd /tmp/di-i27
python3 - <<'PY'
base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
IB = "        pipe.InitBuffer(tbuf0, 32);"
assert base.count(N) == 1 and base.count(IB) == 1 and base.count("        __v0 = 1;") == 1
# RAISE THE DESTINATION TO TWO BLOCKS. At 32 B every arm's "remainder of the final
# partial block" and "whole destination" are the same cells, so a confined fill and
# a whole-destination fill are indistinguishable. At 64 B they are not.
b64 = base.replace(IB, "        pipe.InitBuffer(tbuf0, 64);")
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 16);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
def pad(nb):
    return ("        AscendC::DataCopyExtParams __lp{1, %d, 0, 0, 0};\n"
            "        AscendC::DataCopyPadExtParams<float> __lq{false, 0, 0, 0};\n"
            "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);" % nb)
# 36 B = 9 f32, so cells 0..8 are the logical extent and cell 1 is INSIDE it, in the
# FIRST block, with nothing legitimate to fill there.
#   confined fill          -> cell 1 = gm1[1] = 0.0 -> index 0 -> FAIL, got 0.125
#   whole-destination fill -> cell 1 = gm1[0] = 3.0 -> index 3 -> PASS
# The 8 B and 16 B arms are RE-RUN at the new buffer size: without them a change at
# 36 B could not be told from a change caused by raising the buffer.
for tag, nb, cell in (("36_c1", 36, 1), ("36_c0", 36, 0), ("8_c1", 8, 1), ("16_c3", 16, 3)):
    open("probe-b30/n95_%s.cpp" % tag, "w").write(
        b64.replace(N, POISON + pad(nb)).replace("        __v0 = 1;",
                                                 "        __v0 = %d;" % cell))
print("generated n95_36_c1 (the test), 36_c0 (control), 8_c1 + 16_c3 (buffer-size controls)")
PY
for k in 36_c0 8_c1 16_c3 36_c1; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/n95_$k.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/n95_${k}_rep${rep}.log" 2>&1
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/n95_${k}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/n95_${k}_rep${rep}.log")
    case "$g" in
      "got 0.125000")    d="0.0 = gm1[cell] -> fill CONFINED to the tail" ;;
      "got 2560.125000") d="2.0 = poison     -> cell never written" ;;
      "got 3840.125000") d="3.0 = gm1[0]     -> WHOLE-DESTINATION fill" ;;
      "")                d="no got line -> verdict ${v:-?} (PASS => 3.0)" ;;
      *)                 d="unexpected [$g] -- escape arm" ;;
    esac
    printf '  %-6s rep%s | %-5s | %s\n' "$k" "$rep" "${v:-?}" "$d"
  done
done
echo
echo "=== NM 95 (ii), compile-only: the PAD LOAD's length field ==="
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2
B=$ASCEND_HOME_PATH/bin/bisheng
mkdir -p nm95e && cd nm95e
for k in 8_c1 16_c3; do
  cp ../probe-b30/n95_$k.cpp ../op_kernel/vec_add_kernel.asc
  $B -DKERNEL_ARGS=3 -DKERNEL_DO=v41_embedding_5120_do -DKERNEL_N=5120 -DKERNEL_REPEAT=1 \
     -DMV_K=256 -DMV_N=64 -DORACLE=44 -DRMS_EPS=1e-5f -DTOLERANCE=1e-6f \
     -I/tmp/di-i27/op_kernel -I/tmp/di-i27/op_host -fPIC --npu-arch=dav-2201 \
     -o e$k.o -c -x asc /tmp/di-i27/op_host/main.asc 2>/dev/null
  /usr/bin/objcopy --dump-section .aicore_binary=ed$k.bin e$k.o 2>/dev/null
  printf "  pad load %-6s | .aicore_binary %s B | md5=%s\n" "$k" \
    "$(stat -c%s ed$k.bin 2>/dev/null || echo NONE)" "$(md5sum ed$k.bin 2>/dev/null | cut -c1-12 || echo -)"
done
echo "  --- 8 B vs 16 B pad load, differing bytes (cmp -l, 1-based octal) ---"
cmp -l ed8_c1.bin ed16_c3.bin 2>/dev/null | head -6
