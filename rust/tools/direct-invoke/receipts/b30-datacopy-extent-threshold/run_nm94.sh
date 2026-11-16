set -u
cd /tmp/di-i27
python3 - <<'PY'
base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(N) == 1 and base.count("        __v0 = 1;") == 1
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
def pad(nbytes):
    return ("        AscendC::DataCopyExtParams __lp{1, %d, 0, 0, 0};\n"
            "        AscendC::DataCopyPadExtParams<float> __lq{false, 0, 0, 0};\n"
            "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);" % nbytes)
# The trick: read a cell that is INSIDE the requested extent. gm1 holds
# [3.0, 0.0, 0.0, ...], so for an 8-byte (2 f32) pad:
#   (a) honours the length -> cell 1 = gm1[1] = 0.0 -> index 0 -> FAIL, got 0.125
#   (b) fills the block    -> cell 1 = gm1[0] = 3.0 -> index 3 -> PASS
# Both are legal indices for a 4-row table, so neither path can fault, and the
# readout is the oracle's own verdict rather than a decoded residue.
for (tag, nb, cell) in (("a8_c1", 8, 1), ("a8_c0", 8, 0), ("a16_c3", 16, 3)):
    src = base.replace(N, POISON + pad(nb)).replace("        __v0 = 1;",
                                                    "        __v0 = %d;" % cell)
    open("probe-b30/n94_%s.cpp" % tag, "w").write(src)
print("generated n94_a8_c1 (the test), n94_a8_c0 (control), n94_a16_c3 (second extent)")
PY
# a8_c0 is the CONTROL and must deliver the index: without it an unexpected cell-1
# reading cannot be attributed, which is the defect both of the last two probes had.
for k in a8_c0 a8_c1 a16_c3; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/n94_$k.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/n94_${k}_rep${rep}.log" 2>&1
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/n94_${k}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/n94_${k}_rep${rep}.log")
    case "$g" in
      "got 0.125000")    d="0.0 = gm1[cell]  -> (a) HONOURS THE LENGTH" ;;
      "got 2560.125000") d="2.0 = poison      -> cell not written at all" ;;
      "got 3840.125000") d="3.0 = gm1[0]      -> (b) FILLS THE BLOCK" ;;
      "")                d="no got line -> verdict ${v:-?} (PASS => 3.0 => (b), or control ok)" ;;
      *)                 d="unexpected [$g] -- escape arm, do not interpret" ;;
    esac
    printf '  %-7s rep%s | %-5s | %s\n' "$k" "$rep" "${v:-?}" "$d"
  done
done
echo
echo "=== FREE, no device: is the STORE's length encoded the same way? ==="
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2
B=$ASCEND_HOME_PATH/bin/bisheng
mkdir -p nm87s && cd nm87s
for n in 24 32; do
  cp ../probe-b30/st_$n.cpp ../op_kernel/vec_add_kernel.asc
  $B -DKERNEL_ARGS=3 -DKERNEL_DO=stprobe_do -DKERNEL_N=5120 -DKERNEL_REPEAT=1 \
     -DMV_K=256 -DMV_N=4 -DORACLE=47 -DRMS_EPS=1e-5f -DTOLERANCE=1e-6f \
     -I/tmp/di-i27/op_kernel -I/tmp/di-i27/op_host -fPIC --npu-arch=dav-2201 \
     -o s$n.o -c -x asc /tmp/di-i27/op_host/main.asc 2>/dev/null
  /usr/bin/objcopy --dump-section .aicore_binary=sd$n.bin s$n.o 2>/dev/null
  printf "  store %-3s B | .aicore_binary %s bytes md5=%s\n" "$n" \
    "$(stat -c%s sd$n.bin 2>/dev/null)" "$(md5sum sd$n.bin 2>/dev/null | cut -c1-12)"
done
echo "  differing bytes 24 vs 32:"; cmp -l sd24.bin sd32.bin 2>/dev/null | head -6
