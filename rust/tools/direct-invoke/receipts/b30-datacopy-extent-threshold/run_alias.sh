set -u
cd /tmp/di-i27
python3 - <<'PY'
base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(N) == 1 and base.count("        __v0 = 1;") == 1
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
PAD4 = ("        AscendC::DataCopyExtParams __lp{1, 4, 0, 0, 0};\n"
        "        AscendC::DataCopyPadExtParams<float> __lq{false, 0, 0, 0};\n"
        "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);\n")
# FILL vs ALIAS. The 4-byte pad load leaves cell 1 reading 3.0 = GM[0]. Two readings
# fit: the block was FILLED with element 0's value, or every read ALIASES element 0.
# Overwrite element 0 only, then read cell 1:
#   filled  -> cell 1 still 3.0 -> index 3 -> PASS
#   aliased -> cell 1 reads 1.0 -> index 1 -> got 1280.125 -> FAIL
# Both are legal indices for a 4-row table, so neither path can fault.
DISC = PAD4 + ("        AscendC::PipeBarrier<PIPE_ALL>();\n"
               "        buf0.SetValue(0, 1.0f);\n"
               "        AscendC::PipeBarrier<PIPE_ALL>();")
open("probe-b30/alias_1.cpp", "w").write(base.replace(N, POISON + DISC))
# CONTROL: same build, same SetValue, but reading cell 0 -- must see 1.0, proving the
# SetValue landed at all. got 1280.125 expected.
open("probe-b30/alias_c0.cpp", "w").write(
    base.replace(N, POISON + DISC).replace("        __v0 = 1;", "        __v0 = 0;"))
print("generated alias_1 (cell 1) and alias_c0 (SetValue control)")
PY
for k in alias_1 alias_c0; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/$k.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/alias_${k}_rep${rep}.log" 2>&1
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/alias_${k}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/alias_${k}_rep${rep}.log")
    case "$g" in
      "got 1280.125000") d="value 1.0  -> the SetValue is what this cell reads" ;;
      "got 2560.125000") d="value 2.0  -> poison, untouched by the pad" ;;
      "got 3840.125000") d="value 3.0  -> GM[0], i.e. the block was FILLED" ;;
      "")                d="no got line -> verdict ${v:-?} (PASS => value 3.0, FILLED)" ;;
      *)                 d="unexpected [$g] -- escape arm, do not interpret" ;;
    esac
    printf '  %-9s rep%s | %-5s | %s\n' "$k" "$rep" "${v:-?}" "$d"
  done
done
