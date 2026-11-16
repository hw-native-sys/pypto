set -u
cd /tmp/di-i27
python3 - <<'PY'
base = open("probe-b30/emb_e1_8.cpp").read()
N = "        AscendC::DataCopy(buf0, gm1Tensor, (1) * (8));"
assert base.count(N) == 1 and base.count("        __v0 = 1;") == 1
POISON = ("        AscendC::Duplicate(buf0, 2.0f, 8);\n"
          "        AscendC::PipeBarrier<PIPE_ALL>();\n")
# isPad = TRUE with paddingValue 0: the documented way to make the remainder DEFINED.
# {isPad, leftPadding, rightPadding, paddingValue}
PADT = ("        AscendC::DataCopyExtParams __lp{1, 4, 0, 0, 0};\n"
        "        AscendC::DataCopyPadExtParams<float> __lq{true, 0, 0, 0.0f};\n"
        "        AscendC::DataCopyPad(buf0, gm1Tensor, __lp, __lq);")
for cell in (1, 7):
    src = base.replace(N, POISON + PADT).replace("        __v0 = 1;", "        __v0 = %d;" % cell)
    open("probe-b30/e89t_c%d.cpp" % cell, "w").write(src)
# And the element-0 control: isPad=true must still DELIVER the index.
src0 = base.replace(N, POISON + PADT).replace("        __v0 = 1;", "        __v0 = 0;")
open("probe-b30/e89t_c0.cpp", "w").write(src0)
print("generated e89t_c0 (delivery control), c1, c7")
PY
# Does isPad=true/paddingValue=0 make the block remainder DEFINED (0.0) instead of a
# copy of element 0 (3.0)? got = 1280*value + 0.125.
#   cell 0 : PASS          -> the index still arrives (the repair still works at all)
#   cell 1,7: got 0.125    -> remainder is ZERO-FILLED and DEFINED
#            got 3840.125  -> still GM[0] everywhere; isPad does not control it
#            got 2560.125  -> poison preserved, the pad wrote only its bytes
for cell in 0 1 7; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/e89t_c$cell.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/nm89c_c${cell}_rep${rep}.log" 2>&1
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/nm89c_c${cell}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/nm89c_c${cell}_rep${rep}.log")
    case "$g" in
      "got 2560.125000") d="2.0  poison preserved -- only its bytes written" ;;
      "got 3840.125000") d="3.0  GM[0] still replicated -- isPad does NOT control it" ;;
      "got 0.125000")    d="0.0  ZERO-FILLED: remainder is defined" ;;
      "")                d="no got line -> verdict ${v:-?} (PASS => value 3.0)" ;;
      *)                 d="unexpected [$g]" ;;
    esac
    printf '  cell %-2s rep%s | %-5s | %s\n' "$cell" "$rep" "${v:-?}" "$d"
  done
done
