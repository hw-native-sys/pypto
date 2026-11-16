set -u
cd /tmp/di-i27
python3 - <<'PY'
import re
src = open("probe-b30/i8_32.cpp").read()
assert "AscendC::DataCopy(buf0, gm0Tensor, 32);" in src
for n in (33, 63):
    open("probe-b30/i8_%d.cpp" % n, "w").write(
        src.replace("AscendC::DataCopy(buf0, gm0Tensor, 32);",
                    "AscendC::DataCopy(buf0, gm0Tensor, %d);" % n))
print("generated i8_33 i8_63 (both fit the existing 64 B buffer)")
PY
# NM 91: does ROUND-DOWN happen ABOVE the floor at int8, and is it visible at the
# DESTINATION? Every round-down figure so far is a COMPILE-layer read of the length
# field; none is a destination-side measurement. And every failing int8 arm so far
# (8, 16, 31 B) is SUB-BLOCK, so the multiple rule is still f32-only.
#
# PREDICTIONS, registered before the run:
#   round-down holds at i8   -> 33 B: arrived=32, poison_bytes=1
#                            -> 63 B: arrived=32, poison_bytes=31
#   arrived=0 on either      -> round-down is f32-specific, or the i8 path differs,
#                               which is a BIGGER finding than the confirmation
#   arrived=33 / 63          -> no round-down at i8 at all
# ESCAPE ARM, which the previous enumeration lacked: any other (arrived, poison)
# pair means the space is mis-drawn -- report it and do not interpret.
echo "=== int8 ragged extents above the floor, 3 reps ==="
for n in 33 63; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/i8_$n.cpp" dav-2201 i8probe_do 5120 2 46 1e-5f 1e-6 1 $n 256 \
      > "logs/nm91_n${n}_rep${rep}.log" 2>&1
    rc=$?
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/nm91_n${n}_rep${rep}.log")
    d=$(grep -m1 -oE "i8_bytes=[0-9]+ arrived=[0-9]+ first_mismatch=-?[0-9]+ poison_bytes=[0-9]+ zeros_in_input=[0-9]+" "logs/nm91_n${n}_rep${rep}.log")
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    printf '  %-3s B rep%s | %-5s | %s\n' "$n" "$rep" "${v:-?(rc=$rc)}" "${d:-no readout}"
  done
done
