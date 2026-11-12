set -u
cd /tmp/di-i27
python3 - <<'PY'
TPL = open("probe-b30/st_24.cpp").read()
assert "AscendC::DataCopyExtParams __sp{1, 24, 0, 0, 0};" in TPL
PAD = """        AscendC::DataCopyExtParams __sp{1, 24, 0, 0, 0};
        AscendC::DataCopyPad(gm1Tensor, buf0, __sp);"""
assert TPL.count(PAD) == 1
# ARM A -- GUARD CALIBRATION (NM 92). Declared extent stays 4 B (MV_N=4 on the
# command line) but the kernel stores 64 B with a PLAIN DataCopy. Bytes 4..63 are
# therefore written on purpose, so the guard MUST fire: guard_breach = 60 and
# first_breach = 4. If it reports 0, the detector is dead and all nine of the
# store probe's PASSes are uninterpretable.
open("probe-b30/g92_cal.cpp", "w").write(
    TPL.replace(PAD, "        AscendC::DataCopy(gm1Tensor, buf0, 16);"))
# ARM B -- THE FOURTH CELL (NM 93). A PLAIN DataCopy store of ONE f32 = 4 bytes.
# bisheng encodes a plain DataCopy's length as floor(bytes/32)*32, which is 0 here
# -- but that was measured on LOADS. If a plain 4-byte store delivers, the rounding
# is load-specific and NM 87 needs re-scoping from "a plain DataCopy" to "a plain
# DataCopy LOAD"; if it delivers nothing, the rounding is a property of the plain
# lowering in both directions.
open("probe-b30/g93_plain4.cpp", "w").write(
    TPL.replace(PAD, "        AscendC::DataCopy(gm1Tensor, buf0, 1);"))
print("generated g92_cal (guard calibration) and g93_plain4 (plain 4 B store)")
PY
# Read in this order: the calibration arm DECIDES whether arm B is interpretable.
for k in g92_cal g93_plain4; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/$k.cpp" dav-2201 stprobe_do 5120 2 47 1e-5f 1e-6 1 4 256 \
      > "logs/nm92_${k}_rep${rep}.log" 2>&1
    rc=$?
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/nm92_${k}_rep${rep}.log")
    d=$(grep -m1 -oE "store_bytes=.*guard_end=[0-9]+" "logs/nm92_${k}_rep${rep}.log")
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    printf '  %-11s rep%s | %-5s | %s\n' "$k" "$rep" "${v:-?(rc=$rc)}" "${d:-no readout}"
  done
done
