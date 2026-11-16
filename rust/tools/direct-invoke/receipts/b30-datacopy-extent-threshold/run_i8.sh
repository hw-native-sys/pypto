set -u
cd /tmp/di-i27
python3 gen_i8.py || exit 1
# ORACLE 46: does the 32-byte floor hold at a NARROW dtype, or is it 8 ELEMENTS?
# At f32 those coincide (8 f32 = 32 B) and every extent ever run is consistent with
# both. At int8 they differ fourfold. The emitter enforces the BYTE rule as the
# conservative reading; this says whether that is four times stricter than the part.
#
# PREDICTIONS, registered before the run:
#   a 32-BYTE floor    -> 8, 16, 31 arrive 0 of N (all poison); 32 and 64 arrive fully
#   an 8-ELEMENT floor -> every N >= 8 arrives fully
echo "=== int8 load floor, 3 reps ==="
for n in 64 32 31 16 8; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/i8_$n.cpp" dav-2201 i8probe_do 5120 2 46 1e-5f 1e-6 1 $n 256 \
      > "logs/i8_n${n}_rep${rep}.log" 2>&1
    rc=$?
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/i8_n${n}_rep${rep}.log")
    d=$(grep -m1 -oE "i8_bytes=[0-9]+ arrived=[0-9]+ first_mismatch=-?[0-9]+ poison_bytes=[0-9]+ zeros_in_input=[0-9]+" "logs/i8_n${n}_rep${rep}.log")
    r=$(grep -m1 -oE "ORACLE 46 refuses.*|507[0-9]{3}|error: .*" "logs/i8_n${n}_rep${rep}.log" | cut -c1-70)
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    printf '  %-3s B rep%s | %-5s | %s %s\n' "$n" "$rep" "${v:-?(rc=$rc)}" "${d:-}" "${r:-}"
  done
done
