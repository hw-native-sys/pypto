set -u
cd /tmp/di-i27
python3 gen_st.py || exit 1
# DOES A SUB-BLOCK DataCopyPad STORE WRITE PAST ITS EXTENT?
# This is the shipped path: mlir_to_cpp carries this exact store form at three
# sites, and every oracle in this harness inspects only the bytes the kernel meant
# to write -- so a write OUTSIDE that region has never been looked at.
# 24 bytes is the k=6 shape the emitter's own comment names; 32 is the whole-block
# control; 4 is the extreme.
#
# PREDICTIONS, registered before the run:
#   writes exactly MV_N       -> delivered=MV_N, guard_breach=0, PASS
#   writes a whole 32-B block -> breach starts at MV_N, ends at 32
#   replicates element 0      -> breach_equals_src0 == guard_breach
# ESCAPE ARM: any other (delivered, breach, first_breach) shape means the space is
# mis-drawn -- report first_breach and its value, do not interpret further.
echo "=== sub-block DataCopyPad STORE, guard region 128 B, 3 reps ==="
for n in 32 24 4; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/st_$n.cpp" dav-2201 stprobe_do 5120 2 47 1e-5f 1e-6 1 $n 256 \
      > "logs/st_n${n}_rep${rep}.log" 2>&1
    rc=$?
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/st_n${n}_rep${rep}.log")
    d=$(grep -m1 -oE "store_bytes=.*guard_end=[0-9]+" "logs/st_n${n}_rep${rep}.log")
    r=$(grep -m1 -oE "ORACLE 47 refuses.*|507[0-9]{3}|error: .*" "logs/st_n${n}_rep${rep}.log" | cut -c1-60)
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    printf '  %-2s B rep%s | %-5s | %s %s\n' "$n" "$rep" "${v:-?(rc=$rc)}" "${d:-no readout}" "${r:-}"
  done
done
