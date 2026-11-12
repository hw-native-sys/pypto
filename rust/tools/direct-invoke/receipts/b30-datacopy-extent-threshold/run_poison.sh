set -u
cd /tmp/di-i27
python3 gen_poison.py || exit 1
# POISON PROLOGUE. buf0 (32 B = 8 f32) is filled with 1e30 kernel-side before the DataCopy,
# so every slot the copy does NOT write still holds 1e30 when GetValue reads it. This removes
# the 0.0 collision chip-agent found in ORACLE 44: its decoy is 0.0, which is also what a
# cleared UB hold reads as, so "element1 = 0.0" could not distinguish "the copy wrote the
# decoy" from "nothing wrote it". Under poison those are different values. No new oracle id.
#
# got = 0.125 + 1280*element1, so:   0.125 -> element1 = 0.0 (WRITTEN with the decoy)
#                            ~1.28e33 / inf -> element1 = 1e30 (NOT WRITTEN)
#                                   640.125 -> element1 = 0.5 (actively written WRONG)
#
# Variant A = poison only.  Variant B = poison + an MTE2->S PipeBarrier before the GetValue,
# which the emitted kernel does NOT have: the scalar read of buf0 follows the DMA with no
# wait, so a stale read is a live alternative to a short transfer.
#
# CONTROL: the 8 f32 (32 B) arm must still read 0.0 / PASS in BOTH variants. If the prologue
# itself broke the copy, the control fails and nothing else in the table may be read.
#
# PREDICTIONS, registered before the run:
#   no copy at all below 32 B  -> A: n=8 -> 0.0 ; n=4,2,1 -> 1e30
#   correct partial transfer   -> A: n=8,4,2 -> 0.0 ; n=1 -> 1e30
#   missing barrier (stale)    -> A flickers as before, B is CLEAN: n=8,4,2 -> 0.0, n=1 -> 1e30
#   If B differs from A anywhere, the defect is the ABSENT BARRIER, not the extent, and it is
#   a generator bug in mlir_to_cpp rather than a property of DataCopy.
for V in A B; do
  echo "=== variant $V (poison$([ $V = B ] && echo ' + MTE2->S barrier')), descending arms, 3 reps ==="
  for n in 8 4 2 1; do
    for rep in 1 2 3; do
      env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
        "probe-b30/p${V}_$n.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
        > "logs/poison_${V}_n${n}_rep${rep}.log" 2>&1
      rc=$?
      pkill -9 -x vec_add_run 2>/dev/null; sleep 2
      g=$(grep -oE "got [0-9.e+infa]+" "logs/poison_${V}_n${n}_rep${rep}.log" | head -1)
      vd=$(grep -oE "PASS|FAIL|wrong=[0-9]+" "logs/poison_${V}_n${n}_rep${rep}.log" | head -1)
      case "$g" in
        "got 0.125000")   e1="element1=0.0  WRITTEN(decoy)" ;;
        "got 640.125000") e1="element1=0.5  WRITTEN-WRONG" ;;
        "")               e1="NO READING (rc=$rc)" ;;
        *)                e1="element1=POISON-INTACT -> NOT WRITTEN  [$g]" ;;
      esac
      printf '  %2d f32 (%2d B) rep%s | %-14s | %s\n' "$n" "$((n*4))" "$rep" "$vd" "$e1"
    done
  done
done
