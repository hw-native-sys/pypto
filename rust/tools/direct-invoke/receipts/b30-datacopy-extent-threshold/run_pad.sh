set -u
cd /tmp/di-i27
python3 gen_pad.py || exit 1
# DOES A LOAD-SIDE DataCopyPad DELIVER A SUB-BLOCK GM->UB COPY ON THIS PART?
# The emitter's sub-block refusal says the symmetric repair is a DataCopyPad branch
# and that it is not taken because the load-side pad is unmeasured here. This is that
# measurement. Poison 2.0 at every cell first, then read element 0 -- the index -- so
# the oracle's own verdict answers:
#   PASS / got 3840.125 -> index 3 ARRIVED: the pad repair works, refusal can become a pad
#   got 2560.125        -> poison intact: the pad did not deliver either, refusal is right
# Controls: pad at 32 B must PASS (the pad itself is not the problem), and a PLAIN
# DataCopy at 4 B must FAIL (the defect reproduces in this very build).
echo "=== load-side DataCopyPad, element 0 = the index, 3 reps ==="
for k in pad_1 pad_8 padctl_1; do
  for rep in 1 2 3; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/$k.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/pad_${k}_rep${rep}.log" 2>&1
    rc=$?
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/pad_${k}_rep${rep}.log")
    v=$(grep -m1 -oE "^(PASS|FAIL)" "logs/pad_${k}_rep${rep}.log")
    w=$(grep -m1 -oE "wrong=[0-9]+" "logs/pad_${k}_rep${rep}.log")
    case "$g" in
      "got 3840.125000") d="index 3 ARRIVED" ;;
      "got 2560.125000") d="poison 2.0 intact -> NOT DELIVERED" ;;
      "got 0.125000")    d="0.0 -- the decoy, not the index" ;;
      "")                d="no readout (rc=$rc)" ;;
      *)                 d="unexpected [$g]" ;;
    esac
    printf '  %-10s rep%s | %-5s %-12s | %s\n' "$k" "$rep" "${v:-?}" "${w:-}" "$d"
  done
done
echo "--- faults ---"; grep -hoE "507[0-9]{3}" logs/pad_pad_*.log logs/pad_padctl_*.log 2>/dev/null | sort | uniq -c
