set -u
cd /tmp/di-i27
# POISON WITH A LEGAL INDEX. The 1e30 prologue produced no readout below 32 B: the
# poison flowed into the row index, the GM address went wild, and every such arm
# returned 507035. The fix is the VALUE, not the kernel. EMB_ROWS is 4, so a legal
# index is 0..3, and 2.0 is distinct from all three values already in play --
# the decoy 0.0, the latent-state 0.5, and the true index 3.0. (The knowledge
# owner proposed 7.0, which is OUT OF RANGE for a 4-row table: it would read
# device element 35840 past an 81920-byte allocation and index a[35840] in a
# host vector of 20480, bringing the fault ambiguity straight back.)
#
# ORACLE 44's own decode is got = 1280*index + 0.125, so:
#   got 2560.125 -> element 1 = 2.0  POISON INTACT: the copy did not write this cell
#   got    0.125 -> element 1 = 0.0  the copy WROTE gm1[1], the decoy
#   got  640.125 -> element 1 = 0.5  the latent state, neither of the above
#   got 3840.125 -> element 1 = 3.0  would mean element 1 holds the true index
#
# THE CONTROL THE RETRACTED 32 B LEG LACKED is the n=1 arm: 2560.125 there says in
# ONE cell both that the prologue LANDS and that nothing wrote element 1 below a
# block. Un-landed poison is no longer confusable with a completed write, because
# the two now read 2560.125 and 0.125 instead of both reading 0.125.
#
# Variant A only: variant B (poison + PipeBarrier<PIPE_ALL>) was identical to A in
# every cell of the previous sweep, so the barrier is not a live variable.
# No `timeout` anywhere; pkill only after a launch has returned.
sed 's/1\.0e30f/2.0f/' gen_poison.py > gen_poison2.py
grep -n "2.0f" gen_poison2.py | head -2
python3 gen_poison2.py || exit 1
echo "=== legal-index poison (2.0), variant A, descending arms, 4 reps ==="
for n in 8 4 2 1; do
  for rep in 1 2 3 4; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/pA_$n.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/p2_n${n}_rep${rep}.log" 2>&1
    rc=$?
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/p2_n${n}_rep${rep}.log")
    case "$g" in
      "got 2560.125000") e1="element1=2.0  POISON INTACT -> NOT WRITTEN" ;;
      "got 0.125000")    e1="element1=0.0  WRITTEN (the decoy)" ;;
      "got 640.125000")  e1="element1=0.5  latent state" ;;
      "got 3840.125000") e1="element1=3.0  holds the true index" ;;
      "")                e1="NO READOUT (rc=$rc) -- the legal index did not prevent it" ;;
      *)                 e1="unexpected [$g]" ;;
    esac
    printf '  %2d f32 (%2d B) rep%s | %s\n' "$n" "$((n*4))" "$rep" "$e1"
  done
done
echo "--- faults, if any ---"; grep -hoE "507[0-9]{3}" logs/p2_n*_rep*.log | sort | uniq -c
