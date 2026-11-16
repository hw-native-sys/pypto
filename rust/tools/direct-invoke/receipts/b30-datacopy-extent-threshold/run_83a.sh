set -u
cd /tmp/di-i27
# NM 83a -- PIN THE BRACKET, with the now-proven legal-index poison instrument.
# B30 left the smallest passing extent bracketed in (16 B, 32 B]: 16 B fails, 32 B
# passes, nothing between was tried. 5/6/7 f32 = 20/24/28 B all FIT the existing
# InitBuffer(tbuf0, 32), so this needs no allocation change and is not 83b (9/10/12
# f32 would be a 36/40/48-byte DMA into a 32-byte buffer -- a UB out-of-bounds
# write with tbuf1 allocated right after it, which is why that half needs the
# buffer raised AND the 8 f32 arm re-run at the new size as a control).
#
# Poison 2.0 at every cell, read element 1, same decode:
#   got 2560.125 -> 2.0 POISON INTACT, element 1 not written
#   got    0.125 -> 0.0 WRITTEN, the decoy
# PREDICTIONS, registered before the run:
#   threshold EXACTLY one 32-byte block -> 5,6,7 all POISON INTACT, matching 1/2/4
#   threshold anywhere in (16,28] B     -> at least one of 5,6,7 reads 0.0
#   a per-element copy                  -> all three read 0.0, since element 1 < 5
# The n=8 arm is re-run as the within-sweep positive control.
for n in 8 7 6 5; do
  for rep in 1 2 3 4; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/pA_$n.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/p83a_n${n}_rep${rep}.log" 2>&1
    rc=$?
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -m1 -oE "got [0-9.]+" "logs/p83a_n${n}_rep${rep}.log")
    case "$g" in
      "got 2560.125000") e1="element1=2.0  POISON INTACT -> NOT WRITTEN" ;;
      "got 0.125000")    e1="element1=0.0  WRITTEN (the decoy)" ;;
      "got 640.125000")  e1="element1=0.5  latent state RETURNED" ;;
      "")                e1="NO READOUT (rc=$rc)" ;;
      *)                 e1="unexpected [$g]" ;;
    esac
    printf '  %2d f32 (%2d B) rep%s | %s\n' "$n" "$((n*4))" "$rep" "$e1"
  done
done
echo "--- faults, if any ---"; grep -hoE "507[0-9]{3}" logs/p83a_n*_rep*.log | sort | uniq -c
