set -u
cd /tmp/di-i27
# THE ARM ORDER PERMUTED: 8 4 2 1 instead of 1 2 4 8. Same 20 launches, same arms, same
# oracle, no new code. The first sweep ran the extents in ascending order, so EXTENT and
# LAUNCH POSITION were perfectly collinear and its one transition sits at launch 8 -- the
# third rep of the 8 B arm, MID-ARM, and no transition at all at the 10->11 arm boundary.
# An extent-driven effect must change at an arm boundary. This decides it.
#
# PREDICTIONS, registered before the run:
#   EXTENT-driven   -> per-arm values UNCHANGED from the first sweep:
#                      8 B -> 0.0 x5, 4 B -> 0.5 x5, 2 B -> mixed, 1 B -> 0.0 x5
#   POSITION-driven -> the transition stays near launch 8 whichever arm occupies it, so the
#                      8 B arm (launches 1-5) reads 0.0 and the 4 B arm (6-10) carries the flip
# These are not the same table in any cell but the first arm, so one run separates them.
#
# No `timeout` anywhere; `pkill` only after a launch has returned.
echo "=== element-1 read, arms in DESCENDING order, 5 reps each ==="
launch=0
for n in 8 4 2 1; do
  for rep in 1 2 3 4 5; do
    launch=$((launch+1))
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/emb_e1_$n.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/b30perm_n${n}_rep${rep}.log" 2>&1
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    g=$(grep -oE "got [0-9.]+" "logs/b30perm_n${n}_rep${rep}.log" | head -1)
    case "$g" in
      "got 0.125000")   e1="element1=0.0" ;;
      "got 640.125000") e1="element1=0.5" ;;
      *)                e1="element1=? ($g)" ;;
    esac
    printf 'launch %2d | %2d f32 (%2d B) rep%s | %s\n' "$launch" "$n" "$((n*4))" "$rep" "$e1"
  done
done
