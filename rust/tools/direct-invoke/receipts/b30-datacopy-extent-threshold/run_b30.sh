set -u
cd /tmp/di-i27
mkdir -p logs probe-b30
# B30's open items 77 and 80, on the harness and host B30 was measured on.
#
# 77 (chip-agent: "the cheapest open item in this file"): only 4 B and 32 B were ever run,
#    so nothing distinguishes "a copy must cover one whole 32-byte block" from "a copy
#    smaller than 32 B is wrong in some size-dependent way". 8 B and 16 B decide it.
# 80: B14 says reps 0-1 on a cold 910B2 chip are wrong at rc=0, and B30's report gave no
#    rep count. FIVE reps per arm, every one printed, so the FAIL arm is attributable.
#
# NO `timeout` ANYWHERE, and that is a CORRECTION to this script as first committed.
# It wrapped every launch in `timeout 300`, which on this part's standing rules is a
# forbidden SIGTERM on an in-flight NPU kernel on a shared box. None fired -- all 20
# runs printed a verdict, so every figure in all-20-runs.txt stands -- but the recipe
# inherited it and a wider sweep would have too. `pkill` after a launch that has
# already returned reaps a leftover and is not a kill of a running kernel; it also
# restores the settle the five-rep control had and this sweep lacked.
#
# These are PROBE VARIANTS of an emitter artefact, in their own directory. The corpus
# artefact is untouched and is the 8-f32 form.
SRC=corpus/deepseek_v41/ascendc/embedding_v41_129280x5120.cpp
for n in 1 2 4 8; do
  sed "s/AscendC::DataCopy(buf0, gm1Tensor, (1) \* (8));/AscendC::DataCopy(buf0, gm1Tensor, (1) * ($n));/" \
    "$SRC" > "probe-b30/emb_idxload_$n.cpp"
  grep -c "gm1Tensor, (1) \* ($n))" "probe-b30/emb_idxload_$n.cpp" >/dev/null || { echo "SED FAILED for n=$n"; exit 2; }
done
echo "=== extent sweep: index load of n f32, 5 reps each, ORACLE 44 ==="
for n in 1 2 4 8; do
  bytes=$((n * 4))
  for rep in 1 2 3 4 5; do
    env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh \
      "probe-b30/emb_idxload_$n.cpp" dav-2201 v41_embedding_5120_do 5120 3 44 1e-5f 1e-6 \
      > "logs/b30_n${n}_rep${rep}.log" 2>&1
    pkill -9 -x vec_add_run 2>/dev/null; sleep 2
    v=$(grep -vE "unbound variable" "logs/b30_n${n}_rep${rep}.log" | grep -E "elements=|first disagreement|PASS|FAIL" | tr '\n' ' ')
    printf '%2d f32 (%2d B) rep%s | %s\n' "$n" "$bytes" "$rep" "${v:-NO OUTPUT}"
  done
done
