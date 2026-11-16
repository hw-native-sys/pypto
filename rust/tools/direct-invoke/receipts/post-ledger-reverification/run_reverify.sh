set -u
cd /tmp/di-i27
# RE-RUNNING THE RECORDED COMMANDS VERBATIM, after the ledger was destroyed and
# restored from git. The hash check already showed all 61 verified records point at
# the bytes they claim; this asks the stronger question -- do those bytes still
# PASS? The campaign's own re-earned conditions list "every banked row still
# re-verifiable from its own artefact" as something progress.py cannot compute.
#
# It is also an end-to-end inertness check on ORACLE 48/49: these rows use oracles
# 42, 4 and 12, so if they still pass against the current main.asc, the new
# compile-time-gated blocks did not perturb them.
run() {
  name=$1; shift
  env ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2 bash setup.sh "$@" > "logs/rv_$name.log" 2>&1
  rc=$?
  pkill -9 -x vec_add_run 2>/dev/null; sleep 2
  printf '  %-26s rc=%s | %-5s | %s\n' "$name" "$rc" \
    "$(grep -m1 -oE '^(PASS|FAIL)' logs/rv_$name.log)" \
    "$(grep -m1 -oE 'elements=[0-9]+ wrong=[0-9]+ max_abs=[0-9.e+-]+ max_rel=[0-9.e+-]+' logs/rv_$name.log)"
  grep -m1 -oE "507[0-9]{3}|error: .{0,44}" "logs/rv_$name.log" | sed 's/^/       PROBLEM: /'
}
echo "=== re-verifying three banked rows, one per model, recorded args verbatim ==="
rm -rf build; run topk_v41_1x384_k6     reverify_topk_v41_1x384_k6.cpp     dav-2201 bm_topk_384_6_do       384 3 42 1e-5f 1e-6 1 6 256
rm -rf build; run kda_delta_glm53_1x128 reverify_kda_delta_glm53_1x128.cpp dav-2201 glm53_kda_delta_128_do 128 5 4  1e-5f 1e-6
rm -rf build; run rope_k3_1x64          reverify_rope_k3_1x64.cpp          dav-2201 k3_rope_64_do           64 4 12 1e-5f 1e-6 1 64 256
