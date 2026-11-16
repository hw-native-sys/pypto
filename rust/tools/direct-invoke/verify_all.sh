#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Re-run every kernel this tree has verified on an Ascend part, and report which
# still pass.
#
#   ./verify_all.sh <bench-root>
#
# <bench-root> is a checkout's `rust/bench` (the corpus of `.cpp` artefacts).
# Exits non-zero if any kernel fails, so a claim cannot outlive its evidence.
#
# The table below is the evidence list, not a convenience list: each row records
# the launcher, extent, arity, oracle and tolerance that the run used, so a reader
# can see what "verified" meant for each kernel rather than trusting a percentage.
#
# It listed SEVEN kernels while the campaign had twenty-four -- it would have
# printed "all kernels passed" over a third of the evidence. Regenerate this table
# from the records rather than by hand, and check the count against the report.
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench="${1:?usage: verify_all.sh <bench-root>}"

# kernel                                    arch        launcher                    n     arity oracle eps
# kernel                                    arch        launcher                    n     arity oracle eps     tol
read -r -d '' ROWS <<'ROWS'
deepseek_v41/ascendc/residual_add_v41_1x5120         dav-2201 v41_residual_add_do          5120 3   0 1e-5f    1e-6
glm_5_3/ascendc/residual_add_glm53_1x4096            dav-2201 glm53_residual_add_do        4096 3   0 1e-5f    1e-6
kimi_k3/ascendc/residual_add_k3_1x7168               dav-2201 k3_residual_add_do           7168 3   0 1e-5f    1e-6
deepseek_v41/ascendc/rmsnorm_v41_1x5120_decode       dav-2201 bm_rmsnorm_1_5120_do         5120 2   1 1e-20f   1e-6
deepseek_v41/ascendc/rmsnorm_v41_8x1280_quarter      dav-2201 bm_rmsnorm_8_1280_do        10240 2  40 1e-20f   1e-6 1 8 1280
deepseek_v41/ascendc/rmsnorm_v41_8x5120               dav-2201 bm_rmsnorm_8_5120_do        40960 2  40 1e-20f   1e-6 1 8 5120
deepseek_v41/ascendc/rope_v41_8x64                   dav-2201 bm_rope_8_64_do                512 2  45 1e-5f    1e-6 1 8 64
deepseek_v41/ascendc/attention_decode_v41_1x512      dav-2201 v41_attn_decode_do         4096 4  53 1e-5f    1e-5 1 8 512
deepseek_v41/ascendc/argmax_v41_8x129280             dav-2201 bm_argmax_8_129280_do      1034240 2  58 1e-5f    1e-5 1 8 129280
deepseek_v41/ascendc/argmax_v41_8x131072_pad         dav-2201 bm_argmax_8_131072_do      1048576 2  58 1e-5f    1e-5 1 8 131072
deepseek_v41/ascendc/mx4pk_v41_5120x2304              dav-2201 bm_mx4pk_5120_2304_do      1474560 4  59 1e-5f    1e-4 1 16 5120
glm_5_3/ascendc/attention_decode_glm53_1x256      dav-2201 glm53_attn_decode_256_do     524288 4  56 1e-5f    1e-5 1 2048 256
kimi_k3/ascendc/attention_decode_k3_1x128         dav-2201 k3_attn_decode_128_do      8192 5  57 1e-5f    1e-5 1 64 128
glm_5_3/ascendc/rmsnorm_glm53_1x4096                 dav-2201 glm53_rmsnorm_1_4096_do      4096 2   1 1e-5f    1e-6
kimi_k3/ascendc/rmsnorm_k3_1x7168                    dav-2201 k3_rmsnorm_1_7168_do         7168 2   1 1e-5f    1e-6
deepseek_v41/ascendc/swiglu_v41_1x2304               dav-2201 bm_swiglu_1_2304_do          2304 3   3 1e-5f    1e-6
glm_5_3/ascendc/shexp_gate_up_silu_glm53_1x2048      dav-2201 glm53_shexp_swiglu_do        2048 4   8 1e-5f    1e-6
glm_5_3/ascendc/dense_gate_up_silu_glm53_1x12288     dav-2201 glm53_dense_swiglu_do        12288 4   8 1e-5f    1e-6 1 64 256
glm_5_3/ascendc/routed_gate_up_glm53_8x2048       dav-2201 glm53_routed_gate_up_2048_do   16384 4  8 1e-5f 1e-5
glm_5_3/ascendc/routed_down_glm53_1x2048          dav-2201 glm53_routed_down_2048_do        2048 3  2 1e-5f 1e-5
# Kimi K3's three FFN down-projections, device-verified 2026-11-08. Each contracts 7168 wide
# into ONE output and stores a 1x1 tile, which is ORACLE 2's shape: it checks element 0 against
# an f64 dot and counts anything written past it separately. The three are shape-identical and
# share these inputs, so they print the same dot -- they are here as three distinct ARTEFACTS.
kimi_k3/ascendc/routed_ffn_down_k3_1x7168         dav-2201 k3_routed_down_do                7168 3  2 1e-5f 1e-6
kimi_k3/ascendc/shexp_ffn_down_k3_1x7168          dav-2201 k3_shexp_down_do                 7168 3  2 1e-5f 1e-6
kimi_k3/ascendc/dense_ffn_down_k3_1x7168          dav-2201 k3_dense_down_do                 7168 3  2 1e-5f 1e-6
# Kimi K3's KDA gate projections, device-verified 2026-11-08. These artefacts did not exist
# until that day: the emitter accepted the kernels and no .cpp was recorded. Their GLM 5.3
# twins are the four rows above with the same oracle. NOTE: this w8a8 path intermittently
# HANGS or returns 507034 on the shared 910b -- measured 4 PASS / 1 hang over five identical
# runs -- and never returns a wrong answer, so re-run a row that does not finish.
kimi_k3/ascendc/kda_f_a_k3_1x128_k7168            dav-2201 k3_kda_f_a_do               9000000 5 15 1e-5f 1e-6 1 128 7168
kimi_k3/ascendc/kda_g_a_k3_1x128_k7168            dav-2201 k3_kda_g_a_do               9000000 5 15 1e-5f 1e-6 1 128 7168
kimi_k3/ascendc/kda_f_b_k3_1x8192_k128            dav-2201 k3_kda_f_b_do               9000000 5 15 1e-5f 1e-6 1 8192 128
kimi_k3/ascendc/kda_g_b_k3_1x8192_k128            dav-2201 k3_kda_g_b_do               9000000 5 15 1e-5f 1e-6 1 8192 128
engine/ascendc/rmsnorm_engine_1x512                  dav-2201 engine_rmsnorm_1_512_do          512 2  1 1e-5f 1e-6
engine/ascendc/rmsnorm_engine_1x1024                 dav-2201 engine_rmsnorm_1_1024_do        1024 2  1 1e-5f 1e-6
engine/ascendc/rope_halves_engine_1x64                dav-2201 rope_halves_engine_1x64_do         64 4 12 1e-5f 1e-6
kimi_k3/ascendc/rope_k3_1x64                         dav-2201 k3_rope_64_do                      64 4 12 1e-5f 1e-6 1 64 256
engine/ascendc/rmsnorm_engine_1x16384                 dav-2201 engine_rmsnorm_1_16384_do        16384 2  1 1e-5f 1e-6
engine/ascendc/rmsnorm_engine_rmsmul_1x512            dav-2201 engine_rmsmul_1_512_do           512 3 16 1e-5f 1e-6 3
engine/ascendc/rmsnorm_engine_rmsmul_1x1024           dav-2201 engine_rmsmul_1_1024_do          1024 3 16 1e-5f 1e-6 3
engine/ascendc/rmsnorm_engine_rmsmul_1x4096           dav-2201 engine_rmsmul_1_4096_do          4096 3 16 1e-5f 1e-6 3
engine/ascendc/proj_w8a8_engine_1x1024_k4096          dav-2201 proj_w8a8_engine_1x1024_k4096_do  1200000 5 15 1e-5f 1e-6 3 1024 4096
engine/ascendc/proj_w8a8_engine_1x512_k4096           dav-2201 proj_w8a8_engine_1x512_k4096_do   600000 5 15 1e-5f 1e-6 3 512 4096
engine/ascendc/proj_w8a8_engine_1x32768_k1024         dav-2201 proj_w8a8_engine_1x32768_k1024_do 8400000 5 15 1e-5f 1e-6 3 32768 1024
engine/ascendc/proj_w8a8_engine_1x4096_k8192           dav-2201 proj_w8a8_engine_1x4096_k8192_do   8400000 5 15 1e-5f 1e-6 3 4096 8192
engine/ascendc/proj_w8a8_engine_1x24_k16384           dav-2201 proj_w8a8_engine_1x24_k16384_do   200000 5 15 1e-5f 1e-6 3 24 16384
# The MoE down-projection shape (mx4pk_2048_4096's extents, served by w8a8).
engine/ascendc/w8a8_experts_1x4096_k2048             dav-2201 w8a8_experts_1x4096_k2048_do     9000000 5 15 1e-5f 1e-6 3 4096 2048
engine/ascendc/w8a8_experts_1x2048_k4096             dav-2201 w8a8_experts_1x2048_k4096_do     9000000 5 15 1e-5f 1e-6 3 2048 4096
# The engine's attention extent (64 heads x 512) through ONE scf.for; oracle 34 checks ALL heads.
engine/ascendc/attn_sink_engine_64heads                 dav-2201 attn_sink_1blk_do                70000 4 34 1e-5f 1e-5 1 64 512
# Attention primitives (certified 2026-11-06, bit-exact). The engine's flash kernel needs both.
# NOTE: these are PRIMITIVE probes, not engine slots; they belong here so a regression is caught.
# q8mv rows need EXACT inputs: DS4_IN_DIR/DS4_REF_FILE (mkq8mv.py), because the harness's
# synthetic ramps give a scale plane of ~1e7 and weight bytes that are a by-product of float writes.
engine/ascendc/proj_q8mv_engine_1x1024_k4096          dav-2201 proj_q8mv_engine_1x1024_k4096_do  4300000 4 17 1e-5f 1e-6 3 1024 4096
engine/ascendc/proj_q8mv_engine_1x512_k4096           dav-2201 proj_q8mv_engine_1x512_k4096_do   4300000 4 17 1e-5f 1e-6 3 512 4096
engine/ascendc/proj_q8mv_engine_1x4096_k8192          dav-2201 proj_q8mv_engine_1x4096_k8192_do  34000000 4 17 1e-5f 1e-6 3 4096 8192
engine/ascendc/proj_q8mv_engine_1x32768_k1024         dav-2201 proj_q8mv_engine_1x32768_k1024_do 34000000 4 17 1e-5f 1e-6 3 32768 1024
kimi_k3/ascendc/kda_delta_k3_1x128                   dav-2201 k3_kda_delta_128_do           128 5   4 1e-5f    1e-6
glm_5_3/ascendc/kda_delta_glm53_1x128                dav-2201 glm53_kda_delta_128_do        128 5   4 1e-5f    1e-6
kimi_k3/ascendc/kda_conv_k3_1x7168_k4                dav-2201 k3_kda_conv_7168_do          7168 9  11 1e-5f    1e-5
glm_5_3/ascendc/kda_conv_glm53_1x8192_k4             dav-2201 glm53_kda_conv_8192_do          8192 9  11 1e-5f    1e-5
kimi_k3/ascendc/shexp_gate_up_situ_k3_1x3072         dav-2201 k3_shexp_situ_do             3072 3  10 1e-5f    1e-5
kimi_k3/ascendc/routed_gate_up_situ_k3_1x3072        dav-2201 k3_routed_situ_do            3072 3  10 1e-5f    1e-5
kimi_k3/ascendc/dense_gate_up_situ_k3_1x33792        dav-2201 k3_dense_situ_do            33792 3  10 1e-5f    1e-5
glm_5_3/ascendc/proj_q_a_glm53_1x1536                dav-2201 glm53_proj_q_a_do            1536 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_output_glm53_1x4096             dav-2201 glm53_proj_output_do         4096 3   2 1e-5f    1e-5
glm_5_3/ascendc/kda_f_b_glm53_1x8192_k128                dav-2201 glm53_kda_f_b_do                 9000000 5 15 1e-5f 1e-6 1 8192 128
glm_5_3/ascendc/kda_q_glm53_1x8192_k4096                dav-2201 glm53_kda_q_do               12000000 5 15 1e-5f 1e-6 1 8192 4096
glm_5_3/ascendc/proj_kv_a_mqa_glm53_1x512            dav-2201 glm53_proj_kv_a_mqa_do        512 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_shexp_down_glm53_1x4096         dav-2201 glm53_proj_shexp_down_do     4096 3   2 1e-5f    1e-5
glm_5_3/ascendc/proj_dense_down_glm53_1x4096         dav-2201 glm53_proj_dense_down_do     4096 3   2 1e-5f    1e-5
kimi_k3/ascendc/proj_q_a_k3_1x1536                   dav-2201 k3_proj_q_a_do               1536 3   2 1e-5f    1e-5
kimi_k3/ascendc/proj_kv_a_mqa_k3_1x512               dav-2201 k3_proj_kv_a_mqa_do           512 3   2 1e-5f    1e-5
kimi_k3/ascendc/proj_output_k3_1x7168                dav-2201 k3_proj_output_do            7168 3   2 1e-5f    1e-5
glm_5_3/ascendc/kda_k_glm53_1x8192_k4096                dav-2201 glm53_kda_k_do                   12000000 5 15 1e-5f 1e-6 1 8192 4096
glm_5_3/ascendc/kda_v_glm53_1x8192_k4096                dav-2201 glm53_kda_v_do                   12000000 5 15 1e-5f 1e-6 1 8192 4096
glm_5_3/ascendc/kda_g_a_glm53_1x128_k4096                dav-2201 glm53_kda_g_a_do                 9000000 5 15 1e-5f 1e-6 1 128 4096
glm_5_3/ascendc/kda_g_b_glm53_1x8192_k128                dav-2201 glm53_kda_g_b_do                 9000000 5 15 1e-5f 1e-6 1 8192 128
glm_5_3/ascendc/kda_f_a_glm53_1x128_k4096                dav-2201 glm53_kda_f_a_do                 9000000 5 15 1e-5f 1e-6 1 128 4096
# The three k=7168 KDA projections, device-verified 2026-11-08. Their artefacts are
# byte-identical modulo the kernel symbol, so these three rows re-run ONE arithmetic
# under three names -- a FAIL on one is a FAIL on all three, and a PASS on one is not
# independent evidence for the others.
kimi_k3/ascendc/kda_q_k3_1x8192_k7168                    dav-2201 k3_kda_q_do                     21000000 5 15 1e-5f 1e-6 1 8192 7168
kimi_k3/ascendc/kda_k_k3_1x8192_k7168                    dav-2201 k3_kda_k_do                     21000000 5 15 1e-5f 1e-6 1 8192 7168
kimi_k3/ascendc/kda_v_k3_1x8192_k7168                    dav-2201 k3_kda_v_do                     21000000 5 15 1e-5f 1e-6 1 8192 7168
# The BLOCKED-REDUCTION arm, landed 2026-11-08. Six one-row dots that had each
# refused on the per-buffer UB bound and now emit as a per-chunk chain into a tile
# accumulator, row-summed once after the loop. ORACLE 2 checks element 0 against an
# f64 dot and counts anything written past it, which is the property that says the
# accumulator never reached GM: all six report `undefined=0`.
glm_5_3/ascendc/proj_q_b_glm53_1x16384                   dav-2201 glm53_proj_q_b_do                 16384 3  2 1e-5f 1e-6
glm_5_3/ascendc/proj_kv_b_glm53_1x28672                  dav-2201 glm53_proj_kv_b_do                28672 3  2 1e-5f 1e-6
glm_5_3/ascendc/lm_head_glm53_1x154880                   dav-2201 glm53_lm_head_154880_do          154880 3  2 1e-5f 1e-6
kimi_k3/ascendc/proj_q_b_k3_1x18432                      dav-2201 k3_proj_q_b_do                    18432 3  2 1e-5f 1e-6
kimi_k3/ascendc/proj_kv_b_k3_1x24576                     dav-2201 k3_proj_kv_b_do                   24576 3  2 1e-5f 1e-6
kimi_k3/ascendc/lm_head_k3_1x163840                      dav-2201 k3_lm_head_163840_do             163840 3  2 1e-5f 1e-6
glm_5_3/ascendc/argmax_glm53_1x154880                    dav-2201 glm53_argmax_154880_do          154880 2 41 1e-5f 1e-6
kimi_k3/ascendc/argmax_k3_1x163840                       dav-2201 k3_argmax_163840_do             163840 2 41 1e-5f 1e-6
deepseek_v41/ascendc/topk_v41_1x384_k6                   dav-2201 bm_topk_384_6_do                   384 3 42 1e-5f 1e-6 1 6 256
glm_5_3/ascendc/router_topk_glm53_1x288_k8               dav-2201 glm53_router_288_8_do               288 3 42 1e-5f 1e-6 1 8 256
kimi_k3/ascendc/router_sigmoid_topk_k3_1x896_k16         dav-2201 k3_router_896_16_do                 896 3 43 1e-5f 1e-6 1 16 256
# The four /prim/ probe rows were REMOVED 2026-11-10: their .cpp files no longer
# exist (bench/engine/prim is gone, reverted with the f16 copy playground), so every
# full run counted four phantom FAILs against a corpus that no longer has them.
ROWS

printf 'host: %s\n' "$(hostname)"
if command -v npu-smi >/dev/null 2>&1; then
    # A row of the product table looks like `| 0     910B2   | OK ...`.
    printf 'part: %s\n' "$(npu-smi info 2>/dev/null | awk -F'|' '{n=split($2,f," "); for(i=1;i<=n;i++) if (f[i] ~ /^9[0-9]+[A-Z0-9]/) {print f[i]; exit}}')"
fi
[ -d "${bench}" ] || { echo "no bench at ${bench}" >&2; exit 2; }
printf 'cann: %s\n' "${ASCEND_HOME_PATH:-unset}"
printf 'ptoas: %s\n\n' "$(ptoas --version 2>/dev/null | head -1 || echo absent)"

printf '%-40s %-8s %s\n' kernel elements result
failed=0
# ELEVEN fields, because the table carries eleven. `read` assigns the leftover
# fields to its LAST variable, so with eight names the three trailing columns
# collapsed into `tol`: a row ending `1e-6 1 8192 7168` handed setup.sh the
# tolerance `"1e-6 1 8192 7168"` -- `-DTOLERANCE="1e-6 1 8192 7168f"` -- and no
# shape at all. Measured 2026-11-08 over this table: 28 of 63 rows overflowed,
# and a further 2 parsed cleanly while declaring a shaped oracle (15) with no
# shape, so they silently took setup.sh's MV_N=64 MV_K=256 defaults and would
# have verified a 64x256 corner of a kernel whose name says 8192 outputs. Thirty
# of sixty-three rows -- the script could not re-run half the board it lists.
# Empty trailing fields are fine: setup.sh's `${9:-1}` / `${10:-64}` / `${11:-256}`
# default on empty as well as unset.
# ORACLE 40 REPLACED A STOP-GAP THAT LIVED HERE FOR ONE ITERATION. An optional
# twelfth field, `rowperiodic:R`, used to generate an input of R identical rows so
# that ORACLE 1 -- which computes ONE rms over all n -- became a valid oracle for an
# eight-row kernel. It worked, and it could not refute the bug that mattered: on
# row-periodic input a correct per-row normalisation and one that broadcasts row 0's
# denominator agree exactly. ORACLE 40 holds each row to its own denominator and
# prints the span of them, so the field had exactly one user and no longer has any.
# Removed rather than left as a mechanism nothing exercises; the reasoning survives
# in ORACLE 40's own comment, where it is read.
while read -r kernel arch launcher n arity oracle eps tol repeat mv_n mv_k; do
    [ -z "${kernel}" ] && continue
    # The table carries COMMENT LINES; without this case they became rows with a
    # "#" launcher and counted as FAIL, so the summary always claimed failures.
    case "${kernel}" in \#*) continue ;; esac
    # PER-ROW TIMEOUT. The KDA rows are documented to hang intermittently on the
    # shared 910b (4 PASS / 1 hang over five identical runs); without a timeout a
    # single hang stalls the whole board -- measured 2026-11-10: 23 PASS, then the
    # run sat on a KDA row for over ten minutes. A hang is not a pass, so it is a
    # FAIL after five minutes and the board continues.
    out="$(timeout 300 bash "${here}/setup.sh" "${bench}/${kernel}.cpp" \
        "${arch}" "${launcher}" "${n}" "${arity}" "${oracle}" "${eps}" "${tol}" \
        "${repeat}" "${mv_n}" "${mv_k}" 2>/dev/null | tail -2 | tr '\n' ' ')"
    [ -z "${out}" ] && out="TIMEOUT or no output"
    case "${out}" in
        *PASS*) verdict="PASS  ${out}" ;;
        *)      verdict="FAIL  ${out}"; failed=$((failed + 1)) ;;
    esac
    printf '%-40s %-8s %s\n' "$(basename "${kernel}")" "${n}" "${verdict}"
done <<< "${ROWS}"

printf '\n%s\n' "$([ "${failed}" -eq 0 ] && echo "all kernels passed" || echo "${failed} kernel(s) failed")"
exit $([ "${failed}" -eq 0 ] && echo 0 || echo 1)
