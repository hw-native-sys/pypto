#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Build and run one emitted kernel through the CANN `ASC` language driver.
#
#   ./setup.sh <corpus-kernel.cpp> [arch] [launcher] [n] [args] [oracle] [eps] [tol]
#
# The kernel is emitter output and is copied in verbatim; nothing here edits it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kernel="${1:?usage: setup.sh <corpus-kernel.cpp> [npu-arch] [launcher] [n]}"
arch="${2:-dav-2201}"
launcher="${3:-vec_add_do}"
elements="${4:-256}"
arity="${5:-3}"
oracle="${6:-0}"
eps="${7:-1e-5f}"
# The COMPARISON tolerance, in the same units as the oracle's relative error.
# It was compile-time only (`#define TOLERANCE 1e-6`), so passing a looser value
# on the command line silently did nothing and a run "still failing at 1e-5"
# looked like evidence when it was the same 1e-6 threshold every time.
tol="${8:-1e-6}"
# Launches to time. 1 (the default) keeps a correctness run unchanged and pays
# nothing; above 1 the launcher prints TIME_MS and discards the first launch.
repeat="${9:-1}"
# The matvec oracle's own shape (outputs, contraction width). It cannot be inferred
# from `n`: for a quantised matvec `n` has to cover the RAW BYTE size of the weights.
mv_n="${10:-64}"
mv_k="${11:-256}"
# The THIRD extent, for an oracle needing three (a GEMM's N). 0 means "not passed",
# and an oracle that needs it REFUSES on 0 rather than assuming a square shape. Every
# existing row passes eleven fields or fewer and is unaffected.
mv_p="${12:-0}"
# The embedding table's ROW COUNT, for driving ORACLE 44's index large. 0 means "do not
# override", because EMB_ROWS is `#ifndef`-guarded in main.asc and passing 0 through
# would satisfy the guard with a zero table. Added 2026-11-10 to test whether a
# GlobalTensor's DECLARED extent (16777216 at every emitted site) is advisory: the three
# embedding kernels are verified at a max offset of 15360 and the real tables are up to
# 39.5x the declaration.
emb_rows="${13:-0}"

mkdir -p "${here}/op_kernel"
cp "${kernel}" "${here}/op_kernel/vec_add_kernel.asc"

: "${ASCEND_HOME_PATH:=/opt/cann9-yuyijun/cann-9.0.0-alpha.1}"
export ASCEND_HOME_PATH
# shellcheck disable=SC1090
source "${ASCEND_HOME_PATH}/set_env.sh"

rm -rf "${here}/build"
cmake -S "${here}" -B "${here}/build" \
    -DASC_DIR="${ASCEND_HOME_PATH}/x86_64-linux/lib64/cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DARCH="${arch}" \
    -DKERNEL_DO="${launcher}" -DKERNEL_N="${elements}" \
    -DKERNEL_ARGS="${arity}" -DORACLE="${oracle}" -DRMS_EPS="${eps}" -DTOLERANCE="${tol}f" -DKERNEL_REPEAT="${repeat}" -DMV_N="${mv_n}" -DMV_K="${mv_k}" -DMV_P="${mv_p}" -DEMB_ROWS_OVERRIDE="${emb_rows}" >/dev/null
cmake --build "${here}/build" --target vec_add_run -j 4 2>&1 | grep -E "error|Error" || true

"${here}/build/vec_add_run"
