# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared preprocessing for C++ emitted by PTOAS."""

import re

# Production: A-scales are stored ND-flat on GM; ptoas EmitC only emits MX_A_ZZ,
# so rewrite every e8m0/uint8 MX_A_ZZ GlobalTensor to MX_A_ND (AND2ZZ).
# ZZ-on-GM (rewrite OFF) is not rolled out — disabling this raises RuntimeError.
_ENABLE_MX_A_ZZ_TO_ND_REWRITE = True


def _require_mx_a_zz_to_nd_rewrite() -> None:
    """Fail loudly if the incomplete ZZ data-plane path is selected."""
    if _ENABLE_MX_A_ZZ_TO_ND_REWRITE:
        return
    raise RuntimeError(
        "MX_A_ZZ→MX_A_ND EmitC rewrite is disabled (_ENABLE_MX_A_ZZ_TO_ND_REWRITE=False), "
        "but all v4-pro A-scale GM stores still write ND bytes and rely on this rewrite "
        "for AND2ZZ. Re-enable the rewrite."
    )


def _rewrite_mx_a_zz_e8m0_to_nd(content: str) -> str:
    """Activation MX scales are stored ND-flat on GM; ptoas only accepts mx_a_zz.

    Rewrite float8_e8m0 / uint8 MX_A_ZZ GlobalTensor/TileShape types to MX_A_ND so
    TLoad uses AND2ZZ (ND→ZZ) instead of AZZ2ZZ. Apply per-line so the trailing
    GlobalTensor layout enum is rewritten together with TileShape2D/BaseShape2D.
    """
    out: list[str] = []
    for line in content.splitlines(keepends=True):
        if "MX_A_ZZ" in line and ("float8_e8m0_t" in line or "uint8_t" in line):
            line = line.replace("pto::Layout::MX_A_ZZ", "pto::Layout::MX_A_ND")
        out.append(line)
    return "".join(out)


# Post-EmitC belt-and-suspenders. Preferred path (kept ON in codegen):
#   1) ``pto.barrier <PIPE_ALL>`` before mx_a_* ``pto.tload`` (MakeTileLoadCodegenPTO)
#   2) ``pto.barrier <PIPE_ALL>`` after e8m0 ``pto.tpop_from_*`` (MakeTpopCodegenPTO)
# Board A/B on prefill_indexer: codegen+regex better than codegen-only; keep ON.
# Producer-side TPUSH pipe_barrier+dsb is intentionally omitted (not a hard board
# dependency once AIC TLOAD barriers are present).
_ENABLE_MX_A_SCALE_TLOAD_BARRIER_REGEX = True


def _barrier_before_mx_a_scale_tload(content: str) -> str:
    """Ensure AIV GM TSTORE is visible before AIC MX A-scale (ZZ/ND) TLOAD.

    Two cases (legacy regex; prefer codegen barrier):
    1. ExpandMixed gm_sync: AIC TPOP of flat e8m0 [1,G] then TLOAD — barrier after TPOP.
    2. No gm_sync (e.g. store via reshape view of workspace): AIC TLOAD of MX_A_ZZ
       or MX_A_ND right after V2C data TPOP — insert barrier immediately before that TLOAD.
    """
    if not _ENABLE_MX_A_SCALE_TLOAD_BARRIER_REGEX:
        return content
    content = re.sub(
        r"(TPOP<[^;\n]*float8_e8m0_t,\s*1,\s*\d+[^;\n]*>\([^;\n]*\);)",
        r"\1\n  pipe_barrier(PIPE_ALL);",
        content,
    )
    # TLOAD(dst, gt) where gt was built with MX_A_ZZ or MX_A_ND on any earlier line.
    lines = content.splitlines(keepends=True)
    out: list[str] = []
    mx_a_vars: dict[str, int] = {}
    for i, line in enumerate(lines):
        if "GlobalTensor<" in line and ("MX_A_ZZ" in line or "MX_A_ND" in line) and "=" in line:
            m = re.search(r"GlobalTensor<.*>\s+(\w+)\s*=", line)
            if m:
                mx_a_vars[m.group(1)] = i
        tm = re.match(r"^([ \t]*)TLOAD\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*;", line)
        if tm:
            indent, _dst, src = tm.group(1), tm.group(2), tm.group(3)
            if src in mx_a_vars:
                out.append(f"{indent}pipe_barrier(PIPE_ALL);\n")
        out.append(line)
    return "".join(out)


def preprocess_ptoas_output(content: str) -> str:
    """Prepare PTOAS output for embedding in PyPTO kernel wrappers."""
    _require_mx_a_zz_to_nd_rewrite()

    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#include") and (
            "pto-inst" in stripped or "cstdint" in stripped or "tensor.h" in stripped
        ):
            continue
        if stripped == "using namespace pto;":
            continue
        if stripped.startswith("set_ffts_base_addr("):
            continue
        filtered.append(line)

    result = "".join(filtered)
    result = re.sub(
        r'(?:extern\s*"C"\s*)?(?:__global__\s+)?AICORE\s+void',
        "static __aicore__ void",
        result,
    )
    result = re.sub(r"\bAICORE\b", "__aicore__", result)
    result = _rewrite_mx_a_zz_e8m0_to_nd(result)
    return _barrier_before_mx_a_scale_tload(result)
