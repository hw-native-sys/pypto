# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capability gate for PTOAS logical Cube FP4 geometry."""

import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

from pypto._environment import toolchain_probe_env
from pypto.pypto_core import DataType, ir

from ._ptoas_config import get_ptoas_timeout
from ._ptoas_locate import PTOAS_MIN_VERSION, check_ptoas_version, find_ptoas_binary

_MX_OP_FAMILIES = {
    ir.get_op(f"tile.{family}_mx{suffix}").name: family
    for family in ("matmul", "gemv")
    for suffix in ("", "_acc", "_bias")
}
_PACKED_DTYPES = (DataType.FP4E2M1X2, DataType.FP4E1M2X2)
_probe_lock = threading.Lock()
_verified_binaries: set[tuple[str, int, int, bool]] = set()


# PTOAS Cube operands use logical extents, while PyPTO tracks byte carriers.
# A padded GEMV has one logical row and uses scale groups for its valid K.
def _build_probe(gemv: bool) -> str:
    op = "tgemv" if gemv else "tmatmul"
    m, k, valid_k, scale_k = (1, 1024, 64, 2) if gemv else (32, 128, 128, 4)
    acc_mode = "fractal=1024, pad=0, compact=1" if gemv else "fractal=512, pad=0"
    return (
        f"!A = !pto.tile_buf<loc=left, dtype=!pto.f4E2M1x2, rows={m}, cols={k}, "
        f"v_row={m}, v_col={valid_k}, blayout=col_major, slayout=row_major, fractal=512, pad=0>\n"
        f"!B = !pto.tile_buf<loc=right, dtype=!pto.f4E1M2x2, rows={k}, cols=64, "
        f"v_row={valid_k}, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>\n"
        f"!AS = !pto.tile_buf<loc=scaling, dtype=!pto.f8E8M0, rows={m}, cols={scale_k}, "
        f"v_row={m}, v_col={scale_k}, blayout=row_major, slayout=row_major, fractal=32, pad=0>\n"
        f"!BS = !pto.tile_buf<loc=scaling, dtype=!pto.f8E8M0, rows={scale_k}, cols=64, "
        f"v_row={scale_k}, v_col=64, blayout=col_major, slayout=col_major, fractal=32, pad=0>\n"
        f"!C = !pto.tile_buf<loc=acc, dtype=f32, rows={m}, cols=64, "
        f"v_row={m}, v_col=64, blayout=col_major, slayout=row_major, {acc_mode}>\n"
        "!Bias = !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=64, "
        "v_row=1, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>\n"
        'module attributes {pto.target_arch = "a5"} {\n'
        "  func.func @native_fp4_mx_probe(%a: !A, %b: !B, %bias: !Bias) "
        "attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {\n"
        "    %as = pto.alloc_tile : !AS\n"
        "    %bs = pto.alloc_tile : !BS\n"
        "    %c = pto.alloc_tile : !C\n"
        "    pto.tget_scale_addr ins(%a : !A) outs(%as : !AS)\n"
        "    pto.tget_scale_addr ins(%b : !B) outs(%bs : !BS)\n"
        f"    pto.{op}.mx ins(%a, %as, %b, %bs : !A, !AS, !B, !BS) outs(%c : !C)\n"
        f"    pto.{op}.mx.acc ins(%c, %a, %as, %b, %bs : !C, !A, !AS, !B, !BS) outs(%c : !C)\n"
        f"    pto.{op}.mx.bias ins(%a, %as, %b, %bs, %bias : !A, !AS, !B, !BS, !Bias) outs(%c : !C)\n"
        "    return\n  }\n}\n"
    )


def native_fp4_mx_families(program: ir.Program) -> set[str]:
    """Find MX families that consume packed FP4 operands."""
    families: set[str] = set()

    class Finder(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            ir_op = getattr(op, "op", None)
            if isinstance(ir_op, ir.Op):
                family = _MX_OP_FAMILIES.get(ir_op.name)
                if (
                    family is not None
                    and family not in families
                    and any(
                        isinstance(arg.type, ir.TileType) and arg.type.dtype in _PACKED_DTYPES
                        for arg in op.args
                    )
                ):
                    families.add(family)
            super().visit_call(op)

    finder = Finder()
    for func in program.functions.values():
        finder.visit_stmt(func.body)
    return families


def check_native_fp4_mx_support(*, gemv: bool = False) -> None:
    """Require verified logical FP4 geometry, including for development builds.

    Development builds may report the release version with a different ABI. Cache only
    successes, keyed by resolved executable and file identity, so replacing a
    rejected build or repointing PTOAS_ROOT takes effect in the same process.
    """
    binary = find_ptoas_binary()
    op = "GEMV" if gemv else "matmul"
    hint = (
        f"Native FP4 MX {op} requires PTOAS {PTOAS_MIN_VERSION} or newer with the logical Cube "
        "geometry contract (PTOAS #1700). "
        f"Set PTOAS_ROOT to the released {PTOAS_MIN_VERSION} assembler or a compatible newer/patched build."
    )
    if binary is None:
        raise RuntimeError(f"{hint} No PTOAS executable was found.")
    version = check_ptoas_version(binary)
    with _probe_lock:
        info = os.stat(binary)
        key = (os.path.realpath(binary), info.st_mtime_ns, info.st_size, gemv)
        if key in _verified_binaries:
            return
        try:
            with tempfile.TemporaryDirectory(prefix="pypto-fp4-probe-") as directory:
                source = Path(directory) / "probe.pto"
                source.write_text(_build_probe(gemv))
                result = subprocess.run(
                    [binary, "--pto-arch=a5", str(source)],
                    env=toolchain_probe_env(),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=get_ptoas_timeout(),
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"{hint} Capability probe failed for {version}: {exc}") from exc
        # Acceptance alone is insufficient: emitted tiles must also use logical
        # dimensions, rather than merely relaxing the old verifier checks.
        m, k, scale_k = (1, 1024, 2) if gemv else (32, 128, 4)
        expected = (
            rf"Tile<TileType::Left,\s*float4_e2m1x2_t,\s*{m},\s*{k},",
            rf"Tile<TileType::Right,\s*float4_e1m2x2_t,\s*{k},\s*64,",
            rf"Tile<TileType::Acc,\s*float,\s*{m},\s*64,",
            rf"Tile<TileType::ScaleLeft,\s*float8_e8m0_t,\s*{m},\s*{scale_k},",
            rf"Tile<TileType::ScaleRight,\s*float8_e8m0_t,\s*{scale_k},\s*64,",
        )
        if gemv and result.returncode == 0 and result.stdout.count("TGEMV_MX(") != 3:
            raise RuntimeError(f"{hint} Capability probe failed for {version}: missing TGEMV_MX forms")
        if result.returncode != 0 or not all(re.search(pattern, result.stdout) for pattern in expected):
            detail = result.stderr.strip() or "emitted tile geometry did not match the FP4 carrier contract"
            raise RuntimeError(f"{hint} Capability probe failed for {version}: {detail[:1000]}")
        _verified_binaries.add(key)
