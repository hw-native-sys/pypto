# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Capability gate for the PTOAS v0.72 logical Cube FP4 geometry contract."""

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

_MX_OPS = frozenset(
    ir.get_op(name).name for name in ("tile.matmul_mx", "tile.matmul_mx_acc", "tile.matmul_mx_bias")
)
_PACKED_DTYPES = (DataType.FP4E2M1X2, DataType.FP4E1M2X2)
_probe_lock = threading.Lock()
_verified_binaries: set[tuple[str, int, int]] = set()

# Cube tile operands describe logical [32,128] @ [128,64], while PyPTO
# keeps carrier dimensions internally. Older assemblers doubled those logical
# dimensions in EmitC. Probe both formats, scale binding, and all MX instructions.
_PROBE = (
    "!A = !pto.tile_buf<loc=left, dtype=!pto.f4E2M1x2, rows=32, cols=128, "
    "v_row=32, v_col=128, blayout=col_major, slayout=row_major, fractal=512, pad=0>\n"
    "!B = !pto.tile_buf<loc=right, dtype=!pto.f4E1M2x2, rows=128, cols=64, "
    "v_row=128, v_col=64, blayout=row_major, slayout=col_major, fractal=512, pad=0>\n"
    "!AS = !pto.tile_buf<loc=scaling, dtype=!pto.f8E8M0, rows=32, cols=4, "
    "v_row=32, v_col=4, blayout=row_major, slayout=row_major, fractal=32, pad=0>\n"
    "!BS = !pto.tile_buf<loc=scaling, dtype=!pto.f8E8M0, rows=4, cols=64, "
    "v_row=4, v_col=64, blayout=col_major, slayout=col_major, fractal=32, pad=0>\n"
    "!C = !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=64, "
    "v_row=32, v_col=64, blayout=col_major, slayout=row_major, fractal=512, pad=0>\n"
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
    "    pto.tmatmul.mx ins(%a, %as, %b, %bs : !A, !AS, !B, !BS) outs(%c : !C)\n"
    "    pto.tmatmul.mx.acc ins(%c, %a, %as, %b, %bs : !C, !A, !AS, !B, !BS) outs(%c : !C)\n"
    "    pto.tmatmul.mx.bias ins(%a, %as, %b, %bs, %bias : !A, !AS, !B, !BS, !Bias) outs(%c : !C)\n"
    "    return\n  }\n}\n"
)


def uses_native_fp4_mx(program: ir.Program) -> bool:
    """Inspect operand types, so unrelated FP4 casts never gate FP8 matmul."""

    class Finder(ir.IRVisitor):
        found = False

        def visit_call(self, op: ir.Call) -> None:
            ir_op = getattr(op, "op", None)
            if isinstance(ir_op, ir.Op) and ir_op.name in _MX_OPS:
                self.found |= any(
                    isinstance(arg.type, ir.TileType) and arg.type.dtype in _PACKED_DTYPES for arg in op.args
                )
            super().visit_call(op)

    finder = Finder()
    for func in program.functions.values():
        finder.visit_stmt(func.body)
    return finder.found


def check_native_fp4_mx_support() -> None:
    """Require verified logical FP4 geometry, including for development builds.

    Development builds may report the release version with a different ABI. Cache only
    successes, keyed by resolved executable and file identity, so replacing a
    rejected build or repointing PTOAS_ROOT takes effect in the same process.
    """
    binary = find_ptoas_binary()
    hint = (
        f"Native FP4 MX matmul requires PTOAS {PTOAS_MIN_VERSION} or newer with the logical Cube "
        "geometry contract (PTOAS #1700). "
        f"Set PTOAS_ROOT to the released {PTOAS_MIN_VERSION} assembler or a compatible newer/patched build."
    )
    if binary is None:
        raise RuntimeError(f"{hint} No PTOAS executable was found.")
    version = check_ptoas_version(binary)
    with _probe_lock:
        info = os.stat(binary)
        key = (os.path.realpath(binary), info.st_mtime_ns, info.st_size)
        if key in _verified_binaries:
            return
        try:
            with tempfile.TemporaryDirectory(prefix="pypto-fp4-probe-") as directory:
                source = Path(directory) / "probe.pto"
                source.write_text(_PROBE)
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
        expected = (
            r"Tile<TileType::Left,\s*float4_e2m1x2_t,\s*32,\s*128,",
            r"Tile<TileType::Right,\s*float4_e1m2x2_t,\s*128,\s*64,",
            r"Tile<TileType::Acc,\s*float,\s*32,\s*64,",
            r"Tile<TileType::ScaleLeft,\s*float8_e8m0_t,\s*32,\s*4,",
            r"Tile<TileType::ScaleRight,\s*float8_e8m0_t,\s*4,\s*64,",
        )
        if result.returncode != 0 or not all(re.search(pattern, result.stdout) for pattern in expected):
            detail = result.stderr.strip() or "emitted tile geometry did not match the FP4 carrier contract"
            raise RuntimeError(f"{hint} Capability probe failed for {version}: {detail[:1000]}")
        _verified_binaries.add(key)
