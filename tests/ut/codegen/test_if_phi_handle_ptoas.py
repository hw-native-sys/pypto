# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression: if/else-yield tile phi under memory_planner=PTOAS (issue #1956).

Under PTOAS the opportunistic MemoryReuse pass (whose YieldFixupMutator aliases a
branch's tile yields onto the phi return_var's canonical handle) is skipped. PTO
codegen must therefore itself point each branch's tile producer at the single
head-declared phi handle; otherwise each branch writes its own handle and the
post-if read of the phi handle reads a buffer no branch ever wrote — a silent
miscompile. This test asserts both branch producers write the one phi handle and
that handle is declared exactly once.
"""

# pyright: reportUndefinedVariable=false

import re

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pl.program
class IfPhiProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        a: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        if flag == 0:
            r: pl.Tile[[64, 64], pl.FP32] = pl.add(a, 1.0)
        else:
            r: pl.Tile[[64, 64], pl.FP32] = pl.mul(a, 2.0)
        return pl.store(r, [0, 0], out)


def _ptoas_pto() -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(IfPhiProgram)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span), emit_tile_addr=False)


def _first_group(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    assert m is not None, f"pattern {pattern!r} not found in:\n{text}"
    return m.group(1)


def test_if_phi_branches_write_one_head_declared_handle():
    mlir = _ptoas_pto()
    # The tile the store reads is the phi buffer.
    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    phi = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # It is declared exactly once (at the function head, dominating both branches).
    decls = [ln for ln in mlir.splitlines() if f"{phi} = pto.alloc_tile" in ln]
    assert len(decls) == 1, f"phi handle {phi} must be declared exactly once:\n{mlir}"

    # Both branch producers (tadds / tmuls) write that same phi handle.
    producer_outs = [
        _first_group(r"outs\((%[A-Za-z0-9_]+)", ln)
        for ln in mlir.splitlines()
        if ("pto.tadds" in ln or "pto.tmuls" in ln) and "outs(" in ln
    ]
    assert len(producer_outs) == 2, f"expected two branch producers, got {producer_outs}:\n{mlir}"
    for out in producer_outs:
        assert out == phi, f"branch producer writes {out}, not the phi handle {phi}:\n{mlir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
