# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression: MaterializeAllocTiles dynamic-valid fixup (issue #1956 follow-up).

When memory reuse shares one on-chip buffer across two sibling loops, each
loading with its own *loop-local* dynamic ``valid_shape`` operand, the single
handle is hoisted to the common ancestor — above both operands. Emitting the
handle's dynamic ``valid_col`` there would reference an out-of-scope operand,
which ptoas rejects (``'pto.alloc_tile' op valid_col operand is required because
result type v_col is ?``).

The pass repairs this by declaring the handle with a *static* valid (physical
shape) and re-establishing each use's real valid with an injected
``tile.set_validshape`` where the operand is in scope.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


@pl.program
class ReuseDynamicValidAcrossLoops:
    """Two sibling loops load a [16, 64] Vec tile with a loop-local valid_shape.

    The two tiles have disjoint lifetimes, so MemoryReuse coalesces them onto one
    ``mem_vec`` slot — a single hoisted handle that must not carry either loop's
    dynamic ``valid_len``.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        for i in pl.range(2):
            vlen: pl.Scalar[pl.INDEX] = i + 1
            t: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[16, vlen])] = pl.load(
                a, [i * 16, 0], [16, 64], [16, vlen], target_memory=pl.MemorySpace.Vec
            )
            p: pl.Tile[[16, 64], pl.FP32] = pl.fillpad(t, pad_value=pl.PadValue.min)
            output = pl.store(p, [i * 16, 0], output)
        for j in pl.range(2):
            vlen2: pl.Scalar[pl.INDEX] = j + 1
            t2: pl.Tile[[16, 64], pl.FP32, pl.TileView(valid_shape=[16, vlen2])] = pl.load(
                a, [j * 16, 0], [16, 64], [16, vlen2], target_memory=pl.MemorySpace.Vec
            )
            p2: pl.Tile[[16, 64], pl.FP32] = pl.fillpad(t2, pad_value=pl.PadValue.min)
            output = pl.store(p2, [j * 16, 0], output)
        return output


def _codegen_kernel() -> str:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            ReuseDynamicValidAcrossLoops
        )
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(ir.Program([func], "kernel", optimized.span), emit_tile_addr=True)


def test_hoisted_alloc_declares_static_valid():
    """The buffer's single hoisted alloc_tile must declare a STATIC valid_col.

    Before the fix the loop-local ``valid_len`` was carried onto the hoisted
    handle, so codegen emitted no ``valid_col`` (operand out of scope) and ptoas
    rejected the module. The handle must now carry a constant ``valid_col``.
    """
    mlir = _codegen_kernel()
    # The reused load buffer: the pad=0 Vec alloc_tile that the tloads write to.
    load_allocs = [
        ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln and "loc=vec" in ln and "pad=0" in ln
    ]
    assert load_allocs, f"expected a pad=0 Vec alloc_tile:\n{mlir}"
    for ln in load_allocs:
        assert "valid_col =" in ln, f"hoisted alloc must declare a valid_col operand:\n{ln}"
        # Static valid: the valid_col operand is a numeric constant SSA (%c...),
        # never a loop-local scalar (which would be out of scope at the hoist).
        vcol = ln.split("valid_col =")[1].split(":")[0].strip().split()[0]
        assert vcol.startswith("%c"), (
            f"hoisted alloc valid_col must be a static constant, got {vcol!r}:\n{ln}"
        )


def test_dynamic_valid_reestablished_by_set_validshape():
    """Each loop re-establishes its loop-local valid via an injected set_validshape."""
    mlir = _codegen_kernel()
    sv_lines = [ln for ln in mlir.splitlines() if "pto.set_validshape" in ln]
    # One per loop (two sibling loops share the hoisted handle).
    assert len(sv_lines) >= 2, f"expected an injected set_validshape per loop:\n{mlir}"


def test_set_validshape_precedes_the_consuming_load():
    """The injected set_validshape must run BEFORE the tload that writes the buffer.

    The load's fill/pad extent follows the destination tile's valid, so the real
    dynamic valid must be established before the load — not after. Emitting it
    after left the buffer at its static valid during the load and corrupted
    partial-valid blocks (paged-attention golden mismatch).
    """
    lines = _codegen_kernel().splitlines()
    handle = None
    for ln in lines:
        if "pto.alloc_tile" in ln and "loc=vec" in ln and "pad=0" in ln:
            handle = ln.split("=")[0].strip()
            break
    assert handle, "no static Vec load-buffer alloc_tile found"
    # Every tload into the handle must be preceded (since the prior set_validshape)
    # by a set_validshape on that handle — i.e. no tload occurs while the handle's
    # valid is still stale.
    seen_sv = False
    for ln in lines:
        if f"pto.set_validshape {handle}" in ln:
            seen_sv = True
        elif "pto.tload" in ln and f"outs({handle}" in ln:
            assert seen_sv, f"tload into {handle} not preceded by a set_validshape:\n{ln}"
            seen_sv = False  # consumed; the next load needs its own set_validshape


def test_single_handle_shared_across_both_loops():
    """The reused buffer resolves to exactly one alloc_tile handle used by both loops."""
    mlir = _codegen_kernel()
    tloads = [ln for ln in mlir.splitlines() if "pto.tload" in ln and "outs(" in ln]
    load_handles = {ln.split("outs(")[1].split(":")[0].strip() for ln in tloads}
    # Both loop tloads target the same coalesced handle.
    shared = [h for h in load_handles if "__buf" in h]
    assert shared, f"expected a coalesced tile-buf handle in tload outs:\n{mlir}"
    for handle in shared:
        allocs = [ln for ln in mlir.splitlines() if "pto.alloc_tile" in ln and handle in ln.split("=")[0]]
        assert len(allocs) == 1, f"buffer {handle} must have exactly one alloc_tile:\n{mlir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
