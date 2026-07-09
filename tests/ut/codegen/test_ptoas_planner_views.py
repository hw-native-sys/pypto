# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen under memory_planner=PTOAS: reserved buffers and view def/use types.

With the PTOAS planner, `MemoryReuse` + `AllocateMemoryAddr` are skipped and ptoas
`PlanMemory` owns on-chip placement (--pto-level=level2). Two things that the
default PyPTO planner hides then have to be emitted correctly:

* `system.reserve_buffer(base=AUTO)` never gets a resolved base, so PTO must emit
  ptoas's `auto = true` form (base absent) instead of the manual `base = <n>` one.
* A view chain (`tile.slice` -> `tile.reshape`) no longer folds into per-variable
  `pto.alloc_tile` re-views at one baked address, so it survives as a real
  `pto.subview` + `pto.treshape` pair whose def/use type strings must agree.
"""

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


def _emit_pto(program, planner: passes.MemoryPlanner) -> str:
    """Run the default pipeline under `planner` and return the emitted PTO MLIR."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    emit_tile_addr = planner == passes.MemoryPlanner.PYPTO
    result = codegen.PTOCodegen().generate(optimized, emit_tile_addr=emit_tile_addr)
    return result if isinstance(result, str) else "".join(result.values())


def _sole_line(mlir: str, needle: str) -> str:
    lines = [ln for ln in mlir.splitlines() if needle in ln]
    assert len(lines) == 1, f"expected exactly one {needle!r} line, got {lines}:\n{mlir}"
    return lines[0]


# ── reserve_buffer: base resolution deferred to ptoas ────────────────────────


@pl.program
class AutoReserveBufferProgram:
    """Cross-core pipe whose slot buffers are declared with `base=AUTO`."""

    @pl.function(type=pl.FunctionType.AIV)
    def vector_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096)
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_producer")
        pl.aiv_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_buf, v2c_consumer_buf=v2c_peer
        )

        tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        pl.tpush_to_aic(tile_a, split=0)

        t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        out: pl.Tile[[16, 16], pl.FP32] = pl.exp(t)
        pl.tfree_to_aic(t)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(out, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.AIC)
    def cube_producer(self, arg: pl.Tensor[[16, 16], pl.FP32]):
        v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096)
        c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
        pl.aic_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buf
        )
        received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
        pl.tpush_to_aiv(received, split=0)
        pl.tfree_to_aiv(received)


def test_reserve_buffer_defers_base_to_ptoas():
    """PTOAS planner: `base` is never resolved, so emit ptoas's auto-placement form.

    ptoas rejects `auto = true` alongside a `base` attribute (and `auto = false`
    without one), so the two must move together.
    """
    mlir = _emit_pto(AutoReserveBufferProgram, passes.MemoryPlanner.PTOAS)
    for name in ("c2v_slot_buffer", "v2c_slot_buffer"):
        line = _sole_line(mlir, f'pto.reserve_buffer {{name = "{name}"')
        assert "auto = true" in line, line
        assert "base" not in line, line


def test_reserve_buffer_bakes_resolved_base_under_pypto_planner():
    """Default planner: AllocateMemoryAddr resolves `base`, emitted as manual mode."""
    mlir = _emit_pto(AutoReserveBufferProgram, passes.MemoryPlanner.PYPTO)
    for name in ("c2v_slot_buffer", "v2c_slot_buffer"):
        line = _sole_line(mlir, f'pto.reserve_buffer {{name = "{name}"')
        assert "auto = false" in line, line
        assert "base = 0" in line, line


# ── reshape of a subview: def/use tile_buf types must agree ──────────────────

PAD, VALID, D = 16, 5, 128


@pl.program
class SubviewReshapeProgram:
    """Slice the padded rows off a vec tile, then reshape the slice to one row."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[PAD, D], pl.FP32],
        out: pl.Out[pl.Tensor[[1, VALID * D], pl.FP32]],
    ) -> pl.Tensor[[1, VALID * D], pl.FP32]:
        t: pl.Tile[[PAD, D], pl.FP32] = pl.load(x, [0, 0], [PAD, D])
        v: pl.Tile[[VALID, D], pl.FP32] = pl.tile.slice(t, [VALID, D], [0, 0])
        r: pl.Tile[[1, VALID * D], pl.FP32] = pl.reshape(v, [1, VALID * D])
        return pl.store(r, [0, 0], out)


def _result_type(op_line: str) -> str:
    """The type right of `->` in an `op : <src> -> <dst>` annotation."""
    assert " -> " in op_line, f"expected a src -> dst annotation in: {op_line}"
    return op_line.split(" -> ", 1)[1].strip()


def _operand_type(op_line: str) -> str:
    """The type left of `->` in an `op : <src> -> <dst>` annotation."""
    assert " : " in op_line and " -> " in op_line, f"expected a src -> dst annotation in: {op_line}"
    return op_line.split(" : ", 1)[1].split(" -> ", 1)[0].strip()


def test_reshape_of_subview_annotates_the_subview_def_type():
    """A `pto.treshape` reading a `pto.subview` must annotate the subview's DEF type.

    `pto.subview` infers static valid dims (`v_row=5, v_col=128`) from its slice
    `sizes`, while every IR TileType renders as `v_row=?, v_col=?`. Deriving the
    treshape operand type from the TileType therefore prints `valid=?x?` at the
    use, and MLIR rejects the def/use mismatch.
    """
    mlir = _emit_pto(SubviewReshapeProgram, passes.MemoryPlanner.PTOAS)
    subview = _sole_line(mlir, "pto.subview")
    treshape = _sole_line(mlir, "pto.treshape")

    assert f"v_row={VALID}, v_col={D}" in _result_type(subview), subview
    assert _operand_type(treshape) == _result_type(subview), f"{subview}\n{treshape}"


def test_reshape_of_subview_folds_away_under_pypto_planner():
    """Default planner: the reshape result is pre-declared at the shared baked
    address, so it is a re-view and no `pto.treshape` is emitted at all."""
    mlir = _emit_pto(SubviewReshapeProgram, passes.MemoryPlanner.PYPTO)
    assert "pto.treshape" not in mlir, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
