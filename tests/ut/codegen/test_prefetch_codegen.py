# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen tests for the ``prefetch.*`` async GM->L2 prefetch op family."""

import re

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _generate_mlir(program_cls) -> str:
    """Run PassManager and PTOCodegen on the given program, return MLIR string."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    funcs = list(optimized.functions.values())
    assert funcs, "Program has no functions"
    single = ir.Program([funcs[0]], funcs[0].name, optimized.span)
    return codegen.PTOCodegen().generate(single)


@pl.program
class PrefetchProgram:
    """Prefetch a 1D GM row into L2, then copy a slice of it out."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        x: pl.Tensor[[1, 4096], pl.FP32],
        ws: pl.Tensor[[1024], pl.INT8],
        out: pl.Tensor[[1, 128], pl.FP32],
    ) -> pl.Tensor[[1, 128], pl.FP32]:
        ctx = pl.prefetch.make_context(ws)
        evt = pl.prefetch.async_prefetch(x, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
        tile = pl.load(x, [0, 0], [1, 128])
        return pl.store(tile, [0, 0], out)


class TestPrefetchPTOCodegen:
    """Each prefetch op lowers to its PTOAS counterpart with the right operand types."""

    def test_make_context_lowers_from_i8_pointer(self):
        """The INT8 workspace param feeds ``pto.make_prefetch_async_context`` as ``!pto.ptr<i8>``."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.make_prefetch_async_context\(%\w+ : !pto\.ptr<i8>\) -> !pto\.prefetch_async_context",
            mlir,
        ), mlir

    def test_async_prefetch_lowers_with_partition_view(self):
        """``tprefetch_async`` takes a whole-tensor partition view plus the context."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.tprefetch_async\(%\w+, %\w+ : "
            r"!pto\.partition_tensor_view<1x4096xf32>, !pto\.prefetch_async_context\) "
            r"-> !pto\.async_event",
            mlir,
        ), mlir

    def test_session_uses_projection_assembly_form(self):
        """``get_prefetch_async_session`` is a bare projection — no parenthesised operand list."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.get_prefetch_async_session %\w+ : "
            r"!pto\.prefetch_async_context -> !pto\.async_session",
            mlir,
        ), mlir

    def test_wait_lowers_to_comm_wait_async_event(self):
        """``wait`` pairs the event and session and yields an ``i1``."""
        mlir = _generate_mlir(PrefetchProgram)
        assert re.search(
            r"= pto\.comm\.wait_async_event\(%\w+, %\w+ : "
            r"!pto\.async_event, !pto\.async_session\) -> i1",
            mlir,
        ), mlir

    def test_handle_ssa_values_are_defined_before_use(self):
        """Each handle operand resolves to an SSA name defined earlier in the function.

        Regression guard: the handle types carry no buffer, so an emitter that
        invented a fresh temp instead of defining the assignment's bound LHS name
        would produce operands referencing undefined SSA values.
        """
        mlir = _generate_mlir(PrefetchProgram)
        defined: set[str] = set()
        for line in mlir.splitlines():
            stripped = line.strip()
            operands = re.findall(r"%\w+", stripped)
            if " = " in stripped:
                lhs = stripped.split(" = ", 1)[0].strip()
                operands = re.findall(r"%\w+", stripped.split(" = ", 1)[1])
            else:
                lhs = None
            if "pto.tprefetch_async" in stripped or "pto.comm.wait_async_event" in stripped:
                for operand in operands:
                    assert operand in defined, f"{operand} used before definition in: {stripped}"
            if lhs is not None:
                defined.add(lhs)


@pl.program
class MixedPrefetchProgram:
    """Prefetch alongside a cube op, so ExpandMixedKernel splits AIC/AIV."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(
        self,
        a: pl.Tensor[[128, 128], pl.FP16],
        b: pl.Tensor[[128, 128], pl.FP16],
        ws: pl.Tensor[[65536], pl.INT8],
        out: pl.Tensor[[128, 128], pl.FP32],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        ctx = pl.prefetch.make_context(ws)
        evt = pl.prefetch.async_prefetch(ws, ctx)
        session = pl.prefetch.session(ctx)
        pl.prefetch.wait(evt, session)
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_b = pl.load(b, [0, 0], [128, 128])
        return pl.store(pl.tile.matmul(tile_a, tile_b), [0, 0], out)


class TestPrefetchCoreAffinity:
    """The prefetch family is AIV-only and must not leak onto the cube lane."""

    def test_prefetch_stays_off_the_cube_lane(self):
        """In a mixed kernel, prefetch ops land only in the AIV function.

        ``TPREFETCH_ASYNC`` drives its SDMA tmpBuf from a Vec(UB) scratch tile
        inside ``PrefetchAsyncContext`` (pto-isa static_asserts
        ``ScratchTile::Loc == TileType::Vec``), and UB lives on the vector core.
        These ops carry no tile operand, so without an explicit VECTOR core
        affinity they classify as SHARED and ExpandMixedKernel duplicates them
        onto the cube lane — which has no UB, and which would also run the
        side-effecting prefetch a second time.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(MixedPrefetchProgram)

        per_core: dict[str, int] = {}
        for func in optimized.functions.values():
            if func.func_type.name not in ("AIC", "AIV"):
                continue
            mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))
            per_core[func.func_type.name] = mlir.count("pto.tprefetch_async") + mlir.count(
                "pto.make_prefetch_async_context"
            )

        assert "AIC" in per_core and "AIV" in per_core, f"expected a mixed AIC/AIV split, got {per_core}"
        assert per_core["AIC"] == 0, f"prefetch leaked onto the cube lane: {per_core}"
        assert per_core["AIV"] > 0, f"prefetch missing from the vector lane: {per_core}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
