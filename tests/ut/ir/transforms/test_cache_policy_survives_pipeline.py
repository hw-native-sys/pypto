# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end guard: a declared GM cache policy survives the whole Default pipeline.

``pl.set_cache_policy(b, pl.CachePolicy.BYPASS)`` changes carrier three times on
its way to codegen::

    scope attr ``cache_policy_vars``   (parse .. pass 8)
      -> outlined function attr ``cache_policy``, param indices  (pass 8 .. pass 10)
      -> ``tile.load`` kwarg ``cache``                            (pass 10 .. codegen)

The first two hops are unit-tested where they happen — ``test_outline_incore_scopes.py``
and ``test_convert_tensor_to_tile_ops.py``. This file guards the long tail: the
remaining ~37 passes must carry the ``cache`` kwarg through matmul tiling,
memory planning, and signature rewriting without dropping or misplacing it. The
kwarg rides the same channel ``target_memory`` does, which is what makes the
guard cheap; it is also why a regression here would be silent — a dropped kwarg
compiles fine and merely stops declaring anything.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_BYPASS = int(pl.CachePolicy.BYPASS)


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


class _TileLoadCollector(ir.IRVisitor):
    """Collect every ``tile.load`` Call in the visited IR."""

    def __init__(self) -> None:
        super().__init__()
        self.op_name = ir.get_op("tile.load").name
        self.found: list[ir.Call] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == self.op_name:
            self.found.append(op)
        super().visit_call(op)


def _loads_by_source(program: ir.Program) -> dict[str, list[ir.Call]]:
    """Group every ``tile.load`` in ``program`` by the name of the tensor it reads.

    Keyed on the SSA name because the pipeline renames and re-homes functions;
    the source Var itself stays the kernel parameter the declaration resolved to.
    """
    collector = _TileLoadCollector()
    for func in program.functions.values():
        collector.visit_function(func)

    grouped: dict[str, list[ir.Call]] = {}
    for call in collector.found:
        source = call.args[0]
        assert isinstance(source, ir.Var), f"tile.load source is not a Var: {source}"
        grouped.setdefault(source.name_hint, []).append(call)
    return grouped


def _demo_program(*, declare: bool) -> ir.Program:
    """The two-operand matmul kernel, with and without the declaration on ``b``.

    The pair differs by exactly one line, so the undeclared run is the control
    the declared one is read against.
    """
    if declare:

        @pl.program
        class Declared:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 128], pl.FP32],
                b: pl.Tensor[[128, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
            ) -> pl.Tensor[[256, 256], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                    c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                    out = pl.assemble(out, c, [0, 0])
                return out

        return Declared

    @pl.program
    class Undeclared:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                out = pl.assemble(out, c, [0, 0])
            return out

    return Undeclared


def _run_default_pipeline(program: ir.Program) -> ir.Program:
    return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def test_cache_kwarg_reaches_the_end_of_the_pipeline():
    """The declared tensor's GM read still carries ``cache`` after all 47 passes."""
    after = _run_default_pipeline(_demo_program(declare=True))
    loads = _loads_by_source(after)

    declared = [name for name in loads if name.startswith("b")]
    assert len(declared) == 1, f"expected one source named after 'b', got {sorted(loads)}"
    b_loads = loads[declared[0]]
    assert b_loads, "the declared tensor is never loaded — the guard would pass vacuously"
    assert all(load.kwargs.get("cache") == _BYPASS for load in b_loads)


def test_only_the_declared_tensor_is_annotated():
    """The declaration names one tensor: the other operand's load stays plain."""
    after = _run_default_pipeline(_demo_program(declare=True))
    loads = _loads_by_source(after)

    undeclared = [name for name in loads if not name.startswith("b")]
    assert undeclared, "expected at least one load of an undeclared tensor"
    for name in undeclared:
        assert all("cache" not in load.kwargs for load in loads[name]), f"'{name}' was annotated"


def test_undeclared_program_carries_no_cache_kwarg():
    """Control: without the declaration no load acquires the kwarg anywhere."""
    after = _run_default_pipeline(_demo_program(declare=False))
    loads = _loads_by_source(after)

    assert loads, "expected the pipeline to emit tile.loads"
    for calls in loads.values():
        assert all("cache" not in load.kwargs for load in calls)


def test_no_carrier_attr_outlives_its_consumer():
    """Both intermediate carriers are consumed: the scope attr at pass 8, the
    param-index function attr at pass 10.

    Param indices in particular must not survive — passes after 10 append to
    param lists (InjectGMPipeBuffer, MaterializeDistTensorCtx) and prepend onto
    them (MaterializeValidShapeSymbols), so a surviving index would silently
    name the wrong tensor.
    """
    after = _run_default_pipeline(_demo_program(declare=True))

    for func in after.functions.values():
        attrs = dict(func.attrs)
        assert "cache_policy" not in attrs, f"stale param indices on '{func.name}'"
        assert "cache_policy_vars" not in attrs, f"stale scope declaration on '{func.name}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
