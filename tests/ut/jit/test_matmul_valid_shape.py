# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-level coverage for the matmul valid-shape example."""

import importlib
from typing import Any, cast

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType

_example = importlib.import_module("examples.kernels.12_matmul_valid_shape")
TILE_M = cast(int, getattr(_example, "TILE_M"))
VALID_M = cast(int, getattr(_example, "VALID_M"))
K = cast(int, getattr(_example, "K"))
N = cast(int, getattr(_example, "N"))
matmul_valid_shape = cast(Any, getattr(_example, "matmul_valid_shape"))


class _CallCollector(ir.IRVisitor):
    """Collect calls to the requested built-in operations."""

    def __init__(self, op_names: set[str]) -> None:
        super().__init__()
        self.calls: dict[str, list[ir.Call]] = {name: [] for name in op_names}

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name in self.calls:
            self.calls[op.op.name].append(op)
        super().visit_call(op)


def _const_values(expressions) -> list[int]:
    """Return the integer values of a static IR shape."""
    assert all(isinstance(expression, ir.ConstInt) for expression in expressions)
    return [expression.value for expression in expressions]


@pytest.fixture(scope="module")
def compiled_programs(request: pytest.FixtureRequest) -> tuple[ir.Program, ir.Program, ir.Function]:
    """Compile the example without executing it on hardware."""
    torch = pytest.importorskip("torch")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    request.addfinalizer(backend.reset_for_testing)
    args = (
        torch.randn(VALID_M, K),
        torch.randn(K, N),
        torch.zeros(TILE_M, N),
    )
    _, _, tensor_meta, scalar_values, scalar_dtypes, per_func_dyn = matmul_valid_shape._bind_args(args, {})
    pre_pass = matmul_valid_shape._compile_to_program(
        tensor_meta,
        scalar_values,
        scalar_dtypes,
        per_func_dyn,
        pl,
    )
    post_pass = matmul_valid_shape.lower(*args)
    incore = next(
        function for function in post_pass.functions.values() if ir.is_incore_type(function.func_type)
    )
    return pre_pass, post_pass, incore


def test_matmul_valid_shape_declares_inout_output(compiled_programs):
    """The partially stored output must declare its preserved tail as input state."""
    pre_pass, _, _ = compiled_programs
    entry = pre_pass.get_function("matmul_valid_shape")
    assert entry.param_directions[-1] == ir.ParamDirection.InOut


def test_matmul_valid_shape_store_keeps_logical_result_extent(compiled_programs):
    """A physical 16x16 result stores only its five logically valid rows."""
    _, post_pass, incore = compiled_programs
    collector = _CallCollector({"tile.matmul", "tile.slice", "tile.store"})
    collector.visit_stmt(incore.body)

    assert len(collector.calls["tile.matmul"]) == 1
    assert len(collector.calls["tile.slice"]) == 1
    assert len(collector.calls["tile.store"]) == 1
    matmul_call = collector.calls["tile.matmul"][0]
    slice_call = collector.calls["tile.slice"][0]
    store_call = collector.calls["tile.store"][0]

    assert isinstance(matmul_call.type, ir.TileType)
    assert _const_values(matmul_call.type.shape) == [TILE_M, N]
    assert isinstance(slice_call.type, ir.TileType)
    assert _const_values(slice_call.type.shape) == [VALID_M, N]
    assert _const_values(slice_call.type.get_effective_tile_view().valid_shape) == [VALID_M, N]
    stored_result_type = store_call.args[0].type
    assert isinstance(stored_result_type, ir.TileType)
    assert _const_values(stored_result_type.shape) == [VALID_M, N]
    assert _const_values(stored_result_type.get_effective_tile_view().valid_shape) == [VALID_M, N]

    pto = codegen.PTOCodegen().generate(ir.Program([incore], incore.name, post_pass.span))
    tstore_lines = [line for line in pto.splitlines() if "pto.tstore" in line]
    assert len(tstore_lines) == 1
    assert f"partition_tensor_view<{VALID_M}x{N}xf32>" in tstore_lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
