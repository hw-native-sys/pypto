# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Row recipes keep scratch effects, column layouts and storage overlap explicit."""

from typing import Any

import pytest
from pypto import DataType, ir, passes
from pypto.pypto_core import ir as _ir

_SPAN = ir.Span.unknown()
_RECIPES = ("row_max", "row_sum", "row_expand_sub", "row_expand_div")


def _buffer(name, column=False, **changes):
    options: dict[str, Any] = dict(
        shape=[16, 1] if column else [16, 64],
        dtype=DataType.FP32,
        memory_space=ir.MemorySpace.Vec,
        blayout=ir.TileLayout.col_major if column else ir.TileLayout.row_major,
    )
    options.update(changes)
    return ir.Var(name, ir.BufferType(**options), _SPAN)


def _args(suffix):
    reduction = suffix in ("row_max", "row_sum")
    return [_buffer("src"), _buffer("aux", column=not reduction), _buffer("dst", column=reduction)]


def _call(suffix, arguments):
    return _ir._create_internal_op_call(f"buffer.{suffix}", arguments, {}, _SPAN)


@pytest.mark.parametrize("suffix", _RECIPES)
def test_row_recipe_effects_and_roundtrip(suffix):
    arguments = _args(suffix)
    call = _call(suffix, arguments)
    assert isinstance(call.type, ir.VoidType)
    reduction = suffix in ("row_max", "row_sum")
    for index in range(3):
        effect = ir.get_op_buffer_arg_effect(call.op.name, index)
        writes = index == 2 or (reduction and index == 1)
        assert effect.data == (ir.BufferAccess.Write if writes else ir.BufferAccess.Read)
        assert effect.metadata == ir.BufferAccess.Read
        assert ir.op_arg_is_workspace(call.op.name, index) == (reduction and index == 1)
    statement = ir.EvalStmt(call, _SPAN)
    ir.assert_structural_equal(statement, ir.deserialize(ir.serialize(statement)), enable_auto_mapping=True)
    with pytest.raises(ValueError, match="two inputs and an explicit destination"):
        _call(suffix, arguments[:2])


@pytest.mark.parametrize("suffix", _RECIPES)
@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.BF16, DataType.INT32])
def test_row_recipes_do_not_admit_unverified_dtypes(suffix, dtype):
    arguments = _args(suffix)
    arguments[0] = _buffer("src", dtype=dtype)
    with pytest.raises(ValueError, match="FP32"):
        _call(suffix, arguments)


@pytest.mark.parametrize("suffix", _RECIPES)
@pytest.mark.parametrize("valid", [[-1, -1], [0, 64], [16, 0]])
def test_row_recipes_reject_dynamic_and_empty_reduction_domains(suffix, valid):
    arguments = _args(suffix)
    arguments[0] = _buffer("src", valid_shape=valid)
    with pytest.raises(ValueError, match="nonempty static valid"):
        _call(suffix, arguments)


@pytest.mark.parametrize("suffix", ["row_max", "row_sum"])
@pytest.mark.parametrize("scratch_shape", [[8, 64], [16, 32]])
def test_reduction_workspace_must_cover_every_physical_dimension(suffix, scratch_shape):
    arguments = _args(suffix)
    arguments[1] = _buffer("scratch", shape=scratch_shape)
    with pytest.raises(ValueError, match="scratch covering"):
        _call(suffix, arguments)


@pytest.mark.parametrize("suffix", _RECIPES)
def test_row_recipe_requires_matching_row_geometry(suffix):
    arguments = _args(suffix)
    column_index = 2 if suffix in ("row_max", "row_sum") else 1
    arguments[column_index] = _buffer("column", column=True, valid_shape=[15, 1])
    with pytest.raises(ValueError, match="preserve physical and valid rows"):
        _call(suffix, arguments)


def _verify_storage(suffix, addresses):
    arguments = _args(suffix)
    allocations = [
        ir.AssignStmt(
            value,
            ir.Call(
                ir.get_op("buffer.alloc"),
                [ir.MakeTuple([], _SPAN), ir.ConstInt(address, DataType.INDEX, _SPAN)],
                {},
                None,
                value.type,
                _SPAN,
            ),
            _SPAN,
        )
        for value, address in zip(arguments, addresses)
    ]
    body = ir.SeqStmts([*allocations, ir.EvalStmt(_call(suffix, arguments), _SPAN)], _SPAN)
    program = ir.Program(
        [
            ir.Function(
                "kernel", [], [], body, _SPAN, type=ir.FunctionType.InCore, ir_stage=ir.FunctionIRStage.Buffer
            )
        ],
        "Rows",
        _SPAN,
    )
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.BufferIR)
    return passes.PropertyVerifierRegistry.verify(properties, program)


@pytest.mark.parametrize("suffix", ["row_max", "row_sum"])
@pytest.mark.parametrize("addresses", [(0, 0, 8192), (0, 4096, 0), (0, 4096, 4096)])
def test_reduction_rejects_overlap_between_each_live_pair(suffix, addresses):
    diagnostics = _verify_storage(suffix, addresses)
    assert diagnostics and any("disjoint" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("suffix", ["row_expand_sub", "row_expand_div"])
def test_broadcast_can_reuse_matrix_but_not_its_repeated_column(suffix):
    assert _verify_storage(suffix, (0, 4096, 0)) == []
    diagnostics = _verify_storage(suffix, (0, 4096, 4096))
    assert diagnostics and any("disjoint" in diagnostic.message for diagnostic in diagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
