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
_REDUCTIONS = ("row_max", "row_sum", "row_min", "row_prod")
_BROADCASTS = (
    "row_expand_add",
    "row_expand_sub",
    "row_expand_mul",
    "row_expand_div",
    "row_expand_max",
    "row_expand_min",
)
_RECIPES = (*_REDUCTIONS, *_BROADCASTS)


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
    if suffix.startswith("col_"):
        row = _buffer("row", shape=[1, 64])
        return [_buffer("src"), row, _buffer("dst")] if "expand" in suffix else [_buffer("src"), row]
    reduction = suffix in _REDUCTIONS
    return [_buffer("src"), _buffer("aux", column=not reduction), _buffer("dst", column=reduction)]


def _call(suffix, arguments):
    return _ir._create_internal_op_call(f"buffer.{suffix}", arguments, {}, _SPAN)


@pytest.mark.parametrize("suffix", _RECIPES)
def test_row_recipe_effects_and_roundtrip(suffix):
    arguments = _args(suffix)
    call = _call(suffix, arguments)
    assert isinstance(call.type, ir.VoidType)
    reduction = suffix in _REDUCTIONS
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


@pytest.mark.parametrize("suffix", _REDUCTIONS)
@pytest.mark.parametrize("scratch_shape", [[8, 64], [16, 32]])
def test_reduction_workspace_must_cover_every_physical_dimension(suffix, scratch_shape):
    arguments = _args(suffix)
    arguments[1] = _buffer("scratch", shape=scratch_shape)
    with pytest.raises(ValueError, match="scratch covering"):
        _call(suffix, arguments)


@pytest.mark.parametrize("suffix", _RECIPES)
def test_row_recipe_requires_matching_row_geometry(suffix):
    arguments = _args(suffix)
    column_index = 2 if suffix in _REDUCTIONS else 1
    arguments[column_index] = _buffer("column", column=True, valid_shape=[15, 1])
    with pytest.raises(ValueError, match="preserve physical and valid rows"):
        _call(suffix, arguments)


def _verify_storage(suffix, addresses, source_window=False):
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
    if source_window:
        # The source still advertises [16,64], but its parent has a 128-column
        # row pitch. Descriptor-only validation cannot detect this distinction.
        parent = _buffer("parent", shape=[16, 128])
        allocations[0] = ir.AssignStmt(
            parent,
            ir.Call(
                ir.get_op("buffer.alloc"),
                [ir.MakeTuple([], _SPAN), ir.ConstInt(0, DataType.INDEX, _SPAN)],
                {},
                None,
                parent.type,
                _SPAN,
            ),
            _SPAN,
        )
        offsets = ir.MakeTuple([ir.ConstInt(0, DataType.INDEX, _SPAN)] * 2, _SPAN)
        allocations.append(
            ir.AssignStmt(
                arguments[0],
                ir.Call(ir.get_op("buffer.subview"), [parent, offsets], {}, None, arguments[0].type, _SPAN),
                _SPAN,
            )
        )
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


@pytest.mark.parametrize("suffix", _REDUCTIONS)
@pytest.mark.parametrize("addresses", [(0, 0, 8192), (0, 4096, 0), (0, 4096, 4096)])
def test_reduction_rejects_overlap_between_each_live_pair(suffix, addresses):
    diagnostics = _verify_storage(suffix, addresses)
    assert diagnostics and any("disjoint" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("suffix", _BROADCASTS)
def test_broadcast_can_reuse_matrix_but_not_its_repeated_column(suffix):
    assert _verify_storage(suffix, (0, 4096, 0)) == []
    diagnostics = _verify_storage(suffix, (0, 4096, 4096))
    assert diagnostics and any("disjoint" in diagnostic.message for diagnostic in diagnostics)


_COLUMN_REDUCTIONS = ("col_sum", "col_max", "col_min", "col_prod")
_COLUMN_BROADCASTS = tuple(f"col_expand_{op}" for op in ("add", "sub", "mul", "div", "max", "min"))
_COLUMN_RECIPES = (*_COLUMN_REDUCTIONS, *_COLUMN_BROADCASTS)


@pytest.mark.parametrize("suffix", _COLUMN_RECIPES)
def test_column_effects_and_roundtrip(suffix):
    arguments = _args(suffix)
    call = _call(suffix, arguments)
    assert isinstance(call.type, ir.VoidType)
    for index in range(len(arguments)):
        effect = ir.get_op_buffer_arg_effect(call.op.name, index)
        assert effect.data == (ir.BufferAccess.Write if index == len(arguments) - 1 else ir.BufferAccess.Read)
        assert effect.metadata == ir.BufferAccess.Read
        assert not ir.op_arg_is_workspace(call.op.name, index)
    statement = ir.EvalStmt(call, _SPAN)
    ir.assert_structural_equal(statement, ir.deserialize(ir.serialize(statement)), enable_auto_mapping=True)
    with pytest.raises(ValueError, match="explicit destination"):
        _call(suffix, arguments[:-1])


@pytest.mark.parametrize("suffix", _COLUMN_RECIPES)
@pytest.mark.parametrize(
    "changes,diagnostic",
    [
        ({"dtype": DataType.FP16}, "FP32"),
        ({"blayout": ir.TileLayout.col_major}, "row-major"),
        ({"valid_shape": [0, 64]}, "nonempty static valid"),
        ({"valid_shape": [-1, -1]}, "nonempty static valid"),
    ],
)
def test_column_contract_rejects_unverified_matrix(suffix, changes, diagnostic):
    arguments = _args(suffix)
    arguments[0] = _buffer("src", **changes)
    with pytest.raises(ValueError, match=diagnostic):
        _call(suffix, arguments)


@pytest.mark.parametrize("suffix", _COLUMN_RECIPES)
@pytest.mark.parametrize("changes", [{"shape": [2, 64]}, {"shape": [1, 32]}, {"valid_shape": [1, 57]}])
def test_column_carrier_preserves_valid_and_physical_columns(suffix, changes):
    arguments = _args(suffix)
    options: dict[str, Any] = {"shape": [1, 64], **changes}
    arguments[1] = _buffer("row", **options)
    with pytest.raises(ValueError, match="single row preserving physical and valid columns"):
        _call(suffix, arguments)


@pytest.mark.parametrize("suffix", _COLUMN_REDUCTIONS)
def test_column_reduction_rejects_source_overlap(suffix):
    assert _verify_storage(suffix, (0, 4096)) == []
    diagnostics = _verify_storage(suffix, (0, 0))
    assert diagnostics and any("disjoint" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("suffix", _COLUMN_BROADCASTS)
def test_column_broadcast_can_reuse_matrix_not_row(suffix):
    assert _verify_storage(suffix, (0, 4096, 0)) == []
    diagnostics = _verify_storage(suffix, (0, 4096, 4096))
    assert diagnostics and any("disjoint" in diagnostic.message for diagnostic in diagnostics)


@pytest.mark.parametrize("suffix", _COLUMN_RECIPES)
def test_column_recipes_reject_inherited_wide_pitch(suffix):
    addresses = (0, 16384) if suffix in _COLUMN_REDUCTIONS else (0, 16384, 32768)
    diagnostics = _verify_storage(suffix, addresses, source_window=True)
    assert diagnostics and any("dense physical row pitch" in item.message for item in diagnostics)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
