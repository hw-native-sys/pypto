# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dual-AIV replay assignments retain the substituted value's type."""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _verify_assignments(program):
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.AssignTypeSymmetry)
    diagnostics = passes.PropertyVerifierRegistry.verify(properties, program)
    assert not diagnostics, "\n".join(d.message for d in diagnostics)


def _lanes(program):
    function = program.get_function("main_aiv")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    branch = function.body.stmts[-1]
    assert isinstance(branch, ir.IfStmt)
    assert isinstance(branch.then_body, ir.SeqStmts)
    assert isinstance(branch.else_body, ir.SeqStmts)
    return branch.then_body, branch.else_body


def _assert_empty_tile(type_):
    assert isinstance(type_, ir.TileType)
    assert type_.tile_view is not None
    assert len(type_.tile_view.valid_shape) == 2
    for dim in type_.tile_view.valid_shape:
        assert isinstance(dim, ir.ConstInt)
        assert dim.value == 0


def test_replay_tile_alias_chain_inherits_zero_valid_shape():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
        def main_aiv(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
            slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
            pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
            tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
            alias = tile
            alias2 = alias
            result = pl.mul(alias2, 2.0)
            pl.tfree_to_aic(tile)
            updated = pl.store(result, [0, 0], out)
            return updated

    before = passes.convert_to_ssa()(Before)
    _verify_assignments(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    _, replay = _lanes(after)
    aliases = [s for s in replay.stmts if isinstance(s, ir.AssignStmt) and "alias" in s.var.name_hint]
    assert len(aliases) == 2
    for assign in aliases:
        _assert_empty_tile(assign.var.type)
        ir.assert_structural_equal(assign.var.type, assign.value.type)


def test_store_bypass_uses_destination_shape_symbols():
    before = pl.parse_program("""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV, attrs={"dual_aiv_dispatch": True})
    def main_aiv(
        self,
        outer_n: pl.Scalar[pl.INDEX],
        inner_n: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[outer_n, 16], pl.FP32]],
    ) -> pl.Tensor[[outer_n, 16], pl.FP32]:
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        pl.tfree_to_aic(tile)
        updated = pl.store(tile, [0, 0], out)
        return updated
""")
    function = before.get_function("main_aiv")
    assert function is not None
    assert isinstance(function.body, ir.SeqStmts)
    stmts = list(function.body.stmts)
    assign = stmts[-2]
    assert isinstance(assign, ir.AssignStmt)
    call = assign.value
    assert isinstance(call, ir.Call)
    # Extracted kernels can retain a callee-local symbol on an explicitly typed
    # store result while the destination still carries its caller-local symbol.
    destination_type = function.params[2].type
    assert isinstance(destination_type, ir.TensorType)
    store_type = ir.TensorType([function.params[1], destination_type.shape[1]], pl.FP32)
    value = ir.Call(call.op, call.args, call.kwargs, store_type, call.span)
    result = ir.Var(assign.var.name_hint, store_type, assign.var.span)
    stmts[-2] = ir.AssignStmt(result, value, assign.span)
    stmts[-1] = ir.ReturnStmt([result], function.span)
    function = ir.Function(
        function.name,
        list(zip(function.params, function.param_directions)),
        [store_type],
        ir.SeqStmts(stmts, function.span),
        function.span,
        type=function.func_type,
        attrs=function.attrs,
    )
    before = ir.Program([function], "StoreSymbols", function.span)
    _verify_assignments(before)
    # This explicit intermediate call type is not expressible by the DSL
    # printer: parsing pl.store would infer the destination type instead.
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    function = after.get_function("main_aiv")
    assert function is not None
    _, replay = _lanes(after)
    alias = replay.stmts[-2]
    assert isinstance(alias, ir.AssignStmt)
    assert alias.value is function.params[2]
    ir.assert_structural_equal(alias.var.type, function.params[2].type)


@pytest.mark.parametrize("op", ["fmods", "rems"])
def test_scalar_remainder_replay_materializes_empty_tile(op):
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def main_aiv(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        pl.func_attr({{"dual_aiv_dispatch": True}})
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        tmp = pl.tile.create([16, 16], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        remainder = pl.tile.{op}(tile, 16.0{", tmp" if op == "rems" else ""})
        pl.tfree_to_aic(tile)
        updated = pl.store(remainder, [0, 0], out)
        return updated
""")
    before = passes.convert_to_ssa()(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    lane0, lane1 = _lanes(after)
    for body, expected_op in [(lane0, f"tile.{op}"), (lane1, "tile.create")]:
        remainder = next(
            s for s in body.stmts if isinstance(s, ir.AssignStmt) and "remainder" in s.var.name_hint
        )
        assert isinstance(remainder.value, ir.Call)
        assert remainder.value.op.name == expected_op
        if expected_op == "tile.create":
            _assert_empty_tile(remainder.var.type)


@pytest.mark.parametrize("nested", [False, True])
def test_replay_if_result_inherits_yielded_tile_type(nested):
    inner = "alias = tile"
    if nested:
        inner = (
            "if flag:\n                alias = tile"
            "\n            else:\n                alias = pl.mul(tile, 2.0)"
        )
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def main_aiv(self, flag: pl.Scalar[pl.BOOL],
                 out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
        pl.func_attr({{"dual_aiv_dispatch": True}})
        slot = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x1000)
        pl.aiv_initialize_pipe(dir_mask=1, slot_size=512, c2v_consumer_buf=slot)
        tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        if flag:
            {inner}
        else:
            alias = pl.mul(tile, 3.0)
        result = pl.mul(alias, 4.0)
        pl.tfree_to_aic(tile)
        updated = pl.store(result, [0, 0], out)
        return updated
""")
    before = passes.convert_to_ssa()(before)
    after = passes.split_vector_kernel()(before)
    _verify_assignments(after)
    _, replay = _lanes(after)
    branch = next(stmt for stmt in replay.stmts if isinstance(stmt, ir.IfStmt))
    assert len(branch.return_vars) == 1
    merged = branch.return_vars[0]
    _assert_empty_tile(merged.type)
    for body in [branch.then_body, branch.else_body]:
        assert isinstance(body, ir.SeqStmts)
        yielded = body.stmts[-1]
        assert isinstance(yielded, ir.YieldStmt)
        ir.assert_structural_equal(merged.type, yielded.value[0].type)
    result = next(
        stmt
        for stmt in replay.stmts
        if isinstance(stmt, ir.AssignStmt) and stmt.var.name_hint.startswith("result")
    )
    assert isinstance(result.value, ir.Call)
    assert result.value.args[0] is merged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
