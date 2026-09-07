# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the BlockMxScaleTensorViews pass."""

from collections.abc import Sequence

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import passes

_PREFIX = [
    passes.inline_functions,
    passes.unroll_loops,
    passes.ctrl_flow_transform,
    passes.convert_to_ssa,
    passes.simplify,
    passes.normalize_stmt_structure,
    passes.flatten_call_expr,
    passes.outline_hierarchy_scopes,
    passes.outline_incore_scopes,
    passes.outline_cluster_scopes,
    passes.convert_tensor_to_tile_ops,
    passes.optimize_orch_tensors,
    passes.lower_composite_ops,
    passes.flatten_tile_nd_to_2d,
    passes.block_nz_tensor_views,
    passes.block_mx_scale_tensor_views,
]


def _run(program: ir.Program) -> ir.Program:
    """Run the default pipeline prefix through BlockMxScaleTensorViews."""
    for factory in _PREFIX:
        program = factory()(program)
    return program


def _const(expr: ir.Expr) -> int:
    assert isinstance(expr, ir.ConstInt), f"expected ConstInt, got {type(expr).__name__}"
    return expr.value


def _values(exprs: Sequence[ir.Expr]) -> list[int]:
    return [_const(expr) for expr in exprs]


def _elements(expr: ir.Expr) -> Sequence[ir.Expr]:
    assert isinstance(expr, ir.MakeTuple), f"expected MakeTuple, got {type(expr).__name__}"
    return expr.elements


def _calls_named(program: ir.Program, op_name: str) -> list[ir.Call]:
    registered_name = ir.get_op(op_name).name

    class Collector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op):
            if op.op.name == registered_name:
                self.calls.append(op)
            super().visit_call(op)

    collector = Collector()
    collector.visit_program(program)
    return collector.calls


def _function_calls_named(program: ir.Program, function_name: str) -> list[ir.Call]:
    class Collector(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op):
            if isinstance(op.op, ir.GlobalVar) and op.op.name == function_name:
                self.calls.append(op)
            super().visit_call(op)

    collector = Collector()
    collector.visit_program(program)
    return collector.calls


def _mx_loads(program: ir.Program) -> list[ir.Call]:
    loads = []
    for call in _calls_named(program, "tile.load"):
        view = getattr(call.args[0].type, "tensor_view", None)
        if view is not None and view.layout in {
            ir.TensorLayout.MX_A_ZZ,
            ir.TensorLayout.MX_B_NN,
        }:
            loads.append(call)
    return loads


def _layout_param(program: ir.Program, layout: ir.TensorLayout) -> ir.TensorType:
    for function in program.functions.values():
        for param in function.params:
            param_type = param.type
            if not isinstance(param_type, ir.TensorType):
                continue
            view = param_type.tensor_view
            if view is not None and view.layout == layout:
                return param_type
    raise AssertionError(f"no {layout}-annotated param found")


def test_blocks_mx_a_and_b_shapes_and_load_windows():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a_s: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ],
            b_s: pl.Tensor[[4, 64], pl.FP8E8M0, pl.MX_B_NN],
        ):
            _a_tile = pl.load(a_s, [16, 2], [16, 2], target_memory=pl.Mem.Mat)
            _b_tile = pl.load(b_s, [2, 16], [2, 16], target_memory=pl.Mem.Mat)

    result = _run(Input)
    expected_tensor_shape = [1, 4, 2, 16, 2]
    assert _values(_layout_param(result, ir.TensorLayout.MX_A_ZZ).shape) == expected_tensor_shape
    assert _values(_layout_param(result, ir.TensorLayout.MX_B_NN).shape) == expected_tensor_shape
    for load in _mx_loads(result):
        assert _values(_elements(load.args[1])) == [0, 1, 1, 0, 0]
        assert _values(_elements(load.args[2])) == [1, 1, 1, 16, 2]


def test_remaps_dump_vars_when_blocking_mx_call_argument():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(self, scale: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]):
            _ = pl.load(scale, [0, 0], [16, 2], target_memory=pl.Mem.Mat)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, scale: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]):
            pl.dump_tag(scale)
            self.kernel(scale)

    before_mx = Input
    for factory in _PREFIX[:-1]:
        before_mx = factory()(before_mx)
    before_main = before_mx.get_function("main")
    assert before_main is not None
    old_scale_id = before_main.params[0].unique_id

    result = passes.block_mx_scale_tensor_views()(before_mx)
    after_main = result.get_function("main")
    assert after_main is not None
    new_scale = after_main.params[0]
    assert new_scale.unique_id != old_scale_id

    (call,) = _function_calls_named(result, "kernel")
    assert isinstance(call.args[0], ir.Var)
    assert call.args[0].unique_id == new_scale.unique_id
    assert "dump_vars" in call.attrs
    dump_vars = call.attrs["dump_vars"]
    assert len(dump_vars) == 1
    assert dump_vars[0].unique_id == new_scale.unique_id

    structural = passes.IRPropertySet()
    structural.insert(passes.IRProperty.SSAForm)
    structural.insert(passes.IRProperty.UseAfterDef)
    passes.run_verifier(structural)(result)


def test_narrowed_valid_shape_keeps_complete_physical_window():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, a_s: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]):
            _ = pl.load(a_s, [0, 0], [32, 2], valid_shape=[16, 2], target_memory=pl.Mem.Mat)

    (load,) = _mx_loads(_run(Input))
    assert _values(_elements(load.args[2])) == [1, 2, 1, 16, 2]
    assert _values(_elements(load.args[3])) == [1, 2, 1, 16, 2]
    assert isinstance(load.type, ir.TileType)
    assert load.type.tile_view is not None
    assert _values(load.type.tile_view.valid_shape) == [16, 2]


def test_rejects_unprovable_offset():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            a_s: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ],
            row_offset: pl.Scalar[pl.INDEX],
        ):
            _ = pl.load(a_s, [row_offset, 0], [16, 2], target_memory=pl.Mem.Mat)

    with pytest.raises(ValueError, match=r"cannot be proven|multiple of"):
        _run(Input)


def test_proves_floor_divided_loop_offset():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, a_s: pl.Tensor[[16, 64], pl.FP8E8M0, pl.MX_A_ZZ]):
            for k0 in pl.pipeline(256, 2048, 256, stage=1):
                group_offset = k0 // 32
                _ = pl.load(a_s, [0, group_offset], [16, 8], target_memory=pl.Mem.Mat)

    (load,) = _mx_loads(_run(Input))
    offsets = _elements(load.args[1])
    assert _const(offsets[0]) == 0
    assert _const(offsets[3]) == 0
    assert _const(offsets[4]) == 0
    assert _values(_elements(load.args[2])) == [1, 1, 4, 16, 2]


def test_proves_offset_through_outlined_scalar_params():
    @pl.jit
    def outlined(a_s: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ]):
        for mt in pl.parallel(2):
            row = mt * 16
            for block in pl.spmd(1, name_hint="mx_offset"):
                _ = pl.load(a_s, [row + block * 16, 0], [16, 2], target_memory=pl.Mem.Mat)

    _, _, tensor_map, scalar_values, scalar_dtypes, dynamic_symbols = outlined._bind_args_from_signature({})
    program = outlined._compile_to_program(
        tensor_map,
        scalar_values,
        scalar_dtypes,
        dynamic_symbols,
        pl,
    )
    (load,) = _mx_loads(_run(program))
    block_offset = _elements(load.args[1])[1]
    assert isinstance(block_offset, ir.FloorDiv)
    assert _const(block_offset.right) == 16


def test_rejects_param_offset_if_any_caller_is_unaligned():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a_s: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ],
            row: pl.Scalar[pl.INDEX],
        ):
            _ = pl.load(a_s, [row, 0], [16, 2], target_memory=pl.Mem.Mat)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, a_s: pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ]):
            self.kernel(a_s, 0)
            self.kernel(a_s, 1)

    with pytest.raises(ValueError, match=r"cannot be proven|multiple of"):
        _run(Input)


def test_is_idempotent_via_function_stamp():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, a_s: pl.Tensor[[64, 4], pl.FP8E8M0, pl.MX_A_ZZ]):
            _ = pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat)

    once = passes.block_mx_scale_tensor_views()(Input)
    assert all(function.attrs["mx_tensor_views_blocked"] is True for function in once.functions.values())
    assert passes.block_mx_scale_tensor_views()(once) is once


def test_shaped_nd_mx_backing_aliases_are_rewritten():
    @pl.program
    class MxToNd:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, source: pl.Tensor[[32, 4], pl.FP8E8M0, pl.MX_A_ZZ]) -> pl.Tensor[[1, 128], pl.FP8E8M0]:
            return pl.tensor.view(source, [1, 128], layout=pl.ND)

    mx_to_nd = _run(MxToNd)
    assert _values(_layout_param(mx_to_nd, ir.TensorLayout.MX_A_ZZ).shape) == [1, 2, 2, 16, 2]
    (to_nd,) = _calls_named(mx_to_nd, "tensor.view")
    assert isinstance(to_nd.type, ir.TensorType)
    assert _values(to_nd.type.shape) == [1, 128]

    @pl.program
    class NdToMx:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, source: pl.Tensor[[1, 128], pl.FP8E8M0]) -> pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN]:
            return pl.tensor.view(source, [4, 32], layout=pl.MX_B_NN)

    nd_to_mx = _run(NdToMx)
    (to_mx,) = _calls_named(nd_to_mx, "tensor.view")
    assert isinstance(to_mx.type, ir.TensorType)
    assert _values(to_mx.type.shape) == [1, 2, 2, 16, 2]
    assert _values(_elements(to_mx.args[1])) == [1, 2, 2, 16, 2]


def test_submit_return_type_is_blocked_and_launch_fields_survive():
    span = ir.Span.unknown()
    mx_type = ir.TensorType(
        [ir.ConstInt(32, pl.INDEX, span), ir.ConstInt(4, pl.INDEX, span)],
        pl.FP8E8M0,
        tensor_view=ir.TensorView([], ir.TensorLayout.MX_A_ZZ),
    )
    submit_type = ir.TupleType([mx_type, ir.ScalarType(pl.TASK_ID)])
    dep = ir.Var("dep", ir.ScalarType(pl.TASK_ID), span)
    core_num = ir.ConstInt(4, pl.INDEX, span)
    submit = ir.Submit(
        ir.GlobalVar("kernel"),
        [],
        [dep],
        {"launch_tag": 3},
        {"test_marker": 7},
        submit_type,
        span,
        core_num=core_num,
        sync_start=True,
        allow_early_resolve=True,
    )
    result_var = ir.Var("result", submit_type, span)
    function = ir.Function("caller", [dep], [], ir.AssignStmt(result_var, submit, span), span)

    # The synthetic Submit intentionally carries opaque kwargs that the DSL
    # printer cannot represent, so inspect the pass result without roundtrip
    # instrumentation.
    with passes.PassContext([]):
        result = passes.block_mx_scale_tensor_views()(ir.Program([function], "submit_blocking", span))
    after_assign = next(iter(result.functions.values())).body
    assert isinstance(after_assign, ir.AssignStmt)
    after_submit = after_assign.value
    assert isinstance(after_submit, ir.Submit)
    assert isinstance(after_submit.type, ir.TupleType)
    ir.assert_structural_equal(after_assign.var.type, after_submit.type)
    blocked_mx, task_id = after_submit.type.types
    assert isinstance(blocked_mx, ir.TensorType)
    assert _values(blocked_mx.shape) == [1, 2, 2, 16, 2]
    assert isinstance(task_id, ir.ScalarType) and task_id.dtype == pl.TASK_ID
    assert after_submit.kwargs == {"launch_tag": 3}
    assert after_submit.attrs == {"test_marker": 7}
    assert after_submit.deps == [dep]
    assert after_submit.core_num is core_num
    assert after_submit.sync_start and after_submit.allow_early_resolve


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
