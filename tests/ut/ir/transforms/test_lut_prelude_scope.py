# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Historical owner/control-flow cases now require direct native conversion.

The same loops, branches, aliases and zero-trip joins remain covered without
introducing any owner prelude or compiler-created lookup table.
"""

import re

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


def _emit_incore_mlir(program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend950)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    _assert_lut_owners(optimized)
    parts: list[str] = []
    for func in optimized.functions.values():
        if func.func_type in (pl.FunctionType.Orchestration, pl.FunctionType.Group):
            continue
        single = ir.Program([func], func.name, optimized.span)
        result = codegen.PTOCodegen().generate(single, emit_tile_addr=True)
        parts.append(result if isinstance(result, str) else "".join(result.values()))
    return "\n".join(parts)


def _assert_lut_owners(program) -> list[int]:
    """Every former LUT owner is now free of LUT/index/prelude operations."""
    text = ir.python_print(program)
    assert "tile.gather" not in text and "fp4_lut__table" not in text
    assert "__FREE_VAR" not in text
    owners = []

    class Owners(ir.IRVisitor):
        def visit_function(self, func):
            if func.func_type in (pl.FunctionType.InCore, pl.FunctionType.AIC, pl.FunctionType.AIV):
                owners.append(0)
            super().visit_function(func)

        def visit_in_core_scope_stmt(self, op):
            owners.append(0)
            super().visit_in_core_scope_stmt(op)

        def visit_call(self, op):
            if op.op.name == ir.get_op("tile.cast").name:
                assert isinstance(op.args[0].type, ir.TileType)
                if op.args[0].type.dtype == ir.DataType.FP4E2M1X2:
                    assert isinstance(op.type, ir.TileType)
                    assert op.type.dtype == ir.DataType.FP8E4M3FN
            super().visit_call(op)

    Owners().visit_program(program)
    return owners


@pl.program
class PeeledCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, b: pl.Tensor[[16, 32], pl.FP4E2M1X2], out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]]
    ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
        f8 = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
        return pl.store(f8, [0, 0], out)


@pl.program
class NonPeeledCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self, b: pl.Tensor[[16, 64], pl.FP4E2M1X2], out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]]
    ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
        # Non-peeled: the FIRST cast is inside the pipeline loop (no chunk0).
        # The native cast executes per iteration without a LUT prelude.
        for kb in pl.pipeline(0, 4, stage=1):
            k0 = kb * 16
            f8 = pl.cast(pl.load(b, [0, k0 // 2], [16, 32]), pl.FP8E4M3FN)
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(f8, [0, 0], out)
        return out_t


def test_lut_ci_not_inside_loop_peeled():
    mlir = _emit_incore_mlir(PeeledCast)
    assert "pto.tci" not in mlir and "pto.tgather" not in mlir
    assert mlir.count("pto.tcvt") == 1


def test_lut_ci_not_inside_loop_non_peeled():
    mlir = _emit_incore_mlir(NonPeeledCast)
    assert "pto.tci" not in mlir and "pto.tgather" not in mlir
    assert "scf.for" in mlir and "pto.tcvt" in mlir


@pl.program
class LoopThenTrailingCast:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        b1: pl.Tensor[[16, 64], pl.FP4E2M1X2],
        b2: pl.Tensor[[16, 32], pl.FP4E2M1X2],
        out1: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        out2: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
    ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
        for kb in pl.pipeline(0, 4, stage=1):
            k0 = kb * 8
            f8a = pl.cast(pl.load(b1, [0, k0 // 2], [16, 32]), pl.FP8E4M3FN)
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(f8a, [0, 0], out1)
        # The same owner table must dominate casts inside and after the loop.
        f8b = pl.cast(pl.load(b2, [0, 0], [16, 32]), pl.FP8E4M3FN)
        out_t = pl.store(f8b, [0, 0], out2)
        return out_t


def test_cast_after_loop_reuses_the_region_lut():
    mlir = _emit_incore_mlir(LoopThenTrailingCast)
    tables = set(re.findall(r"%(\S*fp4_lut__table\S*) = ", mlir))
    assert not tables
    assert mlir.count("pto.tcvt") >= 2


def test_if_else_branches_share_the_region_lut():
    """Both branches use the table defined at the execution-region entry."""

    @pl.program
    class IfElseCasts:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            s: pl.Tensor[[1, 1], pl.INT32],
            b1: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            b2: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            cond = pl.read(s, [0, 0])
            t1 = pl.load(b1, [0, 0], [16, 32])
            t2 = pl.load(b2, [0, 0], [16, 32])
            if cond > 0:
                f1 = pl.cast(t1, pl.FP8E4M3FN)
                out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(f1, [0, 0], out)
            else:
                f2 = pl.cast(t2, pl.FP8E4M3FN)
                out_t = pl.store(f2, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(IfElseCasts)
    finally:
        backend.reset_for_testing()
    _assert_lut_owners(after)


def test_cast_before_loop_and_nested_branch_share_the_region_lut():
    """A leading cast and nested branch casts share a dominating region table."""

    @pl.program
    class LoopIfCast:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b1: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            b2: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(b1, [0, 0], [16, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for kb in pl.pipeline(0, 4, stage=1):
                if kb == 0:
                    c1 = pl.cast(pl.load(b1, [0, 0], [16, 32]), pl.FP8E4M3FN)
                    out_t = pl.store(c1, [0, 0], out)
                c2 = pl.cast(pl.load(b2, [0, 0], [16, 32]), pl.FP8E4M3FN)
                out_t = pl.store(c2, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(LoopIfCast)
    finally:
        backend.reset_for_testing()
    _assert_lut_owners(after)


def test_outer_loop_cast_inner_loop_keeps_prelude_ownership():
    """A cast-free inner loop must not disturb its execution owner's prelude."""

    @pl.program
    class OuterCastInnerLoop:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(
                pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN), [0, 0], out
            )
            for k in pl.pipeline(0, 4, stage=1):
                f8 = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
                out_t = pl.store(f8, [0, 0], out)
                for j in pl.range(0, 2):
                    out_t = pl.store(f8, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(OuterCastInnerLoop)
    finally:
        backend.reset_for_testing()
    _assert_lut_owners(after)
    lines = ir.python_print(after).splitlines()
    assert not any("__FREE_VAR" in line for line in lines)
    cast_pos = next(i for i, line in enumerate(lines) if "tile.cast" in line)
    outer_loop = next(i for i, line in enumerate(lines) if "pipeline" in line or "for k" in line)
    assert cast_pos < outer_loop, "The original leading cast must retain its position"


def test_nested_loop_first_cast_in_inner_loop_hoists_above_outer_loop():
    """A first cast in a nested loop still constructs its table outside both loops."""

    @pl.program
    class NestedInnerCast:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            x: pl.Tensor[[16, 64], pl.FP8E4M3FN],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            out_t: pl.Tensor[[16, 64], pl.FP8E4M3FN] = pl.store(pl.load(x, [0, 0], [16, 64]), [0, 0], out)
            for k in pl.pipeline(0, 4, stage=1):
                for j in pl.range(0, 2):
                    f8 = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
                    out_t = pl.store(f8, [0, 0], out)
            return out_t

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(NestedInnerCast)
    finally:
        backend.reset_for_testing()
    _assert_lut_owners(after)
    lines = ir.python_print(after).splitlines()
    assert not any("__FREE_VAR" in line for line in lines)
    table_defs = [
        i for i, line in enumerate(lines) if re.match(r"^\s*\w*fp4_lut__table\w*\s*:\s*pl\.Tile", line)
    ]
    assert not table_defs
    outer_loop = next((i for i, line in enumerate(lines) if "pipeline" in line or "for k" in line), -1)
    inner_loop = next((i for i, line in enumerate(lines) if "range" in line or "for j" in line), -1)
    assert outer_loop >= 0 and inner_loop > outer_loop, (
        f"could not locate the nested loops (outer@{outer_loop}, inner@{inner_loop})"
    )
    cast_pos = next(i for i, line in enumerate(lines) if "tile.cast" in line)
    assert cast_pos > inner_loop, "The inner cast must retain its original execution count"


def test_separate_core_scopes_have_independent_luts():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="first"):
                first = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
                out_first = pl.store(first, [0, 0], out)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="second"):
                second = pl.cast(pl.load(b, [0, 0], [16, 32]), pl.FP8E4M3FN)
                out_second = pl.store(second, [0, 0], out_first)
            return out_second

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(Before)
    finally:
        backend.reset_for_testing()
    assert _assert_lut_owners(after) == [0, 0]


def test_cast_free_region_has_no_lut():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.FP8E4M3FN],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            return pl.store(pl.load(x, [0, 0], [16, 64]), [0, 0], out)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(Before)
    finally:
        backend.reset_for_testing()
    ir.assert_structural_equal(after, Before)
    assert _assert_lut_owners(after) == [0]


def test_lut_result_type_propagates_through_alias_and_loop_carry():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            f8 = pl.cast(pl.load(b, [0, 0], [16, 32], valid_shape=[8, 24]), pl.FP8E4M3FN)
            for i in pl.range(2):
                alias = f8
                if i == 0:
                    f8 = pl.cast(pl.load(b, [0, 0], [16, 32], valid_shape=[8, 24]), pl.FP8E4M3FN)
                else:
                    f8 = alias
            return pl.store(f8, [0, 0], out)

    before = passes.convert_to_ssa()(Before)
    after = passes.legalize_tile_cast()(before)
    assert _assert_lut_owners(after) == [0]
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.SSAForm)
    properties.insert(passes.IRProperty.UseAfterDef)
    properties.insert(passes.IRProperty.AssignTypeSymmetry)
    properties.insert(passes.IRProperty.TypeChecked)
    passes.verify_properties(properties, after, "LUT loop carry")

    class CheckCarries(ir.IRVisitor):
        def visit_for_stmt(self, op):
            for carried in op.iter_args:
                ir.assert_structural_equal(carried.type, carried.initValue.type)
            super().visit_for_stmt(op)

    CheckCarries().visit_program(after)


@pytest.mark.parametrize("loop_kind", ["for", "while"])
@pytest.mark.parametrize("lut_in_then", [True, False])
def test_fp8_seed_and_lut_backedge_keep_the_loop_join_contract(loop_kind, lut_in_then):
    """The initializer remains the result on the zero-trip path."""

    @pl.program
    class BeforeFor:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            x: pl.Tile[
                [16, 64],
                pl.FP8E4M3FN,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[8, 48], stride=[64, 1]),
            ],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
            count: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            f8 = x
            for i in pl.range(count):
                alias = f8
                if i == 0:
                    f8 = pl.cast(pl.load(b, [0, 0], [16, 32], valid_shape=[8, 24]), pl.FP8E4M3FN)
                else:
                    f8 = alias
            return pl.store(f8, [0, 0], out)

    @pl.program
    class BeforeWhile:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            b: pl.Tensor[[16, 32], pl.FP4E2M1X2],
            x: pl.Tile[
                [16, 64],
                pl.FP8E4M3FN,
                pl.Mem.Vec,
                pl.TileView(valid_shape=[8, 48], stride=[64, 1]),
            ],
            out: pl.Out[pl.Tensor[[16, 64], pl.FP8E4M3FN]],
            count: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[16, 64], pl.FP8E4M3FN]:
            f8 = x
            i = 0
            while i < count:
                alias = f8
                if i == 0:
                    f8 = pl.cast(pl.load(b, [0, 0], [16, 32], valid_shape=[8, 24]), pl.FP8E4M3FN)
                else:
                    f8 = alias
                i = i + 1
            return pl.store(f8, [0, 0], out)

    before = passes.convert_to_ssa()(BeforeFor if loop_kind == "for" else BeforeWhile)
    if not lut_in_then:

        class SwapBranches(ir.IRMutator):
            def visit_if_stmt(self, op):
                assert op.else_body is not None
                return ir.IfStmt(op.condition, op.else_body, op.then_body, op.return_vars, op.span)

        before = SwapBranches().visit_program(before)

    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.SSAForm,
        passes.IRProperty.UseAfterDef,
        passes.IRProperty.AssignTypeSymmetry,
        passes.IRProperty.TypeChecked,
    ):
        properties.insert(prop)
    passes.verify_properties(properties, before, "before mixed FP8/LUT join")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(before)
    finally:
        backend.reset_for_testing()
    assert _assert_lut_owners(after) == [0]
    passes.verify_properties(properties, after, "after mixed FP8/LUT join")

    class CheckJoins(ir.IRVisitor):
        def check_loop(self, op):
            for carry, result in zip(op.iter_args, op.return_vars):
                ir.assert_structural_equal(carry.type, carry.initValue.type)
                ir.assert_structural_equal(result.type, carry.type)

        def visit_for_stmt(self, op):
            self.check_loop(op)
            super().visit_for_stmt(op)

        def visit_while_stmt(self, op):
            self.check_loop(op)
            super().visit_while_stmt(op)

    CheckJoins().visit_program(after)


def test_lut_type_propagates_through_tuples_nested_calls_and_native_hops():
    span = ir.Span.unknown()
    n = ir.Var("n", ir.ScalarType(ir.DataType.INDEX), span)
    two_n = ir.Mul(ir.ConstInt(2, ir.DataType.INDEX, span), n, ir.DataType.INDEX, span)
    source = ir.Var(
        "source", ir.TileType([ir.ConstInt(16, ir.DataType.INDEX, span), two_n], ir.DataType.FP4E2M1X2), span
    )
    cast = ir.create_op_call("tile.cast", [source], {"target_type": ir.DataType.FP8E4M3FN, "mode": 2}, span)
    cast = ir.Call(cast.op, cast.args, dict(cast.kwargs), {"test_tag": "lut"}, cast.type, span)
    f8 = ir.Var("f8", cast.type, span)
    alias = ir.Var("alias", f8.type, span)
    pair = ir.MakeTuple([alias, alias], span)
    pair_var = ir.Var("pair", pair.type, span)
    item = ir.TupleGetItemExpr(pair_var, 1, span)
    item_var = ir.Var("item", item.type, span)
    inner = ir.create_op_call("tile.neg", [item_var], {}, span)
    outer = ir.create_op_call("tile.neg", [inner], {}, span)
    outer = ir.Call(outer.op, outer.args, dict(outer.kwargs), {"anchor": alias}, outer.type, span)
    nested = ir.Var("nested", outer.type, span)
    # INT16 admits the conservative FP8 -> FP32 -> integer path: FP32's
    # 24-bit precision exceeds INT16's 15 value bits.
    to_int = ir.create_op_call("tile.cast", [nested], {"target_type": ir.DataType.INT16, "mode": 2}, span)
    to_int = ir.Call(to_int.op, to_int.args, dict(to_int.kwargs), {"test_tag": "native"}, to_int.type, span)
    result = ir.Var("result", to_int.type, span)
    cast_stmt = ir.attach_leading_comments(ir.AssignStmt(f8, cast, span), ["LUT result comment"])
    final_stmt = ir.attach_leading_comments(ir.AssignStmt(result, to_int, span), ["Native result comment"])
    body = ir.SeqStmts(
        [
            cast_stmt,
            ir.AssignStmt(alias, f8, span),
            ir.AssignStmt(pair_var, pair, span),
            ir.AssignStmt(item_var, item, span),
            ir.AssignStmt(nested, outer, span),
            final_stmt,
            ir.ReturnStmt([], span),
        ],
        span,
    )
    function = ir.Function("kernel", [n, source], [], body, span, ir.FunctionType.InCore)
    before = ir.Program([function], "nested_lut_types", span)
    properties = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.SSAForm,
        passes.IRProperty.UseAfterDef,
        passes.IRProperty.AssignTypeSymmetry,
        passes.IRProperty.TypeChecked,
    ):
        properties.insert(prop)
    passes.verify_properties(properties, before, "before nested LUT uses")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    try:
        after = passes.legalize_tile_cast()(before)
    finally:
        backend.reset_for_testing()
    passes.verify_properties(properties, after, "after nested LUT uses")
    assert _assert_lut_owners(after) == [0]

    class CollectAssignments(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.assignments = {}

        def visit_assign_stmt(self, op):
            self.assignments[op.var.name_hint] = op
            super().visit_assign_stmt(op)

    collector = CollectAssignments()
    collector.visit_program(after)
    assignments = collector.assignments
    assert assignments["f8"].leading_comments == ["LUT result comment"]
    assert assignments["result"].leading_comments == ["Native result comment"]
    assert assignments["f8"].value.attrs["test_tag"] == "lut"
    assert assignments["result"].value.attrs["test_tag"] == "native"
    assert assignments["nested"].value.attrs["anchor"].unique_id == assignments["alias"].var.unique_id
    inner_after = assignments["nested"].value.args[0]
    item_after = assignments["item"].var
    expected_inner = ir.create_op_call("tile.neg", [item_after], {}, span)
    ir.assert_structural_equal(inner_after.type, expected_inner.type)
    native_casts = []
    for stmt in assignments.values():
        if isinstance(stmt.value, ir.Call) and stmt.value.op.name == ir.get_op("tile.cast").name:
            source_type = stmt.value.args[0].type
            assert isinstance(source_type, ir.TileType)
            if source_type.dtype != ir.DataType.UINT8:
                native_casts.append(stmt)
    assert len(native_casts) >= 2, "The generic native chain must retain its intermediate definitions"
    assert assignments["result"].var.type.dtype == ir.DataType.INT16
    assert assignments["result"].value.args[0].type.dtype == ir.DataType.FP32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
