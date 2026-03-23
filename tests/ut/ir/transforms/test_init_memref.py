# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for InitMemRefPass."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir import MemorySpace


def _iter_assign_stmts(func):
    """Iterate all AssignStmt in function body."""

    def _visit(stmt):
        if isinstance(stmt, ir.AssignStmt):
            yield stmt
        elif isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                yield from _visit(child)
        elif isinstance(stmt, ir.ForStmt):
            yield from _visit(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            yield from _visit(stmt.then_body)
            if stmt.else_body is not None:
                yield from _visit(stmt.else_body)
        elif isinstance(stmt, ir.WhileStmt):
            yield from _visit(stmt.body)

    yield from _visit(func.body)


def _get_tile_types(func):
    """Get {var_name: tile_type} for all TileType variables with memrefs."""
    result = {}
    for stmt in _iter_assign_stmts(func):
        if isinstance(stmt.var.type, ir.TileType) and stmt.var.type.memref is not None:
            result[stmt.var.name_hint] = stmt.var.type
    return result


def _get_param_types(func):
    """Get {param_name: tensor_type} for all TensorType params with memrefs."""
    result = {}
    for param in func.params:
        if isinstance(param.type, ir.TensorType) and param.type.memref is not None:
            result[param.name_hint] = param.type
    return result


def _first_function(program):
    """Get the first function from a Program."""
    return next(iter(program.functions.values()))


def _is_tile_alloc_assign(stmt):
    """Return True if stmt is an AssignStmt wrapping a tile.alloc call."""
    return (
        isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == "tile.alloc"
    )


def _assert_leading_allocs(func, count):
    """Assert that the first count statements in the body are tile.alloc assigns."""
    assert isinstance(func.body, ir.SeqStmts)
    assert len(func.body.stmts) >= count
    assert all(_is_tile_alloc_assign(stmt) for stmt in func.body.stmts[:count])


def _get_alloc_stmts(func):
    """Get tile.alloc AssignStmts from function body."""
    allocs = []
    for stmt in _iter_assign_stmts(func):
        if _is_tile_alloc_assign(stmt):
            allocs.append(stmt)
    return allocs


def _find_yield_stmt(stmt):
    """Recursively find YieldStmt in a statement tree."""
    if isinstance(stmt, ir.YieldStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for child in stmt.stmts:
            result = _find_yield_stmt(child)
            if result:
                return result
    return None


class TestBasic:
    """Basic MemRef creation, memory space assignment, and alloc generation."""

    def test_simple_load_add_store(self):
        """load-add-store sequence: Vec tiles get MemRef, params get DDR, allocs prepended."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                input_b: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_b, [0, 0], [64, 64])
                tile_sum: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(tile_a, tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_sum, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        _assert_leading_allocs(func, 3)

        # Params: DDR, addr=-1, size=16384
        param_types = _get_param_types(func)
        for name in ("input_a", "input_b", "output"):
            assert name in param_types, f"param {name} should have MemRef"
            assert param_types[name].memory_space == MemorySpace.DDR
            assert param_types[name].memref.addr_.value == -1
            assert param_types[name].memref.size_ == 16384  # 64*64*4

        # Tiles: Vec, addr=-1, size=16384
        tile_types = _get_tile_types(func)
        for name in ("tile_a", "tile_b", "tile_sum"):
            assert name in tile_types, f"tile {name} should have MemRef"
            assert tile_types[name].memory_space == MemorySpace.Vec
            assert tile_types[name].memref.addr_.value == -1
            assert tile_types[name].memref.size_ == 16384

        # Alloc count matches non-DDR tiles
        allocs = _get_alloc_stmts(func)
        assert len(allocs) == 3
        for alloc in allocs:
            assert alloc.value.args[1].value == -1

    def test_matmul_pipeline(self):
        """load→move→matmul→store: Vec/Mat/Left/Right/Acc memory spaces."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16],
                input_b: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a_ub: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
                    input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Left] = pl.move(
                    tile_a_ub, target_memory=pl.MemorySpace.Left
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Right] = pl.move(
                    tile_b_l1, target_memory=pl.MemorySpace.Right
                )
                tile_result: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(
                    tile_a_l0a, tile_b_l0b
                )
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(tile_result, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        _assert_leading_allocs(func, 5)

        # Params: DDR
        param_types = _get_param_types(func)
        for name in ("input_a", "input_b", "output"):
            assert param_types[name].memory_space == MemorySpace.DDR

        # Tiles: correct memory spaces and sizes
        tile_types = _get_tile_types(func)
        expected_spaces = {
            "tile_a_ub": MemorySpace.Vec,
            "tile_b_l1": MemorySpace.Mat,
            "tile_a_l0a": MemorySpace.Left,
            "tile_b_l0b": MemorySpace.Right,
            "tile_result": MemorySpace.Acc,
        }
        for name, expected in expected_spaces.items():
            assert name in tile_types, f"tile {name} should have MemRef"
            assert tile_types[name].memory_space == expected, (
                f"{name}: expected {expected}, got {tile_types[name].memory_space}"
            )
            assert tile_types[name].memref.addr_.value == -1
            if name == "tile_result":
                assert tile_types[name].memref.size_ == 4096  # 32*32*4
            else:
                assert tile_types[name].memref.size_ == 2048  # 32*32*2


class TestMemRefSharing:
    """MemRef sharing: tile.store shares with output param, view ops share with input."""

    def test_store_shares_memref_with_output_param(self):
        """tile.store result shares MemRef with the output tensor parameter."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_a, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        param_types = _get_param_types(func)
        result_memrefs = {}
        for stmt in _iter_assign_stmts(func):
            if isinstance(stmt.var.type, ir.TensorType) and stmt.var.type.memref is not None:
                result_memrefs[stmt.var.name_hint] = stmt.var.type.memref

        assert "result" in result_memrefs
        assert result_memrefs["result"] is param_types["output"].memref

    def test_view_op_shares_memref_with_input(self):
        """tile.reshape output shares MemRef with its input tile."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [64, 64])
                reshaped: pl.Tile[[4096, 1], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(tile_a, [4096, 1])
                flat: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.tile.reshape(reshaped, [64, 64])
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(flat, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        tile_types = _get_tile_types(func)
        assert "tile_a" in tile_types
        assert "reshaped" in tile_types
        assert "flat" in tile_types

        # All three should share the same MemRef (view ops share with input)
        assert tile_types["tile_a"].shares_memref_with(tile_types["reshaped"])
        assert tile_types["reshaped"].shares_memref_with(tile_types["flat"])

        # Only 1 alloc needed (all share same MemRef)
        allocs = _get_alloc_stmts(func)
        assert len(allocs) == 1

    def test_matmul_acc_shares_memref_with_accumulator(self):
        """tile.matmul_acc output shares MemRef with its accumulator input (arg[0])."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_a: pl.Tensor[[32, 32], pl.FP16],
                input_b: pl.Tensor[[32, 32], pl.FP16],
                output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                tile_a_ub: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Vec] = pl.load(input_a, [0, 0], [32, 32])
                tile_b_l1: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Mat] = pl.load(
                    input_b, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat
                )
                tile_a_l0a: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Left] = pl.move(
                    tile_a_ub, target_memory=pl.MemorySpace.Left
                )
                tile_b_l0b: pl.Tile[[32, 32], pl.FP16, pl.MemorySpace.Right] = pl.move(
                    tile_b_l1, target_memory=pl.MemorySpace.Right
                )
                acc: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul(tile_a_l0a, tile_b_l0b)
                acc_next: pl.Tile[[32, 32], pl.FP32, pl.MemorySpace.Acc] = pl.matmul_acc(
                    acc, tile_a_l0a, tile_b_l0b
                )
                result: pl.Tensor[[32, 32], pl.FP32] = pl.store(acc_next, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        tile_types = _get_tile_types(func)
        assert "acc" in tile_types
        assert "acc_next" in tile_types

        # matmul_acc output shares MemRef with its accumulator input
        assert tile_types["acc"].shares_memref_with(tile_types["acc_next"])

        # Only 1 Acc alloc needed (not 2)
        acc_allocs = [a for a in _get_alloc_stmts(func) if a.value.args[0].value == MemorySpace.Acc.value]
        assert len(acc_allocs) == 1


class TestYieldMemRef:
    """MemRef propagation through yield in ForStmt and IfStmt."""

    def test_for_loop_carry_memref_relationships(self):
        """ForStmt: initValue/iter_arg share MemRef (Group A), yield/return_var share MemRef (Group B).

        Group A and B may have different MemRefs — the yield-to-iter_arg mismatch
        is resolved later by MemoryReuse.
        """

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                init_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                other_tile: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                    input_tensor, [0, 0], [64, 64]
                )
                for _i, (acc,) in pl.range(0, 4, init_values=(init_tile,)):
                    acc_next: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.add(acc, other_tile)
                    acc_out = pl.yield_(acc_next)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(acc_out, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        loop = cast(
            ir.ForStmt,
            next(stmt for stmt in cast(ir.SeqStmts, func.body).stmts if isinstance(stmt, ir.ForStmt)),
        )
        iter_arg = loop.iter_args[0]
        return_var = loop.return_vars[0]
        yield_stmt = _find_yield_stmt(loop.body)
        assert yield_stmt is not None
        yield_var = yield_stmt.value[0]

        init_value = iter_arg.initValue
        assert isinstance(init_value, ir.Var)

        # All 4 must have TileType with MemRef in Vec space
        for name, var_type in [
            ("initValue", init_value.type),
            ("iter_arg", iter_arg.type),
            ("yield_value", yield_var.type),
            ("return_var", return_var.type),
        ]:
            assert isinstance(var_type, ir.TileType), f"{name} should be TileType"
            assert var_type.memref is not None, f"{name} should have MemRef"
            assert var_type.memory_space == MemorySpace.Vec, f"{name} should be Vec"

        init_tile_t = cast(ir.TileType, init_value.type)
        iter_arg_t = cast(ir.TileType, iter_arg.type)
        yield_t = cast(ir.TileType, yield_var.type)
        return_t = cast(ir.TileType, return_var.type)

        # Group A: initValue and iter_arg share MemRef
        assert init_tile_t.shares_memref_with(iter_arg_t)
        # Group B: yield value and return_var share MemRef
        assert yield_t.shares_memref_with(return_t)
        # Groups A and B have different MemRefs
        assert not yield_t.shares_memref_with(iter_arg_t)

    def test_if_yield_return_var_shares_memref(self):
        """IfStmt: return_var shares MemRef with the then-branch yield value."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                input_tensor: pl.Tensor[[64, 64], pl.FP32],
                cond: pl.Scalar[pl.INDEX],
                output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                if cond < 2:
                    tile_a: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_a)
                else:
                    tile_b: pl.Tile[[64, 64], pl.FP32, pl.MemorySpace.Vec] = pl.load(
                        input_tensor, [0, 0], [64, 64]
                    )
                    if_result = pl.yield_(tile_b)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.store(if_result, [0, 0], output)
                return result

        After = passes.init_mem_ref()(Before)
        func = _first_function(After)

        if_stmt = next(stmt for stmt in cast(ir.SeqStmts, func.body).stmts if isinstance(stmt, ir.IfStmt))
        assert len(if_stmt.return_vars) == 1
        rv = if_stmt.return_vars[0]

        # return_var must have TileType with MemRef in Vec
        assert isinstance(rv.type, ir.TileType)
        assert rv.type.memref is not None
        assert rv.type.memory_space == MemorySpace.Vec

        # return_var shares MemRef with then-branch yield value
        then_yield = _find_yield_stmt(if_stmt.then_body)
        assert then_yield is not None
        then_var = then_yield.value[0]
        assert isinstance(then_var.type, ir.TileType)
        assert cast(ir.TileType, rv.type).shares_memref_with(cast(ir.TileType, then_var.type))


class TestEdgeCases:
    """Edge cases requiring raw IR construction."""

    def test_rejects_dynamic_tile_shape(self):
        """InitMemRef must fail fast when allocation shape is still dynamic."""
        span = ir.Span.unknown()

        dynamic_len = ir.Var("dynamic_len", ir.ScalarType(ir.DataType.INDEX), span)
        dynamic_tile_type = ir.TileType(
            [ir.ConstInt(1, ir.DataType.INDEX, span), dynamic_len],
            ir.DataType.FP32,
            memory_space=MemorySpace.Vec,
        )
        dynamic_tile = ir.Var("dynamic_tile", dynamic_tile_type, span)

        tpop_call = ir.Call(ir.Op("tile.tpop_from_aic"), [], {"aiv_idx": 0}, dynamic_tile_type, span)
        body = ir.SeqStmts(
            [ir.AssignStmt(dynamic_tile, tpop_call, span), ir.ReturnStmt([dynamic_tile], span)], span
        )
        func = ir.Function("test_func", [], [dynamic_tile_type], body, span)
        program = ir.Program([func], "test_program", span)

        with pytest.raises(Exception, match="InitMemRef requires static shape"):
            passes.init_mem_ref()(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
