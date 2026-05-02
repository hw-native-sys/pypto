# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for LegalizePTOBufferReuse pass.

Verifies that the pass correctly splits MemRefs when multiple tile
variables sharing the same MemRef have incompatible root TileBufSignatures
(different shape/dtype/layout), while preserving legal sharing for
view-like operations (fillpad, reshape).

Test strategy:
- IR-level tests use the Before/Expected pattern with
  ``ir.assert_structural_equal(After, Expected)``.
  DefFields always auto-map, so ``enable_auto_mapping=True`` is unnecessary.
  This makes structural comparison sensitive to MemRef identity sharing:
  two tiles that share a MemRef in ``After`` must also share in ``Expected``.
- TestLegalizeWithCodegen retains the IRBuilder-based construction and MLIR
  string assertions, since those tests verify codegen output (alloc counts,
  addresses, dynamic-shape rendering) rather than IR shape.
"""

import math

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType
from pypto.pypto_core import DataType

_SPAN = ir.Span.unknown()
_IDX = DataType.INDEX
_FP32 = DataType.FP32
_FP16 = DataType.FP16


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure backend before each test."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


# ---------------------------------------------------------------------------
# Tests: identical signatures keep shared MemRef
# ---------------------------------------------------------------------------


class TestLegalSharingPreserved:
    """Same-shape same-dtype tiles sharing a MemRef should keep sharing."""

    def test_same_signature_keeps_shared(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                t1: pl.Tile[[32, 32], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                t2: pl.Tile[[32, 32], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = pl.tile.adds(
                    t1, 1.0
                )
                result: pl.Tensor[[32, 32], pl.FP32] = pl.tile.store(t2, [0, 0], b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[32, 32], pl.FP32],
                b: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
            ) -> pl.Tensor[[32, 32], pl.FP32]:
                t1: pl.Tile[[32, 32], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [32, 32], [32, 32], target_memory=pl.Mem.Vec, transpose=False
                )
                t2: pl.Tile[[32, 32], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = pl.tile.adds(
                    t1, 1.0
                )
                result: pl.Tensor[[32, 32], pl.FP32] = pl.tile.store(t2, [0, 0], b)
                return result

        After = passes.legalize_pto_buffer_reuse()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_fillpad_view_keeps_shared(self):
        """fillpad changes pad but keeps same shape -> legal view."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                loaded: pl.Tile[[128, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                padded: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef("mem_vec_0", -1, 65536),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(loaded, pad_value=pl.PadValue.max)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(padded, [0, 0], b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                loaded: pl.Tile[[128, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                padded: pl.Tile[
                    [128, 128],
                    pl.FP32,
                    pl.MemRef("mem_vec_0", -1, 65536),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(loaded, pad_value=pl.PadValue.max)
                result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(padded, [0, 0], b)
                return result

        After = passes.legalize_pto_buffer_reuse()(Before)
        ir.assert_structural_equal(After, Expected)


class TestAscend910BSplitLoadTpopHazard:
    """Ascend910B split AIV kernels should split load+tpop writer reuse."""

    def test_ascend910b_split_aiv_splits_load_plus_tpop_reuse(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, down: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                down_prev: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.load(down, [0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_1", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.tpop_from_aic(split=1)
                )
                down_next: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.add(down_prev, pipe_chunk)
                )
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(down_next, [0, 0], down)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, down: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                mem_vec_2: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 4096)
                down_prev: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.load(down, [0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_1", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.tpop_from_aic(split=1)
                )
                down_next: pl.Tile[[8, 128], pl.FP32, pl.MemRef(mem_vec_2, 0, 4096), pl.Mem.Vec] = (
                    pl.tile.add(down_prev, pipe_chunk)
                )
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(down_next, [0, 0], down)
                return result

        After = passes.legalize_pto_buffer_reuse()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_ascend950_keeps_compatible_share(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend950)

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, down: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                down_prev: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.load(down, [0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_1", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.tpop_from_aic(split=1)
                )
                down_next: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.add(down_prev, pipe_chunk)
                )
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(down_next, [0, 0], down)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
            def main(self, down: pl.InOut[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
                down_prev: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.load(down, [0, 0], [8, 128], [8, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                pipe_chunk: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_1", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.tpop_from_aic(split=1)
                )
                down_next: pl.Tile[[8, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 4096), pl.Mem.Vec] = (
                    pl.tile.add(down_prev, pipe_chunk)
                )
                result: pl.Tensor[[16, 128], pl.FP32] = pl.tile.store(down_next, [0, 0], down)
                return result

        After = passes.legalize_pto_buffer_reuse()(Before)
        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Tests: incompatible signatures cause split
# ---------------------------------------------------------------------------


class TestIllegalSharingSplit:
    """Tiles with incompatible root signatures should be split."""

    def test_different_shape_same_memref_splits(self):
        """Two writers with different shapes -> must split."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                t1: pl.Tile[[128, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(t2, [0, 0], b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
                t1: pl.Tile[[128, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_1, 0, 65536), pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                result: pl.Tensor[[128, 128], pl.FP32] = pl.tile.store(t2, [0, 0], b)
                return result

        After = passes.legalize_pto_buffer_reuse()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_split_propagates_through_view_chain(self):
        """A split writer's legal views should follow the new MemRef."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                t1: pl.Tile[[128, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                t3: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef("mem_vec_0", -1, 65536),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(t2, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(t3, [0, 0], b)
                return result

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                mem_vec_1: pl.Ptr = pl.tile.alloc(pl.Mem.Vec, 65536)
                t1: pl.Tile[[128, 128], pl.FP32, pl.MemRef("mem_vec_0", -1, 65536), pl.Mem.Vec] = (
                    pl.tile.load(a, [0, 0], [128, 128], [128, 128], target_memory=pl.Mem.Vec, transpose=False)
                )
                t2: pl.Tile[[64, 64], pl.FP32, pl.MemRef(mem_vec_1, 0, 65536), pl.Mem.Vec] = pl.tile.load(
                    a, [0, 0], [64, 64], [64, 64], target_memory=pl.Mem.Vec, transpose=False
                )
                t3: pl.Tile[
                    [64, 64],
                    pl.FP32,
                    pl.MemRef(mem_vec_1, 0, 65536),
                    pl.Mem.Vec,
                    pl.TileView(pad=pl.PadValue.max),
                ] = pl.tile.fillpad(t2, pad_value=pl.PadValue.max)
                result: pl.Tensor[[64, 64], pl.FP32] = pl.tile.store(t3, [0, 0], b)
                return result

        After = passes.legalize_pto_buffer_reuse()(Before)
        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# Integration tests: legalize + codegen
#
# These tests verify codegen output (alloc counts, addresses, dynamic-shape
# rendering in MLIR text), so they keep IRBuilder construction and MLIR
# string assertions rather than the structural-equality pattern. The
# pass-level IR shape is already covered by TestLegalSharingPreserved /
# TestIllegalSharingSplit above.
# ---------------------------------------------------------------------------


def _ci(val: int) -> ir.ConstInt:
    return ir.ConstInt(val, _IDX, _SPAN)


def _dtype_bytes(dtype: DataType) -> int:
    if dtype in (_FP32,):
        return 4
    if dtype in (_FP16,):
        return 2
    raise ValueError(f"Unsupported dtype: {dtype}")


class _MemRefAlloc:
    def __init__(self, start_id: int = 0) -> None:
        self._next_id = start_id

    def vec(self, shape: list[int], dtype: DataType) -> ir.MemRef:
        size = math.prod(shape) * _dtype_bytes(dtype)
        mr = ir.MemRef(ir.MemorySpace.Vec, _ci(-1), size, self._next_id)
        self._next_id += 1
        return mr


def _tile_t(
    shape: list[int], dtype: DataType, memref: ir.MemRef, space: ir.MemorySpace = ir.MemorySpace.Vec
) -> ir.TileType:
    return ir.TileType(shape, dtype, memref, None, space)


def _tile_t_with_view(
    shape: list[int],
    dtype: DataType,
    memref: ir.MemRef,
    tile_view: ir.TileView,
    space: ir.MemorySpace = ir.MemorySpace.Vec,
) -> ir.TileType:
    return ir.TileType(shape, dtype, memref, tile_view, space)


def _tensor_t(shape: list[int], dtype: DataType) -> ir.TensorType:
    return ir.TensorType(shape, dtype)


def _load_call(
    source: ir.Var, offsets: ir.MakeTuple, shapes: ir.MakeTuple, tile_type: ir.TileType
) -> ir.Call:
    """Build a tile.load Call with default valid_shapes=shapes, target_memory=Vec, transpose=False."""
    return ir.Call(
        ir.Op("tile.load"),
        [source, offsets, shapes, shapes],
        {"target_memory": ir.MemorySpace.Vec, "transpose": False},
        tile_type,
        _SPAN,
    )


def _get_mlir_code(result: str | dict[str, str]) -> str:
    """Normalize generate() output to a single MLIR string."""
    return result if isinstance(result, str) else "".join(result.values())


def _get_alloc_tile_lines(mlir_code: str) -> list[str]:
    """Return normalized alloc_tile lines from generated MLIR."""
    return [line.strip() for line in mlir_code.splitlines() if "pto.alloc_tile" in line]


def _generate_legalized_mlir(program: ir.Program) -> str:
    """Run legalization then generate MLIR."""
    legalized = passes.legalize_pto_buffer_reuse()(program)
    return _get_mlir_code(codegen.PTOCodegen().generate(legalized))


def _get_alloc_addrs(alloc_lines: list[str]) -> list[str]:
    """Extract alloc_tile addr values after asserting that each line carries one."""
    for line in alloc_lines:
        assert "addr =" in line, f"Expected addr attribute in alloc_tile: {line}"
    return [line.split("addr = ")[1].split()[0] for line in alloc_lines]


class TestLegalizeWithCodegen:
    """End-to-end: legalize pass + codegen produces valid MLIR."""

    def test_fillpad_shared_still_single_alloc(self):
        """After legalization, fillpad sharing still produces one alloc_tile."""
        shared = _MemRefAlloc().vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)
        load_type = _tile_t([128, 128], _FP32, shared)

        padded_view = ir.TileView(valid_shape=[_ci(128), _ci(128)], pad=ir.PadValue.max)
        padded_type = _tile_t_with_view([128, 128], _FP32, shared, padded_view)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", load_type, _SPAN)
        t2 = ir.Var("t2", padded_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)

        load_call = _load_call(a_var, offsets, shapes, load_type)
        fillpad_call = ir.Call(
            ir.Op("tile.fillpad"),
            [t1],
            {"pad_value": ir.PadValue.max},
            padded_type,
            _SPAN,
        )
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load_call, _SPAN),
                ir.AssignStmt(t2, fillpad_call, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        mlir_code = _generate_legalized_mlir(program)
        alloc_lines = _get_alloc_tile_lines(mlir_code)
        assert len(alloc_lines) == 2, (
            f"Expected two alloc_tiles for per-var alloc (same MemRef, same addr), got: {alloc_lines}"
        )
        assert "%c-1" not in mlir_code
        addr_values = _get_alloc_addrs(alloc_lines)
        assert addr_values[0] == addr_values[1], f"Expected same addr for shared MemRef, got: {addr_values}"

    def test_fillpad_dynamic_valid_row_keeps_shared_addr(self):
        """Dynamic valid_row and fillpad should keep one shared address after legalization."""
        shared = _MemRefAlloc().vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)
        valid_rows = ir.Var("m", ir.ScalarType(_IDX), _SPAN)

        load_view = ir.TileView(valid_shape=[valid_rows, _ci(128)])
        load_type = _tile_t_with_view([128, 128], _FP32, shared, load_view)

        padded_view = ir.TileView(valid_shape=[_ci(128), _ci(128)], pad=ir.PadValue.max)
        padded_type = _tile_t_with_view([128, 128], _FP32, shared, padded_view)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", load_type, _SPAN)
        t2 = ir.Var("t2", padded_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)

        load_call = _load_call(a_var, offsets, shapes, load_type)
        fillpad_call = ir.Call(
            ir.Op("tile.fillpad"),
            [t1],
            {"pad_value": ir.PadValue.max},
            padded_type,
            _SPAN,
        )
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load_call, _SPAN),
                ir.AssignStmt(t2, fillpad_call, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [
                (a_var, ir.ParamDirection.In),
                (b_var, ir.ParamDirection.Out),
                (valid_rows, ir.ParamDirection.In),
            ],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        mlir_code = _generate_legalized_mlir(program)
        alloc_lines = _get_alloc_tile_lines(mlir_code)

        assert len(alloc_lines) == 2, (
            f"Expected two alloc_tiles for per-var alloc (same MemRef, same addr), got: {alloc_lines}"
        )
        addr_values = _get_alloc_addrs(alloc_lines)
        assert addr_values[0] == addr_values[1], f"Expected same addr for shared MemRef, got: {addr_values}"

        # Dynamic valid_shape: type has v_row=?, v_col=? (PTOAS requires both dynamic)
        assert "v_row=?" in alloc_lines[0], (
            f"Expected dynamic v_row=? in tile_buf type, got: {alloc_lines[0]}"
        )
        assert "v_col=?" in alloc_lines[0], (
            f"Expected dynamic v_col=? in tile_buf type, got: {alloc_lines[0]}"
        )

        padded_allocs = [line for line in alloc_lines if "pad=2>" in line]
        assert len(padded_allocs) == 1, f"Expected one padded alloc_tile after fillpad, got: {alloc_lines}"

    def test_incompatible_shape_split_produces_two_allocs(self):
        """After legalization, split MemRefs produce separate alloc_tiles."""
        alloc = _MemRefAlloc()
        shared = alloc.vec([128, 128], _FP32)

        input_t = _tensor_t([128, 128], _FP32)
        output_t = _tensor_t([128, 128], _FP32)

        view_128 = ir.TileView(valid_shape=[_ci(128), _ci(128)])
        view_64 = ir.TileView(valid_shape=[_ci(64), _ci(64)])

        tile1_type = _tile_t_with_view([128, 128], _FP32, shared, view_128)
        tile2_type = _tile_t_with_view([64, 64], _FP32, shared, view_64)

        a_var = ir.Var("a", input_t, _SPAN)
        b_var = ir.Var("b", output_t, _SPAN)
        t1 = ir.Var("t1", tile1_type, _SPAN)
        t2 = ir.Var("t2", tile2_type, _SPAN)
        result_var = ir.Var("result", output_t, _SPAN)

        offsets = ir.MakeTuple([_ci(0), _ci(0)], _SPAN)
        shapes_128 = ir.MakeTuple([_ci(128), _ci(128)], _SPAN)
        shapes_64 = ir.MakeTuple([_ci(64), _ci(64)], _SPAN)

        load1 = _load_call(a_var, offsets, shapes_128, tile1_type)
        load2 = _load_call(a_var, offsets, shapes_64, tile2_type)
        store_call = ir.Call(ir.Op("tile.store"), [t2, offsets, b_var], result_var.type, _SPAN)

        body = ir.SeqStmts(
            [
                ir.AssignStmt(t1, load1, _SPAN),
                ir.AssignStmt(t2, load2, _SPAN),
                ir.AssignStmt(result_var, store_call, _SPAN),
                ir.ReturnStmt([result_var], _SPAN),
            ],
            _SPAN,
        )

        func = ir.Function(
            "main",
            [(a_var, ir.ParamDirection.In), (b_var, ir.ParamDirection.Out)],
            [output_t],
            body,
            _SPAN,
            ir.FunctionType.InCore,
        )
        program = ir.Program([func], "Test", _SPAN)

        mlir_code = _generate_legalized_mlir(program)
        alloc_lines = _get_alloc_tile_lines(mlir_code)
        assert len(alloc_lines) == 2, f"Expected two alloc_tiles after split, got: {alloc_lines}"

        _get_alloc_addrs(alloc_lines)

        sizes_128 = [line for line in alloc_lines if "rows=128" in line and "cols=128" in line]
        sizes_64 = [line for line in alloc_lines if "rows=64" in line and "cols=64" in line]
        assert len(sizes_128) == 1, f"Expected one 128x128 alloc: {alloc_lines}"
        assert len(sizes_64) == 1, f"Expected one 64x64 alloc: {alloc_lines}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
