# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for tile.gather_compare (compare-form pto.tgather, DPS-via-args).

Type contract (enforced by the op's type deduction):
    * ``src`` dtype in {FP16, FP32, INT16, INT32}; tile lives in Vec.
    * ``dst`` dtype is always INT32 (gathered indices); tile lives in Vec.
    * ``kvalue`` is a scalar whose dtype equals ``src``.
    * ``tmp`` is a UINT8 workspace tile (synthesized by the
      tensor→tile conversion pass; carried through as-is at the tile surface).
"""

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.language.parser.diagnostics import InvalidOperationError

_VALID_SRC_DTYPES = [pl.FP16, pl.FP32, pl.INT16, pl.INT32]
_INVALID_SRC_DTYPES = [pl.UINT16, pl.UINT32, pl.UINT8, pl.INT64]


def _find_call(program, op_name: str):
    """The single Call to ``op_name`` in ``program``'s only function.

    Asserting on the deduced Call is what makes these type tests actually test
    the type contract: a substring match on the printed program passes whenever
    the operator merely *appears*, whatever shape or dtype it deduced.
    """
    wanted = _ir.get_op(op_name).name
    found = []

    class _Collector(_ir.IRVisitor):
        def visit_call(self, expr):  # type: ignore[override]
            if expr.op.name == wanted:
                found.append(expr)
            super().visit_call(expr)

    for func in program.functions.values():
        _Collector().visit_stmt(func.body)
    assert len(found) == 1, f"expected exactly one {op_name} call, found {len(found)}"
    return found[0]


def _static_shape(shaped_type: _ir.ShapedType) -> list[int]:
    dims: list[int] = []
    for axis, dim in enumerate(shaped_type.shape):
        assert isinstance(dim, _ir.ConstInt), f"dimension {axis} is not a static extent"
        dims.append(dim.value)
    return dims


def _tuple_elements(call) -> list[_ir.ShapedType]:
    """The deduced result type of a multi-output call, as its output elements."""
    result_type = call.type
    assert isinstance(result_type, _ir.TupleType), (
        f"{call.op.name} must deduce a TupleType, got {type(result_type).__name__}"
    )
    elements: list[_ir.ShapedType] = []
    for i, element in enumerate(result_type.types):
        assert isinstance(element, _ir.ShapedType), (
            f"{call.op.name} output {i} must be a shaped type, got {type(element).__name__}"
        )
        elements.append(element)
    return elements


def _build_program(cmp_mode: str | int = "eq", offset=0, src_dtype=pl.FP32, count_dtype=pl.INT32, rows=32):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[rows, 64], src_dtype],
            kvalue: pl.Scalar[src_dtype],
            out_dst: pl.Tensor[[rows, 8], pl.INT32],
            out_cdst: pl.Tensor[[1, rows], count_dtype],
        ):
            s: pl.Tile[[rows, 64], src_dtype] = pl.load(src, [0, 0], [rows, 64])
            tmp: pl.Tile[[32, 64], pl.UINT8] = pl.tile.create([32, 64], pl.UINT8)
            d, c = pl.tile.gather_compare(
                s, kvalue, tmp, cmp_mode=cmp_mode, offset=offset, out_cols=8, count_dtype=count_dtype
            )
            pl.store(d, [0, 0], out_dst)
            pl.store(c, [0, 0], out_cdst)

    return Program


class TestTileGatherCompareTypes:
    """Type-contract tests: dtype allowed / disallowed, dst INT32, kvalue matches src."""

    @pytest.mark.parametrize("src_dtype", _VALID_SRC_DTYPES)
    def test_valid_src_dtype(self, src_dtype):
        call = _find_call(_build_program(src_dtype=src_dtype), "tile.gather_compare")
        dst, cdst = _tuple_elements(call)
        # dst holds the gathered indices: [rows, out_cols], INT32 whatever src is.
        assert _static_shape(dst) == [32, 8]
        assert dst.dtype == pl.INT32
        # cdst holds one match count per row: [1, rows], count_dtype (INT32 by default).
        assert _static_shape(cdst) == [1, 32]
        assert cdst.dtype == pl.INT32
        # Both destinations live in Vec — set_output_memory reaches every element
        # of the TupleType, not just the first.
        assert dst.memory_space == cdst.memory_space == _ir.MemorySpace.Vec

    def test_count_dtype_selects_cdst_dtype(self):
        call = _find_call(_build_program(count_dtype=pl.UINT32), "tile.gather_compare")
        dst, cdst = _tuple_elements(call)
        assert dst.dtype == pl.INT32, "dst indices stay INT32 regardless of count_dtype"
        assert cdst.dtype == pl.UINT32

    @pytest.mark.parametrize("rows", [1, 2, 7, 8, 9, 16])
    @pytest.mark.parametrize("count_dtype", [pl.INT32, pl.UINT32])
    def test_count_capacity_is_aligned_without_padding_valid_counts(self, rows, count_dtype):
        """The tuple result owns its storage; an oversized store target cannot fix it."""
        call = _find_call(_build_program(rows=rows, count_dtype=count_dtype), "tile.gather_compare")
        dst, counts = _tuple_elements(call)
        assert isinstance(counts, _ir.TileType)
        assert _static_shape(dst) == [rows, 8]
        # Eight 32-bit slots fill the smallest legal 32-byte UB row. Aligned
        # source row counts retain their capacity instead of gaining a block.
        assert _static_shape(counts) == [1, ((rows + 7) // 8) * 8]
        assert counts.dtype == count_dtype
        assert counts.memory_space == _ir.MemorySpace.Vec
        view = counts.tile_view
        if rows % 8:
            # A padded tile must carry an explicit logical extent. A fully
            # valid row-major tile may canonicalize to the implicit view.
            assert view is not None
        if view is not None:
            assert view.blayout == _ir.TileLayout.row_major
        valid_shape = counts.shape if view is None else view.valid_shape
        assert len(valid_shape) == 2
        for dim, expected in zip(valid_shape, [1, rows]):
            assert isinstance(dim, _ir.ConstInt)
            assert dim.value == expected

    @pytest.mark.parametrize("dynamic_valid", [False, True])
    @pytest.mark.parametrize("count_dtype", [pl.INT32, pl.UINT32])
    def test_partial_source_rows_remain_the_valid_count_extent(self, dynamic_valid, count_dtype):
        """Physical capacity sizes storage; only source valid rows produce counts."""
        span = _ir.Span.unknown()
        valid_rows = (
            _ir.Var("valid_rows", _ir.ScalarType(pl.INDEX), span)
            if dynamic_valid
            else _ir.ConstInt(7, pl.INDEX, span)
        )
        source = _ir.Var(
            "source",
            _ir.TileType([9, 64], pl.FP32, tile_view=_ir.TileView(valid_shape=[valid_rows, 64])),
            span,
        )
        workspace = _ir.Var("workspace", _ir.TileType([1, 512], pl.UINT8), span)
        call = _ir.create_op_call(
            "tile.gather_compare",
            [source, _ir.ConstFloat(0.5, pl.FP32, span), workspace],
            {"cmp_mode": 4, "out_cols": 64, "count_dtype": count_dtype},
            span,
        )
        dst, counts = _tuple_elements(call)
        assert isinstance(dst, _ir.TileType)
        assert isinstance(counts, _ir.TileType)
        assert _static_shape(dst) == [9, 64]
        assert _static_shape(counts) == [1, 16]
        assert dst.tile_view is not None and counts.tile_view is not None
        # Preserve the dynamic extent itself through the tuple type instead of
        # replacing it with the physical source rows or the aligned capacity.
        assert dst.tile_view.valid_shape[0].same_as(valid_rows)
        assert counts.tile_view.valid_shape[1].same_as(valid_rows)
        projected = _ir.TupleGetItemExpr(call, 1, span)
        assert isinstance(projected.type, _ir.TileType)
        assert projected.type.tile_view is not None
        assert projected.type.tile_view.valid_shape[1].same_as(valid_rows)

    @pytest.mark.parametrize(
        ("rows", "error"), [(0, "positive"), (-1, "positive"), ((1 << 63) - 1, "overflow")]
    )
    def test_invalid_count_capacity_is_rejected_before_allocation(self, rows, error):
        """Rounding an invalid/overflowing capacity must not create an unsafe tile."""
        span = _ir.Span.unknown()
        source = _ir.Var("source", _ir.TileType([rows, 64], pl.FP32), span)
        workspace = _ir.Var("workspace", _ir.TileType([1, 512], pl.UINT8), span)
        with pytest.raises(ValueError, match=error):
            _ir.create_op_call(
                "tile.gather_compare",
                [source, _ir.ConstFloat(0.5, pl.FP32, span), workspace],
                {"cmp_mode": 4, "out_cols": 64},
                span,
            )

    def test_destinations_are_not_arguments(self):
        """The op takes three inputs; dst and cdst reach the caller as results.

        Naming them in the argument list instead would make the caller allocate
        buffers InitMemRef owns — see docs/en/dev/ir/08-multi_output_ops.md.
        """
        call = _find_call(_build_program(), "tile.gather_compare")
        assert len(call.args) == 3
        assert _ir.get_op_output_arity("tile.gather_compare") == 2
        assert _ir.get_op_argument_count("tile.gather_compare") == 3

    @pytest.mark.parametrize("src_dtype", _INVALID_SRC_DTYPES)
    def test_invalid_src_dtype_raises(self, src_dtype):
        with pytest.raises(InvalidOperationError, match="src dtype"):
            _build_program(src_dtype=src_dtype)

    def test_kvalue_dtype_mismatch_raises(self):
        with pytest.raises(InvalidOperationError, match="kvalue dtype"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[32, 64], pl.FP32],
                    kvalue: pl.Scalar[pl.FP16],  # mismatch: kvalue FP16 vs src FP32
                    out_dst: pl.Tensor[[32, 8], pl.INT32],
                    out_cdst: pl.Tensor[[1, 32], pl.INT32],
                ):
                    s: pl.Tile[[32, 64], pl.FP32] = pl.load(src, [0, 0], [32, 64])
                    tmp: pl.Tile[[32, 64], pl.UINT8] = pl.tile.create([32, 64], pl.UINT8)
                    d, c = pl.tile.gather_compare(s, kvalue, tmp, cmp_mode="eq", out_cols=8)
                    pl.store(d, [0, 0], out_dst)
                    pl.store(c, [0, 0], out_cdst)


class TestTileGatherCompareCmpMode:
    """cmp_mode accepts strings and ints in [0, 5]; otherwise raises."""

    def test_default_eq(self):
        call = _find_call(_build_program(), "tile.gather_compare")
        assert call.kwargs["cmp_mode"] == 0
        assert call.kwargs["offset"] == 0

    def test_gt_with_offset(self):
        call = _find_call(_build_program(cmp_mode="gt", offset=4), "tile.gather_compare")
        assert call.kwargs["cmp_mode"] == 4
        assert call.kwargs["offset"] == 4

    def test_int_cmp_mode(self):
        call = _find_call(_build_program(cmp_mode=2), "tile.gather_compare")
        assert call.kwargs["cmp_mode"] == 2

    def test_invalid_cmp_mode_string(self):
        with pytest.raises(InvalidOperationError, match="cmp_mode"):
            _build_program(cmp_mode="bogus")

    def test_invalid_cmp_mode_int(self):
        with pytest.raises(InvalidOperationError, match="cmp_mode"):
            _build_program(cmp_mode=99)


def _build_tensor_compare_program(cmp_mode="eq", offset=0, src_dtype=pl.FP32):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[32, 64], src_dtype],
            kvalue: pl.Scalar[src_dtype],
        ) -> tuple[pl.Tensor[[32, 8], pl.INT32], pl.Tensor[[1, 32], pl.INT32]]:
            d, c = pl.tensor.gather(src, kvalue=kvalue, cmp_mode=cmp_mode, offset=offset, out_cols=8)
            return d, c

    return Program


class TestTensorGatherCompareTypes:
    """Tensor-level unified gather routing to tensor.gather_compare."""

    @pytest.mark.parametrize("src_dtype", _VALID_SRC_DTYPES)
    def test_valid_src_dtype(self, src_dtype):
        call = _find_call(_build_tensor_compare_program(src_dtype=src_dtype), "tensor.gather_compare")
        dst, cdst = _tuple_elements(call)
        assert _static_shape(dst) == [32, 8]
        assert dst.dtype == pl.INT32
        assert _static_shape(cdst) == [1, 32]
        assert cdst.dtype == pl.INT32
        # The tensor-level op takes only (input, kvalue): the UINT8 workspace is
        # synthesized later, by ConvertTensorToTileOps.
        assert len(call.args) == 2

    @pytest.mark.parametrize("src_dtype", _INVALID_SRC_DTYPES)
    def test_invalid_src_dtype_raises(self, src_dtype):
        with pytest.raises(InvalidOperationError, match="input dtype"):
            _build_tensor_compare_program(src_dtype=src_dtype)

    def test_compare_with_offset(self):
        call = _find_call(_build_tensor_compare_program(cmp_mode="gt", offset=4), "tensor.gather_compare")
        assert call.kwargs["cmp_mode"] == 4
        assert call.kwargs["offset"] == 4

    def test_mutually_exclusive_index_and_compare(self):
        with pytest.raises(InvalidOperationError, match="mutually exclusive"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[32, 64], pl.FP32],
                    idx: pl.Tensor[[32, 8], pl.INT32],
                    kv: pl.Scalar[pl.FP32],
                ) -> pl.Tensor[[32, 8], pl.FP32]:
                    return pl.tensor.gather(src, dim=-1, index=idx, kvalue=kv, cmp_mode="eq", out_cols=8)

    def test_mutually_exclusive_mask_and_compare(self):
        with pytest.raises(InvalidOperationError, match="mutually exclusive"):

            @pl.program
            class Bad:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    src: pl.Tensor[[32, 64], pl.FP32],
                    kv: pl.Scalar[pl.FP32],
                ) -> pl.Tensor[[32, 8], pl.INT32]:
                    return pl.tensor.gather(src, mask_pattern=1, kvalue=kv, cmp_mode="eq", out_cols=8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
