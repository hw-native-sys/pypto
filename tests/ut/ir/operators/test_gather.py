# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ``tile.gather`` (index-form ``pto.tgather``) type deduction.

Drives the deducer directly via the IR op so the dtype combinations are exercised
without depending on the DSL parser. The end-to-end behaviour (including the A5
device path) is covered by the system tests in ``tests/st/runtime/ops/test_gather.py``.

Type contract after the A5 (Ascend950) index-form alignment — the IR deducer is
backend-agnostic, so it accepts the union of what any target permits *up to the
universal index-width constraint* (PyPTO has no internal tgather verifier; the
arch-specific A2/A3 i32-only rule is left to the external PTOAS assembler — see
the PTO IR manual, ``pto.tgather`` index-form checks):

* ``src`` dtype in {FP16, FP32, INT16, INT32}; tile lives in Vec.
* ``indices`` dtype is INT32 (with any ``src``), or INT16 — but INT16 indices are
  only valid with a 16-bit ``src`` (FP16/INT16): the tgather b32 form reads each
  index as a u32, so an INT16 index with a 32-bit ``src`` is unsafe on every
  target, not merely arch-gated. That universal rule is checked by the deducer.
* ``tmp`` is a workspace operand required by the IR but not read by the A5 index
  form, so any Vec tile dtype is accepted (A2/A3 constrains it at PTOAS).
* ``dst`` dtype equals ``src``; ``dst`` shape equals ``indices``.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.ir.op import tensor, tile


def _gather(src_dtype: DataType, idx_dtype: DataType, tmp_dtype: DataType):
    span = ir.Span.unknown()
    src = ir.Var("src", ir.TileType([1, 64], src_dtype), span)
    idx = ir.Var("idx", ir.TileType([1, 64], idx_dtype), span)
    tmp = ir.Var("tmp", ir.TileType([1, 64], tmp_dtype), span)
    return tile.gather(src, idx, tmp)


class TestTileGatherIndexTypes:
    """Deducer type-contract tests for the index form."""

    @pytest.mark.parametrize("src_dtype", [DataType.FP16, DataType.FP32, DataType.INT16, DataType.INT32])
    def test_valid_src_dtype(self, src_dtype):
        call = _gather(src_dtype, DataType.INT32, DataType.INT32)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == src_dtype  # dst dtype follows src

    @pytest.mark.parametrize(
        ("src_dtype", "idx_dtype"),
        [
            (DataType.FP32, DataType.INT32),
            (DataType.INT32, DataType.INT32),
            (DataType.FP16, DataType.INT16),  # INT16 indices require a 16-bit src.
            (DataType.INT16, DataType.INT16),
        ],
    )
    def test_valid_index_dtype(self, src_dtype, idx_dtype):
        # INT32 indices are valid with any src; INT16 indices require a 16-bit src.
        call = _gather(src_dtype, idx_dtype, DataType.INT32)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == src_dtype  # dst dtype follows src

    @pytest.mark.parametrize("tmp_dtype", [DataType.FP32, DataType.FP16, DataType.INT32, DataType.INT16])
    def test_tmp_dtype_unconstrained(self, tmp_dtype):
        # tmp is not read by the A5 index form; any Vec tile dtype is accepted at IR level.
        call = _gather(DataType.FP32, DataType.INT32, tmp_dtype)
        assert isinstance(call.type, ir.TileType)

    def test_dst_shape_follows_indices(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([1, 128], DataType.FP16), span)
        idx = ir.Var("idx", ir.TileType([1, 16], DataType.INT16), span)
        tmp = ir.Var("tmp", ir.TileType([1, 16], DataType.FP32), span)
        call = tile.gather(src, idx, tmp)
        assert isinstance(call.type, ir.TileType)
        assert call.type.dtype == DataType.FP16  # follows src
        # dst shape follows indices [1, 16], not src [1, 128].
        assert len(call.type.shape) == 2
        assert isinstance(call.type.shape[0], ir.ConstInt) and call.type.shape[0].value == 1
        assert isinstance(call.type.shape[1], ir.ConstInt) and call.type.shape[1].value == 16

    def test_invalid_src_dtype_raises(self):
        with pytest.raises(ValueError, match="src dtype"):
            _gather(DataType.UINT8, DataType.INT32, DataType.INT32)

    @pytest.mark.parametrize("bad_idx_dtype", [DataType.FP32, DataType.INT8])
    def test_invalid_index_dtype_raises(self, bad_idx_dtype):
        with pytest.raises(ValueError, match="indices dtype"):
            _gather(DataType.FP32, bad_idx_dtype, DataType.INT32)

    @pytest.mark.parametrize("wide_src_dtype", [DataType.FP32, DataType.INT32])
    def test_int16_index_requires_16bit_src(self, wide_src_dtype):
        # INT16 indices with a 32-bit src are unsafe on every target (tgather b32
        # reads them as u32), so the deducer rejects the combination outright.
        with pytest.raises(ValueError, match="16-bit src"):
            _gather(wide_src_dtype, DataType.INT16, DataType.INT32)

    def test_non_tile_indices_raises(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TileType([1, 64], DataType.FP32), span)
        scalar_idx = ir.Var("idx", ir.ScalarType(DataType.INT32), span)
        tmp = ir.Var("tmp", ir.TileType([1, 64], DataType.INT32), span)
        with pytest.raises(ValueError, match="TileType"):
            tile.gather(src, scalar_idx, tmp)


class TestTensorGatherFlat:
    @pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32, DataType.INT16, DataType.INT32])
    def test_flat_source_dtype(self, dtype):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([2, 3, 32], dtype), span)
        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)
        call = tensor.gather(src, index=idx)
        assert isinstance(call.type, ir.TensorType)
        assert call.type.dtype == dtype

    @pytest.mark.parametrize("source_type", [ir.TensorType, ir.TileType, ir.DistributedTensorType])
    @pytest.mark.parametrize("index_type", [ir.TensorType, ir.TileType, ir.DistributedTensorType])
    def test_flat_index_form(self, source_type, index_type):
        span = ir.Span.unknown()
        src_shape = [ir.ConstInt(n, DataType.INDEX, span) for n in (4, 32)]
        idx_shape = [ir.ConstInt(n, DataType.INDEX, span) for n in (2, 16)]
        src = ir.Var("src", source_type(src_shape, DataType.FP32), span)
        idx = ir.Var("idx", index_type(idx_shape, DataType.INT32), span)
        call = tensor.gather(src, index=idx)
        assert call.op.name == ir.get_op("tensor.gather").name
        assert "dim" not in call.kwargs
        ir.assert_structural_equal(call.type, ir.TensorType([2, 16], DataType.FP32))
        ir.assert_structural_equal(tensor.gather(src, idx), call)

    @pytest.mark.parametrize("memory", [ir.MemorySpace.Mat, ir.MemorySpace.Left, ir.MemorySpace.Right])
    @pytest.mark.parametrize("source_type", [ir.TensorType, ir.TileType])
    def test_flat_rejects_non_vec_index(self, memory, source_type):
        span = ir.Span.unknown()
        src = ir.Var("src", source_type([4, 32], DataType.FP32), span)
        idx = ir.Var("idx", ir.TileType([1, 16], DataType.INT32, memory_space=memory), span)
        with pytest.raises(ValueError, match="indices in Vec"):
            tensor.gather(src, index=idx)

    @pytest.mark.parametrize("source_type", [ir.TensorType, ir.TileType])
    @pytest.mark.parametrize("wrapper", [tensor.gather, pl.gather])
    @pytest.mark.parametrize(
        "view",
        [ir.TileView(blayout=ir.TileLayout.col_major), ir.TileView(slayout=ir.TileLayout.row_major)],
    )
    def test_flat_rejects_unsupported_index_layout(self, source_type, wrapper, view):
        span = ir.Span.unknown()
        src = ir.Var("src", source_type([4, 32], DataType.FP32), span)
        idx = ir.Var(
            "idx",
            ir.TileType([16, 16], DataType.INT32, tile_view=view, memory_space=ir.MemorySpace.Vec),
            span,
        )
        if wrapper is pl.gather:
            src = (pl.Tile if source_type is ir.TileType else pl.Tensor)(expr=src)
            idx = pl.Tile(expr=idx)
        with pytest.raises(ValueError, match="indices with an unboxed row-major layout"):
            wrapper(src, index=idx)

    def test_flat_rejects_transpose_view_index(self):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([1024], DataType.FP32), span)
        idx = ir.Var("idx", ir.TileType([16, 16], DataType.INT32, memory_space=ir.MemorySpace.Vec), span)
        with pytest.raises(ValueError, match="indices with an unboxed row-major layout"):
            tensor.gather(src, tile.transpose_view(idx))

    @pytest.mark.parametrize("wrapper", [tensor.gather, pl.gather])
    @pytest.mark.parametrize("options", [{"offset": 4}, {"count_dtype": DataType.INT32}])
    def test_mask_rejects_compare_options(self, wrapper, options):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([1, 32], DataType.FP32), span)
        if wrapper is pl.gather:
            src = pl.Tensor(expr=src)
        with pytest.raises(ValueError, match="only valid for the compare form"):
            wrapper(src, mask_pattern=1, **options)

    @pytest.mark.parametrize("wrapper", [tensor.gather, pl.gather])
    @pytest.mark.parametrize("source_type", [ir.TensorType, ir.TileType])
    @pytest.mark.parametrize(
        ("dtype", "cols"),
        [
            (dtype, cols)
            for dtype in (DataType.FP16, DataType.FP32, DataType.INT16, DataType.INT32)
            for cols in (5, 17)
        ]
        + [(DataType.FP16, 8), (DataType.INT16, 8)],
    )
    def test_flat_rejects_unaligned_physical_rows(self, wrapper, source_type, dtype, cols):
        span = ir.Span.unknown()
        src = ir.Var("src", source_type([4, 32], dtype), span)
        idx = ir.Var("idx", ir.TensorType([1, cols], DataType.INT32), span)
        if wrapper is pl.gather:
            src = (pl.Tile if source_type is ir.TileType else pl.Tensor)(expr=src)
            idx = pl.Tensor(expr=idx)
        with pytest.raises(ValueError, match="32-byte aligned physical index/output rows"):
            wrapper(src, index=idx)

    def test_flat_rejects_dynamic_index_columns(self):
        span = ir.Span.unknown()
        cols = ir.Var("cols", ir.ScalarType(DataType.INDEX), span)
        src = ir.Var("src", ir.TensorType([1024], DataType.FP32), span)
        idx = ir.Var("idx", ir.TensorType([ir.ConstInt(1, DataType.INDEX, span), cols], DataType.INT32), span)
        with pytest.raises(ValueError, match="positive static index column count"):
            tensor.gather(src, index=idx)

    @pytest.mark.parametrize("dtype", [DataType.INT16, DataType.FP32])
    def test_flat_requires_int32_indices(self, dtype):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([1024], DataType.FP16), span)
        idx = ir.Var("idx", ir.TensorType([1, 32], dtype), span)
        with pytest.raises(ValueError, match="INT32"):
            tensor.gather(src, index=idx)

    @pytest.mark.parametrize("shape", [[32], [1, 2, 16]])
    def test_flat_requires_2d_indices(self, shape):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([1024], DataType.FP32), span)
        idx = ir.Var("idx", ir.TensorType(shape, DataType.INT32), span)
        with pytest.raises(ValueError, match="2D index"):
            tensor.gather(src, index=idx)

    @pytest.mark.parametrize(
        "view",
        [ir.TensorView([64, 1], ir.TensorLayout.ND), ir.TensorView([1, 4], ir.TensorLayout.DN)],
    )
    def test_flat_rejects_noncontiguous_gm_source(self, view):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([4, 32], DataType.FP32, None, view), span)
        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)
        with pytest.raises(ValueError, match="contiguous ND"):
            tensor.gather(src, index=idx)

    @pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32, DataType.INT16, DataType.INT32])
    @pytest.mark.parametrize("index_type", [ir.TensorType, ir.TileType, ir.DistributedTensorType])
    def test_flat_preserves_index_valid_shape(self, dtype, index_type):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([1024], dtype), span)
        idx = ir.Var(
            "idx",
            index_type(
                [2, 32],
                DataType.INT32,
                None,
                ir.TileView(valid_shape=[1, 13])
                if index_type is ir.TileType
                else ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[1, 13]),
            ),
            span,
        )
        call = tensor.gather(src, index=idx)
        ir.assert_structural_equal(
            call.type,
            ir.TensorType(
                [2, 32], dtype, None, ir.TensorView(layout=ir.TensorLayout.ND, valid_shape=[1, 13])
            ),
        )

    @pytest.mark.parametrize("wrapper", [tensor.gather, pl.gather])
    def test_flat_wrapper_forms_and_validation(self, wrapper):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([1024], DataType.FP32), span)
        idx = ir.Var("idx", ir.TensorType([1, 16], DataType.INT32), span)
        if wrapper is pl.gather:
            src, idx = pl.Tensor(expr=src), pl.Tensor(expr=idx)
        positional, keyword = wrapper(src, idx), wrapper(src, index=idx)
        if wrapper is pl.gather:
            assert isinstance(positional, pl.Tensor)
            positional, keyword = positional.unwrap(), keyword.unwrap()
        ir.assert_structural_equal(positional, keyword)
        with pytest.raises(ValueError, match="both positionally"):
            wrapper(src, idx, index=idx)
        with pytest.raises(ValueError, match="mutually exclusive"):
            wrapper(src, index=idx, mask_pattern=1)
        with pytest.raises(ValueError, match="requires index"):
            wrapper(src, dim=0)


class TestGatherWrapperDelegation:
    @pytest.mark.parametrize(
        "options", [{"dim": -1}, {}, {"mask_pattern": 1, "output_dtype": DataType.UINT32}]
    )
    def test_single_result_forms_match_ir_wrapper(self, options):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([2, 32], DataType.FP32), span)
        idx = ir.Var("idx", ir.TensorType([2, 16], DataType.INT32), span)
        ir_options = options if "mask_pattern" in options else {**options, "index": idx}
        dsl_options = options if "mask_pattern" in options else {**options, "index": pl.Tensor(expr=idx)}
        actual = pl.gather(pl.Tensor(expr=src), **dsl_options)
        assert isinstance(actual, pl.Tensor)
        ir.assert_structural_equal(actual.unwrap(), tensor.gather(src, **ir_options))

    @pytest.mark.parametrize("scalar_form", ["dsl", "ir", "literal"])
    def test_compare_normalizes_scalar_and_wraps_tuple(self, scalar_form):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([2, 32], DataType.FP32), span)
        kv = ir.ConstFloat(1.0, DataType.FP32, span)
        threshold = {"dsl": pl.Scalar(expr=kv), "ir": kv, "literal": 1.0}[scalar_form]
        options = {"cmp_mode": "gt", "out_cols": 8, "offset": 4, "count_dtype": DataType.UINT32}
        actual = pl.gather(pl.Tensor(expr=src), kvalue=threshold, **options)
        expected = tensor.gather(src, kvalue=kv, **options)
        assert isinstance(actual, tuple) and len(actual) == 2
        for i, value in enumerate(actual):
            ir.assert_structural_equal(value.unwrap(), ir.TupleGetItemExpr(expected, i, span))

    @pytest.mark.parametrize(
        "options",
        [
            {},
            {"dim": 0},
            {"offset": 4},
            {"count_dtype": DataType.UINT32},
            {"cmp_mode": "gt"},
            {"mask_pattern": 1, "out_cols": 8},
            {"cmp_mode": "gt", "out_cols": 8, "output_dtype": DataType.UINT32},
        ],
    )
    def test_invalid_forms_share_ir_diagnostics(self, options):
        span = ir.Span.unknown()
        src = ir.Var("src", ir.TensorType([2, 32], DataType.FP32), span)
        with pytest.raises(ValueError) as expected:
            tensor.gather(src, **options)
        with pytest.raises(ValueError) as actual:
            pl.gather(pl.Tensor(expr=src), **options)
        assert str(actual.value) == str(expected.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
