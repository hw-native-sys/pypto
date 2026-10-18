# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Registered cast scratch rules and their read-only query contract."""

import pytest
from pypto import DataType, backend, ir


@pytest.mark.parametrize(
    "target,is_a2a3",
    [(backend.BackendType.Ascend910, True), (backend.BackendType.Ascend950, False)],
)
@pytest.mark.parametrize("saturating", [False, True])
@pytest.mark.parametrize(
    "src,dst,needs_scratch",
    [
        (DataType.FP32, DataType.INT16, True),
        (DataType.FP16, DataType.INT16, True),
        (DataType.FP16, DataType.INT8, True),
        (DataType.FP16, DataType.UINT8, True),
        (DataType.FP32, DataType.FP16, False),
        (DataType.FP16, DataType.FP32, False),
        (DataType.FP16, DataType.INT4, False),
    ],
)
def test_tcvt_scratch_pairs(target, is_a2a3, saturating, src, dst, needs_scratch):
    instance = backend.get_backend_instance(target)
    source = ir.TileType([1, 128], src)
    plan = instance.plan_op_scratch("tile.cast", source_tile=source, target_dtype=dst, saturating=saturating)
    assert plan is not None
    if is_a2a3 and not saturating and needs_scratch:
        assert plan.status == backend.OpScratchStatus.Required
        assert plan.spec is not None
        assert (plan.spec.dtype, plan.spec.rows, plan.spec.cols) == (
            DataType.INT8,
            1,
            512 if src == DataType.FP32 else 256,
        )
    else:
        assert plan.status == backend.OpScratchStatus.NotRequired
        assert plan.spec is None


@pytest.mark.parametrize(
    "src,dst,shape,valid_shape,expected_bytes",
    [
        (DataType.FP32, DataType.INT16, [4, 128], [3, 96], 1152),
        (DataType.FP16, DataType.INT16, [4, 128], [3, 7], 32),
        (DataType.FP16, DataType.INT8, [4, 128], [3, 7], 160),
        (DataType.FP16, DataType.UINT8, [4, 256], [3, 192], 256),
    ],
)
def test_tcvt_scratch_spec_uses_valid_shape(src, dst, shape, valid_shape, expected_bytes):
    """Backend sizing distinguishes valid extents from the physical row pitch."""
    instance = backend.get_backend_instance(backend.BackendType.Ascend910)
    source = ir.TileType(shape, src, tile_view=ir.TileView(valid_shape=valid_shape))
    plan = instance.plan_op_scratch("tile.cast", source_tile=source, target_dtype=dst, saturating=False)
    assert plan is not None and plan.status == backend.OpScratchStatus.Required
    assert plan.spec is not None
    assert (plan.spec.dtype, plan.spec.rows, plan.spec.cols) == (DataType.INT8, 1, expected_bytes)


@pytest.mark.parametrize(
    "target,saturating,needs_static",
    [
        (backend.BackendType.Ascend910, False, True),
        (backend.BackendType.Ascend910, True, False),
        (backend.BackendType.Ascend950, False, False),
    ],
)
@pytest.mark.parametrize("dynamic_dimension", ["valid_rows", "valid_cols", "physical_cols"])
def test_tcvt_scratch_spec_dynamic_validity(target, saturating, needs_static, dynamic_dimension):
    """Only the backend that needs scratch validates static source geometry."""
    instance = backend.get_backend_instance(target)
    span = ir.Span.unknown()
    rows = ir.Var("rows", ir.ScalarType(DataType.INDEX), span)
    shape = [4, rows if dynamic_dimension == "physical_cols" else 128]
    valid_shape = [
        rows if dynamic_dimension == "valid_rows" else 3,
        rows if dynamic_dimension == "valid_cols" else 96,
    ]
    source = ir.TileType(shape, DataType.FP32, tile_view=ir.TileView(valid_shape=valid_shape))
    plan = instance.plan_op_scratch(
        "tile.cast", source_tile=source, target_dtype=DataType.INT16, saturating=saturating
    )
    assert plan is not None and plan.spec is None
    assert plan.status == (
        backend.OpScratchStatus.Unknown if needs_static else backend.OpScratchStatus.NotRequired
    )
    if needs_static:
        assert "requires static source shape" in plan.reason


@pytest.mark.parametrize(
    "configured_type", [None, backend.BackendType.Ascend910, backend.BackendType.Ascend950]
)
@pytest.mark.parametrize("target_type", [backend.BackendType.Ascend910, backend.BackendType.Ascend950])
def test_cast_scratch_query_uses_explicit_backend(configured_type, target_type):
    """Queries neither require nor change the global target, including a different target."""
    backend.reset_for_testing()
    try:
        if configured_type is not None:
            backend.set_backend_type(configured_type)
        target = backend.get_backend_instance(target_type)
        source = ir.TileType([4, 128], DataType.FP32, tile_view=ir.TileView(valid_shape=[3, 96]))
        plan = target.plan_op_scratch(
            "tile.cast", source_tile=source, target_dtype=DataType.INT16, saturating=False
        )
        assert plan is not None
        if target_type == backend.BackendType.Ascend910:
            assert plan.status == backend.OpScratchStatus.Required
            assert plan.spec is not None and plan.spec.cols == 1152
            with pytest.raises(AttributeError):
                setattr(plan.spec, "cols", 1)
        else:
            assert plan.status == backend.OpScratchStatus.NotRequired and plan.spec is None
        assert source.memref is None
        assert backend.is_backend_configured() == (configured_type is not None)
        if configured_type is not None:
            assert backend.get_backend_type() == configured_type
    finally:
        backend.reset_for_testing()


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({}, "dtypes"),
        ({"source_dtype": DataType.FP32}, "dtypes"),
        ({"source_dtype": DataType.FP32, "target_dtype": DataType.INT16}, "saturation"),
        ({"source_dtype": DataType.FP32, "target_dtype": DataType.INT16, "saturating": False}, "geometry"),
    ],
)
def test_cast_scratch_query_reports_missing_information(kwargs, reason):
    target = backend.get_backend_instance(backend.BackendType.Ascend910)
    plan = target.plan_op_scratch("tile.cast", **kwargs)
    assert plan is not None and plan.status == backend.OpScratchStatus.Unknown
    assert plan.spec is None and reason in plan.reason


@pytest.mark.parametrize("shape,dtype,reason", [([128], None, "2D"), ([4, 128], DataType.FP16, "disagrees")])
def test_cast_scratch_query_rejects_inconsistent_input(shape, dtype, reason):
    target = backend.get_backend_instance(backend.BackendType.Ascend910)
    plan = target.plan_op_scratch(
        "tile.cast",
        dtype,
        source_tile=ir.TileType(shape, DataType.FP32),
        target_dtype=DataType.INT16,
        saturating=False,
    )
    assert plan is not None and plan.status == backend.OpScratchStatus.Unsupported
    assert plan.spec is None and reason in plan.reason


def test_cast_no_scratch_verdict_does_not_require_geometry():
    target = backend.get_backend_instance(backend.BackendType.Ascend910)
    for plan in (
        target.plan_op_scratch("tile.cast", saturating=True),
        target.plan_op_scratch("tile.cast", source_dtype=DataType.FP32, target_dtype=DataType.FP16),
    ):
        assert plan is not None and plan.status == backend.OpScratchStatus.NotRequired
        assert plan.spec is None
    assert target.plan_op_scratch("tile.add") is None
