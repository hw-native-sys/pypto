# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR-level tests for the v2 ``WindowBuffer`` / ``CommGroup`` schema (N1.3)."""

import pytest
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    CommGroup,
    ConstInt,
    Span,
    WindowBuffer,
    structural_equal,
)


def _const(value: int) -> ConstInt:
    return ConstInt(value, DataType.INT64, Span.unknown())


# ---------------------------------------------------------------------------
# WindowBuffer
# ---------------------------------------------------------------------------


def test_window_buffer_minimal_construction():
    wb = WindowBuffer("data", _const(256), DataType.FP32)
    assert wb.name == "data"
    assert wb.dtype == DataType.FP32
    assert wb.load_from_host is False
    assert wb.store_to_host is False


def test_window_buffer_load_store_to_host_flags():
    """v2 schema: load/store_to_host are bool flags; the actual host tensor
    binding (if any) is recorded on the alloc op, not here."""
    wb = WindowBuffer("lut", _const(64), DataType.FP32, load_from_host=True, store_to_host=True)
    assert wb.load_from_host is True
    assert wb.store_to_host is True


# ---------------------------------------------------------------------------
# CommGroup structural equality
# ---------------------------------------------------------------------------


def test_comm_group_empty_devices_means_all():
    """``devices == []`` is the convention for "covers all DistributedConfig.device_ids"."""
    g = CommGroup([], [WindowBuffer("data", _const(64), DataType.FP32)])
    assert list(g.devices) == []


def test_comm_group_explicit_device_subset():
    g = CommGroup([0, 1, 2], [WindowBuffer("data", _const(64), DataType.FP32)])
    assert list(g.devices) == [0, 1, 2]


def test_comm_group_structural_equal():
    g1 = CommGroup([], [WindowBuffer("data", _const(256), DataType.FP32)])
    g2 = CommGroup([], [WindowBuffer("data", _const(256), DataType.FP32)])
    assert structural_equal(g1, g2)


def test_comm_group_structural_not_equal_when_devices_differ():
    g_all = CommGroup([], [WindowBuffer("data", _const(64), DataType.FP32)])
    g_subset = CommGroup([0, 1], [WindowBuffer("data", _const(64), DataType.FP32)])
    assert not structural_equal(g_all, g_subset)


def test_comm_group_structural_not_equal_when_subsets_differ():
    a = CommGroup([0, 1], [WindowBuffer("data", _const(64), DataType.FP32)])
    b = CommGroup([2, 3], [WindowBuffer("data", _const(64), DataType.FP32)])
    assert not structural_equal(a, b)


def test_comm_group_structural_not_equal_when_slots_differ():
    a = CommGroup([], [WindowBuffer("data", _const(64), DataType.FP32)])
    b = CommGroup([], [WindowBuffer("data", _const(128), DataType.FP32)])
    assert not structural_equal(a, b)


def test_comm_group_structural_not_equal_when_load_flag_differs():
    a = CommGroup([], [WindowBuffer("data", _const(64), DataType.FP32, load_from_host=False)])
    b = CommGroup([], [WindowBuffer("data", _const(64), DataType.FP32, load_from_host=True)])
    assert not structural_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
