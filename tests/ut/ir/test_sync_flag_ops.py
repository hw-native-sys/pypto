# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for system.sync_src / system.sync_dst (intra-core two-pipe flag sync).

event_id is either a static compile-time int attribute, or a dynamic
ScalarType(INDEX) operand (mirroring system.sync_set / system.sync_wait,
cross-core — see tests/ut/ir/test_cross_core_ops.py) — never both. A bare
Python int is treated as static; any Expr (including a ConstInt) is routed
through the dynamic operand path, matching _create_sync_op in
pypto/ir/op/system_ops.py.
"""

import pytest
from pypto import DataType, ir
from pypto.ir.op import system_ops


@pytest.mark.parametrize("op_name,fn", [("system.sync_src", system_ops.sync_src), ("system.sync_dst", system_ops.sync_dst)])
def test_sync_flag_ops_registered(op_name, fn):
    """system.sync_src / system.sync_dst are registered ops."""
    assert ir.is_op_registered(op_name), f"{op_name} should be registered"


@pytest.mark.parametrize("fn", [system_ops.sync_src, system_ops.sync_dst])
def test_sync_flag_static_event_id(fn):
    """A plain int event_id is stored as a static kwarg, with no positional operand."""
    call = fn(set_pipe=ir.PipeType.MTE2, wait_pipe=ir.PipeType.V, event_id=3)
    assert isinstance(call.type, ir.UnknownType)
    assert call.args == []
    assert call.kwargs == {
        "set_pipe": int(ir.PipeType.MTE2),
        "wait_pipe": int(ir.PipeType.V),
        "event_id": 3,
    }


@pytest.mark.parametrize("fn", [system_ops.sync_src, system_ops.sync_dst])
def test_sync_flag_dynamic_event_id(fn):
    """An index Scalar event id is carried as the dynamic operand, not a kwarg."""
    event_id = ir.Var("event_id", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    call = fn(set_pipe=ir.PipeType.MTE2, wait_pipe=ir.PipeType.V, event_id=event_id)
    assert isinstance(call.type, ir.UnknownType)
    assert call.args == [event_id]
    assert "event_id" not in call.kwargs


@pytest.mark.parametrize("fn", [system_ops.sync_src, system_ops.sync_dst])
@pytest.mark.parametrize("event_id", [-1, 8])
def test_sync_flag_rejects_out_of_range_static_event_id(fn, event_id):
    """PTO_EventEnum only defines EVENT_ID0..EVENT_ID7 (the [0, 7] range)."""
    with pytest.raises(ValueError, match="event_id"):
        fn(set_pipe=ir.PipeType.MTE2, wait_pipe=ir.PipeType.V, event_id=event_id)


@pytest.mark.parametrize("fn", [system_ops.sync_src, system_ops.sync_dst])
def test_sync_flag_rejects_non_index_dynamic_event_id(fn):
    """PTO's dynamic event operand is index-typed."""
    event_id = ir.Var("event_id", ir.ScalarType(DataType.INT32), ir.Span.unknown())
    with pytest.raises(ValueError, match=r"ScalarType\(INDEX\)"):
        fn(set_pipe=ir.PipeType.MTE2, wait_pipe=ir.PipeType.V, event_id=event_id)


@pytest.mark.parametrize("fn", [system_ops.sync_src, system_ops.sync_dst])
def test_sync_flag_rejects_missing_event_id(fn):
    """Exactly one of static event_id / dynamic operand must be present."""
    with pytest.raises(TypeError, match="event_id"):
        fn(set_pipe=ir.PipeType.MTE2, wait_pipe=ir.PipeType.V, event_id=None)
