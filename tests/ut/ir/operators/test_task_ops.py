# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""IR-level tests for the manual_scope task ops (``system.task_*``).

``system.task_invalid`` is the lowering target of the DSL ``None`` TaskId
sentinel; ``system.task_is_valid`` is its boolean predicate. The producer
TaskId of a ``pl.submit(...)`` call is carried as a tuple element of the
kernel ``Call`` itself, not as a standalone op. The tests construct ``Call``
nodes directly via ``ir.create_op_call``.
"""

import pytest
from pypto import DataType, ir


def _span():
    return ir.Span.unknown()


# ----------------------------------------------------------------------------
# Type deduction
# ----------------------------------------------------------------------------


def test_task_invalid_returns_scalar_task_id():
    call = ir.create_op_call("system.task_invalid", [], {}, _span())
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.TASK_ID


def test_task_is_valid_returns_scalar_bool():
    task_id = ir.Var("tid", ir.ScalarType(DataType.TASK_ID), _span())
    call = ir.create_op_call("system.task_is_valid", [task_id], {}, _span())
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.BOOL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
