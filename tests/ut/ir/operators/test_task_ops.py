# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""IR-level tests for the manual_scope task ops (``system.task_*``).

These ops are synthesized internally by ``LowerManualDepsToTaskId`` and the
upcoming phase-fence lowering pass. They are not exposed as DSL surfaces;
the tests construct ``Call`` nodes directly via ``ir.create_op_call``.
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


def test_task_id_of_returns_scalar_task_id():
    # Producer is a Var of any type — the op only carries the SSA companion
    # threading and doesn't constrain its argument's dtype in the IR.
    producer = ir.Var("v", ir.ScalarType(DataType.INT32), _span())
    call = ir.create_op_call("system.task_id_of", [producer], {}, _span())
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.TASK_ID


def test_task_is_valid_returns_scalar_bool():
    task_id = ir.Var("tid", ir.ScalarType(DataType.TASK_ID), _span())
    call = ir.create_op_call("system.task_is_valid", [task_id], {}, _span())
    assert isinstance(call.type, ir.ScalarType)
    assert call.type.dtype == DataType.BOOL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
