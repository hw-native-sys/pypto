# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Preserve runtime lane control when PTOAS adds an AIV0 prologue gate."""

import pytest
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.backend.pto_backend import _honor_runtime_aiv_lane_control


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def _function(*, dual=True, runtime=True):
    span = ir.Span.unknown()
    statements = []
    if runtime:
        lane_type = ir.ScalarType(DataType.INDEX)
        lane = ir.Var("lane", lane_type, span)
        call = ir.Call(ir.get_op("tile.get_subblock_idx"), [], {}, lane_type, span)
        statements.append(ir.AssignStmt(lane, call, span))
    statements.append(ir.ReturnStmt([], span))
    return ir.Function(
        "target",
        [],
        [],
        ir.SeqStmts(statements, span),
        span,
        type=ir.FunctionType.AIV,
        attrs={"dual_aiv_dispatch": dual},
    )


CPP = """
static __aicore__ void target(int32_t lane) {
  using T = float;
  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  if (get_subblockid() == 0) {
    initialize_pipe();
    if (lane == 0) {
      consume_payload();
    }
  }
  #endif // __DAV_VEC__
  ptoas_auto_sync_tail();
  return;
}
"""


def test_runtime_control_removes_only_the_generated_outer_gate():
    other = CPP.replace("target", "other")
    result = _honor_runtime_aiv_lane_control(_function(), other + CPP)
    assert result.startswith(other)
    target = result[len(other) :]
    assert "if (get_subblockid() == 0)" not in target
    assert target.count("initialize_pipe();") == target.count("consume_payload();") == 1
    assert (
        target.index("set_vector_mask") < target.index("initialize_pipe();") < target.index("if (lane == 0)")
    )
    assert "#endif // __DAV_VEC__" in target
    assert "ptoas_auto_sync_tail();" in target


@pytest.mark.parametrize("dual,runtime", [(False, True), (True, False), (False, False)])
def test_single_lane_or_no_runtime_control_preserves_the_gate(dual, runtime):
    assert _honor_runtime_aiv_lane_control(_function(dual=dual, runtime=runtime), CPP) == CPP


def test_a2a3_preserves_existing_lane_handling():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    assert _honor_runtime_aiv_lane_control(_function(), CPP) == CPP


def test_unrecognized_or_absent_prologue_is_not_rewritten():
    code = CPP.replace("set_mask_norm();", "custom_mask_setup();")
    with pytest.raises(RuntimeError, match="unrecognized PTOAS single-lane guard"):
        _honor_runtime_aiv_lane_control(_function(), code)
    assert _honor_runtime_aiv_lane_control(_function(), "// no native prologue") == "// no native prologue"


def test_malformed_prologue_fails_before_native_compilation():
    code = CPP.replace("  #endif", "  unexpected_post_guard_work();\n  #endif")
    with pytest.raises(RuntimeError, match="malformed PTOAS no-split vector prologue guard"):
        _honor_runtime_aiv_lane_control(_function(), code)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
