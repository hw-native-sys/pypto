# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qualification is a conservative, profile-dependent proof, not hardware evidence."""

import pytest
from pypto import DataType, ir, passes


def loop(
    lower: int | ir.Expr = 0,
    upper: int | ir.Expr = 8,
    step: int | ir.Expr = 1,
    body=None,
    policy=None,
    iter_args=None,
):
    span = ir.Span.unknown()

    def expr(value):
        return value if isinstance(value, ir.Expr) else ir.ConstInt(value, DataType.INDEX, span)

    args = iter_args or []
    return ir.ForStmt(
        ir.Var("i", ir.ScalarType(DataType.INDEX), span),
        expr(lower),
        expr(upper),
        expr(step),
        args,
        body if body is not None else ir.SeqStmts([], span),
        [ir.Var("out", a.type, span) for a in args],
        span,
        hardware_loop_policy=policy or ir.HardwareLoopPolicy.Auto,
    )


def profile(**kwargs):
    p = ir.HardwareLoopProfile()
    p.supported = True
    for name, value in kwargs.items():
        setattr(p, name, value)
    return p


@pytest.mark.parametrize(
    "bounds,state,count,post",
    [
        ((0, 8, 1), "Eligible", 8, 8),
        ((0, 4, 0), "Ineligible", None, None),
        ((8, 0, -1), "Ineligible", None, None),
        ((32760, 32767, 4), "Ineligible", None, None),
        ((0, 32767, 1), "Eligible", 32767, 32767),
        ((0, 32768, 1), "Eligible", 32768, 32768),
        ((0, 65535, 1), "Eligible", 65535, 65535),
        ((0, 65536, 1), "Ineligible", 65536, 65536),
        ((8, 0, 1), "Ineligible", None, None),
        ((0, 8, -1), "Ineligible", None, None),
        ((-(2**63), 2**63 - 1, 1), "Ineligible", None, None),
    ],
)
def test_bounds(bounds, state, count, post):
    result = ir.analyze_hardware_loop(loop(*bounds), profile())
    assert result.state.name == state
    assert result.trip_count == count
    assert result.post_iv == post
    assert result.reason
    assert result.loop_id == "root/loop0"


def test_unsigned_counter_range_is_independent_of_trip_limit():
    p = profile(max_trip_count=70000)
    largest = ir.analyze_hardware_loop(loop(upper=65535), p)
    assert largest.state == ir.HardwareLoopState.Eligible
    assert largest.post_iv == 65535
    overflow = ir.analyze_hardware_loop(loop(upper=65536), p)
    assert overflow.state == ir.HardwareLoopState.Ineligible
    assert overflow.reason == "HWL_COUNTER_RANGE"


def test_trip_limit_can_be_stricter_than_unsigned_counter_range():
    result = ir.analyze_hardware_loop(loop(upper=65535), profile(max_trip_count=65534))
    assert result.state == ir.HardwareLoopState.Ineligible
    assert result.reason == "HWL_TRIP_COUNT"


def test_dynamic_require_step_fails():
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    op = loop(step=n, policy=ir.HardwareLoopPolicy.Require)
    result = ir.analyze_hardware_loop(op, profile())
    assert result.state == ir.HardwareLoopState.Ineligible
    with pytest.raises(ValueError, match=r"Ineligible.*root/loop0"):
        ir.verify_hardware_loop_requirement(op, profile())


@pytest.mark.parametrize("dtype", [DataType.INDEX, DataType.INT32, DataType.UINT16, DataType.UINT64])
def test_dynamic_integer_stop_qualifies_without_range_proof(dtype):
    n = ir.Var("n", ir.ScalarType(dtype), ir.Span.unknown())
    op = loop(upper=n, policy=ir.HardwareLoopPolicy.Require)
    result = ir.analyze_hardware_loop(op, profile())
    assert result.state == ir.HardwareLoopState.Eligible
    assert result.upper is None and result.trip_count is None and result.post_iv is None
    # Dynamic acceptance must not fabricate static counts or hardware evidence.
    with pytest.raises(ValueError, match="backend evidence missing"):
        ir.verify_hardware_loop_requirement(op, profile())


@pytest.mark.parametrize("dtype", [DataType.FP16, DataType.FP32, DataType.BOOL])
def test_noninteger_stop_is_ineligible(dtype):
    n = ir.Var("n", ir.ScalarType(dtype), ir.Span.unknown())
    result = ir.analyze_hardware_loop(loop(upper=n), profile())
    assert result.state == ir.HardwareLoopState.Ineligible
    assert result.reason == "HWL_STOP_TYPE"

    required = loop(upper=n, policy=ir.HardwareLoopPolicy.Require)
    function = ir.Function("invalid", [n], [], ir.RegScopeStmt("", required, n.span), n.span)
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.HardwareLoopContractValid)
    diagnostics = passes.PropertyVerifierRegistry.verify(
        properties, ir.Program([function], "Invalid", n.span)
    )
    assert any("integer scalar stop" in diagnostic.message for diagnostic in diagnostics)


def test_dynamic_stop_still_checks_nested_depth_and_control_flow():
    span = ir.Span.unknown()
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), span)
    inner = loop(upper=n)
    assert ir.analyze_hardware_loop(loop(body=inner), profile()).state == ir.HardwareLoopState.Eligible
    assert ir.analyze_hardware_loop(loop(body=inner), profile(max_depth=1)).reason == "HWL_DEPTH"
    op = loop(upper=n, body=ir.ReturnStmt([], span))
    assert ir.analyze_hardware_loop(op, profile()).reason == "HWL_CONTROL_FLOW"
    assert ir.analyze_hardware_loop(inner, profile(supported=False)).reason == "HWL_UNSUPPORTED_PROFILE"
    # A dynamic trip count is accepted even when no static limit proof exists.
    assert ir.analyze_hardware_loop(inner, profile(max_trip_count=0)).state == ir.HardwareLoopState.Eligible


def test_eligible_is_not_evidence():
    op = loop(policy=ir.HardwareLoopPolicy.Require)
    with pytest.raises(ValueError, match="evidence"):
        ir.verify_hardware_loop_requirement(op, profile())
    # Auto never requests evidence. Static empty Require is exempt as well.
    ir.verify_hardware_loop_requirement(loop(), profile(supported=False))
    ir.verify_hardware_loop_requirement(
        loop(0, 0, policy=ir.HardwareLoopPolicy.Require), profile(supported=False)
    )


def test_body_and_profile_changes_are_reanalyzed():
    inner = loop()
    outer = loop(body=inner)
    assert ir.analyze_hardware_loop(outer, profile(max_depth=1)).state == ir.HardwareLoopState.Ineligible
    assert ir.analyze_hardware_loop(outer, profile(max_depth=2)).state == ir.HardwareLoopState.Eligible
    assert ir.analyze_hardware_loop(loop(), profile(max_depth=1)).state == ir.HardwareLoopState.Eligible
    # Resource accounting is static; large trip counts never expand the body.
    assert ir.analyze_hardware_loop(outer, profile(max_body_nodes=0)).state == ir.HardwareLoopState.Ineligible


def test_hardware_loop_nesting_limit_is_four():
    nested = loop()
    for _ in range(3):
        nested = loop(body=nested)

    assert ir.analyze_hardware_loop(nested, profile()).state == ir.HardwareLoopState.Eligible
    assert ir.analyze_hardware_loop(nested, profile(max_depth=3)).state == ir.HardwareLoopState.Ineligible

    fifth = loop(body=nested)
    assert ir.analyze_hardware_loop(fifth, profile()).state == ir.HardwareLoopState.Ineligible


def test_required_hardware_loop_requires_nested_qualification():
    inner_auto = loop()
    outer_required = loop(body=inner_auto, policy=ir.HardwareLoopPolicy.Require)
    result = ir.analyze_hardware_loop(outer_required, profile())
    assert result.state == ir.HardwareLoopState.Eligible

    span = ir.Span.unknown()
    unknown_call = ir.Call(ir.Op("unknown_nested_effect"), [], span)
    inner_unknown = loop(body=ir.EvalStmt(unknown_call, span))
    outer_required = loop(body=inner_unknown, policy=ir.HardwareLoopPolicy.Require)
    result = ir.analyze_hardware_loop(outer_required, profile())
    assert result.state == ir.HardwareLoopState.Unknown


@pytest.mark.parametrize("start,step", [(1, 1), (0, 2)])
def test_nested_auto_loop_must_have_hardware_bounds(start, step):
    inner = loop(start, 8, step)
    outer = loop(body=inner, policy=ir.HardwareLoopPolicy.Require)
    assert ir.analyze_hardware_loop(inner, profile()).state == ir.HardwareLoopState.Ineligible
    # The parent conservatively reports an unqualified nested body as Unknown.
    assert ir.analyze_hardware_loop(outer, profile()).state != ir.HardwareLoopState.Eligible


def test_illegal_control_flow():
    op = loop(body=ir.ReturnStmt([], ir.Span.unknown()))
    assert ir.analyze_hardware_loop(op, profile()).state == ir.HardwareLoopState.Ineligible


def test_context_auto_report_is_optional():
    with passes.PassContext([], report_hardware_loops=True) as context:
        assert context.get_report_hardware_loops()
        assert context.get_hardware_loop_reports() == []


def test_contract_property_is_registered():
    assert passes.IRProperty.HardwareLoopContractValid is not None


def test_empty_outer_does_not_require_inner_evidence():
    inner = loop(policy=ir.HardwareLoopPolicy.Require)
    outer = loop(0, 0, body=inner, policy=ir.HardwareLoopPolicy.Require)
    ir.verify_hardware_loop_requirement(outer, profile())


def test_unknown_calls_are_not_eligible():
    span = ir.Span.unknown()
    call = ir.Call(ir.Op("unknown_effect"), [], span)
    op = loop(body=ir.EvalStmt(call, span))
    assert ir.analyze_hardware_loop(op, profile()).state == ir.HardwareLoopState.Unknown


@pytest.mark.parametrize("carry_kind", ["scalar", "reg", "tile"])
def test_carry_capabilities(carry_kind):
    span = ir.Span.unknown()
    types = {
        "scalar": ir.ScalarType(DataType.INDEX),
        "reg": ir.RegType(64, DataType.FP32, ir.RegKind.VecReg),
        "tile": ir.TileType([64], DataType.FP32),
    }
    typ = types[carry_kind]
    value = ir.Var("initial", typ, span)
    carry = ir.IterArg("carry", typ, value, span)
    op = loop(iter_args=[carry], body=ir.YieldStmt([carry], span))
    p = profile(**{f"allow_{carry_kind}_carry": False})
    assert ir.analyze_hardware_loop(op, p).state == ir.HardwareLoopState.Ineligible
    setattr(p, f"allow_{carry_kind}_carry", True)
    assert ir.analyze_hardware_loop(op, p).state == ir.HardwareLoopState.Eligible


def test_nested_unknown_initializer_is_not_eligible():
    span = ir.Span.unknown()
    typ = ir.ScalarType(DataType.INDEX)
    initial = ir.Call(ir.Op("unknown_initializer"), [], typ, span)
    carry = ir.IterArg("carry", typ, initial, span)
    inner = loop(iter_args=[carry], body=ir.YieldStmt([ir.ConstInt(0, DataType.INDEX, span)], span))
    outer = loop(body=inner)
    assert ir.analyze_hardware_loop(outer, profile(max_depth=2)).state == ir.HardwareLoopState.Unknown


@pytest.mark.parametrize("bounds", [(0, 8, 1), (32760, 32767, 4)])
def test_auto_analysis_does_not_rewrite(bounds):
    op = loop(*bounds)
    before = ir.serialize(op)
    ir.analyze_hardware_loop(op, profile())
    ir.verify_hardware_loop_requirement(op, profile())
    assert ir.serialize(op) == before


def test_context_manager_returns_active_context():
    original = passes.PassContext([], report_hardware_loops=True)
    with original as context:
        assert context is original
        assert passes.PassContext.current() is context


@pytest.mark.parametrize("start,step", [(1, 1), (0, 2), (0, -1)])
def test_required_bounds_cannot_be_bypassed_by_empty_loop(start, step):
    op = loop(start, 0, step, policy=ir.HardwareLoopPolicy.Require)
    result = ir.analyze_hardware_loop(op, profile())
    assert result.state == ir.HardwareLoopState.Ineligible
    assert result.reason == "HWL_CANONICAL_BOUNDS"
    with pytest.raises(ValueError, match="HWL_CANONICAL_BOUNDS"):
        ir.verify_hardware_loop_requirement(op, profile())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
