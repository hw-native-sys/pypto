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
    p.profile_id = "synthetic-v1"
    p.toolchain_id = "test-compiler-v1"
    p.supported = True
    for name, value in kwargs.items():
        setattr(p, name, value)
    return p


@pytest.mark.parametrize(
    "bounds,state,count,post",
    [
        ((0, 8, 1), "Eligible", 8, 8),
        ((0, 4, 0), "Ineligible", None, None),
        ((8, 0, -1), "Ineligible", 8, 0),
        ((32760, 32767, 4), "Ineligible", 2, 32768),
        ((0, 32767, 1), "Eligible", 32767, 32767),
        ((0, 32768, 1), "Ineligible", 32768, 32768),
        ((8, 0, 1), "Eligible", 0, 8),
        ((0, 8, -1), "Eligible", 0, 0),
        ((-(2**63), 2**63 - 1, 1), "Ineligible", None, None),
    ],
)
def test_bounds(bounds, state, count, post):
    result = ir.analyze_hardware_loop(loop(*bounds), profile())
    assert result.state.name == state
    assert result.trip_count == count
    assert result.post_iv == post
    assert result.reason
    assert result.profile_id == "synthetic-v1"
    assert result.toolchain_id == "test-compiler-v1"


@pytest.mark.parametrize("dynamic_arg", ["upper", "step"])
def test_dynamic_is_unknown_and_require_fails(dynamic_arg):
    n = ir.Var("n", ir.ScalarType(DataType.INDEX), ir.Span.unknown())
    op = loop(**{dynamic_arg: n}, policy=ir.HardwareLoopPolicy.Require)
    result = ir.analyze_hardware_loop(op, profile())
    assert result.state == ir.HardwareLoopState.Unknown
    with pytest.raises(ValueError, match=r"Unknown.*root/loop0.*synthetic-v1.*test-compiler-v1"):
        ir.verify_hardware_loop_requirement(op, profile())


def test_eligible_is_not_evidence():
    op = loop(policy=ir.HardwareLoopPolicy.Require)
    with pytest.raises(ValueError, match="evidence"):
        ir.verify_hardware_loop_requirement(op, profile())
    # Auto never requests evidence. Static empty Require is exempt as well.
    ir.verify_hardware_loop_requirement(loop(), profile(supported=False))
    ir.verify_hardware_loop_requirement(
        loop(8, 0, policy=ir.HardwareLoopPolicy.Require), profile(supported=False)
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
    outer = loop(4, 0, body=inner, policy=ir.HardwareLoopPolicy.Require)
    ir.verify_hardware_loop_requirement(outer, profile())


def test_profile_version_and_unknown_calls():
    first = ir.analyze_hardware_loop(loop(), profile())
    second = ir.analyze_hardware_loop(loop(), profile(toolchain_id="test-compiler-v2"))
    assert first.toolchain_id != second.toolchain_id
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
