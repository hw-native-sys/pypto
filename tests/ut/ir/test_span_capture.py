# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pinned spans win over the builder's frame capture.

The DSL parser pins a span per DSL call and the trace interpreter pins one
per instrumented row; without the pin check, ``_capture_call_span`` grabbed
two frames up — framework plumbing in traced execution, attributing IR
statements to the wrong file.
"""

import inspect

import pytest
from pypto.ir import IRBuilder
from pypto.ir.utils import reset_pinned_span, set_pinned_span, use_parser_span
from pypto.pypto_core import DataType, ir


def test_pinned_span_wins_for_builder_var():
    b = IRBuilder()
    pin = ir.Span("pinned.py", 42, 4)
    with b.function("f", ir.Span.unknown()):
        with use_parser_span(pin):
            v = b.var("x", ir.ScalarType(DataType.INT32))
    assert str(v.span.filename).endswith("pinned.py")
    assert v.span.begin_line == 42


def test_pinned_span_wins_for_builder_let():
    b = IRBuilder()
    pin = ir.Span("pinned.py", 7, 0)
    with b.function("f", ir.Span.unknown()):
        base = b.var("x", ir.ScalarType(DataType.INT32))
        with use_parser_span(pin):
            bound = b.let("y", base)
    assert str(bound.span.filename).endswith("pinned.py")
    assert bound.span.begin_line == 7


def test_pinned_span_wins_for_operator_overload():
    """Arithmetic built while a row pin is active carries the pinned location.

    The trace interpreter pins a span per instrumented row; without the pin
    check on the operator path, ``x + 1`` inside a traced kernel walked the
    Python stack and landed on the interpreter's frame instead.
    """
    b = IRBuilder()
    pin = ir.Span("user_kernel.py", 42, 0)
    with b.function("f", ir.Span.unknown()):
        x = b.var("x", ir.ScalarType(DataType.INT32))
        token = set_pinned_span(pin)
        try:
            pinned = x + 1
        finally:
            reset_pinned_span(token)
    assert str(pinned.span.filename).endswith("user_kernel.py")
    assert pinned.span.begin_line == 42


def test_unpinned_operator_overload_keeps_caller_location():
    """With no pin active, the operator path still captures the caller —
    the pure-DSL fallback must survive the pinned-span rerouting."""
    b = IRBuilder()
    with b.function("f", ir.Span.unknown()):
        x = b.var("x", ir.ScalarType(DataType.INT32))
        unpinned = _unpinned_add(x)
    assert str(unpinned.span.filename) == __file__
    # the span names the ``x + 1`` line in this file, not framework plumbing
    assert unpinned.span.begin_line == inspect.getsourcelines(_unpinned_add)[1] + 1


def _unpinned_add(x):
    return x + 1


def test_pinned_span_wins_for_direct_begin_end_surface():
    """The direct begin/end primitives resolve ``span=None`` through the pin.

    A value-oriented frontend (the trace interpreter) calls
    ``begin_for_loop``/``begin_while_loop``/``begin_if``/``begin_scope``
    without a span while a per-row pin is active; without the pin check
    every structural node degraded to ``Span.unknown()``.
    """
    b = IRBuilder()
    pin = ir.Span("user_kernel.py", 42, 0)
    with b.function("f", ir.Span.unknown()):
        i = b.var("i", ir.ScalarType(DataType.INT32))
        c = b.var("c", ir.ScalarType(DataType.BOOL))
        token = set_pinned_span(pin)
        try:
            b.begin_for_loop(i, 0, 4, 1)
            loop = b.end_for_loop()
            b.begin_while_loop(c)
            while_stmt = b.end_while_loop()
            b.begin_if(c)
            if_stmt = b.end_if()
            b.begin_scope(ir.ScopeKind.InCore)
            scope = b.end_scope()
        finally:
            reset_pinned_span(token)
    for stmt in (loop, while_stmt, if_stmt, scope):
        assert str(stmt.span.filename).endswith("user_kernel.py"), type(stmt).__name__
        assert stmt.span.begin_line == 42, type(stmt).__name__


def test_unpinned_direct_begin_end_keeps_unknown_span():
    """With no pin active, the direct surface stays ``Span.unknown()`` —
    it never walks the Python stack (frame capture would land on framework
    plumbing), matching the pre-pin contract for direct callers."""
    b = IRBuilder()
    with b.function("f", ir.Span.unknown()):
        i = b.var("i", ir.ScalarType(DataType.INT32))
        b.begin_for_loop(i, 0, 4, 1)
        loop = b.end_for_loop()
    assert loop.span.begin_line == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
