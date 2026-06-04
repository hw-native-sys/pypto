# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Framework wire-up tests for :class:`WindowBuffer`.

After N2, ``WindowBuffer`` is a :class:`Var` subclass with its own
``ObjectKind``. The dispatch tables in ``ExprFunctor`` / ``IRVisitor`` /
``IRMutator`` must each have an explicit ``WindowBuffer`` entry — otherwise
any pass that visits IR containing a constructed ``WindowBuffer`` would
fall through and raise ``TypeError("Unknown expression type")``.

These tests construct a ``WindowBuffer`` directly and drive it through the
base ``IRVisitor`` and ``IRMutator`` to confirm the wire-up is in place
before the ``MaterializeCommDomainScopes`` pass (which materialises ``WindowBuffer``
instances during normal compilation) lands.
"""

import pytest
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    ConstInt,
    IRMutator,
    IRVisitor,
    PtrType,
    Span,
    Var,
    WindowBuffer,
)


def _const(value: int) -> ConstInt:
    return ConstInt(value, DataType.INT64, Span.unknown())


def _ptr(name: str) -> Var:
    return Var(name, PtrType(), Span.unknown())


def test_visitor_dispatch_does_not_throw():
    """``IRVisitor::VisitExpr(window_buffer)`` must route to the dedicated
    overload, not fall through to the catch-all ``TypeError``."""
    wb = WindowBuffer(_ptr("data"), _const(256))
    visitor = IRVisitor()
    # Default impl is a no-op; the key invariant is "no TypeError".
    visitor.visit_expr(wb)


def test_visitor_custom_subclass_sees_window_buffer():
    """A Python-side subclass overriding ``visit_window_buffer`` must be
    invoked when the dispatcher sees a ``WindowBuffer`` instance."""

    class Recorder(IRVisitor):
        def __init__(self):
            super().__init__()
            self.seen: list[WindowBuffer] = []

        def visit_window_buffer(self, op):  # noqa: D401 - test hook
            self.seen.append(op)

    wb = WindowBuffer(_ptr("data"), _const(256))
    rec = Recorder()
    rec.visit_expr(wb)
    assert len(rec.seen) == 1
    assert rec.seen[0] is wb


def test_mutator_identity_returns_same_instance():
    """The base ``IRMutator`` is copy-on-write: if no child changed, it must
    return the original ``WindowBuffer`` instance, not mint a fresh one."""
    wb = WindowBuffer(_ptr("data"), _const(256))
    mutator = IRMutator()
    result = mutator.visit_expr(wb)
    assert result is wb


def test_mutator_remaps_base_via_var_substitution():
    """When the mutator's ``var_remap_`` (seeded through a subclass) swaps
    the underlying Ptr ``base``, the WindowBuffer override must mint a fresh
    ``WindowBuffer`` referencing the substituted base."""

    old_base = _ptr("data")
    new_base = _ptr("data")  # Different Var identity, same name_hint
    wb = WindowBuffer(old_base, _const(256))

    class SubstituteBaseMutator(IRMutator):
        def visit_var(self, op):
            if op is old_base:
                return new_base
            return super().visit_var(op)

    fresh = SubstituteBaseMutator().visit_expr(wb)
    assert fresh is not wb
    assert isinstance(fresh, WindowBuffer)
    assert fresh.base is new_base
    assert fresh.load_from_host == wb.load_from_host
    assert fresh.store_to_host == wb.store_to_host


def test_mutator_preserves_host_staging_flags():
    """Host-staging flags must round-trip through identity mutation."""
    wb = WindowBuffer(_ptr("data"), _const(64), load_from_host=True, store_to_host=True)
    mutator = IRMutator()
    result = mutator.visit_expr(wb)
    assert result is wb  # identity (no children changed)
    assert isinstance(result, WindowBuffer)
    assert result.load_from_host is True
    assert result.store_to_host is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
