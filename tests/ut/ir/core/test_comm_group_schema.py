# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IR-level tests for the v2 ``WindowBuffer`` / ``CommDomainScopeStmt`` schema.

Post-redesign, ``WindowBuffer`` is a :class:`Var` subclass that mirrors
:class:`MemRef` exactly:

* Wraps a :class:`Var` of type :class:`PtrType` (``base``) — the
  allocation-identity token from ``pld.tensor.alloc_window_buffer``.
* Carries ``size`` (per-rank bytes), ``load_from_host`` / ``store_to_host``
  flags. **No** dtype, **no** name field — the runtime-unique identifier
  comes from the inherited ``Var.name_hint`` (taken from ``base.name_hint``).

``CommDomainScopeStmt`` wraps host_orch use sites of one comm domain's
slots; it carries a ``devices`` list (empty = "all devices") and an
ordered ``slots`` vector of :class:`WindowBuffer`s. Structural equality
follows the field descriptors of :class:`ScopeStmt` (``body``,
``name_hint``, ``attrs``) plus ``devices`` and ``slots``.
"""

import pytest
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    CommDomainScopeStmt,
    ConstInt,
    PtrType,
    SeqStmts,
    Span,
    Var,
    WindowBuffer,
    structural_equal,
)


def _empty_body() -> SeqStmts:
    return SeqStmts([], Span.unknown())


def _scope(devices: list[int], slots: list[WindowBuffer]) -> CommDomainScopeStmt:
    """Build a minimal CommDomainScopeStmt with an empty body — the body
    field is required for ScopeStmt structural equality but is orthogonal
    to the devices/slots semantics under test here."""
    return CommDomainScopeStmt(devices, slots, "", body=_empty_body(), span=Span.unknown())


def _const(value: int) -> ConstInt:
    return ConstInt(value, DataType.INT64, Span.unknown())


def _ptr(name: str) -> Var:
    return Var(name, PtrType(), Span.unknown())


# ---------------------------------------------------------------------------
# WindowBuffer
# ---------------------------------------------------------------------------


def test_window_buffer_minimal_construction():
    base = _ptr("data")
    wb = WindowBuffer(base, _const(256))
    # name_hint flows from the base Ptr Var (mirrors MemRef).
    assert wb.name_hint == "data"
    assert wb.base is base
    assert isinstance(wb.size, ConstInt)
    assert wb.size.value == 256
    assert wb.load_from_host is False
    assert wb.store_to_host is False


def test_window_buffer_load_store_to_host_flags():
    """v2 schema: load/store_to_host are bool flags; the actual host tensor
    binding (if any) is recorded on the alloc op, not here."""
    base = _ptr("lut")
    wb = WindowBuffer(base, _const(64), load_from_host=True, store_to_host=True)
    assert wb.load_from_host is True
    assert wb.store_to_host is True


def test_window_buffer_is_var_subclass():
    """Mirror MemRef: WindowBuffer inherits from Var so visitor / mutator
    machinery treats it the same as any other Var."""
    base = _ptr("data")
    wb = WindowBuffer(base, _const(64))
    assert isinstance(wb, Var)


# ---------------------------------------------------------------------------
# CommDomainScopeStmt structural equality
# ---------------------------------------------------------------------------


def test_comm_domain_scope_empty_devices_means_all():
    """``devices == []`` is the convention for "covers all DistributedConfig.device_ids"."""
    s = _scope([], [WindowBuffer(_ptr("data"), _const(64))])
    assert list(s.devices) == []


def test_comm_domain_scope_explicit_device_subset():
    s = _scope([0, 1, 2], [WindowBuffer(_ptr("data"), _const(64))])
    assert list(s.devices) == [0, 1, 2]


def test_comm_domain_scope_structural_equal_when_slot_var_is_shared():
    """Two CommDomainScopeStmts whose ``slots`` contain the **same**
    WindowBuffer Var instances compare structurally equal.

    Mirrors :class:`MemRef` semantics: independently-constructed Var
    instances each have a unique identity, so structural equality requires
    sharing the underlying Var (same ``shared_ptr``). This is the form the
    comm-collection pass produces — slots in the synthesised scope and the
    ``DistributedTensorType.window_buffer`` references on every consuming
    view both alias the same WindowBuffer.
    """
    wb = WindowBuffer(_ptr("data"), _const(256))
    s1 = _scope([], [wb])
    s2 = _scope([], [wb])
    assert structural_equal(s1, s2)


def test_comm_domain_scope_structural_equal_for_independent_slot_vars_under_auto_mapping():
    """Two CommDomainScopeStmts whose ``slots`` are *separately constructed*
    compare structurally equal under ``enable_auto_mapping=True`` —
    corresponding Vars are matched by their position in the structure, so
    distinct UniqueIds don't break equality when the surrounding shape,
    base ``name_hint``, ``size``, and host-staging flags all match.

    Mirrors :class:`MemRef`'s auto-mapping semantics: the default identity
    path requires shared Var instances; the auto-mapping path is the
    structural-isomorphism check used when comparing IR produced by
    independent runs.
    """
    s1 = _scope([], [WindowBuffer(_ptr("data"), _const(256))])
    s2 = _scope([], [WindowBuffer(_ptr("data"), _const(256))])
    assert structural_equal(s1, s2, enable_auto_mapping=True)


def test_comm_domain_scope_structural_not_equal_when_devices_differ():
    """Sharing the slot Var isolates the ``devices``-only difference."""
    wb = WindowBuffer(_ptr("data"), _const(64))
    s_all = _scope([], [wb])
    s_subset = _scope([0, 1], [wb])
    assert not structural_equal(s_all, s_subset)


def test_comm_domain_scope_structural_not_equal_when_subsets_differ():
    wb = WindowBuffer(_ptr("data"), _const(64))
    a = _scope([0, 1], [wb])
    b = _scope([2, 3], [wb])
    assert not structural_equal(a, b)


def test_comm_domain_scope_structural_not_equal_when_slots_differ():
    """Same base Var but different ``size`` → distinct slot Vars → distinct scopes."""
    base = _ptr("data")
    a = _scope([], [WindowBuffer(base, _const(64))])
    b = _scope([], [WindowBuffer(base, _const(128))])
    assert not structural_equal(a, b)


def test_comm_domain_scope_structural_not_equal_when_load_flag_differs():
    base = _ptr("data")
    a = _scope([], [WindowBuffer(base, _const(64), load_from_host=False)])
    b = _scope([], [WindowBuffer(base, _const(64), load_from_host=True)])
    assert not structural_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
