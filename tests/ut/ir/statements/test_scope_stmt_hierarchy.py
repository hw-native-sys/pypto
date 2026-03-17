# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for ScopeStmt Hierarchy kind (Step 03)."""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir, passes


def _empty_body():
    return ir.SeqStmts([], ir.Span("test", 1, 0))


def _span():
    return ir.Span("test", 1, 0)


# ─── ScopeKind.Hierarchy value ────────────────────────────────────────────────


def test_hierarchy_scope_kind_exists():
    """ScopeKind.Hierarchy is a valid enum value."""
    assert hasattr(ir.ScopeKind, "Hierarchy")


def test_hierarchy_scope_kind_distinct():
    """Hierarchy is distinct from existing ScopeKind values."""
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.InCore
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.AutoInCore
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.Cluster


# ─── ScopeStmt backward compatibility ────────────────────────────────────────


def test_scope_stmt_backward_compat_3arg():
    """Existing 3-arg ScopeStmt(InCore, body, span) still works."""
    s = ir.ScopeStmt(ir.ScopeKind.InCore, _empty_body(), _span())
    assert s.scope_kind == ir.ScopeKind.InCore
    assert s.level is None
    assert s.role is None


def test_scope_stmt_backward_compat_cluster():
    """Existing ScopeStmt(Cluster) still works."""
    s = ir.ScopeStmt(ir.ScopeKind.Cluster, _empty_body(), _span())
    assert s.scope_kind == ir.ScopeKind.Cluster
    assert s.level is None
    assert s.role is None


# ─── ScopeStmt with Hierarchy kind ───────────────────────────────────────────


def test_scope_stmt_hierarchy_with_level_and_role():
    """ScopeStmt(Hierarchy) carries level and role."""
    s = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    assert s.scope_kind == ir.ScopeKind.Hierarchy
    assert s.level == ir.Level.HOST
    assert s.role == ir.Role.Worker


def test_scope_stmt_hierarchy_orchestrator():
    """Orchestrator role at cluster level."""
    s = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.POD,
        role=ir.Role.Orchestrator,
    )
    assert s.role == ir.Role.Orchestrator
    # POD and CLUSTER_0 share the same Linqu level
    assert s.level is not None
    assert ir.level_to_linqu_level(s.level) == 4


def test_scope_stmt_hierarchy_level_only():
    """Hierarchy scope with level but no explicit role."""
    s = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.GLOBAL,
    )
    assert s.level == ir.Level.GLOBAL
    assert s.role is None


def test_scope_stmt_hierarchy_global():
    """Global coordinator hierarchy scope."""
    s = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.GLOBAL,
        role=ir.Role.Orchestrator,
    )
    assert s.level == ir.Level.GLOBAL
    assert ir.level_to_linqu_level(s.level) == 7


# ─── structural_equal ────────────────────────────────────────────────────────


def test_structural_equal_hierarchy_scope():
    """structural_equal compares level and role on ScopeStmt."""
    s1 = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    s2 = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    ir.assert_structural_equal(s1, s2)


def test_structural_equal_different_level():
    """structural_equal detects different levels."""
    s1 = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    s2 = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.GLOBAL,
        role=ir.Role.Worker,
    )
    with pytest.raises(Exception):
        ir.assert_structural_equal(s1, s2)


def test_structural_equal_different_role():
    """structural_equal detects different roles."""
    s1 = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    s2 = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Orchestrator,
    )
    with pytest.raises(Exception):
        ir.assert_structural_equal(s1, s2)


def test_structural_equal_none_vs_set_level():
    """structural_equal detects None vs set level on ScopeStmt."""
    s_none = ir.ScopeStmt(ir.ScopeKind.InCore, _empty_body(), _span())
    s_set = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        _empty_body(),
        _span(),
        level=ir.Level.HOST,
    )
    with pytest.raises(Exception):
        ir.assert_structural_equal(s_none, s_set)


# ─── Python printer ──────────────────────────────────────────────────────────


def test_printer_hierarchy_scope():
    """Python printer renders Hierarchy scope as pl.at(...)."""
    body = _empty_body()
    scope = ir.ScopeStmt(
        ir.ScopeKind.Hierarchy,
        body,
        _span(),
        level=ir.Level.HOST,
        role=ir.Role.Worker,
    )
    # Wrap in a function to print
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(" in printed
    assert "Level.HOST" in printed
    assert "Role.Worker" in printed


def test_printer_incore_scope_unchanged():
    """Python printer still renders InCore scope as pl.incore()."""
    body = _empty_body()
    scope = ir.ScopeStmt(ir.ScopeKind.InCore, body, _span())
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.incore()" in printed


# ─── Outline pass safety ─────────────────────────────────────────────────────


def test_outline_incore_works_with_normal_program():
    """OutlineIncoreScopes works normally on programs without Hierarchy scopes."""

    @pl.program
    class P:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.incore():
                y = pl.add(x, x)
            return y

    After = passes.outline_incore_scopes()(P)
    assert After is not None


def test_scope_outliner_ignores_hierarchy_kind():
    """ScopeOutliner (used by OutlineIncoreScopes) only targets its configured
    ScopeKind and naturally ignores Hierarchy scopes via the ScopeKind check."""
    # The ScopeOutliner matches on target_scope_kind_ (InCore or Cluster).
    # ScopeKind::Hierarchy (value 3) != InCore (0) != Cluster (2), so
    # the outliner's VisitStmt_ will skip it via: if (scope_kind_ != target_) return.
    # We verify this property at the enum level since we can't inject a Hierarchy
    # scope via the DSL parser yet (pl.at() parsing is Step 04).
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.InCore
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.Cluster
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.AutoInCore


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
