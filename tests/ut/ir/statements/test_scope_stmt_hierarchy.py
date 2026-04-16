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


# ─── Construction with derived classes (issue #1047) ────────────────────────


def test_in_core_scope_construction():
    """InCoreScopeStmt construction works."""
    s = ir.InCoreScopeStmt(body=_empty_body(), span=_span())
    assert s.scope_kind == ir.ScopeKind.InCore
    assert isinstance(s, ir.ScopeStmt)


def test_cluster_scope_construction():
    """ClusterScopeStmt construction works."""
    s = ir.ClusterScopeStmt(body=_empty_body(), span=_span())
    assert s.scope_kind == ir.ScopeKind.Cluster
    assert isinstance(s, ir.ScopeStmt)


# ─── HierarchyScopeStmt ─────────────────────────────────────────────────────


def test_scope_stmt_hierarchy_with_level_and_role():
    """HierarchyScopeStmt carries level and role."""
    s = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=_empty_body(), span=_span())
    assert s.scope_kind == ir.ScopeKind.Hierarchy
    assert s.level == ir.Level.HOST
    assert s.role == ir.Role.Worker


def test_scope_stmt_hierarchy_orchestrator():
    """Orchestrator role at cluster level."""
    s = ir.HierarchyScopeStmt(level=ir.Level.POD, role=ir.Role.Orchestrator, body=_empty_body(), span=_span())
    assert s.role == ir.Role.Orchestrator
    assert ir.level_to_linqu_level(s.level) == 4


def test_scope_stmt_hierarchy_level_only():
    """Hierarchy scope with level but no explicit role."""
    s = ir.HierarchyScopeStmt(level=ir.Level.GLOBAL, body=_empty_body(), span=_span())
    assert s.level == ir.Level.GLOBAL
    assert s.role is None


def test_scope_stmt_hierarchy_global():
    """Global coordinator hierarchy scope."""
    s = ir.HierarchyScopeStmt(
        level=ir.Level.GLOBAL, role=ir.Role.Orchestrator, body=_empty_body(), span=_span()
    )
    assert s.level == ir.Level.GLOBAL
    assert ir.level_to_linqu_level(s.level) == 7


# ─── structural_equal ────────────────────────────────────────────────────────


def test_structural_equal_hierarchy_scope():
    s1 = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=_empty_body(), span=_span())
    s2 = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=_empty_body(), span=_span())
    ir.assert_structural_equal(s1, s2)


def test_structural_equal_different_level():
    s1 = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=_empty_body(), span=_span())
    s2 = ir.HierarchyScopeStmt(level=ir.Level.GLOBAL, role=ir.Role.Worker, body=_empty_body(), span=_span())
    with pytest.raises(ValueError):
        ir.assert_structural_equal(s1, s2)


def test_structural_equal_different_role():
    s1 = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=_empty_body(), span=_span())
    s2 = ir.HierarchyScopeStmt(
        level=ir.Level.HOST, role=ir.Role.Orchestrator, body=_empty_body(), span=_span()
    )
    with pytest.raises(ValueError):
        ir.assert_structural_equal(s1, s2)


def test_structural_equal_different_kinds():
    """Different scope kinds (InCore vs Hierarchy) compare as unequal."""
    s_in = ir.InCoreScopeStmt(body=_empty_body(), span=_span())
    s_hier = ir.HierarchyScopeStmt(level=ir.Level.HOST, body=_empty_body(), span=_span())
    with pytest.raises(Exception):
        ir.assert_structural_equal(s_in, s_hier)


# ─── Python printer ──────────────────────────────────────────────────────────


def test_printer_hierarchy_scope():
    body = _empty_body()
    scope = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=body, span=_span())
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(" in printed
    assert "Level.HOST" in printed
    assert "Role.Worker" in printed


def test_printer_incore_scope_unchanged():
    body = _empty_body()
    scope = ir.InCoreScopeStmt(body=body, span=_span())
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(level=pl.Level.CORE_GROUP)" in printed


def test_printer_incore_scope_with_split():
    body = _empty_body()
    scope = ir.InCoreScopeStmt(split=ir.SplitMode.UP_DOWN, body=body, span=_span())
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.UP_DOWN)" in printed


def test_scope_stmt_incore_with_split():
    s = ir.InCoreScopeStmt(split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span())
    assert s.scope_kind == ir.ScopeKind.InCore
    assert s.split == ir.SplitMode.UP_DOWN


def test_structural_equal_incore_with_split():
    s1 = ir.InCoreScopeStmt(split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span())
    s2 = ir.InCoreScopeStmt(split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span())
    ir.assert_structural_equal(s1, s2)


def test_structural_equal_incore_different_split():
    s1 = ir.InCoreScopeStmt(split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span())
    s2 = ir.InCoreScopeStmt(split=ir.SplitMode.LEFT_RIGHT, body=_empty_body(), span=_span())
    with pytest.raises(ValueError):
        ir.assert_structural_equal(s1, s2)


# ─── Outline pass safety ─────────────────────────────────────────────────────


def test_outline_incore_works_with_normal_program():
    """OutlineIncoreScopes works normally on programs without Hierarchy scopes."""

    @pl.program
    class P:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
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
