# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the typed ScopeStmt class hierarchy."""

import pypto.language as pl
import pytest
from pypto.pypto_core import ir, passes


def _empty_body():
    return ir.SeqStmts([], ir.Span("test", 1, 0))


def _span():
    return ir.Span("test", 1, 0)


# ─── ScopeKind values ────────────────────────────────────────────────────────


def test_hierarchy_scope_kind_exists():
    """ScopeKind.Hierarchy is a valid enum value."""
    assert hasattr(ir.ScopeKind, "Hierarchy")


def test_scope_kinds_are_distinct():
    """Each surviving ScopeKind is distinct."""
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.Cluster
    assert ir.ScopeKind.Hierarchy != ir.ScopeKind.Spmd
    assert ir.ScopeKind.Cluster != ir.ScopeKind.Spmd


# ─── Construction with derived classes (issue #1047) ────────────────────────


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
    """Different scope kinds compare as unequal."""
    s_cluster = ir.ClusterScopeStmt(body=_empty_body(), span=_span())
    s_hier = ir.HierarchyScopeStmt(level=ir.Level.HOST, body=_empty_body(), span=_span())
    with pytest.raises(ValueError):
        ir.assert_structural_equal(s_cluster, s_hier)


# ─── Python printer ──────────────────────────────────────────────────────────


def test_printer_hierarchy_scope():
    body = _empty_body()
    scope = ir.HierarchyScopeStmt(level=ir.Level.HOST, role=ir.Role.Worker, body=body, span=_span())
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(" in printed
    assert "Level.HOST" in printed
    assert "Role.Worker" in printed


def test_printer_core_group_scope():
    body = _empty_body()
    scope = ir.HierarchyScopeStmt(level=ir.Level.CORE_GROUP, body=body, span=_span())
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(level=pl.Level.CORE_GROUP)" in printed


def test_printer_core_group_scope_with_split():
    body = _empty_body()
    scope = ir.HierarchyScopeStmt(
        level=ir.Level.CORE_GROUP, split=ir.SplitMode.UP_DOWN, body=body, span=_span()
    )
    func = ir.Function("test_fn", [], [], scope, _span())
    printed = str(func)
    assert "pl.at(level=pl.Level.CORE_GROUP" in printed
    assert "pl.split(pl.SplitMode.UP_DOWN)" in printed


def test_scope_stmt_core_group_with_split():
    s = ir.HierarchyScopeStmt(
        level=ir.Level.CORE_GROUP, split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span()
    )
    assert s.scope_kind == ir.ScopeKind.Hierarchy
    assert s.split == ir.SplitMode.UP_DOWN


def test_structural_equal_core_group_with_split():
    s1 = ir.HierarchyScopeStmt(
        level=ir.Level.CORE_GROUP, split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span()
    )
    s2 = ir.HierarchyScopeStmt(
        level=ir.Level.CORE_GROUP, split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span()
    )
    ir.assert_structural_equal(s1, s2)


def test_structural_equal_core_group_different_split():
    s1 = ir.HierarchyScopeStmt(
        level=ir.Level.CORE_GROUP, split=ir.SplitMode.UP_DOWN, body=_empty_body(), span=_span()
    )
    s2 = ir.HierarchyScopeStmt(
        level=ir.Level.CORE_GROUP, split=ir.SplitMode.LEFT_RIGHT, body=_empty_body(), span=_span()
    )
    with pytest.raises(ValueError):
        ir.assert_structural_equal(s1, s2)


# ─── Outline pass ────────────────────────────────────────────────────────────


def test_outline_hierarchy_works_with_core_group_program():
    """OutlineHierarchyScopes outlines CORE_GROUP scopes into Function(InCore)."""

    @pl.program
    class P:
        @pl.function
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                y = pl.add(x, x)
            return y

    After = passes.outline_hierarchy_scopes()(P)
    assert After is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
