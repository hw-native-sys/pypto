# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for parsing pl.at(level=..., role=...) (Step 04)."""

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.pypto_core import ir


def _find_scope_stmt(stmt):
    """Recursively find first ScopeStmt in an IR tree."""
    if isinstance(stmt, ir.ScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _find_scope_stmt(s)
            if r is not None:
                return r
    return None


# ─── Basic pl.at() parsing ────────────────────────────────────────────────


def test_parse_pl_at_host_worker():
    """Parse with pl.at(level=HOST, role=Worker)."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.Hierarchy
    assert scope.level == ir.Level.HOST
    assert scope.role == ir.Role.Worker


def test_parse_pl_at_global_orchestrator():
    """Parse with pl.at(level=GLOBAL, role=Orchestrator)."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.Hierarchy
    assert scope.level == ir.Level.GLOBAL
    assert scope.role == ir.Role.Orchestrator


def test_parse_pl_at_level_only():
    """Parse with pl.at(level=CHIP) — no role."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CHIP):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.Hierarchy
    assert scope.level == ir.Level.CHIP
    assert scope.role is None


def test_parse_pl_at_alias_pod():
    """Parse with pl.at(level=POD) — alias for CLUSTER_0."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.POD, role=pl.Role.Orchestrator):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.level is not None
    # POD is an alias for CLUSTER_0; nanobind enums compare by underlying value
    assert ir.level_to_linqu_level(scope.level) == ir.level_to_linqu_level(ir.Level.CLUSTER_0)


# ─── Nested pl.at() blocks ────────────────────────────────────────────────


def test_parse_pl_at_nested():
    """Parse nested pl.at blocks."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
            with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
                y = pl.add(x, x)
        return y

    outer = _find_scope_stmt(f.body)
    assert outer is not None
    assert outer.level == ir.Level.GLOBAL

    inner = _find_scope_stmt(outer.body)
    assert inner is not None
    assert inner.level == ir.Level.HOST
    assert inner.role == ir.Role.Worker


# ─── Error cases ──────────────────────────────────────────────────────────


def test_parse_pl_at_missing_level():
    """pl.at() without level= raises error."""
    with pytest.raises(ParserSyntaxError, match="level"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(role=pl.Role.Worker):
                y = pl.add(x, x)
            return y


def test_parse_pl_at_unknown_kwarg():
    """pl.at() with unknown keyword raises error."""
    with pytest.raises(ParserSyntaxError, match="Unknown keyword"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, bogus=42):
                y = pl.add(x, x)
            return y


# ─── Backward compatibility ───────────────────────────────────────────────


def test_backward_compat_incore():
    """Existing pl.incore() still works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.incore():
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert scope.level is None
    assert scope.role is None


def test_backward_compat_cluster():
    """Existing pl.cluster() still works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.incore():
                y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.Cluster
    assert scope.level is None


# ─── Printer round-trip ───────────────────────────────────────────────────


def test_printer_hierarchy_scope_roundtrip():
    """Python printer renders Hierarchy scope with level/role."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Worker):
            y = pl.add(x, x)
        return y

    printed = str(f)
    assert "pl.at(" in printed
    assert "Level.HOST" in printed
    assert "Role.Worker" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
