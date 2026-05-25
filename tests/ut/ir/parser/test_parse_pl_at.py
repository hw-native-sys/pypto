# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for parsing pl.at(level=..., role=...) (Step 04)."""

import warnings
from typing import Protocol, cast

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.pypto_core import ir


class _HasSplit(Protocol):
    split: ir.SplitMode | None


class _HasLevelRole(Protocol):
    level: ir.Level | None
    role: ir.Role | None


def _find_scope_stmt(stmt: ir.Stmt) -> ir.ScopeStmt | None:
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


def test_parse_pl_at_host_worker_rejected():
    """Inline ``with pl.at(role=SubWorker)`` is rejected; use @pl.function instead."""
    with pytest.raises(ParserSyntaxError, match="not supported"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker):
                _ = x
            return x


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
    hierarchy_scope = cast(_HasLevelRole, scope)
    assert hierarchy_scope.level == ir.Level.GLOBAL
    assert hierarchy_scope.role == ir.Role.Orchestrator


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
    hierarchy_scope = cast(_HasLevelRole, scope)
    assert hierarchy_scope.level == ir.Level.CHIP
    assert hierarchy_scope.role is None


def test_parse_pl_at_alias_pod():
    """Parse with pl.at(level=POD) — alias for CLUSTER_0."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.POD, role=pl.Role.Orchestrator):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    hierarchy_scope = cast(_HasLevelRole, scope)
    assert hierarchy_scope.level is not None
    # POD is an alias for CLUSTER_0; nanobind enums compare by underlying value
    assert ir.level_to_linqu_level(hierarchy_scope.level) == ir.level_to_linqu_level(ir.Level.CLUSTER_0)


# ─── Nested pl.at() blocks ────────────────────────────────────────────────


def test_parse_pl_at_nested():
    """Parse nested pl.at Hierarchy blocks (Orchestrator scopes)."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.GLOBAL, role=pl.Role.Orchestrator):
            with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
                _ = x
        return x

    outer = _find_scope_stmt(f.body)
    assert outer is not None
    assert outer.scope_kind == ir.ScopeKind.Hierarchy
    outer_scope = cast(_HasLevelRole, outer)
    assert outer_scope.level == ir.Level.GLOBAL

    inner = _find_scope_stmt(outer.body)
    assert inner is not None
    assert inner.scope_kind == ir.ScopeKind.Hierarchy
    inner_scope = cast(_HasLevelRole, inner)
    assert inner_scope.level == ir.Level.HOST
    assert inner_scope.role == ir.Role.Orchestrator


# ─── Error cases ──────────────────────────────────────────────────────────


def test_parse_pl_at_missing_level():
    """pl.at() without level= raises error."""
    with pytest.raises(ParserSyntaxError, match="level"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(role=pl.Role.SubWorker):
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


# ─── Printer round-trip ───────────────────────────────────────────────────


def test_printer_hierarchy_scope_roundtrip():
    """Python printer renders Hierarchy scope with level/role."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
            _ = x
        return x

    printed = str(f)
    assert "pl.at(" in printed
    assert "Level.HOST" in printed
    assert "Role.Orchestrator" in printed


# ─── New pl.at() InCore / AutoInCore forms ───────────────────────────────────


def test_parse_pl_at_core_group_incore():
    """pl.at(level=CORE_GROUP) creates InCoreScopeStmt."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore


def test_parse_pl_at_role_with_core_group_errors():
    """role= combined with level=CORE_GROUP raises error."""
    with pytest.raises(ParserSyntaxError, match="role"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker):
                y = pl.add(x, x)
            return y


def test_incore_deprecation_warning():
    """pl.incore() emits DeprecationWarning at parse time."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.incore():
                y = pl.add(x, x)
            return y

    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert any("pl.incore()" in str(warning.message) for warning in w)


def test_auto_incore_deprecation_warning():
    """pl.auto_incore() emits DeprecationWarning at parse time."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.auto_incore():
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x

    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert any("pl.auto_incore()" in str(warning.message) for warning in w)


# ─── InCore with split ──────────────────────────────────────────────────────


def test_parse_pl_incore_with_split():
    """pl.incore(split=UP_DOWN) creates InCoreScopeStmt with split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.incore(split=pl.SplitMode.UP_DOWN):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split == ir.SplitMode.UP_DOWN


def test_parse_pl_incore_with_split_left_right():
    """pl.incore(split=LEFT_RIGHT) creates InCoreScopeStmt with LEFT_RIGHT split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.incore(split=pl.SplitMode.LEFT_RIGHT):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split == ir.SplitMode.LEFT_RIGHT


def test_printer_incore_with_split_roundtrip():
    """Python printer renders InCore scope with split and it can be re-parsed."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            y = pl.add(x, x)
        return y

    printed = str(f)
    assert "pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)])" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
