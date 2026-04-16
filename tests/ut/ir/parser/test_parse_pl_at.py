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
from typing import TypeVar

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.pypto_core import ir

T = TypeVar("T", bound=ir.ScopeStmt)


def _find_scope(stmt, scope_type: type[T]) -> T | None:
    """Recursively find first scope statement of `scope_type` in an IR tree."""
    if isinstance(stmt, scope_type):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _find_scope(s, scope_type)
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

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
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

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
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

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
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

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
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

    outer = _find_scope(f.body, ir.HierarchyScopeStmt)
    assert outer is not None
    assert outer.level == ir.Level.GLOBAL

    inner = _find_scope(outer.body, ir.HierarchyScopeStmt)
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

    scope = _find_scope(f.body, ir.InCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert not isinstance(scope, ir.HierarchyScopeStmt)


def test_backward_compat_cluster():
    """Existing pl.cluster() still works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.incore():
                y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.ClusterScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.Cluster


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


# ─── New pl.at() InCore / AutoInCore forms ───────────────────────────────────


def test_parse_pl_at_core_group_incore():
    """pl.at(level=CORE_GROUP) creates InCoreScopeStmt."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.InCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore


def test_parse_pl_at_core_group_chunked_loop_optimizer_bare():
    """pl.at(level=CORE_GROUP, optimization=pl.chunked_loop_optimizer) → AutoInCore, split=UP_DOWN."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    scope = _find_scope(f.body, ir.AutoInCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.AutoInCore
    assert scope.split == ir.SplitMode.UP_DOWN


def test_parse_pl_at_core_group_chunked_loop_optimizer_with_split():
    """pl.at(level=CORE_GROUP, optimization=chunked_loop_optimizer(split=LEFT_RIGHT)) → AutoInCore."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT),
        ):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    scope = _find_scope(f.body, ir.AutoInCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.AutoInCore
    assert scope.split == ir.SplitMode.LEFT_RIGHT


def test_parse_pl_at_optimization_on_non_core_group_errors():
    """optimization= is not supported for non-CORE_GROUP levels."""
    with pytest.raises(ParserSyntaxError, match="CORE_GROUP"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, optimization=pl.chunked_loop_optimizer):
                y = pl.add(x, x)
            return y


def test_parse_pl_at_unknown_optimization_errors():
    """optimization= with unsupported value raises error."""
    with pytest.raises(ParserSyntaxError, match="chunked_loop_optimizer"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=42):  # type: ignore[arg-type]
                y = pl.add(x, x)
            return y


def test_parse_pl_at_split_mode_none_errors():
    """chunked_loop_optimizer(split=SplitMode.NONE) raises error."""
    with pytest.raises(ParserSyntaxError, match=r"SplitMode\.NONE"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
            ):
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x


def test_parse_pl_at_role_with_core_group_errors():
    """role= combined with level=CORE_GROUP raises error."""
    with pytest.raises(ParserSyntaxError, match="role"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, role=pl.Role.Worker):
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

    scope = _find_scope(f.body, ir.InCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert scope.split == ir.SplitMode.UP_DOWN


def test_parse_pl_incore_with_split_left_right():
    """pl.incore(split=LEFT_RIGHT) creates InCoreScopeStmt with LEFT_RIGHT split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.incore(split=pl.SplitMode.LEFT_RIGHT):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.InCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert scope.split == ir.SplitMode.LEFT_RIGHT


def test_parse_pl_at_core_group_with_split():
    """pl.at(level=CORE_GROUP, split=UP_DOWN) creates InCoreScopeStmt with split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.UP_DOWN):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.InCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert scope.split == ir.SplitMode.UP_DOWN


def test_parse_pl_at_core_group_with_split_left_right():
    """pl.at(level=CORE_GROUP, split=LEFT_RIGHT) creates InCoreScopeStmt with LEFT_RIGHT."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.LEFT_RIGHT):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.InCoreScopeStmt)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert scope.split == ir.SplitMode.LEFT_RIGHT


def test_parse_pl_at_optimization_and_split_conflict():
    """Cannot use both optimization= and split= in pl.at()."""
    with pytest.raises(ParserSyntaxError, match="Cannot use both"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimization=pl.chunked_loop_optimizer,
                split=pl.SplitMode.UP_DOWN,
            ):
                y = pl.add(x, x)
            return y


def test_parse_pl_at_split_on_non_core_group_errors():
    """split= is not supported for non-CORE_GROUP levels."""
    with pytest.raises(ParserSyntaxError, match="CORE_GROUP"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, split=pl.SplitMode.UP_DOWN):
                y = pl.add(x, x)
            return y


def test_printer_incore_with_split_roundtrip():
    """Python printer renders InCore scope with split and it can be re-parsed."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.UP_DOWN):
            y = pl.add(x, x)
        return y

    printed = str(f)
    assert "pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.UP_DOWN)" in printed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
