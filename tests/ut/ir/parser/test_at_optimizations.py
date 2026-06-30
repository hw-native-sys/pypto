# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for pl.at(..., optimizations=[...]) parsing.

The optimizations= list lets users express ``pl.split(...)``. The legacy
``optimization=`` kwarg and the legacy top-level ``split=`` kwarg have been
removed; passing them now falls through to the generic unknown-keyword error
from pl.at().
"""

import warnings
from typing import Protocol, cast

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.pypto_core import ir


class _HasSplit(Protocol):
    split: ir.SplitMode | None


def _find_scope_stmt(stmt: ir.Stmt) -> ir.ScopeStmt | None:
    """Recursively find the first scope statement in an IR tree."""
    if isinstance(stmt, ir.ScopeStmt):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _find_scope_stmt(s)
            if r is not None:
                return r
    return None


# ─── New API: optimizations=[pl.split(...)] → InCore with split ──────────────


def test_parse_optimizations_split_only_up_down():
    """optimizations=[pl.split(UP_DOWN)] → InCore with split=UP_DOWN."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split == ir.SplitMode.UP_DOWN


def test_parse_optimizations_split_only_left_right():
    """optimizations=[pl.split(LEFT_RIGHT)] → InCore with split=LEFT_RIGHT."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split == ir.SplitMode.LEFT_RIGHT


def test_parse_optimizations_empty_list_is_plain_incore():
    """optimizations=[] → InCore with no split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[]):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split is None


# ─── No DeprecationWarning for the optimizations= API ─────────────────────────


def test_new_optimizations_kwarg_emits_no_warning():
    """The new optimizations= API emits no DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                y = pl.add(x, x)
            return y


# ─── Validation errors on optimizations= entries ──────────────────────────────


def test_optimizations_must_be_list():
    """optimizations= must be a list literal."""
    with pytest.raises(ParserSyntaxError, match="must be a list literal"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=pl.split(pl.SplitMode.UP_DOWN),  # type: ignore[arg-type]
            ):
                y = pl.add(x, x)
            return y


def test_duplicate_split_errors():
    """Two pl.split(...) entries in the same list is an error."""
    with pytest.raises(ParserSyntaxError, match="Duplicate.*split"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.split(pl.SplitMode.LEFT_RIGHT)],
            ):
                y = pl.add(x, x)
            return y


def test_unsupported_entry_errors():
    """Unknown entries in optimizations=[...] are rejected."""
    with pytest.raises(ParserSyntaxError, match="Unsupported entry"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[42]):  # type: ignore[list-item]
                y = pl.add(x, x)
            return y


def test_split_none_in_list_is_explicit_nosplit():
    """pl.split(SplitMode.NONE) is accepted and preserved explicitly."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split == ir.SplitMode.NONE


def test_split_factory_accepts_none_at_runtime():
    """pl.split() accepts explicit SplitMode.NONE construction at runtime."""

    entry = pl.split(pl.SplitMode.NONE)
    assert entry.mode == ir.SplitMode.NONE


def test_split_on_non_core_group_errors():
    """pl.split(...) is only valid at CORE_GROUP."""
    with pytest.raises(ParserSyntaxError, match="CORE_GROUP"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                y = pl.add(x, x)
            return y


# ─── Fully qualified pl.optimizations.* forms ────────────────────────────────


def test_fully_qualified_split():
    """pl.optimizations.split(...) also works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.optimizations.split(pl.SplitMode.UP_DOWN)],
        ):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert cast(_HasSplit, scope).split == ir.SplitMode.UP_DOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
