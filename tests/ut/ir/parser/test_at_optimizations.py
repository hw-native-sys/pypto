# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for pl.at(..., optimizations=[...]) parsing.

After the removal of InCoreScopeStmt / AutoInCoreScopeStmt, ``pl.at(...)`` always
produces a HierarchyScopeStmt. At ``Level.CORE_GROUP``, the ``optimizations=``
list accepts ``pl.split(mode)`` to populate the scope's ``split`` field.
"""

from typing import TypeVar

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.pypto_core import ir

T = TypeVar("T", bound=ir.ScopeStmt)


def _find_scope(stmt, scope_type: type[T]) -> T | None:
    """Recursively find the first scope of ``scope_type`` in an IR tree."""
    if isinstance(stmt, scope_type):
        return stmt
    if isinstance(stmt, ir.SeqStmts):
        for s in stmt.stmts:
            r = _find_scope(s, scope_type)
            if r is not None:
                return r
    return None


# ─── optimizations=[pl.split(...)] → HierarchyScopeStmt with split ───────────


def test_parse_optimizations_split_only_up_down():
    """optimizations=[pl.split(UP_DOWN)] → HierarchyScopeStmt with split=UP_DOWN."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
    assert scope is not None
    assert scope.level == ir.Level.CORE_GROUP
    assert scope.split == ir.SplitMode.UP_DOWN


def test_parse_optimizations_split_only_left_right():
    """optimizations=[pl.split(LEFT_RIGHT)] → HierarchyScopeStmt with split=LEFT_RIGHT."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
    assert scope is not None
    assert scope.split == ir.SplitMode.LEFT_RIGHT


def test_parse_optimizations_empty_list_is_plain_hierarchy():
    """optimizations=[] → HierarchyScopeStmt with no split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[]):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
    assert scope is not None
    assert scope.split is None


def test_parse_core_group_no_optimizations():
    """pl.at(level=CORE_GROUP) without optimizations → plain HierarchyScopeStmt."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            y = pl.add(x, x)
        return y

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
    assert scope is not None
    assert scope.level == ir.Level.CORE_GROUP
    assert scope.split is None


# ─── Validation errors on optimizations= entries ──────────────────────────────


def test_optimizations_must_be_list():
    """optimizations= must be a list literal."""
    with pytest.raises(ParserSyntaxError, match="must be a list literal"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=pl.split(pl.SplitMode.UP_DOWN)):  # type: ignore[arg-type]
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


def test_split_none_in_list_errors():
    """pl.split(SplitMode.NONE) is rejected."""
    with pytest.raises(ParserSyntaxError, match=r"SplitMode\.NONE"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE)]):
                y = pl.add(x, x)
            return y


def test_split_factory_rejects_none_at_runtime():
    """pl.split() also rejects SplitMode.NONE at construction time."""
    with pytest.raises(ValueError, match=r"SplitMode\.NONE"):
        pl.split(pl.SplitMode.NONE)


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

    scope = _find_scope(f.body, ir.HierarchyScopeStmt)
    assert scope is not None
    assert scope.split == ir.SplitMode.UP_DOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
