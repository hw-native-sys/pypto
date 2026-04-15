# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for pl.at(..., optimizations=[...]) parsing.

Covers issue #1030: the optimizations= list lets users express ``pl.split(...)``
and ``pl.auto_chunk`` independently. The legacy ``optimization=`` and top-level
``split=`` kwargs remain functional but emit DeprecationWarning, and mixing the
new ``optimizations=`` with either deprecated kwarg is a hard error.
"""

import warnings

import pypto.language as pl
import pytest
from pypto.language.parser.diagnostics import ParserSyntaxError
from pypto.pypto_core import ir


def _find_scope_stmt(stmt):
    """Recursively find the first ScopeStmt in an IR tree."""
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
    assert scope.split == ir.SplitMode.UP_DOWN


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
    assert scope.split == ir.SplitMode.LEFT_RIGHT


# ─── New API: optimizations=[pl.auto_chunk] → AutoInCore (no split) ──────────


def test_parse_optimizations_auto_chunk_only():
    """optimizations=[pl.auto_chunk] → AutoInCore with no split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.AutoInCore
    assert scope.split is None


# ─── New API: optimizations=[pl.auto_chunk, pl.split(...)] → AutoInCore + split


def test_parse_optimizations_auto_chunk_with_split():
    """optimizations=[pl.auto_chunk, pl.split(UP_DOWN)] → AutoInCore with split."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)],
        ):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.AutoInCore
    assert scope.split == ir.SplitMode.UP_DOWN


def test_parse_optimizations_order_independent():
    """List order does not affect the produced IR."""

    @pl.function
    def f1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.LEFT_RIGHT)],
        ):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    @pl.function
    def f2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT), pl.auto_chunk],
        ):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    s1 = _find_scope_stmt(f1.body)
    s2 = _find_scope_stmt(f2.body)
    assert s1 is not None and s2 is not None
    assert s1.scope_kind == s2.scope_kind == ir.ScopeKind.AutoInCore
    assert s1.split == s2.split == ir.SplitMode.LEFT_RIGHT


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
    assert scope.split is None


# ─── Equivalence with deprecated API ──────────────────────────────────────────


def test_legacy_chunked_loop_optimizer_matches_new_form():
    """Legacy bare optimizer (defaults to UP_DOWN) ≡ new auto_chunk + split(UP_DOWN)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        @pl.function
        def legacy(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x

    @pl.function
    def new(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)],
        ):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    s_legacy = _find_scope_stmt(legacy.body)
    s_new = _find_scope_stmt(new.body)
    assert s_legacy is not None and s_new is not None
    assert s_legacy.scope_kind == s_new.scope_kind == ir.ScopeKind.AutoInCore
    assert s_legacy.split == s_new.split == ir.SplitMode.UP_DOWN


def test_legacy_split_kwarg_matches_new_form():
    """Legacy top-level split= ≡ new optimizations=[pl.split(...)]."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        @pl.function
        def legacy(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.LEFT_RIGHT):
                y = pl.add(x, x)
            return y

    @pl.function
    def new(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)]):
            y = pl.add(x, x)
        return y

    s_legacy = _find_scope_stmt(legacy.body)
    s_new = _find_scope_stmt(new.body)
    assert s_legacy is not None and s_new is not None
    assert s_legacy.scope_kind == s_new.scope_kind == ir.ScopeKind.InCore
    assert s_legacy.split == s_new.split == ir.SplitMode.LEFT_RIGHT


# ─── DeprecationWarning emission ──────────────────────────────────────────────


def test_legacy_optimization_kwarg_emits_deprecation_warning():
    """Using the legacy optimization= kwarg emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="optimizations=\\[pl.auto_chunk\\]"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x


def test_legacy_split_kwarg_emits_deprecation_warning():
    """Using the legacy top-level split= kwarg emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="optimizations=\\[pl.split"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, split=pl.SplitMode.UP_DOWN):
                y = pl.add(x, x)
            return y


def test_new_optimizations_kwarg_emits_no_warning():
    """The new optimizations= API emits no DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                y = pl.add(x, x)
            return y


# ─── Hard errors when mixing new with deprecated kwargs ──────────────────────


def test_mix_optimizations_with_legacy_optimization_errors():
    """Cannot combine optimizations= with deprecated optimization=."""
    with pytest.raises(ParserSyntaxError, match="Cannot mix"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
                optimization=pl.chunked_loop_optimizer,
            ):
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x


def test_mix_optimizations_with_legacy_split_errors():
    """Cannot combine optimizations= with deprecated split=."""
    with pytest.raises(ParserSyntaxError, match="Cannot mix"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[pl.auto_chunk],
                split=pl.SplitMode.UP_DOWN,
            ):
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x


# ─── Validation errors on optimizations= entries ──────────────────────────────


def test_optimizations_must_be_list():
    """optimizations= must be a list literal."""
    with pytest.raises(ParserSyntaxError, match="must be a list literal"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=pl.auto_chunk):  # type: ignore[arg-type]
                y = pl.add(x, x)
            return y


def test_duplicate_auto_chunk_errors():
    """Two pl.auto_chunk entries in the same list is an error."""
    with pytest.raises(ParserSyntaxError, match="Duplicate.*auto_chunk"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.auto_chunk]):
                for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                    x = pl.add(x, x)
            return x


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


def test_auto_chunk_on_non_core_group_errors():
    """pl.auto_chunk is only valid at CORE_GROUP."""
    with pytest.raises(ParserSyntaxError, match="CORE_GROUP"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, optimizations=[pl.auto_chunk]):
                y = pl.add(x, x)
            return y


def test_split_on_non_core_group_errors():
    """pl.split(...) is only valid at CORE_GROUP."""
    with pytest.raises(ParserSyntaxError, match="CORE_GROUP"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                y = pl.add(x, x)
            return y


# ─── Fully qualified pl.optimizations.* forms ────────────────────────────────


def test_fully_qualified_auto_chunk():
    """pl.optimizations.auto_chunk also works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.optimizations.auto_chunk]):
            for i in pl.parallel(0, 8, 1, chunk=4, chunk_policy="leading_full"):
                x = pl.add(x, x)
        return x

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.AutoInCore


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
    assert scope.split == ir.SplitMode.UP_DOWN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
