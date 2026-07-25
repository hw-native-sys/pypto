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


def test_fully_qualified_cross_core_slot():
    """pl.optimizations.cross_core_slot(...) also works."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.optimizations.cross_core_slot(slot_num=4)],
        ):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.attrs.get("slot_num") == 4


# ─── pl.cross_core_slot(slot_num=N): the orthogonal pipe-sizing entry ─────────


def test_parse_optimizations_cross_core_slot_only():
    """optimizations=[pl.cross_core_slot(...)] sets slot_num and leaves split unset.

    This is the point of the entry: a scope can size the cross-core ring
    without naming a SplitMode it does not want.
    """

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=4)]):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.scope_kind == ir.ScopeKind.InCore
    assert scope.attrs.get("slot_num") == 4
    assert cast(_HasSplit, scope).split is None


def test_parse_optimizations_split_and_cross_core_slot():
    """The two entries are orthogonal and combine in one list."""

    @pl.function
    def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=16)],
        ):
            y = pl.add(x, x)
        return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert cast(_HasSplit, scope).split == ir.SplitMode.UP_DOWN
    assert scope.attrs.get("slot_num") == 16


def test_duplicate_cross_core_slot_errors():
    """Two pl.cross_core_slot(...) entries in the same list is an error."""
    with pytest.raises(ParserSyntaxError, match="Duplicate.*cross_core_slot"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[pl.cross_core_slot(slot_num=4), pl.cross_core_slot(slot_num=8)],
            ):
                y = pl.add(x, x)
            return y


def test_cross_core_slot_conflicts_with_deprecated_split_slot_num():
    """Setting the slot count twice — via pl.split(slot_num=) and via the
    dedicated entry — is rejected rather than resolved by list order."""
    with pytest.raises(ParserSyntaxError, match="sets the cross-core slot count twice"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[
                    pl.split(pl.SplitMode.UP_DOWN, slot_num=4),
                    pl.cross_core_slot(slot_num=8),
                ],
            ):
                y = pl.add(x, x)
            return y


def test_cross_core_slot_conflict_detected_in_either_order():
    """Same conflict, entries swapped — the deprecated spelling appearing last."""
    with pytest.raises(ParserSyntaxError, match="sets the cross-core slot count twice"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[
                    pl.cross_core_slot(slot_num=8),
                    pl.split(pl.SplitMode.UP_DOWN, slot_num=4),
                ],
            ):
                y = pl.add(x, x)
            return y


def test_cross_core_slot_requires_slot_num_keyword():
    """pl.cross_core_slot() without slot_num= is an error."""
    with pytest.raises(ParserSyntaxError, match="requires the slot_num="):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot()]):  # type: ignore[call-arg]
                y = pl.add(x, x)
            return y


def test_cross_core_slot_rejects_positional_arg():
    """The slot count must be passed by keyword."""
    with pytest.raises(ParserSyntaxError, match="no positional arguments"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(4)]):  # type: ignore[misc]
                y = pl.add(x, x)
            return y


def test_cross_core_slot_rejects_unknown_keyword():
    """Only slot_num= is accepted."""
    with pytest.raises(ParserSyntaxError, match="Unknown keyword argument 'depth'"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(depth=4)]):  # type: ignore[call-arg]
                y = pl.add(x, x)
            return y


@pytest.mark.parametrize(
    ("bad", "expected"),
    [
        ("0", "must be positive"),
        # -1 is ast.UnaryOp(USub, Constant(1)), not an ast.Constant, so it is
        # rejected one step earlier by the integer-literal check. Same shape as
        # the pre-existing pl.split(slot_num=) behaviour.
        ("-1", "must be an integer literal"),
    ],
)
def test_cross_core_slot_rejects_non_positive(bad: str, expected: str):
    """slot_num must be a positive integer literal."""
    src = (
        "import pypto.language as pl\n"
        "@pl.function\n"
        "def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:\n"
        "    with pl.at(level=pl.Level.CORE_GROUP, "
        f"optimizations=[pl.cross_core_slot(slot_num={bad})]):\n"
        "        y = pl.add(x, x)\n"
        "    return y\n"
    )
    with pytest.raises(ParserSyntaxError, match=expected):
        pl.parse(src)


def test_cross_core_slot_rejects_bool_literal():
    """bool is a subclass of int — True must not pose as a slot count."""
    with pytest.raises(ParserSyntaxError, match="must be an integer literal"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[pl.cross_core_slot(slot_num=True)],  # type: ignore[arg-type]
            ):
                y = pl.add(x, x)
            return y


def test_cross_core_slot_on_non_core_group_errors():
    """pl.cross_core_slot(...) is only valid at CORE_GROUP — the slot_num attr
    has no meaning on a Hierarchy scope, so it must not silently attach."""
    with pytest.raises(ParserSyntaxError, match="CORE_GROUP"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.HOST, optimizations=[pl.cross_core_slot(slot_num=4)]):
                y = pl.add(x, x)
            return y


def test_cross_core_slot_factory_validates_at_runtime():
    """The runtime factory rejects non-positive / non-int slot counts."""
    assert pl.cross_core_slot(slot_num=4).slot_num == 4
    with pytest.raises(ValueError, match="must be a positive integer"):
        pl.cross_core_slot(slot_num=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        pl.cross_core_slot(slot_num=True)  # type: ignore[arg-type]


# ─── Deprecated pl.split(slot_num=...) spelling ──────────────────────────────


def test_split_slot_num_emits_deprecation_warning():
    """The old carrier still parses, but warns and points at the new entry."""
    with pytest.warns(DeprecationWarning, match="pl.cross_core_slot"):

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(
                level=pl.Level.CORE_GROUP,
                optimizations=[pl.split(pl.SplitMode.UP_DOWN, slot_num=4)],
            ):
                y = pl.add(x, x)
            return y

    scope = _find_scope_stmt(f.body)
    assert scope is not None
    assert scope.attrs.get("slot_num") == 4
    assert cast(_HasSplit, scope).split == ir.SplitMode.UP_DOWN


def test_split_slot_num_factory_emits_deprecation_warning():
    """The runtime factory warns on the same deprecated argument."""
    with pytest.warns(DeprecationWarning, match="pl.cross_core_slot"):
        entry = pl.split(pl.SplitMode.UP_DOWN, slot_num=4)
    assert entry.slot_num == 4


def test_split_without_slot_num_emits_no_warning():
    """Only the slot_num= argument is deprecated, not pl.split itself."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)

        @pl.function
        def f(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
                y = pl.add(x, x)
            return y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
