# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Scoped region state shared by scope, dispatch and operator parsing."""

import ast
from collections.abc import Generator
from contextlib import contextmanager

from pypto.pypto_core import ir

from .diagnostics import (
    ParserSyntaxError,
)
from .parser_state import ControlFlowState, ParserServices


class RegionContext:
    """Scoped region state shared by scope, dispatch and operator parsing."""

    def __init__(
        self,
        *,
        flow: ControlFlowState,
        services: ParserServices,
    ) -> None:
        self.flow = flow
        self.services = services
        self.scope_kind_stack: list[ir.ScopeKind] = []

        # Set while parsing the body of an InCore scope the user declared with an
        # explicit ``optimizations=[pl.split(MODE)]`` — MODE included when it is
        # ``NONE``. ``InCoreScopeStmt.split_`` cannot answer this: it has a single
        # encoding of "no split" (``SplitMode.NONE``), so a literal
        # ``pl.split(pl.SplitMode.NONE)`` is indistinguishable there from writing
        # no ``pl.split`` at all (issue #2205). The literal is only visible here,
        # which is where the RFC #1820 mutual-exclusion rejection now lives.
        # InCore scopes never nest (NoNestedInCore), so one slot suffices.
        self.incore_user_split: ir.SplitMode | None = None

        # Active ``pl.split_aiv(mode=...)`` modes (innermost last). ``pl.aiv_shard`` /
        # ``pl.aic_gather`` inherit the split mode from this stack rather than
        # taking it as an argument.
        self.split_aiv_mode_stack: list[ir.SplitMode] = []

        # Depth of nested ``with pl.manual_scope():`` blocks. Used to gate the
        # ``deps=[var]`` kwarg recognition on kernel calls.
        self.manual_scope_depth: int = 0

    @contextmanager
    def scope_kind_context(self, scope_kind: ir.ScopeKind) -> Generator[None, None, None]:
        """Track scope nesting during parsing for context-sensitive validation."""
        self.scope_kind_stack.append(scope_kind)
        try:
            yield
        finally:
            self.scope_kind_stack.pop()

    def is_inside_scope(self, scope_kind: ir.ScopeKind) -> bool:
        """Return whether parsing is currently nested inside the given scope kind."""
        return scope_kind in self.scope_kind_stack

    @contextmanager
    def incore_user_split_context(
        self, scope_kind: "ir.ScopeKind", split_mode: "ir.SplitMode | None"
    ) -> Generator[None, None, None]:
        """Record the enclosing InCore scope's *literal* ``pl.split(MODE)`` entry.

        ``split_mode`` is the parsed ``optimizations=[pl.split(MODE)]`` entry, or
        ``None`` when the user wrote no ``pl.split`` at all. Only the parser can
        tell the two apart once ``MODE`` is ``NONE`` — see
        :meth:`RegionContext.reject_user_split_with_split_aiv_region`.

        A no-op for every other scope kind: ``optimizations=[pl.split(...)]`` only
        lowers onto an InCore scope, so another kind must neither set nor clear the
        record of an enclosing one.
        """
        if scope_kind != ir.ScopeKind.InCore:
            yield
            return
        previous = self.incore_user_split
        self.incore_user_split = split_mode
        try:
            yield
        finally:
            self.incore_user_split = previous

    def reject_user_split_with_split_aiv_region(self, stmt: ast.For, hint: str) -> None:
        """Reject ``pl.split_aiv`` inside an InCore scope declaring ``pl.split(...)``.

        A function-level AUTO split (``optimizations=[pl.split(MODE)]``) and
        explicit ``pl.split_aiv`` regions are mutually exclusive AIV-split
        mechanisms: downstream lowering takes the per-region path and would
        silently drop the function-level split. **Any** ``pl.split(...)`` is
        rejected, ``pl.SplitMode.NONE`` included (RFC #1820) — NONE carries no
        split of its own, but writing it still reads as "auto and manual split
        mixed on one scope", and the cross-core slot count that once forced the
        NONE spelling now has its own orthogonal ``pl.cross_core_slot(slot_num=N)``
        entry.

        ``OutlineIncoreScopes`` keeps the same rejection for the modes that reach
        the IR, and remains the backstop for scopes that never went through this
        parser. It cannot see a literal ``NONE``, so that spelling is caught here.
        """
        if self.incore_user_split is None:
            return
        raise ParserSyntaxError(
            f"scope combines a function-level pl.split(pl.SplitMode.{self.incore_user_split.name}) "
            "(optimizations=[pl.split(...)]) with a pl.split_aiv region; these are mutually "
            "exclusive AIV-split mechanisms — the function-level split would be silently "
            "dropped (the per-region split governs the lanes)",
            span=self.services.span_tracker.get_span(stmt),
            hint="Remove optimizations=[pl.split(...)] or the pl.split_aiv region. To pin a "
            "custom cross-core slot count, use optimizations=[pl.cross_core_slot(slot_num=N)], "
            f"which is orthogonal to splitting. {hint}",
        )

    @contextmanager
    def split_aiv_mode_context(self, mode: ir.SplitMode) -> Generator[None, None, None]:
        """Track the active ``pl.split_aiv`` split mode during body parsing.

        ``pl.aiv_shard`` / ``pl.aic_gather`` read the innermost entry to inherit
        the split mode from the enclosing ``for ... in pl.split_aiv(mode=...)``
        scope instead of taking it as an explicit argument.

        Also pushes a ``"split_aiv"`` sentinel onto :attr:`_loop_kind_stack` for
        the duration of the body parse. A ``pl.split_aiv`` loop lowers to a scope,
        not a ``ForStmt``, so ``break`` / ``continue`` inside it have no loop to
        target; the sentinel makes :meth:`ControlFlowParser.validate_loop_control` reject them
        instead of letting them silently bind to an enclosing Python loop.
        """
        self.split_aiv_mode_stack.append(mode)
        self.flow.loop_kind_stack.append("split_aiv")
        try:
            yield
        finally:
            self.flow.loop_kind_stack.pop()
            self.split_aiv_mode_stack.pop()
