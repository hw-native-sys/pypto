# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime, hierarchy, SPMD and explicit split scopes and their body carriers."""

import ast
import keyword as _keyword_mod
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

from pypto.ir import op as ir_op
from pypto.ir.printer import python_print
from pypto.language.optimizations import SPLIT_SLOT_NUM_DEPRECATION, CrossCoreSlot, cross_core_slot
from pypto.pypto_core import DataType, ir

from .attr_parser import AttrParser
from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
from .dispatch_semantics import DispatchSemantics
from .enum_utils import (
    CACHE_POLICY_MAP,
    CACHE_POLICY_NAMES,
    LEVEL_MAP,
    ROLE_MAP,
    SCOPE_MODE_MAP,
    SPLIT_MODE_MAP,
    extract_enum_value,
)
from .parser_state import FunctionContext, ParseCallbacks, ParserServices
from .region_context import RegionContext
from .source_comments import SourceCommentTracker
from .syntax_helpers import (
    is_cache_policy_marker,
    is_pl_call,
    is_self_method_call,
    spmd_body_calls,
    spmd_body_has_call,
)
from .type_resolver import (
    _is_dump_call,  # noqa: PLC2701
)

# Scope attr seeded by ``pl.set_cache_policy(t, policy)``; mirrors
# ``ir::kAttrCachePolicyVars``. Holds ``list[tuple[Var, int]]`` and is consumed
# by the scope outliner, which re-emits it as outlined-function param indices.
CACHE_POLICY_VARS_ATTR = "cache_policy_vars"

# ``pl.set_cache_policy`` may only be written where the declaration has a
# scope to attach to. Spelling out the legal positions once keeps the
# position rejection and the fallback kind rejection from drifting apart.
CACHE_POLICY_POSITION_HINT = (
    "Write pl.set_cache_policy(t, pl.CachePolicy.BYPASS) as a standalone statement "
    "directly inside a `with pl.at(...):` / `with pl.spmd(...):` scope body (or a "
    "`for i in pl.spmd(...):` body). It declares a property of the whole scope, so it "
    "cannot sit outside one, nor nested inside an `if` / `for` within one — a "
    "conditionally-executed declaration is a promise the compiler cannot check. To "
    "annotate a single access instead, use pl.load(..., cache=pl.CachePolicy.BYPASS)."
)

# Scope kinds that carry ``kAttrCachePolicyVars``: the two the outliner turns
# into a device kernel, and the only two whose printer re-emits the marker
# statements (see ``IRPythonPrinter::PrintScopeCachePolicyStmts``). A Spmd
# scope reaches here only in the wrapper-less *dispatch* shape — an inline
# body is carried by a nested InCore, which is what this set matches — and a
# dispatch body's reads live in the pre-defined callee, not here. Accepting
# the marker on any other kind would attach an attr nothing consumes and the
# printer drops, silently discarding a contract the author stated.
CACHE_POLICY_SCOPE_KINDS: "frozenset[ir.ScopeKind]" = frozenset({ir.ScopeKind.Hierarchy, ir.ScopeKind.InCore})

# Per-kind explanation of why the declaration has nowhere to attach.
CACHE_POLICY_REJECT_HINTS: "dict[ir.ScopeKind, str]" = {
    ir.ScopeKind.Spmd: (
        "This `pl.spmd(...)` body dispatches a pre-defined kernel, so the GM reads happen "
        "inside that callee, not here. Declare the policy in the callee — a "
        "`with pl.at(level=pl.Level.CORE_GROUP):` scope in its body, or "
        "pl.load(..., cache=pl.CachePolicy.BYPASS) on the access itself. An *inline* "
        "pl.spmd body does accept the marker: it is carried by the InCore scope the body "
        "is outlined into."
    ),
    ir.ScopeKind.Cluster: (
        "A pl.cluster(...) scope co-schedules AIC and AIV work rather than becoming a "
        "kernel of its own. Move the declaration into the `with pl.at(...):` scope that "
        "actually reads the tensor."
    ),
    ir.ScopeKind.Runtime: (
        "A runtime scope (pl.manual_scope / pl.auto_scope) only chooses dependency "
        "semantics for the tasks it contains; it issues no GM read of its own. Move the "
        "declaration into the `with pl.at(...):` scope that reads the tensor."
    ),
}

# ``pl.spmd()`` keywords whose *value* the caller interprets. ``deps`` is
# legal on every form: declaring edges is orthogonal to capturing the
# dispatch TaskId with ``as tid`` (see _parse_spmd_kwargs).
SPMD_KWARGS = frozenset(
    {
        "core_num",
        "sync_start",
        "name_hint",
        "optimizations",
        "deps",
        "dumps",
        "allow_early_resolve",
        "predicate",
    }
)

# AIV sub-core count is hardware-fixed at 2 (the two AIV lanes of one AICore).
SPLIT_AIV_SUBCORE_NUM = 2

#: Function types whose body is a device-side kernel, where a Graph region
#: has nothing to record. Opaque / Orchestration / Graph / Inline are allowed.
GRAPH_FORBIDDEN_FUNC_TYPES = frozenset(
    {
        ir.FunctionType.InCore,
        ir.FunctionType.AIC,
        ir.FunctionType.AIV,
        ir.FunctionType.Group,
        ir.FunctionType.Spmd,
    }
)


@dataclass
class AtKwargState:
    """Mutable accumulator used while parsing pl.at() keyword arguments."""

    level: "ir.Level | None" = None
    role: "ir.Role | None" = None
    name_hint: str = ""
    # ``allow_early_resolve=True`` — speculative early-dispatch opt-in, stored as
    # the scope's ``allow_early_resolve`` attr and threaded onto the synthesised
    # ``Submit`` by the outliner (mirrors ``pl.submit(..., allow_early_resolve=)``).
    allow_early_resolve: bool = False
    split_mode: "ir.SplitMode | None" = None
    # Optional cross-core pipe slot count from ``pl.cross_core_slot(slot_num=N)``
    # (or the deprecated ``pl.split(mode, slot_num=N)`` spelling). Stored on the
    # scope attrs and propagated to the outlined function attr.
    cross_core_slots: "CrossCoreSlot | None" = None
    # Tracks the ``optimizations=`` kwarg AST so a duplicate ``optimizations=``
    # can be rejected in ``_handle_at_optimizations_kw``.
    new_optimizations_kw: "ast.keyword | None" = field(default=None)
    # ``deps=[tid1, tid2]`` AST kept verbatim; resolved into Var refs by the
    # caller once it has decided this scope opts into the manual_dep_edges path.
    deps_kw: "ast.keyword | None" = field(default=None)
    # ``no_dep_args=[t1, t2]`` AST kept verbatim; resolved into outer-scope
    # Var refs by the caller, written to
    # ScopeStmt.attrs[arg_direction_overrides_vars], and translated into
    # per-arg-index overrides by the outliner.
    no_dep_args_kw: "ast.keyword | None" = field(default=None)
    # ``dumps=[t1, t2]`` AST kept verbatim; resolved into outer-scope Var
    # refs by the caller and written to ScopeStmt.attrs[dump_vars]. The
    # scope-level selective-dump surface, symmetric with ``deps=``: the printer
    # emits it for any scope carrying ``kAttrDumpVars`` (seeded by ``pl.dump_tag``
    # at parse, by an explicit ``dumps=`` list, and by the inline-call
    # ``dump_vars`` transfer), and the outliner translates it into the
    # synthesised dispatch's ``kAttrDumpVars``.
    dumps_kw: "ast.keyword | None" = field(default=None)
    # ``predicate=(t[i] > 0)`` AST kept verbatim; resolved by
    # ``_parse_at_predicate`` only after the level / nesting gates have run, so a
    # misplaced predicate reports *that* rather than a shape error. The parsed
    # comparison Expr lands on ScopeStmt.attrs[predicate] and is moved onto
    # ``Submit::predicate_`` by the outliner.
    predicate_kw: "ast.keyword | None" = field(default=None)
    windowize: bool = False


SPMD_SCOPE_NAME_SUFFIX = "_spmd"


# ``pl.at()`` kwargs whose AST node is stashed verbatim on ``AtKwargState`` for
# later resolution: ``deps`` / ``no_dep_args`` / ``dumps`` are duplicate-checked
# then resolved by ``_parse_at_meta``, and ``predicate`` by
# ``_parse_at_predicate``.
# Maps the kwarg name to its ``AtKwargState`` field.
AT_STASH_KWARGS = {
    "deps": "deps_kw",
    "no_dep_args": "no_dep_args_kw",
    "dumps": "dumps_kw",
    "predicate": "predicate_kw",
}


def split_spmd_for_loop_name_hints(name_hint: str) -> tuple[str, str]:
    """Map one ``for i in pl.spmd(..., name_hint=...)`` hint to Spmd vs InCore names.

    The for-form auto-wraps an inner ``InCoreScopeStmt``; ``OutlineIncoreScopes``
    names the AIC kernel from that inner ``name_hint_``. The outer ``SpmdScopeStmt``
    hint is consumed by ``OutlineClusterScopes`` (``_spmd_`` suffix on the wrapper).

    - ``name_hint="q_proj"`` → Spmd ``q_proj_spmd``, InCore ``q_proj``
    - ``name_hint="q_proj_spmd"`` → Spmd ``q_proj_spmd``, InCore ``q_proj``
    """
    if not name_hint:
        return "", ""
    if name_hint.endswith(SPMD_SCOPE_NAME_SUFFIX):
        return name_hint, name_hint[: -len(SPMD_SCOPE_NAME_SUFFIX)]
    return f"{name_hint}{SPMD_SCOPE_NAME_SUFFIX}", name_hint


class ScopeParser:
    """Runtime, hierarchy, SPMD and explicit split scopes and their body carriers."""

    def __init__(
        self,
        *,
        attrs: AttrParser,
        callbacks: ParseCallbacks,
        comments: SourceCommentTracker,
        dispatch: DispatchSemantics,
        function: FunctionContext,
        regions: RegionContext,
        services: ParserServices,
    ) -> None:
        self.attrs = attrs
        self.callbacks = callbacks
        self.comments = comments
        self.dispatch = dispatch
        self.function = function
        self.regions = regions
        self.services = services
        # ``pl.set_cache_policy(...)`` markers already hoisted onto a scope attr
        # by :meth:`ScopeParser.collect_scope_cache_policy_decls`, keyed by ``id()`` of the
        # AST statement. The pre-scan runs *before* the scope body is parsed
        # (scope attrs are fixed at ``begin_scope``), so the marker is still in
        # the body's statement list when ``parse_evaluation_statement`` reaches
        # it; this registry is how the handler tells an already-consumed marker
        # from one written somewhere the declaration cannot attach. The AST node
        # is kept as the value so it stays alive and its ``id()`` cannot be
        # reused by another object mid-parse. See ``_handle_set_cache_policy``.
        self.consumed_cache_policy_markers: dict[int, ast.Expr] = {}

    def handle_set_cache_policy(self, stmt: ast.Expr) -> None:
        """Handle ``pl.set_cache_policy(<name>, pl.CachePolicy.<NAME>)`` at statement position.

        The declarative GM cache-access marker. Unlike ``pl.dump_tag`` — which is
        forward-sticky and consumed right here — this one attaches to the
        *enclosing scope*, and a ``ScopeStmt``'s attrs are fixed when the scope
        begins, before its body is parsed. So the real work happens earlier, in
        :meth:`ScopeParser.collect_scope_cache_policy_decls`, which pre-scans the body's
        top-level statements at each scope-construction site and records what it
        consumed in ``_consumed_cache_policy_markers``.

        By the time the body parse reaches the marker, therefore, a legal one is
        already on the scope attr and only has to be skipped so it emits no IR.
        Reaching here *unconsumed* means the marker was written where no scope
        could take it — outside a scope, or nested inside an ``if`` / ``for``
        within one — which is the rejection this handler exists for.
        """
        if id(stmt) in self.consumed_cache_policy_markers:
            return
        raise ParserSyntaxError(
            "pl.set_cache_policy() must be a standalone statement directly inside a "
            "pl.at(...) / pl.spmd(...) scope body",
            span=self.services.span_tracker.get_span(stmt),
            hint=CACHE_POLICY_POSITION_HINT,
        )

    def resolve_cache_policy_marker(self, stmt: ast.Expr) -> "tuple[ir.Var, int]":
        """Validate one ``pl.set_cache_policy(...)`` marker and resolve its operands.

        Returns the ``(Var, CachePolicy-as-int)`` pair the scope attr carries.
        The policy is stored as a plain ``int`` because that is what the C++
        attr (``std::vector<std::pair<VarPtr, int>>``) and the downstream
        ``tile.load`` ``cache`` kwarg both hold.
        """
        call = stmt.value
        assert isinstance(call, ast.Call)
        span = self.services.span_tracker.get_span(stmt)

        expected_args = 2
        if len(call.args) != expected_args or call.keywords:
            raise ParserSyntaxError(
                "pl.set_cache_policy() takes exactly two positional arguments (no keywords)",
                span=span,
                hint="Use: pl.set_cache_policy(tensor_var, pl.CachePolicy.BYPASS)",
            )
        if not isinstance(call.args[0], ast.Name):
            raise ParserSyntaxError(
                "pl.set_cache_policy() first argument must be a bare variable name",
                span=self.services.span_tracker.get_span(call.args[0]),
                hint=(
                    "Write pl.set_cache_policy(b, pl.CachePolicy.BYPASS) where b is a tensor "
                    "bound outside the scope and read inside it. The declaration is tracked by "
                    "variable identity, so attribute / subscript / call expressions — which name "
                    "no binding — are not supported."
                ),
            )
        policy = extract_enum_value(call.args[1], CACHE_POLICY_MAP, "CachePolicy", "pl.CachePolicy")

        # Resolved against the bindings live *at the scope*, not inside it: the
        # declaration names a tensor the scope body captures from outside, which
        # is exactly what the scope outliner later resolves to an outlined
        # parameter index. Identity matching is reliable because the scope
        # manager returns a stable object per binding, so a later rebinding of
        # the same name yields a new (undeclared) Var.
        name = call.args[0].id
        var = self.services.scope_manager.lookup_var(name)
        if var is None:
            raise ParserSyntaxError(
                f"pl.set_cache_policy() argument '{name}' is not defined at this point",
                span=span,
                hint=(
                    "Declare a policy only for a tensor already bound where the scope starts — a "
                    "parameter or an earlier assignment. A tensor created inside the scope body "
                    "is not captured by it and has no policy to declare."
                ),
            )
        # ``lookup_var`` may return a non-Var placeholder (e.g. a loop-yield name
        # string), and only tensors are read from GM. Reject early so the scope
        # attr never carries a typeless binding for the outliner to trip over.
        if not isinstance(var, ir.Var) or not isinstance(var.type, ir.TensorType):
            got = type(var.type).__name__ if isinstance(var, ir.Var) else type(var).__name__
            raise ParserTypeError(
                f"pl.set_cache_policy() argument '{name}' is not a tensor (got {got})",
                span=self.services.span_tracker.get_span(call.args[0]),
                hint="A cache-access policy only applies to a tensor read from global memory.",
            )
        return var, int(policy)

    def collect_scope_cache_policy_decls(
        self, body: "list[ast.stmt]", scope_kind: "ir.ScopeKind"
    ) -> "list[tuple[ir.Var, int]]":
        """Hoist the ``pl.set_cache_policy(...)`` markers out of a scope body.

        Called at every scope-construction site, just before ``builder.scope()``,
        because a ``ScopeStmt``'s attrs are fixed when the scope begins. Only
        the body's *top-level* statements are scanned: a marker nested inside an
        ``if`` / ``for`` is deliberately left for
        :meth:`ScopeParser.handle_set_cache_policy` to reject.

        Position within the body does not matter — the printer re-emits the
        declarations first, so the roundtrip is position-normalising.
        """
        decls: list[tuple[ir.Var, int]] = []
        for child in body:
            if not isinstance(child, ast.Expr) or not is_cache_policy_marker(child.value):
                continue
            if scope_kind not in CACHE_POLICY_SCOPE_KINDS:
                raise ParserSyntaxError(
                    f"pl.set_cache_policy() has nothing to attach to on this {scope_kind.name} scope",
                    span=self.services.span_tracker.get_span(child),
                    hint=CACHE_POLICY_REJECT_HINTS.get(scope_kind, CACHE_POLICY_POSITION_HINT),
                )
            var, policy = self.resolve_cache_policy_marker(child)
            self.consumed_cache_policy_markers[id(child)] = child
            # A repeated declaration for the same binding is redundant when it
            # restates the same policy — keep the first so the attr stays a set
            # of distinct tensors. A repeat that names a *different* policy is a
            # contradiction, not a redundancy: silently keeping the first would
            # resolve it toward BYPASS, the direction that also asserts the
            # coherency contract the second statement retracts.
            existing = next((p for v, p in decls if v is var), None)
            if existing is None:
                decls.append((var, policy))
            elif existing != policy:
                raise ParserSyntaxError(
                    f"pl.set_cache_policy() declares conflicting policies for '{var.name_hint}' "
                    f"in one scope: {CACHE_POLICY_NAMES[existing]} then "
                    f"{CACHE_POLICY_NAMES[policy]}",
                    span=self.services.span_tracker.get_span(child),
                    hint="One tensor takes one policy per scope. Drop the redundant "
                    "declaration, or move the differing one into its own scope. To vary "
                    "the policy per access, use pl.load(..., cache=...) instead.",
                )
        return decls

    def merge_cache_policy_attr(
        self,
        attrs: "list[tuple[str, Any]] | None",
        decls: "list[tuple[ir.Var, int]]",
    ) -> "list[tuple[str, Any]] | None":
        """Attach hoisted cache-policy declarations to a scope's attr list.

        Placed right after ``dump_vars`` and before ``task_id_var`` / ``slot_num``
        for the same reason :meth:`ScopeParser.merge_forward_sticky_dump` picks that slot:
        keeping a canonical order makes first parse and print -> reparse stable.
        Structural comparison treats attrs as an order-insensitive key map.
        """
        if not decls:
            return attrs
        new_attrs: list[tuple[str, Any]] = list(attrs) if attrs else []
        insert_at = next(
            (i for i, (k, _) in enumerate(new_attrs) if k in {"task_id_var", "slot_num"}),
            len(new_attrs),
        )
        new_attrs.insert(insert_at, (CACHE_POLICY_VARS_ATTR, decls))
        return new_attrs

    def parse_at_kwargs(self, call: ast.Call) -> "AtKwargState":
        """Extract level, role, split mode, deps, and name from pl.at(...).

        Supports both positional and keyword forms. The split mode is configured
        through the ``optimizations=[...]`` list with ``pl.split(...)`` entries.

        Returns the populated :class:`_AtKwargState`. ``deps_kw`` carries the
        verbatim ``deps=`` keyword AST when present, so the caller can resolve it
        into ``Var`` refs once it has decided this scope opts into the
        ``manual_dep_edges`` path.
        """
        if len(call.args) > 2:
            raise ParserSyntaxError(
                f"pl.at() takes at most 2 positional arguments, got {len(call.args)}",
                span=self.services.span_tracker.get_span(call),
                hint="Use pl.at(level) or pl.at(level, role)",
            )

        state = AtKwargState()
        if len(call.args) >= 1:
            state.level = extract_enum_value(call.args[0], LEVEL_MAP, "Level", "pl.Level")
        if len(call.args) >= 2:
            state.role = extract_enum_value(call.args[1], ROLE_MAP, "Role", "pl.Role")

        for kw in call.keywords:
            self.dispatch_at_keyword(kw, state)

        if state.level is None:
            raise ParserSyntaxError(
                "pl.at() requires a level argument",
                span=self.services.span_tracker.get_span(call),
                hint="Use pl.at(pl.Level.HOST) or pl.at(level=pl.Level.HOST)",
            )

        return state

    def dispatch_at_keyword(self, kw: ast.keyword, state: "AtKwargState") -> None:
        """Dispatch a single pl.at() keyword argument and update state."""
        if kw.arg == "level":
            if state.level is not None:
                raise ParserSyntaxError(
                    "pl.at() got multiple values for argument 'level'",
                    span=self.services.span_tracker.get_span(kw),
                )
            state.level = extract_enum_value(kw.value, LEVEL_MAP, "Level", "pl.Level")
        elif kw.arg == "role":
            if state.role is not None:
                raise ParserSyntaxError(
                    "pl.at() got multiple values for argument 'role'",
                    span=self.services.span_tracker.get_span(kw),
                )
            state.role = extract_enum_value(kw.value, ROLE_MAP, "Role", "pl.Role")
        elif kw.arg == "optimizations":
            self.handle_at_optimizations_kw(kw, state)
        elif kw.arg == "name_hint":
            state.name_hint = self.parse_scope_name_hint(kw.value, "pl.at()")
        elif kw.arg == "allow_early_resolve":
            if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                raise ParserSyntaxError(
                    "pl.at() allow_early_resolve must be a boolean literal (True/False)",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint="Write allow_early_resolve=True to opt this scope into early-dispatch.",
                )
            state.allow_early_resolve = kw.value.value
        elif kw.arg == "windowize":
            if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                raise ParserSyntaxError(
                    "pl.at() windowize must be a boolean literal (True/False)",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint="Write windowize=True to allow local windowization for this InCore scope.",
                )
            state.windowize = kw.value.value
        elif kw.arg in AT_STASH_KWARGS:
            self.stash_at_kwarg(kw, state)
        elif kw.arg is None:
            raise ParserSyntaxError(
                "Unsupported **kwargs in pl.at()",
                span=self.services.span_tracker.get_span(kw),
                hint="Use pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker)",
            )
        else:
            raise ParserSyntaxError(
                f"Unknown keyword argument '{kw.arg}' in pl.at()",
                span=self.services.span_tracker.get_span(kw),
                hint=(
                    "Supported arguments: level, role, optimizations, deps, no_dep_args, dumps, "
                    "allow_early_resolve, predicate, name_hint, windowize"
                ),
            )

    def stash_at_kwarg(self, kw: ast.keyword, state: "AtKwargState") -> None:
        """Stash a verbatim-kept ``pl.at()`` kwarg (``deps`` / ``no_dep_args`` /
        ``dumps``) onto ``state``, rejecting a duplicate."""
        assert kw.arg is not None  # caller dispatches here only for _AT_STASH_KWARGS keys
        attr = AT_STASH_KWARGS[kw.arg]
        if getattr(state, attr) is not None:
            raise ParserSyntaxError(
                f"pl.at() got multiple values for argument '{kw.arg}'",
                span=self.services.span_tracker.get_span(kw),
            )
        setattr(state, attr, kw)

    def handle_at_optimizations_kw(self, kw: ast.keyword, state: "AtKwargState") -> None:
        if state.new_optimizations_kw is not None:
            raise ParserSyntaxError(
                "pl.at() got multiple values for argument 'optimizations'",
                span=self.services.span_tracker.get_span(kw),
            )
        state.new_optimizations_kw = kw
        (
            state.split_mode,
            state.cross_core_slots,
        ) = self.parse_optimizations_list(kw.value)

    def parse_optimizations_list(
        self,
        value: ast.expr,
        *,
        owner: str = "pl.at",
        list_hint: str | None = None,
        entry_hint: str | None = None,
    ) -> tuple["ir.SplitMode | None", "CrossCoreSlot | None"]:
        """Parse ``optimizations=[...]`` for ``pl.at`` or ``pl.spmd``.

        Two entries are recognised, freely combinable because they are
        orthogonal — ``pl.split(MODE)`` sets the cross-core split mode, and
        ``pl.cross_core_slot(slot_num=N)`` sets the cross-core pipe slot count.
        The fully qualified forms (``pl.optimizations.split(MODE)`` etc.) are
        also accepted.

        Args:
            owner: API name used in error messages (e.g. ``"pl.at"``, ``"pl.spmd"``).
            list_hint: Override hint when ``optimizations=`` is not a list literal.
            entry_hint: Override hint for unsupported list entries.

        Returns:
            Tuple ``(split_mode, cross_core_slots)``.
        """
        if list_hint is None:
            list_hint = "Use optimizations=[pl.split(pl.SplitMode.NONE)]."
        if entry_hint is None:
            entry_hint = "Each entry must be pl.split(pl.SplitMode.X) or pl.cross_core_slot(slot_num=N)."
        if not isinstance(value, ast.List):
            raise ParserSyntaxError(
                f"{owner}(optimizations=...) must be a list literal",
                span=self.services.span_tracker.get_span(value),
                hint=list_hint,
            )

        split_mode: ir.SplitMode | None = None
        cross_core_slots: CrossCoreSlot | None = None
        seen_split = False
        seen_cross_core_slot = False
        # Tracks whether cross_core_slots came from the deprecated
        # ``pl.split(slot_num=)`` spelling, so a list carrying both sources for
        # one value is rejected instead of letting entry order decide.
        slot_num_from_split = False

        for entry in value.elts:
            if (parsed := self.try_parse_pl_split(entry)) is not None:
                if seen_split:
                    raise ParserSyntaxError(
                        "Duplicate 'pl.split(...)' in optimizations=[...]",
                        span=self.services.span_tracker.get_span(entry),
                    )
                seen_split = True
                split_mode, slot_num = parsed
                if slot_num is not None:
                    if seen_cross_core_slot:
                        raise ParserSyntaxError(
                            "optimizations=[...] sets the cross-core slot count twice: via the "
                            "deprecated pl.split(slot_num=...) and via pl.cross_core_slot(...)",
                            span=self.services.span_tracker.get_span(entry),
                            hint="Keep only pl.cross_core_slot(slot_num=N).",
                        )
                    cross_core_slots = CrossCoreSlot(slot_num=slot_num)
                    slot_num_from_split = True
            elif (slots := self.try_parse_pl_cross_core_slot(entry)) is not None:
                if seen_cross_core_slot:
                    raise ParserSyntaxError(
                        "Duplicate 'pl.cross_core_slot(...)' in optimizations=[...]",
                        span=self.services.span_tracker.get_span(entry),
                    )
                if slot_num_from_split:
                    raise ParserSyntaxError(
                        "optimizations=[...] sets the cross-core slot count twice: via the "
                        "deprecated pl.split(slot_num=...) and via pl.cross_core_slot(...)",
                        span=self.services.span_tracker.get_span(entry),
                        hint="Drop slot_num= from pl.split(...) and keep pl.cross_core_slot(slot_num=N).",
                    )
                seen_cross_core_slot = True
                cross_core_slots = slots
            else:
                raise ParserSyntaxError(
                    f"Unsupported entry in {owner}(optimizations=[...])",
                    span=self.services.span_tracker.get_span(entry),
                    hint=entry_hint,
                )

        return split_mode, cross_core_slots

    def parse_spmd_optimizations_list(
        self, value: ast.expr
    ) -> "tuple[ir.SplitMode | None, CrossCoreSlot | None]":
        """Parse split mode and pipe depths on ``pl.spmd(..., optimizations=[...])``.

        Returns ``(split_mode, cross_core_slots)``.
        """
        return self.parse_optimizations_list(
            value,
            owner="pl.spmd",
            list_hint="Use optimizations=[pl.split(pl.SplitMode.NONE)].",
            entry_hint="Each entry must be pl.split(pl.SplitMode.X) or pl.cross_core_slot(slot_num=N).",
        )

    def is_pl_optimization_call(self, node: ast.expr, name: str) -> bool:
        """True when ``node`` is ``pl.<name>(...)`` or ``pl.optimizations.<name>(...)``."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != name:
            return False
        # pl.<name>(...)
        if isinstance(func.value, ast.Name) and func.value.id == "pl":
            return True
        # pl.optimizations.<name>(...)
        return (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == "optimizations"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pl"
        )

    def try_parse_pl_split(self, node: ast.expr) -> "tuple[ir.SplitMode, int | None] | None":
        """Return ``(SplitMode, slot_num)`` if the AST node is ``pl.split(MODE)``; else None.

        ``slot_num`` is ``None`` unless the deprecated ``slot_num=N`` keyword is
        given. Also accepts the fully qualified form
        ``pl.optimizations.split(MODE)``.
        """
        if not self.is_pl_optimization_call(node, "split"):
            return None
        assert isinstance(node, ast.Call)  # narrowed by _is_pl_optimization_call

        slot_num: int | None = None
        for kw in node.keywords:
            if kw.arg == "slot_num":
                slot_num = self.eval_slot_num_literal(
                    kw.value,
                    "pl.split(slot_num=...)",
                    hint="Use e.g. pl.cross_core_slot(slot_num=16).",
                )
                # Deprecated spelling: the slot count is orthogonal to the split
                # mode, so it has its own entry now. Warn rather than reject so
                # existing kernels keep working.
                warnings.warn(SPLIT_SLOT_NUM_DEPRECATION, DeprecationWarning, stacklevel=2)
            else:
                raise ParserSyntaxError(
                    f"Unknown keyword argument '{kw.arg}' in pl.split()",
                    span=self.services.span_tracker.get_span(kw),
                    hint="pl.split() accepts only the deprecated slot_num= keyword.",
                )
        if len(node.args) != 1:
            raise ParserSyntaxError(
                f"pl.split() takes exactly 1 positional argument, got {len(node.args)}",
                span=self.services.span_tracker.get_span(node),
                hint="Use pl.split(pl.SplitMode.NONE).",
            )
        mode = extract_enum_value(node.args[0], SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
        return mode, slot_num

    def try_parse_pl_cross_core_slot(self, node: ast.expr) -> "CrossCoreSlot | None":
        """Return the pipe depths if the node is ``pl.cross_core_slot(slot_num=N)``; else None.

        Also accepts ``pl.optimizations.cross_core_slot(slot_num=N)``. The
        ``slot_num=`` keyword is required; ``local_slot_num=`` is optional.
        """
        if not self.is_pl_optimization_call(node, "cross_core_slot"):
            return None
        assert isinstance(node, ast.Call)  # narrowed by _is_pl_optimization_call

        usage_hint = "Use pl.cross_core_slot(slot_num=4)."
        if node.args:
            raise ParserSyntaxError(
                f"pl.cross_core_slot() takes no positional arguments, got {len(node.args)}",
                span=self.services.span_tracker.get_span(node),
                hint=usage_hint,
            )
        slot_num: int | None = None
        local_slot_num: int | None = None
        for kw in node.keywords:
            if kw.arg == "slot_num":
                slot_num = self.eval_slot_num_literal(
                    kw.value, "pl.cross_core_slot(slot_num=...)", hint=usage_hint
                )
            elif kw.arg == "local_slot_num":
                # Explicit None has the same default semantics as omission.
                if not (isinstance(kw.value, ast.Constant) and kw.value.value is None):
                    local_slot_num = self.eval_slot_num_literal(
                        kw.value, "pl.cross_core_slot(local_slot_num=...)", hint=usage_hint
                    )
            else:
                raise ParserSyntaxError(
                    f"Unknown keyword argument '{kw.arg}' in pl.cross_core_slot()",
                    span=self.services.span_tracker.get_span(kw),
                    hint=usage_hint,
                )
        if slot_num is None:
            raise ParserSyntaxError(
                "pl.cross_core_slot() requires the slot_num= keyword argument",
                span=self.services.span_tracker.get_span(node),
                hint=usage_hint,
            )
        try:
            return cross_core_slot(slot_num=slot_num, local_slot_num=local_slot_num)
        except ValueError as exc:
            raise ParserSyntaxError(str(exc), span=self.services.span_tracker.get_span(node)) from exc

    def eval_slot_num_literal(self, value: ast.expr, api: str, *, hint: str) -> int:
        """Evaluate a ``slot_num=`` argument as a positive int literal.

        Args:
            value: The keyword's value AST node.
            api: API spelling used in error messages (e.g. ``"pl.cross_core_slot(slot_num=...)"``).
            hint: Usage hint attached to the "not an integer literal" error.
        """
        # bool is a subclass of int — reject it explicitly.
        if (
            not isinstance(value, ast.Constant)
            or isinstance(value.value, bool)
            or not isinstance(value.value, int)
        ):
            raise ParserSyntaxError(
                f"{api} must be an integer literal",
                span=self.services.span_tracker.get_span(value),
                hint=hint,
            )
        if value.value <= 0:
            raise ParserSyntaxError(
                f"{api} must be positive, got {value.value}",
                span=self.services.span_tracker.get_span(value),
            )
        return value.value

    def parse_scope_name_hint(
        self, value: ast.expr, func_name: str, *, require_identifier: bool = True
    ) -> str:
        """Extract and validate a scope name hint from an AST expression.

        Args:
            value: AST expression node for the name_hint value
            func_name: Function name for error messages (e.g. "pl.at()")
            require_identifier: Whether the name may become an outlined function identifier.

        Returns:
            Validated name hint string.
        """
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ParserSyntaxError(
                f"{func_name} 'name_hint' argument must be a string literal",
                span=self.services.span_tracker.get_span(value),
                hint='Use name_hint="my_scope_name"',
            )
        name_hint = value.value
        if (
            require_identifier
            and name_hint
            and (not name_hint.isidentifier() or _keyword_mod.iskeyword(name_hint))
        ):
            raise ParserSyntaxError(
                f"{func_name} 'name_hint' must be a valid non-keyword identifier, got {name_hint!r}",
                span=self.services.span_tracker.get_span(value),
                hint="Use a valid Python identifier like 'fused_matmul_add'",
            )
        return name_hint

    def parse_scope(self, stmt: ast.With, context_expr: ast.Call) -> None:
        """Parse ``with pl.scope(mode=...):`` into a Runtime scope.

        ``mode`` defaults to ``ScopeMode.AUTO``. AUTO scopes are the explicit IR
        form of the orchestration ``SIMPLER_SCOPE()`` block; MANUAL scopes turn off
        auto dependency tracking (``pl.scope(mode=pl.ScopeMode.MANUAL)`` — the
        former ``pl.manual_scope()``).
        """
        if context_expr.args:
            raise ParserSyntaxError(
                "pl.scope() takes only 'mode=' and 'name_hint=' keywords, not positional arguments",
                span=self.services.span_tracker.get_span(stmt),
                hint="Use 'with pl.scope():' or 'with pl.scope(mode=pl.ScopeMode.MANUAL):'.",
            )
        manual = False
        name_hint = ""
        for kw in context_expr.keywords:
            if kw.arg == "mode":
                manual = extract_enum_value(kw.value, SCOPE_MODE_MAP, "ScopeMode", "pl.ScopeMode")
            elif kw.arg == "name_hint":
                name_hint = self.parse_scope_name_hint(kw.value, "pl.scope()", require_identifier=False)
            else:
                raise ParserSyntaxError(
                    f"pl.scope() got an unexpected keyword '{kw.arg}'",
                    span=self.services.span_tracker.get_span(stmt),
                    hint='Use mode=pl.ScopeMode.AUTO|MANUAL and/or name_hint="label".',
                )
        self.emit_runtime_scope(stmt, manual=manual, name_hint=name_hint)

    def parse_manual_scope(self, stmt: ast.With, context_expr: ast.Call) -> None:
        """Parse ``with pl.manual_scope():`` — an alias for ``pl.scope(mode=MANUAL)``."""
        if context_expr.args:
            raise ParserSyntaxError(
                "pl.manual_scope() does not accept positional arguments",
                span=self.services.span_tracker.get_span(stmt),
                hint="Use 'with pl.manual_scope():' (or 'with pl.scope(mode=pl.ScopeMode.MANUAL):').",
            )
        name_hint = ""
        for kw in context_expr.keywords:
            if kw.arg == "name_hint":
                name_hint = self.parse_scope_name_hint(
                    kw.value, "pl.manual_scope()", require_identifier=False
                )
            else:
                raise ParserSyntaxError(
                    f"pl.manual_scope() got an unexpected keyword '{kw.arg}'",
                    span=self.services.span_tracker.get_span(stmt),
                    hint='The only accepted keyword is name_hint="label".',
                )
        self.emit_runtime_scope(stmt, manual=True, name_hint=name_hint)

    def emit_runtime_scope(self, stmt: ast.With, manual: bool, name_hint: str = "") -> None:
        """Build a RuntimeScopeStmt for a ``with pl.scope(...)`` / ``pl.manual_scope()`` block.

        Enforces the scope-placement rules:
          - MANUAL may not nest inside another MANUAL (runtime forbids).
          - AUTO may not nest inside a MANUAL (runtime forbids AUTO-in-MANUAL).
          - A hand-placed AUTO scope requires ``@pl.function(auto_scope=False)``;
            in the default auto_scope=True the compiler owns AUTO placement, so a
            stray ``with pl.scope():`` would be a no-op — reject it instead.
        """
        span = self.services.span_tracker.get_span(stmt)
        if manual:
            if self.regions.manual_scope_depth > 0:
                raise ParserSyntaxError(
                    "a manual scope may not be nested inside another manual scope",
                    span=span,
                    hint="Flatten the nested manual scope into the enclosing one, or move it outside.",
                )
            self.regions.manual_scope_depth += 1
            try:
                self.parse_scope_body(stmt, ir.ScopeKind.Runtime, span, manual=True, name_hint=name_hint)
            finally:
                self.regions.manual_scope_depth -= 1
            return

        # AUTO scope.
        if self.function.func_auto_scope:
            raise ParserSyntaxError(
                "a hand-placed AUTO 'with pl.scope()' requires @pl.function(auto_scope=False)",
                span=span,
                hint="In the default auto_scope=True mode the compiler places AUTO scopes; set "
                "auto_scope=False to place them yourself, or use pl.scope(mode=pl.ScopeMode.MANUAL).",
            )
        if self.regions.manual_scope_depth > 0:
            raise ParserSyntaxError(
                "an AUTO scope may not be nested inside a manual scope",
                span=span,
                hint="The runtime forbids AUTO scope nested in MANUAL scope; move it outside.",
            )
        self.parse_scope_body(stmt, ir.ScopeKind.Runtime, span, manual=False, name_hint=name_hint)

    def parse_legacy_scope(
        self,
        stmt: ast.With,
        context_expr: ast.Call,
        func_attr: str,
        scope_kind_map: dict[str, "ir.ScopeKind"],
        optional_vars: "ast.expr | None" = None,
    ) -> None:
        """Parse pl.cluster / pl.spmd scope context managers.

        ``optional_vars`` (the ``as <target>`` clause) is only meaningful for
        ``pl.spmd`` (``with pl.spmd(...) as tid:``); the caller rejects it on the
        other kinds before dispatching here.
        """
        name_hint = ""
        if func_attr == "cluster":
            if context_expr.args:
                raise ParserSyntaxError(
                    f"pl.{func_attr}() does not accept positional arguments",
                    span=self.services.span_tracker.get_span(stmt),
                    hint=f"Use 'with pl.{func_attr}():'",
                )
            dump_vars: list[ir.Var] = []
            seen_dumps = False
            for kw in context_expr.keywords:
                if kw.arg == "name_hint":
                    name_hint = self.parse_scope_name_hint(kw.value, f"pl.{func_attr}()")
                elif kw.arg == "dumps":
                    dump_vars = self.parse_scope_dumps_once(kw, "pl.cluster", seen_dumps)
                    seen_dumps = True
                else:
                    raise ParserSyntaxError(
                        f"pl.{func_attr}() got unexpected keyword argument '{kw.arg}'",
                        span=self.services.span_tracker.get_span(stmt),
                        hint="Supported keywords: 'name_hint', 'dumps'. For SPMD dispatch, use pl.spmd(4):",
                    )
            scope_kind = scope_kind_map[func_attr]
            span = self.services.span_tracker.get_span(stmt)
            # ``dumps=`` marks tensors for selective dump on the Group dispatch the
            # cluster outlines to — the same ``dump_vars`` carrier a forward-sticky
            # ``pl.dump_tag`` merges in, and the surface the printer emits it back as.
            self.parse_scope_body(
                stmt,
                scope_kind,
                span,
                name_hint=name_hint,
                attrs=[("dump_vars", dump_vars)] if dump_vars else None,
            )
            return
        self.parse_spmd_scope(stmt, context_expr, scope_kind_map, optional_vars=optional_vars)

    def validate_spmd_kwarg_name(self, kw: ast.keyword, anchor: ast.AST, *, usage_hint: str) -> None:
        """Reject a ``pl.spmd()`` keyword that is not legal in this context.

        Split out of :meth:`ScopeParser.parse_spmd_kwargs` so that method handles only what
        each keyword's *value* means, while "is this keyword name accepted here"
        lives in one place. Covers two rejections:

        * ``**kwargs`` unpacking (``kw.arg is None``) — the parser must see each
          keyword literally.
        * any unrecognised name, whose hint lists the keywords valid here.
        """
        if kw.arg is None:
            # `pl.spmd(**cfg)` — ast.keyword.arg is None for **kwargs unpacking.
            raise ParserSyntaxError(
                "pl.spmd() does not accept **kwargs; pass core_num (positional) "
                "and sync_start=/name_hint=/optimizations= explicitly",
                span=self.services.span_tracker.get_span(kw.value),
                hint=usage_hint,
            )
        if kw.arg in SPMD_KWARGS:
            return
        raise ParserSyntaxError(
            f"pl.spmd() got unexpected keyword argument '{kw.arg}'",
            span=self.services.span_tracker.get_span(anchor),
            hint="Supported keywords: 'sync_start', 'name_hint', 'optimizations', 'deps', "
            "'allow_early_resolve', 'predicate'",
        )

    def parse_spmd_kwargs(
        self,
        anchor: ast.AST,
        call: ast.Call,
        *,
        usage_hint: str,
    ) -> tuple[
        "ir.Expr",
        bool,
        str,
        "ir.SplitMode | None",
        "CrossCoreSlot | None",
        "list[ir.Var]",
        bool,
        "ir.Expr | None",
        "list[ir.Var]",
    ]:
        """Parse the ``pl.spmd(core_num, *, sync_start=, name_hint=, optimizations=, deps=, ...)`` arguments.

        Also accepts ``allow_early_resolve=`` (the speculative early-dispatch
        hint), ``predicate=`` (the dispatch predicate) and ``dumps=`` (selective
        tensor dump). The first positional argument is ``core_num`` (range-like).
        Returns ``(core_num, sync_start, name_hint, split_mode, cross_core_slots,
        dep_vars, allow_early_resolve, predicate, dump_vars)`` with
        ``sync_start`` / ``allow_early_resolve`` defaulting to ``False``,
        ``split_mode`` / ``cross_core_slots`` / ``predicate`` to ``None``, and
        ``dep_vars`` / ``dump_vars`` to ``[]``.

        ``optimizations=[...]`` accepts split and pipe-depth entries — see
        :meth:`ScopeParser.parse_spmd_optimizations_list`.

        ``deps=[...]`` is accepted on every form. It takes the same shapes as
        ``pl.submit(..., deps=)`` / ``pl.at(..., deps=)`` — producer TaskId
        ``Scalar[TASK_ID]`` Vars, an ``Array[N, TASK_ID]`` carry, or the ``None``
        sentinel — resolved via :meth:`DispatchSemantics.parse_submit_deps_kwarg`. Capturing the
        dispatch TaskId with ``as tid`` is orthogonal: a scope that declares
        edges without capturing one still lowers to a ``Submit`` (the Spmd
        outliner synthesises the unused TaskId Var), exactly as
        ``pl.at(..., deps=[...])`` already does without an ``as`` clause.

        ``allow_early_resolve=True/False`` is a speculative early-dispatch hint
        (same as ``pl.submit`` / ``pl.at``); it is always accepted here (it needs
        no ``as tid``), and a cluster-nesting guard at the call site rejects it
        when the dispatch would be unwrapped into a Group function.

        ``predicate=(t[i] > 0)`` is a dispatch predicate — the same surface and
        the same validation as ``pl.spmd_submit(..., predicate=)``, shared via
        :meth:`DispatchSemantics.parse_submit_predicate_kwarg`. It rides on the ``SpmdScopeStmt``
        until ``OutlineSpmdScopes`` moves it onto the synthesised ``Submit``.
        Like ``allow_early_resolve`` and ``deps=`` it needs no ``as tid`` (the
        outliner synthesises a TaskId Var when the scope has none) and is
        rejected by the same cluster-nesting guard. The producer-in-``deps=``
        contract is checked below, once ``dep_vars`` is resolved.

        ``dumps=[t, ...]`` marks tensors for selective dump on the grid dispatch,
        resolved via :meth:`ScopeParser.parse_scope_dumps_kwarg` exactly as on ``pl.at``. It
        rides on the ``SpmdScopeStmt`` (never on an auto-synthesised InCore
        carrier) — which is also where the printer reads it back — and the Spmd
        outliner moves it onto the synthesised dispatch. Unlike the Submit-only
        kwargs it is legal on a ``pl.cluster()``-nested scope:
        ``OutlineClusterScopes`` lifts it onto the Group dispatch together with
        the launch spec.
        """
        if len(call.args) > 1:
            raise ParserSyntaxError(
                "pl.spmd() accepts at most one positional argument (core_num)",
                span=self.services.span_tracker.get_span(call.args[1]),
                hint=usage_hint,
            )
        core_num: ir.Expr | None = None
        if call.args:
            core_num = self.dispatch.parse_and_validate_core_num(call.args[0], call.args[0], usage_hint)
        sync_start: bool = False
        name_hint = ""
        split_mode: ir.SplitMode | None = None
        cross_core_slots: CrossCoreSlot | None = None
        deps_kw: ast.keyword | None = None
        dump_vars: list[ir.Var] = []
        seen_dumps = False
        allow_early_resolve: bool = False
        predicate: ir.Expr | None = None
        for kw in call.keywords:
            self.validate_spmd_kwarg_name(kw, anchor, usage_hint=usage_hint)
            if kw.arg == "name_hint":
                name_hint = self.parse_scope_name_hint(kw.value, "pl.spmd()")
            elif kw.arg == "core_num":
                if core_num is not None:
                    raise ParserSyntaxError(
                        "pl.spmd() got multiple values for argument 'core_num'",
                        span=self.services.span_tracker.get_span(kw.value),
                        hint=usage_hint,
                    )
                core_num = self.dispatch.parse_and_validate_core_num(kw.value, kw.value, usage_hint)
            elif kw.arg == "sync_start":
                sync_start = self.dispatch.parse_spmd_bool_literal_kwarg(kw, usage_hint)
            elif kw.arg == "optimizations":
                split_mode, cross_core_slots = self.parse_spmd_optimizations_list(kw.value)
            elif kw.arg == "deps":
                deps_kw = kw
            elif kw.arg == "dumps":
                dump_vars = self.parse_scope_dumps_once(kw, "pl.spmd", seen_dumps)
                seen_dumps = True
            elif kw.arg == "allow_early_resolve":
                allow_early_resolve = self.dispatch.parse_spmd_bool_literal_kwarg(kw, usage_hint)
            elif kw.arg == "predicate":
                # Not a bool literal — parsed as an ordinary expression and
                # shape-validated, exactly as on pl.spmd_submit.
                predicate = self.dispatch.parse_submit_predicate_kwarg("pl.spmd()", [kw])
        if core_num is None:
            raise ParserSyntaxError(
                "pl.spmd() requires core_num (first positional argument)",
                span=self.services.span_tracker.get_span(anchor),
                hint=usage_hint,
            )
        dep_vars: list[ir.Var] = []
        if deps_kw is not None:
            anchor_span = self.services.span_tracker.get_span(anchor)
            dep_vars = self.dispatch.parse_submit_deps_kwarg("pl.spmd()", [deps_kw], anchor_span)
        # Producer-in-deps contract, checked once dep_vars is known. Same
        # best-effort spot check as pl.spmd_submit: the scheduler reads the
        # operand at the dispatch point, so its producing task must be a
        # dependency or the read may observe a stale value. Every pl.spmd form
        # accepts deps=, so the remediation hint is always "add the producer".
        if predicate is not None:
            self.dispatch.validate_predicate_deps(
                "pl.spmd()", predicate, dep_vars, self.services.span_tracker.get_span(anchor)
            )
        return (
            core_num,
            sync_start,
            name_hint,
            split_mode,
            cross_core_slots,
            dep_vars,
            allow_early_resolve,
            predicate,
            dump_vars,
        )

    @staticmethod
    def build_spmd_scope_attrs(
        dep_vars: "list[ir.Var]",
        allow_early_resolve: bool,
        predicate: "ir.Expr | None",
        task_id_var: "ir.Var | None" = None,
        dump_vars: "list[ir.Var] | None" = None,
    ) -> "list[tuple[str, Any]]":
        """Build a ``SpmdScopeStmt``'s attr list in canonical order.

        The single source of the ordering shared by all three ``pl.spmd`` forms:
        ``manual_dep_edges``, ``dump_vars``, ``task_id_var`` (``as tid`` only),
        then ``allow_early_resolve``, then ``predicate`` — mirroring
        :meth:`ScopeParser.parse_at_meta` for ``pl.at`` scopes and the printer's emission
        order in ``python_printer.cpp``. ``structural_equal`` matches attrs by
        key, so the order is not needed for a print -> reparse round-trip; one
        canonical order keeps freshly parsed IR deterministic, and building it in
        one place is what makes adding a kwarg a two-file change instead of a
        four-site one.

        Each entry is omitted when unset, so a plain ``with pl.spmd(n):`` yields
        an empty list (the caller passes ``attrs=... or None``).
        """
        attrs: list[tuple[str, Any]] = []
        if dep_vars:
            attrs.append(("manual_dep_edges", dep_vars))
        if dump_vars:
            attrs.append(("dump_vars", dump_vars))
        if task_id_var is not None:
            attrs.append(("task_id_var", task_id_var))
        if allow_early_resolve:
            attrs.append(("allow_early_resolve", True))
        if predicate is not None:
            attrs.append(("predicate", predicate))
        return attrs

    @staticmethod
    def spmd_body_reads_block_idx(body: "list[ast.stmt]") -> bool:
        """True if any statement in an SPMD body calls ``get_block_idx()``.

        This is what marks a body as *inline* — it does the per-block work itself
        and so must be outlined into a per-block kernel. A body that instead
        dispatches to a kernel reads the index inside the callee and is classified
        by :meth:`_spmd_body_dispatches_kernel`; see :meth:`ScopeParser.emit_spmd_body`.

        Matched at the AST layer (no IR ``Op`` exists yet) by the trailing call name,
        so every valid spelling of the API counts regardless of receiver:
        ``pl.get_block_idx()`` (the top-level alias real models use), the qualified
        ``pl.tile.get_block_idx()`` / ``tile.get_block_idx()``, and a bare
        ``get_block_idx()`` imported directly. Matching by name only is deliberately
        lenient: ``get_block_idx`` is unique to this API (no other DSL object exposes
        it), and being lenient here is far safer than rejecting a real body that
        distinguishes blocks.
        """
        return spmd_body_has_call(
            body,
            lambda node: (isinstance(node.func, ast.Attribute) and node.func.attr == "get_block_idx")
            or (isinstance(node.func, ast.Name) and node.func.id == "get_block_idx"),
        )

    def explicit_incore_carrier(self, body: "list[ast.stmt]") -> "AtKwargState | None":
        """The parsed ``pl.at`` state if ``body`` is a lone ``with pl.at(<CORE_GROUP>, ...):``.

        That nested scope *is* the ``InCoreScopeStmt`` carrier, so
        :meth:`ScopeParser.emit_spmd_body` must parse it as an ordinary nested scope rather
        than synthesising a second one around it (which would violate the
        ``NoNestedInCore`` structural property).

        This is the shape the printer emits for ``Spmd(InCore(...))`` whenever
        neither the ``as tid`` nor the ``for``-form branch applies — see
        ``IRPythonPrinter::VisitStmt_(SpmdScopeStmtPtr)``'s plain-with
        fallthrough. Matching it here is what makes that output re-parseable.

        The level is resolved through :meth:`ScopeParser.parse_at_kwargs`, so the positional
        form (``pl.at(pl.Level.CORE_GROUP)``) counts too and the enum name is
        validated against ``LEVEL_MAP`` rather than string-matched. An
        ``as tid`` capture on the carrier is irrelevant here — it rides on the
        nested scope either way. Only ``CORE_GROUP`` qualifies: every other level
        builds a ``HierarchyScopeStmt``, which is not an InCore carrier (see
        :meth:`ScopeParser.parse_at_scope`).

        Returns the parsed :class:`_AtKwargState` (never ``None`` for a match) so
        the caller can see whether the carrier itself carries an ``optimizations=``
        entry; ``None`` when ``body`` is not a carrier.
        """
        if len(body) != 1 or not isinstance(body[0], ast.With):
            return None
        items = body[0].items
        if len(items) != 1 or not is_pl_call(items[0].context_expr, "at"):
            return None
        state = self.parse_at_kwargs(items[0].context_expr)
        return state if state.level == ir.Level.CORE_GROUP else None

    def spmd_body_kernel_dispatch_count(self, body: "list[ast.stmt]") -> int:
        """Number of cross-function kernel dispatches in an SPMD body.

        A body with at least one dispatch is a *dispatch* body: the per-block work
        lives in the callee, which reads the block index internally, so the body
        itself need not mention ``get_block_idx``.

        The count matters because an unwrapped dispatch body is lowered by
        ``FindFirstInnerCall`` (``wrapper_call_utils.cpp``), which stops at the
        first call — orchestration codegen would emit a launch for that callee only
        and silently drop the rest. Hoisted temporaries and tuple projections are
        not dispatches, so the shapes ``FlattenCallExpr`` and the multi-output
        desugar produce still count as exactly one.

        Both call spellings :meth:`ASTParser.parse_call` accepts count: the ``self.<kernel>``
        cross-function call, and a bare name resolving through ``closure_vars`` to
        an external ``@pl.function`` / ``@pl.inline``.
        """
        from .decorator import InlineFunction  # noqa: PLC0415 (circular import)

        def is_dispatch(node: ast.Call) -> bool:
            if is_self_method_call(node):
                return True
            if isinstance(node.func, ast.Name):
                resolved = self.services.expr_evaluator.closure_vars.get(node.func.id)
                return isinstance(resolved, (ir.Function, InlineFunction))
            return False

        return sum(1 for _ in spmd_body_calls(body, is_dispatch))

    def emit_spmd_body(  # noqa: PLR0913 — args map 1:1 to the SpmdScopeStmt fields
        self,
        stmt: ast.With,
        span: "ir.Span",
        scope_kind: "ir.ScopeKind",
        core_num: "ir.Expr",
        sync_start: bool,
        name_hint: str,
        split_mode: "ir.SplitMode | None",
        cross_core_slots: "CrossCoreSlot | None",
        scope_attrs: "list[tuple[str, Any]]",
    ) -> None:
        """Emit the ``SpmdScopeStmt`` body shared by the plain and ``as tid`` with-forms.

        The two forms differ only in ``scope_attrs`` (the ``as tid`` form adds
        ``task_id_var`` / ``manual_dep_edges``); the body dispatch is identical:

        The body shape decides whether an inner ``InCoreScopeStmt`` carrier is
        synthesised. The test is semantic — does the body do per-block work
        itself, or dispatch it? — not a statement count: a dispatch body may
        legally hold several statements once ``FlattenCallExpr`` has hoisted a
        nested call arg, or the multi-output tuple desugar has split an assign
        into a temp plus projections.

        * an explicit ``with pl.at(<CORE_GROUP level>, ...):`` body → that nested
          scope already *is* the carrier; parse it as an ordinary nested scope
          (see :meth:`_is_explicit_incore_carrier`). This is the form the printer
          emits, so recognising it here is what makes ``Spmd(InCore(...))``
          round-trip.
        * dispatch body (does not read the per-block index), no split and no slot
          count → ``SpmdScopeStmt(body=<stmts>)`` with no inner InCore wrapper —
          the callee is a pre-defined kernel that reads the block index
          internally. This is also the shape ``OutlineIncoreScopes`` leaves behind
          once an inline body is outlined, so the IR round-trips identically
          across passes.
        * inline body (reads the per-block index), or any body carrying an
          ``optimizations=`` entry → wrap in ``InCoreScopeStmt(split, <body>)`` for
          ``OutlineIncoreScopes`` to outline into a synthetic per-block kernel,
          exactly like ``for i in pl.spmd(n):``. Both entries lower onto the InCore
          scope (``split_`` and the ``slot_num`` attr), so either one forces the
          wrapper.

        A body that neither reads the index nor dispatches a kernel does no
        per-block differentiation at all and is rejected (see below). An explicit
        carrier is exempt from that check: the user wrote the InCore scope
        themselves, so the body is taken as deliberate.
        """
        # ``cross_core_slots`` is weighed alongside ``split_mode`` throughout: the
        # slot count lands on the *InCore* scope (OutlineIncoreScopes reads
        # ``slot_num`` off InCoreScopeStmt only), so either entry alone is enough
        # to require a carrier — and enough to collide with one the body already
        # provides. Before pl.cross_core_slot existed the two were coupled:
        # slot_num could only arrive via pl.split(mode, slot_num=N), which always
        # set a mode.
        has_optimization_entry = split_mode is not None or cross_core_slots is not None
        carrier = self.explicit_incore_carrier(stmt.body)
        if carrier is not None:
            # The nested ``pl.at(CORE_GROUP)`` already is the carrier, and an
            # ``optimizations=`` entry has to land on it. One on the
            # ``pl.spmd(...)`` line as well is either a duplicate or would have to
            # be pushed into a scope the user wrote by hand — reject both rather
            # than silently picking one or dropping it.
            if has_optimization_entry:
                carrier_has_entry = carrier.split_mode is not None or carrier.cross_core_slots is not None
                where = (
                    "is specified twice: on `pl.spmd(...)` and on the inner"
                    if carrier_has_entry
                    else "belongs on the InCore scope, but the body already provides one as the inner"
                )
                raise ParserSyntaxError(
                    f"`optimizations=` {where} `pl.at(level=pl.Level.CORE_GROUP, ...)`",
                    span=span,
                    hint="Keep `optimizations=[...]` on the inner `pl.at(...)` carrier only.",
                )
            needs_incore = False
        else:
            is_inline_body = self.spmd_body_reads_block_idx(stmt.body)
            dispatch_count = 0 if is_inline_body else self.spmd_body_kernel_dispatch_count(stmt.body)
            if not is_inline_body and dispatch_count == 0:
                raise ParserSyntaxError(
                    "`with pl.spmd(...)` body neither reads the per-block index via "
                    "`pl.tile.get_block_idx()` nor dispatches a `self.<kernel>(...)` call, "
                    "so every block would run identical work.",
                    span=span,
                    hint="Add `i = pl.tile.get_block_idx()` inside the scope, dispatch a kernel "
                    "that reads it, or use `for i in pl.spmd(n):` to bind the block index "
                    "automatically.",
                )
            # An unwrapped dispatch body is lowered via FindFirstInnerCall, which
            # stops at the first call — a second dispatch would be silently
            # dropped by orchestration codegen rather than launched.
            if dispatch_count > 1:
                raise ParserSyntaxError(
                    f"`with pl.spmd(...)` dispatch body launches {dispatch_count} kernels; "
                    "only one kernel dispatch per SPMD scope is supported.",
                    span=span,
                    hint="Use one `pl.spmd(...)` scope per kernel, or make the body inline by "
                    "reading `pl.tile.get_block_idx()` so the whole body is outlined into a "
                    "single per-block kernel.",
                )
            needs_incore = is_inline_body or has_optimization_entry
        if not needs_incore:
            # Either the body already carries its own InCore, or it is a dispatch
            # body whose per-block work lives in the callee. Any ``scope_attrs``
            # (allow_early_resolve, and for the ``as tid`` form task_id_var /
            # manual_dep_edges) ride on the SpmdScopeStmt.
            #
            # ``cross_core_slots`` is checked alongside ``split_mode``: the slot
            # count lands on the *InCore* scope (OutlineIncoreScopes reads
            # ``slot_num`` off InCoreScopeStmt only), so taking this wrapper-less
            # path with a slot count set would silently discard it. Before
            # pl.cross_core_slot existed the two were coupled — slot_num could
            # only arrive via pl.split(mode, slot_num=N), which always set a mode.
            self.parse_scope_body(
                stmt,
                scope_kind,
                span,
                name_hint=name_hint,
                core_num=core_num,
                sync_start=sync_start,
                attrs=scope_attrs or None,
            )
            return
        # split= hint or an inline multi-statement body requires an inner
        # InCoreScopeStmt to carry split_ / be outlined. Build the scope directly
        # instead of routing through _parse_scope_body, so merge any forward-sticky
        # pl.dump_tag tensors onto it here (see _parse_spmd_for_loop for the full
        # rationale).
        spmd_name_hint, incore_name_hint = split_spmd_for_loop_name_hints(name_hint)
        incore_attrs = self.merge_forward_sticky_dump(None, ir.ScopeKind.InCore)
        incore_attrs = self.append_cross_core_slots_attr(incore_attrs, cross_core_slots)
        # Cache-policy declarations attach to the InCore carrier, not the Spmd
        # scope: the carrier is what gets outlined into the per-block kernel
        # whose params the policy resolves against. That is also the scope the
        # printer re-emits them from, since the Spmd printer inlines this
        # carrier's header away.
        incore_attrs = self.merge_cache_policy_attr(
            incore_attrs, self.collect_scope_cache_policy_decls(stmt.body, ir.ScopeKind.InCore)
        )
        with self.services.builder.scope(
            scope_kind,
            span,
            name_hint=spmd_name_hint,
            core_num=core_num,
            sync_start=sync_start,
            attrs=scope_attrs or None,
        ):
            with self.regions.scope_kind_context(scope_kind):
                self.services.scope_manager.enter_scope("spmd_with")
                with self.services.builder.scope(
                    ir.ScopeKind.InCore,
                    span,
                    split=split_mode,
                    name_hint=incore_name_hint,
                    attrs=incore_attrs,
                ):
                    with (
                        self.regions.scope_kind_context(ir.ScopeKind.InCore),
                        self.regions.incore_user_split_context(ir.ScopeKind.InCore, split_mode),
                    ):
                        self.services.scope_manager.enter_scope("spmd_with_incore")
                        self.callbacks.body(stmt.body)
                        self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                        self.services.scope_manager.exit_scope(leak_vars=True)
                self.services.scope_manager.exit_scope(leak_vars=True)

    def parse_spmd_scope(
        self,
        stmt: ast.With,
        context_expr: ast.Call,
        scope_kind_map: dict[str, "ir.ScopeKind"],
        optional_vars: "ast.expr | None" = None,
    ) -> None:
        """Parse ``with pl.spmd(...):`` / ``with pl.spmd(...) as tid:`` into a ScopeStmt(Spmd).

        Two forms, differing only in whether the grid dispatch's producer TaskId is
        captured — the body shape is identical (see :meth:`ScopeParser.emit_spmd_body`):

        * ``with pl.spmd(n):`` — no captured TaskId. Accepts either a dispatch
          body calling a pre-defined kernel (direct dispatch) or an inline body
          auto-outlined into an InCore kernel (like ``for i in pl.spmd(n):``,
          minus the auto-bound loop var — read the per-block index inside via
          ``pl.tile.get_block_idx()``).
        * ``with pl.spmd(n, deps=[...]) as tid:`` — same body shapes, and
          additionally captures the producer ``Scalar[TASK_ID]`` (mirrors
          ``with pl.at(...) as tid:``) so it can feed a ``deps=`` edge.

        TaskId capture, ``deps=`` and inline bodies are all orthogonal: the inline
        body is outlined the same way with or without ``as tid``; ``as tid`` only
        adds the ``task_id_var`` attr. ``deps=`` alone is enough to force the
        ``ir.Submit`` shape — the Spmd outliner synthesises an unused TaskId Var
        for a scope that declares edges without capturing one, the same way
        ``pl.at(..., deps=[...])`` works without an ``as`` clause.
        """
        with_hint = (
            "Use 'with pl.spmd(4):' with a body that dispatches a 'self.<kernel>(...)' call "
            "or reads 'pl.tile.get_block_idx()', or 'with pl.spmd(4) as tid:' to also "
            "capture the dispatch TaskId."
        )
        (
            core_num,
            sync_start,
            name_hint,
            split_mode,
            cross_core_slots,
            dep_vars,
            allow_early_resolve,
            predicate,
            dump_vars,
        ) = self.parse_spmd_kwargs(stmt, context_expr, usage_hint=with_hint)
        scope_kind = scope_kind_map["spmd"]
        span = self.services.span_tracker.get_span(stmt)

        if optional_vars is not None:
            self.parse_spmd_scope_with_tid(
                stmt,
                span,
                scope_kind,
                core_num,
                sync_start,
                name_hint,
                split_mode,
                cross_core_slots,
                dep_vars,
                allow_early_resolve,
                predicate,
                dump_vars,
                optional_vars,
            )
            return

        # ``deps=`` declares explicit producer edges, ``allow_early_resolve`` opts
        # the grid dispatch into speculative early-dispatch, and ``predicate``
        # gates it at the dispatch point (all three mirror pl.submit /
        # pl.spmd_submit). A cluster-nested pl.spmd is unwrapped into the Group
        # function by OutlineClusterScopes and never lowers to a Submit, so any
        # of them would be silently dropped — reject them here (mirrors the
        # ``as tid`` cluster rejection in _parse_spmd_scope_with_tid).
        self.dispatch.reject_spmd_submit_only_kwargs_in_cluster(
            dep_vars, allow_early_resolve, predicate, span
        )
        # No task_id_var — this form captures none; the rest of the canonical
        # order is shared with the other two forms.
        spmd_attrs = self.build_spmd_scope_attrs(
            dep_vars, allow_early_resolve, predicate, dump_vars=dump_vars
        )

        # No ``as tid``: the plain with-form. Any ``deps=`` rides on the scope as
        # ``manual_dep_edges`` and the Spmd outliner synthesises the TaskId Var it
        # needs to emit a Submit. The shared helper leaves a dispatch body
        # unwrapped and outlines an inline body into a synthetic InCore kernel —
        # identical to the ``as tid`` form, minus the captured TaskId.
        self.emit_spmd_body(
            stmt,
            span,
            scope_kind,
            core_num,
            sync_start,
            name_hint,
            split_mode,
            cross_core_slots,
            spmd_attrs,
        )

    def parse_spmd_scope_with_tid(  # noqa: PLR0913 — args map 1:1 to the SpmdScopeStmt + capture
        self,
        stmt: ast.With,
        span: "ir.Span",
        scope_kind: "ir.ScopeKind",
        core_num: "ir.Expr",
        sync_start: bool,
        name_hint: str,
        split_mode: "ir.SplitMode | None",
        cross_core_slots: "CrossCoreSlot | None",
        dep_vars: "list[ir.Var]",
        allow_early_resolve: bool,
        predicate: "ir.Expr | None",
        dump_vars: "list[ir.Var]",
        optional_vars: "ast.expr",
    ) -> None:
        """Parse ``with pl.spmd(n, deps=[...]) as tid:`` capturing the dispatch TaskId.

        Records ``{manual_dep_edges?, task_id_var}`` on the ``SpmdScopeStmt`` and
        emits the ``system.task_invalid()`` placeholder. The body shape mirrors the
        plain with-form so the IR is identical to the post-``OutlineIncoreScopes``
        shape (a single Call) and round-trips through print -> reparse:

        * dispatch body, no split → ``SpmdScopeStmt(attrs, body=<stmts>)`` (no
          InCore wrapper) — the same shape ``OutlineIncoreScopes`` leaves behind
          once the inline body is outlined, so reparse is stable across passes.
        * inline body (reads the per-block index), or any body with split → wrap in
          ``InCoreScopeStmt(split, <body>)``, exactly like the for-form (minus the
          synthesised ``loop_var = tile.get_block_idx()``; the body reads the block
          index explicitly via ``pl.tile.get_block_idx()``).
        * an explicit ``pl.at(<CORE_GROUP level>)`` body already carries its own
          InCore, so it is parsed as an ordinary nested scope.

        The ``kAttrTaskIdVar`` on the outer Spmd scope makes ``OutlineClusterScopes``
        lower the dispatch to an ``ir.Submit`` whose trailing tuple element is the
        grid-wide producer TaskId — identical to the ``pl.at(...) as tid:`` rail.
        """
        if self.regions.is_inside_scope(ir.ScopeKind.Cluster):
            raise ParserSyntaxError(
                "`with pl.spmd(...) as tid:` cannot capture a TaskId when nested inside "
                "`pl.cluster()` — a cluster-nested pl.spmd is unwrapped into the Group "
                "function and never produces a Submit.",
                span=span,
                hint="Use a standalone `with pl.spmd(...) as tid:` (implicit cluster) to "
                "capture the dispatch TaskId.",
            )
        if not isinstance(optional_vars, ast.Name):
            raise ParserSyntaxError(
                "`as` target on `with pl.spmd(...)` must be a plain variable name",
                span=span,
                hint="Use `with pl.spmd(...) as tid:` (single name; nested tuples are not allowed).",
            )

        # ``task_id_var`` is the only attr unique to this form. The Spmd outliner
        # reads each one off the scope and threads it onto the synthesised Submit;
        # ``predicate`` in particular carries live SSA Vars (the operand tensor and
        # its indices), which ConvertToSSA versions via SubstScopeAttrs.
        tid_var = self.services.builder.var(optional_vars.id, ir.ScalarType(DataType.TASK_ID), span=span)
        self.services.scope_manager.define_var(optional_vars.id, tid_var, span=span)
        scope_attrs = self.build_spmd_scope_attrs(
            dep_vars, allow_early_resolve, predicate, tid_var, dump_vars=dump_vars
        )

        # Emit the transient ``AssignStmt(tid, system.task_invalid())`` placeholder
        # one stmt BEFORE the scope so ConvertToSSA has a def for the tid Var; the
        # spmd outliner drops it and synthesises the real
        # ``AssignStmt(tid, TupleGetItem(ret_tmp, n_outputs))`` binding. Pop/re-push
        # pending leading comments so they land on the surviving SpmdScopeStmt, not
        # the placeholder (mirrors _parse_at_scope).
        leading = self.services.builder.pop_pending_leading_comments()
        placeholder_rhs = ir.create_op_call("system.task_invalid", [], {}, span)
        self.services.builder.assign(tid_var, placeholder_rhs, span=span)
        self.services.builder.push_pending_leading_comments(leading)

        # Emit the body via the shared helper — identical shape to the plain
        # with-form, plus the task_id_var / manual_dep_edges built into scope_attrs
        # above: a lone call (no split) keeps the no-InCore-wrapper direct-dispatch
        # shape; an inline multi-statement body (or single call + split) wraps in an
        # InCoreScopeStmt for outlining and must read the per-block index.
        self.emit_spmd_body(
            stmt,
            span,
            scope_kind,
            core_num,
            sync_start,
            name_hint,
            split_mode,
            cross_core_slots,
            scope_attrs,
        )
        self.dispatch.record_scope_producer(stmt.body, tid_var)

    def parse_spmd_for_loop(self, stmt: ast.For, iter_call: ast.Call) -> None:
        """Parse ``for i in pl.spmd(N, ...): body`` into
        ``SpmdScopeStmt(body=InCoreScopeStmt(body=<bind i; body>))``.

        The loop variable is bound to ``pl.tile.get_block_idx()`` as the first
        statement of the auto-outlined InCore function, giving per-block code
        direct access to the block index without a separate kernel
        declaration.
        """
        spmd_hint = (
            "Use 'for i in pl.spmd(4):' — the loop variable is bound to the "
            "per-block index (equivalent to pl.tile.get_block_idx())."
        )
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "for ... in pl.spmd(...) must use a single loop variable",
                span=self.services.span_tracker.get_span(stmt.target),
                hint=spmd_hint,
            )
        loop_var_name = stmt.target.id

        # Reject loop-specific kwargs that make no sense for SPMD blocks.
        disallowed_loop_kwargs = {"init_values", "chunk", "chunk_policy", "attrs", "step", "stage"}
        for kw in iter_call.keywords:
            if kw.arg in disallowed_loop_kwargs:
                raise ParserSyntaxError(
                    f"pl.spmd() loop form does not accept '{kw.arg}='",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint=spmd_hint,
                )

        # The for-form captures no TaskId, but ``deps=`` does not need one: the
        # edges ride on the SpmdScopeStmt and the Spmd outliner synthesises the
        # TaskId Var required to emit a Submit. Use the with-form
        # `with pl.spmd(n, deps=[...]) as tid:` when the dispatch must also be
        # nameable as a later task's dependency.
        (
            core_num,
            sync_start,
            name_hint,
            split_mode,
            cross_core_slots,
            dep_vars,
            allow_early_resolve,
            predicate,
            dump_vars,
        ) = self.parse_spmd_kwargs(stmt, iter_call, usage_hint=spmd_hint)
        spmd_name_hint, incore_name_hint = split_spmd_for_loop_name_hints(name_hint)

        span = self.services.span_tracker.get_span(stmt)
        # ``deps=`` / ``allow_early_resolve`` / ``predicate`` ride on the
        # SpmdScopeStmt (read by the Spmd outliner onto the synthesised Submit).
        # A cluster-nested pl.spmd is unwrapped into the Group and never produces
        # a Submit, so reject them there (mirrors the with-form / as-tid guards).
        self.dispatch.reject_spmd_submit_only_kwargs_in_cluster(
            dep_vars, allow_early_resolve, predicate, span
        )
        # No task_id_var — the for-form captures none (see the with-form).
        spmd_attrs = self.build_spmd_scope_attrs(
            dep_vars, allow_early_resolve, predicate, dump_vars=dump_vars
        )
        # Merge forward-sticky pl.dump_tag tensors onto the auto-outlined InCore
        # scope — the kernel the loop body lowers to. The with-form (pl.at /
        # pl.spmd / pl.cluster) routes through _parse_scope_body for this; the
        # for-form builds its scope directly, so attach here to keep the two
        # paths symmetric. OutlineIncoreScopes then carries the dump_vars onto
        # the synthesised inner-kernel Call; the wrapper-dispatch codegen
        # (BuildWrapperReorderedParams) honours that inner call's dump_vars.
        incore_attrs = self.merge_forward_sticky_dump(None, ir.ScopeKind.InCore)
        incore_attrs = self.append_cross_core_slots_attr(incore_attrs, cross_core_slots)
        # Same as the with-form: the declarations belong to the InCore carrier
        # the loop body lowers to, which is also the scope the printer re-emits
        # them from (the ``for i in pl.spmd(...)`` form drops the carrier's
        # header entirely).
        incore_attrs = self.merge_cache_policy_attr(
            incore_attrs, self.collect_scope_cache_policy_decls(stmt.body, ir.ScopeKind.InCore)
        )
        with self.services.builder.scope(
            ir.ScopeKind.Spmd,
            span,
            name_hint=spmd_name_hint,
            core_num=core_num,
            sync_start=sync_start,
            attrs=spmd_attrs or None,
        ):
            with self.regions.scope_kind_context(ir.ScopeKind.Spmd):
                self.services.scope_manager.enter_scope("spmd_for")
                with self.services.builder.scope(
                    ir.ScopeKind.InCore,
                    span,
                    split=split_mode,
                    name_hint=incore_name_hint,
                    attrs=incore_attrs,
                ):
                    with (
                        self.regions.scope_kind_context(ir.ScopeKind.InCore),
                        self.regions.incore_user_split_context(ir.ScopeKind.InCore, split_mode),
                    ):
                        # Bind `i = pl.tile.get_block_idx()` as the first
                        # statement of the outlined InCore body.
                        loop_var = self.services.builder.var(
                            loop_var_name, ir.ScalarType(DataType.INDEX), span=span
                        )
                        self.services.scope_manager.define_var(loop_var_name, loop_var)
                        self.services.builder.assign(loop_var, ir_op.tile.get_block_idx(span=span), span=span)
                        self.callbacks.body(stmt.body)
                self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                # Leak vars to parent so post-store rebindings (e.g.
                # ``out = pl.store(...)`` inside the body) remain visible to
                # subsequent statements like ``return out``. Matches Python's
                # own for-loop variable-leaking semantics.
                self.services.scope_manager.exit_scope(leak_vars=True)

    def parse_split_aiv_for_loop(self, stmt: ast.For, iter_call: ast.Call) -> None:
        """Parse ``for aiv_id in pl.split_aiv(2, mode=...): body`` into a
        first-class ``SplitAivScopeStmt`` region.

        The region is a structural node that may appear anywhere in an InCore
        body — including inside a ``pl.range`` / ``pl.pipeline`` loop or an
        ``if`` — and carries the requested ``SplitMode`` on
        ``SplitAivScopeStmt::split_``. The loop variable is bound to
        ``pl.tile.get_subblock_idx()`` (the AIV lane / sub-core index) as the
        first statement of the region body. ``LowerAutoVectorSplit`` (pass 23)
        consumes and erases the node; it never reaches codegen.
        """
        split_aiv_hint = (
            "Use 'for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):' — n is the AIV "
            "sub-core count (hardware-fixed at 2); 'mode' is required (NONE = task-parallel, "
            "no halving; UP_DOWN / LEFT_RIGHT = data-parallel halving); and the loop variable "
            "binds the AIV lane index (equivalent to pl.tile.get_subblock_idx())."
        )
        # A pl.split_aiv loop must not be nested inside another pl.split_aiv body:
        # one split_aiv body already represents the two AIV lanes, so re-partitioning
        # them is not a meaningful (or lowerable) pattern. _split_aiv_mode_stack is
        # non-empty exactly while parsing inside an enclosing split_aiv body.
        if self.regions.split_aiv_mode_stack:
            raise ParserSyntaxError(
                "nested 'for ... in pl.split_aiv(...)' is not allowed: a split_aiv body already "
                "represents the two AIV lanes, so it cannot contain another split_aiv loop",
                span=self.services.span_tracker.get_span(stmt),
                hint=split_aiv_hint,
            )
        # Placement check, like the nested-region one above: an enclosing InCore
        # scope that declares its own optimizations=[pl.split(...)] already picked
        # the AUTO split mechanism, which this region would silently override.
        self.regions.reject_user_split_with_split_aiv_region(stmt, split_aiv_hint)
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "for ... in pl.split_aiv(...) must use a single loop variable",
                span=self.services.span_tracker.get_span(stmt.target),
                hint=split_aiv_hint,
            )
        loop_var_name = stmt.target.id

        # ``pl.split_aiv`` IS the split declaration, so a co-present
        # ``optimizations=[pl.split(...)]`` would be a second, conflicting split
        # spec. Loop-carried / chunking kwargs make no sense for an SPMD-style
        # split body either. Reject both with targeted diagnostics.
        disallowed_loop_kwargs = {"init_values", "chunk", "chunk_policy", "attrs", "step", "stage"}
        for kw in iter_call.keywords:
            if kw.arg in disallowed_loop_kwargs:
                raise ParserSyntaxError(
                    f"pl.split_aiv() loop form does not accept '{kw.arg}='",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint=split_aiv_hint,
                )
            if kw.arg == "optimizations":
                raise ParserSyntaxError(
                    "pl.split_aiv() does not accept 'optimizations=' — pl.split_aiv() IS the "
                    "split declaration; a co-present pl.split(...) is a conflicting split spec",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint=split_aiv_hint,
                )

        # ``n`` (the AIV sub-core count) is positional and hardware-fixed at 2.
        if len(iter_call.args) != 1:
            raise ParserSyntaxError(
                "pl.split_aiv() takes exactly one positional argument (n, the AIV sub-core count)",
                span=self.services.span_tracker.get_span(iter_call),
                hint=split_aiv_hint,
            )
        n_expr = self.callbacks.expression(iter_call.args[0])
        if not (isinstance(n_expr, ir.ConstInt) and n_expr.value == SPLIT_AIV_SUBCORE_NUM):
            got = n_expr.value if isinstance(n_expr, ir.ConstInt) else python_print(n_expr, format=False)
            raise ParserSyntaxError(
                f"pl.split_aiv(n) requires n == {SPLIT_AIV_SUBCORE_NUM} "
                f"(AIV sub-core count is hardware-fixed at {SPLIT_AIV_SUBCORE_NUM}), got {got}",
                span=self.services.span_tracker.get_span(iter_call.args[0]),
                hint=split_aiv_hint,
            )

        # ``mode`` is a required keyword — no silent default.
        split_mode: ir.SplitMode | None = None
        for kw in iter_call.keywords:
            if kw.arg is None:
                raise ParserSyntaxError(
                    "pl.split_aiv() does not accept **kwargs; pass n (positional) and mode= explicitly",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint=split_aiv_hint,
                )
            if kw.arg == "mode":
                split_mode = extract_enum_value(kw.value, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
            elif kw.arg not in disallowed_loop_kwargs and kw.arg != "optimizations":
                raise ParserSyntaxError(
                    f"pl.split_aiv() got unexpected keyword argument '{kw.arg}'",
                    span=self.services.span_tracker.get_span(kw.value),
                    hint=split_aiv_hint,
                )
        if split_mode is None:
            raise ParserSyntaxError(
                "pl.split_aiv() requires mode= (e.g. mode=pl.SplitMode.NONE for task-parallel, "
                "or pl.SplitMode.UP_DOWN / pl.SplitMode.LEFT_RIGHT for data-parallel halving)",
                span=self.services.span_tracker.get_span(iter_call),
                hint=split_aiv_hint,
            )

        # Build a first-class SplitAivScopeStmt region. The region body begins
        # with ``aiv_id = pl.tile.get_subblock_idx()`` and carries the requested
        # SplitMode on the node; LowerAutoVectorSplit (pass 23) consumes it.
        #
        # FLATTEN: when the region is ALREADY in a core context, emit it in place.
        # The wrapper below exists only to give OutlineIncoreScopes something to
        # outline, so it is meaningless once we are core-side. Two ways to be:
        #   1. Inside an open CORE_GROUP InCore scope (directly, or through an
        #      intervening pl.range/pl.pipeline/if): the region nests in the open
        #      context and survives the enclosing function's outlining.
        #   2. The enclosing FUNCTION is already InCore, so no scope is open and
        #      the open-scope test alone reads False.
        #
        # Arm 2 is what makes print -> parse a fixed point: after outlining, the
        # region is bare at the top of an InCore ``*_incore_0``, and reparsing that
        # printed form must rebuild it bare. Emitting what the syntax means is the
        # parser's job; whether a region is ALLOWED in this function is a placement
        # rule, enforced by AivSplitValid check (h) — which keys on the ``split_aiv``
        # attr OutlineIncoreScopes stamps, so a hand-authored InCore function (no
        # such attr) is still rejected while the outlined one is accepted.
        if (
            self.regions.is_inside_scope(ir.ScopeKind.InCore)
            or self.function.func_type == ir.FunctionType.InCore
        ):
            self.emit_split_aiv_region(stmt, loop_var_name, split_mode)
            return

        # Bare top-level form (no enclosing InCore scope, and the function is not
        # itself InCore — e.g. a plain ``@pl.function`` / ``@pl.jit`` Opaque body):
        # a top-level split_aiv must live inside a core function, so synthesize an
        # InCore wrapper first and nest the region inside it (keeps it eligible for
        # OutlineIncoreScopes — else the region would have no enclosing InCore to
        # outline). Merge any forward-sticky pl.dump_tag tensors onto the wrapper
        # (mirrors the other InCore-creating paths); the split mode + split_aiv
        # marker ride the nested SplitAivScopeStmt region node, not the wrapper.
        span = self.services.span_tracker.get_span(stmt)
        incore_attrs = self.merge_forward_sticky_dump(None, ir.ScopeKind.InCore)
        with self.services.builder.scope(ir.ScopeKind.InCore, span, attrs=incore_attrs):
            with self.regions.scope_kind_context(ir.ScopeKind.InCore):
                self.emit_split_aiv_region(stmt, loop_var_name, split_mode)

    def emit_split_aiv_region(self, stmt: ast.For, loop_var_name: str, split_mode: ir.SplitMode) -> None:
        """Emit a first-class ``SplitAivScopeStmt`` region at the current point.

        Opens a ``ScopeKind.SplitAiv`` scope carrying ``split_mode`` and binds
        ``aiv_id = pl.tile.get_subblock_idx()`` as its first statement, then
        parses the loop body inside it. Used by both arms of
        :meth:`ScopeParser.parse_split_aiv_for_loop` (inside-InCore and the bare,
        InCore-wrapped form).
        """
        span = self.services.span_tracker.get_span(stmt)
        with self.services.builder.scope(ir.ScopeKind.SplitAiv, span, split=split_mode):
            with self.regions.scope_kind_context(ir.ScopeKind.SplitAiv):
                # A fresh var scope keeps the loop var / body bindings tidy;
                # leak_vars pushes them up so subsequent statements stay visible.
                self.services.scope_manager.enter_scope("split_aiv_for")
                # Bind `aiv_id = pl.tile.get_subblock_idx()` as the first
                # statement of the region body.
                loop_var = self.services.builder.var(loop_var_name, ir.ScalarType(DataType.INDEX), span=span)
                self.services.scope_manager.define_var(loop_var_name, loop_var)
                self.services.builder.assign(loop_var, ir_op.tile.get_subblock_idx(span=span), span=span)
                # Expose the split mode to ``pl.aiv_shard`` / ``pl.aic_gather``
                # calls in the body, which inherit it from this region.
                with self.regions.split_aiv_mode_context(split_mode):
                    self.callbacks.body(stmt.body)
                self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                self.services.scope_manager.exit_scope(leak_vars=True)

    def merge_forward_sticky_dump(
        self,
        attrs: "list[tuple[str, Any]] | None",
        scope_kind: "ir.ScopeKind",
    ) -> "list[tuple[str, Any]] | None":
        """Merge forward-sticky ``pl.dump_tag`` tensors into a scope's dump_vars attr.

        The single injection point for the scope-level selective-dump carrier on
        first parse — the explicit / round-trip ``dumps=`` surface (on ``pl.at`` /
        ``pl.spmd`` / ``pl.cluster`` / ``pl.graph``) is handled separately by
        :meth:`ScopeParser.parse_scope_dumps_kwarg`, and merged here with the tags. Both the ``pl.at`` and the
        ``pl.cluster`` paths route through
        :meth:`ScopeParser.parse_scope_body`, so attaching here covers every scope kind that
        becomes a kernel dispatch. Runtime scopes (``pl.manual_scope`` /
        ``pl.auto_scope``) are skipped: they are not outlined into a dispatch, and
        ``pl.submit`` inside a manual scope carries its own per-call ``dump_vars``.

        Tags are captured at scope entry (forward-sticky), so they are bound
        before the scope and live at its entry — exactly the SSA version the
        synthesised dispatch receives as an arg. Entries the scope never consumes
        are dropped later by the outliner. ``dump_vars`` is kept before
        ``task_id_var`` so a print -> reparse (which rebuilds the canonical order
        via :meth:`ScopeParser.parse_at_meta`) preserves the printed attr order.
        Structural comparison itself ignores the order of attr keys.
        """
        if scope_kind == ir.ScopeKind.Runtime:
            return attrs
        tagged = [v for v in self.function.dump_tagged_vars if isinstance(v.type, ir.TensorType)]
        if not tagged:
            return attrs

        new_attrs: list[tuple[str, Any]] = list(attrs) if attrs else []
        for i, (k, v) in enumerate(new_attrs):
            if k == "dump_vars":
                merged = list(v)
                seen = {id(x) for x in merged}
                for t in tagged:
                    if id(t) not in seen:
                        merged.append(t)
                        seen.add(id(t))
                new_attrs[i] = ("dump_vars", merged)
                return new_attrs
        # Insert ``dump_vars`` before ``task_id_var`` AND before the trailing
        # ``slot_num`` so the canonical order (dump_vars ... task_id_var,
        # slot_num) matches what a print -> reparse rebuilds via _parse_at_meta +
        # _append_cross_core_slots_attr; structural equality ignores attr key order.
        insert_at = next(
            (i for i, (k, _) in enumerate(new_attrs) if k in {"task_id_var", "slot_num"}),
            len(new_attrs),
        )
        new_attrs.insert(insert_at, ("dump_vars", tagged))
        return new_attrs

    def parse_scope_body(  # noqa: PLR0913 — kwargs map 1:1 to ScopeStmt fields
        self,
        stmt: ast.With,
        scope_kind: "ir.ScopeKind",
        span: "ir.Span",
        *,
        level: "ir.Level | None" = None,
        role: "ir.Role | None" = None,
        split: "ir.SplitMode | None" = None,
        name_hint: str = "",
        core_num: "ir.Expr | None" = None,
        sync_start: bool | None = None,
        manual: bool | None = None,
        attrs: "list[tuple[str, Any]] | None" = None,
    ) -> None:
        """Build a scope statement from a with-statement body."""
        attrs = self.merge_forward_sticky_dump(attrs, scope_kind)
        # Hoist the body's ``pl.set_cache_policy`` markers onto this scope before
        # it begins — ``begin_scope`` fixes the attrs, so there is no later point
        # at which a marker found mid-body could still attach.
        attrs = self.merge_cache_policy_attr(
            attrs, self.collect_scope_cache_policy_decls(stmt.body, scope_kind)
        )
        with self.services.builder.scope(
            scope_kind,
            span,
            level=level,
            role=role,
            split=split,
            name_hint=name_hint,
            core_num=core_num,
            sync_start=sync_start,
            manual=manual,
            attrs=attrs,
        ):
            with (
                self.regions.scope_kind_context(scope_kind),
                self.regions.incore_user_split_context(scope_kind, split),
            ):
                self.services.scope_manager.enter_scope("scope")
                self.callbacks.body(stmt.body)
                self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                self.services.scope_manager.exit_scope(leak_vars=True)

    def parse_graph_scope(self, stmt: ast.With, context_expr: ast.Call) -> None:
        """Parse ``with pl.graph("name"):`` into a GraphScopeStmt.

        The region name is positional and required: ``OutlineGraphScopes`` names
        the outlined function after it, codegen derives the emitted symbol from
        that name, and the runtime keys its recorded Definition on the symbol. An
        auto-generated name would still work but would silently change whenever
        an unrelated region was added, so the user owns it.
        """
        span = self.services.span_tracker.get_span(stmt)

        # An Opaque body is orchestration that has not been classified yet (a plain
        # ``@pl.function`` / ``@pl.jit`` parses Opaque and is promoted by the
        # outlining passes), and an Inline body is spliced into its caller before
        # OutlineGraphScopes runs — so the rejection set is the device-side kernel
        # types, not "everything that is not Orchestration yet".
        if self.function.func_type in GRAPH_FORBIDDEN_FUNC_TYPES:
            raise ParserSyntaxError(
                f"pl.graph() is not valid inside a {self.function.func_type.name} function",
                span=span,
                hint="A Graph region records orchestration task launches, so it belongs in the "
                "@pl.jit entry (or another @pl.jit.graph function), not in a kernel body. To make a "
                "whole function a Graph, decorate it with @pl.jit.graph instead.",
            )

        # Hierarchy is here for a different reason than the other three: a
        # `pl.at(level=HOST)` region is distributed orchestration, and Graph x
        # distributed is explicitly out of scope for now (RFC #2399), so reject it
        # at the surface rather than let it reach a pass that has never seen it.
        for device_kind, construct in (
            (ir.ScopeKind.InCore, "pl.at(level=...)"),
            (ir.ScopeKind.Hierarchy, "pl.at(level=..., role=...)"),
            (ir.ScopeKind.Cluster, "pl.cluster()"),
            (ir.ScopeKind.Spmd, "pl.spmd(...)"),
        ):
            if self.regions.is_inside_scope(device_kind):
                raise ParserSyntaxError(
                    f"pl.graph() cannot be nested inside `{construct}`",
                    span=span,
                    hint=(
                        "A Graph region records a topology of task launches, so it must sit "
                        "around the dispatches rather than inside one. Move the "
                        "`with pl.graph(...):` outside."
                        if device_kind is not ir.ScopeKind.Hierarchy
                        else "Graph regions inside a distributed hierarchy region are not "
                        "supported yet. Move the `with pl.graph(...):` outside."
                    ),
                )

        if self.regions.is_inside_scope(ir.ScopeKind.Graph):
            raise ParserSyntaxError(
                "pl.graph() cannot be nested inside another pl.graph()",
                span=span,
                hint="The host_build_graph runtime cannot record a graph inside a graph. Mark "
                "either the outer region or the inner one, not both.",
            )

        if len(context_expr.args) != 1:
            raise ParserSyntaxError(
                f"pl.graph() takes exactly one positional argument (the region name), "
                f"got {len(context_expr.args)}",
                span=span,
                hint="Use 'with pl.graph(\"decoder_layer\"):'",
            )
        # ``dumps=`` marks tensors for selective dump on the Graph task the region
        # outlines to (codegen emits it on the graph dispatch itself, distinct from
        # the kernel dispatches recorded inside) — the same ``dump_vars`` carrier a
        # forward-sticky ``pl.dump_tag`` merges in, and the surface the printer
        # emits it back as.
        dump_vars: list[ir.Var] = []
        seen_dumps = False
        for kw in context_expr.keywords:
            if kw.arg != "dumps":
                raise ParserSyntaxError(
                    f"pl.graph() got unexpected keyword argument '{kw.arg or '**kwargs'}'",
                    span=span,
                    hint="pl.graph() takes the region name and an optional `dumps=[...]`: "
                    "'with pl.graph(\"decoder_layer\"):'",
                )
            dump_vars = self.parse_scope_dumps_once(kw, "pl.graph", seen_dumps)
            seen_dumps = True

        name = self.parse_scope_name_hint(context_expr.args[0], "pl.graph()")
        if not name:
            raise ParserSyntaxError(
                "pl.graph() region name must not be empty",
                span=span,
                hint="The name becomes the recorded graph's symbol: 'with pl.graph(\"decoder_layer\"):'",
            )

        self.parse_scope_body(
            stmt,
            ir.ScopeKind.Graph,
            span,
            name_hint=name,
            attrs=[("dump_vars", dump_vars)] if dump_vars else None,
        )

    def parse_at_scope(
        self, stmt: ast.With, context_expr: ast.Call, optional_vars: "ast.expr | None" = None
    ) -> None:
        """Parse pl.at(...) context manager into a ScopeStmt."""
        state = self.parse_at_kwargs(context_expr)
        level = state.level
        role = state.role
        split_mode = state.split_mode
        name_hint = state.name_hint
        deps_kw = state.deps_kw
        no_dep_args_kw = state.no_dep_args_kw
        dumps_kw = state.dumps_kw
        assert level is not None  # _parse_at_kwargs raises if level is missing
        span = self.services.span_tracker.get_span(stmt)

        is_core_group = level == ir.Level.CORE_GROUP

        # Both optimizations= entries lower onto the InCore scope: pl.split(...)
        # into split_, pl.cross_core_slot(...) into the slot_num attr. Neither
        # has a meaning on a Hierarchy scope, so reject either at a non-CORE_GROUP
        # level rather than silently attaching a dead attr.
        if (split_mode is not None or state.cross_core_slots is not None) and not is_core_group:
            raise ParserSyntaxError(
                "optimizations=[pl.split(...)] / optimizations=[pl.cross_core_slot(...)] are only "
                "supported with level=pl.Level.CORE_GROUP",
                span=span,
                hint="Use pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]).",
            )

        if is_core_group and role is not None:
            raise ParserSyntaxError(
                "role= is not supported with level=pl.Level.CORE_GROUP",
                span=span,
                hint="Drop role= for InCore scopes, or use a non-CORE_GROUP level for Hierarchy scope",
            )

        # Build the optional ``manual_dep_edges`` / ``task_id_var`` attrs for the
        # synthesised ScopeStmt. ``deps=[tid1, tid2]`` accepts the same shapes
        # as ``pl.submit(..., deps=)`` — TaskId Vars, the ``None`` sentinel, or
        # an ``Array[N, TASK_ID]`` carry. ``with pl.at(...) as tid:`` binds a
        # fresh ``Scalar[TASK_ID]`` Var in the outer scope; the outliner
        # later wires it to ``TupleGetItem(call_lhs, last_idx)``.
        predicate = self.parse_at_predicate(state.predicate_kw, is_core_group, span)
        scope_attrs = self.parse_at_meta(
            deps_kw, no_dep_args_kw, dumps_kw, optional_vars, state.allow_early_resolve, predicate, span
        )
        scope_attrs = self.append_cross_core_slots_attr(scope_attrs, state.cross_core_slots)
        if state.windowize:
            scope_attrs = list(scope_attrs or [])
            scope_attrs.append(("windowize", True))

        # ``with pl.at(...) as tid:`` allocates ``tid`` as an outer-scope Var
        # whose real definition is synthesised later by ``OutlineIncoreScopes``
        # (as ``AssignStmt(tid, TupleGetItem(call_lhs, last_idx))``). The
        # outlining pass runs *after* ConvertToSSA, so we emit a placeholder
        # ``AssignStmt(tid, system.task_invalid())`` *before* the scope to give
        # ConvertToSSA a def that gets renamed consistently with the scope's
        # ``task_id_var`` attr reference and subsequent ``deps=[tid]`` uses.
        # ``ScopeOutliner::VisitStmt_(SeqStmtsPtr)`` detects this placeholder
        # by looking one stmt *behind* each target scope and drops it once it
        # has generated the real binding.
        #
        # The placeholder is *transient*: the outliner drops it. Any leading
        # comments attached to the ``with`` statement therefore must NOT land
        # on the placeholder — they belong on the surviving scope. We pop the
        # pending-comment stack here and re-push it so the scope, emitted
        # next, absorbs them.
        tid_var: ir.Var | None = None
        if scope_attrs is not None:
            for k, v in scope_attrs:
                if k == "task_id_var":
                    tid_var = v
                    leading = self.services.builder.pop_pending_leading_comments()
                    placeholder_rhs = ir.create_op_call("system.task_invalid", [], {}, span)
                    self.services.builder.assign(v, placeholder_rhs, span=span)
                    self.services.builder.push_pending_leading_comments(leading)
                    break

        if not is_core_group:
            # SubWorker scopes are no longer supported as inline `with pl.at(...)`
            # blocks. Declare a SubWorker via @pl.function(level=..., role=Worker).
            if ir.level_to_linqu_level(level) >= 3 and role == ir.Role.SubWorker:
                raise ParserSyntaxError(
                    "Inline 'with pl.at(level>=HOST, role=pl.Role.SubWorker)' is not supported.",
                    span=span,
                    hint="Declare a SubWorker via @pl.function(level=..., role=pl.Role.SubWorker) "
                    "as a self-contained function (no 'self' parameter) inside @pl.program.",
                )
            self.parse_scope_body(
                stmt,
                ir.ScopeKind.Hierarchy,
                span,
                level=level,
                role=role,
                name_hint=name_hint,
                attrs=scope_attrs,
            )
        else:
            self.parse_scope_body(
                stmt,
                ir.ScopeKind.InCore,
                span,
                split=split_mode,
                name_hint=name_hint,
                attrs=scope_attrs,
            )

        # Record the tensors this scope writes as produced by its ``tid``, so a
        # later ``predicate=`` over one of them is checked against ``deps=``.
        # Same helper (and same best-effort limits) as the ``pl.spmd`` scope
        # form; ``_parse_scope_body`` leaks body-bound names into the enclosing
        # scope, so the lookups below resolve.
        if tid_var is not None:
            self.dispatch.record_scope_producer(stmt.body, tid_var)

    def parse_at_predicate(
        self, kw: "ast.keyword | None", is_core_group: bool, span: "ir.Span"
    ) -> "ir.Expr | None":
        """Parse ``pl.at(..., predicate=(t[i] > 0))`` — the placements that dispatch.

        A dispatch predicate only has a runtime carrier where the scope becomes
        an independently submitted L0 task, i.e. an InCore scope that
        ``OutlineIncoreScopes`` turns into a ``Submit`` orchestration codegen
        dispatches. Three placements never get there and would drop it silently:

        * a non-CORE_GROUP ``pl.at`` builds a Hierarchy scope, which is outlined
          into an Opaque level/role function orchestration codegen never
          dispatches as a task;
        * a ``pl.at`` nested inside ``pl.cluster()`` / ``pl.spmd()`` / another
          CORE_GROUP ``pl.at`` is folded into the enclosing Group / Spmd wrapper
          dispatch (or into the enclosing kernel body), and codegen emits
          ``set_predicate`` from the *outer* call only;
        * a ``pl.at`` written directly in a function body ``OutlineIncoreScopes``
          skips — ``Group`` / ``Spmd`` / an InCore-family kernel — is never
          outlined into a dispatch at all, so the scope reaches codegen still
          carrying the attr and nothing reads it.

        These gates read the function being parsed, so they cannot see through an
        ``@pl.function(type=Inline)`` helper: its body parses on its own, and only
        ``InlineFunctions`` (pass 1) reveals which call site it lands in. A helper
        called at orchestration top level is a legal placement, so ``Inline`` is
        accepted here and the outlining passes re-check the placement they can
        actually see (``AssertDispatchPredicatesHaveACarrier`` in
        ``OutlineIncoreScopes``, ``AssertNoInnerDispatchPredicate`` in
        ``OutlineClusterScopes``), reporting a user-facing error there.

        Nesting inside a *non*-CORE_GROUP ``pl.at`` is **not** rejected: the
        inner InCore scope is still outlined into its own ``Submit`` (whether the
        enclosing Hierarchy scope is itself outlined or survives in place), and
        the emitted orchestration C++ carries the same ``set_predicate`` as the
        un-nested form.

        Each rejection is raised here rather than miscompiled, mirroring
        :meth:`DispatchSemantics.reject_spmd_submit_only_kwargs_in_cluster`. The shape checks are
        shared with the call form via
        :meth:`DispatchSemantics.parse_submit_predicate_kwarg`; the operand's producer must also
        be in ``deps=`` (checked in :meth:`ScopeParser.parse_at_meta`, which owns the
        resolved dep list).
        """
        if kw is None:
            return None
        if not is_core_group:
            raise ParserSyntaxError(
                "pl.at(predicate=...) is only supported with level=pl.Level.CORE_GROUP",
                span=span,
                hint="A non-CORE_GROUP pl.at builds a Hierarchy scope, which is never outlined into a "
                "task dispatch, so the dispatch predicate would be lost. Use "
                "pl.at(level=pl.Level.CORE_GROUP, predicate=...) or pl.spmd_submit(..., predicate=...).",
            )
        for kind, owner in (
            (ir.ScopeKind.Cluster, "pl.cluster()"),
            (ir.ScopeKind.Spmd, "pl.spmd()"),
            (ir.ScopeKind.InCore, "another pl.at(level=pl.Level.CORE_GROUP)"),
        ):
            if self.regions.is_inside_scope(kind):
                raise ParserSyntaxError(
                    f"`pl.at(..., predicate=(...))` cannot be nested inside `{owner}` — the inner "
                    "dispatch is folded into the enclosing task, so the dispatch predicate would be lost.",
                    span=span,
                    hint="Move the predicate to the enclosing dispatch, or use a top-level "
                    "`with pl.at(level=pl.Level.CORE_GROUP, predicate=...):`.",
                )
        # ``OutlineIncoreScopes`` (pass 9) only rewrites ``Opaque`` and
        # orchestration-like bodies; a Group / Spmd / InCore-family function is
        # left untouched, so a scope written there never becomes a dispatch and
        # the attr rides straight through to codegen. ``Inline`` is accepted:
        # ``InlineFunctions`` (pass 1) substitutes the body into its caller, and
        # a caller at orchestration top level is a legal placement. Which caller
        # it is cannot be known here, so the outlining passes re-check it.
        if self.function.func_type not in (
            ir.FunctionType.Opaque,
            ir.FunctionType.Inline,
        ) and not ir.is_orchestration_like(self.function.func_type):
            raise ParserSyntaxError(
                "`pl.at(..., predicate=(...))` is not supported inside a "
                f"{self.function.func_type.name} function — the scope is never outlined into its own task "
                "dispatch there, so the dispatch predicate would be lost.",
                span=span,
                hint="Write the predicated `with pl.at(level=pl.Level.CORE_GROUP, predicate=...):` "
                "in the orchestration function that launches this kernel.",
            )
        return self.dispatch.parse_submit_predicate_kwarg("pl.at()", [kw])

    def parse_at_meta(
        self,
        deps_kw: "ast.keyword | None",
        no_dep_args_kw: "ast.keyword | None",
        dumps_kw: "ast.keyword | None",
        optional_vars: "ast.expr | None",
        allow_early_resolve: bool,
        predicate: "ir.Expr | None",
        span: "ir.Span",
    ) -> "list[tuple[str, Any]] | None":
        """Build the ScopeStmt ``attrs`` list from a ``pl.at(...)`` ``deps=`` /
        ``no_dep_args=`` / ``dumps=`` / ``predicate=`` kwarg set and a
        ``with ... as <tid>:`` capture target.

        Returns ``None`` when none are present, leaving the scope's ``attrs_``
        empty (the typical plain ``pl.at(...)`` case). Otherwise returns a list
        with up to six reserved keys, always in this canonical order (so a
        print -> reparse cycle reproduces it byte-for-byte; structural_equal
        compares scope attrs positionally):

          * ``manual_dep_edges``: ``list[VarPtr]`` — same shape as the
            ``pl.submit(..., deps=)`` attr; consumed by codegen via
            ``Arg::set_dependencies``.
          * ``arg_direction_overrides_vars``: ``list[VarPtr]`` — outer-scope
            tensor Vars whose corresponding arg slots on the synthesised Call
            must be ``ArgDirection.NoDep``. The outliner translates this Var
            list into positional indices using the captured-var order and
            writes the result back as ``arg_direction_overrides`` on the Call.
          * ``dump_vars``: ``list[VarPtr]`` — outer-scope tensor Vars to mark for
            selective tensor dump. Seeded by ``pl.dump_tag`` (forward-sticky,
            from :attr:`_dump_tagged_vars`) at parse and from an explicit
            ``dumps=`` kwarg (also the print/reparse roundtrip surface). The
            outliner translates this into the synthesised dispatch's ``kAttrDumpVars``.
          * ``task_id_var``: ``VarPtr`` — the outer-scope ``Scalar[TASK_ID]``
            Var the outliner binds to the producer TaskId tuple element.
          * ``allow_early_resolve``: ``bool`` — speculative early-dispatch hint.
          * ``predicate``: ``ExprPtr`` — the dispatch-predicate comparison,
            already shape- and placement-validated by
            :meth:`ScopeParser.parse_at_predicate`. Unlike the keys above this is a whole
            Expr carrying live SSA Vars (the operand tensor and its indices);
            the outliner moves it onto ``Submit::predicate_``.

        The ``tid`` Var is defined in the outer scope so subsequent statements
        (e.g. another ``pl.at(..., deps=[tid])``) can reference it.
        """
        # ``dumps=`` is the explicit scope-level dump surface (symmetric with
        # ``deps=``) and also the print/reparse round-trip surface. The
        # forward-sticky ``pl.dump_tag`` seed is merged in later by
        # :meth:`ScopeParser.parse_scope_body` (the single injection point shared by the
        # ``pl.at`` and ``pl.cluster`` paths), so it is
        # not consulted here.
        dump_vars: list[ir.Var] = self.parse_scope_dumps_kwarg(dumps_kw, "pl.at") if dumps_kw else []

        if (
            deps_kw is None
            and no_dep_args_kw is None
            and not dump_vars
            and optional_vars is None
            and not allow_early_resolve
            and predicate is None
        ):
            return None

        attrs: list[tuple[str, Any]] = []

        dep_vars = (
            self.dispatch.parse_submit_deps_kwarg("pl.at()", [deps_kw], span) if deps_kw is not None else []
        )
        if dep_vars:
            # Attr keys mirror the C++ ``kAttrManualDepEdges`` /
            # ``kAttrTaskIdVar`` / ``kAttrArgDirOverrideVars`` /
            # ``kAttrDumpVars`` constants (include/pypto/ir/expr.h); passed
            # as raw strings since they are not exposed to Python.
            attrs.append(("manual_dep_edges", dep_vars))

        if no_dep_args_kw is not None:
            no_dep_vars = self.parse_at_no_dep_args_kwarg(no_dep_args_kw)
            if no_dep_vars:
                attrs.append(("arg_direction_overrides_vars", no_dep_vars))

        if dump_vars:
            attrs.append(("dump_vars", dump_vars))

        if optional_vars is not None:
            if not isinstance(optional_vars, ast.Name):
                raise ParserSyntaxError(
                    "`as` target on `with pl.at(...)` must be a plain variable name",
                    span=span,
                    hint="Use `with pl.at(...) as tid:` (single name; nested tuples are not allowed).",
                )
            tid_var = self.services.builder.var(optional_vars.id, ir.ScalarType(DataType.TASK_ID), span=span)
            self.services.scope_manager.define_var(optional_vars.id, tid_var, span=span)
            attrs.append(("task_id_var", tid_var))

        # ``allow_early_resolve`` — a plain bool the outliner reads off the scope
        # and threads onto the synthesised Submit.
        if allow_early_resolve:
            attrs.append(("allow_early_resolve", True))

        # ``predicate`` last (canonical order, matching the pl.spmd scope form) —
        # an Expr, not a flag, moved onto ``Submit::predicate_`` by the outliner.
        # The producer-in-deps contract is checked here because this is where the
        # resolved dep list lives; ``pl.at`` accepts ``deps=`` on every form, so
        # the default ``deps_accepted=True`` hint is always the right advice.
        if predicate is not None:
            self.dispatch.validate_predicate_deps("pl.at()", predicate, dep_vars, span)
            attrs.append(("predicate", predicate))

        return attrs if attrs else None

    def parse_at_no_dep_args_kwarg(self, kw: "ast.keyword") -> list[ir.Var]:
        """Resolve ``pl.at(no_dep_args=[t1, t2])`` entries to outer-scope tensor Vars.

        Each entry must be a bare Name that resolves, via the surrounding
        scope manager, to a Tensor-typed Var captured by the scope body. The
        outliner later translates the returned Var list into positional
        indices into the synthesised Call's ``args_``, then attaches them as
        ``arg_direction_overrides`` so ``DeriveCallDirections`` overwrites
        those slots to ``ArgDirection.NoDep`` — the same effect as
        ``pl.no_dep(t)`` at an explicit kernel call site.

        The kwarg is named ``no_dep_args=`` (not the more symmetric-looking
        ``no_deps=``) because it describes *kernel-call argument slots*
        that should skip auto-dep tracking — distinct from ``deps=`` on
        the same call, which takes *producer TaskIds* and adds explicit
        edges. Same word "dep", different layer; the ``_args`` suffix
        flags that the entries are call-argument tensors, not TaskIds.

        Returns an empty list for ``no_dep_args=[]`` so callers can treat
        it as a no-op.
        """
        if not isinstance(kw.value, (ast.List, ast.Tuple)):
            raise ParserTypeError(
                "pl.at(no_dep_args=...) must be a list literal of tensor names",
                span=self.services.span_tracker.get_span(kw),
                hint="Use `no_dep_args=[t1, t2]` with bare tensor names captured by the scope body.",
            )

        resolved: list[ir.Var] = []
        seen: set[int] = set()
        for elt in kw.value.elts:
            if not isinstance(elt, ast.Name):
                raise ParserTypeError(
                    "pl.at(no_dep_args=[...]) entries must be bare tensor names",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Use `no_dep_args=[t]` where `t` is a tensor variable visible "
                    "to the enclosing function scope.",
                )
            var = self.services.scope_manager.lookup_var(elt.id)
            if var is None:
                raise ParserTypeError(
                    f"pl.at(no_dep_args=[...]) references unknown name '{elt.id}'",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Each entry must resolve to a tensor visible in the enclosing function scope.",
                )
            if not isinstance(var.type, ir.TensorType):
                raise ParserTypeError(
                    f"pl.at(no_dep_args=[...]) entry '{elt.id}' is not a tensor (got type {var.type})",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Only tensors can be marked NoDep; scalars and other "
                    "types do not participate in dependency tracking.",
                )
            if id(var) in seen:
                raise ParserTypeError(
                    f"pl.at(no_dep_args=[...]) lists '{elt.id}' more than once",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Each tensor may appear at most once in `no_dep_args=`.",
                )
            seen.add(id(var))
            resolved.append(var)
        return resolved

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports:
        - with pl.cluster(): ... (creates ScopeStmt with Cluster scope)
        - with pl.graph("name"): ... (creates ScopeStmt with Graph scope)
        - with pl.at(level=..., role=...): ... (creates ScopeStmt with InCore/Hierarchy scope)
        - with pl.at(level=CORE_GROUP): ... (creates ScopeStmt with InCore scope)
        - with pl.at(level=CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]): ...
          (InCore with split)

        Args:
            stmt: With AST node
        """
        # Check that we have exactly one context manager
        if len(stmt.items) != 1:
            raise ParserSyntaxError(
                "Only single context manager supported in with statement",
                span=self.services.span_tracker.get_span(stmt),
                hint="Use 'with pl.cluster():', 'with pl.spmd(...):',"
                " 'with pl.at(level=...):', 'with pl.scope():', or"
                " 'with pl.manual_scope():'"
                " without multiple context managers",
            )

        item = stmt.items[0]
        context_expr = item.context_expr
        optional_vars = item.optional_vars  # the ``as <target>`` clause, if any

        if _is_dump_call(context_expr, "comm_domain"):
            self.parse_comm_domain_scope(stmt, context_expr)
            return

        # Map DSL function names to ScopeKind values
        _SCOPE_KIND_MAP = {
            "cluster": ir.ScopeKind.Cluster,
            "spmd": ir.ScopeKind.Spmd,
        }

        if isinstance(context_expr, ast.Call):
            func = context_expr.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "pl":
                # ``as <target>`` binds the producer TaskId of the outlined kernel
                # dispatch. It is meaningful on ``pl.at(...) as tid:`` (InCore /
                # Hierarchy scope) and ``pl.spmd(...) as tid:`` (grid dispatch).
                # Reject it on every other scope construct so misuses surface at
                # parse time rather than silently dropping the binding.
                if optional_vars is not None and func.attr not in ("at", "spmd"):
                    raise ParserSyntaxError(
                        f"`with pl.{func.attr}(...) as ...:` is not supported "
                        "— the `as` clause only applies to `with pl.at(...) as tid:` and "
                        "`with pl.spmd(...) as tid:`",
                        span=self.services.span_tracker.get_span(stmt),
                        hint="Drop the `as` target, or use `with pl.at(...) as tid:` / "
                        "`with pl.spmd(...) as tid:` to capture the outlined dispatch's "
                        "producer TaskId.",
                    )

                # Unified runtime scope: with pl.scope(mode=...): ...
                if func.attr == "scope":
                    self.parse_scope(stmt, context_expr)
                    return

                # Manual scope alias: with pl.manual_scope(): ...
                if func.attr == "manual_scope":
                    self.parse_manual_scope(stmt, context_expr)
                    return

                # Existing scope kinds: pl.cluster(), pl.spmd()
                if func.attr in _SCOPE_KIND_MAP:
                    self.parse_legacy_scope(
                        stmt, context_expr, func.attr, _SCOPE_KIND_MAP, optional_vars=optional_vars
                    )
                    return

                # with pl.graph("name"): ...
                if func.attr == "graph":
                    self.parse_graph_scope(stmt, context_expr)
                    return

                # pl.at(level=..., role=..., deps=...) [as tid]
                if func.attr == "at":
                    self.parse_at_scope(stmt, context_expr, optional_vars=optional_vars)
                    return

        # Unsupported context manager
        raise UnsupportedFeatureError(
            "Unsupported context manager in with statement",
            span=self.services.span_tracker.get_span(stmt),
            hint="Supported: 'with pl.cluster():', 'with pl.spmd(...):',"
            " 'with pl.at(level=..., optimizations=[...]):', 'with pl.graph(\"name\"):',"
            " 'with pl.scope():', or 'with pl.manual_scope():'",
        )

    @staticmethod
    def append_cross_core_slots_attr(
        attrs: "list[tuple[str, Any]] | None", slots: "CrossCoreSlot | None"
    ) -> "list[tuple[str, Any]] | None":
        """Append pipe depths in a stable order for print/parse round-trips.

        Keep omitted local depth absent so existing scope IR stays unchanged.
        """
        if slots is None:
            return attrs
        result: list[tuple[str, Any]] = list(attrs) if attrs else []
        result.append(("slot_num", slots.slot_num))
        if slots.local_slot_num is not None:
            result.append(("local_slot_num", slots.local_slot_num))
        return result

    def parse_scope_dumps_once(self, kw: "ast.keyword", api: str, seen: bool) -> list[ir.Var]:
        """Reject a duplicate scope ``dumps=`` before resolving its entries."""
        if seen:
            raise ParserSyntaxError(
                f"{api}() got multiple values for argument 'dumps'",
                span=self.services.span_tracker.get_span(kw),
            )
        return self.parse_scope_dumps_kwarg(kw, api)

    def parse_scope_dumps_kwarg(self, kw: "ast.keyword", api: str) -> list[ir.Var]:
        """Resolve a scope's ``dumps=[t1, t2]`` entries to outer-scope tensor Vars.

        Shared by every scope construct with a ``dumps=`` kwarg — ``pl.at``,
        ``pl.spmd``, ``pl.cluster`` and ``pl.graph`` — each of which may carry
        ``kAttrDumpVars``; ``api`` (e.g. ``"pl.at"``) names the construct in
        error messages.

        Mirrors :meth:`ScopeParser.parse_at_no_dep_args_kwarg`: each entry must be a bare
        Name resolving to a Tensor-typed Var. ``dumps=`` is the explicit
        scope-level selective-dump surface (symmetric with ``deps=``) and also
        the print/reparse round-trip surface — the printer emits it for any
        scope carrying ``kAttrDumpVars`` (seeded by ``pl.dump_tag`` at parse, by
        an explicit ``dumps=`` list, and by the inline-call ``dump_vars``
        transfer). The outliner translates the returned Var list into the
        synthesised dispatch's ``kAttrDumpVars`` by Var identity; entries the
        scope does not actually capture are skipped there (no error), so unlike
        ``no_dep_args`` there is no capture requirement at parse.

        Returns an empty list for ``dumps=[]`` so callers treat it as a no-op.
        """
        if not isinstance(kw.value, (ast.List, ast.Tuple)):
            raise ParserTypeError(
                f"{api}(dumps=...) must be a list literal of tensor names",
                span=self.services.span_tracker.get_span(kw),
                hint="Use `dumps=[t1, t2]` with bare tensor names visible to the enclosing function.",
            )

        resolved: list[ir.Var] = []
        seen: set[int] = set()
        for elt in kw.value.elts:
            if not isinstance(elt, ast.Name):
                raise ParserTypeError(
                    f"{api}(dumps=[...]) entries must be bare tensor names",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Use `dumps=[t]` where `t` is a tensor variable visible "
                    "to the enclosing function scope.",
                )
            var = self.services.scope_manager.lookup_var(elt.id)
            if var is None:
                raise ParserTypeError(
                    f"{api}(dumps=[...]) references unknown name '{elt.id}'",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Each entry must resolve to a tensor visible in the enclosing function scope.",
                )
            if not isinstance(var.type, ir.TensorType):
                raise ParserTypeError(
                    f"{api}(dumps=[...]) entry '{elt.id}' is not a tensor (got type {var.type})",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Only tensors can be selectively dumped.",
                )
            if id(var) in seen:
                raise ParserTypeError(
                    f"{api}(dumps=[...]) lists '{elt.id}' more than once",
                    span=self.services.span_tracker.get_span(elt),
                    hint="Each tensor may appear at most once in `dumps=`.",
                )
            seen.add(id(var))
            resolved.append(var)
        return resolved

    def parse_comm_domain_scope(self, stmt: ast.With, call: ast.Call) -> None:
        """Restore a printed scope, binding its slots after parsing the body."""
        span = self.services.span_tracker.get_span(stmt)
        if call.args or stmt.items[0].optional_vars is not None:
            raise ParserSyntaxError("comm_domain accepts keyword arguments and no 'as' target", span=span)
        keywords: dict[str, ast.expr] = {}
        for keyword in call.keywords:
            if keyword.arg not in {"name", "devices", "slots", "attrs"} or keyword.arg in keywords:
                raise ParserSyntaxError("Unknown or duplicate comm_domain keyword", span=span)
            keywords[keyword.arg] = keyword.value
        if not {"name", "devices", "slots"} <= keywords.keys():
            raise ParserSyntaxError("comm_domain requires name, devices, and slots", span=span)
        name_node, devices_node, slots_node = (keywords[key] for key in ("name", "devices", "slots"))
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            raise ParserSyntaxError("comm_domain name must be a string literal", span=span)
        if not isinstance(devices_node, ast.List) or any(
            not isinstance(device, ast.Constant) or type(device.value) is not int
            for device in devices_node.elts
        ):
            raise ParserSyntaxError("comm_domain devices must be a list of integer literals", span=span)
        if not isinstance(slots_node, ast.List):
            raise ParserSyntaxError(
                "comm_domain slots must be a list of window_buffer descriptors", span=span
            )
        for descriptor in slots_node.elts:
            self.services.type_resolver.register_window_buffer(descriptor)
        devices = [cast(int, cast(ast.Constant, device).value) for device in devices_node.elts]
        attrs: list[tuple[str, Any]] = []
        slots: list[ir.WindowBuffer] = []
        with self.services.builder.comm_domain_scope(devices, slots, name_node.value, span, attrs):
            with self.regions.scope_kind_context(ir.ScopeKind.CommDomain):
                self.services.scope_manager.enter_scope("scope")
                self.callbacks.body(stmt.body)
                if "attrs" in keywords:
                    attrs_node = keywords["attrs"]
                    if not isinstance(attrs_node, ast.Dict):
                        raise ParserSyntaxError("comm_domain attrs must be a dictionary literal", span=span)
                    for key, value in zip(attrs_node.keys, attrs_node.values):
                        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                            raise ParserSyntaxError(
                                "comm_domain attr names must be string literals", span=span
                            )
                        attrs.append(
                            (key.value, self.attrs.parse_attr_value("comm_domain", key.value, value))
                        )
                slots.extend(
                    self.services.type_resolver.resolve_window_buffer(node) for node in slots_node.elts
                )
                self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                self.services.scope_manager.exit_scope(leak_vars=True)
