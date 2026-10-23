# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared launch, TaskId producer and predicate/dependency semantics."""

import ast
import copy
from typing import Any, cast

from pypto.ir.printer import python_print
from pypto.language.dsl_api import RangeIterator as _DslRangeIterator
from pypto.pypto_core import DataType, ir

from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
)
from .parser_state import ParseCallbacks, ParserServices
from .region_context import RegionContext
from .syntax_helpers import is_dep_var_type, is_per_element_task_id_read

# The op a bare ``predicate=None`` lowers to. Resolved through the registry getter so a rename
# fails at import instead of silently dropping the hint from the error message below.
TASK_INVALID_OP = ir.get_op("system.task_invalid").name

# Mirrors the runtime's ``NUM_TASK_TIMING_SLOTS`` (runtime/src/common/platform/include/common/device_phase.h),
# re-spelled here since the parser validates ``timing_slot=`` before it ever reaches
# codegen's own mirror, ``kRuntimeNumTaskTimingSlots`` in orchestration_codegen.cpp.
NUM_TASK_TIMING_SLOTS = 16

# Integer dtypes accepted wherever the DSL requires an integer-typed scalar
# expression: an SPMD ``core_num`` (block count, via ``pl.spmd`` and
# ``pl.spmd_submit``) and the ``predicate=`` element indices.
INTEGER_DTYPES = frozenset(
    {
        DataType.INT4,
        DataType.INT8,
        DataType.INT16,
        DataType.INT32,
        DataType.INT64,
        DataType.UINT4,
        DataType.UINT8,
        DataType.UINT16,
        DataType.UINT32,
        DataType.UINT64,
        DataType.INDEX,
    }
)

# Comparison IR node kinds the runtime's single-comparison DispatchPredicate
# can express. Reuses the IR's own comparison nodes — the runtime PredicateOp
# mapping lives in orchestration codegen, not here.
PREDICATE_CMP_TYPES = (ir.Eq, ir.Ne, ir.Gt, ir.Lt, ir.Ge, ir.Le)


class DispatchSemantics:
    """Shared launch, TaskId producer and predicate/dependency semantics."""

    def __init__(
        self,
        *,
        callbacks: ParseCallbacks,
        regions: RegionContext,
        services: ParserServices,
    ) -> None:
        self.callbacks = callbacks
        self.regions = regions
        self.services = services
        # Maps a Var bound from a ``pl.submit(...)`` result tuple, by
        # ``unique_id``, to the producer TaskId Var bound by that same submit.
        # Lets ``_validate_predicate_deps`` enforce the "predicate operand's
        # producer must be one of deps=" contract at the parser boundary.
        self.submit_producer_tid: dict[int, tuple[ir.Var, int]] = {}

        # How many times each TaskId Var object has been bound by a submit.
        # Strict SSA reuses one ``ir.Var`` object across same-named rebindings
        # (``_, tid = ...`` twice yields the same object), so object identity
        # alone cannot tell *which* task a later ``deps=[tid]`` refers to.
        # Recording the generation at producer time lets the contract check
        # detect that the name was rebound in between, instead of matching on
        # identity and falsely certifying the submit as compliant.
        self.tid_binding_generation: dict[int, int] = {}

    def parse_and_validate_core_num(self, value_node: ast.AST, source: ast.AST, usage_hint: str) -> "ir.Expr":
        """Parse and validate an SPMD ``core_num`` expression.

        ``core_num`` must be an integer-typed IR expression; a compile-time
        constant must be strictly positive. Shared by ``pl.spmd`` (scope) and
        ``pl.spmd_submit`` (task launch) so both reject the same bad inputs.
        """
        # ast.AST covers any expression; parse_expression expects ast.expr. The
        # grammar for keyword values and positional args always gives ast.expr
        # here, but cast for mypy's sake.
        expr = self.callbacks.expression(cast("ast.expr", value_node))
        expr_type = expr.type
        is_integer = isinstance(expr_type, ir.ScalarType) and expr_type.dtype in INTEGER_DTYPES
        if not is_integer:
            raise ParserSyntaxError(
                f"core_num must be an integer expression, got {python_print(expr_type, format=False)}",
                span=self.services.span_tracker.get_span(source),
                hint=usage_hint,
            )
        if isinstance(expr, ir.ConstInt) and expr.value <= 0:
            raise ParserSyntaxError(
                f"core_num must be a positive integer, got {expr.value}",
                span=self.services.span_tracker.get_span(source),
                hint=usage_hint,
            )
        return expr

    def parse_spmd_submit_kwargs(
        self, method_name: str, keywords: list[ast.keyword], span: ir.Span
    ) -> tuple["ir.Expr", bool]:
        """Parse the SPMD launch-spec kwargs of ``pl.spmd_submit(...)``.

        ``core_num=N`` is required (keyword-only — the positional slots are the
        kernel's own arguments) and must be a positive integer expression.
        ``sync_start=True/False`` is optional and must be a boolean literal.
        Returns ``(core_num_expr, sync_start)``.
        """
        # Concrete, arg-syntax-free hint: this fires on core_num / sync_start
        # validation, so point at the keyword itself (core_num is required, a
        # positive int; sync_start is optional and defaults to False).
        usage_hint = (
            f"pl.spmd_submit(self.{method_name}, ...) requires a positive integer "
            "'core_num' keyword (e.g. core_num=8); 'sync_start' is optional (default False)."
        )
        core_num: ir.Expr | None = None
        sync_start = False
        for kw in keywords:
            if kw.arg == "core_num":
                core_num = self.parse_and_validate_core_num(kw.value, kw.value, usage_hint)
            elif kw.arg == "sync_start":
                if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                    raise ParserSyntaxError(
                        "sync_start must be a boolean literal (True/False)",
                        span=self.services.span_tracker.get_span(kw.value),
                        hint=usage_hint,
                    )
                sync_start = kw.value.value
            # 'deps' / 'attrs' / 'device' are validated and consumed elsewhere
            # in _parse_kernel_call; ignore them here.
        if core_num is None:
            raise ParserSyntaxError(
                f"pl.spmd_submit(self.{method_name}, ...) requires the core_num keyword argument",
                span=span,
                hint=usage_hint,
            )
        return core_num, sync_start

    def parse_spmd_bool_literal_kwarg(self, kw: ast.keyword, usage_hint: str) -> bool:
        """Validate and return a boolean-literal ``pl.spmd()`` kwarg value.

        Shared by ``sync_start=`` and ``allow_early_resolve=`` (both require a
        plain ``True`` / ``False`` literal so the parser can record the flag
        without evaluating an expression). The error message names ``kw.arg`` so
        each kwarg reports its own diagnostic.
        """
        if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
            raise ParserSyntaxError(
                f"{kw.arg} must be a boolean literal (True/False)",
                span=self.services.span_tracker.get_span(kw.value),
                hint=usage_hint,
            )
        return kw.value.value

    def reject_spmd_submit_only_kwargs_in_cluster(
        self,
        dep_vars: "list[ir.Var]",
        allow_early_resolve: bool,
        predicate: "ir.Expr | None",
        span: "ir.Span",
    ) -> None:
        """Reject Submit-only ``pl.spmd()`` kwargs on a ``pl.cluster()``-nested scope.

        Covers ``deps=[...]``, ``allow_early_resolve=True`` and
        ``predicate=(...)``: each is carried by a ``Submit`` field (``deps_`` /
        ``allow_early_resolve_`` / ``predicate_``) with no plain-``Call``
        equivalent. A cluster-nested Spmd scope is unwrapped into the Group
        function by ``OutlineClusterScopes`` (``UnwrapNestedSpmd``) and never
        lowers to a ``Submit``, so any of them would be silently dropped. Raise a
        clear parse-time error instead, mirroring the ``as tid`` cluster
        rejection in :meth:`ScopeParser.parse_spmd_scope_with_tid`. ``UnwrapNestedSpmd``
        re-asserts this for hand-built / deserialized IR.
        """
        if not self.regions.is_inside_scope(ir.ScopeKind.Cluster):
            return
        # Report the kwarg the user actually wrote, so the message names it.
        if dep_vars:
            kwarg, lost = "deps=[...]", "the dependency edges"
        elif allow_early_resolve:
            kwarg, lost = "allow_early_resolve=True", "the early-dispatch hint"
        elif predicate is not None:
            kwarg, lost = "predicate=(...)", "the dispatch predicate"
        else:
            return
        raise ParserSyntaxError(
            f"`pl.spmd(..., {kwarg})` cannot be nested inside `pl.cluster()` — "
            "a cluster-nested pl.spmd is unwrapped into the Group function and never produces a "
            f"Submit, so {lost} would be lost.",
            span=span,
            hint=f"Use a standalone `with pl.spmd(..., {kwarg}):` (implicit cluster) to keep it.",
        )

    def reject_submit_dispatch_metadata_in_cluster(
        self,
        *,
        user_dep_vars: list["ir.Var"],
        core_num: "ir.Expr | None",
        sync_start: bool,
        allow_early_resolve: bool,
        predicate: "ir.Expr | None",
        timing_slot: int | None,
        span: "ir.Span",
    ) -> None:
        """Reject per-task metadata on a Submit nested inside ``pl.cluster()``.

        A cluster body is outlined into one Group task. Calls or plain Submit
        nodes inside that body identify the Group's AIC/AIV callees, but they
        are not dispatched independently, so their task-level controls have no
        runtime carrier and would otherwise be silently ignored.
        """
        if not self.regions.is_inside_scope(ir.ScopeKind.Cluster):
            return

        metadata_fields: list[str] = []
        if user_dep_vars:
            metadata_fields.append("deps")
        if core_num is not None:
            metadata_fields.append("core_num")
        if sync_start:
            metadata_fields.append("sync_start")
        if allow_early_resolve:
            metadata_fields.append("allow_early_resolve")
        if predicate is not None:
            metadata_fields.append("predicate")
        if timing_slot is not None:
            metadata_fields.append("timing_slot")
        if not metadata_fields:
            return

        construct = "pl.spmd_submit" if core_num is not None else "pl.submit"
        raise ParserSyntaxError(
            f"`{construct}(...)` inside `pl.cluster()` cannot carry per-task dispatch metadata "
            f"({', '.join(metadata_fields)}) — the cluster is dispatched as one Group task, so metadata on "
            "an inner Submit would be ignored",
            span=span,
            hint="Move the dependency or launch control to the dispatch surrounding the cluster, "
            "or omit it from the inner Submit.",
        )

    def record_scope_producer(self, body: "list[ast.stmt]", tid_var: "ir.Var") -> None:
        """Record ``body``'s assigned Vars as produced by the scope's ``tid_var``.

        The scope-form counterpart of the tracking ``_parse_submit_tuple_lhs``
        does for ``out, tid = pl.submit(...)``. Without it a
        ``with pl.spmd(...) as tid:`` that writes a tensor leaves no producer
        record, so :meth:`DispatchSemantics.validate_predicate_deps` silently certifies a later
        ``predicate=`` over that tensor even when the scope is absent from
        ``deps=`` — the exact stale-read the contract exists to prevent, in the
        form most natural to write.

        Same best-effort scope as the call form: only names bound by a
        top-level assignment in the scope body are tracked (aliases, writes
        through ``pl.Out`` arguments, and rebinds via an intervening call are
        not). The generation counter mirrors the submit path so a ``tid`` name
        rebound by a later scope cannot match by Var identity alone.
        """
        names: list[str] = []

        def collect(target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    collect(elt)

        for node in body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    collect(target)
            elif isinstance(node, ast.AnnAssign):
                collect(node.target)
        if not names:
            return
        gen = self.tid_binding_generation.get(tid_var.unique_id, 0) + 1
        self.tid_binding_generation[tid_var.unique_id] = gen
        for name in names:
            produced = self.services.scope_manager.lookup_var(name)
            if isinstance(produced, ir.Var):
                self.submit_producer_tid[produced.unique_id] = (tid_var, gen)

    def is_python_resolvable_ast(
        self, node: ast.AST, extra_python_names: frozenset[str] = frozenset()
    ) -> bool:
        """True iff no ``ast.Name`` in ``node`` shadows a DSL-scope IR Var.

        Used by Form 2 (``deps=[arr[i] for i in ...]``) to decide whether the
        comprehension's iterable + filter clauses can be evaluated natively in
        Python. Returning True means it is *safe* to hand the AST to
        ``ExprEvaluator.eval_expr``; the evaluator will surface a ``NameError``
        (wrapped as ``ParserTypeError``) if any name still cannot be resolved
        at runtime, which is the right error path for typos / missing imports.

        Returning False (an IR Var is in scope) guarantees the eval would
        either crash on an IR-side type or produce IR objects we cannot use
        as a Python sequence — surface a clear "depends on IR variable" error
        before reaching eval.

        ``extra_python_names`` lets the caller temporarily inject names that
        are bound by the comprehension's own loop targets (and so are valid
        Python references during filter evaluation even though they are not
        in ``closure_vars`` yet).

        Note: ``ast.walk`` recurses through nested ``ast.ListComp`` /
        ``ast.GeneratorExp`` nodes too. Any inner generator's loop target
        is *not* added to ``extra_python_names`` automatically, so the
        predicate is conservative for nested comprehensions. This is fine
        in v1 — ``_unroll_deps_comprehension`` only accepts a single ``for``
        clause and so never produces nested generators inside the iterable
        / filter — but tighten the walk if multi-generator support lands.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if child.id in extra_python_names:
                    continue
                if self.services.scope_manager.lookup_var(child.id) is not None:
                    # An IR Var shadows the name — cannot evaluate at parse time.
                    return False
        return True

    @staticmethod
    def substitute_python_name_in_ast(node: ast.expr, name: str, value: Any) -> ast.expr:
        """Return a copy of ``node`` with every bare reference to ``name``
        replaced by an ``ast.Constant(value)``.

        Used by Form 2's body substitution: the comprehension's loop variable
        is bound to a per-iteration Python int, and the body AST is rewritten
        accordingly before being handed back to ``parse_expression``.
        """

        class _Substituter(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
                if node.id == name and isinstance(node.ctx, ast.Load):
                    return ast.copy_location(ast.Constant(value=value), node)
                return node

        copied = copy.deepcopy(node)
        return cast(ast.expr, _Substituter().visit(copied))

    def unroll_deps_comprehension(
        self, comp: ast.ListComp, method_name: str, span: ir.Span
    ) -> list[ast.expr]:
        """Form 2: unroll ``[<body> for x in <iter> if <filter>]`` to a flat
        list of ``ast.expr`` per kept element. The body is returned as AST
        (not parsed) so the caller can route it through Form-1 acceptance,
        substituting the loop variable as an ``ast.Constant`` first.

        Requires every name in ``<iter>`` and ``<filter>`` to be Python-only
        (closure / globals / the comprehension's own loop var). Multi-target
        ``for (i, j) in ...`` and multi-generator comprehensions are rejected
        at v1.
        """
        if len(comp.generators) != 1:
            raise ParserSyntaxError(
                f"'{method_name}' deps= comprehension supports a single `for` clause (v1)",
                span=span,
                hint="Inline the inner generator(s) or write a single flat `for ... in ...`.",
            )
        gen = comp.generators[0]
        if gen.is_async:
            raise ParserSyntaxError(
                f"'{method_name}' deps= comprehension cannot be `async for` (v1)",
                span=span,
            )
        if not isinstance(gen.target, ast.Name):
            raise ParserSyntaxError(
                f"'{method_name}' deps= comprehension target must be a single name (v1)",
                span=span,
                hint="Use `for i in ...`, not `for i, j in ...`.",
            )
        loop_name = gen.target.id

        if not self.is_python_resolvable_ast(gen.iter):
            raise ParserTypeError(
                f"'{method_name}' deps= comprehension iterable depends on an IR variable; "
                f"cannot unroll at parse time",
                span=span,
                hint="Use a Python `range(...)`, a tuple of const ints, or a module-level list. "
                "If you need a runtime-bounded loop, write Form 1 (`arr[i]`) inside a `pl.range` body.",
            )
        for if_ in gen.ifs:
            if not self.is_python_resolvable_ast(if_, extra_python_names=frozenset([loop_name])):
                raise ParserTypeError(
                    f"'{method_name}' deps= comprehension filter depends on an IR variable; "
                    f"cannot unroll at parse time",
                    span=span,
                    hint="Filter clauses may reference only Python-level constants "
                    "(closures, globals, the comprehension's own loop var).",
                )

        iterable = self.services.expr_evaluator.eval_expr(gen.iter)
        # Reject DSL loop builders (``pl.range`` / ``pl.parallel`` / ``pl.pipeline``)
        # — they signal "runtime IR loop," and unrolling them at parse time
        # would silently flatten what the user intended as a per-iteration
        # construct.
        if isinstance(iterable, _DslRangeIterator):
            raise ParserTypeError(
                f"'{method_name}' deps= comprehension iterable is a DSL loop "
                f"(`pl.range` / `pl.parallel` / `pl.pipeline`); cannot unroll at parse time",
                span=span,
                hint="Use a Python `range(...)` for a parse-time-unrolled list, or write "
                "Form 1 (`arr[i]`) inside a `pl.range` body for a runtime loop.",
            )
        try:
            iter_values = list(iterable)
        except TypeError as e:
            raise ParserTypeError(
                f"'{method_name}' deps= comprehension iterable did not produce a sequence: {e}",
                span=span,
                hint="The iterable must be a Python `range`, list, or tuple.",
            ) from e
        for v in iter_values:
            # ``bool`` is an ``int`` subclass — exclude it first so a stray
            # ``True``/``False`` does not silently index slot 0/1.
            if isinstance(v, bool) or not isinstance(v, int):
                raise ParserTypeError(
                    f"'{method_name}' deps= comprehension iterable yielded a non-integer "
                    f"value ({type(v).__name__}); only integer indices are supported",
                    span=span,
                )

        saved_closure = self.services.expr_evaluator.closure_vars
        result: list[ast.expr] = []
        try:
            for value in iter_values:
                self.services.expr_evaluator.closure_vars = {**saved_closure, loop_name: value}
                keep = all(bool(self.services.expr_evaluator.eval_expr(if_)) for if_ in gen.ifs)
                if not keep:
                    continue
                result.append(self.substitute_python_name_in_ast(comp.elt, loop_name, value))
        finally:
            self.services.expr_evaluator.closure_vars = saved_closure
        return result

    def synthesize_deps_array(self, entries: list[ir.Expr], span: ir.Span) -> ir.Var:
        """Emit ``_submit_deps_buf = pl.array.create(N, pl.TASK_ID)`` followed
        by N ``array.update_element`` rebinds, returning the final array Var.

        Triggered when any ``deps=`` entry needs Form 1 / Form 2 desugaring.
        The synthesized array is single-use (referenced only as the produced
        Call's ``manual_dep_edges`` source) and is locally scoped — never
        threaded through a loop iter_arg — so existing array-carry, SSA, and
        DCE machinery treat it as any other locally-bound ``pl.array.create``
        followed by writes.
        """
        n = len(entries)
        size_expr = ir.ConstInt(n, DataType.INDEX, span)
        create_call = ir.create_op_call("array.create", [size_expr], {"dtype": DataType.TASK_ID}, span)
        buf_var = self.services.builder.let("_submit_deps_buf", create_call, span=span)
        for i, entry in enumerate(entries):
            idx_expr = ir.ConstInt(i, DataType.INDEX, span)
            upd_call = ir.create_op_call("array.update_element", [buf_var, idx_expr, entry], {}, span)
            buf_var = self.services.builder.let("_submit_deps_buf", upd_call, span=span)
        return buf_var

    def parse_submit_deps_kwarg(  # noqa: PLR0912 — single-purpose validation flow; splitting hurts readability
        self, method_name: str, keywords: list[ast.keyword], span: ir.Span
    ) -> list[ir.Var]:
        """Extract the optional ``deps=[tid1, tid2]`` kwarg on a ``pl.submit(...)`` call.

        Accepted entry shapes:

        * ``Scalar[TASK_ID]`` Var — a single producer TaskId (from a prior
          ``_, tid = pl.submit(...)`` or a loop iter_arg carry).
        * ``Array[N, TASK_ID]`` Var — a per-slot TaskId array (whole-array
          fence; codegen expands one ``is_valid()``-guarded slot per element).
        * ``None`` literal — the "no producer yet" sentinel; dropped here.
        * ``arr[idx]`` where ``arr`` is an ``Array[_, TASK_ID]`` — **Form 1**.
          Lowered into a fresh ``Array[K, TASK_ID]`` populated by
          ``array.update_element`` so the existing whole-array dep codegen
          fires unchanged. ``idx`` is any int-typed IR expression.
        * ``[<entry> for <name> in <iter> if <pred>]`` where ``<iter>`` /
          ``<pred>`` are evaluable in pure Python at parse time — **Form 2**.
          Unrolled into N Form-1 entries.

        When at least one entry needs desugaring, the synthesized array Var
        replaces the entire dep list (single ``Array[K, TASK_ID]`` entry).
        When every entry is a bare TaskId / Array Var, the direct path is
        kept verbatim so existing codegen golden output is byte-identical.

        Returns an empty list when ``deps=`` is absent.
        """
        deps_kw = next((kw for kw in keywords if kw.arg == "deps"), None)
        if deps_kw is None:
            return []

        # Top-level acceptable shapes:
        #   * ast.List / ast.Tuple — element-wise iteration (mixing direct
        #     entries with Form 1 subscripts).
        #   * ast.ListComp — the entire dep list is a single comprehension
        #     (Form 2). The comprehension is unrolled into a flat list of
        #     ast.expr entries which are then processed like a list literal.
        if isinstance(deps_kw.value, ast.ListComp):
            ast_entries: list[ast.expr] = self.unroll_deps_comprehension(deps_kw.value, method_name, span)
        elif isinstance(deps_kw.value, (ast.List, ast.Tuple)):
            ast_entries = []
            for elt in deps_kw.value.elts:
                if isinstance(elt, ast.ListComp):
                    ast_entries.extend(self.unroll_deps_comprehension(elt, method_name, span))
                else:
                    ast_entries.append(elt)
        else:
            raise ParserTypeError(
                f"'{method_name}' deps= must be a list / tuple / comprehension of TaskId values",
                span=self.services.span_tracker.get_span(deps_kw.value),
                hint="Use deps=[tid] where tid is a TaskId from a prior "
                "`_, tid = pl.submit(...)`, a loop iter_arg, `arr[i]` on a TASK_ID array, "
                "or `None`.",
            )

        direct_entries: list[ir.Expr] = []
        needs_desugar = False
        saw_whole_array = False
        for elt in ast_entries:
            elt_span = self.services.span_tracker.get_span(elt)
            # A bare ``None`` entry is the "no producer yet" sentinel — it
            # contributes no edge (an invalid TaskId would be skipped by the
            # codegen ``is_valid()`` guard anyway), so drop it here.
            if isinstance(elt, ast.Constant) and elt.value is None:
                continue
            expr = self.callbacks.expression(elt)
            if isinstance(expr, ir.Var) and is_dep_var_type(expr.type):
                is_array_var = isinstance(expr.type, ir.ArrayType)
                # Mixing a whole-array Var with per-element / comprehension
                # entries cannot be lowered: the synthesizer would feed the
                # ArrayType Var into ``array.update_element``'s scalar value
                # slot, tripping a C++ type-deducer CHECK. Refuse the mixed
                # form rather than silently re-shaping the user's intent.
                if is_array_var and needs_desugar:
                    raise ParserTypeError(
                        f"'{method_name}' deps= cannot mix a whole TASK_ID array "
                        f"with per-element entries (got array entry "
                        f"'{ast.unparse(elt)}' after a per-element / comprehension entry)",
                        span=elt_span,
                        hint="Pass the array alone (`deps=[arr]`) for a whole-array "
                        "fence, or index it slot-by-slot (`deps=[arr[i], arr[j], ...]`).",
                    )
                if is_array_var:
                    saw_whole_array = True
                direct_entries.append(expr)
                continue
            if is_per_element_task_id_read(expr):
                if saw_whole_array:
                    raise ParserTypeError(
                        f"'{method_name}' deps= cannot mix a whole TASK_ID array "
                        f"with per-element entries (got per-element entry "
                        f"'{ast.unparse(elt)}' after a whole-array entry)",
                        span=elt_span,
                        hint="Pass the array alone (`deps=[arr]`) for a whole-array "
                        "fence, or index it slot-by-slot (`deps=[arr[i], arr[j], ...]`).",
                    )
                direct_entries.append(expr)
                needs_desugar = True
                continue
            raise ParserTypeError(
                f"'{method_name}' deps= entries must be a TaskId variable, "
                f"a TASK_ID array element (e.g. `arr[i]`), or None — "
                f"got '{ast.unparse(elt)}'",
                span=elt_span,
                hint="Bind a TaskId with `_, tid = pl.submit(self.producer, ...)`, "
                "read a slot with `arr[i]` from a `pl.array.create(N, pl.TASK_ID)`, "
                "or pass `None` for no producer.",
            )

        if not needs_desugar:
            # All-direct path — preserve existing codegen byte-for-byte.
            # Invariant: ``needs_desugar`` is only flipped on the Call branch
            # above, so every entry on this path is an ``ir.Var``. The assert
            # documents that invariant and surfaces a clear failure if any
            # future edit threads a non-Var into ``direct_entries`` without
            # setting the desugar gate.
            assert all(isinstance(e, ir.Var) for e in direct_entries)
            return cast(list[ir.Var], direct_entries)
        if not direct_entries:
            return []
        synth = self.synthesize_deps_array(direct_entries, span)
        return [synth]

    def parse_submit_dumps_kwarg(
        self, method_name: str, args: list[ir.Expr], keywords: list[ast.keyword], span: ir.Span
    ) -> list[ir.Var]:
        """Extract the optional ``dumps=[t1, t2]`` kwarg on a ``pl.submit(...)`` call.

        The submit-side selective-dump surface, symmetric with ``deps=``: each
        entry marks one tensor argument of this submit for selective tensor dump
        (simpler#844). Entries feed the same ``attrs['dump_vars']`` set as a
        ``pl.dump_tag`` declaration, tracked by Var identity through SSA /
        inline / codegen.

        Each entry must be a tensor-typed Var that is a positional argument of
        this submit (matched by identity). Returns an empty list when ``dumps=``
        is absent.
        """
        dumps_kw = next((kw for kw in keywords if kw.arg == "dumps"), None)
        if dumps_kw is None:
            return []
        if not isinstance(dumps_kw.value, (ast.List, ast.Tuple)):
            raise ParserTypeError(
                f"'{method_name}' dumps= must be a list / tuple of tensor arguments",
                span=self.services.span_tracker.get_span(dumps_kw.value),
                hint="Write dumps=[x, y] listing tensors passed to this submit.",
            )
        result: list[ir.Var] = []
        for elt in dumps_kw.value.elts:
            elt_span = self.services.span_tracker.get_span(elt)
            val = self.callbacks.expression(elt)
            if not isinstance(val, ir.Var) or not isinstance(val.type, ir.TensorType):
                raise ParserTypeError(
                    f"'{method_name}' dumps= entries must be tensor variables — got '{ast.unparse(elt)}'",
                    span=elt_span,
                    hint="List tensors passed to this submit, e.g. dumps=[x].",
                )
            if not any(val is a for a in args):
                raise ParserTypeError(
                    f"'{method_name}' dumps= entry '{ast.unparse(elt)}' is not an argument of this submit",
                    span=elt_span,
                    hint="dumps= may only name tensors passed positionally to the submitted kernel.",
                )
            if not any(val is e for e in result):  # dedup by identity
                result.append(val)
        return result

    def parse_submit_allow_early_resolve_kwarg(self, method_name: str, keywords: list[ast.keyword]) -> bool:
        """Extract the optional ``allow_early_resolve=True/False`` kwarg.

        Accepted on ``pl.submit(...)`` and ``pl.spmd_submit(...)``. Opts this
        task in as a speculative early-dispatch producer — the scheduler may
        pre-stage its consumers before it completes (simpler#1065). Must be a
        boolean literal; defaults to ``False`` when absent.
        """
        kw = next((k for k in keywords if k.arg == "allow_early_resolve"), None)
        if kw is None:
            return False
        if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
            raise ParserSyntaxError(
                f"'{method_name}' allow_early_resolve must be a boolean literal (True/False)",
                span=self.services.span_tracker.get_span(kw.value),
                hint="Write allow_early_resolve=True to opt this submit into early-dispatch.",
            )
        return kw.value.value

    def parse_submit_timing_slot_kwarg(self, method_name: str, keywords: list[ast.keyword]) -> int | None:
        """Extract an optional ``timing_slot=0..15`` task-timing tag.

        The runtime has sixteen fixed task-timing slots. A tagged task contributes
        its scheduler dispatch/finish window to the selected slot, which is
        returned as a device-domain trace span. The slot is a compile-time
        configuration value, so it must be an integer literal rather than an IR
        expression.
        """
        kw = next((k for k in keywords if k.arg == "timing_slot"), None)
        if kw is None:
            return None
        if (
            not isinstance(kw.value, ast.Constant)
            or not isinstance(kw.value.value, int)
            or isinstance(kw.value.value, bool)
        ):
            raise ParserSyntaxError(
                f"'{method_name}' timing_slot must be a non-negative integer literal in 0..15",
                span=self.services.span_tracker.get_span(kw.value),
                hint="Write timing_slot=0 to tag this task with task-timing slot 0.",
            )
        slot = kw.value.value
        if not 0 <= slot < NUM_TASK_TIMING_SLOTS:
            raise ParserSyntaxError(
                f"'{method_name}' timing_slot must be in 0..{NUM_TASK_TIMING_SLOTS - 1}, got {slot}",
                span=self.services.span_tracker.get_span(kw.value),
                hint="Choose one of the runtime's sixteen task-timing slots (0 through 15).",
            )
        return slot

    def record_submit_timing_slot_attr(
        self, method_name: str, keywords: list[ast.keyword], extra_attrs: dict[str, Any]
    ) -> int | None:
        """Parse ``timing_slot=`` and record its Call attr when present."""
        slot = self.parse_submit_timing_slot_kwarg(method_name, keywords)
        if slot is not None:
            extra_attrs["task_timing_slot"] = slot
        return slot

    def parse_submit_predicate_kwarg(self, method_name: str, keywords: list[ast.keyword]) -> ir.Expr | None:
        """Extract the optional ``predicate=(<tensor>[<indices>] <op> <int>)`` kwarg.

        Accepted on ``pl.submit(...)``, ``pl.spmd_submit(...)`` and the
        ``with pl.spmd(...)`` scope form (``method_name`` names the caller in
        error messages). Encodes a
        dispatch predicate the scheduler evaluates at the dispatch point; a
        false result retires the task inline without dispatching to a core,
        while still settling fanin/fanout. Returns the comparison
        :class:`ir.Expr` (e.g. ``Gt(Cast(tensor.read(rc, [0, 0])), 0)``) or
        ``None`` when absent.

        The kwarg is parsed as an **ordinary expression** — ``rc[0, 0]`` is the
        DSL's usual sugar for ``pl.read``, so it lowers to ``tensor.read`` and
        the comparison to a ``Gt``/``Lt``/... node. No bespoke syntax and no
        private representation: the predicate rides on ``Submit.predicate`` as
        plain IR, and orchestration codegen decomposes it into the runtime's
        ``operand OP target`` triple.

        Note the Expr is stored on the Submit, never in a statement position, so
        the ``tensor.read`` is never executed in orchestration — reading the
        value there is exactly the ``wait_for_tensor_ready`` stall the predicate
        exists to avoid.

        Only what the runtime can express is accepted: a single comparison
        between a tensor element and an integer literal. That shape is checked by
        :meth:`DispatchSemantics.validate_predicate_shape`; the operand's producer must also be in
        ``deps=`` (:meth:`DispatchSemantics.validate_predicate_deps`).
        """
        kw = next((k for k in keywords if k.arg == "predicate"), None)
        if kw is None:
            return None
        predicate = self.callbacks.expression(kw.value)
        self.validate_predicate_shape(method_name, predicate, kw.value)
        return predicate

    @staticmethod
    def predicate_read(predicate: ir.BinaryExpr) -> "ir.Call | None":
        """Return the ``tensor.read`` Call of a well-shaped predicate, else None.

        Accepts either operand order (``t[i] > 0`` / ``0 < t[i]``) and sees
        through the Cast the DSL inserts when the read's dtype differs from the
        compared literal's.
        """

        def strip_cast(e: ir.Expr) -> ir.Expr:
            while isinstance(e, ir.Cast):
                e = e.operand
            return e

        def is_read(e: ir.Expr) -> bool:
            return isinstance(e, ir.Call) and e.op.name == ir.get_op("tensor.read").name

        lhs, rhs = strip_cast(predicate.left), strip_cast(predicate.right)
        if isinstance(lhs, ir.Call) and is_read(lhs) and isinstance(rhs, ir.ConstInt):
            return lhs
        if isinstance(rhs, ir.Call) and is_read(rhs) and isinstance(lhs, ir.ConstInt):
            return rhs
        return None

    @staticmethod
    def read_indices(read: ir.Call) -> list[ir.Expr]:
        """Per-axis indices of a ``tensor.read``.

        The op takes ``(tensor, indices)`` where ``indices`` is a ``MakeTuple``
        for a multi-axis read and a bare expression for a single axis.
        """
        if len(read.args) < 2:
            return []
        idx = read.args[1]
        return list(idx.elements) if isinstance(idx, ir.MakeTuple) else [idx]

    @classmethod
    def predicate_operand_tensor(cls, predicate: ir.BinaryExpr) -> "ir.Expr | None":
        """Return the tensor a predicate reads, or None when not well-shaped."""
        read = cls.predicate_read(predicate)
        return read.args[0] if read is not None and read.args else None

    def validate_predicate_shape(self, method_name: str, predicate: ir.Expr, node: ast.expr) -> None:
        """Reject predicates the runtime cannot express.

        The runtime evaluates one comparison of one tensor element against one
        constant, so only ``tensor[indices] OP int-literal`` (in either operand
        order) is accepted. Chained comparisons, arithmetic on the element,
        boolean combination, and non-literal right-hand sides are all rejected
        here rather than surfacing as an opaque codegen failure.
        """
        hint = "Use predicate=(tensor[i0, ...] > 0) — one comparison against an integer literal."
        span = self.services.span_tracker.get_span(node)
        if not isinstance(predicate, PREDICATE_CMP_TYPES):
            got = type(predicate).__name__
            extra = (
                " (to dispatch unconditionally, omit predicate= entirely)"
                if isinstance(predicate, ir.Call) and predicate.op.name == TASK_INVALID_OP
                else ""
            )
            raise ParserSyntaxError(
                f"'{method_name}' predicate must be a single comparison (==, !=, >, <, >=, <=); "
                f"got {got}{extra}",
                span=span,
                hint=hint,
            )

        read = self.predicate_read(predicate)
        if read is None:
            raise ParserSyntaxError(
                f"'{method_name}' predicate must compare one tensor element (e.g. t[i]) against an "
                "integer literal; the runtime evaluates a single element-vs-constant comparison",
                span=span,
                hint=hint,
            )
        # tensor.read(tensor, indices): arg 0 is the tensor, arg 1 the index list.
        operand_type = read.args[0].type if read.args else None
        if not isinstance(operand_type, ir.TensorType):
            got = python_print(operand_type, format=False) if operand_type is not None else "nothing"
            raise ParserTypeError(
                f"predicate operand must be a tensor, got {got}",
                span=span,
                hint=hint,
            )
        # The runtime reads `elem_size` bytes at the operand address and
        # sign-extends to int64 before comparing. An unsigned operand whose top
        # bit is set therefore compares as negative and silently inverts the
        # dispatch decision (a UINT32 count of 3_000_000_000 reads as
        # -1_294_967_296, so `count[e] > 0` is false and the task is skipped with
        # no diagnostic). Sub-byte dtypes have no addressable single-element
        # read. Reject both here for a source-level message; orchestration
        # codegen re-checks as the backstop for IR built outside the DSL.
        operand_dtype = operand_type.dtype
        if not (operand_dtype.is_signed_int() and operand_dtype.get_bit() in (8, 16, 32, 64)):
            raise ParserTypeError(
                f"predicate operand must be a signed 8/16/32/64-bit integer tensor, got "
                f"{operand_dtype}. The runtime sign-extends the value it reads, so an unsigned "
                f"operand can compare as negative and silently invert the dispatch decision",
                span=span,
                hint=hint,
            )
        # Each index renders as a scalar C++ expression into the runtime predicate
        # index array, so it must be an integer scalar — not, e.g., a tensor.
        indices = self.read_indices(read)
        for idx_expr in indices:
            if not (isinstance(idx_expr.type, ir.ScalarType) and idx_expr.type.dtype in INTEGER_DTYPES):
                raise ParserTypeError(
                    "predicate indices must be integer scalars, got "
                    f"{python_print(idx_expr.type, format=False)}",
                    span=span,
                    hint=hint,
                )
            # The runtime index array is uint32_t, so a negative constant wraps
            # to a huge value and yields an out-of-bounds GM address.
            if isinstance(idx_expr, ir.ConstInt) and idx_expr.value < 0:
                raise ParserTypeError(
                    f"predicate index must be non-negative, got {idx_expr.value}; the runtime stores "
                    f"indices as uint32_t, so a negative value wraps to an out-of-bounds address",
                    span=span,
                    hint=hint,
                )
        # The indices must locate exactly one element, so their count has to
        # match the operand tensor's rank.
        rank = len(operand_type.shape)
        if len(indices) != rank:
            raise ParserTypeError(
                f"predicate needs {rank} index(es) to locate a single element of a rank-{rank} tensor, "
                f"got {len(indices)}",
                span=span,
                hint=hint,
            )

    def validate_predicate_deps(
        self,
        method_name: str,
        predicate: ir.Expr,
        dep_vars: list[ir.Var],
        span: ir.Span,
    ) -> None:
        """Enforce that a ``predicate=`` operand's producer is one of ``deps=``.

        The scheduler evaluates the predicate at the dispatch point. If the task
        that writes the operand tensor is not a dependency, the predicate may be
        evaluated before that producer has completed and the dispatch decision
        is made from stale data. The contract is therefore that the operand's
        producer appears in ``deps=``.

        Only violations that are statically provable here are reported: the
        check applies when the operand is a Var bound from a prior
        ``pl.submit(...)`` result. An operand from any other source (a function
        parameter, say) has no tracked producer, and an ``Array[N, TASK_ID]``
        dep entry does not name its producers individually — both are skipped
        rather than risk rejecting a correct program.
        """
        if not isinstance(predicate, PREDICATE_CMP_TYPES):
            return
        operand = self.predicate_operand_tensor(predicate)
        if not isinstance(operand, ir.Var):
            return
        recorded = self.submit_producer_tid.get(operand.unique_id)
        if recorded is None:
            return
        producer_tid, producer_gen = recorded
        if any(isinstance(d.type, ir.ArrayType) for d in dep_vars):
            return
        # Identity match is necessary but not sufficient: if the TaskId name was
        # rebound by a later submit, the same Var object now denotes a different
        # task, so the dep does not actually order this submit after the
        # producer.
        current_gen = self.tid_binding_generation.get(producer_tid.unique_id, producer_gen)
        if any(d is producer_tid for d in dep_vars) and current_gen == producer_gen:
            return
        tid_name = producer_tid.name_hint
        hint = (
            f"Add the producer to the dependency list: deps=[{tid_name}]. Without it the "
            "scheduler may evaluate the predicate before the producing task has written the "
            "tensor."
        )
        raise ParserSyntaxError(
            f"'{method_name}' predicate reads '{operand.name_hint}', which is produced by the task "
            f"bound to '{tid_name}', but '{tid_name}' is not in deps=",
            span=span,
            hint=hint,
        )

    def parse_dispatch_device_kwarg(
        self,
        keywords: list[ast.keyword],
    ) -> ir.Expr | None:
        """Extract the optional ``device=`` kwarg on a ``self.<orch>(...)`` call.

        ``device=`` selects the physical device for an orchestrator dispatch
        and is legal only when the callee has :class:`ir.Role.Orchestrator`
        (i.e. either chip-level ``FunctionType.Orchestration`` or a function
        declared with ``level=..., role=Role.Orchestrator``; both forms set
        the same ``role`` field on the IR Function). Callee-role validation
        happens in :meth:`ASTParser.parse_call` via the ``allowed_kwargs`` filter
        before this method runs, so by the time we reach this point the
        kwarg is known to be legal. The parser is intentionally permissive
        about the value form: any IR expression works. Downstream passes
        (the comm-collection pass, simplify) inspect the resolved expression
        and decide:

        * ``ConstInt`` (literal or after simplify-time constant folding) →
          fixed-device dispatch.
        * IR Var that is a ``for``-loop induction variable with a known
          range → device subset / kAll.
        * Anything else → rejected by the comm-collection pass, with the
          full def-use chain available for a good error message.

        Returns:
            The parsed IR expression, or ``None`` when ``device=`` is absent.
        """
        device_kw = next((kw for kw in keywords if kw.arg == "device"), None)
        if device_kw is None:
            return None
        return self.callbacks.expression(device_kw.value)
