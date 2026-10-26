# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Kernel, external and inline calls, submit bindings and return-type deduction."""

import ast
from typing import TYPE_CHECKING, Any

from pypto.language import _dump
from pypto.pypto_core import DataType, ir

from .attr_parser import AttrParser
from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
)
from .dispatch_semantics import DispatchSemantics
from .parser_state import FunctionContext, InlineState, ParseCallbacks, ParserServices, ProgramBindings
from .syntax_helpers import is_pl_call
from .type_resolver import (
    _is_dump_call,  # noqa: PLC2701
)
from .type_semantics import (
    InlineReturnProvenanceCollector,
    derive_return_types_from_body,
    inline_argument_return_type,
    inline_immutable_aliases,
)

if TYPE_CHECKING:
    from .decorator import InlineFunction


class CallParser:
    """Kernel, external and inline calls, submit bindings and return-type deduction."""

    def __init__(
        self,
        *,
        attrs: AttrParser,
        bindings: ProgramBindings,
        callbacks: ParseCallbacks,
        dispatch: DispatchSemantics,
        function: FunctionContext,
        inline: InlineState,
        services: ParserServices,
    ) -> None:
        self.attrs = attrs
        self.bindings = bindings
        self.callbacks = callbacks
        self.dispatch = dispatch
        self.function = function
        self.inline = inline
        self.services = services
        # Cache of return types derived from a callee body (for annotation-less
        # callees), keyed by the callee Function. Avoids re-walking the body on
        # every call site within one function's parse. Keying by the Function
        # object (not id()) keeps a strong reference, so there is no id-reuse
        # hazard from a collected-and-reallocated Function.
        self.derived_return_types_cache: dict[ir.Function, list[ir.Type]] = {}
        self.inline_return_provenance_cache: dict[
            ir.Function, tuple[tuple[ir.Expr, ...], frozenset[int], dict[int, ir.Expr]]
        ] = {}

    def parse_submit_assignment(self, target: ast.Tuple, call: ast.Call, *, is_spmd: bool = False) -> None:
        """Parse ``out, tid = pl.submit(self.kernel, *args, deps=[...])`` and the
        ``pl.spmd_submit(self.kernel, *args, core_num=N, sync_start=...)`` SPMD
        variant (``is_spmd=True``).

        ``pl.submit`` / ``pl.spmd_submit`` are manual_scope constructs: they
        submit a kernel call and bind both the kernel result(s) and the
        producer TaskId. The desugared IR is a single :class:`ir.Submit` whose
        return type is the flat ``TupleType([*<kernel results>, Scalar[TASK_ID]])``
        — elements ``0..N-1`` are the kernel's results (one element per output,
        matching the kernel's declared return arity) and element ``N`` is the
        producer ``Scalar[TASK_ID]``. ``pl.spmd_submit`` additionally records
        the SPMD launch spec (``core_num`` / ``sync_start``) on the Submit.
        """
        construct = "pl.spmd_submit" if is_spmd else "pl.submit"
        span = self.services.span_tracker.get_span(call)
        # ``pl.submit`` / ``pl.spmd_submit`` and ``deps=`` are orthogonal to
        # ``manual_scope``: the runtime's ``Arg::set_dependencies`` adds
        # explicit edges on top of the auto-tracked deps (final fanin = auto
        # union explicit), so both flavours of orchestrator scope (auto or
        # manual) accept ``deps=[...]``. In auto scope it is a precision tool —
        # auto handles most of the dep graph; explicit edges patch the cases
        # the runtime cannot infer (or infers too conservatively).
        if len(target.elts) != 2:
            raise ParserSyntaxError(
                f"{construct}(...) must be unpacked as exactly 2 targets "
                f"(result, task_id), got {len(target.elts)}",
                span=span,
                hint=f"Use `out, tid = {construct}(self.kernel, ...)`.",
            )
        out_target, tid_target = target.elts
        if not isinstance(tid_target, ast.Name):
            raise ParserSyntaxError(
                f"{construct}(...) task_id target must be a plain variable name",
                span=span,
                hint=f"Use `out, tid = {construct}(self.kernel, ...)`.",
            )
        # The result target is either a single Name (single-output kernel) or
        # a flat tuple of Names (multi-output kernel). Nested tuples are
        # rejected: each kernel output is a single value, so a nested target
        # would silently pass the arity check and then fail later with an
        # opaque tuple-index error.
        if isinstance(out_target, ast.Name):
            out_names: list[ast.expr] = [out_target]
        elif isinstance(out_target, ast.Tuple):
            out_names = list(out_target.elts)
        else:
            raise ParserSyntaxError(
                f"{construct}(...) result target must be a variable name or a tuple of names",
                span=span,
                hint=f"Use `out, tid = {construct}(...)` or `(a, b), tid = {construct}(...)`.",
            )
        for elt in out_names:
            if not isinstance(elt, ast.Name):
                raise ParserSyntaxError(
                    f"{construct}(...) result target must contain plain variable names only "
                    f"(no nested tuples), got '{ast.unparse(elt)}'",
                    span=span,
                    hint=f"Use `out, tid = {construct}(...)` or `(a, b), tid = {construct}(...)`.",
                )
        n_outs = len(out_names)
        if not call.args:
            raise ParserSyntaxError(
                f"{construct}(...) requires the kernel as its first argument",
                span=span,
                hint=f"Use `out, tid = {construct}(self.kernel, *kernel_args, deps=[...])`.",
            )
        method_attr = call.args[0]
        if not (
            isinstance(method_attr, ast.Attribute)
            and isinstance(method_attr.value, ast.Name)
            and method_attr.value.id == "self"
        ):
            raise ParserSyntaxError(
                f"{construct}(...) first argument must be a `self.<kernel>` method reference",
                span=span,
                hint="Pass the kernel itself, not a call: "
                f"`{construct}(self.kernel, x, y)` — not `{construct}(self.kernel(x, y))`.",
            )

        call_expr = self.parse_kernel_call(
            method_attr, call.args[1:], call.keywords, span, as_submit=True, as_spmd=is_spmd
        )

        # The submit Call returns a flat Tuple{*<kernel results>, TaskId};
        # validate the result-target arity against the kernel's return arity.
        ret_type = call_expr.type
        if isinstance(ret_type, ir.TupleType) and len(ret_type.types) != n_outs + 1:
            raise ParserTypeError(
                f"{construct}(...) unpacks {n_outs} result value(s) but kernel "
                f"'{method_attr.attr}' returns {len(ret_type.types) - 1}",
                span=span,
                hint="Match the number of result targets to the kernel's return arity.",
            )

        # Bind the flat tuple: elements 0..N-1 -> kernel results, element N ->
        # the producer TaskId.
        submit_var = self.services.builder.let("_submit_tmp", call_expr, span=span)
        produced: list[ir.Var] = []
        for i, elt in enumerate(out_names):
            produced.extend(
                self.callbacks.bind_unpack_target(elt, ir.TupleGetItemExpr(submit_var, i, span), span)
            )
        task_id_expr = ir.TupleGetItemExpr(submit_var, n_outs, span)
        tid_var = self.callbacks.assign(tid_target.id, task_id_expr, span)
        self.services.scope_manager.define_var(tid_target.id, tid_var, span=span)
        # Record this submit's result Vars against its producer TaskId so a
        # later submit's predicate= operand can be checked against its deps=.
        gen = self.dispatch.tid_binding_generation.get(tid_var.unique_id, 0) + 1
        self.dispatch.tid_binding_generation[tid_var.unique_id] = gen
        for out_var in produced:
            self.dispatch.submit_producer_tid[out_var.unique_id] = (tid_var, gen)

    def build_submit_single_lhs_expr(self, call: ast.Call, *, is_spmd: bool = False) -> ir.Expr:
        """Build the ``ir.Submit`` expression for the single-LHS form
        ``result = pl.submit(self.kernel, ...)`` (or ``pl.spmd_submit`` when
        ``is_spmd=True``) without binding it.

        Separated from :meth:`CallParser.parse_submit_single_lhs` so the annotated
        assignment path can feed the inferred Submit type through the
        standard annotation-consistency / ``override_type`` validation flow
        that all other RHS expressions go through.
        """
        construct = "pl.spmd_submit" if is_spmd else "pl.submit"
        span = self.services.span_tracker.get_span(call)
        if not call.args:
            raise ParserSyntaxError(
                f"{construct}(...) requires the kernel as its first argument",
                span=span,
                hint=f"Use `result = {construct}(self.kernel, *kernel_args, deps=[...])`.",
            )
        method_attr = call.args[0]
        if not (
            isinstance(method_attr, ast.Attribute)
            and isinstance(method_attr.value, ast.Name)
            and method_attr.value.id == "self"
        ):
            raise ParserSyntaxError(
                f"{construct}(...) first argument must be a `self.<kernel>` method reference",
                span=span,
                hint="Pass the kernel itself, not a call: "
                f"`{construct}(self.kernel, x, y)` — not `{construct}(self.kernel(x, y))`.",
            )
        return self.parse_kernel_call(
            method_attr, call.args[1:], call.keywords, span, as_submit=True, as_spmd=is_spmd
        )

    def parse_submit_single_lhs(self, target: ast.Name, call: ast.Call, *, is_spmd: bool = False) -> None:
        """Parse the bare single-LHS form ``result = pl.submit(self.kernel, ...)``
        (or ``pl.spmd_submit`` when ``is_spmd=True``).

        Used by :meth:`ASTParser.parse_assignment` (no annotation). The annotated form
        ``result: pl.Tuple[..., TASK_ID] = pl.submit(...)`` builds the Submit
        via :meth:`CallParser.build_submit_single_lhs_expr` and runs the regular
        annotation-consistency flow before binding.
        """
        span = self.services.span_tracker.get_span(call)
        call_expr = self.build_submit_single_lhs_expr(call, is_spmd=is_spmd)
        var = self.callbacks.assign(target.id, call_expr, span)
        self.services.scope_manager.define_var(target.id, var, span=span)

    @staticmethod
    def validate_kernel_call_kwargs(
        method_name: str,
        func_obj: ir.Function | None,
        keywords: list[ast.keyword],
        as_submit: bool,
        as_spmd: bool,
        span: ir.Span,
    ) -> None:
        """Reject unknown keyword arguments on a cross-function kernel call.

        ``attrs=`` surfaces call-site directions and is always allowed.
        ``deps=`` / ``dumps=`` / ``timing_slot=`` are accepted only on
        ``pl.submit(...)``;
        ``core_num=`` / ``sync_start=`` only on ``pl.spmd_submit(...)``;
        ``device=`` only when the callee is an Orchestrator. Raises
        ``ParserTypeError`` with a targeted hint for the common
        deps/dumps-on-a-plain-call mistakes.
        """
        allowed_kwargs = {"attrs"}
        if as_submit:
            allowed_kwargs.add("deps")
            allowed_kwargs.add("dumps")
            allowed_kwargs.add("allow_early_resolve")
            allowed_kwargs.add("predicate")
            allowed_kwargs.add("timing_slot")
        if as_spmd:
            allowed_kwargs.update({"core_num", "sync_start"})
        if func_obj is not None and func_obj.role == ir.Role.Orchestrator:
            allowed_kwargs.add("device")
        for kw in keywords:
            if kw.arg in allowed_kwargs:
                continue
            hint = f"Allowed keyword arguments: {sorted(allowed_kwargs)}"
            if kw.arg == "deps" and not as_submit:
                hint = (
                    "Plain self.kernel(...) is fire-and-forget. To attach "
                    "dependency edges, submit it: "
                    "`out, tid = pl.submit(self.kernel, ..., deps=[...])`."
                )
            elif kw.arg in ("core_num", "sync_start") and not as_spmd:
                hint = (
                    f"'{kw.arg}' is an SPMD launch parameter. Launch the kernel "
                    "across multiple blocks with "
                    "`out, tid = pl.spmd_submit(self.kernel, ..., core_num=N)`."
                )
            elif kw.arg == "dumps" and not as_submit:
                hint = (
                    "dumps= is only valid on pl.submit(...) / pl.at(...). For a "
                    "plain self.kernel(...) call, declare the dump target with a "
                    "`pl.dump_tag(x)` statement before the call instead."
                )
            raise ParserTypeError(
                f"Function '{method_name}' does not accept keyword argument '{kw.arg}'",
                span=span,
                hint=hint,
            )

    @staticmethod
    def call_args_for_return_deduction(
        func_obj: ir.Function,
        args: list[ir.Expr],
        *,
        as_submit: bool,
    ) -> tuple[list[ir.Var], list[ir.Expr]]:
        """Pair callee params with actual args for return-type substitution."""
        if not as_submit or len(args) == len(func_obj.params):
            return list(func_obj.params), args

        callee_params: list[ir.Var] = []
        paired_args: list[ir.Expr] = []
        arg_idx = 0
        for param_idx, (param, direction) in enumerate(zip(func_obj.params, func_obj.param_directions)):
            if direction in (ir.ParamDirection.Out, ir.ParamDirection.InOut):
                remaining_required = sum(
                    1
                    for d in func_obj.param_directions[param_idx + 1 :]
                    if d not in (ir.ParamDirection.Out, ir.ParamDirection.InOut)
                )
                if len(args) - arg_idx <= remaining_required:
                    continue
            callee_params.append(param)
            paired_args.append(args[arg_idx])
            arg_idx += 1
        return callee_params, paired_args

    def validate_kernel_call_arity(
        self,
        method_name: str,
        func_obj: Any,
        n_args: int,
        span: ir.Span,
        as_submit: bool,
    ) -> None:
        """Check a ``self.<method>(...)`` call site's positional-argument count.

        One place decides the call flavour; the resulting ``(lo, hi)`` window
        then covers all three checks — an exact-arity check is just ``lo == hi``.

        Args:
            method_name: Callee name, for the diagnostic.
            func_obj: The callee ``ir.Function``.
            n_args: Positional argument count at the call site.
            span: Source span of the call.
            as_submit: Whether the site is ``pl.submit`` / ``pl.spmd_submit``.

        Raises:
            ParserTypeError: The argument count is outside the callee's window.
        """
        n_params = len(func_obj.params)
        if as_submit and self.function.func_level == ir.Level.HOST:
            # A distributed HOST submit dispatches to L3, which allocates no
            # output tensors, so it cannot omit Out/InOut params the way an
            # Orchestration-level submit can.
            flavour, lo, hi = "host_submit", n_params, n_params
        elif as_submit:
            # For submit (pl.submit / pl.spmd_submit), Out- and InOut-directed
            # parameters are runtime-allocated outputs that MAY be omitted at
            # the call site. The lower bound is the count of non-Out/InOut
            # params; the upper bound is all params (when Out params are passed
            # explicitly).
            flavour = "submit"
            lo = sum(
                1
                for d in func_obj.param_directions
                if d not in (ir.ParamDirection.Out, ir.ParamDirection.InOut)
            )
            hi = n_params
        else:
            flavour, lo, hi = "call", n_params, n_params

        if lo <= n_args <= hi:
            return

        param_info = [f"{p.name_hint}: {d.name}" for p, d in zip(func_obj.params, func_obj.param_directions)]
        hint = f"Parameters: {param_info}"
        if flavour == "host_submit":
            message = (
                "Distributed HOST pl.submit(...) requires every callee argument, including "
                f"Out/InOut tensors; function '{method_name}' expects {hi} argument(s), got {n_args}"
            )
        elif flavour == "submit":
            message = f"Function '{method_name}' expects {lo}..{hi} argument(s), got {n_args}"
            hint += ". Out/InOut params may be omitted in submit calls."
        else:
            message = f"Function '{method_name}' expects {hi} argument(s), got {n_args}"
        raise ParserTypeError(message, span=span, hint=hint)

    def parse_kernel_call(
        self,
        method_attr: ast.Attribute,
        arg_nodes: list[ast.expr],
        keywords: list[ast.keyword],
        span: ir.Span,
        *,
        as_submit: bool,
        as_spmd: bool = False,
    ) -> ir.Expr:
        """Build the ``ir.Call`` for a cross-function ``self.<method>(...)`` invocation.

        Shared by three call surfaces:

        * the plain-call path in :meth:`ASTParser.parse_call` (``as_submit=False``) — a
          fire-and-forget kernel call; ``deps=`` is rejected and the call's
          return type is the callee's own.
        * the ``pl.submit(...)`` path in :meth:`CallParser.parse_submit_assignment`
          (``as_submit=True``) — the call captures a producer TaskId; ``deps=``
          is accepted and the return type is augmented to the flat
          ``TupleType([*<callee returns>, Scalar[TASK_ID]])``.
        * the ``pl.spmd_submit(...)`` path (``as_submit=True, as_spmd=True``) —
          as above, plus the ``core_num=`` / ``sync_start=`` SPMD launch-spec
          kwargs are accepted and recorded on the resulting ``ir.Submit``.

        Args:
            method_attr: The ``self.<method>`` attribute node.
            arg_nodes: Positional kernel argument AST nodes (no ``self.method``).
            keywords: Keyword AST nodes from the call site.
            span: Source span of the call.
            as_submit: Whether this originates from a ``pl.submit``/
                ``pl.spmd_submit`` call.
            as_spmd: Whether this originates from a ``pl.spmd_submit(...)`` call
                (accepts the ``core_num`` / ``sync_start`` launch-spec kwargs).
        """
        method_name = method_attr.attr
        if method_name not in self.bindings.global_vars:
            raise UndefinedVariableError(
                f"Function '{method_name}' not defined in program",
                span=span,
                hint=f"Available functions: {list(self.bindings.global_vars.keys())}",
            )
        gvar = self.bindings.global_vars[method_name]
        func_obj = self.bindings.gvar_to_func.get(gvar)

        # Reject unknown kwargs (``attrs`` always; ``deps`` / ``dumps`` only on
        # submit; ``core_num`` / ``sync_start`` only on spmd_submit; ``device``
        # only for an Orchestrator callee).
        self.validate_kernel_call_kwargs(method_name, func_obj, keywords, as_submit, as_spmd, span)

        # Validate argument count before parsing args to fail fast.
        if func_obj is not None:
            self.validate_kernel_call_arity(method_name, func_obj, len(arg_nodes), span, as_submit)

        arg_directions = self.extract_arg_directions_from_attrs(method_name, keywords, len(arg_nodes), span)
        if arg_directions is None:
            arg_directions = []
        # Manual deps live on Submit::deps_ only (ManualDepsOnSubmitOnly
        # invariant): there is no attrs-dict surface for them on any call kind.
        self.reject_manual_dep_edges_in_attrs(method_name, keywords, span)
        # Generic round-trip safety net: recover every attrs={...} key that has
        # no dedicated extractor (arg_direction_overrides, dummy_task, ...). The
        # printer's PrintAttrValue is the matching writer.
        extra_attrs = self.attrs.extract_generic_call_attrs(method_name, keywords, span)
        # Detect ``pl.no_dep(...)`` wrappers at call-arg positions and collect
        # their indices for the arg_direction_overrides attr.
        unwrapped_args, no_dep_indices = self.strip_call_arg_markers(arg_nodes)
        args = [self.callbacks.expression(arg) for arg in unwrapped_args]
        # ``pl.submit`` parses the optional ``deps=[tid, ...]`` (explicit-edge
        # attr) and ``dumps=[tensor, ...]`` (selective dump) kwargs. A plain
        # Call recovers its selective-dump targets from the machine-only
        # ``attrs={"dump_vars": [...]}`` round-trip dict (printed metadata), not
        # a user-facing kwarg.
        user_dep_vars: list[ir.Var] = []
        explicit_dump_vars: list[ir.Var] = []
        if as_submit:
            user_dep_vars = self.dispatch.parse_submit_deps_kwarg(method_name, keywords, span)
            explicit_dump_vars = self.dispatch.parse_submit_dumps_kwarg(method_name, args, keywords, span)
        else:
            explicit_dump_vars = self.extract_dump_vars_from_attrs(method_name, args, keywords, span)
        # Build the selective-dump set in arg order (stable round-trip):
        # forward-sticky ``pl.dump_tag`` matches, ``dumps=`` entries (submit),
        # and round-tripped ``attrs['dump_vars']`` entries (plain Call) — all by
        # Var identity. Stored as ``attrs['dump_vars']`` (VarPtr list) on the
        # Call/Submit, so the dump target is tracked by Var through SSA / inline
        # / codegen. There is no call-arg wrapper surface: ``dump_vars`` is an
        # IR-level attr, never spelled as a user kwarg at a plain
        # ``self.kernel(...)`` call site.
        dump_vars: list[ir.Var] = []
        for arg in args:
            tagged = any(arg is t for t in self.function.dump_tagged_vars)
            in_dumps = any(arg is d for d in explicit_dump_vars)
            if (tagged or in_dumps) and isinstance(arg, ir.Var):
                dump_vars.append(arg)
        # Orchestration dispatch ``device=`` kwarg: resolves to a ConstInt or
        # an enclosing-loop induction Var.
        device_expr = self.dispatch.parse_dispatch_device_kwarg(keywords)
        # ``pl.spmd_submit`` parses the SPMD launch spec (``core_num=`` /
        # ``sync_start=``) into a (positive int Expr, bool) pair recorded on
        # the resulting Submit.
        core_num_expr: ir.Expr | None = None
        sync_start = False
        if as_spmd:
            core_num_expr, sync_start = self.dispatch.parse_spmd_submit_kwargs(method_name, keywords, span)
        # ``allow_early_resolve=True`` opts this task in as a speculative
        # early-dispatch producer. Accepted on both pl.submit and pl.spmd_submit
        # (recorded on the Submit; no-op for a plain self.kernel(...) call).
        allow_early_resolve = False
        predicate: ir.Expr | None = None
        timing_slot: int | None = None
        if as_submit:
            allow_early_resolve = self.dispatch.parse_submit_allow_early_resolve_kwarg(method_name, keywords)
            predicate = self.dispatch.parse_submit_predicate_kwarg(method_name, keywords)
            timing_slot = self.dispatch.record_submit_timing_slot_attr(method_name, keywords, extra_attrs)
            if predicate is not None:
                self.dispatch.validate_predicate_deps(method_name, predicate, user_dep_vars, span)
            self.dispatch.reject_submit_dispatch_metadata_in_cluster(
                user_dep_vars=user_dep_vars,
                core_num=core_num_expr,
                sync_start=sync_start,
                allow_early_resolve=allow_early_resolve,
                predicate=predicate,
                timing_slot=timing_slot,
                span=span,
            )
        return_types = func_obj.return_types if func_obj else []
        # A callee that declares no ``-> `` annotation has empty ``return_types``
        # but may ``return <value>`` (e.g. an InCore kernel returning its
        # ``pl.Out`` tensor param). For a plain cross-function Call, derive the
        # effective result type from the callee body so the call recovers a
        # concrete type instead of ``UnknownType`` and the print -> parse
        # round-trip stays symmetric. Submit return augmentation (``as_submit``)
        # is intentionally left on the declared types only — its tuple arity is
        # governed by pl.submit conventions, not by the callee's implicit return.
        if func_obj is not None and not return_types and not as_submit:
            return_types = self.effective_return_types(func_obj)
        # Argument format checks also apply to void calls and submits whose
        # only result is a TaskId; retain the existing omitted-output mapping.
        if func_obj is not None:
            callee_params_for_return, args_for_return = self.call_args_for_return_deduction(
                func_obj,
                args,
                as_submit=as_submit,
            )
            # Void inline wrappers can still carry independent placeholders.
            # Defer their Tile contracts to expansion; format checks run in either mode.
            return_types = ir.deduce_call_return_type(
                callee_params_for_return,
                args_for_return,
                return_types,
                is_inline=bool(return_types) and func_obj.func_type == ir.FunctionType.Inline,
            )
            return_types = self.inline_argument_return_types(func_obj, args, return_types)
        return self.make_call_with_return_type(
            gvar,
            args,
            return_types,
            span,
            arg_directions=arg_directions,
            no_dep_indices=no_dep_indices,
            dump_vars=dump_vars,
            user_dep_vars=user_dep_vars,
            device_expr=device_expr,
            augment_task_id=as_submit,
            core_num=core_num_expr,
            sync_start=sync_start,
            allow_early_resolve=allow_early_resolve,
            predicate=predicate,
            extra_attrs=extra_attrs,
        )

    def effective_return_types(self, func: ir.Function | None) -> list[ir.Type]:
        """Effective callee return types for cross-function call type inference.

        Returns the callee's declared ``return_types`` when present. When the
        callee declared no ``-> `` annotation (empty ``return_types``) but its
        body does ``return <value>``, derive the result types from the first
        such return so a plain cross-function ``Call`` recovers a concrete value
        type instead of ``UnknownType`` — keeping the print -> parse round-trip
        symmetric. Derived results are cached per Function so repeated call sites
        within one body do not re-walk the callee.
        """
        if func is None:
            return []
        declared = list(func.return_types)
        if declared:
            return declared
        cached = self.derived_return_types_cache.get(func)
        if cached is None:
            cached = derive_return_types_from_body(func.body)
            self.derived_return_types_cache[func] = cached
        return list(cached)

    def inline_argument_return_types(
        self, func: ir.Function, args: list[ir.Expr], return_types: list[ir.Type]
    ) -> list[ir.Type]:
        """Keep direct Tile returns in caller scope after native contract checking.

        Different inline operands may specialize one generic symbol differently.
        An unresolved callee symbol cannot survive in a caller's temporary type,
        so recover proven parameter returns from the corresponding actual type.
        A single trailing return and an unmodified formal (possibly reached
        through immutable aliases) are required; computed values and competing
        return paths retain native inference.
        """
        if func.func_type != ir.FunctionType.Inline or len(args) != len(func.params):
            return return_types
        cached = self.inline_return_provenance_cache.get(func)
        if cached is None:
            collector = InlineReturnProvenanceCollector()
            collector.visit_stmt(func.body)
            trailing = func.body
            if isinstance(trailing, ir.SeqStmts) and trailing.stmts:
                trailing = trailing.stmts[-1]
            values = (
                tuple(trailing.value)
                if collector.return_count == 1 and isinstance(trailing, ir.ReturnStmt)
                else ()
            )
            aliases = inline_immutable_aliases(func, collector) if values else {}
            cached = (values, frozenset(collector.assigned_ids), aliases)
            self.inline_return_provenance_cache[func] = cached
        values, assigned_ids, aliases = cached
        if len(values) != len(return_types):
            return return_types
        argument_types = {p.unique_id: a.type for p, a in zip(func.params, args)}
        return [
            inline_argument_return_type(v, t, argument_types, assigned_ids, aliases) or t
            for v, t in zip(values, return_types)
        ]

    def make_call_with_return_type(  # noqa: PLR0913, PLR0912 — cohesive call-builder; bundling args/branches hurts clarity
        self,
        gvar: ir.GlobalVar,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
        arg_directions: list[ir.ArgDirection] | None = None,
        no_dep_indices: list[int] | None = None,
        dump_vars: list[ir.Var] | None = None,
        user_dep_vars: list[ir.Var] | None = None,
        device_expr: ir.Expr | None = None,
        augment_task_id: bool = False,
        core_num: ir.Expr | None = None,
        sync_start: bool = False,
        allow_early_resolve: bool = False,
        predicate: ir.Expr | None = None,
        extra_attrs: dict[str, Any] | None = None,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type, optional call-site directions
        and optional per-arg ``NoDep`` overrides.

        Args:
            gvar: GlobalVar identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
            arg_directions: Optional explicit per-argument :class:`ir.ArgDirection`
                vector. When non-empty its length must equal ``len(args)`` and the
                resulting :class:`ir.Call` is constructed via the attrs overload
                with ``attrs['arg_directions']`` populated. When ``None`` or empty
                the call is constructed without explicit directions (legacy form;
                ``DeriveCallDirections`` will fill them in later).
            no_dep_indices: Optional list of arg positions wrapped in ``pl.no_dep(...)``
                at the call site. Stored as ``attrs['arg_direction_overrides']`` so
                ``DeriveCallDirections`` can overwrite the auto-derived direction at
                each indicated slot to ``ArgDirection.NoDep``.
            dump_vars: Optional list of argument Vars marked for selective
                tensor dump (via ``pl.dump_tag`` / ``dumps=``). Stored as ``attrs['dump_vars']``
                (``vector<VarPtr>``); orchestration codegen marks each matching
                ``Arg`` slot via ``Arg::dump(...)``. Tracked by Var identity so it
                stays consistent with ``args_`` through SSA / inline / codegen.
            user_dep_vars: Optional list of TaskId Vars from a ``pl.submit(...)``
                ``deps=[tid1, tid2]`` kwarg. Each entry is a
                ``Scalar[TASK_ID]`` (from a prior ``_, tid = pl.submit(...)`` /
                a ``None`` sentinel / a loop iter_arg) or an
                ``Array[N, TASK_ID]`` (from ``pl.array.create(N, pl.TASK_ID)``).
                Stored in the typed ``Submit.deps`` field (only valid with
                ``augment_task_id=True`` — ManualDepsOnSubmitOnly invariant);
                codegen emits a ``set_dependencies(arr, count)`` call built
                from the entries (array entries expand to one slot each).
            device_expr: Optional :class:`ir.Expr` from the ``device=`` kwarg
                on an Orchestration dispatch call. Stored under
                ``attrs['device']`` so the comm-collection pass can recover
                per-dispatch device subsets without re-parsing the AST.
            augment_task_id: When True (``pl.submit(...)`` path), the call's
                return type is wrapped as the flat
                ``TupleType([*<callee returns>, Scalar[TASK_ID]])`` — elements
                ``0..N-1`` are the kernel results, element ``N`` is the
                producer TaskId.
            core_num: Optional SPMD block-count :class:`ir.Expr` from the
                ``pl.spmd_submit(..., core_num=N)`` path. Recorded on the
                resulting ``ir.Submit.core_num`` (only valid with
                ``augment_task_id=True``).
            sync_start: SPMD sync-start flag from ``pl.spmd_submit(...,
                sync_start=...)``. Recorded on ``ir.Submit.sync_start``.
            allow_early_resolve: Speculative early-dispatch opt-in from
                ``pl.submit(..., allow_early_resolve=True)`` /
                ``pl.spmd_submit(...)``. Recorded on
                ``ir.Submit.allow_early_resolve`` (only valid with
                ``augment_task_id=True``).
            extra_attrs: Generic round-trip attrs recovered from a printed
                ``attrs={...}`` dict that have no dedicated param above
                (e.g. ``arg_direction_overrides``, ``dummy_task``). Merged into
                the call's attrs without overwriting a dedicated value.
        """
        if not return_types:
            return_type: ir.Type = ir.UnknownType()
        elif len(return_types) == 1:
            return_type = return_types[0]
        else:
            return_type = ir.TupleType(return_types)

        attrs: dict[str, Any] | None = None
        # Build the attrs dict from the dedicated params. Order is no longer
        # load-bearing: structural_equal compares attrs as an order-insensitive
        # key->value map, so the canonical order here is for readability only.
        if dump_vars:
            attrs = {"dump_vars": list(dump_vars)}
        if arg_directions:
            attrs = attrs or {}
            attrs["arg_directions"] = list(arg_directions)
        if no_dep_indices:
            attrs = attrs or {}
            attrs["arg_direction_overrides"] = list(no_dep_indices)
        if device_expr is not None:
            attrs = attrs or {}
            attrs["device"] = device_expr
        # Generic round-trip attrs (e.g. arg_direction_overrides from a printed
        # attrs dict, dummy_task, future keys). setdefault so a value already
        # supplied by a dedicated param above wins — the two sources are
        # mutually exclusive in practice (printed IR carries no pl.no_dep
        # wrappers), this is purely defensive.
        if extra_attrs:
            attrs = attrs or {}
            for _k, _v in extra_attrs.items():
                attrs.setdefault(_k, _v)

        if augment_task_id:
            # pl.submit emits a first-class Submit node (not an augmented
            # Call) so passes / printers can dispatch on the kind directly
            # and the typed deps_ field replaces the prior attrs-encoded
            # manual_dep_edges. The return type is still the flat
            # Tuple{*<kernel results>, TaskId} so tuple projection of the
            # producer TaskId continues to work the same way as before.
            return_type = ir.TupleType([*return_types, ir.ScalarType(DataType.TASK_ID)])
            deps_list: list[ir.Expr] = list(user_dep_vars) if user_dep_vars else []
            # deps live only on Submit::deps_; attrs never carries
            # manual_dep_edges (ManualDepsOnSubmitOnly invariant).
            submit_attrs: dict[str, Any] | None = attrs if attrs else None
            # pl.spmd_submit carries an SPMD launch spec (core_num/sync_start) on
            # the Submit, and allow_early_resolve carries the early-dispatch
            # opt-in; either requires the full ctor form even when there are no
            # attrs. A plain pl.submit with no attrs / hints keeps the minimal
            # form so existing golden output is byte-identical.
            if (
                submit_attrs is not None
                or core_num is not None
                or allow_early_resolve
                or predicate is not None
            ):
                return ir.Submit(
                    gvar,
                    args,
                    deps_list,
                    {},
                    submit_attrs,
                    return_type,
                    span,
                    core_num=core_num,
                    sync_start=sync_start,
                    allow_early_resolve=allow_early_resolve,
                    predicate=predicate,
                )
            return ir.Submit(gvar, args, deps_list, return_type, span)

        if attrs is not None:
            return ir.Call(gvar, args, {}, attrs, return_type, span)

        if not return_types:
            return ir.Call(gvar, args, span)
        return ir.Call(gvar, args, return_type, span)

    @staticmethod
    def match_call_arg_marker(node: ast.expr, name: str) -> ast.expr | None:
        """Return the inner arg if *node* is a ``pl.<name>(arg)`` / ``<name>(arg)``
        single-positional-arg marker call, else ``None``.

        The match is intentionally tight — only the bare ``pl.<name>`` attribute
        access (or a bare ``<name>`` import) with exactly one positional arg and
        no keywords qualifies. ``obj.<name>(x)`` on a user-defined object is left
        in place so the parser surfaces a normal-call error instead of silently
        stripping the wrapper.
        """
        if (
            isinstance(node, ast.Call)
            and len(node.args) == 1
            and not node.keywords
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == name
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "pl"
                )
                or (isinstance(node.func, ast.Name) and node.func.id == name)
            )
        ):
            return node.args[0]
        return None

    @classmethod
    def strip_call_arg_markers(
        cls,
        arg_nodes: list[ast.expr],
    ) -> tuple[list[ast.expr], list[int]]:
        """Peel ``pl.no_dep(arg)`` wrappers from a kernel-call argument list.

        Returns the unwrapped arg ASTs (innermost expression parsed into IR)
        plus the indices of arguments wrapped by ``pl.no_dep(...)`` →
        ``ArgDirection.NoDep`` overrides applied later by
        ``DeriveCallDirections`` (stored as ``attrs['arg_direction_overrides']``).

        Selective tensor dump is *not* a call-arg wrapper: the only dump
        surfaces are the declarative ``pl.dump_tag(t)`` statement and the
        explicit ``dumps=[...]`` kwarg on ``pl.submit(...)`` / ``pl.at(...)``.
        A wrapper with multiple args or any keyword is not recognized and is
        left in place (the parser hits it later as a normal call and surfaces a
        clear error).
        """
        unwrapped: list[ast.expr] = []
        no_dep_indices: list[int] = []
        for i, raw in enumerate(arg_nodes):
            node = raw
            saw_no_dep = False
            while True:
                inner = cls.match_call_arg_marker(node, "no_dep")
                if inner is not None:
                    saw_no_dep = True
                    node = inner
                    continue
                break
            unwrapped.append(node)
            if saw_no_dep:
                no_dep_indices.append(i)
        return unwrapped, no_dep_indices

    @staticmethod
    def reject_keyword_args(
        func_name: str,
        call: ast.Call,
        span: ir.Span,
        allowed: set[str] | None = None,
    ) -> None:
        """Reject keyword arguments not in *allowed*.

        Args:
            func_name: Function name used in error messages.
            call: AST call node.
            span: Source span of the call.
            allowed: Optional set of keyword names that are permitted. When
                ``None``, all keyword arguments are rejected (the default).
        """
        allowed = allowed or set()
        for kw in call.keywords:
            if kw.arg not in allowed:
                hint = (
                    f"Allowed keyword arguments: {sorted(allowed)}"
                    if allowed
                    else "Pass all arguments positionally"
                )
                raise ParserTypeError(
                    f"Function '{func_name}' does not accept keyword argument '{kw.arg}'",
                    span=span,
                    hint=hint,
                )

    def extract_arg_directions_from_attrs(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        arg_count: int,
        span: ir.Span,
    ) -> list[ir.ArgDirection] | None:
        """Extract ``arg_directions`` from an ``attrs={...}`` kwarg."""
        from pypto.language.arg_direction import NAME_TO_DIRECTION  # noqa: PLC0415

        attrs_dict = self.attrs.get_call_attrs_dict(method_name, keywords, span)
        if attrs_dict is None:
            return None

        directions: list[ir.ArgDirection] | None = None
        for key_node, value_node in zip(attrs_dict.keys, attrs_dict.values):
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                continue
            if key_node.value != "arg_directions":
                continue
            if not isinstance(value_node, ast.List):
                raise ParserTypeError(
                    f"attrs['arg_directions'] on call to '{method_name}' must be a list literal",
                    span=self.services.span_tracker.get_span(value_node),
                )
            parsed: list[ir.ArgDirection] = []
            for elt in value_node.elts:
                if not (
                    isinstance(elt, ast.Attribute)
                    and isinstance(elt.value, ast.Attribute)
                    and elt.value.attr == "adir"
                    and isinstance(elt.value.value, ast.Name)
                    and elt.value.value.id == "pl"
                ):
                    raise ParserSyntaxError(
                        f"attrs['arg_directions'] on call to '{method_name}' must contain "
                        "'pl.adir.<name>' references only",
                        span=self.services.span_tracker.get_span(elt),
                    )
                if elt.attr not in NAME_TO_DIRECTION:
                    raise ParserSyntaxError(
                        f"Unknown call-site direction marker 'pl.adir.{elt.attr}' in call to '{method_name}'",
                        span=self.services.span_tracker.get_span(elt),
                        hint=("Valid markers: " + ", ".join(f"pl.adir.{n}" for n in NAME_TO_DIRECTION)),
                    )
                parsed.append(NAME_TO_DIRECTION[elt.attr])
            directions = parsed

        if directions is None:
            return []

        if directions and len(directions) != arg_count:
            raise ParserTypeError(
                f"attrs['arg_directions'] length ({len(directions)}) on call to "
                f"'{method_name}' must match positional arg count ({arg_count})",
                span=span,
            )
        return directions

    def reject_manual_dep_edges_in_attrs(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        span: ir.Span,
    ) -> None:
        """Reject ``manual_dep_edges`` inside an ``attrs={...}`` kwarg.

        Manual dependency edges live in the typed ``Submit::deps_`` field
        (ManualDepsOnSubmitOnly invariant); the printer never emits the key in
        an attrs dict, so seeing it in source is a user error — fail loudly
        instead of silently dropping the edges.
        """
        attrs_dict = self.attrs.get_call_attrs_dict(method_name, keywords, span)
        if attrs_dict is None:
            return
        for key_node in attrs_dict.keys:
            if isinstance(key_node, ast.Constant) and key_node.value == "manual_dep_edges":
                raise ParserTypeError(
                    f"attrs={{'manual_dep_edges': ...}} is not accepted on a call to '{method_name}'",
                    span=span,
                    hint=(
                        "Dependency edges belong on a submit: "
                        "`out, tid = pl.submit(self.kernel, ..., deps=[...])`."
                    ),
                )

    def extract_dump_vars_from_attrs(
        self,
        method_name: str,
        args: list[ir.Expr],
        keywords: list[ast.keyword],
        span: ir.Span,
    ) -> list[ir.Var]:
        """Extract selective-dump targets from an ``attrs={"dump_vars": [...]}`` dict.

        This is the machine-only round-trip surface for the Call's ``dump_vars``
        attr: a plain ``self.kernel(...)`` exposes no user-facing ``dumps=``
        kwarg (the dump targets are seeded by ``pl.dump_tag`` / scope ``dumps=``
        and live in IR only), so the printer surfaces them inside the same
        ``attrs={...}`` dict as ``arg_directions``. Each entry must be a bare
        tensor variable that is a positional argument of this call (matched by
        identity, same contract as the submit-side ``dumps=`` kwarg). Returns an
        empty list when the key is absent.
        """
        attrs_dict = self.attrs.get_call_attrs_dict(method_name, keywords, span)
        if attrs_dict is None:
            return []
        for key_node, value_node in zip(attrs_dict.keys, attrs_dict.values):
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                continue
            if key_node.value != "dump_vars":
                continue
            if not isinstance(value_node, (ast.List, ast.Tuple)):
                raise ParserTypeError(
                    f"attrs['dump_vars'] on call to '{method_name}' must be a list literal",
                    span=self.services.span_tracker.get_span(value_node),
                )
            result: list[ir.Var] = []
            seen: set[int] = set()
            for elt in value_node.elts:
                elt_span = self.services.span_tracker.get_span(elt)
                val = self.callbacks.expression(elt)
                if not isinstance(val, ir.Var) or not isinstance(val.type, ir.TensorType):
                    raise ParserTypeError(
                        f"attrs['dump_vars'] entries on call to '{method_name}' must be tensor "
                        f"variables — got '{ast.unparse(elt)}'",
                        span=elt_span,
                    )
                if not any(val is a for a in args):
                    raise ParserTypeError(
                        f"attrs['dump_vars'] entry '{ast.unparse(elt)}' is not an argument of "
                        f"call to '{method_name}'",
                        span=elt_span,
                    )
                if id(val) not in seen:  # dedup by identity, matching _parse_scope_dumps_kwarg
                    seen.add(id(val))
                    result.append(val)
            return result
        return []

    @staticmethod
    def validate_call_arg_count(func_name: str, func: ir.Function, got: int, span: ir.Span) -> None:
        """Validate that the number of call arguments matches the function's parameter count.

        Args:
            func_name: Name used at the call site (for error messages)
            func: The target ir.Function
            got: Number of arguments provided at the call site
            span: Source span of the call (for error location)
        """
        expected = len(func.params)
        if got != expected:
            param_info = [f"{p.name_hint}: {d.name}" for p, d in zip(func.params, func.param_directions)]
            raise ParserTypeError(
                f"Function '{func_name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Parameters: {param_info}",
            )

    def parse_external_function_call(
        self, _local_name: str, ext_func: ir.Function, call: ast.Call
    ) -> ir.Expr:
        """Parse a call to an externally-defined ir.Function.

        Args:
            _local_name: The name used in the caller's scope (may be aliased)
            ext_func: The external ir.Function object
            call: The AST Call node
        """
        func_name = ext_func.name
        span = self.services.span_tracker.get_span(call)

        # Validate no naming conflict with internal program functions
        if func_name in self.bindings.global_vars:
            raise ParserSyntaxError(
                f"External function '{func_name}' conflicts with program function '{func_name}'",
                span=span,
                hint="Rename either the external or program function to avoid the name conflict",
            )

        # Check for conflicting externals with same .name but different objects
        if (
            func_name in self.bindings.external_funcs
            and self.bindings.external_funcs[func_name] is not ext_func
        ):
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func_name}'",
                span=span,
                hint="External functions must have unique names; rename one of the functions",
            )

        # Track the external function
        self.bindings.external_funcs[func_name] = ext_func

        # Reject keyword args and validate argument count before parsing
        self.reject_keyword_args(func_name, call, span)
        self.validate_call_arg_count(func_name, ext_func, len(call.args), span)

        args = [self.callbacks.expression(arg) for arg in call.args]

        gvar = ir.GlobalVar(func_name)
        return_types = self.effective_return_types(ext_func)
        return_types = ir.deduce_call_return_type(
            list(ext_func.params),
            args,
            return_types,
            # Match method calls: void wrappers validate Tile contracts only
            # after expansion supplies their final arguments. Format checks remain active.
            is_inline=bool(return_types) and ext_func.func_type == ir.FunctionType.Inline,
        )
        return_types = self.inline_argument_return_types(ext_func, args, return_types)
        return self.make_call_with_return_type(gvar, args, return_types, span)

    @staticmethod
    def is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def parse_inline_call(self, _local_name: str, inline_func: "InlineFunction", call: ast.Call) -> ir.Expr:
        """Parse a call to an InlineFunction, expanding its body in-place.

        Args:
            _local_name: The name used in the caller's scope
            inline_func: The InlineFunction object
            call: The AST Call node
        """
        span = self.services.span_tracker.get_span(call)
        self.reject_keyword_args(inline_func.name, call, span)

        expected = len(inline_func.param_names)
        got = len(call.args)
        if got != expected:
            raise ParserTypeError(
                f"Inline function '{inline_func.name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Check the inline function's parameter list: {inline_func.param_names}",
            )

        # Parse call arguments in the caller's context before entering inline scope
        arg_exprs = [self.callbacks.expression(arg) for arg in call.args]

        self.services.scope_manager.enter_scope("inline")
        for param_name, arg_expr in zip(inline_func.param_names, arg_exprs):
            self.services.scope_manager.define_var(param_name, arg_expr, allow_redef=True)

        # Save parser state and switch to the inline function's context
        prev_inline_state = (self.inline.inline_mode, self.inline.inline_return_expr)
        self.inline.inline_mode = True
        self.inline.inline_return_expr = None

        prev_closure_vars = self.services.expr_evaluator.closure_vars
        self.services.expr_evaluator.closure_vars = {**inline_func.closure_vars, **prev_closure_vars}

        prev_span_state = (
            self.services.span_tracker.source_file,
            self.services.span_tracker.source_lines,
            self.services.span_tracker.line_offset,
            self.services.span_tracker.col_offset,
        )
        self.services.span_tracker.source_file = inline_func.source_file
        self.services.span_tracker.source_lines = inline_func.source_lines
        self.services.span_tracker.line_offset = inline_func.line_offset
        self.services.span_tracker.col_offset = inline_func.col_offset

        try:
            for i, stmt in enumerate(inline_func.func_def.body):
                if i == 0 and self.is_docstring(stmt):
                    continue
                self.callbacks.statement(stmt)
        finally:
            # Restore parser state
            (
                self.services.span_tracker.source_file,
                self.services.span_tracker.source_lines,
                self.services.span_tracker.line_offset,
                self.services.span_tracker.col_offset,
            ) = prev_span_state
            self.services.expr_evaluator.closure_vars = prev_closure_vars
            return_expr = self.inline.inline_return_expr
            self.inline.inline_mode, self.inline.inline_return_expr = prev_inline_state
            # Keep the existing statement-level inline semantics for local
            # definitions, but never expose formal parameter bindings. A
            # formal may share a name with an unrelated caller variable; if it
            # escaped, a later inline call could receive the first call's
            # actual argument instead of the caller variable it names.
            self.services.scope_manager.exit_scope(
                leak_vars=True,
                exclude_names=set(inline_func.param_names),
            )

        if return_expr is None:
            raise ParserTypeError(
                f"Inline function '{inline_func.name}' has no return value",
                span=span,
                hint="Inline functions used as expressions must return a value",
            )

        return return_expr

    def parse_recorded_submit(self, call: ast.Call) -> ir.Submit:
        """Restore a Submit result type independently of callee inference."""
        span = self.services.span_tracker.get_span(call)
        if len(call.args) != 2 or call.keywords:
            raise ParserSyntaxError("_dump.submit requires a submit expression and result type", span=span)
        inner = call.args[0]
        if not (is_pl_call(inner, "submit") or is_pl_call(inner, "spmd_submit")):
            raise ParserSyntaxError("_dump.submit requires pl.submit or pl.spmd_submit", span=span)
        submit = self.build_submit_single_lhs_expr(inner, is_spmd=is_pl_call(inner, "spmd_submit"))
        assert isinstance(submit, ir.Submit)
        result_type = self.services.type_resolver.resolve_type(call.args[1])
        if isinstance(result_type, list):
            result_type = ir.TupleType(result_type)
        return ir.Submit(
            submit.op,
            submit.args,
            submit.deps,
            submit.kwargs,
            submit.attrs,
            result_type,
            span,
            core_num=submit.core_num,
            sync_start=submit.sync_start,
            allow_early_resolve=submit.allow_early_resolve,
            predicate=submit.predicate,
        )

    def parse_recorded_call(self, call: ast.Call) -> ir.Call:
        """Restore a Call snapshot without repeating type inference."""
        span = self.services.span_tracker.get_span(call)
        if len(call.args) != 3 or not isinstance(call.args[1], ast.List):
            raise ParserSyntaxError(
                "_dump.call requires an operator, an argument list, and a result type", span=span
            )
        op_node, args_node, type_node = call.args
        if isinstance(op_node, ast.Constant) and isinstance(op_node.value, str):
            op = ir.get_op(op_node.value)
        elif (
            isinstance(op_node, ast.Attribute)
            and isinstance(op_node.value, ast.Name)
            and op_node.value.id == "self"
            and op_node.attr in self.bindings.global_vars
        ):
            op = self.bindings.global_vars[op_node.attr]
        else:
            raise ParserSyntaxError("_dump.call requires a registered op name or a program method", span=span)
        result_type = self.services.type_resolver.resolve_type(type_node)
        if isinstance(result_type, list):
            result_type = ir.TupleType(result_type)
        metadata: dict[str, dict[str, Any]] = {"kwargs": {}, "attrs": {}}
        seen: set[str] = set()
        for keyword in call.keywords:
            if keyword.arg not in metadata or keyword.arg in seen or not isinstance(keyword.value, ast.Dict):
                raise ParserSyntaxError("_dump.call accepts unique kwargs and attrs dictionaries", span=span)
            seen.add(keyword.arg)
            values = metadata[keyword.arg]
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(key.value, str) or key.value in values:
                    raise ParserSyntaxError(
                        "_dump.call metadata keys must be unique string literals", span=span
                    )
                values[key.value] = self.attrs.parse_attr_value("_dump.call", key.value, value)
        assert isinstance(args_node, ast.List)
        args = [self.callbacks.expression(arg) for arg in args_node.elts]
        return ir.Call(op, args, metadata["kwargs"], metadata["attrs"], result_type, span)

    def parse_dump_call(self, call: ast.Call) -> ir.Expr:
        """Read private spellings that preserve the recorded IR structure."""
        span = self.services.span_tracker.get_span(call)
        if _is_dump_call(call, "submit"):
            return self.parse_recorded_submit(call)
        if _is_dump_call(call, "call"):
            return self.parse_recorded_call(call)
        if _is_dump_call(call, "window_buffer") or _is_dump_call(call, "window_ref"):
            return self.services.type_resolver.resolve_window_buffer(call)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "logical_xor":
            if len(call.args) != 2 or call.keywords:
                raise ParserSyntaxError("logical_xor requires exactly two positional operands", span=span)
            left, right = [self.callbacks.expression(arg) for arg in call.args]
            if any(
                not isinstance(expr.type, ir.ScalarType) or expr.type.dtype != DataType.BOOL
                for expr in (left, right)
            ):
                raise ParserTypeError("logical_xor requires BOOL scalar operands", span=span)
            return ir.Xor(left, right, DataType.BOOL, span)
        if isinstance(call.func, ast.Attribute) and call.func.attr == "float64":
            if (
                len(call.args) != 1
                or call.keywords
                or not isinstance(call.args[0], ast.Constant)
                or not isinstance(call.args[0].value, str)
            ):
                raise ParserSyntaxError("float64 requires one hexadecimal string literal", span=span)
            try:
                value = _dump.float64(call.args[0].value)
            except ValueError as exc:
                raise ParserSyntaxError(str(exc), span=span) from exc
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        raise ParserSyntaxError(f"Unknown IR dump expression: {ast.unparse(call.func)}", span=span)
