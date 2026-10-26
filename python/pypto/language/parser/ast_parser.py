# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST entry point composing focused DSL parsers and shared parse services."""

import ast
from collections.abc import Sequence
from typing import Any, NoReturn, cast

from pypto._function_attrs import AUTO_SCOPE_ATTR, EXTERNAL_SOURCE_ATTR
from pypto.ir import IRBuilder
from pypto.ir import op as ir_op
from pypto.ir.printer import python_print
from pypto.ir.type import _merge_annotation_metadata  # noqa: PLC2701
from pypto.language.distributed import op as _dsl_pld
from pypto.language.typing.dynamic import DynVar, dynamic
from pypto.pypto_core import DataType, ir

from ._dsl_invoker import invoke_dsl
from .attr_parser import AttrParser
from .call_parser import CallParser
from .control_flow_parser import ControlFlowParser
from .diagnostics import (
    BUG_CLASS_EXCEPTIONS,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
    concise_error_message,
)
from .dispatch_semantics import DispatchSemantics
from .enum_utils import (
    SPLIT_MODE_MAP,
    extract_enum_value,
)
from .expr_evaluator import ExprEvaluator
from .op_parser import OpParser
from .parser_state import (
    ControlFlowState,
    FunctionContext,
    InlineState,
    ParseCallbacks,
    ParserServices,
    ProgramBindings,
)
from .region_context import RegionContext
from .scope_manager import ScopeManager
from .scope_parser import ScopeParser
from .source_comments import SourceCommentTracker
from .span_tracker import SpanTracker
from .subscript_parser import SubscriptParser
from .syntax_helpers import (
    is_cache_policy_marker,
    is_docstring_stmt,
    is_dsl_call,
    is_empty_body,
    is_pl_call,
    is_pld_call,
)
from .type_resolver import (
    TypeResolver,
    _is_dump_call,  # noqa: PLC2701
)
from .type_semantics import (
    ShapeDimAliasExpander,
    collect_type_symbol_ids,
    normalize_inferred_type_for_annotation,
    shape_exprs_match,
    tensor_types_differ_only_by_valid_shape,
    tile_types_differ_only_by_unset_memory_space,
    types_match,
)

# ``pl.func_attr({...})`` — the body-prologue directive carrying function-level
# attributes. Body position is what lets an attribute reference a parameter: a
# decorator is evaluated before the signature binds any name.
FUNC_ATTR_DIRECTIVE = "func_attr"

# The ``split`` attr stores an int but is spelled (and printed) as the
# ``pl.SplitMode.X`` enum, so it needs enum handling on both attr paths.
SPLIT_ATTR = "split"

# Sentinel: a parsed attr value that must not be stored at all. ``None`` cannot
# serve here — it is a legitimate attr value — so identity against this object
# is the signal.
OMIT_ATTR: Any = object()

# Function attrs the parser consumes BEFORE it walks the body, so a body-position
# declaration would arrive too late to have any effect. These keep their
# dedicated ``@pl.function(...)`` keyword and are rejected in ``pl.func_attr``:
#   - ``auto_scope``      gates implicit scope insertion during the body walk
#   - ``external_source`` selects the no-DSL-body path entirely (the body must
#                         be a bare ``...``, so there is nowhere to put a
#                         prologue in the first place)
DECORATOR_ONLY_FUNC_ATTRS: frozenset[str] = frozenset({AUTO_SCOPE_ATTR, EXTERNAL_SOURCE_ATTR})


class ASTParser:
    """Compose the DSL parsers while preserving one recursive parsing entry point."""

    def __init__(  # noqa: PLR0913
        self,
        source_file: str,
        source_lines: list[str],
        line_offset: int = 0,
        col_offset: int = 0,
        global_vars: dict[str, ir.GlobalVar] | None = None,
        gvar_to_func: dict[ir.GlobalVar, ir.Function] | None = None,
        strict_ssa: bool = False,
        closure_vars: dict[str, Any] | None = None,
        buffer_name_meta: dict[tuple[str, str], dict[str, Any]] | None = None,
        dyn_var_cache: dict[str, ir.Var] | None = None,
        pending_comments: dict[int, list[tuple[int, str]]] | None = None,
        alloc_window_buffer_names: set[str] | None = None,
    ):
        """Initialize AST parser.

        Args:
            source_file: Path to source file
            source_lines: Lines of source code (dedented for parsing)
            line_offset: Line number offset to add to AST line numbers (for dedented code)
            col_offset: Column offset to add to AST column numbers (for dedented code)
            global_vars: Optional map of function names to GlobalVars for cross-function calls
            gvar_to_func: Optional map of GlobalVars to parsed Functions for type inference
            strict_ssa: If True, enforce SSA (single assignment). If False (default), allow reassignment.
            closure_vars: Optional variables from the enclosing scope for dynamic shape resolution
            buffer_name_meta: Optional shared (func_name, buffer_name) → metadata registry for cross-function
                import_peer_buffer resolution. When multiple functions in a @pl.program share this
                dict, import_peer_buffer can resolve .base from a peer function's reserve_buffer.
            dyn_var_cache: Optional shared cache mapping dynamic var names to ir.Var objects.
                When multiple functions in a @pl.program share this dict, the same DynVar
                produces the same ir.Var across functions.
            pending_comments: Map from 1-based line number to ``#``-stripped comment lines
                (produced by :func:`extract_line_comments`). Drained in source order and
                attached as ``leading_comments`` metadata to the stmt that follows.
            alloc_window_buffer_names: Optional shared set of declared buffer names
                (``pld.tensor.alloc_window_buffer`` LHS). Shared across all functions in
                a ``@pl.program`` to enforce program-wide name uniqueness.
        """
        self.span_tracker = SpanTracker(source_file, source_lines, line_offset, col_offset)
        self.scope_manager = ScopeManager(strict_ssa=strict_ssa)
        self.expr_evaluator = ExprEvaluator(
            closure_vars=closure_vars or {},
            span_tracker=self.span_tracker,
        )
        self.type_resolver = TypeResolver(
            expr_evaluator=self.expr_evaluator,
            scope_lookup=self.scope_manager.lookup_var,
            span_tracker=self.span_tracker,
            dyn_var_cache=dyn_var_cache,
            parse_expression=self.parse_expression,
        )
        self.builder = IRBuilder()
        self.services = ParserServices(
            self.span_tracker, self.scope_manager, self.expr_evaluator, self.type_resolver, self.builder
        )
        self.callbacks = ParseCallbacks(
            self.parse_expression,
            self.parse_statement,
            self._parse_body_siblings,
            self._assign_or_let,
            self._bind_unpack_target,
        )
        self.comments = SourceCommentTracker(
            span_tracker=self.span_tracker, pending_comments=pending_comments
        )
        self.subscripts = SubscriptParser(callbacks=self.callbacks, services=self.services)
        self.attrs = AttrParser(callbacks=self.callbacks, services=self.services)
        self.function = FunctionContext()
        self.flow = ControlFlowState()
        self.inline = InlineState()
        self.bindings = ProgramBindings(
            global_vars=global_vars,
            gvar_to_func=gvar_to_func,
            buffer_name_meta=buffer_name_meta,
            alloc_window_buffer_names=alloc_window_buffer_names,
        )
        self.regions = RegionContext(flow=self.flow, services=self.services)
        self.dispatch = DispatchSemantics(
            callbacks=self.callbacks, regions=self.regions, services=self.services
        )
        self.scopes = ScopeParser(
            attrs=self.attrs,
            callbacks=self.callbacks,
            comments=self.comments,
            dispatch=self.dispatch,
            function=self.function,
            regions=self.regions,
            services=self.services,
        )
        self.calls = CallParser(
            attrs=self.attrs,
            bindings=self.bindings,
            callbacks=self.callbacks,
            dispatch=self.dispatch,
            function=self.function,
            inline=self.inline,
            services=self.services,
        )
        self.ops = OpParser(
            attrs=self.attrs,
            bindings=self.bindings,
            callbacks=self.callbacks,
            dispatch=self.dispatch,
            function=self.function,
            regions=self.regions,
            services=self.services,
        )
        self.control = ControlFlowParser(
            attrs=self.attrs,
            callbacks=self.callbacks,
            comments=self.comments,
            flow=self.flow,
            scopes=self.scopes,
            services=self.services,
        )
        # Preserve the decorator-facing references to the program symbol tables.
        self.global_vars = self.bindings.global_vars
        self.gvar_to_func = self.bindings.gvar_to_func
        self.external_funcs = self.bindings.external_funcs

    def _bind_dynvar_to_scalar_param(
        self,
        param_name: str,
        param_type: ir.Type,
        param_var: ir.Var,
        param_span: ir.Span,
        symbols_used_so_far: set[int],
    ) -> None:
        """Re-point a same-named ``pl.dynamic()`` symbol at this scalar parameter.

        A ``pl.Scalar[pl.INDEX]`` parameter that shadows a ``pl.dynamic()`` symbol
        of the same name IS that symbol. ``MaterializeValidShapeSymbols`` turns an
        unbindable valid_shape symbol into a parameter, and the printer keeps its
        ``pl.dynamic()`` declaration so the annotations naming it still resolve
        (Python evaluates annotations in the enclosing scope, before parameters
        exist). Without this, the annotation would read a second, unbound Var of
        the same name.

        Only ``INDEX`` scalars qualify: a ``DynVar`` is an INDEX-valued dimension,
        so rebinding one to, say, a ``pl.Scalar[pl.FP32]`` parameter would put an
        FP32 Var in a shape expression.

        Annotations resolve in declaration order, so the parameter has to precede
        every annotation that names it. When an earlier parameter's type already
        read the symbol, rebinding now would leave that earlier type pointing at a
        different Var — which later reads as an unbound symbol and materializes a
        second, redundant argument. Reject that ordering instead.

        Args:
            param_name: Parameter name as written in the signature
            param_type: Resolved parameter type
            param_var: The Var just created for the parameter
            param_span: Span of the parameter, for diagnostics
            symbols_used_so_far: ``id()`` of every Var already read by an earlier
                parameter's type annotation

        Raises:
            ParserTypeError: If an earlier parameter annotation already read the
                shadowed symbol.
        """
        declared = self.expr_evaluator.closure_vars.get(param_name)
        if not isinstance(declared, DynVar):
            return
        if not isinstance(param_type, ir.ScalarType) or param_type.dtype != DataType.INDEX:
            return
        # An alias can resolve through the shared cache without populating its
        # own _ir_var. Also retain the object's candidate for evaluated compound
        # annotations, which can have read it directly through DynVar.unwrap().
        previous = (self.type_resolver.get_dynamic_symbol(declared.name), declared._ir_var)
        if any(var is not None and id(var) in symbols_used_so_far for var in previous):
            raise ParserTypeError(
                f"Parameter '{param_name}' shadows the dynamic symbol '{declared.name}', "
                f"which an earlier parameter's type annotation already uses",
                span=param_span,
                hint=(
                    f"Declare '{param_name}: pl.Scalar[pl.INDEX]' before the parameter whose "
                    f"pl.TensorView(valid_shape=...) names it, so the annotation resolves to "
                    f"the parameter"
                ),
            )
        # This is a function-local shadow, not a mutation of the program's
        # DynVar. Sharing the replacement with another function would make its
        # tensor annotation refer to this callee's scalar parameter (#3137).
        local = dynamic(declared.name)
        local._ir_var = param_var
        local.expr = param_var
        self.expr_evaluator.closure_vars = {**self.expr_evaluator.closure_vars, param_name: local}
        # The closure uses the Python alias, but both dimension caches use the
        # symbolic name: D = dynamic("B") must shadow the cached B, not D.
        self.expr_evaluator.bind_local_dynamic_symbol(declared.name, param_var)
        self.type_resolver.bind_local_dynamic_symbol(declared.name, param_var)

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
        func_level: ir.Level | None = None,
        func_role: ir.Role | None = None,
        func_attrs: dict[str, Any] | None = None,
        inline_body: str | None = None,
        requires_runtime_binding: bool = False,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)
            func_level: Hierarchy level (default: None)
            func_role: Function role (default: None)
            func_attrs: Function-level attributes dict (default: None)
            inline_body: If provided, the function body is replaced by a single
                ``InlineStmt`` carrying this verbatim Python source instead of
                parsing the AST body as DSL.
            requires_runtime_binding: True for an abstract SubWorker (``...``
                body) whose implementation is supplied at runtime. The function
                carries an empty ``InlineStmt`` body and this flag set.

        Returns:
            IR Function object
        """
        func_name = func_def.name
        # Window definitions may follow a signature that references them. Their
        # allocation Vars are interned and adopted at the actual definition site.
        self.type_resolver.register_window_definitions(func_def)
        self.function.func_name = func_name
        self.function.func_level = func_level
        # auto_scope rides in func_attrs; absent ⇒ default True.
        self.function.func_auto_scope = bool((func_attrs or {}).get(AUTO_SCOPE_ATTR, True))
        self.function.func_type = func_type
        self.function.param_dim_symbols = set()
        self.function.shape_dim_aliases = {}
        func_span = self.span_tracker.get_span(func_def)

        # Enter function scope
        self.scope_manager.enter_scope("function")

        # Forward-sticky selective tensor dump (simpler#844): a
        # ``pl.dump_tag(t)`` statement (handled in ``_handle_dump_tag``) records
        # the bound Var; every subsequent kernel dispatch consuming that exact
        # Var gets it merged into the dispatch's ``dump_vars`` attr (Call or
        # Submit). Tracked by Var identity — the scope
        # manager returns a stable object per binding, so a later reassignment
        # of the same name yields a new Var that is not tagged. Reset per
        # function; populated only inside Orchestration / Inline bodies (other
        # scopes reject the marker in ``_handle_dump_tag``). Inlining needs no
        # special migration: ``dump_vars`` rides on the spliced Call nodes and
        # the mutator substitutes the callee Var for the caller's arg.
        self.function.dump_tagged_vars = []
        # Reset alongside it: the consumed-marker registry only has to outlive
        # the body parse of the function whose scopes seeded it.
        self.scopes.consumed_cache_policy_markers = {}

        # Begin building function
        with self.builder.function(
            func_name,
            func_span,
            type=func_type,
            level=func_level,
            role=func_role,
            attrs=func_attrs,
            requires_runtime_binding=requires_runtime_binding,
        ) as f:
            # Vars already read by an earlier parameter's type annotation, by id().
            # A scalar parameter shadowing one of these arrives too late to bind it.
            symbols_used_so_far: set[int] = set()

            # Parse parameters (skip 'self' if it's the first parameter without annotation)
            for arg in func_def.args.args:
                param_name = arg.arg

                # Skip 'self' parameter if it has no annotation (shouldn't happen if decorator stripped it)
                if param_name == "self" and arg.annotation is None:
                    continue

                if arg.annotation is None:
                    raise ParserTypeError(
                        f"Parameter '{param_name}' missing type annotation",
                        span=self.span_tracker.get_span(arg),
                        hint="Add a type annotation like: x: pl.Tensor[[64], pl.FP32]",
                    )

                param_type, param_direction = self.type_resolver.resolve_param_type(arg.annotation)
                param_span = self.span_tracker.get_span(arg)

                # Add parameter to function with direction
                param_var = f.param(param_name, param_type, param_span, direction=param_direction)

                self._bind_dynvar_to_scalar_param(
                    param_name, param_type, param_var, param_span, symbols_used_so_far
                )
                symbols_used_so_far.update(collect_type_symbol_ids(param_type))

                # A bare ``pl.dynamic()`` Var in a tensor param's shape names that
                # argument's runtime extent. Orchestration codegen defines exactly
                # these symbols from the task-arg descriptors, which is what makes
                # them usable as values — and what ``_fold_tensor_dim`` folds onto.
                if isinstance(param_type, ir.TensorType):
                    for extent in param_type.shape:
                        if isinstance(extent, ir.Var):
                            self.function.param_dim_symbols.add(extent.unique_id)

                # Register in scope
                self.scope_manager.define_var(param_name, param_var, allow_redef=True)

            # Parse return type
            if func_def.returns:
                return_type = self.type_resolver.resolve_type(func_def.returns)
                if isinstance(return_type, list):
                    # tuple[T1, T2, ...] -> multiple return types
                    for rt in return_type:
                        f.return_type(rt)
                else:
                    f.return_type(return_type)

            # ``pl.func_attr({...})`` prologue. Merged here — after the params
            # are bound above, which is what lets an attr reference one — and
            # stripped from the body so no later stage sees the directives.
            # Consumed before the body-shape dispatch below so that a
            # signature-only external kernel carrying other attrs still counts
            # as having an empty body.
            body_stmts = self._consume_func_attr_prologue(func_def.body, func_name)

            # Parse function body. HOST SubWorkers carry pure-Python source
            # via ``inline_body`` and are not parsed as DSL.
            external_source = (func_attrs or {}).get(EXTERNAL_SOURCE_ATTR)
            if external_source is not None:
                # Header-only external C++ kernel: no DSL body to parse. The
                # signature (params + directions + return types) is the contract
                # the orchestration submits against; the backend compiles the
                # referenced ``.cpp`` as the InCore kernel by func_id.
                if func_type not in (ir.FunctionType.AIC, ir.FunctionType.AIV):
                    raise ParserTypeError(
                        f"external_source is only valid on FunctionType.AIC or "
                        f"FunctionType.AIV, got {func_type!r} for function '{func_name}'",
                        span=func_span,
                        hint="Declare the external kernel as pl.FunctionType.AIC or pl.FunctionType.AIV.",
                    )
                if not is_empty_body(body_stmts):
                    raise ParserSyntaxError(
                        f"External kernel '{func_name}' must have an empty '...' body "
                        "(signature only) — its implementation is the C++ source "
                        "given by external_source.",
                        span=func_span,
                        hint="Replace the function body with a bare '...'.",
                    )
            elif inline_body is not None:
                self.builder.inline_stmt(inline_body, ir.InlineLanguage.Python, func_span)
            else:
                self._parse_body_siblings(body_stmts)
                self.comments.discard_tail_block_comments(func_def.body, upper_line=None)

        # Exit function scope
        self.scope_manager.exit_scope()

        return f.get_result()

    def _consume_func_attr_prologue(self, body: list[ast.stmt], func_name: str) -> list[ast.stmt]:
        """Merge leading ``pl.func_attr({...})`` directives, return the rest of the body.

        ``pl.func_attr`` is a parse-time directive, not a statement: the dict is
        merged into the function's attrs and no IR node is emitted. It must
        appear in the **prologue** — before every other statement — because an
        attribute describes the whole function and must not appear to start
        applying partway down a body. That restriction is also what bounds the
        referenceable names to the parameters, which are the only bindings in
        scope at this point.

        A leading docstring is not "another statement": it precedes the prologue
        in ordinary Python style and carries no semantics here. It is skipped and
        kept in the returned body, so comment rerouting still sees it.

        Args:
            body: The function's AST body statements
            func_name: Enclosing function name, for diagnostics

        Returns:
            The body statements after the prologue, to be parsed as DSL. Any
            leading docstring is retained at the front.
        """
        index = 0
        docstring: list[ast.stmt] = []
        if body and is_docstring_stmt(body[0]):
            docstring = [body[0]]
            index = 1

        while index < len(body) and is_pl_call(getattr(body[index], "value", None), FUNC_ATTR_DIRECTIVE):
            self._merge_func_attr_directive(cast(ast.Expr, body[index]), func_name)
            index += 1

        # Anything further down the body is a misplacement, not a second prologue.
        for stmt in body[index:]:
            if is_pl_call(getattr(stmt, "value", None), FUNC_ATTR_DIRECTIVE):
                raise ParserSyntaxError(
                    f"pl.{FUNC_ATTR_DIRECTIVE}() must appear before every other statement in '{func_name}'",
                    span=self.span_tracker.get_span(stmt),
                    hint=(
                        f"Move the pl.{FUNC_ATTR_DIRECTIVE}(...) call to the top of the function "
                        "body. A function attribute describes the whole function, so it is "
                        "declared in the prologue; only parameters are referenceable there."
                    ),
                )
        return docstring + body[index:]

    def _merge_func_attr_directive(self, stmt: ast.Expr, func_name: str) -> None:
        """Evaluate one ``pl.func_attr({...})`` call and merge it into the function."""
        call = cast(ast.Call, stmt.value)
        span = self.span_tracker.get_span(stmt)

        if len(call.args) != 1 or call.keywords:
            raise ParserSyntaxError(
                f"pl.{FUNC_ATTR_DIRECTIVE}() takes exactly one positional dict argument (no keywords)",
                span=span,
                hint=f'Use: pl.{FUNC_ATTR_DIRECTIVE}({{"split": pl.SplitMode.UP_DOWN}})',
            )
        if not isinstance(call.args[0], ast.Dict):
            raise ParserSyntaxError(
                f"pl.{FUNC_ATTR_DIRECTIVE}() argument must be a dict literal",
                span=self.span_tracker.get_span(call.args[0]),
                hint=f'Use: pl.{FUNC_ATTR_DIRECTIVE}({{"split": pl.SplitMode.UP_DOWN}})',
            )

        attrs: dict[str, Any] = {}
        for key_node, value_node in zip(call.args[0].keys, call.args[0].values):
            if key_node is None:
                raise ParserSyntaxError(
                    f"Unsupported `**` unpacking in pl.{FUNC_ATTR_DIRECTIVE}({{...}})",
                    span=span,
                    hint="Use only string literal keys in the attrs dict.",
                )
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise ParserSyntaxError(
                    f"pl.{FUNC_ATTR_DIRECTIVE}() key must be a string literal, got {ast.unparse(key_node)}",
                    span=self.span_tracker.get_span(key_node),
                    hint=f'Use string literal keys, e.g. pl.{FUNC_ATTR_DIRECTIVE}({{"split": ...}}).',
                )
            key = key_node.value
            if key in attrs:
                raise ParserSyntaxError(
                    f"Duplicate function attribute '{key}' in pl.{FUNC_ATTR_DIRECTIVE}({{...}})",
                    span=span,
                    hint="Each attribute may be declared only once.",
                )
            if key in DECORATOR_ONLY_FUNC_ATTRS:
                raise ParserSyntaxError(
                    f"'{key}' cannot be set with pl.{FUNC_ATTR_DIRECTIVE}()",
                    span=span,
                    hint=(
                        f"Pass it as the '{key}=' keyword of @pl.function(...) instead. The parser "
                        "reads it before the body is parsed, so a body-position declaration comes "
                        "too late to take effect."
                    ),
                )
            value = self._parse_func_attr_value(key, value_node)
            if value is OMIT_ATTR:
                continue
            attrs[key] = value

        # The builder rejects a key already present, which covers a duplicate
        # across two pl.func_attr calls as well as one between the body and a
        # decorator attrs= (those were merged in at begin_function).
        try:
            self.builder.add_function_attrs(attrs)
        except ValueError as exc:
            # The duplicate-key rejection is a C++ ``CHECK`` in IRBuilder::AddFunctionAttrs,
            # so strip FatalLogger's implementation-facing tail before it reaches the header.
            raise ParserSyntaxError(concise_error_message(exc), span=span, hint=None) from exc

    def _parse_func_attr_value(self, key: str, value_node: ast.expr) -> Any:
        """Reconstruct one ``pl.func_attr`` value from its Python AST node.

        Shares ``_parse_attr_value``'s syntax-inference contract with the
        printer, so a printed prologue reparses to the same attrs. The one rule
        worth stating outright: a **bare name is always a parameter reference**,
        never the value of a same-named Python variable in the enclosing scope.
        The decorator form evaluated its dict as Python and could capture such a
        value; body position resolves names against the signature instead.
        """
        if key == SPLIT_ATTR:
            # ``split`` is an enum whose printed spelling (pl.SplitMode.X) the
            # generic path would resolve to its int value, changing the stored
            # attr type. Mirrors the decorator's dedicated handling, including
            # dropping ``SplitMode.NONE``: storing the 0 would be a non-canonical
            # spelling of "no split" that the printer filters, so print -> parse
            # would lose the key and structural round-trip would fail.
            split_mode = extract_enum_value(value_node, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
            if split_mode == ir.SplitMode.NONE:
                return OMIT_ATTR
            return split_mode.value
        return self.attrs.parse_attr_value(f"pl.{FUNC_ATTR_DIRECTIVE}", key, value_node)

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Drains pending ``#`` comments on lines up to ``stmt.end_lineno`` and
        attaches them as leading comments on the emitted IR stmt. Bare-string
        expressions (docstrings) are not emitted as IR; their text is rerouted
        into ``_pending_comments`` so the next stmt picks them up as leading
        comments.

        Args:
            stmt: AST statement node
        """
        leading = self.comments.drain_pending_comments(stmt)

        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # Bare-string expression — treat as comment text on the next stmt.
            # Prepend any `#` comments already collected for this line so the
            # printed order matches source order.
            doc_lines = stmt.value.value.splitlines() or [""]
            self.comments.requeue_comments_after(stmt, [*leading, *doc_lines])
            return

        # Push leading comments onto the builder's pending stack BEFORE
        # dispatching. The first stmt the helper emits in the same context as
        # this push absorbs the queued comments through its ctor path. The
        # stack (rather than a single slot) lets nested parse_statement calls
        # (this compound stmt + its body stmts) each have their own pending
        # entry without clobbering the outer one. If the helper emits nothing
        # (e.g. pl.static_assert, pass), pop returns the entry untouched and
        # we re-queue it into _pending_comments so the next stmt picks it up.
        self.builder.push_pending_leading_comments(leading)

        if isinstance(stmt, ast.AnnAssign):
            self.parse_annotated_assignment(stmt)
        elif isinstance(stmt, ast.Assign):
            self.parse_assignment(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._reject_augmented_assignment(stmt)
        elif isinstance(stmt, ast.For):
            self.control.parse_for_loop(stmt)
        elif isinstance(stmt, ast.While):
            self.control.parse_while_loop(stmt)
        elif isinstance(stmt, ast.If):
            self.control.parse_if_statement(stmt)
        elif isinstance(stmt, ast.With):
            self.scopes.parse_with_statement(stmt)
        elif isinstance(stmt, ast.Return):
            self.parse_return(stmt)
        elif isinstance(stmt, ast.Expr):
            self.parse_evaluation_statement(stmt)
        elif isinstance(stmt, ast.Break):
            self.control.parse_break(stmt)
        elif isinstance(stmt, ast.Continue):
            self.control.parse_continue(stmt)
        elif isinstance(stmt, ast.Pass):
            pass  # Produces no IR; residue handling below re-queues the comments.
        else:
            raise UnsupportedFeatureError(
                f"Unsupported statement type: {type(stmt).__name__}",
                span=self.span_tracker.get_span(stmt),
                hint="Only assignments, for loops, while loops, if statements, "
                "with statements, returns, break, continue, and pass are supported in DSL functions",
            )

        # Pop the pending entry we pushed above. If the helper emitted a
        # matching stmt, the result is empty. Otherwise the unconsumed
        # comments get re-queued into _pending_comments so the next source
        # stmt picks them up.
        residue = self.builder.pop_pending_leading_comments()
        if residue:
            self.comments.requeue_comments_after(stmt, residue)

    def _reject_augmented_assignment(self, stmt: ast.AugAssign) -> NoReturn:
        target = ast.unparse(stmt.target)
        hint = f"Use an explicit assignment to {target}."
        if isinstance(stmt.op, ast.Add):
            hint += (
                f" For matrix accumulation, write {target} = pl.matmul_acc("
                f"{target}, lhs, rhs, init_cond=(k0 == 0)); init_cond selects "
                "overwrite on the first reduction step."
            )
        raise UnsupportedFeatureError(
            f"Augmented assignment is not supported in DSL functions: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint=hint,
        )

    def _parse_body_siblings(self, body: Sequence[ast.stmt]) -> None:
        """Parse a block's body stmts, applying stale-pending sweeps between siblings.

        The inter-sibling sweep catches tail-of-block comments that live in the
        gap between one sibling's body (which has already closed) and the next
        sibling. Skipped for the first sibling since there is no prior closed
        block — any pending comments at that point belong to the enclosing
        compound stmt's header or leading attachments.
        """
        for i, stmt in enumerate(body):
            if i > 0:
                self.comments.discard_stale_pending_before(stmt)
            self.parse_statement(stmt)

    def parse_annotated_assignment(self, stmt: ast.AnnAssign) -> None:  # noqa: PLR0912
        """Parse annotated assignment: var: type = value.

        Args:
            stmt: AnnAssign AST node
        """
        if not isinstance(stmt.target, ast.Name):
            raise ParserSyntaxError(
                "Only simple variable assignments supported",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use a simple variable name for assignment targets",
            )

        var_name = stmt.target.id
        span = self.span_tracker.get_span(stmt)

        # Check if this is a yield assignment: var: type = pl.yield_(...)
        if isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "yield_":
                # Handle yield assignment
                yield_exprs: list[ir.Expr] = []
                for arg in stmt.value.args:
                    expr = self.parse_expression(arg)
                    yield_exprs.append(expr)

                # Emit yield statement
                yield_span = self.span_tracker.get_span(stmt.value)
                self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))
                self.control.track_yield_var(var_name, yield_exprs)

                # Don't register in scope yet - will be done when if statement completes
                return

        # Parse value expression
        if stmt.value is None:
            raise UnsupportedFeatureError(
                "Yield assignment with no value is not supported",
                span=self.span_tracker.get_span(stmt),
                hint="Provide a value for the assignment",
            )
        is_ptr_type_annotation = (
            isinstance(stmt.annotation, ast.Attribute)
            and isinstance(stmt.annotation.value, ast.Name)
            and stmt.annotation.value.id == "pl"
            and stmt.annotation.attr in ("Ptr", "MemRefType")
        ) or (isinstance(stmt.annotation, ast.Name) and stmt.annotation.id in ("Ptr", "MemRefType"))

        if (
            is_ptr_type_annotation
            and isinstance(stmt.value, ast.Call)
            and self.ops.is_printed_alloc_call(stmt.value)
        ):
            value_expr = self.ops.parse_printed_alloc_call(stmt.value)
            # Adopt the Var a signature annotation already interned for this
            # name. A parameter's MemRef may name a base Ptr allocated here, and
            # the signature parses first; minting a second Var would leave the
            # parameter's MemRef pointing at a different allocation identity
            # than the alloc that defines it.
            ptr_var = self.type_resolver.interned_base_ptr(var_name) or ir.Var(var_name, ir.PtrType(), span)
            self.builder.emit(ir.AssignStmt(ptr_var, value_expr, span))
            self.scope_manager.define_var(var_name, ptr_var, span=span)
            return

        # Printer round-trip: ``buf: pl.Ptr = pld.[tensor.]alloc_window_buffer(N)``.
        # The printer adds a ``pl.Ptr`` annotation for clarity; route the
        # annotated form through the same dedicated alloc parser as the
        # unannotated form.
        if is_ptr_type_annotation and (
            is_pld_call(stmt.value, "alloc_window_buffer") or _is_dump_call(stmt.value, "alloc_window_buffer")
        ):
            self._parse_alloc_window_buffer_assignment(stmt.target, stmt.value)
            return

        # Printer round-trip: ``res: pl.Tuple[..., TASK_ID] = pl.submit(...)``.
        # The annotation is checked against the inferred Submit type below
        # via the standard ``override_type`` validation path; the single-LHS
        # Submit handler builds the Submit and binds it to ``var_name`` directly.
        if is_pl_call(stmt.value, "submit"):
            # AnnAssign targets can also be Attribute/Subscript at runtime; the
            # typeshed type is narrower than the grammar.
            if not isinstance(stmt.target, ast.Name):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ParserSyntaxError(
                    "Annotated assignment of pl.submit must target a simple variable name",
                    span=self.span_tracker.get_span(stmt.target),
                    hint="Use `result: pl.Tuple[..., TASK_ID] = pl.submit(self.kernel, ...)`.",
                )
            # Build the Submit Expr and fall through to the standard
            # annotation-consistency / override_type validation path below —
            # so ``res: pl.Tuple[..., TASK_ID] = pl.submit(...)`` actually
            # validates the annotation against the inferred Submit return
            # type the way every other annotated RHS does.
            value_expr = self.calls.build_submit_single_lhs_expr(stmt.value)
        elif is_pl_call(stmt.value, "spmd_submit"):
            if not isinstance(stmt.target, ast.Name):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ParserSyntaxError(
                    "Annotated assignment of pl.spmd_submit must target a simple variable name",
                    span=self.span_tracker.get_span(stmt.target),
                    hint="Use `result: pl.Tuple[..., TASK_ID] = pl.spmd_submit(self.kernel, core_num=N)`.",
                )
            value_expr = self.calls.build_submit_single_lhs_expr(stmt.value, is_spmd=True)
        else:
            value_expr = self.parse_expression(stmt.value)

        # Validate annotation value types and preserve omitted inferred metadata.
        override_type = None
        # Skip annotations the resolver can't handle:
        # - String forward refs (e.g. "SomeType")
        # - pl.UnknownType (emitted by printer for unrepresentable types)
        # - Singleton marker types the resolver has no entry for
        #   (pl.MemRefType / pl.Ptr / pld.WindowBufferType): no shape/dtype to
        #   validate; the Var's type is fully determined by the RHS-inferred
        #   type. Markers listed in `_MARKER_TYPE_GETTERS` (pl.AsyncEvent,
        #   pld.CommCtx, ...) resolve normally — for them the kind check is a
        #   no-op, so they need no entry here under either spelling.
        ann = stmt.annotation
        is_unresolvable = (isinstance(ann, ast.Constant) and isinstance(ann.value, str)) or (
            isinstance(ann, ast.Attribute)
            and isinstance(ann.value, ast.Name)
            and (
                (ann.value.id == "pl" and ann.attr in ("UnknownType", "MemRefType", "Ptr"))
                or (ann.value.id == "pld" and ann.attr == "WindowBufferType")
            )
        )
        if is_unresolvable:
            resolved = None
        else:
            resolved = self.type_resolver.resolve_type(ann)
        if resolved is not None and not isinstance(resolved, list):
            inferred_for_validation = normalize_inferred_type_for_annotation(resolved, value_expr)
            self.type_resolver.validate_annotation_consistency(
                resolved, inferred_for_validation, var_name, span
            )
            if isinstance(resolved, ir.TupleType):
                override_type = _merge_annotation_metadata(resolved, value_expr.type)
            elif isinstance(value_expr.type, ir.UnknownType):
                # Inferred type is unknown (e.g. tpop_from_aiv): use annotation as type
                override_type = resolved
            elif isinstance(resolved, ir.TileType) and isinstance(value_expr.type, ir.TileType):
                normalized_inferred = normalize_inferred_type_for_annotation(resolved, value_expr)
                assert isinstance(normalized_inferred, ir.TileType)
                # Merge annotation metadata with inferred type: annotation fields
                # take priority, but inferred fields fill gaps the annotation doesn't specify.
                # This handles memref, memory_space, and tile_view in a single path.
                ann_ms = resolved.memory_space
                ann_tv = resolved.tile_view
                inf_ms = normalized_inferred.memory_space
                # For the ND→2D flattening case (FlattenTileNdTo2D), the call infers
                # an ND TileType whose tile_view.valid_shape is also ND.  Merging that
                # into a 2D type would produce an inconsistent valid_shape.  Detect this
                # by comparing dimensionality: if normalized_inferred (which carries the
                # annotation's 2D shape for ND→2D) has fewer dims than value_expr.type,
                # we're in the ND→2D case and must NOT use the ND tile_view.
                # For the 2D→2D case (fresh compilation), the C++ inferred tile_view
                # (e.g. col_major for [N,1] Vec) must be preserved so downstream passes
                # can see the correct layout.
                if len(normalized_inferred.shape) != len(value_expr.type.shape):
                    # ND→2D: avoid carrying ND valid_shape into 2D type
                    inf_tv = normalized_inferred.tile_view
                else:
                    # 2D→2D: preserve the actual C++ inferred tile_view
                    inf_tv = value_expr.type.tile_view
                merged_ms = ann_ms if ann_ms is not None else inf_ms
                merged_tv = ann_tv if ann_tv is not None else inf_tv
                # Build the override whenever the annotation contributes anything
                # the raw inferred type does not already carry.  The shape test
                # compares against ``value_expr.type`` (the *raw* inferred type),
                # not ``normalized_inferred`` — the latter has already adopted the
                # annotation's shape for the ND->2D case, so comparing against it
                # always matches and the flattened shape would be silently dropped.
                # Before memory spaces became optional this was masked: ``merged_ms``
                # was never None, so the override was always built.
                if (
                    resolved.memref is not None
                    or merged_ms is not None
                    or merged_tv is not None
                    or not shape_exprs_match(resolved.shape, value_expr.type.shape)
                ):
                    override_type = ir.TileType(
                        resolved.shape, resolved.dtype, resolved.memref, merged_tv, merged_ms
                    )
            elif isinstance(resolved, ir.DistributedTensorType) and resolved.window_buffer is not None:
                override_type = resolved
            elif isinstance(resolved, ir.ShapedType) and resolved.memref is not None:
                override_type = resolved
            elif isinstance(resolved, ir.TensorType) and resolved.tensor_view is not None:
                # Annotation specifies tensor view (stride/layout); preserve it
                override_type = resolved
            elif type(resolved) is ir.TensorType and type(value_expr.type) is ir.TensorType:
                # Restore declarative shapes only when direct tensor.dim
                # temporaries prove them identical to the inferred shape.
                # Other annotations must not override computed extents.
                expander = ShapeDimAliasExpander(self.function.shape_dim_aliases)
                if all(
                    ir.structural_equal(ann_dim, expander.visit_expr(inf_dim))
                    for ann_dim, inf_dim in zip(resolved.shape, value_expr.type.shape)
                ):
                    override_type = ir.TensorType(
                        resolved.shape, resolved.dtype, value_expr.type.memref, value_expr.type.tensor_view
                    )
            elif (
                isinstance(resolved, ir.ScalarType)
                and isinstance(value_expr.type, ir.ScalarType)
                and value_expr.type.dtype == DataType.INDEX
            ):
                override_type = resolved
        if isinstance(override_type, (ir.ScalarType, ir.TupleType)):
            value_expr = self._convert_annotation_scalars(value_expr, override_type, var_name)
        # If annotation syntax determines the result type more precisely than the
        # raw call inference, rebuild the Call with that type so structural
        # equality sees the same IR after print→parse.
        if (
            override_type is not None
            and not _is_dump_call(stmt.value, "call")
            and isinstance(value_expr, ir.Call)
            and (
                (
                    isinstance(override_type, ir.TileType)
                    and isinstance(value_expr.type, ir.UnknownType)
                    and value_expr.op.name
                    in {
                        ir.get_op("tile.tpop_from_aiv").name,
                        ir.get_op("tile.tpop_from_aic").name,
                    }
                )
                or not types_match(value_expr.type, override_type)
            )
        ):
            # Carry attrs through the rebuild: a forward-referenced callee has no
            # known return type, so the annotation always overrides it and every
            # call attr (arg_directions, arg_direction_overrides, dump_vars, ...)
            # would otherwise be silently dropped.
            value_expr = ir.Call(
                value_expr.op,
                value_expr.args,
                value_expr.kwargs,
                value_expr.attrs,
                override_type,
                value_expr.span,
            )

        # Reuse existing Var on reassignment (override_type is intentionally
        # discarded — the Var's type was fixed at first definition; the SSA
        # pass will create properly typed versioned copies later).
        var = self._assign_or_let(var_name, value_expr, span, override_type)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)

        # Track buffer metadata for attribute access (e.g., pipe_buf.base)
        if isinstance(stmt.value, ast.Call):
            self.ops.track_buffer_meta(var_name, stmt.value)

    def _assign_or_let(
        self,
        var_name: str,
        value_expr: ir.Expr,
        span: ir.Span,
        override_type: ir.Type | None = None,
    ) -> ir.Var:
        """Assign to existing Var if possible, otherwise create a new let binding."""
        existing_var = self.scope_manager.lookup_var(var_name)

        # ``n = pl.tensor.dim(x, 0)`` folded to a dyn-dim symbol (see
        # ``_fold_tensor_dim``): bind the Python name straight to the symbol. A Let
        # would copy it, and every shape later built from ``n`` would then carry the
        # copy rather than the symbol — the aliasing the fold exists to remove.
        # Both guards test the Var the name is bound to, never a parallel set of
        # names: scopes are a stack, so a name-keyed alias set desyncs the moment a
        # non-leaking scope rebinds the name and exits.
        # An LHS annotation that simply restates the symbol's own type (``n:
        # pl.Scalar[pl.INDEX] = pl.tensor.dim(x, 0)`` — the form the printer emits)
        # must not defeat the alias; only an annotation asking for a *different*
        # type falls through to a Let of its own.
        if (
            ir.is_orchestration_like(self.function.func_type)
            and self.ops.is_param_dim_symbol(value_expr)
            and (existing_var is None or self.ops.is_param_dim_symbol(existing_var))
            and (override_type is None or types_match(override_type, value_expr.type))
        ):
            return value_expr

        # The name is bound to a symbol, which is immutable — it names an argument's
        # extent. Rebind the name to a fresh let rather than assigning *through* the
        # alias into the symbol itself.
        if self.ops.is_param_dim_symbol(existing_var):
            existing_var = None

        if existing_var is not None and type(existing_var) is ir.Var and not self.scope_manager.strict_ssa:
            # Reject changes to the storage contract (#642). Pre-SSA tensor
            # validity may refine per definition; SSA keeps each RHS's view.
            value_type = override_type or value_expr.type
            if (
                not isinstance(value_type, ir.UnknownType)
                and not isinstance(existing_var.type, ir.UnknownType)
                and not types_match(existing_var.type, value_type)
                and not tensor_types_differ_only_by_valid_shape(existing_var.type, value_type)
            ):
                if not tile_types_differ_only_by_unset_memory_space(existing_var.type, value_type):
                    raise ParserTypeError(
                        f"Cannot reassign '{var_name}' with a different type: "
                        f"was {ir.python_print_type(existing_var.type)}, "
                        f"got {ir.python_print_type(value_type)}",
                        span=span,
                        hint="Use a different variable name for tensors with different shapes or dtypes",
                    )
                # Compatible: one side just has not been placed yet. Keep the
                # binding on the Var's own type so the AssignStmt stays
                # symmetric; InferTileMemorySpace resolves both sides together.
                if (
                    isinstance(value_expr, ir.Call)
                    and isinstance(existing_var.type, ir.TileType)
                    and existing_var.type.memory_space is not None
                ):
                    value_expr = ir.Call(
                        value_expr.op,
                        value_expr.args,
                        value_expr.kwargs,
                        value_expr.attrs,
                        existing_var.type,
                        value_expr.span,
                    )
            self.function.shape_dim_aliases.pop(existing_var.unique_id, None)
            self.builder.assign(existing_var, value_expr, span=span)
            return existing_var
        var = self.builder.let(var_name, value_expr, type=override_type, span=span)
        if isinstance(value_expr, ir.Call) and value_expr.op.name == ir.get_op("tensor.dim").name:
            self.function.shape_dim_aliases[var.unique_id] = value_expr
        return var

    def parse_assignment(self, stmt: ast.Assign) -> None:  # noqa: PLR0912
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        # Intercept ``buf = pld.[tensor.]alloc_window_buffer(...)``: the alloc
        # op derives its ``name`` kwarg from the LHS, and the LHS name must be
        # globally unique within the @pl.program.
        if len(stmt.targets) == 1 and (
            is_pld_call(stmt.value, "alloc_window_buffer") or _is_dump_call(stmt.value, "alloc_window_buffer")
        ):
            self._parse_alloc_window_buffer_assignment(stmt.targets[0], stmt.value)
            return

        # Handle tuple unpacking for yields
        if len(stmt.targets) == 1:
            target = stmt.targets[0]

            # Handle tuple unpacking: (a, b, c) = pl.yield_(...) or self.func(...)
            if isinstance(target, ast.Tuple):
                # Check if value is a pl.yield_() call
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # This is handled in yield parsing
                        self.control.parse_yield_assignment(target, stmt.value)
                        return

                # ``out, tid = pl.submit(self.kernel, *args, deps=[...])`` —
                # manual_scope submit construct (desugars to a single Call
                # with an augmented Tuple{<kernel return>, TaskId} type).
                if is_pl_call(stmt.value, "submit"):
                    self.calls.parse_submit_assignment(target, stmt.value)
                    return
                # ``out, tid = pl.spmd_submit(self.kernel, *args, core_num=N,
                # sync_start=..., deps=[...])`` — SPMD task launch. Same desugar
                # as pl.submit, but the Submit carries the SPMD launch spec.
                if is_pl_call(stmt.value, "spmd_submit"):
                    self.calls.parse_submit_assignment(target, stmt.value, is_spmd=True)
                    return

                # General tuple unpacking for function calls returning TupleType
                span = self.span_tracker.get_span(stmt)
                value_expr = self.parse_expression(stmt.value)

                if (
                    isinstance(stmt.value, ast.Attribute)
                    and stmt.value.attr == "shape"
                    and isinstance(stmt.value.value, ast.Name)
                    and isinstance(value_expr, ir.MakeTuple)
                ):
                    if len(target.elts) != len(value_expr.elements):
                        raise ParserTypeError(
                            "Tensor shape unpacking requires one target per dimension",
                            span=span,
                        )
                    # Preserve parameter-dimension aliases instead of copying
                    # their symbols through a tuple temporary and projections.
                    for elt, item_expr in zip(target.elts, value_expr.elements, strict=True):
                        if not isinstance(elt, ast.Name):
                            raise ParserSyntaxError(
                                "Tensor shape unpacking targets must be variable names",
                                span=self.span_tracker.get_span(elt),
                            )
                        var = self._assign_or_let(elt.id, item_expr, span)
                        self.scope_manager.define_var(elt.id, var, span=span)
                    return

                # Bind the tuple result to a temporary variable
                tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)

                # Extract each element using TupleGetItemExpr
                for i, elt in enumerate(target.elts):
                    if not isinstance(elt, ast.Name):
                        raise ParserSyntaxError(
                            f"Tuple unpacking target must be a variable name, got {ast.unparse(elt)}",
                            span=self.span_tracker.get_span(elt),
                            hint="Use simple variable names in tuple unpacking: a, b, c = func()",
                        )
                    item_expr = ir.TupleGetItemExpr(tuple_var, i, span)
                    var = self._assign_or_let(elt.id, item_expr, span)
                    self.scope_manager.define_var(elt.id, var, span=span)
                return

            # Handle simple assignment
            if isinstance(target, ast.Name):
                var_name = target.id
                span = self.span_tracker.get_span(stmt)

                # ``result = pl.submit(self.kernel, ...)`` — single-LHS bind of
                # a Submit to a Tuple-typed Var. Round-trips the printer's
                # single-LHS form for a Submit whose AssignStmt LHS is a fresh
                # tuple temp (e.g. after a pass rewrite). The unpacking form
                # ``out, tid = pl.submit(...)`` continues to go through
                # ``_parse_submit_assignment``.
                if is_pl_call(stmt.value, "submit"):
                    self.calls.parse_submit_single_lhs(target, stmt.value)
                    return
                if is_pl_call(stmt.value, "spmd_submit"):
                    self.calls.parse_submit_single_lhs(target, stmt.value, is_spmd=True)
                    return

                # Check if this is a yield assignment: var = pl.yield_(...)
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # Handle yield assignment
                        yield_exprs: list[ir.Expr] = []
                        for arg in stmt.value.args:
                            expr = self.parse_expression(arg)
                            # Deliberate backstop: the delegated parser's declared ir.Expr
                            # return is a lie for a few DSL constructors.
                            if not isinstance(expr, ir.Expr):  # pyright: ignore[reportUnnecessaryIsInstance]
                                raise ParserSyntaxError(
                                    f"Yield argument must be an IR expression, got {type(expr)}",
                                    span=self.span_tracker.get_span(arg),
                                    hint="Ensure yield arguments are valid expressions",
                                )
                            yield_exprs.append(expr)

                        # Emit yield statement
                        yield_span = self.span_tracker.get_span(stmt.value)
                        self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))
                        self.control.track_yield_var(var_name, yield_exprs)

                        # Don't register in scope yet - will be done when loop/if completes
                        return

                value_expr = self.parse_expression(stmt.value)
                var = self._assign_or_let(var_name, value_expr, span)
                self.scope_manager.define_var(var_name, var, span=span)

                # Track buffer metadata for attribute access (e.g., pipe_buf.base)
                if isinstance(stmt.value, ast.Call):
                    self.ops.track_buffer_meta(var_name, stmt.value)

                return

            # Handle subscript-write: dst[i:i+64, j:j+64] = src
            if isinstance(target, ast.Subscript):
                self.subscripts.parse_subscript_assignment(target, stmt.value)
                return

        raise ParserSyntaxError(
            f"Unsupported assignment: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
        )

    def _bind_unpack_target(self, target: ast.expr, value_expr: ir.Expr, span: ir.Span) -> list[ir.Var]:
        """Bind a tuple-unpacking target (a Name or a nested Tuple) to an IR expr.

        Used by the ``pl.submit(...)`` desugaring so a multi-output kernel can
        be unpacked as ``(a, b), tid = pl.submit(...)``.

        Returns the Vars bound, in binding order (flattened across nesting).
        """
        if isinstance(target, ast.Name):
            var = self._assign_or_let(target.id, value_expr, span)
            self.scope_manager.define_var(target.id, var, span=span)
            return [var]
        if isinstance(target, ast.Tuple):
            tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)
            bound: list[ir.Var] = []
            for i, elt in enumerate(target.elts):
                item_expr = ir.TupleGetItemExpr(tuple_var, i, span)
                bound.extend(self._bind_unpack_target(elt, item_expr, span))
            return bound
        raise ParserSyntaxError(
            f"Tuple unpacking target must be a variable name or nested tuple, got {ast.unparse(target)}",
            span=span,
            hint="Use simple variable names in tuple unpacking: a, b, c = func()",
        )

    def _parse_alloc_window_buffer_assignment(self, target: ast.expr, value: ast.Call) -> None:
        """Parse ``buf = pld.[tensor.]alloc_window_buffer(size)``.

        Parser-only concerns (everything else delegates to the DSL wrapper /
        IR builder / C++ deducer via :func:`invoke_dsl`):

        - HOST-only, like ``world_size`` (see ``_validate_pld_op_call``): the
          window buffer is a host-orchestration resource, not lowerable inside
          InCore / SPMD scopes.
        - LHS must be a single ``ast.Name``.
        - That name must be globally unique within the ``@pl.program``.
        - User can't pass ``name=`` (it's parser-injected from the LHS).
        """
        span = self.span_tracker.get_span(value)
        is_printed = _is_dump_call(value, "alloc_window_buffer")

        in_device_scope = any(
            self.regions.is_inside_scope(kind) for kind in (ir.ScopeKind.InCore, ir.ScopeKind.Spmd)
        )
        if not is_printed and (self.function.func_level != ir.Level.HOST or in_device_scope):
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer() can only be called in HOST orchestration "
                "context (not inside InCore / SPMD scopes); "
                f"current function level: {self.function.func_level}",
                span=span,
                hint="Use '@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)' "
                "on the enclosing function and call outside any nested device-side scope",
            )

        if not isinstance(target, ast.Name):
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer must be assigned to a single variable name "
                f"(got '{ast.unparse(target)}')",
                span=span,
                hint="Use 'buf = pld.tensor.alloc_window_buffer(...)' (no tuple unpacking, no subscripts)",
            )

        name = target.id

        if not is_printed and name in self.bindings.alloc_window_buffer_names:
            raise ParserSyntaxError(
                f"pld.tensor.alloc_window_buffer name '{name}' is already declared in this program",
                span=span,
                hint="Each window buffer must have a globally unique name across all functions",
            )

        args = [
            self.parse_expression(a) if is_printed else self.ops.parse_op_positional_arg(a)
            for a in value.args
        ]
        user_kwargs = self.ops.parse_op_kwargs(value)
        if "name" in user_kwargs and not is_printed:
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer 'name' kwarg cannot be passed explicitly — "
                f"it is auto-derived from the assignment LHS ('{name}')",
                span=span,
                hint="Drop the 'name=...' kwarg; the LHS variable name becomes the buffer name",
            )

        # Printed snapshots retain their fields directly. Authoring syntax goes
        # through invoke_dsl for the usual arity, size, and keyword validation.
        if is_printed:
            alloc_call = ir.Call(
                ir.get_op("pld.tensor.alloc_window_buffer"),
                args,
                user_kwargs,
                self.attrs.parse_op_attrs(value) or {},
                ir.PtrType(),
                span,
            )
        else:
            alloc_call = invoke_dsl(
                _dsl_pld.alloc_window_buffer,
                args,
                {"name": name, **user_kwargs},
                span,
            )

        self.bindings.alloc_window_buffer_names.add(name)
        var = self.type_resolver.interned_base_ptr(name)
        if var is None:
            var = self._assign_or_let(name, alloc_call, span)
        else:
            self.builder.emit(ir.AssignStmt(var, alloc_call, span))
        self.scope_manager.define_var(name, var, span=span)

    @staticmethod
    def _format_ir_arg(expr: ir.Expr) -> str:
        """Format an IR expression for static_print output."""
        if isinstance(expr, ir.Var):
            return f"{expr.name_hint}: {python_print(expr.type)}"
        if isinstance(expr, (ir.ConstInt, ir.ConstFloat, ir.ConstBool)):
            return f"{expr.value}: {python_print(expr.type)}"
        return python_print(expr)

    def _format_fstring(self, node: ast.JoinedStr) -> str:
        """Format an f-string (ast.JoinedStr) for static_print output.

        Processes each part of the f-string: string literals are kept as-is,
        and expression placeholders are formatted via IR.
        """
        segments: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                segments.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                if value.conversion != -1 or value.format_spec is not None:
                    raise UnsupportedFeatureError(
                        "F-string conversion (!r, !s, !a) and format specs (:.2f) "
                        "are not supported in static_print",
                        span=self.span_tracker.get_span(value.value),
                        hint='Use plain f-string placeholders like f"{variable}"',
                    )
                expr = self.parse_expression(value.value)
                segments.append(self._format_ir_arg(expr))
            else:
                segments.append(str(value))
        return "".join(segments)

    def _handle_static_print(self, stmt: ast.Expr) -> None:
        """Handle pl.static_print() — print IR info to stdout at parse time."""
        call = stmt.value
        assert isinstance(call, ast.Call)
        span = self.span_tracker.get_span(stmt)

        if not call.args:
            raise ParserSyntaxError(
                "static_print() requires at least 1 argument",
                span=span,
                hint="Use: pl.static_print(variable) or pl.static_print('label', variable)",
            )

        parts: list[str] = []
        for arg in call.args:
            # String literals are printed as-is (labels)
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                parts.append(arg.value)
            elif isinstance(arg, ast.JoinedStr):
                parts.append(self._format_fstring(arg))
            else:
                expr = self.parse_expression(arg)
                parts.append(self._format_ir_arg(expr))

        location = f"{span.filename}:{span.begin_line}" if span.filename else f"line {span.begin_line}"
        print(f"static_print [{location}]: {' '.join(parts)}")

    def _handle_static_assert(self, stmt: ast.Expr) -> None:
        """Handle pl.static_assert() — assert condition at parse time."""
        call = stmt.value
        assert isinstance(call, ast.Call)
        span = self.span_tracker.get_span(stmt)

        args = call.args
        if len(args) < 1 or len(args) > 2:
            raise ParserSyntaxError(
                "static_assert() requires 1 or 2 arguments",
                span=span,
                hint="Use: pl.static_assert(condition) or pl.static_assert(condition, 'message')",
            )

        # Extract optional message (must be string literal)
        msg = ""
        if len(args) == 2:
            if not isinstance(args[1], ast.Constant) or not isinstance(args[1].value, str):
                raise ParserSyntaxError(
                    "static_assert() message must be a string literal",
                    span=self.span_tracker.get_span(args[1]),
                )
            msg = args[1].value

        error_msg = f"static_assert failed: {msg}" if msg else "static_assert failed"

        # Try Python-level evaluation first (handles closure variable expressions)
        success, value = self.expr_evaluator.try_eval_expr(args[0])
        if success:
            if not value:
                condition_src = ast.unparse(args[0])
                raise ParserError(
                    error_msg,
                    span=span,
                    hint=f"Condition `{condition_src}` evaluated to {value!r}",
                )
            return

        # Fall back to IR parsing — constants are compile-time evaluable
        expr = self.parse_expression(args[0])
        if isinstance(expr, (ir.ConstBool, ir.ConstInt, ir.ConstFloat)):
            if not expr.value:
                condition_src = ast.unparse(args[0])
                raise ParserError(
                    error_msg,
                    span=span,
                    hint=f"Condition `{condition_src}` evaluated to {expr.value!r}",
                )
            return

        condition_src = ast.unparse(args[0])
        raise ParserError(
            "static_assert condition must be compile-time evaluable",
            span=span,
            hint=f"Condition `{condition_src}` produced a non-constant IR expression",
        )

    def _handle_dump_tag(self, stmt: ast.Expr) -> None:
        """Handle ``pl.dump_tag(<name>)`` at statement position.

        ``pl.dump_tag(t)`` is the declarative per-tensor dump marker. It records
        the bound Var so that every *subsequent* kernel dispatch consuming that
        exact Var gets it merged into the dispatch's ``dump_vars`` attr (Call or
        Submit; see :meth:`CallParser.parse_kernel_call`). No IR statement is emitted; the
        marker is consumed here.

        ``FunctionType.Inline`` is accepted because the InlineFunctions pass
        splices the inline body — including any ``dump_vars`` attrs on its
        kernel calls — into the caller orchestration, with the mutator
        substituting the callee Var for the caller's arg. No attr migration is
        needed.
        """
        call = stmt.value
        assert isinstance(call, ast.Call)
        span = self.span_tracker.get_span(stmt)

        if (
            not ir.is_orchestration_like(self.function.func_type)
            and self.function.func_type != ir.FunctionType.Inline
        ):
            raise ParserSyntaxError(
                "pl.dump_tag() is only valid inside an Orchestration or Inline function",
                span=span,
                hint=(
                    "Move the pl.dump_tag(...) marker into the @pl.function(type=pl."
                    "FunctionType.Orchestration) function whose tasks consume the tagged "
                    "tensor, or into an @pl.jit.inline / @pl.function(type=pl."
                    "FunctionType.Inline) helper that the orchestration inlines. Selective "
                    "tensor dump is filtered per kernel call by the orchestration codegen; "
                    "kernel-body (AIV / AIC / Group) usage has no effect."
                ),
            )
        if len(call.args) != 1 or call.keywords:
            raise ParserSyntaxError(
                "pl.dump_tag() takes exactly one positional argument (no keywords)",
                span=span,
                hint="Use: pl.dump_tag(tensor_var)",
            )
        if not isinstance(call.args[0], ast.Name):
            raise ParserSyntaxError(
                "pl.dump_tag() argument must be a bare variable name",
                span=self.span_tracker.get_span(call.args[0]),
                hint=(
                    "Write pl.dump_tag(q) where q is a tensor variable bound in this "
                    "orchestration scope. Attribute / subscript / call expressions are "
                    "not supported."
                ),
            )
        # Forward-sticky: record the bound Var so subsequent kernel calls that
        # consume this exact Var add it to their ``dump_vars`` attr (see
        # ``_parse_kernel_call``). The scope manager returns a stable object per
        # binding, so identity matching is reliable and a later reassignment of
        # the same name yields a new (untagged) Var.
        name = call.args[0].id
        var = self.scope_manager.lookup_var(name)
        if var is None:
            raise ParserSyntaxError(
                f"pl.dump_tag() argument '{name}' is not defined at this point",
                span=span,
                hint="Tag a tensor only after it is bound (a parameter or an earlier assignment).",
            )
        # ``lookup_var`` may return a non-Var placeholder (e.g. a loop-yield
        # name string), and only tensors are dumpable. Reject early so
        # ``_merge_forward_sticky_dump`` never sees a typeless binding.
        if not isinstance(var, ir.Var) or not isinstance(var.type, ir.TensorType):
            raise ParserTypeError(
                f"pl.dump_tag() argument '{name}' is not a tensor (got {type(var).__name__})",
                span=self.span_tracker.get_span(call.args[0]),
                hint="Only tensors can be selectively dumped.",
            )
        if not any(var is t for t in self.function.dump_tagged_vars):
            self.function.dump_tagged_vars.append(var)

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        In inline mode, captures the return expression instead of emitting ReturnStmt.

        Args:
            stmt: Return AST node
        """
        if self.inline.inline_mode:
            if stmt.value is None:
                return  # void inline, no return value
            if isinstance(stmt.value, ast.Tuple):
                exprs = [self.parse_expression(elt) for elt in stmt.value.elts]
                self.inline.inline_return_expr = ir.MakeTuple(exprs, self.span_tracker.get_span(stmt))
            else:
                self.inline.inline_return_expr = self.parse_expression(stmt.value)
            return

        span = self.span_tracker.get_span(stmt)

        if stmt.value is None:
            self.builder.return_stmt(None, span)
            return

        # Handle tuple return
        if isinstance(stmt.value, ast.Tuple):
            return_exprs: list[ir.Expr] = []
            for elt in stmt.value.elts:
                return_exprs.append(self.parse_expression(elt))
            self.builder.return_stmt(return_exprs, span)
        else:
            # Single return value
            return_expr = self.parse_expression(stmt.value)
            self.builder.return_stmt([return_expr], span)

    def parse_evaluation_statement(self, stmt: ast.Expr) -> None:
        """Parse evaluation statement (EvalStmt).

        Evaluation statements represent operations executed for their side effects,
        with the return value discarded (e.g., synchronization barriers).

        Args:
            stmt: Expr AST node
        """
        # Intercept compile-time-only constructs (produce no IR)
        if is_dsl_call(stmt, "static_print"):
            self._handle_static_print(stmt)
            return
        if is_dsl_call(stmt, "static_assert"):
            self._handle_static_assert(stmt)
            return
        if is_pl_call(stmt.value, "dump_tag"):
            self._handle_dump_tag(stmt)
            return
        if is_cache_policy_marker(stmt.value):
            self.scopes.handle_set_cache_policy(stmt)
            return

        # Special case: bare pl.yield_() emits a YieldStmt via parse_yield_call.
        # Do not create an additional EvalStmt for the returned expression.
        if (
            isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Attribute)
            and stmt.value.func.attr == "yield_"
            and isinstance(stmt.value.func.value, ast.Name)
            and stmt.value.func.value.id == "pl"
        ):
            self.control.parse_yield_call(stmt.value)
            return

        expr = self.parse_expression(stmt.value)
        span = self.span_tracker.get_span(stmt)

        # Validate that we got an IR expression (not a list literal, etc.); the
        # delegated parser's declared return is a lie for a few DSL constructors.
        if not isinstance(expr, ir.Expr):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ParserSyntaxError(
                f"Evaluation statement must be an IR expression, got {type(expr).__name__}",
                span=span,
                hint="Only function calls and operations can be used as standalone statements",
            )

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

    def parse_expression(self, expr: ast.expr) -> ir.Expr:
        """Parse expression and return IR Expr.

        Args:
            expr: AST expression node

        Returns:
            IR expression
        """
        if isinstance(expr, ast.Name):
            return self.parse_name(expr)
        elif isinstance(expr, ast.Constant):
            return self.parse_constant(expr)
        elif isinstance(expr, ast.BinOp):
            return self.parse_binop(expr)
        elif isinstance(expr, ast.Compare):
            return self.parse_compare(expr)
        elif isinstance(expr, ast.Call):
            return self.parse_call(expr)
        elif isinstance(expr, ast.Attribute):
            return self.parse_attribute(expr)
        elif isinstance(expr, ast.UnaryOp):
            return self.parse_unaryop(expr)
        elif isinstance(expr, ast.List):
            return self.parse_list(expr)
        elif isinstance(expr, ast.Tuple):
            return self.parse_tuple_literal(expr)
        elif isinstance(expr, ast.Subscript):
            return self.subscripts.parse_subscript(expr)
        elif isinstance(expr, ast.BoolOp):
            return self.parse_boolop(expr)
        else:
            raise UnsupportedFeatureError(
                f"Unsupported expression type: {type(expr).__name__}",
                span=self.span_tracker.get_span(expr),
                hint="Use supported expressions like variables, constants, operations, or function calls",
            )

    def parse_name(self, name: ast.Name) -> ir.Expr:
        """Parse variable name reference.

        Resolves names by checking the DSL scope first, then falling back
        to closure variables from the enclosing Python scope.

        Args:
            name: Name AST node

        Returns:
            IR expression (Var from scope, or constant/tuple from closure)
        """
        var_name = name.id
        var = self.scope_manager.lookup_var(var_name)

        if var is not None:
            return var

        # Fall back to closure variables
        result = self.expr_evaluator.try_eval_as_ir(name)
        if result is not None:
            return result

        raise UndefinedVariableError(
            f"Undefined variable '{var_name}'",
            span=self.span_tracker.get_span(name),
            hint="Check if the variable is defined before using it or is available in the enclosing scope",
        )

    def parse_constant(self, const: ast.Constant) -> ir.Expr:
        """Parse constant value.

        Args:
            const: Constant AST node

        Returns:
            IR constant expression
        """
        span = self.span_tracker.get_span(const)
        value = const.value

        if isinstance(value, bool):
            return ir.ConstBool(value, span)
        elif isinstance(value, int):
            return ir.ConstInt(value, DataType.INDEX, span)
        elif isinstance(value, float):
            return ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, span)
        elif value is None:
            # ``None`` is the "no producer yet" TaskId sentinel — the Pythonic
            # spelling of an invalid TaskId. Used to seed a TaskId loop
            # carry (``prev_tid = None``) or as a ``deps=[None]`` entry.
            # Lowers to ``system.task_invalid`` -> Scalar[TASK_ID]; codegen
            # emits ``TaskId::invalid()`` and downstream ``set_dependencies``
            # skips it via an ``is_valid()`` guard.
            return ir.create_op_call("system.task_invalid", [], {}, span)
        else:
            raise ParserTypeError(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, bool, or None (TaskId sentinel) constants",
            )

    def _can_fold_expr(self, node: ast.expr) -> bool:
        """Check whether an expression tree is safe to constant-fold.

        Returns True only when every ast.Name leaf is absent from the DSL
        scope (i.e. it must come from closure variables) and the tree
        contains only foldable node types (Name, Constant, BinOp, UnaryOp).
        This prevents incorrect folding when a DSL variable shadows a
        closure name and avoids unnecessary eval() overhead.
        """
        _FOLDABLE_AST_TYPES = (
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.operator,
            ast.unaryop,
            ast.expr_context,
        )
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if self.scope_manager.lookup_var(child.id) is not None:
                    return False
            elif not isinstance(child, _FOLDABLE_AST_TYPES):
                return False
        return True

    def parse_binop(self, binop: ast.BinOp) -> ir.Expr:
        """Parse binary operation.

        Attempts compile-time constant folding first: if every leaf of the
        BinOp tree can be resolved from closure variables (and no DSL-scoped
        name shadows a closure name), the whole expression is evaluated in
        Python and emitted as a single ConstInt / ConstFloat.

        Args:
            binop: BinOp AST node

        Returns:
            IR binary expression
        """
        if self._can_fold_expr(binop):
            folded = self.expr_evaluator.try_eval_as_ir(binop)
            if folded is not None:
                return folded

        span = self.span_tracker.get_span(binop)
        left = self.parse_expression(binop.left)
        right = self.parse_expression(binop.right)

        op_map = {
            ast.Add: ir.add,
            ast.Sub: ir.sub,
            ast.Mult: ir.mul,
            ast.Div: ir.truediv,
            ast.FloorDiv: ir.floordiv,
            ast.Mod: ir.mod,
            ast.Pow: ir.pow,
            ast.BitAnd: ir.bit_and,
            ast.BitOr: ir.bit_or,
            ast.BitXor: ir.bit_xor,
            ast.LShift: ir.bit_shift_left,
            ast.RShift: ir.bit_shift_right,
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %, **, &, |, ^, <<, >>",
            )

        return op_map[op_type](left, right, span)

    def parse_compare(self, compare: ast.Compare) -> ir.Expr:
        """Parse comparison operation.

        Args:
            compare: Compare AST node

        Returns:
            IR comparison expression
        """
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ParserSyntaxError(
                "Only simple comparisons supported",
                span=self.span_tracker.get_span(compare),
                hint="Use single comparison operators like: a < b, not chained comparisons",
            )

        span = self.span_tracker.get_span(compare)
        left = self.parse_expression(compare.left)
        right = self.parse_expression(compare.comparators[0])

        op_map = {
            ast.Eq: ir.eq,
            ast.NotEq: ir.ne,
            ast.Lt: ir.lt,
            ast.LtE: ir.le,
            ast.Gt: ir.gt,
            ast.GtE: ir.ge,
        }

        op_type = type(compare.ops[0])
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported comparison: {op_type.__name__}",
                span=self.span_tracker.get_span(compare),
                hint="Use supported comparisons: ==, !=, <, <=, >, >=",
            )

        return op_map[op_type](left, right, span)

    def parse_unaryop(self, unary: ast.UnaryOp) -> ir.Expr:
        """Parse unary operation.

        Attempts compile-time constant folding first, same as parse_binop.
        Skips folding for ``not`` (ast.Not) and ``~`` (ast.Invert) because
        their Python semantics differ from the DSL's IR operators.

        Args:
            unary: UnaryOp AST node

        Returns:
            IR unary expression
        """
        if not isinstance(unary.op, (ast.Not, ast.Invert)) and self._can_fold_expr(unary):
            folded = self.expr_evaluator.try_eval_as_ir(unary)
            if folded is not None:
                return folded

        span = self.span_tracker.get_span(unary)
        operand = self.parse_expression(unary.operand)

        op_map = {
            ast.USub: ir.neg,
            ast.Not: ir.not_,
            ast.Invert: ir.bit_not,
        }

        op_type = type(unary.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported unary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(unary),
                hint="Use supported unary operators: -, not, ~",
            )

        # Fold constant negation: -ConstInt(n) -> ConstInt(-n), -ConstFloat(n) -> ConstFloat(-n).
        # Python parses negative literals (e.g. -1 in function args) as UnaryOp(USub, Constant(1)),
        # but the IR builder creates ConstInt(-1) directly. Folding here preserves roundtrip.
        if op_type == ast.USub:
            if isinstance(operand, ir.ConstInt):
                return ir.ConstInt(-operand.value, operand.dtype, span)
            if isinstance(operand, ir.ConstFloat):
                return ir.ConstFloat(-operand.value, operand.dtype, span)

        return op_map[op_type](operand, span)

    def parse_boolop(self, boolop: ast.BoolOp) -> ir.Expr:
        """Parse boolean operation (and/or).

        Chains multiple values with left-to-right associativity:
        ``a and b and c`` becomes ``And(And(a, b), c)``.

        Args:
            boolop: BoolOp AST node

        Returns:
            IR boolean expression
        """
        span = self.span_tracker.get_span(boolop)
        op_map: dict[type, Any] = {
            ast.And: ir.and_,
            ast.Or: ir.or_,
        }
        op_type = type(boolop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported boolean operator: {op_type.__name__}",
                span=span,
                hint="Use 'and' or 'or'",
            )
        ir_op = op_map[op_type]
        values = [self.parse_expression(v) for v in boolop.values]
        result = values[0]
        for val in values[1:]:
            result = ir_op(result, val, span)
        return result

    def parse_call(self, call: ast.Call) -> ir.Expr:
        """Parse function call.

        Args:
            call: Call AST node

        Returns:
            IR expression from call
        """
        func = call.func

        if _is_dump_call(call):
            return self.calls.parse_dump_call(call)

        # Handle pl.yield_() specially
        if isinstance(func, ast.Attribute) and func.attr == "yield_":
            return self.control.parse_yield_call(call)

        # ``pl.submit(...)`` is an assignment-only construct — it must be
        # unpacked as ``out, tid = pl.submit(self.kernel, ...)``. Reaching it
        # in an expression context (bare statement or simple assignment) is a
        # usage error.
        if is_pl_call(call, "submit"):
            raise ParserSyntaxError(
                "pl.submit(...) must be unpacked as a 2-tuple: `out, tid = pl.submit(self.kernel, ...)`",
                span=self.span_tracker.get_span(call),
                hint="pl.submit returns (kernel result, producer TaskId); bind both with tuple unpacking.",
            )
        if is_pl_call(call, "spmd_submit"):
            raise ParserSyntaxError(
                "pl.spmd_submit(...) must be unpacked as a 2-tuple: "
                "`out, tid = pl.spmd_submit(self.kernel, ..., core_num=N)`",
                span=self.span_tracker.get_span(call),
                hint="pl.spmd_submit returns (kernel result, producer TaskId); unpack both as a 2-tuple.",
            )

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                span = self.span_tracker.get_span(call)
                # Plain ``self.kernel(...)`` is fire-and-forget — no producer
                # TaskId and ``deps=`` is rejected. Use ``pl.submit(...)`` to
                # capture a TaskId and attach explicit dependency edges.
                return self.calls.parse_kernel_call(func, call.args, call.keywords, span, as_submit=False)

            # Handle pl.tensor.*, pl.tile.*, and pl.* operation calls
            return self.ops.parse_op_call(call)

        # Handle bare-name calls to external ir.Function or InlineFunction
        if isinstance(func, ast.Name):
            from .decorator import InlineFunction  # noqa: PLC0415 (circular import)

            func_name = func.id
            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if isinstance(resolved, ir.Function):
                return self.calls.parse_external_function_call(func_name, resolved, call)
            if isinstance(resolved, InlineFunction):
                return self.calls.parse_inline_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, pl.yield_(), self.method() for cross-function calls, "
            "or call an external @pl.function / @pl.inline by name",
        )

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        if attr.attr == "shape" and isinstance(attr.value, ast.Name):
            tensor = self.scope_manager.lookup_var(attr.value.id)
            if isinstance(tensor, ir.Expr) and isinstance(tensor.type, ir.TensorType):
                span = self.span_tracker.get_span(attr)
                # Match tensor.dim's signature-symbol folding while deriving
                # the rank from the current IR value, including local results.
                orchestration = ir.is_orchestration_like(self.function.func_type)
                dimensions = [
                    extent
                    if isinstance(extent, ir.ConstInt)
                    or (orchestration and self.ops.is_param_dim_symbol(extent))
                    else ir_op.tensor.dim(tensor, axis, span)
                    for axis, extent in enumerate(tensor.type.shape)
                ]
                return ir.MakeTuple(dimensions, span)

        # Try to evaluate as a Python enum value (e.g., pl.MemorySpace.Vec -> ConstInt)
        try:
            value = self.expr_evaluator.eval_expr(attr)
            if isinstance(value, ir.MemorySpace):
                return ir.ConstInt(value.value, DataType.INDEX, self.span_tracker.get_span(attr))
        except BUG_CLASS_EXCEPTIONS:
            # This eval is speculative, so its failure is normally not worth reporting - but a
            # failed internal invariant is, and swallowing it here would replace the diagnostic
            # with the unrelated "standalone attribute access" error raised below.
            raise
        except Exception:
            pass
        # This might be accessing a DataType enum or similar
        # For now, this is primarily used in calls, not standalone
        raise UnsupportedFeatureError(
            f"Standalone attribute access not supported: {ast.unparse(attr)}",
            span=self.span_tracker.get_span(attr),
            hint="Attribute access is only supported within function calls",
        )

    def _parse_sequence_literal(self, node: ast.List | ast.Tuple) -> ir.MakeTuple:
        """Parse list or tuple literal into MakeTuple IR expression.

        Args:
            node: List or Tuple AST node

        Returns:
            MakeTuple IR expression
        """
        span = self.span_tracker.get_span(node)
        elements = [self.parse_expression(elt) for elt in node.elts]
        return ir.MakeTuple(elements, span)

    def parse_list(self, list_node: ast.List) -> ir.MakeTuple:
        """Parse list literal into MakeTuple IR expression."""
        return self._parse_sequence_literal(list_node)

    def parse_tuple_literal(self, tuple_node: ast.Tuple) -> ir.MakeTuple:
        """Parse tuple literal like (x, y, z)."""
        return self._parse_sequence_literal(tuple_node)

    @staticmethod
    def _spmd_body_reads_block_idx(body: list[ast.stmt]) -> bool:
        """Keep the existing internal inspection entry point during migration."""
        return ScopeParser.spmd_body_reads_block_idx(body)

    def _convert_annotation_scalars(self, value: ir.Expr, annotation: ir.Type, name: str) -> ir.Expr:
        """Convert annotated scalars, including scalars nested inside tuples.

        Integer-compatible annotations can request real dtype conversions.
        Plan them from types first, so metadata-only annotations preserve the
        original expression. Retain type objects with the memo keys to prevent
        Python object IDs from being recycled during tuple-DAG traversal.
        """
        plans: dict[tuple[int, int], tuple[ir.Type, ir.Type, bool]] = {}

        def needs_conversion(actual: ir.Type, target: ir.Type) -> bool:
            key = (id(actual), id(target))
            if key in plans:
                return plans[key][2]
            if isinstance(actual, ir.ScalarType) and isinstance(target, ir.ScalarType):
                needed = actual.dtype != target.dtype
            elif isinstance(actual, ir.TupleType) and isinstance(target, ir.TupleType):
                needed = any(needs_conversion(a, t) for a, t in zip(actual.types, target.types))
            else:
                needed = False
            plans[key] = (actual, target, needed)
            return needed

        def convert(expr: ir.Expr, target: ir.Type, hint: str) -> ir.Expr:
            if not needs_conversion(expr.type, target):
                return expr
            if isinstance(target, ir.ScalarType):
                if isinstance(expr, ir.ConstInt):
                    # Bare literals start as INDEX placeholders. Stamp the value
                    # too, or SSA substitution can give loop carries the wrong dtype.
                    return ir.ConstInt(expr.value, target.dtype, expr.span)
                if isinstance(expr, ir.Call):
                    # Preserve the dtype actually emitted by the producer (#2779)
                    # and keep calls at statement level before casting their result.
                    expr = self.builder.let(f"{hint}_index", expr, span=expr.span)
                return ir.cast(expr, target.dtype, expr.span)
            assert isinstance(target, ir.TupleType)
            if isinstance(expr, ir.MakeTuple):
                elements = list(expr.elements)
            else:
                # A tuple-producing Call/Submit must run once, with its original
                # result type. Cast the extracted values rather than relabeling
                # the producer's ABI or nesting repeated calls in MakeTuple.
                if not isinstance(expr, ir.Var):
                    expr = self.builder.let(f"{hint}_source", expr, span=expr.span)
                elements = [ir.TupleGetItemExpr(expr, i, expr.span) for i in range(len(target.types))]
            return ir.MakeTuple(
                [
                    convert(element, t, f"{hint}_{i}")
                    for i, (element, t) in enumerate(zip(elements, target.types))
                ],
                expr.span,
            )

        return convert(value, annotation, name)


__all__ = ["ASTParser"]
