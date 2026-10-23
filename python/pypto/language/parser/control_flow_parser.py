# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structured control flow, loop-carried values and yield bindings."""

import ast
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from pypto.ir.printer import python_print
from pypto.pypto_core import DataType, ir

from .attr_parser import AttrParser
from .diagnostics import (
    InvalidOperationError,
    ParserSyntaxError,
    ParserTypeError,
)
from .parser_state import ControlFlowState, ParseCallbacks, ParserServices
from .scope_parser import ScopeParser
from .source_comments import SourceCommentTracker
from .syntax_helpers import is_const_int, is_dsl_call
from .type_resolver import (
    _const_int_value,  # noqa: PLC2701
    _is_dump_call,  # noqa: PLC2701
)

VALID_ITERATORS = {"range", "parallel", "unroll", "pipeline", "while_", "spmd", "split_aiv"}

ITERATOR_ERROR = (
    "For loop must use pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), pl.while_(), "
    "pl.spmd(), or pl.split_aiv()"
)

ITERATOR_HINT = (
    "Use pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), pl.while_(), pl.spmd(), "
    "or pl.split_aiv() as the iterator"
)

ITERATOR_KEYWORDS = {
    "range": ("init_values", "attrs"),
    "parallel": ("init_values", "attrs"),
    "unroll": ("init_values", "attrs"),
    "pipeline": ("init_values", "stage", "attrs"),
}

# Maps iterator type name to ForKind enum value.
ITERATOR_TO_KIND = {
    "range": ir.ForKind.Sequential,
    "parallel": ir.ForKind.Parallel,
    "unroll": ir.ForKind.Unroll,
    "pipeline": ir.ForKind.Pipeline,
}


class ControlFlowParser:
    """Structured control flow, loop-carried values and yield bindings."""

    def __init__(
        self,
        *,
        attrs: AttrParser,
        callbacks: ParseCallbacks,
        comments: SourceCommentTracker,
        flow: ControlFlowState,
        scopes: ScopeParser,
        services: ParserServices,
    ) -> None:
        self.attrs = attrs
        self.callbacks = callbacks
        self.comments = comments
        self.flow = flow
        self.scopes = scopes
        self.services = services

    @contextmanager
    def yield_tracking_scope(self) -> Generator[None, None, None]:
        """Save and restore yield tracking state around a block.

        Initializes fresh _current_yield_vars and _current_yield_types for the
        block, then restores the previous values (including None) when the block
        exits. None means yield tracking is inactive.
        """
        prev_vars = self.flow.current_yield_vars
        prev_types = self.flow.current_yield_types
        self.flow.current_yield_vars = []
        self.flow.current_yield_types = {}
        try:
            yield
        finally:
            self.flow.current_yield_vars = prev_vars
            self.flow.current_yield_types = prev_types

    def track_yield_var(self, var_name: str, yield_exprs: list[ir.Expr]) -> None:
        """Track a yielded variable name and infer its type.

        Called after emitting a YieldStmt to record the variable name
        for if/for/while output registration and capture the yield
        expression type for unannotated yield inference.

        Args:
            var_name: The variable name being assigned from the yield
            yield_exprs: The list of yielded IR expressions
        """
        if self.flow.current_yield_vars is not None:
            self.flow.current_yield_vars.append(var_name)
        if self.flow.current_yield_types is not None:
            if len(yield_exprs) == 1:
                self.flow.current_yield_types.setdefault(var_name, yield_exprs[0].type)

    def parse_yield_assignment(self, target: ast.Tuple, value: ast.Call) -> None:
        """Parse yield assignment: (a, b) = pl.yield_(x, y).

        Args:
            target: Tuple of target variable names
            value: Call to pl.yield_()
        """
        # Parse yield expressions
        yield_exprs: list[ir.Expr] = []
        for arg in value.args:
            expr = self.callbacks.expression(arg)
            # Deliberate backstop: the delegated parser's declared ir.Expr return
            # is a lie for a few DSL constructors.
            if not isinstance(expr, ir.Expr):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ParserSyntaxError(
                    f"Yield argument must be an IR expression, got {type(expr)}",
                    span=self.services.span_tracker.get_span(arg),
                    hint="Ensure yield arguments are valid expressions",
                )
            yield_exprs.append(expr)

        # Emit yield statement
        span = self.services.span_tracker.get_span(value)
        self.services.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variable names and types for if/for statement processing
        for i, elt in enumerate(target.elts):
            if isinstance(elt, ast.Name) and i < len(yield_exprs):
                self.track_yield_var(elt.id, [yield_exprs[i]])

        # For tuple yields at the for/while loop level, register the variables
        # (they'll be available as loop.get_result().return_vars)
        if (self.flow.in_for_loop or self.flow.in_while_loop) and not self.flow.in_if_stmt:
            # Register yielded variable names in scope
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    # Will be resolved from loop outputs
                    self.services.scope_manager.define_var(var_name, f"loop_yield_{i}")

    def validate_for_loop_iterator(self, stmt: ast.For) -> tuple[ast.Call, str]:
        """Validate that for loop uses pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), or pl.while_().

        Returns:
            Tuple of (call_node, iterator_type) where iterator_type is
            "range", "parallel", "unroll", "pipeline", or "while_"
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ParserSyntaxError(
                ITERATOR_ERROR,
                span=self.services.span_tracker.get_span(stmt.iter),
                hint=ITERATOR_HINT,
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute) and func.attr in VALID_ITERATORS:
            return iter_call, func.attr

        raise ParserSyntaxError(
            ITERATOR_ERROR,
            span=self.services.span_tracker.get_span(stmt.iter),
            hint=ITERATOR_HINT,
        )

    def parse_for_loop_target(self, stmt: ast.For) -> tuple[str, ast.AST | None, bool]:
        """Parse for loop target, returning (loop_var_name, iter_args_node, is_simple_for)."""
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id, None, True

        if isinstance(stmt.target, ast.Tuple) and len(stmt.target.elts) == 2:
            loop_var_node = stmt.target.elts[0]
            iter_args_node = stmt.target.elts[1]

            if not isinstance(loop_var_node, ast.Name):
                raise ParserSyntaxError(
                    "Loop variable must be a simple name",
                    span=self.services.span_tracker.get_span(loop_var_node),
                    hint="Use a simple variable name for the loop counter",
                )
            return loop_var_node.id, iter_args_node, False

        raise ParserSyntaxError(
            "For loop target must be a simple name or: (loop_var, (iter_args...))",
            span=self.services.span_tracker.get_span(stmt.target),
            hint="Use: for i in pl.range(n) or for i, (var1,) in pl.range(n, init_values=(...,))",
        )

    def setup_iter_args(self, loop: Any, iter_args_node: ast.AST, init_values: list[ir.Expr]) -> None:
        """Set up iter_args and return_vars for Pattern A loops."""
        if not isinstance(iter_args_node, ast.Tuple):
            raise ParserSyntaxError(
                "Iter args must be a tuple",
                span=self.services.span_tracker.get_span(iter_args_node),
                hint="Wrap iteration variables in parentheses: (var1, var2)",
            )

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.services.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.services.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.services.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def parse_for_loop(self, stmt: ast.For) -> None:  # noqa: PLR0912
        """Parse for loop with pl.range(), pl.parallel(), pl.unroll(), or pl.while_().

        Supports patterns for range/parallel/unroll:
          Pattern A (explicit): for i, (vars,) in pl.range(..., init_values=(...,))
          Pattern B (simple):   for i in pl.range(n)

        Supports pattern for while-as-for:
          for (vars,) in pl.while_(init_values=(...,)):
              pl.cond(condition)

        Both patterns also work with pl.parallel() and pl.unroll(). Explicit
        init_values on pl.unroll() represent transitional SSA IR; UnrollLoops
        still requires non-SSA input and normally runs before ConvertToSSA.
        Pattern B produces a ForStmt without iter_args/return_vars/yield.
        The C++ ConvertToSSA pass handles converting to SSA form.
        """
        iter_call, iterator_type = self.validate_for_loop_iterator(stmt)

        # Handle pl.while_() case
        if iterator_type == "while_":
            self.parse_while_as_for(stmt, iter_call)
            return

        # Handle pl.spmd() loop form — auto-outlines into Spmd(InCore(body)).
        if iterator_type == "spmd":
            self.scopes.parse_spmd_for_loop(stmt, iter_call)
            return

        # Handle pl.split_aiv() loop form — opens a single explicit-split InCore scope.
        if iterator_type == "split_aiv":
            self.scopes.parse_split_aiv_for_loop(stmt, iter_call)
            return

        loop_var_name, iter_args_node, is_simple_for = self.parse_for_loop_target(stmt)
        range_args = self.parse_range_call(iter_call, iterator_type)

        if is_simple_for and range_args["init_values"]:
            raise ParserSyntaxError(
                "For loop target must be a tuple when init_values is provided",
                span=self.services.span_tracker.get_span(stmt.target),
                hint="Use: for i, (var1,) in pl.range(n, init_values=(val1,)) to include iter_args",
            )

        self.require_yield_lhs_for_init_values(stmt, bool(range_args["init_values"]))

        # For pl.unroll(), require compile-time constant integer bounds
        # and reject step=0. Fail early with clear parser errors instead of
        # later generic failures in the UnrollLoops C++ pass.
        # Note: negative literals like -1 become ir.Neg(ir.ConstInt(1)).
        if iterator_type == "unroll":
            for _bound_name in ("start", "stop", "step"):
                _bound_value = range_args.get(_bound_name)
                if _bound_value is not None and not is_const_int(_bound_value):
                    raise ParserSyntaxError(
                        "pl.unroll() requires compile-time constant integer bounds",
                        span=self.services.span_tracker.get_span(iter_call),
                        hint="Use integer literals for start, stop, and step in pl.unroll().",
                    )
            _step = range_args.get("step")
            if _const_int_value(_step) == 0:
                raise ParserSyntaxError(
                    "pl.unroll() step cannot be zero",
                    span=self.services.span_tracker.get_span(iter_call),
                    hint="Use a non-zero step in pl.unroll(start, stop, step).",
                )

        # Validate stage= on pl.pipeline() and merge into attrs as "pipeline_stages".
        # stage= is required on pl.pipeline() and forbidden everywhere else.
        pipeline_stages: int | None = None
        stage_expr = range_args.get("stage")
        if iterator_type == "pipeline":
            if stage_expr is None:
                raise ParserSyntaxError(
                    "pl.pipeline() requires stage= (positive integer)",
                    span=self.services.span_tracker.get_span(iter_call),
                    hint="Use pl.pipeline(stop, stage=F).",
                )
            if not is_const_int(stage_expr):
                raise ParserSyntaxError(
                    "pl.pipeline() stage must be a compile-time constant integer",
                    span=self.services.span_tracker.get_span(iter_call),
                    hint="Use an integer literal: stage=4",
                )
            stage_val = _const_int_value(stage_expr)
            if stage_val is None or stage_val < 1:
                raise ParserSyntaxError(
                    f"pl.pipeline() stage must be >= 1, got {stage_val}",
                    span=self.services.span_tracker.get_span(iter_call),
                    hint="Use a positive integer for stage: stage=4",
                )
            pipeline_stages = stage_val
        elif stage_expr is not None:
            raise ParserSyntaxError(
                f"stage= is only supported on pl.pipeline(), not pl.{iterator_type}()",
                span=self.services.span_tracker.get_span(iter_call),
                hint="Use pl.pipeline() for software pipelining.",
            )

        kind = ITERATOR_TO_KIND[iterator_type]
        # Infer loop var dtype from range bounds to preserve roundtrip fidelity.
        # Bare Python ints still map to INDEX, but explicitly typed INT64 bounds
        # should rebuild an INT64 loop var when the printer emitted them that way.
        _loop_var_dtype = DataType.INDEX
        saw_index_bound = False
        saw_int64_bound = False
        for _bound in (range_args.get("start"), range_args.get("stop"), range_args.get("step")):
            if isinstance(_bound, ir.ConstInt):
                if _bound.dtype == DataType.INDEX:
                    saw_index_bound = True
                elif _bound.dtype == DataType.INT64:
                    saw_int64_bound = True
                else:
                    _loop_var_dtype = _bound.dtype
                    break
        if _loop_var_dtype == DataType.INDEX and saw_int64_bound and not saw_index_bound:
            _loop_var_dtype = DataType.INT64
        loop_var = self.services.builder.var(loop_var_name, ir.ScalarType(_loop_var_dtype))
        span = self.services.span_tracker.get_span(stmt)
        loop_output_vars: list[str] = []
        prev_loop_builder = self.flow.current_loop_builder
        prev_in_for_loop = self.flow.in_for_loop
        prev_in_while_loop = self.flow.in_while_loop

        attrs_dict: dict[str, object] | None = range_args.get("attrs") or None
        if pipeline_stages is not None:
            attrs_dict = dict(attrs_dict) if attrs_dict else {}
            attrs_dict["pipeline_stages"] = pipeline_stages
        with self.services.builder.for_loop(
            loop_var,
            range_args["start"],
            range_args["stop"],
            range_args["step"],
            span,
            kind,
            attrs=attrs_dict,
        ) as loop:
            self.flow.current_loop_builder = loop
            self.flow.in_for_loop = True
            self.flow.loop_kind_stack.append(iterator_type)
            self.services.scope_manager.enter_scope("for")
            self.services.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

            if not is_simple_for:
                assert iter_args_node is not None  # Guaranteed by _parse_for_loop_target
                self.setup_iter_args(loop, iter_args_node, range_args["init_values"])

            with self.yield_tracking_scope():
                self.callbacks.body(stmt.body)
                self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                assert self.flow.current_yield_vars is not None  # Guaranteed by _yield_tracking_scope
                loop_output_vars = self.flow.current_yield_vars[:]

            # Yield-LHS names if present; else synthetic `_out` names (pre-SSA).
            if not is_simple_for and range_args["init_values"]:
                if loop_output_vars:
                    annotations = dict(self.scan_for_yields(stmt.body))
                    for rv_name in loop_output_vars:
                        annotation = annotations.get(rv_name)
                        loop.return_var(
                            rv_name, self.resolve_yield_var_type(annotation) if annotation else None
                        )
                else:
                    assert iter_args_node is not None
                    assert isinstance(iter_args_node, ast.Tuple)
                    for iter_arg_node in iter_args_node.elts:
                        assert isinstance(iter_arg_node, ast.Name)
                        loop.return_var(f"{iter_arg_node.id}_out")

            should_leak = is_simple_for and not loop_output_vars
            self.services.scope_manager.exit_scope(leak_vars=should_leak)
            self.flow.loop_kind_stack.pop()
            self.flow.in_for_loop = prev_in_for_loop
            self.flow.in_while_loop = prev_in_while_loop
            self.flow.current_loop_builder = prev_loop_builder

        if not is_simple_for:
            loop_result = loop.get_result()
            if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
                for i, var_name in enumerate(loop_output_vars):
                    if i < len(loop_result.return_vars):
                        self.services.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def parse_range_keyword(self, keyword: ast.keyword, result: dict[str, Any], iterator_type: str) -> None:
        """Parse a single keyword argument from a range-like iterator call."""
        if keyword.arg == "init_values":
            if isinstance(keyword.value, (ast.List, ast.Tuple)):
                result["init_values"] = [self.callbacks.expression(elt) for elt in keyword.value.elts]
            else:
                raise ParserSyntaxError(
                    "init_values must be a list or tuple",
                    span=self.services.span_tracker.get_span(keyword.value),
                    hint="Use a tuple for init_values: init_values=(var1, var2)",
                )
        elif keyword.arg == "stage":
            result["stage"] = self.callbacks.expression(keyword.value)
        elif keyword.arg == "attrs":
            if not isinstance(keyword.value, ast.Dict):
                raise ParserSyntaxError(
                    "attrs must be a dict literal",
                    span=self.services.span_tracker.get_span(keyword.value),
                    hint='Use a dict like attrs={"my_attr": 1}',
                )
            result["attrs"] = self.attrs.parse_attrs_dict(keyword.value)
        else:
            supported = ITERATOR_KEYWORDS.get(iterator_type, ITERATOR_KEYWORDS["range"])
            raise ParserSyntaxError(
                f"Unknown keyword argument '{keyword.arg}' in pl.{iterator_type}()",
                span=self.services.span_tracker.get_span(keyword),
                hint=f"Supported keywords for pl.{iterator_type}(): {', '.join(supported)}",
            )

    def parse_range_call(self, call: ast.Call, iterator_type: str = "range") -> dict[str, Any]:
        """Parse pl.range()/parallel()/unroll()/pipeline() call arguments.

        Args:
            call: AST Call node for the iterator
            iterator_type: One of "range", "parallel", "unroll", "pipeline"

        Returns:
            Dictionary with start, stop, step, init_values
        """
        if len(call.args) < 1:
            raise ParserSyntaxError(
                f"pl.{iterator_type}() requires at least 1 argument (stop)",
                span=self.services.span_tracker.get_span(call),
                hint=f"Provide at least the stop value: pl.{iterator_type}(10) or pl.{iterator_type}(0, 10)",
            )

        start = 0
        step = 1

        if len(call.args) == 1:
            stop = self.callbacks.expression(call.args[0])
        elif len(call.args) == 2:
            start = self.callbacks.expression(call.args[0])
            stop = self.callbacks.expression(call.args[1])
        else:
            start = self.callbacks.expression(call.args[0])
            stop = self.callbacks.expression(call.args[1])
            step = self.callbacks.expression(call.args[2])

        result: dict[str, Any] = {
            "init_values": [],
            "unroll": None,
            "attrs": {},
        }
        for keyword in call.keywords:
            self.parse_range_keyword(keyword, result, iterator_type)

        result["start"] = start
        result["stop"] = stop
        result["step"] = step
        return result

    def is_cond_call(self, stmt: ast.stmt) -> bool:
        """Check if statement is a pl.cond() call (without parsing).

        Args:
            stmt: AST statement node

        Returns:
            True if statement is pl.cond() call, False otherwise
        """
        if not isinstance(stmt, ast.Expr):
            return False
        return is_dsl_call(stmt, "cond")

    def extract_cond_call(self, stmt: ast.stmt) -> ir.Expr | None:
        """Extract condition from pl.cond() call statement.

        Args:
            stmt: AST statement node

        Returns:
            Parsed condition expression if statement is pl.cond(), None otherwise
        """
        if not self.is_cond_call(stmt):
            return None

        assert isinstance(stmt, ast.Expr)
        call = stmt.value
        assert isinstance(call, ast.Call)

        # Parse the condition argument
        if len(call.args) != 1:
            raise ParserSyntaxError(
                "pl.cond() requires exactly 1 argument",
                span=self.services.span_tracker.get_span(call),
                hint="Use: pl.cond(condition)",
            )

        return self.callbacks.expression(call.args[0])

    def validate_while_call_args(self, while_call: ast.Call) -> None:
        """Validate that pl.while_() has no positional arguments."""
        if len(while_call.args) > 0:
            raise ParserSyntaxError(
                "pl.while_() takes no positional arguments",
                span=self.services.span_tracker.get_span(while_call),
                hint="Use: pl.while_(init_values=(...,)) with pl.cond(condition) as first statement in body",
            )

    def parse_while_init_values(self, while_call: ast.Call) -> list[ir.Expr]:
        """Parse init_values from pl.while_() keyword arguments."""
        init_values: list[ir.Expr] = []
        for keyword in while_call.keywords:
            if keyword.arg == "init_values":
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.callbacks.expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a tuple or list",
                        span=self.services.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values (lists also accepted): init_values=(var1, var2)",
                    )

        if not init_values:
            raise ParserSyntaxError(
                "pl.while_() requires init_values",
                span=self.services.span_tracker.get_span(while_call),
                hint="Provide init_values: pl.while_(init_values=(val1, val2))",
            )

        return init_values

    @classmethod
    def find_first_bare_yield(cls, stmts: list[ast.stmt]) -> ast.Expr | None:
        """First bare `pl.yield_(...)` expression-statement in `stmts`, or None.

        Recurses into `if/else` branches and `with` bodies at the same scope
        (e.g. `with pl.at(...): pl.yield_(...)` inside a loop body). Does NOT
        descend into nested For/While bodies — yields inside an inner loop
        bind to that inner loop's return_vars, not the outer one.
        """
        for s in stmts:
            if isinstance(s, ast.Expr) and is_dsl_call(s, "yield_"):
                return s
            if isinstance(s, ast.If):
                found = cls.find_first_bare_yield(s.body) or cls.find_first_bare_yield(s.orelse)
                if found is not None:
                    return found
            elif isinstance(s, ast.With):
                found = cls.find_first_bare_yield(s.body)
                if found is not None:
                    return found
        return None

    def require_yield_lhs_for_init_values(self, stmt: ast.For, init_values_present: bool) -> None:
        """Reject bare `pl.yield_(...)` when init_values is non-empty.

        Valid body shapes for an init-values loop:
          1. Assignment-form yield: `x_next = pl.yield_(updated)` — the LHS
             supplies the return_var name and the post-loop binding.
          2. No yield at all (pre-SSA): body mutates iter_args via AssignStmt;
             ConvertToSSA later synthesizes the yield. Synthetic `_out`
             return_var names are used.

        Bare `pl.yield_(...)` mixed with init_values is rejected: it would
        leave the post-loop binding nameless, and the user gets a downstream
        arity mismatch from the IR builder instead of a clear parser error.
        """
        if not init_values_present:
            return
        first_bare = self.find_first_bare_yield(stmt.body)
        if first_bare is None:
            return
        raise ParserSyntaxError(
            "Loop with init_values requires an assignment-form pl.yield_(...) "
            "— the LHS supplies the post-loop binding name. Bare pl.yield_(...) "
            "is rejected here.",
            span=self.services.span_tracker.get_span(first_bare),
            hint="Use yield-LHS form: `x_next = pl.yield_(updated)`. Then refer to `x_next` after the loop.",
        )

    def validate_while_body(self, stmt: ast.For) -> None:
        """Validate pl.while_() body structure."""
        if not stmt.body:
            raise ParserSyntaxError(
                "pl.while_() body cannot be empty",
                span=self.services.span_tracker.get_span(stmt),
                hint="Add pl.cond(condition) as first statement",
            )

        if not self.is_cond_call(stmt.body[0]):
            raise ParserSyntaxError(
                "First statement in pl.while_() body must be pl.cond(condition)",
                span=self.services.span_tracker.get_span(stmt.body[0]),
                hint="Add pl.cond(condition) as first statement",
            )

    def validate_while_target(self, stmt: ast.For, init_values: list[ir.Expr]) -> ast.Tuple:
        """Validate and return pl.while_() target tuple."""
        if not isinstance(stmt.target, ast.Tuple):
            raise ParserSyntaxError(
                "While loop target must be a tuple for pl.while_()",
                span=self.services.span_tracker.get_span(stmt.target),
                hint="Use: for (var1, var2) in pl.while_(init_values=(...,))",
            )

        iter_args_node = stmt.target

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.services.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        return iter_args_node

    def setup_while_iter_args(self, loop: Any, iter_args_node: ast.Tuple, init_values: list[ir.Expr]) -> None:
        """Set up iter_args for pl.while_() loop."""
        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.services.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.services.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def parse_while_body_statements(self, stmt: ast.For) -> list[str]:
        """Parse body statements for pl.while_() loop, return yielded vars."""
        with self.yield_tracking_scope():
            # Validate body (skip first statement which is pl.cond()).
            for body_stmt in stmt.body[1:]:
                if self.is_cond_call(body_stmt):
                    raise ParserSyntaxError(
                        "pl.cond() can only be the first statement in a pl.while_() loop body",
                        span=self.services.span_tracker.get_span(body_stmt),
                        hint="Remove this pl.cond() - condition is already specified at the start",
                    )
            self.callbacks.body(stmt.body[1:])
            self.comments.discard_tail_block_comments(stmt.body[1:], upper_line=stmt.end_lineno)

            assert self.flow.current_yield_vars is not None  # Guaranteed by _yield_tracking_scope
            return self.flow.current_yield_vars[:]

    def register_while_outputs(self, loop: Any, loop_output_vars: list[str]) -> None:
        """Register output variables from pl.while_() loop.

        Mirrors `parse_for_loop`: the post-loop binding name is the yield-LHS,
        not the header tuple. Header-tuple names are loop-scoped only.
        """
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
            for i, var_name in enumerate(loop_output_vars):
                if i < len(loop_result.return_vars):
                    self.services.scope_manager.define_var(
                        var_name, loop_result.return_vars[i], allow_redef=True
                    )

    def parse_while_as_for(self, stmt: ast.For, while_call: ast.Call) -> None:
        """Parse while loop using for...in pl.while_() pattern.

        Pattern: for (var1, var2) in pl.while_(init_values=(val1, val2)):
                     pl.cond(condition)
                     ...

        Args:
            stmt: For AST node
            while_call: Call to pl.while_()
        """
        # Validate and parse arguments
        self.validate_while_call_args(while_call)
        init_values = self.parse_while_init_values(while_call)
        self.validate_while_body(stmt)
        iter_args_node = self.validate_while_target(stmt, init_values)

        # init_values is always non-empty for pl.while_ (enforced earlier).
        self.require_yield_lhs_for_init_values(stmt, True)

        span = self.services.span_tracker.get_span(stmt)
        placeholder_condition = ir.ConstBool(True, span)
        prev_loop_builder = self.flow.current_loop_builder
        prev_in_for_loop = self.flow.in_for_loop
        prev_in_while_loop = self.flow.in_while_loop

        with self.services.builder.while_loop(placeholder_condition, span) as loop:
            self.flow.current_loop_builder = loop
            self.flow.in_while_loop = True
            self.flow.loop_kind_stack.append("while")
            self.services.scope_manager.enter_scope("while")

            # Set up iter_args
            self.setup_while_iter_args(loop, iter_args_node, init_values)

            # Parse and set the condition (now that iter_args are in scope)
            condition = self.extract_cond_call(stmt.body[0])
            if condition is None:
                raise ParserSyntaxError(
                    "First statement in pl.while_() body must be pl.cond(condition)",
                    span=self.services.span_tracker.get_span(stmt.body[0]),
                    hint="Add pl.cond(condition) as first statement",
                )
            loop.set_condition(condition)

            # Parse body statements first to get actual output variable names
            loop_output_vars = self.parse_while_body_statements(stmt)

            # Yield-LHS names if present; else synthetic `_out` names (pre-SSA).
            if loop_output_vars:
                annotations = dict(self.scan_for_yields(stmt.body))
                for var_name in loop_output_vars:
                    annotation = annotations.get(var_name)
                    loop.return_var(var_name, self.resolve_yield_var_type(annotation) if annotation else None)
            else:
                for iter_arg_node in iter_args_node.elts:
                    assert isinstance(iter_arg_node, ast.Name)
                    loop.return_var(f"{iter_arg_node.id}_out")

            # Enforce Bool-typed condition after iter_args and return_vars are both
            # set up, so that the while_loop context manager's __exit__ validation
            # does not mask this error with an arity mismatch.
            self.check_condition_is_bool(
                condition, "while", self.services.span_tracker.get_span(stmt.body[0])
            )

            self.services.scope_manager.exit_scope(leak_vars=False)
            self.flow.loop_kind_stack.pop()
            self.flow.in_for_loop = prev_in_for_loop
            self.flow.in_while_loop = prev_in_while_loop
            self.flow.current_loop_builder = prev_loop_builder

        # Register output variables
        self.register_while_outputs(loop, loop_output_vars)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Parse natural while loop syntax.

        Natural while syntax: while condition: body

        This creates a WhileStmt without iter_args (non-SSA form).
        The C++ ConvertToSSA pass will convert it to SSA form if needed.

        Args:
            stmt: While AST node
        """
        # Parse natural while syntax: while condition:
        condition = self.callbacks.expression(stmt.test)
        span = self.services.span_tracker.get_span(stmt)
        self.check_condition_is_bool(condition, "while", span)
        prev_loop_builder = self.flow.current_loop_builder
        prev_in_for_loop = self.flow.in_for_loop
        prev_in_while_loop = self.flow.in_while_loop

        with self.services.builder.while_loop(condition, span) as loop:
            self.flow.current_loop_builder = loop
            self.flow.in_while_loop = True
            self.flow.loop_kind_stack.append("while")
            self.services.scope_manager.enter_scope("while")

            # Parse body statements
            self.callbacks.body(stmt.body)
            self.comments.discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)

            # Variables leak to outer scope (ConvertToSSA will handle)
            self.services.scope_manager.exit_scope(leak_vars=True)
            self.flow.loop_kind_stack.pop()
            self.flow.in_for_loop = prev_in_for_loop
            self.flow.in_while_loop = prev_in_while_loop
            self.flow.current_loop_builder = prev_loop_builder

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement with phi nodes.

        When pl.yield_() is used, phi nodes are created via return_vars.
        When no yields are used (plain syntax), variables leak to outer scope
        and the C++ ConvertToSSA pass handles creating phi nodes.

        Args:
            stmt: If AST node
        """
        # Parse condition
        condition = self.callbacks.expression(stmt.test)
        span = self.services.span_tracker.get_span(stmt)
        self.check_condition_is_bool(condition, "if", span)

        # Track yield output variable names from both branches
        then_yield_vars: list[tuple[str, ast.expr | None]] = []

        # Begin if statement
        with self.services.builder.if_stmt(condition, span) as if_builder:
            self.flow.current_if_builder = if_builder
            self.flow.in_if_stmt = True

            with self.yield_tracking_scope():
                # Scan for yield variable names (without executing)
                then_yield_vars = self.scan_for_yields(stmt.body)

                # Also scan else branch to handle yields in both branches
                if stmt.orelse:
                    else_yield_vars = self.scan_for_yields(stmt.orelse)
                    # Merge with then branch yields (then branch takes precedence for type)
                    then_names = {name for name, _ in then_yield_vars}
                    for name, annotation in else_yield_vars:
                        if name not in then_names:
                            then_yield_vars.append((name, annotation))

                # Determine if we should leak variables (no explicit yields)
                should_leak = not bool(then_yield_vars)

                # Parse then branch (yield types captured via _current_yield_types)
                self.services.scope_manager.enter_scope("if")
                self.callbacks.body(stmt.body)
                self.comments.discard_tail_block_comments(
                    stmt.body, upper_line=self.comments.then_branch_tail_upper(stmt)
                )
                self.services.scope_manager.exit_scope(leak_vars=should_leak)

                # Parse else branch if present
                if stmt.orelse:
                    if_builder.else_()
                    self.services.scope_manager.enter_scope("else")
                    self.callbacks.body(stmt.orelse)
                    self.comments.discard_tail_block_comments(stmt.orelse, upper_line=stmt.end_lineno)
                    self.services.scope_manager.exit_scope(leak_vars=should_leak)

                # Declare return vars AFTER parsing branches so captured yield types
                # are available for unannotated yields (fixes issue #233 / #234)
                for var_name, annotation in then_yield_vars:
                    if annotation is not None:
                        var_type = self.resolve_yield_var_type(annotation)
                    elif (
                        self.flow.current_yield_types is not None
                        and var_name in self.flow.current_yield_types
                    ):
                        var_type = self.flow.current_yield_types[var_name]
                    else:
                        var_type = self.resolve_yield_var_type(None)
                    if_builder.return_var(var_name, var_type)

        # After if statement completes, register the output variables in the outer scope
        if then_yield_vars:
            # Get the output variables from the if statement
            if_result = if_builder.get_result()
            if hasattr(if_result, "return_vars") and if_result.return_vars:
                # Register each output variable with its name (extract name from tuple)
                for i, (var_name, _) in enumerate(then_yield_vars):
                    if i < len(if_result.return_vars):
                        output_var = if_result.return_vars[i]
                        self.services.scope_manager.define_var(var_name, output_var)

        self.flow.in_if_stmt = False
        self.flow.current_if_builder = None

    def parse_break(self, stmt: ast.Break) -> None:
        """Parse break statement.

        Args:
            stmt: Break AST node
        """
        span = self.services.span_tracker.get_span(stmt)
        self.validate_loop_control("break", span)
        self.services.builder.break_stmt(span=span)

    def parse_continue(self, stmt: ast.Continue) -> None:
        """Parse continue statement.

        Args:
            stmt: Continue AST node
        """
        span = self.services.span_tracker.get_span(stmt)
        self.validate_loop_control("continue", span)
        self.services.builder.continue_stmt(span=span)

    def validate_loop_control(self, keyword: str, span: ir.Span) -> None:
        """Validate that a break/continue statement is in a valid loop context.

        Args:
            keyword: "break" or "continue"
            span: Source span for error reporting

        Raises:
            InvalidOperationError: If not inside a loop, or inside a parallel/unrolled loop
        """
        if not self.flow.loop_kind_stack:
            raise InvalidOperationError(
                f"'{keyword}' outside loop",
                span=span,
                hint=f"'{keyword}' can only be used inside a for or while loop",
            )
        current_kind = self.flow.loop_kind_stack[-1]
        if current_kind == "parallel":
            raise InvalidOperationError(
                f"'{keyword}' not supported in parallel loops",
                span=span,
                hint=f"'{keyword}' can only be used in sequential (pl.range) or while loops",
            )
        if current_kind == "unroll":
            raise InvalidOperationError(
                f"'{keyword}' not supported in unrolled loops",
                span=span,
                hint=f"'{keyword}' can only be used in sequential (pl.range) or while loops",
            )
        if current_kind == "split_aiv":
            raise InvalidOperationError(
                f"'{keyword}' not supported inside a 'for ... in pl.split_aiv(...)' body",
                span=span,
                hint=f"'{keyword}' can only be used in sequential (pl.range) or while loops; a "
                "pl.split_aiv body is a scope over the two AIV lanes, not a loop",
            )

    def parse_yield_call(self, call: ast.Call) -> ir.Expr:
        """Parse pl.yield_() call.

        Args:
            call: Call to pl.yield_() or pl.yield_()

        Returns:
            IR expression (first yielded value for single yield)
        """
        span = self.services.span_tracker.get_span(call)
        yield_exprs: list[ir.Expr] = []

        for arg in call.args:
            expr = self.callbacks.expression(arg)
            yield_exprs.append(expr)

        # Emit yield statement
        self.services.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Return first expression as the "value" of the yield, so an
        # assignment-form yield like `var = pl.yield_(expr)` binds `var` to
        # `expr` for the body's purposes. The IR builder resolves the actual
        # return_var once the enclosing scope closes.
        if len(yield_exprs) == 0:
            # Bare pl.yield_() with no arguments — return None (used as bare stmt)
            return None  # type: ignore[return-value]
        if len(yield_exprs) == 1:
            return yield_exprs[0]

        # Bare yield statements may legally yield multiple loop-carried values;
        # expression contexts still require assignment-form unpacking.
        if self.flow.in_for_loop or self.flow.in_while_loop or self.flow.in_if_stmt:
            return None  # type: ignore[return-value]

        raise ParserSyntaxError(
            "Multiple yields should use tuple unpacking assignment",
            span=self.services.span_tracker.get_span(call),
            hint="Use tuple unpacking: (a, b) = pl.yield_(x, y)",
        )

    def resolve_yield_var_type(self, annotation: ast.expr | None) -> ir.Type:
        """Resolve type annotation for a yield variable.

        Args:
            annotation: Type annotation AST node, or None if not annotated

        Returns:
            Resolved IR type
        """
        if annotation is None:
            # Fallback to generic tensor type when no annotation present
            return ir.TensorType([1], DataType.INT32)

        resolved = self.services.type_resolver.resolve_type(annotation)
        # resolve_type can return list[Type] for tuple[...] annotations
        if isinstance(resolved, list):
            if len(resolved) == 0:
                # Empty tuple type - use fallback
                return ir.TensorType([1], DataType.INT32)
            if len(resolved) == 1:
                # Single element - unwrap
                return resolved[0]
            # Multiple elements - create TupleType
            return ir.TupleType(resolved)
        # Single type
        return resolved

    def check_condition_is_bool(self, condition: Any, kind: str, span: Any) -> None:
        """Check that an if/while condition is a scalar BOOL expression.

        Raises ParserTypeError if the condition is not Bool-typed. No auto-coercion:
        users must write an explicit comparison such as `if x != 0:` or `if x > 0:`
        instead of `if x:` when x is not already a bool.

        Args:
            condition: Parsed condition expression (ir.Expr)
            kind: "if" or "while" (used in error message)
            span: Span for error reporting
        """
        cond_type = condition.type
        if not isinstance(cond_type, ir.ScalarType) or cond_type.dtype != DataType.BOOL:
            type_str = python_print(cond_type) if isinstance(cond_type, ir.Type) else str(cond_type)
            raise ParserTypeError(
                f"{kind} condition must be a Bool-typed scalar expression, got {type_str}",
                span=span,
                hint=f"Use an explicit boolean comparison, e.g. `{kind} x != 0:` or `{kind} x > 0:`",
            )

    def scan_for_yields(self, stmts: list[ast.stmt]) -> list[tuple[str, ast.expr | None]]:
        """Scan statements for yield assignments to determine output variable names and types.

        Args:
            stmts: List of statements to scan

        Returns:
            List of tuples (variable_name, type_annotation) where type_annotation is None if not annotated
        """
        yield_vars: list[tuple[str, ast.expr | None]] = []

        for stmt in stmts:
            # Check for annotated assignment with yield_: var: type = pl.yield_(...)
            if isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name) and isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        yield_vars.append((stmt.target.id, stmt.annotation))

            # Check for regular assignment with yield_: var = pl.yield_(...)
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    # Single variable assignment
                    if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            yield_vars.append((target.id, None))
                    # Tuple unpacking: (a, b) = pl.yield_(...)
                    elif isinstance(target, ast.Tuple) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Attribute) and func.attr == "yield_":
                            annotations = self.dump_yield_types(stmt.value, len(target.elts))
                            for i, elt in enumerate(target.elts):
                                if isinstance(elt, ast.Name):
                                    yield_vars.append((elt.id, annotations[i]))

            # Skip nested if statements — each nested if's yields are
            # its own return_vars, handled by parse_if_statement when
            # it processes that specific if node. Recursing would
            # incorrectly count inner yields as outer return_vars.
            elif isinstance(stmt, ast.If):
                pass

            # Descend into `with pl.scope():` blocks — they are transparent to
            # yield association, so a trailing `var = pl.yield_(...)` inside the
            # scope still defines the enclosing for/if return-var. These are
            # inserted by the MaterializeRuntimeScopes pass (or written by hand).
            elif isinstance(stmt, ast.With) and self.is_runtime_scope_with(stmt):
                yield_vars.extend(self.scan_for_yields(stmt.body))

        return yield_vars

    @staticmethod
    def is_runtime_scope_with(stmt: ast.With) -> bool:
        """True if @p stmt is a runtime-scope block (``with pl.scope(...):`` any
        mode, or the ``with pl.manual_scope():`` alias).

        Must mirror the ``func.attr`` dispatch in ``parse_with_statement`` —
        both ``scope`` and ``manual_scope`` build a RuntimeScopeStmt, so both
        must be transparent to yield scanning.
        """
        for item in stmt.items:
            ce = item.context_expr
            if (
                isinstance(ce, ast.Call)
                and isinstance(ce.func, ast.Attribute)
                and ce.func.attr in {"scope", "manual_scope"}
            ):
                return True
        return False

    def dump_yield_types(self, call: ast.Call, count: int) -> list[ast.expr | None]:
        """Read lossless result annotations from a printed multi-result yield."""
        if not _is_dump_call(call, "yield_"):
            return [None] * count
        if len(call.keywords) != 1 or call.keywords[0].arg != "types":
            raise ParserSyntaxError(
                "_dump.yield_ requires a types list", span=self.services.span_tracker.get_span(call)
            )
        types = call.keywords[0].value
        if not isinstance(types, ast.List) or len(types.elts) != count:
            raise ParserSyntaxError(
                "_dump.yield_ types must match the number of results",
                span=self.services.span_tracker.get_span(call),
            )
        return list(types.elts)
