# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST parsing for converting Python DSL to IR builder calls."""

import ast
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from pypto.ir import IRBuilder
from pypto.ir import op as ir_op
from pypto.ir.printer import python_print
from pypto.pypto_core import DataType, ir

from .diagnostics import (
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
    concise_error_message,
)
from .enum_utils import LEVEL_MAP, LOOP_ORIGIN_MAP, ROLE_MAP, SPLIT_MODE_MAP, extract_enum_value
from .expr_evaluator import ExprEvaluator
from .scope_manager import ScopeManager
from .span_tracker import SpanTracker
from .type_resolver import (
    TypeResolver,
    _const_int_value,  # noqa: PLC2701
    _implicit_tile_view_defaults,  # noqa: PLC2701
)

if TYPE_CHECKING:
    from .decorator import InlineFunction


def _is_const_int(value: object) -> bool:
    """Check if a value is a compile-time constant integer.

    Handles plain int, ir.ConstInt, and ir.Neg(ir.ConstInt) (negative literals).
    """
    if isinstance(value, (int, ir.ConstInt)):
        return True
    return isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt)


def _fold_const_slice_extent(upper: object, lower: object) -> int | None:
    """Fold a slice extent when both bounds are compile-time constants."""
    upper_value = _const_int_value(upper)
    lower_value = _const_int_value(lower)
    if upper_value is None or lower_value is None:
        return None
    return upper_value - lower_value


def _shape_exprs_match(lhs: Sequence[ir.Expr], rhs: Sequence[ir.Expr]) -> bool:
    """Return whether two shape-like expression lists are statically identical."""
    if len(lhs) != len(rhs):
        return False
    for lhs_dim, rhs_dim in zip(lhs, rhs):
        if lhs_dim is rhs_dim:
            continue
        lhs_value = _const_int_value(lhs_dim)
        rhs_value = _const_int_value(rhs_dim)
        if lhs_value is None or rhs_value is None or lhs_value != rhs_value:
            return False
    return True


def _has_printable_tile_view(
    tile_view: ir.TileView | None,
    shape: Sequence[ir.Expr],
    memory_space: ir.MemorySpace | None = None,
) -> bool:
    """Return whether a TileView carries non-default metadata in Python text form."""
    if tile_view is None:
        return False
    if tile_view.valid_shape and not _shape_exprs_match(tile_view.valid_shape, shape):
        return True
    if tile_view.stride:
        return True
    if tile_view.start_offset is not None:
        return True
    default_blayout, default_slayout, default_fractal = _implicit_tile_view_defaults(shape, memory_space)
    if tile_view.blayout != default_blayout:
        return True
    if tile_view.slayout != default_slayout:
        return True
    if tile_view.fractal != default_fractal:
        return True
    if tile_view.pad != ir.PadValue.null:
        return True
    return False


def _normalize_type_for_syntax_match(type_: ir.Type | None) -> ir.Type | None:
    """Normalize omitted default TileView metadata before syntax-level comparison."""
    if isinstance(type_, ir.TileType):
        tile_view = type_.tile_view
        if tile_view is not None and not _has_printable_tile_view(tile_view, type_.shape, type_.memory_space):
            tile_view = None
        if tile_view is type_.tile_view:
            return type_
        return ir.TileType(type_.shape, type_.dtype, type_.memref, tile_view, type_.memory_space)

    return type_


def _types_match(lhs: ir.Type | None, rhs: ir.Type | None) -> bool:
    """Return whether two IR types are structurally equal after TileView normalization.

    Uses C++ ``structural_equal`` (via ``==``) instead of comparing printed
    strings.  Note: ``structural_equal`` intentionally skips the ``memref``
    field on TileType/TensorType — memref identity is not relevant for
    parser-level type matching because the Var (not the Call) carries the
    authoritative memref via ``override_type``.
    """
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return lhs is rhs
    lhs = _normalize_type_for_syntax_match(lhs)
    rhs = _normalize_type_for_syntax_match(rhs)
    assert lhs is not None
    assert rhs is not None
    if type(lhs) is not type(rhs):
        return False
    return lhs == rhs


def _normalize_inferred_type_for_annotation(
    annotation_type: ir.Type,
    value_expr: ir.Expr,
) -> ir.Type:
    """Normalize inferred type before comparing it with an explicit annotation.

    For some printer-emitted forms, the explicit annotation carries result-type
    information that the RHS syntax does not fully encode.  A key example is
    ``FlattenTileNdTo2D`` output:

    - LHS annotation prints the flattened 2D tile shape
    - RHS ``pl.tile.load(...)`` / ``pl.tile.create(...)`` still carries the
      original ND tensor-region operands

    In that case the annotation is the only place where the flattened result
    shape is visible in Python syntax, so roundtrip parsing must trust it.
    """
    inferred_type = _normalize_type_for_syntax_match(value_expr.type)
    assert inferred_type is not None
    if not (
        isinstance(annotation_type, ir.TileType)
        and isinstance(inferred_type, ir.TileType)
        and isinstance(value_expr, ir.Call)
        and value_expr.op.name in {"tile.load", "tile.create"}
        and len(annotation_type.shape) <= 2
        and len(inferred_type.shape) > 2
    ):
        return inferred_type

    return ir.TileType(
        annotation_type.shape,
        inferred_type.dtype,
        inferred_type.memref,
        inferred_type.tile_view,
        inferred_type.memory_space,
    )


class ASTParser:
    """Parses Python AST and builds IR using IRBuilder."""

    def __init__(
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
        )
        self.builder = IRBuilder()
        self.global_vars = global_vars or {}  # Track GlobalVars for cross-function calls
        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference
        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

        # Track loop kinds for break/continue validation
        self._loop_kind_stack: list[str] = []
        self._scope_kind_stack: list[ir.ScopeKind] = []

        # Inline function expansion state
        self._inline_mode = False
        self._inline_return_expr: ir.Expr | None = None

        # Yield tracking state — None means tracking is inactive (outside loops/ifs)
        self._current_yield_vars: list[str] | None = None
        self._current_yield_types: dict[str, ir.Type] | None = None

        # Buffer metadata registry for reserve_buffer().base attribute access
        self._buffer_meta: dict[str, dict[str, Any]] = {}
        # Secondary index:
        # (func_name, buffer_name) → metadata (for cross-function import_peer_buffer resolution)
        # Shared across parser instances when parsing a @pl.program with multiple functions.
        self._buffer_name_meta: dict[tuple[str, str], dict[str, Any]] = (
            buffer_name_meta if buffer_name_meta is not None else {}
        )
        # Current function name (set during parse_function)
        self._func_name: str = ""

    @contextmanager
    def _yield_tracking_scope(self) -> Iterator[None]:
        """Save and restore yield tracking state around a block.

        Initializes fresh _current_yield_vars and _current_yield_types for the
        block, then restores the previous values (including None) when the block
        exits. None means yield tracking is inactive.
        """
        prev_vars = self._current_yield_vars
        prev_types = self._current_yield_types
        self._current_yield_vars = []
        self._current_yield_types = {}
        try:
            yield
        finally:
            self._current_yield_vars = prev_vars
            self._current_yield_types = prev_types

    def _track_yield_var(self, var_name: str, yield_exprs: list[ir.Expr]) -> None:
        """Track a yielded variable name and infer its type.

        Called after emitting a YieldStmt to record the variable name
        for if/for/while output registration and capture the yield
        expression type for unannotated yield inference.

        Args:
            var_name: The variable name being assigned from the yield
            yield_exprs: The list of yielded IR expressions
        """
        if self._current_yield_vars is not None:
            self._current_yield_vars.append(var_name)
        if self._current_yield_types is not None:
            if len(yield_exprs) == 1:
                self._current_yield_types.setdefault(var_name, yield_exprs[0].type)

    @contextmanager
    def _scope_kind_context(self, scope_kind: ir.ScopeKind) -> Iterator[None]:
        """Track scope nesting during parsing for context-sensitive validation."""
        self._scope_kind_stack.append(scope_kind)
        try:
            yield
        finally:
            self._scope_kind_stack.pop()

    def _is_inside_scope(self, scope_kind: ir.ScopeKind) -> bool:
        """Return whether parsing is currently nested inside the given scope kind."""
        return scope_kind in self._scope_kind_stack

    def parse_function(
        self,
        func_def: ast.FunctionDef,
        func_type: ir.FunctionType = ir.FunctionType.Opaque,
        func_level: ir.Level | None = None,
        func_role: ir.Role | None = None,
        func_attrs: dict[str, Any] | None = None,
    ) -> ir.Function:
        """Parse function definition and build IR.

        Args:
            func_def: AST FunctionDef node
            func_type: Function type (default: Opaque)
            func_level: Hierarchy level (default: None)
            func_role: Function role (default: None)
            func_attrs: Function-level attributes dict (default: None)

        Returns:
            IR Function object
        """
        func_name = func_def.name
        self._func_name = func_name
        func_span = self.span_tracker.get_span(func_def)

        # Enter function scope
        self.scope_manager.enter_scope("function")

        # Begin building function
        with self.builder.function(
            func_name, func_span, type=func_type, level=func_level, role=func_role, attrs=func_attrs
        ) as f:
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

            # Parse function body (skip docstrings)
            for i, stmt in enumerate(func_def.body):
                # Skip docstrings (string constants as first statement or after decorators)
                if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        continue  # Skip docstring
                self.parse_statement(stmt)

        # Exit function scope
        self.scope_manager.exit_scope()

        return f.get_result()

    def parse_statement(self, stmt: ast.stmt) -> None:
        """Parse a statement node.

        Args:
            stmt: AST statement node
        """
        if isinstance(stmt, ast.AnnAssign):
            self.parse_annotated_assignment(stmt)
        elif isinstance(stmt, ast.Assign):
            self.parse_assignment(stmt)
        elif isinstance(stmt, ast.For):
            self.parse_for_loop(stmt)
        elif isinstance(stmt, ast.While):
            self.parse_while_loop(stmt)
        elif isinstance(stmt, ast.If):
            self.parse_if_statement(stmt)
        elif isinstance(stmt, ast.With):
            self.parse_with_statement(stmt)
        elif isinstance(stmt, ast.Return):
            self.parse_return(stmt)
        elif isinstance(stmt, ast.Expr):
            self.parse_evaluation_statement(stmt)
        elif isinstance(stmt, ast.Break):
            self.parse_break(stmt)
        elif isinstance(stmt, ast.Continue):
            self.parse_continue(stmt)
        elif isinstance(stmt, ast.Pass):
            pass  # No-op: pass statements produce no IR
        else:
            raise UnsupportedFeatureError(
                f"Unsupported statement type: {type(stmt).__name__}",
                span=self.span_tracker.get_span(stmt),
                hint="Only assignments, for loops, while loops, if statements, "
                "with statements, returns, break, continue, and pass are supported in DSL functions",
            )

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
                yield_exprs = []
                for arg in stmt.value.args:
                    expr = self.parse_expression(arg)
                    yield_exprs.append(expr)

                # Emit yield statement
                yield_span = self.span_tracker.get_span(stmt.value)
                self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))
                self._track_yield_var(var_name, yield_exprs)

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
            and self._is_printed_alloc_call(stmt.value)
        ):
            value_expr = self._parse_printed_alloc_call(stmt.value)
            ptr_var = ir.Var(var_name, ir.PtrType(), span)
            self.builder.emit(ir.AssignStmt(ptr_var, value_expr, span))
            self.scope_manager.define_var(var_name, ptr_var, span=span)
            return

        value_expr = self.parse_expression(stmt.value)

        # Validate annotation against inferred type; use annotation as override only for memref
        override_type = None
        if stmt.annotation is not None:
            # Skip annotations the resolver can't handle:
            # - String forward refs (e.g. "SomeType")
            # - pl.UnknownType (emitted by printer for unrepresentable types)
            ann = stmt.annotation
            is_unresolvable = (isinstance(ann, ast.Constant) and isinstance(ann.value, str)) or (
                isinstance(ann, ast.Attribute)
                and isinstance(ann.value, ast.Name)
                and ann.value.id == "pl"
                and ann.attr in ("UnknownType", "MemRefType", "Ptr")
            )
            if is_unresolvable:
                resolved = None
            else:
                resolved = self.type_resolver.resolve_type(ann)
            if resolved is not None and not isinstance(resolved, list):
                inferred_for_validation = _normalize_inferred_type_for_annotation(resolved, value_expr)
                self.type_resolver.validate_annotation_consistency(
                    resolved, inferred_for_validation, var_name, span
                )
                if isinstance(value_expr.type, ir.UnknownType):
                    # Inferred type is unknown (e.g. tpop_from_aiv): use annotation as type
                    override_type = resolved
                elif isinstance(resolved, ir.TileType) and isinstance(value_expr.type, ir.TileType):
                    normalized_inferred = _normalize_inferred_type_for_annotation(resolved, value_expr)
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
                    if (
                        resolved.memref is not None
                        or merged_ms is not None
                        or merged_tv is not None
                        or not _shape_exprs_match(resolved.shape, normalized_inferred.shape)
                    ):
                        override_type = ir.TileType(
                            resolved.shape, resolved.dtype, resolved.memref, merged_tv, merged_ms
                        )
                elif isinstance(resolved, ir.ShapedType) and resolved.memref is not None:
                    override_type = resolved
                elif isinstance(resolved, ir.TensorType) and resolved.tensor_view is not None:
                    # Annotation specifies tensor view (stride/layout); preserve it
                    override_type = resolved
                elif (
                    isinstance(resolved, ir.ScalarType)
                    and isinstance(value_expr.type, ir.ScalarType)
                    and value_expr.type.dtype == DataType.INDEX
                ):
                    override_type = resolved
        # If annotation syntax determines the result type more precisely than the
        # raw call inference, rebuild the Call with that type so structural
        # equality sees the same IR after print→parse.
        if (
            override_type is not None
            and isinstance(value_expr, ir.Call)
            and (
                (
                    isinstance(override_type, ir.TileType)
                    and isinstance(value_expr.type, ir.UnknownType)
                    and value_expr.op.name in ("tile.tpop_from_aiv", "tile.tpop_from_aic")
                )
                or not _types_match(value_expr.type, override_type)
            )
        ):
            value_expr = ir.Call(
                value_expr.op, value_expr.args, value_expr.kwargs, override_type, value_expr.span
            )

        # Reuse existing Var on reassignment (override_type is intentionally
        # discarded — the Var's type was fixed at first definition; the SSA
        # pass will create properly typed versioned copies later).
        var = self._assign_or_let(var_name, value_expr, span, override_type)

        # Register in scope
        self.scope_manager.define_var(var_name, var, span=span)

        # Track buffer metadata for attribute access (e.g., pipe_buf.base)
        if isinstance(stmt.value, ast.Call):
            self._track_buffer_meta(var_name, stmt.value)

    def _assign_or_let(
        self,
        var_name: str,
        value_expr: ir.Expr,
        span: ir.Span,
        override_type: ir.Type | None = None,
    ) -> ir.Var:
        """Assign to existing Var if possible, otherwise create a new let binding."""
        existing_var = self.scope_manager.lookup_var(var_name)
        if existing_var is not None and type(existing_var) is ir.Var and not self.scope_manager.strict_ssa:
            # Reject reassignment with a different type (#642).  Same Python
            # variable maps to the same Var node, so the type must match.
            value_type = override_type or value_expr.type
            if (
                not isinstance(value_type, ir.UnknownType)
                and not isinstance(existing_var.type, ir.UnknownType)
                and not _types_match(existing_var.type, value_type)
            ):
                raise ParserTypeError(
                    f"Cannot reassign '{var_name}' with a different type: "
                    f"was {ir.python_print_type(existing_var.type)}, "
                    f"got {ir.python_print_type(value_type)}",
                    span=span,
                    hint="Use a different variable name for tensors with different shapes or dtypes",
                )
            self.builder.assign(existing_var, value_expr, span=span)
            return existing_var
        return self.builder.let(var_name, value_expr, type=override_type, span=span)

    def parse_assignment(self, stmt: ast.Assign) -> None:
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
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
                        self.parse_yield_assignment(target, stmt.value)
                        return

                # General tuple unpacking for function calls returning TupleType
                span = self.span_tracker.get_span(stmt)
                value_expr = self.parse_expression(stmt.value)

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

                # Check if this is a yield assignment: var = pl.yield_(...)
                if isinstance(stmt.value, ast.Call):
                    func = stmt.value.func
                    if isinstance(func, ast.Attribute) and func.attr == "yield_":
                        # Handle yield assignment
                        yield_exprs = []
                        for arg in stmt.value.args:
                            expr = self.parse_expression(arg)
                            if not isinstance(expr, ir.Expr):
                                raise ParserSyntaxError(
                                    f"Yield argument must be an IR expression, got {type(expr)}",
                                    span=self.span_tracker.get_span(arg),
                                    hint="Ensure yield arguments are valid expressions",
                                )
                            yield_exprs.append(expr)

                        # Emit yield statement
                        yield_span = self.span_tracker.get_span(stmt.value)
                        self.builder.emit(ir.YieldStmt(yield_exprs, yield_span))
                        self._track_yield_var(var_name, yield_exprs)

                        # Don't register in scope yet - will be done when loop/if completes
                        return

                value_expr = self.parse_expression(stmt.value)
                var = self._assign_or_let(var_name, value_expr, span)
                self.scope_manager.define_var(var_name, var, span=span)

                # Track buffer metadata for attribute access (e.g., pipe_buf.base)
                if isinstance(stmt.value, ast.Call):
                    self._track_buffer_meta(var_name, stmt.value)

                return

        raise ParserSyntaxError(
            f"Unsupported assignment: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
        )

    def parse_yield_assignment(self, target: ast.Tuple, value: ast.Call) -> None:
        """Parse yield assignment: (a, b) = pl.yield_(x, y).

        Args:
            target: Tuple of target variable names
            value: Call to pl.yield_()
        """
        # Parse yield expressions
        yield_exprs = []
        for arg in value.args:
            expr = self.parse_expression(arg)
            # Ensure it's an IR Expr
            if not isinstance(expr, ir.Expr):
                raise ParserSyntaxError(
                    f"Yield argument must be an IR expression, got {type(expr)}",
                    span=self.span_tracker.get_span(arg),
                    hint="Ensure yield arguments are valid expressions",
                )
            yield_exprs.append(expr)

        # Emit yield statement
        span = self.span_tracker.get_span(value)
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Track yielded variable names and types for if/for statement processing
        for i, elt in enumerate(target.elts):
            if isinstance(elt, ast.Name) and i < len(yield_exprs):
                self._track_yield_var(elt.id, [yield_exprs[i]])

        # For tuple yields at the for/while loop level, register the variables
        # (they'll be available as loop.get_result().return_vars)
        if (self.in_for_loop or self.in_while_loop) and not self.in_if_stmt:
            # Register yielded variable names in scope
            for i, elt in enumerate(target.elts):
                if isinstance(elt, ast.Name):
                    var_name = elt.id
                    # Will be resolved from loop outputs
                    self.scope_manager.define_var(var_name, f"loop_yield_{i}")

    _VALID_ITERATORS = {"range", "parallel", "unroll", "while_"}
    _ITERATOR_ERROR = "For loop must use pl.range(), pl.parallel(), pl.unroll(), or pl.while_()"
    _ITERATOR_HINT = "Use pl.range(), pl.parallel(), pl.unroll(), or pl.while_() as the iterator"

    def _validate_for_loop_iterator(self, stmt: ast.For) -> tuple[ast.Call, str]:
        """Validate that for loop uses pl.range(), pl.parallel(), pl.unroll(), or pl.while_().

        Returns:
            Tuple of (call_node, iterator_type) where iterator_type is
            "range", "parallel", "unroll", or "while_"
        """
        if not isinstance(stmt.iter, ast.Call):
            raise ParserSyntaxError(
                self._ITERATOR_ERROR,
                span=self.span_tracker.get_span(stmt.iter),
                hint=self._ITERATOR_HINT,
            )

        iter_call = stmt.iter
        func = iter_call.func
        if isinstance(func, ast.Attribute) and func.attr in self._VALID_ITERATORS:
            return iter_call, func.attr

        raise ParserSyntaxError(
            self._ITERATOR_ERROR,
            span=self.span_tracker.get_span(stmt.iter),
            hint=self._ITERATOR_HINT,
        )

    def _parse_for_loop_target(self, stmt: ast.For) -> tuple[str, ast.AST | None, bool]:
        """Parse for loop target, returning (loop_var_name, iter_args_node, is_simple_for)."""
        if isinstance(stmt.target, ast.Name):
            return stmt.target.id, None, True

        if isinstance(stmt.target, ast.Tuple) and len(stmt.target.elts) == 2:
            loop_var_node = stmt.target.elts[0]
            iter_args_node = stmt.target.elts[1]

            if not isinstance(loop_var_node, ast.Name):
                raise ParserSyntaxError(
                    "Loop variable must be a simple name",
                    span=self.span_tracker.get_span(loop_var_node),
                    hint="Use a simple variable name for the loop counter",
                )
            return loop_var_node.id, iter_args_node, False

        raise ParserSyntaxError(
            "For loop target must be a simple name or: (loop_var, (iter_args...))",
            span=self.span_tracker.get_span(stmt.target),
            hint="Use: for i in pl.range(n) or for i, (var1,) in pl.range(n, init_values=(...,))",
        )

    def _setup_iter_args(self, loop: Any, iter_args_node: ast.AST, init_values: list) -> None:
        """Set up iter_args and return_vars for Pattern A loops."""
        if not isinstance(iter_args_node, ast.Tuple):
            raise ParserSyntaxError(
                "Iter args must be a tuple",
                span=self.span_tracker.get_span(iter_args_node),
                hint="Wrap iteration variables in parentheses: (var1, var2)",
            )

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def parse_for_loop(self, stmt: ast.For) -> None:  # noqa: PLR0912
        """Parse for loop with pl.range(), pl.parallel(), pl.unroll(), or pl.while_().

        Supports patterns for range/parallel/unroll:
          Pattern A (explicit): for i, (vars,) in pl.range(..., init_values=(...,))
          Pattern B (simple):   for i in pl.range(n)

        Supports pattern for while-as-for:
          for (vars,) in pl.while_(init_values=(...,)):
              pl.cond(condition)

        Both patterns also work with pl.parallel() for parallel loops.
        pl.unroll() is for compile-time loop unrolling (no init_values).
        Pattern B produces a ForStmt without iter_args/return_vars/yield.
        The C++ ConvertToSSA pass handles converting to SSA form.
        """
        iter_call, iterator_type = self._validate_for_loop_iterator(stmt)

        # Handle pl.while_() case
        if iterator_type == "while_":
            self._parse_while_as_for(stmt, iter_call)
            return

        loop_var_name, iter_args_node, is_simple_for = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call)

        if is_simple_for and range_args["init_values"]:
            raise ParserSyntaxError(
                "For loop target must be a tuple when init_values is provided",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i, (var1,) in pl.range(n, init_values=(val1,)) to include iter_args",
            )

        # For pl.unroll(), require compile-time constant integer bounds
        # and reject step=0. Fail early with clear parser errors instead of
        # later generic failures in the UnrollLoops C++ pass.
        # Note: negative literals like -1 become ir.Neg(ir.ConstInt(1)).
        if iterator_type == "unroll":
            if range_args["init_values"]:
                raise ParserSyntaxError(
                    "pl.unroll() cannot be combined with init_values",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use pl.range() to carry state across iterations.",
                )
            for _bound_name in ("start", "stop", "step"):
                _bound_value = range_args.get(_bound_name)
                if _bound_value is not None and not _is_const_int(_bound_value):
                    raise ParserSyntaxError(
                        "pl.unroll() requires compile-time constant integer bounds",
                        span=self.span_tracker.get_span(iter_call),
                        hint="Use integer literals for start, stop, and step in pl.unroll().",
                    )
            _step = range_args.get("step")
            if _const_int_value(_step) == 0:
                raise ParserSyntaxError(
                    "pl.unroll() step cannot be zero",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use a non-zero step in pl.unroll(start, stop, step).",
                )

        # Validate chunk arguments
        chunk_expr = range_args.get("chunk")
        chunk_policy_str = range_args.get("chunk_policy", "leading_full")
        if chunk_expr is not None:
            self._validate_chunk_args(chunk_expr, range_args["init_values"], iter_call)

        kind = self._ITERATOR_TO_KIND[iterator_type]
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
        loop_var = self.builder.var(loop_var_name, ir.ScalarType(_loop_var_dtype))
        span = self.span_tracker.get_span(stmt)
        loop_output_vars: list[str] = []
        prev_loop_builder = self.current_loop_builder
        prev_in_for_loop = self.in_for_loop
        prev_in_while_loop = self.in_while_loop

        attrs_dict = range_args.get("attrs") or None
        with self.builder.for_loop(
            loop_var,
            range_args["start"],
            range_args["stop"],
            range_args["step"],
            span,
            kind,
            chunk_size=chunk_expr,
            chunk_policy=chunk_policy_str,
            attrs=attrs_dict,
        ) as loop:
            self.current_loop_builder = loop
            self.in_for_loop = True
            self._loop_kind_stack.append(iterator_type)
            self.scope_manager.enter_scope("for")
            self.scope_manager.define_var(loop_var_name, loop_var, allow_redef=True)

            if not is_simple_for:
                assert iter_args_node is not None  # Guaranteed by _parse_for_loop_target
                self._setup_iter_args(loop, iter_args_node, range_args["init_values"])

            with self._yield_tracking_scope():
                for body_stmt in stmt.body:
                    self.parse_statement(body_stmt)
                assert self._current_yield_vars is not None  # Guaranteed by _yield_tracking_scope
                loop_output_vars = self._current_yield_vars[:]

            # Create return_vars using yield LHS names (or fallback to _out names)
            if not is_simple_for and range_args["init_values"]:
                if loop_output_vars:
                    for rv_name in loop_output_vars:
                        loop.return_var(rv_name)
                else:
                    # Fallback: no yield vars found, use auto-generated names
                    assert iter_args_node is not None
                    assert isinstance(iter_args_node, ast.Tuple)
                    for iter_arg_node in iter_args_node.elts:
                        assert isinstance(iter_arg_node, ast.Name)
                        loop.return_var(f"{iter_arg_node.id}_out")

            should_leak = is_simple_for and not loop_output_vars
            self.scope_manager.exit_scope(leak_vars=should_leak)
            self._loop_kind_stack.pop()
            self.in_for_loop = prev_in_for_loop
            self.in_while_loop = prev_in_while_loop
            self.current_loop_builder = prev_loop_builder

        if not is_simple_for:
            loop_result = loop.get_result()
            if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
                for i, var_name in enumerate(loop_output_vars):
                    if i < len(loop_result.return_vars):
                        self.scope_manager.define_var(var_name, loop_result.return_vars[i])

    def _validate_chunk_args(self, chunk_expr: Any, init_values: list[Any], iter_call: ast.Call) -> None:
        """Validate chunk arguments for range/parallel/unroll loops."""
        if not self._is_inside_scope(ir.ScopeKind.AutoInCore):
            raise ParserSyntaxError(
                "chunk=... loops are only valid inside with pl.auto_incore(): or "
                "with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):",
                span=self.span_tracker.get_span(iter_call),
                hint="Wrap the loop in 'with pl.at(level=pl.Level.CORE_GROUP, "
                "optimization=pl.chunked_loop_optimizer):' or remove the chunk= argument.",
            )
        if not _is_const_int(chunk_expr):
            raise ParserSyntaxError(
                "chunk must be a compile-time constant positive integer",
                span=self.span_tracker.get_span(iter_call),
                hint="Use an integer literal for chunk: chunk=5",
            )
        chunk_val = _const_int_value(chunk_expr)
        if chunk_val is not None and chunk_val <= 0:
            raise ParserSyntaxError(
                f"chunk must be a positive integer, got {chunk_val}",
                span=self.span_tracker.get_span(iter_call),
                hint="Use a positive integer for chunk: chunk=5",
            )

    def _parse_range_keyword(self, keyword: ast.keyword, result: dict[str, Any]) -> None:
        """Parse a single keyword argument from a pl.range() call."""
        if keyword.arg == "init_values":
            if isinstance(keyword.value, (ast.List, ast.Tuple)):
                result["init_values"] = [self.parse_expression(elt) for elt in keyword.value.elts]
            else:
                raise ParserSyntaxError(
                    "init_values must be a list or tuple",
                    span=self.span_tracker.get_span(keyword.value),
                    hint="Use a tuple for init_values: init_values=(var1, var2)",
                )
        elif keyword.arg == "chunk":
            result["chunk"] = self.parse_expression(keyword.value)
        elif keyword.arg == "chunk_policy":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                _VALID_CHUNK_POLICIES = {"leading_full"}
                if keyword.value.value not in _VALID_CHUNK_POLICIES:
                    raise ParserSyntaxError(
                        f"Unsupported chunk_policy: {keyword.value.value!r}",
                        span=self.span_tracker.get_span(keyword.value),
                        hint=f"Supported values: {', '.join(sorted(_VALID_CHUNK_POLICIES))}",
                    )
                result["chunk_policy"] = keyword.value.value
            else:
                raise ParserSyntaxError(
                    "chunk_policy must be a string literal",
                    span=self.span_tracker.get_span(keyword.value),
                    hint='Use a string like chunk_policy="leading_full"',
                )
        elif keyword.arg == "attrs":
            if not isinstance(keyword.value, ast.Dict):
                raise ParserSyntaxError(
                    "attrs must be a dict literal",
                    span=self.span_tracker.get_span(keyword.value),
                    hint='Use a dict like attrs={"loop_origin": pl.LoopOrigin.ChunkOuter}',
                )
            result["attrs"] = self._parse_attrs_dict(keyword.value)
        else:
            raise ParserSyntaxError(
                f"Unknown keyword argument '{keyword.arg}' in range()",
                span=self.span_tracker.get_span(keyword),
                hint="Supported keywords: init_values, chunk, chunk_policy, attrs",
            )

    def _parse_range_call(self, call: ast.Call) -> dict[str, Any]:
        """Parse pl.range() call arguments.

        Args:
            call: AST Call node for pl.range()

        Returns:
            Dictionary with start, stop, step, init_values
        """
        if len(call.args) < 1:
            raise ParserSyntaxError(
                "pl.range() requires at least 1 argument (stop)",
                span=self.span_tracker.get_span(call),
                hint="Provide at least the stop value: pl.range(10) or pl.range(0, 10)",
            )

        start = 0
        step = 1

        if len(call.args) == 1:
            stop = self.parse_expression(call.args[0])
        elif len(call.args) == 2:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
        elif len(call.args) >= 3:
            start = self.parse_expression(call.args[0])
            stop = self.parse_expression(call.args[1])
            step = self.parse_expression(call.args[2])

        result: dict[str, Any] = {
            "init_values": [],
            "chunk": None,
            "chunk_policy": "leading_full",
            "attrs": {},
        }
        for keyword in call.keywords:
            self._parse_range_keyword(keyword, result)

        result["start"] = start
        result["stop"] = stop
        result["step"] = step
        return result

    def _parse_attrs_dict(self, node: ast.Dict) -> dict[str, object]:
        """Parse an attrs dict literal from AST.

        Supports string keys with values that are:
        - Integer/float/bool/string constants
        - pl.LoopOrigin.<name> enum references
        """
        # Map of known enum attr keys to their (enum_map, enum_name, qualified) configs
        _ENUM_ATTRS: dict[str, tuple[dict[str, object], str, str]] = {
            "loop_origin": (LOOP_ORIGIN_MAP, "LoopOrigin", "pl.LoopOrigin"),
        }

        result: dict[str, object] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                raise ParserSyntaxError(
                    "attrs keys must be string literals",
                    span=self.span_tracker.get_span(key_node) if key_node else None,
                )
            key = key_node.value

            if key in _ENUM_ATTRS:
                enum_map, enum_name, qualified = _ENUM_ATTRS[key]
                result[key] = extract_enum_value(value_node, enum_map, enum_name, qualified)
            elif isinstance(value_node, ast.Constant):
                result[key] = value_node.value
            else:
                raise ParserSyntaxError(
                    f"Unsupported value type for attrs key '{key}'",
                    span=self.span_tracker.get_span(value_node),
                    hint="Supported values: integer, float, bool, string,"
                    " or enum (e.g., pl.LoopOrigin.ChunkOuter)",
                )
        return result

    def _is_cond_call(self, stmt: ast.stmt) -> bool:
        """Check if statement is a pl.cond() call (without parsing).

        Args:
            stmt: AST statement node

        Returns:
            True if statement is pl.cond() call, False otherwise
        """
        if not isinstance(stmt, ast.Expr):
            return False
        return self._is_dsl_call(stmt, "cond")

    def _extract_cond_call(self, stmt: ast.stmt) -> ir.Expr | None:
        """Extract condition from pl.cond() call statement.

        Args:
            stmt: AST statement node

        Returns:
            Parsed condition expression if statement is pl.cond(), None otherwise
        """
        if not self._is_cond_call(stmt):
            return None

        assert isinstance(stmt, ast.Expr)
        call = stmt.value
        assert isinstance(call, ast.Call)

        # Parse the condition argument
        if len(call.args) != 1:
            raise ParserSyntaxError(
                "pl.cond() requires exactly 1 argument",
                span=self.span_tracker.get_span(call),
                hint="Use: pl.cond(condition)",
            )

        return self.parse_expression(call.args[0])

    @staticmethod
    def _is_dsl_call(stmt: ast.Expr, func_name: str) -> bool:
        """Check if statement is a call to a named DSL function (e.g. pl.func_name() or func_name()).

        Args:
            stmt: AST Expr node
            func_name: The function name to match (e.g. "static_print")

        Returns:
            True if statement is a call to the named function
        """
        call = stmt.value
        if not isinstance(call, ast.Call):
            return False
        func = call.func
        if isinstance(func, ast.Attribute):
            return func.attr == func_name
        if isinstance(func, ast.Name):
            return func.id == func_name
        return False

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

    def _validate_while_call_args(self, while_call: ast.Call) -> None:
        """Validate that pl.while_() has no positional arguments."""
        if len(while_call.args) > 0:
            raise ParserSyntaxError(
                "pl.while_() takes no positional arguments",
                span=self.span_tracker.get_span(while_call),
                hint="Use: pl.while_(init_values=(...,)) with pl.cond(condition) as first statement in body",
            )

    def _parse_while_init_values(self, while_call: ast.Call) -> list[ir.Expr]:
        """Parse init_values from pl.while_() keyword arguments."""
        init_values = []
        for keyword in while_call.keywords:
            if keyword.arg == "init_values":
                if isinstance(keyword.value, (ast.List, ast.Tuple)):
                    for elt in keyword.value.elts:
                        init_values.append(self.parse_expression(elt))
                else:
                    raise ParserSyntaxError(
                        "init_values must be a tuple or list",
                        span=self.span_tracker.get_span(keyword.value),
                        hint="Use a tuple for init_values (lists also accepted): init_values=(var1, var2)",
                    )

        if not init_values:
            raise ParserSyntaxError(
                "pl.while_() requires init_values",
                span=self.span_tracker.get_span(while_call),
                hint="Provide init_values: pl.while_(init_values=(val1, val2))",
            )

        return init_values

    def _validate_while_body(self, stmt: ast.For) -> None:
        """Validate pl.while_() body structure."""
        if not stmt.body:
            raise ParserSyntaxError(
                "pl.while_() body cannot be empty",
                span=self.span_tracker.get_span(stmt),
                hint="Add pl.cond(condition) as first statement",
            )

        if not self._is_cond_call(stmt.body[0]):
            raise ParserSyntaxError(
                "First statement in pl.while_() body must be pl.cond(condition)",
                span=self.span_tracker.get_span(stmt.body[0]),
                hint="Add pl.cond(condition) as first statement",
            )

    def _validate_while_target(self, stmt: ast.For, init_values: list[ir.Expr]) -> ast.Tuple:
        """Validate and return pl.while_() target tuple."""
        if not isinstance(stmt.target, ast.Tuple):
            raise ParserSyntaxError(
                "While loop target must be a tuple for pl.while_()",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for (var1, var2) in pl.while_(init_values=(...,))",
            )

        iter_args_node = stmt.target

        if len(iter_args_node.elts) != len(init_values):
            raise ParserSyntaxError(
                f"Mismatch: {len(iter_args_node.elts)} iter_args but {len(init_values)} init_values",
                span=self.span_tracker.get_span(iter_args_node),
                hint=f"Provide exactly {len(init_values)} iteration variable(s) to match init_values",
            )

        return iter_args_node

    def _setup_while_iter_args(
        self, loop: Any, iter_args_node: ast.Tuple, init_values: list[ir.Expr]
    ) -> None:
        """Set up iter_args for pl.while_() loop."""
        for i, iter_arg_node in enumerate(iter_args_node.elts):
            if not isinstance(iter_arg_node, ast.Name):
                raise ParserSyntaxError(
                    "Iter arg must be a simple name",
                    span=self.span_tracker.get_span(iter_arg_node),
                    hint="Use simple variable names for iteration variables",
                )
            iter_arg_var = loop.iter_arg(iter_arg_node.id, init_values[i])
            self.scope_manager.define_var(iter_arg_node.id, iter_arg_var, allow_redef=True)

    def _parse_while_body_statements(self, stmt: ast.For) -> list[str]:
        """Parse body statements for pl.while_() loop, return yielded vars."""
        with self._yield_tracking_scope():
            # Parse body (skip first statement which is pl.cond())
            for body_stmt in stmt.body[1:]:
                # Check if pl.cond() appears anywhere else in body
                if self._is_cond_call(body_stmt):
                    raise ParserSyntaxError(
                        "pl.cond() can only be the first statement in a pl.while_() loop body",
                        span=self.span_tracker.get_span(body_stmt),
                        hint="Remove this pl.cond() - condition is already specified at the start",
                    )
                self.parse_statement(body_stmt)

            assert self._current_yield_vars is not None  # Guaranteed by _yield_tracking_scope
            return self._current_yield_vars[:]

    def _register_while_outputs(
        self, loop: Any, iter_args_node: ast.Tuple, loop_output_vars: list[str]
    ) -> None:
        """Register output variables from pl.while_() loop."""
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
            for i, iter_arg_node in enumerate(iter_args_node.elts):
                if i >= len(loop_result.return_vars):
                    break
                if isinstance(iter_arg_node, ast.Name):
                    self.scope_manager.define_var(
                        iter_arg_node.id, loop_result.return_vars[i], allow_redef=True
                    )

            for i, var_name in enumerate(loop_output_vars):
                if i < len(loop_result.return_vars):
                    self.scope_manager.define_var(var_name, loop_result.return_vars[i], allow_redef=True)

    def _parse_while_as_for(self, stmt: ast.For, while_call: ast.Call) -> None:
        """Parse while loop using for...in pl.while_() pattern.

        Pattern: for (var1, var2) in pl.while_(init_values=(val1, val2)):
                     pl.cond(condition)
                     ...

        Args:
            stmt: For AST node
            while_call: Call to pl.while_()
        """
        # Validate and parse arguments
        self._validate_while_call_args(while_call)
        init_values = self._parse_while_init_values(while_call)
        self._validate_while_body(stmt)
        iter_args_node = self._validate_while_target(stmt, init_values)

        span = self.span_tracker.get_span(stmt)
        placeholder_condition = ir.ConstBool(True, span)
        prev_loop_builder = self.current_loop_builder
        prev_in_for_loop = self.in_for_loop
        prev_in_while_loop = self.in_while_loop

        with self.builder.while_loop(placeholder_condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self._loop_kind_stack.append("while")
            self.scope_manager.enter_scope("while")

            # Set up iter_args
            self._setup_while_iter_args(loop, iter_args_node, init_values)

            # Parse and set the condition (now that iter_args are in scope)
            condition = self._extract_cond_call(stmt.body[0])
            if condition is None:
                raise ParserSyntaxError(
                    "First statement in pl.while_() body must be pl.cond(condition)",
                    span=self.span_tracker.get_span(stmt.body[0]),
                    hint="Add pl.cond(condition) as first statement",
                )
            loop.set_condition(condition)

            # Parse body statements first to get actual output variable names
            loop_output_vars = self._parse_while_body_statements(stmt)

            # Add return_vars using actual output variable names from body
            if not loop_output_vars:
                raise ParserSyntaxError(
                    "pl.while_() with init_values requires a pl.yield_(...) in the body",
                    span=self.span_tracker.get_span(stmt),
                    hint="Yield the updated loop-carried values before the end of the body",
                )
            for var_name in loop_output_vars:
                loop.return_var(var_name)

            self.scope_manager.exit_scope(leak_vars=False)
            self._loop_kind_stack.pop()
            self.in_for_loop = prev_in_for_loop
            self.in_while_loop = prev_in_while_loop
            self.current_loop_builder = prev_loop_builder

        # Register output variables
        self._register_while_outputs(loop, iter_args_node, loop_output_vars)

    def parse_while_loop(self, stmt: ast.While) -> None:
        """Parse natural while loop syntax.

        Natural while syntax: while condition: body

        This creates a WhileStmt without iter_args (non-SSA form).
        The C++ ConvertToSSA pass will convert it to SSA form if needed.

        Args:
            stmt: While AST node
        """
        # Parse natural while syntax: while condition:
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)
        prev_loop_builder = self.current_loop_builder
        prev_in_for_loop = self.in_for_loop
        prev_in_while_loop = self.in_while_loop

        with self.builder.while_loop(condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self._loop_kind_stack.append("while")
            self.scope_manager.enter_scope("while")

            # Parse body statements
            for body_stmt in stmt.body:
                self.parse_statement(body_stmt)

            # Variables leak to outer scope (ConvertToSSA will handle)
            self.scope_manager.exit_scope(leak_vars=True)
            self._loop_kind_stack.pop()
            self.in_for_loop = prev_in_for_loop
            self.in_while_loop = prev_in_while_loop
            self.current_loop_builder = prev_loop_builder

    def parse_if_statement(self, stmt: ast.If) -> None:
        """Parse if statement with phi nodes.

        When pl.yield_() is used, phi nodes are created via return_vars.
        When no yields are used (plain syntax), variables leak to outer scope
        and the C++ ConvertToSSA pass handles creating phi nodes.

        Args:
            stmt: If AST node
        """
        # Parse condition
        condition = self.parse_expression(stmt.test)
        span = self.span_tracker.get_span(stmt)

        # Track yield output variable names from both branches
        then_yield_vars = []

        # Begin if statement
        with self.builder.if_stmt(condition, span) as if_builder:
            self.current_if_builder = if_builder
            self.in_if_stmt = True

            with self._yield_tracking_scope():
                # Scan for yield variable names (without executing)
                then_yield_vars = self._scan_for_yields(stmt.body)

                # Also scan else branch to handle yields in both branches
                if stmt.orelse:
                    else_yield_vars = self._scan_for_yields(stmt.orelse)
                    # Merge with then branch yields (then branch takes precedence for type)
                    then_names = {name for name, _ in then_yield_vars}
                    for name, annotation in else_yield_vars:
                        if name not in then_names:
                            then_yield_vars.append((name, annotation))

                # Determine if we should leak variables (no explicit yields)
                should_leak = not bool(then_yield_vars)

                # Parse then branch (yield types captured via _current_yield_types)
                self.scope_manager.enter_scope("if")
                for then_stmt in stmt.body:
                    self.parse_statement(then_stmt)
                self.scope_manager.exit_scope(leak_vars=should_leak)

                # Parse else branch if present
                if stmt.orelse:
                    if_builder.else_()
                    self.scope_manager.enter_scope("else")
                    for else_stmt in stmt.orelse:
                        self.parse_statement(else_stmt)
                    self.scope_manager.exit_scope(leak_vars=should_leak)

                # Declare return vars AFTER parsing branches so captured yield types
                # are available for unannotated yields (fixes issue #233 / #234)
                for var_name, annotation in then_yield_vars:
                    if annotation is not None:
                        var_type = self._resolve_yield_var_type(annotation)
                    elif self._current_yield_types is not None and var_name in self._current_yield_types:
                        var_type = self._current_yield_types[var_name]
                    else:
                        var_type = self._resolve_yield_var_type(None)
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
                        self.scope_manager.define_var(var_name, output_var)

        self.in_if_stmt = False
        self.current_if_builder = None

    def _parse_at_kwargs(self, call: ast.Call) -> tuple[ir.Level, ir.Role | None, ir.SplitMode | None]:
        """Extract level, role, and optimization from pl.at(...) call.

        Supports both positional and keyword forms:
        - pl.at(pl.Level.HOST)
        - pl.at(pl.Level.HOST, pl.Role.Worker)
        - pl.at(level=pl.Level.HOST, role=pl.Role.Worker)
        - pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer)
        - pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=...))

        Args:
            call: AST Call node for pl.at(...)

        Returns:
            Tuple of (level, role, opt_split) where opt_split is None when no
            optimization= kwarg is present, or a SplitMode when
            chunked_loop_optimizer is used.
        """
        if len(call.args) > 2:
            raise ParserSyntaxError(
                f"pl.at() takes at most 2 positional arguments, got {len(call.args)}",
                hint="Use pl.at(level) or pl.at(level, role)",
            )

        level = None
        role = None
        opt_split: ir.SplitMode | None = None

        # Parse positional arguments
        if len(call.args) >= 1:
            level = extract_enum_value(call.args[0], LEVEL_MAP, "Level", "pl.Level")
        if len(call.args) >= 2:
            role = extract_enum_value(call.args[1], ROLE_MAP, "Role", "pl.Role")

        # Parse keyword arguments
        for kw in call.keywords:
            if kw.arg == "level":
                if level is not None:
                    raise ParserSyntaxError(
                        "pl.at() got multiple values for argument 'level'",
                    )
                level = extract_enum_value(kw.value, LEVEL_MAP, "Level", "pl.Level")
            elif kw.arg == "role":
                if role is not None:
                    raise ParserSyntaxError(
                        "pl.at() got multiple values for argument 'role'",
                    )
                role = extract_enum_value(kw.value, ROLE_MAP, "Role", "pl.Role")
            elif kw.arg == "optimization":
                if opt_split is not None:
                    raise ParserSyntaxError(
                        "pl.at() got multiple values for argument 'optimization'",
                        span=self.span_tracker.get_span(kw),
                    )
                opt_split = self._parse_chunked_loop_optimizer(kw.value)
            elif kw.arg is None:
                raise ParserSyntaxError(
                    "Unsupported **kwargs in pl.at()",
                    hint="Use pl.at(level=pl.Level.HOST, role=pl.Role.Worker)",
                )
            else:
                raise ParserSyntaxError(
                    f"Unknown keyword argument '{kw.arg}' in pl.at()",
                    hint="Supported arguments: level, role, optimization",
                )
        if level is None:
            raise ParserSyntaxError(
                "pl.at() requires a level argument",
                hint="Use pl.at(pl.Level.HOST) or pl.at(level=pl.Level.HOST)",
            )
        return level, role, opt_split

    def _parse_chunked_loop_optimizer(self, value: ast.expr) -> "ir.SplitMode":
        """Parse pl.chunked_loop_optimizer or pl.chunked_loop_optimizer(split=...) AST node.

        Returns the split mode to use for the AutoInCore scope.
        """
        # Bare: pl.chunked_loop_optimizer
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "chunked_loop_optimizer"
            and isinstance(value.value, ast.Name)
            and value.value.id == "pl"
        ):
            return ir.SplitMode.UP_DOWN

        # Called: pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "chunked_loop_optimizer"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "pl"
        ):
            if value.args:
                raise ParserSyntaxError(
                    "pl.chunked_loop_optimizer() does not accept positional arguments",
                    span=self.span_tracker.get_span(value),
                    hint="Use: pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)",
                )
            split = ir.SplitMode.UP_DOWN
            for opt_kw in value.keywords:
                if opt_kw.arg == "split":
                    split = extract_enum_value(opt_kw.value, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
                    if split == ir.SplitMode.NONE:
                        raise ParserSyntaxError(
                            "pl.chunked_loop_optimizer() does not support split=pl.SplitMode.NONE",
                            span=self.span_tracker.get_span(opt_kw.value),
                            hint="Use pl.SplitMode.UP_DOWN or pl.SplitMode.LEFT_RIGHT",
                        )
                else:
                    raise ParserSyntaxError(
                        f"pl.chunked_loop_optimizer() got unexpected keyword '{opt_kw.arg}'",
                        span=self.span_tracker.get_span(opt_kw),
                        hint="Only 'split' is supported: "
                        "pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)",
                    )
            return split

        raise ParserSyntaxError(
            "optimization= only accepts pl.chunked_loop_optimizer or "
            "pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)",
            span=self.span_tracker.get_span(value),
            hint="Use optimization=pl.chunked_loop_optimizer or "
            "optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)",
        )

    def _eval_split_mode(self, value: ast.expr, stmt: ast.With) -> "ir.SplitMode":
        """Extract SplitMode enum value from AST expression."""
        return extract_enum_value(value, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")

    def _parse_scope_body(
        self,
        stmt: ast.With,
        scope_kind: "ir.ScopeKind",
        span: "ir.Span",
        *,
        level: "ir.Level | None" = None,
        role: "ir.Role | None" = None,
        split: "ir.SplitMode | None" = None,
    ) -> None:
        """Build a scope statement from a with-statement body."""
        with self.builder.scope(scope_kind, span, level=level, role=role, split=split):
            with self._scope_kind_context(scope_kind):
                self.scope_manager.enter_scope("scope")
                for body_stmt in stmt.body:
                    self.parse_statement(body_stmt)
                self.scope_manager.exit_scope(leak_vars=True)

    def _parse_at_scope(self, stmt: ast.With, context_expr: ast.Call) -> None:
        """Parse pl.at(...) context manager into a ScopeStmt."""
        level, role, opt_split = self._parse_at_kwargs(context_expr)
        span = self.span_tracker.get_span(stmt)

        is_core_group = level == ir.Level.CORE_GROUP

        if opt_split is not None and not is_core_group:
            raise ParserSyntaxError(
                "optimization=chunked_loop_optimizer is only supported with level=pl.Level.CORE_GROUP",
                span=span,
                hint="Use pl.at(level=pl.Level.CORE_GROUP, "
                "optimization=pl.chunked_loop_optimizer) for AutoInCore scope",
            )

        if is_core_group and role is not None:
            raise ParserSyntaxError(
                "role= is not supported with level=pl.Level.CORE_GROUP",
                span=span,
                hint="Drop role= for InCore/AutoInCore scopes, "
                "or use a non-CORE_GROUP level for Hierarchy scope",
            )

        if not is_core_group:
            self._parse_scope_body(stmt, ir.ScopeKind.Hierarchy, span, level=level, role=role)
        elif opt_split is not None:
            self._parse_scope_body(stmt, ir.ScopeKind.AutoInCore, span, split=opt_split)
        else:
            self._parse_scope_body(stmt, ir.ScopeKind.InCore, span)

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports:
        - with pl.incore(): ... (deprecated; creates ScopeStmt with InCore scope)
        - with pl.auto_incore(): ... (deprecated; creates ScopeStmt with AutoInCore scope)
        - with pl.auto_incore(split=pl.SplitMode.UP_DOWN): ... (deprecated; with split mode)
        - with pl.cluster(): ... (creates ScopeStmt with Cluster scope)
        - with pl.at(level=..., role=...): ... (creates ScopeStmt with InCore/Hierarchy scope)
        - with pl.at(level=CORE_GROUP): ... (creates ScopeStmt with InCore scope)
        - with pl.at(level=CORE_GROUP, optimization=pl.chunked_loop_optimizer): ...
          (creates ScopeStmt with AutoInCore scope)

        Args:
            stmt: With AST node
        """
        # Check that we have exactly one context manager
        if len(stmt.items) != 1:
            raise ParserSyntaxError(
                "Only single context manager supported in with statement",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.incore():', 'with pl.auto_incore():',"
                " 'with pl.cluster():', or 'with pl.at(level=...):'"
                " without multiple context managers",
            )

        item = stmt.items[0]
        context_expr = item.context_expr

        # Map DSL function names to ScopeKind values
        _SCOPE_KIND_MAP = {
            "incore": ir.ScopeKind.InCore,
            "auto_incore": ir.ScopeKind.AutoInCore,
            "cluster": ir.ScopeKind.Cluster,
        }

        if isinstance(context_expr, ast.Call):
            func = context_expr.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "pl":
                # Existing scope kinds: pl.incore(), pl.auto_incore(), pl.cluster()
                if func.attr in _SCOPE_KIND_MAP:
                    # Default split mode: None (validated later in ExpandMixedKernel for mixed kernels)
                    split_mode = None
                    if func.attr in ("auto_incore", "incore"):
                        if context_expr.args:
                            raise ParserSyntaxError(
                                f"pl.{func.attr}() does not accept positional arguments",
                                span=self.span_tracker.get_span(stmt),
                                hint=f"Use 'with pl.{func.attr}(split=pl.SplitMode.UP_DOWN):'",
                            )
                        for kw in context_expr.keywords:
                            if kw.arg == "split":
                                split_mode = self._eval_split_mode(kw.value, stmt)
                            else:
                                raise ParserSyntaxError(
                                    f"pl.{func.attr}() got unexpected keyword argument '{kw.arg}'",
                                    span=self.span_tracker.get_span(stmt),
                                    hint="Only 'split' keyword is supported",
                                )
                        # Emit deprecation warnings for incore/auto_incore
                        if func.attr == "incore":
                            warnings.warn(
                                "pl.incore() is deprecated; use "
                                "'with pl.at(level=pl.Level.CORE_GROUP):' instead",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                        else:
                            warnings.warn(
                                "pl.auto_incore() is deprecated; use "
                                "'with pl.at(level=pl.Level.CORE_GROUP, "
                                "optimization=pl.chunked_loop_optimizer):' instead",
                                DeprecationWarning,
                                stacklevel=2,
                            )
                    elif context_expr.args or context_expr.keywords:
                        raise ParserSyntaxError(
                            f"pl.{func.attr}() does not accept arguments",
                            span=self.span_tracker.get_span(stmt),
                            hint=f"Use 'with pl.{func.attr}():' without arguments",
                        )
                    scope_kind = _SCOPE_KIND_MAP[func.attr]
                    span = self.span_tracker.get_span(stmt)
                    self._parse_scope_body(stmt, scope_kind, span, split=split_mode)
                    return

                # pl.at(level=..., role=..., optimization=...)
                if func.attr == "at":
                    self._parse_at_scope(stmt, context_expr)
                    return

        # Unsupported context manager
        raise UnsupportedFeatureError(
            "Unsupported context manager in with statement",
            span=self.span_tracker.get_span(stmt),
            hint="Supported: 'with pl.incore():', 'with pl.auto_incore():',"
            " 'with pl.cluster():', 'with pl.at(level=..., optimization=...):'",
        )

    def parse_return(self, stmt: ast.Return) -> None:
        """Parse return statement.

        In inline mode, captures the return expression instead of emitting ReturnStmt.

        Args:
            stmt: Return AST node
        """
        if self._inline_mode:
            if stmt.value is None:
                return  # void inline, no return value
            if isinstance(stmt.value, ast.Tuple):
                exprs = [self.parse_expression(elt) for elt in stmt.value.elts]
                self._inline_return_expr = ir.MakeTuple(exprs, self.span_tracker.get_span(stmt))
            else:
                self._inline_return_expr = self.parse_expression(stmt.value)
            return

        span = self.span_tracker.get_span(stmt)

        if stmt.value is None:
            self.builder.return_stmt(None, span)
            return

        # Handle tuple return
        if isinstance(stmt.value, ast.Tuple):
            return_exprs = []
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
        if self._is_dsl_call(stmt, "static_print"):
            self._handle_static_print(stmt)
            return
        if self._is_dsl_call(stmt, "static_assert"):
            self._handle_static_assert(stmt)
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
            self.parse_yield_call(stmt.value)
            return

        expr = self.parse_expression(stmt.value)
        span = self.span_tracker.get_span(stmt)

        # Validate that we got an IR expression (not a list literal, etc.)
        if not isinstance(expr, ir.Expr):
            raise ParserSyntaxError(
                f"Evaluation statement must be an IR expression, got {type(expr).__name__}",
                span=span,
                hint="Only function calls and operations can be used as standalone statements",
            )

        # Emit EvalStmt using builder method
        self.builder.eval_stmt(expr, span)

    def parse_break(self, stmt: ast.Break) -> None:
        """Parse break statement.

        Args:
            stmt: Break AST node
        """
        span = self.span_tracker.get_span(stmt)
        self._validate_loop_control("break", span)
        self.builder.break_stmt(span=span)

    def parse_continue(self, stmt: ast.Continue) -> None:
        """Parse continue statement.

        Args:
            stmt: Continue AST node
        """
        span = self.span_tracker.get_span(stmt)
        self._validate_loop_control("continue", span)
        self.builder.continue_stmt(span=span)

    def _validate_loop_control(self, keyword: str, span: ir.Span) -> None:
        """Validate that a break/continue statement is in a valid loop context.

        Args:
            keyword: "break" or "continue"
            span: Source span for error reporting

        Raises:
            InvalidOperationError: If not inside a loop, or inside a parallel/unrolled loop
        """
        if not self._loop_kind_stack:
            raise InvalidOperationError(
                f"'{keyword}' outside loop",
                span=span,
                hint=f"'{keyword}' can only be used inside a for or while loop",
            )
        current_kind = self._loop_kind_stack[-1]
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
            return self.parse_subscript(expr)
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
        else:
            raise ParserTypeError(
                f"Unsupported constant type: {type(value)}",
                span=self.span_tracker.get_span(const),
                hint="Use int, float, or bool constants",
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
            ast.LShift: ir.bit_shift_left,
            ast.RShift: ir.bit_shift_right,
        }

        op_type = type(binop.op)
        if op_type not in op_map:
            raise UnsupportedFeatureError(
                f"Unsupported binary operator: {op_type.__name__}",
                span=self.span_tracker.get_span(binop),
                hint="Use supported operators: +, -, *, /, //, %, <<, >>",
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

        # Handle pl.yield_() specially
        if isinstance(func, ast.Attribute) and func.attr == "yield_":
            return self.parse_yield_call(call)

        # Handle cross-function calls via self.method_name() in @pl.program classes
        if isinstance(func, ast.Attribute):
            # Check for self.method_name pattern
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                method_name = func.attr
                if method_name in self.global_vars:
                    gvar = self.global_vars[method_name]
                    span = self.span_tracker.get_span(call)
                    self._reject_keyword_args(method_name, call, span)

                    # Validate argument count before parsing args to fail fast
                    func_obj = self.gvar_to_func.get(gvar)
                    if func_obj is not None:
                        self._validate_call_arg_count(method_name, func_obj, len(call.args), span)

                    args = [self.parse_expression(arg) for arg in call.args]
                    return_types = func_obj.return_types if func_obj else []
                    if func_obj is not None and return_types:
                        return_types = ir.deduce_call_return_type(
                            list(func_obj.params),
                            args,
                            return_types,
                        )
                    return self._make_call_with_return_type(gvar, args, return_types, span)
                else:
                    raise UndefinedVariableError(
                        f"Function '{method_name}' not defined in program",
                        span=self.span_tracker.get_span(call),
                        hint=f"Available functions: {list(self.global_vars.keys())}",
                    )

            # Handle pl.tensor.*, pl.tile.*, and pl.* operation calls
            return self.parse_op_call(call)

        # Handle bare-name calls to external ir.Function or InlineFunction
        if isinstance(func, ast.Name):
            from .decorator import InlineFunction  # noqa: PLC0415 (circular import)

            func_name = func.id
            resolved = self.expr_evaluator.closure_vars.get(func_name)
            if isinstance(resolved, ir.Function):
                return self._parse_external_function_call(func_name, resolved, call)
            if isinstance(resolved, InlineFunction):
                return self._parse_inline_call(func_name, resolved, call)

        raise UnsupportedFeatureError(
            f"Unsupported function call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.* operations, pl.yield_(), self.method() for cross-function calls, "
            "or call an external @pl.function / @pl.inline by name",
        )

    def parse_yield_call(self, call: ast.Call) -> ir.Expr:
        """Parse pl.yield_() call.

        Args:
            call: Call to pl.yield_() or pl.yield_()

        Returns:
            IR expression (first yielded value for single yield)
        """
        span = self.span_tracker.get_span(call)
        yield_exprs = []

        for arg in call.args:
            expr = self.parse_expression(arg)
            yield_exprs.append(expr)

        # Emit yield statement
        self.builder.emit(ir.YieldStmt(yield_exprs, span))

        # Bare top-level loop/while yields carry the loop output vars directly.
        # Record their names so pl.while_()/init_values and similar printed forms
        # can be parsed back into return_vars without requiring assignment-form
        # yields.
        if (
            self._current_yield_vars is not None
            and not self.in_if_stmt
            and (self.in_for_loop or self.in_while_loop)
        ):
            for expr in yield_exprs:
                if isinstance(expr, ir.Var):
                    self._track_yield_var(expr.name_hint, [expr])

        # Track yielded variables for if statement processing
        # This is for single assignment like: var = pl.yield_(expr)
        # We'll return a placeholder that gets resolved when if statement completes

        # Return first expression as the "value" of the yield
        # This handles: var = pl.yield_(expr)
        if len(yield_exprs) == 0:
            # Bare pl.yield_() with no arguments — return None (used as bare stmt)
            return None  # type: ignore[return-value]
        if len(yield_exprs) == 1:
            return yield_exprs[0]

        # Bare yield statements may legally yield multiple loop-carried values;
        # expression contexts still require assignment-form unpacking.
        if self.in_for_loop or self.in_while_loop or self.in_if_stmt:
            return None  # type: ignore[return-value]

        raise ParserSyntaxError(
            "Multiple yields should use tuple unpacking assignment",
            span=self.span_tracker.get_span(call),
            hint="Use tuple unpacking: (a, b) = pl.yield_(x, y)",
        )

    def parse_op_call(self, call: ast.Call) -> ir.Expr:
        """Parse operation call like pl.tensor.create_tensor() or pl.add().

        Args:
            call: Call AST node

        Returns:
            IR expression from operation
        """
        func = call.func

        # Navigate through attribute chain to find operation
        # e.g., pl.tensor.create_tensor -> ["pl", "tensor", "create_tensor"]
        # e.g., pl.add -> ["pl", "add"]
        attrs = []
        node = func
        while isinstance(node, ast.Attribute):
            attrs.insert(0, node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            attrs.insert(0, node.id)

        # pl.tensor.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "tensor":
            op_name = attrs[2]
            return self._parse_tensor_op(op_name, call)

        # pl.tile.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "tile":
            op_name = attrs[2]
            return self._parse_tile_op(op_name, call)

        # pl.system.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "system":
            op_name = attrs[2]
            return self._parse_system_op(op_name, call)

        # pl.const(value, dtype) — typed constant literal
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] == "const":
            return self._parse_typed_constant(call)

        # pl.{operation} (2-segment, unified dispatch or promoted ops)
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] not in ("tensor", "tile", "system"):
            op_name = attrs[1]
            return self._parse_unified_op(op_name, call)

        raise UnsupportedFeatureError(
            f"Unsupported operation call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.*, pl.tensor.*, pl.tile.*, or pl.system.* operations",
        )

    def _make_call_with_return_type(
        self,
        gvar: ir.GlobalVar,
        args: list[ir.Expr],
        return_types: list[ir.Type],
        span: ir.Span,
    ) -> ir.Expr:
        """Create an ir.Call, attaching the return type when known.

        Args:
            gvar: GlobalVar identifying the callee
            args: Parsed argument expressions
            return_types: The callee's return type list (may be empty)
            span: Source span for the call
        """
        if not return_types:
            return ir.Call(gvar, args, span)
        if len(return_types) == 1:
            return ir.Call(gvar, args, return_types[0], span)
        return ir.Call(gvar, args, ir.TupleType(return_types), span)

    @staticmethod
    def _reject_keyword_args(func_name: str, call: ast.Call, span: ir.Span) -> None:
        """Reject keyword arguments on function calls that only support positional args."""
        if call.keywords:
            raise ParserTypeError(
                f"Function '{func_name}' does not accept keyword arguments",
                span=span,
                hint="Pass all arguments positionally",
            )

    @staticmethod
    def _validate_call_arg_count(func_name: str, func: ir.Function, got: int, span: ir.Span) -> None:
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

    def _parse_external_function_call(
        self, _local_name: str, ext_func: ir.Function, call: ast.Call
    ) -> ir.Expr:
        """Parse a call to an externally-defined ir.Function.

        Args:
            _local_name: The name used in the caller's scope (may be aliased)
            ext_func: The external ir.Function object
            call: The AST Call node
        """
        func_name = ext_func.name
        span = self.span_tracker.get_span(call)

        # Validate no naming conflict with internal program functions
        if func_name in self.global_vars:
            raise ParserSyntaxError(
                f"External function '{func_name}' conflicts with program function '{func_name}'",
                span=span,
                hint="Rename either the external or program function to avoid the name conflict",
            )

        # Check for conflicting externals with same .name but different objects
        if func_name in self.external_funcs and self.external_funcs[func_name] is not ext_func:
            raise ParserSyntaxError(
                f"Conflicting external functions with name '{func_name}'",
                span=span,
                hint="External functions must have unique names; rename one of the functions",
            )

        # Track the external function
        self.external_funcs[func_name] = ext_func

        # Reject keyword args and validate argument count before parsing
        self._reject_keyword_args(func_name, call, span)
        self._validate_call_arg_count(func_name, ext_func, len(call.args), span)

        args = [self.parse_expression(arg) for arg in call.args]

        gvar = ir.GlobalVar(func_name)
        return_types = ir.deduce_call_return_type(
            list(ext_func.params),
            args,
            list(ext_func.return_types),
        )
        return self._make_call_with_return_type(gvar, args, return_types, span)

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        """Check if an AST statement is a docstring (string constant expression)."""
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    def _parse_inline_call(self, _local_name: str, inline_func: "InlineFunction", call: ast.Call) -> ir.Expr:
        """Parse a call to an InlineFunction, expanding its body in-place.

        Args:
            _local_name: The name used in the caller's scope
            inline_func: The InlineFunction object
            call: The AST Call node
        """
        span = self.span_tracker.get_span(call)
        self._reject_keyword_args(inline_func.name, call, span)

        expected = len(inline_func.param_names)
        got = len(call.args)
        if got != expected:
            raise ParserTypeError(
                f"Inline function '{inline_func.name}' expects {expected} argument(s), got {got}",
                span=span,
                hint=f"Check the inline function's parameter list: {inline_func.param_names}",
            )

        # Parse call arguments in the caller's context before entering inline scope
        arg_exprs = [self.parse_expression(arg) for arg in call.args]

        self.scope_manager.enter_scope("inline")
        for param_name, arg_expr in zip(inline_func.param_names, arg_exprs):
            self.scope_manager.define_var(param_name, arg_expr, allow_redef=True)

        # Save parser state and switch to the inline function's context
        prev_inline_state = (self._inline_mode, self._inline_return_expr)
        self._inline_mode = True
        self._inline_return_expr = None

        prev_closure_vars = self.expr_evaluator.closure_vars
        self.expr_evaluator.closure_vars = {**inline_func.closure_vars, **prev_closure_vars}

        prev_span_state = (
            self.span_tracker.source_file,
            self.span_tracker.source_lines,
            self.span_tracker.line_offset,
            self.span_tracker.col_offset,
        )
        self.span_tracker.source_file = inline_func.source_file
        self.span_tracker.source_lines = inline_func.source_lines
        self.span_tracker.line_offset = inline_func.line_offset
        self.span_tracker.col_offset = inline_func.col_offset

        try:
            for i, stmt in enumerate(inline_func.func_def.body):
                if i == 0 and self._is_docstring(stmt):
                    continue
                self.parse_statement(stmt)
        finally:
            # Restore parser state
            (
                self.span_tracker.source_file,
                self.span_tracker.source_lines,
                self.span_tracker.line_offset,
                self.span_tracker.col_offset,
            ) = prev_span_state
            self.expr_evaluator.closure_vars = prev_closure_vars
            return_expr = self._inline_return_expr
            self._inline_mode, self._inline_return_expr = prev_inline_state
            # Leak vars so inlined definitions are visible to the caller
            self.scope_manager.exit_scope(leak_vars=True)

        if return_expr is None:
            raise ParserTypeError(
                f"Inline function '{inline_func.name}' has no return value",
                span=span,
                hint="Inline functions used as expressions must return a value",
            )

        return return_expr

    def _parse_op_kwargs(self, call: ast.Call) -> dict[str, Any]:
        """Parse keyword arguments for an operation call.

        Shared helper for tensor, tile, system, and unified op parsing.

        Args:
            call: Call AST node

        Returns:
            Dictionary of keyword argument names to values
        """
        kwargs = {}
        for keyword in call.keywords:
            key = keyword.arg
            value = keyword.value

            # Handle dtype specially
            if key == "dtype":
                kwargs[key] = self.type_resolver.resolve_dtype(value)
            elif isinstance(value, ast.Constant):
                kwargs[key] = value.value
            elif isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                kwargs[key] = self._resolve_unary_kwarg(value)
            elif isinstance(value, ast.Name):
                kwargs[key] = self._resolve_name_kwarg(value)
            elif isinstance(value, ast.Attribute):
                kwargs[key] = self._resolve_attribute_kwarg(value)
            elif isinstance(value, ast.List):
                kwargs[key] = self._resolve_list_kwarg(value)
            else:
                kwargs[key] = self.parse_expression(value)
        return kwargs

    def _resolve_unary_kwarg(self, value: ast.UnaryOp) -> Any:
        """Resolve a unary op kwarg value (e.g., -1)."""
        if isinstance(value.operand, ast.Constant) and isinstance(value.operand.value, (int, float)):
            return -value.operand.value
        return self.parse_expression(value)

    def _resolve_name_kwarg(self, value: ast.Name) -> Any:
        """Resolve a Name kwarg value via scope lookup or closure eval."""
        if value.id in ["True", "False"]:
            return value.id == "True"
        if self.scope_manager.lookup_var(value.id) is not None:
            return self.parse_expression(value)  # IR var from scope
        # Not in IR scope — evaluate from closure (raises ParserTypeError if undefined)
        return self.expr_evaluator.eval_expr(value)

    def _resolve_attribute_kwarg(self, value: ast.Attribute) -> Any:
        """Resolve an Attribute kwarg value (e.g., pl.FP32, config.field, pipe_buf.base)."""
        try:
            return self.type_resolver.resolve_dtype(value)
        except ParserTypeError:
            pass

        # Check buffer metadata registry (e.g., pipe_buf.base from reserve_buffer)
        if isinstance(value.value, ast.Name):
            meta = self._buffer_meta.get(value.value.id)
            if meta is not None and value.attr in meta:
                return meta[value.attr]

        # Not a dtype or buffer attr — evaluate as a general expression from closure.
        return self.expr_evaluator.eval_expr(value)

    def _track_buffer_meta(self, var_name: str, call: ast.Call) -> None:
        """Track kwargs from reserve_buffer and import_peer_buffer calls for later attribute access.

        Enables patterns like:
            pipe_buf = pl.reserve_buffer(..., base=0x1000)
            pl.aic_initialize_pipe(..., v2c_consumer_buf=pipe_buf.base)

            peer_buf = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_bidir")
            pl.aic_initialize_pipe(..., c2v_consumer_buf=peer_buf.base)
        """
        func = call.func
        if not isinstance(func, ast.Attribute):
            return
        if func.attr not in ("reserve_buffer", "import_peer_buffer"):
            return
        meta: dict[str, Any] = {}
        for kw in call.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Constant):
                meta[kw.arg] = kw.value.value
        if func.attr == "reserve_buffer":
            # Index by (func_name, buffer_name) for cross-function import_peer_buffer resolution
            buf_name = meta.get("name")
            if buf_name is not None:
                self._buffer_name_meta[(self._func_name, buf_name)] = meta
        elif func.attr == "import_peer_buffer":
            # Resolve base from peer's reserve_buffer if available
            buf_name = meta.get("name")
            peer_func_name = meta.get("peer_func")
            if buf_name is not None and peer_func_name is not None:
                peer_key = (peer_func_name, buf_name)
                if peer_key in self._buffer_name_meta:
                    peer_meta = self._buffer_name_meta[peer_key]
                    if "base" in peer_meta:
                        meta["base"] = peer_meta["base"]
            if "base" not in meta:
                meta["base"] = -1  # AUTO sentinel
        if meta:
            self._buffer_meta[var_name] = meta

    def _resolve_list_kwarg(self, value: ast.List) -> Any:
        """Resolve a List kwarg value, trying closure eval first."""
        # If any element refers to a name in IR scope, parse as IR expressions
        # (mirrors _resolve_name_kwarg: IR scope takes priority over closure)
        if any(
            isinstance(elt, ast.Name) and self.scope_manager.lookup_var(elt.id) is not None
            for elt in value.elts
        ):
            return self.parse_list(value)
        success, result = self.expr_evaluator.try_eval_expr(value)
        if success and isinstance(result, list):
            return result
        return self.parse_list(value)

    def _dispatch_op(self, module: Any, module_name: str, op_name: str, call: ast.Call) -> ir.Expr:
        """Dispatch an operation call to the given ir_op module.

        Args:
            module: The ir_op sub-module (e.g., ir_op.tensor, ir_op.tile, ir_op.system)
            module_name: Human-readable module name for error messages
            op_name: Name of the operation to look up on the module
            call: Call AST node

        Returns:
            IR expression from the operation
        """
        if not hasattr(module, op_name):
            raise InvalidOperationError(
                f"Unknown {module_name} operation: {op_name}",
                span=self.span_tracker.get_span(call),
                hint=f"Check if '{op_name}' is a valid {module_name} operation",
            )
        args = [self.parse_expression(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        op_func = getattr(module, op_name)
        try:
            return op_func(*args, **kwargs, span=self.span_tracker.get_span(call))
        except ParserError:
            raise
        except Exception as e:
            raise InvalidOperationError(
                f"Error in {module_name} operation '{op_name}': {concise_error_message(e)}",
                span=self.span_tracker.get_span(call),
            ) from e

    def _parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation."""
        if op_name == "alloc":
            return self._parse_printed_alloc_call(call)
        return self._dispatch_op(ir_op.tensor, "tensor", op_name, call)

    def _parse_tile_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tile operation."""
        if op_name == "alloc":
            return self._parse_printed_alloc_call(call)
        return self._dispatch_op(ir_op.tile, "tile", op_name, call)

    @staticmethod
    def _is_printed_alloc_call(call: ast.Call) -> bool:
        """Return whether the AST call matches printer-emitted ``pl.tile.alloc(...)``
        or ``pl.tensor.alloc(...)``."""
        func = call.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "alloc"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr in ("tile", "tensor")
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pl"
        )

    def _parse_printed_alloc_call(self, call: ast.Call) -> ir.Call:
        """Parse printer-emitted ``pl.{tile,tensor}.alloc(...)`` into a raw IR call."""
        func = call.func
        assert isinstance(func, ast.Attribute) and isinstance(func.value, ast.Attribute)
        namespace = func.value.attr  # "tile" or "tensor"
        op_name = f"{namespace}.alloc"

        if call.keywords:
            raise InvalidOperationError(
                f"{op_name} in printed IR must use positional arguments only",
                span=self.span_tracker.get_span(call),
            )
        args = [self.parse_expression(arg) for arg in call.args]
        if len(args) != 2:
            raise InvalidOperationError(
                f"{op_name} in printed IR expects 2 positional args (memory_space, size), got {len(args)}",
                span=self.span_tracker.get_span(call),
            )
        return ir.create_op_call(op_name, args, {}, self.span_tracker.get_span(call))

    def _parse_system_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse system operation."""
        return self._dispatch_op(ir_op.system, "system", op_name, call)

    # Maps iterator type name to ForKind enum value.
    _ITERATOR_TO_KIND = {
        "range": ir.ForKind.Sequential,
        "parallel": ir.ForKind.Parallel,
        "unroll": ir.ForKind.Unroll,
    }

    # Maps unified op names to the scalar variant for tile ops.
    # Only binary arithmetic ops have scalar auto-dispatch.
    _TILE_SCALAR_OPS: dict[str, str] = {
        "add": "adds",
        "sub": "subs",
        "mul": "muls",
        "div": "divs",
    }

    # Maps unified op names to ir scalar expression functions.
    _SCALAR_BINARY_OPS: dict[str, str] = {
        "min": "min_",
        "max": "max_",
    }

    _SCALAR_UNARY_OPS: dict[str, str] = {}

    # Maps unified op names to ir scalar functions that take (expr, dtype, span).
    _SCALAR_DTYPE_OPS: dict[str, str] = {
        "cast": "cast",
    }

    # Ops that exist only in one module (no dispatch needed).
    _TENSOR_ONLY_OPS = {
        "create_tensor",
        "dim",
        "assemble",
        "full",
    }
    _TILE_ONLY_OPS = {
        "load",
        "store",
        "move",
        "log",
        "relu",
        "minimum",
        "cmp",
        "cmps",
        "sum",
        "matmul_bias",
        "gemv",
        "gemv_acc",
        "gemv_bias",
        "abs",
        "create_tile",
        "tpush_to_aiv",
        "tpush_to_aic",
        "tpop_from_aic",
        "tpop_from_aiv",
    }
    _SYSTEM_OPS = {
        "tfree_to_aic",
        "tfree_to_aiv",
        "aic_initialize_pipe",
        "aiv_initialize_pipe",
        "reserve_buffer",
        "import_peer_buffer",
    }

    def _parse_unified_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse unified operation call (pl.{op_name}).

        Dispatches to tensor or tile IR op based on the first argument's type.

        Args:
            op_name: Name of the operation
            call: Call AST node

        Returns:
            IR expression from the dispatched operation
        """
        # Short-circuit for ops that only exist in one module
        if op_name in self._TENSOR_ONLY_OPS:
            return self._parse_tensor_op(op_name, call)
        if op_name in self._TILE_ONLY_OPS:
            return self._parse_tile_op(op_name, call)
        if op_name in self._SYSTEM_OPS:
            return self._parse_system_op(op_name, call)

        call_span = self.span_tracker.get_span(call)

        if not call.args:
            raise InvalidOperationError(
                f"Unified operation '{op_name}' requires at least one argument for type dispatch",
                span=call_span,
                hint="Provide a Tensor or Tile as the first argument",
            )

        # Parse only the first arg to determine dispatch target
        first_arg = self.parse_expression(call.args[0])
        first_type = first_arg.type

        if isinstance(first_type, ir.TensorType):
            return self._parse_tensor_op(op_name, call)

        if isinstance(first_type, ir.TileType):
            # For binary arithmetic ops, check if rhs is scalar → use scalar variant
            scalar_op = self._TILE_SCALAR_OPS.get(op_name)
            if scalar_op and len(call.args) >= 2:
                rhs_arg = self.parse_expression(call.args[1])
                if isinstance(rhs_arg.type, ir.ScalarType):
                    return self._parse_tile_op(scalar_op, call)

            return self._parse_tile_op(op_name, call)

        if isinstance(first_type, ir.ScalarType):
            return self._parse_scalar_op(op_name, call, call_span)

        raise InvalidOperationError(
            f"Cannot dispatch '{op_name}': first argument has type {type(first_type).__name__}, "
            f"expected TensorType, TileType, or ScalarType",
            span=call_span,
            hint="Use pl.tensor.* or pl.tile.* for explicit dispatch",
        )

    def _parse_typed_constant(self, call: ast.Call) -> ir.Expr:
        """Parse pl.const(value, dtype) → ConstInt or ConstFloat.

        Args:
            call: Call AST node for pl.const(value, dtype)

        Returns:
            ConstInt or ConstFloat with the specified dtype
        """
        span = self.span_tracker.get_span(call)

        if len(call.args) != 2:
            raise ParserSyntaxError(
                "pl.const() requires exactly 2 arguments: value and dtype",
                span=span,
                hint="Use pl.const(42, pl.INT32) or pl.const(1.0, pl.FP16)",
            )

        # Extract numeric value from first argument (handles Constant and -Constant)
        value_node = call.args[0]
        negate = False
        if isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
            negate = True
            value_node = value_node.operand

        if not isinstance(value_node, ast.Constant) or not isinstance(value_node.value, (int, float)):
            raise ParserSyntaxError(
                "pl.const() first argument must be a numeric literal",
                span=span,
                hint="Use an int or float literal: pl.const(42, pl.INT32)",
            )

        value = value_node.value
        if negate:
            value = -value

        # Resolve dtype from second argument
        dtype = self.type_resolver.resolve_dtype(call.args[1])

        if isinstance(value, float):
            return ir.ConstFloat(value, dtype, span)
        else:
            return ir.ConstInt(value, dtype, span)

    def _parse_scalar_op(self, op_name: str, call: ast.Call, call_span: ir.Span) -> ir.Expr:
        """Parse scalar operation (e.g. pl.min(s1, s2) where s1, s2 are scalars).

        Args:
            op_name: Name of the operation
            call: Call AST node
            call_span: Source span for error reporting

        Returns:
            IR scalar expression
        """
        if call.keywords:
            raise InvalidOperationError(
                f"Scalar operation '{op_name}' does not accept keyword arguments",
                span=call_span,
            )

        if op_name in self._SCALAR_DTYPE_OPS:
            if len(call.args) != 2:
                raise InvalidOperationError(
                    f"Scalar operation '{op_name}' requires exactly 2 arguments (value, dtype), "
                    f"got {len(call.args)}",
                    span=call_span,
                )
            operand = self.parse_expression(call.args[0])
            dtype = self.type_resolver.resolve_dtype(call.args[1])
            ir_func_name = self._SCALAR_DTYPE_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(operand, dtype, call_span)

        if op_name in self._SCALAR_BINARY_OPS:
            if len(call.args) != 2:
                raise InvalidOperationError(
                    f"Scalar binary operation '{op_name}' requires exactly 2 arguments, got {len(call.args)}",
                    span=call_span,
                )
            lhs = self.parse_expression(call.args[0])
            rhs = self.parse_expression(call.args[1])
            ir_func_name = self._SCALAR_BINARY_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(lhs, rhs, call_span)

        if op_name in self._SCALAR_UNARY_OPS:
            if len(call.args) != 1:
                raise InvalidOperationError(
                    f"Scalar unary operation '{op_name}' requires exactly 1 argument, got {len(call.args)}",
                    span=call_span,
                )
            arg = self.parse_expression(call.args[0])
            ir_func_name = self._SCALAR_UNARY_OPS[op_name]
            ir_func = getattr(ir, ir_func_name)
            return ir_func(arg, call_span)

        supported = sorted(
            set(self._SCALAR_BINARY_OPS) | set(self._SCALAR_UNARY_OPS) | set(self._SCALAR_DTYPE_OPS)
        )
        raise InvalidOperationError(
            f"Operation '{op_name}' is not supported for scalar arguments",
            span=call_span,
            hint=f"Supported scalar ops: {', '.join(supported)}",
        )

    def parse_attribute(self, attr: ast.Attribute) -> ir.Expr:
        """Parse attribute access.

        Args:
            attr: Attribute AST node

        Returns:
            IR expression
        """
        # Try to evaluate as a Python enum value (e.g., pl.MemorySpace.Vec -> ConstInt)
        try:
            value = self.expr_evaluator.eval_expr(attr)
            if isinstance(value, ir.MemorySpace):
                return ir.ConstInt(value.value, DataType.INDEX, self.span_tracker.get_span(attr))
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

    def parse_subscript(self, subscript: ast.Subscript) -> ir.Expr:
        """Parse subscript expression: tuple[0], tensor[0:16, :], tile[0:16, :].

        Supports:
        - TupleType: ``t[0]`` -> TupleGetItemExpr
        - TensorType: ``A[0:16, :]`` -> tensor.slice, ``A[i, j]`` -> tensor.read
        - TileType: ``A[0:16, :]`` -> tile.slice, ``A[i, j]`` -> tile.read
        """
        span = self.span_tracker.get_span(subscript)
        value_expr = self.parse_expression(subscript.value)
        value_type = value_expr.type

        if isinstance(value_type, ir.TupleType):
            return self._parse_tuple_subscript(subscript, value_expr, span)
        if isinstance(value_type, ir.TensorType):
            return self._parse_tensor_subscript(subscript, value_expr, value_type, span)
        if isinstance(value_type, ir.TileType):
            return self._parse_tile_subscript(subscript, value_expr, value_type, span)

        raise ParserTypeError(
            f"Subscript requires Tuple, Tensor, or Tile type, got {type(value_type).__name__}",
            span=span,
            hint="Subscript syntax is supported for Tuple, Tensor, and Tile types",
        )

    def _parse_tuple_subscript(self, subscript: ast.Subscript, value_expr: ir.Expr, span: ir.Span) -> ir.Expr:
        """Parse tuple subscript: ``t[0]`` -> TupleGetItemExpr."""
        if isinstance(subscript.slice, ast.Constant):
            index = subscript.slice.value
            if not isinstance(index, int):
                raise ParserSyntaxError(
                    "Tuple index must be an integer",
                    span=span,
                    hint="Use integer index like tuple[0]",
                )
        else:
            raise UnsupportedFeatureError(
                "Only constant integer indices supported for tuple access",
                span=span,
                hint="Use a constant integer index like tuple[0]",
            )
        return ir.TupleGetItemExpr(value_expr, index, span)

    def _normalize_subscript_indices(self, subscript: ast.Subscript, span: ir.Span) -> list[ast.expr]:
        """Normalize subscript.slice into a flat list of index components.

        ``A[x]`` -> [x], ``A[x, y]`` -> [x, y]
        Each component is an ast.Slice, ast.Constant, ast.Name, ast.BinOp, etc.
        """
        slc = subscript.slice
        if isinstance(slc, ast.Tuple):
            return list(slc.elts)
        return [slc]

    def _build_slice_shape_expr(
        self,
        upper_expr: int | ir.Expr,
        lower_expr: int | ir.Expr,
    ) -> int | ir.Expr:
        """Build a slice extent, folding compile-time constants when possible."""
        folded_extent = _fold_const_slice_extent(upper_expr, lower_expr)
        if folded_extent is not None:
            return folded_extent
        lhs = (
            upper_expr
            if isinstance(upper_expr, ir.Expr)
            else ir.ConstInt(upper_expr, DataType.INDEX, ir.Span.unknown())
        )
        rhs = (
            lower_expr
            if isinstance(lower_expr, ir.Expr)
            else ir.ConstInt(lower_expr, DataType.INDEX, ir.Span.unknown())
        )
        return ir.sub(lhs, rhs)

    def _to_index_expr(self, expr: int | ir.Expr) -> ir.Expr:
        """Convert an integer-like slice bound into an INDEX expression."""
        if isinstance(expr, ir.Expr):
            return expr
        return ir.ConstInt(expr, DataType.INDEX, ir.Span.unknown())

    def _build_clamped_slice_shape_expr(
        self,
        upper_expr: int | ir.Expr,
        lower_expr: int | ir.Expr,
        span: ir.Span,
    ) -> int | ir.Expr:
        """Build a non-negative slice extent, clamping dynamic cases at zero."""
        folded_extent = _fold_const_slice_extent(upper_expr, lower_expr)
        if folded_extent is not None:
            return max(folded_extent, 0)
        return ir.max_(
            self._to_index_expr(self._build_slice_shape_expr(upper_expr, lower_expr)),
            ir.ConstInt(0, DataType.INDEX, ir.Span.unknown()),
            span,
        )

    def _intersect_slice_upper_bound(
        self,
        requested_upper: int | ir.Expr,
        source_upper: int | ir.Expr,
        span: ir.Span,
    ) -> int | ir.Expr:
        """Intersect an explicit slice upper bound with the source logical extent."""
        requested_const = _fold_const_slice_extent(requested_upper, 0)
        source_const = _fold_const_slice_extent(source_upper, 0)
        if requested_const is not None and source_const is not None:
            return min(requested_const, source_const)
        return ir.min_(self._to_index_expr(requested_upper), self._to_index_expr(source_upper), span)

    def _build_tile_alloc_extent(
        self,
        static_extent: int | ir.Expr,
        lower_expr: int | ir.Expr,
        upper_expr: int | ir.Expr | None,
        span: ir.Span,
    ) -> int | ir.Expr:
        """Build the static tile extent for a slice.

        Tile slices must keep a compile-time allocation shape even when the logical
        valid extent is dynamic. This helper computes the largest static extent that
        can safely back the slice while `valid_shape` carries any runtime narrowing.
        """
        lower_const = _fold_const_slice_extent(lower_expr, 0)
        static_const = _fold_const_slice_extent(static_extent, 0)
        upper_const = None if upper_expr is None else _fold_const_slice_extent(upper_expr, 0)

        max_extent = None
        if lower_const is not None and static_const is not None:
            max_extent = max(static_const - lower_const, 0)

        if upper_const is not None:
            extent = max(upper_const - lower_const, 0) if lower_const is not None else upper_const
            if max_extent is not None:
                extent = min(extent, max_extent)
            if extent <= 0:
                raise UnsupportedFeatureError(
                    "Tile subscript must produce a positive static extent",
                    span=span,
                    hint="Keep tile slice bounds within the source static shape and ensure upper > lower",
                )
            return extent

        if max_extent is not None:
            if max_extent <= 0:
                raise UnsupportedFeatureError(
                    "Tile subscript must produce a positive static extent",
                    span=span,
                    hint="Keep tile slice bounds within the source static shape and ensure upper > lower",
                )
            return max_extent

        return static_extent

    def _slice_extents_match(
        self,
        lhs: int | ir.Expr,
        rhs: int | ir.Expr,
    ) -> bool:
        """Return True when two slice extents are obviously equivalent."""
        lhs_const = _fold_const_slice_extent(lhs, 0)
        rhs_const = _fold_const_slice_extent(rhs, 0)
        if lhs_const is not None and rhs_const is not None:
            return lhs_const == rhs_const
        return lhs is rhs

    def _build_subscript_slice_args(
        self,
        indices: list[ast.expr],
        full_shape: list[ir.Expr],
        span: ir.Span,
        kind_name: str,
    ) -> tuple[list[int | ir.Expr], list[int | ir.Expr]]:
        """Convert mixed subscript indices into slice shape/offset args."""
        shape_exprs: list[int | ir.Expr] = []
        offset_exprs: list[int | ir.Expr] = []

        for dim_idx, idx in enumerate(indices):
            if not isinstance(idx, ast.Slice):
                offset_exprs.append(self.parse_expression(idx))
                shape_exprs.append(1)
                continue

            if idx.step is not None:
                raise UnsupportedFeatureError(
                    f"Slice step is not supported in {kind_name} subscript",
                    span=span,
                    hint="Use A[start:stop] without step",
                )

            if idx.lower is None:
                offset_exprs.append(0)
                shape_exprs.append(
                    full_shape[dim_idx] if idx.upper is None else self.parse_expression(idx.upper)
                )
                continue

            lower_expr = self.parse_expression(idx.lower)
            offset_exprs.append(lower_expr)

            upper_expr = full_shape[dim_idx] if idx.upper is None else self.parse_expression(idx.upper)
            shape_exprs.append(self._build_slice_shape_expr(upper_expr, lower_expr))

        return shape_exprs, offset_exprs

    def _build_tile_subscript_slice_args(
        self,
        indices: list[ast.expr],
        tile_type: ir.TileType,
        span: ir.Span,
    ) -> tuple[list[int | ir.Expr], list[int | ir.Expr], list[int | ir.Expr] | None]:
        """Convert tile subscripts into static shape/offset args plus optional valid_shape."""
        static_shape = list(tile_type.shape)
        tile_view = tile_type.tile_view
        logical_shape = (
            list(tile_view.valid_shape)
            if tile_view is not None and len(tile_view.valid_shape) == len(static_shape)
            else list(static_shape)
        )

        shape_exprs: list[int | ir.Expr] = []
        offset_exprs: list[int | ir.Expr] = []
        valid_shape_exprs: list[int | ir.Expr] = []
        needs_valid_shape = False

        for dim_idx, idx in enumerate(indices):
            if not isinstance(idx, ast.Slice):
                index_expr = self.parse_expression(idx)
                offset_exprs.append(index_expr)
                shape_exprs.append(1)
                valid_shape_exprs.append(1)
                continue

            if idx.step is not None:
                raise UnsupportedFeatureError(
                    "Slice step is not supported in tile subscript",
                    span=span,
                    hint="Use A[start:stop] without step",
                )

            lower_expr = 0 if idx.lower is None else self.parse_expression(idx.lower)
            if idx.lower is not None and _fold_const_slice_extent(lower_expr, 0) is None:
                raise UnsupportedFeatureError(
                    "Dynamic lower bounds are not supported in tile subscript",
                    span=span,
                    hint="Use a constant tile slice lower bound or rewrite the logic with explicit tile ops",
                )
            offset_exprs.append(lower_expr)

            parsed_upper = None if idx.upper is None else self.parse_expression(idx.upper)
            valid_upper = (
                logical_shape[dim_idx]
                if parsed_upper is None
                else self._intersect_slice_upper_bound(parsed_upper, logical_shape[dim_idx], span)
            )

            valid_extent = (
                valid_upper
                if idx.lower is None
                else self._build_clamped_slice_shape_expr(valid_upper, lower_expr, span)
            )
            alloc_extent = self._build_tile_alloc_extent(
                static_shape[dim_idx], lower_expr, parsed_upper, span
            )

            shape_exprs.append(alloc_extent)
            valid_shape_exprs.append(valid_extent)
            needs_valid_shape = needs_valid_shape or not self._slice_extents_match(alloc_extent, valid_extent)

        return shape_exprs, offset_exprs, valid_shape_exprs if needs_valid_shape else None

    def _parse_tensor_subscript(
        self,
        subscript: ast.Subscript,
        value_expr: ir.Expr,
        tensor_type: ir.TensorType,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse tensor subscript: A[0:16, :] -> tensor.slice, A[i, j] -> tensor.read."""
        indices = self._normalize_subscript_indices(subscript, span)
        rank = len(tensor_type.shape)
        if len(indices) != rank:
            raise ParserTypeError(
                f"Tensor subscript requires {rank} indices, got {len(indices)}",
                span=span,
                hint=f"Provide exactly {rank} indices for a {rank}D tensor",
            )

        has_slice = any(isinstance(idx, ast.Slice) for idx in indices)

        if not has_slice:
            # All integer indices -> tensor.read(tensor, [i, j])
            idx_exprs: list[int | ir.Expr] = [self.parse_expression(idx) for idx in indices]
            return ir_op.tensor.read(value_expr, idx_exprs, span=span)

        shape_exprs, offset_exprs = self._build_subscript_slice_args(
            indices, list(tensor_type.shape), span, "tensor"
        )
        return ir_op.tensor.slice(value_expr, shape_exprs, offset_exprs, span=span)

    def _parse_tile_subscript(
        self,
        subscript: ast.Subscript,
        value_expr: ir.Expr,
        tile_type: ir.TileType,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse tile subscript: A[0:16, :] -> tile.slice, A[i, j] -> tile.read."""
        indices = self._normalize_subscript_indices(subscript, span)
        rank = len(tile_type.shape)
        if len(indices) != rank:
            raise ParserTypeError(
                f"Tile subscript requires {rank} indices, got {len(indices)}",
                span=span,
                hint=f"Provide exactly {rank} indices for a {rank}D tile",
            )

        has_slice = any(isinstance(idx, ast.Slice) for idx in indices)

        if not has_slice:
            # All integer indices -> tile.read(tile, [i, j])
            idx_exprs: list[int | ir.Expr] = [self.parse_expression(idx) for idx in indices]
            return ir_op.tile.read(value_expr, idx_exprs, span=span)

        shape_exprs, offset_exprs, valid_shape_exprs = self._build_tile_subscript_slice_args(
            indices, tile_type, span
        )
        return ir_op.tile.slice(value_expr, shape_exprs, offset_exprs, valid_shape_exprs, span=span)

    def _resolve_yield_var_type(self, annotation: ast.expr | None) -> ir.Type:
        """Resolve type annotation for a yield variable.

        Args:
            annotation: Type annotation AST node, or None if not annotated

        Returns:
            Resolved IR type
        """
        if annotation is None:
            # Fallback to generic tensor type when no annotation present
            return ir.TensorType([1], DataType.INT32)

        resolved = self.type_resolver.resolve_type(annotation)
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

    def _scan_for_yields(self, stmts: list[ast.stmt]) -> list[tuple[str, ast.expr | None]]:
        """Scan statements for yield assignments to determine output variable names and types.

        Args:
            stmts: List of statements to scan

        Returns:
            List of tuples (variable_name, type_annotation) where type_annotation is None if not annotated
        """
        yield_vars = []

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
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    yield_vars.append((elt.id, None))

            # Skip nested if statements — each nested if's yields are
            # its own return_vars, handled by parse_if_statement when
            # it processes that specific if node. Recursing would
            # incorrectly count inner yields as outer return_vars.
            elif isinstance(stmt, ast.If):
                pass

        return yield_vars


__all__ = ["ASTParser"]
