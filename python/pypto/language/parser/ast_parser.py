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
import copy
import keyword as _keyword_mod
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from pypto.ir import IRBuilder
from pypto.ir import op as ir_op
from pypto.ir.printer import python_print
from pypto.language.distributed import op as _dsl_pld
from pypto.language.dsl_api import RangeIterator as _DslRangeIterator
from pypto.language.op import array_ops as _dsl_array
from pypto.language.op import system_ops as _dsl_system
from pypto.language.op import tensor_ops as _dsl_tensor
from pypto.language.op import tile_ops as _dsl_tile
from pypto.pypto_core import DataType, ir
from pypto.pypto_core import arith as _arith

from ._dsl_invoker import invoke_dsl
from .diagnostics import (
    InvalidOperationError,
    ParserError,
    ParserSyntaxError,
    ParserTypeError,
    UndefinedVariableError,
    UnsupportedFeatureError,
    concise_error_message,
)
from .enum_utils import (
    LEVEL_MAP,
    LOOP_ORIGIN_MAP,
    ROLE_MAP,
    SCOPE_MODE_MAP,
    SPLIT_MODE_MAP,
    extract_enum_value,
)
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


# Canonical pld.<category>.<op> middle segments. Kept in sync with the
# submodules exposed by pypto.language.distributed.op (system_ops / tensor_ops
# / tile_ops); also surfaced as the hint in _parse_pld_category_op.
_PLD_CATEGORIES: frozenset[str] = frozenset({"system", "tensor", "tile"})


def _is_pld_call(node: object, attr_name: str) -> TypeGuard[ast.Call]:
    """Match ``pld.<attr_name>(...)`` or ``pld.<category>.<attr_name>(...)``.

    Anchored on the literal ``pld.`` prefix; aliased imports don't match. For
    the 3-segment form the middle segment must be one of the canonical
    categories (``system`` / ``tensor`` / ``tile``) so typos like
    ``pld.typo.alloc_window_buffer(...)`` don't bypass dispatch.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != attr_name:
        return False
    parent = func.value
    # 2-segment: pld.<attr_name>
    if isinstance(parent, ast.Name) and parent.id == "pld":
        return True
    # 3-segment: pld.<category>.<attr_name>
    return (
        isinstance(parent, ast.Attribute)
        and parent.attr in _PLD_CATEGORIES
        and isinstance(parent.value, ast.Name)
        and parent.value.id == "pld"
    )


def _is_pl_call(node: object, attr_name: str) -> TypeGuard[ast.Call]:
    """Return True when ``node`` is the AST for a ``pl.<attr_name>(...)`` call.

    Recognises only the dotted ``pl.<attr>`` form. Aliasing under another
    name (``from pypto.language import submit``) is intentionally not matched —
    the parser anchors construct detection on the ``pl.`` prefix.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == attr_name
        and isinstance(func.value, ast.Name)
        and func.value.id == "pl"
    )


def _is_const_int(value: object) -> bool:
    """Check if a value is a compile-time constant integer.

    Handles plain int, ir.ConstInt, and ir.Neg(ir.ConstInt) (negative literals).
    """
    if isinstance(value, (int, ir.ConstInt)):
        return True
    return isinstance(value, ir.Neg) and isinstance(value.operand, ir.ConstInt)


def _is_dep_var_type(type_: ir.Type | None) -> bool:
    """Return True if ``type_`` is acceptable as a ``pl.submit(...)`` ``deps=`` entry.

    Two shapes are accepted:

    * ``ScalarType(TASK_ID)`` — a single TaskId (typically the ``tid`` bound
      by a prior ``_, tid = pl.submit(...)``, or the ``system.task_invalid``
      sentinel a bare ``None`` lowers to).
    * ``ArrayType(..., TASK_ID)`` — a per-slot TaskId array (typically from
      ``pl.array.create(N, pl.TASK_ID)`` threaded through a loop as an
      iter_arg). Codegen expands it to one entry per slot, each guarded by
      ``is_valid()`` (early-phase slots may still hold the invalid sentinel).
    """
    if isinstance(type_, ir.ScalarType) and type_.dtype == DataType.TASK_ID:
        return True
    if isinstance(type_, ir.ArrayType) and type_.dtype == DataType.TASK_ID:
        return True
    return False


def _is_per_element_task_id_read(expr: ir.Expr) -> bool:
    """Return True if ``expr`` is ``array.get_element(arr, idx)`` reading a
    TASK_ID from an ``Array[_, TASK_ID]``.

    Accepted as a ``deps=`` entry under Form 1 (per-element subscript). The
    parser desugars these into a fresh ``Array[N, TASK_ID]`` populated by
    ``array.update_element`` calls so the existing whole-array dep codegen
    path can emit the per-slot ``is_valid()``-guarded ``set_dependencies``.
    """
    if not isinstance(expr, ir.Call):
        return False
    if expr.op.name != "array.get_element":
        return False
    if not isinstance(expr.type, ir.ScalarType) or expr.type.dtype != DataType.TASK_ID:
        return False
    if not expr.args:
        return False
    arr_type = expr.args[0].type
    return isinstance(arr_type, ir.ArrayType) and arr_type.dtype == DataType.TASK_ID


def _fold_const_slice_extent(upper: object, lower: object) -> int | None:
    """Fold a slice extent when both bounds are compile-time constants."""
    upper_value = _const_int_value(upper)
    lower_value = _const_int_value(lower)
    if upper_value is None or lower_value is None:
        return None
    return upper_value - lower_value


def _get_source_valid_shape(source_type: ir.Type) -> list[ir.Expr] | None:
    """Return the source's view ``valid_shape``, or ``None`` when no view metadata.

    Pure accessor — does NOT decide whether the source is actually narrowed
    (whether ``valid_shape`` differs from ``shape``). Callers that need the
    narrowing semantics should compare against ``source_type.shape`` themselves
    (e.g. via :func:`_shape_exprs_match`).

    TensorType reads ``tensor_view.valid_shape``; TileType uses
    ``get_effective_tile_view()`` so a canonicalized implicit view (stored as
    ``None``) still surfaces its semantic ``valid_shape``.
    """
    if isinstance(source_type, ir.TensorType):
        view = source_type.tensor_view
        if view is None or not view.valid_shape:
            return None
        return list(view.valid_shape)
    if isinstance(source_type, ir.TileType):
        view = source_type.get_effective_tile_view()
        if not view.valid_shape:
            return None
        return list(view.valid_shape)
    return None


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


def _shape_has_symbolic_dim(type_: ir.Type) -> bool:
    """Return True when *type_* is a shaped type with at least one non-ConstInt dim."""
    if not isinstance(type_, (ir.TensorType, ir.TileType)):
        return False
    return any(not isinstance(d, ir.ConstInt) for d in type_.shape)


def _simplify_shape_dims(type_: ir.Type, analyzer: "_arith.Analyzer") -> ir.Type:
    """Rebuild a TensorType/TileType with each outer shape dim passed through
    the arithmetic simplifier.

    Needed so that symbolic dims that reduce to the same value — e.g.
    ``(k + C) - k`` and ``C`` — compare equal under structural ``==``.

    Scope: only the outer ``shape`` is simplified. TileView fields
    (``valid_shape``, ``stride``, ``start_offset``) and TensorView
    fields retain their original expressions and still compare structurally.
    """
    if not _shape_has_symbolic_dim(type_):
        return type_
    assert isinstance(type_, (ir.TensorType, ir.TileType))
    simplified_shape = [analyzer.simplify(d) if isinstance(d, ir.Expr) else d for d in type_.shape]
    if isinstance(type_, ir.TensorType):
        return ir.TensorType(simplified_shape, type_.dtype, type_.memref, type_.tensor_view)
    return ir.TileType(simplified_shape, type_.dtype, type_.memref, type_.tile_view, type_.memory_space)


def _types_match(lhs: ir.Type | None, rhs: ir.Type | None) -> bool:
    """Return whether two IR types are structurally equal after TileView normalization.

    Uses C++ ``structural_equal`` (via ``==``) instead of comparing printed
    strings.  Note: ``structural_equal`` intentionally skips the ``memref``
    field on TileType/TensorType — memref identity is not relevant for
    parser-level type matching because the Var (not the Call) carries the
    authoritative memref via ``override_type``.

    Outer shape dimensions are simplified via a shared arithmetic analyzer
    before comparison so expressions like ``(k + C) - k`` match ``C``.  The
    analyzer is only constructed when at least one shape actually contains a
    symbolic dim, keeping the all-static fast path free of arith overhead.
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
    if _shape_has_symbolic_dim(lhs) or _shape_has_symbolic_dim(rhs):
        analyzer = _arith.Analyzer()
        lhs = _simplify_shape_dims(lhs, analyzer)
        rhs = _simplify_shape_dims(rhs, analyzer)
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


class _FirstReturnTypeCollector(ir.IRVisitor):
    """Capture the value types of the first ``return <values>`` in a body.

    Used to derive a callee's *effective* return types when it declared no
    ``-> `` annotation but does ``return <value>``. Recording only the first
    value-bearing return is sufficient: a function's returns are type-consistent,
    so any of them yields the same call result type.
    """

    def __init__(self) -> None:
        super().__init__()
        self.types: list[ir.Type] | None = None

    def visit_return_stmt(self, op: ir.ReturnStmt) -> None:
        if self.types is None and op.value:
            self.types = [v.type for v in op.value]
        super().visit_return_stmt(op)


def _derive_return_types_from_body(body: ir.Stmt | None) -> list[ir.Type]:
    """Effective return types derived from a function body's first return.

    Returns an empty list when *body* is ``None``, has no value-bearing
    ``return`` (a void function), or the returned values are not all concretely
    typed. In the last case there is nothing to recover — leaving the result
    empty keeps the call's type ``UnknownType`` exactly as before.
    """
    if body is None:
        return []
    collector = _FirstReturnTypeCollector()
    collector.visit_stmt(body)
    types = collector.types
    if not types or any(isinstance(t, ir.UnknownType) for t in types):
        return []
    return types


@dataclass
class _AtKwargState:
    """Mutable accumulator used while parsing pl.at() keyword arguments."""

    level: "ir.Level | None" = None
    role: "ir.Role | None" = None
    name_hint: str = ""
    requests_auto_chunk: bool = False
    split_mode: "ir.SplitMode | None" = None
    # Optional cross-core ring-buffer depth from ``pl.split(mode, slot_num=N)``.
    # Stored on the scope attrs and propagated to the outlined function attr.
    split_slot_num: "int | None" = None
    # Tracks which kwarg produced the AutoChunk / split state so the validation
    # step can reject mixing the new `optimizations=` list with the deprecated
    # `optimization=`/`split=` kwargs and emit DeprecationWarning at the end.
    new_optimizations_kw: "ast.keyword | None" = field(default=None)
    legacy_optimization_kw: "ast.keyword | None" = field(default=None)
    legacy_split_kw: "ast.keyword | None" = field(default=None)
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


_SPMD_SCOPE_NAME_SUFFIX = "_spmd"

# ``pl.at()`` kwargs whose AST node is stashed verbatim on ``_AtKwargState`` for
# later resolution (duplicate-checked, then resolved by ``_parse_at_meta``).
# Maps the kwarg name to its ``_AtKwargState`` field.
_AT_STASH_KWARGS = {
    "deps": "deps_kw",
    "no_dep_args": "no_dep_args_kw",
    "dumps": "dumps_kw",
}


def _split_spmd_for_loop_name_hints(name_hint: str) -> tuple[str, str]:
    """Map one ``for i in pl.spmd(..., name_hint=...)`` hint to Spmd vs InCore names.

    The for-form auto-wraps an inner ``InCoreScopeStmt``; ``OutlineIncoreScopes``
    names the AIC kernel from that inner ``name_hint_``. The outer ``SpmdScopeStmt``
    hint is consumed by ``OutlineClusterScopes`` (``_spmd_`` suffix on the wrapper).

    - ``name_hint="q_proj"`` → Spmd ``q_proj_spmd``, InCore ``q_proj``
    - ``name_hint="q_proj_spmd"`` → Spmd ``q_proj_spmd``, InCore ``q_proj``
    """
    if not name_hint:
        return "", ""
    if name_hint.endswith(_SPMD_SCOPE_NAME_SUFFIX):
        return name_hint, name_hint[: -len(_SPMD_SCOPE_NAME_SUFFIX)]
    return f"{name_hint}{_SPMD_SCOPE_NAME_SUFFIX}", name_hint


class ASTParser:
    """Parses Python AST and builds IR using IRBuilder."""

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
        self.global_vars = global_vars or {}  # Track GlobalVars for cross-function calls
        self.gvar_to_func = gvar_to_func or {}  # Track parsed functions for type inference
        self.external_funcs: dict[str, ir.Function] = {}  # Track external functions referenced
        # Cache of return types derived from a callee body (for annotation-less
        # callees), keyed by the callee Function. Avoids re-walking the body on
        # every call site within one function's parse. Keying by the Function
        # object (not id()) keeps a strong reference, so there is no id-reuse
        # hazard from a collected-and-reallocated Function.
        self._derived_return_types_cache: dict[ir.Function, list[ir.Type]] = {}

        # Track context for handling yields and returns
        self.in_for_loop = False
        self.in_while_loop = False
        self.in_if_stmt = False
        self.current_if_builder = None
        self.current_loop_builder = None

        # Track loop kinds for break/continue validation
        self._loop_kind_stack: list[str] = []
        self._scope_kind_stack: list[ir.ScopeKind] = []
        # Depth of nested ``with pl.manual_scope():`` blocks. Used to gate the
        # ``deps=[var]`` kwarg recognition on kernel calls.
        self._manual_scope_depth: int = 0

        # Forward-sticky ``pl.dump_tag`` set (per function, reset at function
        # entry). Holds the bound Vars whose subsequent kernel-call uses get a
        # per-call ``dump_vars`` entry. See ``_handle_dump_tag``.
        self._dump_tagged_vars: list[Any] = []

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
        # Current function level (set during parse_function). Drives
        # context-scoped op constraints (e.g. pld.system.world_size is host-only).
        self._func_level: ir.Level | None = None
        # Current function type (set during parse_function). Used by markers
        # whose validity is scoped to a specific function type — e.g.
        # ``pl.dump_tag`` only makes sense in Orchestration functions.
        self._func_type: ir.FunctionType = ir.FunctionType.Opaque

        # Current function's auto_scope flag (set during parse_function). When
        # True (default) the compiler owns AUTO scope placement, so a hand-placed
        # AUTO `with pl.scope()` is rejected; set False to take control. MANUAL
        # scopes are allowed regardless (they are a dependency-semantics choice).
        self._func_auto_scope: bool = True

        # Pending comments keyed by 1-based line number, drained by parse_statement.
        # Each entry is (col_offset, text) so the parser can distinguish
        # tail-of-block comments (inside body indent) from outer-scope comments.
        self._pending_comments: dict[int, list[tuple[int, str]]] = pending_comments or {}

        # Declared pld.tensor.alloc_window_buffer names — globally unique across
        # all functions in a @pl.program (the decorator shares this set).
        self._alloc_window_buffer_names: set[str] = (
            alloc_window_buffer_names if alloc_window_buffer_names is not None else set()
        )

        # Cached arithmetic analyzer used to simplify symbolic slice extents at
        # construction time. One instance per parser amortises the sub-analyzer
        # setup across the many subscripts found in a typical function body.
        self._arith_analyzer: _arith.Analyzer | None = None

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
        self._func_name = func_name
        self._func_level = func_level
        # auto_scope rides in func_attrs (key "auto_scope"); absent ⇒ default True.
        self._func_auto_scope = bool((func_attrs or {}).get("auto_scope", True))
        self._func_type = func_type
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
        self._dump_tagged_vars: list[Any] = []

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

            # Parse function body. HOST SubWorkers carry pure-Python source
            # via ``inline_body`` and are not parsed as DSL.
            if inline_body is not None:
                self.builder.inline_stmt(inline_body, ir.InlineLanguage.Python, func_span)
            else:
                self._parse_body_siblings(func_def.body)
                self._discard_tail_block_comments(func_def.body, upper_line=None)

        # Exit function scope
        self.scope_manager.exit_scope()

        return f.get_result()

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
        leading = self._drain_pending_comments(stmt)

        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            # Bare-string expression — treat as comment text on the next stmt.
            # Prepend any `#` comments already collected for this line so the
            # printed order matches source order.
            doc_lines = stmt.value.value.splitlines() or [""]
            self._requeue_comments_after(stmt, [*leading, *doc_lines])
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
            self._requeue_comments_after(stmt, residue)

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
                self._discard_stale_pending_before(stmt)
            self.parse_statement(stmt)

    def _then_branch_tail_upper(self, stmt: ast.If) -> int:
        """Inclusive upper-bound line for the then-branch's tail-drop range.

        Points to the line just before the ``else:`` keyword when an else
        clause is present (so the then-branch tail does not extend into
        else-body leading comments), and falls back to the if-stmt's end line
        when there is no else clause.
        """
        if not stmt.orelse:
            return stmt.end_lineno or stmt.lineno
        else_line = self._find_else_keyword_line(stmt)
        if else_line is not None:
            return else_line - 1
        return stmt.orelse[0].lineno - 1

    def _find_else_keyword_line(self, stmt: ast.If) -> int | None:
        """Return the source line of the ``else:`` keyword for an if-stmt.

        Python's AST has no node for the ``else:`` keyword itself; its line can
        only be recovered by scanning the source text between the last then-body
        stmt and the first orelse stmt. Used to compute the exact upper bound of
        the then-branch for :meth:`_discard_tail_block_comments` so that
        comments physically inside the else branch (e.g. leading comments for
        the first else stmt) are not mistaken for then-branch tails.

        Returns ``None`` if not found (e.g. ``elif`` chains lack an ``else:``
        keyword line).
        """
        if not stmt.orelse:
            return None
        src = self.span_tracker.source_lines
        start = (stmt.body[-1].end_lineno or stmt.body[-1].lineno) + 1
        end = stmt.orelse[0].lineno  # exclusive
        for line_no in range(start, end):
            idx = line_no - 1
            if not (0 <= idx < len(src)):
                continue
            stripped = src[idx].lstrip()
            # Match `else:` (optionally followed by whitespace/comment) but not
            # `elif ...` (which is a separate else-branch chain lowered into
            # ast.If.orelse[0] and has no `else:` keyword).
            if stripped.startswith("else") and stripped[4:5] in (":", " ", "\t"):
                return line_no
        return None

    @staticmethod
    def _header_ast_end_line(stmt: ast.stmt) -> int:
        """Last AST line of a compound stmt's header, before the body's colon.

        For a wrapped multi-line header (e.g. ``for i in pl.range(\n  10,\n):``)
        this returns the line of the closing ``)``/``:``; for a simple
        single-line header it returns ``stmt.lineno``.

        Used to classify comments between ``stmt.lineno`` and ``body[0].lineno``:
        comments on a line ``<= header_ast_end`` belong to the header; comments
        on later lines are body-leading for ``body[0]``.
        """
        ends: list[int] = [stmt.lineno]
        if isinstance(stmt, ast.For):
            ends.append(stmt.target.end_lineno or stmt.target.lineno)
            ends.append(stmt.iter.end_lineno or stmt.iter.lineno)
        elif isinstance(stmt, (ast.While, ast.If)):
            ends.append(stmt.test.end_lineno or stmt.test.lineno)
        elif isinstance(stmt, ast.With):
            for item in stmt.items:
                ends.append(item.context_expr.end_lineno or item.context_expr.lineno)
                if item.optional_vars is not None:
                    ends.append(item.optional_vars.end_lineno or item.optional_vars.lineno)
        return max(ends)

    def _drain_pending_comments(self, stmt: ast.stmt) -> list[str]:
        """Collect pending comments whose line numbers fall at or before ``stmt``.

        For simple stmts, drain through ``stmt.end_lineno`` so trailing comments
        on the last logical line (e.g. ``y = 1  # note``) attach to the stmt.

        For compound stmts (``for``/``while``/``if``/``with``) with a non-empty
        body, the header can span multiple physical lines. We drain up to the
        line just before the first body stmt and split by column: comments at
        a column *less than* the body's indent are header-level and attach to
        the compound stmt; comments at the body's indent are left pending for
        the first body stmt to pick up.

        Returns comments in source order and removes header/simple entries from
        the pending map.
        """
        if not self._pending_comments:
            return []
        body = getattr(stmt, "body", None)
        leading: list[str] = []
        if isinstance(body, list) and body:
            header_end = body[0].lineno - 1
            header_ast_end = self._header_ast_end_line(stmt)
            body_col = body[0].col_offset
            for line in sorted(k for k in self._pending_comments if k <= header_end):
                if stmt.lineno <= line <= header_ast_end:
                    # Header-level comment: either an inline trailer on the
                    # header's first line (e.g. `for i in range(16):  # tiles`)
                    # or a continuation-line comment inside a wrapped multi-line
                    # header (e.g. `for i in pl.range(\n  10,  # comment\n):`).
                    # Attach to the compound stmt regardless of column.
                    leading.extend(text for _col, text in self._pending_comments.pop(line))
                    continue
                if line > header_ast_end:
                    # Past the header but before body[0] — this is a body-leading
                    # comment for body[0]. Leave pending for body[0] to pick up.
                    continue
                # line < stmt.lineno: pre-stmt leftover. Split by column —
                # shallower-than-body attaches to this compound stmt; same-or-
                # deeper stays pending (belongs to body).
                kept: list[tuple[int, str]] = []
                for col, text in self._pending_comments[line]:
                    if col < body_col:
                        leading.append(text)
                    else:
                        kept.append((col, text))
                if kept:
                    self._pending_comments[line] = kept
                else:
                    del self._pending_comments[line]
        else:
            end_line = stmt.end_lineno or stmt.lineno
            for line in sorted(k for k in self._pending_comments if k <= end_line):
                leading.extend(text for _col, text in self._pending_comments.pop(line))
        return leading

    def _requeue_comments_after(self, stmt: ast.stmt, comments: list[str]) -> None:
        """Re-enqueue ``comments`` onto the line after ``stmt`` ends.

        Used when a stmt does not produce IR (docstring reroute, bare ``pass``)
        but may have collected leading comments that must survive to the next
        stmt. No-op when ``comments`` is empty. Synthetic comments inherit the
        stmt's column offset so they are treated as body-level for tail-drop
        purposes.
        """
        if not comments:
            return
        next_line = (stmt.end_lineno or stmt.lineno) + 1
        col = stmt.col_offset
        entries = [(col, text) for text in comments]
        existing = self._pending_comments.setdefault(next_line, [])
        self._pending_comments[next_line] = [*entries, *existing]

    def _discard_tail_block_comments(self, body: list[ast.stmt], upper_line: int | None) -> None:
        """Drop pending tail comments inside a block's physical line range.

        Called after a block's body finishes parsing. Sweeps pending comments on
        lines in ``[body[0].lineno, upper_line]`` whose column is
        ``>= body[0].col_offset`` and drops them with a warning.

        ``upper_line`` must be the inclusive upper bound of the block's physical
        extent (typically the enclosing compound stmt's ``end_lineno``, or the
        line before the next sibling clause such as ``else:``). Passing ``None``
        means "no upper bound" — sweep all remaining pending comments from
        ``body_start`` onward; used for the function body (outermost scope).

        Bounding is necessary because ``_pending_comments`` is populated
        up-front from the entire source, so comments in yet-to-parse sibling
        blocks at the same indent would otherwise be swept in error.
        """
        if not body or not self._pending_comments:
            return
        block_col = body[0].col_offset
        block_start = body[0].lineno
        tail: list[tuple[int, str]] = []
        if upper_line is None:
            candidates = [k for k in self._pending_comments if k >= block_start]
        else:
            candidates = [k for k in self._pending_comments if block_start <= k <= upper_line]
        for line in sorted(candidates):
            remaining: list[tuple[int, str]] = []
            for col, text in self._pending_comments[line]:
                if col >= block_col:
                    tail.append((line, text))
                else:
                    remaining.append((col, text))
            if remaining:
                self._pending_comments[line] = remaining
            else:
                del self._pending_comments[line]
        self._emit_tail_warning(tail)

    def _discard_stale_pending_before(self, stmt: ast.stmt) -> None:
        """Drop pending comments on earlier lines at deeper indent than ``stmt``.

        When parsing advances to ``stmt``, any pending comment on a line
        ``< stmt.lineno`` whose column is strictly greater than
        ``stmt.col_offset`` physically lives inside a previous sibling block
        whose body has already closed (dedent). Those comments cannot attach to
        ``stmt`` (wrong indent level) and would otherwise be misattributed by
        the simple-stmt drain, so sweep them here as tail-of-block.

        This is the mechanism that catches tail comments in the "gap" between
        one block's body and the next outer-scope stmt — complementing the
        bounded :meth:`_discard_tail_block_comments` that runs at block exit.
        """
        if not self._pending_comments:
            return
        stale: list[tuple[int, str]] = []
        for line in sorted(k for k in self._pending_comments if k < stmt.lineno):
            remaining: list[tuple[int, str]] = []
            for col, text in self._pending_comments[line]:
                if col > stmt.col_offset:
                    stale.append((line, text))
                else:
                    remaining.append((col, text))
            if remaining:
                self._pending_comments[line] = remaining
            else:
                del self._pending_comments[line]
        self._emit_tail_warning(stale)

    @staticmethod
    def _emit_tail_warning(tail: list[tuple[int, str]]) -> None:
        """Emit a UserWarning summarizing dropped tail-of-block comments."""
        if not tail:
            return
        preview = ", ".join(f"line {line}: {text!r}" for line, text in tail[:3])
        if len(tail) > 3:
            preview += f", … (+{len(tail) - 3} more)"
        warnings.warn(
            f"Dropped {len(tail)} tail-of-block comment(s): {preview}. "
            "Tail-of-block comments are not preserved in IR; move them above a "
            "statement or into the outer scope to retain them.",
            UserWarning,
            stacklevel=3,
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

        # Printer round-trip: ``buf: pl.Ptr = pld.[tensor.]alloc_window_buffer(N)``.
        # The printer adds a ``pl.Ptr`` annotation for clarity; route the
        # annotated form through the same dedicated alloc parser as the
        # unannotated form.
        if is_ptr_type_annotation and _is_pld_call(stmt.value, "alloc_window_buffer"):
            self._parse_alloc_window_buffer_assignment(stmt.target, stmt.value)
            return

        # Printer round-trip: ``res: pl.Tuple[..., TASK_ID] = pl.submit(...)``.
        # The annotation is checked against the inferred Submit type below
        # via the standard ``override_type`` validation path; the single-LHS
        # Submit handler builds the Submit and binds it to ``var_name`` directly.
        if _is_pl_call(stmt.value, "submit"):
            if not isinstance(stmt.target, ast.Name):
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
            value_expr = self._build_submit_single_lhs_expr(stmt.value)
        elif _is_pl_call(stmt.value, "spmd_submit"):
            if not isinstance(stmt.target, ast.Name):
                raise ParserSyntaxError(
                    "Annotated assignment of pl.spmd_submit must target a simple variable name",
                    span=self.span_tracker.get_span(stmt.target),
                    hint="Use `result: pl.Tuple[..., TASK_ID] = pl.spmd_submit(self.kernel, core_num=N)`.",
                )
            value_expr = self._build_submit_single_lhs_expr(stmt.value, is_spmd=True)
        else:
            value_expr = self.parse_expression(stmt.value)

        # Validate annotation against inferred type; use annotation as override only for memref
        override_type = None
        if stmt.annotation is not None:
            # Skip annotations the resolver can't handle:
            # - String forward refs (e.g. "SomeType")
            # - pl.UnknownType (emitted by printer for unrepresentable types)
            # - Singleton marker types (pl.MemRefType / pl.Ptr / pld.WindowBufferType
            #   / pld.CommCtxType): no shape/dtype to validate; the Var's type is
            #   fully determined by the RHS-inferred type.
            ann = stmt.annotation
            is_unresolvable = (isinstance(ann, ast.Constant) and isinstance(ann.value, str)) or (
                isinstance(ann, ast.Attribute)
                and isinstance(ann.value, ast.Name)
                and (
                    (ann.value.id == "pl" and ann.attr in ("UnknownType", "MemRefType", "Ptr"))
                    or (ann.value.id == "pld" and ann.attr in ("WindowBufferType", "CommCtxType"))
                )
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

    def parse_assignment(self, stmt: ast.Assign) -> None:  # noqa: PLR0912
        """Parse regular assignment: var = value or tuple unpacking.

        Args:
            stmt: Assign AST node
        """
        # Intercept ``buf = pld.[tensor.]alloc_window_buffer(...)``: the alloc
        # op derives its ``name`` kwarg from the LHS, and the LHS name must be
        # globally unique within the @pl.program.
        if len(stmt.targets) == 1 and _is_pld_call(stmt.value, "alloc_window_buffer"):
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
                        self.parse_yield_assignment(target, stmt.value)
                        return

                # ``out, tid = pl.submit(self.kernel, *args, deps=[...])`` —
                # manual_scope submit construct (desugars to a single Call
                # with an augmented Tuple{<kernel return>, TaskId} type).
                if _is_pl_call(stmt.value, "submit"):
                    self._parse_submit_assignment(target, stmt.value)
                    return
                # ``out, tid = pl.spmd_submit(self.kernel, *args, core_num=N,
                # sync_start=..., deps=[...])`` — SPMD task launch. Same desugar
                # as pl.submit, but the Submit carries the SPMD launch spec.
                if _is_pl_call(stmt.value, "spmd_submit"):
                    self._parse_submit_assignment(target, stmt.value, is_spmd=True)
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

                # ``result = pl.submit(self.kernel, ...)`` — single-LHS bind of
                # a Submit to a Tuple-typed Var. Round-trips the printer's
                # single-LHS form for a Submit whose AssignStmt LHS is a fresh
                # tuple temp (e.g. after a pass rewrite). The unpacking form
                # ``out, tid = pl.submit(...)`` continues to go through
                # ``_parse_submit_assignment``.
                if _is_pl_call(stmt.value, "submit"):
                    self._parse_submit_single_lhs(target, stmt.value)
                    return
                if _is_pl_call(stmt.value, "spmd_submit"):
                    self._parse_submit_single_lhs(target, stmt.value, is_spmd=True)
                    return

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

            # Handle subscript-write: dst[i:i+64, j:j+64] = src
            if isinstance(target, ast.Subscript):
                self._parse_subscript_assignment(target, stmt.value)
                return

        raise ParserSyntaxError(
            f"Unsupported assignment: {ast.unparse(stmt)}",
            span=self.span_tracker.get_span(stmt),
            hint="Use simple variable assignments or tuple unpacking with pl.yield_()",
        )

    def _bind_unpack_target(self, target: ast.expr, value_expr: ir.Expr, span: ir.Span) -> None:
        """Bind a tuple-unpacking target (a Name or a nested Tuple) to an IR expr.

        Used by the ``pl.submit(...)`` desugaring so a multi-output kernel can
        be unpacked as ``(a, b), tid = pl.submit(...)``.
        """
        if isinstance(target, ast.Name):
            var = self._assign_or_let(target.id, value_expr, span)
            self.scope_manager.define_var(target.id, var, span=span)
            return
        if isinstance(target, ast.Tuple):
            tuple_var = self.builder.let("_tuple_tmp", value_expr, span=span)
            for i, elt in enumerate(target.elts):
                item_expr = ir.TupleGetItemExpr(tuple_var, i, span)
                self._bind_unpack_target(elt, item_expr, span)
            return
        raise ParserSyntaxError(
            f"Tuple unpacking target must be a variable name or nested tuple, got {ast.unparse(target)}",
            span=span,
            hint="Use simple variable names in tuple unpacking: a, b, c = func()",
        )

    def _parse_submit_assignment(self, target: ast.Tuple, call: ast.Call, *, is_spmd: bool = False) -> None:
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
        span = self.span_tracker.get_span(call)
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

        call_expr = self._parse_kernel_call(
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
        submit_var = self.builder.let("_submit_tmp", call_expr, span=span)
        for i, elt in enumerate(out_names):
            self._bind_unpack_target(elt, ir.TupleGetItemExpr(submit_var, i, span), span)
        task_id_expr = ir.TupleGetItemExpr(submit_var, n_outs, span)
        tid_var = self._assign_or_let(tid_target.id, task_id_expr, span)
        self.scope_manager.define_var(tid_target.id, tid_var, span=span)

    def _build_submit_single_lhs_expr(self, call: ast.Call, *, is_spmd: bool = False) -> ir.Expr:
        """Build the ``ir.Submit`` expression for the single-LHS form
        ``result = pl.submit(self.kernel, ...)`` (or ``pl.spmd_submit`` when
        ``is_spmd=True``) without binding it.

        Separated from :meth:`_parse_submit_single_lhs` so the annotated
        assignment path can feed the inferred Submit type through the
        standard annotation-consistency / ``override_type`` validation flow
        that all other RHS expressions go through.
        """
        construct = "pl.spmd_submit" if is_spmd else "pl.submit"
        span = self.span_tracker.get_span(call)
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
        return self._parse_kernel_call(
            method_attr, call.args[1:], call.keywords, span, as_submit=True, as_spmd=is_spmd
        )

    def _parse_submit_single_lhs(self, target: ast.Name, call: ast.Call, *, is_spmd: bool = False) -> None:
        """Parse the bare single-LHS form ``result = pl.submit(self.kernel, ...)``
        (or ``pl.spmd_submit`` when ``is_spmd=True``).

        Used by :meth:`parse_assignment` (no annotation). The annotated form
        ``result: pl.Tuple[..., TASK_ID] = pl.submit(...)`` builds the Submit
        via :meth:`_build_submit_single_lhs_expr` and runs the regular
        annotation-consistency flow before binding.
        """
        span = self.span_tracker.get_span(call)
        call_expr = self._build_submit_single_lhs_expr(call, is_spmd=is_spmd)
        var = self._assign_or_let(target.id, call_expr, span)
        self.scope_manager.define_var(target.id, var, span=span)

    def _parse_alloc_window_buffer_assignment(self, target: ast.expr, value: ast.Call) -> None:
        """Parse ``buf = pld.[tensor.]alloc_window_buffer(size)``.

        Parser-only concerns (everything else delegates to the DSL wrapper /
        IR builder / C++ deducer via :func:`invoke_dsl`):

        - LHS must be a single ``ast.Name``.
        - That name must be globally unique within the ``@pl.program``.
        - User can't pass ``name=`` (it's parser-injected from the LHS).
        """
        span = self.span_tracker.get_span(value)

        if not isinstance(target, ast.Name):
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer must be assigned to a single variable name "
                f"(got '{ast.unparse(target)}')",
                span=span,
                hint="Use 'buf = pld.tensor.alloc_window_buffer(...)' (no tuple unpacking, no subscripts)",
            )

        name = target.id

        if name in self._alloc_window_buffer_names:
            raise ParserSyntaxError(
                f"pld.tensor.alloc_window_buffer name '{name}' is already declared in this program",
                span=span,
                hint="Each window buffer must have a globally unique name across all functions",
            )

        args = [self._parse_op_positional_arg(a) for a in value.args]
        user_kwargs = self._parse_op_kwargs(value)
        if "name" in user_kwargs:
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer 'name' kwarg cannot be passed explicitly — "
                f"it is auto-derived from the assignment LHS ('{name}')",
                span=span,
                hint="Drop the 'name=...' kwarg; the LHS variable name becomes the buffer name",
            )

        # Route through invoke_dsl — same path as _dispatch_op. Arity, size
        # type, unknown kwargs are validated by the DSL wrapper / IR / C++.
        alloc_call = invoke_dsl(
            _dsl_pld.alloc_window_buffer,
            args,
            {**user_kwargs, "name": name},
            span,
        )

        self._alloc_window_buffer_names.add(name)
        var = self._assign_or_let(name, alloc_call, span)
        self.scope_manager.define_var(name, var, span=span)

    def _parse_subscript_assignment(self, target: ast.Subscript, value_node: ast.expr) -> None:
        """Desugar ``dst[<slices...>] = src`` to ``dst = pl.assemble(dst, src, offsets)``.

        This sugar is parser-time only and therefore only meaningful before
        ConvertToSSA: the rewrite rebinds ``dst``, which strict-SSA forbids.
        """
        span = self.span_tracker.get_span(target)

        if self.scope_manager.strict_ssa:
            raise UnsupportedFeatureError(
                "Subscript-write syntax 'dst[...] = src' is only supported before SSA conversion",
                span=span,
                hint="Rewrite as an explicit pl.assemble(...) call, or remove strict_ssa=True",
            )

        if not isinstance(target.value, ast.Name):
            raise ParserSyntaxError(
                f"Subscript-write target must be a variable name, got {ast.unparse(target.value)}",
                span=span,
                hint="Assign through a named Tensor/Tile, e.g. 'dst[...] = src'",
            )

        var_name = target.value.id
        base_expr = self.parse_expression(target.value)
        base_type = base_expr.type

        if isinstance(base_type, ir.ArrayType):
            self._parse_array_subscript_assignment(target, base_expr, var_name, value_node, span)
            return

        # ``base_kind`` is the *kind* class (Tensor or Tile) — not the concrete
        # subclass — so the source-type check below accepts any tensor-shaped
        # sibling (``DistributedTensorType`` for tensor target, future tile
        # subclasses for tile target). Without this, mixed-kind writes such as
        # ``dist_win[i:j, :] = plain_tensor_src`` would be rejected even though
        # the underlying ``tensor.assemble`` deducer accepts both via
        # ``AsTensorTypeLike``.
        if isinstance(base_type, ir.TensorType):
            kind_name = "tensor"
            assemble_op = ir_op.tensor.assemble
            base_kind: type = ir.TensorType
        elif isinstance(base_type, ir.TileType):
            kind_name = "tile"
            assemble_op = ir_op.tile.assemble
            base_kind = ir.TileType
        else:
            raise ParserTypeError(
                f"Subscript-write requires a Tensor, Tile, or Array target, got {type(base_type).__name__}",
                span=span,
                hint="Subscript-write 'dst[...] = src' is only supported for Tensor, Tile, and Array",
            )

        indices = self._normalize_subscript_indices(target, span)
        offsets, extents, drop_dims = self._build_assemble_offsets_and_extents(
            indices, base_type.shape, span, kind_name
        )
        dropped = set(drop_dims)
        kept_dims = [d for d in range(len(extents)) if d not in dropped]
        natural_rank = len(kept_dims)
        # The tile.slice deducer floors a sub-2D result back to 2D by prepending
        # unit axes, so a tile rhs obtained by reading an indexed view is 2D even
        # when the "natural" rank-reduced rank is < 2 — accept that shape here.
        floored_rank = max(2, natural_rank) if isinstance(base_type, ir.TileType) else natural_rank

        source_expr = self.parse_expression(value_node)
        source_type = source_expr.type
        # Union form gives pyright the narrowing it needs to use ``.shape`` below;
        # ``base_kind`` is the runtime kind discriminator (Tensor vs Tile).
        if not (isinstance(source_type, (ir.TensorType, ir.TileType)) and isinstance(source_type, base_kind)):
            raise ParserTypeError(
                f"Subscript-write source must also be a {kind_name}, got {type(source_type).__name__}",
                span=span,
                hint=f"The rhs of 'dst[...] = src' must be a {kind_name} of matching shape",
            )

        src_rank = len(source_type.shape)
        if src_rank not in (natural_rank, floored_rank):
            expected = (
                f"{natural_rank}D" if natural_rank == floored_rank else f"{natural_rank}D or {floored_rank}D"
            )
            raise ParserTypeError(
                f"Subscript-write source must be {expected} to match the rank-reduced "
                f"{kind_name} window, got {src_rank}D",
                span=span,
                hint="A scalar index removes its axis from the lhs window; the rhs rank must "
                "match the remaining axes (tiles may also keep leading unit axes to stay 2D)",
            )

        # Constant extents the source axes must match — with leading unit axes
        # prepended when the source carries the tile 2D-floor's padding.
        lead_units = src_rank - natural_rank
        expected_extents = [1] * lead_units + [extents[d] for d in kept_dims]
        # A narrowed source (static_shape padded for ISA alignment, valid_shape
        # carrying the logical extent — same pattern pl.store accepts on the
        # tile path) is allowed when its valid_shape matches the window.
        source_valid_shape = _get_source_valid_shape(source_type)
        for src_axis, want in enumerate(expected_extents):
            requested_const = _fold_const_slice_extent(want, 0)
            source_const = _fold_const_slice_extent(source_type.shape[src_axis], 0)
            if requested_const is None or source_const is None:
                continue
            if requested_const == source_const:
                continue
            if source_valid_shape is not None:
                valid_const = _fold_const_slice_extent(source_valid_shape[src_axis], 0)
                # Dynamic valid_shape (folds to None) — parser cannot disprove
                # the match, so trust it, symmetric to dynamic slot / dynamic
                # static_shape above. Constant valid_shape only accepts on a
                # match.
                if valid_const is None or valid_const == requested_const:
                    continue
            raise ParserTypeError(
                f"Subscript-write shape mismatch on source axis {src_axis}: "
                f"window expects {requested_const} elements, source has {source_const}",
                span=span,
                hint="Make the source's static_shape or valid_shape match the "
                "rank-reduced lhs window, or adjust the slice bounds",
            )

        if src_rank != len(extents):
            source_expr = self._lift_subscript_write_source_rank(
                source_expr,
                source_type,
                source_valid_shape,
                drop_dims,
                extents,
                lead_units,
                kind_name,
                span,
            )

        assemble_call = assemble_op(base_expr, source_expr, offsets, span=span)
        var = self._assign_or_let(var_name, assemble_call, span)
        self.scope_manager.define_var(var_name, var, span=span)

    def _lift_subscript_write_source_rank(
        self,
        source_expr: ir.Expr,
        source_type: ir.TensorType | ir.TileType,
        source_valid_shape: list[ir.Expr] | None,
        drop_dims: list[int],
        extents: list[int | ir.Expr],
        lead_units: int,
        kind_name: str,
        span: ir.Span,
    ) -> ir.Expr:
        """Reshape source up to the full-rank target window for rank-reducing writes.

        Inserts unit dims at the ``drop_dims`` positions so the assemble's
        source/window ranks match — mirrors the implicit reshape numpy does on
        write. Reshape target always derives from the source's *own* padded
        ``static_shape`` (minus any tile 2D-floor lead unit axes), so the
        reshape product check passes whether or not the source carries a
        narrower ``valid_shape`` (issue #1509).

        When the source carries an explicit narrower ``valid_shape``, that
        narrowing is carried forward via ``tensor.reshape``'s optional 3rd
        argument; ``tile.reshape`` has no such parameter, so a narrowed tile
        + rank-lift combo is rejected here (the user should switch to
        ``pl.store``).
        """
        # "Narrowed" means valid_shape is actually smaller than shape; a view
        # with valid_shape == shape (e.g. a tile's canonical implicit view) is
        # semantically equivalent to no view and must NOT be treated as a
        # narrow source — otherwise canonical 1D-tile rank-lift writes would
        # hit the issue #1509 path meant for ISA-padded sources.
        is_narrowed = source_valid_shape is not None and not _shape_exprs_match(
            source_type.shape, source_valid_shape
        )

        # tile.reshape has no valid_shape parameter, so a narrowed tile +
        # rank-lift would silently drop the narrowing. Reject upfront and point
        # users to pl.store.
        if is_narrowed and kind_name != "tensor":
            raise UnsupportedFeatureError(
                "Subscript-write with rank reduction is not supported when "
                "the source tile carries an explicit valid_shape — "
                "tile.reshape cannot carry valid_shape across the rank lift.",
                span=span,
                hint="Write the tile via pl.store directly, or use slice "
                "indices on every axis instead of scalar indices to avoid "
                "rank reduction.",
            )

        drop_set = set(drop_dims)
        unit: ir.Expr = ir.ConstInt(1, DataType.INDEX, span)

        def _lift_shape(seq: Sequence[int | ir.Expr]) -> list[int | ir.Expr]:
            # Skip the tile 2D-floor lead unit axes — they were padded into seq
            # only to satisfy the tile ≥2D invariant; the lift reconstructs the
            # target rank from the kept positions and the drop_dims unit fillers.
            it = iter(list(seq)[lead_units:])
            return [unit if d in drop_set else next(it) for d in range(len(extents))]

        reshape_op = ir_op.tensor.reshape if kind_name == "tensor" else ir_op.tile.reshape
        reshape_args: list[list[int | ir.Expr]] = [_lift_shape(source_type.shape)]
        if is_narrowed:
            # Tensor path with a genuinely narrower valid_shape — carry it via
            # reshape's optional 3rd arg so the narrowing survives the rank
            # lift (issue #1509). Asserted non-None by the is_narrowed guard.
            assert source_valid_shape is not None
            reshape_args.append(_lift_shape(source_valid_shape))
        return reshape_op(source_expr, *reshape_args, span=span)

    def _parse_array_subscript_assignment(
        self,
        target: ast.Subscript,
        base_expr: ir.Expr,
        var_name: str,
        value_node: ast.expr,
        span: ir.Span,
    ) -> None:
        """Desugar ``arr[i] = value`` for an Array target.

        Lowers to ``arr = array.update_element(arr, i, value)`` — SSA-functional
        update. Like the Tensor/Tile sugar, this rebinds ``arr`` and is therefore
        only valid before SSA conversion.
        """
        index_node = target.slice
        if isinstance(index_node, ast.Tuple):
            raise ParserTypeError(
                "Array subscript-write requires a single index (rank-1 in v1)",
                span=span,
                hint="Use arr[i] = v, not arr[i, j] = v",
            )
        index_expr = self.parse_expression(index_node)
        value_expr = self.parse_expression(value_node)
        # Bare int literals come back as ConstInt(INDEX); the update_element
        # deducer requires the value dtype to match the array's element dtype.
        # Retag the literal so users can write `arr[i] = 7` without explicit casts.
        array_type = base_expr.type
        if isinstance(value_expr, ir.ConstInt) and isinstance(array_type, ir.ArrayType):
            value_expr = ir.ConstInt(value_expr.value, array_type.dtype, value_expr.span)
        update_call = ir.create_op_call("array.update_element", [base_expr, index_expr, value_expr], {}, span)
        var = self._assign_or_let(var_name, update_call, span)
        self.scope_manager.define_var(var_name, var, span=span)

    def _build_assemble_offsets_and_extents(
        self,
        indices: list[ast.expr],
        target_shape: Sequence[ir.Expr],
        span: ir.Span,
        kind_name: str,
    ) -> tuple[list[int | ir.Expr], list[int | ir.Expr], list[int]]:
        """Parse a subscript-write index list into per-axis offsets/extents plus drop_dims.

        Same numpy-style rules as the read path: ``a[lower:upper]`` writes the
        ``lower``..``upper`` window of that axis (offset ``lower``, extent
        ``upper - lower``, defaulting to the full axis); ``a[i]`` writes a
        unit-extent window at offset ``i`` and adds that axis to ``drop_dims``
        (the source is rank-reduced over those axes); dims past ``indices`` are
        implicit ``:``. ``offsets``/``extents`` are always full-rank. An
        all-scalar *full-rank* index (a true element write) and ``step`` are
        still rejected.
        """
        rank = len(target_shape)
        if len(indices) > rank:
            raise ParserTypeError(
                f"{kind_name.capitalize()} subscript-write has {len(indices)} indices but the "
                f"{kind_name} is {rank}D",
                span=span,
                hint=f"Provide at most {rank} indices for a {rank}D {kind_name}",
            )
        if len(indices) == rank and not any(isinstance(idx, ast.Slice) for idx in indices):
            raise UnsupportedFeatureError(
                f"Element-write 'dst[i, j] = scalar' is not supported for {kind_name} targets",
                span=span,
                hint="Use slice indices (e.g. 'dst[i:i+1, j:j+1] = tile_1x1') or index fewer axes",
            )

        offsets: list[int | ir.Expr] = []
        extents: list[int | ir.Expr] = []
        drop_dims: list[int] = []
        for dim_idx, idx in enumerate(indices):
            if not isinstance(idx, ast.Slice):
                offsets.append(self.parse_expression(idx))
                extents.append(1)
                drop_dims.append(dim_idx)
                continue
            if idx.step is not None:
                raise UnsupportedFeatureError(
                    f"Slice step is not supported in {kind_name} subscript-write",
                    span=span,
                    hint="Use 'dst[start:stop] = src' without step",
                )
            lower: int | ir.Expr = 0 if idx.lower is None else self.parse_expression(idx.lower)
            upper: int | ir.Expr = (
                target_shape[dim_idx] if idx.upper is None else self.parse_expression(idx.upper)
            )
            offsets.append(lower)
            extents.append(self._build_slice_shape_expr(upper, lower))
        for dim_idx in range(len(indices), rank):
            offsets.append(0)
            extents.append(target_shape[dim_idx])
        return offsets, extents, drop_dims

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

    _VALID_ITERATORS = {"range", "parallel", "unroll", "pipeline", "while_", "spmd"}
    _ITERATOR_ERROR = (
        "For loop must use pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), pl.while_(), or pl.spmd()"
    )
    _ITERATOR_HINT = (
        "Use pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), pl.while_(), or pl.spmd() as the iterator"
    )

    def _validate_for_loop_iterator(self, stmt: ast.For) -> tuple[ast.Call, str]:
        """Validate that for loop uses pl.range(), pl.parallel(), pl.unroll(), pl.pipeline(), or pl.while_().

        Returns:
            Tuple of (call_node, iterator_type) where iterator_type is
            "range", "parallel", "unroll", "pipeline", or "while_"
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

        # Handle pl.spmd() loop form — auto-outlines into Spmd(InCore(body)).
        if iterator_type == "spmd":
            self._parse_spmd_for_loop(stmt, iter_call)
            return

        loop_var_name, iter_args_node, is_simple_for = self._parse_for_loop_target(stmt)
        range_args = self._parse_range_call(iter_call, iterator_type)

        if is_simple_for and range_args["init_values"]:
            raise ParserSyntaxError(
                "For loop target must be a tuple when init_values is provided",
                span=self.span_tracker.get_span(stmt.target),
                hint="Use: for i, (var1,) in pl.range(n, init_values=(val1,)) to include iter_args",
            )

        self._require_yield_lhs_for_init_values(stmt, bool(range_args["init_values"]))

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
        chunk_policy_str = range_args.get("chunk_policy", "guarded")
        if chunk_expr is not None:
            self._validate_chunk_args(chunk_expr, range_args["init_values"], iter_call)

        # Validate stage= on pl.pipeline() and merge into attrs as "pipeline_stages".
        # stage= is required on pl.pipeline() and forbidden everywhere else.
        pipeline_stages: int | None = None
        stage_expr = range_args.get("stage")
        if iterator_type == "pipeline":
            if stage_expr is None:
                raise ParserSyntaxError(
                    "pl.pipeline() requires stage= (positive integer)",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use pl.pipeline(stop, stage=F).",
                )
            if chunk_expr is not None:
                raise ParserSyntaxError(
                    "stage= and chunk= are mutually exclusive on pl.pipeline()",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Drop chunk= when pipelining — pl.pipeline replicates the body at tile level.",
                )
            if not _is_const_int(stage_expr):
                raise ParserSyntaxError(
                    "pl.pipeline() stage must be a compile-time constant integer",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use an integer literal: stage=4",
                )
            stage_val = _const_int_value(stage_expr)
            if stage_val is None or stage_val < 1:
                raise ParserSyntaxError(
                    f"pl.pipeline() stage must be >= 1, got {stage_val}",
                    span=self.span_tracker.get_span(iter_call),
                    hint="Use a positive integer for stage: stage=4",
                )
            pipeline_stages = stage_val
        elif stage_expr is not None:
            raise ParserSyntaxError(
                f"stage= is only supported on pl.pipeline(), not pl.{iterator_type}()",
                span=self.span_tracker.get_span(iter_call),
                hint="Use pl.pipeline() for software pipelining.",
            )

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

        attrs_dict: dict[str, object] | None = range_args.get("attrs") or None
        if pipeline_stages is not None:
            attrs_dict = dict(attrs_dict) if attrs_dict else {}
            attrs_dict["pipeline_stages"] = pipeline_stages
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
                self._parse_body_siblings(stmt.body)
                self._discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                assert self._current_yield_vars is not None  # Guaranteed by _yield_tracking_scope
                loop_output_vars = self._current_yield_vars[:]

            # Yield-LHS names if present; else synthetic `_out` names (pre-SSA).
            if not is_simple_for and range_args["init_values"]:
                if loop_output_vars:
                    for rv_name in loop_output_vars:
                        loop.return_var(rv_name)
                else:
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
                "chunk=... loops are only valid inside "
                "with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):",
                span=self.span_tracker.get_span(iter_call),
                hint="Wrap the loop in 'with pl.at(level=pl.Level.CORE_GROUP, "
                "optimizations=[pl.auto_chunk]):' or remove the chunk= argument.",
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

    _ITERATOR_KEYWORDS = {
        "range": ("init_values", "chunk", "chunk_policy", "attrs"),
        "parallel": ("init_values", "chunk", "chunk_policy", "attrs"),
        "unroll": ("init_values", "attrs"),
        "pipeline": ("init_values", "stage", "attrs"),
    }

    def _parse_range_keyword(self, keyword: ast.keyword, result: dict[str, Any], iterator_type: str) -> None:
        """Parse a single keyword argument from a range-like iterator call."""
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
        elif keyword.arg == "stage":
            result["stage"] = self.parse_expression(keyword.value)
        elif keyword.arg == "chunk_policy":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                _VALID_CHUNK_POLICIES = {"leading_full", "guarded"}
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
            supported = self._ITERATOR_KEYWORDS.get(iterator_type, self._ITERATOR_KEYWORDS["range"])
            raise ParserSyntaxError(
                f"Unknown keyword argument '{keyword.arg}' in pl.{iterator_type}()",
                span=self.span_tracker.get_span(keyword),
                hint=f"Supported keywords for pl.{iterator_type}(): {', '.join(supported)}",
            )

    def _parse_range_call(self, call: ast.Call, iterator_type: str = "range") -> dict[str, Any]:
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
                span=self.span_tracker.get_span(call),
                hint=f"Provide at least the stop value: pl.{iterator_type}(10) or pl.{iterator_type}(0, 10)",
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
            "chunk_policy": "guarded",
            "unroll": None,
            "attrs": {},
        }
        for keyword in call.keywords:
            self._parse_range_keyword(keyword, result, iterator_type)

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
    def _is_dsl_call(node: ast.AST, func_name: str) -> bool:
        """Check if `node` is a call to a named DSL function (`pl.func_name(...)`
        or bare `func_name(...)`). Accepts either an `ast.Expr` statement (the
        body-stmt form, e.g. a bare `pl.cond(...)` line) or an expression node
        directly (e.g. the RHS of an `ast.Assign`).
        """
        if isinstance(node, ast.Expr):
            node = node.value
        if not isinstance(node, ast.Call):
            return False
        func = node.func
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

    def _handle_dump_tag(self, stmt: ast.Expr) -> None:
        """Handle ``pl.dump_tag(<name>)`` at statement position.

        ``pl.dump_tag(t)`` is the declarative per-tensor dump marker. It records
        the bound Var so that every *subsequent* kernel dispatch consuming that
        exact Var gets it merged into the dispatch's ``dump_vars`` attr (Call or
        Submit; see :meth:`_parse_kernel_call`). No IR statement is emitted; the
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

        if self._func_type not in (ir.FunctionType.Orchestration, ir.FunctionType.Inline):
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
        if not any(var is t for t in self._dump_tagged_vars):
            self._dump_tagged_vars.append(var)

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

    @classmethod
    def _find_first_bare_yield(cls, stmts: list[ast.stmt]) -> ast.Expr | None:
        """First bare `pl.yield_(...)` expression-statement in `stmts`, or None.

        Recurses into `if/else` branches and `with` bodies at the same scope
        (e.g. `with pl.at(...): pl.yield_(...)` inside a loop body). Does NOT
        descend into nested For/While bodies — yields inside an inner loop
        bind to that inner loop's return_vars, not the outer one.
        """
        for s in stmts:
            if isinstance(s, ast.Expr) and cls._is_dsl_call(s, "yield_"):
                return s
            if isinstance(s, ast.If):
                found = cls._find_first_bare_yield(s.body) or cls._find_first_bare_yield(s.orelse)
                if found is not None:
                    return found
            elif isinstance(s, ast.With):
                found = cls._find_first_bare_yield(s.body)
                if found is not None:
                    return found
        return None

    def _require_yield_lhs_for_init_values(self, stmt: ast.For, init_values_present: bool) -> None:
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
        first_bare = self._find_first_bare_yield(stmt.body)
        if first_bare is None:
            return
        raise ParserSyntaxError(
            "Loop with init_values requires an assignment-form pl.yield_(...) "
            "— the LHS supplies the post-loop binding name. Bare pl.yield_(...) "
            "is rejected here.",
            span=self.span_tracker.get_span(first_bare),
            hint="Use yield-LHS form: `x_next = pl.yield_(updated)`. Then refer to `x_next` after the loop.",
        )

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
            # Validate body (skip first statement which is pl.cond()).
            for body_stmt in stmt.body[1:]:
                if self._is_cond_call(body_stmt):
                    raise ParserSyntaxError(
                        "pl.cond() can only be the first statement in a pl.while_() loop body",
                        span=self.span_tracker.get_span(body_stmt),
                        hint="Remove this pl.cond() - condition is already specified at the start",
                    )
            self._parse_body_siblings(stmt.body[1:])
            self._discard_tail_block_comments(stmt.body[1:], upper_line=stmt.end_lineno)

            assert self._current_yield_vars is not None  # Guaranteed by _yield_tracking_scope
            return self._current_yield_vars[:]

    def _register_while_outputs(self, loop: Any, loop_output_vars: list[str]) -> None:
        """Register output variables from pl.while_() loop.

        Mirrors `parse_for_loop`: the post-loop binding name is the yield-LHS,
        not the header tuple. Header-tuple names are loop-scoped only.
        """
        loop_result = loop.get_result()
        if hasattr(loop_result, "return_vars") and loop_result.return_vars and loop_output_vars:
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

        # init_values is always non-empty for pl.while_ (enforced earlier).
        self._require_yield_lhs_for_init_values(stmt, True)

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

            # Yield-LHS names if present; else synthetic `_out` names (pre-SSA).
            if loop_output_vars:
                for var_name in loop_output_vars:
                    loop.return_var(var_name)
            else:
                for iter_arg_node in iter_args_node.elts:
                    assert isinstance(iter_arg_node, ast.Name)
                    loop.return_var(f"{iter_arg_node.id}_out")

            # Enforce Bool-typed condition after iter_args and return_vars are both
            # set up, so that the while_loop context manager's __exit__ validation
            # does not mask this error with an arity mismatch.
            self._check_condition_is_bool(condition, "while", self.span_tracker.get_span(stmt.body[0]))

            self.scope_manager.exit_scope(leak_vars=False)
            self._loop_kind_stack.pop()
            self.in_for_loop = prev_in_for_loop
            self.in_while_loop = prev_in_while_loop
            self.current_loop_builder = prev_loop_builder

        # Register output variables
        self._register_while_outputs(loop, loop_output_vars)

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
        self._check_condition_is_bool(condition, "while", span)
        prev_loop_builder = self.current_loop_builder
        prev_in_for_loop = self.in_for_loop
        prev_in_while_loop = self.in_while_loop

        with self.builder.while_loop(condition, span) as loop:
            self.current_loop_builder = loop
            self.in_while_loop = True
            self._loop_kind_stack.append("while")
            self.scope_manager.enter_scope("while")

            # Parse body statements
            self._parse_body_siblings(stmt.body)
            self._discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)

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
        self._check_condition_is_bool(condition, "if", span)

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
                self._parse_body_siblings(stmt.body)
                self._discard_tail_block_comments(stmt.body, upper_line=self._then_branch_tail_upper(stmt))
                self.scope_manager.exit_scope(leak_vars=should_leak)

                # Parse else branch if present
                if stmt.orelse:
                    if_builder.else_()
                    self.scope_manager.enter_scope("else")
                    self._parse_body_siblings(stmt.orelse)
                    self._discard_tail_block_comments(stmt.orelse, upper_line=stmt.end_lineno)
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

    def _parse_at_kwargs(self, call: ast.Call) -> "_AtKwargState":
        """Extract level, role, AutoChunk request, split mode, deps, and name from pl.at(...).

        Supports both positional and keyword forms. Preferred new API uses the
        ``optimizations=[...]`` list with ``pl.split(...)`` and ``pl.auto_chunk``
        entries. The legacy ``optimization=`` and top-level ``split=`` kwargs
        are still accepted but emit a DeprecationWarning. Mixing the new
        ``optimizations=`` list with either deprecated kwarg is a hard error.

        Returns the populated :class:`_AtKwargState`. ``requests_auto_chunk`` is
        True when the resulting scope must be ``AutoInCore`` rather than
        ``InCore``. ``deps_kw`` carries the verbatim ``deps=`` keyword AST when
        present, so the caller can resolve it into ``Var`` refs once it has
        decided this scope opts into the ``manual_dep_edges`` path.
        """
        if len(call.args) > 2:
            raise ParserSyntaxError(
                f"pl.at() takes at most 2 positional arguments, got {len(call.args)}",
                span=self.span_tracker.get_span(call),
                hint="Use pl.at(level) or pl.at(level, role)",
            )

        state = _AtKwargState()
        if len(call.args) >= 1:
            state.level = extract_enum_value(call.args[0], LEVEL_MAP, "Level", "pl.Level")
        if len(call.args) >= 2:
            state.role = extract_enum_value(call.args[1], ROLE_MAP, "Role", "pl.Role")

        for kw in call.keywords:
            self._dispatch_at_keyword(kw, state)

        if state.level is None:
            raise ParserSyntaxError(
                "pl.at() requires a level argument",
                span=self.span_tracker.get_span(call),
                hint="Use pl.at(pl.Level.HOST) or pl.at(level=pl.Level.HOST)",
            )

        self._validate_at_kwarg_combinations(state)
        return state

    def _dispatch_at_keyword(self, kw: ast.keyword, state: "_AtKwargState") -> None:
        """Dispatch a single pl.at() keyword argument and update state."""
        if kw.arg == "level":
            if state.level is not None:
                raise ParserSyntaxError(
                    "pl.at() got multiple values for argument 'level'",
                    span=self.span_tracker.get_span(kw),
                )
            state.level = extract_enum_value(kw.value, LEVEL_MAP, "Level", "pl.Level")
        elif kw.arg == "role":
            if state.role is not None:
                raise ParserSyntaxError(
                    "pl.at() got multiple values for argument 'role'",
                    span=self.span_tracker.get_span(kw),
                )
            state.role = extract_enum_value(kw.value, ROLE_MAP, "Role", "pl.Role")
        elif kw.arg == "optimizations":
            self._handle_at_optimizations_kw(kw, state)
        elif kw.arg == "optimization":
            self._handle_at_legacy_optimization_kw(kw, state)
        elif kw.arg == "split":
            self._handle_at_legacy_split_kw(kw, state)
        elif kw.arg == "name_hint":
            state.name_hint = self._parse_scope_name_hint(kw.value, "pl.at()")
        elif kw.arg in _AT_STASH_KWARGS:
            self._stash_at_kwarg(kw, state)
        elif kw.arg is None:
            raise ParserSyntaxError(
                "Unsupported **kwargs in pl.at()",
                span=self.span_tracker.get_span(kw),
                hint="Use pl.at(level=pl.Level.HOST, role=pl.Role.SubWorker)",
            )
        else:
            raise ParserSyntaxError(
                f"Unknown keyword argument '{kw.arg}' in pl.at()",
                span=self.span_tracker.get_span(kw),
                hint=("Supported arguments: level, role, optimizations, deps, no_dep_args, dumps, name_hint"),
            )

    def _stash_at_kwarg(self, kw: ast.keyword, state: "_AtKwargState") -> None:
        """Stash a verbatim-kept ``pl.at()`` kwarg (``deps`` / ``no_dep_args`` /
        ``dumps``) onto ``state``, rejecting a duplicate."""
        assert kw.arg is not None  # caller dispatches here only for _AT_STASH_KWARGS keys
        attr = _AT_STASH_KWARGS[kw.arg]
        if getattr(state, attr) is not None:
            raise ParserSyntaxError(
                f"pl.at() got multiple values for argument '{kw.arg}'",
                span=self.span_tracker.get_span(kw),
            )
        setattr(state, attr, kw)

    def _handle_at_optimizations_kw(self, kw: ast.keyword, state: "_AtKwargState") -> None:
        if state.new_optimizations_kw is not None:
            raise ParserSyntaxError(
                "pl.at() got multiple values for argument 'optimizations'",
                span=self.span_tracker.get_span(kw),
            )
        state.new_optimizations_kw = kw
        (
            state.requests_auto_chunk,
            state.split_mode,
            state.split_slot_num,
        ) = self._parse_optimizations_list(kw.value)

    def _handle_at_legacy_optimization_kw(self, kw: ast.keyword, state: "_AtKwargState") -> None:
        if state.legacy_optimization_kw is not None:
            raise ParserSyntaxError(
                "pl.at() got multiple values for argument 'optimization'",
                span=self.span_tracker.get_span(kw),
            )
        state.legacy_optimization_kw = kw
        # Bare or called legacy optimizer always implies AutoChunk.
        state.requests_auto_chunk = True
        state.split_mode = self._parse_chunked_loop_optimizer(kw.value)

    def _handle_at_legacy_split_kw(self, kw: ast.keyword, state: "_AtKwargState") -> None:
        if state.legacy_split_kw is not None:
            raise ParserSyntaxError(
                "pl.at() got multiple values for argument 'split'",
                span=self.span_tracker.get_span(kw),
            )
        state.legacy_split_kw = kw
        state.split_mode = self._eval_split_mode(kw.value)

    def _validate_at_kwarg_combinations(self, state: "_AtKwargState") -> None:
        """Reject illegal kwarg combinations and emit DeprecationWarnings."""
        # Hard error when mixing new optimizations= with deprecated kwargs.
        if state.new_optimizations_kw is not None and (
            state.legacy_optimization_kw is not None or state.legacy_split_kw is not None
        ):
            offending = state.legacy_optimization_kw or state.legacy_split_kw
            assert offending is not None
            raise ParserSyntaxError(
                "Cannot mix 'optimizations=' with deprecated 'optimization=' or 'split=' kwargs in pl.at()",
                span=self.span_tracker.get_span(offending),
                hint="Use only optimizations=[pl.split(...), pl.auto_chunk] — drop the deprecated kwargs.",
            )

        # Preserve the pre-existing rule that the two deprecated kwargs cannot be
        # combined: legacy `optimization=` always implied AutoInCore + a baked-in
        # split, so combining it with legacy top-level `split=` was ambiguous.
        if state.legacy_optimization_kw is not None and state.legacy_split_kw is not None:
            raise ParserSyntaxError(
                "Cannot use both 'optimization' and 'split' in pl.at()",
                span=self.span_tracker.get_span(state.legacy_split_kw),
                hint="Use optimizations=[pl.auto_chunk, pl.split(...)] for AutoInCore + "
                "split, or optimizations=[pl.split(...)] for plain InCore + split.",
            )

        # Emit deprecation warnings for legacy kwargs (after mixing checks, so the
        # user sees the structural error first if both apply).
        if state.legacy_optimization_kw is not None:
            warnings.warn(
                "pl.at(optimization=pl.chunked_loop_optimizer[(...)]) is deprecated; "
                "use pl.at(optimizations=[pl.auto_chunk]) — combine with pl.split(...) "
                "if a split mode is needed.",
                DeprecationWarning,
                stacklevel=2,
            )
        if state.legacy_split_kw is not None:
            warnings.warn(
                "pl.at(split=...) is deprecated; use pl.at(optimizations=[pl.split(...)]).",
                DeprecationWarning,
                stacklevel=2,
            )

    def _parse_optimizations_list(
        self,
        value: ast.expr,
        *,
        owner: str = "pl.at",
        list_hint: str | None = None,
        entry_hint: str | None = None,
    ) -> tuple[bool, "ir.SplitMode | None", "int | None"]:
        """Parse ``optimizations=[...]`` for ``pl.at`` or ``pl.spmd``.

        Each entry must be one of:

        - ``pl.auto_chunk`` — request AutoInCore semantics.
        - ``pl.split(MODE)`` — set the cross-core split mode.

        Both fully qualified forms (``pl.optimizations.auto_chunk``,
        ``pl.optimizations.split(MODE)``) are also accepted.

        Args:
            owner: API name used in error messages (e.g. ``"pl.at"``, ``"pl.spmd"``).
            list_hint: Override hint when ``optimizations=`` is not a list literal.
            entry_hint: Override hint for unsupported list entries.

        Returns:
            Tuple ``(requests_auto_chunk, split_mode, split_slot_num)``.
        """
        if list_hint is None:
            list_hint = (
                "Use optimizations=[pl.split(pl.SplitMode.NONE)]."
                if owner == "pl.spmd"
                else "Use optimizations=[pl.split(pl.SplitMode.NONE)] or optimizations=[pl.auto_chunk]."
            )
        if entry_hint is None:
            entry_hint = (
                "Each entry must be pl.split(pl.SplitMode.X)."
                if owner == "pl.spmd"
                else "Each entry must be pl.auto_chunk or pl.split(pl.SplitMode.X)."
            )
        if not isinstance(value, ast.List):
            raise ParserSyntaxError(
                f"{owner}(optimizations=...) must be a list literal",
                span=self.span_tracker.get_span(value),
                hint=list_hint,
            )

        requests_auto_chunk = False
        split_mode: ir.SplitMode | None = None
        split_slot_num: int | None = None
        seen_auto_chunk = False
        seen_split = False

        for entry in value.elts:
            if self._is_pl_auto_chunk(entry):
                if seen_auto_chunk:
                    raise ParserSyntaxError(
                        "Duplicate 'pl.auto_chunk' in optimizations=[...]",
                        span=self.span_tracker.get_span(entry),
                    )
                seen_auto_chunk = True
                requests_auto_chunk = True
            elif (parsed := self._try_parse_pl_split(entry)) is not None:
                if seen_split:
                    raise ParserSyntaxError(
                        "Duplicate 'pl.split(...)' in optimizations=[...]",
                        span=self.span_tracker.get_span(entry),
                    )
                seen_split = True
                split_mode, slot_num = parsed
                if slot_num is not None and (
                    split_mode is None or split_mode == ir.SplitMode.NONE
                ):
                    raise ParserSyntaxError(
                        "pl.split(slot_num=...) is only valid with a cross-core split mode",
                        span=self.span_tracker.get_span(entry),
                        hint="Set mode to pl.SplitMode.UP_DOWN or pl.SplitMode.LEFT_RIGHT, "
                        "or drop slot_num=.",
                    )
                split_slot_num = slot_num
            else:
                raise ParserSyntaxError(
                    f"Unsupported entry in {owner}(optimizations=[...])",
                    span=self.span_tracker.get_span(entry),
                    hint=entry_hint,
                )

        return requests_auto_chunk, split_mode, split_slot_num

    def _parse_spmd_optimizations_list(
        self, value: ast.expr, *, span_anchor: ast.AST
    ) -> "tuple[ir.SplitMode | None, int | None]":
        """Parse ``pl.spmd(..., optimizations=[...])`` — ``pl.split`` only.

        ``pl.auto_chunk`` is not supported on ``pl.spmd``; use
        ``pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk])`` inside
        the scope body instead.

        Returns ``(split_mode, split_slot_num)``.
        """
        requests_auto_chunk, split_mode, split_slot_num = self._parse_optimizations_list(
            value,
            owner="pl.spmd",
            list_hint="Use optimizations=[pl.split(pl.SplitMode.NONE)].",
            entry_hint="Each entry must be pl.split(pl.SplitMode.X).",
        )
        if requests_auto_chunk:
            raise ParserSyntaxError(
                "pl.auto_chunk is not supported in pl.spmd(optimizations=[...])",
                span=self.span_tracker.get_span(span_anchor),
                hint="Use optimizations=[pl.split(pl.SplitMode.X)] on pl.spmd(), "
                "or move pl.auto_chunk into an inner "
                "pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]).",
            )
        return split_mode, split_slot_num

    @staticmethod
    def _is_pl_auto_chunk(node: ast.expr) -> bool:
        """Return True if the AST node is ``pl.auto_chunk`` or ``pl.optimizations.auto_chunk``."""
        if not isinstance(node, ast.Attribute) or node.attr != "auto_chunk":
            return False
        # pl.auto_chunk
        if isinstance(node.value, ast.Name) and node.value.id == "pl":
            return True
        # pl.optimizations.auto_chunk
        if (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "optimizations"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "pl"
        ):
            return True
        return False

    def _try_parse_pl_split(self, node: ast.expr) -> "tuple[ir.SplitMode, int | None] | None":
        """Return ``(SplitMode, slot_num)`` if the AST node is ``pl.split(MODE)``; else None.

        ``slot_num`` is ``None`` unless the optional ``slot_num=N`` keyword is
        given. Also accepts the fully qualified form
        ``pl.optimizations.split(MODE)``.
        """
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "split":
            return None
        # pl.split(...)
        if isinstance(func.value, ast.Name) and func.value.id == "pl":
            pass
        # pl.optimizations.split(...)
        elif (
            isinstance(func.value, ast.Attribute)
            and func.value.attr == "optimizations"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "pl"
        ):
            pass
        else:
            return None

        slot_num: int | None = None
        for kw in node.keywords:
            if kw.arg == "slot_num":
                slot_num = self._eval_pl_split_slot_num(kw.value)
            else:
                raise ParserSyntaxError(
                    f"Unknown keyword argument '{kw.arg}' in pl.split()",
                    span=self.span_tracker.get_span(kw),
                    hint="pl.split() accepts only the optional slot_num= keyword.",
                )
        if len(node.args) != 1:
            raise ParserSyntaxError(
                f"pl.split() takes exactly 1 positional argument, got {len(node.args)}",
                span=self.span_tracker.get_span(node),
                hint="Use pl.split(pl.SplitMode.NONE).",
            )
        mode = extract_enum_value(node.args[0], SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
        return mode, slot_num

    def _eval_pl_split_slot_num(self, value: ast.expr) -> int:
        """Evaluate ``slot_num=`` in ``pl.split(...)`` as a positive int literal."""
        # bool is a subclass of int — reject it explicitly.
        if (
            not isinstance(value, ast.Constant)
            or isinstance(value.value, bool)
            or not isinstance(value.value, int)
        ):
            raise ParserSyntaxError(
                "pl.split(slot_num=...) must be an integer literal",
                span=self.span_tracker.get_span(value),
                hint="Use e.g. pl.split(pl.SplitMode.UP_DOWN, slot_num=16).",
            )
        if value.value <= 0:
            raise ParserSyntaxError(
                f"pl.split(slot_num=...) must be positive, got {value.value}",
                span=self.span_tracker.get_span(value),
            )
        return value.value

    def _parse_chunked_loop_optimizer(self, value: ast.expr) -> "ir.SplitMode | None":
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
            return None

        # Called: pl.chunked_loop_optimizer(split=pl.SplitMode.<MODE>)
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
                    hint="Use: pl.chunked_loop_optimizer(split=pl.SplitMode.<MODE>)",
                )
            split: ir.SplitMode | None = None
            for opt_kw in value.keywords:
                if opt_kw.arg == "split":
                    split = extract_enum_value(opt_kw.value, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")
                else:
                    raise ParserSyntaxError(
                        f"pl.chunked_loop_optimizer() got unexpected keyword '{opt_kw.arg}'",
                        span=self.span_tracker.get_span(opt_kw),
                        hint="Only 'split' is supported: "
                        "pl.chunked_loop_optimizer(split=pl.SplitMode.<MODE>)",
                    )
            return split

        raise ParserSyntaxError(
            "optimization= only accepts pl.chunked_loop_optimizer or "
            "pl.chunked_loop_optimizer(split=pl.SplitMode.<MODE>)",
            span=self.span_tracker.get_span(value),
            hint="Use optimization=pl.chunked_loop_optimizer or "
            "optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.<MODE>)",
        )

    def _eval_split_mode(self, value: ast.expr) -> "ir.SplitMode":
        """Extract SplitMode enum value from AST expression."""
        return extract_enum_value(value, SPLIT_MODE_MAP, "SplitMode", "pl.SplitMode")

    def _parse_scope_name_hint(self, value: ast.expr, func_name: str) -> str:
        """Extract and validate a scope name hint from an AST expression.

        Args:
            value: AST expression node for the name_hint value
            func_name: Function name for error messages (e.g. "pl.at()")

        Returns:
            Validated name hint string.
        """
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ParserSyntaxError(
                f"{func_name} 'name_hint' argument must be a string literal",
                span=self.span_tracker.get_span(value),
                hint='Use name_hint="my_scope_name"',
            )
        name_hint = value.value
        if name_hint and (not name_hint.isidentifier() or _keyword_mod.iskeyword(name_hint)):
            raise ParserSyntaxError(
                f"{func_name} 'name_hint' must be a valid non-keyword identifier, got {name_hint!r}",
                span=self.span_tracker.get_span(value),
                hint="Use a valid Python identifier like 'fused_matmul_add'",
            )
        return name_hint

    def _parse_scope(self, stmt: ast.With, context_expr: ast.Call) -> None:
        """Parse ``with pl.scope(mode=...):`` into a Runtime scope.

        ``mode`` defaults to ``ScopeMode.AUTO``. AUTO scopes are the explicit IR
        form of the orchestration ``PTO2_SCOPE()`` block; MANUAL scopes turn off
        auto dependency tracking (``pl.scope(mode=pl.ScopeMode.MANUAL)`` — the
        former ``pl.manual_scope()``).
        """
        if context_expr.args:
            raise ParserSyntaxError(
                "pl.scope() takes only a 'mode=' keyword, not positional arguments",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.scope():' or 'with pl.scope(mode=pl.ScopeMode.MANUAL):'.",
            )
        manual = False
        for kw in context_expr.keywords:
            if kw.arg == "mode":
                manual = extract_enum_value(kw.value, SCOPE_MODE_MAP, "ScopeMode", "pl.ScopeMode")
            else:
                raise ParserSyntaxError(
                    f"pl.scope() got an unexpected keyword '{kw.arg}'",
                    span=self.span_tracker.get_span(stmt),
                    hint="The only accepted keyword is mode=pl.ScopeMode.AUTO|MANUAL.",
                )
        self._emit_runtime_scope(stmt, manual=manual)

    def _parse_manual_scope(self, stmt: ast.With, context_expr: ast.Call) -> None:
        """Parse ``with pl.manual_scope():`` — an alias for ``pl.scope(mode=MANUAL)``."""
        if context_expr.args or context_expr.keywords:
            raise ParserSyntaxError(
                "pl.manual_scope() does not accept arguments",
                span=self.span_tracker.get_span(stmt),
                hint="Use 'with pl.manual_scope():' (or 'with pl.scope(mode=pl.ScopeMode.MANUAL):').",
            )
        self._emit_runtime_scope(stmt, manual=True)

    def _emit_runtime_scope(self, stmt: ast.With, manual: bool) -> None:
        """Build a RuntimeScopeStmt for a ``with pl.scope(...)`` / ``pl.manual_scope()`` block.

        Enforces the scope-placement rules:
          - MANUAL may not nest inside another MANUAL (runtime forbids).
          - AUTO may not nest inside a MANUAL (runtime forbids AUTO-in-MANUAL).
          - A hand-placed AUTO scope requires ``@pl.function(auto_scope=False)``;
            in the default auto_scope=True the compiler owns AUTO placement, so a
            stray ``with pl.scope():`` would be a no-op — reject it instead.
        """
        span = self.span_tracker.get_span(stmt)
        if manual:
            if self._manual_scope_depth > 0:
                raise ParserSyntaxError(
                    "a manual scope may not be nested inside another manual scope",
                    span=span,
                    hint="Flatten the nested manual scope into the enclosing one, or move it outside.",
                )
            self._manual_scope_depth += 1
            try:
                self._parse_scope_body(stmt, ir.ScopeKind.Runtime, span, manual=True)
            finally:
                self._manual_scope_depth -= 1
            return

        # AUTO scope.
        if self._func_auto_scope:
            raise ParserSyntaxError(
                "a hand-placed AUTO 'with pl.scope()' requires @pl.function(auto_scope=False)",
                span=span,
                hint="In the default auto_scope=True mode the compiler places AUTO scopes; set "
                "auto_scope=False to place them yourself, or use pl.scope(mode=pl.ScopeMode.MANUAL).",
            )
        if self._manual_scope_depth > 0:
            raise ParserSyntaxError(
                "an AUTO scope may not be nested inside a manual scope",
                span=span,
                hint="The runtime forbids AUTO scope nested in MANUAL scope; move it outside.",
            )
        self._parse_scope_body(stmt, ir.ScopeKind.Runtime, span, manual=False)

    def _parse_legacy_scope(
        self,
        stmt: ast.With,
        context_expr: ast.Call,
        func_attr: str,
        scope_kind_map: dict[str, "ir.ScopeKind"],
        optional_vars: "ast.expr | None" = None,
    ) -> None:
        """Parse legacy scope context managers (pl.incore, pl.auto_incore, pl.cluster, pl.spmd).

        ``optional_vars`` (the ``as <target>`` clause) is only meaningful for
        ``pl.spmd`` (``with pl.spmd(...) as tid:``); the caller rejects it on the
        other kinds before dispatching here.
        """
        split_mode = None
        name_hint = ""
        if func_attr in ("auto_incore", "incore"):
            if context_expr.args:
                raise ParserSyntaxError(
                    f"pl.{func_attr}() does not accept positional arguments",
                    span=self.span_tracker.get_span(stmt),
                    hint=f"Use 'with pl.{func_attr}(split=pl.SplitMode.UP_DOWN):'",
                )
            for kw in context_expr.keywords:
                if kw.arg == "split":
                    split_mode = self._eval_split_mode(kw.value)
                elif kw.arg == "name_hint":
                    name_hint = self._parse_scope_name_hint(kw.value, f"pl.{func_attr}()")
                else:
                    raise ParserSyntaxError(
                        f"pl.{func_attr}() got unexpected keyword argument '{kw.arg}'",
                        span=self.span_tracker.get_span(stmt),
                        hint="Supported keywords: 'split', 'name_hint'",
                    )
            if func_attr == "incore":
                warnings.warn(
                    "pl.incore() is deprecated; use 'with pl.at(level=pl.Level.CORE_GROUP):' "
                    "(optionally with optimizations=[pl.split(pl.SplitMode.X)]) instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "pl.auto_incore() is deprecated; use "
                    "'with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):' "
                    "(combine with pl.split(pl.SplitMode.X) if a split mode is needed) instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
        elif func_attr == "cluster":
            if context_expr.args:
                raise ParserSyntaxError(
                    f"pl.{func_attr}() does not accept positional arguments",
                    span=self.span_tracker.get_span(stmt),
                    hint=f"Use 'with pl.{func_attr}():'",
                )
            for kw in context_expr.keywords:
                if kw.arg == "name_hint":
                    name_hint = self._parse_scope_name_hint(kw.value, f"pl.{func_attr}()")
                else:
                    raise ParserSyntaxError(
                        f"pl.{func_attr}() got unexpected keyword argument '{kw.arg}'",
                        span=self.span_tracker.get_span(stmt),
                        hint="Supported keyword: 'name_hint'. For SPMD dispatch, use pl.spmd(4):",
                    )
            scope_kind = scope_kind_map[func_attr]
            span = self.span_tracker.get_span(stmt)
            self._parse_scope_body(stmt, scope_kind, span, name_hint=name_hint)
            return
        elif func_attr == "spmd":
            self._parse_spmd_scope(stmt, context_expr, scope_kind_map, optional_vars=optional_vars)
            return
        elif context_expr.args or context_expr.keywords:
            raise ParserSyntaxError(
                f"pl.{func_attr}() does not accept arguments",
                span=self.span_tracker.get_span(stmt),
                hint=f"Use 'with pl.{func_attr}():' without arguments",
            )
        scope_kind = scope_kind_map[func_attr]
        span = self.span_tracker.get_span(stmt)
        self._parse_scope_body(stmt, scope_kind, span, split=split_mode, name_hint=name_hint)

    # Integer dtypes accepted for an SPMD ``core_num`` (block count). Shared by
    # the ``pl.spmd`` scope path and the ``pl.spmd_submit`` task-launch path.
    _CORE_NUM_INTEGER_DTYPES = frozenset(
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

    def _parse_and_validate_core_num(
        self, value_node: ast.AST, source: ast.AST, usage_hint: str
    ) -> "ir.Expr":
        """Parse and validate an SPMD ``core_num`` expression.

        ``core_num`` must be an integer-typed IR expression; a compile-time
        constant must be strictly positive. Shared by ``pl.spmd`` (scope) and
        ``pl.spmd_submit`` (task launch) so both reject the same bad inputs.
        """
        # ast.AST covers any expression; parse_expression expects ast.expr. The
        # grammar for keyword values and positional args always gives ast.expr
        # here, but cast for mypy's sake.
        expr = self.parse_expression(cast("ast.expr", value_node))
        expr_type = expr.type
        is_integer = isinstance(expr_type, ir.ScalarType) and expr_type.dtype in self._CORE_NUM_INTEGER_DTYPES
        if not is_integer:
            raise ParserSyntaxError(
                f"core_num must be an integer expression, got {python_print(expr_type, format=False)}",
                span=self.span_tracker.get_span(source),
                hint=usage_hint,
            )
        if isinstance(expr, ir.ConstInt) and expr.value <= 0:
            raise ParserSyntaxError(
                f"core_num must be a positive integer, got {expr.value}",
                span=self.span_tracker.get_span(source),
                hint=usage_hint,
            )
        return expr

    def _parse_spmd_submit_kwargs(
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
                core_num = self._parse_and_validate_core_num(kw.value, kw.value, usage_hint)
            elif kw.arg == "sync_start":
                if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                    raise ParserSyntaxError(
                        "sync_start must be a boolean literal (True/False)",
                        span=self.span_tracker.get_span(kw.value),
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

    def _parse_spmd_kwargs(
        self,
        anchor: ast.AST,
        call: ast.Call,
        *,
        usage_hint: str,
        allow_deps: bool = False,
    ) -> tuple["ir.Expr", bool, str, "ir.SplitMode | None", "int | None", "list[ir.Var]"]:
        """Parse ``pl.spmd(core_num, *, sync_start=, name_hint=, optimizations=, deps=)`` arguments.

        The first positional argument is ``core_num`` (range-like). Returns
        ``(core_num, sync_start, name_hint, split_mode, split_slot_num, dep_vars)``
        with ``sync_start`` defaulting to ``False``, ``split_mode`` /
        ``split_slot_num`` to ``None``, and ``dep_vars`` to ``[]``.

        ``optimizations=[...]`` accepts only ``pl.split(MODE)`` — see
        :meth:`_parse_spmd_optimizations_list`.

        ``deps=[...]`` is accepted only when ``allow_deps`` is True (the
        ``with pl.spmd(...) as tid:`` form). It takes the same shapes as
        ``pl.submit(..., deps=)`` / ``pl.at(..., deps=)`` — producer TaskId
        ``Scalar[TASK_ID]`` Vars, an ``Array[N, TASK_ID]`` carry, or the ``None``
        sentinel — resolved via :meth:`_parse_submit_deps_kwarg`.
        """
        if len(call.args) > 1:
            raise ParserSyntaxError(
                "pl.spmd() accepts at most one positional argument (core_num)",
                span=self.span_tracker.get_span(call.args[1]),
                hint=usage_hint,
            )
        core_num: ir.Expr | None = None
        if call.args:
            core_num = self._parse_and_validate_core_num(call.args[0], call.args[0], usage_hint)
        sync_start: bool = False
        name_hint = ""
        split_mode: ir.SplitMode | None = None
        split_slot_num: int | None = None
        deps_kw: ast.keyword | None = None
        for kw in call.keywords:
            if kw.arg is None:
                # `pl.spmd(**cfg)` — ast.keyword.arg is None for **kwargs unpacking.
                raise ParserSyntaxError(
                    "pl.spmd() does not accept **kwargs; pass core_num (positional) "
                    "and sync_start=/name_hint=/optimizations= explicitly",
                    span=self.span_tracker.get_span(kw.value),
                    hint=usage_hint,
                )
            if kw.arg == "name_hint":
                name_hint = self._parse_scope_name_hint(kw.value, "pl.spmd()")
            elif kw.arg == "core_num":
                if core_num is not None:
                    raise ParserSyntaxError(
                        "pl.spmd() got multiple values for argument 'core_num'",
                        span=self.span_tracker.get_span(kw.value),
                        hint=usage_hint,
                    )
                core_num = self._parse_and_validate_core_num(kw.value, kw.value, usage_hint)
            elif kw.arg == "sync_start":
                if not isinstance(kw.value, ast.Constant) or not isinstance(kw.value.value, bool):
                    raise ParserSyntaxError(
                        "sync_start must be a boolean literal (True/False)",
                        span=self.span_tracker.get_span(anchor),
                        hint=usage_hint,
                    )
                sync_start = kw.value.value
            elif kw.arg == "optimizations":
                split_mode, split_slot_num = self._parse_spmd_optimizations_list(
                    kw.value, span_anchor=anchor
                )
            elif kw.arg == "deps":
                if not allow_deps:
                    raise ParserSyntaxError(
                        "pl.spmd() does not accept 'deps=' here",
                        span=self.span_tracker.get_span(kw.value),
                        hint="Use `with pl.spmd(n, deps=[...]) as tid:` (the with-form) to "
                        "declare explicit TaskId deps, or `out, tid = pl.spmd_submit(..., "
                        "deps=[...])` for the single-call form.",
                    )
                deps_kw = kw
            else:
                supported = (
                    "Supported keywords: 'sync_start', 'name_hint', 'optimizations', 'deps'"
                    if allow_deps
                    else "Supported keywords: 'sync_start', 'name_hint', 'optimizations'"
                )
                raise ParserSyntaxError(
                    f"pl.spmd() got unexpected keyword argument '{kw.arg}'",
                    span=self.span_tracker.get_span(anchor),
                    hint=supported,
                )
        if core_num is None:
            raise ParserSyntaxError(
                "pl.spmd() requires core_num (first positional argument)",
                span=self.span_tracker.get_span(anchor),
                hint=usage_hint,
            )
        dep_vars: list[ir.Var] = []
        if deps_kw is not None:
            anchor_span = self.span_tracker.get_span(anchor)
            dep_vars = self._parse_submit_deps_kwarg("pl.spmd()", [deps_kw], anchor_span)
        return core_num, sync_start, name_hint, split_mode, split_slot_num, dep_vars

    def _parse_spmd_scope(
        self,
        stmt: ast.With,
        context_expr: ast.Call,
        scope_kind_map: dict[str, "ir.ScopeKind"],
        optional_vars: "ast.expr | None" = None,
    ) -> None:
        """Parse ``with pl.spmd(...):`` / ``with pl.spmd(...) as tid:`` into a ScopeStmt(Spmd).

        Two forms:

        * ``with pl.spmd(n): self.kernel(...)`` — wraps a single kernel call
          (historical shape; no producer TaskId captured, no ``deps=``).
        * ``with pl.spmd(n, deps=[...]) as tid:`` — captures the grid dispatch's
          producer ``Scalar[TASK_ID]`` (mirrors ``with pl.at(...) as tid:``) and
          accepts an inline multi-statement body that is auto-outlined into an
          InCore kernel, exactly like ``for i in pl.spmd(n):``. The per-block
          index is read inside the body via ``pl.tile.get_block_idx()``.
        """
        with_hint = (
            "Use 'with pl.spmd(4):' with a single function call inside, or "
            "'with pl.spmd(4) as tid:' to capture the dispatch TaskId."
        )
        # ``deps=`` is accepted ONLY with ``as tid`` — gate it by keyword presence,
        # not by the resolved list being non-empty. _parse_submit_deps_kwarg
        # normalizes ``deps=[]`` / ``deps=[None]`` to ``[]``, so a truthiness check
        # would silently accept those unsupported forms on the plain with-form.
        # Passing allow_deps=(optional_vars is not None) makes _parse_spmd_kwargs
        # reject any ``deps=`` on the non-capturing form (and keeps its "supported
        # keywords" hint accurate).
        core_num, sync_start, name_hint, split_mode, split_slot_num, dep_vars = self._parse_spmd_kwargs(
            stmt, context_expr, usage_hint=with_hint, allow_deps=optional_vars is not None
        )
        scope_kind = scope_kind_map["spmd"]
        span = self.span_tracker.get_span(stmt)

        if optional_vars is not None:
            self._parse_spmd_scope_with_tid(
                stmt,
                span,
                scope_kind,
                core_num,
                sync_start,
                name_hint,
                split_mode,
                split_slot_num,
                dep_vars,
                optional_vars,
            )
            return

        # No ``as tid``: the historical single-kernel-call with-form. ``deps=`` was
        # already rejected above (allow_deps=False), so dep_vars is empty here.
        # Validate body is exactly one statement that is a function call.
        # The loop form (for i in pl.spmd(n):) and the `as tid` with-form are
        # what accept inline multi-statement bodies.
        spmd_hint = (
            "The 'with pl.spmd()' form (without 'as tid') wraps a single kernel call. "
            "Use 'with pl.spmd(4) as tid:' to write inline tile/tensor ops and capture the "
            "dispatch TaskId, or 'for i in pl.spmd(4):' for an inline loop body."
        )
        if len(stmt.body) != 1:
            raise ParserSyntaxError(
                f"pl.spmd() body must contain exactly one statement, got {len(stmt.body)}",
                span=self.span_tracker.get_span(stmt),
                hint=spmd_hint,
            )
        body_stmt = stmt.body[0]
        is_call = (
            (isinstance(body_stmt, ast.Assign) and isinstance(body_stmt.value, ast.Call))
            or (isinstance(body_stmt, ast.AnnAssign) and isinstance(body_stmt.value, ast.Call))
            or (isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Call))
        )
        if not is_call:
            raise ParserSyntaxError(
                "pl.spmd() body statement must be a function call",
                span=self.span_tracker.get_span(stmt),
                hint=spmd_hint,
            )
        if split_mode is None:
            # No optimizations — preserve the historical IR shape:
            # SpmdScopeStmt(<call>) with no inner InCore wrapper.
            self._parse_scope_body(
                stmt,
                scope_kind,
                span,
                name_hint=name_hint,
                core_num=core_num,
                sync_start=sync_start,
            )
        else:
            # split= hint requires an inner InCoreScopeStmt to carry the
            # split_ field. Build SpmdScopeStmt(InCoreScopeStmt(split_=mode, <call>)).
            spmd_name_hint, incore_name_hint = _split_spmd_for_loop_name_hints(name_hint)
            # Like the for-form, this path builds the InCore scope directly
            # instead of routing through _parse_scope_body, so merge any
            # forward-sticky pl.dump_tag tensors onto it here (see
            # _parse_spmd_for_loop for the full rationale).
            incore_attrs = self._merge_forward_sticky_dump(None, ir.ScopeKind.InCore)
            incore_attrs = self._append_split_slot_num_attr(incore_attrs, split_slot_num)
            with self.builder.scope(
                scope_kind,
                span,
                name_hint=spmd_name_hint,
                core_num=core_num,
                sync_start=sync_start,
            ):
                with self._scope_kind_context(scope_kind):
                    self.scope_manager.enter_scope("spmd_with")
                    with self.builder.scope(
                        ir.ScopeKind.InCore,
                        span,
                        split=split_mode,
                        name_hint=incore_name_hint,
                        attrs=incore_attrs,
                    ):
                        with self._scope_kind_context(ir.ScopeKind.InCore):
                            self.scope_manager.enter_scope("spmd_with_incore")
                            self._parse_body_siblings(stmt.body)
                            self._discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                            self.scope_manager.exit_scope(leak_vars=True)
                    self.scope_manager.exit_scope(leak_vars=True)

    def _parse_spmd_scope_with_tid(  # noqa: PLR0913 — args map 1:1 to the SpmdScopeStmt + capture
        self,
        stmt: ast.With,
        span: "ir.Span",
        scope_kind: "ir.ScopeKind",
        core_num: "ir.Expr",
        sync_start: bool,
        name_hint: str,
        split_mode: "ir.SplitMode | None",
        split_slot_num: "int | None",
        dep_vars: "list[ir.Var]",
        optional_vars: "ast.expr",
    ) -> None:
        """Parse ``with pl.spmd(n, deps=[...]) as tid:`` capturing the dispatch TaskId.

        Records ``{manual_dep_edges?, task_id_var}`` on the ``SpmdScopeStmt`` and
        emits the ``system.task_invalid()`` placeholder. The body shape mirrors the
        plain with-form so the IR is identical to the post-``OutlineIncoreScopes``
        shape (a single Call) and round-trips through print -> reparse:

        * single call, no split → ``SpmdScopeStmt(attrs, body=Call)`` (no InCore
          wrapper) — the same shape ``OutlineIncoreScopes`` leaves behind once the
          inline body is outlined, so reparse is stable across passes.
        * inline multi-statement body, or single-call with split → wrap in
          ``InCoreScopeStmt(split, <body>)``, exactly like the for-form (minus the
          synthesised ``loop_var = tile.get_block_idx()``; the body reads the block
          index explicitly via ``pl.tile.get_block_idx()``).

        The ``kAttrTaskIdVar`` on the outer Spmd scope makes ``OutlineClusterScopes``
        lower the dispatch to an ``ir.Submit`` whose trailing tuple element is the
        grid-wide producer TaskId — identical to the ``pl.at(...) as tid:`` rail.
        """
        if self._is_inside_scope(ir.ScopeKind.Cluster):
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

        # Canonical attr order (deps before tid) mirrors _parse_at_meta so a
        # print -> reparse cycle compares equal under structural_equal's
        # positional attr check.
        scope_attrs: list[tuple[str, Any]] = []
        if dep_vars:
            scope_attrs.append(("manual_dep_edges", dep_vars))
        tid_var = self.builder.var(optional_vars.id, ir.ScalarType(DataType.TASK_ID), span=span)
        self.scope_manager.define_var(optional_vars.id, tid_var, span=span)
        scope_attrs.append(("task_id_var", tid_var))

        # Emit the transient ``AssignStmt(tid, system.task_invalid())`` placeholder
        # one stmt BEFORE the scope so ConvertToSSA has a def for the tid Var; the
        # spmd outliner drops it and synthesises the real
        # ``AssignStmt(tid, TupleGetItem(ret_tmp, n_outputs))`` binding. Pop/re-push
        # pending leading comments so they land on the surviving SpmdScopeStmt, not
        # the placeholder (mirrors _parse_at_scope).
        leading = self.builder.pop_pending_leading_comments()
        placeholder_rhs = ir.create_op_call("system.task_invalid", [], {}, span)
        self.builder.assign(tid_var, placeholder_rhs, span=span)
        self.builder.push_pending_leading_comments(leading)

        # A lone kernel call (no split) keeps the no-InCore-wrapper shape — the
        # same SpmdScopeStmt(body=Call) that OutlineIncoreScopes leaves behind once
        # an inline body is outlined — so the IR round-trips identically across
        # passes. Anything else (inline multi-statement body, or single-call with a
        # split hint) wraps in an InCoreScopeStmt, exactly like the for-form.
        body_stmt = stmt.body[0] if len(stmt.body) == 1 else None
        is_single_call = body_stmt is not None and (
            (isinstance(body_stmt, ast.Assign) and isinstance(body_stmt.value, ast.Call))
            or (isinstance(body_stmt, ast.AnnAssign) and isinstance(body_stmt.value, ast.Call))
            or (isinstance(body_stmt, ast.Expr) and isinstance(body_stmt.value, ast.Call))
        )
        if is_single_call and split_mode is None:
            self._parse_scope_body(
                stmt,
                scope_kind,
                span,
                name_hint=name_hint,
                core_num=core_num,
                sync_start=sync_start,
                attrs=scope_attrs,
            )
            return

        spmd_name_hint, incore_name_hint = _split_spmd_for_loop_name_hints(name_hint)
        incore_attrs = self._merge_forward_sticky_dump(None, ir.ScopeKind.InCore)
        incore_attrs = self._append_split_slot_num_attr(incore_attrs, split_slot_num)
        with self.builder.scope(
            scope_kind,
            span,
            name_hint=spmd_name_hint,
            core_num=core_num,
            sync_start=sync_start,
            attrs=scope_attrs,
        ):
            with self._scope_kind_context(scope_kind):
                self.scope_manager.enter_scope("spmd_with")
                with self.builder.scope(
                    ir.ScopeKind.InCore,
                    span,
                    split=split_mode,
                    name_hint=incore_name_hint,
                    attrs=incore_attrs,
                ):
                    with self._scope_kind_context(ir.ScopeKind.InCore):
                        self.scope_manager.enter_scope("spmd_with_incore")
                        self._parse_body_siblings(stmt.body)
                        self._discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                        self.scope_manager.exit_scope(leak_vars=True)
                self.scope_manager.exit_scope(leak_vars=True)

    def _parse_spmd_for_loop(self, stmt: ast.For, iter_call: ast.Call) -> None:
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
                span=self.span_tracker.get_span(stmt.target),
                hint=spmd_hint,
            )
        loop_var_name = stmt.target.id

        # Reject loop-specific kwargs that make no sense for SPMD blocks.
        disallowed_loop_kwargs = {"init_values", "chunk", "chunk_policy", "attrs", "step", "stage"}
        for kw in iter_call.keywords:
            if kw.arg in disallowed_loop_kwargs:
                raise ParserSyntaxError(
                    f"pl.spmd() loop form does not accept '{kw.arg}='",
                    span=self.span_tracker.get_span(kw.value),
                    hint=spmd_hint,
                )

        # The for-form does not capture a TaskId, so it rejects deps= (allow_deps
        # defaults False): use the with-form `with pl.spmd(n, deps=[...]) as tid:`
        # to wire explicit deps. dep_vars is therefore always empty here.
        core_num, sync_start, name_hint, split_mode, split_slot_num, _ = self._parse_spmd_kwargs(
            stmt, iter_call, usage_hint=spmd_hint
        )
        spmd_name_hint, incore_name_hint = _split_spmd_for_loop_name_hints(name_hint)

        span = self.span_tracker.get_span(stmt)
        # Merge forward-sticky pl.dump_tag tensors onto the auto-outlined InCore
        # scope — the kernel the loop body lowers to. The with-form (pl.at /
        # pl.spmd / pl.incore) routes through _parse_scope_body for this; the
        # for-form builds its scope directly, so attach here to keep the two
        # paths symmetric. OutlineIncoreScopes then carries the dump_vars onto
        # the synthesised inner-kernel Call; the wrapper-dispatch codegen
        # (BuildWrapperReorderedParams) honours that inner call's dump_vars.
        incore_attrs = self._merge_forward_sticky_dump(None, ir.ScopeKind.InCore)
        incore_attrs = self._append_split_slot_num_attr(incore_attrs, split_slot_num)
        with self.builder.scope(
            ir.ScopeKind.Spmd,
            span,
            name_hint=spmd_name_hint,
            core_num=core_num,
            sync_start=sync_start,
        ):
            with self._scope_kind_context(ir.ScopeKind.Spmd):
                self.scope_manager.enter_scope("spmd_for")
                with self.builder.scope(
                    ir.ScopeKind.InCore,
                    span,
                    split=split_mode,
                    name_hint=incore_name_hint,
                    attrs=incore_attrs,
                ):
                    with self._scope_kind_context(ir.ScopeKind.InCore):
                        # Bind `i = pl.tile.get_block_idx()` as the first
                        # statement of the outlined InCore body.
                        loop_var = self.builder.var(loop_var_name, ir.ScalarType(DataType.INDEX), span=span)
                        self.scope_manager.define_var(loop_var_name, loop_var)
                        self.builder.assign(loop_var, ir_op.tile.get_block_idx(span=span), span=span)
                        self._parse_body_siblings(stmt.body)
                self._discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                # Leak vars to parent so post-store rebindings (e.g.
                # ``out = pl.store(...)`` inside the body) remain visible to
                # subsequent statements like ``return out``. Matches Python's
                # own for-loop variable-leaking semantics.
                self.scope_manager.exit_scope(leak_vars=True)

    def _merge_forward_sticky_dump(
        self,
        attrs: "list[tuple[str, Any]] | None",
        scope_kind: "ir.ScopeKind",
    ) -> "list[tuple[str, Any]] | None":
        """Merge forward-sticky ``pl.dump_tag`` tensors into a scope's dump_vars attr.

        The single injection point for the scope-level selective-dump carrier on
        first parse — the explicit / round-trip ``dumps=`` surface is handled
        separately by :meth:`_parse_at_meta`. Both the ``pl.at`` and the legacy
        ``pl.incore`` / ``pl.cluster`` paths route through
        :meth:`_parse_scope_body`, so attaching here covers every scope kind that
        becomes a kernel dispatch. Runtime scopes (``pl.manual_scope`` /
        ``pl.auto_scope``) are skipped: they are not outlined into a dispatch, and
        ``pl.submit`` inside a manual scope carries its own per-call ``dump_vars``.

        Tags are captured at scope entry (forward-sticky), so they are bound
        before the scope and live at its entry — exactly the SSA version the
        synthesised dispatch receives as an arg. Entries the scope never consumes
        are dropped later by the outliner. ``dump_vars`` is kept before
        ``task_id_var`` so a print -> reparse (which rebuilds the canonical order
        via :meth:`_parse_at_meta`) compares equal under structural_equal's
        positional attr check.
        """
        if scope_kind == ir.ScopeKind.Runtime:
            return attrs
        tagged = [v for v in self._dump_tagged_vars if isinstance(v.type, ir.TensorType)]
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
        insert_at = next((i for i, (k, _) in enumerate(new_attrs) if k == "task_id_var"), len(new_attrs))
        new_attrs.insert(insert_at, ("dump_vars", tagged))
        return new_attrs

    def _parse_scope_body(  # noqa: PLR0913 — kwargs map 1:1 to ScopeStmt fields
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
        attrs = self._merge_forward_sticky_dump(attrs, scope_kind)
        with self.builder.scope(
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
            with self._scope_kind_context(scope_kind):
                self.scope_manager.enter_scope("scope")
                self._parse_body_siblings(stmt.body)
                self._discard_tail_block_comments(stmt.body, upper_line=stmt.end_lineno)
                self.scope_manager.exit_scope(leak_vars=True)

    def _parse_at_scope(
        self, stmt: ast.With, context_expr: ast.Call, optional_vars: "ast.expr | None" = None
    ) -> None:
        """Parse pl.at(...) context manager into a ScopeStmt."""
        state = self._parse_at_kwargs(context_expr)
        level = state.level
        role = state.role
        requests_auto_chunk = state.requests_auto_chunk
        split_mode = state.split_mode
        name_hint = state.name_hint
        deps_kw = state.deps_kw
        no_dep_args_kw = state.no_dep_args_kw
        dumps_kw = state.dumps_kw
        assert level is not None  # _parse_at_kwargs raises if level is missing
        span = self.span_tracker.get_span(stmt)

        is_core_group = level == ir.Level.CORE_GROUP

        if requests_auto_chunk and not is_core_group:
            raise ParserSyntaxError(
                "auto-chunk optimization is only supported with level=pl.Level.CORE_GROUP "
                "(via optimizations=[pl.auto_chunk] or the deprecated "
                "optimization=pl.chunked_loop_optimizer)",
                span=span,
                hint="Use pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]) "
                "for an AutoInCore scope.",
            )

        if split_mode is not None and not is_core_group:
            raise ParserSyntaxError(
                "split mode is only supported with level=pl.Level.CORE_GROUP "
                "(via optimizations=[pl.split(...)] or the deprecated split= kwarg)",
                span=span,
                hint="Use pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]).",
            )

        if is_core_group and role is not None:
            raise ParserSyntaxError(
                "role= is not supported with level=pl.Level.CORE_GROUP",
                span=span,
                hint="Drop role= for InCore/AutoInCore scopes, "
                "or use a non-CORE_GROUP level for Hierarchy scope",
            )

        # Build the optional ``manual_dep_edges`` / ``task_id_var`` attrs for the
        # synthesised ScopeStmt. ``deps=[tid1, tid2]`` accepts the same shapes
        # as ``pl.submit(..., deps=)`` — TaskId Vars, the ``None`` sentinel, or
        # an ``Array[N, TASK_ID]`` carry. ``with pl.at(...) as tid:`` binds a
        # fresh ``Scalar[TASK_ID]`` Var in the outer scope; the outliner
        # later wires it to ``TupleGetItem(call_lhs, last_idx)``.
        scope_attrs = self._parse_at_meta(deps_kw, no_dep_args_kw, dumps_kw, optional_vars, span)
        scope_attrs = self._append_split_slot_num_attr(scope_attrs, state.split_slot_num)

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
        if scope_attrs is not None:
            for k, v in scope_attrs:
                if k == "task_id_var":
                    leading = self.builder.pop_pending_leading_comments()
                    placeholder_rhs = ir.create_op_call("system.task_invalid", [], {}, span)
                    self.builder.assign(v, placeholder_rhs, span=span)
                    self.builder.push_pending_leading_comments(leading)
                    break

        if not is_core_group:
            # SubWorker scopes are no longer supported as inline `with pl.at(...)`
            # blocks. Declare a SubWorker via @pl.function(level=..., role=Worker).
            if level is not None and ir.level_to_linqu_level(level) >= 3 and role == ir.Role.SubWorker:
                raise ParserSyntaxError(
                    "Inline 'with pl.at(level>=HOST, role=pl.Role.SubWorker)' is not supported.",
                    span=span,
                    hint="Declare a SubWorker via @pl.function(level=..., role=pl.Role.SubWorker) "
                    "as a self-contained function (no 'self' parameter) inside @pl.program.",
                )
            self._parse_scope_body(
                stmt,
                ir.ScopeKind.Hierarchy,
                span,
                level=level,
                role=role,
                name_hint=name_hint,
                attrs=scope_attrs,
            )
        elif requests_auto_chunk:
            self._parse_scope_body(
                stmt,
                ir.ScopeKind.AutoInCore,
                span,
                split=split_mode,
                name_hint=name_hint,
                attrs=scope_attrs,
            )
        else:
            self._parse_scope_body(
                stmt,
                ir.ScopeKind.InCore,
                span,
                split=split_mode,
                name_hint=name_hint,
                attrs=scope_attrs,
            )

    @staticmethod
    def _append_split_slot_num_attr(
        attrs: "list[tuple[str, Any]] | None", slot_num: "int | None"
    ) -> "list[tuple[str, Any]] | None":
        """Append the ``slot_num`` scope attr (from ``pl.split(mode, slot_num=N)``).

        Appended last so a print -> reparse cycle reproduces the same attr order
        (``optimizations=[pl.split(...)]`` is printed alongside ``deps=`` /
        ``dumps=``, and slot_num always lands at the tail here on both passes).
        Returns ``attrs`` unchanged when ``slot_num`` is ``None``.
        """
        if slot_num is None:
            return attrs
        result: list[tuple[str, Any]] = list(attrs) if attrs else []
        result.append(("slot_num", slot_num))
        return result

    def _parse_at_meta(
        self,
        deps_kw: "ast.keyword | None",
        no_dep_args_kw: "ast.keyword | None",
        dumps_kw: "ast.keyword | None",
        optional_vars: "ast.expr | None",
        span: "ir.Span",
    ) -> "list[tuple[str, Any]] | None":
        """Build the ScopeStmt ``attrs`` list from a ``pl.at(...)`` ``deps=`` /
        ``no_dep_args=`` / ``dumps=`` kwarg set and a ``with ... as <tid>:``
        capture target.

        Returns ``None`` when none are present, leaving the scope's ``attrs_``
        empty (the typical plain ``pl.at(...)`` case). Otherwise returns a list
        with up to four reserved keys, always in this canonical order (so a
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

        The ``tid`` Var is defined in the outer scope so subsequent statements
        (e.g. another ``pl.at(..., deps=[tid])``) can reference it.
        """
        # ``dumps=`` is the explicit scope-level dump surface (symmetric with
        # ``deps=``) and also the print/reparse round-trip surface. The
        # forward-sticky ``pl.dump_tag`` seed is merged in later by
        # :meth:`_parse_scope_body` (the single injection point shared by the
        # ``pl.at`` and legacy ``pl.incore`` / ``pl.cluster`` paths), so it is
        # not consulted here.
        dump_vars: list[ir.Var] = self._parse_at_dumps_kwarg(dumps_kw) if dumps_kw else []

        if deps_kw is None and no_dep_args_kw is None and not dump_vars and optional_vars is None:
            return None

        attrs: list[tuple[str, Any]] = []

        if deps_kw is not None:
            dep_vars = self._parse_submit_deps_kwarg("pl.at()", [deps_kw], span)
            if dep_vars:
                # Attr keys mirror the C++ ``kAttrManualDepEdges`` /
                # ``kAttrTaskIdVar`` / ``kAttrArgDirOverrideVars`` /
                # ``kAttrDumpVars`` constants (include/pypto/ir/expr.h); passed
                # as raw strings since they are not exposed to Python.
                attrs.append(("manual_dep_edges", dep_vars))

        if no_dep_args_kw is not None:
            no_dep_vars = self._parse_at_no_dep_args_kwarg(no_dep_args_kw)
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
            tid_var = self.builder.var(optional_vars.id, ir.ScalarType(DataType.TASK_ID), span=span)
            self.scope_manager.define_var(optional_vars.id, tid_var, span=span)
            attrs.append(("task_id_var", tid_var))

        return attrs if attrs else None

    def _parse_at_no_dep_args_kwarg(self, kw: "ast.keyword") -> list[ir.Var]:
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
                span=self.span_tracker.get_span(kw),
                hint="Use `no_dep_args=[t1, t2]` with bare tensor names captured by the scope body.",
            )

        resolved: list[ir.Var] = []
        seen: set[int] = set()
        for elt in kw.value.elts:
            if not isinstance(elt, ast.Name):
                raise ParserTypeError(
                    "pl.at(no_dep_args=[...]) entries must be bare tensor names",
                    span=self.span_tracker.get_span(elt),
                    hint="Use `no_dep_args=[t]` where `t` is a tensor variable visible "
                    "to the enclosing function scope.",
                )
            var = self.scope_manager.lookup_var(elt.id)
            if var is None:
                raise ParserTypeError(
                    f"pl.at(no_dep_args=[...]) references unknown name '{elt.id}'",
                    span=self.span_tracker.get_span(elt),
                    hint="Each entry must resolve to a tensor visible in the enclosing function scope.",
                )
            if not isinstance(var.type, ir.TensorType):
                raise ParserTypeError(
                    f"pl.at(no_dep_args=[...]) entry '{elt.id}' is not a tensor (got type {var.type})",
                    span=self.span_tracker.get_span(elt),
                    hint="Only tensors can be marked NoDep; scalars and other "
                    "types do not participate in dependency tracking.",
                )
            if id(var) in seen:
                raise ParserTypeError(
                    f"pl.at(no_dep_args=[...]) lists '{elt.id}' more than once",
                    span=self.span_tracker.get_span(elt),
                    hint="Each tensor may appear at most once in `no_dep_args=`.",
                )
            seen.add(id(var))
            resolved.append(var)
        return resolved

    def _parse_at_dumps_kwarg(self, kw: "ast.keyword") -> list[ir.Var]:
        """Resolve ``pl.at(dumps=[t1, t2])`` entries to outer-scope tensor Vars.

        Mirrors :meth:`_parse_at_no_dep_args_kwarg`: each entry must be a bare
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
                "pl.at(dumps=...) must be a list literal of tensor names",
                span=self.span_tracker.get_span(kw),
                hint="Use `dumps=[t1, t2]` with bare tensor names visible to the enclosing function.",
            )

        resolved: list[ir.Var] = []
        seen: set[int] = set()
        for elt in kw.value.elts:
            if not isinstance(elt, ast.Name):
                raise ParserTypeError(
                    "pl.at(dumps=[...]) entries must be bare tensor names",
                    span=self.span_tracker.get_span(elt),
                    hint="Use `dumps=[t]` where `t` is a tensor variable visible "
                    "to the enclosing function scope.",
                )
            var = self.scope_manager.lookup_var(elt.id)
            if var is None:
                raise ParserTypeError(
                    f"pl.at(dumps=[...]) references unknown name '{elt.id}'",
                    span=self.span_tracker.get_span(elt),
                    hint="Each entry must resolve to a tensor visible in the enclosing function scope.",
                )
            if not isinstance(var.type, ir.TensorType):
                raise ParserTypeError(
                    f"pl.at(dumps=[...]) entry '{elt.id}' is not a tensor (got type {var.type})",
                    span=self.span_tracker.get_span(elt),
                    hint="Only tensors can be selectively dumped.",
                )
            if id(var) in seen:
                raise ParserTypeError(
                    f"pl.at(dumps=[...]) lists '{elt.id}' more than once",
                    span=self.span_tracker.get_span(elt),
                    hint="Each tensor may appear at most once in `dumps=`.",
                )
            seen.add(id(var))
            resolved.append(var)
        return resolved

    def parse_with_statement(self, stmt: ast.With) -> None:
        """Parse with statement for scope contexts.

        Currently supports:
        - with pl.incore(): ... (deprecated; creates ScopeStmt with InCore scope)
        - with pl.incore(split=pl.SplitMode.UP_DOWN): ... (deprecated; InCore with split)
        - with pl.auto_incore(): ... (deprecated; creates ScopeStmt with AutoInCore scope)
        - with pl.auto_incore(split=pl.SplitMode.UP_DOWN): ... (deprecated; with split mode)
        - with pl.cluster(): ... (creates ScopeStmt with Cluster scope)
        - with pl.at(level=..., role=...): ... (creates ScopeStmt with InCore/Hierarchy scope)
        - with pl.at(level=CORE_GROUP): ... (creates ScopeStmt with InCore scope)
        - with pl.at(level=CORE_GROUP, split=pl.SplitMode.UP_DOWN): ... (InCore with split)
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
        optional_vars = item.optional_vars  # the ``as <target>`` clause, if any

        # Map DSL function names to ScopeKind values
        _SCOPE_KIND_MAP = {
            "incore": ir.ScopeKind.InCore,
            "auto_incore": ir.ScopeKind.AutoInCore,
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
                        span=self.span_tracker.get_span(stmt),
                        hint="Drop the `as` target, or use `with pl.at(...) as tid:` / "
                        "`with pl.spmd(...) as tid:` to capture the outlined dispatch's "
                        "producer TaskId.",
                    )

                # Unified runtime scope: with pl.scope(mode=...): ...
                if func.attr == "scope":
                    self._parse_scope(stmt, context_expr)
                    return

                # Manual scope alias: with pl.manual_scope(): ...
                if func.attr == "manual_scope":
                    self._parse_manual_scope(stmt, context_expr)
                    return

                # Existing scope kinds: pl.incore(), pl.auto_incore(), pl.cluster(), pl.spmd()
                if func.attr in _SCOPE_KIND_MAP:
                    self._parse_legacy_scope(
                        stmt, context_expr, func.attr, _SCOPE_KIND_MAP, optional_vars=optional_vars
                    )
                    return

                # pl.at(level=..., role=..., deps=...) [as tid]
                if func.attr == "at":
                    self._parse_at_scope(stmt, context_expr, optional_vars=optional_vars)
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
        if _is_pl_call(stmt.value, "dump_tag"):
            self._handle_dump_tag(stmt)
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
        elif value is None:
            # ``None`` is the "no producer yet" TaskId sentinel — the Pythonic
            # spelling of an invalid PTO2TaskId. Used to seed a TaskId loop
            # carry (``prev_tid = None``) or as a ``deps=[None]`` entry.
            # Lowers to ``system.task_invalid`` -> Scalar[TASK_ID]; codegen
            # emits ``PTO2TaskId::invalid()`` and downstream ``set_dependencies``
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

        # ``pl.submit(...)`` is an assignment-only construct — it must be
        # unpacked as ``out, tid = pl.submit(self.kernel, ...)``. Reaching it
        # in an expression context (bare statement or simple assignment) is a
        # usage error.
        if _is_pl_call(call, "submit"):
            raise ParserSyntaxError(
                "pl.submit(...) must be unpacked as a 2-tuple: `out, tid = pl.submit(self.kernel, ...)`",
                span=self.span_tracker.get_span(call),
                hint="pl.submit returns (kernel result, producer TaskId); bind both with tuple unpacking.",
            )
        if _is_pl_call(call, "spmd_submit"):
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
                return self._parse_kernel_call(func, call.args, call.keywords, span, as_submit=False)

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

        # pld.<op> (2-segment unified short form)
        if len(attrs) == 2 and attrs[0] == "pld":
            return self._parse_pld_op(attrs[1], call)

        # pld.<category>.<op> (3-segment canonical form; category = system/tensor/tile)
        if len(attrs) == 3 and attrs[0] == "pld":
            return self._parse_pld_category_op(attrs[1], attrs[2], call)

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

        # pl.array.{operation} (3-segment)
        if len(attrs) >= 3 and attrs[0] == "pl" and attrs[1] == "array":
            op_name = attrs[2]
            return self._parse_array_op(op_name, call)

        # pl.const(value, dtype) — typed constant literal
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] == "const":
            return self._parse_typed_constant(call)

        # pl.{operation} (2-segment, unified dispatch or promoted ops)
        if len(attrs) >= 2 and attrs[0] == "pl" and attrs[1] not in ("tensor", "tile", "system", "array"):
            op_name = attrs[1]
            return self._parse_unified_op(op_name, call)

        raise UnsupportedFeatureError(
            f"Unsupported operation call: {ast.unparse(call)}",
            span=self.span_tracker.get_span(call),
            hint="Use pl.*, pl.tensor.*, pl.tile.*, or pl.system.* operations",
        )

    @staticmethod
    def _validate_kernel_call_kwargs(
        method_name: str,
        func_obj: ir.Function | None,
        keywords: list[ast.keyword],
        as_submit: bool,
        as_spmd: bool,
        span: ir.Span,
    ) -> None:
        """Reject unknown keyword arguments on a cross-function kernel call.

        ``attrs=`` surfaces call-site directions and is always allowed.
        ``deps=`` / ``dumps=`` are accepted only on ``pl.submit(...)``;
        ``core_num=`` / ``sync_start=`` only on ``pl.spmd_submit(...)``;
        ``device=`` only when the callee is an Orchestrator. Raises
        ``ParserTypeError`` with a targeted hint for the common
        deps/dumps-on-a-plain-call mistakes.
        """
        allowed_kwargs = {"attrs"}
        if as_submit:
            allowed_kwargs.add("deps")
            allowed_kwargs.add("dumps")
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

    def _parse_kernel_call(
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

        * the plain-call path in :meth:`parse_call` (``as_submit=False``) — a
          fire-and-forget kernel call; ``deps=`` is rejected and the call's
          return type is the callee's own.
        * the ``pl.submit(...)`` path in :meth:`_parse_submit_assignment`
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
        if method_name not in self.global_vars:
            raise UndefinedVariableError(
                f"Function '{method_name}' not defined in program",
                span=span,
                hint=f"Available functions: {list(self.global_vars.keys())}",
            )
        gvar = self.global_vars[method_name]
        func_obj = self.gvar_to_func.get(gvar)

        # Reject unknown kwargs (``attrs`` always; ``deps`` / ``dumps`` only on
        # submit; ``core_num`` / ``sync_start`` only on spmd_submit; ``device``
        # only for an Orchestrator callee).
        self._validate_kernel_call_kwargs(method_name, func_obj, keywords, as_submit, as_spmd, span)

        # Validate argument count before parsing args to fail fast.
        if func_obj is not None:
            self._validate_call_arg_count(method_name, func_obj, len(arg_nodes), span)

        arg_directions = self._extract_arg_directions_from_attrs(method_name, keywords, len(arg_nodes), span)
        if arg_directions is None:
            arg_directions = []
        # Manual deps live on Submit::deps_ only (ManualDepsOnSubmitOnly
        # invariant): there is no attrs-dict surface for them on any call kind.
        self._reject_manual_dep_edges_in_attrs(method_name, keywords, span)
        # Generic round-trip safety net: recover every attrs={...} key that has
        # no dedicated extractor (arg_direction_overrides, dummy_task, ...). The
        # printer's PrintAttrValue is the matching writer.
        extra_attrs = self._extract_generic_call_attrs(method_name, keywords, span)
        # Detect ``pl.no_dep(...)`` wrappers at call-arg positions and collect
        # their indices for the arg_direction_overrides attr.
        unwrapped_args, no_dep_indices = self._strip_call_arg_markers(arg_nodes)
        args = [self.parse_expression(arg) for arg in unwrapped_args]
        # ``pl.submit`` parses the optional ``deps=[tid, ...]`` (explicit-edge
        # attr) and ``dumps=[tensor, ...]`` (selective dump) kwargs. A plain
        # Call recovers its selective-dump targets from the machine-only
        # ``attrs={"dump_vars": [...]}`` round-trip dict (printed metadata), not
        # a user-facing kwarg.
        user_dep_vars: list[ir.Var] = []
        explicit_dump_vars: list[ir.Var] = []
        if as_submit:
            user_dep_vars = self._parse_submit_deps_kwarg(method_name, keywords, span)
            explicit_dump_vars = self._parse_submit_dumps_kwarg(method_name, args, keywords, span)
        else:
            explicit_dump_vars = self._extract_dump_vars_from_attrs(method_name, args, keywords, span)
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
            tagged = any(arg is t for t in self._dump_tagged_vars)
            in_dumps = any(arg is d for d in explicit_dump_vars)
            if (tagged or in_dumps) and isinstance(arg, ir.Var):
                dump_vars.append(arg)
        # Orchestration dispatch ``device=`` kwarg: resolves to a ConstInt or
        # an enclosing-loop induction Var.
        device_expr = self._parse_dispatch_device_kwarg(keywords)
        # ``pl.spmd_submit`` parses the SPMD launch spec (``core_num=`` /
        # ``sync_start=``) into a (positive int Expr, bool) pair recorded on
        # the resulting Submit.
        core_num_expr: ir.Expr | None = None
        sync_start = False
        if as_spmd:
            core_num_expr, sync_start = self._parse_spmd_submit_kwargs(method_name, keywords, span)
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
            return_types = self._effective_return_types(func_obj)
        if func_obj is not None and return_types:
            return_types = ir.deduce_call_return_type(
                list(func_obj.params),
                args,
                return_types,
            )
        return self._make_call_with_return_type(
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
            extra_attrs=extra_attrs,
        )

    def _is_python_resolvable_ast(
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
                if self.scope_manager.lookup_var(child.id) is not None:
                    # An IR Var shadows the name — cannot evaluate at parse time.
                    return False
        return True

    @staticmethod
    def _substitute_python_name_in_ast(node: ast.expr, name: str, value: Any) -> ast.expr:
        """Return a copy of ``node`` with every bare reference to ``name``
        replaced by an ``ast.Constant(value)``.

        Used by Form 2's body substitution: the comprehension's loop variable
        is bound to a per-iteration Python int, and the body AST is rewritten
        accordingly before being handed back to ``parse_expression``.
        """

        class _Substituter(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:  # noqa: N802
                if n.id == name and isinstance(n.ctx, ast.Load):
                    return ast.copy_location(ast.Constant(value=value), n)
                return n

        copied = copy.deepcopy(node)
        return cast(ast.expr, _Substituter().visit(copied))

    def _unroll_deps_comprehension(
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

        if not self._is_python_resolvable_ast(gen.iter):
            raise ParserTypeError(
                f"'{method_name}' deps= comprehension iterable depends on an IR variable; "
                f"cannot unroll at parse time",
                span=span,
                hint="Use a Python `range(...)`, a tuple of const ints, or a module-level list. "
                "If you need a runtime-bounded loop, write Form 1 (`arr[i]`) inside a `pl.range` body.",
            )
        for if_ in gen.ifs:
            if not self._is_python_resolvable_ast(if_, extra_python_names=frozenset([loop_name])):
                raise ParserTypeError(
                    f"'{method_name}' deps= comprehension filter depends on an IR variable; "
                    f"cannot unroll at parse time",
                    span=span,
                    hint="Filter clauses may reference only Python-level constants "
                    "(closures, globals, the comprehension's own loop var).",
                )

        iterable = self.expr_evaluator.eval_expr(gen.iter)
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

        saved_closure = self.expr_evaluator.closure_vars
        result: list[ast.expr] = []
        try:
            for value in iter_values:
                self.expr_evaluator.closure_vars = {**saved_closure, loop_name: value}
                keep = all(bool(self.expr_evaluator.eval_expr(if_)) for if_ in gen.ifs)
                if not keep:
                    continue
                result.append(self._substitute_python_name_in_ast(comp.elt, loop_name, value))
        finally:
            self.expr_evaluator.closure_vars = saved_closure
        return result

    def _synthesize_deps_array(self, entries: list[ir.Expr], span: ir.Span) -> ir.Var:
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
        buf_var = self.builder.let("_submit_deps_buf", create_call, span=span)
        for i, entry in enumerate(entries):
            idx_expr = ir.ConstInt(i, DataType.INDEX, span)
            upd_call = ir.create_op_call("array.update_element", [buf_var, idx_expr, entry], {}, span)
            buf_var = self.builder.let("_submit_deps_buf", upd_call, span=span)
        return buf_var

    def _parse_submit_deps_kwarg(  # noqa: PLR0912 — single-purpose validation flow; splitting hurts readability
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
            ast_entries: list[ast.expr] = self._unroll_deps_comprehension(deps_kw.value, method_name, span)
        elif isinstance(deps_kw.value, (ast.List, ast.Tuple)):
            ast_entries = []
            for elt in deps_kw.value.elts:
                if isinstance(elt, ast.ListComp):
                    ast_entries.extend(self._unroll_deps_comprehension(elt, method_name, span))
                else:
                    ast_entries.append(elt)
        else:
            raise ParserTypeError(
                f"'{method_name}' deps= must be a list / tuple / comprehension of TaskId values",
                span=self.span_tracker.get_span(deps_kw.value),
                hint="Use deps=[tid] where tid is a TaskId from a prior "
                "`_, tid = pl.submit(...)`, a loop iter_arg, `arr[i]` on a TASK_ID array, "
                "or `None`.",
            )

        direct_entries: list[ir.Expr] = []
        needs_desugar = False
        saw_whole_array = False
        for elt in ast_entries:
            elt_span = self.span_tracker.get_span(elt)
            # A bare ``None`` entry is the "no producer yet" sentinel — it
            # contributes no edge (an invalid TaskId would be skipped by the
            # codegen ``is_valid()`` guard anyway), so drop it here.
            if isinstance(elt, ast.Constant) and elt.value is None:
                continue
            expr = self.parse_expression(elt)
            if isinstance(expr, ir.Var) and _is_dep_var_type(expr.type):
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
            if _is_per_element_task_id_read(expr):
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
        synth = self._synthesize_deps_array(direct_entries, span)
        return [synth]

    def _parse_submit_dumps_kwarg(
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
                span=self.span_tracker.get_span(dumps_kw.value),
                hint="Write dumps=[x, y] listing tensors passed to this submit.",
            )
        result: list[ir.Var] = []
        for elt in dumps_kw.value.elts:
            elt_span = self.span_tracker.get_span(elt)
            val = self.parse_expression(elt)
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

    def _parse_dispatch_device_kwarg(
        self,
        keywords: list[ast.keyword],
    ) -> ir.Expr | None:
        """Extract the optional ``device=`` kwarg on a ``self.<orch>(...)`` call.

        ``device=`` selects the physical device for an orchestrator dispatch
        and is legal only when the callee has :class:`ir.Role.Orchestrator`
        (i.e. either chip-level ``FunctionType.Orchestration`` or a function
        declared with ``level=..., role=Role.Orchestrator``; both forms set
        the same ``role`` field on the IR Function). Callee-role validation
        happens in :meth:`parse_call` via the ``allowed_kwargs`` filter
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
        return self.parse_expression(device_kw.value)

    def _effective_return_types(self, func: ir.Function | None) -> list[ir.Type]:
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
        cached = self._derived_return_types_cache.get(func)
        if cached is None:
            cached = _derive_return_types_from_body(func.body)
            self._derived_return_types_cache[func] = cached
        return list(cached)

    def _make_call_with_return_type(  # noqa: PLR0913, PLR0912 — cohesive call-builder; bundling args/branches hurts clarity
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
            # the Submit; that requires the full ctor form even when there are
            # no attrs. A plain pl.submit with no attrs keeps the minimal form
            # so existing golden output is byte-identical.
            if submit_attrs is not None or core_num is not None:
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
                )
            return ir.Submit(gvar, args, deps_list, return_type, span)

        if attrs is not None:
            return ir.Call(gvar, args, {}, attrs, return_type, span)

        if not return_types:
            return ir.Call(gvar, args, span)
        return ir.Call(gvar, args, return_type, span)

    @staticmethod
    def _match_call_arg_marker(node: ast.expr, name: str) -> ast.expr | None:
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
    def _strip_call_arg_markers(
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
                inner = cls._match_call_arg_marker(node, "no_dep")
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
    def _reject_keyword_args(
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

    def _get_call_attrs_dict(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        span: ir.Span,
    ) -> ast.Dict | None:
        attrs_kw: ast.keyword | None = next((kw for kw in keywords if kw.arg == "attrs"), None)
        if attrs_kw is None:
            return None

        if not isinstance(attrs_kw.value, ast.Dict):
            raise ParserTypeError(
                f"'attrs=' on call to '{method_name}' must be a dict literal",
                span=self.span_tracker.get_span(attrs_kw.value),
                hint='e.g. attrs={"arg_directions": [pl.adir.input, ...]}',
            )
        # No key allowlist: every attrs key round-trips. The well-known keys
        # (arg_directions / manual_dep_edges / dump_vars) keep dedicated
        # extractors; every other key is recovered generically by
        # ``_extract_generic_call_attrs`` -> ``_parse_attr_value`` (the printer's
        # ``PrintAttrValue`` is the matching writer). Only the string-literal-key
        # invariant is enforced here.
        for key_node in attrs_kw.value.keys:
            if not (isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)):
                raise ParserSyntaxError(
                    f"'attrs=' on call to '{method_name}' must use string-literal keys",
                    span=self.span_tracker.get_span(key_node) if key_node else span,
                )
        return attrs_kw.value

    def _parse_attr_value(self, method_name: str, key: str, value_node: ast.expr) -> Any:
        """Reconstruct one generic ``attrs={...}`` value from its Python AST node.

        The matching writer is ``PrintAttrValue`` in the C++ printer
        (``src/ir/transforms/python_printer.cpp``). The value type is inferred
        from syntax (the "syntax inference only" round-trip contract): scalars
        from constants, ``[int, ...]`` -> index list, ``[pl.adir.X, ...]`` ->
        direction list, ``[name, ...]`` / bare ``name`` -> Var(s), anything else
        -> an IR expression. Syntactically ambiguous shapes (empty / mixed-kind
        lists) are REJECTED rather than guessed — never silently dropped —
        mirroring the printer's fail-loud behaviour.
        """
        from pypto.language.arg_direction import NAME_TO_DIRECTION  # noqa: PLC0415

        node_span = self.span_tracker.get_span(value_node)
        if isinstance(value_node, ast.Constant):
            v = value_node.value
            # bool is a subclass of int, so this also covers True/False.
            if isinstance(v, (bool, int, float, str)):
                return v
            raise ParserSyntaxError(
                f"attrs['{key}'] on call to '{method_name}': unsupported constant "
                f"'{ast.unparse(value_node)}'",
                span=node_span,
            )
        if isinstance(value_node, (ast.List, ast.Tuple)):
            elts = list(value_node.elts)
            if not elts:
                # Empty list: element type is not inferable from syntax, but the
                # binding (ConvertKwargsDict) types it by attr key (e.g.
                # arg_direction_overrides -> vector<int32_t>, the Var-list keys
                # -> vector<VarPtr>). Return [] so a printed empty vector attr
                # round-trips instead of raising.
                return []
            # All integer literals -> index list (e.g. arg_direction_overrides).
            if all(
                isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool)
                for e in elts
            ):
                return [e.value for e in elts]  # type: ignore[attr-defined]
            # All pl.adir.<name> -> ArgDirection list.
            if all(
                isinstance(e, ast.Attribute)
                and isinstance(e.value, ast.Attribute)
                and e.value.attr == "adir"
                and isinstance(e.value.value, ast.Name)
                and e.value.value.id == "pl"
                for e in elts
            ):
                dirs: list[ir.ArgDirection] = []
                for e in elts:
                    assert isinstance(e, ast.Attribute)
                    if e.attr not in NAME_TO_DIRECTION:
                        raise ParserSyntaxError(
                            f"attrs['{key}'] on call to '{method_name}': unknown direction "
                            f"'pl.adir.{e.attr}'",
                            span=self.span_tracker.get_span(e),
                        )
                    dirs.append(NAME_TO_DIRECTION[e.attr])
                return dirs
            # All bare names -> Var list (resolved in the current scope).
            if all(isinstance(e, ast.Name) for e in elts):
                return [self.parse_expression(e) for e in elts]
            raise ParserSyntaxError(
                f"attrs['{key}'] on call to '{method_name}': list elements are mixed or of an "
                "unsupported kind (expected all ints, all pl.adir.<name>, or all names)",
                span=node_span,
            )
        # Bare name -> Var; any other expression -> the parsed IR expression.
        return self.parse_expression(value_node)

    def _extract_generic_call_attrs(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        span: ir.Span,
    ) -> dict[str, Any]:
        """Recover every non-bespoke ``attrs={...}`` key via ``_parse_attr_value``.

        The well-known keys (arg_directions / manual_dep_edges / dump_vars) keep
        their dedicated extractors and are skipped here; this is the generic
        round-trip path for everything else (arg_direction_overrides,
        dummy_task, and any future attr the printer emits via ``PrintAttrValue``).
        """
        attrs_dict = self._get_call_attrs_dict(method_name, keywords, span)
        if attrs_dict is None:
            return {}
        bespoke = {"arg_directions", "manual_dep_edges", "dump_vars"}
        result: dict[str, Any] = {}
        for key_node, value_node in zip(attrs_dict.keys, attrs_dict.values):
            # _get_call_attrs_dict already validated string-literal keys.
            assert isinstance(key_node, ast.Constant) and isinstance(key_node.value, str)
            key = key_node.value
            if key in bespoke:
                continue
            result[key] = self._parse_attr_value(method_name, key, value_node)
        return result

    def _extract_arg_directions_from_attrs(
        self,
        method_name: str,
        keywords: list[ast.keyword],
        arg_count: int,
        span: ir.Span,
    ) -> list[ir.ArgDirection] | None:
        """Extract ``arg_directions`` from an ``attrs={...}`` kwarg."""
        from pypto.language.arg_direction import NAME_TO_DIRECTION  # noqa: PLC0415

        attrs_dict = self._get_call_attrs_dict(method_name, keywords, span)
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
                    span=self.span_tracker.get_span(value_node),
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
                        span=self.span_tracker.get_span(elt),
                    )
                if elt.attr not in NAME_TO_DIRECTION:
                    raise ParserSyntaxError(
                        f"Unknown call-site direction marker 'pl.adir.{elt.attr}' in call to '{method_name}'",
                        span=self.span_tracker.get_span(elt),
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

    def _reject_manual_dep_edges_in_attrs(
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
        attrs_dict = self._get_call_attrs_dict(method_name, keywords, span)
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

    def _extract_dump_vars_from_attrs(
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
        attrs_dict = self._get_call_attrs_dict(method_name, keywords, span)
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
                    span=self.span_tracker.get_span(value_node),
                )
            result: list[ir.Var] = []
            seen: set[int] = set()
            for elt in value_node.elts:
                elt_span = self.span_tracker.get_span(elt)
                val = self.parse_expression(elt)
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
                if id(val) not in seen:  # dedup by identity, matching _parse_at_dumps_kwarg
                    seen.add(id(val))
                    result.append(val)
            return result
        return []

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
            self._effective_return_types(ext_func),
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

    def _parse_op_positional_arg(self, arg: ast.expr) -> Any:
        """Parse a positional op argument.

        For ``ast.Attribute`` nodes (e.g. ``pl.INDEX``, ``pl.FP32``), try
        dtype resolution first so wrappers receive a ``DataType`` for slots
        like ``pl.cast(value, dtype)``. Falls through to ``parse_expression``
        for everything else, which keeps Tensor/Tile/Scalar var lookups,
        list literals, etc. on the existing path.
        """
        if isinstance(arg, ast.Attribute):
            try:
                return self.type_resolver.resolve_dtype(arg)
            except ParserError:
                pass
        return self.parse_expression(arg)

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
        """Dispatch an op call to a DSL wrapper module.

        The wrapper owns type-checking and dispatch (e.g. ``tile.add`` already
        routes scalar rhs to ``tile.adds``); the parser only parses args, wraps
        them as DSL types via :func:`invoke_dsl`, pins the call-site span
        through a contextvar, and unwraps the result.

        Args:
            module: A DSL op submodule (``_dsl_tensor`` / ``_dsl_tile`` /
                ``_dsl_system``) or the unified ``pypto.language.op`` namespace.
            module_name: Human-readable module name for error messages.
            op_name: Name of the operation to look up on the module.
            call: Call AST node.

        Returns:
            IR expression from the operation.
        """
        if not hasattr(module, op_name):
            raise InvalidOperationError(
                f"Unknown {module_name} operation: {op_name}",
                span=self.span_tracker.get_span(call),
                hint=f"Check if '{op_name}' is a valid {module_name} operation",
            )
        args = [self._parse_op_positional_arg(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        op_func = getattr(module, op_name)
        span = self.span_tracker.get_span(call)
        try:
            return invoke_dsl(op_func, args, kwargs, span)
        except ParserError:
            raise
        except (TypeError, ValueError) as e:
            # Wrapper may have prefixed its message (``pl.<op>:`` from
            # ``_raise_type_dispatch_error`` or ``pl.<module>.<op>:``) or raised
            # bare text from a deeper helper (e.g. ``ValueError("Invalid
            # rounding mode ...")``). Make sure the surfaced error always names
            # the op so users can locate the bad call. When any operand was a
            # Scalar, append a hint pointing at Python operators.
            msg = str(e)
            if not msg.startswith(f"pl.{op_name}") and not msg.startswith(f"{module_name}.{op_name}"):
                msg = f"{module_name} operation '{op_name}': {msg}"
            hint = self._scalar_operand_hint(args, kwargs)
            raise InvalidOperationError(msg, span=span, hint=hint) from e
        except Exception as e:
            raise InvalidOperationError(
                f"Error in {module_name} operation '{op_name}': {concise_error_message(e)}",
                span=span,
            ) from e

    @staticmethod
    def _scalar_operand_hint(args: list[Any], kwargs: dict[str, Any]) -> str | None:
        """Return the Python-operators hint when any operand has ScalarType.

        Replaces a fragile ``"Scalar" in msg`` substring check; the parsed
        operands carry their IR type explicitly, so we can decide based on
        that rather than the wrapper's error wording.
        """
        for v in (*args, *kwargs.values()):
            if isinstance(v, ir.Expr) and isinstance(v.type, ir.ScalarType):
                return (
                    "For scalar arithmetic / comparison, use Python operators directly "
                    "(e.g. `s1 + s2`, `s1 % s2`, `s1 == s2`)."
                )
        return None

    def _parse_tensor_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tensor operation."""
        if op_name == "alloc":
            return self._parse_printed_alloc_call(call)
        # Prefer the DSL wrapper (owns type-checking + dispatch); fall back to
        # the IR-builder layer for pass-internal ops like ``gather_mask``
        # that are emitted by the printer but have no DSL wrapper.
        if hasattr(_dsl_tensor, op_name):
            return self._dispatch_op(_dsl_tensor, "pl.tensor", op_name, call)
        return self._dispatch_ir_builder_op(ir_op.tensor, "pl.tensor", op_name, call)

    def _parse_tile_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse tile operation."""
        if op_name == "alloc":
            return self._parse_printed_alloc_call(call)
        if hasattr(_dsl_tile, op_name):
            return self._dispatch_op(_dsl_tile, "pl.tile", op_name, call)
        return self._dispatch_ir_builder_op(ir_op.tile, "pl.tile", op_name, call)

    def _dispatch_ir_builder_op(self, module: Any, module_name: str, op_name: str, call: ast.Call) -> ir.Expr:
        """Dispatch to a raw IR-builder op (no DSL wrapping).

        Used as a fallback when the DSL layer doesn't expose an op that the
        printer emitted (e.g. pass-internal lowerings like ``tile.gather_mask``,
        ``tile.mrgsort_format1``). IR builders take ``span=`` explicitly and
        accept raw ``ir.Expr`` arguments — no DSL wrap/unwrap round-trip.
        """
        if not hasattr(module, op_name):
            raise InvalidOperationError(
                f"Unknown {module_name} operation: {op_name}",
                span=self.span_tracker.get_span(call),
                hint=f"Check if '{op_name}' is a valid {module_name} operation",
            )
        args = [self._parse_op_positional_arg(arg) for arg in call.args]
        kwargs = self._parse_op_kwargs(call)
        op_func = getattr(module, op_name)
        span = self.span_tracker.get_span(call)
        try:
            return op_func(*args, **kwargs, span=span)
        except ParserError:
            raise
        except Exception as e:
            raise InvalidOperationError(
                f"Error in {module_name} operation '{op_name}': {concise_error_message(e)}",
                span=span,
            ) from e

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
        if op_name == "task_dummy":
            return self._parse_task_dummy(call)
        return self._dispatch_op(_dsl_system, "pl.system", op_name, call)

    def _parse_task_dummy(self, call: ast.Call) -> ir.Expr:
        """Parse user-written ``pl.system.task_dummy(deps=[...])``."""
        span = self.span_tracker.get_span(call)
        if call.args:
            raise InvalidOperationError(
                "pl.system.task_dummy must not use positional arguments",
                span=span,
            )
        allowed_kwargs = {"deps"}
        for kw in call.keywords:
            if kw.arg not in allowed_kwargs:
                raise ParserTypeError(
                    f"pl.system.task_dummy does not accept keyword argument '{kw.arg}'",
                    span=span,
                    hint="Use pl.system.task_dummy(deps=[task_id]) where each dep is a TaskId value",
                )
        if not any(kw.arg == "deps" for kw in call.keywords):
            raise ParserTypeError(
                "pl.system.task_dummy requires keyword argument 'deps'",
                span=span,
                hint="Use pl.system.task_dummy(deps=[]) for an empty dummy barrier",
            )
        deps = self._parse_submit_deps_kwarg("pl.system.task_dummy", call.keywords, span)
        base = ir.create_op_call("system.task_dummy", [], {}, span)
        attrs: list[tuple[str, Any]] = [("dummy_task", True)]
        if deps:
            attrs.append(("manual_dep_edges", deps))
        return ir.Call(base.op, base.args, base.kwargs, attrs, base.type, base.span)

    def _parse_array_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse array operation (create / get_element / update_element)."""
        return self._dispatch_op(_dsl_array, "pl.array", op_name, call)

    def _validate_pld_op_call(self, op_name: str, call: ast.Call) -> None:
        """Parser-context checks shared by 2-segment and 3-segment pld paths.

        - ``alloc_window_buffer``: must be intercepted at assignment-LHS.
          Reaching here means it was used outside that context.
        - ``world_size``: host-only.
        """
        span = self.span_tracker.get_span(call)

        if op_name == "alloc_window_buffer":
            raise ParserSyntaxError(
                "pld.tensor.alloc_window_buffer must appear as the RHS of a simple assignment "
                "(its result must be bound to a named variable)",
                span=span,
                hint="Write 'buf = pld.tensor.alloc_window_buffer(N)' "
                "(or the short form 'buf = pld.alloc_window_buffer(N)')",
            )

        if op_name == "world_size":
            in_device_scope = any(
                self._is_inside_scope(kind)
                for kind in (ir.ScopeKind.InCore, ir.ScopeKind.AutoInCore, ir.ScopeKind.Spmd)
            )
            if self._func_level != ir.Level.HOST or in_device_scope:
                raise ParserSyntaxError(
                    "pld.system.world_size() can only be called in HOST orchestration context "
                    "(not inside InCore / AutoInCore / SPMD scopes); "
                    f"current function level: {self._func_level}",
                    span=span,
                    hint="Use '@pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)' "
                    "on the enclosing function and call outside any nested device-side scope",
                )

    def _parse_pld_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse ``pld.<op>(...)`` — 2-segment unified short form.

        Forwards to the 3-segment surface via the unified-dispatch shim in
        :mod:`pypto.language.distributed.op.unified_ops`.
        """
        self._validate_pld_op_call(op_name, call)
        span = self.span_tracker.get_span(call)

        if not hasattr(_dsl_pld, op_name):
            raise InvalidOperationError(
                f"Unknown distributed operation 'pld.{op_name}'",
                span=span,
                hint="Available short forms: pld.world_size, pld.get_comm_ctx, pld.rank, "
                "pld.nranks, pld.alloc_window_buffer, pld.window, pld.remote_load",
            )

        return self._dispatch_op(_dsl_pld, "pld", op_name, call)

    def _parse_pld_category_op(self, category: str, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse ``pld.<category>.<op>(...)`` — 3-segment canonical form.

        Categories: ``system`` / ``tensor`` / ``tile`` (mirrors ``pl.system`` /
        ``pl.tensor`` / ``pl.tile``).
        """
        span = self.span_tracker.get_span(call)
        self._validate_pld_op_call(op_name, call)

        if category not in _PLD_CATEGORIES or not hasattr(_dsl_pld, category):
            raise InvalidOperationError(
                f"Unknown distributed category 'pld.{category}'",
                span=span,
                hint="Available categories: pld.system, pld.tensor, pld.tile",
            )

        submodule = getattr(_dsl_pld, category)
        if not hasattr(submodule, op_name):
            raise InvalidOperationError(
                f"Unknown distributed operation 'pld.{category}.{op_name}'",
                span=span,
                hint=f"Check spelling, or list available ops in pypto.language.distributed.op.{category}_ops",
            )

        return self._dispatch_op(submodule, f"pld.{category}", op_name, call)

    # Maps iterator type name to ForKind enum value.
    _ITERATOR_TO_KIND = {
        "range": ir.ForKind.Sequential,
        "parallel": ir.ForKind.Parallel,
        "unroll": ir.ForKind.Unroll,
        "pipeline": ir.ForKind.Pipeline,
    }

    def _parse_unified_op(self, op_name: str, call: ast.Call) -> ir.Expr:
        """Parse a ``pl.<op>(...)`` call by delegating to the matching DSL wrapper.

        Lookup is restricted to ``pypto.language.op`` rather than the broader
        ``pypto.language`` namespace. The op package re-exports only callable
        ops (unified + promoted tensor/tile/system), so non-op DSL symbols
        like ``pl.range``, ``pl.dynamic``, or ``pl.const`` cannot be invoked
        here as ops — they're handled by their own parser entry points
        upstream of this method. The wrapper itself owns type-checking and
        dispatch (e.g. ``unified_ops.add`` routes Tile+Scalar to
        ``tile.adds``); the parser only forwards arguments and the call-site
        span.
        """
        import pypto.language.op as _pl_op  # noqa: PLC0415 (circular import — `pypto.language` re-exports parser)

        op_func = getattr(_pl_op, op_name, None)
        if op_func is None or not callable(op_func):
            raise InvalidOperationError(
                f"Unknown operation 'pl.{op_name}'",
                span=self.span_tracker.get_span(call),
                hint="Check spelling, or use pl.tensor.*/pl.tile.*/pl.system.* for explicit namespacing",
            )
        return self._dispatch_op(_pl_op, "pl", op_name, call)

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
        """Parse a subscript expression: ``tuple[0]``, ``tensor[i, j]``, ``tile[i:j]``, ...

        Supports:
        - TupleType: ``t[0]`` -> TupleGetItemExpr
        - TensorType / TileType: numpy-style indexing — a scalar index removes its
          dim, a slice keeps it, missing trailing indices are implicit ``:``.
          ``A[i, j]`` on a 2D operand (all-scalar, full-rank) -> scalar read;
          everything else (``A[i]``, ``A[i, j]`` on a >2D operand, ``A[i:i+8, j]``,
          chained ``A[i][j]``) -> a ``tensor.slice`` / ``tile.slice`` view carrying
          ``drop_dims``. Tiles are clamped to a 2D minimum (see _parse_tile_subscript).
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
        if isinstance(value_type, ir.ArrayType):
            return self._parse_array_subscript(subscript, value_expr, span)

        raise ParserTypeError(
            f"Subscript requires Tuple, Tensor, Tile, or Array type, got {type(value_type).__name__}",
            span=span,
            hint="Subscript syntax is supported for Tuple, Tensor, Tile, and Array types",
        )

    def _parse_array_subscript(self, subscript: ast.Subscript, value_expr: ir.Expr, span: ir.Span) -> ir.Expr:
        """Parse array subscript: ``arr[i]`` -> array.get_element."""
        index_node = subscript.slice
        if isinstance(index_node, ast.Tuple):
            raise ParserTypeError(
                "Array subscript requires a single index (rank-1 in v1)",
                span=span,
                hint="Use arr[i], not arr[i, j]",
            )
        index_expr = self.parse_expression(index_node)
        return ir.create_op_call("array.get_element", [value_expr, index_expr], {}, span)

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

    def _get_arith_analyzer(self) -> "_arith.Analyzer":
        """Return a parser-scoped, lazily-constructed arithmetic analyzer."""
        if self._arith_analyzer is None:
            self._arith_analyzer = _arith.Analyzer()
        return self._arith_analyzer

    def _build_slice_shape_expr(
        self,
        upper_expr: int | ir.Expr,
        lower_expr: int | ir.Expr,
    ) -> int | ir.Expr:
        """Build a slice extent, folding compile-time constants when possible.

        Symbolic extents are passed through the arithmetic simplifier so common
        patterns like ``x * s + s - x * s`` or ``(k + C) - k`` collapse to the
        scalar extent at construction time. This keeps downstream shape checks
        and reassignment guards from seeing structurally-distinct-but-equal
        expressions.
        """
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
        simplified = self._get_arith_analyzer().simplify(ir.sub(lhs, rhs))
        const_value = _const_int_value(simplified)
        if const_value is not None:
            return const_value
        return simplified

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
    ) -> tuple[list[int | ir.Expr], list[int | ir.Expr], list[int]]:
        """Convert mixed/partial subscript indices into slice shape/offset/drop_dims args.

        ``shape``/``offset`` are always full-rank: a scalar index ``i`` at dim
        ``d`` contributes ``shape[d] = 1, offset[d] = i`` and adds ``d`` to
        ``drop_dims`` (numpy-style rank reduction); a slice keeps its dim; dims
        not covered by ``indices`` become implicit ``:`` and are not dropped.
        """
        shape_exprs: list[int | ir.Expr] = []
        offset_exprs: list[int | ir.Expr] = []
        drop_dims: list[int] = []

        for dim_idx, idx in enumerate(indices):
            if not isinstance(idx, ast.Slice):
                offset_exprs.append(self.parse_expression(idx))
                shape_exprs.append(1)
                drop_dims.append(dim_idx)
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

        # Dims past the supplied indices are implicit ``:`` (kept, full extent).
        for dim_idx in range(len(indices), len(full_shape)):
            offset_exprs.append(0)
            shape_exprs.append(full_shape[dim_idx])

        return shape_exprs, offset_exprs, drop_dims

    def _build_tile_subscript_slice_args(
        self,
        indices: list[ast.expr],
        tile_type: ir.TileType,
        span: ir.Span,
    ) -> tuple[list[int | ir.Expr], list[int | ir.Expr], list[int | ir.Expr] | None, list[int]]:
        """Convert tile subscripts into static shape/offset args plus optional valid_shape.

        Same rank-reducing rules as :meth:`_build_subscript_slice_args`: a scalar
        index at dim ``d`` contributes a unit dim and adds ``d`` to ``drop_dims``;
        a slice keeps its dim; dims past ``indices`` are implicit ``:``. The
        ``tile.slice`` deducer clamps the result back to 2D if rank reduction
        would take it below 2D.
        """
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
        drop_dims: list[int] = []
        needs_valid_shape = False

        for dim_idx, idx in enumerate(indices):
            if not isinstance(idx, ast.Slice):
                index_expr = self.parse_expression(idx)
                offset_exprs.append(index_expr)
                shape_exprs.append(1)
                valid_shape_exprs.append(1)
                drop_dims.append(dim_idx)
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

        # Dims past the supplied indices are implicit ``:`` (kept, full extent).
        for dim_idx in range(len(indices), len(static_shape)):
            offset_exprs.append(0)
            shape_exprs.append(static_shape[dim_idx])
            valid_shape_exprs.append(logical_shape[dim_idx])
            needs_valid_shape = needs_valid_shape or not self._slice_extents_match(
                static_shape[dim_idx], logical_shape[dim_idx]
            )

        return shape_exprs, offset_exprs, valid_shape_exprs if needs_valid_shape else None, drop_dims

    def _parse_tensor_subscript(
        self,
        subscript: ast.Subscript,
        value_expr: ir.Expr,
        tensor_type: ir.TensorType,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse a tensor subscript with numpy-style rank-reducing / partial / chained indexing.

        Each scalar index removes its dim; a slice keeps it; missing trailing
        indices are implicit ``:``. ``A[i, j]`` on a 2D tensor (all-scalar,
        full-rank) reads a scalar via ``tensor.read``; everything else (``A[i]``,
        ``A[i, j]`` on a >2D tensor, ``A[i:i+8, j]``, ``A[i][j]``, ...) is a
        ``tensor.slice`` view, carrying ``drop_dims`` for the scalar-indexed axes.
        """
        indices = self._normalize_subscript_indices(subscript, span)
        rank = len(tensor_type.shape)
        if len(indices) > rank:
            raise ParserTypeError(
                f"Tensor subscript has {len(indices)} indices but the tensor is {rank}D",
                span=span,
                hint=f"Provide at most {rank} indices for a {rank}D tensor",
            )

        has_slice = any(isinstance(idx, ast.Slice) for idx in indices)

        if not has_slice and len(indices) == rank:
            # All-scalar, full-rank -> scalar element read.
            idx_exprs: list[int | ir.Expr] = [self.parse_expression(idx) for idx in indices]
            return ir_op.tensor.read(value_expr, idx_exprs, span=span)

        shape_exprs, offset_exprs, drop_dims = self._build_subscript_slice_args(
            indices, list(tensor_type.shape), span, "tensor"
        )
        return ir_op.tensor.slice(value_expr, shape_exprs, offset_exprs, drop_dims=drop_dims, span=span)

    def _parse_tile_subscript(
        self,
        subscript: ast.Subscript,
        value_expr: ir.Expr,
        tile_type: ir.TileType,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse a tile subscript with numpy-style rank-reducing / partial / chained indexing.

        Same rules as :meth:`_parse_tensor_subscript`, plus a tile-only floor:
        tiles are physically 2D, so a result that would naturally be < 2D is
        auto-promoted to 2D (unit axes prepended) with a non-fatal warning.
        """
        indices = self._normalize_subscript_indices(subscript, span)
        rank = len(tile_type.shape)
        if len(indices) > rank:
            raise ParserTypeError(
                f"Tile subscript has {len(indices)} indices but the tile is {rank}D",
                span=span,
                hint=f"Provide at most {rank} indices for a {rank}D tile",
            )

        has_slice = any(isinstance(idx, ast.Slice) for idx in indices)

        if not has_slice and len(indices) == rank:
            # All-scalar, full-rank -> scalar element read.
            idx_exprs: list[int | ir.Expr] = [self.parse_expression(idx) for idx in indices]
            return ir_op.tile.read(value_expr, idx_exprs, span=span)

        shape_exprs, offset_exprs, valid_shape_exprs, drop_dims = self._build_tile_subscript_slice_args(
            indices, tile_type, span
        )
        natural_rank = rank - len(drop_dims)
        if drop_dims and natural_rank < 2:
            kept = [shape_exprs[i] for i in range(rank) if i not in set(drop_dims)]
            promoted = [1] * (2 - len(kept)) + list(kept)
            promoted_repr = "[" + ", ".join(self._render_dim_for_msg(d) for d in promoted) + "]"
            warnings.warn(
                f"tile subscript reduces to {natural_rank}D; auto-promoting to 2D shape {promoted_repr} "
                f"— use an explicit tile.reshape if you want a different layout",
                UserWarning,
                stacklevel=2,
            )
        return ir_op.tile.slice(
            value_expr, shape_exprs, offset_exprs, valid_shape_exprs, drop_dims=drop_dims, span=span
        )

    @staticmethod
    def _render_dim_for_msg(dim: int | ir.Expr) -> str:
        """Render a shape dim for a warning/error message (constant value, else ``?``)."""
        if isinstance(dim, int):
            return str(dim)
        value = _const_int_value(dim) if isinstance(dim, ir.Expr) else None
        return str(value) if value is not None else "?"

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

    def _check_condition_is_bool(self, condition: Any, kind: str, span: Any) -> None:
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

            # Descend into `with pl.scope():` blocks — they are transparent to
            # yield association, so a trailing `var = pl.yield_(...)` inside the
            # scope still defines the enclosing for/if return-var. These are
            # inserted by the MaterializeRuntimeScopes pass (or written by hand).
            elif isinstance(stmt, ast.With) and self._is_runtime_scope_with(stmt):
                yield_vars.extend(self._scan_for_yields(stmt.body))

        return yield_vars

    @staticmethod
    def _is_runtime_scope_with(stmt: ast.With) -> bool:
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


__all__ = ["ASTParser"]
