# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure IR shape/type comparison and annotation normalization helpers."""

from collections.abc import Sequence

from pypto.pypto_core import arith as _arith
from pypto.pypto_core import ir

from .type_resolver import (
    _const_int_value,  # noqa: PLC2701
    _implicit_tile_view_defaults,  # noqa: PLC2701
)


def fold_const_slice_extent(upper: object, lower: object) -> int | None:
    """Fold a slice extent when both bounds are compile-time constants."""
    upper_value = _const_int_value(upper)
    lower_value = _const_int_value(lower)
    if upper_value is None or lower_value is None:
        return None
    return upper_value - lower_value


def get_source_valid_shape(source_type: ir.Type) -> list[ir.Expr] | None:
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


def collect_type_symbol_ids(param_type: ir.Type) -> set[int]:
    """Return ``id()`` of every Var read by a parameter type's shape metadata.

    Used to detect a scalar parameter that shadows a ``pl.dynamic()`` symbol an
    earlier parameter annotation already resolved, which cannot be rebound.
    """
    symbol_ids: set[int] = set()

    def walk(expr: ir.Expr | None) -> None:
        if expr is None:
            return
        if isinstance(expr, ir.Var):
            symbol_ids.add(id(expr))
            return
        for attr in ("left", "right", "operand"):
            operand = getattr(expr, attr, None)
            if isinstance(operand, ir.Expr):
                walk(operand)

    if isinstance(param_type, ir.TensorType):
        for dim in param_type.shape:
            walk(dim)
        view = param_type.tensor_view
        if view is not None:
            for dim in list(view.valid_shape) + list(view.stride):
                walk(dim)
    return symbol_ids


def shape_exprs_match(lhs: Sequence[ir.Expr], rhs: Sequence[ir.Expr]) -> bool:
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


def has_printable_tile_view(
    tile_view: ir.TileView | None,
    shape: Sequence[ir.Expr],
    memory_space: ir.MemorySpace | None = None,
) -> bool:
    """Return whether a TileView carries non-default metadata in Python text form."""
    if tile_view is None:
        return False
    if tile_view.valid_shape and not shape_exprs_match(tile_view.valid_shape, shape):
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
    if tile_view.compact != ir.CompactMode.null:
        return True
    return False


def normalize_type_for_syntax_match(type_: ir.Type | None) -> ir.Type | None:
    """Normalize omitted default TileView metadata before syntax-level comparison."""
    if isinstance(type_, ir.TileType):
        tile_view = type_.tile_view
        if tile_view is not None and not has_printable_tile_view(tile_view, type_.shape, type_.memory_space):
            tile_view = None
        if tile_view is type_.tile_view:
            return type_
        return ir.TileType(type_.shape, type_.dtype, type_.memref, tile_view, type_.memory_space)

    return type_


def shape_has_symbolic_dim(type_: ir.Type) -> bool:
    """Return True when *type_* is a shaped type with at least one non-ConstInt dim."""
    if not isinstance(type_, (ir.TensorType, ir.TileType)):
        return False
    return any(not isinstance(d, ir.ConstInt) for d in type_.shape)


def simplify_shape_dims(type_: ir.Type, analyzer: "_arith.Analyzer") -> ir.Type:
    """Rebuild a TensorType/TileType with each outer shape dim passed through
    the arithmetic simplifier.

    Needed so that symbolic dims that reduce to the same value — e.g.
    ``(k + C) - k`` and ``C`` — compare equal under structural ``==``.

    Scope: only the outer ``shape`` is simplified. TileView fields
    (``valid_shape``, ``stride``, ``start_offset``) and TensorView
    fields retain their original expressions and still compare structurally.
    """
    if not shape_has_symbolic_dim(type_):
        return type_
    assert isinstance(type_, (ir.TensorType, ir.TileType))
    # The isinstance keeps a mistyped shape element diagnosable instead of
    # crashing the simplify call.
    simplified_shape = [
        analyzer.simplify(d)
        if isinstance(d, ir.Expr)  # pyright: ignore[reportUnnecessaryIsInstance]
        else d
        for d in type_.shape
    ]
    if isinstance(type_, ir.DistributedTensorType):
        return ir.DistributedTensorType(
            simplified_shape, type_.dtype, type_.memref, type_.tensor_view, type_.window_buffer
        )
    if isinstance(type_, ir.TensorType):
        return ir.TensorType(simplified_shape, type_.dtype, type_.memref, type_.tensor_view)
    return ir.TileType(simplified_shape, type_.dtype, type_.memref, type_.tile_view, type_.memory_space)


def types_match(lhs: ir.Type | None, rhs: ir.Type | None) -> bool:
    """Return whether two IR types are structurally equal after TileView normalization.

    Uses C++ ``structural_equal`` (via ``==``) instead of comparing printed
    strings. This caller explicitly removes ``memref`` metadata: it is irrelevant for
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
    lhs = normalize_type_for_syntax_match(ir._without_type_memref(lhs))
    rhs = normalize_type_for_syntax_match(ir._without_type_memref(rhs))
    assert lhs is not None
    assert rhs is not None
    if type(lhs) is not type(rhs):
        return False
    if shape_has_symbolic_dim(lhs) or shape_has_symbolic_dim(rhs):
        analyzer = _arith.Analyzer()
        lhs = simplify_shape_dims(lhs, analyzer)
        rhs = simplify_shape_dims(rhs, analyzer)
    return lhs == rhs


def tile_types_differ_only_by_unset_memory_space(lhs: ir.Type | None, rhs: ir.Type | None) -> bool:
    """Return whether two TileTypes are equal once an unset space is a wildcard.

    ``memory_space=None`` is the IR's "the compiler will place this", not a
    distinct type. Ops that leave it unset (``pl.load``, ``pl.tile.create``) and
    ops whose space is fixed by hardware (every vector op, via
    ``set_output_memory(Vec)``) therefore mix freely in ordinary user code::

        t = pl.load(a, [0, 0], [16, 64])   # unset
        t = pl.add(t, t)                   # Vec

    Treating that as a type change would reject the most common shape of DSL
    there is. All other fields still compare exactly.

    Kept separate from ``_types_match``: annotation-vs-inferred comparisons must
    stay strict, so that a *stated* space which contradicts the producer is
    still an error.
    """
    if not isinstance(lhs, ir.TileType) or not isinstance(rhs, ir.TileType):
        return False
    if (lhs.memory_space is None) == (rhs.memory_space is None):
        return False  # both set or both unset -> plain _types_match already decided
    unified = ir.TileType(rhs.shape, rhs.dtype, rhs.memref, rhs.tile_view, lhs.memory_space)
    return types_match(lhs, unified)


def normalize_inferred_type_for_annotation(
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
    inferred_type = normalize_type_for_syntax_match(value_expr.type)
    assert inferred_type is not None
    if not (
        isinstance(annotation_type, ir.TileType)
        and isinstance(inferred_type, ir.TileType)
        and isinstance(value_expr, ir.Call)
        and value_expr.op.name in {ir.get_op("tile.load").name, ir.get_op("tile.create").name}
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


class FirstReturnTypeCollector(ir.IRVisitor):
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


def derive_return_types_from_body(body: ir.Stmt | None) -> list[ir.Type]:
    """Effective return types derived from a function body's first return.

    Returns an empty list when *body* is ``None``, has no value-bearing
    ``return`` (a void function), or the returned values are not all concretely
    typed. In the last case there is nothing to recover — leaving the result
    empty keeps the call's type ``UnknownType`` exactly as before.
    """
    if body is None:
        return []
    collector = FirstReturnTypeCollector()
    collector.visit_stmt(body)
    types = collector.types
    if not types or any(isinstance(t, ir.UnknownType) for t in types):
        return []
    return types


class ShapeDimAliasExpander(ir.IRMutator):
    """Expand direct dimension-read temporaries only for annotation comparison."""

    def __init__(self, aliases: dict[int, ir.Call]) -> None:
        super().__init__()
        self.aliases = aliases

    def visit_var(self, op: ir.Var) -> ir.Expr:
        return self.aliases.get(op.unique_id, op)

    def visit_iter_arg(self, op: ir.IterArg) -> ir.Expr:
        return op

    def visit_call(self, op: ir.Call) -> ir.Expr:
        # A dimension read is an atomic shape value. Do not traverse the
        # tensor operand's type or substitute unrelated scalar computations.
        return op


def tensor_types_differ_only_by_valid_shape(lhs: ir.Type | None, rhs: ir.Type | None) -> bool:
    """Allow pre-SSA rebinding to refine only an ordinary tensor's valid region.

    A fresh matmul product can narrow a full-box accumulator seed. Preserve the
    RHS type for ConvertToSSA to version, while keeping shape, dtype and storage
    layout checks strict. This is not annotation compatibility or a rule for
    distributed windows.
    """
    if type(lhs) is not ir.TensorType or type(rhs) is not ir.TensorType:
        return False

    def without_valid_shape(tensor_type: ir.TensorType) -> ir.TensorType:
        view = tensor_type.tensor_view
        if view is not None:
            view = ir.TensorView(view.stride, view.layout, [], view.pad)
        return ir.TensorType(tensor_type.shape, tensor_type.dtype, tensor_type.memref, view)

    return types_match(without_valid_shape(lhs), without_valid_shape(rhs))


class InlineReturnProvenanceCollector(ir.IRVisitor):
    """Check that a trailing inline return has no competing return or rebound formal."""

    def __init__(self) -> None:
        super().__init__()
        self.return_count = 0
        self.assigned_ids: set[int] = set()
        self.assignment_counts: dict[int, int] = {}

    def visit_return_stmt(self, op: ir.ReturnStmt) -> None:
        self.return_count += 1
        super().visit_return_stmt(op)

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        self.assigned_ids.add(op.var.unique_id)
        self.assignment_counts[op.var.unique_id] = self.assignment_counts.get(op.var.unique_id, 0) + 1
        super().visit_assign_stmt(op)


def inline_alias_origin(value: ir.Expr, origins: dict[int, ir.Expr]) -> ir.Expr | None:
    """Resolve transparent aliases to an earlier formal or tuple projection."""
    if isinstance(value, ir.Var):
        return origins.get(value.unique_id)
    if isinstance(value, ir.TupleGetItemExpr):
        container = inline_alias_origin(value.tuple, origins)
        if (
            container is not None
            and isinstance(container.type, ir.TupleType)
            and 0 <= value.index < len(container.type.types)
        ):
            return ir.TupleGetItemExpr(container, value.index, value.span)
    return None


def inline_immutable_aliases(
    func: ir.Function, collector: InlineReturnProvenanceCollector
) -> dict[int, ir.Expr]:
    """Cache single-definition, top-level aliases without tracing computed values.

    Definition order compresses a Var chain as it is read, so long chains are
    resolved once per callee. Only already-proven origins qualify; forward
    references, cycles, nested assignments, and rebound formals stay unresolved.
    """
    parameter_ids = {param.unique_id for param in func.params}
    origins: dict[int, ir.Expr] = {
        param.unique_id: param for param in func.params if param.unique_id not in collector.assigned_ids
    }
    statements = func.body.stmts if isinstance(func.body, ir.SeqStmts) else (func.body,)
    for stmt in statements:
        if not isinstance(stmt, ir.AssignStmt):
            continue
        target = stmt.var.unique_id
        if target in parameter_ids or collector.assignment_counts[target] != 1:
            continue
        origin = inline_alias_origin(stmt.value, origins)
        if origin is not None:
            origins[target] = origin
    return {identity: origin for identity, origin in origins.items() if identity not in parameter_ids}


def returned_tile_types(actual: ir.Type, fallback: ir.Type) -> ir.Type | None:
    """Recover only returned Tile metadata; other result kinds retain native deduction."""
    if isinstance(actual, ir.TileType) and isinstance(fallback, ir.TileType):
        return actual
    if isinstance(actual, ir.TupleType) and isinstance(fallback, ir.TupleType):
        if len(actual.types) != len(fallback.types):
            return None
        projected = [returned_tile_types(a, f) for a, f in zip(actual.types, fallback.types)]
        if any(t is not None for t in projected):
            return ir.TupleType([p if p is not None else f for p, f in zip(projected, fallback.types)])
    return None


def inline_argument_return_type(
    value: ir.Expr,
    fallback: ir.Type,
    argument_types: dict[int, ir.Type],
    assigned_ids: frozenset[int],
    aliases: dict[int, ir.Expr],
) -> ir.Type | None:
    """Trace formal/immutable-alias returns by identity, without guessing computed results."""
    if isinstance(value, ir.Var):
        actual = argument_types.get(value.unique_id)
        if actual is not None and value.unique_id not in assigned_ids:
            return returned_tile_types(actual, fallback)
        origin = aliases.get(value.unique_id)
        if origin is not None:
            return inline_argument_return_type(origin, fallback, argument_types, assigned_ids, aliases)
    elif isinstance(value, ir.MakeTuple) and isinstance(fallback, ir.TupleType):
        if len(value.elements) != len(fallback.types):
            return None
        projected = [
            inline_argument_return_type(v, f, argument_types, assigned_ids, aliases)
            for v, f in zip(value.elements, fallback.types)
        ]
        if any(t is not None for t in projected):
            return ir.TupleType([p if p is not None else f for p, f in zip(projected, fallback.types)])
    elif isinstance(value, ir.TupleGetItemExpr):
        container = inline_argument_return_type(
            value.tuple, value.tuple.type, argument_types, assigned_ids, aliases
        )
        if isinstance(container, ir.TupleType) and 0 <= value.index < len(container.types):
            return returned_tile_types(container.types[value.index], fallback)
    return None
