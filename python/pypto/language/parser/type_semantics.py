# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure IR shape/type comparison and annotation normalization helpers.

The syntax-level comparison family is shared through
``pypto.language.typing.type_utils``. Parser-specific normalization and
return-provenance analysis remain here; compatibility aliases preserve the
parser's existing import paths.
"""

from collections.abc import Callable

from pypto.language.typing.type_utils import (  # noqa: PLC2701 — the shared syntax-level seam
    _const_int_value,
    _has_printable_tile_view,
    _normalize_type_for_syntax_match,
    _shape_exprs_match,
    _shape_has_symbolic_dim,
    _simplify_shape_dims,
    _tensor_types_differ_only_by_valid_shape,
    _tile_types_differ_only_by_unset_memory_space,
    _types_match,
)
from pypto.pypto_core import ir

# The parser lane's public spellings of the shared seam's comparison family:
# one implementation (``typing.type_utils``), re-exposed under the names the
# parser's callers import, so both frontends compare identically.
normalize_type_for_syntax_match = _normalize_type_for_syntax_match
shape_exprs_match = _shape_exprs_match
tensor_types_differ_only_by_valid_shape = _tensor_types_differ_only_by_valid_shape
tile_types_differ_only_by_unset_memory_space = _tile_types_differ_only_by_unset_memory_space
types_match = _types_match

# Shape/view helpers that were also part of this module's surface before the
# seam existed, so they stay importable from here. A frontend (or a downstream
# user) that imported them from this path must not break because the
# implementation moved one package over.
has_printable_tile_view = _has_printable_tile_view
shape_has_symbolic_dim = _shape_has_symbolic_dim
simplify_shape_dims = _simplify_shape_dims


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


def collect_type_symbol_ids(param_type: ir.Type, key: Callable[[ir.Var], int] = id) -> set[int]:
    """Return the identity of every Var read by a parameter type's shape metadata.

    Used to detect a scalar parameter that shadows a ``pl.dynamic()`` symbol an
    earlier parameter annotation already resolved, which cannot be rebound.
    """
    symbol_ids: set[int] = set()

    def walk(expr: ir.Expr | None) -> None:
        if expr is None:
            return
        if isinstance(expr, ir.Var):
            symbol_ids.add(key(expr))
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


class ParameterReturnDims(ir.IRMutator):
    """Expand shape computations that can be expressed in the signature."""

    def __init__(self, func: ir.Function) -> None:
        super().__init__()
        assignments = InlineReturnProvenanceCollector()
        assignments.visit_stmt(func.body)
        parameter_ids = {param.unique_id for param in func.params}
        self.params = parameter_ids - assignments.assigned_ids
        self.tensor_origins = inline_immutable_aliases(func, assignments)
        self.signature_symbols = {
            symbol
            for param in func.params
            for symbol in collect_type_symbol_ids(param.type, key=lambda var: var.unique_id)
        }
        self.aliases: dict[int, ir.Expr] = {}
        self.unresolved = False

        # Only unconditional, single-definition scalar bindings are eligible.
        # Substitute each definition once, in dominance order, so a later
        # reference can share the already-expanded expression tree.
        statements = func.body.stmts if isinstance(func.body, ir.SeqStmts) else (func.body,)
        for stmt in statements:
            if not isinstance(stmt, ir.AssignStmt) or not isinstance(stmt.var.type, ir.ScalarType):
                continue
            target = stmt.var.unique_id
            if target in parameter_ids or assignments.assignment_counts[target] != 1:
                continue
            self.unresolved = False
            value = self.visit_expr(stmt.value)
            if not self.unresolved:
                self.aliases[target] = value
        self.unresolved = False

    def visit_var(self, op: ir.Var) -> ir.Expr:
        if op.unique_id in self.aliases:
            return self.aliases[op.unique_id]
        self.unresolved |= op.unique_id not in self.signature_symbols
        return op

    def visit_iter_arg(self, op: ir.IterArg) -> ir.Expr:
        self.unresolved = True
        return op

    def visit_call(self, op: ir.Call) -> ir.Expr:
        if op.op.name != ir.get_op("tensor.dim").name or len(op.args) != 2:
            self.unresolved = True
            return op
        tensor, axis = op.args
        if isinstance(tensor, ir.Var):
            tensor = self.tensor_origins.get(tensor.unique_id, tensor)
        if (
            isinstance(tensor, ir.Var)
            and tensor.unique_id in self.params
            and isinstance(tensor.type, ir.TensorType)
            and isinstance(axis, ir.ConstInt)
            and -len(tensor.type.shape) <= axis.value < len(tensor.type.shape)
        ):
            return tensor.type.shape[axis.value]
        self.unresolved = True
        return op

    def return_type(self, typ: ir.Type) -> ir.Type:
        if isinstance(typ, ir.TupleType):
            return ir.TupleType([self.return_type(element) for element in typ.types])
        if not isinstance(typ, ir.TensorType):
            return typ
        shape = [self.visit_expr(dim) for dim in typ.shape]
        view = typ.tensor_view
        if view is not None:
            view = ir.TensorView(
                [self.visit_expr(dim) for dim in view.stride],
                view.layout,
                [self.visit_expr(dim) for dim in view.valid_shape],
                view.pad,
            )
        if isinstance(typ, ir.DistributedTensorType):
            return ir.DistributedTensorType(shape, typ.dtype, typ.memref, view, typ.window_buffer)
        return ir.TensorType(shape, typ.dtype, typ.memref, view)


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


def returned_descriptor_types(actual: ir.Type, fallback: ir.Type) -> ir.Type | None:
    """Recover returned tensor/Tile descriptors; scalar results retain native deduction."""
    if isinstance(actual, (ir.TileType, ir.TensorType)) and type(actual) is type(fallback):
        return actual
    if isinstance(actual, ir.TupleType) and isinstance(fallback, ir.TupleType):
        if len(actual.types) != len(fallback.types):
            return None
        projected = [returned_descriptor_types(a, f) for a, f in zip(actual.types, fallback.types)]
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
            return returned_descriptor_types(actual, fallback)
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
            return returned_descriptor_types(container.types[value.index], fallback)
    return None
