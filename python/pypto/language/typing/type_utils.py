# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The syntax-level type comparison the parser and the tracer share.

Both frontends must accept and reject the same kernels, so the comparison
that decides a rebind or an annotation match cannot live in one of them.
The tile-view defaults are the parser's ``GetImplicitTileLayout`` mirror;
the type resolver materializes them from here.
"""

from collections.abc import Sequence

from pypto.pypto_core import arith as _arith
from pypto.pypto_core import ir as _ir


def _implicit_tile_layout_from_shape(shape: Sequence[_ir.Expr | int]) -> _ir.TileLayout:
    """Infer the implicit block layout represented by omitted TileView syntax.

    Mirrors C++ ``InferImplicitTileLayoutFromShape`` (tile_view_semantics.h):
    a 2-D tile whose column extent is 1 (and rows > 1) implicitly carries
    ``col_major``; everything else is ``row_major``.
    """
    if len(shape) != 2:
        return _ir.TileLayout.row_major

    rows_value = _const_int_value(shape[0])
    cols_value = _const_int_value(shape[1])
    if rows_value is None or cols_value is None:
        return _ir.TileLayout.row_major
    return _ir.TileLayout.col_major if cols_value == 1 and rows_value > 1 else _ir.TileLayout.row_major


def implicit_tile_view_defaults(
    shape: Sequence[_ir.Expr | int],
    memory_space: _ir.MemorySpace | None = None,
) -> tuple[_ir.TileLayout, _ir.TileLayout, int]:
    """Return the ``(blayout, slayout, fractal)`` implicit defaults for a tile
    of ``shape`` living in ``memory_space``.

    The single Python mirror of C++ ``GetImplicitTileLayout``
    (tile_view_semantics.h) — the space→layout table both frontends share.
    The parser materializes these when TileView syntax omits a field; the
    tracer's annotation funnel (``annotation_to_type``) materializes them
    when a field sits at the bare ``TileView()`` default, which is the
    same convention the printer's elision uses (fields equal to the bare
    default are indistinguishable from omitted).
    """
    default_blayout = _implicit_tile_layout_from_shape(shape)
    default_slayout = _ir.TileLayout.none_box
    default_fractal = _ir.TileView().fractal
    if memory_space in (_ir.MemorySpace.Mat, _ir.MemorySpace.Left):
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.row_major
    elif memory_space == _ir.MemorySpace.Right:
        default_slayout = _ir.TileLayout.col_major
    elif memory_space == _ir.MemorySpace.Acc:
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.row_major
        default_fractal = 1024
    elif memory_space == _ir.MemorySpace.LeftScale:
        # ISA TileLeftScale: RowMajor / RowMajor, MX scale fractal size 32.
        default_blayout = _ir.TileLayout.row_major
        default_slayout = _ir.TileLayout.row_major
        default_fractal = 32
    elif memory_space == _ir.MemorySpace.RightScale:
        # ISA TileRightScale: ColMajor / ColMajor, MX scale fractal size 32.
        default_blayout = _ir.TileLayout.col_major
        default_slayout = _ir.TileLayout.col_major
        default_fractal = 32
    return default_blayout, default_slayout, default_fractal


def _const_int_value(value: object) -> int | None:
    """Extract integer value from a compile-time constant, or None."""
    if isinstance(value, int):
        return value
    if isinstance(value, _ir.ConstInt):
        return value.value
    if isinstance(value, _ir.Neg) and isinstance(value.operand, _ir.ConstInt):
        return -value.operand.value
    return None


def _shape_exprs_match(lhs: Sequence[_ir.Expr], rhs: Sequence[_ir.Expr]) -> bool:
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
    tile_view: _ir.TileView | None,
    shape: Sequence[_ir.Expr],
    memory_space: _ir.MemorySpace | None = None,
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
    default_blayout, default_slayout, default_fractal = implicit_tile_view_defaults(shape, memory_space)
    if tile_view.blayout != default_blayout:
        return True
    if tile_view.slayout != default_slayout:
        return True
    if tile_view.fractal != default_fractal:
        return True
    if tile_view.pad != _ir.PadValue.null:
        return True
    if tile_view.compact != _ir.CompactMode.null:
        return True
    return False


def _normalize_type_for_syntax_match(type_: _ir.Type | None) -> _ir.Type | None:
    """Normalize omitted default TileView metadata before syntax-level comparison."""
    if isinstance(type_, _ir.TileType):
        tile_view = type_.tile_view
        if tile_view is not None and not _has_printable_tile_view(tile_view, type_.shape, type_.memory_space):
            tile_view = None
        if tile_view is type_.tile_view:
            return type_
        return _ir.TileType(type_.shape, type_.dtype, type_.memref, tile_view, type_.memory_space)

    return type_


def _shape_has_symbolic_dim(type_: _ir.Type) -> bool:
    """Return True when *type_* is a shaped type with at least one non-ConstInt dim."""
    if not isinstance(type_, (_ir.TensorType, _ir.TileType)):
        return False
    return any(not isinstance(d, _ir.ConstInt) for d in type_.shape)


def _simplify_shape_dims(type_: _ir.Type, analyzer: _arith.Analyzer) -> _ir.Type:
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
    assert isinstance(type_, (_ir.TensorType, _ir.TileType))
    # The isinstance keeps a mistyped shape element diagnosable instead of
    # crashing the simplify call.
    simplified_shape = [
        analyzer.simplify(d)
        if isinstance(d, _ir.Expr)  # pyright: ignore[reportUnnecessaryIsInstance]
        else d
        for d in type_.shape
    ]
    if isinstance(type_, _ir.DistributedTensorType):
        return _ir.DistributedTensorType(
            simplified_shape, type_.dtype, type_.memref, type_.tensor_view, type_.window_buffer
        )
    if isinstance(type_, _ir.TensorType):
        return _ir.TensorType(simplified_shape, type_.dtype, type_.memref, type_.tensor_view)
    return _ir.TileType(simplified_shape, type_.dtype, type_.memref, type_.tile_view, type_.memory_space)


def _types_match(lhs: _ir.Type | None, rhs: _ir.Type | None) -> bool:
    """Return whether two IR types are structurally equal after TileView normalization.

    Uses C++ ``structural_equal`` (via ``==``) instead of comparing printed
    strings. The seam strips ``memref`` metadata first
    (``_ir._without_type_memref``): ``structural_equal`` compares it, but
    allocation identity is not part of frontend-level value-type matching —
    the parser's Var (not the Call) carries the authoritative memref via
    ``override_type``, so a new producer must not retain the previous
    value's allocation merely because shape and dtype agree.

    Outer shape dimensions are simplified via a shared arithmetic analyzer
    before comparison so expressions like ``(k + C) - k`` match ``C``.  The
    analyzer is only constructed when at least one shape actually contains a
    symbolic dim, keeping the all-static fast path free of arith overhead.
    """
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return lhs is rhs
    lhs = _normalize_type_for_syntax_match(_ir._without_type_memref(lhs))
    rhs = _normalize_type_for_syntax_match(_ir._without_type_memref(rhs))
    assert lhs is not None
    assert rhs is not None
    if type(lhs) is not type(rhs):
        return False
    if _shape_has_symbolic_dim(lhs) or _shape_has_symbolic_dim(rhs):
        analyzer = _arith.Analyzer()
        lhs = _simplify_shape_dims(lhs, analyzer)
        rhs = _simplify_shape_dims(rhs, analyzer)
    return lhs == rhs


def _tile_types_differ_only_by_unset_memory_space(lhs: _ir.Type | None, rhs: _ir.Type | None) -> bool:
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
    if not isinstance(lhs, _ir.TileType) or not isinstance(rhs, _ir.TileType):
        return False
    if (lhs.memory_space is None) == (rhs.memory_space is None):
        return False  # both set or both unset -> plain _types_match already decided
    unified = _ir.TileType(rhs.shape, rhs.dtype, rhs.memref, rhs.tile_view, lhs.memory_space)
    return _types_match(lhs, unified)


def _tensor_types_differ_only_by_valid_shape(lhs: _ir.Type | None, rhs: _ir.Type | None) -> bool:
    """Allow pre-SSA rebinding to refine only an ordinary tensor's valid region.

    A fresh matmul product can narrow a full-box accumulator seed. Preserve the
    RHS type for ConvertToSSA to version, while keeping shape, dtype and storage
    layout checks strict. This is not annotation compatibility or a rule for
    distributed windows.
    """
    if type(lhs) is not _ir.TensorType or type(rhs) is not _ir.TensorType:
        return False

    def without_valid_shape(tensor_type: _ir.TensorType) -> _ir.TensorType:
        view = tensor_type.tensor_view
        if view is not None:
            view = _ir.TensorView(view.stride, view.layout, [], view.pad)
        return _ir.TensorType(tensor_type.shape, tensor_type.dtype, tensor_type.memref, view)

    return _types_match(without_valid_shape(lhs), without_valid_shape(rhs))


# ``_tile_types_differ_only_by_unset_memory_space`` is shared with the parser's
# variable-type reconciliation; the listing keeps pyright's unused-function
# check satisfied.
__all__ = [
    "_tensor_types_differ_only_by_valid_shape",
    "_tile_types_differ_only_by_unset_memory_space",
]
