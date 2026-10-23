# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor, Tile, Array and tuple indexing, slicing and indexed assignment."""

import ast
import warnings
from collections.abc import Sequence

from pypto.ir import op as ir_op
from pypto.pypto_core import DataType, ir
from pypto.pypto_core import arith as _arith

from .diagnostics import (
    ParserSyntaxError,
    ParserTypeError,
    UnsupportedFeatureError,
)
from .parser_state import ParseCallbacks, ParserServices
from .type_resolver import (
    _const_int_value,  # noqa: PLC2701
)
from .type_semantics import fold_const_slice_extent, get_source_valid_shape, shape_exprs_match


class SubscriptParser:
    """Tensor, Tile, Array and tuple indexing, slicing and indexed assignment."""

    def __init__(
        self,
        *,
        callbacks: ParseCallbacks,
        services: ParserServices,
    ) -> None:
        self.callbacks = callbacks
        self.services = services
        # Cached arithmetic analyzer used to simplify symbolic slice extents at
        # construction time. One instance per parser amortises the sub-analyzer
        # setup across the many subscripts found in a typical function body.
        self.arith_analyzer: _arith.Analyzer | None = None

    def parse_subscript_assignment(self, target: ast.Subscript, value_node: ast.expr) -> None:
        """Desugar ``dst[<slices...>] = src`` to ``dst = pl.assemble(dst, src, offsets)``.

        This sugar is parser-time only and therefore only meaningful before
        ConvertToSSA: the rewrite rebinds ``dst``, which strict-SSA forbids.
        """
        span = self.services.span_tracker.get_span(target)

        if self.services.scope_manager.strict_ssa:
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
        base_expr = self.callbacks.expression(target.value)
        base_type = base_expr.type

        if isinstance(base_type, ir.ArrayType):
            self.parse_array_subscript_assignment(target, base_expr, var_name, value_node, span)
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

        indices = self.normalize_subscript_indices(target, span)
        offsets, extents, drop_dims = self.build_assemble_offsets_and_extents(
            indices, base_type.shape, span, kind_name
        )
        dropped = set(drop_dims)
        kept_dims = [d for d in range(len(extents)) if d not in dropped]
        natural_rank = len(kept_dims)
        # The tile.slice deducer floors a sub-2D result back to 2D by prepending
        # unit axes, so a tile rhs obtained by reading an indexed view is 2D even
        # when the "natural" rank-reduced rank is < 2 — accept that shape here.
        floored_rank = max(2, natural_rank) if isinstance(base_type, ir.TileType) else natural_rank

        source_expr = self.callbacks.expression(value_node)
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
        source_valid_shape = get_source_valid_shape(source_type)
        for src_axis, want in enumerate(expected_extents):
            requested_const = fold_const_slice_extent(want, 0)
            source_const = fold_const_slice_extent(source_type.shape[src_axis], 0)
            if requested_const is None or source_const is None:
                continue
            if requested_const == source_const:
                continue
            if source_valid_shape is not None:
                valid_const = fold_const_slice_extent(source_valid_shape[src_axis], 0)
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
            source_expr = self.lift_subscript_write_source_rank(
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
        var = self.callbacks.assign(var_name, assemble_call, span)
        self.services.scope_manager.define_var(var_name, var, span=span)

    def lift_subscript_write_source_rank(
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

        A narrower ``valid_shape`` survives the lift on both paths. Reshape maps
        the source's valid region through to the result, and a lift that only
        inserts unit axes is a coordinate-only rank change it reproduces exactly.
        The tensor path additionally restates the lifted region through
        ``tensor.reshape``'s optional 3rd argument (issue #1509); ``tile.reshape``
        has no such parameter and does not need one.
        """
        # "Narrowed" means valid_shape is actually smaller than shape; a view
        # with valid_shape == shape (e.g. a tile's canonical implicit view) is
        # semantically equivalent to no view and must NOT be treated as a
        # narrow source — otherwise canonical 1D-tile rank-lift writes would
        # hit the issue #1509 path meant for ISA-padded sources.
        is_narrowed = source_valid_shape is not None and not shape_exprs_match(
            source_type.shape, source_valid_shape
        )

        drop_set = set(drop_dims)
        unit: ir.Expr = ir.ConstInt(1, DataType.INDEX, span)

        def _lift_shape(seq: Sequence[int | ir.Expr]) -> list[int | ir.Expr]:
            # Skip the tile 2D-floor lead unit axes — they were padded into seq
            # only to satisfy the tile ≥2D invariant; the lift reconstructs the
            # target rank from the kept positions and the drop_dims unit fillers.
            it = iter(list(seq)[lead_units:])
            return [unit if d in drop_set else next(it) for d in range(len(extents))]

        is_tensor = kind_name == "tensor"
        reshape_op = ir_op.tensor.reshape if is_tensor else ir_op.tile.reshape
        reshape_args: list[list[int | ir.Expr]] = [_lift_shape(source_type.shape)]
        if is_narrowed and is_tensor:
            # Tensor path with a genuinely narrower valid_shape — restate it via
            # reshape's optional 3rd arg (issue #1509). Reshape derives the same
            # region from the source anyway, so this only pins it explicitly.
            # Asserted non-None by the is_narrowed guard.
            assert source_valid_shape is not None
            reshape_args.append(_lift_shape(source_valid_shape))
        return reshape_op(source_expr, *reshape_args, span=span)

    def parse_array_subscript_assignment(
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
        index_expr = self.callbacks.expression(index_node)
        value_expr = self.callbacks.expression(value_node)
        # Bare int literals come back as ConstInt(INDEX); the update_element
        # deducer requires the value dtype to match the array's element dtype.
        # Retag the literal so users can write `arr[i] = 7` without explicit casts.
        array_type = base_expr.type
        if isinstance(value_expr, ir.ConstInt) and isinstance(array_type, ir.ArrayType):
            value_expr = ir.ConstInt(value_expr.value, array_type.dtype, value_expr.span)
        update_call = ir.create_op_call("array.update_element", [base_expr, index_expr, value_expr], {}, span)
        var = self.callbacks.assign(var_name, update_call, span)
        self.services.scope_manager.define_var(var_name, var, span=span)

    def build_assemble_offsets_and_extents(
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
                offsets.append(self.callbacks.expression(idx))
                extents.append(1)
                drop_dims.append(dim_idx)
                continue
            if idx.step is not None:
                raise UnsupportedFeatureError(
                    f"Slice step is not supported in {kind_name} subscript-write",
                    span=span,
                    hint="Use 'dst[start:stop] = src' without step",
                )
            lower: int | ir.Expr = 0 if idx.lower is None else self.callbacks.expression(idx.lower)
            upper: int | ir.Expr = (
                target_shape[dim_idx] if idx.upper is None else self.callbacks.expression(idx.upper)
            )
            offsets.append(lower)
            extents.append(self.build_slice_shape_expr(upper, lower))
        for dim_idx in range(len(indices), rank):
            offsets.append(0)
            extents.append(target_shape[dim_idx])
        return offsets, extents, drop_dims

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
        span = self.services.span_tracker.get_span(subscript)
        value_expr = self.callbacks.expression(subscript.value)
        value_type = value_expr.type

        if isinstance(value_type, ir.TupleType):
            return self.parse_tuple_subscript(subscript, value_expr, span)
        if isinstance(value_type, ir.TensorType):
            return self.parse_tensor_subscript(subscript, value_expr, value_type, span)
        if isinstance(value_type, ir.TileType):
            return self.parse_tile_subscript(subscript, value_expr, value_type, span)
        if isinstance(value_type, ir.ArrayType):
            return self.parse_array_subscript(subscript, value_expr, span)

        raise ParserTypeError(
            f"Subscript requires Tuple, Tensor, Tile, or Array type, got {type(value_type).__name__}",
            span=span,
            hint="Subscript syntax is supported for Tuple, Tensor, Tile, and Array types",
        )

    def parse_array_subscript(self, subscript: ast.Subscript, value_expr: ir.Expr, span: ir.Span) -> ir.Expr:
        """Parse array subscript: ``arr[i]`` -> array.get_element."""
        index_node = subscript.slice
        if isinstance(index_node, ast.Tuple):
            raise ParserTypeError(
                "Array subscript requires a single index (rank-1 in v1)",
                span=span,
                hint="Use arr[i], not arr[i, j]",
            )
        index_expr = self.callbacks.expression(index_node)
        return ir.create_op_call("array.get_element", [value_expr, index_expr], {}, span)

    def parse_tuple_subscript(self, subscript: ast.Subscript, value_expr: ir.Expr, span: ir.Span) -> ir.Expr:
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

    def normalize_subscript_indices(self, subscript: ast.Subscript, span: ir.Span) -> list[ast.expr]:
        """Normalize subscript.slice into a flat list of index components.

        ``A[x]`` -> [x], ``A[x, y]`` -> [x, y]
        Each component is an ast.Slice, ast.Constant, ast.Name, ast.BinOp, etc.
        """
        slc = subscript.slice
        if isinstance(slc, ast.Tuple):
            return list(slc.elts)
        return [slc]

    def get_arith_analyzer(self) -> "_arith.Analyzer":
        """Return a parser-scoped, lazily-constructed arithmetic analyzer."""
        if self.arith_analyzer is None:
            self.arith_analyzer = _arith.Analyzer()
        return self.arith_analyzer

    def build_slice_shape_expr(
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
        folded_extent = fold_const_slice_extent(upper_expr, lower_expr)
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
        simplified = self.get_arith_analyzer().simplify(ir.sub(lhs, rhs))
        const_value = _const_int_value(simplified)
        if const_value is not None:
            return const_value
        return simplified

    def to_index_expr(self, expr: int | ir.Expr) -> ir.Expr:
        """Convert an integer-like slice bound into an INDEX expression."""
        if isinstance(expr, ir.Expr):
            return expr
        return ir.ConstInt(expr, DataType.INDEX, ir.Span.unknown())

    def build_clamped_slice_shape_expr(
        self,
        upper_expr: int | ir.Expr,
        lower_expr: int | ir.Expr,
        span: ir.Span,
    ) -> int | ir.Expr:
        """Build a non-negative slice extent, clamping dynamic cases at zero."""
        folded_extent = fold_const_slice_extent(upper_expr, lower_expr)
        if folded_extent is not None:
            return max(folded_extent, 0)
        return ir.max_(
            self.to_index_expr(self.build_slice_shape_expr(upper_expr, lower_expr)),
            ir.ConstInt(0, DataType.INDEX, ir.Span.unknown()),
            span,
        )

    def intersect_slice_upper_bound(
        self,
        requested_upper: int | ir.Expr,
        source_upper: int | ir.Expr,
        span: ir.Span,
    ) -> int | ir.Expr:
        """Intersect an explicit slice upper bound with the source logical extent."""
        requested_const = fold_const_slice_extent(requested_upper, 0)
        source_const = fold_const_slice_extent(source_upper, 0)
        if requested_const is not None and source_const is not None:
            return min(requested_const, source_const)
        return ir.min_(self.to_index_expr(requested_upper), self.to_index_expr(source_upper), span)

    def build_tile_alloc_extent(
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
        lower_const = fold_const_slice_extent(lower_expr, 0)
        static_const = fold_const_slice_extent(static_extent, 0)
        upper_const = None if upper_expr is None else fold_const_slice_extent(upper_expr, 0)

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

    def slice_extents_match(
        self,
        lhs: int | ir.Expr,
        rhs: int | ir.Expr,
    ) -> bool:
        """Return True when two slice extents are obviously equivalent."""
        lhs_const = fold_const_slice_extent(lhs, 0)
        rhs_const = fold_const_slice_extent(rhs, 0)
        if lhs_const is not None and rhs_const is not None:
            return lhs_const == rhs_const
        return lhs is rhs

    def build_subscript_slice_args(
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
                offset_exprs.append(self.callbacks.expression(idx))
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
                    full_shape[dim_idx] if idx.upper is None else self.callbacks.expression(idx.upper)
                )
                continue

            lower_expr = self.callbacks.expression(idx.lower)
            offset_exprs.append(lower_expr)

            upper_expr = full_shape[dim_idx] if idx.upper is None else self.callbacks.expression(idx.upper)
            shape_exprs.append(self.build_slice_shape_expr(upper_expr, lower_expr))

        # Dims past the supplied indices are implicit ``:`` (kept, full extent).
        for dim_idx in range(len(indices), len(full_shape)):
            offset_exprs.append(0)
            shape_exprs.append(full_shape[dim_idx])

        return shape_exprs, offset_exprs, drop_dims

    def build_tile_subscript_slice_args(
        self,
        indices: list[ast.expr],
        tile_type: ir.TileType,
        span: ir.Span,
    ) -> tuple[list[int | ir.Expr], list[int | ir.Expr], list[int | ir.Expr] | None, list[int]]:
        """Convert tile subscripts into static shape/offset args plus optional valid_shape.

        Same rank-reducing rules as :meth:`SubscriptParser.build_subscript_slice_args`: a scalar
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
                index_expr = self.callbacks.expression(idx)
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

            lower_expr = 0 if idx.lower is None else self.callbacks.expression(idx.lower)
            if idx.lower is not None and fold_const_slice_extent(lower_expr, 0) is None:
                raise UnsupportedFeatureError(
                    "Dynamic lower bounds are not supported in tile subscript",
                    span=span,
                    hint="Use a constant tile slice lower bound or rewrite the logic with explicit tile ops",
                )
            offset_exprs.append(lower_expr)

            parsed_upper = None if idx.upper is None else self.callbacks.expression(idx.upper)
            valid_upper = (
                logical_shape[dim_idx]
                if parsed_upper is None
                else self.intersect_slice_upper_bound(parsed_upper, logical_shape[dim_idx], span)
            )

            valid_extent = (
                valid_upper
                if idx.lower is None
                else self.build_clamped_slice_shape_expr(valid_upper, lower_expr, span)
            )
            alloc_extent = self.build_tile_alloc_extent(static_shape[dim_idx], lower_expr, parsed_upper, span)

            shape_exprs.append(alloc_extent)
            valid_shape_exprs.append(valid_extent)
            needs_valid_shape = needs_valid_shape or not self.slice_extents_match(alloc_extent, valid_extent)

        # Dims past the supplied indices are implicit ``:`` (kept, full extent).
        for dim_idx in range(len(indices), len(static_shape)):
            offset_exprs.append(0)
            shape_exprs.append(static_shape[dim_idx])
            valid_shape_exprs.append(logical_shape[dim_idx])
            needs_valid_shape = needs_valid_shape or not self.slice_extents_match(
                static_shape[dim_idx], logical_shape[dim_idx]
            )

        return shape_exprs, offset_exprs, valid_shape_exprs if needs_valid_shape else None, drop_dims

    def parse_tensor_subscript(
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
        indices = self.normalize_subscript_indices(subscript, span)
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
            idx_exprs: list[int | ir.Expr] = [self.callbacks.expression(idx) for idx in indices]
            return ir_op.tensor.read(value_expr, idx_exprs, span=span)

        shape_exprs, offset_exprs, drop_dims = self.build_subscript_slice_args(
            indices, list(tensor_type.shape), span, "tensor"
        )
        return ir_op.tensor.slice(value_expr, shape_exprs, offset_exprs, drop_dims=drop_dims, span=span)

    def parse_tile_subscript(
        self,
        subscript: ast.Subscript,
        value_expr: ir.Expr,
        tile_type: ir.TileType,
        span: ir.Span,
    ) -> ir.Expr:
        """Parse a tile subscript with numpy-style rank-reducing / partial / chained indexing.

        Same rules as :meth:`SubscriptParser.parse_tensor_subscript`, plus a tile-only floor:
        tiles are physically 2D, so a result that would naturally be < 2D is
        auto-promoted to 2D (unit axes prepended) with a non-fatal warning.
        """
        indices = self.normalize_subscript_indices(subscript, span)
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
            idx_exprs: list[int | ir.Expr] = [self.callbacks.expression(idx) for idx in indices]
            return ir_op.tile.read(value_expr, idx_exprs, span=span)

        shape_exprs, offset_exprs, valid_shape_exprs, drop_dims = self.build_tile_subscript_slice_args(
            indices, tile_type, span
        )
        natural_rank = rank - len(drop_dims)
        if drop_dims and natural_rank < 2:
            kept = [shape_exprs[i] for i in range(rank) if i not in set(drop_dims)]
            promoted = [1] * (2 - len(kept)) + list(kept)
            promoted_repr = "[" + ", ".join(self.render_dim_for_msg(d) for d in promoted) + "]"
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
    def render_dim_for_msg(dim: int | ir.Expr) -> str:
        """Render a shape dim for a warning/error message (constant value, else ``?``)."""
        if isinstance(dim, int):
            return str(dim)
        # The isinstance keeps a mistyped dim diagnosable instead of crashing the
        # constant lookup.
        value = _const_int_value(dim) if isinstance(dim, ir.Expr) else None  # pyright: ignore[reportUnnecessaryIsInstance]
        return str(value) if value is not None else "?"
