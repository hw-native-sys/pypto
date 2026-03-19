# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Operator specifications and op registry for fuzzing.

Defines ValueRange, OpSpec, and the built-in BLOCK_*_OPS lists.
Adding a new op: append an OpSpec to the appropriate list below.
"""

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np  # Used in lambda functions for op equivalents


@dataclass
class ValueRange:
    """Track value range properties for a variable.

    Attributes:
        can_be_negative: Whether the value can be negative
        can_be_zero: Whether the value can be zero
        can_be_positive: Whether the value can be positive
    """

    can_be_negative: bool = True
    can_be_zero: bool = True
    can_be_positive: bool = True

    def is_always_positive(self) -> bool:
        """Check if value is guaranteed to be positive (> 0)."""
        return self.can_be_positive and not self.can_be_negative and not self.can_be_zero

    def is_always_nonzero(self) -> bool:
        """Check if value is guaranteed to be non-zero."""
        return not self.can_be_zero

    def is_safe_for_sqrt(self) -> bool:
        """Check if value is safe for sqrt (>= 0)."""
        return not self.can_be_negative

    def is_safe_for_log(self) -> bool:
        """Check if value is safe for log (> 0)."""
        return self.is_always_positive()

    def is_safe_for_div(self) -> bool:
        """Check if value is safe as divisor (non-zero)."""
        return self.is_always_nonzero()


@dataclass
class OpSpec:
    """Operator specification for fuzzing.

    Attributes:
        name: Operator name (e.g., "tile.add")
        input_types: List of input types (e.g., ["tile", "tile"])
        output_type: Output type (e.g., "tile")
        constraints: Additional constraints (e.g., {"min_shape": [64, 64]})
        np_equivalent: NumPy equivalent function for golden reference
        shape_transform: Optional callable that computes output shape from input shapes
        param_generator: Optional callable that generates operator parameters
        requires_params: Whether this operator requires parameters (default: False)
        second_can_be_scalar: If True, the second input may be randomly replaced
            with a scalar at generation time. The parser auto-dispatches
            pl.add(tile, scalar) to the scalar variant, so no separate
            tile.adds / tile.subs / … ops are needed.
    """

    name: str
    input_types: list[str]
    output_type: str
    constraints: dict[str, Any]
    np_equivalent: Any | None = None
    shape_transform: Any | None = None
    param_generator: Any | None = None
    requires_params: bool = False
    second_can_be_scalar: bool = False

    # ------------------------------------------------------------------
    # Eligibility checks — called by the fuzzer to decide if this op
    # can be used given the current generation state.  New constraint
    # types only need changes here, not in the fuzzer loop.
    # ------------------------------------------------------------------

    def is_range_eligible(
        self,
        available_tiles: list[str],
        variable_ranges: dict[str, "ValueRange"],
        allow_scalars: bool,
    ) -> bool:
        """Check if value-range constraints can be satisfied.

        Returns False when the available tiles cannot satisfy this op's
        ``positive_only`` or ``avoid_zero`` constraints.
        """
        tile_inputs = sum(1 for t in self.input_types if t == "tile")
        if tile_inputs == 0:
            return True

        if self.constraints.get("positive_only", False):
            is_safe = (
                (lambda r: r.is_safe_for_log()) if "log" in self.name else (lambda r: r.is_safe_for_sqrt())
            )
            safe_count = sum(
                1 for t in available_tiles if t in variable_ranges and is_safe(variable_ranges[t])
            )
            if safe_count < tile_inputs:
                return False

        if self.constraints.get("avoid_zero", False):
            if tile_inputs == 2:
                if self.second_can_be_scalar and allow_scalars:
                    pass  # scalar divisor is guaranteed non-zero
                else:
                    has_nonzero = any(
                        t in variable_ranges and variable_ranges[t].is_safe_for_div() for t in available_tiles
                    )
                    if not has_nonzero:
                        return False
            elif tile_inputs == 1:
                has_nonzero = any(
                    t in variable_ranges and variable_ranges[t].is_safe_for_div() for t in available_tiles
                )
                if not has_nonzero:
                    return False

        return True

    def is_shape_eligible(
        self,
        available_tiles: list[str],
        variable_shapes: dict[str, tuple[int, int]] | None,
    ) -> bool:
        """Check if shape constraints can be satisfied.

        Returns False when the available tiles don't have the required
        shapes (e.g. row vectors, col vectors) for this op.
        """
        if self.constraints.get("row_vec_required", False):
            if variable_shapes is None:
                return False
            row_vecs = [t for t in available_tiles if variable_shapes.get(t, (0, 0))[1] == 1]
            regulars = [t for t in available_tiles if variable_shapes.get(t, (0, 0))[1] != 1]
            if not row_vecs or not regulars:
                return False
            # Require at least one dimension-compatible pair (matching row count)
            row_vec_rows = {variable_shapes[t][0] for t in row_vecs}
            has_compatible = any(variable_shapes[t][0] in row_vec_rows for t in regulars)
            if not has_compatible:
                return False

        if self.constraints.get("col_vec_required", False):
            if variable_shapes is None:
                return False
            col_vecs = [t for t in available_tiles if variable_shapes.get(t, (0, 0))[0] == 1]
            regulars = [t for t in available_tiles if variable_shapes.get(t, (0, 0))[0] != 1]
            if not col_vecs:
                return False
            # Require at least one dimension-compatible pair (matching col count)
            if regulars:
                col_vec_cols = {variable_shapes[t][1] for t in col_vecs}
                has_compatible = any(variable_shapes[t][1] in col_vec_cols for t in regulars)
                if not has_compatible:
                    return False

        return True

    def has_enough_inputs(
        self,
        available_tiles: list[str],
        available_scalars: list[str],
        allow_scalars: bool,
    ) -> bool:
        """Check if enough tile/scalar inputs are available."""
        tile_inputs = sum(1 for t in self.input_types if t == "tile")
        scalar_inputs = sum(1 for t in self.input_types if t == "scalar")

        min_tiles = max(1, tile_inputs - 1) if (self.second_can_be_scalar and allow_scalars) else tile_inputs
        has_tiles = len(available_tiles) >= min_tiles
        has_scalars = (scalar_inputs == 0) or (
            allow_scalars and (len(available_scalars) >= scalar_inputs or scalar_inputs > 0)
        )
        return has_tiles and has_scalars

    # ------------------------------------------------------------------
    # Shape & range computation
    # ------------------------------------------------------------------

    def compute_output_shape(
        self, input_shapes: list[tuple[int, int]], params: dict[str, Any] | None = None
    ) -> tuple[int, int]:
        """Compute output shape from input shapes."""
        if self.shape_transform:
            sig = inspect.signature(self.shape_transform)
            if len(sig.parameters) >= 2 and params is not None:
                return self.shape_transform(input_shapes, params)
            else:
                return self.shape_transform(input_shapes)
        return input_shapes[0] if input_shapes else (128, 128)

    def generate_params(self, input_shapes: list[tuple[int, int]], rng) -> dict[str, Any]:
        """Generate operator parameters based on input shapes."""
        if self.param_generator and self.requires_params:
            return self.param_generator(input_shapes, rng)
        return {}

    def _compute_mul_range(self, r1: ValueRange, r2: ValueRange) -> ValueRange:
        """Compute range for multiplication operations."""
        return ValueRange(
            can_be_zero=r1.can_be_zero or r2.can_be_zero,
            can_be_positive=(r1.can_be_positive and r2.can_be_positive)
            or (r1.can_be_negative and r2.can_be_negative),
            can_be_negative=(r1.can_be_positive and r2.can_be_negative)
            or (r1.can_be_negative and r2.can_be_positive),
        )

    def _compute_unary_range(self, input_range: ValueRange) -> ValueRange:
        """Compute range for unary operations."""
        op_map = {
            "tile.abs": ValueRange(
                False, input_range.can_be_zero, input_range.can_be_positive or input_range.can_be_negative
            ),
            "tile.relu": ValueRange(False, True, input_range.can_be_positive),
            "tile.sqrt": ValueRange(False, input_range.can_be_zero, True),
            "tile.rsqrt": ValueRange(False, input_range.can_be_zero, True),
            "tile.exp": ValueRange(False, False, True),
            "tile.log": ValueRange(True, True, True),
            "tile.neg": ValueRange(
                input_range.can_be_positive, input_range.can_be_zero, input_range.can_be_negative
            ),
            "tile.recip": ValueRange(input_range.can_be_negative, False, input_range.can_be_positive),
        }
        return op_map.get(self.name, ValueRange())

    def _compute_binary_range(self, input_ranges: list[ValueRange]) -> ValueRange:
        """Compute range for binary operations."""
        if self.name == "tile.add":
            r0 = input_ranges[0]
            if len(input_ranges) >= 2:
                r1 = input_ranges[1]
                return ValueRange(
                    r0.can_be_negative or r1.can_be_negative,
                    True,
                    r0.can_be_positive or r1.can_be_positive,
                )
            return ValueRange(r0.can_be_negative, True, r0.can_be_positive)
        if self.name == "tile.sub":
            return ValueRange(True, True, True)
        if self.name == "tile.mul":
            if len(input_ranges) >= 2:
                return self._compute_mul_range(input_ranges[0], input_ranges[1])
            # Scalar second arg is always positive (0.1–10.0): sign is preserved
            r0 = input_ranges[0]
            return ValueRange(r0.can_be_negative, r0.can_be_zero, r0.can_be_positive)
        if self.name in ["tile.div", "tile.row_expand_div"]:
            return ValueRange(True, input_ranges[0].can_be_zero, True)
        if self.name in ["tile.maximum", "tile.minimum"] and len(input_ranges) >= 2:
            return ValueRange(
                input_ranges[0].can_be_negative or input_ranges[1].can_be_negative,
                input_ranges[0].can_be_zero or input_ranges[1].can_be_zero,
                input_ranges[0].can_be_positive or input_ranges[1].can_be_positive,
            )
        return ValueRange()

    def _compute_expand_range(self, input_ranges: list[ValueRange], op_type: str) -> ValueRange:
        """Compute range for row/col expand operations."""
        if len(input_ranges) < 2:
            return ValueRange()
        if "add" in op_type:
            return ValueRange(input_ranges[0].can_be_negative, True, input_ranges[0].can_be_positive)
        if "sub" in op_type:
            return ValueRange(True, True, True)
        if "mul" in op_type:
            return self._compute_mul_range(input_ranges[0], input_ranges[1])
        if "div" in op_type:
            return ValueRange(True, input_ranges[0].can_be_zero, True)
        return ValueRange()

    def compute_output_range(self, input_ranges: list[ValueRange]) -> ValueRange:
        """Compute output value range from input ranges."""
        if not input_ranges:
            return ValueRange()

        # Unary operations
        if self.name in [
            "tile.abs",
            "tile.relu",
            "tile.sqrt",
            "tile.rsqrt",
            "tile.exp",
            "tile.log",
            "tile.neg",
            "tile.recip",
        ]:
            return self._compute_unary_range(input_ranges[0])

        # Binary operations
        if self.name in [
            "tile.add",
            "tile.sub",
            "tile.mul",
            "tile.div",
            "tile.maximum",
            "tile.minimum",
        ]:
            return self._compute_binary_range(input_ranges)

        # Row/col expand operations
        if self.name.startswith("tile.row_expand_") or self.name.startswith("tile.col_expand_"):
            return self._compute_expand_range(input_ranges, self.name)

        # Reduction operations
        if self.name in ["tile.row_sum", "tile.col_sum"]:
            return ValueRange(input_ranges[0].can_be_negative, True, input_ranges[0].can_be_positive)
        if self.name in ["tile.row_max", "tile.row_min", "tile.col_max", "tile.col_min"]:
            return ValueRange(
                input_ranges[0].can_be_negative, input_ranges[0].can_be_zero, input_ranges[0].can_be_positive
            )

        # Matrix operations
        if self.name == "tile.matmul":
            return ValueRange(True, True, True)

        return ValueRange()


# ---------------------------------------------------------------------------
# Operator registries
# To add a new op: append an OpSpec to the appropriate list.
# ---------------------------------------------------------------------------

# Block-level binary operators
BLOCK_BINARY_OPS: list[OpSpec] = [
    OpSpec(
        "tile.add",
        ["tile", "tile"],
        "tile",
        {"exact_shape": True},
        lambda a, b: a + b,
        second_can_be_scalar=True,
    ),
    OpSpec(
        "tile.sub",
        ["tile", "tile"],
        "tile",
        {"exact_shape": True},
        lambda a, b: a - b,
        second_can_be_scalar=True,
    ),
    OpSpec(
        "tile.mul",
        ["tile", "tile"],
        "tile",
        {"exact_shape": True},
        lambda a, b: a * b,
        second_can_be_scalar=True,
    ),
    OpSpec(
        "tile.div",
        ["tile", "tile"],
        "tile",
        {"exact_shape": True, "avoid_zero": True},
        lambda a, b: a / b,
        second_can_be_scalar=True,
    ),
    OpSpec("tile.maximum", ["tile", "tile"], "tile", {"exact_shape": True}, lambda a, b: np.maximum(a, b)),
    OpSpec("tile.minimum", ["tile", "tile"], "tile", {"exact_shape": True}, lambda a, b: np.minimum(a, b)),
]

# Block-level unary operators
BLOCK_UNARY_OPS: list[OpSpec] = [
    OpSpec("tile.sqrt", ["tile"], "tile", {"positive_only": True}, lambda a: np.sqrt(a)),
    OpSpec(
        "tile.rsqrt",
        ["tile"],
        "tile",
        {"positive_only": True, "avoid_zero": True},
        lambda a: 1.0 / np.sqrt(a),
    ),
    OpSpec("tile.exp", ["tile"], "tile", {}, lambda a: np.exp(np.clip(a, -10, 10))),
    OpSpec("tile.neg", ["tile"], "tile", {}, lambda a: -a),
    OpSpec("tile.recip", ["tile"], "tile", {"avoid_zero": True}, lambda a: 1.0 / a),
    OpSpec("tile.log", ["tile"], "tile", {"positive_only": True}, lambda a: np.log(a)),
    OpSpec("tile.abs", ["tile"], "tile", {}, lambda a: np.abs(a)),
    OpSpec("tile.relu", ["tile"], "tile", {}, lambda a: np.maximum(0, a)),
]

# Block-level row expand operators
# Input: one [M, N] tile and one [M, 1] row vector
# The row vector is broadcast to [M, N] before the operation
# NOTE: row_expand_add is excluded because the CPU simulator (SimKernel)
# does not implement TROWEXPANDADD_IMPL.
BLOCK_ROW_EXPAND_OPS: list[OpSpec] = [
    OpSpec(
        "tile.row_expand_sub",
        ["tile", "tile"],
        "tile",
        {"row_vec_required": True},
        lambda a, b: a - b,
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),
    OpSpec(
        "tile.row_expand_mul",
        ["tile", "tile"],
        "tile",
        {"row_vec_required": True},
        lambda a, b: a * b,
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),
    OpSpec(
        "tile.row_expand_div",
        ["tile", "tile"],
        "tile",
        {"row_vec_required": True, "avoid_zero": True},
        lambda a, b: a / b,
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),
]

# Block-level reduction operators
# Reduce along axis=1: [M,N] -> [M,1]
# Produces [M, 1] row vectors that can be used with row_expand ops
# Note: Second input is a temporary tile placeholder, not an actual input
BLOCK_REDUCTION_OPS: list[OpSpec] = [
    OpSpec(
        "tile.row_sum",
        ["tile"],  # Only one actual input, tmp_tile is created during codegen
        "tile",
        {"produces_row_vec": True, "requires_tmp_tile": True},
        lambda a: np.sum(a, axis=1, keepdims=True),
        shape_transform=lambda shapes: (shapes[0][0], 1) if len(shapes) >= 1 else (128, 1),
    ),
    OpSpec(
        "tile.row_max",
        ["tile"],  # Only one actual input, tmp_tile is created during codegen
        "tile",
        {"produces_row_vec": True, "requires_tmp_tile": True},
        lambda a: np.max(a, axis=1, keepdims=True),
        shape_transform=lambda shapes: (shapes[0][0], 1) if len(shapes) >= 1 else (128, 1),
    ),
    OpSpec(
        "tile.row_min",
        ["tile"],  # Only one actual input, tmp_tile is created during codegen
        "tile",
        {"produces_row_vec": True, "requires_tmp_tile": True},
        lambda a: np.min(a, axis=1, keepdims=True),
        shape_transform=lambda shapes: (shapes[0][0], 1) if len(shapes) >= 1 else (128, 1),
    ),
]

# Block-level column reduction operators (column-wise reduction)
# axis=0: column reduction, [M, N] -> [1, N]
# Output [1, N] can be used with col_expand operations
# Note: Uses general reduction ops with axis=0, keepdim=True
BLOCK_COL_REDUCTION_OPS: list[OpSpec] = [
    OpSpec(
        "tile.col_sum",
        ["tile"],
        "tile",
        {"produces_col_vec": True, "requires_params": True},
        lambda a: np.sum(a, axis=0, keepdims=True),
        shape_transform=lambda shapes: (1, shapes[0][1]) if len(shapes) >= 1 else (1, 128),
        param_generator=lambda shapes, rng: {"axis": 0, "keepdim": True},
        requires_params=True,
    ),
    OpSpec(
        "tile.col_max",
        ["tile"],
        "tile",
        {"produces_col_vec": True, "requires_params": True},
        lambda a: np.max(a, axis=0, keepdims=True),
        shape_transform=lambda shapes: (1, shapes[0][1]) if len(shapes) >= 1 else (1, 128),
        param_generator=lambda shapes, rng: {"axis": 0, "keepdim": True},
        requires_params=True,
    ),
    OpSpec(
        "tile.col_min",
        ["tile"],
        "tile",
        {"produces_col_vec": True, "requires_params": True},
        lambda a: np.min(a, axis=0, keepdims=True),
        shape_transform=lambda shapes: (1, shapes[0][1]) if len(shapes) >= 1 else (1, 128),
        param_generator=lambda shapes, rng: {"axis": 0, "keepdim": True},
        requires_params=True,
    ),
]

# Block-level column expand operators (column broadcast)
# Requirement: second operand must be [1, N] shaped (column vector)
# Operation: broadcasts column vector to each column of the tile
BLOCK_COL_EXPAND_OPS: list[OpSpec] = [
    OpSpec(
        "tile.col_expand",
        ["tile", "tile"],
        "tile",
        {"col_vec_required": True},
        lambda a, b: np.broadcast_to(b, a.shape),
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),  # b is [1, N], broadcasts to [M, N], output is [M, N]
    OpSpec(
        "tile.col_expand_mul",
        ["tile", "tile"],
        "tile",
        {"col_vec_required": True},
        lambda a, b: a * b,
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),
    OpSpec(
        "tile.col_expand_div",
        ["tile", "tile"],
        "tile",
        {"col_vec_required": True, "avoid_zero": True},
        lambda a, b: a / b,
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),
    OpSpec(
        "tile.col_expand_sub",
        ["tile", "tile"],
        "tile",
        {"col_vec_required": True},
        lambda a, b: a - b,
        shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
    ),
]

# Block-level matrix operators
# Note: matmul requires special memory handling (Left, Right, Acc)
# The kernel generator will handle the memory management sequence
BLOCK_MATRIX_OPS: list[OpSpec] = [
    OpSpec(
        "tile.matmul",
        ["tile", "tile"],
        "tile",
        {"matmul_shape": True, "requires_memory_management": True},
        lambda a, b: a @ b,
        shape_transform=lambda shapes, params=None: (shapes[0][0], shapes[1][1])
        if len(shapes) >= 2
        else shapes[0],
    ),
]

# Block-level reshape operator
# Reshapes a tile to a different shape with the same number of elements
BLOCK_RESHAPE_OPS: list[OpSpec] = [
    OpSpec(
        "tile.reshape",
        ["tile"],
        "tile",
        {"requires_target_shape": True},
        lambda a, shape: a.reshape(shape),
        shape_transform=lambda shapes, params=None: params.get("target_shape", shapes[0])
        if params
        else shapes[0],
        param_generator=lambda shapes, rng: {"target_shape": _generate_compatible_reshape(shapes[0], rng)},
        requires_params=True,
    ),
]


def _generate_compatible_reshape(original_shape: tuple[int, int], rng) -> tuple[int, int]:
    """Generate a compatible reshape target for a 2D tile.

    For vectors ([M, 1] or [1, N]), transposes them to the opposite orientation.
    For regular tiles [M, N], swaps dimensions.
    This is commonly used in normalization operations to work around layout constraints.

    Args:
        original_shape: Original tile shape (M, N)
        rng: Random number generator

    Returns:
        Compatible target shape
    """
    m, n = original_shape

    # For row vectors [M, 1], reshape to column vectors [1, M]
    if n == 1 and m > 1:
        return (1, m)

    # For column vectors [1, N], reshape to row vectors [N, 1]
    if m == 1 and n > 1:
        return (n, 1)

    # For regular tiles, swap dimensions (transpose-like reshape)
    return (n, m)
