# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Operator fuzzer for generating random operator combinations.
"""

import inspect
import random
from dataclasses import dataclass
from typing import Any

import numpy as np  # Used in lambda functions for op equivalents

# Data type byte sizes
DTYPE_SIZES = {
    "FP32": 4,
    "FP16": 2,
    "INT8": 1,
    "INT32": 4,
}


def is_shape_aligned(shape: tuple[int, int], dtype: str = "FP32") -> bool:
    """Check if shape satisfies 32-byte alignment constraint

    Args:
        shape: (rows, cols) Shape tuple
        dtype: Data type (default FP32)

    Returns:
        True if shape satisfies alignment requirement

    Rules:
        - trailing axis (cols) must be 1, or
        - (trailing axis * sizeof(dtype)) must be a multiple of 32

    Example (FP32, sizeof=4):
        - (128, 1) ✓  trailing axis=1
        - (128, 8) ✓  8*4=32, aligned
        - (128, 16) ✓ 16*4=64, aligned
        - (128, 32) ✓ 32*4=128, aligned
        - (128, 5) ✗  5*4=20, not aligned
    """
    _rows, cols = shape
    dtype_size = DTYPE_SIZES.get(dtype, 4)

    # trailing axis is 1, always aligned
    if cols == 1:
        return True

    # Check if (trailing axis * sizeof(dtype)) is a multiple of 32
    return (cols * dtype_size) % 32 == 0


def get_aligned_shapes(dtype: str = "FP32", max_size: int = 128) -> list[tuple[int, int]]:
    """Get all common shapes that satisfy alignment constraint

    Args:
        dtype: Data type (default FP32)
        max_size: Maximum dimension size (default 128, avoid memory overflow)

    Returns:
        List of aligned shapes
    """
    dtype_size = DTYPE_SIZES.get(dtype, 4)
    # Calculate minimum aligned column count (except 1)
    min_aligned_cols = 32 // dtype_size  # FP32: 8, FP16: 16, INT8: 32

    aligned_shapes = []

    # Common row counts - limit maximum to max_size
    common_rows = [32, 64, 80, 96, 128]
    common_rows = [r for r in common_rows if r <= max_size]

    # Aligned column counts: 1, min_aligned_cols, 2*min_aligned_cols, ...
    for rows in common_rows:
        # Case where cols = 1
        aligned_shapes.append((rows, 1))

        # Aligned column counts
        max_multiplier = max_size // min_aligned_cols
        for multiplier in range(1, max_multiplier + 1):
            cols = min_aligned_cols * multiplier
            if cols <= max_size:
                aligned_shapes.append((rows, cols))

    return aligned_shapes


def generate_aligned_shape(rng, dtype: str = "FP32", max_size: int = 128) -> tuple[int, int]:
    """Randomly generate an aligned shape

    Args:
        rng: Random number generator
        dtype: Data type
        max_size: Maximum dimension size (default 128, avoid memory overflow)

    Returns:
        Shape tuple satisfying alignment constraint
    """
    aligned_shapes = get_aligned_shapes(dtype, max_size)
    return rng.choice(aligned_shapes) if aligned_shapes else (128, 128)


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
        name: Operator name (e.g., "block.add")
        input_types: List of input types (e.g., ["tile", "tile"])
        output_type: Output type (e.g., "tile")
        constraints: Additional constraints (e.g., {"min_shape": [64, 64]})
        np_equivalent: NumPy equivalent function for golden reference
        shape_transform: Optional callable that computes output shape from input shapes
        param_generator: Optional callable that generates operator parameters
        requires_params: Whether this operator requires parameters (default: False)
    """

    name: str
    input_types: list[str]
    output_type: str
    constraints: dict[str, Any]
    np_equivalent: Any | None = None
    shape_transform: Any | None = None
    param_generator: Any | None = None
    requires_params: bool = False

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
            "block.abs": ValueRange(
                False, input_range.can_be_zero, input_range.can_be_positive or input_range.can_be_negative
            ),
            "block.relu": ValueRange(False, True, input_range.can_be_positive),
            "block.sqrt": ValueRange(False, input_range.can_be_zero, True),
            "block.rsqrt": ValueRange(False, input_range.can_be_zero, True),
            "block.exp": ValueRange(False, False, True),
            "block.log": ValueRange(True, True, True),
            "block.neg": ValueRange(
                input_range.can_be_positive, input_range.can_be_zero, input_range.can_be_negative
            ),
            "block.recip": ValueRange(input_range.can_be_negative, False, input_range.can_be_positive),
        }
        return op_map.get(self.name, ValueRange())

    def _compute_binary_range(self, input_ranges: list[ValueRange]) -> ValueRange:
        """Compute range for binary operations."""
        if self.name in ["block.add", "block.adds"]:
            return ValueRange(input_ranges[0].can_be_negative, True, input_ranges[0].can_be_positive)
        if self.name in ["block.sub", "block.subs"]:
            return ValueRange(True, True, True)
        if self.name in ["block.mul", "block.muls"] and len(input_ranges) >= 2:
            return self._compute_mul_range(input_ranges[0], input_ranges[1])
        if self.name in ["block.div", "block.divs", "block.row_expand_div"]:
            return ValueRange(True, input_ranges[0].can_be_zero, True)
        if self.name in ["block.maximum", "block.minimum"] and len(input_ranges) >= 2:
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
            "block.abs",
            "block.relu",
            "block.sqrt",
            "block.rsqrt",
            "block.exp",
            "block.log",
            "block.neg",
            "block.recip",
        ]:
            return self._compute_unary_range(input_ranges[0])

        # Binary operations
        if self.name in [
            "block.add",
            "block.adds",
            "block.sub",
            "block.subs",
            "block.mul",
            "block.muls",
            "block.div",
            "block.divs",
            "block.maximum",
            "block.minimum",
        ]:
            return self._compute_binary_range(input_ranges)

        # Row/col expand operations
        if self.name.startswith("block.row_expand_") or self.name.startswith("block.col_expand_"):
            return self._compute_expand_range(input_ranges, self.name)

        # Reduction operations
        if self.name in ["block.row_sum", "block.col_sum"]:
            return ValueRange(input_ranges[0].can_be_negative, True, input_ranges[0].can_be_positive)
        if self.name in ["block.row_max", "block.row_min", "block.col_max", "block.col_min"]:
            return ValueRange(
                input_ranges[0].can_be_negative, input_ranges[0].can_be_zero, input_ranges[0].can_be_positive
            )

        # Matrix operations
        if self.name == "block.matmul":
            return ValueRange(True, True, True)

        return ValueRange()


class OpFuzzer:
    """Generates random operator combinations for fuzzing."""

    # Block-level binary operators
    BLOCK_BINARY_OPS = [
        OpSpec("block.add", ["tile", "tile"], "tile", {}, lambda a, b: a + b),
        OpSpec("block.sub", ["tile", "tile"], "tile", {}, lambda a, b: a - b),
        OpSpec("block.mul", ["tile", "tile"], "tile", {}, lambda a, b: a * b),
        OpSpec("block.div", ["tile", "tile"], "tile", {"avoid_zero": True}, lambda a, b: a / b),
        OpSpec("block.maximum", ["tile", "tile"], "tile", {}, lambda a, b: np.maximum(a, b)),
        OpSpec("block.minimum", ["tile", "tile"], "tile", {}, lambda a, b: np.minimum(a, b)),
    ]

    # Block-level scalar operators
    BLOCK_SCALAR_OPS = [
        OpSpec("block.adds", ["tile", "scalar"], "tile", {}, lambda a, s: a + s),
        OpSpec("block.subs", ["tile", "scalar"], "tile", {}, lambda a, s: a - s),
        OpSpec("block.muls", ["tile", "scalar"], "tile", {}, lambda a, s: a * s),
        OpSpec("block.divs", ["tile", "scalar"], "tile", {"avoid_zero": True}, lambda a, s: a / s),
    ]

    # Block-level unary operators
    BLOCK_UNARY_OPS = [
        OpSpec("block.sqrt", ["tile"], "tile", {"positive_only": True}, lambda a: np.sqrt(a)),
        OpSpec("block.rsqrt", ["tile"], "tile", {"positive_only": True}, lambda a: 1.0 / np.sqrt(a)),
        OpSpec("block.exp", ["tile"], "tile", {}, lambda a: np.exp(np.clip(a, -10, 10))),
        OpSpec("block.neg", ["tile"], "tile", {}, lambda a: -a),
        OpSpec("block.recip", ["tile"], "tile", {"avoid_zero": True}, lambda a: 1.0 / a),
        OpSpec("block.log", ["tile"], "tile", {"positive_only": True}, lambda a: np.log(a)),
        OpSpec("block.abs", ["tile"], "tile", {}, lambda a: np.abs(a)),
        OpSpec("block.relu", ["tile"], "tile", {}, lambda a: np.maximum(0, a)),
    ]

    # Block-level row expand operators
    # Input: one [M, N] tile and one [M, 1] row vector
    # The row vector is broadcast to [M, N] before the operation
    BLOCK_ROW_EXPAND_OPS = [
        OpSpec(
            "block.row_expand_add",
            ["tile", "tile"],
            "tile",
            {"row_vec_required": True},
            lambda a, b: a + b,
            shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
        ),  # b is [M,1], broadcasts to [M,N], output is [M,N]
        OpSpec(
            "block.row_expand_sub",
            ["tile", "tile"],
            "tile",
            {"row_vec_required": True},
            lambda a, b: a - b,
            shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
        ),
        OpSpec(
            "block.row_expand_mul",
            ["tile", "tile"],
            "tile",
            {"row_vec_required": True},
            lambda a, b: a * b,
            shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
        ),
        OpSpec(
            "block.row_expand_div",
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
    BLOCK_REDUCTION_OPS = [
        OpSpec(
            "block.row_sum",
            ["tile"],  # Only one actual input, tmp_tile is created during codegen
            "tile",
            {"produces_row_vec": True, "requires_tmp_tile": True},
            lambda a: np.sum(a, axis=1, keepdims=True),
            shape_transform=lambda shapes: (shapes[0][0], 1) if len(shapes) >= 1 else (128, 1),
        ),
        OpSpec(
            "block.row_max",
            ["tile"],  # Only one actual input, tmp_tile is created during codegen
            "tile",
            {"produces_row_vec": True, "requires_tmp_tile": True},
            lambda a: np.max(a, axis=1, keepdims=True),
            shape_transform=lambda shapes: (shapes[0][0], 1) if len(shapes) >= 1 else (128, 1),
        ),
        OpSpec(
            "block.row_min",
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
    BLOCK_COL_REDUCTION_OPS = [
        OpSpec(
            "block.col_sum",
            ["tile"],
            "tile",
            {"produces_col_vec": True, "requires_params": True},
            lambda a: np.sum(a, axis=0, keepdims=True),
            shape_transform=lambda shapes: (1, shapes[0][1]) if len(shapes) >= 1 else (1, 128),
            param_generator=lambda shapes, rng: {"axis": 0, "keepdim": True},
            requires_params=True,
        ),
        OpSpec(
            "block.col_max",
            ["tile"],
            "tile",
            {"produces_col_vec": True, "requires_params": True},
            lambda a: np.max(a, axis=0, keepdims=True),
            shape_transform=lambda shapes: (1, shapes[0][1]) if len(shapes) >= 1 else (1, 128),
            param_generator=lambda shapes, rng: {"axis": 0, "keepdim": True},
            requires_params=True,
        ),
        OpSpec(
            "block.col_min",
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
    BLOCK_COL_EXPAND_OPS = [
        OpSpec(
            "block.col_expand_mul",
            ["tile", "tile"],
            "tile",
            {"col_vec_required": True},
            lambda a, b: a * b,
            shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
        ),  # b is [1, N], broadcasts to [M, N], output is [M, N]
        OpSpec(
            "block.col_expand_div",
            ["tile", "tile"],
            "tile",
            {"col_vec_required": True, "avoid_zero": True},
            lambda a, b: a / b,
            shape_transform=lambda shapes: shapes[0] if len(shapes) >= 1 else (128, 128),
        ),
        OpSpec(
            "block.col_expand_sub",
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
    BLOCK_MATRIX_OPS = [
        OpSpec(
            "block.matmul",
            ["tile", "tile"],
            "tile",
            {"matmul_shape": True, "requires_memory_management": True},
            lambda a, b: a @ b,
            shape_transform=lambda shapes, params=None: (shapes[0][0], shapes[1][1])
            if len(shapes) >= 2
            else shapes[0],
        ),
    ]

    def __init__(
        self,
        seed: int | None = None,
        enable_advanced_ops: bool = False,
        advanced_ops_probability: float = 0.5,
    ):
        """Initialize fuzzer with optional seed for reproducibility.

        Args:
            seed: Random seed for reproducibility
            enable_advanced_ops: Enable advanced operations (row_expand, row_sum, matmul, etc.)
            advanced_ops_probability: Probability of selecting advanced ops (default: 0.5)
        """
        self.rng = random.Random(seed)
        # Basic operators (PipeType::V - VECTOR core)
        self.basic_vector_ops = self.BLOCK_BINARY_OPS + self.BLOCK_SCALAR_OPS + self.BLOCK_UNARY_OPS
        self.vector_ops = self.basic_vector_ops

        # Advanced operators (PipeType::V - VECTOR core)
        self.advanced_vector_ops = []

        # Matrix operators (PipeType::M - CUBE core)
        self.matrix_ops = []

        # Advanced ops probability control
        self.enable_advanced_ops = enable_advanced_ops
        self.advanced_ops_probability = advanced_ops_probability

        if enable_advanced_ops:
            # Add reduction (row and col), row_expand, and col_expand operators
            self.advanced_vector_ops = (
                self.BLOCK_REDUCTION_OPS
                # + self.BLOCK_COL_REDUCTION_OPS
                + self.BLOCK_ROW_EXPAND_OPS
                + self.BLOCK_COL_EXPAND_OPS
            )
            self.vector_ops = self.basic_vector_ops + self.advanced_vector_ops
            self.matrix_ops = self.BLOCK_MATRIX_OPS

        # Default to VECTOR operators
        self.ops = self.vector_ops

        # Track operator usage to avoid overuse
        self.op_usage_count = {}
        self.exp_count = 0
        self.div_count = 0
        self.matmul_count = 0  # Track matmul usage
        # Current kernel's pipe type (None, 'M', 'V')
        self.current_pipe_type = None

    def generate_op_chain(  # noqa: PLR0912, PLR0915
        self,
        num_ops: int = 5,
        input_count: int = 2,
        allow_scalars: bool = True,
        track_shapes: bool = False,
        default_shape: tuple[int, int] = (128, 128),
        prefer_matrix_ops: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Generate a chain of operator calls.

        All input tensors and intermediate results are guaranteed to contribute
        to the final output through smart generation and post-processing.

        Algorithm:
        1. Select operations from the eligible pool
        2. Assign inputs and track variable shapes
        3. Limit expensive operations (exp, div)
        4. Ensure all input tensors are used
        5. Track value ranges to avoid numerical issues
        6. Route to M-pipe (matmul with memory management) or V-pipe (element-wise)

        Args:
            prefer_matrix_ops: If True, use matrix operations (M-pipe); if False, use vector
                              operations (V-pipe). If None (default), randomly choose based on
                              availability. Once a pipe type is chosen, all operations in the
                              chain will use that type.
        """
        self.op_usage_count = {}
        self.exp_count = 0
        self.div_count = 0
        self.matmul_count = 0  # Track matmul usage
        self.current_pipe_type = None

        # Decide pipe type: M-pipe (matrix) or V-pipe (vector)
        # Once chosen, all ops in this chain use the same pipe type
        if prefer_matrix_ops is None:
            # Auto-select: use matrix ops with 1% probability (when available)
            if self.matrix_ops and self.rng.random() < 0.01:
                prefer_matrix_ops = True
            else:
                prefer_matrix_ops = False

        if prefer_matrix_ops and self.matrix_ops:
            self.ops = self.matrix_ops
            self.current_pipe_type = "M"
        else:
            self.ops = self.vector_ops
            self.current_pipe_type = "V"

        # Initialize available variables
        available_tiles = [f"tile_{chr(97 + i)}" for i in range(input_count)]
        # Limit scalars: most of the time 1, at most 2
        available_scalars = ["2.0"] if allow_scalars else []

        # Track which initial inputs have been used
        initial_inputs = set(available_tiles)
        used_inputs = set()

        # Track usage count for each variable
        variable_usage_count = {tile: 0 for tile in available_tiles}

        # Shape tracking (optional)
        variable_shapes = {}
        if track_shapes:
            for tile in available_tiles:
                variable_shapes[tile] = default_shape

        # Value range tracking (always enabled for safety)
        variable_ranges = {}
        # Initialize input ranges: torch.randn can produce negative, zero, or positive values
        for tile in available_tiles:
            variable_ranges[tile] = ValueRange(can_be_negative=True, can_be_zero=True, can_be_positive=True)

        operations = []
        max_retries = 3
        next_tmp_index = 0

        for i in range(num_ops):
            # Calculate urgency for using unused inputs
            unused_count = len(initial_inputs - used_inputs)
            remaining_ops = num_ops - i

            # Dynamic priority
            use_unused_priority = 0.7
            if unused_count > 0:
                if unused_count >= remaining_ops:
                    use_unused_priority = 1.0
                elif remaining_ops > 0:
                    use_unused_priority = min(0.9, 0.7 + 0.3 * (unused_count / remaining_ops))

            # Select eligible operators with safety checks
            eligible_ops = self._get_eligible_ops_safe(
                available_tiles,
                available_scalars,
                allow_scalars,
                variable_shapes if track_shapes else None,
                variable_ranges,  # Pass value ranges for constraint checking
            )

            if not eligible_ops:
                break

            # Prioritize binary ops if we need to use unused inputs
            if unused_count > 0 and use_unused_priority >= 0.9:
                binary_ops = [op for op in eligible_ops if sum(1 for t in op.input_types if t == "tile") >= 2]
                if binary_ops:
                    eligible_ops = binary_ops

            op = None
            for _retry in range(max_retries):
                candidate_op = self.rng.choice(eligible_ops)

                # Check operator usage frequency to avoid overuse
                op_name = candidate_op.name
                usage = self.op_usage_count.get(op_name, 0)

                # If this op already used >40% of the time, only allow with 30% probability
                if usage > num_ops * 0.4:
                    if self.rng.random() < 0.3:
                        op = candidate_op
                        break
                else:
                    op = candidate_op
                    break

            if op is None:
                # Exhausted retries, fall back to the first eligible op
                op = eligible_ops[0]

            # Select inputs
            inputs = []
            scalar_value = None
            input_idx = 0  # Track which input we're selecting
            skip_operation = False  # Flag to skip this operation if inputs are incompatible

            for input_type in op.input_types:
                if input_type == "tile":
                    candidate_tiles = available_tiles
                    needs_abs_wrapper = False  # Flag to insert abs operation

                    # Filter by value range constraints first
                    if op.constraints.get("positive_only", False):
                        # For sqrt/rsqrt/log, need non-negative or positive inputs
                        if "log" in op.name:
                            # log requires strictly positive
                            safe_tiles = [
                                t
                                for t in candidate_tiles
                                if t in variable_ranges and variable_ranges[t].is_safe_for_log()
                            ]
                            if safe_tiles:
                                candidate_tiles = safe_tiles
                            else:
                                # No safe tiles, need to insert abs operation
                                needs_abs_wrapper = True
                        else:  # sqrt, rsqrt
                            # Need non-negative inputs
                            safe_tiles = [
                                t
                                for t in candidate_tiles
                                if t in variable_ranges and variable_ranges[t].is_safe_for_sqrt()
                            ]
                            if safe_tiles:
                                candidate_tiles = safe_tiles
                            else:
                                # No safe tiles, need to insert abs operation
                                needs_abs_wrapper = True

                    if op.constraints.get("avoid_zero", False):
                        # For div/recip, divisor must be non-zero
                        if len(op.input_types) == 2 and input_idx == 1:
                            # Second operand of binary op (divisor)
                            safe_tiles = [
                                t
                                for t in candidate_tiles
                                if t in variable_ranges and variable_ranges[t].is_safe_for_div()
                            ]
                            if safe_tiles:
                                candidate_tiles = safe_tiles
                            else:
                                # No guaranteed non-zero tiles, but we can't easily fix this
                                # Skip this operation
                                skip_operation = True
                                break
                        elif len(op.input_types) == 1:
                            # Unary op like recip
                            safe_tiles = [
                                t
                                for t in candidate_tiles
                                if t in variable_ranges and variable_ranges[t].is_safe_for_div()
                            ]
                            if safe_tiles:
                                candidate_tiles = safe_tiles
                            else:
                                skip_operation = True
                                break

                    if track_shapes:
                        # Special handling for row_vec_required operators
                        if op.constraints.get("row_vec_required", False):
                            if input_idx == 0:
                                # First argument must NOT be [M, 1] (should be [M, N] tile)
                                candidate_tiles = [
                                    t
                                    for t in candidate_tiles
                                    if t not in variable_shapes or variable_shapes[t][1] != 1
                                ]
                            elif input_idx == 1:
                                # Second argument MUST be [M, 1] shaped
                                candidate_tiles = [
                                    t
                                    for t in candidate_tiles
                                    if t in variable_shapes and variable_shapes[t][1] == 1
                                ]

                            if not candidate_tiles:
                                # No suitable tiles available, skip this operator
                                skip_operation = True
                                break
                        # Special handling for col_vec_required operators
                        elif op.constraints.get("col_vec_required", False):
                            if input_idx == 0:
                                # First argument must NOT be [1, N] (should be [M, N] tile)
                                candidate_tiles = [
                                    t
                                    for t in candidate_tiles
                                    if t not in variable_shapes or variable_shapes[t][0] != 1
                                ]
                            elif input_idx == 1:
                                # Second argument MUST be [1, N] shaped
                                candidate_tiles = [
                                    t
                                    for t in candidate_tiles
                                    if t in variable_shapes and variable_shapes[t][0] == 1
                                ]

                            if not candidate_tiles:
                                # No suitable tiles available, skip this operator
                                skip_operation = True
                                break
                        else:
                            # For non-row_expand operators, exclude [M, 1] vectors from ALL inputs
                            # because they have ColMajor layout and cause TADD/TSUB/etc errors
                            # CRITICAL: [M, 1] tiles can ONLY be used with row_expand_* operators
                            candidate_tiles = [
                                t
                                for t in candidate_tiles
                                if t not in variable_shapes or variable_shapes[t][1] != 1
                            ]

                            # Also exclude [M, 1] tiles from reduction operators (row_sum, row_max, row_min)
                            # because they produce [M, 1] output which is ColMajor
                            if op.constraints.get("produces_row_vec", False):
                                # For reduction ops, input must NOT be [M, 1]
                                candidate_tiles = [
                                    t
                                    for t in candidate_tiles
                                    if t not in variable_shapes or variable_shapes[t][1] != 1
                                ]

                            # Also exclude [1, N] tiles from col reduction operators
                            # (col_sum, col_max, col_min) because they produce [1, N] output
                            if op.constraints.get("produces_col_vec", False):
                                # For col reduction ops, input must NOT be [1, N]
                                candidate_tiles = [
                                    t
                                    for t in candidate_tiles
                                    if t not in variable_shapes or variable_shapes[t][0] != 1
                                ]

                            # Apply general shape compatibility check
                            candidate_tiles = [
                                t
                                for t in candidate_tiles
                                if self._is_shape_compatible(op, t, variable_shapes)
                            ]
                            if not candidate_tiles:
                                # No suitable tiles available, skip this operator
                                skip_operation = True
                                break

                    # Smart selection: prioritize unused inputs
                    unused_initial_inputs = {
                        t for t in candidate_tiles if t in initial_inputs and t not in used_inputs
                    }

                    candidate_scores = []
                    for t in candidate_tiles:
                        score = 0

                        if t in unused_initial_inputs:
                            score += 50
                            if use_unused_priority >= 0.9:
                                score += 30

                        usage = variable_usage_count.get(t, 0)
                        score += max(0, 20 - usage * 5)

                        if t.startswith("tmp_"):
                            score += 5

                        candidate_scores.append((t, score))

                    if candidate_scores:
                        max_score = max(score for _, score in candidate_scores)

                        if max_score >= 40:
                            threshold = max(max_score * 0.6, 30)
                            top_candidates = [t for t, score in candidate_scores if score >= threshold]

                            if top_candidates and self.rng.random() < 0.85:
                                candidate_tiles = top_candidates
                        else:
                            min_score_needed = max(max_score * 0.7, 10)
                            preferred = [t for t, score in candidate_scores if score >= min_score_needed]
                            if preferred and self.rng.random() < 0.75:
                                candidate_tiles = preferred

                    selected_input = self.rng.choice(candidate_tiles)

                    # If needs_abs_wrapper, insert abs operation before using this input
                    if needs_abs_wrapper:
                        abs_op = next((op for op in self.BLOCK_UNARY_OPS if op.name == "block.abs"), None)
                        if abs_op:
                            abs_output = f"tmp_{next_tmp_index}"
                            next_tmp_index += 1

                            abs_op_dict = {
                                "op": abs_op,
                                "inputs": [selected_input],
                                "output": abs_output,
                                "scalar_value": None,
                                "params": None,
                            }

                            if track_shapes:
                                abs_op_dict["output_shape"] = variable_shapes.get(
                                    selected_input, default_shape
                                )
                                variable_shapes[abs_output] = abs_op_dict["output_shape"]

                            # Compute output range for abs
                            input_range = variable_ranges.get(selected_input, ValueRange())
                            abs_output_range = abs_op.compute_output_range([input_range])
                            variable_ranges[abs_output] = abs_output_range

                            operations.append(abs_op_dict)
                            available_tiles.append(abs_output)
                            variable_usage_count[abs_output] = 0
                            variable_usage_count[selected_input] = (
                                variable_usage_count.get(selected_input, 0) + 1
                            )

                            # Use the abs output instead of the original input
                            selected_input = abs_output

                    inputs.append(selected_input)

                    current_count = variable_usage_count.get(selected_input, 0)
                    variable_usage_count[selected_input] = current_count + 1

                    if selected_input in initial_inputs:
                        used_inputs.add(selected_input)

                    input_idx += 1

                elif input_type == "scalar":
                    # Limit to at most 2 unique scalars, prefer reusing existing ones
                    if available_scalars:
                        scalar_value = self.rng.choice(available_scalars)
                    elif len(available_scalars) < 2:
                        # Generate a new scalar only if we have less than 2
                        scalar_value = f"{self.rng.uniform(0.1, 10.0):.2f}"
                        available_scalars.append(scalar_value)
                    else:
                        # Already have 2 scalars, reuse one
                        scalar_value = self.rng.choice(available_scalars)
                    inputs.append(scalar_value)
                    input_idx += 1

            # Skip this operation if we couldn't find suitable inputs or if flagged to skip
            if skip_operation or len(inputs) < len(op.input_types):
                continue

            output = f"tmp_{next_tmp_index}"
            next_tmp_index += 1

            # Generate operator parameters if required
            params = None
            if op.requires_params:
                input_shapes = [variable_shapes[inp] for inp in inputs if inp in variable_shapes]
                if input_shapes:
                    params = op.generate_params(input_shapes, self.rng)

            op_dict = {
                "op": op,
                "inputs": inputs,
                "output": output,
                "scalar_value": scalar_value,
                "params": params,
            }

            # Compute output shape if tracking
            if track_shapes:
                input_shapes = [variable_shapes[inp] for inp in inputs if inp in variable_shapes]
                output_shape = op.compute_output_shape(input_shapes, params)
                op_dict["output_shape"] = output_shape
                op_dict["input_shapes"] = input_shapes  # Store input shapes for reduction ops
                variable_shapes[output] = output_shape

            # Compute output value range
            input_ranges = [variable_ranges[inp] for inp in inputs if inp in variable_ranges]
            output_range = op.compute_output_range(input_ranges)
            variable_ranges[output] = output_range

            operations.append(op_dict)
            available_tiles.append(output)
            variable_usage_count[output] = 0

            op_name = op.name
            self.op_usage_count[op_name] = self.op_usage_count.get(op_name, 0) + 1

            if "exp" in op_name:
                self.exp_count += 1
            if "div" in op_name:
                self.div_count += 1
            if "matmul" in op_name:
                self.matmul_count += 1

            # Auto-expand [M, 1] row vectors via row_expand to [M, N]
            # Note: row_expand is V-pipe only, so check current_pipe_type
            if track_shapes and output in variable_shapes and self.current_pipe_type == "V":
                output_shape = variable_shapes[output]
                if output_shape[1] == 1:  # Output is [M, 1]
                    # Try to expand via row_expand to [M, N]
                    next_tmp_index = self._try_expand_row_vec(
                        output,
                        operations,
                        available_tiles,
                        variable_shapes,
                        variable_usage_count,
                        default_shape,
                        next_tmp_index,
                    )
                elif output_shape[0] == 1:  # Output is [1, N]
                    # Try to expand via col_expand to [M, N]
                    next_tmp_index = self._try_expand_col_vec(
                        output,
                        operations,
                        available_tiles,
                        variable_shapes,
                        variable_usage_count,
                        variable_ranges,
                        default_shape,
                        next_tmp_index,
                    )

        # Ensure all initial inputs are used
        unused_inputs = initial_inputs - used_inputs
        if unused_inputs:
            # Select merge operator based on pipe type
            if self.current_pipe_type == "M" and self.matrix_ops:
                merge_op = self.matrix_ops[0]  # matmul
            else:
                merge_op = next((op for op in self.BLOCK_BINARY_OPS if op.name == "block.add"), None)

            for unused_input in unused_inputs:
                if operations:
                    current_final = operations[-1]["output"]
                    output = f"tmp_{len(operations)}"

                    op_dict = {
                        "op": merge_op,
                        "inputs": [unused_input, current_final],
                        "output": output,
                        "scalar_value": None,
                        "params": None,
                    }

                    if track_shapes:
                        input_shapes = [
                            variable_shapes.get(unused_input, default_shape),
                            variable_shapes.get(current_final, default_shape),
                        ]
                        output_shape = merge_op.compute_output_shape(input_shapes)
                        op_dict["output_shape"] = output_shape
                        variable_shapes[output] = output_shape

                    operations.append(op_dict)
                    available_tiles.append(output)
                    used_inputs.add(unused_input)
                    variable_usage_count[output] = 0
                    variable_usage_count[unused_input] = variable_usage_count.get(unused_input, 0) + 1
                    variable_usage_count[current_final] = variable_usage_count.get(current_final, 0) + 1

        # Ensure all intermediate results contribute to the final output
        if operations:
            final_output = operations[-1]["output"]
            unused_intermediates = []

            for var_name, usage_count in variable_usage_count.items():
                if var_name.startswith("tmp_") and usage_count == 0 and var_name != final_output:
                    unused_intermediates.append(var_name)

            if unused_intermediates:
                # Select merge operator based on pipe type
                if self.current_pipe_type == "M" and self.matrix_ops:
                    merge_op = self.matrix_ops[0]  # matmul
                else:
                    merge_op = next((op for op in self.BLOCK_BINARY_OPS if op.name == "block.add"), None)

                for unused_var in unused_intermediates:
                    current_final = operations[-1]["output"]
                    output = f"tmp_{len(operations)}"

                    op_dict = {
                        "op": merge_op,
                        "inputs": [unused_var, current_final],
                        "output": output,
                        "scalar_value": None,
                        "params": None,
                    }

                    if track_shapes:
                        input_shapes = [
                            variable_shapes.get(unused_var, default_shape),
                            variable_shapes.get(current_final, default_shape),
                        ]
                        output_shape = merge_op.compute_output_shape(input_shapes)
                        op_dict["output_shape"] = output_shape
                        variable_shapes[output] = output_shape

                    operations.append(op_dict)
                    available_tiles.append(output)
                    variable_usage_count[output] = 0
                    variable_usage_count[unused_var] = variable_usage_count.get(unused_var, 0) + 1
                    variable_usage_count[current_final] = variable_usage_count.get(current_final, 0) + 1

        return operations

    def _try_expand_row_vec(
        self,
        row_vec_name: str,
        operations: list[dict[str, Any]],
        available_tiles: list[str],
        variable_shapes: dict[str, tuple[int, int]],
        variable_usage_count: dict[str, int],
        default_shape: tuple[int, int],
        next_tmp_index: int,
    ) -> int:
        """Expand a [M, 1] tile to [M, N] via a row_expand operation.

        Steps:
        1. Find an [M, N] tile where N != 1
        2. Choose a row_expand operation
        3. Apply: expanded = row_expand_op(regular_tile, row_vec)
        4. Add the expanded result to available tiles

        This broadcasts a [M, 1] ColMajor tile with a [M, N] RowMajor tile,
        producing a full [M, N] result.

        Returns:
            Updated next_tmp_index
        """
        # Find an [M, N] tile where N != 1
        row_vec_shape = variable_shapes[row_vec_name]
        M = row_vec_shape[0]

        # Find candidate tiles with matching M dimension and N != 1
        candidate_regular_tiles = [
            t
            for t in available_tiles
            if t in variable_shapes and variable_shapes[t][0] == M and variable_shapes[t][1] != 1
        ]

        if not candidate_regular_tiles:
            # No suitable tile found, skip expansion
            return next_tmp_index

        # Pick a regular tile
        regular_tile = self.rng.choice(candidate_regular_tiles)
        regular_shape = variable_shapes[regular_tile]

        # Choose a row_expand operation
        row_expand_ops = [
            op
            for op in self.BLOCK_ROW_EXPAND_OPS
            if "div" not in op.name or self.div_count < 5  # Limit div operations
        ]

        if not row_expand_ops:
            return next_tmp_index

        expand_op = self.rng.choice(row_expand_ops)

        output_name = f"tmp_{next_tmp_index}"
        next_tmp_index += 1
        op_dict = {
            "op": expand_op,
            "inputs": [regular_tile, row_vec_name],
            "output": output_name,
            "scalar_value": None,
            "params": None,
            "output_shape": regular_shape,  # Output matches regular_tile shape
        }

        operations.append(op_dict)
        available_tiles.append(output_name)
        variable_shapes[output_name] = regular_shape
        variable_usage_count[output_name] = 0

        variable_usage_count[regular_tile] = variable_usage_count.get(regular_tile, 0) + 1
        variable_usage_count[row_vec_name] = variable_usage_count.get(row_vec_name, 0) + 1

        op_name = expand_op.name
        self.op_usage_count[op_name] = self.op_usage_count.get(op_name, 0) + 1
        if "div" in op_name:
            self.div_count += 1

        return next_tmp_index

    def _try_expand_col_vec(
        self,
        col_vec_name: str,
        operations: list[dict[str, Any]],
        available_tiles: list[str],
        variable_shapes: dict[str, tuple[int, int]],
        variable_usage_count: dict[str, int],
        variable_ranges: dict[str, "ValueRange"],
        default_shape: tuple[int, int],
        next_tmp_index: int,
    ) -> int:
        """Expand [1, N] column vector via col_expand to [M, N]

        Steps:
        1. Find a [M, N] tile (M != 1) with matching N dimension
        2. Select a col_expand operation
        3. Generate: expanded = col_expand_op(regular_tile, col_vec)
        4. Add expanded to available tiles

        [1, N] vectors are RowMajor, safer than [M, 1] ColMajor vectors.

        Returns:
            Updated next_tmp_index
        """
        # Get [1, N] shape
        col_vec_shape = variable_shapes[col_vec_name]
        N = col_vec_shape[1]

        # Find [M, N] tiles where M != 1 and N matches
        candidate_regular_tiles = [
            t
            for t in available_tiles
            if t in variable_shapes and variable_shapes[t][1] == N and variable_shapes[t][0] != 1
        ]

        if not candidate_regular_tiles:
            # No suitable tile, skip expansion
            return next_tmp_index

        # Select a regular tile
        regular_tile = self.rng.choice(candidate_regular_tiles)
        regular_shape = variable_shapes[regular_tile]

        # Select col_expand operation
        col_expand_ops = [
            op
            for op in self.BLOCK_COL_EXPAND_OPS
            if "div" not in op.name or self.div_count < 5  # Limit div operations
        ]

        if not col_expand_ops:
            return next_tmp_index

        # For div operations, check if col_vec is safe (non-zero)
        if "div" in col_expand_ops[0].name:
            col_vec_range = variable_ranges.get(col_vec_name)
            if col_vec_range and not col_vec_range.is_safe_for_div():
                # Filter out div operations
                col_expand_ops = [op for op in col_expand_ops if "div" not in op.name]
                if not col_expand_ops:
                    return next_tmp_index

        expand_op = self.rng.choice(col_expand_ops)

        output_name = f"tmp_{next_tmp_index}"
        next_tmp_index += 1
        op_dict = {
            "op": expand_op,
            "inputs": [regular_tile, col_vec_name],
            "output": output_name,
            "scalar_value": None,
            "params": None,
            "output_shape": regular_shape,  # Output shape matches regular_tile
        }

        operations.append(op_dict)
        available_tiles.append(output_name)
        variable_shapes[output_name] = regular_shape
        variable_usage_count[output_name] = 0

        # Compute output value range
        input_ranges = [variable_ranges.get(regular_tile), variable_ranges.get(col_vec_name)]
        if all(r is not None for r in input_ranges):
            output_range = expand_op.compute_output_range(input_ranges)
            variable_ranges[output_name] = output_range

        variable_usage_count[regular_tile] = variable_usage_count.get(regular_tile, 0) + 1
        variable_usage_count[col_vec_name] = variable_usage_count.get(col_vec_name, 0) + 1

        op_name = expand_op.name
        self.op_usage_count[op_name] = self.op_usage_count.get(op_name, 0) + 1
        if "div" in op_name:
            self.div_count += 1

        return next_tmp_index

    def _get_eligible_ops_safe(  # noqa: PLR0912
        self,
        available_tiles: list[str],
        available_scalars: list[str],
        allow_scalars: bool,
        variable_shapes: dict[str, tuple[int, int]] | None = None,
        variable_ranges: dict[str, ValueRange] | None = None,
    ) -> list[OpSpec]:
        """Get operators that can be applied with current variables (with safety checks).

        Checks:
        1. Limit exp operator usage (max 3 times)
        2. Limit div operator usage (max 5 times)
        3. Limit matmul operator usage (max 3 times)
        4. Check shape compatibility
        5. Check value range constraints (positive_only, avoid_zero)
        6. Select operator pool based on probability (only for V-pipe)
        """
        # self.ops is already set to the correct pool (matrix or vector)
        # based on pipe type selection in generate_op_chain
        eligible = []

        for op in self.ops:
            # Check: limit expensive operations
            if "exp" in op.name and self.exp_count >= 3:
                continue  # Skip exp: limit reached to avoid overflow

            if "div" in op.name and self.div_count >= 5:
                continue  # Skip div: limit reached to avoid precision loss

            # Limit matmul operations to 2-3 times max
            if "matmul" in op.name and self.matmul_count >= 3:
                continue  # Skip matmul if already used 3 times

            tile_inputs = sum(1 for t in op.input_types if t == "tile")
            scalar_inputs = sum(1 for t in op.input_types if t == "scalar")

            has_tiles = len(available_tiles) >= tile_inputs
            has_scalars = (scalar_inputs == 0) or (
                allow_scalars and (len(available_scalars) >= scalar_inputs or scalar_inputs > 0)
            )

            # Check value range constraints
            if variable_ranges is not None and tile_inputs > 0:
                # Check if we have tiles that satisfy the operator's constraints
                if op.constraints.get("positive_only", False):
                    # Need at least one tile that is safe for sqrt/log
                    safe_tiles = [
                        t
                        for t in available_tiles
                        if t in variable_ranges
                        and (
                            variable_ranges[t].is_safe_for_sqrt()
                            if "sqrt" in op.name or "rsqrt" in op.name
                            else variable_ranges[t].is_safe_for_log()
                        )
                    ]
                    if len(safe_tiles) < tile_inputs:
                        continue  # Not enough safe tiles

                if op.constraints.get("avoid_zero", False):
                    # Need at least one tile that is guaranteed non-zero (for divisor)
                    # For binary ops, only the second operand (divisor) needs to be non-zero
                    if tile_inputs == 2:
                        # For div operations, we need at least one non-zero tile for divisor
                        nonzero_tiles = [
                            t
                            for t in available_tiles
                            if t in variable_ranges and variable_ranges[t].is_safe_for_div()
                        ]
                        if len(nonzero_tiles) < 1:
                            continue  # No safe divisor available
                    elif tile_inputs == 1:
                        # For recip, the single input must be non-zero
                        nonzero_tiles = [
                            t
                            for t in available_tiles
                            if t in variable_ranges and variable_ranges[t].is_safe_for_div()
                        ]
                        if len(nonzero_tiles) < 1:
                            continue  # No safe input for recip

            # Skip operators that require special shapes when shape tracking is disabled
            if variable_shapes is None:
                # Without shape tracking, skip row_expand ops (need [M, 1] vectors)
                if op.constraints.get("row_vec_required", False):
                    continue
                # Without shape tracking, skip col_expand ops (need [1, N] vectors)
                if op.constraints.get("col_vec_required", False):
                    continue
            else:
                # With shape tracking, check if we have compatible shapes
                if op.constraints.get("row_vec_required", False):
                    # Need at least one [M, 1] shaped tile for second argument
                    has_row_vec = any(variable_shapes.get(t, (0, 0))[1] == 1 for t in available_tiles)
                    # Also need at least one non-[M, 1] tile for first argument
                    has_regular_tile = any(variable_shapes.get(t, (0, 0))[1] != 1 for t in available_tiles)
                    if not (has_row_vec and has_regular_tile):
                        continue

                if op.constraints.get("col_vec_required", False):
                    # Need at least one [1, N] shaped tile for second argument
                    has_col_vec = any(variable_shapes.get(t, (0, 0))[0] == 1 for t in available_tiles)
                    if not has_col_vec:
                        continue

            if has_tiles and has_scalars:
                eligible.append(op)

        return eligible

    def _is_shape_compatible(self, op: OpSpec, var: str, variable_shapes: dict[str, tuple[int, int]]) -> bool:
        """Check if a variable's shape is compatible with an operator.

        Args:
            op: The operator specification
            var: Variable name to check
            variable_shapes: Dictionary mapping variable names to their shapes

        Returns:
            True if the variable's shape is compatible with the operator
        """
        if var not in variable_shapes:
            return True

        var_shape = variable_shapes[var]

        # Check row_vec_required constraint: second argument must be [M, 1]
        if op.constraints.get("row_vec_required", False):
            # For row_expand operations, the second operand must have shape [M, 1]
            # We can't determine which operand this is without more context,
            # so we check if this variable could be a valid row vector
            if var_shape[1] != 1:
                return False

        # Check col_vec_required constraint: second argument must be [1, N]
        if op.constraints.get("col_vec_required", False):
            # For col_expand operations, the second operand must have shape [1, N]
            if var_shape[0] != 1:
                return False

        return True
