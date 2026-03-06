# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Shape utility functions for tensor alignment constraints.
"""

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
    if aligned_shapes:
        return rng.choice(aligned_shapes)
    # No preset shapes fit: clamp to (max_size, max_size) with minimum alignment
    clamped = max(1, max_size)
    return (clamped, clamped)
