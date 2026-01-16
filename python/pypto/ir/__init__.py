# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO IR module with tensor operations.

This module provides:
- Re-exports of all core IR types from pypto_core.ir
- Organized operation namespaces (e.g., op.tensor.create)
- Helper utilities
"""

from typing import Tuple, Union

from pypto.pypto_core import DataType  # noqa: F401

# Re-export all core IR types and functions
from pypto.pypto_core.ir import *  # noqa: F401, F403

# Import operation modules
from . import op


# Helper functions for control flow
def yield_value(*values: "Expr") -> Union["Expr", Tuple["Expr", ...]]:  # noqa: F405
    """Yield value(s) in control flow constructs.

    Args:
        *values: One or more expressions to yield

    Returns:
        Single expression if one value, tuple of expressions if multiple
    """
    # For now, yield_value is just a passthrough in Python
    # The actual yield semantics are handled by the control flow constructs
    if len(values) == 1:
        return values[0]
    return values


def yield_tensor(tensor: "Expr") -> "Expr":  # noqa: F405
    """Yield a tensor value in control flow constructs.

    Args:
        tensor: Tensor expression to yield

    Returns:
        The tensor expression
    """
    # For now, yield_tensor is just a passthrough in Python
    return tensor


# Expose common DataType constants for convenience
FP16 = DataType.FP16
FP32 = DataType.FP32
INT32 = DataType.INT32
INT64 = DataType.INT64


__all__ = [
    "op",
    "yield_value",
    "yield_tensor",
    "FP16",
    "FP32",
    "INT32",
    "INT64",
]
