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
- IR Builder for incremental IR construction
- Helper utilities
- Enhanced type constructors (e.g., TensorType with integer shape support)
"""

# Re-export all core IR types and functions from native module
from pypto.pypto_core.ir import *  # noqa: F401, F403

# Import operation modules
# Import operator overloading with span capture and normalization
# This patches Var and ScalarExpr with Python operators
from . import (
    op,
    operators,  # noqa: F401
)

# Import IR Builder
from .builder import IRBuilder  # noqa: F401

# Import TensorType and TileType with enhanced __init__ that supports integer shapes
# This patches the native TensorType and TileType classes to accept integer shapes
from .type import TensorType, TileType  # noqa: F401


def python_print(node, prefix="pi"):  # type: ignore[misc]
    """
    Print IR node or Type object in Python IR syntax.

    This is a unified wrapper that dispatches to the appropriate C++ function
    based on the type of the input object.

    Args:
        node: IR node (Expr, Stmt, Function, Program) or Type object to print
        prefix: Module prefix (default 'pi' for 'import pypto.ir as pi')

    Returns:
        str: Python-style string representation
    """
    from pypto.pypto_core import ir as _ir_core  # noqa: PLC0415

    # Check if node is a Type object
    if isinstance(node, _ir_core.Type):
        # Use the separate function for Type objects
        return _ir_core.python_print_type(node, prefix)
    else:
        # Use the standard function for IRNode objects
        return _ir_core.python_print(node, prefix)


__all__ = ["op", "IRBuilder", "TensorType", "TileType", "python_print"]
