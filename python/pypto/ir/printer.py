# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Utilities for printing IR nodes in Python syntax."""

from pypto.pypto_core import ir as _ir_core


def python_print(node: _ir_core.IRNode | _ir_core.Type, prefix: str = "pl", concise: bool = False) -> str:
    """Print IR node or Type object in Python IR syntax.

    This is a unified wrapper that dispatches to the appropriate C++ function
    based on the type of the input object.

    Args:
        node: IR node (Expr, Stmt, Function, Program) or Type object to print
        prefix: Module prefix (default 'pl' for 'import pypto.language as pl')
        concise: If true, omit intermediate type annotations (default false)

    Returns:
        Python-style string representation
    """
    if isinstance(node, _ir_core.Type):
        return _ir_core.python_print_type(node, prefix)
    else:
        return _ir_core.python_print(node, prefix, concise)
