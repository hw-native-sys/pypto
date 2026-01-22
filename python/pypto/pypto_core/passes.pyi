# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Type stubs for PyPTO IR Pass transformations."""

from pypto.pypto_core.ir import Function

class Pass:
    """Base class for IR transformation passes.

    A Pass represents a transformation that can be applied to a Function.
    Concrete pass implementations should inherit from this class and
    implement the run() method.
    """

    def run(self, func: Function) -> Function:
        """Execute the pass on a function.

        Args:
            func: Input Function to transform

        Returns:
            Transformed Function after the pass has been applied
        """

class IdentityPass(Pass):
    """A pass that appends '_identity' suffix to function name for testing.

    This is a simple pass implementation used primarily for testing the
    pass infrastructure. It modifies the function name by appending
    '_identity' to demonstrate that the pass was executed.
    """

    def __init__(self) -> None:
        """Create an identity pass."""

class InitMemRefPass(Pass):
    """A pass that initializes memref for variables.

    This pass traverses the function and initializes the MemRef field for all
    Var nodes. It sets memory space to UB by default, or DDR for variables
    used in block.load/block.store operations.
    """

    def __init__(self) -> None:
        """Create an InitMemRef pass."""

__all__ = ["Pass", "IdentityPass", "InitMemRefPass"]
