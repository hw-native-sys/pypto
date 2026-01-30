# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# pylint: disable=unused-argument
"""Code generation module for converting IR to pto-isa C++"""

from pypto import DataType
from pypto.pypto_core.ir import Function, MemorySpace, PipeType

class TypeConverter:
    """Utility for converting IR types to pto-isa C++ types"""

    def __init__(self) -> None:
        """Create a type converter"""

    def ConvertDataType(self, dtype: DataType) -> str:
        """Convert DataType to C++ type string

        Args:
            dtype: PyPTO DataType

        Returns:
            C++ type string (e.g., 'float', 'half', 'int32_t')
        """

    def ConvertMemorySpace(self, space: MemorySpace) -> str:
        """Convert MemorySpace to C++ memory space annotation

        Args:
            space: Memory space type

        Returns:
            Annotation string (e.g., '__gm__' for DDR, empty string for on-chip)
        """

    def ConvertPipeType(self, pipe: PipeType) -> str:
        """Convert PipeType to pto-isa pipe type string

        Args:
            pipe: Pipeline type

        Returns:
            C++ pipe type string with 'PIPE_' prefix (e.g., 'PIPE_MTE1', 'PIPE_V')
        """

    def ConvertEventId(self, event_id: int) -> str:
        """Convert event ID to pto-isa event ID string

        Args:
            event_id: Event ID (must be in range [0, 7])

        Returns:
            C++ event ID string with 'EVENT_ID' prefix (e.g., 'EVENT_ID0')
        """

    def GenerateShapeType(self, dims: list[int]) -> str:
        """Generate Shape type instantiation

        Args:
            dims: Shape dimensions

        Returns:
            Shape type string with 5D padding (e.g., 'Shape<1, 1, 1, 128, 64>')
        """

    def GenerateStrideType(self, shape: list[int]) -> str:
        """Generate Stride type instantiation for row-major layout

        Args:
            shape: Shape dimensions

        Returns:
            Stride type string with 5D padding
        """

class CceCodegen:
    """CCE code generator for converting PyPTO IR to pto-isa C++ code"""

    def __init__(self) -> None:
        """Create a code generator"""

    def Generate(self, func: Function) -> str:
        """Generate C++ code from a PyPTO IR function

        Args:
            func: The IR Function to generate code for

        Returns:
            Generated C++ code as a string
        """

__all__ = [
    "TypeConverter",
    "CceCodegen",
]
