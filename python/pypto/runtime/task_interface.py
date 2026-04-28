# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Re-exports from ``simpler.task_interface`` and ``simpler.worker``.

All C++ nanobind types (DataType, ChipCallable, ChipStorageTaskArgs, etc.) and
torch-aware helpers (make_tensor_arg, scalar_to_uint64) come from the
``simpler`` package installed via ``pip install simpler``.
"""

from simpler.task_interface import (  # pyright: ignore[reportMissingImports]
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    CoreCallable,
    scalar_to_uint64,
)
from simpler.worker import Worker  # pyright: ignore[reportMissingImports]
from simpler_setup.torch_interop import make_tensor_arg  # pyright: ignore[reportMissingImports]

__all__ = [
    "CallConfig",
    "ChipCallable",
    "ChipStorageTaskArgs",
    "CoreCallable",
    "Worker",
    "make_tensor_arg",
    "scalar_to_uint64",
]
