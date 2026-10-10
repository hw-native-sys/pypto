# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Consumed surface of ``simpler.task_interface``.

Mirrors the real module's re-exports: the nanobind wire types from
``_task_interface`` plus ``Buffer``/``Tensor`` re-exported through
``simpler.buffer``.
"""

from _task_interface import (
    ArgDirection,
    CallConfig,
    ChipCallable,
    ChipStorageTaskArgs,
    ChipTensor,
    CoreCallable,
    DataType,
    TaskArgs,
    scalar_to_uint64,
)

from .buffer import Buffer, Tensor

__all__ = [
    "ArgDirection",
    "Buffer",
    "CallConfig",
    "ChipCallable",
    "ChipStorageTaskArgs",
    "ChipTensor",
    "CoreCallable",
    "DataType",
    "TaskArgs",
    "Tensor",
    "scalar_to_uint64",
]
