# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PyPTO runtime module.

Provides utilities for compiling a ``@pl.program`` and running it on an
Ascend NPU (or simulator).

Example::

    import torch
    from pypto.runtime import run, RunConfig

    a = torch.full((128, 128), 2.0)
    b = torch.full((128, 128), 3.0)
    c = torch.zeros(128, 128)
    compiled = run(MyProgram, a, b, c, config=RunConfig(platform="a2a3sim"))
"""

from .device_tensor import DeviceTensor
from .runner import RunConfig, RunResult, compile_program, execute_compiled, run
from .tensor_spec import ScalarSpec, TensorSpec
from .worker import Worker

__all__ = [
    "run",
    "compile_program",
    "execute_compiled",
    "DeviceTensor",
    "RunConfig",
    "RunResult",
    "ScalarSpec",
    "TensorSpec",
    "Worker",
]
