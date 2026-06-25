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

from typing import TYPE_CHECKING, Any

from .bench import BenchmarkStats, benchmark
from .device_tensor import DeviceTensor
from .distributed_runner import DistributedWorker, execute_distributed_compiled
from .log_config import _ensure_configured as _ensure_log_configured
from .log_config import configure_log
from .log_config import current_level as log_level
from .runner import RunConfig, RunResult, compile_program, execute_compiled, run
from .runtime_base import Worker
from .tensor_spec import ScalarSpec, TensorSpec
from .worker import ChipWorker, RegistrationHandle

if TYPE_CHECKING:
    # ``RunTiming`` is a simpler nanobind type. Re-exported lazily (see
    # ``__getattr__`` below) so ``import pypto.runtime`` does not pull in the
    # optional ``simpler`` package; under TYPE_CHECKING we expose it for IDEs.
    from .task_interface import RunTiming  # pyright: ignore[reportAttributeAccessIssue]

# Honour ``PYPTO_RUNTIME_LOG`` before any runtime entry point runs.
_ensure_log_configured()


def __getattr__(name: str) -> Any:
    """Lazily re-export ``RunTiming`` from :mod:`pypto.runtime.task_interface`.

    ``RunTiming`` is the type users read off ``last_run_timing`` /
    ``execute_compiled`` (issue #1679), so exposing it from the package root
    keeps it discoverable (``from pypto.runtime import RunTiming``). It is
    resolved on first access rather than at import time because
    ``task_interface`` eagerly imports the optional ``simpler`` package, which
    must not become a hard import-time dependency of ``pypto.runtime``.
    """
    if name == "RunTiming":
        # pyright: ignore — RunTiming is re-exported from the stub-less
        # ``simpler.worker``, so pyright cannot resolve the symbol.
        from .task_interface import (  # noqa: PLC0415
            RunTiming,  # pyright: ignore[reportAttributeAccessIssue]
        )

        return RunTiming
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "run",
    "benchmark",
    "compile_program",
    "execute_compiled",
    "execute_distributed_compiled",
    "configure_log",
    "log_level",
    "BenchmarkStats",
    "ChipWorker",
    "DeviceTensor",
    "DistributedWorker",
    "RegistrationHandle",
    "RunConfig",
    "RunResult",
    "RunTiming",
    "ScalarSpec",
    "TensorSpec",
    "Worker",
]
