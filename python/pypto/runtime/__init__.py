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

Provides the execution half of the pipeline: dispatching a program compiled by
:func:`pypto.ir.compile` onto an Ascend NPU (or simulator), and the device
memory that outlives a single dispatch.

Compilation itself belongs to :func:`pypto.ir.compile`; this module never
compiles on your behalf. :meth:`RunConfig.compile_kwargs` bridges the two, so
one config object can drive both phases::

    import torch
    from pypto import ir
    from pypto.runtime import RunConfig

    config = RunConfig(platform="a2a3sim")
    compiled = ir.compile(MyProgram, **config.compile_kwargs())

    a = torch.full((128, 128), 2.0)
    b = torch.full((128, 128), 3.0)
    c = torch.zeros(128, 128)
    compiled(a, b, c, config=config)

``execute_compiled`` and ``execute_distributed_compiled`` dispatch a build
directory instead of an artifact handle. Both are **deprecated**: rebuild the
handle with :meth:`pypto.ir.CompiledProgram.from_dir` or
:meth:`pypto.ir.DistributedCompiledProgram.from_dir` and call it. They still
work, and forward to the same implementation, but each emits a
``DeprecationWarning`` and will be removed in a future release. Migrating
``execute_compiled`` means moving its explicit ``platform`` / ``device_id`` /
``dfx`` / ``aicpu_thread_num`` onto the :class:`RunConfig` first — see its
docstring for why a plain rename changes where the run lands.

``RunConfig`` aggregates three concerns, and ``compile_options()`` /
``run_options()`` / ``dfx_options()`` return each on its own. It keeps every
field and every caller; the parts are the vocabulary underneath it.

Two of the three are exported because something takes them:
:class:`CompileOptions` unpacks straight into ``ir.compile``
(``ir.compile(program, **options.as_compile_kwargs())``), so a caller that only
compiles needs no ``RunConfig``; :class:`DfxOptions` is the ``dfx=`` parameter
of ``execute_compiled`` and the artifact-directory CLI entry points.
``RunOptions`` is **not** exported: no dispatch entry point accepts one yet —
they all take a ``RunConfig`` — so it is currently the internal shape the
dispatch plumbing reads, reachable as ``pypto.runtime.runner.RunOptions``.
Exporting it is for the change that migrates those signatures.

``docs/en/dev/08-entry-points.md`` maps every compile and execution entry point
to the layer it belongs to.
"""

from .bench import BenchmarkStats, TraceInvocation, TraceSpan, benchmark
from .device_tensor import DeviceTensor, StackedDeviceTensor
from .distributed_runner import (
    DistributedRunHandle,
    DistributedWorker,
    ReadOnlyHostTensor,
    execute_distributed_compiled,
)
from .log_config import _ensure_configured as _ensure_log_configured
from .log_config import configure_log
from .log_config import current_level as log_level
from .pto_isa import ensure_pto_isa_root, pto_isa_include_dir
from .runner import CompileOptions, DfxOptions, RunConfig, RunResult, execute_compiled
from .runtime_base import Worker
from .tensor_spec import ScalarSpec, TensorSpec
from .worker import ChipWorker, RegistrationHandle

# Honour ``PYPTO_RUNTIME_LOG`` before any runtime entry point runs.
_ensure_log_configured()

__all__ = [
    "benchmark",
    "execute_compiled",
    "execute_distributed_compiled",
    "configure_log",
    "log_level",
    "ensure_pto_isa_root",
    "pto_isa_include_dir",
    "BenchmarkStats",
    "TraceInvocation",
    "TraceSpan",
    "ChipWorker",
    "DeviceTensor",
    "StackedDeviceTensor",
    "DistributedWorker",
    "DistributedRunHandle",
    "ReadOnlyHostTensor",
    "RegistrationHandle",
    "CompileOptions",
    "DfxOptions",
    "RunConfig",
    "RunResult",
    "ScalarSpec",
    "TensorSpec",
    "Worker",
]
