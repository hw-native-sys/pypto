# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""Synchronous direct dispatch for caller-owned NPU buffers."""

import importlib
from time import perf_counter
from typing import Any

import torch
from pypto.ir.compiled_program import CompiledProgram
from pypto.runtime import ChipWorker, DeviceTensor, RunConfig


class DispatchTimer:
    """Accumulate non-overlapping setup intervals only for diagnostic runs."""

    def __init__(self, totals: dict[str, float]):
        self.totals = totals
        self.start = perf_counter()

    def mark(self, name: str) -> None:
        end = perf_counter()
        self.totals[name] = self.totals.get(name, 0.0) + (end - self.start) * 1000
        self.start = perf_counter()


def _no_timing(name: str) -> None:
    pass


class DispatchPlan:
    """Fixed callable/configuration; never retain tensor addresses or registrations.

    Recreate after changing the compiled artifact or desired run configuration.
    TaskArgs, scalar values, Buffer registrations and the caller stream stay per-call.
    """

    def __init__(self, compiled: CompiledProgram):
        self.compiled = compiled
        self.task_interface = importlib.import_module("pypto.runtime.task_interface")
        self.buffer_type = importlib.import_module("simpler.buffer").Buffer
        tags = importlib.import_module("simpler.task_interface").TensorArgType
        directions = {"In": tags.INPUT, "Out": tags.OUTPUT_EXISTING, "InOut": tags.INOUT}
        infos, _, _ = compiled._get_metadata()
        self.tensor_tags = [directions[info.direction.name] for info in infos if info.shape is not None]
        self.callable = compiled.chip_callable
        self.config = compiled._build_call_config(RunConfig(platform=compiled.platform))


def run_kernel(
    chip: ChipWorker,
    compiled: CompiledProgram,
    *values: torch.Tensor | DeviceTensor | int,
    plan: DispatchPlan | None = None,
    timings: dict[str, float] | None = None,
) -> None:
    """Bind caller-owned device allocations to address-free TaskArgs.

    Tensor and scalar pools follow the lowered chip ABI. The raw context pointer
    is a scalar; every data operand carries a live Buffer registration. The
    synchronous run retires uses before registrations are revoked.
    """
    mark = DispatchTimer(timings).mark if timings is not None else _no_timing
    if plan is None:
        plan = DispatchPlan(compiled)
    elif plan.compiled is not compiled:
        raise ValueError("DispatchPlan belongs to a different compiled artifact")
    mark("plan")
    task_interface = plan.task_interface
    args = task_interface.TaskArgs()
    buffers: list[Any] = []
    tensors = [v for v in values if not isinstance(v, int)]
    scalars = [v for v in values if isinstance(v, int)]
    mark("taskargs_init")
    try:
        for val, tag in zip(tensors, plan.tensor_tags, strict=True):
            if isinstance(val, DeviceTensor):
                address, nbytes, shape, dtype = val.data_ptr, val.nbytes, val.shape, val.dtype
            else:
                if val.device.type != "npu" or not val.is_contiguous():
                    raise ValueError("Borrowed execution requires contiguous NPU tensors")
                address, nbytes = val.data_ptr(), val.numel() * val.element_size()
                shape, dtype = tuple(val.shape), val.dtype
            mark("tensor_inspect")
            buffer = plan.buffer_type.wrap(
                address=address, nbytes=nbytes, location=chip._impl.device_location
            )
            buffers.append(buffer)
            mark("buffer_register")
            args.add_tensor(
                buffer.tensor(shape, task_interface.torch_dtype_to_datatype(dtype)),
                tag,
            )
            mark("tensor_bind")
        for val in scalars:
            args.add_scalar(int(val))
        mark("scalar_bind")
        # Drain torch_npu's host emission queue and device work before entering
        # simpler on this Python thread. This example deliberately runs serially.
        getattr(torch, "npu").synchronize()
        mark("torch_sync")
        stream = int(getattr(torch, "npu").current_stream().npu_stream)
        mark("stream_lookup")
        chip._run_chip(
            plan.callable,
            args,
            plan.config,
            caller_stream=stream,
        )
        mark("native_dispatch")
    except BaseException:
        # A failed dispatch can retain native uses. Teardown must retire those
        # uses before registrations may be revoked; a failed close leaves them live.
        chip.close()
        for buffer in buffers:
            buffer.close()
        raise
    else:
        for buffer in buffers:
            buffer.close()
        mark("buffer_release")
