# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Owned, uncached CANNBotDSL bindings for ordinary JIT orchestration."""

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pypto.jit.external.compatibility import request_from_context
from pypto.jit.external.contracts import CompileTarget, Core, ParameterSpec, TensorSpec
from pypto.jit.external.validation import ContractError
from pypto.jit.specializer import SpecializeContext
from pypto.pypto_core import passes

from .cannbotdsl_runtime import lower
from .external_bridge import RuntimeTarget
from .tmr_bridge import _CAPABILITIES, publish

if TYPE_CHECKING:
    from pypto.jit.external.providers.cannbotdsl import CANNBotDSLProvider


@dataclass(frozen=True)
class CANNBotDSLBinding:
    """Strongly own the kernel until each specialization has exported its bytes.

    The first JIT subset accepts static, packed ND views occupying whole cache
    lines. These are requirements checked on every invocation, not facts inferred
    about caller storage. Complete input capture and cache reuse are separate work.
    """

    provider: "CANNBotDSLProvider"
    kernel: object

    def prepare(
        self, context: SpecializeContext, platform: str, runtime: passes.RuntimeKind, output_dir: str
    ) -> None:
        if platform != "a5" or runtime not in (
            passes.RuntimeKind.TENSORMAP_AND_RINGBUFFER,
            passes.RuntimeKind.HOST_BUILD_GRAPH,
        ):
            raise ContractError("CANNBotDSL JIT requires platform a5 and the TMR or percounter runtime")
        request = request_from_context(context, CompileTarget("a5", "ccec", "consumer-selected"))
        parameters: list[ParameterSpec] = []
        for parameter in request.parameters:
            spec = parameter.type
            if isinstance(spec, TensorSpec):
                if any(type(dim) is not int or dim <= 0 for dim in spec.shape):
                    raise ContractError("CANNBotDSL JIT requires positive static tensor shapes")
                shape = tuple(int(dim) for dim in spec.shape)
                if math.prod(shape) * spec.dtype.get_byte() % 64:
                    raise ContractError(
                        "CANNBotDSL JIT requires tensor views occupying whole 64-byte cache lines"
                    )
                strides = tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))
                spec = replace(spec, strides=strides, alignment=64)
            parameters.append(replace(parameter, type=spec))
        request = replace(request, parameters=tuple(parameters))
        prepared = self.provider.prepare(self.kernel, request)
        if prepared.request != request:
            raise ContractError("CANNBotDSL JIT provider returned a different request")
        host_graph = runtime == passes.RuntimeKind.HOST_BUILD_GRAPH
        target = RuntimeTarget(
            "percounter" if host_graph else "tmr",
            "1",
            request.target,
            frozenset({("cannbotdsl.device_callable", 1)}),
            _CAPABILITIES,
        )
        task = lower(prepared, target)
        # Lazy binary loading outlives this preparation call.
        native = publish(task, output_dir)
        entry = native.source
        execution = prepared.execution
        context.external_core_type = "mixed" if len(execution.cores) == 2 else execution.cores[0].value
        context.external_aic_source = entry if Core.AIC in execution.cores else None
        context.external_aiv_source = entry if Core.AIV in execution.cores else None
        context.external_dual_aiv_dispatch = execution.aiv_lanes == 2
        context.external_include_dirs = native.include_dirs
        context.external_contract = prepared.contract
        context.external_runtime = "percounter" if host_graph else "tensormap_and_ringbuffer"
        symbol = f"{task.preflight_symbol}_jit"
        context.external_preflight_source = (
            task.preflight_source.removeprefix("#pragma once\n")
            + f"""
inline void {symbol}(const CoreTaskArgs& args) {{
    always_assert(args.launch_spec.block_num() == 1 && "CANNBotDSL: one block required");
    always_assert(!args.launch_spec.require_sync_start() && "CANNBotDSL: no sync start");
    {task.preflight_symbol}(args);
}}
"""
        )
        context.external_preflight_symbol = symbol
        context.external_binary_policy = task.binary_policy
