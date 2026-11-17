# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Single-block A5 AIV execution of owned CANNBotDSL exports on TMR.

TMR executes callbacks serially on each physical core, including queued tasks.
Local memory and sync resources belong to that callback until it returns. This
bridge drains before entry and before return, restores the default device state,
and only submits single-participant tasks with completion-based dependencies.
"""

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from pypto.jit.external.contracts import Core, PreparedKernel, SourceArtifact
from pypto.jit.external.validation import ContractError

from .cannbotdsl_abi import _manifest, build_binding_checker, build_call_adapter
from .cannbotdsl_submit import build_ordered_submitter
from .external_bridge import ExternalTask, RuntimeTarget, validate_target
from .tmr_bridge import _CAPABILITIES, _file

if TYPE_CHECKING:
    from pypto.runtime.kernel_compiler import KernelCompiler


_VERSION = "tmr-cannbotdsl-aiv-1"
_RESOURCES: dict[str, object] = {
    "allocations": [],
    "entry_requires_previous_task_complete": True,
    "exclusive_local_memory": True,
    "exclusive_sync_resources": True,
    "implicit_workspace_bytes": 0,
    "local_memory_bytes": [],
    "reentrant": False,
    "stack": "consumer_must_validate_compiler_requirement",
    "subblock_id": "physical_aiv_lane",
    "topology": "single_core",
}
_STATE = {
    "clobbered": ["DMA", "padding", "quantization", "mask", "local_memory", "sync_resources"],
    "entry": "asc_init_and_default_dma_state_atomic_none",
    "exit_atomic": "none",
    "preserved": ["CTRL"],
}
_COMPLETION = {
    "ddr_barrier": True,
    "local_pipelines_drained": True,
    "pipe_barrier_is_cross_core": False,
    "publication": "consumer_after_all_participants_return",
    "scalar_gm_flushed": True,
    "wait_for_all_participants": False,
}


def _validate(prepared: PreparedKernel, target: RuntimeTarget) -> None:
    validate_target(prepared, target)
    if (
        target.runtime,
        target.version,
        target.compile_target.architecture,
        target.compile_target.compiler,
    ) != ("tmr", "1", "a5", "ccec"):
        raise ContractError("CANNBotDSL AIV bridge requires TMR v1 on a5/ccec")
    execution = prepared.execution
    if (
        execution.cores != (Core.AIV,)
        or execution.aiv_lanes != 1
        or any((execution.full_residency, execution.sync_start, execution.logical_launch_context))
    ):
        raise ContractError("CANNBotDSL AIV bridge requires one independent AIV participant")
    if prepared.contract.required_capabilities - _CAPABILITIES:
        raise ContractError("CANNBotDSL AIV bridge has unsupported required capabilities")
    manifest = _manifest(prepared)
    resources = manifest["resources"]
    expected = {**_RESOURCES, "gm_write_isolation": resources.get("gm_write_isolation")}
    build = manifest["build_requirements"]
    # Type-sensitive comparison rejects True in place of a byte count and
    # unknown fields. Do not silently accept a stronger exporter contract.
    from pypto.jit.external.providers.cannbotdsl import _expect_metadata  # noqa: PLC0415

    for actual, required, label in (
        (resources, expected, "AIV resources (no local allocations or workspace)"),
        (manifest["state"], _STATE, "AIV state"),
        (manifest["completion"], _COMPLETION, "AIV completion"),
        (
            build,
            {
                "cann_headers": "consumer_resolved_sdk",
                "device_function_mode": True,
                "dsl_headers": build["dsl_headers"],
                "dsl_headers_sha256": build.get("dsl_headers_sha256"),
                "language": "AscendC",
                "separate_translation_unit": True,
                "source_includes": ["c_api/asc_simd.h"],
                "standard": "c++17",
                "targets": ["aiv"],
            },
            "AIV build requirements",
        ),
        (
            manifest["loading"],
            {"host_runtime": False, "text_only": "requires_binary_validation"},
            "AIV loading",
        ),
    ):
        _expect_metadata(actual, required, label)
    # Also validates the exact 64-byte isolation policy for a singleton.
    build_ordered_submitter((prepared,))


def lower(prepared: PreparedKernel, target: RuntimeTarget) -> ExternalTask:
    """Lower the supported descriptor ABI to a synchronous TMR AIV entry."""
    _validate(prepared, target)
    adapter = build_call_adapter(prepared)
    checker = build_binding_checker(prepared)
    lines = [
        '#include "c_api/asc_simd.h"',
        f'#include "{adapter.entry}"',
        'extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {',
        # TMR's scalar executor does not use local memory or sync events while
        # the callback runs. Drain inherited work before acquiring those resources.
        "    pipe_barrier(PIPE_ALL);",
        "    asc_dcci_entire_out();",
        "    dsb(DSB_DDR);",
        "    set_atomic_none();",
        "    uint64_t saved_ctrl = asc_get_ctrl();",
        f"    {adapter.symbol}(args);",
        # Return is the executor's permission to publish FIN and run the next
        # callback. No async operations or writes may outlive this point.
        "    pipe_barrier(PIPE_ALL);",
        "    asc_dcci_entire_out();",
        "    dsb(DSB_DDR);",
        # Re-establish the same public ASC defaults used at export entry. The
        # following task must not inherit DMA/padding/mask configuration.
        "    asc_init();",
        "    asc_set_gm2l1_loop_size(1, 1);",
        "    asc_set_gm2ub_loop_size(1, 1);",
        "    asc_set_ub2gm_loop_size(1, 1);",
        "    asc_set_l0c2gm_config(0, 0, false);",
        "    asc_set_l0c2gm_nz2nd(0, 0, 0);",
        "    asc_set_l0c_copy_prequant(0);",
        "    asc_set_l0c2gm_quant_post(0);",
        "    asc_set_l0c2gm_relu_alpha(0);",
        "    float lrelu_alpha = 0.0f;",
        "    asc_set_l0c2gm_lrelu_alpha(lrelu_alpha);",
        "    asc_set_l0c2gm_channel_para(0);",
        "    asc_set_copy_pad_val(static_cast<uint32_t>(0));",
        "    asc_set_ndim_pad_value(static_cast<uint32_t>(0));",
        "    asc_set_gm2l1_pad(static_cast<uint32_t>(0));",
        "    asc_set_l13d_padding(static_cast<uint64_t>(0));",
        "    asc_set_l12l0_padding_val(static_cast<uint64_t>(0));",
        "    set_atomic_none();",
        "    asc_set_ctrl(saved_ctrl);",
        "}",
        "",
    ]
    source = SourceArtifact(
        (*adapter.files, _file("bridge.cpp", "\n".join(lines).encode())),
        "bridge.cpp",
        "kernel_entry",
        adapter.include_dirs,
    )
    return ExternalTask(
        source,
        prepared,
        target,
        _VERSION,
        next(f.content.decode() for f in checker.files if f.path == checker.entry),
        checker.symbol,
        binary_policy="a5-text-only",
    )


def compile_aiv_task(task: ExternalTask, compiler: "KernelCompiler", *, pto_isa_root: str) -> bytes:
    """Compile checked AIV task bytes for CoreCallable; sources live through compilation.

    The returned text owns its bytes. Register it at the function ID used by
    build_aiv_submitter, and retain the callable and tensor storage through run.
    This direct path deliberately has no binary/preparation cache.
    """
    from pypto.runtime.elf_parser import extract_text_section  # noqa: PLC0415

    from .tmr_bridge import materialize  # noqa: PLC0415

    if task != lower(task.prepared, task.target) or compiler.platform != "a5":
        raise ContractError("CANNBotDSL AIV compilation requires unmodified bridge output and platform a5")
    # Resolve C API headers from the same SDK that selected CCEC, never from
    # a provider directory or an unrelated installation found on PATH.
    sdk_root = getattr(compiler.sdk.ccec, "ascend_home_path", None)
    if not isinstance(sdk_root, str) or not sdk_root:
        raise ContractError("CANNBotDSL AIV compilation requires a CCEC SDK root")
    cann_include = Path(sdk_root) / "asc" / "include"
    if not (cann_include / "c_api" / "asc_simd.h").is_file():
        raise ContractError(f"CANNBotDSL AIV compilation requires C API headers at {cann_include}")
    with materialize(task) as source:
        image = compiler.compile_incore(
            source.source,
            core_type="aiv",
            pto_isa_root=pto_isa_root,
            runtime_name="tensormap_and_ringbuffer",
            extra_include_dirs=[*source.include_dirs, str(cann_include), str(cann_include.parent)],
            text_only=True,
        )
    return extract_text_section(image)


def build_aiv_submitter(tasks: tuple[ExternalTask, ...], function_ids: tuple[int, ...]) -> SourceArtifact:
    """Bind checked tasks to real TMR submission, one block and one AIV per call.

    All IDs must refer to the corresponding compile_aiv_task output. Invoke in
    one active TMR scope; external concurrent calls require their own GM ordering.
    Tensor storage must remain alive until all returned task IDs complete.
    """
    if (
        not tasks
        or type(tasks) is not tuple
        or type(function_ids) is not tuple
        or len(tasks) != len(function_ids)
    ):
        raise ContractError("CANNBotDSL AIV submitter requires matching nonempty task/ID tuples")
    if any(type(i) is not int or not 0 <= i < 2**31 for i in function_ids):
        raise ContractError("CANNBotDSL AIV function IDs must be nonnegative int32 values")
    for task in tasks:
        if task != lower(task.prepared, task.target):
            raise ContractError("CANNBotDSL AIV submitter requires unmodified bridge output")
    if len(set(function_ids)) != len(function_ids):
        raise ContractError("CANNBotDSL AIV function IDs must be unique")
    ordered = build_ordered_submitter(tuple(task.prepared for task in tasks))
    # Include IDs in the symbol to allow multiple registrations in one TU.
    symbol = f"{ordered.symbol}_aiv_{'_'.join(str(i) for i in function_ids)}"
    count = len(tasks)
    lines = [
        "#pragma once",
        '#include "orchestration_api.h"',
        f'#include "{ordered.entry}"',
        f"inline std::array<TaskId, {count}> {symbol}(const std::array<CoreTaskArgs*, {count}>& args) {{",
        "    for (auto* arg : args) {",
        '        always_assert(arg != nullptr && "CANNBotDSL AIV: null binding");',
        '        always_assert(arg->launch_spec.block_num() == 1 && "CANNBotDSL AIV: one block required");',
        '        always_assert(!arg->launch_spec.require_sync_start() && "CANNBotDSL AIV: no sync start");',
        "    }",
        f"    constexpr int32_t ids[] = {{{', '.join(str(i) for i in function_ids)}}};",
        f"    return {ordered.symbol}(args, [&](size_t i, CoreTaskArgs& arg) {{",
        "        return rt_submit_aiv_task(ids[i], arg).task_id();",
        "    });",
        "}",
        "",
    ]
    return replace(
        ordered,
        files=(*ordered.files, _file("aiv_submit.hpp", "\n".join(lines).encode())),
        entry="aiv_submit.hpp",
        symbol=symbol,
    )
