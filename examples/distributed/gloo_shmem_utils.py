# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

"""CPU preparation and eager KERNEL dispatch for caller-owned SHMEM windows."""

from pathlib import Path
from typing import Any

import torch
from pypto.ir._artifact_metadata import ArtifactMetadata
from pypto.ir.compiled_artifact import CompiledArtifact
from pypto.jit.decorator import JITFunction


def compile_eager_kernel(jit: JITFunction, *meta: torch.Tensor | int, output_dir: str) -> CompiledArtifact:
    """Compile an L2 kernel with explicit SHMEM context scalars, without a device.

    The compiler still verifies external storage, return aliases and absence of
    HOST access. Only the generated CommCtx suffix is added to the caller ABI.
    Native assembly is deliberately done here, before requesting device resources.
    """
    import pypto.language as pl  # noqa: PLC0415
    from pypto._artifact_contract import ArtifactExecutionMode  # noqa: PLC0415
    from pypto.ir.compile import _compile_artifact  # noqa: PLC0415

    specialization, _ = jit._resolve_specialization(meta, {}, allow_signature_mode=True)
    program, _ = jit._compile_to_program_with_rename_map(
        specialization.tensor_meta,
        specialization.scalar_dtypes,
        specialization.constexpr_values,
        specialization.per_func_dyn,
        pl,
    )
    artifact = _compile_artifact(
        program,
        contract=ArtifactExecutionMode.KERNEL,
        _explicit_comm_contexts=True,
        platform="a2a3",
        output_dir=output_dir,
    )
    artifact.load()
    return artifact


def _require_shmem_metadata(artifact: CompiledArtifact) -> ArtifactMetadata:
    metadata = artifact.metadata
    if metadata is None or metadata.return_aliases is None or metadata.comm_context_bindings is None:
        raise ValueError(
            f"SHMEM artifact {artifact.output_dir} requires explicit communication-context metadata. "
            "Recompile the example."
        )
    return metadata


def load_eager_kernel(output_dir: Path) -> CompiledArtifact:
    """Read and check a prepared example artifact before assembling its callable."""
    artifact = CompiledArtifact.from_dir(output_dir)
    _require_shmem_metadata(artifact)
    return artifact


def run_eager_kernel(artifact: CompiledArtifact, *values: Any) -> Any:
    """Enqueue a prebuilt benchmark artifact with the same arguments as its JIT function."""
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415
    from pypto.torch import launch  # noqa: PLC0415
    from pypto.torch.interop import _describe_kernel_call  # noqa: PLC0415

    config = get_process_kernel_state().require_config()
    metadata = _require_shmem_metadata(artifact)
    assert metadata.return_aliases is not None and metadata.comm_context_bindings is not None
    frame = _describe_kernel_call(
        metadata.parameters,
        metadata.return_aliases,
        values,
        platform=config.platform,
        device_index=config.device_id,
        distributed_indices=tuple(index for _, index in metadata.comm_context_bindings),
        append_contexts=True,
    )
    return launch.invoke(artifact, frame, config)
