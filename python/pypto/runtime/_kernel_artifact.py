# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free kernel binary assembly and artifact recovery for the internal JIT path."""

import importlib
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
from pypto._kernel_abi import KernelABI
from pypto.ir.compiled_program import load_kernel_metadata
from pypto.jit._artifact_manifest import BuildKind


def validate_kernel_config(config: ModuleType, abi: KernelABI) -> None:
    """Match the compiler's per-tensor signature and runtime to the logical ABI."""
    from ._prebuilt import _encode_signature  # noqa: PLC0415

    signature = _encode_signature(config.ORCHESTRATION.get("signature", []))
    expected = [p.direction.upper() for p in abi.parameters if p.shape is not None]
    if signature != expected or config.RUNTIME_CONFIG.get("runtime") != abi.runtime:
        raise ValueError("Generated kernel signature/runtime does not match its ABI descriptor")
    if config.ORCHESTRATION.get("function_name") != "aicpu_orchestration_entry":
        raise ValueError("Kernel artifact requires the inspected Simpler orchestration entry")


def require_kernel_native(abi: KernelABI) -> None:
    """Reject a native callable builder from a different or unidentified SDK revision."""
    native = importlib.import_module("_task_interface")
    revision = getattr(native, "__build_commit__", None)
    if revision != abi.simpler_revision:
        raise ValueError(
            f"Kernel ABI requires Simpler {abi.simpler_revision}; native binding is {revision!r}"
        )


def validate_kernel_record(record: dict[str, Any], abi: KernelABI) -> None:
    """Require the correct target, pooled signature and descriptor-bearing binary."""
    expected = [p.direction.upper() for p in abi.parameters if p.shape is not None]
    orch = record["orchestration"]
    if (
        record["platform"] != abi.platform
        or record["runtime_name"] != abi.runtime
        or orch["signature"] != expected
        or orch["function_name"] != "aicpu_orchestration_entry"
        or abi.binary_tag() not in orch["binary"]
    ):
        raise ValueError("Kernel binary does not match its ABI descriptor")


class KernelArtifact:
    """Own a compiled kernel artifact and its callable, never a Worker or launch API."""

    def __init__(self, directory: Path, abi: KernelABI):
        self.output_dir = directory
        self.kernel_abi = abi
        self.platform = abi.platform
        self.execution_capabilities = ExecutionCapabilities((ArtifactExecutionMode.KERNEL,))
        self._artifact_runtime: Any = None
        self._loaded: tuple[Any, str, dict[str, Any]] | None = None
        self._lock = threading.Lock()
        load_kernel_metadata(directory, abi)

    def load(self) -> Any:
        """Compile missing binaries or restore verified bytes, retaining the callable."""
        from ._prebuilt import load_prebuilt, prepare_prebuilt  # noqa: PLC0415

        with self._lock:
            if self._loaded is None:
                if self._artifact_runtime is not None:
                    self._loaded = self._artifact_runtime.load()["."]
                else:
                    load_kernel_metadata(self.output_dir, self.kernel_abi)
                    prepare_prebuilt(
                        self.output_dir, self.platform, BuildKind.SINGLE_CHIP, kernel_abi=self.kernel_abi
                    )
                    self._loaded = load_prebuilt(
                        self.output_dir, self.platform, BuildKind.SINGLE_CHIP, kernel_abi=self.kernel_abi
                    )["."]
            return self._loaded[0]

    @property
    def chip_callable(self) -> Any:
        """Return the lazily assembled callable without registering or executing it."""
        return self.load()
