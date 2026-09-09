# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal bridge from immutable artifact handles to lazy runtime loading."""

import threading
from pathlib import Path
from typing import Any

from pypto.jit._artifact_manifest import ArtifactState, BuildKind, read_manifest
from pypto.jit.artifact_cache import ArtifactHandle, ArtifactStore

from ._prebuilt import load_prebuilt, prepare_prebuilt, ready_spec


class ArtifactRuntime:
    """Own a stage transition and live callables without owning device execution.

    Generated paths remain alive after promotion; the compiled object's path
    identity is never changed after attachment (workers may already hash it).
    Private build results also remain alive for this object's entire lifetime.
    """

    def __init__(self, store: ArtifactStore, handle: ArtifactHandle, platform: str, run_directory: Path):
        read_manifest(handle.directory, handle.key, handle.spec)
        if store.root != handle.cache_root:
            raise ValueError("Artifact handle and store have different cache roots")
        run_directory = run_directory.resolve()
        if run_directory == store.root or store.root in run_directory.parents:
            raise ValueError(f"Runtime output must be outside the artifact cache: {run_directory}")
        self.store = store
        self.handle = handle
        self.platform = platform
        self.run_directory = run_directory
        self.directory = handle.directory
        self._lock = threading.Lock()
        self._chips: dict[str, tuple[Any, str, dict[str, Any]]] | None = None

    def load(self) -> dict[str, tuple[Any, str, dict[str, Any]]]:
        """Promote once if needed, then return reusable callables; never execute."""
        with self._lock:
            if self._chips is not None:
                return self._chips
            handle = self.handle
            read_manifest(handle.directory, handle.key, handle.spec)
            if handle.spec.state is ArtifactState.GENERATED:
                spec = ready_spec(handle.directory, handle.spec)

                def build(directory: Path) -> None:
                    handle.materialize(directory)
                    from ._artifact_sources import validate_generated_sources  # noqa: PLC0415

                    validate_generated_sources(directory, handle.spec.build_kind)
                    prepare_prebuilt(directory, self.platform, handle.spec.build_kind)

                result = self.store.get_or_build(handle.key, spec, build)
                if result.handle is not None:
                    handle = result.handle
                    self.handle = handle
                    self.directory = handle.directory
                else:
                    assert result.private_directory is not None
                    self.directory = result.private_directory
            if self.directory == handle.directory:
                read_manifest(handle.directory, handle.key, handle.spec)
            chips = load_prebuilt(self.directory, self.platform, handle.spec.build_kind)
            self._chips = chips
            return chips


def bind_artifact(compiled: Any, store: ArtifactStore, handle: ArtifactHandle, run_directory: Path) -> None:
    """Attach a validated artifact before runtime loading, retaining fresh IR.

    This is an internal adapter entry point, not automatic JIT cache lookup.
    The caller owns matching the compiled program's input snapshot to the key.
    All attached metadata must be recoverable and match the persisted platform.
    """
    from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

    read_manifest(handle.directory, handle.key, handle.spec)
    if isinstance(compiled, DistributedCompiledProgram):
        expected = BuildKind.DISTRIBUTED
        restored = DistributedCompiledProgram.from_dir(handle.directory)
    elif isinstance(compiled, CompiledProgram):
        if compiled.orchestration_names:
            raise ValueError("Persist individual single-chip builds instead of a multi-orchestration parent")
        if compiled._chip_callable is not None:
            raise ValueError("Attach an artifact before loading or registering the compiled program")
        expected = BuildKind.SINGLE_CHIP
        restored = CompiledProgram.from_dir(handle.directory)
    else:
        raise TypeError(f"Unsupported artifact-backed program: {type(compiled).__name__}")
    if handle.spec.build_kind is not expected or restored.platform != compiled.platform:
        raise ValueError("Artifact build kind/platform does not match the compiled program")
    if vars(compiled).get("_artifact_runtime") is not None:
        raise ValueError("Compiled program already has an artifact runtime")
    runtime = ArtifactRuntime(store, handle, compiled.platform, run_directory)
    # This is the only rebind. Later promotion retains this immutable source
    # directory and changes only the runtime's binary handle, never object hashes.
    compiled._output_dir = handle.directory
    compiled._artifact_runtime = runtime


def restore_artifact(store: ArtifactStore, handle: ArtifactHandle, run_directory: Path) -> Any:
    """Restore metadata without IR, and attach the explicit artifact loading policy."""
    from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

    read_manifest(handle.directory, handle.key, handle.spec)
    cls = CompiledProgram if handle.spec.build_kind is BuildKind.SINGLE_CHIP else DistributedCompiledProgram
    compiled = cls.from_dir(handle.directory)
    bind_artifact(compiled, store, handle, run_directory)
    return compiled


def runtime_output_directory(compiled: Any) -> Path:
    """Return writable run storage without changing the immutable artifact root."""
    runtime = vars(compiled).get("_artifact_runtime")
    return compiled.output_dir if runtime is None else runtime.run_directory
