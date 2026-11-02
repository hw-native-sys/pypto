# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal single-chip build product shared by program and kernel consumers."""

import threading
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pypto._artifact_contract import ExecutionCapabilities

from ._artifact_metadata import ArtifactMetadata

if TYPE_CHECKING:
    from pypto.runtime._artifact_runtime import ArtifactRuntime


class CompiledArtifact:
    """Own metadata and lazy callables, never Worker, stream or call arguments.

    The source directory is rebound once at cache attachment. Stage promotion
    changes ArtifactRuntime's binary directory without changing this identity.
    """

    def __init__(
        self, output_dir: str | PathLike[str], platform: str, metadata: ArtifactMetadata | None = None
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.platform = platform
        self.metadata = metadata
        self._artifact_runtime: ArtifactRuntime | None = None
        self._chip_callable: Any = None
        self._runtime_name: str | None = None
        self._runtime_config: dict[str, Any] | None = None
        self._load_lock = threading.Lock()

    @property
    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities() if self.metadata is None else self.metadata.execution_capabilities

    @classmethod
    def from_dir(cls, output_dir: str | PathLike[str]) -> "CompiledArtifact":
        directory = Path(output_dir).resolve()
        metadata = ArtifactMetadata.read(directory)
        return cls(directory, metadata.platform, metadata)

    def load(self) -> None:
        """Build/assemble once; failed loads publish no successful state."""
        with self._load_lock:
            if self._chip_callable is not None:
                return
            if self._artifact_runtime is None:
                from pypto.runtime.device_runner import _compile_and_assemble  # noqa: PLC0415

                cc, rn, rc = _compile_and_assemble(self.output_dir, self.platform)
            else:
                cc, rn, rc = self._artifact_runtime.load()["."]
            self._runtime_name, self._runtime_config = rn, rc
            self._chip_callable = cc

    @property
    def chip_callable(self) -> Any:
        self.load()
        return self._chip_callable

    @property
    def runtime_name(self) -> str:
        self.load()
        assert self._runtime_name is not None
        return self._runtime_name

    @property
    def runtime_config(self) -> dict[str, Any]:
        self.load()
        assert self._runtime_config is not None
        return self._runtime_config
