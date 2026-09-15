# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Execution capabilities shared by compiled metadata and artifact manifests.

This module has no compiler, torch, or runtime dependencies. Capabilities state
which executor may consume an artifact; they neither select a JIT calling mode
nor promise that device binaries have been built or loaded.
"""

from dataclasses import dataclass
from enum import Enum


class ArtifactExecutionMode(Enum):
    """Artifact consumers, independent of hardware/simulator execution modes."""

    PROGRAM = "program"
    KERNEL = "kernel"


@dataclass(frozen=True)
class ExecutionCapabilities:
    """Validated, immutable capabilities for program and internal kernel producers."""

    modes: tuple[ArtifactExecutionMode, ...] = (ArtifactExecutionMode.PROGRAM,)

    def __post_init__(self) -> None:
        if not self.modes or any(not isinstance(mode, ArtifactExecutionMode) for mode in self.modes):
            raise ValueError("Execution capabilities require a nonempty collection of ArtifactExecutionMode")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("Execution capabilities must not contain duplicate modes")
        object.__setattr__(self, "modes", tuple(sorted(self.modes, key=lambda mode: mode.value)))

    def record(self) -> list[str]:
        """Return the canonical JSON representation, without runtime state."""
        return [mode.value for mode in self.modes]

    @classmethod
    def from_record(cls, value: object) -> "ExecutionCapabilities":
        """Read an explicit capability list; missing/unknown values are errors."""
        if not isinstance(value, list) or any(not isinstance(mode, str) for mode in value):
            raise ValueError(f"'supported_execution_modes' must be a list of mode names, got {value!r}")
        try:
            return cls(tuple(ArtifactExecutionMode(mode) for mode in value))
        except ValueError as exc:
            raise ValueError(f"Invalid 'supported_execution_modes' {value!r}: {exc}") from exc

    def require(self, mode: ArtifactExecutionMode) -> None:
        """Reject incompatible consumers before compilation/loading/execution."""
        if not isinstance(mode, ArtifactExecutionMode):
            raise TypeError(f"Expected ArtifactExecutionMode, got {mode!r}")
        if mode not in self.modes:
            raise ValueError(f"Artifact supports {self.record()}, but the consumer requires {mode.value!r}")
