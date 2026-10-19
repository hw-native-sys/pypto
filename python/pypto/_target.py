# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Target architecture and execution-mode normalization shared by compiler and runtime."""

import warnings
from enum import Enum, auto

from pypto.backend import BackendType
from pypto.pypto_core import backend as _backend_core


class ExecutionMode(Enum):
    """Whether a run reaches real silicon or the simulator.

    One of the two axes a ``platform`` string packs. It decides how kernels are
    assembled (``.so`` for the simulator, ``.o`` plus a text-section extract for
    silicon) and whether the two-pass swimlane capture applies; it never reaches
    codegen, which sees only the architecture.
    """

    ONBOARD = auto()
    SIM = auto()


_ARCHES: tuple[BackendType, ...] = (BackendType.Ascend910, BackendType.Ascend950)


def _arch_name(arch: BackendType) -> str:
    """Return the wire name of an architecture (``"a2a3"`` / ``"a5"``).

    Asks the backend handler rather than mapping it here: the C++ side already
    owns this string — it is what codegen stamps as ``pto.target_arch`` — and a
    second copy in Python is a second thing to keep in step.
    """
    return _backend_core.get_backend_instance(arch).get_handler().get_pto_target_arch()


def _platform_string(arch: BackendType, execution_mode: ExecutionMode) -> str:
    """Join the two axes into the wire spelling the runtime and CLI take."""
    return f"{_arch_name(arch)}{'sim' if execution_mode is ExecutionMode.SIM else ''}"


def _parse_platform(platform: str) -> tuple[BackendType, ExecutionMode]:
    """Split a wire platform string back into its two axes.

    The single place that sniffs the string. Everything else asks the axes.
    """
    for arch in _ARCHES:
        name = _arch_name(arch)
        if platform == name:
            return arch, ExecutionMode.ONBOARD
        if platform == f"{name}sim":
            return arch, ExecutionMode.SIM
    expected = ", ".join(f"{_arch_name(a)!r}, {_arch_name(a) + 'sim'!r}" for a in _ARCHES)
    raise ValueError(f"Invalid platform {platform!r}. Expected {expected}.")


def _resolve_target(
    backend_type: BackendType,
    execution_mode: ExecutionMode | None = None,
    platform: str | None = None,
) -> tuple[BackendType, ExecutionMode]:
    """Normalize a target, retaining legacy platform precedence.

    An explicit new execution mode must agree with a legacy platform. An
    omitted mode inherits that platform, or defaults to the simulator.
    """
    if execution_mode is not None and type(execution_mode) is not ExecutionMode:
        raise TypeError(f"execution_mode must be an ExecutionMode, got {execution_mode!r}")
    if platform is not None:
        arch, mode = _parse_platform(platform)
        if execution_mode is not None and execution_mode is not mode:
            raise ValueError(f"execution_mode={execution_mode.name} conflicts with platform={platform!r}")
        return arch, mode
    if type(backend_type) is not BackendType:
        raise TypeError(f"backend_type must be a BackendType, got {backend_type!r}")
    return backend_type, execution_mode if execution_mode is not None else ExecutionMode.SIM


def _warn_deprecated_platform(api: str) -> None:
    """Warn at a public legacy input boundary, pointing to its caller."""
    warnings.warn(
        f"{api}(platform=...) is deprecated; use backend_type and execution_mode instead.",
        DeprecationWarning,
        stacklevel=3,
    )


__all__ = [
    "ExecutionMode",
    "_parse_platform",
    "_platform_string",
    "_resolve_target",
    "_warn_deprecated_platform",
]
