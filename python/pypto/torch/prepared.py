# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicitly frozen eager kernels with fresh validation and ownership on every call."""

import inspect
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pypto._kernel_abi import KernelABI
from pypto.runtime.kernel.callable import KernelRegistration

from . import launch

if TYPE_CHECKING:
    from pypto.jit.decorator import JITFunction
    from pypto.runtime import CompileOptions


@dataclass(frozen=True, eq=False)
class PreparedKernel:
    """One specialization prepared by :func:`prepare` for a live process Worker.

    Call with every runtime argument, positionally or by name; constexpr and
    compilation options are already frozen. Dynamic tensor extents, allocations,
    scalar values and the current stream may change. Static shape, dtype and
    layout must still match. Calls never re-read source, constants or ambient
    compilation settings; prepare again to use changes to those inputs.

    This object retains the artifact and registration, never the sample tensors
    or a previous call frame. It cannot be used after Worker shutdown or fork.
    """

    _registration: KernelRegistration = field(repr=False)
    _plan: launch._LaunchPlan = field(init=False, repr=False)
    _binding: inspect.Signature = field(init=False, repr=False)

    def __post_init__(self) -> None:
        abi = self._registration.artifact.kernel_abi
        object.__setattr__(self, "_plan", launch._LaunchPlan.from_abi(abi))
        # The compiled ABI excludes constexpr slots. All remaining slots are
        # required, including scalar defaults and borrowed Out/InOut tensors.
        object.__setattr__(
            self,
            "_binding",
            inspect.Signature(
                [inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in abi.parameters]
            ),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Validate current arguments and enqueue once, returning output aliases."""
        registration = self._registration
        # Check the PID before any inherited locks, framework state or arguments.
        # Admission rechecks liveness under the owner's submission lock as well.
        registration.require_live()
        if kwargs:
            args = tuple(self._binding.bind(*args, **kwargs).arguments.values())
        frame = launch.describe_eager_call(
            self._plan.abi, args, registration.owner.config, signature=self._plan.signature
        )
        return launch._enqueue_frame(registration, frame, plan=self._plan)


def prepare(
    kernel: "JITFunction",
    *args: Any,
    config: "CompileOptions | None" = None,
    **kwargs: Any,
) -> PreparedKernel:
    """Compile and register one eager specialization without executing it.

    Call after ``pypto.torch.init`` and outside graph capture. Sample arguments
    follow the JIT signature, including constexpr values and all output buffers.
    Source, constants, static dimensions, layouts and effective compiler settings
    are frozen into the returned callable. Diagnostics and cache policy apply
    only during preparation. Ordinary JIT calls retain their existing semantics.

    Args:
        kernel: A ``@pl.jit`` kernel entry.
        *args: Sample positional arguments; tensor contents are not read.
        config: Compilation options for the frozen artifact.
        **kwargs: Sample keyword arguments, including constexpr bindings.

    Returns:
        A callable accepting only the compiled runtime parameters. It retains
        neither the sample arguments nor their device addresses.

    Raises:
        TypeError: The kernel, options or sample arguments are invalid.
        ValueError: Arguments or compiler settings conflict with the bound target.
        RuntimeError: Kernel mode is not initialized, the Worker is stale, or
            preparation is attempted inside graph capture.
    """
    from pypto.jit.decorator import JITFunction, _eager_kernel_config  # noqa: PLC0415
    from pypto.runtime.kernel.context import get_process_kernel_state  # noqa: PLC0415

    if not isinstance(kernel, JITFunction):
        raise TypeError(f"prepare requires a @pl.jit function, got {type(kernel).__name__}")
    target = _eager_kernel_config(config)
    state = get_process_kernel_state()
    specialization: Hashable | None = None

    def preflight(abi: KernelABI, arguments: list[Any], key: Hashable) -> bool:
        nonlocal specialization
        frame = launch.describe_eager_call(abi, arguments, target)
        if frame.capture_id:
            raise RuntimeError("pypto.torch.prepare must run outside graph capture")
        specialization = key
        return False

    artifact, _, _ = kernel._resolve_compiled(
        args, kwargs | {"config": config}, _kernel=True, _preflight=preflight, _kernel_config=target
    )
    registration = state.ensure_callable(artifact, target)
    assert specialization is not None
    state.publish_specialization(specialization, registration)
    return PreparedKernel(registration)
