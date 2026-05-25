# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed compiled program wrapper for L3+ programs.

Provides a callable API similar to :class:`CompiledProgram` but executes
through simpler's distributed runtime (Worker level=3)::

    compiled = ir.compile(MyDistributedProgram)
    compiled(a, b, c)   # executes via simpler Worker(level=3)
"""

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pypto.backend import BackendType
from pypto.pypto_core.ir import Program, Role, level_to_linqu_level
from pypto.runtime.device_tensor import DeviceTensor

from .compiled_program import (
    CallArg,
    _default_platform,
    _extract_param_infos,
    _ParamInfo,
    _to_torch_dtype,
    _validate_device_tensor,
)

if TYPE_CHECKING:
    from pypto.runtime.distributed_runner import DistributedRuntime


def _extract_param_infos_from_func(func):
    """Extract parameter metadata from a specific function."""
    from pypto.pypto_core.ir import ConstInt, ParamDirection, ScalarType, ShapedType  # noqa: PLC0415

    param_infos = []
    output_indices = []

    for i, (param, direction) in enumerate(zip(func.params, func.param_directions, strict=True)):
        param_type = param.type
        shape = None

        if isinstance(param_type, ShapedType):
            dtype = param_type.dtype
            shape = [dim.value if isinstance(dim, ConstInt) else -1 for dim in param_type.shape]
        elif isinstance(param_type, ScalarType):
            dtype = param_type.dtype
        else:
            raise TypeError(
                f"Unsupported parameter type for {param.name_hint!r}: {type(param_type).__name__}"
            )

        param_infos.append(_ParamInfo(name=param.name_hint, direction=direction, shape=shape, dtype=dtype))
        if direction == ParamDirection.Out:
            output_indices.append(i)

    return param_infos, output_indices, list(func.return_types)


@dataclass
class DistributedConfig:
    """Configuration for L3 distributed execution."""

    device_ids: list[int] = field(default_factory=lambda: [0])
    num_sub_workers: int = 0
    runtime: str = "tensormap_and_ringbuffer"
    block_dim: int = 1
    aicpu_thread_num: int = 1


class DistributedCompiledProgram:
    """A compiled L3+ distributed program that executes via simpler Worker(level=3).

    Returned by :func:`ir.compile` when the program contains HOST-level
    or higher hierarchy functions.

    Calling conventions match :class:`CompiledProgram`:

    **In-place** (output passed as argument)::

        compiled(a, b, c)

    **Return** (program has a return value)::

        c = compiled(a, b)
    """

    __test__ = False

    def __init__(
        self,
        program: Program,
        output_dir: str,
        *,
        backend_type: BackendType = BackendType.Ascend910B,
        platform: str | None = None,
        distributed_config: DistributedConfig | None = None,
    ) -> None:
        # ``program`` is the post-pass IR. The runtime needs post-pass IR for
        # orchestrator metadata (post-SSA names that match the generated
        # host_orch.py) and to iterate Orchestration functions synthesized by
        # passes such as OutlineHierarchyScopes.
        self._program = program
        self._output_dir = Path(output_dir).resolve()
        self._backend_type = backend_type
        self._platform = platform or _default_platform(backend_type)
        self._distributed_config = distributed_config or DistributedConfig()
        self._param_infos = None
        self._output_indices = None
        self._return_types = None

        self._emit_debug_runner()

    def _emit_debug_runner(self) -> None:
        """Write ``<output_dir>/debug/run.py`` for replaying this program.

        Best-effort: distributed programs without a clean orchestration entry
        will skip emission (the replay CLI is still usable directly).

        Disable globally by setting ``PYPTO_EMIT_DEBUG_RUNNER=0`` (also accepts
        ``false`` / ``no``).
        """
        if os.environ.get("PYPTO_EMIT_DEBUG_RUNNER", "").strip().lower() in ("0", "false", "no"):
            return

        from pypto.runtime.debug.run_script_writer import write_run_script  # noqa: PLC0415

        try:
            param_infos, _, _ = self._get_metadata()
        except (ValueError, TypeError):
            return
        write_run_script(self._output_dir, param_infos, platform=self._platform)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def program(self) -> Program:
        return self._program

    @property
    def platform(self) -> str:
        return self._platform

    def __str__(self) -> str:
        return str(self._output_dir)

    def __repr__(self) -> str:
        return f"DistributedCompiledProgram({self._output_dir!s})"

    def __fspath__(self) -> str:
        return str(self._output_dir)

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        if self._param_infos is None:
            # Find the HOST orchestrator function (post-SSA names match the
            # generated Python code).
            host_orch = None
            for func in self._program.functions.values():
                if (
                    func.level is not None
                    and level_to_linqu_level(func.level) >= 3
                    and func.role is not None
                    and func.role == Role.Orchestrator
                ):
                    host_orch = func
                    break

            if host_orch is not None:
                self._param_infos, self._output_indices, self._return_types = _extract_param_infos_from_func(
                    host_orch
                )
            else:
                self._param_infos, self._output_indices, self._return_types = _extract_param_infos(
                    self._program
                )
        assert self._output_indices is not None and self._return_types is not None
        return self._param_infos, self._output_indices, self._return_types

    def __call__(
        self,
        *args: CallArg,
        config: Any = None,
    ) -> torch.Tensor | DeviceTensor | tuple[torch.Tensor | DeviceTensor, ...] | None:
        """Execute the distributed program via simpler Worker(level=3)."""
        from pypto.runtime.distributed_runner import execute_distributed  # noqa: PLC0415

        param_infos, output_indices, return_types = self._get_metadata()
        n_params = len(param_infos)
        n_inputs = n_params - len(output_indices)
        has_return = len(return_types) > 0
        return_style = has_return and len(args) == n_inputs

        if len(args) == n_params:
            all_args: list[CallArg] = list(args)
        elif return_style:
            all_args = self._build_full_args(args, param_infos, output_indices)
        else:
            expected = f"{n_params} (in-place)"
            if has_return:
                expected += f" or {n_inputs} (return)"
            raise TypeError(
                f"DistributedCompiledProgram expects {expected} arguments, got {len(args)}. "
                f"Parameters: {[p.name for p in param_infos]}"
            )

        # Validate and coerce args. Tensor params accept a host ``torch.Tensor``
        # or a worker-resident ``DeviceTensor`` (skips H2D/D2H), matching the L2
        # ``CompiledProgram`` calling convention.
        coerced: list[torch.Tensor | DeviceTensor] = []
        for info, arg in zip(param_infos, all_args, strict=True):
            if isinstance(arg, DeviceTensor):
                _validate_device_tensor(arg, info)
                coerced.append(arg)
                continue
            if not isinstance(arg, torch.Tensor):
                raise TypeError(
                    f"Distributed programs only support tensor parameters "
                    f"(torch.Tensor host or DeviceTensor worker-resident). "
                    f"Parameter {info.name!r} got {type(arg).__name__}"
                )
            coerced.append(arg)

        execute_distributed(self, coerced, config)

        if not return_style:
            return None
        outputs = [coerced[i] for i in output_indices]
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def prepare(
        self,
        config: Any = None,
        *,
        sub_worker_overrides: dict[str, Callable[..., Any]] | None = None,
    ) -> "DistributedRuntime":
        """Prepare a reusable L3 execution handle (setup once, dispatch many).

        Runs the expensive setup (``compile_and_assemble``, generated-module
        loading, ``Worker(level=3)`` construction + registration + ``init()``)
        exactly once and returns a :class:`DistributedRuntime` that dispatches
        many times on the held Worker. The handle also exposes device-memory
        helpers (``alloc_tensor`` / ``malloc`` / ``copy_to`` / ``copy_from`` /
        ``free``) for building worker-resident :class:`DeviceTensor` buffers
        that survive across dispatches.

        Per-call inputs and outputs are reused-in-place **shared-memory** host
        ``torch.Tensor`` buffers (allocated before ``prepare()``) and/or
        worker-resident ``DeviceTensor`` / ``ContinuousTensor`` arguments.
        Non-shared host tensors are rejected (the forked chip worker cannot see
        a buffer allocated after the fork). The convenience host-to-device
        upload of arbitrary host ``torch.Tensor`` inputs is only available on
        the one-shot ``compile(...)(*args)`` / ``execute_distributed`` path.

        Args:
            config: Optional run configuration (reserved; currently unused).
            sub_worker_overrides: Replace a generated sub-worker placeholder
                (matched by name) with your own callable — e.g. a real sampling
                closure in place of the codegen stub. Each name must be a
                sub-worker the program declares; an unknown name raises
                ``ValueError``.

        Returns:
            A :class:`DistributedRuntime`; use it as a context manager or call
            ``close()`` when done.
        """
        from pypto.runtime.distributed_runner import DistributedRuntime  # noqa: PLC0415

        return DistributedRuntime(self, config, sub_worker_overrides=sub_worker_overrides)

    @staticmethod
    def _build_full_args(input_args, param_infos, output_indices):
        output_set = set(output_indices)
        all_tensors = []
        input_idx = 0

        for i, info in enumerate(param_infos):
            if i in output_set:
                if info.shape is None:
                    raise ValueError(f"Cannot allocate output tensor {info.name!r}: no shape in IR")
                if any(d < 0 for d in info.shape):
                    raise ValueError(
                        f"Cannot allocate output tensor {info.name!r}: shape {info.shape} "
                        f"contains dynamic dimensions."
                    )
                torch_dtype = _to_torch_dtype(info.dtype)
                if torch_dtype is None:
                    raise ValueError(f"Unsupported dtype {info.dtype} for output tensor {info.name!r}")
                all_tensors.append(torch.zeros(info.shape, dtype=torch_dtype))
            else:
                all_tensors.append(input_args[input_idx])
                input_idx += 1

        return all_tensors
