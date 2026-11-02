# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiled program wrapper returned by :func:`ir.compile`.

Provides a Triton-like callable API: compile once, then call with
torch tensors::

    compiled = ir.compile(MyProgram)
    compiled(a, b, c)                    # in-place on default sim, device 0
    c = compiled(a, b)                   # return style
    compiled(a, b, c, device=1)          # specify device at call time

**Why the ``pypto.runtime`` imports are function-local.** This module provides
the program execution handle around a shared artifact, so it reaches forward into
the layer below it. Each deferral has its own reason, and they are not the same:

- ``device_runner`` needs the optional ``simpler`` package at import time.
  Hoisting it would make ``simpler`` a hard requirement of ``import pypto.ir``,
  which the unit-test runners do not have.
- ``runner``, ``debug.run_script_writer`` and (in the L3 sibling)
  ``distributed_runner`` import cleanly at module scope; they stay deferred to
  keep ``pypto.ir``'s own import graph free of the dispatch layer, not because
  hoisting them fails.

The one import that genuinely could not be hoisted was the metadata the replay
writer reads back — that cycle is gone: it lives in the ``param_info`` leaf now.
"""

import ctypes
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import torch

from pypto._artifact_contract import ArtifactExecutionMode, ExecutionCapabilities
from pypto._target import (
    ExecutionMode,
    _parse_platform,
    _platform_string,
    _resolve_target,
    _warn_deprecated_platform,
)
from pypto.backend import BackendType
from pypto.pypto_core import DataType
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core.ir import (
    Function,
    ParamDirection,
    Program,
)
from pypto.runtime.device_tensor import DeviceTensor, StackedDeviceTensor

# Re-exported: these moved to a leaf module so a consumer of the metadata can
# depend on it without depending on this module. See ``param_info``.
from ._artifact_metadata import (  # noqa: F401 -- compatibility re-exports for L3 consumers
    _COMPILED_META_FILENAME,
    _COMPILED_META_SCHEMA,
    _DISTRIBUTED_META_FILENAME,
    ArtifactMetadata,
    _drop_foreign_build_markers,
    _is_obj_list,
    _is_str_dict,
    _load_meta,
    _meta_error,
    _remove_meta,
    _write_meta_atomically,
)
from .compiled_artifact import CompiledArtifact
from .param_info import (  # noqa: F401  -- re-export
    _DATATYPE_TO_CTYPE,
    _extract_func_param_infos,
    _extract_param_infos,
    _matches_torch_dtype,
    _param_info_from_dict,
    _param_info_to_dict,
    _ParamInfo,
    _to_torch_dtype,
    bind_complete_args,
)
from .param_info import (
    _DATATYPE_TO_TORCH as _DATATYPE_TO_TORCH,  # noqa: PLC0414
)
from .param_info import (
    ParamInfo as ParamInfo,  # noqa: PLC0414
)
from .param_info import (
    _datatype_from_string as _datatype_from_string,  # noqa: PLC0414
)
from .param_info import (
    _is_int_list as _is_int_list,  # noqa: PLC0414
)
from .param_info import arg_shape_as_declared as _arg_shape_as_declared

# Type alias for arguments accepted by CompiledProgram.__call__().
# Tensor params accept ``torch.Tensor`` (host), :class:`DeviceTensor`
# (worker-resident — skips H2D/D2H, see ``pypto.runtime.DeviceTensor``), or, for
# distributed programs, a :class:`StackedDeviceTensor` (per-card resident shards).
# Scalar params accept Python primitives or ctypes scalars (which are
# coerced to the correct ctypes type internally).
CallArg: TypeAlias = (
    "torch.Tensor | DeviceTensor | StackedDeviceTensor | int | float | bool | ctypes._SimpleCData[Any]"
)

if TYPE_CHECKING:
    from pypto.runtime._artifact_runtime import ArtifactRuntime


def _validate_fp4_carrier_shape(shape: Sequence[int], info: "_ParamInfo") -> None:
    """Validate the runtime byte-carrier shape for a packed FP4 parameter."""
    if info.dtype not in (DataType.FP4E2M1X2, DataType.FP4E1M2X2):
        return
    if not shape:
        raise TypeError(f"Packed FP4 parameter {info.name!r} must have rank >= 1")
    if shape[-1] <= 0:
        raise TypeError(
            f"Packed FP4 parameter {info.name!r} requires a positive runtime x2 carrier last dimension; "
            f"got shape {tuple(shape)}"
        )


def _validate_device_tensor(arg: DeviceTensor, info: _ParamInfo) -> None:
    """Check a ``DeviceTensor`` arg against IR parameter metadata.

    Raises:
        TypeError: when shape or dtype disagrees with ``info``.

    Rank is always enforced. Each static dim in ``info.shape`` is
    checked individually, so partially-dynamic signatures like
    ``[128, -1]`` still reject a tensor with the wrong leading
    dimension. Dynamic dims (``-1``) are skipped. Dtypes that the
    runtime can't map to torch are also skipped.
    """
    if info.shape is not None:
        blocked_arg_shape = _arg_shape_as_declared(arg.shape, info)
        if len(info.shape) != len(blocked_arg_shape):
            raise TypeError(
                f"Parameter {info.name!r} expects rank {len(info.shape)} "
                f"(shape {tuple(info.shape)}); got DeviceTensor shape {arg.shape}"
            )
        for expected_dim, actual_dim in zip(info.shape, blocked_arg_shape, strict=True):
            if expected_dim >= 0 and expected_dim != actual_dim:
                raise TypeError(
                    f"Parameter {info.name!r} expects shape {tuple(info.shape)}; "
                    f"got DeviceTensor shape {arg.shape}"
                )
    _validate_fp4_carrier_shape(arg.shape, info)
    expected_dtype = _to_torch_dtype(info.dtype)
    if expected_dtype is not None and not _matches_torch_dtype(info.dtype, arg.dtype):
        raise TypeError(
            f"Parameter {info.name!r} expects dtype {expected_dtype}; got DeviceTensor dtype {arg.dtype}"
        )


def _validate_stacked_tensor(arg: StackedDeviceTensor, info: _ParamInfo) -> None:
    """Check a ``StackedDeviceTensor`` arg against IR parameter metadata.

    The stacked tensor stands in for a ``[B, *tail]`` parameter the orchestrator
    slices along its leading dimension, so its ``full_shape`` is validated the
    same way as a :class:`DeviceTensor`'s shape: rank and every static dim must
    agree; dynamic dims (``-1``) are skipped. Per-shard shapes and dtype
    consistency are already enforced by ``StackedDeviceTensor.__init__``.

    Raises:
        TypeError: when ``full_shape`` rank/dims or dtype disagree with ``info``.
    """
    if info.shape is not None:
        blocked_full_shape = _arg_shape_as_declared(arg.full_shape, info)
        if len(info.shape) != len(blocked_full_shape):
            raise TypeError(
                f"Parameter {info.name!r} expects rank {len(info.shape)} "
                f"(shape {tuple(info.shape)}); got StackedDeviceTensor full_shape {arg.full_shape}"
            )
        for expected_dim, actual_dim in zip(info.shape, blocked_full_shape, strict=True):
            if expected_dim >= 0 and expected_dim != actual_dim:
                raise TypeError(
                    f"Parameter {info.name!r} expects shape {tuple(info.shape)}; "
                    f"got StackedDeviceTensor full_shape {arg.full_shape}"
                )
    _validate_fp4_carrier_shape(arg.full_shape, info)
    expected_dtype = _to_torch_dtype(info.dtype)
    if expected_dtype is not None and not _matches_torch_dtype(info.dtype, arg.dtype):
        raise TypeError(
            f"Parameter {info.name!r} expects dtype {expected_dtype}; "
            f"got StackedDeviceTensor dtype {arg.dtype}"
        )


def _build_full_args(
    input_args: tuple["CallArg", ...],
    param_infos: list[_ParamInfo],
    output_indices: list[int],
) -> list["CallArg"]:
    """Allocate output tensors and interleave with input args."""
    output_set = set(output_indices)
    all_tensors: list[CallArg] = []
    input_idx = 0

    for i, info in enumerate(param_infos):
        if i in output_set:
            if info.shape is None:
                raise ValueError(f"Cannot allocate output tensor {info.name!r}: no shape in IR")
            if any(d < 0 for d in info.shape):
                raise ValueError(
                    f"Cannot allocate output tensor {info.name!r}: shape {info.shape} "
                    f"contains dynamic dimensions. Pass all tensors explicitly (in-place style)."
                )
            torch_dtype = _to_torch_dtype(info.dtype)
            if torch_dtype is None:
                raise ValueError(f"Unsupported dtype {info.dtype} for output tensor {info.name!r}")
            all_tensors.append(torch.zeros(info.shape, dtype=torch_dtype))
        else:
            all_tensors.append(input_args[input_idx])
            input_idx += 1

    return all_tensors


def _coerce_args(  # noqa: PLR0912 — branches for in-place vs return + scalar/tensor coercion
    args: tuple["CallArg", ...],
    param_infos: list[_ParamInfo],
    output_indices: list[int],
    return_types: list[Any],
    *,
    caller_name: str,
) -> "tuple[list[torch.Tensor | DeviceTensor | ctypes._SimpleCData[Any]], bool]":
    """Validate user-provided args against IR metadata and coerce them.

    Returns ``(coerced, return_style)`` where ``coerced`` is a full positional
    list (length ``len(param_infos)``) and ``return_style`` is ``True`` when
    the caller passed only inputs and expects outputs to be returned.

    For return-style calls, output ``torch.Tensor`` slots are auto-allocated
    and placed at ``output_indices``. Tensor args are checked for shape/dtype
    against IR; scalar args are wrapped in the matching ctypes type.
    """
    n_params = len(param_infos)
    n_inputs = n_params - len(output_indices)
    has_return = len(return_types) > 0
    return_style = has_return and len(args) == n_inputs

    if len(args) == n_params:
        all_args = bind_complete_args(args, param_infos, caller_name=caller_name)
    elif return_style:
        all_args = _build_full_args(args, param_infos, output_indices)
    else:
        expected = f"{n_params} (in-place)"
        if has_return:
            expected += f" or {n_inputs} (return)"
        raise TypeError(
            f"{caller_name} expects {expected} arguments, got {len(args)}. "
            f"Parameters: {[p.name for p in param_infos]}"
        )

    coerced: list[torch.Tensor | DeviceTensor | ctypes._SimpleCData[Any]] = []
    for info, arg in zip(param_infos, all_args, strict=True):
        if info.shape is None:
            if isinstance(arg, torch.Tensor):
                raise TypeError(f"Parameter {info.name!r} is a scalar ({info.dtype}); got torch.Tensor")
            if isinstance(arg, ctypes._SimpleCData):
                expected_ctype = _DATATYPE_TO_CTYPE.get(str(info.dtype))
                if expected_ctype is not None and not isinstance(arg, expected_ctype):
                    raise TypeError(
                        f"Parameter {info.name!r} expects {expected_ctype.__name__} "
                        f"for dtype {info.dtype}; got {type(arg).__name__}"
                    )
                coerced.append(arg)
            else:
                ctype = _DATATYPE_TO_CTYPE.get(str(info.dtype))
                if ctype is None:
                    raise TypeError(f"Unsupported scalar dtype {info.dtype} for parameter {info.name!r}")
                coerced.append(ctype(arg))
        else:
            if not isinstance(arg, (torch.Tensor, DeviceTensor)):
                raise TypeError(
                    f"Parameter {info.name!r} is a tensor; got {type(arg).__name__}. "
                    f"Pass a torch.Tensor (host) or DeviceTensor (worker-resident)."
                )
            if isinstance(arg, DeviceTensor):
                _validate_device_tensor(arg, info)
            else:
                _validate_fp4_carrier_shape(arg.shape, info)
                if info.dtype in (DataType.FP4E2M1X2, DataType.FP4E1M2X2):
                    if not _matches_torch_dtype(info.dtype, arg.dtype):
                        raise TypeError(
                            f"Parameter {info.name!r} expects dtype {_to_torch_dtype(info.dtype)} "
                            f"or torch.uint8 carriers; got torch.Tensor dtype {arg.dtype}"
                        )
                    declared_shape = _arg_shape_as_declared(arg.shape, info)
                    if len(declared_shape) != len(info.shape) or any(
                        expected >= 0 and expected != actual
                        for expected, actual in zip(info.shape, declared_shape)
                    ):
                        raise TypeError(
                            f"Parameter {info.name!r} expects shape {tuple(info.shape)}; "
                            f"got torch.Tensor shape {tuple(arg.shape)}"
                        )
            coerced.append(arg)

    return coerced, return_style


def _invoke_compiled(
    *,
    output_dir: Path,
    platform: str,
    param_infos: list[_ParamInfo],
    output_indices: list[int],
    return_types: list[Any],
    args: tuple["CallArg", ...],
    config: Any,
    caller_name: str,
    artifact_runtime: Any = None,
) -> "torch.Tensor | tuple[torch.Tensor, ...] | None":
    """Shared dispatch: coerce args, call the runtime, pack outputs.

    Used by both :meth:`CompiledProgram.__call__` (single-orch case) and
    :meth:`_SubChipCallable.__call__` (multi-orch case). The two callers
    differ only in *where* the artifacts live and *whose* metadata they
    apply — everything from argument coercion onward is identical.
    An explicit run config selects its platform; without one, the platform
    bound to the compiled artifact is preserved.

    Returns *outputs*: ``None`` for in-place calls or the packed return
    tensors otherwise. Per-run timing is no longer returned — read it from
    the runtime's ``[STRACE]`` log markers (simpler PR #1177).
    """
    coerced, return_style = _coerce_args(
        args, param_infos, output_indices, return_types, caller_name=caller_name
    )

    # Lazy import from outside the overlay scope; its precision lives there.
    from pypto.runtime.runner import (  # noqa: PLC0415
        RunConfig,
        _execute_compiled,
    )

    execution_platform = platform if config is None else config.platform
    if config is None:
        config = RunConfig()

    _execute_compiled(
        output_dir,
        coerced,
        platform=execution_platform,
        device_id=config.device_id,
        dfx=config.dfx_options(),
        aicpu_thread_num=config.aicpu_thread_num,
        config=config,
        **({"artifact_runtime": artifact_runtime} if artifact_runtime is not None else {}),
    )

    if not return_style:
        return None
    outputs = [coerced[i] for i in output_indices]
    assert all(isinstance(o, torch.Tensor) for o in outputs)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)  # type: ignore[return-value]


def _default_platform(backend_type: BackendType) -> str:
    """Return the default simulator platform for a backend type.

    The mapping from backend to platform name lives on the per-backend
    BackendHandler so adding a new backend only requires implementing the
    handler.
    """
    return _backend_core.get_backend_instance(backend_type).get_handler().get_default_sim_platform()


def _reload_platform(
    persisted_backend: BackendType,
    persisted_platform: str | None,
    *,
    backend_type: BackendType | None,
    execution_mode: ExecutionMode | None,
    platform: str | None,
) -> str | None:
    """Apply target overrides, retaining the persisted value of an omitted axis."""
    if backend_type is None and execution_mode is None:
        return platform or persisted_platform
    if execution_mode is None and platform is None:
        execution_mode = _parse_platform(persisted_platform or _default_platform(persisted_backend))[1]
    target_backend = persisted_backend if backend_type is None else backend_type
    return _platform_string(*_resolve_target(target_backend, execution_mode, platform))


def _write_debug_runner(
    output_dir: Path,
    platform: str,
    get_metadata: Callable[[], tuple[list[_ParamInfo], list[int], list[Any]]],
) -> None:
    """Write ``<output_dir>/debug/run.py`` so the kernel can be replayed via
    ``python .../debug/run.py``.

    Best-effort: programs that lack a clean orchestration entry (unusual shapes,
    edge-case codegen) cannot have their param signature extracted, so the file
    is skipped — the replay CLI is still usable directly against the output dir.

    Shared by :class:`CompiledProgram` and
    :class:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram`;
    ``get_metadata`` is the caller's ``_get_metadata`` bound method.

    Disable globally by setting ``PYPTO_EMIT_DEBUG_RUNNER=0`` (also accepts
    ``false`` / ``no``).
    """
    if os.environ.get("PYPTO_EMIT_DEBUG_RUNNER", "").strip().lower() in ("0", "false", "no"):
        return

    from pypto.runtime.debug.run_script_writer import write_run_script  # noqa: PLC0415

    # Best-effort: neither metadata extraction nor writing the replay script may
    # crash the compile/execute pipeline, so any failure just skips the file.
    try:
        param_infos, _, _ = get_metadata()
        write_run_script(output_dir, param_infos, platform=platform)
    except Exception:  # noqa: BLE001
        return


class _RuntimeFacade:
    """Program-facing accessors delegated to the common compilation product.

    Keep compatibility accessors for runtime consumers while the shared artifact
    owns every callable and loading transition. Multi-entry parents can redirect
    loading to their selected child without introducing a second loader.
    """

    _artifact: CompiledArtifact

    @property
    def _output_dir(self) -> Path:
        return self._artifact.output_dir

    @_output_dir.setter
    def _output_dir(self, value: Path) -> None:
        self._artifact.output_dir = value

    @property
    def _platform(self) -> str:
        return self._artifact.platform

    @property
    def _chip_callable(self) -> Any:
        return self._artifact._chip_callable

    @property
    def _runtime_name(self) -> str | None:
        return self._artifact._runtime_name

    @property
    def _runtime_config(self) -> dict[str, Any] | None:
        return self._artifact._runtime_config

    @property
    def _artifact_runtime(self) -> "ArtifactRuntime | None":
        return self._artifact._artifact_runtime

    @_artifact_runtime.setter
    def _artifact_runtime(self, value: "ArtifactRuntime | None") -> None:
        self._artifact._artifact_runtime = value

    @property
    def execution_capabilities(self) -> ExecutionCapabilities:
        return self._artifact.execution_capabilities

    @property
    def output_dir(self) -> Path:
        """Directory owned by the shared artifact."""
        return self._artifact.output_dir

    @property
    def backend_type(self) -> BackendType:
        """Effective backend, including an explicit reload target override."""
        return _parse_platform(self.platform)[0]

    @property
    def execution_mode(self) -> ExecutionMode:
        """Simulator or onboard execution for this artifact."""
        return _parse_platform(self.platform)[1]

    @property
    def platform(self) -> str:
        """Effective target platform of the shared artifact."""
        return self._artifact.platform

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        """Derive the program runner's argument-allocation view from metadata.

        The runner only inspects the return count. Reconstruct its placeholders
        and Out indices here instead of retaining a second mutable signature.
        Preserve the original caller signature for PROGRAM binding and allocation;
        lowered parameters describe the assembly ABI.
        """
        metadata = self._artifact.metadata
        assert metadata is not None
        params = list(metadata.parameters)
        return (
            params,
            [i for i, p in enumerate(params) if p.direction == ParamDirection.Out],
            [None] * metadata.num_return_types,
        )

    @property
    def param_names(self) -> list[str]:
        """Parameter names of the orchestration function."""
        params, _, _ = self._get_metadata()
        return [p.name for p in params]

    @property
    def output_indices(self) -> list[int]:
        """Indices of pure Out parameters eligible for program auto-allocation."""
        _, outputs, _ = self._get_metadata()
        return outputs

    @property
    def has_return(self) -> bool:
        """Whether the orchestration function has return values."""
        _, _, returns = self._get_metadata()
        return bool(returns)

    def _check_runtime_access(self) -> None:
        """Allow wrappers to redirect multi-entry loading to their children."""

    def _ensure_runtime_loaded(self) -> None:
        self._check_runtime_access()
        self._artifact.load()

    def load(self) -> None:
        """Eagerly compile-and-load the runtime artefacts.

        Optional — :attr:`chip_callable`, :attr:`runtime_name`, and
        :attr:`runtime_config` all auto-load on first access.
        """
        self._ensure_runtime_loaded()

    @property
    def chip_callable(self) -> Any:
        """Simpler ``ChipCallable`` — hand to ``simpler.worker.Worker.register``."""
        self._ensure_runtime_loaded()
        return self._chip_callable

    @property
    def runtime_name(self) -> str:
        """Runtime ABI name baked into ``kernel_config.py`` (e.g. ``"tensormap_and_ringbuffer"``)."""
        self._ensure_runtime_loaded()
        assert self._runtime_name is not None
        return self._runtime_name

    def _host_tensor_indices(self) -> tuple[int, ...] | list[int]:
        """Read argument transfers without compiling binaries or opening a worker."""
        if self._runtime_config is not None:
            return self._runtime_config.get("host_tensor_indices", ())
        from pypto.runtime._artifact_sources import read_kernel_config  # noqa: PLC0415

        config_path = self._output_dir / "kernel_config.py"
        if not config_path.exists():
            return ()
        config = read_kernel_config(config_path)
        return getattr(config, "RUNTIME_CONFIG", {}).get("host_tensor_indices", ())

    @property
    def runtime_config(self) -> dict[str, Any]:
        """``RUNTIME_CONFIG`` dict from ``kernel_config.py`` (e.g. ``aicpu_thread_num``)."""
        self._ensure_runtime_loaded()
        assert self._runtime_config is not None
        return self._runtime_config


class CompiledProgram(_RuntimeFacade):
    """A compiled PyPTO program that can be called with torch tensors.

    Returned by :func:`ir.compile`. The shared :class:`CompiledArtifact` owns
    the compilation output, target and signature. This wrapper adds live IR,
    entry selection and program execution. The device is selected at call time.

    Two calling conventions:

    **In-place** (output passed as argument)::

        compiled = ir.compile(MyProgram)
        compiled(a, b, c)  # c modified in-place on device

    **Return** (program has a return value)::

        compiled = ir.compile(MyProgram)
        c = compiled(a, b)  # output allocated and returned

    Device selection is a keyword argument on each call::

        compiled(a, b, c, device=1)

    For backward compatibility, ``CompiledProgram`` also behaves like a
    path string via ``__str__`` and ``__fspath__``, so existing code that
    does ``os.path.join(ir.compile(prog), "kernels")`` continues to work.
    """

    __test__ = False  # Not a pytest test class

    def __init__(
        self,
        program: Program | None,
        output_dir: str,
        *,
        backend_type: BackendType = BackendType.Ascend910,
        platform: str | None = None,
        execution_mode: ExecutionMode | None = None,
        _sub_chip_names: Sequence[str] | None = None,
        _execution_capabilities: ExecutionCapabilities = ExecutionCapabilities(),
        _artifact: CompiledArtifact | None = None,
        _sub_artifacts: dict[str, CompiledArtifact] | None = None,
    ) -> None:
        # Reload and shared compilation supply the artifact directly. Only
        # legacy direct construction needs to extract metadata from live IR.
        _execution_capabilities.require(ArtifactExecutionMode.PROGRAM)
        self._program = program
        target = (
            platform or _default_platform(backend_type)
            if execution_mode is None
            else _platform_string(*_resolve_target(backend_type, execution_mode, platform))
        )
        self._artifact = _artifact or CompiledArtifact(output_dir, target)
        self._artifact.execution_capabilities.require(ArtifactExecutionMode.PROGRAM)
        # Multi-orch (L2-only) programs emit each Orchestration as a
        # self-contained sub-build under ``next_levels/<name>/``. Bind those
        # sub-dirs eagerly so ``__call__`` can error early and ``__getitem__``
        # / ``__getattr__`` can dispatch by orch name.
        self._sub_chip_dirs = self._resolve_sub_chip_dirs(_sub_chip_names)
        self._sub_artifacts = _sub_artifacts or {}

        # Only the fresh-compile path (live IR) writes artifacts: the reload
        # path must not clobber a user's hand-edited debug/run.py, nor rewrite
        # the sidecar it just read.
        if program is not None:
            if self._sub_chip_dirs:
                # Multi-orch: the parent has no single canonical signature, but
                # each next_levels/<name>/ IS a complete single-orch build dir,
                # so give each one its own sidecar. That makes the per-sub-build
                # reload the parent's error message points at actually work.
                # The debug runner stays parent-less for the same reason as
                # before: one script per sub-build, from its own pipeline.
                #
                # This build writes no top-level marker at all, so every one it
                # finds there -- its own stale parent sidecar included -- belongs
                # to an earlier compile into this reused directory and would
                # otherwise keep pointing consumers at that build.
                _drop_foreign_build_markers(self._output_dir, keep=())
                if _sub_artifacts is None:
                    self._persist_sub_chip_metadata()
            else:
                _drop_foreign_build_markers(
                    self._output_dir, keep=(_COMPILED_META_FILENAME, "kernel_config.py")
                )
                if _artifact is None:
                    self._persist_metadata()
                _write_debug_runner(self._output_dir, self._platform, self._get_metadata)

    def _resolve_sub_chip_dirs(self, sub_chip_names: Sequence[str] | None) -> dict[str, Path]:
        """Bind each L2 orchestration name to its ``next_levels/<name>/`` sub-build.

        *sub_chip_names* is the layout the caller's codegen just emitted --
        ``ir.compile`` passes
        :func:`~pypto.backend.pto_backend.multi_chip_orch_names` of the program it
        compiled, and :meth:`from_dir` passes ``[]`` because a top-level
        ``compiled_meta.json`` is only ever written for a single-orch build. It is
        authoritative *including when empty*: an empty list states "this build's one
        orchestration lives at the top level".

        That declaration is what makes recompiling into a reused ``output_dir``
        correct. ``ir.compile`` never clears the directory, so a ``next_levels/``
        left by an earlier multi-orch compile of a *different* program outlives it;
        deciding the layout by scanning for that directory would classify the new
        single-orch build as multi-orch, hide its own top-level artifacts behind the
        stale sub-builds, and delete the sidecar it just wrote. Sub-builds are not
        touched by a single-chip codegen, so a leftover one keeps its own artifacts
        *and* its matching sidecar -- stale as a pair, never mismatched.

        ``None`` means the caller declared no layout (a direct construction outside
        ``ir.compile``), leaving the on-disk scan as the only evidence of where the
        artifacts were written. The marker is ``orchestration/`` (always present
        after codegen) rather than ``kernel_config.py`` (only present after ptoas)
        so the dispatch surface is inspectable in ``skip_ptoas=True`` builds --
        calling a sub-callable without ``kernel_config.py`` then fails cleanly
        inside the dispatch path with a ``FileNotFoundError``. Distributed (L3+)
        builds are excluded: they also lay out ``next_levels/<chip_task>/`` but
        expose a single canonical entry point via ``orchestration/host_orch.py`` and
        must be invoked through :meth:`__call__` directly, not by subscript.
        """
        next_levels = self._output_dir / "next_levels"
        if sub_chip_names is not None:
            return {name: next_levels / name for name in sub_chip_names}

        sub_chip_dirs: dict[str, Path] = {}
        has_host_orch = (self._output_dir / "orchestration" / "host_orch.py").is_file()
        if next_levels.is_dir() and not has_host_orch:
            for child in sorted(next_levels.iterdir()):
                if child.is_dir() and (child / "orchestration").is_dir():
                    sub_chip_dirs[child.name] = child
        return sub_chip_dirs

    def _persist_metadata(self) -> None:
        """Write ``<output_dir>/compiled_meta.json`` for :meth:`from_dir`.

        Captures exactly what the dispatch path reads from the IR -- the
        orchestration param metadata (names, directions, shapes, dtypes) plus
        the return-type count -- alongside the platform / backend. Runtime
        artefacts (``chip_callable`` / ``runtime_name`` / ``runtime_config``)
        need no persistence: ``_compile_and_assemble`` already rederives them
        from the generated ``kernel_config.py``.

        Best-effort: a program without a resolvable orchestration signature
        emits nothing (mirrors :func:`_write_debug_runner`) *and* deletes any
        sidecar a previous compile into this directory left behind, so
        :meth:`from_dir` reports the missing file with a recompile hint instead
        of handing out a signature that no longer describes these artifacts.
        """
        try:
            self._get_metadata()
        except (ValueError, TypeError):
            _remove_meta(self._output_dir)
            return
        assert self._artifact.metadata is not None
        self._artifact.metadata.write(self.output_dir)

    def _persist_sub_chip_metadata(self) -> None:
        """Write one sidecar per ``next_levels/<name>/`` sub-build.

        Each sub-build is dispatched on its own (``compiled[<name>]``, or a
        direct ``CompiledProgram.from_dir(next_levels/<name>)``), so each needs
        the signature of *its* orchestration rather than the parent's. Only the
        sub-builds this compile emitted are visited (see
        :meth:`_resolve_sub_chip_dirs`), which is what keeps every sidecar
        describing the artifacts sitting next to it.

        Same best-effort contract as :meth:`_persist_metadata`: a sub-build the
        IR carries no matching function for, or whose signature cannot be
        extracted, emits no sidecar and drops whatever an earlier compile left
        in that directory, rather than failing the compile.
        """
        assert self._program is not None
        for name, sub_dir in self._sub_chip_dirs.items():
            func = self._program.get_function(name)
            if func is None:
                _remove_meta(sub_dir)
                continue
            try:
                params, _, returns = _extract_func_param_infos(func)
            except (ValueError, TypeError):
                _remove_meta(sub_dir)
                continue
            metadata = ArtifactMetadata(
                tuple(params), tuple(params), len(returns), self.platform, self.backend_type
            )
            metadata.write(sub_dir)
            self._sub_artifacts[name] = CompiledArtifact(sub_dir, self.platform, metadata)

    @classmethod
    def from_dir(
        cls,
        output_dir: str | os.PathLike[str],
        *,
        platform: str | None = None,
        backend_type: BackendType | None = None,
        execution_mode: ExecutionMode | None = None,
    ) -> "CompiledProgram":
        """Reconstruct a single-chip program from an existing ``build_output/`` dir.

        Rebuilds param metadata from ``compiled_meta.json`` (written at compile
        time) so the program is callable **and benchmarkable** without re-running
        the pypto compile -- the L2 counterpart of
        :meth:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram.from_dir`
        and the basis of the ``runtime_dir`` replay workflow (edit the generated
        orchestration cpp / ``.pto``, then re-measure)::

            from pypto.ir import CompiledProgram
            from pypto.runtime import ExecutionMode, benchmark

            compiled = CompiledProgram.from_dir(work_dir, execution_mode=ExecutionMode.ONBOARD)
            stats = benchmark(compiled, args, rounds=100)

        Args:
            output_dir: A build directory produced by a prior ``ir.compile`` of a
                single-orchestration (L2) program. Must contain
                ``compiled_meta.json``.
            platform: Deprecated; use backend_type and execution_mode.
                Override the persisted platform (e.g. swap ``a2a3sim`` ->
                ``a2a3`` to replay on hardware). ``None`` keeps the persisted
                value.
            execution_mode: Override the execution mode, retaining the persisted
                backend unless backend_type is also supplied.
            backend_type: Override the persisted codegen backend. ``None`` keeps
                the persisted value. An override retains the saved execution mode
                unless execution_mode or a legacy platform is also supplied.

        Returns:
            A :class:`CompiledProgram` whose ``__call__`` and runtime-artefact
            accessors behave exactly like the freshly-compiled object.
            :attr:`program` is ``None`` -- the IR itself is not persisted, so
            :meth:`validate_ir` still reads the directory's ``passes_dump/``
            rather than the live program. Always a single-orchestration program:
            the sidecar is only ever written for one, so a ``next_levels/`` an
            earlier multi-orch compile left in the same directory is ignored
            rather than mistaken for this build's dispatch surface.

        Raises:
            FileNotFoundError: ``compiled_meta.json`` is absent -- the directory
                predates this feature, is a distributed (L3+) build, or is the
                *parent* of a multi-orch build (each ``next_levels/<name>/``
                sub-build carries its own sidecar; reload one of those).
            ValueError: ``compiled_meta.json`` is unreadable, malformed, or
                records a ``schema`` version incompatible with this pypto build.
        """
        if platform is not None:
            _warn_deprecated_platform("CompiledProgram.from_dir")
        meta_path = Path(output_dir).resolve() / _COMPILED_META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — cannot reconstruct a compiled program from this "
                f"directory. It predates the single-chip replay feature, is a distributed "
                f"(L3+) build (use DistributedCompiledProgram.from_dir), or is the parent of "
                f"a multi-orch build (each next_levels/<name>/ sub-build has its own sidecar "
                f"— reload one of those). Recompile via ir.compile() to refresh."
            )
        artifact = CompiledArtifact.from_dir(output_dir)
        metadata = artifact.metadata
        assert metadata is not None
        metadata.execution_capabilities.require(ArtifactExecutionMode.PROGRAM)
        target = _reload_platform(
            metadata.backend_type,
            metadata.platform,
            backend_type=backend_type,
            execution_mode=execution_mode,
            platform=platform,
        )
        artifact.platform = target or metadata.platform
        return cls(
            None,
            str(output_dir),
            backend_type=backend_type or metadata.backend_type,
            platform=artifact.platform,
            _sub_chip_names=[],
            _artifact=artifact,
        )

    # --- Properties -----------------------------------------------------------

    @property
    def program(self) -> Program | None:
        """The original IR Program (pre-optimization passes).

        ``None`` for a program reconstructed via :meth:`from_dir` -- the IR is
        not persisted alongside the build artifacts.
        """
        return self._program

    # --- Pre-runtime IR validation -------------------------------------------

    def validate_ir(
        self,
        tensors: dict[str, torch.Tensor],
        expected: dict[str, torch.Tensor],
        *,
        rtol: float = 5e-2,
        atol: float = 5e-2,
    ) -> None:
        """Re-run ``torch_codegen`` on each dumped pass IR and numerically
        compare against golden outputs.

        This gives per-pass correctness checking before ever touching the
        device: each ``passes_dump/`` file is re-executed via
        :func:`pypto.debug.torch_codegen` and compared to *expected*.

        Requires the program to have been compiled with ``dump_passes=True``
        (the default), which produces ``<output_dir>/passes_dump/``.

        Args:
            tensors: Input tensors for executing generated functions, keyed
                by function parameter name.
            expected: Golden output tensors keyed by tensor name.
            rtol: Relative tolerance forwarded to ``torch.allclose``.
            atol: Absolute tolerance forwarded to ``torch.allclose``.

        Raises:
            FileNotFoundError: If no ``passes_dump/`` directory exists
                (i.e. compiled with ``dump_passes=False``).
            AssertionError: If any pass IR's numeric result diverges from
                *expected*.

        Example:
            >>> compiled = ir.compile(MyProgram)        # dump_passes=True
            >>> compiled.validate_ir(inputs, expected)  # per-pass check
        """
        # Lazy import keeps the core ir layer free of a debug-layer
        # dependency at import time.
        from pypto.debug import validate_pass_ir_codegen_results  # noqa: PLC0415

        passes_dump = self._output_dir / "passes_dump"
        if not passes_dump.is_dir():
            raise FileNotFoundError(
                f"No passes_dump/ under {self._output_dir}. "
                "Compile with dump_passes=True to enable IR validation."
            )
        validate_pass_ir_codegen_results(str(passes_dump), tensors, expected, rtol=rtol, atol=atol)

    # --- Runtime artefacts (lazy) — see _RuntimeFacade -----------------------
    #
    # load() / chip_callable / runtime_name / runtime_config live on the shared
    # _RuntimeFacade base. A single-orch program loads on first access; a
    # multi-orch program has no single canonical runtime, so the override below
    # steers callers to the per-orch sub-callable instead.

    def _check_runtime_access(self) -> None:
        if self._sub_chip_dirs:
            raise TypeError(
                f"Multi-orch program has {len(self._sub_chip_dirs)} orchestrations "
                f"{sorted(self._sub_chip_dirs)}; access runtime artefacts via "
                f"compiled[<name>] instead."
            )

    # --- Argument builders (internal; used by the runtime workers) ----------

    def _build_orch_args(
        self,
        *args: "CallArg",
        worker: Any | None = None,
    ) -> "tuple[Any, list[torch.Tensor | DeviceTensor | ctypes._SimpleCData[Any]], bool]":
        """Coerce user args and pack into simpler address-free ``TaskArgs``.

        Returns ``(orch_args, coerced, return_style)``:

        - ``orch_args``: simpler dispatch arg pack. Hand to
          ``Worker.run(cid, orch_args, cfg)``. ``worker`` owns the wire tensor
          identities and must be the same simpler Worker used for dispatch.
        - ``coerced``: full positional list of length ``len(param_infos)``.
          Scalar values are wrapped in their target ``ctypes`` type. For
          return-style callers, output ``torch.Tensor``s are auto-allocated
          and placed at :attr:`output_indices`; read those after dispatch
          to get the run's outputs.
        - ``return_style``: ``True`` if the caller passed only inputs.

        Raises:
            TypeError: Arg count / type mismatch, or called on a multi-orch
                program (use ``compiled[<name>]._build_orch_args(...)`` instead).
        """
        if self._sub_chip_dirs:
            raise TypeError(
                f"Multi-orch program has {len(self._sub_chip_dirs)} orchestrations "
                f"{sorted(self._sub_chip_dirs)}; use compiled[<name>]._build_orch_args(...)."
            )
        param_infos, output_indices, return_types = self._get_metadata()
        coerced, return_style = _coerce_args(
            args, param_infos, output_indices, return_types, caller_name="CompiledProgram"
        )
        if worker is None:
            raise TypeError(
                "_build_orch_args requires the simpler Worker that will own and dispatch these TaskArgs"
            )
        from pypto.runtime.runner import (  # noqa: PLC0415 -- lazy import from outside the overlay scope; its precision lives there
            _coerced_to_orch_args,
        )

        orch_args = _coerced_to_orch_args(coerced, worker, self._host_tensor_indices())
        return orch_args, coerced, return_style

    def _build_call_config(
        self,
        config: Any = None,
        *,
        aicpu_thread_num: int | None = None,
        dfx_dir: "Path | None" = None,
    ) -> Any:
        """Translate a pypto :class:`RunConfig` into a simpler ``CallConfig``.

        Precedence for ``aicpu_thread_num``: explicit kwarg > ``config``
        field > ``runtime_config`` baked into ``kernel_config.py``. When
        all three are unset, the simpler runtime's own default applies.

        DFX flags are copied straight from ``config``; ``dfx_dir`` (when
        given) becomes ``output_prefix``. Callers that enable DFX flags
        are responsible for creating ``dfx_dir`` beforehand — simpler's
        ``validate()`` rejects DFX-enabled calls without a valid prefix.
        """
        if self._sub_chip_dirs:
            raise TypeError(
                f"Multi-orch program has {len(self._sub_chip_dirs)} orchestrations "
                f"{sorted(self._sub_chip_dirs)}; use compiled[<name>]._build_call_config(...)."
            )
        from pypto.runtime.runner import RunConfig  # noqa: PLC0415
        from pypto.runtime.runner import _build_call_config as _runner_build_call_config  # noqa: PLC0415

        run_config = config if config is not None else RunConfig()
        return _runner_build_call_config(
            run_config,
            runtime_config=self.runtime_config,
            aicpu_thread_num_override=aicpu_thread_num,
            dfx_dir=dfx_dir,
        )

    # --- Backward compatibility: behave like a path string --------------------

    def __str__(self) -> str:
        return str(self._output_dir)

    def __repr__(self) -> str:
        return f"CompiledProgram({self._output_dir!s})"

    def __fspath__(self) -> str:
        """Allow ``os.path.join(compiled, ...)`` and ``Path(compiled)``."""
        return str(self._output_dir)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CompiledProgram):
            return self._output_dir == other._output_dir
        if isinstance(other, (str, os.PathLike)):
            # PathLike[Unknown] from the isinstance narrow; str() accepts any fspath result.
            return str(self._output_dir) == str(other)  # type: ignore[reportUnknownArgumentType]
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._output_dir)

    # --- Metadata (lazy) ------------------------------------------------------

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        """Adapt the shared signature, extracting legacy live IR only once."""
        if self._artifact.metadata is None:
            if self._program is None:
                raise RuntimeError(
                    "CompiledProgram has neither live IR nor persisted param metadata; "
                    "reconstruct via CompiledProgram.from_dir()."
                )
            params, _, returns = _extract_param_infos(self._program)
            self._artifact.metadata = ArtifactMetadata(
                tuple(params), tuple(params), len(returns), self.platform, self.backend_type
            )
        return super()._get_metadata()

    # --- Multi-orch dispatch (L2-only programs with >1 Orchestration) -------

    @property
    def orchestration_names(self) -> list[str]:
        """Names of L2 orchestrations addressable via ``compiled[name]``.

        Empty for single-orch programs (use ``compiled(...)`` directly).
        """
        return sorted(self._sub_chip_dirs)

    def __getitem__(self, name: str) -> "_SubChipCallable":
        if name not in self._sub_chip_dirs:
            raise KeyError(
                f"No orchestration {name!r} under {self._output_dir / 'next_levels'}. "
                f"Available: {sorted(self._sub_chip_dirs)}"
            )
        if self._program is None:
            raise RuntimeError(
                f"Multi-orch dispatch needs live IR; this CompiledProgram for {self._output_dir} "
                f"has none. Reload next_levels/{name}/ directly via CompiledProgram.from_dir(), "
                f"or recompile via ir.compile()."
            )
        func = self._program.get_function(name)
        if func is None:
            raise KeyError(
                f"next_levels/{name}/ exists but function {name!r} is missing from the program IR."
            )
        child = _SubChipCallable(
            name,
            func,
            self._sub_chip_dirs[name],
            self._platform,
            self.execution_capabilities,
            artifact=self._sub_artifacts.get(name),
        )
        self._sub_artifacts[name] = child._artifact
        return child

    def __getattr__(self, name: str) -> "_SubChipCallable":
        # __getattr__ only fires when normal attribute lookup fails. Read
        # _sub_chip_dirs from __dict__ to avoid recursion through __getattr__
        # itself during early-construction or pickle/copy edge cases.
        sub_dirs = self.__dict__.get("_sub_chip_dirs", {})
        if name in sub_dirs:
            return self[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    # --- Callable API ---------------------------------------------------------

    def __call__(
        self,
        *args: CallArg,
        config: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Execute the compiled program with torch tensors and/or scalars.

        Args match the orchestration function's parameter order.  For
        **in-place** style, pass all tensors (including outputs) and the
        output tensors are modified on device.  For **return** style,
        pass only input tensors and the outputs are allocated and returned.

        Scalar parameters (``pl.Scalar[...]``) accept Python ``int``,
        ``float``, ``bool``, or ``ctypes`` scalar values.

        Args:
            *args: Positional arguments — ``torch.Tensor`` for tensor
                params, ``int | float | bool | ctypes._SimpleCData`` for
                scalar params.
            config: Optional :class:`~pypto.runtime.runner.RunConfig` for
                execution platform, device index, profiling, etc. When omitted,
                the compiled artifact's platform and other runtime defaults apply.

        Returns:
            ``None`` for in-place calls, a single ``torch.Tensor`` or a
            ``tuple`` for return-style calls. Per-run on-device timing is no
            longer surfaced as an attribute — read it from the runtime's
            ``[STRACE]`` log markers (simpler PR #1177).

        Raises:
            TypeError: If the program has multiple L2 orchestrations (use
                ``compiled[name](...)``), or if argument count/types do
                not match the orchestration signature.
        """
        if self._sub_chip_dirs:
            raise TypeError(
                f"Program has {len(self._sub_chip_dirs)} L2 orchestrations "
                f"{sorted(self._sub_chip_dirs)}; select one explicitly via "
                f"compiled['<name>'](...) or compiled.<name>(...)."
            )
        param_infos, output_indices, return_types = self._get_metadata()
        return _invoke_compiled(
            output_dir=self._output_dir,
            platform=self._platform,
            param_infos=param_infos,
            output_indices=output_indices,
            return_types=return_types,
            args=args,
            config=config,
            caller_name="CompiledProgram",
            artifact_runtime=getattr(self, "_artifact_runtime", None),
        )


class _SubChipCallable(_RuntimeFacade):
    """One L2 orchestration of a multi-orch :class:`CompiledProgram`.

    Returned by ``compiled[name]`` / ``compiled.<name>``. Self-contained:
    binds the orch's IR function, its sub-build directory, and the parent's
    platform, so calling it dispatches to that sub-dir only.
    """

    __test__ = False

    def __init__(
        self,
        name: str,
        func: Function,
        sub_dir: Path,
        platform: str,
        execution_capabilities: ExecutionCapabilities = ExecutionCapabilities(),
        *,
        artifact: CompiledArtifact | None = None,
    ) -> None:
        execution_capabilities.require(ArtifactExecutionMode.PROGRAM)
        self._name = name
        if artifact is not None:
            self._artifact = artifact
        elif (sub_dir / _COMPILED_META_FILENAME).exists():
            self._artifact = CompiledArtifact.from_dir(sub_dir)
            self._artifact.platform = platform
            assert self._artifact.metadata is not None
        else:
            param_infos, _, return_types = _extract_func_param_infos(func)
            self._artifact = CompiledArtifact(
                sub_dir,
                platform,
                ArtifactMetadata(
                    tuple(param_infos),
                    tuple(param_infos),
                    len(return_types),
                    platform,
                    _parse_platform(platform)[0],
                    execution_capabilities=execution_capabilities,
                ),
            )

        assert self._artifact.metadata is not None
        self._artifact.execution_capabilities.require(ArtifactExecutionMode.PROGRAM)

    @property
    def name(self) -> str:
        return self._name

    def _build_orch_args(
        self,
        *args: "CallArg",
        worker: Any | None = None,
    ) -> "tuple[Any, list[torch.Tensor | DeviceTensor | ctypes._SimpleCData[Any]], bool]":
        param_infos, output_indices, return_types = self._get_metadata()
        coerced, return_style = _coerce_args(
            args,
            param_infos,
            output_indices,
            return_types,
            caller_name=f"orchestration {self._name!r}",
        )
        if worker is None:
            raise TypeError(
                "_build_orch_args requires the simpler Worker that will own and dispatch these TaskArgs"
            )
        from pypto.runtime.runner import (  # noqa: PLC0415 -- lazy import from outside the overlay scope; its precision lives there
            _coerced_to_orch_args,
        )

        orch_args = _coerced_to_orch_args(coerced, worker, self._host_tensor_indices())
        return orch_args, coerced, return_style

    def _build_call_config(
        self,
        config: Any = None,
        *,
        aicpu_thread_num: int | None = None,
        dfx_dir: "Path | None" = None,
    ) -> Any:
        from pypto.runtime.runner import RunConfig  # noqa: PLC0415
        from pypto.runtime.runner import _build_call_config as _runner_build_call_config  # noqa: PLC0415

        run_config = config if config is not None else RunConfig()
        return _runner_build_call_config(
            run_config,
            runtime_config=self.runtime_config,
            aicpu_thread_num_override=aicpu_thread_num,
            dfx_dir=dfx_dir,
        )

    def __repr__(self) -> str:
        return f"_SubChipCallable({self._name!r} @ {self._output_dir})"

    def __call__(
        self,
        *args: CallArg,
        config: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        param_infos, output_indices, return_types = self._get_metadata()
        return _invoke_compiled(
            output_dir=self._output_dir,
            platform=self._platform,
            param_infos=param_infos,
            output_indices=output_indices,
            return_types=return_types,
            args=args,
            config=config,
            caller_name=f"orchestration {self._name!r}",
        )


# Public re-exports for callers (e.g. ir.compile()) that need orchestration
# parameter metadata without instantiating a full CompiledProgram.

extract_param_infos = _extract_param_infos

# ``_validate_stacked_tensor`` is shared (lazy-imported) with the runtime's
# prepared-dispatch boundary and the ir suite; the listing keeps pyright's
# unused-function check satisfied without widening the public surface.
__all__ = [
    "_validate_stacked_tensor",
    "_COMPILED_META_SCHEMA",
    "_DISTRIBUTED_META_FILENAME",
    "_is_obj_list",
    "_is_str_dict",
    "_load_meta",
    "_meta_error",
    "_write_meta_atomically",
    "_param_info_from_dict",
    "_param_info_to_dict",
]
