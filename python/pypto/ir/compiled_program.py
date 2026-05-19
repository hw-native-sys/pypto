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
"""

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pypto.backend import BackendType
from pypto.pypto_core import DataType
from pypto.pypto_core import backend as _backend_core
from pypto.pypto_core.ir import (
    ConstInt,
    Function,
    FunctionType,
    ParamDirection,
    Program,
    ScalarType,
    ShapedType,
)
from pypto.runtime.device_tensor import DeviceTensor

# Type alias for arguments accepted by CompiledProgram.__call__().
# Tensor params accept ``torch.Tensor`` (host) or :class:`DeviceTensor`
# (worker-resident — skips H2D/D2H, see ``pypto.runtime.DeviceTensor``).
# Scalar params accept Python primitives or ctypes scalars (which are
# coerced to the correct ctypes type internally).
CallArg = torch.Tensor | DeviceTensor | int | float | bool | ctypes._SimpleCData

# IR DataType -> torch.dtype mapping.
# Keyed by string because nanobind DataType instances are not singletons,
# so dict lookup by object identity / hash may fail even for equal values.
_DATATYPE_TO_TORCH: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "index": torch.int64,
}
# uint16/32/64 were added in PyTorch 2.3; register only if available
for _name in ("uint16", "uint32", "uint64"):
    _torch_dtype = getattr(torch, _name, None)
    if _torch_dtype is not None:
        _DATATYPE_TO_TORCH[_name] = _torch_dtype
del _name, _torch_dtype

# IR DataType -> ctypes scalar constructor mapping.
# Used to wrap Python int/float/bool values into the correct ctypes scalar
# when calling a compiled program with scalar parameters.
_DATATYPE_TO_CTYPE: dict[str, type[ctypes._SimpleCData]] = {
    "fp16": ctypes.c_float,  # no native half; promote to float
    "fp32": ctypes.c_float,
    "fp64": ctypes.c_double,
    "bfloat16": ctypes.c_float,  # no native bfloat16; promote to float
    "int8": ctypes.c_int8,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "bool": ctypes.c_bool,
    "index": ctypes.c_int64,
}


def _to_torch_dtype(dtype: DataType) -> torch.dtype | None:
    """Convert an IR DataType to the corresponding torch.dtype."""
    return _DATATYPE_TO_TORCH.get(str(dtype))


@dataclass
class _ParamInfo:
    """Metadata for a single orchestration function parameter."""

    name: str
    direction: ParamDirection
    shape: list[int] | None  # None for scalar params
    dtype: DataType


def _extract_func_param_infos(func: Function) -> tuple[list[_ParamInfo], list[int], list[Any]]:
    """Extract parameter metadata from a specific IR function.

    Returns:
        Tuple of ``(param_infos, output_indices, return_types)``.
    """
    param_infos: list[_ParamInfo] = []
    output_indices: list[int] = []

    for i, (param, direction) in enumerate(zip(func.params, func.param_directions, strict=True)):
        param_type = param.type
        shape: list[int] | None = None

        if isinstance(param_type, ShapedType):
            dtype = param_type.dtype
            shape = [dim.value if isinstance(dim, ConstInt) else -1 for dim in param_type.shape]
        elif isinstance(param_type, ScalarType):
            dtype = param_type.dtype
        else:
            raise TypeError(
                f"Unsupported parameter type for {param.name_hint!r}: {type(param_type).__name__}. "
                f"Expected ShapedType or ScalarType."
            )

        param_infos.append(_ParamInfo(name=param.name_hint, direction=direction, shape=shape, dtype=dtype))

        # Only pure Out params can be auto-allocated in return-style calls.
        # InOut params require an initial value from the caller, so they must
        # be passed explicitly like inputs.
        if direction == ParamDirection.Out:
            output_indices.append(i)

    return param_infos, output_indices, list(func.return_types)


def _extract_param_infos(program: Program) -> tuple[list[_ParamInfo], list[int], list[Any]]:
    """Extract parameter metadata from the program's orchestration function.

    Args:
        program: A compiled IR Program.

    Returns:
        Tuple of (param_infos, output_indices, return_types).

    Raises:
        ValueError: If no Orchestration function is found.
    """
    # Prefer the Orchestration function; fall back to the sole function when
    # orchestration is auto-generated by codegen (single-kernel programs).
    orch_func = next(
        (f for f in program.functions.values() if f.func_type == FunctionType.Orchestration),
        None,
    )
    if orch_func is None:
        funcs = list(program.functions.values())
        if len(funcs) == 1:
            orch_func = funcs[0]
        else:
            raise ValueError(
                "Program has no Orchestration function and multiple InCore functions. "
                "Add an explicit Orchestration function to define the call signature."
            )

    return _extract_func_param_infos(orch_func)


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
        if len(info.shape) != len(arg.shape):
            raise TypeError(
                f"Parameter {info.name!r} expects rank {len(info.shape)} "
                f"(shape {tuple(info.shape)}); got DeviceTensor shape {arg.shape}"
            )
        for expected_dim, actual_dim in zip(info.shape, arg.shape, strict=True):
            if expected_dim >= 0 and expected_dim != actual_dim:
                raise TypeError(
                    f"Parameter {info.name!r} expects shape {tuple(info.shape)}; "
                    f"got DeviceTensor shape {arg.shape}"
                )
    expected_dtype = _to_torch_dtype(info.dtype)
    if expected_dtype is not None and arg.dtype != expected_dtype:
        raise TypeError(
            f"Parameter {info.name!r} expects dtype {expected_dtype}; got DeviceTensor dtype {arg.dtype}"
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


def _invoke_compiled(  # noqa: PLR0912 — branches for in-place vs return + scalar/tensor coercion
    *,
    output_dir: Path,
    platform: str,
    param_infos: list[_ParamInfo],
    output_indices: list[int],
    return_types: list[Any],
    args: tuple["CallArg", ...],
    config: Any,
    caller_name: str,
) -> "torch.Tensor | tuple[torch.Tensor, ...] | None":
    """Shared dispatch: coerce args, call the runtime, pack outputs.

    Used by both :meth:`CompiledProgram.__call__` (single-orch case) and
    :meth:`_SubChipCallable.__call__` (multi-orch case). The two callers
    differ only in *where* the artifacts live and *whose* metadata they
    apply — everything from argument coercion onward is identical.
    """
    n_params = len(param_infos)
    n_inputs = n_params - len(output_indices)
    has_return = len(return_types) > 0
    return_style = has_return and len(args) == n_inputs

    if len(args) == n_params:
        all_args: list[CallArg] = list(args)
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

    coerced: list[torch.Tensor | DeviceTensor | ctypes._SimpleCData] = []
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
            coerced.append(arg)

    from pypto.runtime.runner import RunConfig, _DfxOpts, execute_compiled  # noqa: PLC0415

    if config is None:
        config = RunConfig()

    execute_compiled(
        output_dir,
        coerced,
        platform=platform,
        device_id=config.device_id,
        pto_isa_commit=config.pto_isa_commit,
        dfx=_DfxOpts.from_run_config(config),
        block_dim=config.block_dim,
        aicpu_thread_num=config.aicpu_thread_num,
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


class CompiledProgram:
    """A compiled PyPTO program that can be called with torch tensors.

    Returned by :func:`ir.compile`.  ``CompiledProgram`` is a **compiled
    artifact** -- it stores the compilation output, target platform, and IR
    metadata.  The ``device`` index is provided at call time.

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
        program: Program,
        output_dir: str,
        *,
        backend_type: BackendType = BackendType.Ascend910B,
        platform: str | None = None,
    ) -> None:
        self._program = program
        self._output_dir = Path(output_dir).resolve()
        self._backend_type = backend_type
        self._platform = platform or _default_platform(backend_type)
        # Lazy metadata -- extracted on first call
        self._param_infos: list[_ParamInfo] | None = None
        self._output_indices: list[int] | None = None
        self._return_types: list[Any] | None = None

        # Multi-orch (L2-only) programs emit each Orchestration as a
        # self-contained sub-build under ``next_levels/<name>/``. Detect
        # those sub-dirs eagerly so ``__call__`` can error early and
        # ``__getitem__`` / ``__getattr__`` can dispatch by orch name.
        # The marker is ``orchestration/`` (always present after
        # codegen) rather than ``kernel_config.py`` (only present after
        # ptoas) so the dispatch surface works for inspection in
        # ``skip_ptoas=True`` builds — actually calling a sub-callable
        # without ``kernel_config.py`` fails cleanly inside
        # ``execute_compiled`` with a ``FileNotFoundError``.
        #
        # Skip detection for distributed (L3+) builds: those also lay
        # out ``next_levels/<chip_task>/`` but expose a single canonical
        # entry point via ``orchestration/host_orch.py`` and must be
        # invoked through ``CompiledProgram.__call__`` directly, not
        # via subscript dispatch.
        self._sub_chip_dirs: dict[str, Path] = {}
        has_host_orch = (self._output_dir / "orchestration" / "host_orch.py").is_file()
        next_levels = self._output_dir / "next_levels"
        if next_levels.is_dir() and not has_host_orch:
            for child in sorted(next_levels.iterdir()):
                if child.is_dir() and (child / "orchestration").is_dir():
                    self._sub_chip_dirs[child.name] = child

        # Debug runner only makes sense when there's a single canonical entry
        # point; multi-orch programs have one sub-build (and one debug script)
        # per orch, emitted by each sub-build's own pipeline.
        if not self._sub_chip_dirs:
            self._emit_debug_runner()

    def _emit_debug_runner(self) -> None:
        """Write ``<output_dir>/debug/run.py`` so the kernel can be replayed via
        ``python .../debug/run.py``.

        Best-effort: programs that lack a clean orchestration entry (unusual
        shapes, edge-case codegen) cannot have their param signature extracted
        here. In that case the file is skipped — the replay CLI is still usable
        directly against the output directory.

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

    # --- Properties -----------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Path to compiled artifacts (kernels/, orchestration/, etc.)."""
        return self._output_dir

    @property
    def program(self) -> Program:
        """The original IR Program (pre-optimization passes)."""
        return self._program

    @property
    def backend_type(self) -> BackendType:
        """Backend type used during compilation."""
        return self._backend_type

    @property
    def platform(self) -> str:
        """Target execution platform (e.g. ``"a2a3sim"``, ``"a5"``)."""
        return self._platform

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
            return str(self._output_dir) == str(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._output_dir)

    # --- Metadata (lazy) ------------------------------------------------------

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        """Return (param_infos, output_indices, return_types), extracting on first call."""
        if self._param_infos is None:
            self._param_infos, self._output_indices, self._return_types = _extract_param_infos(self._program)
        return self._param_infos, self._output_indices, self._return_types  # type: ignore[return-value]

    @property
    def param_names(self) -> list[str]:
        """Parameter names of the orchestration function."""
        param_infos, _, _ = self._get_metadata()
        return [p.name for p in param_infos]

    @property
    def output_indices(self) -> list[int]:
        """Indices of pure Out parameters (eligible for auto-allocation)."""
        _, out_idx, _ = self._get_metadata()
        return list(out_idx)

    @property
    def has_return(self) -> bool:
        """Whether the orchestration function has return values."""
        _, _, return_types = self._get_metadata()
        return len(return_types) > 0

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
        func = self._program.get_function(name)
        if func is None:
            raise KeyError(
                f"next_levels/{name}/ exists but function {name!r} is missing from the program IR."
            )
        return _SubChipCallable(name, func, self._sub_chip_dirs[name], self._platform)

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
                device index, profiling, etc.  Defaults to ``RunConfig()``.

        Returns:
            ``None`` for in-place calls, a single ``torch.Tensor`` or a
            ``tuple`` for return-style calls.

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
        )


class _SubChipCallable:
    """One L2 orchestration of a multi-orch :class:`CompiledProgram`.

    Returned by ``compiled[name]`` / ``compiled.<name>``. Self-contained:
    binds the orch's IR function, its sub-build directory, and the parent's
    platform, so calling it dispatches to that sub-dir only.
    """

    __test__ = False

    def __init__(self, name: str, func: Function, sub_dir: Path, platform: str) -> None:
        self._name = name
        self._func = func
        self._output_dir = sub_dir
        self._platform = platform
        self._param_infos, self._output_indices, self._return_types = _extract_func_param_infos(func)

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self._param_infos]

    def __repr__(self) -> str:
        return f"_SubChipCallable({self._name!r} @ {self._output_dir})"

    def __call__(
        self,
        *args: CallArg,
        config: Any = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        return _invoke_compiled(
            output_dir=self._output_dir,
            platform=self._platform,
            param_infos=self._param_infos,
            output_indices=self._output_indices,
            return_types=self._return_types,
            args=args,
            config=config,
            caller_name=f"orchestration {self._name!r}",
        )


# Public re-exports for callers (e.g. ir.compile()) that need orchestration
# parameter metadata without instantiating a full CompiledProgram.
ParamInfo = _ParamInfo
extract_param_infos = _extract_param_infos
