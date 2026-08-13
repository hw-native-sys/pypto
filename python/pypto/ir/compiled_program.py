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
import json
import os
from collections.abc import Callable, Sequence
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
from pypto.runtime.device_tensor import DeviceTensor, StackedDeviceTensor

# Type alias for arguments accepted by CompiledProgram.__call__().
# Tensor params accept ``torch.Tensor`` (host), :class:`DeviceTensor`
# (worker-resident — skips H2D/D2H, see ``pypto.runtime.DeviceTensor``), or, for
# distributed programs, a :class:`StackedDeviceTensor` (per-card resident shards).
# Scalar params accept Python primitives or ctypes scalars (which are
# coerced to the correct ctypes type internally).
CallArg = torch.Tensor | DeviceTensor | StackedDeviceTensor | int | float | bool | ctypes._SimpleCData

# Filename of the small JSON sidecar persisted alongside the build artifacts so
# a single-orchestration program can be reconstructed (``from_dir``) without the
# live IR -- the L2 counterpart of ``distributed_meta.json``. Bump
# ``_COMPILED_META_SCHEMA`` on any incompatible format change.
#
# Both sidecars embed the same ``_param_info_to_dict`` payload but version it
# independently, so a change to *that* shared format must bump this schema AND
# ``distributed_compiled_program._META_SCHEMA``.
_COMPILED_META_FILENAME = "compiled_meta.json"
_COMPILED_META_SCHEMA = 1

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
# Float8 / MX scale dtypes (PyTorch 2.1+ / 2.3+ / 2.7+); map IR string → torch.dtype.
# Packed MXFP4 (fp4 ↔ float4_e2m1fn_x2) must be here so return-style execution
# can allocate FP4 outputs after JIT specialization accepts the torch dtype.
for _ir_name, _torch_name in (
    ("fp8e4m3fn", "float8_e4m3fn"),
    ("fp8e5m2", "float8_e5m2"),
    ("fp8e8m0", "float8_e8m0fnu"),
    ("fp4", "float4_e2m1fn_x2"),
):
    _torch_dtype = getattr(torch, _torch_name, None)
    if _torch_dtype is not None:
        _DATATYPE_TO_TORCH[_ir_name] = _torch_dtype
del _ir_name, _torch_name, _torch_dtype

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


def _to_runtime_shape(shape: list[int], dtype: DataType) -> list[int]:
    """Convert logical IR shape to the runtime carrier shape.

    Only packed FP4 differs: Torch/runtime use one x2 element per byte while
    PyPTO IR counts logical nibbles. The conversion remains at the call ABI
    boundary, so no persistent storage_shape is needed in IR types.
    """
    runtime_shape = list(shape)
    if dtype == DataType.FP4 and runtime_shape[-1] != -1:
        # Static FP4 shape validity is centralized in ShapedType construction.
        runtime_shape[-1] //= 2
    return runtime_shape


def _validate_fp4_carrier_shape(shape: Sequence[int], info: "_ParamInfo") -> None:
    """Validate the runtime x2 carrier shape for one FP4 parameter."""
    if info.dtype != DataType.FP4:
        return
    if not shape:
        raise TypeError(f"Packed FP4 parameter {info.name!r} must have rank >= 1")
    if shape[-1] <= 0:
        raise TypeError(
            f"Packed FP4 parameter {info.name!r} requires a positive runtime x2 carrier last dimension; "
            f"got shape {tuple(shape)}"
        )


@dataclass
class _ParamInfo:
    """Metadata for a single orchestration function parameter."""

    name: str
    direction: ParamDirection
    shape: list[int] | None  # None for scalar params
    dtype: DataType


# Reverse map ``str(DataType) -> DataType`` for JSON round-tripping (e.g. the L3
# ``distributed_meta.json`` consumed by ``DistributedCompiledProgram.from_dir``).
# ``DataType`` is a non-singleton nanobind type, so reconstruction goes through
# the canonical string (``str(DataType.FP32) == "fp32"``) rather than identity.
# Built from the named constants exposed on ``DataType`` so it tracks the C++
# enum without a hand-maintained literal table.
_STR_TO_DATATYPE: dict[str, DataType] = {}
for _dt_name in (
    "BOOL", "INT4", "INT8", "INT16", "INT32", "INT64", "UINT4", "UINT8", "UINT16",
    "UINT32", "UINT64", "FP4", "FP8E4M3FN", "FP8E5M2", "FP8E8M0", "FP16", "FP32", "BF16",
    "HF4", "HF8", "INDEX", "TASK_ID",
):  # fmt: skip
    _dt = getattr(DataType, _dt_name, None)
    if _dt is not None:
        _STR_TO_DATATYPE[str(_dt)] = _dt
del _dt_name, _dt


def _datatype_from_string(s: str) -> DataType:
    """Reconstruct a :class:`DataType` from its ``str()`` form (e.g. ``"fp32"``)."""
    dt = _STR_TO_DATATYPE.get(s)
    if dt is None:
        raise ValueError(f"Unknown DataType string {s!r}; known: {sorted(_STR_TO_DATATYPE)}")
    return dt


def _param_info_to_dict(info: _ParamInfo) -> dict[str, Any]:
    """Serialise a :class:`_ParamInfo` to a JSON-safe dict (see :func:`_param_info_from_dict`)."""
    return {
        "name": info.name,
        "direction": info.direction.name,
        "shape": info.shape,
        "dtype": str(info.dtype),
    }


def _param_info_from_dict(d: dict[str, Any]) -> _ParamInfo:
    """Reconstruct a :class:`_ParamInfo` from :func:`_param_info_to_dict` output.

    Every field is validated here rather than trusted: the sidecars this parses
    live in a user-supplied build directory, so a hand-edited or version-skewed
    value must fail at load time with an actionable :class:`ValueError` instead
    of surfacing much later as an unrelated ``TypeError`` from output
    allocation or shape handling.

    Raises:
        KeyError: A required field is absent.
        ValueError: A field is present but not a valid value for its slot.
    """
    name = d["name"]
    if not isinstance(name, str):
        raise ValueError(f"'name' must be a string, got {name!r}")

    # Strict enum-member lookup, not getattr(): ``getattr(ParamDirection, "mro")``
    # resolves to a bound method and would sail through as a bogus direction.
    raw_direction = d["direction"]
    direction = ParamDirection.__members__.get(raw_direction) if isinstance(raw_direction, str) else None
    if direction is None:
        raise ValueError(
            f"'direction' must be one of {sorted(ParamDirection.__members__)}, got {raw_direction!r}"
        )

    # ``bool`` is an ``int`` subclass; a JSON ``true`` is never a dimension.
    shape = d["shape"]
    if shape is not None:
        is_dim_list = isinstance(shape, list) and all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
        )
        if not is_dim_list:
            raise ValueError(f"'shape' must be null or a list of ints, got {shape!r}")

    raw_dtype = d["dtype"]
    if not isinstance(raw_dtype, str):
        raise ValueError(f"'dtype' must be a string, got {raw_dtype!r}")

    return _ParamInfo(
        name=name,
        direction=direction,
        shape=shape,
        dtype=_datatype_from_string(raw_dtype),
    )


def _remove_meta(output_dir: Path) -> None:
    """Drop ``<output_dir>/compiled_meta.json`` if present.

    ``ir.compile`` never clears ``output_dir`` (``os.makedirs(exist_ok=True)``),
    so recompiling a *different* program shape into a reused directory can leave
    a sidecar describing the previous one. Every fresh-compile path that does
    not overwrite the sidecar must therefore remove it: a stale signature would
    drive the new artifacts with the old parameter ABI, which
    :meth:`CompiledProgram.from_dir` has no way to detect.
    """
    (output_dir / _COMPILED_META_FILENAME).unlink(missing_ok=True)


def _load_compiled_meta(meta_path: Path) -> dict[str, Any]:
    """Read and fully validate a ``compiled_meta.json`` sidecar.

    Every malformed-payload failure surfaces as a single :class:`ValueError`
    naming *meta_path* and the recompile fix, rather than leaking whichever
    ``JSONDecodeError`` / ``KeyError`` / ``AttributeError`` / ``TypeError`` the
    offending field happened to raise. A build directory is user-supplied
    input, so a hand-edited or truncated sidecar must fail like bad input, not
    like an internal error.

    Returns:
        ``{"param_infos", "num_return_types", "platform", "backend_type"}``,
        already converted to their runtime types.

    Raises:
        ValueError: the file is not readable as the expected schema.
    """

    def _bad(detail: str) -> ValueError:
        return ValueError(
            f"Invalid {_COMPILED_META_FILENAME} in {meta_path}: {detail}. The metadata was "
            f"written by a different pypto version or hand-edited — recompile via "
            f"ir.compile() to refresh."
        )

    try:
        raw_text = meta_path.read_text()
    except OSError as exc:  # unreadable / vanished / replaced by a directory
        raise _bad(f"cannot be read ({exc})") from exc
    except UnicodeDecodeError as exc:  # binary or mis-encoded file
        raise _bad(f"is not valid UTF-8 text ({exc})") from exc
    try:
        meta = json.loads(raw_text)
    except ValueError as exc:  # JSONDecodeError is a ValueError subclass
        raise _bad(f"not valid JSON ({exc})") from exc
    if not isinstance(meta, dict):
        raise _bad(f"expected a JSON object, got {type(meta).__name__}")

    schema = meta.get("schema")
    if schema != _COMPILED_META_SCHEMA:
        raise ValueError(
            f"Incompatible {_COMPILED_META_FILENAME} schema {schema!r} (expected "
            f"{_COMPILED_META_SCHEMA}) in {meta_path}. The metadata was written by a "
            f"different pypto version — recompile via ir.compile() to refresh."
        )

    raw_params = meta.get("params")
    if not isinstance(raw_params, list):
        raise _bad(f"'params' must be a list, got {type(raw_params).__name__}")
    param_infos: list[_ParamInfo] = []
    for i, entry in enumerate(raw_params):
        if not isinstance(entry, dict):
            raise _bad(f"params[{i}] must be an object, got {type(entry).__name__}")
        try:
            param_infos.append(_param_info_from_dict(entry))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise _bad(f"params[{i}] is malformed ({exc})") from exc

    num_return_types = meta.get("num_return_types", 0)
    if not isinstance(num_return_types, int) or isinstance(num_return_types, bool) or num_return_types < 0:
        raise _bad(f"'num_return_types' must be a non-negative integer, got {num_return_types!r}")

    # Strict enum-member lookup, not getattr(): ``getattr(BackendType, "mro")``
    # resolves to a bound method and would sail through as a bogus backend.
    raw_backend = meta.get("backend_type", "Ascend910B")
    backend_type = BackendType.__members__.get(raw_backend) if isinstance(raw_backend, str) else None
    if backend_type is None:
        raise _bad(f"'backend_type' must be one of {sorted(BackendType.__members__)}, got {raw_backend!r}")

    platform = meta.get("platform")
    if platform is not None and not isinstance(platform, str):
        raise _bad(f"'platform' must be a string or absent, got {type(platform).__name__}")

    return {
        "param_infos": param_infos,
        "num_return_types": num_return_types,
        "platform": platform,
        "backend_type": backend_type,
    }


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
            logical_shape = [dim.value if isinstance(dim, ConstInt) else -1 for dim in param_type.shape]
            shape = _to_runtime_shape(logical_shape, dtype)
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
    _validate_fp4_carrier_shape(arg.shape, info)
    expected_dtype = _to_torch_dtype(info.dtype)
    if expected_dtype is not None and arg.dtype != expected_dtype:
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
        if len(info.shape) != len(arg.full_shape):
            raise TypeError(
                f"Parameter {info.name!r} expects rank {len(info.shape)} "
                f"(shape {tuple(info.shape)}); got StackedDeviceTensor full_shape {arg.full_shape}"
            )
        for expected_dim, actual_dim in zip(info.shape, arg.full_shape, strict=True):
            if expected_dim >= 0 and expected_dim != actual_dim:
                raise TypeError(
                    f"Parameter {info.name!r} expects shape {tuple(info.shape)}; "
                    f"got StackedDeviceTensor full_shape {arg.full_shape}"
                )
    _validate_fp4_carrier_shape(arg.full_shape, info)
    expected_dtype = _to_torch_dtype(info.dtype)
    if expected_dtype is not None and arg.dtype != expected_dtype:
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
) -> tuple[list[torch.Tensor | DeviceTensor | ctypes._SimpleCData], bool]:
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
            else:
                _validate_fp4_carrier_shape(arg.shape, info)
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

    from pypto.runtime.runner import RunConfig, _DfxOpts, execute_compiled  # noqa: PLC0415

    execution_platform = platform if config is None else config.platform
    if config is None:
        config = RunConfig()

    execute_compiled(
        output_dir,
        coerced,
        platform=execution_platform,
        device_id=config.device_id,
        dfx=_DfxOpts.from_run_config(config),
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
    """Lazy compile-and-load of runtime artefacts.

    Shared by :class:`CompiledProgram` and :class:`_SubChipCallable`. The host
    class must define the backing fields ``_output_dir`` (Path), ``_platform``
    (str), and the lazily-populated ``_chip_callable`` / ``_runtime_name`` /
    ``_runtime_config`` (initialised to ``None``). A host may override
    :meth:`_check_runtime_access` to forbid direct loading.
    """

    _output_dir: Path
    _platform: str
    _chip_callable: Any
    _runtime_name: str | None
    _runtime_config: dict[str, Any] | None

    def _check_runtime_access(self) -> None:
        """Hook run before the first compile-and-load. Default: allow.

        Hosts that have no single canonical runtime (e.g. a multi-orch
        :class:`CompiledProgram`) override this to redirect callers elsewhere.
        """

    def _ensure_runtime_loaded(self) -> None:
        if self._chip_callable is not None:
            return
        self._check_runtime_access()
        from pypto.runtime.device_runner import compile_and_assemble  # noqa: PLC0415

        cc, rn, rc = compile_and_assemble(self._output_dir, self._platform)
        # Publish the "loaded" sentinel (_chip_callable) last so a reader can
        # never observe it set while _runtime_name / _runtime_config are None.
        self._runtime_name = rn
        self._runtime_config = rc
        self._chip_callable = cc

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

    @property
    def runtime_config(self) -> dict[str, Any]:
        """``RUNTIME_CONFIG`` dict from ``kernel_config.py`` (e.g. ``aicpu_thread_num``)."""
        self._ensure_runtime_loaded()
        assert self._runtime_config is not None
        return self._runtime_config


class CompiledProgram(_RuntimeFacade):
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
        program: Program | None,
        output_dir: str,
        *,
        backend_type: BackendType = BackendType.Ascend910B,
        platform: str | None = None,
        _param_infos: list[_ParamInfo] | None = None,
        _output_indices: list[int] | None = None,
        _return_types: list[Any] | None = None,
        _sub_chip_names: Sequence[str] | None = None,
    ) -> None:
        # ``program`` is ``None`` on the :meth:`from_dir` reload path: param
        # metadata is supplied pre-derived via the ``_param_infos`` /
        # ``_output_indices`` / ``_return_types`` kwargs (read back from
        # ``compiled_meta.json``), and the runtime artefacts are assembled from
        # the on-disk ``kernel_config.py`` -- so no live IR is needed.
        self._program = program
        self._output_dir = Path(output_dir).resolve()
        self._backend_type = backend_type
        self._platform = platform or _default_platform(backend_type)
        # Lazy metadata -- extracted on first call, or supplied by from_dir()
        self._param_infos = _param_infos
        self._output_indices = _output_indices
        self._return_types = _return_types

        # Lazy runtime artefacts -- compiled-and-assembled on first access
        # of chip_callable / runtime_name / runtime_config (or via load()).
        self._chip_callable: Any = None
        self._runtime_name: str | None = None
        self._runtime_config: dict[str, Any] | None = None

        # Multi-orch (L2-only) programs emit each Orchestration as a
        # self-contained sub-build under ``next_levels/<name>/``. Bind those
        # sub-dirs eagerly so ``__call__`` can error early and ``__getitem__``
        # / ``__getattr__`` can dispatch by orch name.
        self._sub_chip_dirs = self._resolve_sub_chip_dirs(_sub_chip_names)

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
                # Drop any parent sidecar left by an earlier single-orch compile
                # into this same directory -- ``ir.compile`` never clears
                # output_dir, and a stale parent signature would silently drive
                # the new artifacts with the old parameter ABI.
                _remove_meta(self._output_dir)
                self._persist_sub_chip_metadata()
            else:
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
        inside ``execute_compiled`` with a ``FileNotFoundError``. Distributed (L3+)
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

    def _write_meta(
        self,
        output_dir: Path,
        param_infos: list[_ParamInfo],
        return_types: list[Any],
    ) -> None:
        """Write one ``compiled_meta.json`` describing a single orchestration.

        Written atomically (temp file + :func:`os.replace`) so a reader never
        observes a half-written sidecar, and so a crash mid-write leaves the
        previous file intact rather than a truncated one.
        """
        meta = {
            "schema": _COMPILED_META_SCHEMA,
            "params": [_param_info_to_dict(p) for p in param_infos],
            "num_return_types": len(return_types),
            "platform": self._platform,
            "backend_type": self._backend_type.name,
        }
        target = output_dir / _COMPILED_META_FILENAME
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, target)

    def _persist_metadata(self) -> None:
        """Write ``<output_dir>/compiled_meta.json`` for :meth:`from_dir`.

        Captures exactly what the dispatch path reads from the IR -- the
        orchestration param metadata (names, directions, shapes, dtypes) plus
        the return-type count -- alongside the platform / backend. Runtime
        artefacts (``chip_callable`` / ``runtime_name`` / ``runtime_config``)
        need no persistence: ``compile_and_assemble`` already rederives them
        from the generated ``kernel_config.py``.

        Best-effort: a program without a resolvable orchestration signature
        emits nothing (mirrors :func:`_write_debug_runner`) *and* deletes any
        sidecar a previous compile into this directory left behind, so
        :meth:`from_dir` reports the missing file with a recompile hint instead
        of handing out a signature that no longer describes these artifacts.
        """
        try:
            param_infos, _, return_types = self._get_metadata()
        except (ValueError, TypeError):
            _remove_meta(self._output_dir)
            return
        self._write_meta(self._output_dir, param_infos, return_types)

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
                param_infos, _, return_types = _extract_func_param_infos(func)
            except (ValueError, TypeError):
                _remove_meta(sub_dir)
                continue
            self._write_meta(sub_dir, param_infos, return_types)

    @classmethod
    def from_dir(
        cls,
        output_dir: str | os.PathLike[str],
        *,
        platform: str | None = None,
        backend_type: BackendType | None = None,
    ) -> "CompiledProgram":
        """Reconstruct a single-chip program from an existing ``build_output/`` dir.

        Rebuilds param metadata from ``compiled_meta.json`` (written at compile
        time) so the program is callable **and benchmarkable** without re-running
        the pypto compile -- the L2 counterpart of
        :meth:`~pypto.ir.distributed_compiled_program.DistributedCompiledProgram.from_dir`
        and the basis of the ``runtime_dir`` replay workflow (edit the generated
        orchestration cpp / ``.pto``, then re-measure)::

            from pypto.ir import CompiledProgram
            from pypto.runtime import benchmark

            compiled = CompiledProgram.from_dir(work_dir, platform="a2a3")
            stats = benchmark(compiled, args, rounds=100)

        Args:
            output_dir: A build directory produced by a prior ``ir.compile`` of a
                single-orchestration (L2) program. Must contain
                ``compiled_meta.json``.
            platform: Override the persisted platform (e.g. swap ``a2a3sim`` ->
                ``a2a3`` to replay on hardware). ``None`` keeps the persisted
                value.
            backend_type: Override the persisted codegen backend. ``None`` keeps
                the persisted value.

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
        meta_path = Path(output_dir).resolve() / _COMPILED_META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — cannot reconstruct a compiled program from this "
                f"directory. It predates the single-chip replay feature, is a distributed "
                f"(L3+) build (use DistributedCompiledProgram.from_dir), or is the parent of "
                f"a multi-orch build (each next_levels/<name>/ sub-build has its own sidecar "
                f"— reload one of those). Recompile via ir.compile() to refresh."
            )
        meta = _load_compiled_meta(meta_path)
        param_infos = meta["param_infos"]
        output_indices = [i for i, p in enumerate(param_infos) if p.direction == ParamDirection.Out]
        # ``return_types`` contents are never inspected at runtime — only the
        # count matters (has_return = len(...) > 0), so placeholders suffice.
        return_types: list[Any] = [None] * meta["num_return_types"]
        bt = backend_type or meta["backend_type"]
        # A top-level sidecar is written only for a single-orch build, so it
        # settles the layout: dispatch through this object, not through a
        # ``next_levels/`` an earlier multi-orch compile may have left here.
        return cls(
            None,
            str(output_dir),
            backend_type=bt,
            platform=platform or meta.get("platform"),
            _param_infos=param_infos,
            _output_indices=output_indices,
            _return_types=return_types,
            _sub_chip_names=[],
        )

    # --- Properties -----------------------------------------------------------

    @property
    def output_dir(self) -> Path:
        """Path to compiled artifacts (kernels/, orchestration/, etc.)."""
        return self._output_dir

    @property
    def program(self) -> Program | None:
        """The original IR Program (pre-optimization passes).

        ``None`` for a program reconstructed via :meth:`from_dir` -- the IR is
        not persisted alongside the build artifacts.
        """
        return self._program

    @property
    def backend_type(self) -> BackendType:
        """Backend type used during compilation."""
        return self._backend_type

    @property
    def platform(self) -> str:
        """Target execution platform (e.g. ``"a2a3sim"``, ``"a5"``)."""
        return self._platform

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

    # --- Argument builders (for users driving a simpler.Worker directly) -----

    def build_orch_args(
        self,
        *args: "CallArg",
    ) -> tuple[Any, list[torch.Tensor | DeviceTensor | ctypes._SimpleCData], bool]:
        """Coerce user args and pack into a simpler ``ChipStorageTaskArgs``.

        Returns ``(orch_args, coerced, return_style)``:

        - ``orch_args``: simpler dispatch arg pack. Hand to
          ``Worker.run(cid, orch_args, cfg)``.
        - ``coerced``: full positional list of length ``len(param_infos)``.
          Scalar values are wrapped in their target ``ctypes`` type. For
          return-style callers, output ``torch.Tensor``s are auto-allocated
          and placed at :attr:`output_indices`; read those after dispatch
          to get the run's outputs.
        - ``return_style``: ``True`` if the caller passed only inputs.

        Raises:
            TypeError: Arg count / type mismatch, or called on a multi-orch
                program (use ``compiled[<name>].build_orch_args(...)`` instead).
        """
        if self._sub_chip_dirs:
            raise TypeError(
                f"Multi-orch program has {len(self._sub_chip_dirs)} orchestrations "
                f"{sorted(self._sub_chip_dirs)}; use compiled[<name>].build_orch_args(...)."
            )
        param_infos, output_indices, return_types = self._get_metadata()
        coerced, return_style = _coerce_args(
            args, param_infos, output_indices, return_types, caller_name="CompiledProgram"
        )
        from pypto.runtime.runner import _coerced_to_orch_args  # noqa: PLC0415

        orch_args = _coerced_to_orch_args(coerced)
        return orch_args, coerced, return_style

    def build_call_config(
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
                f"{sorted(self._sub_chip_dirs)}; use compiled[<name>].build_call_config(...)."
            )
        from pypto.runtime.runner import RunConfig, _build_call_config  # noqa: PLC0415

        run_config = config if config is not None else RunConfig()
        return _build_call_config(
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
            return str(self._output_dir) == str(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._output_dir)

    # --- Metadata (lazy) ------------------------------------------------------

    def _get_metadata(self) -> tuple[list[_ParamInfo], list[int], list[Any]]:
        """Return (param_infos, output_indices, return_types), extracting on first call."""
        if self._param_infos is None:
            if self._program is None:
                # Reload path with no pre-filled metadata — should not happen
                # (``from_dir`` always supplies it); guard rather than deref None.
                raise RuntimeError(
                    "CompiledProgram has neither live IR nor persisted param metadata; "
                    "reconstruct via CompiledProgram.from_dir()."
                )
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
        )


class _SubChipCallable(_RuntimeFacade):
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
        # Lazy runtime artefacts — mirror CompiledProgram.
        self._chip_callable: Any = None
        self._runtime_name: str | None = None
        self._runtime_config: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def platform(self) -> str:
        return self._platform

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self._param_infos]

    @property
    def output_indices(self) -> list[int]:
        return list(self._output_indices)

    def build_orch_args(
        self,
        *args: "CallArg",
    ) -> tuple[Any, list[torch.Tensor | DeviceTensor | ctypes._SimpleCData], bool]:
        coerced, return_style = _coerce_args(
            args,
            self._param_infos,
            self._output_indices,
            self._return_types,
            caller_name=f"orchestration {self._name!r}",
        )
        from pypto.runtime.runner import _coerced_to_orch_args  # noqa: PLC0415

        orch_args = _coerced_to_orch_args(coerced)
        return orch_args, coerced, return_style

    def build_call_config(
        self,
        config: Any = None,
        *,
        aicpu_thread_num: int | None = None,
        dfx_dir: "Path | None" = None,
    ) -> Any:
        from pypto.runtime.runner import RunConfig, _build_call_config  # noqa: PLC0415

        run_config = config if config is not None else RunConfig()
        return _build_call_config(
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
