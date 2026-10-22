# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration parameter metadata and shared torch/ctypes dtype maps.

A leaf: it imports nothing from ``pypto.runtime`` and nothing from the rest of
``pypto.ir`` beyond the core bindings, so anything may depend on it.

That is the point. This metadata used to live in ``compiled_program``, which
made a consumer of it — ``pypto.runtime.debug.run_script_writer``, which
renders a replay script from a program's parameters — import back into the
module that reaches forward into ``pypto.runtime``. That is a genuine cycle:
hoisting ``run_script_writer``'s import to module scope fails with
``cannot import name 'ParamInfo' from partially initialized module``. Splitting
the metadata out removes the back edge; ``compiled_program`` re-exports these
names, so nothing else has to know they moved.
"""

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeGuard, TypeVar, cast

import torch

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import (
    ConstInt,
    Function,
    FunctionType,
    ParamDirection,
    Program,
    ScalarType,
    ShapedType,
    TensorLayout,
)

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
    # Torch has no E1M2 dtype; annotated kernels exchange one byte per x2 carrier.
    "fp4e1m2x2": torch.uint8,
    "fp4e2m1x2": torch.uint8,
    "bool": torch.bool,
    "index": torch.int64,
}
# uint16/32/64 were added in PyTorch 2.3; register only if available
_name: str = ""
_torch_dtype: torch.dtype | None = None
for _name in ("uint16", "uint32", "uint64"):
    _torch_dtype = getattr(torch, _name, None)
    if _torch_dtype is not None:
        _DATATYPE_TO_TORCH[_name] = _torch_dtype
del _name, _torch_dtype
# Float8 / MX scale dtypes (PyTorch 2.1+ / 2.3+ / 2.7+); map IR string → torch.dtype.
# Prefer Torch's native E2M1 x2 carrier when available; E1M2 uses raw UINT8.
# Both formats retain the IR carrier shape during return-style allocation.
_ir_name: str = ""
_torch_name: str = ""
_torch_dtype: torch.dtype | None = None
for _ir_name, _torch_name in (
    ("fp8e4m3fn", "float8_e4m3fn"),
    ("fp8e5m2", "float8_e5m2"),
    ("fp8e8m0", "float8_e8m0fnu"),
    ("fp4e2m1x2", "float4_e2m1fn_x2"),
):
    _torch_dtype = getattr(torch, _torch_name, None)
    if _torch_dtype is not None:
        _DATATYPE_TO_TORCH[_ir_name] = _torch_dtype
del _ir_name, _torch_name, _torch_dtype


# IR DataType -> ctypes scalar constructor mapping.
# Used to wrap Python int/float/bool values into the correct ctypes scalar
# when calling a compiled program with scalar parameters.
_DATATYPE_TO_CTYPE: "dict[str, type[ctypes._SimpleCData[Any]]]" = {
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


def _matches_torch_dtype(dtype: DataType, actual: torch.dtype) -> bool:
    """Accept the declared dtype's native Torch representation or packed byte carrier."""
    return actual == _to_torch_dtype(dtype) or (
        dtype in (DataType.FP4E2M1X2, DataType.FP4E1M2X2) and actual == torch.uint8
    )


@dataclass(frozen=True)
class _ParamInfo:
    """Metadata for a single orchestration function parameter."""

    name: str
    direction: ParamDirection
    shape: Sequence[int] | None  # Normalized to tuple; None for scalar params
    dtype: DataType
    # Layout describes either logical or lowered metadata; NZ/MX need not
    # already have their blocked rank-5 shape here.
    layout: str | None = None

    def __post_init__(self) -> None:
        # IR construction and passes own parameter legality. Metadata only
        # snapshots the shape, including L3 axes removed before L2 submission.
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(self.shape))


# Public spelling for code outside ``pypto.ir`` (the replay-script writer, and
# harnesses that bind arguments themselves).
ParamInfo = _ParamInfo

# pto-isa's NZ blocking: c0 elements per 32-byte C0 line, 16 rows per fractal.
_NZ_C0_BYTES = 32
_NZ_FRACTAL_ROWS = 16


def block_nz_shape(shape: Sequence[int], dtype: torch.dtype) -> list[int]:
    """The blocked rank-5 shape an NZ parameter of logical *shape* is compiled to.

    ``[..., R, C]`` becomes ``[prod(lead), C/c0, R/16, 16, c0]``: every leading
    axis folds into the single batch slot pto-isa declares, and the trailing
    matrix expands into the fractal plane. This mirrors ``BlockNzShape`` in
    ``tensor_view_semantics.h`` — a caller allocates the logical shape, the
    compiled parameter names the blocked one, and the two describe the same bytes.
    """
    if len(shape) < 2:
        raise ValueError(f"an NZ shape needs a trailing [R, C] pair, got {tuple(shape)}")
    rows, cols = shape[-2], shape[-1]
    c0 = _NZ_C0_BYTES // torch.empty((), dtype=dtype).element_size()
    if rows % _NZ_FRACTAL_ROWS or cols % c0:
        raise ValueError(
            f"an NZ shape needs {_NZ_FRACTAL_ROWS}-row fractals and whole C0 lines of {c0} "
            f"elements, got {tuple(shape)}"
        )
    batch = 1
    for dim in shape[:-2]:
        batch *= dim
    return [batch, cols // c0, rows // _NZ_FRACTAL_ROWS, _NZ_FRACTAL_ROWS, c0]


def arg_shape_as_declared(arg_shape: Sequence[int], info: _ParamInfo) -> list[int]:
    """*arg_shape* in the parameter's own terms, ready to compare with ``info.shape``.

    An NZ parameter is compiled to the blocked rank-5 shape the backend
    addresses, while its caller allocates and passes the logical one — the same
    bytes under two spellings. Comparing them means blocking the caller's shape
    with the rule the compiler used. Every other parameter is already in its
    caller's terms, and a shape that does not block at all (an odd row count,
    say) is left alone so it is reported as the mismatch it is.
    """
    if info.layout != "NZ":
        return list(arg_shape)
    # A dynamic blocked batch can only come from logical [B, R, C]: the NZ
    # pass rejects dynamic leading extents at rank > 3. Entry code reads B
    # directly from the incoming shapes[0], so accepting another logical rank
    # here would disagree with that lookup even if blocking hides the mismatch.
    if info.shape and info.shape[0] < 0 and len(arg_shape) != 3:
        raise TypeError(
            f"Parameter {info.name!r} with a dynamic NZ batch expects logical rank 3 "
            f"(shape [B, R, C]); got shape {tuple(arg_shape)}"
        )
    expected_dtype = _to_torch_dtype(info.dtype)
    if expected_dtype is None:
        return list(arg_shape)
    try:
        return block_nz_shape(arg_shape, expected_dtype)
    except ValueError:
        return list(arg_shape)


_Arg = TypeVar("_Arg")


def bind_complete_args(
    args: Sequence[_Arg], param_infos: Sequence[_ParamInfo], *, caller_name: str
) -> list[_Arg]:
    """Bind every positional parameter without allocating outputs or converting values.

    Out and InOut slots are mandatory, even when the IR has return values.
    Tensor aliases and runtime scalar values are preserved. Consumer-specific
    dtype, shape, storage and ABI validation happens after this shared binding.
    """
    if len(args) != len(param_infos):
        raise TypeError(
            f"{caller_name} expects {len(param_infos)} arguments including all Out/InOut parameters, "
            f"got {len(args)}. Parameters: {[p.name for p in param_infos]}"
        )
    return list(args)


# ``_to_torch_dtype`` is shared (leaf-imported) by the compiled-program tiers;
# the listing keeps pyright's unused-function check satisfied.
__all__ = [
    "_matches_torch_dtype",
    "_to_torch_dtype",
    "_param_info_to_dict",
    "_param_info_from_dict",
    "_extract_func_param_infos",
    "_extract_param_infos",
]


# Reverse map ``str(DataType) -> DataType`` for JSON round-tripping (e.g. the L3
# ``distributed_meta.json`` consumed by ``DistributedCompiledProgram.from_dir``).
# ``DataType`` is a non-singleton nanobind type, so reconstruction goes through
# the canonical string (``str(DataType.FP32) == "fp32"``) rather than identity.
# Built from the named constants exposed on ``DataType`` so it tracks the C++
# enum without a hand-maintained literal table.
_STR_TO_DATATYPE: dict[str, DataType] = {}
_dt: DataType | None = None
_dt_name: str = ""
for _dt_name in (
    "BOOL", "INT4", "INT8", "INT16", "INT32", "INT64", "UINT4", "UINT8", "UINT16",
    "UINT32", "UINT64", "FP4E2M1X2", "FP4E1M2X2", "FP8E4M3FN", "FP8E5M2", "FP8E8M0",
    "FP16", "FP32", "BF16", "HF4", "HF8", "INDEX", "TASK_ID",
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
        "shape": list(info.shape) if info.shape is not None else None,
        "dtype": str(info.dtype),
        "layout": info.layout,
    }


def _is_int_list(v: object) -> TypeGuard[list[int]]:
    """True for a JSON array of non-bool ints (``bool`` is an ``int`` subclass)."""
    return isinstance(v, list) and all(
        isinstance(x, int) and not isinstance(x, bool) for x in cast(list[object], v)
    )


def _param_info_from_dict(d: dict[str, object]) -> _ParamInfo:
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
    if not isinstance(name, str) or not name:
        raise ValueError(f"'name' must be a nonempty string, got {name!r}")

    # Strict enum-member lookup, not getattr(): ``getattr(ParamDirection, "mro")``
    # resolves to a bound method and would sail through as a bogus direction.
    raw_direction = d["direction"]
    direction = ParamDirection.__members__.get(raw_direction) if isinstance(raw_direction, str) else None
    if direction is None:
        raise ValueError(
            f"'direction' must be one of {sorted(ParamDirection.__members__)}, got {raw_direction!r}"
        )

    # ``bool`` is an ``int`` subclass; a JSON ``true`` is never a dimension.
    raw_shape: object = d["shape"]
    shape: list[int] | None = None
    if raw_shape is not None:
        if _is_int_list(raw_shape):
            shape = raw_shape
        else:
            raise ValueError(f"'shape' must be null or a list of ints, got {raw_shape!r}")

    raw_dtype = d["dtype"]
    if not isinstance(raw_dtype, str):
        raise ValueError(f"'dtype' must be a string, got {raw_dtype!r}")

    # A sidecar written before parameters carried a layout says nothing about
    # one, which is exactly what ND means.
    raw_layout = d.get("layout")
    if raw_layout is not None and (
        not isinstance(raw_layout, str) or raw_layout not in TensorLayout.__members__
    ):
        raise ValueError(f"'layout' must be null or a TensorLayout name, got {raw_layout!r}")

    return _ParamInfo(
        name=name,
        direction=direction,
        shape=shape,
        dtype=_datatype_from_string(raw_dtype),
        layout=raw_layout,
    )


def validate_parameter_names(parameters: Sequence[ParamInfo]) -> None:
    """Reject ambiguous binding without imposing an L2 limit on L3 signatures."""
    names: set[str] = set()
    for param in parameters:
        if param.name in names:
            raise ValueError(f"Duplicate parameter name {param.name!r}; names must be unique")
        names.add(param.name)


def validate_return_aliases(parameters: Sequence[ParamInfo], aliases: Sequence[int]) -> None:
    """Validate full-signature tensor indices, preserving order and duplicates."""
    for index in aliases:
        if type(index) is not int or not 0 <= index < len(parameters) or parameters[index].shape is None:
            raise ValueError(f"Return alias must reference a tensor parameter, got {index!r}")


MAX_KERNEL_TENSORS = 256
MAX_KERNEL_SCALARS = 128


def validate_l2_capacity(parameters: Sequence[ParamInfo]) -> None:
    """Check ChipStorageTaskArgs pools only at the single-chip boundary."""
    tensors = sum(p.shape is not None for p in parameters)
    if tensors > MAX_KERNEL_TENSORS or len(parameters) - tensors > MAX_KERNEL_SCALARS:
        raise ValueError("Signature exceeds L2 argument capacity")


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

        layout: str | None = None

        if isinstance(param_type, ShapedType):
            dtype = param_type.dtype
            shape = [dim.value if isinstance(dim, ConstInt) else -1 for dim in param_type.shape]
            view = getattr(param_type, "tensor_view", None)
            layout = view.layout.name if view is not None else "ND"
        elif isinstance(param_type, ScalarType):
            dtype = param_type.dtype
        else:
            raise TypeError(
                f"Unsupported parameter type for {param.name_hint!r}: {type(param_type).__name__}. "
                f"Expected ShapedType or ScalarType."
            )

        param_infos.append(
            _ParamInfo(name=param.name_hint, direction=direction, shape=shape, dtype=dtype, layout=layout)
        )

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
