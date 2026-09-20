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
from typing import TypeVar

import torch

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import ParamDirection

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


@dataclass
class _ParamInfo:
    """Metadata for a single orchestration function parameter."""

    name: str
    direction: ParamDirection
    shape: list[int] | None  # None for scalar params
    dtype: DataType
    # ``TensorLayout`` name ("ND", "NZ", ...) for a tensor param, None for a
    # scalar. An NZ param's ``shape`` is the *blocked* rank-5 one the backend
    # addresses, not the logical shape its caller allocates, so a consumer
    # comparing shapes has to block the caller's shape first (``block_nz_shape``).
    layout: str | None = None


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
