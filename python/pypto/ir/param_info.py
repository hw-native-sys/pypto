# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration parameter metadata, and the IR-dtype to torch-dtype map.

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

from dataclasses import dataclass

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


# Public spelling for code outside ``pypto.ir`` (the replay-script writer, and
# harnesses that bind arguments themselves).
ParamInfo = _ParamInfo
