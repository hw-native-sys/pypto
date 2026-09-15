# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Internal dispatcher schema and Fake/Meta helpers; no operators register on import.

Definitions belong to a caller-owned torch.library.Library. Device kernels and
public kernel-mode registration are deliberately supplied by later integration.
"""

import ctypes
import keyword
import re
from collections.abc import Sequence
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor

from pypto.ir.param_info import _DATATYPE_TO_CTYPE, ParamInfo, _to_torch_dtype, bind_complete_args
from pypto.pypto_core.ir import ParamDirection

from .interop import _scalar_value

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _check_name(name: str) -> None:
    """Accept plain dispatcher identifiers without schema punctuation or keywords."""
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or keyword.iskeyword(name):
        raise ValueError(f"Expected a plain operator or parameter identifier, got {name!r}")


def _scalar_schema(info: ParamInfo) -> str:
    """Represent supported scalar types without narrowing the dispatcher integer range."""
    dtype = str(info.dtype)
    if dtype == "bool":
        return "bool"
    if dtype in ("fp16", "fp32", "fp64", "bfloat16"):
        return "float"
    if dtype in _DATATYPE_TO_CTYPE and dtype != "uint64":
        return "int"
    raise TypeError(f"Parameter {info.name!r} has no supported dispatcher scalar type for {info.dtype}")


def _check_scalar(value: Any, info: ParamInfo) -> None:
    """Validate abstract scalar values without converting symbolic integers to Python."""
    kind = _scalar_schema(info)
    symbolic_type = {"int": torch.SymInt, "float": torch.SymFloat, "bool": torch.SymBool}[kind]
    if isinstance(value, symbolic_type):
        if kind == "int":
            ctype = _DATATYPE_TO_CTYPE[str(info.dtype)]
            bits = ctypes.sizeof(ctype) * 8
            signed = ctype(-1).value < 0
            minimum, maximum = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
            torch._check(value >= minimum, lambda: f"Parameter {info.name!r} is below {info.dtype} range")
            torch._check(value <= maximum, lambda: f"Parameter {info.name!r} exceeds {info.dtype} range")
        return
    if type(value) not in (int, float, bool):
        raise TypeError(f"Parameter {info.name!r} expects a {kind} scalar, got {type(value).__name__}")
    _scalar_value(value, info)


class RegistrationSignature:
    """Describe a borrowed-output operator using the shared ParamInfo contract.

    Tensor returns must alias complete Out/InOut slots. Scalar outputs, implicit
    output allocation, scalar-only dispatcher operators and UINT64 scalars are
    unsupported. Construction and import have no registration side effects.
    """

    def __init__(self, params: Sequence[ParamInfo], *, return_aliases: Sequence[int] = ()) -> None:
        """Copy the signature and reject contracts the helper cannot express."""
        self._params = tuple(
            ParamInfo(p.name, p.direction, list(p.shape) if p.shape is not None else None, p.dtype)
            for p in params
        )
        self._return_aliases = tuple(return_aliases)
        names: set[str] = set()
        tensor_count = 0
        for info in self._params:
            _check_name(info.name)
            if info.name in names:
                raise ValueError(f"Duplicate parameter name {info.name!r}")
            names.add(info.name)
            if info.direction not in (ParamDirection.In, ParamDirection.Out, ParamDirection.InOut):
                raise ValueError(f"Parameter {info.name!r} has unsupported direction {info.direction}")
            if info.shape is None:
                if info.direction != ParamDirection.In:
                    raise ValueError(f"Scalar parameter {info.name!r} must have direction In")
                _scalar_schema(info)
            else:
                tensor_count += 1
                if _to_torch_dtype(info.dtype) is None:
                    raise TypeError(f"Parameter {info.name!r} has unsupported tensor dtype {info.dtype}")
                if any(type(dim) is not int or dim < -1 for dim in info.shape):
                    raise ValueError(f"Parameter {info.name!r} requires nonnegative dimensions or -1")
        if not tensor_count:
            raise ValueError("Dispatcher metadata registration requires at least one tensor parameter")
        for index in self._return_aliases:
            if (
                type(index) is not int
                or not 0 <= index < len(self._params)
                or self._params[index].shape is None
            ):
                raise ValueError(f"Return alias must reference a tensor parameter, got {index!r}")
            if self._params[index].direction == ParamDirection.In:
                raise ValueError(
                    "Return aliases require Out/InOut tensors; read-only identity returns are unsupported"
                )

    def schema(self, name: str) -> str:
        """Build a schema with explicit mutation and exact input-to-output aliases."""
        _check_name(name)
        types = []
        for index, info in enumerate(self._params):
            if info.shape is None:
                types.append(_scalar_schema(info))
            else:
                mutation = "!" if info.direction != ParamDirection.In else ""
                types.append(f"Tensor(a{index}{mutation})")
        arguments = ", ".join(f"{kind} {info.name}" for kind, info in zip(types, self._params, strict=True))
        returns = [types[index] for index in self._return_aliases]
        result = returns[0] if len(returns) == 1 else f"({', '.join(returns)})"
        return f"{name}({arguments}) -> {result}"

    def fake(self, *args: Any) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
        """Check Fake/Meta arguments and return the declared original alias objects.

        Never read pointers, storage, values, physical NPU formats or framework
        device/stream state. Symbolic dimensions stay symbolic. Physical layout
        and storage overlap remain real-call validation responsibilities.
        """
        bound = bind_complete_args(args, self._params, caller_name="Fake/Meta call")
        device = None
        for arg, info in zip(bound, self._params, strict=True):
            if info.shape is None:
                _check_scalar(arg, info)
                continue
            if not isinstance(arg, torch.Tensor) or not (isinstance(arg, FakeTensor) or arg.is_meta):
                raise TypeError(f"Parameter {info.name!r} requires a FakeTensor or Meta tensor")
            if device is not None and arg.device != device:
                raise ValueError(f"Parameter {info.name!r} has device {arg.device}, expected {device}")
            device = arg.device
            dtype = _to_torch_dtype(info.dtype)
            if arg.dtype != dtype:
                raise TypeError(f"Parameter {info.name!r} expects dtype {dtype}, got {arg.dtype}")
            if arg.layout != torch.strided or not arg.is_contiguous():
                raise ValueError(f"Parameter {info.name!r} requires a contiguous strided tensor")
            if arg.is_conj() or arg.is_neg():
                raise ValueError(f"Parameter {info.name!r} has an unresolved conjugate or negative view")
            if arg.requires_grad and torch.is_grad_enabled():
                raise ValueError(
                    f"Parameter {info.name!r} requires gradients; no autograd contract is supplied"
                )
            if arg.ndim != len(info.shape):
                raise ValueError(f"Parameter {info.name!r} expects rank {len(info.shape)}, got {arg.ndim}")
            for expected, actual in zip(info.shape, arg.shape, strict=True):
                if expected != -1:
                    torch._check(
                        actual == expected, lambda: f"Parameter {info.name!r} has incompatible shape"
                    )
        result = tuple(bound[index] for index in self._return_aliases)
        return result[0] if len(result) == 1 else result or None

    def define(self, library: torch.library.Library, name: str) -> None:
        """Define metadata in a caller-owned library, failing on duplicate names.

        The library owns definition and fake-kernel lifetimes. This method does
        not install a CPU/NPU kernel, a functionalization rule or autograd.
        Aliased-return schemas alone do not imply torch.compile support.
        """
        register_fake = getattr(torch.library, "register_fake", None)
        if register_fake is None:
            raise RuntimeError("Metadata registration requires torch.library.register_fake support")
        schema = self.schema(name)
        library.define(schema)
        register_fake(f"{library.ns}::{name}", self.fake, lib=library)
