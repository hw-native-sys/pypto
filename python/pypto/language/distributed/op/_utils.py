# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared helpers for ``pld.*`` DSL wrappers."""

from collections.abc import Sequence
from typing import Any

from pypto.language.typing import IntLike, Scalar
from pypto.pypto_core import ir as _ir
from pypto.pypto_core.ir import Expr


def _unwrap(value: Any) -> Any:
    """Unwrap a DSL wrapper (Tensor / Tile / Scalar / Ptr / CommCtx / ...) to ``ir.Expr``.

    Falls through unchanged for raw ``ir.Expr`` and primitive ``int`` /
    ``float`` values (which downstream IR builders normalise to ``ConstInt`` /
    ``ConstFloat``).
    """
    if hasattr(value, "unwrap"):
        return value.unwrap()
    return value


def _normalize_intlike(seq: Sequence[IntLike]) -> list[int | Expr]:
    """Unwrap Scalar elements to Expr so the sequence matches C++ binding types."""
    return [elem.unwrap() if isinstance(elem, Scalar) else elem for elem in seq]


def _unwrap_distributed_tensors(op_name: str, **named: Any) -> tuple[Expr, ...]:
    """Unwrap and validate DistributedTensor positional args for a ``pld.*`` op.

    Each kwarg is one positional argument by name (``dst=...``, ``src=...``,
    ``target=...``). Returns the unwrapped ``ir.Expr`` instances in the same
    order. Raises :class:`TypeError` with an op-named, role-tagged diagnostic
    when an argument is not a window-bound :class:`DistributedTensor`.

    This is the shared validation path for ops whose operands are *all*
    window-bound :class:`DistributedTensor`s — ``pld.tensor.get`` /
    ``pld.tensor.allreduce`` (and future tensor-level collectives). Ops with
    an asymmetric signature (e.g. ``pld.tensor.put``, whose ``src`` may be a
    plain :class:`pl.Tensor`) validate inline instead.
    """
    exprs: list[Expr] = []
    for role, value in named.items():
        expr = _unwrap(value)
        if not isinstance(expr, Expr) or not isinstance(expr.type, _ir.DistributedTensorType):
            got = _ir.python_print_type(expr.type) if isinstance(expr, Expr) else type(expr).__name__
            raise TypeError(f"{op_name} expects a DistributedTensor {role} (window-bound); got {got}")
        exprs.append(expr)
    return tuple(exprs)


__all__ = ["_normalize_intlike", "_unwrap", "_unwrap_distributed_tensors"]
