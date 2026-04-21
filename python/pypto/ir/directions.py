# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Friendly module-level helpers for building :class:`ir.Call` arg_directions vectors.

These helpers exist for tests, examples and hand-written IR code that need to
attach :class:`ir.ArgDirection` values to a call without importing the enum
explicitly. They mirror the runtime ``add_input/add_output/add_output(Tensor)
/add_inout/add_no_dep/add_scalar`` API on the call site.

The user-facing DSL (``pypto.language``) keeps using :class:`pl.Out` and
:class:`pl.InOut` on the *callee* parameter list — those map to
:class:`ir.ParamDirection`. The helpers here are about the *call* site and
populate :attr:`ir.Call.arg_directions`. A :class:`Pass` (``DeriveCallDirections``)
can recompute the same vector from the callee's :class:`ir.ParamDirection` and
buffer lineage, so explicit usage is reserved for hand-built IR fragments.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pypto.pypto_core import ir as _ir

ArgDirection = _ir.ArgDirection

# Per-call-site direction aliases (lowercase, named after the runtime methods).
# ``input`` intentionally shadows the Python builtin within this module; users
# access it via ``ir.directions.input`` or ``ir.input``, never as a free name.
input = ArgDirection.Input
output = ArgDirection.Output
output_existing = ArgDirection.OutputExisting
inout = ArgDirection.InOut
no_dep = ArgDirection.NoDep
scalar = ArgDirection.Scalar


def make_call(
    op: _ir.Op,
    args: Sequence[_ir.Expr],
    *,
    directions: Sequence[ArgDirection] | None = None,
    kwargs: dict[str, Any] | None = None,
    type: _ir.Type | None = None,
    span: _ir.Span | None = None,
) -> _ir.Call:
    """Build a :class:`ir.Call` while populating ``arg_directions`` in one place.

    Args:
        op: Callee operation (typically an :class:`ir.GlobalVar` for a function).
        args: Positional argument expressions.
        directions: Optional explicit per-argument :class:`ArgDirection` vector.
            When ``None`` the resulting call has an empty ``arg_directions``
            vector (legacy / pre-DeriveCallDirections state). Run
            ``DeriveCallDirections`` before consumers that require resolved
            call-site directions. When provided the length must match ``args``.
        kwargs: Optional keyword arguments to attach to the call.
        type: Optional explicit return type.
        span: Optional source span; defaults to :meth:`ir.Span.unknown`.

    Returns:
        The constructed :class:`ir.Call`.

    Raises:
        ValueError: If ``directions`` has a length other than ``len(args)``.
    """
    actual_span = span if span is not None else _ir.Span.unknown()
    actual_type = type if type is not None else _ir.UnknownType()
    actual_kwargs = dict(kwargs or {})
    args_list = list(args)

    if directions is None:
        return _ir.Call(op, args_list, actual_kwargs, actual_type, actual_span)

    directions_list = list(directions)
    if len(directions_list) != len(args_list):
        raise ValueError(
            f"make_call: directions length ({len(directions_list)}) must match args length ({len(args_list)})"
        )
    attrs: dict[str, Any] = {"arg_directions": directions_list}
    return _ir.Call(op, args_list, actual_kwargs, attrs, actual_type, actual_span)


__all__ = [
    "ArgDirection",
    "input",
    "output",
    "output_existing",
    "inout",
    "no_dep",
    "scalar",
    "make_call",
]
