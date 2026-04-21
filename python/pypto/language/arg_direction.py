# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-call-site direction markers for PyPTO Language DSL.

These wrappers attach an explicit :class:`pypto.ir.ArgDirection` to a call
argument. They are identity functions at Python runtime and are recognized
by the parser, which strips them and stores the direction vector on
``ir.Call.attrs['arg_directions']`` (also accessible via the
``ir.Call.arg_directions`` shortcut property).

Usage in a printed (or hand-written) DSL Orchestration function::

    import pypto.language as pl

    result = self.kernel_process(
        pl.adir.input(a),
        pl.adir.scalar(is_first),
        pl.adir.inout(result),
    )

The wrappers correspond 1:1 with the runtime task-submission methods on
``PTOParam`` (``add_input`` / ``add_output`` / ``add_inout`` /
``add_no_dep`` / ``add_scalar``) and with :class:`ir.ArgDirection` enum
values:

==================== ================================
Helper               ``ArgDirection``
==================== ================================
``pl.adir.input``            ``Input``
``pl.adir.output``           ``Output``
``pl.adir.output_existing``  ``OutputExisting``
``pl.adir.inout``            ``InOut``
``pl.adir.no_dep``           ``NoDep``
``pl.adir.scalar``           ``Scalar``
==================== ================================

The wrappers exist so that ``Call.attrs['arg_directions']`` survives a
``python_print`` → ``parse`` round-trip (it is otherwise a derived attr
populated by the ``DeriveCallDirections`` pass and not visible in the
DSL surface syntax).

User-facing parameter direction is still expressed via ``pl.Out[T]`` /
``pl.InOut[T]`` on the *callee*'s parameter list — those map to
``ir.ParamDirection`` and remain the recommended way to declare a
function's contract.
"""

from __future__ import annotations

from typing import TypeVar

from pypto.pypto_core import ir as _ir

T = TypeVar("T")

ArgDirection = _ir.ArgDirection


def input(x: T) -> T:  # noqa: A001 -- shadows builtin within this module only
    """Mark a call argument as an ``Input`` (read-only tensor) at the call site."""
    return x


def output(x: T) -> T:
    """Mark a call argument as an ``Output`` (freshly allocated, single writer)."""
    return x


def output_existing(x: T) -> T:
    """Mark a call argument as an ``OutputExisting`` (writes into an externally provided buffer)."""
    return x


def inout(x: T) -> T:
    """Mark a call argument as ``InOut`` (read-modify-write, e.g. WAW promotion)."""
    return x


def no_dep(x: T) -> T:
    """Mark a call argument as ``NoDep`` (explicitly opt out of the dataflow)."""
    return x


def scalar(x: T) -> T:
    """Mark a call argument as a ``Scalar`` (passed by value, not as a tensor handle)."""
    return x


# Mapping from the wrapper's leaf attribute name to the IR enum value.
# Used by both the printer (enum → name) and parser (name → enum).
NAME_TO_DIRECTION: dict[str, ArgDirection] = {
    "input": ArgDirection.Input,
    "output": ArgDirection.Output,
    "output_existing": ArgDirection.OutputExisting,
    "inout": ArgDirection.InOut,
    "no_dep": ArgDirection.NoDep,
    "scalar": ArgDirection.Scalar,
}

DIRECTION_TO_NAME: dict[ArgDirection, str] = {v: k for k, v in NAME_TO_DIRECTION.items()}


__all__ = [
    "ArgDirection",
    "DIRECTION_TO_NAME",
    "NAME_TO_DIRECTION",
    "inout",
    "input",
    "no_dep",
    "output",
    "output_existing",
    "scalar",
]
