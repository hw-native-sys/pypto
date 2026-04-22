# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-call-site direction markers for PyPTO Language DSL.

These markers attach an explicit :class:`pypto.ir.ArgDirection` to a call
argument and are stored on ``ir.Call.attrs['arg_directions']`` (also
accessible via the ``ir.Call.arg_directions`` shortcut property).

The printer surfaces the direction vector via a trailing ``attrs=`` keyword
on cross-function calls so that ``python_print`` → ``parse`` round-trips
preserve the metadata::

    import pypto.language as pl

    result = self.kernel_process(
        a, is_first, result,
        attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.inout]},
    )

Each ``pl.adir.<name>`` symbol is a direct alias of the matching
:class:`ir.ArgDirection` enum value — they are not callable. Per-argument
wrapper forms such as ``pl.adir.input(x)`` are intentionally not supported.

The markers correspond 1:1 with the runtime task-submission methods on
``PTOParam`` (``add_input`` / ``add_output`` / ``add_inout`` /
``add_no_dep`` / ``add_scalar``) and with :class:`ir.ArgDirection` enum
values:

==================== ================================
Marker               ``ArgDirection``
==================== ================================
``pl.adir.input``            ``Input``
``pl.adir.output``           ``Output``
``pl.adir.output_existing``  ``OutputExisting``
``pl.adir.inout``            ``InOut``
``pl.adir.no_dep``           ``NoDep``
``pl.adir.scalar``           ``Scalar``
==================== ================================

User-facing parameter direction is still expressed via ``pl.Out[T]`` /
``pl.InOut[T]`` on the *callee*'s parameter list — those map to
``ir.ParamDirection`` and remain the recommended way to declare a
function's contract.
"""

from __future__ import annotations

from pypto.pypto_core import ir as _ir

ArgDirection = _ir.ArgDirection

# Direct aliases of the enum values. Bare references like ``pl.adir.input`` are
# the only supported surface form; ``pl.adir.input(x)`` would be a TypeError
# at runtime since enum values are not callable.
input = ArgDirection.Input  # noqa: A001 -- shadows builtin within this module only
output = ArgDirection.Output
output_existing = ArgDirection.OutputExisting
inout = ArgDirection.InOut
no_dep = ArgDirection.NoDep
scalar = ArgDirection.Scalar

# Mapping from the marker's leaf attribute name to the IR enum value.
# Used by both the printer (enum -> name) and parser (name -> enum).
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
