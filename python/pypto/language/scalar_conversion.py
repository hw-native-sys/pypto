# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The declared-slot scalar conversion: admission rule and canonical naming.

A declared scalar dtype is the type of its slot, so a staged integer value whose
dtype differs from it is converted to the declared dtype. Both frontends ask this
module the same question — *is this pair convertible, and to what* — so the two
lanes cannot diverge on admission, on the produced dtype, or on the name of the
temporary that carries the conversion.

The admitted lattice is exactly the one the language spec grants a declared slot
(`docs/en/user/language/00-types.md`, "Declared scalar dtypes convert"): one end
is ``pl.INDEX``, the other a fixed-width integer. That is narrower than "any two
integers" on purpose — the language states no signedness or width policy between
two fixed-width types, so admitting those pairs would invent one. Everything
outside the lattice keeps whatever refusal its own layer already produces; this
module never widens a refusal into an acceptance.

Values convert by wrapping, not by saturating: the conversion happens where a
value is *placed*, not where a numerical operation was requested, and there is no
mode to state at that position. The wrapping table itself lives in the user
contract (`docs/en/user/language/00-types.md`, "What converts"); this module only
decides which dtype the conversion produces.

Why the conversion is bound to a temporary instead of being written inline as a
call argument: the device task-argument path reads a variable or a constant
literal, and one `let` keeps the emitted IR in the shape both the backend and the
printed round-trip already accept. :func:`conversion_temp_name` is the single
source of that temporary's name.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pypto.pypto_core import DataType, ir

__all__ = [
    "CONVERT",
    "REJECT",
    "ArgumentDecision",
    "conversion_target",
    "conversion_temp_name",
    "classify_scalar_conversion",
]

# Fixed-width integer widths that exist as *scalar* values. Four-bit dtypes are
# packed byte carriers in this IR (``pl.INT4`` / ``pl.UINT4`` are storage
# formats, and both backends reject them end-to-end), so a 4-bit scalar slot has
# no register to occupy. ``TASK_ID`` is an opaque handle and ``BOOL`` is a
# logical value: ``pl.cast`` spells the intended 0/1 mapping for the latter, and
# neither is an integer of some width.
_ADMITTED_INTEGER_BITS = frozenset({8, 16, 32, 64})


class ArgumentDisposition(Enum):
    """What the declaration says about one call-site argument."""

    ACCEPT = "accept"
    """The argument already is the declared type: nothing to emit."""

    CONVERT = "convert"
    """The argument is an admitted integer pair: convert to ``target``."""

    REJECT = "reject"
    """Outside the lattice: the layer's own refusal stands."""


CONVERT = ArgumentDisposition.CONVERT
REJECT = ArgumentDisposition.REJECT


@dataclass(frozen=True)
class ArgumentDecision:
    """One argument's verdict against its declared parameter type."""

    disposition: ArgumentDisposition
    target: DataType | None = None
    """The dtype the conversion must produce (``CONVERT`` only)."""
    reason: str | None = None
    """Why the pair is refused (``REJECT`` only), in the checker's wording."""


def _is_admitted_integer(dtype: DataType) -> bool:
    """Whether ``dtype`` is a fixed-width integer that exists as a scalar."""
    return dtype.is_int() and dtype.get_bit() in _ADMITTED_INTEGER_BITS


def conversion_target(declared: ir.Type, actual: ir.Type) -> DataType | None:
    """The dtype a conversion must produce, or ``None`` when no conversion applies.

    ``None`` covers both "already the declared type" and "not admitted"; callers
    distinguish them by comparing the dtypes themselves, because the two need
    different diagnostics.
    """
    if not isinstance(declared, ir.ScalarType) or not isinstance(actual, ir.ScalarType):
        return None
    declared_dtype = declared.dtype
    actual_dtype = actual.dtype
    if declared_dtype == actual_dtype:
        return None
    # Exactly one end is INDEX. INDEX -> INDEX and fixed-width -> fixed-width are
    # both outside the lattice: the first is an identity, the second is the
    # signedness/width policy the language does not state.
    if (declared_dtype == DataType.INDEX) == (actual_dtype == DataType.INDEX):
        return None
    other = actual_dtype if declared_dtype == DataType.INDEX else declared_dtype
    if not _is_admitted_integer(other):
        return None
    return declared_dtype


def classify_scalar_conversion(declared: ir.Type, actual: ir.Type) -> ArgumentDecision:
    """Decide one scalar argument against its declared parameter type.

    ``reason`` is filled for a refused dtype pair so the caller can raise in its
    own diagnostic shape without re-deriving the wording; the caller still owns
    every non-dtype mismatch (shape, layout, memory space, kind).
    """
    if not isinstance(declared, ir.ScalarType) or not isinstance(actual, ir.ScalarType):
        return ArgumentDecision(ArgumentDisposition.REJECT, reason="a different type kind")
    target = conversion_target(declared, actual)
    if target is not None:
        return ArgumentDecision(ArgumentDisposition.CONVERT, target=target)
    if declared.dtype == actual.dtype:
        return ArgumentDecision(ArgumentDisposition.ACCEPT)
    return ArgumentDecision(
        ArgumentDisposition.REJECT,
        reason=f"dtype {actual.dtype} against the declared {declared.dtype}",
    )


def conversion_temp_name(param_name: str, target: DataType) -> str:
    """The name both frontends give the temporary that carries a conversion.

    Derived from the parameter and the target dtype, so the same call site
    produces the same name on either lane — which is what lets the parity gate
    compare the two programs structurally instead of tolerating a rename. The
    leading underscore keeps it out of the author's namespace.
    """
    return f"_{param_name}_{str(target).lower()}"
