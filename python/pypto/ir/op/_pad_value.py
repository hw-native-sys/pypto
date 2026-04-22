# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Normalization for the ``fillpad`` ``pad_value`` keyword argument.

The hardware only supports the three padding modes encoded by ``PadValue``
(``zero`` / ``max`` / ``min``). To make the DSL friendlier we accept three
numeric literals as sugar that map onto those modes; anything else raises
with a clear hint so the user is never silently given a different value.
"""

import math

from pypto.pypto_core.ir import PadValue

_HINT = (
    "Use pl.PadValue.zero / pl.PadValue.max / pl.PadValue.min, or one of the literals 0, math.inf, -math.inf."
)


def normalize_pad_value(pad_value: object) -> PadValue:
    """Coerce a fillpad ``pad_value`` argument to a ``PadValue`` enum.

    Accepted inputs:
      * ``PadValue.zero`` / ``max`` / ``min`` — returned unchanged.
      * ``0`` / ``0.0`` — mapped to ``PadValue.zero``.
      * ``math.inf`` — mapped to ``PadValue.max``.
      * ``-math.inf`` — mapped to ``PadValue.min``.

    ``PadValue.null`` is rejected because "no padding mode" is meaningless
    for an op that exists to write a fill value. Anything else (other
    numbers, ``NaN``, ``bool``, ``str``, ``None``, ...) also raises.
    """
    if isinstance(pad_value, PadValue):
        if pad_value == PadValue.null:
            raise ValueError(
                f"fillpad pad_value cannot be PadValue.null — that means no padding mode. {_HINT}"
            )
        return pad_value
    # bool subclasses int — reject explicitly so True/False don't sneak in as 0/1.
    if isinstance(pad_value, bool):
        raise TypeError(f"fillpad pad_value cannot be bool ({pad_value!r}). {_HINT}")
    if isinstance(pad_value, int):
        if pad_value == 0:
            return PadValue.zero
        raise ValueError(
            f"fillpad pad_value only accepts the integer literal 0 "
            f"(mapped to PadValue.zero); got {pad_value!r}. {_HINT}"
        )
    if isinstance(pad_value, float):
        if math.isnan(pad_value):
            raise ValueError(f"fillpad pad_value cannot be NaN. {_HINT}")
        if pad_value == 0.0:
            return PadValue.zero
        if math.isinf(pad_value):
            return PadValue.max if pad_value > 0 else PadValue.min
        raise ValueError(
            f"fillpad pad_value only accepts the float literals 0.0, "
            f"math.inf, or -math.inf; got {pad_value!r}. {_HINT}"
        )
    raise TypeError(
        f"fillpad pad_value must be a PadValue or one of "
        f"0 / math.inf / -math.inf; got {type(pad_value).__name__} "
        f"{pad_value!r}. {_HINT}"
    )
