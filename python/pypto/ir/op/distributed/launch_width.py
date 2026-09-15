# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Launch-width admission mapping for managed multi-AIV collectives (RFC #2521).

Tested, canonical reference for the ``(P, L) -> B`` formula the RFC's frozen contract
item 3 specifies. This module builds no IR and is not itself part of any lowering
pass or codegen path: the actual runtime call site
(``all_to_all_v/templates/entry.cpp.in``, plan 120 in ``pypto-3.0-notes``) is a
self-contained C++ template compiled separately from these compiler sources, so it
carries its own C++ mirror of this exact formula rather than linking against it.
This function exists to pin the formula down and test it once, in a form that is
directly testable today.
"""


def cal_all_to_all_v_blocks(p: int, l: int) -> int:  # noqa: E741 - RFC's own symbol names
    """Map a requested launch width to the admitted block count.

    RFC #2521's frozen contract item 3: a pure function of ``(P, L)`` only, no
    payload input, computed once at the entry and never recomputed by the kernel
    (contract item 11).

    Args:
        p: Rank count (the communication domain size).
        l: Requested AIV block count (``core_num``'s "maximum" semantics).

    Returns:
        The admitted block count ``B``: ``L`` unchanged when it doesn't reach a
        full rank cohort (``L < P``), otherwise the largest multiple of ``P`` not
        exceeding ``L``.

    Raises:
        TypeError: If ``p`` or ``l`` is not a non-boolean ``int`` (``bool`` and
            floats are rejected so the annotated ``int`` return contract holds).
        ValueError: If ``p`` or ``l`` is not positive.
    """
    if type(p) is not int:
        raise TypeError(f"cal_all_to_all_v_blocks: P (rank count) must be int, got {type(p).__name__}")
    if type(l) is not int:
        raise TypeError(
            f"cal_all_to_all_v_blocks: L (requested core_num) must be int, got {type(l).__name__}"
        )
    if p <= 0:
        raise ValueError(f"cal_all_to_all_v_blocks: P (rank count) must be positive, got {p}")
    if l <= 0:
        raise ValueError(f"cal_all_to_all_v_blocks: L (requested core_num) must be positive, got {l}")
    return l if l < p else (l // p) * p
