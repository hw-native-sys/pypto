# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression guards binding ``pto_rebuild`` to ``pto_backend`` invariants.

The rebuild path in ``pypto.runtime.debug.pto_rebuild`` deliberately holds
local copies of three pto_backend internals (sentinel literals, the body
preprocess pipeline, and the base ptoas flag list) so the debug entry point
does not pull in the full codegen stack. These tests catch silent drift if
either side evolves.
"""

from __future__ import annotations

import inspect

import pytest
from pypto.backend import pto_backend
from pypto.runtime.debug import pto_rebuild


def _wrapper_src() -> str:
    return inspect.getsource(pto_backend._generate_kernel_wrapper)


def test_begin_sentinel_appears_in_pto_backend() -> None:
    assert pto_rebuild.PTOAS_BODY_BEGIN in _wrapper_src(), (
        f"BEGIN sentinel {pto_rebuild.PTOAS_BODY_BEGIN!r} no longer found in "
        "pto_backend._generate_kernel_wrapper — splice will fail to locate the body."
    )


def test_end_sentinel_appears_in_pto_backend() -> None:
    assert pto_rebuild.PTOAS_BODY_END in _wrapper_src(), (
        f"END sentinel {pto_rebuild.PTOAS_BODY_END!r} no longer found in "
        "pto_backend._generate_kernel_wrapper — splice will fail to locate the body."
    )


_PTOAS_SAMPLE = """#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
using namespace pto;

extern "C" __global__ AICORE void main_kernel(__gm__ int64_t* args) {
    AICORE void helper();
    helper();
}

AICORE void helper() {}
"""


def test_preprocess_matches_pto_backend() -> None:
    """``pto_rebuild._preprocess_ptoas_body`` must produce byte-identical
    output to ``pto_backend._preprocess_ptoas_output`` on the same input —
    otherwise a spliced kernel cpp will diverge from a fresh ``ir.compile()``
    output and miscompile silently.
    """
    expected = pto_backend._preprocess_ptoas_output(_PTOAS_SAMPLE)
    actual = pto_rebuild._preprocess_ptoas_body(_PTOAS_SAMPLE)
    assert actual == expected, (
        "pto_rebuild._preprocess_ptoas_body drifted from pto_backend._preprocess_ptoas_output."
    )


def test_base_ptoas_flags_subset_of_backend_flags() -> None:
    """The rebuild path uses base flags only (no backend-handler extras).

    Both ``pto_rebuild._ptoas_flags`` and ``pto_backend._get_ptoas_flags`` pick
    the ``--pto-level`` from the memory planner (rebuild infers it from the
    ``.pto`` content; a fresh compile from ``memory_planner``), so the drift
    guard checks that every distinct token the rebuild path can emit — across
    both a level3 (``addr =`` present) and level2 (absent) ``.pto`` — still
    appears in ``_get_ptoas_flags``'s source. Source-level check avoids needing
    a configured backend in this test.
    """
    src = inspect.getsource(pto_backend._get_ptoas_flags)
    # Union of tokens the rebuild path emits for both level3 and level2 inputs.
    rebuild_flags = {
        tok
        for content in ("qk = pto.alloc_tile addr = %c0 : ...", "qk = pto.alloc_tile : ...")
        for flag in pto_rebuild._ptoas_flags(content)
        for tok in flag.replace("--pto-level=", "").split()
    }
    missing = [tok for tok in rebuild_flags if repr(tok) not in src and tok not in src]
    assert not missing, (
        f"pto_rebuild base flag tokens {missing!r} no longer found in pto_backend._get_ptoas_flags source."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
