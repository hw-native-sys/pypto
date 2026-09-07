# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Softmax System Tests for PyPTO.

One tile reduction pattern is demonstrated:
  1. Softmax    — exp(x - max(x)) / sum(exp(x - max(x)))
"""

import pytest
import torch
from examples.intermediate.softmax import tile_softmax
from harness import st


def _softmax_case():
    """Row-wise softmax over a 64x64 tile."""
    torch.manual_seed(0)
    a = torch.randn(64, 64, dtype=torch.float32)
    output = torch.zeros_like(a)
    return st.case(tile_softmax, a, output, name="tile_softmax", golden=lambda _: torch.softmax(a, dim=-1))


@st.cases(_softmax_case())
def test_tile_softmax(case_run):
    """Row-wise softmax matches the torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
