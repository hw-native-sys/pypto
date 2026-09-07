# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime tests for tile.concat (column-wise concatenation) using @pl.jit."""

import pytest
import torch
from examples.beginner.concat import tile_concat_32x32
from harness import st


def _concat_case():
    """Tile concatenation: 32x16 + 32x16 -> 32x32.

    Uses random data on purpose: with a constant-filled ``a``, a concat that
    overwrites rows of its own source before reading them still produces the
    expected output (every row holds the same value), so the corruption is
    invisible. Distinct per-row values are what make such a defect fail here.
    """
    torch.manual_seed(0)
    a = torch.randn(32, 16, dtype=torch.float32)
    b = torch.randn(32, 16, dtype=torch.float32)
    c = torch.zeros((32, 32), dtype=torch.float32)
    return st.case(
        tile_concat_32x32,
        a,
        b,
        c,
        name="tile_concat_32x32",
        golden=lambda _: torch.cat([a, b], dim=1),
    )


@st.cases(_concat_case())
def test_tile_concat_32x32(case_run):
    """Column-wise concat matches the torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
