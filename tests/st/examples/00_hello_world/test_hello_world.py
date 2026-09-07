# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Hello World Example for PyPTO — element-wise tensor addition.

Verifies the simplest end-to-end @pl.jit kernel: load → add → store.
"""

import pytest
import torch
from examples.beginner.hello_world import tile_add
from harness import st


def _add_case():
    """Element-wise addition: c = a + b."""
    a = torch.full((128, 128), 2.0, dtype=torch.float32)
    b = torch.full((128, 128), 3.0, dtype=torch.float32)
    c = torch.zeros((128, 128), dtype=torch.float32)
    return st.case(tile_add, a, b, c, name="hello_world_add", golden=lambda _: a + b)


@st.cases(_add_case())
def test_hello_world_add(case_run):
    """Compile and run element-wise addition; compare result to torch reference."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
