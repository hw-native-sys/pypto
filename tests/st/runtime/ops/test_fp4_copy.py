# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 regression coverage for packed-FP4 GM row addressing (issue #2754)."""

import pypto.language as pl
import pytest
import torch
from harness import st

if not hasattr(torch, "float4_e2m1fn_x2"):
    pytest.skip("torch.float4_e2m1fn_x2 is required", allow_module_level=True)

ROWS = 2
LOGICAL_WIDTH = 512
PHYSICAL_WIDTH = LOGICAL_WIDTH // 2


@pl.jit
def fp4_copy(
    source: pl.Tensor[[ROWS, LOGICAL_WIDTH], pl.FP4],
    output: pl.Out[pl.Tensor[[ROWS, LOGICAL_WIDTH], pl.FP4]],
):
    for row in pl.spmd(ROWS):
        value = pl.load(source, [row, 0], [1, LOGICAL_WIDTH])
        pl.store(value, [row, 0], output)
    return output


def _values() -> torch.Tensor:
    physical = torch.empty((ROWS, PHYSICAL_WIDTH), dtype=torch.uint8)
    physical[0] = torch.arange(PHYSICAL_WIDTH, dtype=torch.uint8)
    physical[1] = torch.arange(PHYSICAL_WIDTH, dtype=torch.uint8).bitwise_xor(0xA5)
    return physical.view(torch.float4_e2m1fn_x2)


def _golden(tensors):
    tensors["output"].view(torch.uint8).copy_(tensors["source"].view(torch.uint8))


def _compare_bytes(actual, expected):
    actual_bytes = actual["output"].view(torch.uint8)
    expected_bytes = expected["output"].view(torch.uint8)
    torch.testing.assert_close(actual_bytes, expected_bytes, rtol=0, atol=0)


_FP4_COPY_CASE = st.case(
    fp4_copy,
    _values(),
    torch.zeros((ROWS, PHYSICAL_WIDTH), dtype=torch.uint8).view(torch.float4_e2m1fn_x2),
    name="fp4_copy_two_rows",
    golden=_golden,
    compare=_compare_bytes,
    platform="a5",
)
# Keep this A5 regression on the machine's supported AICPU affinity set [1, 2, 3].
_FP4_COPY_CASE.config.aicpu_thread_num = 3


@pytest.mark.platforms("a5")
@st.cases(_FP4_COPY_CASE)
def test_fp4_copy_two_rows(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
