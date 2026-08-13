# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device-free tests for dtype conversion in the system-test harness."""

import pytest
import torch

from harness.core.harness import DataType


@pytest.mark.parametrize(
    ("member", "torch_name"),
    [
        ("FP8E4M3FN", "float8_e4m3fn"),
        ("FP8E8M0", "float8_e8m0fnu"),
        ("FP4", "float4_e2m1fn_x2"),
    ],
)
def test_mx_dtype_maps_to_torch(member, torch_name):
    """Harness MX enum values map to the matching optional torch dtype."""
    dtype = getattr(torch, torch_name, None)
    if not isinstance(dtype, torch.dtype):
        pytest.skip(f"torch.{torch_name} is unavailable")
    assert getattr(DataType, member).torch_dtype is dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
