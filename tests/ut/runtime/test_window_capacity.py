# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dense window capacity checks need no device or simpler runtime."""

import pytest
from pypto.runtime.tensor_arg import check_window_capacity


@pytest.mark.parametrize("slot_nbytes", [64, 128])
def test_fitting_window(slot_nbytes):
    check_window_capacity(name="data", shapes=(4, 4), element_bytes=4, slot_nbytes=slot_nbytes)


def test_logical_capacity_excludes_alignment_padding():
    with pytest.raises(ValueError, match="64 bytes exceeds its backing window slot of 60 bytes"):
        check_window_capacity(name="data", shapes=(16,), element_bytes=4, slot_nbytes=60)


@pytest.mark.parametrize("dim", [0, -1])
def test_non_positive_dimension(dim):
    with pytest.raises(ValueError, match="non-positive dimension"):
        check_window_capacity(name="data", shapes=(dim,), element_bytes=4, slot_nbytes=64)


def test_capacity_arithmetic_does_not_overflow():
    with pytest.raises(ValueError, match="exceeds its backing window slot"):
        check_window_capacity(name="data", shapes=(2**62, 4), element_bytes=4, slot_nbytes=2**63 - 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
