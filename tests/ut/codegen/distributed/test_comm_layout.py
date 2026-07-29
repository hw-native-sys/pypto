# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time CommContext layout pin test.

The runtime `CommContext` struct (runtime/src/common/platform_comm/comm_context.h)
is consumed by emitted distributed kernels via the `CommRemoteOffset` helper,
which indexes the struct using literal byte offsets. PyPTO codegen mirrors those
offsets in `include/pypto/codegen/distributed/comm_layout.h` and pins them with
`static_assert`. This test re-asserts the same literals from Python so a runtime
ABI change is caught even if the C++ static_asserts ever get edited together
with the runtime header.
"""

import pytest
from pypto.pypto_core import ir


def test_comm_context_layout_constants():
    layout = ir.comm_layout
    assert layout.RANK_ID_OFFSET == 16
    assert layout.RANK_NUM_OFFSET == 20
    assert layout.WINDOWS_IN_OFFSET == 32
    assert layout.WINDOWS_OUT_OFFSET == 544
    assert layout.WINDOW_SLOT_STRIDE == 8
    assert layout.COMM_CTX_SIZE == 1056


def test_window_slot_array_fits_between_in_and_out():
    """windowsIn occupies (WINDOWS_OUT_OFFSET - WINDOWS_IN_OFFSET) bytes — i.e.
    64 slots of 8 bytes each."""
    layout = ir.comm_layout
    span = layout.WINDOWS_OUT_OFFSET - layout.WINDOWS_IN_OFFSET
    assert span == 64 * layout.WINDOW_SLOT_STRIDE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
