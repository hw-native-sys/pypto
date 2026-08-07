# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side MX scale pack coverage for multi-box ZZ / NN layouts.

Cube LeftScale / RightScale consume ``[16, 2]`` (MX_A_ZZ) and ``[2, 16]``
(MX_B_NN) boxes. Logical ND scales that span more than one box must be
reordered before they are written to GM for an MX TLOAD. This matches the
PTOAS / pto-isa golden ``convert_scale_a_format`` / ``convert_scale_b_format``
transforms used by the multi-box matmul_mx ST sample.
"""

import pytest
import torch

SCALE_BLOCK_SIZE = 16
SCALE_C0_SIZE = 2


def _pack_a_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical A scales into the MX_A_ZZ physical layout."""
    m, k_groups = scale_codes.shape
    assert m % SCALE_BLOCK_SIZE == 0
    assert k_groups % SCALE_C0_SIZE == 0
    return (
        scale_codes.reshape(
            m // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _pack_b_scale(scale_codes: torch.Tensor) -> torch.Tensor:
    """Pack logical B scales into the MX_B_NN physical layout."""
    k_groups, n = scale_codes.shape
    assert k_groups % SCALE_C0_SIZE == 0
    assert n % SCALE_BLOCK_SIZE == 0
    return (
        scale_codes.reshape(
            k_groups // SCALE_C0_SIZE,
            SCALE_C0_SIZE,
            n // SCALE_BLOCK_SIZE,
            SCALE_BLOCK_SIZE,
        )
        .permute(2, 0, 3, 1)
        .contiguous()
        .reshape(k_groups, n)
    )


def _unpack_a_scale(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_a_scale`."""
    m, k_groups = packed.shape
    return (
        packed.reshape(
            m // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(m, k_groups)
    )


def _unpack_b_scale(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_pack_b_scale`."""
    k_groups, n = packed.shape
    return (
        packed.reshape(
            n // SCALE_BLOCK_SIZE,
            k_groups // SCALE_C0_SIZE,
            SCALE_BLOCK_SIZE,
            SCALE_C0_SIZE,
        )
        .permute(1, 3, 0, 2)
        .contiguous()
        .reshape(k_groups, n)
    )


class TestMxScalePack:
    """Verify multi-box A/B scale pack/unpack against the box address formula."""

    def test_single_a_box_is_identity(self):
        logical = torch.arange(16 * 2, dtype=torch.uint8).reshape(16, 2)
        assert torch.equal(_pack_a_scale(logical), logical)

    def test_multibox_a_pack_reorders_and_roundtrips(self):
        # [32, 4] = 2 (m-boxes) x 2 (k-boxes) of [16, 2].
        logical = torch.arange(32 * 4, dtype=torch.uint8).reshape(32, 4)
        packed = _pack_a_scale(logical)
        assert not torch.equal(packed, logical)

        # Physical GM is box-major: box (mb, kb) occupies a contiguous [16, 2].
        boxes = packed.reshape(2, 2, 16, 2)
        assert torch.equal(boxes[0, 0], logical[0:16, 0:2])
        assert torch.equal(boxes[0, 1], logical[0:16, 2:4])
        assert torch.equal(boxes[1, 0], logical[16:32, 0:2])
        assert torch.equal(boxes[1, 1], logical[16:32, 2:4])
        assert torch.equal(_unpack_a_scale(packed), logical)

    def test_multibox_b_pack_reorders_and_roundtrips(self):
        # [4, 64] = 2 (k-boxes) x 4 (n-boxes) of [2, 16].
        logical = torch.arange(4 * 64, dtype=torch.uint8).reshape(4, 64)
        packed = _pack_b_scale(logical)
        assert not torch.equal(packed, logical)

        # After pack, flat order is n-major then k-major over [2, 16] boxes
        # (reshape as [N/16, K/64, 16, 2] before the final reshape).
        box_grid = packed.reshape(4, 2, 16, 2)
        for nb in range(4):
            for kb in range(2):
                expected = logical[kb * 2 : (kb + 1) * 2, nb * 16 : (nb + 1) * 16].T.contiguous()
                assert torch.equal(box_grid[nb, kb], expected)
        assert torch.equal(_unpack_b_scale(packed), logical)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
