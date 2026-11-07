# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 round trip of FP8E5M2 data through the runtime's UINT8 byte carrier.

The pinned runtime has no E5M2 ``DataType``; PyPTO passes ``torch.float8_e5m2``
host tensors as zero-copy UINT8 views while the kernel keeps its FP8E5M2 type.
A load/store copy read at a non-zero offset must return every finite E5M2 code
unchanged -- normals, subnormals, the 57344 maximum, and *both* signed zeros.

The comparison is bitwise, on the raw bytes, which the signed zeros require:
the harness default compares elementwise with ``torch.isclose``
(``device_runner.py``), and ``isclose(-0.0, +0.0)`` is true even at
``rtol=atol=0``, so a path that flipped the sign bit of a zero would pass an
elementwise check. Comparing ``view(torch.uint8)`` makes "the bytes survive"
the actual assertion.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

_REQUIRED_TORCH_DTYPES = ("float8_e5m2",)
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch.float8_e5m2 required", allow_module_level=True)

ROWS, COLS = 32, 256
SRC_ROWS = 2 * ROWS
ROW_OFFSET = ROWS


def _finite_e5m2_codes() -> torch.Tensor:
    """All E5M2 byte codes except Inf (0x7C) and NaN (0x7D-0x7F), for both signs.

    Both signed zeros (0x00 and 0x80) are included, which is why the comparison
    has to be bitwise.
    """
    codes = torch.arange(256, dtype=torch.int32)
    magnitude = codes & 0x7F
    return codes[magnitude < 0x7C].to(torch.uint8)


def _source_bytes() -> torch.Tensor:
    """Deterministic E5M2 source whose copied window holds every finite code."""
    finite = _finite_e5m2_codes()
    generator = torch.Generator().manual_seed(5)
    picks = torch.randint(0, finite.numel(), (SRC_ROWS, COLS), generator=generator)
    codes = finite[picks]
    # Row ROW_OFFSET enumerates the finite codes in order, so the copied window
    # provably contains each one at least once, subnormals and zeros included.
    codes[ROW_OFFSET] = torch.cat([finite, finite[: COLS - finite.numel()]])
    assert codes[ROW_OFFSET].eq(0x00).any(), "+0.0 must be covered"
    assert codes[ROW_OFFSET].eq(0x80).any(), "-0.0 must be covered"
    return codes


@pl.jit.incore
def e5m2_offset_copy_kernel(
    src: pl.Tensor[[SRC_ROWS, COLS], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP8E5M2]],
) -> pl.Tensor[[ROWS, COLS], pl.FP8E5M2]:
    out = pl.store(pl.load(src, [ROW_OFFSET, 0], [ROWS, COLS]), [0, 0], out)
    return out


@pl.jit
def e5m2_offset_copy(
    src: pl.Tensor[[SRC_ROWS, COLS], pl.FP8E5M2],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP8E5M2]],
) -> pl.Tensor[[ROWS, COLS], pl.FP8E5M2]:
    return e5m2_offset_copy_kernel(src, out)


def _golden(tensors: dict[str, torch.Tensor]) -> None:
    tensors["out"][:] = tensors["src"][ROW_OFFSET : ROW_OFFSET + ROWS]


def _compare_bytes(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    """Assert byte-for-byte equality, so a corrupted -0.0 cannot pass."""
    for name, got in actual.items():
        want = expected[name]
        got_bytes = got.view(torch.uint8)
        want_bytes = want.view(torch.uint8)
        mismatches = int((got_bytes != want_bytes).sum())
        assert mismatches == 0, (
            f"Output '{name}' differs in {mismatches}/{want_bytes.numel()} bytes; "
            f"first actual={got_bytes.flatten()[:8].tolist()} "
            f"expected={want_bytes.flatten()[:8].tolist()}"
        )


@pytest.mark.platforms("a5")
@st.cases(
    st.case(
        e5m2_offset_copy,
        _source_bytes().view(torch.float8_e5m2),
        torch.zeros(ROWS, COLS, dtype=torch.float8_e5m2),
        name="fp8e5m2_offset_copy",
        golden=_golden,
        compare=_compare_bytes,
        platform="a5",
    ),
)
def test_e5m2_offset_copy_is_byte_exact(case_run):
    """Every finite E5M2 byte code survives the UINT8 wire carrier unchanged."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
