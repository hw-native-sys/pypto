# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 runtime: multi-row FP4 GM pitch (copy/view) and FP4→BF16 cast via @pl.jit."""

import pypto.language as pl
import pytest
import torch
from harness import st

if not hasattr(torch, "float4_e2m1fn_x2"):
    pytest.skip("torch.float4_e2m1fn_x2 required", allow_module_level=True)

ROWS, LOGICAL_K, PACKED_K = 2, 512, 256
CAST_ROWS, CAST_LOGICAL_K = 4, 64


def _xor_row_fp4() -> torch.Tensor:
    physical = torch.empty((ROWS, PACKED_K), dtype=torch.uint8)
    physical[0] = torch.arange(PACKED_K, dtype=torch.uint8)
    physical[1] = torch.arange(PACKED_K, dtype=torch.uint8).bitwise_xor(0xA5)
    return physical.view(torch.float4_e2m1fn_x2)


def _make_fp4_cast_src(shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = shape
    assert cols % 2 == 0
    generator = torch.Generator().manual_seed(41)
    codes = torch.randint(0, 16, (rows, cols), generator=generator).to(torch.uint8)
    codes[1:] = codes[1:].bitwise_xor(0x05)
    packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
    return packed.contiguous().view(torch.float4_e2m1fn_x2)


def _decode_fp4_data(data: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    packed = data.contiguous().view(torch.uint8).reshape(rows, cols // 2)
    codes = torch.empty((rows, cols), dtype=torch.long)
    codes[:, 0::2] = (packed & 0x0F).to(torch.long)
    codes[:, 1::2] = ((packed >> 4) & 0x0F).to(torch.long)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float64,
    )
    return values[codes]


@pl.jit.incore
def _fp4_copy_kernel(
    src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
    out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
):
    return pl.store(pl.load(src, [0, 0], [ROWS, LOGICAL_K]), [0, 0], out)


@pl.jit
def _fp4_copy_entry(
    src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
    out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
):
    return _fp4_copy_kernel(src, out)


@pl.jit.incore
def _fp4_view_copy_kernel(
    src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
    out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
):
    viewed: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4] = pl.tensor.view(src, [ROWS, LOGICAL_K])
    return pl.store(pl.load(viewed, [0, 0], [ROWS, LOGICAL_K]), [0, 0], out)


@pl.jit
def _fp4_view_copy_entry(
    src: pl.Tensor[[ROWS, LOGICAL_K], pl.FP4],
    out: pl.Out[pl.Tensor[[ROWS, LOGICAL_K], pl.FP4]],
):
    return _fp4_view_copy_kernel(src, out)


@pl.jit.incore
def _fp4_cast_kernel(
    src: pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.FP4],
    out: pl.Out[pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]],
):
    return pl.store(pl.cast(pl.load(src, [0, 0], [CAST_ROWS, CAST_LOGICAL_K]), pl.BF16), [0, 0], out)


@pl.jit
def _fp4_cast_entry(
    src: pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.FP4],
    out: pl.Out[pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]],
):
    return _fp4_cast_kernel(src, out)


# Unannotated Torch float4_e2m1fn_x2 → logical FP4 + last-dim expand (PackFp4 frontend).
@pl.jit.incore
def _fp4_copy_torch_default_kernel(src: pl.Tensor, out: pl.Out[pl.Tensor]):
    return pl.store(pl.load(src, [0, 0], [ROWS, LOGICAL_K]), [0, 0], out)


@pl.jit
def _fp4_copy_torch_default_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _fp4_copy_torch_default_kernel(src, out)


def _copy_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"out": tensors["src"].contiguous().clone()}


def _cast_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    decoded = _decode_fp4_data(tensors["src"], CAST_ROWS, CAST_LOGICAL_K)
    return {"out": decoded.to(torch.bfloat16)}


@pytest.mark.platforms("a5")
@st.cases(
    st.case(
        _fp4_copy_entry,
        _xor_row_fp4(),
        torch.empty((ROWS, PACKED_K), dtype=torch.float4_e2m1fn_x2),
        name="fp4_two_row_copy",
        golden=_copy_golden,
        platform="a5",
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        _fp4_view_copy_entry,
        _xor_row_fp4(),
        torch.empty((ROWS, PACKED_K), dtype=torch.float4_e2m1fn_x2),
        name="fp4_two_row_view_copy",
        golden=_copy_golden,
        platform="a5",
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        _fp4_cast_entry,
        _make_fp4_cast_src((CAST_ROWS, CAST_LOGICAL_K)),
        torch.empty((CAST_ROWS, CAST_LOGICAL_K), dtype=torch.bfloat16),
        name="fp4_multi_row_cast_bf16",
        golden=_cast_golden,
        platform="a5",
        rtol=1e-2,
        atol=1e-2,
    ),
    st.case(
        _fp4_copy_torch_default_entry,
        _xor_row_fp4(),
        torch.empty((ROWS, PACKED_K), dtype=torch.float4_e2m1fn_x2),
        name="fp4_two_row_copy_torch_default",
        golden=_copy_golden,
        platform="a5",
        rtol=0.0,
        atol=0.0,
    ),
)
def test_fp4_runtime(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
