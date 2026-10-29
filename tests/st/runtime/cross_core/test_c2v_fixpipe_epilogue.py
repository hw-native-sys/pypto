# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 C2V narrowing preserves RNE ties and repeated FIFO lifetimes."""

import pypto.language as pl
import pytest
import torch
from harness import st


@pl.jit.incore
def c2v_fixpipe_kernel(
    a: pl.Tensor[[64, 128], pl.BF16],
    b: pl.Tensor[[128, 128], pl.BF16],
    out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
    RELU: pl.constexpr,
    BIDIRECTIONAL: pl.constexpr,
    FOLD: pl.constexpr,
):
    _block = pl.tile.get_block_idx()
    for _iteration in pl.range(3):
        if BIDIRECTIONAL:
            vector = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Vec)
            left = pl.move(vector, target_memory=pl.Mem.Left)
        else:
            mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            left = pl.move(mat, target_memory=pl.Mem.Left)
        right_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        right = pl.move(right_mat, target_memory=pl.Mem.Right)
        acc = pl.matmul(left, right)
        if FOLD:
            narrowed = pl.cast(acc, pl.BF16, mode="rint")
        else:
            # Explicit saturation requests retain the Vector conversion path.
            narrowed = pl.cast(acc, pl.BF16, mode="rint", saturation_mode="off")
        widened = pl.cast(narrowed, pl.FP32)
        if RELU:
            result = pl.maximum(widened, 0.0)
        else:
            result = widened
        out = pl.store(pl.cast(result, pl.BF16, mode="rint"), [_block * 64, 0], out)
    return out


@pl.jit
def c2v_fixpipe_main(
    a: pl.Tensor[[64, 128], pl.BF16],
    b: pl.Tensor[[128, 128], pl.BF16],
    out: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
    reference: pl.Out[pl.Tensor[[64, 128], pl.BF16]],
    RELU: pl.constexpr,
    BIDIRECTIONAL: pl.constexpr,
):
    with pl.spmd(1, sync_start=True):
        out = c2v_fixpipe_kernel(a, b, out, RELU, BIDIRECTIONAL, True)
    with pl.spmd(1, sync_start=True):
        reference = c2v_fixpipe_kernel(a, b, reference, RELU, BIDIRECTIONAL, False)
    return out, reference


def _compare(actual, expected):
    got, reference = actual["out"], actual["reference"]
    assert torch.equal(torch.isnan(got), torch.isnan(reference))
    finite_or_inf = ~torch.isnan(reference)
    assert torch.equal(got.view(torch.int16)[finite_or_inf], reference.view(torch.int16)[finite_or_inf])
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)


def _case(relu, bidirectional, special=False):
    # Two BF16 products sum exactly in FP32. Adjacent columns put ties above
    # even and odd BF16 mantissas, with both signs, and values on either side.
    a = torch.zeros(64, 128, dtype=torch.bfloat16)
    a[:, :2] = 1
    b = torch.zeros(128, 128, dtype=torch.bfloat16)
    bases = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125] * 32)
    residuals = torch.tensor([2**-8, 2**-8, -(2**-8), -(2**-8)] * 32)
    residuals[4::8] *= 0.5
    residuals[5::8] *= 1.5
    b[0] = bases.to(torch.bfloat16)
    b[1] = residuals.to(torch.bfloat16)
    if special:
        # Isolate each special value in one output column. Other K products
        # stay finite zero, so they cannot introduce spurious 0*Inf NaNs.
        b[0, :8] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 0.0, 0.0, 0.0]).to(
            torch.bfloat16
        )
        b[1, :8] = 0
        b[0, 8:10] = torch.finfo(torch.bfloat16).max
        b[1, 8:10] = float(2**119)
    expected = (a.float() @ b.float()).to(torch.bfloat16).float()
    if relu:
        expected = torch.maximum(expected, torch.tensor(0.0))
    return st.case(
        c2v_fixpipe_main,
        a,
        b,
        torch.zeros(64, 128, dtype=torch.bfloat16),
        torch.zeros(64, 128, dtype=torch.bfloat16),
        relu,
        bidirectional,
        name=f"c2v_bf16_relu{relu}_v2c{bidirectional}_special{special}",
        golden=lambda _: {"out": expected.to(torch.bfloat16), "reference": expected.to(torch.bfloat16)},
        compare=_compare,
        rtol=0,
        atol=0,
    )


@pytest.mark.platforms("a5")
@st.cases(
    *(_case(relu, bidirectional) for relu in (False, True) for bidirectional in (False, True)),
    *(_case(relu, False, special=True) for relu in (False, True)),
)
def test_c2v_fixpipe_rounding_and_reuse(case_run):
    case_run.assert_passed()
    if case_run.work_dir is not None:
        sources = list((case_run.work_dir / "ptoas").glob("*.pto"))
        assert sources, "Expected persisted PTO sources for the compiled test"
        assert any("acc_push_epilogue" in source.read_text() for source in sources)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
