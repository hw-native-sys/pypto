# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical copy coverage, including SRAM reuse across InCore calls."""

import importlib

import pypto.language as pl
import pytest
import torch


@pl.jit.incore
def stage(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    pl.copy(dst, src, [0, 0], [0, 0], [4, 600])
    return dst


@pl.jit.incore
def restore(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    pl.copy(
        dst,
        src,
        [0, 0],
        [0, 0],
        [4, 600],
        source_memory=pl.Mem.SRAM,
        target_memory=pl.Mem.DDR,
    )
    return dst


@pl.jit
def roundtrip(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    scratch = pl.create_tensor([4, 600], dtype=pl.FP32)
    scratch = stage(src, scratch)
    return restore(scratch, dst)


@pl.jit.incore
def copy_region(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.InOut[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    pl.copy(dst, src, [1, 8], [0, 16], [2, 577])
    return dst


@pl.jit
def region(
    src: pl.Tensor[[4, 600], pl.FP32], dst: pl.InOut[pl.Tensor[[4, 600], pl.FP32]]
) -> pl.Tensor[[4, 600], pl.FP32]:
    return copy_region(src, dst)


def test_cross_incore_roundtrip(test_config):
    src = torch.arange(2400, dtype=torch.float32).reshape(4, 600)
    dst = torch.zeros_like(src)
    roundtrip(src, dst, config=test_config)
    torch.testing.assert_close(dst, src, rtol=0, atol=0)


def test_subregion_preserves_surrounding_data(test_config):
    src = torch.arange(2400, dtype=torch.float32).reshape(4, 600)
    dst = torch.full_like(src, -1)
    expected = dst.clone()
    expected[1:3, 8:585] = src[:2, 16:593]
    region(src, dst, config=test_config)
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["random", "identity_left", "identity_right"])
def test_beginner_matmul_ddr_copy(test_config, mode):
    """Run the actual beginner example, not a separately implemented matmul."""
    matmul = importlib.import_module("examples.beginner.05_matmul").matmul_64
    torch.manual_seed(17)
    a = torch.randn(64, 64, dtype=torch.float32)
    b = torch.randn(64, 64, dtype=torch.float32)
    if mode == "identity_left":
        a = torch.eye(64)
    elif mode == "identity_right":
        b = torch.eye(64)
    before_a, before_b = a.clone(), b.clone()
    c = torch.full((64, 64), float("nan"))
    compiled = matmul.compile(a, b, c, config=test_config)
    compiled(a, b, c, config=test_config)
    torch.testing.assert_close(c, a @ b, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(a, before_a, rtol=0, atol=0)
    torch.testing.assert_close(b, before_b, rtol=0, atol=0)

    # The example must execute real DDR copies rather than aliasing A/B away.
    kernels = [path.read_text() for path in compiled.output_dir.rglob("*.pto")]
    copy_kernels = [text for text in kernels if 'target_memory = "gm"' in text]
    assert copy_kernels, "The example must emit a DDR copy kernel"
    assert sum(text.count('target_memory = "gm"') for text in copy_kernels) >= 2
    assert all('source_memory = "gm"' in text and "pto.tload " in text for text in copy_kernels)
    assert all('"sram"' not in text for text in kernels)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
