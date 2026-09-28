# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Borrow native A2/A3 FRACTAL_NZ weights in direct, registered and captured calls."""

import os
import subprocess
import sys

import pypto.language as pl
import pytest


@pl.jit
def bf16_nt(
    x: pl.Tensor[[16, 64], pl.BF16],
    w: pl.Tensor[[32, 64], pl.BF16, pl.NZ],
    out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:, :] = pl.matmul(x, w, b_trans=True)
    return out


@pl.jit
def int8_kn_group(
    x: pl.Tensor[[16, 64], pl.INT8],
    w: pl.Tensor[[128, 32], pl.INT8, pl.NZ],
    out: pl.Out[pl.Tensor[[16, 32], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        # Read the second K group from the original two-dimensional NZ matrix.
        out[:, :] = pl.matmul(x, w[64:128, :], out_dtype=pl.INT32)
    return out


def _run(device, directory):
    import torch  # noqa: PLC0415
    import torch_npu  # noqa: PLC0415
    from pypto import CacheConfig, configure_cache  # noqa: PLC0415
    from pypto.torch import init, register  # noqa: PLC0415

    os.chdir(directory)
    os.environ.pop("PYPTO_PROG_BUILD_DIR", None)
    configure_cache(CacheConfig(enabled=False))
    torch_npu.npu.set_device(device)
    torch_npu.npu.config.allow_internal_format = True
    init()
    for kernel, dtype, shape, output_dtype in (
        (bf16_nt, torch.bfloat16, (32, 64), torch.float32),
        (int8_kn_group, torch.int8, (128, 32), torch.int32),
    ):
        logical = (torch.arange(shape[0] * shape[1]).reshape(shape) % 3 - 1).to(dtype)
        host_x = (torch.arange(16 * 64).reshape(16, 64) % 3 - 1).to(dtype)
        x = torch_npu.npu_format_cast(host_x.to(f"npu:{device}"), 2)
        weight = torch_npu.npu_format_cast(logical.to(x.device), 29)
        original = weight.data_ptr(), weight.untyped_storage().data_ptr()
        out = torch.empty((16, 32), dtype=output_dtype, device=x.device)
        matrix = logical.T if kernel is bf16_nt else logical[64:128]
        expected = host_x.to(output_dtype) @ matrix.to(output_dtype)
        if dtype == torch.bfloat16:
            native = torch.nn.functional.linear(x, weight)
        else:
            # Native quant matmul consumes the very same two-dimensional NZ bytes.
            full_x = torch.cat((torch.zeros_like(x), x), dim=1)
            native = torch_npu.npu_quant_matmul(
                full_x,
                weight,
                torch.ones(32, device=x.device),
                pertoken_scale=torch.ones(16, device=x.device),
                output_dtype=torch.bfloat16,
            )
        torch.testing.assert_close(native.cpu().to(output_dtype), expected, rtol=0, atol=0)
        op = register(kernel, f"pypto_native_nz_st::{kernel.__name__}")
        # Any accidental Python-side conversion in the call adapter is an error.
        with pytest.MonkeyPatch.context() as patch:

            def forbidden(*args, **kwargs):
                raise AssertionError("launch must borrow NZ storage without a format conversion")

            patch.setattr(torch_npu, "npu_format_cast", forbidden)
            assert kernel(x, weight, out) is out
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            assert op(x, weight, out) is out
            torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
            graph = torch_npu.npu.NPUGraph()
            with torch_npu.npu.graph(graph):
                op(x, weight, out)
            for sign in (1, -1, 1):
                x.copy_(host_x * sign)
                graph.replay()
                torch.testing.assert_close(out.cpu(), expected * sign, rtol=0, atol=0)
        assert torch_npu.get_npu_format(weight) == 29
        assert (weight.data_ptr(), weight.untyped_storage().data_ptr()) == original
        del graph


def test_native_nz(test_config, tmp_path):
    if test_config.codegen_only or test_config.platform != "a2a3":
        pytest.skip("Requires an A2/A3 NPU and the optional torch adapter")
    pytest.importorskip("torch_npu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.st.runtime.kernel.test_native_nz import _run; "
            "import sys; _run(int(sys.argv[1]), sys.argv[2])",
            str(test_config.device_id),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
