# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime tests for MX quantization."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec

MX_GROUP_SIZE = 32


def _case_shapes(group_axis: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return ((16, 64), (16, 64)) if group_axis == 1 else ((32, 64), (64, 32))


def _pack_scale_codes(codes: torch.Tensor, group_axis: int) -> torch.Tensor:
    """Pack logical scale codes into the public A-ZZ or B-NN byte order."""
    if group_axis == 1:
        rows, groups = codes.shape
        packed = codes.reshape(rows // 16, 16, groups // 2, 2).permute(0, 2, 1, 3)
    else:
        groups, cols = codes.shape
        packed = codes.reshape(groups // 2, 2, cols // 16, 16).permute(2, 0, 3, 1)
    return packed.contiguous().reshape(1, -1).view(torch.float8_e8m0fnu)


def _quant_inputs_and_golden(group_axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact MXFP8 blocks with known quantized values and E8M0 scales."""
    fp8_values = torch.tensor(
        [
            -448,
            -256,
            -128,
            -64,
            -32,
            -16,
            -8,
            -4,
            -2,
            -1,
            -0.5,
            -0.25,
            -0.125,
            -0.0625,
            -0.015625,
            0,
            0.015625,
            0.0625,
            0.125,
            0.25,
            0.5,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            448,
            0,
        ],
        dtype=torch.float32,
    )
    src_shape, quant_shape = _case_shapes(group_axis)
    groups = src_shape[1] // MX_GROUP_SIZE
    exponent_shape = (src_shape[0], groups) if group_axis == 1 else (groups, src_shape[0])
    group_exponents = (torch.arange(exponent_shape[0] * exponent_shape[1]) % 4 - 1).reshape(exponent_shape)
    scales = torch.pow(2.0, group_exponents).to(torch.float32)
    if group_axis == 1:
        src = (fp8_values.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(src_shape)
        quantized = fp8_values.to(torch.float8_e4m3fn).repeat(src_shape[0], groups)
    else:
        src_kn = (fp8_values.reshape(1, MX_GROUP_SIZE, 1) * scales.unsqueeze(1)).reshape(quant_shape)
        src = src_kn.transpose(0, 1)
        quantized = (
            fp8_values.to(torch.float8_e4m3fn)
            .reshape(1, MX_GROUP_SIZE, 1)
            .expand(groups, MX_GROUP_SIZE, src_shape[0])
            .reshape(quant_shape)
        )
    scale = _pack_scale_codes((group_exponents + 127).to(torch.uint8), group_axis)
    return src.contiguous(), quantized.contiguous(), scale.contiguous()


def _make_fp8_quant(group_axis: int):
    """Quantize exactly representable FP32 blocks through one public MX path."""
    (rows, cols), (quant_rows, quant_cols) = _case_shapes(group_axis)
    scale_elements = rows * cols // MX_GROUP_SIZE

    @pl.jit(auto_scope=False)
    def quantize(
        src: pl.Tensor[[rows, cols], pl.FP32],
        out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.FP8E4M3FN]],
        out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.FP8E8M0]],
    ):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            values = pl.tile.load(src, [block * rows, 0], [rows, cols])
            quantized, scale = pl.quant_mx(values, group_axis=group_axis)
            pl.tile.store(quantized, [0, 0], out_quant)
            pl.tile.store(pl.reshape(scale, [1, scale_elements]), [0, 0], out_scale)
        return out_quant, out_scale

    return quantize


def _fp8_quant_golden(group_axis: int):
    def golden(tensors):
        _src, quant, scale = _quant_inputs_and_golden(group_axis)
        return {"out_quant": quant, "out_scale": scale}

    return golden


def _fp4_quant_golden(group_axis, algorithm, raw=False, interleave=False):
    """Reference block scaling and nearest-even E2M1 encoding in carrier order."""

    def golden(tensors):
        source = tensors["src"].float()
        if raw and group_axis == 0:
            source = source.transpose(0, 1)
        blocks = source.reshape(source.shape[0], -1, MX_GROUP_SIZE)
        maximum = blocks.abs().amax(dim=-1)
        exponents = (
            torch.floor(torch.log2(maximum)) - 2
            if algorithm == "ocp"
            else torch.ceil(torch.log2(maximum / 6))
        )
        normalized = blocks / torch.pow(2.0, exponents).unsqueeze(-1)
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
        distance = (normalized.abs().unsqueeze(-1) - levels).abs()
        # Even nibble codes win equidistant ties, independently of the device cast.
        order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        codes = order[distance[..., order].argmin(dim=-1)]
        codes = (codes | (torch.signbit(normalized).to(torch.long) << 3)).to(torch.uint8)
        codes = codes.reshape(source.shape)
        if group_axis == 0:
            codes = codes.transpose(0, 1)
            exponents = exponents.transpose(0, 1)
        payload = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        scale_codes = (exponents + 127).to(torch.uint8)
        if interleave:
            groups, cols = scale_codes.shape
            scale = scale_codes.reshape(groups // 2, 2, cols).transpose(1, 2).contiguous().flatten(1)
        else:
            scale = scale_codes.reshape(1, -1) if raw else _pack_scale_codes(scale_codes, group_axis)
        return {"out_quant": payload, "out_scale": scale.view(torch.uint8)}

    return golden


def _fp4_quant_source(group_axis, dtype):
    rows = 16 if group_axis == 1 else 64
    # Distinct block exponents and signs expose orientation and packed-pitch errors.
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.arange(rows * 64).reshape(rows, 2, 32)
    values = levels[(codes + codes // 32) % 8]
    values = torch.where((codes // 8) % 2 == 0, values, -values)
    # A maximum of seven distinguishes OCP scale selection from NV's ceil(max/6).
    values[:, 1, 0] = 7
    exponents = (torch.arange(rows * 2).reshape(rows, 2) % 5) - 2
    return (values * torch.pow(2.0, exponents).unsqueeze(-1)).reshape(rows, 64).to(dtype)


def _make_fp4_quant(group_axis, dtype, algorithm):
    rows = 16 if group_axis == 1 else 64
    quant_rows, quant_cols = (rows, 32) if group_axis == 1 else (64, rows // 2)
    scale_elements = rows * 2

    @pl.jit(auto_scope=False)
    def quantize(
        src: pl.Tensor[[rows, 64], dtype],
        out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.fp4e2m1x2]],
        out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.UINT8]],
    ):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            values = pl.tile.load(src, [block * rows, 0], [rows, 64])
            quantized, scale = pl.quant_mx(
                values, group_axis=group_axis, dtype=pl.fp4e2m1x2, quant_scale_alg=algorithm
            )
            pl.tile.store(quantized, [0, 0], out_quant)
            scale_bytes = pl.reinterpret_view(pl.reshape(scale, [1, scale_elements]), pl.UINT8)
            pl.tile.store(scale_bytes, [0, 0], out_scale)
        return out_quant, out_scale

    return quantize


def _compare_quant_bytes(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name].view(torch.uint8), expected[name].view(torch.uint8)), name


@pl.jit(auto_scope=False)
def _fp4_raw_quant(
    src: pl.Tensor[[3, 64], pl.BF16],
    out_quant: pl.Out[pl.Tensor[[3, 32], pl.fp4e2m1x2]],
    out_scale: pl.Out[pl.Tensor[[1, 6], pl.UINT8]],
):
    with pl.spmd(1):
        block = pl.tile.get_block_idx()
        values = pl.tile.load(src, [block * 3, 0], [3, 64])
        max_workspace = pl.tile.set_validshape(pl.tile.create([1, 16], pl.BF16), 1, 6)
        scaling_workspace = pl.tile.set_validshape(pl.tile.create([1, 128], pl.BF16), 1, 6)
        quantized, exp = pl.tile.quant_mx_raw(
            values, max_workspace, scaling_workspace, group_axis=1, dtype=pl.fp4e2m1x2
        )
        pl.tile.store(quantized, [0, 0], out_quant)
        pl.tile.store(exp, [0, 0], out_scale)
    return out_quant, out_scale


def _make_fp4_raw_interleave(algorithm):
    @pl.jit(auto_scope=False)
    def quantize(
        src: pl.Tensor[[128, 64], pl.BF16],
        out_quant: pl.Out[pl.Tensor[[128, 32], pl.fp4e2m1x2]],
        out_scale: pl.Out[pl.Tensor[[2, 128], pl.UINT8]],
    ):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            values = pl.tile.load(src, [block * 128, 0], [128, 64])
            max_workspace = pl.tile.create([4, 64], pl.BF16)
            scaling_workspace = pl.tile.create([4, 64], pl.BF16)
            quantized, exp = pl.tile.quant_mx_raw(
                values,
                max_workspace,
                scaling_workspace,
                group_axis=0,
                dtype=pl.fp4e2m1x2,
                quant_scale_alg=algorithm,
                interleave=True,
            )
            pl.tile.store(quantized, [0, 0], out_quant)
            pl.tile.store(exp, [0, 0], out_scale)
        return out_quant, out_scale

    return quantize


def _fp4_quant_output(name: str, shape: list[int]) -> TensorSpec:
    return TensorSpec(
        name,
        shape,
        DataType.UINT8,
        init_value=lambda: torch.empty(shape, dtype=torch.uint8),
        is_output=True,
    )


# Defer Torch operations until after pytest-forked forks the test process;
# collection-time computation can leave inherited CPU thread-pool locks held.
@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_fp4_quant(axis, dtype, algorithm),
            tensors=[
                TensorSpec(
                    "src",
                    [16 if axis == 1 else 64, 64],
                    data_dtype,
                    init_value=lambda axis=axis, torch_dtype=torch_dtype: _fp4_quant_source(
                        axis, torch_dtype
                    ),
                ),
                _fp4_quant_output("out_quant", [16 if axis == 1 else 64, 32]),
                _fp4_quant_output("out_scale", [1, 32 if axis == 1 else 128]),
            ],
            name=f"fp4_quant_axis{axis}_{dtype}_{algorithm}",
            golden=_fp4_quant_golden(axis, algorithm),
            compare=_compare_quant_bytes,
        )
        for dtype, torch_dtype, data_dtype in (
            (pl.BF16, torch.bfloat16, DataType.BF16),
            (pl.FP16, torch.float16, DataType.FP16),
        )
        for axis, algorithm in ((1, "ocp"), (0, "ocp"), (0, "nv"))
    ],
    st.case(
        _fp4_raw_quant,
        tensors=[
            TensorSpec(
                "src",
                [3, 64],
                DataType.BF16,
                init_value=lambda: _fp4_quant_source(1, torch.bfloat16)[:3, :64].contiguous(),
            ),
            _fp4_quant_output("out_quant", [3, 32]),
            _fp4_quant_output("out_scale", [1, 6]),
        ],
        name="fp4_quant_raw_small_rows",
        golden=_fp4_quant_golden(1, "ocp", raw=True),
        compare=_compare_quant_bytes,
    ),
    *[
        st.case(
            _make_fp4_raw_interleave(algorithm),
            tensors=[
                TensorSpec(
                    "src",
                    [128, 64],
                    DataType.BF16,
                    init_value=lambda: torch.cat(
                        (_fp4_quant_source(0, torch.bfloat16), _fp4_quant_source(0, torch.bfloat16) * 4),
                        dim=1,
                    ).T.contiguous(),
                ),
                _fp4_quant_output("out_quant", [128, 32]),
                _fp4_quant_output("out_scale", [2, 128]),
            ],
            name=f"fp4_quant_raw_interleave_{algorithm}",
            golden=_fp4_quant_golden(0, algorithm, raw=True, interleave=True),
            compare=_compare_quant_bytes,
        )
        for algorithm in ("ocp", "nv")
    ],
)
def test_fp4_quant_runtime(case_run):
    case_run.assert_passed()


# The FP8 paths used to run through a PTOTestCase, whose default validator is
# `torch.isclose`: at rtol=atol=0 that still accepts a -0.0 for a +0.0, so the
# "exact" claim did not actually cover the sign bit of a zero payload byte.
# These cases compare carrier bytes instead, like the FP4 ones above.
@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_fp8_quant(axis),
            tensors=[
                TensorSpec(
                    "src",
                    list(_case_shapes(axis)[0]),
                    DataType.FP32,
                    init_value=lambda axis=axis: _quant_inputs_and_golden(axis)[0],
                ),
                TensorSpec(
                    "out_quant",
                    list(_case_shapes(axis)[1]),
                    DataType.FP8E4M3FN,
                    init_value=lambda axis=axis: torch.zeros(_case_shapes(axis)[1], dtype=torch.uint8).view(
                        torch.float8_e4m3fn
                    ),
                    is_output=True,
                ),
                TensorSpec(
                    "out_scale",
                    [1, _case_shapes(axis)[0][0] * _case_shapes(axis)[0][1] // MX_GROUP_SIZE],
                    DataType.FP8E8M0,
                    init_value=lambda axis=axis: torch.zeros(
                        (1, _case_shapes(axis)[0][0] * _case_shapes(axis)[0][1] // MX_GROUP_SIZE),
                        dtype=torch.uint8,
                    ).view(torch.float8_e8m0fnu),
                    is_output=True,
                ),
            ],
            name=f"quant_mx_axis{axis}",
            golden=_fp8_quant_golden(axis),
            compare=_compare_quant_bytes,
        )
        for axis in (1, 0)
    ],
)
def test_fp8_quant_runtime(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
