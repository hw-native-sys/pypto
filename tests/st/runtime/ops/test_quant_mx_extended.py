# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 numerical tests for canonical exponent tiles and E5M2 quantization."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec

if not all(hasattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2")):
    pytest.skip("torch FP8 types required", allow_module_level=True)

MX_GROUP_SIZE = 32


def _make_e5m2_quant(axis: int):
    rows = 16 if axis == 1 else 32
    q_rows, q_cols = (rows, 64) if axis == 1 else (64, rows)

    @pl.jit(auto_scope=False)
    def quantize(
        src: pl.Tensor[[rows, 64], pl.FP32],
        out_quant: pl.Out[pl.Tensor[[q_rows, q_cols], pl.FP8E5M2]],
        out_scale: pl.Out[pl.Tensor[[1, rows * 2], pl.UINT8]],
    ):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            values = pl.tile.load(src, [block * rows, 0], [rows, 64])
            quantized, scale = pl.quant_mx(values, group_axis=axis, dtype=pl.FP8E5M2)
            pl.tile.store(quantized, [0, 0], out_quant)
            scale_bytes = pl.reinterpret_view(pl.reshape(scale, [1, rows * 2]), pl.UINT8)
            pl.tile.store(scale_bytes, [0, 0], out_scale)
        return out_quant, out_scale

    return quantize


def _e5m2_source(rows: int) -> torch.Tensor:
    # Each block contains a distinct E5M2 maximum and a rotation of the
    # payload so an incorrect exponent or scale-box order changes the output.
    levels = torch.tensor([-57344, -32768, -64, -1, -0.5, 0, 1, 64], dtype=torch.float32)
    block = torch.arange(rows * 2).reshape(rows, 2)
    lane = torch.arange(MX_GROUP_SIZE).reshape(1, 1, -1)
    values = levels[(lane + block.unsqueeze(-1)) % len(levels)]
    values = values * torch.pow(2.0, (block % 13 - 6).unsqueeze(-1))
    values[0, 0] = 0
    values[0, 1, ::2] = 2.0**-130
    values[0, 1, 1::2] = -(2.0**-130)
    values[1, 0, 7] = float("nan")
    values[1, 1, 15] = torch.finfo(torch.float32).max
    return values.reshape(rows, 64)


def _e5m2_golden(axis: int):
    def golden(tensors):
        rows = tensors["src"].shape[0]
        blocks = tensors["src"].double().reshape(rows, 2, MX_GROUP_SIZE)
        maximum = blocks.abs().amax(dim=-1)
        # frexp is exact at powers of two, including the largest finite FP32.
        # Double-precision ldexp keeps the reference independent of the
        # device's reciprocal-scale encoding and covers FP32 subnormals.
        _, binary_exp = torch.frexp(maximum)
        exponent = (binary_exp - 1 - 15).clamp_min(-127)
        exponent = torch.where(maximum == 0, -127, exponent)
        code = torch.where(maximum.isnan(), 255, exponent + 127).to(torch.uint8)
        normalized = torch.ldexp(blocks, -exponent.unsqueeze(-1))
        normalized = torch.where(maximum.isnan().unsqueeze(-1), float("nan"), normalized)
        payload = normalized.clamp(-57344, 57344).reshape(rows, 64)
        if axis == 0:
            payload = payload.transpose(0, 1)
        # K=64 gives a single group pair, so both ZZ and the transposed
        # NN orientation physically store the two codes for each source row
        # together, in row order.
        return {
            "out_quant": payload.contiguous().to(torch.float8_e5m2),
            "out_scale": code.reshape(1, -1),
        }

    return golden


def _compare_e5m2_quant(actual, expected):
    assert torch.equal(actual["out_scale"], expected["out_scale"])
    expected_nan = expected["out_quant"].float().isnan()
    assert torch.equal(actual["out_quant"].float().isnan(), expected_nan)
    actual_bytes = actual["out_quant"].view(torch.uint8)
    expected_bytes = expected["out_quant"].view(torch.uint8)
    mismatch = (actual_bytes != expected_bytes) & ~expected_nan
    assert not bool(mismatch.any()), (
        f"first E5M2 byte mismatch at {mismatch.nonzero()[0].tolist()}: "
        f"actual={actual_bytes[mismatch][0].item():#04x}, "
        f"expected={expected_bytes[mismatch][0].item():#04x}"
    )


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_e5m2_quant(axis),
            tensors=[
                TensorSpec("src", [rows, 64], DataType.FP32, init_value=lambda rows=rows: _e5m2_source(rows)),
                TensorSpec(
                    "out_quant",
                    [rows, 64] if axis == 1 else [64, rows],
                    DataType.FP8E5M2,
                    is_output=True,
                ),
                TensorSpec(
                    "out_scale",
                    [1, rows * 2],
                    DataType.UINT8,
                    init_value=lambda rows=rows: torch.empty((1, rows * 2), dtype=torch.uint8),
                    is_output=True,
                ),
            ],
            name=f"e5m2_quant_axis{axis}",
            golden=_e5m2_golden(axis),
            compare=_compare_e5m2_quant,
        )
        for axis, rows in ((1, 16), (0, 32))
    ]
)
def test_e5m2_quantization(case_run):
    case_run.assert_passed()


def _make_tensor_e5m2_quant(axis: int):
    rows = 16 if axis == 1 else 32
    q_shape = (rows, 64) if axis == 1 else (64, rows)
    scale_shape = (rows, 2) if axis == 1 else (2, rows)
    scale_layout = pl.MX_A_ZZ if axis == 1 else pl.MX_B_NN

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def quantize(
            self, src: pl.Tensor[[rows, 64], pl.FP32]
        ) -> tuple[pl.Tensor[q_shape, pl.FP8E5M2], pl.Tensor[scale_shape, pl.FP8E8M0, scale_layout]]:
            quant, scale = pl.quant_mx(src, group_axis=axis, dtype=pl.FP8E5M2)
            return quant, scale

        @pl.function(type=pl.FunctionType.InCore)
        def copy_scale(
            self, src: pl.Tensor[[1, rows * 2], pl.FP8E8M0], dst: pl.Out[pl.Tensor[[1, rows * 2], pl.UINT8]]
        ) -> pl.Tensor[[1, rows * 2], pl.UINT8]:
            values = pl.load(src, [0, 0], [1, rows * 2])
            return pl.store(pl.reinterpret_view(values, pl.UINT8), [0, 0], dst)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            src: pl.Tensor[[rows, 64], pl.FP32],
            out_quant: pl.Out[pl.Tensor[q_shape, pl.FP8E5M2]],
            out_scale: pl.Out[pl.Tensor[[1, rows * 2], pl.UINT8]],
        ) -> tuple[pl.Tensor[q_shape, pl.FP8E5M2], pl.Tensor[[1, rows * 2], pl.UINT8]]:
            quant, scale = self.quantize(src)
            out_quant = pl.assemble(out_quant, quant, [0, 0])
            # Read back the returned MX tensor's exact physical bytes without
            # requiring a host-side torch FP8E8M0 dtype.
            scale_bytes = pl.tensor.view(scale, shape=[1, rows * 2], layout=pl.ND)
            out_scale = self.copy_scale(scale_bytes, out_scale)
            return out_quant, out_scale

    return Program


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_tensor_e5m2_quant(axis),
            tensors=[
                TensorSpec("src", [rows, 64], DataType.FP32, init_value=lambda rows=rows: _e5m2_source(rows)),
                TensorSpec(
                    "out_quant", [rows, 64] if axis == 1 else [64, rows], DataType.FP8E5M2, is_output=True
                ),
                TensorSpec(
                    "out_scale",
                    [1, rows * 2],
                    DataType.UINT8,
                    init_value=lambda rows=rows: torch.zeros((1, rows * 2), dtype=torch.uint8),
                    is_output=True,
                ),
            ],
            name=f"tensor_e5m2_quant_axis{axis}",
            golden=_e5m2_golden(axis),
            compare=_compare_e5m2_quant,
        )
        for axis, rows in ((1, 16), (0, 32))
    ]
)
def test_tensor_e5m2_quantization(case_run):
    case_run.assert_passed()


@pl.jit(auto_scope=False)
def _canonical_axis1_quant(
    src: pl.Tensor[[16, 128], pl.FP32],
    out_quant: pl.Out[pl.Tensor[[16, 128], pl.INT8]],
    out_exp: pl.Out[pl.Tensor[[16, 32], pl.UINT8]],
):
    with pl.spmd(1):
        block = pl.tile.get_block_idx()
        values = pl.tile.load(src, [block * 16, 0], [16, 128])
        valid = pl.tile.set_validshape(values, 16, 64)
        amax = pl.tile.create([1, 32], pl.FP32)
        scaling = pl.tile.create([1, 32], pl.FP32)
        quant, exp = pl.tile.quant_mx_raw(valid, amax, scaling, group_axis=1, canonical_axis1=True)
        pl.tile.store(quant, [0, 0], out_quant)
        pl.tile.store(exp, [0, 0], out_exp)
    return out_quant, out_exp


def _canonical_blocks():
    q = torch.tensor([-448, -256, -16, -1, 0, 1, 16, 448], dtype=torch.float32)
    block = torch.arange(32).reshape(16, 2)
    lane = torch.arange(MX_GROUP_SIZE).reshape(1, 1, -1)
    normalized = q[(lane + block.unsqueeze(-1)) % len(q)]
    return normalized, (block % 15) - 7


def _canonical_source():
    normalized, exponent = _canonical_blocks()
    source = torch.full((16, 128), 1.0e20)
    source[:, :64] = torch.ldexp(normalized, exponent.unsqueeze(-1)).reshape(16, 64)
    return source


def _canonical_golden(_tensors):
    normalized, exponent = _canonical_blocks()
    quant = torch.zeros((16, 128), dtype=torch.int8)
    exp = torch.zeros((16, 32), dtype=torch.uint8)
    quant[:, :64] = normalized.reshape(16, 64).to(torch.float8_e4m3fn).view(torch.int8)
    exp[:, :2] = (exponent + 127).to(torch.uint8)
    return {"out_quant": quant, "out_exp": exp}


def _compare_canonical(actual, expected):
    assert torch.equal(actual["out_quant"][:, :64], expected["out_quant"][:, :64])
    assert torch.equal(actual["out_exp"][:, :2], expected["out_exp"][:, :2])


@pytest.mark.platforms("a5")
@st.cases(
    st.case(
        _canonical_axis1_quant,
        tensors=[
            TensorSpec("src", [16, 128], DataType.FP32, init_value=_canonical_source),
            TensorSpec("out_quant", [16, 128], DataType.INT8, is_output=True),
            TensorSpec(
                "out_exp",
                [16, 32],
                DataType.UINT8,
                init_value=lambda: torch.empty((16, 32), dtype=torch.uint8),
                is_output=True,
            ),
        ],
        name="canonical_axis1_quant",
        golden=_canonical_golden,
        compare=_compare_canonical,
    )
)
def test_canonical_axis1_quantization(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
