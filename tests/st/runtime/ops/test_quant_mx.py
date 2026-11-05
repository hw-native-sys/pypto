# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-board A5 runtime tests for MX quantization."""

from typing import Any

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.runtime.runner import RunConfig

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


# The 32 values an E4M3 group can carry with no rounding at all, so a quantized
# payload compares byte for byte against the source divided by its scale.
_FP8_EXACT_VALUES = torch.tensor(
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


def _quant_inputs_and_golden(group_axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact MXFP8 blocks with known quantized values and E8M0 scales."""
    src_shape, quant_shape = _case_shapes(group_axis)
    groups = src_shape[1] // MX_GROUP_SIZE
    exponent_shape = (src_shape[0], groups) if group_axis == 1 else (groups, src_shape[0])
    group_exponents = (torch.arange(exponent_shape[0] * exponent_shape[1]) % 4 - 1).reshape(exponent_shape)
    scales = torch.pow(2.0, group_exponents).to(torch.float32)
    if group_axis == 1:
        src = (_FP8_EXACT_VALUES.reshape(1, 1, MX_GROUP_SIZE) * scales.unsqueeze(-1)).reshape(src_shape)
        quantized = _FP8_EXACT_VALUES.to(torch.float8_e4m3fn).repeat(src_shape[0], groups)
    else:
        src_kn = (_FP8_EXACT_VALUES.reshape(1, MX_GROUP_SIZE, 1) * scales.unsqueeze(1)).reshape(quant_shape)
        src = src_kn.transpose(0, 1)
        quantized = (
            _FP8_EXACT_VALUES.to(torch.float8_e4m3fn)
            .reshape(1, MX_GROUP_SIZE, 1)
            .expand(groups, MX_GROUP_SIZE, src_shape[0])
            .reshape(quant_shape)
        )
    scale = _pack_scale_codes((group_exponents + 127).to(torch.uint8), group_axis)
    return src.contiguous(), quantized.contiguous(), scale.contiguous()


class TestQuantMx(PTOTestCase):
    """Quantize FP32 blocks through either public MX grouping path."""

    __test__ = False

    def __init__(self, group_axis: int, *, platform: str | None = None):
        super().__init__(RunConfig(rtol=0, atol=0), platform=platform)
        self.group_axis = group_axis
        self.src_shape, self.quant_shape = _case_shapes(group_axis)
        self.scale_elements = self.src_shape[0] * self.src_shape[1] // MX_GROUP_SIZE

    def get_name(self) -> str:
        return f"quant_mx_axis{self.group_axis}_{self.src_shape[0]}x{self.src_shape[1]}"

    def define_tensors(self) -> list[TensorSpec]:
        src, _expected_quant, _expected_scale = _quant_inputs_and_golden(self.group_axis)
        return [
            TensorSpec("src", list(self.src_shape), DataType.FP32, init_value=src),
            TensorSpec("out_quant", list(self.quant_shape), DataType.FP8E4M3FN, is_output=True),
            TensorSpec(
                "out_scale",
                [1, self.scale_elements],
                DataType.FP8E8M0,
                is_output=True,
            ),
        ]

    def get_program(self) -> Any:
        src_rows, src_cols = self.src_shape
        quant_rows, quant_cols = self.quant_shape
        scale_elements = self.scale_elements
        group_axis = self.group_axis

        @pl.program
        class QuantMxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def quant_mx(
                self,
                src: pl.Tensor[[src_rows, src_cols], pl.FP32],
                out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.FP8E4M3FN]],
                out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.FP8E8M0]],
            ):
                src_tile = pl.load(src, [0, 0], [src_rows, src_cols])
                quantized, scale = pl.quant_mx(src_tile, group_axis=group_axis)
                out_quant = pl.store(quantized, [0, 0], out_quant)
                out_scale = pl.store(pl.reshape(scale, [1, scale_elements]), [0, 0], out_scale)
                return out_quant, out_scale

            @pl.function(type=pl.FunctionType.Orchestration)
            def orchestrator(
                self,
                src: pl.Tensor[[src_rows, src_cols], pl.FP32],
                out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.FP8E4M3FN]],
                out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.FP8E8M0]],
            ):
                out_quant, out_scale = self.quant_mx(src, out_quant, out_scale)
                return out_quant, out_scale

        return QuantMxProgram

    def compute_expected(self, tensors: dict[str, torch.Tensor], params=None) -> None:
        _src, expected_quant, expected_scale = _quant_inputs_and_golden(self.group_axis)
        tensors["out_quant"][:] = expected_quant
        tensors["out_scale"][:] = expected_scale


@pytest.mark.platforms("a5")
class TestQuantMxOperations:
    """Numerical execution coverage for the Ascend950-only quantization ops."""

    @pytest.mark.parametrize(
        ("platform", "group_axis"),
        [pytest.param("a5", 1, id="a5-axis1"), pytest.param("a5", 0, id="a5-axis0")],
    )
    def test_quant_mx(self, test_runner, platform, group_axis):
        result = test_runner.run(TestQuantMx(group_axis, platform=platform))
        assert result.passed, f"Test failed: {result.error}"


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


# Writing the scale into a tensor that carries its MX layout exercises the
# store's own ND descriptor and its flat offset arithmetic: the destination is
# the blocked SFractal box view, so a wrong stride, box order or block offset
# lands the bytes in the wrong place while still compiling.
#
# Discriminating power is the whole point here, and it does not come for free.
# An earlier form of this case built its exponents from `arange % 4`, which gave
# a 16x2 box only TWO distinct row signatures out of 16 — so 56 of the 120
# row swaps inside a box were byte-identical, and the exact comparison was blind
# to them. Measured on silicon: swapping rows 0 and 2 of the golden still
# PASSED. `_distinct_group_exponents` / `_distinct_group_payload` below give
# every (row, group) its own exponent and its own payload permutation instead,
# and `_assert_construction_is_discriminating` counts the signatures rather than
# arguing about them.


def _distinct_group_exponents(rows: int, groups: int) -> torch.Tensor:
    """A distinct E8M0 exponent for every (row, group) of a scale tile.

    Centred on 0 so the codes stay well inside E8M0's `[0, 254]`. The payload
    values are exact powers of two times a group maximum of 448, so any
    exponent whose scaled extremes stay in normal FP32 reproduces bit-exactly;
    +-32 is far inside that.
    """
    return torch.arange(rows * groups).reshape(rows, groups) - (rows * groups) // 2


def _distinct_group_payload(index: int) -> torch.Tensor:
    """A distinct exact-E4M3 permutation of one 32-value group.

    A roll alone yields only 32 permutations, which repeat once a tile holds
    more than 32 groups and would leave two boxes byte-identical. Composing the
    roll with two independent involutions (reverse, swap adjacent pairs) raises
    that to 128 while keeping every value exactly representable, so the group
    maximum — and therefore the chosen exponent — is unchanged. Distinctness is
    asserted, not assumed: see `_assert_construction_is_discriminating`.
    """
    payload = torch.roll(_FP8_EXACT_VALUES, index % MX_GROUP_SIZE)
    variant = index // MX_GROUP_SIZE
    if variant & 1:
        payload = payload.flip(0)
    if variant & 2:
        payload = payload.reshape(MX_GROUP_SIZE // 2, 2).flip(1).reshape(-1)
    return payload.contiguous()


def _assert_construction_is_discriminating(payload: torch.Tensor, codes: torch.Tensor) -> None:
    """Fail loudly if the fixture stops telling the layout cases apart.

    Counts the distinct signatures the construction actually produces, per
    component: an aggregate count hides a component that discriminates nothing,
    and a swap confined to that component would then sail through.
    """
    groups = payload.reshape(-1, MX_GROUP_SIZE)
    distinct_payload = len({tuple(row.tolist()) for row in groups})
    distinct_codes = len(set(codes.flatten().tolist()))
    assert distinct_payload == groups.shape[0], (
        f"payload groups are not all distinct: {distinct_payload} of {groups.shape[0]}"
    )
    assert distinct_codes == codes.numel(), (
        f"scale codes are not all distinct: {distinct_codes} of {codes.numel()}"
    )


def _mx_layout_case(
    rows: int, groups: int, group_axis: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source, exact quantized payload, and logical scale codes for one case.

    ``rows`` x ``groups`` are the scale's logical extents. For ``group_axis=1``
    (A-ZZ) a scale row is a source row; for ``group_axis=0`` (B-NN) a scale
    column is a source column and the groups run down K.
    """
    exponents = _distinct_group_exponents(rows, groups)
    payloads = torch.stack([_distinct_group_payload(index) for index in range(rows * groups)]).reshape(
        rows, groups, MX_GROUP_SIZE
    )
    _assert_construction_is_discriminating(payloads, exponents + 127)
    scaled = payloads * torch.pow(2.0, exponents).to(torch.float32).unsqueeze(-1)
    # Both axes take the same [N or M, K] source; only the quantized result's
    # orientation differs. `tile.tquant_mx` always reads K as the tile's second
    # dimension, and with group_axis=0 it returns the payload transposed as
    # [K, N] (the shape `_case_shapes(0)` also encodes).
    src = scaled.reshape(rows, groups * MX_GROUP_SIZE)
    if group_axis == 1:
        quantized = payloads.reshape(rows, groups * MX_GROUP_SIZE)
    else:
        quantized = payloads.permute(1, 2, 0).reshape(groups * MX_GROUP_SIZE, rows)
    return (
        src.contiguous(),
        quantized.to(torch.float8_e4m3fn).contiguous(),
        (exponents + 127).to(torch.uint8).contiguous(),
    )


def _mx_layout_golden(rows: int, groups: int, group_axis: int, scale_shape: list[int]):
    """Golden for an MX-layout scale store: exact payload plus packed scale."""

    def golden(tensors):
        _src, quantized, codes = _mx_layout_case(rows, groups, group_axis)
        packed = _pack_scale_codes(codes if group_axis == 1 else codes.T, group_axis)
        return {
            "out_quant": quantized,
            # Hand back the declared dtype: the harness assigns the golden into
            # the output buffer, so a uint8 code would be converted as a
            # *number* (126 -> the nearest E8M0 power of two) rather than kept
            # as a byte.
            "out_scale": packed.reshape(scale_shape),
        }

    return golden


# --- MX_A_ZZ: two blocks, one 16x2 scale box row each -----------------------
@pl.jit(auto_scope=False)
def _mx_a_zz_scale_store(
    src: pl.Tensor[[32, 64], pl.FP32],
    out_quant: pl.Out[pl.Tensor[[32, 64], pl.FP8E4M3FN]],
    out_scale: pl.Out[pl.Tensor[[32, 2], pl.FP8E8M0, pl.MX_A_ZZ]],
):
    with pl.spmd(2):
        row = pl.tile.get_block_idx() * 16
        values = pl.tile.load(src, [row, 0], [16, 64])
        quantized, scale = pl.quant_mx(values, group_axis=1)
        pl.tile.store(quantized, [row, 0], out_quant)
        pl.tile.store(scale, [row, 0], out_scale)
    return out_quant, out_scale


# --- MX_B_NN: two blocks, 32 columns of N each ------------------------------
# A B-NN scale is [K/32, N] and its boxes run along N, so the blocks split N.
# `tile.tquant_mx` reads K as the tile's second dimension on both axes, so the
# source stays [N, K] while the quantized payload comes back as [K, N].
# Nothing else in this file stores a B-NN scale on a card: the MX_B_NN codegen
# cases are assembly-only, which proves the descriptors parse, not that the
# bytes land where the layout says.
@pl.jit(auto_scope=False)
def _mx_b_nn_scale_store(
    src: pl.Tensor[[64, 64], pl.FP32],
    out_quant: pl.Out[pl.Tensor[[64, 64], pl.FP8E4M3FN]],
    out_scale: pl.Out[pl.Tensor[[2, 64], pl.FP8E8M0, pl.MX_B_NN]],
):
    with pl.spmd(2):
        col = pl.tile.get_block_idx() * 32
        values = pl.tile.load(src, [col, 0], [32, 64])
        quantized, scale = pl.quant_mx(values, group_axis=0)
        pl.tile.store(quantized, [0, col], out_quant)
        pl.tile.store(scale, [0, col], out_scale)
    return out_quant, out_scale


def _mx_layout_source(rows: int, groups: int, group_axis: int):
    def source() -> torch.Tensor:
        src, _quantized, _codes = _mx_layout_case(rows, groups, group_axis)
        return src

    return source


def _mx_layout_tensors(src_shape, quant_shape, scale_shape, rows, groups, group_axis):
    return [
        TensorSpec("src", src_shape, DataType.FP32, init_value=_mx_layout_source(rows, groups, group_axis)),
        TensorSpec(
            "out_quant",
            quant_shape,
            DataType.FP8E4M3FN,
            init_value=lambda: torch.zeros(tuple(quant_shape), dtype=torch.uint8).view(torch.float8_e4m3fn),
            is_output=True,
        ),
        TensorSpec(
            "out_scale",
            scale_shape,
            DataType.FP8E8M0,
            init_value=lambda: torch.zeros(tuple(scale_shape), dtype=torch.uint8).view(torch.float8_e8m0fnu),
            is_output=True,
        ),
    ]


@pytest.mark.platforms("a5")
@st.cases(
    st.case(
        _mx_a_zz_scale_store,
        tensors=_mx_layout_tensors([32, 64], [32, 64], [32, 2], 32, 2, 1),
        name="quant_mx_scale_into_mx_a_zz",
        golden=_mx_layout_golden(32, 2, 1, [32, 2]),
        compare=_compare_quant_bytes,
    ),
    st.case(
        _mx_b_nn_scale_store,
        tensors=_mx_layout_tensors([64, 64], [64, 64], [2, 64], 64, 2, 0),
        name="quant_mx_scale_into_mx_b_nn",
        golden=_mx_layout_golden(64, 2, 0, [2, 64]),
        compare=_compare_quant_bytes,
    ),
)
def test_mx_layout_scale_store_runtime(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
