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

# These dtypes are optional in the torch versions the project supports
# (>=2.0), and the case list below dereferences them while collecting — before
# the a5 platform marker could skip. Guard at import, like test_matmul_mx.py.
_REQUIRED_TORCH_DTYPES = ("float4_e2m1fn_x2", "float8_e4m3fn", "float8_e8m0fnu")
if not all(hasattr(torch, name) for name in _REQUIRED_TORCH_DTYPES):
    pytest.skip("torch MXFP4/MXFP8/E8M0 dtypes required", allow_module_level=True)

MX_GROUP_SIZE = 32
# One MX scale box is 16 rows x 2 groups of exponent codes.
_SCALE_BOX_ROWS = 16
_SCALE_BOX_COLS = 2


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


# Values an E4M3 block reproduces bit-exactly once its own 2**e scale is divided
# out, so the goldens below can assert equality instead of a tolerance.
_FP8_E4M3_EXACT = torch.tensor(
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


def _quant_inputs_and_golden(
    group_axis: int,
    src_shape: tuple[int, int] | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact MXFP8 blocks with known quantized values and E8M0 scales.

    ``src_shape`` defaults to the small per-axis shape; the larger cases pass one
    that spans several 16x2 scale boxes along both axes. OCP selects
    ``floor(log2(amax)) - 8`` for E4M3, which is exactly ``e`` for a block of
    ``_FP8_E4M3_EXACT * 2**e`` — hence an expected scale code of ``e + 127`` and
    a quantized payload equal to ``_FP8_E4M3_EXACT`` itself.
    """
    fp8_values = _FP8_E4M3_EXACT
    if src_shape is None:
        src_shape, quant_shape = _case_shapes(group_axis)
    else:
        quant_shape = src_shape if group_axis == 1 else (src_shape[1], src_shape[0])
    groups = src_shape[1] // MX_GROUP_SIZE
    exponent_shape = (src_shape[0], groups) if group_axis == 1 else (groups, src_shape[0])
    # Both the exponent and the payload must vary with *both* scale coordinates,
    # or the fixture cannot see a layout bug: a row-major `arange % 4` gives
    # every row the same exponents and every group the same 32 values, so
    # swapping two box rows or two group boxes leaves the bytes identical.
    rows_index = torch.arange(exponent_shape[0]).reshape(-1, 1)
    groups_index = torch.arange(exponent_shape[1]).reshape(1, -1)
    # One exponent per 16x2 box, so no two boxes carry the same scale bytes and
    # a swapped box is visible. `_pack_scale_codes` boxes (row/16, group/2) on
    # axis1 and (group/2, row/16) on axis0, so index the box -- and the lane
    # inside it -- the same way round.
    if group_axis == 1:
        box_index = (rows_index // _SCALE_BOX_ROWS) * (
            (exponent_shape[1] + _SCALE_BOX_COLS - 1) // _SCALE_BOX_COLS
        ) + (groups_index // _SCALE_BOX_COLS)
        lane_in_box = (rows_index % _SCALE_BOX_ROWS) * _SCALE_BOX_COLS + (groups_index % _SCALE_BOX_COLS)
    else:
        box_index = (groups_index // _SCALE_BOX_ROWS) * (
            (exponent_shape[0] + _SCALE_BOX_COLS - 1) // _SCALE_BOX_COLS
        ) + (rows_index // _SCALE_BOX_COLS)
        lane_in_box = (groups_index % _SCALE_BOX_ROWS) * _SCALE_BOX_COLS + (rows_index % _SCALE_BOX_COLS)
    # FP16 bounds the range: the largest payload is 448, so 448 * 2**e must stay
    # finite (e <= 7), and the smallest is 2**-6.
    group_exponents = (box_index % 12) - 4
    scales = torch.pow(2.0, group_exponents).to(torch.float32)
    # Within a box every lane shares the exponent, so the payload rotation is
    # the only thing that can distinguish them -- and it has to be *injective*
    # over the box, not merely varying. `(row * 5 + group * 11) % 32` was not:
    # for axis1 (row=15, group=0) and (row=0, group=1) both rotate by 11, which
    # made those two blocks byte-identical and a swap between them invisible.
    # `lane_in_box` is the 16x2 lane index, so it runs 0..31 exactly once per
    # box and `box_index + lane_in_box` is a permutation of the 32 rotations
    # within every box, while still differing between boxes. Rotation is itself
    # a permutation of the block, so each block's amax stays 448 and OCP still
    # selects exactly `e`.
    rotation = (box_index + lane_in_box) % MX_GROUP_SIZE
    lane = torch.arange(MX_GROUP_SIZE)
    if group_axis == 1:
        # [rows, groups, 32] payload, rotated per group.
        values = fp8_values[(lane.reshape(1, 1, -1) - rotation.unsqueeze(-1)) % MX_GROUP_SIZE]
        src = (values * scales.unsqueeze(-1)).reshape(src_shape)
        quantized = values.to(torch.float8_e4m3fn).reshape(src_shape)
    else:
        # [groups, 32, N]: the group axis is outermost in the quantized result.
        values = fp8_values[(lane.reshape(1, -1, 1) - rotation.unsqueeze(1)) % MX_GROUP_SIZE]
        src_kn = (values * scales.unsqueeze(1)).reshape(quant_shape)
        src = src_kn.transpose(0, 1)
        quantized = values.to(torch.float8_e4m3fn).reshape(quant_shape)
    scale = _pack_scale_codes((group_exponents + 127).to(torch.uint8), group_axis)
    return src.to(dtype).contiguous(), quantized.contiguous(), scale.contiguous()


def _fp8_case_shapes(group_axis: int, src_shape: tuple[int, int]) -> tuple[int, int]:
    """Quantized shape for a source of ``src_shape`` on ``group_axis``."""
    rows, cols = src_shape
    return (rows, cols) if group_axis == 1 else (cols, rows)


def _make_fp8_quant(group_axis: int, src_shape: tuple[int, int], dtype, algorithm: str):
    """Quantize exactly representable blocks through one public MX path."""
    rows, cols = src_shape
    quant_rows, quant_cols = _fp8_case_shapes(group_axis, src_shape)
    scale_elements = rows * cols // MX_GROUP_SIZE

    @pl.jit(auto_scope=False)
    def quantize(
        src: pl.Tensor[[rows, cols], dtype],
        out_quant: pl.Out[pl.Tensor[[quant_rows, quant_cols], pl.FP8E4M3FN]],
        out_scale: pl.Out[pl.Tensor[[1, scale_elements], pl.FP8E8M0]],
    ):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            values = pl.tile.load(src, [block * rows, 0], [rows, cols])
            quantized, scale = pl.quant_mx(values, group_axis=group_axis, quant_scale_alg=algorithm)
            pl.tile.store(quantized, [0, 0], out_quant)
            pl.tile.store(pl.reshape(scale, [1, scale_elements]), [0, 0], out_scale)
        return out_quant, out_scale

    return quantize


def _fp8_quant_golden(group_axis: int, src_shape: tuple[int, int], torch_dtype):
    def golden(tensors):
        _src, quant, scale = _quant_inputs_and_golden(group_axis, src_shape, torch_dtype)
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


_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _decode_packed_e2m1(payload: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 carriers to float, low nibble first.

    The inverse of the packing the encoder golden performs: byte ``b`` holds the
    even column in its low nibble and the odd column in its high nibble, bit 3
    is the sign and bits 0-2 index the eight E2M1 magnitudes.
    """
    carriers = payload.view(torch.uint8)
    nibbles = torch.stack((carriers & 0x0F, carriers >> 4), dim=-1).flatten(-2)
    magnitude = _E2M1_LEVELS[(nibbles & 0x07).long()]
    return torch.where((nibbles & 0x08) != 0, -magnitude, magnitude)


class _Fp4EncodeAndDequantReference:
    """Check the FP4 payload twice: against the encoder, and by decoding it.

    The encoder golden reimplements block scaling and nearest-even E2M1
    rounding, so a mistake shared with the implementation would cancel. This
    adds the opposite direction: decode what the device actually wrote, scale it
    by the exponent the device actually chose, and require the product to equal
    the source **exactly**. That holds because every source value is an E2M1
    level times a power of two, and it shares no arithmetic with the encoder.

    Axis1 only, and only for a single 16x2 scale box, where the packed byte
    order is plain row-major so the exponents can be read back with a reshape.
    The axis0 box order needs the inverse of ``_pack_scale_codes``; its byte
    comparison against the encoder golden already covers that path.
    """

    def __init__(self, rows: int, groups: int, torch_dtype):
        assert rows == 16, "the packed order is row-major only for one box row"
        self._rows = rows
        self._groups = groups
        self._torch_dtype = torch_dtype
        self._encoder = _fp4_quant_golden(1, "ocp")

    def golden(self, tensors):
        return self._encoder(tensors)

    def compare(self, actual, expected):
        _compare_quant_bytes(actual, expected)
        # Rebuild the source rather than remembering it from `golden`: with
        # `--save-kernels` into an existing directory the harness reuses the
        # persisted goldens and never calls `golden` in this process, so any
        # state stashed there would still be unset here.
        source = _fp4_quant_source(1, self._torch_dtype).float()
        decoded = _decode_packed_e2m1(actual["out_quant"])
        codes = actual["out_scale"].view(torch.uint8).reshape(self._rows, self._groups)
        scales = torch.pow(2.0, codes.float() - 127.0)
        rebuilt = decoded.reshape(self._rows, self._groups, MX_GROUP_SIZE) * scales.unsqueeze(-1)
        blocks = source.reshape(self._rows, self._groups, MX_GROUP_SIZE)
        limit = (_E2M1_LEVELS[-1] * scales).unsqueeze(-1)
        expected_values = torch.clamp(blocks, -limit, limit)
        wrong = rebuilt != expected_values
        assert not bool(wrong.any()), (
            "dequantized payload does not reproduce the source: first mismatch at "
            f"{torch.nonzero(wrong)[0].tolist()} (got {rebuilt[wrong][0].item()}, "
            f"want {expected_values[wrong][0].item()})"
        )


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


# One reference per axis1 source dtype, so each instance can rebuild that
# dtype's source in `compare` without depending on `golden` having run.
_FP4_AXIS1_REFERENCES = {
    pl.BF16: _Fp4EncodeAndDequantReference(16, 2, torch.bfloat16),
    pl.FP16: _Fp4EncodeAndDequantReference(16, 2, torch.float16),
}


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
            # Axis1 additionally gets the decode-side reference; see the class.
            golden=(_FP4_AXIS1_REFERENCES[dtype].golden if axis == 1 else _fp4_quant_golden(axis, algorithm)),
            compare=(_FP4_AXIS1_REFERENCES[dtype].compare if axis == 1 else _compare_quant_bytes),
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
#
# The matrix covers every source dtype on both grouping axes at 64x128, which
# spans four 16x2 scale box rows and two box columns, so a wrong box order or
# row pitch cannot pass by coincidence. The NV rows are path coverage, not a
# numerical discriminator: OCP picks `floor(log2(amax)) - 8` and NV
# `ceil(log2(amax / 448))`, which agree for every block here because each
# block's amax is `448 * 2**e`. They differ only when amax's mantissa exceeds
# 1.75, which within a binade means amax > 448 and OCP clipping — so the
# algorithms are separated numerically by the FP4 axis0 NV cases above (amax 7
# against an E2M1 max of 6), while these rows prove the MXFP8 NV instantiation
# runs on device and is bit-identical where it should be.
_FP8_SRC_SHAPE = (64, 128)
_FP8_DTYPES = (
    (pl.FP32, torch.float32, DataType.FP32),
    (pl.FP16, torch.float16, DataType.FP16),
    (pl.BF16, torch.bfloat16, DataType.BF16),
)


def _fp8_scale_elements(src_shape: tuple[int, int]) -> int:
    return src_shape[0] * src_shape[1] // MX_GROUP_SIZE


def _fp8_output_spec(name: str, shape: tuple[int, int], dtype: DataType, torch_dtype) -> TensorSpec:
    return TensorSpec(
        name,
        list(shape),
        dtype,
        init_value=lambda: torch.zeros(shape, dtype=torch.uint8).view(torch_dtype),
        is_output=True,
    )


def _fp8_case(axis: int, dtype, torch_dtype, data_dtype, algorithm: str):
    src_shape = _FP8_SRC_SHAPE
    elements = _fp8_scale_elements(src_shape)
    return st.case(
        _make_fp8_quant(axis, src_shape, dtype, algorithm),
        tensors=[
            TensorSpec(
                "src",
                list(src_shape),
                data_dtype,
                init_value=lambda: _quant_inputs_and_golden(axis, src_shape, torch_dtype)[0],
            ),
            _fp8_output_spec(
                "out_quant",
                _fp8_case_shapes(axis, src_shape),
                DataType.FP8E4M3FN,
                torch.float8_e4m3fn,
            ),
            _fp8_output_spec("out_scale", (1, elements), DataType.FP8E8M0, torch.float8_e8m0fnu),
        ],
        name=f"quant_mx_axis{axis}_{data_dtype.name.lower()}_{algorithm}",
        golden=_fp8_quant_golden(axis, src_shape, torch_dtype),
        compare=_compare_quant_bytes,
    )


@pytest.mark.platforms("a5")
@st.cases(
    # OCP on both axes for all three source dtypes.
    *[
        _fp8_case(axis, dtype, torch_dtype, data_dtype, "ocp")
        for dtype, torch_dtype, data_dtype in _FP8_DTYPES
        for axis in (1, 0)
    ],
    # NV where PTOAS allows it: axis0 for every dtype, axis1 for FP32 only.
    *[_fp8_case(0, dtype, torch_dtype, data_dtype, "nv") for dtype, torch_dtype, data_dtype in _FP8_DTYPES],
    _fp8_case(1, pl.FP32, torch.float32, DataType.FP32, "nv"),
)
def test_fp8_quant_runtime(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
