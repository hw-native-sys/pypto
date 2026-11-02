# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 packed FP4 transfers and ND numerical casts."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec

_HAS_TORCH_FP4 = hasattr(torch, "float4_e2m1fn_x2")


_TORCH_FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", torch.uint8)


ROWS, PACKED_K = 2, 256


def _xor_row_fp4() -> torch.Tensor:
    physical = torch.empty((ROWS, PACKED_K), dtype=torch.uint8)
    physical[0] = torch.arange(PACKED_K, dtype=torch.uint8)
    physical[1] = torch.arange(PACKED_K, dtype=torch.uint8).bitwise_xor(0xA5)
    return physical.view(_TORCH_FP4_DTYPE)


@pl.jit.incore
def _fp4_copy_kernel(
    src: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2]],
):
    return pl.store(pl.load(src, [0, 0], [ROWS, PACKED_K]), [0, 0], out)


@pl.jit
def _fp4_copy_entry(
    src: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2]],
):
    return _fp4_copy_kernel(src, out)


@pl.jit.incore
def _fp4_view_copy_kernel(
    src: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2]],
):
    viewed: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2] = pl.tensor.view(src, [ROWS, PACKED_K])
    return pl.store(pl.load(viewed, [0, 0], [ROWS, PACKED_K]), [0, 0], out)


@pl.jit
def _fp4_view_copy_entry(
    src: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2]],
):
    return _fp4_view_copy_kernel(src, out)


@pl.jit.incore
def _fp4_offset_copy_kernel(
    src: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[ROWS, PACKED_K // 2], pl.fp4e2m1x2]],
):
    tile = pl.load(src, [0, PACKED_K // 4], [ROWS, PACKED_K // 2])
    return pl.store(tile, [0, 0], out)


@pl.jit
def _fp4_offset_copy_entry(
    src: pl.Tensor[[ROWS, PACKED_K], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[ROWS, PACKED_K // 2], pl.fp4e2m1x2]],
):
    return _fp4_offset_copy_kernel(src, out)


@pl.jit.incore
def _fp4_copy_torch_default_kernel(src: pl.Tensor, out: pl.Out[pl.Tensor]):
    return pl.store(pl.load(src, [0, 0], [ROWS, PACKED_K]), [0, 0], out)


@pl.jit
def _fp4_copy_torch_default_entry(src: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _fp4_copy_torch_default_kernel(src, out)


def _copy_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"out": tensors["src"].contiguous().view(torch.uint8).clone().view(_TORCH_FP4_DTYPE)}


def _offset_copy_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    raw = tensors["src"].view(torch.uint8)[:, PACKED_K // 4 : 3 * PACKED_K // 4]
    return {"out": raw.contiguous().view(_TORCH_FP4_DTYPE)}


def _compare_packed_bytes(actual, expected):
    """Compare every carrier byte, including distinct nibble encodings."""
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name].view(torch.uint8), expected[name].view(torch.uint8)), name


def _copy_input():
    return TensorSpec("src", [ROWS, PACKED_K], DataType.FP4E2M1X2, init_value=lambda: _xor_row_fp4())


def _copy_output(columns):
    return TensorSpec(
        "out",
        [ROWS, columns],
        DataType.FP4E2M1X2,
        is_output=True,
        init_value=lambda: torch.empty((ROWS, columns), dtype=_TORCH_FP4_DTYPE),
    )


def _fp4_copy_torch_default_program():
    # This case intentionally derives its contract from native Torch tensors.
    # Construct the samples during program building, after collection.
    return _fp4_copy_torch_default_entry.specialize(
        _xor_row_fp4(), torch.empty((ROWS, PACKED_K), dtype=_TORCH_FP4_DTYPE)
    )


@pytest.mark.skipif(not _HAS_TORCH_FP4, reason="torch.float4_e2m1fn_x2 required")
@pytest.mark.platforms("a5", "a5sim")
@st.cases(
    st.case(
        _fp4_copy_entry,
        tensors=[_copy_input(), _copy_output(PACKED_K)],
        name="fp4_two_row_copy",
        golden=_copy_golden,
        compare=_compare_packed_bytes,
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        _fp4_view_copy_entry,
        tensors=[_copy_input(), _copy_output(PACKED_K)],
        name="fp4_two_row_view_copy",
        golden=_copy_golden,
        compare=_compare_packed_bytes,
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        _fp4_offset_copy_entry,
        tensors=[_copy_input(), _copy_output(PACKED_K // 2)],
        name="fp4_two_row_offset_copy",
        golden=_offset_copy_golden,
        compare=_compare_packed_bytes,
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        _fp4_copy_torch_default_program,
        tensors=[_copy_input(), _copy_output(PACKED_K)],
        name="fp4_two_row_copy_torch_default",
        golden=_copy_golden,
        compare=_compare_packed_bytes,
        rtol=0.0,
        atol=0.0,
    ),
)
def test_fp4_copy_runtime(case_run):
    case_run.assert_passed()


CAST_ROWS, CAST_LOGICAL_K = 4, 64


def _make_fp4_cast_src(shape: tuple[int, int]) -> torch.Tensor:
    rows, cols = shape
    assert cols % 2 == 0
    generator = torch.Generator().manual_seed(41)
    codes = torch.randint(0, 16, (rows, cols), generator=generator).to(torch.uint8)
    codes[1:] = codes[1:].bitwise_xor(0x05)
    packed = ((codes[:, 1::2] & 0x0F) << 4) | (codes[:, 0::2] & 0x0F)
    return packed.contiguous().view(_TORCH_FP4_DTYPE)


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
def _fp4_cast_kernel(
    src: pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K // 2], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]],
):
    return pl.store(pl.cast(pl.load(src, [0, 0], [CAST_ROWS, CAST_LOGICAL_K // 2]), pl.BF16), [0, 0], out)


@pl.jit
def _fp4_cast_entry(
    src: pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K // 2], pl.fp4e2m1x2],
    out: pl.Out[pl.Tensor[[CAST_ROWS, CAST_LOGICAL_K], pl.BF16]],
):
    return _fp4_cast_kernel(src, out)


def _cast_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    decoded = _decode_fp4_data(tensors["src"], CAST_ROWS, CAST_LOGICAL_K)
    return {"out": decoded.to(torch.bfloat16)}


@pytest.mark.skipif(not _HAS_TORCH_FP4, reason="torch.float4_e2m1fn_x2 required")
@pytest.mark.platforms("a5")
@st.cases(
    st.case(
        _fp4_cast_entry,
        tensors=[
            TensorSpec(
                "src",
                [CAST_ROWS, CAST_LOGICAL_K // 2],
                DataType.FP4E2M1X2,
                init_value=lambda: _make_fp4_cast_src((CAST_ROWS, CAST_LOGICAL_K)),
            ),
            TensorSpec(
                "out",
                [CAST_ROWS, CAST_LOGICAL_K],
                DataType.BF16,
                is_output=True,
                init_value=lambda: torch.empty((CAST_ROWS, CAST_LOGICAL_K), dtype=torch.bfloat16),
            ),
        ],
        name="fp4_multi_row_cast_bf16",
        golden=_cast_golden,
        rtol=0.0,
        atol=0.0,
    ),
)
def test_fp4_cast_runtime(case_run):
    case_run.assert_passed()


def _all_fp4_carriers() -> torch.Tensor:
    codes = torch.arange(64, dtype=torch.uint8).reshape(1, 64) % 16
    rows = torch.cat([(codes + row * 3) % 16 for row in range(4)])
    return ((rows[:, 1::2] << 4) | rows[:, 0::2]).contiguous()


def _packed_fp4_values(data: torch.Tensor, dtype: pl.DataType) -> torch.Tensor:
    raw = data.contiguous().view(torch.uint8)
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2).to(torch.long)
    positive = (
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        if dtype == pl.fp4e2m1x2
        else [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    )
    lut = torch.tensor(positive + [-value for value in positive], dtype=torch.bfloat16)
    return lut[codes]


def _make_packed_fp4_decode(dtype):
    @pl.jit(auto_scope=False)
    def decode(src: pl.Tensor[[4, 32], dtype], out: pl.Out[pl.Tensor[[4, 64], pl.BF16]]):
        with pl.spmd(1):
            row = pl.tile.get_block_idx() * 4
            packed = pl.tile.load(src, [row, 0], [4, 32])
            decoded = pl.cast(packed, pl.BF16)
            pl.tile.store(decoded, [row, 0], out)
        return out

    return decode


def _make_packed_fp4_encode(dtype, mode):
    @pl.jit(auto_scope=False)
    def encode(src: pl.Tensor[[4, 64], pl.BF16], out: pl.Out[pl.Tensor[[4, 32], dtype]]):
        with pl.spmd(1):
            row = pl.tile.get_block_idx() * 4
            values = pl.tile.load(src, [row, 0], [4, 64])
            packed = pl.cast(values, dtype, mode=mode)
            pl.tile.store(packed, [row, 0], out)
        return out

    return encode


def _make_packed_fp4_roundtrip(dtype):
    @pl.jit(auto_scope=False)
    def roundtrip(src: pl.Tensor[[4, 32], dtype], out: pl.Out[pl.Tensor[[4, 32], dtype]]):
        with pl.spmd(1):
            row = pl.tile.get_block_idx() * 4
            packed = pl.tile.load(src, [row, 0], [4, 32])
            decoded = pl.cast(packed, pl.BF16)
            encoded = pl.cast(decoded, dtype, mode="rint")
            pl.tile.store(encoded, [row, 0], out)
        return out

    return roundtrip


def _make_packed_fp4_convert(source, target):
    @pl.jit(auto_scope=False)
    def convert(src: pl.Tensor[[4, 32], source], out: pl.Out[pl.Tensor[[4, 32], target]]):
        with pl.spmd(1):
            row = pl.tile.get_block_idx() * 4
            packed = pl.tile.load(src, [row, 0], [4, 32])
            converted = pl.cast(packed, target, mode="rint")
            pl.tile.store(converted, [row, 0], out)
        return out

    return convert


def _packed_decode_golden(dtype):
    def golden(tensors):
        return {"out": _packed_fp4_values(tensors["src"], dtype)}

    return golden


def _packed_encode_values(dtype):
    exact = _packed_fp4_values(_all_fp4_carriers(), dtype).flatten()
    levels = _packed_fp4_values(torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8), dtype).flatten()
    # Quarter steps and ties distinguish all five modes; exact code points
    # retain the complete encoding and signed-zero coverage in the first row.
    interior = torch.cat([levels[:-1] + (levels[1:] - levels[:-1]) * part for part in (0.25, 0.5, 0.75)])
    interior = torch.cat((interior, -interior))
    values = exact.clone()
    values[64:] = interior.repeat(5)[:192]
    return values.reshape(4, 64)


def _packed_encode_golden(dtype, mode):
    def golden(tensors):
        values = tensors["src"].float()
        levels = (
            _packed_fp4_values(torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8), dtype)
            .flatten()
            .float()
        )
        magnitude = values.abs().contiguous()
        upper = torch.searchsorted(levels, magnitude).clamp(max=7)
        lower = (upper - 1).clamp(min=0)
        lower = torch.where(levels[upper] == magnitude, upper, lower)
        negative = torch.signbit(values)
        if mode in ("rint", "round"):
            lower_distance = magnitude - levels[lower]
            upper_distance = levels[upper] - magnitude
            tie_up = lower % 2 != 0 if mode == "rint" else torch.ones_like(negative)
            choose_upper = (upper_distance < lower_distance) | ((upper_distance == lower_distance) & tie_up)
        elif mode == "floor":
            choose_upper = negative
        elif mode == "ceil":
            choose_upper = ~negative
        else:
            choose_upper = torch.zeros_like(negative)
        codes = (torch.where(choose_upper, upper, lower) | (negative.to(torch.long) << 3)).to(torch.uint8)
        return {"out": (codes[:, 1::2] << 4) | codes[:, 0::2]}

    return golden


def _packed_roundtrip_golden(tensors):
    return {"out": tensors["src"].clone()}


def _packed_conversion_golden(source, target):
    def golden(tensors):
        decoded = _packed_fp4_values(tensors["src"], source)
        return _packed_encode_golden(target, "rint")({"src": decoded})

    return golden


def _compare_bf16_bits(actual, expected):
    # Equality on the raw BF16 words preserves the sign of zero as well.
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name].view(torch.int16), expected[name].view(torch.int16)), name


def _packed_input(mask=0xFF):
    # Defer tensor operations until execution: collection precedes forked compilation.
    return TensorSpec("src", [4, 32], DataType.UINT8, init_value=lambda: _all_fp4_carriers() & mask)


def _encoding_input(dtype):
    return TensorSpec("src", [4, 64], DataType.BF16, init_value=lambda: _packed_encode_values(dtype))


def _output(dtype, columns):
    torch_dtype = torch.bfloat16 if dtype == DataType.BF16 else torch.uint8
    return TensorSpec(
        "out",
        [4, columns],
        dtype,
        is_output=True,
        init_value=lambda: torch.empty([4, columns], dtype=torch_dtype),
    )


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_packed_fp4_decode(dtype),
            tensors=[_packed_input(), _output(DataType.BF16, 64)],
            name=f"{dtype}_raw_decode_all_codes",
            golden=_packed_decode_golden(dtype),
            compare=_compare_bf16_bits,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
    ],
    *[
        st.case(
            _make_packed_fp4_encode(dtype, mode),
            tensors=[_encoding_input(dtype), _output(DataType.UINT8, 32)],
            name=f"{dtype}_raw_encode_all_codes_{mode}",
            golden=_packed_encode_golden(dtype, mode),
            compare=_compare_packed_bytes,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for mode in ("rint", "round", "floor", "ceil", "trunc")
    ],
    *[
        st.case(
            _make_packed_fp4_roundtrip(dtype),
            tensors=[_packed_input(), _output(DataType.UINT8, 32)],
            name=f"{dtype}_raw_roundtrip_all_codes",
            golden=_packed_roundtrip_golden,
            compare=_compare_packed_bytes,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
    ],
    *[
        st.case(
            _make_packed_fp4_convert(source, target),
            # Keep E2M1 magnitudes <=1.5, within E1M2's finite range. E1M2
            # uses all codes, including ties that exercise nearest-even E2M1.
            tensors=[
                _packed_input(0xBB if source == pl.fp4e2m1x2 else 0xFF),
                _output(DataType.UINT8, 32),
            ],
            name=f"{source}_to_{target}_raw_numeric_cast",
            golden=_packed_conversion_golden(source, target),
            compare=_compare_packed_bytes,
        )
        for source, target in ((pl.fp4e2m1x2, pl.fp4e1m2x2), (pl.fp4e1m2x2, pl.fp4e2m1x2))
    ],
)
def test_packed_fp4_raw_carrier_runtime(case_run):
    case_run.assert_passed()


def _make_packed_fp4_dn_decode(dtype):
    @pl.jit(auto_scope=False)
    def decode(src: pl.Tensor[[4, 32], dtype], out: pl.Out[pl.Tensor[[4, 64], pl.BF16]]):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            source = pl.tensor.view(src, [32, 4], layout=pl.TensorLayout.DN)
            target = pl.tensor.view(out, [64, 4], layout=pl.TensorLayout.DN)
            packed = pl.tile.load(source, [block * 32, 0], [32, 4])
            decoded = pl.cast(packed, pl.BF16)
            pl.tile.store(decoded, [block * 64, 0], target)
        return out

    return decode


def _make_packed_fp4_dn_encode(dtype, mode):
    @pl.jit(auto_scope=False)
    def encode(src: pl.Tensor[[4, 64], pl.BF16], out: pl.Out[pl.Tensor[[4, 32], dtype]]):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            # The metadata transpose gives TCVT a dense column-major source
            # without depending on the ordinary BF16 DN load path.
            values = pl.tile.load(src, [block * 4, 0], [4, 64])
            column_values = pl.tile.transpose_view(values)
            packed = pl.cast(column_values, dtype, mode=mode)
            target = pl.tensor.view(out, [32, 4], layout=pl.TensorLayout.DN)
            pl.tile.store(packed, [block * 32, 0], target)
        return out

    return encode


def _make_packed_fp4_dn_convert(source_dtype, target_dtype):
    @pl.jit(auto_scope=False)
    def convert(src: pl.Tensor[[4, 32], source_dtype], out: pl.Out[pl.Tensor[[4, 32], target_dtype]]):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            source = pl.tensor.view(src, [32, 4], layout=pl.TensorLayout.DN)
            target = pl.tensor.view(out, [32, 4], layout=pl.TensorLayout.DN)
            packed = pl.tile.load(source, [block * 32, 0], [32, 4])
            converted = pl.cast(packed, target_dtype, mode="rint")
            pl.tile.store(converted, [block * 32, 0], target)
        return out

    return convert


def _make_packed_fp4_dn_tensor_cast(source_dtype, target_dtype):
    source_cols = 64 if source_dtype == pl.BF16 else 32
    target_cols = 64 if target_dtype == pl.BF16 else 32

    @pl.jit(auto_scope=False)
    def convert(
        src: pl.Tensor[[4, source_cols], source_dtype],
        out: pl.Out[pl.Tensor[[4, target_cols], target_dtype]],
    ):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            source = pl.tensor.view(src, [source_cols, 4], layout=pl.TensorLayout.DN)
            target = pl.tensor.view(out, [target_cols, 4], layout=pl.TensorLayout.DN)
            converted = pl.tensor.cast(source, target_dtype, mode="rint")
            pl.tensor.assemble(target, converted, [block * target_cols, 0])
        return out

    return convert


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_packed_fp4_dn_decode(dtype),
            tensors=[_packed_input(), _output(DataType.BF16, 64)],
            name=f"{dtype}_dense_dn_decode_all_codes",
            golden=_packed_decode_golden(dtype),
            compare=_compare_bf16_bits,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
    ],
    *[
        st.case(
            _make_packed_fp4_dn_encode(dtype, mode),
            tensors=[_encoding_input(dtype), _output(DataType.UINT8, 32)],
            name=f"{dtype}_dense_dn_encode_{mode}",
            golden=_packed_encode_golden(dtype, mode),
            compare=_compare_packed_bytes,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for mode in ("rint", "round", "floor", "ceil", "trunc")
    ],
    *[
        st.case(
            _make_packed_fp4_dn_convert(source, target),
            tensors=[
                _packed_input(0xBB if source == pl.fp4e2m1x2 else 0xFF),
                _output(DataType.UINT8, 32),
            ],
            name=f"{source}_to_{target}_dense_dn_cast",
            golden=_packed_conversion_golden(source, target),
            compare=_compare_packed_bytes,
        )
        for source, target in ((pl.fp4e2m1x2, pl.fp4e1m2x2), (pl.fp4e1m2x2, pl.fp4e2m1x2))
    ],
)
def test_packed_fp4_dense_dn_runtime(case_run):
    case_run.assert_passed()


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _make_packed_fp4_dn_tensor_cast(dtype, pl.BF16),
            tensors=[_packed_input(), _output(DataType.BF16, 64)],
            name=f"{dtype}_dense_dn_tensor_decode",
            golden=_packed_decode_golden(dtype),
            compare=_compare_bf16_bits,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
    ],
    *[
        st.case(
            _make_packed_fp4_dn_tensor_cast(pl.BF16, dtype),
            tensors=[_encoding_input(dtype), _output(DataType.UINT8, 32)],
            name=f"{dtype}_dense_dn_tensor_encode",
            golden=_packed_encode_golden(dtype, "rint"),
            compare=_compare_packed_bytes,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
    ],
    *[
        st.case(
            _make_packed_fp4_dn_tensor_cast(source, target),
            tensors=[
                _packed_input(0xBB if source == pl.fp4e2m1x2 else 0xFF),
                _output(DataType.UINT8, 32),
            ],
            name=f"{source}_to_{target}_dense_dn_tensor_cast",
            golden=_packed_conversion_golden(source, target),
            compare=_compare_packed_bytes,
        )
        for source, target in ((pl.fp4e2m1x2, pl.fp4e1m2x2), (pl.fp4e1m2x2, pl.fp4e2m1x2))
    ],
)
def test_packed_fp4_dense_dn_tensor_runtime(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--platform", "a5"])
