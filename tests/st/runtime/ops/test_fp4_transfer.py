# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact carrier transfers and padding for both PTOAS FP4 formats."""

import pypto.language as pl
import pytest
import torch
from harness import st
from harness.core.harness import DataType, TensorSpec
from pypto.pypto_core.passes import MemoryPlanner


def _transfer_entry(dtype: pl.DataType, dn: bool):
    @pl.jit(auto_scope=False)
    def entry(
        src: pl.Tensor[[32, 64], dtype],
        out: pl.InOut[pl.Tensor[[32, 64], dtype]],
    ):
        with pl.spmd(1):
            offset = pl.tile.get_block_idx() * 32 + 32
            if dn:
                # A canonical ND/DN view changes coordinates, not the bytes.
                source = pl.tensor.view(src, [64, 32], layout=pl.TensorLayout.DN)
                target = pl.tensor.view(out, [64, 32], layout=pl.TensorLayout.DN)
                dn_tile = pl.tile.load(source, [offset, 0], [32, 32])
                pl.tile.store(dn_tile, [0, 0], target)
            else:
                nd_tile = pl.tile.load(src, [0, offset], [32, 32])
                pl.tile.store(nd_tile, [0, 0], out)
        return out

    return entry


def _golden(tensors):
    expected = tensors["out"].view(torch.uint8).clone()
    expected[:, :32] = tensors["src"].view(torch.uint8)[:, 32:]
    return {"out": expected}


def _compare(actual, expected):
    # Include untouched sentinel bytes: an expanded DMA row pitch must fail.
    assert actual.keys() == expected.keys()
    assert torch.equal(actual["out"].view(torch.uint8), expected["out"])


def _carrier_input():
    return TensorSpec(
        "src",
        [32, 64],
        DataType.UINT8,
        init_value=lambda: (torch.arange(2048, dtype=torch.int32) * 29 + 7).to(torch.uint8).reshape(32, 64),
    )


def _sentinel_output():
    return TensorSpec(
        "out",
        [32, 64],
        DataType.UINT8,
        is_output=True,
        init_value=lambda: torch.full((32, 64), 0xA7, dtype=torch.uint8),
    )


# The pinned simulator corrupts partial DN transfers. Partial ND byte
# comparisons pass, but the process crashes during cleanup.
@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _transfer_entry(dtype, dn),
            tensors=[_carrier_input(), _sentinel_output()],
            name=f"{dtype}_carrier_{'dn' if dn else 'nd'}_offset_copy",
            golden=_golden,
            compare=_compare,
            rtol=0.0,
            atol=0.0,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for dn in (False, True)
    ]
)
def test_packed_fp4_carrier_transfers(case_run):
    case_run.assert_passed()


def _fillpad_entry(dtype, operation, pad_value):
    expand = operation == "expand"
    inplace = operation == "inplace"

    @pl.jit(auto_scope=False)
    def entry(src: pl.Tensor[[32, 64], dtype], out: pl.Out[pl.Tensor[[32, 64], dtype]]):
        with pl.spmd(1):
            row = pl.tile.get_block_idx() * 32
            if expand:
                small_tile = pl.tile.load(src, [row, 0], [16, 32], valid_shape=[13, 23])
                padded = pl.tile.fillpad_expand(small_tile, [32, 64], pad_value=pad_value)
            else:
                full_tile = pl.tile.load(src, [row, 0], [32, 64], valid_shape=[13, 23])
                if inplace:
                    padded = pl.tile.fillpad_inplace(full_tile, pad_value=pad_value)
                else:
                    padded = pl.tile.fillpad(full_tile, pad_value=pad_value)
            pl.tile.store(padded, [row, 0], out)
        return out

    return entry


def _fillpad_golden(pad_byte):
    def golden(tensors):
        expected = torch.full_like(tensors["out"], pad_byte)
        expected[:13, :23] = tensors["src"][:13, :23]
        return {"out": expected}

    return golden


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _fillpad_entry(dtype, operation, pad_value),
            tensors=[_carrier_input(), _sentinel_output()],
            name=f"{dtype}_fillpad_{operation}_{mode}",
            golden=_fillpad_golden(pad_byte),
            compare=_compare,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for operation in ("copy", "expand", "inplace")
        # pto-isa PadValueMap specifies two zero/min/max nibbles per byte
        # for both formats: E2M1 extrema are +/-6, E1M2 extrema +/-1.75.
        for mode, pad_value, pad_byte in (
            ("zero", pl.PadValue.zero, 0x00),
            ("min", pl.PadValue.min, 0xFF),
            ("max", pl.PadValue.max, 0x77),
        )
    ]
)
def test_packed_fp4_fillpad_carriers(case_run):
    case_run.assert_passed()


def _int8_extract_entry(dtype):
    @pl.jit(auto_scope=False)
    def entry(src: pl.Tensor[[32, 64], dtype], out: pl.Out[pl.Tensor[[16, 32], dtype]]):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            packed = pl.tile.load(src, [block * 32, 0], [32, 64])
            # A5 TEXTRACT Vec->Vec accepts signed byte carriers, not UINT8.
            payload = pl.reinterpret_view(packed, pl.INT8)
            window = pl.tile.extract(payload, 8, 32, [16, 32], target_memory=pl.Mem.Vec)
            result = pl.reinterpret_view(window, dtype)
            pl.tile.store(result, [block * 16, 0], out)
        return out

    return entry


def _byte_assemble_entry(dtype):
    @pl.jit(auto_scope=False)
    def entry(src: pl.Tensor[[16, 32], dtype], out: pl.InOut[pl.Tensor[[32, 64], dtype]]):
        with pl.spmd(1):
            block = pl.tile.get_block_idx()
            source = pl.tile.load(src, [block * 16, 0], [16, 32])
            target = pl.tile.load(out, [block * 32, 0], [32, 64])
            source_bytes = pl.reinterpret_view(source, pl.UINT8)
            target_bytes = pl.reinterpret_view(target, pl.UINT8)
            updated = pl.tile.assemble(target_bytes, source_bytes, [8, 32])
            result = pl.reinterpret_view(updated, dtype)
            pl.tile.store(result, [block * 32, 0], out)
        return out

    return entry


def _window_source(rows, cols):
    # Independent row/column patterns expose an incorrect nonzero row offset.
    return (torch.arange(rows)[:, None] * 11 + torch.arange(cols)[None, :] * 29 + 7).to(torch.uint8)


def _window_input(rows, columns):
    return TensorSpec(
        "src", [rows, columns], DataType.UINT8, init_value=lambda: _window_source(rows, columns)
    )


def _byte_extract_golden(tensors):
    return {"out": tensors["src"][8:24, 32:64].clone()}


def _byte_assemble_golden(tensors):
    expected = tensors["out"].clone()
    expected[8:24, 32:64] = tensors["src"]
    return {"out": expected}


@pytest.mark.platforms("a5")
@st.cases(
    *[
        st.case(
            _int8_extract_entry(dtype),
            tensors=[
                _window_input(32, 64),
                TensorSpec(
                    "out",
                    [16, 32],
                    DataType.UINT8,
                    is_output=True,
                    init_value=lambda: torch.empty((16, 32), dtype=torch.uint8),
                ),
            ],
            name=f"{dtype}_int8_alias_extract_{planner.name}",
            golden=_byte_extract_golden,
            compare=_compare,
            memory_planner=planner,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP)
    ],
    *[
        st.case(
            _byte_assemble_entry(dtype),
            tensors=[_window_input(16, 32), _sentinel_output()],
            name=f"{dtype}_byte_alias_assemble_{planner.name}",
            golden=_byte_assemble_golden,
            compare=_compare,
            memory_planner=planner,
        )
        for dtype in (pl.fp4e2m1x2, pl.fp4e1m2x2)
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.DSA_RP)
    ],
)
def test_packed_fp4_byte_alias_windows(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
