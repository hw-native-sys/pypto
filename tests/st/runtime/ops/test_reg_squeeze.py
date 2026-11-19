# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bit-exact squeeze and loop append acceptance, including UB guard regions."""

from functools import partial

import pypto.language as pl
import pytest
import torch
from harness import st


def _stream_entry(bits, trips=7):
    dtype = pl.UINT16 if bits == 16 else pl.UINT32
    carrier = pl.INT16 if bits == 16 else pl.INT32
    predicate_dtype = pl.FP16 if bits == 16 else pl.FP32
    lanes = 2048 // bits
    chunks = 7
    capacity = chunks * lanes + 64

    @pl.jit(auto_scope=False)
    def entry(
        src: pl.Tensor[[chunks, lanes], carrier],
        flags: pl.Tensor[[chunks, lanes], predicate_dtype],
        out: pl.Out[pl.Tensor[[1, capacity], carrier]],
    ):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            raw = pl.load(src, [worker * chunks, 0], [chunks, lanes], target_memory=pl.Mem.Vec)
            source = pl.reinterpret_view(pl.reshape(raw, [chunks * lanes]), dtype)
            predicates = pl.reshape(
                pl.load(flags, [worker * chunks, 0], [chunks, lanes], target_memory=pl.Mem.Vec),
                [chunks * lanes],
            )
            guarded = pl.tile.full([capacity], dtype=carrier, value=123)
            destination = pl.reinterpret_view(guarded, dtype)
            with pl.reg_scope():
                state = pl.reg.squeeze_store_init(destination, offset=7)
                for chunk in pl.range(0, trips):
                    value = pl.reg.load(source, offset=chunk * lanes, lanes=lanes)
                    flag = pl.reg.load(predicates, offset=chunk * lanes, lanes=lanes)
                    seed = pl.reg.full_mask(lanes=lanes)
                    keep = pl.reg.compares(flag, 0.0, seed, predicate="ne")
                    state = pl.reg.squeeze_store_step(state, value, keep)
                pl.reg.squeeze_store_flush(state)
                # A second independent stream checks state reset and original
                # destination preservation after POST_UPDATE of the first cursor.
                last = pl.reg.load(source, offset=(chunks - 1) * lanes, lanes=lanes)
                three = pl.reg.active_mask(3, lanes=lanes)
                second = pl.reg.squeeze_store_init(destination, offset=capacity - 32)
                second = pl.reg.squeeze_store_step(second, last, three)
                pl.reg.squeeze_store_flush(second)
            _out = pl.store(pl.reshape(guarded, [1, capacity]), [0, 0], out)
        return out

    return entry


def _register_entry(bits):
    dtype = pl.UINT16 if bits == 16 else pl.UINT32
    carrier = pl.INT16 if bits == 16 else pl.INT32
    predicate_dtype = pl.FP16 if bits == 16 else pl.FP32
    lanes = 2048 // bits

    @pl.jit(auto_scope=False)
    def entry(
        src: pl.Tensor[[7, lanes], carrier],
        flags: pl.Tensor[[7, lanes], predicate_dtype],
        counts: pl.Tensor[[7], pl.INT32],
        out: pl.Out[pl.Tensor[[7, lanes + 64], carrier]],
    ):
        with pl.spmd(1):
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, 7):
                raw = pl.load(src, [row, 0], [1, lanes], target_memory=pl.Mem.Vec)
                source = pl.reinterpret_view(pl.reshape(raw, [lanes]), dtype)
                predicates = pl.reshape(
                    pl.load(flags, [row, 0], [1, lanes], target_memory=pl.Mem.Vec), [lanes]
                )
                count = pl.cast(pl.read(counts, [row]), pl.INDEX)
                guarded = pl.tile.full([lanes + 64], dtype=carrier, value=123)
                destination = pl.reinterpret_view(guarded, dtype)
                with pl.reg_scope():
                    value = pl.reg.load(source, lanes=lanes)
                    flag = pl.reg.load(predicates, lanes=lanes)
                    seed = pl.reg.full_mask(lanes=lanes)
                    keep = pl.reg.compares(flag, 0.0, seed, predicate="ne")
                    packed = pl.reg.squeeze(value, keep)
                    prefix = pl.reg.active_mask(count, lanes=lanes)
                    _written = pl.reg.store(destination, 32, packed, prefix)
                _out = pl.store(pl.reshape(guarded, [1, lanes + 64]), [row, 0], out)
        return out

    return entry


def _inputs(bits, pattern):
    lanes = 2048 // bits
    counts = [0, 1, lanes // 2 - 1, lanes // 2, lanes // 2 + 1, lanes - 1, lanes]
    if pattern == "empty":
        counts = [0] * 7
    elif pattern == "full":
        counts = [lanes] * 7
    # Payloads exercise unsigned high bits and UINT32 values not representable
    # in FP32. Signed host tensors carry the exact same bits to/from the device.
    values = torch.arange(7 * lanes, dtype=torch.int64).reshape(7, lanes) * 65537 + 0x80000123
    src = values.to(torch.int16 if bits == 16 else torch.int32)
    lane_order = (torch.arange(lanes) * 13) % lanes
    flags = (lane_order[None, :] < torch.tensor(counts)[:, None]).to(
        torch.float16 if bits == 16 else torch.float32
    )
    return src, flags, torch.tensor(counts, dtype=torch.int32)


def _golden(tensors, stream, trips=7):
    src, flags = tensors["src"], tensors["flags"]
    expected = torch.full_like(tensors["out"], 123)
    if stream:
        selected = src[:trips][flags[:trips] != 0]
        expected[0, 7 : 7 + selected.numel()] = selected
        expected[0, -32:-29] = src[-1, :3]
    else:
        for row in range(7):
            selected = src[row][flags[row] != 0]
            expected[row, 32 : 32 + selected.numel()] = selected
    return {"out": expected}


def _cases():
    for bits in (16, 32):
        lanes = 2048 // bits
        carrier = torch.int16 if bits == 16 else torch.int32
        for pattern in ("mixed", "empty", "full", "zero_trip"):
            src, flags, counts = _inputs(bits, pattern)
            yield st.case(
                _stream_entry(bits, trips=0 if pattern == "zero_trip" else 7),
                src,
                flags,
                torch.full((1, 7 * lanes + 64), 123, dtype=carrier),
                name=f"squeeze_stream_u{bits}_{pattern}",
                golden=partial(_golden, stream=True, trips=0 if pattern == "zero_trip" else 7),
                enable_buffer_ir=False,
                rtol=0,
                atol=0,
            )
            if pattern == "mixed":
                yield st.case(
                    _register_entry(bits),
                    src,
                    flags,
                    counts,
                    torch.full((7, lanes + 64), 123, dtype=carrier),
                    name=f"squeeze_register_u{bits}",
                    golden=partial(_golden, stream=False),
                    enable_buffer_ir=False,
                    rtol=0,
                    atol=0,
                )


@pytest.mark.platforms("a5")
@st.cases(*_cases())
def test_squeeze(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
