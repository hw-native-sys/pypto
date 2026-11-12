# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 numerical acceptance for extended register and logical channel operations.

Build with --build OUTPUT in a task-submit --no-device job. Execute the saved
artifacts with --worker OUTPUT DEVICE in a task-submit --device auto job, using
TASK_DEVICE. Compilation never reserves a device; execution never skips pin
validation or substitutes simulator code. Inputs and outputs are CPU tensors
transferred by the production ChipWorker, as in test_reg_scope_onboard.py.
"""

import json
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pypto.language as pl
import pytest
import torch
from pypto import CacheConfig
from pypto.backend import BackendType
from pypto.ir.compiled_program import CompiledProgram
from pypto.runtime import ChipWorker, ExecutionMode, RunConfig
from pypto.runtime._artifact_sources import read_kernel_config
from pypto.runtime.runtime_pin import check_runtime_pin

pytestmark = pytest.mark.platforms("a5", reason="Extended VMI operations require A5 hardware")
ROOT = Path(__file__).resolve().parents[4]
LENGTHS = (0, 1, 31, 32, 37, 63, 64)
CASES = (
    "float32",
    "integer_u32",
    "integer_i32",
    "indexed_f32",
    "indexed_u32",
    "indexed_f16",
    "indexed_base_f32",
    "indexed_base_f16",
    "histogram128",
    "histogram256",
    "reinterpret",
    "mx_quant",
    "unmasked_f32",
    "unmasked_bf16",
    "channels_f32",
    "channels_bf16_fp8",
)
FLOAT_OPS = ("expdif", "axpy", "mula", "lrelu", "prelu")
INTEGER_OPS = ("mull", "addc", "subc", "addcs", "subcs", "intlv", "dintlv")
MX_OPS = ("reduce_max", "reduce_sum", "group_broadcast")
# PyPTO operation name -> PTOAS mnemonic, for the ops whose mnemonic is not the
# name with a `v` prefix. The reductions share one opcode with their grouped
# form, and the grouped broadcast carries no prefix at all.
MNEMONICS = {
    "reduce_max": "vcmax",
    "reduce_sum": "vcadd",
    "create_group_mask": "create_group_mask",
    "group_broadcast": "group_broadcast",
    "channel_split": "channel_split",
    "channel_merge": "channel_merge",
    "convert": "vcvt",
}


def _histogram_body(bins):
    dtype, output_dtype = pl.UINT8, pl.UINT16

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], dtype],
        out: pl.InOut[pl.Tensor[[1, 1024], output_dtype]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], output_dtype]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        limit_count = pl.cast(n * 4, pl.INT32)
        with pl.reg_scope():
            x = pl.reg.load(a, lanes=256)
            acc = pl.reg.load(c, lanes=bins)
            grouped = pl.reg.create_group_mask(37, num_groups=4, group_size=64)
            idx = pl.reg.index_gen(0, lanes=256, dtype=pl.INT32)
            limit = pl.reg.broadcast(limit_count, lanes=256, dtype=pl.INT32)
            seed = pl.reg.full_mask(lanes=256)
            prefix = pl.reg.compare(idx, limit, seed, predicate="lt")
            mask = pl.reg.mask_and(grouped, prefix)
            h = pl.reg.chist(acc, x, mask)
            d = pl.reg.dhist(acc, x, mask)
            h2 = pl.reg.chist(h, x, mask)
            c = pl.reg.store(c, offset=256, value=h)
            c = pl.reg.store(c, offset=512, value=d)
            c = pl.reg.store(c, offset=768, value=h2)
            # chist writes over its original accumulator. dhist reloads its
            # output and updates that same interval, exercising both SSA-ordered
            # writeback and PTOAS store-to-load synchronization on real hardware.
            c = pl.reg.store(c, offset=0, value=h)
            d_acc = pl.reg.load(c, offset=512, lanes=bins)
            d2 = pl.reg.dhist(d_acc, x, mask)
            c = pl.reg.store(c, offset=512, value=d2)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _reinterpret_body():
    dtype = pl.UINT32

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], dtype],
        out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], dtype]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        active = pl.cast(n // 4, pl.INDEX)
        with pl.reg_scope():
            x = pl.reg.load(a, lanes=64)
            bytes_ = pl.reg.interpret_cast(x, dtype=pl.UINT8)
            mask = pl.reg.create_group_mask(37, num_groups=4, group_size=64)
            zero = pl.reg.broadcast(0, lanes=256, dtype=pl.UINT8)
            selected = pl.reg.select(mask, bytes_, zero)
            words = pl.reg.interpret_cast(selected, dtype=pl.UINT32)
            floats = pl.reg.interpret_cast(x, dtype=pl.FP32)
            restored = pl.reg.interpret_cast(floats, dtype=pl.UINT32)
            halves = pl.reg.interpret_cast(x, dtype=pl.FP16)
            half_bits = pl.reg.interpret_cast(halves, dtype=pl.UINT16)
            bfloat = pl.reg.interpret_cast(half_bits, dtype=pl.BF16)
            restored_half = pl.reg.interpret_cast(bfloat, dtype=pl.UINT32)
            dynamic_mask = pl.reg.create_group_mask(active, num_groups=4, group_size=16)
            zeros = pl.reg.broadcast(0, lanes=64, dtype=pl.UINT32)
            dynamic_selected = pl.reg.select(dynamic_mask, x, zeros)
            signed = pl.reg.interpret_cast(x, dtype=pl.INT32)
            restored_signed = pl.reg.interpret_cast(signed, dtype=pl.UINT32)
            c = pl.reg.store(c, offset=64, value=words)
            c = pl.reg.store(c, offset=128, value=restored)
            c = pl.reg.store(c, offset=192, value=restored_half)
            c = pl.reg.store(c, offset=256, value=restored_signed)
            c = pl.reg.store(c, offset=320, value=dynamic_selected)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _indexed_body(dtype):
    index_dtype = pl.UINT16 if dtype == pl.FP16 else pl.UINT32

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], dtype],
        out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], dtype]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        with pl.reg_scope():
            idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
            limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
            seed = pl.reg.full_mask(lanes=64)
            mask = pl.reg.compare(idx, limit, seed, predicate="lt")
            offsets = pl.reg.index_gen(0, lanes=64, dtype=index_dtype)
            three = pl.reg.broadcast(3, lanes=64, dtype=index_dtype)
            last = pl.reg.broadcast(255, lanes=64, dtype=index_dtype)
            scaled = pl.reg.mul(offsets, three)
            reverse = pl.reg.sub(last, scaled)
            values = pl.reg.gather(a, reverse, mask)
            x = pl.reg.load(a, lanes=64)
            y = pl.reg.load(a, offset=64, lanes=64)
            z = pl.reg.load(a, offset=128, lanes=64)
            # Preserve reads from both sides of a write (WAR/RAW), then
            # overwrite twice with independent payloads (WAW). PTOAS must
            # provide ordering without frontend VV_ALL fences.
            before = pl.reg.gather(c, reverse, mask)
            c = pl.reg.scatter(c, reverse, x, mask)
            reread = pl.reg.gather(c, reverse, mask)
            c = pl.reg.scatter(c, reverse, y, mask)
            c = pl.reg.scatter(c, reverse, z, mask)
            final = pl.reg.gather(c, reverse, mask)
            repeated = pl.reg.broadcast(7, lanes=64, dtype=index_dtype)
            repeats = pl.reg.gather(a, repeated, mask)
            c = pl.reg.store(c, offset=320, value=values)
            c = pl.reg.store(c, offset=448, value=reread)
            c = pl.reg.store(c, offset=576, value=repeats)
            c = pl.reg.store(c, offset=704, value=before)
            c = pl.reg.store(c, offset=832, value=final)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _indexed_base_body(dtype):
    """Compose scalar element bases with nonzero Tile view origins."""
    index_dtype = pl.UINT16 if dtype == pl.FP16 else pl.UINT32

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], dtype],
        out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], dtype]:
        loaded = pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec)
        a = pl.reshape(pl.slice(loaded, [1, 224], [0, 32]), [224])
        c = pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec)
        destination = pl.reshape(pl.slice(c, [1, 256], [0, 32]), [256])
        c = pl.reshape(c, [1024])
        base = pl.cast(32 + n, pl.INDEX)
        active = pl.cast(n, pl.INDEX)
        with pl.reg_scope():
            lane = pl.reg.index_gen(0, lanes=64, dtype=index_dtype)
            full = pl.reg.full_mask(lanes=64)
            scaled = pl.reg.shift_lefts(lane, 1, full)
            last = pl.reg.broadcast(127, lanes=64, dtype=index_dtype)
            offsets = pl.reg.sub(last, scaled, full)
            mask = pl.reg.active_mask(active, lanes=64)
            fixed = pl.reg.gather(a, offsets, mask, base_offset=16)
            dynamic = pl.reg.gather(a, offsets, mask, base_offset=base)
            updated = pl.reg.scatter(destination, offsets, fixed, mask, base_offset=16)
            fixed_read = pl.reg.gather(updated, offsets, mask, base_offset=16)
            updated = pl.reg.scatter(updated, offsets, dynamic, mask, base_offset=base)
            dynamic_read = pl.reg.gather(updated, offsets, mask, base_offset=base)
            c = pl.reg.store(c, offset=384, value=fixed)
            c = pl.reg.store(c, offset=512, value=dynamic)
            c = pl.reg.store(c, offset=640, value=fixed_read)
            c = pl.reg.store(c, offset=768, value=dynamic_read)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _mx_quant_body():
    """MX group scaling entirely in registers: grouped amax, broadcast, divide.

    This is the shape that could not be expressed before. The per-group maximum
    had no in-register form at all: `reduce_max` collapses to one lane, and a
    one-lane result cannot be re-spread, because binary ops require matching
    lane counts and 32-bit `gather_lanes` is restricted to 64 lanes. The only
    alternative was a five-step rotate-and-max butterfly, ten register ops per
    64 lanes in the innermost loop of every MX-quantized kernel.
    """

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], pl.FP32],
        out: pl.InOut[pl.Tensor[[1, 1024], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], pl.FP32]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        with pl.reg_scope():
            value = pl.reg.load(a, lanes=64)
            full = pl.reg.full_mask(lanes=64)
            idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
            limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
            active = pl.reg.compare(idx, limit, full, predicate="lt")
            magnitude = pl.reg.abs(value, full)
            # The MX geometry: 64 lanes in groups of 32, hence two groups. Both
            # reductions run under a full mask, because an empty group's
            # identity (-inf or the dtype minimum) is deliberately not part of
            # the contract this test pins down.
            amax = pl.reg.reduce_max(magnitude, full, group_size=32)
            total = pl.reg.reduce_sum(value, full, reassoc=True, group_size=32)
            scale = pl.reg.group_broadcast(amax, lanes=64)
            sums = pl.reg.group_broadcast(total, lanes=64)
            # Only the scaling is masked, so inactive lanes leave an exact zero
            # and the reduction results stay independent of the active count.
            scaled = pl.reg.div(value, scale, active)
            c = pl.reg.store(c, offset=64, value=value)
            c = pl.reg.store(c, offset=128, value=scale)
            c = pl.reg.store(c, offset=192, value=scaled)
            c = pl.reg.store(c, offset=256, value=sums)
            # Two lanes, not 64: a grouped reduction holds one lane per group in
            # group order. A bug that broadcast lane 0 over the whole register
            # could satisfy the stores above but not this one.
            c = pl.reg.store(c, offset=320, value=amax)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _unmasked_body(dtype):
    register_dtype = pl.BF16 if dtype == pl.UINT16 else pl.FP32

    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], dtype],
        out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], dtype]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        with pl.reg_scope():
            first = pl.reg.load(a, lanes=128)
            second = pl.reg.load(a, offset=128, lanes=128)
            x = pl.reg.interpret_cast(first, dtype=register_dtype)
            y = pl.reg.interpret_cast(second, dtype=register_dtype)
            low, high = pl.reg.intlv(x, y)
            even, odd = pl.reg.dintlv(x, y)
            low_bits = pl.reg.interpret_cast(low, dtype=dtype)
            high_bits = pl.reg.interpret_cast(high, dtype=dtype)
            even_bits = pl.reg.interpret_cast(even, dtype=dtype)
            odd_bits = pl.reg.interpret_cast(odd, dtype=dtype)
            c = pl.reg.store(c, offset=64, value=low_bits)
            c = pl.reg.store(c, offset=192, value=high_bits)
            c = pl.reg.store(c, offset=448, value=even_bits)
            c = pl.reg.store(c, offset=576, value=odd_bits)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _channels_body():
    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], pl.FP32],
        out: pl.InOut[pl.Tensor[[1, 1024], pl.FP32]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], pl.FP32]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        with pl.reg_scope():
            source = pl.reg.load(a, lanes=128)
            even, odd = pl.reg.channel_split(source)
            c = pl.reg.store(c, offset=64, value=even)
            c = pl.reg.store(c, offset=128, value=odd)
            # Independently loaded inputs keep split/merge permutations from
            # cancelling each other and hiding a reversed channel contract.
            x0 = pl.reg.load(a, offset=0, lanes=64)
            x1 = pl.reg.load(a, offset=64, lanes=64)
            x2 = pl.reg.load(a, offset=128, lanes=64)
            x3 = pl.reg.load(a, offset=192, lanes=64)
            pair = pl.reg.channel_merge(x0, x1)
            four = pl.reg.channel_merge(x0, x1, x2, x3)
            c = pl.reg.store(c, offset=256, value=pair)
            c = pl.reg.store(c, offset=512, value=four)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _channels_bf16_fp8_body():
    @pl.jit.incore
    def body(
        src: pl.Tensor[[1, 256], pl.UINT16],
        out: pl.InOut[pl.Tensor[[1, 1024], pl.UINT8]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], pl.UINT8]:
        a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
        c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
        with pl.reg_scope():
            first_bits = pl.reg.load(a, lanes=128)
            second_bits = pl.reg.load(a, offset=128, lanes=128)
            first = pl.reg.interpret_cast(first_bits, dtype=pl.BF16)
            second = pl.reg.interpret_cast(second_bits, dtype=pl.BF16)
            even, odd = pl.reg.dintlv(first, second)
            even_float = pl.reg.convert(even, dtype=pl.FP32)
            odd_float = pl.reg.convert(odd, dtype=pl.FP32)
            x0, x2 = pl.reg.channel_split(even_float)
            x1, x3 = pl.reg.channel_split(odd_float)
            full = pl.reg.full_mask(lanes=64)
            # Distinct scales expose channel swaps before the merged FP8 byte
            # payload is checked against an independent CPU conversion.
            y0 = pl.reg.muls(x0, 0.5, full)
            y1 = pl.reg.muls(x1, 1.0, full)
            y2 = pl.reg.muls(x2, 2.0, full)
            y3 = pl.reg.muls(x3, -1.0, full)
            restored = pl.reg.channel_merge(y0, y1, y2, y3)
            quantized = pl.reg.convert(restored, dtype=pl.FP8E4M3FN, rounding="R", saturate=False)
            carrier = pl.reg.interpret_cast(quantized, dtype=pl.UINT8)
            c = pl.reg.store(c, offset=256, value=carrier)
        c = pl.reshape(c, [1, 1024])
        pl.store(c, [0, 0], out)
        return out

    return body


def _program(case):
    dtype = {
        "integer_u32": pl.UINT32,
        "integer_i32": pl.INT32,
        "indexed_u32": pl.UINT32,
        "indexed_f16": pl.FP16,
        "indexed_base_f32": pl.FP32,
        "indexed_base_f16": pl.FP16,
        "histogram128": pl.UINT8,
        "histogram256": pl.UINT8,
        "reinterpret": pl.UINT32,
        "unmasked_bf16": pl.UINT16,
        "channels_bf16_fp8": pl.UINT16,
    }.get(case, pl.FP32)
    output_dtype = pl.UINT16 if case.startswith("histogram") else dtype
    if case == "channels_bf16_fp8":
        output_dtype = pl.UINT8
    bins = 128 if case == "histogram128" else 256
    if case == "float32":

        @pl.jit.incore
        def body(
            src: pl.Tensor[[1, 256], dtype],
            out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1024], dtype]:
            a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
            c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
            with pl.reg_scope():
                x = pl.reg.load(a, offset=0, lanes=64)
                y = pl.reg.load(a, offset=64, lanes=64)
                z = pl.reg.load(a, offset=128, lanes=64)
                idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                mask = pl.reg.compare(idx, limit, seed, predicate="lt")
                e = pl.reg.expdif(x, y, mask)
                ax = pl.reg.axpy(x, y, 0.75, mask)
                ma = pl.reg.mula(z, x, y, mask)
                lr = pl.reg.lrelu(x, 0.125, mask)
                pr = pl.reg.prelu(x, y, mask)
                c = pl.reg.store(c, offset=64, value=e)
                c = pl.reg.store(c, offset=128, value=ax)
                c = pl.reg.store(c, offset=192, value=ma)
                c = pl.reg.store(c, offset=256, value=lr)
                c = pl.reg.store(c, offset=320, value=pr)
            c = pl.reshape(c, [1, 1024])
            pl.store(c, [0, 0], out)
            return out
    elif case.startswith("histogram"):
        body = _histogram_body(bins)
    elif case == "reinterpret":
        body = _reinterpret_body()
    elif case == "mx_quant":
        body = _mx_quant_body()
    elif case.startswith("unmasked_"):
        body = _unmasked_body(dtype)
    elif case == "channels_f32":
        body = _channels_body()
    elif case == "channels_bf16_fp8":
        body = _channels_bf16_fp8_body()
    elif case.startswith("indexed_base_"):
        body = _indexed_base_body(dtype)
    elif case.startswith("integer_"):

        @pl.jit.incore
        def body(
            src: pl.Tensor[[1, 256], dtype],
            out: pl.InOut[pl.Tensor[[1, 1024], dtype]],
            n: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, 1024], dtype]:
            a = pl.reshape(pl.load(src, [0, 0], [1, 256], target_memory=pl.MemorySpace.Vec), [256])
            c = pl.reshape(pl.load(out, [0, 0], [1, 1024], target_memory=pl.MemorySpace.Vec), [1024])
            with pl.reg_scope():
                x = pl.reg.load(a, offset=0, lanes=64)
                y = pl.reg.load(a, offset=64, lanes=64)
                ci = pl.reg.load(a, offset=128, lanes=64)
                idx = pl.reg.index_gen(0, lanes=64, dtype=pl.INT32)
                limit = pl.reg.broadcast(n, lanes=64, dtype=pl.INT32)
                seed = pl.reg.full_mask(lanes=64)
                mask = pl.reg.compare(idx, limit, seed, predicate="lt")
                zero = pl.reg.broadcast(0, lanes=64, dtype=dtype)
                one = pl.reg.broadcast(1, lanes=64, dtype=dtype)
                carry_in = pl.reg.compare(ci, zero, seed, predicate="ne")
                lo, hi = pl.reg.mull(x, y, mask)
                add, ac = pl.reg.addc(x, y, mask)
                sub, sc = pl.reg.subc(x, y, mask)
                adds, asc = pl.reg.addcs(x, y, carry_in, mask)
                subs, ssc = pl.reg.subcs(x, y, carry_in, mask)
                il, ih = pl.reg.intlv(x, y, mask)
                de, do = pl.reg.dintlv(x, y, mask)
                acv = pl.reg.select(ac, one, zero)
                scv = pl.reg.select(sc, one, zero)
                ascv = pl.reg.select(asc, one, zero)
                sscv = pl.reg.select(ssc, one, zero)
                c = pl.reg.store(c, offset=64, value=lo)
                c = pl.reg.store(c, offset=128, value=hi)
                c = pl.reg.store(c, offset=192, value=add)
                c = pl.reg.store(c, offset=256, value=acv)
                c = pl.reg.store(c, offset=320, value=sub)
                c = pl.reg.store(c, offset=384, value=scv)
                c = pl.reg.store(c, offset=448, value=adds)
                c = pl.reg.store(c, offset=512, value=ascv)
                c = pl.reg.store(c, offset=576, value=subs)
                c = pl.reg.store(c, offset=640, value=sscv)
                c = pl.reg.store(c, offset=704, value=il)
                c = pl.reg.store(c, offset=768, value=ih)
                c = pl.reg.store(c, offset=832, value=de)
                c = pl.reg.store(c, offset=896, value=do)
            c = pl.reshape(c, [1, 1024])
            pl.store(c, [0, 0], out)
            return out
    else:
        body = _indexed_body(dtype)

    @pl.jit
    def dispatch(
        src: pl.Tensor[[1, 256], dtype],
        out: pl.InOut[pl.Tensor[[1, 1024], output_dtype]],
        n: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1024], output_dtype]:
        return body(src, out, n)

    return dispatch


def _environment():
    assert os.environ.get("PYPTO_SKIP_RUNTIME_PIN_CHECK", "0") == "0"
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        assert not any(
            token in os.environ.get(key, "").lower() for token in ("simulator", "camodel", "msopprof")
        )
    pin = check_runtime_pin(checkout_root=ROOT)
    assert pin.matches and pin.pinned == pin.installed, pin
    return pin


def _build(output):
    pin = _environment()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"runtime_commit": pin.installed, "cases": []}
    recorder = runpy.run_path(str(ROOT / "examples/runtime/vf_emitc_vpto.py"))["_record_compilation"]
    for case in CASES:
        directory = output / case
        compiled = _program(case).compile(
            config=RunConfig(
                backend_type=BackendType.Ascend950,
                execution_mode=ExecutionMode.ONBOARD,
                cache_config=CacheConfig(enabled=False),
                save_kernels_dir=str(directory),
                dump_passes=True,
            )
        )
        config = read_kernel_config(directory / "kernel_config.py")
        helpers = [Path(p).read_text() for kernel in config.KERNELS for p in kernel["vpto_sources"]]
        ops = (
            FLOAT_OPS
            if case == "float32"
            else INTEGER_OPS
            if case.startswith("integer_")
            else ("chist", "dhist", "create_group_mask")
            if case.startswith("histogram")
            else ("interpret_cast", "create_group_mask")
            if case == "reinterpret"
            else MX_OPS
            if case == "mx_quant"
            else ("intlv", "dintlv")
            if case.startswith("unmasked_")
            else ("channel_split", "channel_merge")
            if case == "channels_f32"
            else ("dintlv", "channel_split", "channel_merge", "convert", "interpret_cast")
            if case == "channels_bf16_fp8"
            else ("gather", "scatter")
        )
        for op in ops:
            mnemonic = MNEMONICS.get(op, f"v{op}")
            assert any(f"pto.vmi.{mnemonic}" in helper for helper in helpers), (case, op)
        if case == "mx_quant":
            # The grouped forms must carry the group COUNT, not the group width,
            # and the reduction result must be narrowed to one lane per group.
            assert any("{group = 2" in helper for helper in helpers), case
            assert any("vreg<2xf32>" in helper for helper in helpers), case
        if case.startswith("indexed_"):
            assert all('pto.mem_bar "VV_ALL"' not in helper for helper in helpers)
        evidence = Path(tempfile.mkdtemp(prefix="compilation-", dir=directory))
        with recorder(evidence):
            compiled.load()  # Production PTOAS -> LLVM -> Bisheng -> linked .text, no device.
        manifest["cases"].append(
            {
                "name": case,
                "directory": str(directory.resolve()),
                "runtime": compiled.runtime_name,
                "ops": list(ops),
            }
        )
        print(f"BUILT {case}: {', '.join(ops)}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _inputs(case, repeat):
    if case == "unmasked_bf16":
        # Distinct finite BF16 bit patterns test lane ordering without rounding
        # or NaN equality obscuring a misplaced payload. UINT16 transports the
        # bits through NumPy; the kernel reinterprets them as BF16 registers.
        bits = (0x3E00 + np.arange(256, dtype=np.uint16) * 13).astype(np.uint16)
        bits[::3] |= np.uint16(0x8000)
        return np.roll(bits, repeat * 37)
    if case == "channels_bf16_fp8":
        values = (((np.arange(256) * 37) % 223 - 111) * 0.5).astype(np.float32)
        values = np.roll(values, repeat * 37)
        # All values are exactly representable in BF16, including the smallest
        # FP8 subnormal and normal values and signs at the channel boundaries.
        values[0:8] = (0.0, -0.0, 2**-9, -(2**-9), 2**-6, -(2**-6), 1.0, -1.0)
        return (values.view(np.uint32) >> 16).astype(np.uint16)
    if case in ("unmasked_f32", "channels_f32"):
        return (np.arange(256, dtype=np.float32) * 0.25 - 31.75 + repeat * 0.125).astype(np.float32)
    if case.startswith("indexed_base_"):
        dtype = np.float16 if case.endswith("f16") else np.float32
        return (np.arange(256) * 3 - 256 + repeat * 1024).astype(dtype)
    if case == "mx_quant":
        # Dyadic values, so every quotient below is exact and the golden stays
        # an equality check. The two groups have DIFFERENT maxima (8.0 and 1.0):
        # a broadcast that wrongly spread lane 0 across all 64 lanes would still
        # satisfy a single-maximum input. Rolling by repeat permutes each group
        # without changing its multiset, so the expected maxima and sums stay
        # the same while the data layout differs.
        pattern = np.array([8.0, -4.0, 2.0, -1.0, 0.0, 6.0, -5.0, 3.0], dtype=np.float32)
        values = np.zeros(256, dtype=np.float32)
        values[:32] = np.resize(np.roll(pattern, repeat), 32)
        values[32:64] = np.resize(np.roll(pattern / 8.0, repeat), 32)
        return values
    if case.startswith("histogram"):
        patterns = np.array([0, 1, 126, 127, 128, 129, 254, 255], dtype=np.uint8)
        return np.resize(np.roll(patterns, repeat), 256)
    if case == "reinterpret":
        patterns = np.array(
            [0, 0x80000000, 0x3F800000, 0x7FC12345, 0x7F800000, 0xFF800000, 0x12345678, 0xFFFFFFFF],
            dtype=np.uint32,
        )
        return np.resize(np.roll(patterns, repeat), 256)
    if case.startswith("integer_"):
        patterns = np.array(
            [0, 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF, 123456789, 0xFFFFFFFE, 987654321], dtype=np.uint32
        )
        data = np.resize(np.roll(patterns, repeat), 256)
        data[64:128] = np.resize(np.roll(patterns, 3 + repeat), 64)
        data[128:192] = (np.arange(64) + repeat) % 2
        return data.view(np.int32) if case == "integer_i32" else data
    if case == "indexed_u32":
        return np.arange(256, dtype=np.uint32) * 104729 + repeat * 31
    dtype = np.float16 if case == "indexed_f16" else np.float32
    values = ((np.arange(256) % 29) * 0.125 - 1.75 + repeat * 0.25).astype(dtype)
    if case == "float32":
        values[64:128] = (np.arange(64) % 11) * 0.125 + 0.5
    return values


def _initial(case, data):
    if case == "channels_bf16_fp8":
        return np.full(1024, 0xA5, dtype=np.uint8)
    if case == "unmasked_bf16":
        return np.full(1024, 0xA55A, dtype=np.uint16)
    if case.startswith("histogram"):
        initial = np.full(1024, 0xCCCC, dtype=np.uint16)
        bins = 128 if case == "histogram128" else 256
        initial[:bins] = ((65530 + np.arange(bins) % 11) % 65536).astype(np.uint16)
        return initial
    return np.full(1024, 0xDEADBEEF if data.dtype == np.uint32 else -777, dtype=data.dtype)


def _golden(case, data, n):
    expected = _initial(case, data)
    if case.startswith("unmasked_"):
        x, y = data[:128], data[128:]
        interleaved = np.column_stack((x, y)).reshape(-1)
        expected[64:192] = interleaved[:128]
        expected[192:320] = interleaved[128:]
        expected[448:576] = data[::2]
        expected[576:704] = data[1::2]
        return expected
    if case == "channels_f32":
        expected[64:128] = data[:128:2]
        expected[128:192] = data[1:128:2]
        expected[256:384] = data[:128].reshape(2, 64).T.reshape(-1)
        expected[512:768] = data.reshape(4, 64).T.reshape(-1)
        return expected
    if case == "channels_bf16_fp8":
        values = (data.astype(np.uint32) << 16).view(np.float32)
        scales = np.resize(np.array([0.5, 1.0, 2.0, -1.0], dtype=np.float32), 256)
        quantized = torch.from_numpy(values * scales).to(torch.float8_e4m3fn)
        expected[256:512] = quantized.view(torch.uint8).numpy()
        return expected
    if case.startswith("indexed_base_"):
        offsets = 127 - np.arange(64) * 2
        fixed = 32 + 16 + offsets
        dynamic = 32 + 32 + n + offsets
        # Compare both shifted scatter regions, including overlapping writes,
        # and every untouched guard element in the complete output allocation.
        expected[fixed[:n]] = data[fixed[:n]]
        expected[dynamic[:n]] = data[dynamic[:n]]
        for start, indices in ((384, fixed), (512, dynamic), (640, fixed), (768, dynamic)):
            expected[start : start + 64] = 0
            expected[start : start + n] = data[indices[:n]]
        return expected
    if case == "mx_quant":
        lane = np.arange(64)
        first_group = lane < 32
        # Group 0 holds the pattern, group 1 the same pattern scaled by 1/8, so
        # the maxima are 8.0 and 1.0 and the sums 4 * 9.0 and 4 * 1.125. Every
        # value is exactly representable, so a reassociated sum is exact too.
        scale = np.where(first_group, 8.0, 1.0).astype(np.float32)
        sums = np.where(first_group, 36.0, 4.5).astype(np.float32)
        expected[64:128] = data[:64]
        expected[128:192] = scale
        # Inactive lanes carry an exact zero, because the div is masked.
        expected[192:256] = np.where(lane < n, data[:64] / scale, 0.0)
        expected[256:320] = sums
        expected[320:322] = (8.0, 1.0)
        return expected
    if case.startswith("histogram"):
        bins = 128 if case == "histogram128" else 256
        active = (np.arange(256) % 64 < 37) & (np.arange(256) < n * 4)
        counts = np.bincount(data[active].astype(np.int64), minlength=256)
        cumulative = np.cumsum(counts)[:bins]
        acc = expected[:bins].astype(np.uint64)
        expected[256 : 256 + bins] = ((acc + cumulative) % 65536).astype(np.uint16)
        expected[:bins] = ((acc + cumulative) % 65536).astype(np.uint16)
        expected[512 : 512 + bins] = ((acc + 2 * counts[:bins]) % 65536).astype(np.uint16)
        expected[768 : 768 + bins] = ((acc + 2 * cumulative) % 65536).astype(np.uint16)
        return expected
    if case == "reinterpret":
        bits = data[:64].view(np.uint8).copy()
        bits[np.arange(256) % 64 >= 37] = 0
        expected[64:128] = bits.view(np.uint32)
        expected[320:384] = np.where(np.arange(64) % 16 < n // 4, data[:64], 0)
        for offset in (128, 192, 256):
            expected[offset : offset + 64] = data[:64]
        return expected
    x, y, z = (data[start : start + 64] for start in (0, 64, 128))
    if case == "float32":
        xd, yd, zd = x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)
        results = [
            np.exp(xd - yd),
            xd * 0.75 + yd,
            zd + xd * yd,
            np.where(xd > 0, xd, xd * 0.125),
            np.maximum(xd, 0) + yd * np.minimum(xd, 0),
        ]
    elif case.startswith("integer_"):
        ux, uy = x.view(np.uint32).astype(object), y.view(np.uint32).astype(object)
        ci = z.astype(object)
        product = x.astype(object) * y.astype(object)
        summed, difference = ux + uy, ux - uy
        summed_c, difference_c = ux + uy + ci, ux - uy - (1 - ci)
        interleaved = np.column_stack((x, y)).reshape(-1)
        combined = np.concatenate((x, y))
        raw = [
            product,
            product >> 32,
            summed,
            summed > 0xFFFFFFFF,
            difference,
            ux >= uy,
            summed_c,
            summed_c > 0xFFFFFFFF,
            difference_c,
            ux >= uy + (1 - ci),
            interleaved[:64],
            interleaved[64:],
            combined[::2],
            combined[1::2],
        ]
        results = [
            np.array([int(v) & 0xFFFFFFFF for v in result], dtype=np.uint32).view(data.dtype)
            for result in raw
        ]
    else:
        indices = 255 - np.arange(64) * 3
        before = expected[indices].copy()
        expected[indices[:n]] = z[:n]
        for start, values in (
            (320, data[indices]),
            (448, x),
            (576, np.full(64, data[7], dtype=data.dtype)),
            (704, before),
            (832, z),
        ):
            expected[start : start + 64] = 0
            expected[start : start + n] = values[:n]
        return expected
    for index, result in enumerate(results, 1):
        start = index * 64
        expected[start : start + 64] = 0
        expected[start : start + n] = result[:n]
    return expected


def _run(output, device):
    pin = _environment()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["runtime_commit"] == pin.installed
    assert {case["name"] for case in manifest["cases"]} == set(CASES)
    config = RunConfig(
        backend_type=BackendType.Ascend950, execution_mode=ExecutionMode.ONBOARD, device_id=device
    )
    checks = []
    failures = []
    with ChipWorker(config=config, runtime=manifest["cases"][0]["runtime"]) as worker:
        for entry in manifest["cases"]:
            case = entry["name"]
            compiled = CompiledProgram.from_dir(entry["directory"])
            handle = worker.register(compiled)
            for repeat in range(2):
                data = _inputs(case, repeat)
                src = torch.from_numpy(data.reshape(1, 256).copy())
                for n in LENGTHS:
                    expected = _golden(case, data, n)
                    actual = torch.from_numpy(_initial(case, data).reshape(1, 1024))
                    handle(src, actual, n)
                    try:
                        if case == "float32":
                            np.testing.assert_allclose(
                                actual.numpy()[0],
                                expected,
                                rtol=3e-5,
                                atol=2e-6,
                                err_msg=f"{case}, repeat={repeat}, n={n}",
                            )
                        else:
                            np.testing.assert_array_equal(
                                actual.numpy()[0], expected, err_msg=f"{case}, repeat={repeat}, n={n}"
                            )
                    except AssertionError as error:
                        mismatches = (
                            np.flatnonzero(~np.isclose(actual.numpy()[0], expected, rtol=3e-5, atol=2e-6))
                            if case == "float32"
                            else np.flatnonzero(actual.numpy()[0] != expected)
                        )
                        failures.append(
                            {
                                "case": case,
                                "repeat": repeat,
                                "active_lanes": n,
                                "error": str(error),
                                "offsets": mismatches[:16].tolist(),
                                "actual": actual.numpy()[0][mismatches[:16]].tolist(),
                                "expected": expected[mismatches[:16]].tolist(),
                            }
                        )
                        print(
                            f"FAIL {case} repeat={repeat} n={n} offsets={mismatches[:8].tolist()}", flush=True
                        )
                    else:
                        print(
                            f"PASS {case} repeat={repeat} n={n} (results, inactive lanes, guards)", flush=True
                        )
                    checks.append({"case": case, "repeat": repeat, "active_lanes": n, "ops": entry["ops"]})
    (output / "result.json").write_text(
        json.dumps(
            {
                "device_id": device,
                "hardware_executed": True,
                "runtime_commit": pin.installed,
                "numerical_checks": len(checks),
                "checks": checks,
                "failures": failures,
                "passed_checks": len(checks) - len(failures),
            },
            indent=2,
        )
        + "\n"
    )

    (output / f"result-{time.time_ns()}.json").write_bytes((output / "result.json").read_bytes())
    assert not failures, f"{len(failures)} of {len(checks)} hardware numerical cases failed; see result.json"


def test_reg_extended_onboard(vpto_onboard_session: Path | None):
    if os.environ.get("PYPTO_RUN_VPTO_ONBOARD") != "1":
        pytest.skip("Build first, then opt in with PYPTO_RUN_VPTO_ONBOARD=1 under task-submit")
    # The autouse fixture redirects the per-test output environment. This
    # precompiled acceptance consumes the original session artifact directory.
    assert vpto_onboard_session is not None
    _run(vpto_onboard_session, int(os.environ["PYPTO_VPTO_DEVICE_ID"]))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--build":
        _build(Path(sys.argv[2]).resolve())
    elif len(sys.argv) == 4 and sys.argv[1] == "--worker":
        _run(Path(sys.argv[2]).resolve(), int(sys.argv[3]))
    else:
        pytest.main([__file__, "-v", *sys.argv[1:]])
