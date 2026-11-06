# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Default whole-kernel rollout: independent device goldens for each recipe."""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import ir, passes

_PLANNERS = (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)


@pl.jit.incore
def _sub(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.load(b, [0, 0], [16, 32])
    value = pl.tile.sub(x, y)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _div(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.load(b, [0, 0], [16, 32])
    value = pl.tile.div(x, y)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _maximum(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.load(b, [0, 0], [16, 32])
    value = pl.tile.maximum(x, y)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _minimum(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.load(b, [0, 0], [16, 32])
    value = pl.tile.minimum(x, y)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _abs(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.abs(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _exp(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.exp(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _sqrt(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.sqrt(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _neg(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.neg(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _relu(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.relu(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _log(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.log(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _recip(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.recip(x)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _adds(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.adds(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _subs(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.subs(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _muls(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.muls(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _divs(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.divs(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _maximums(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.maximums(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _minimums(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.minimums(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _lrelu(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    value = pl.tile.lrelu(x, 0.5)
    return pl.store(value, [0, 0], out)


@pl.jit.incore
def _full(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    value = pl.tile.full([16, 32], pl.FP32, 0.5)
    return pl.store(value, [0, 0], out)


def _entry(kernel):
    @pl.jit
    def entry(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
        return kernel(a, b, out)

    return entry


_GOLDENS = {
    "sub": lambda a, b: a - b,
    "div": lambda a, b: a / b,
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "abs": lambda a, b: a.abs(),
    "exp": lambda a, b: a.exp(),
    "sqrt": lambda a, b: a.sqrt(),
    "neg": lambda a, b: -a,
    "relu": lambda a, b: a.relu(),
    "log": lambda a, b: a.log(),
    "recip": lambda a, b: a.reciprocal(),
    "adds": lambda a, b: a + 0.5,
    "subs": lambda a, b: a - 0.5,
    "muls": lambda a, b: a * 0.5,
    "divs": lambda a, b: a / 0.5,
    "maximums": lambda a, b: a.clamp(min=0.5),
    "minimums": lambda a, b: a.clamp(max=0.5),
    "lrelu": lambda a, b: torch.where(a > 0, a, a * 0.5),
    "full": lambda a, b: torch.full_like(a, 0.5),
}
_INPUT = torch.linspace(0.25, 1.25, 512).reshape(16, 32)
_RHS = torch.linspace(0.5, 2, 512).reshape(16, 32)
_KERNELS = (
    _sub,
    _div,
    _maximum,
    _minimum,
    _abs,
    _exp,
    _sqrt,
    _neg,
    _relu,
    _log,
    _recip,
    _adds,
    _subs,
    _muls,
    _divs,
    _maximums,
    _minimums,
    _lrelu,
    _full,
)


@st.cases(
    *[
        st.case(
            _entry(kernel),
            _INPUT - 0.75 if name in ("abs", "neg", "relu", "lrelu") else _INPUT,
            _RHS,
            torch.full_like(_INPUT, -777),
            name=f"buffer_default_{name}_{planner.name.lower()}",
            golden=lambda tensors, fn=_GOLDENS[name]: fn(tensors["a"], tensors["b"]),
            memory_planner=planner,
            rtol=1e-4,
            atol=1e-5,
        )
        for kernel, name in zip(_KERNELS, _GOLDENS)
        for planner in _PLANNERS
    ]
)
def test_default_recipe_device_results(case_run, request):
    _assert_buffer_result(case_run, request)


def _assert_buffer_result(case_run, request):
    case_run.assert_passed()
    if request.config.getoption("--precompile-workers") is not None:
        assert case_run.work_dir is not None
        program = ir.deserialize((case_run.work_dir / "buffer_ir.msgpack").read_bytes())
        kernels = [f for f in program.functions.values() if ir.is_incore_type(f.func_type)]
        expected = (
            ir.FunctionIRStage.Buffer
            if case_run.case.platform in ("a2a3", "a2a3sim")
            else ir.FunctionIRStage.Functional
        )
        assert kernels and all(f.ir_stage == expected for f in kernels)


@pl.jit.incore
def _typed_kernel(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32], valid_shape=[13, 24])
    return pl.store(x, [3, 8], out)


@pl.jit
def _typed_entry(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    return _typed_kernel(a, out)


def _typed_golden(tensors):
    result = tensors["out"].clone()
    result[3:16, 8:32] = tensors["a"][:13, :24]
    return result


@st.cases(
    *[
        st.case(
            _typed_entry,
            (torch.arange(512).reshape(16, 32) * 13 - 1777).to(dtype),
            torch.full((24, 48), -777, dtype=dtype),
            name=f"buffer_partial_{str(dtype).removeprefix('torch.')}_{planner.name.lower()}",
            golden=_typed_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)
        for planner in _PLANNERS
    ]
)
def test_default_partial_typed_transfer(case_run, request):
    _assert_buffer_result(case_run, request)


@pl.jit.incore
def _reciprocal_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.tile.recip(x, high_precision=True)
    out = pl.store(y, [0, 0], out)
    return pl.store(x, [16, 0], out)


@pl.jit
def _reciprocal_entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    return _reciprocal_kernel(a, out)


@st.cases(
    *[
        st.case(
            _reciprocal_entry,
            _INPUT,
            torch.full((32, 32), -777.0),
            name=f"buffer_recip_live_input_{planner.name.lower()}",
            golden=lambda tensors: torch.cat((tensors["a"].reciprocal(), tensors["a"])),
            memory_planner=planner,
            rtol=1e-5,
            atol=1e-6,
        )
        for planner in _PLANNERS
    ]
)
def test_default_recip_preserves_live_input(case_run, request):
    _assert_buffer_result(case_run, request)


@pl.jit.incore
def _typed_add(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.load(b, [0, 0], [16, 32])
    return pl.store(pl.add(x, y), [0, 0], out)


@pl.jit.incore
def _typed_mul(a: pl.Tensor, b: pl.Tensor, out: pl.Out[pl.Tensor]):
    x = pl.load(a, [0, 0], [16, 32])
    y = pl.load(b, [0, 0], [16, 32])
    return pl.store(pl.mul(x, y), [0, 0], out)


@st.cases(
    *[
        st.case(
            _entry(kernel),
            (torch.arange(512).reshape(16, 32) % 29 - 14).to(dtype),
            (torch.arange(512).reshape(16, 32) % 17 + 1).to(dtype),
            torch.full((16, 32), -777, dtype=dtype),
            name=f"buffer_{kernel.__name__}_{str(dtype).removeprefix('torch.')}_{planner.name.lower()}",
            golden=lambda tensors, fn=golden: fn(tensors["a"], tensors["b"]),
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for kernel, golden in ((_typed_add, torch.add), (_typed_mul, torch.mul))
        for dtype in (torch.float16, torch.int32)
        for planner in _PLANNERS
    ]
)
def test_default_typed_arithmetic(case_run, request):
    _assert_buffer_result(case_run, request)


@pl.jit.incore
def _softmax_full(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    source = pl.load(a, [0, 0], [64, 64])
    scratch = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    maximum = pl.row_max(source, scratch)
    shifted = pl.row_expand_sub(source, maximum)
    exponent = pl.exp(shifted)
    total = pl.row_sum(exponent, scratch)
    result = pl.row_expand_div(exponent, total)
    out = pl.store(result, [0, 0], out)
    return pl.store(source, [64, 0], out)


@pl.jit.incore
def _softmax_partial(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    source = pl.load(a, [0, 0], [16, 128], valid_shape=[13, 117])
    scratch = pl.create_tile([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    maximum = pl.row_max(source, scratch)
    shifted = pl.row_expand_sub(source, maximum)
    exponent = pl.exp(shifted)
    total = pl.row_sum(exponent, scratch)
    result = pl.row_expand_div(exponent, total)
    out = pl.store(result, [0, 0], out)
    return pl.store(source, [16, 0], out)


@pl.jit
def _softmax_core_group(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    # Match the public softmax example's outlining and lane-splitting path.
    with pl.at(level=pl.Level.CORE_GROUP):
        source = pl.load(a, [0, 0], [64, 64])
        scratch = pl.create_tile([64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        maximum = pl.row_max(source, scratch)
        shifted = pl.row_expand_sub(source, maximum)
        exponent = pl.exp(shifted)
        total = pl.row_sum(exponent, scratch)
        result = pl.row_expand_div(exponent, total)
        out = pl.store(result, [0, 0], out)
        out = pl.store(source, [64, 0], out)
    return out


def _softmax_entry(kernel):
    @pl.jit
    def entry(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return kernel(a, out)

    return entry


def _softmax_golden(tensors, valid):
    source, output = tensors["a"], tensors["out"].clone()
    rows, cols = valid
    output[:rows, :cols] = torch.softmax(source[:rows, :cols], dim=-1)
    start = source.shape[0]
    output[start : start + rows, :cols] = source[:rows, :cols]
    return output


@st.cases(
    *[
        st.case(
            kernel,
            torch.randn(shape, generator=torch.Generator().manual_seed(19)) * 4,
            torch.full((2 * shape[0], shape[1]), -777.0),
            name=f"buffer_softmax_{name}_{planner.name.lower()}",
            golden=lambda tensors, valid=valid: _softmax_golden(tensors, valid),
            memory_planner=planner,
            rtol=1e-4,
            atol=1e-5,
        )
        for kernel, name, shape, valid in (
            (_softmax_entry(_softmax_full), "full", (64, 64), (64, 64)),
            (_softmax_entry(_softmax_partial), "partial", (16, 128), (13, 117)),
            (_softmax_core_group, "core_group", (64, 64), (64, 64)),
        )
        for planner in _PLANNERS
    ]
)
def test_default_softmax_device_results(case_run, request):
    _assert_buffer_result(case_run, request)


@pl.jit.incore
def _row_vector_states(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    source = pl.load(a, [0, 0], [16, 64])
    scratch = pl.create_tile([16, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    maximum = pl.row_max(source, scratch)
    narrowed = pl.tile.set_validshape(maximum, 13, 1)
    partial = pl.tile.set_validshape(source, 13, 64)
    shifted_partial = pl.row_expand_sub(partial, narrowed)
    out = pl.store(shifted_partial, [0, 0], out)
    shifted = pl.row_expand_sub(source, maximum)
    out = pl.store(shifted, [16, 0], out)
    total = pl.row_sum(source, scratch)
    scaled = pl.row_expand_div(source, total)
    return pl.store(scaled, [32, 0], out)


def _row_vector_golden(tensors):
    source, output = tensors["a"], tensors["out"].clone()
    maximum = torch.amax(source, dim=-1, keepdim=True)
    output[:13] = source[:13] - maximum[:13]
    output[16:32] = source - maximum
    output[32:] = source / torch.sum(source, dim=-1, keepdim=True)
    return output


@st.cases(
    *[
        st.case(
            _softmax_entry(_row_vector_states),
            torch.randn((16, 64), generator=torch.Generator().manual_seed(37)) + 4,
            torch.full((48, 64), -777.0),
            name=f"buffer_row_vectors_{planner.name.lower()}",
            golden=_row_vector_golden,
            memory_planner=planner,
            rtol=1e-4,
            atol=1e-5,
        )
        for planner in _PLANNERS
    ]
)
def test_default_row_vector_states(case_run, request):
    # A narrowed column descriptor must not change the original value's valid
    # rows; matrix broadcasts expose the values and preserve untouched GM rows.
    _assert_buffer_result(case_run, request)


def _cast_select_entry(scalar, valid):
    rows, cols = valid

    @pl.jit.incore
    def kernel(a: pl.Tensor, b: pl.Tensor, out: pl.InOut[pl.Tensor]):
        a_tile = pl.load(a, [0, 0], [8, 64], valid_shape=[rows, cols])
        b_tile = pl.load(b, [0, 0], [8, 64], valid_shape=[rows, cols])
        x = pl.cast(a_tile, pl.FP32, mode="rint")
        y = pl.cast(b_tile, pl.FP32, mode="rint")
        if scalar:
            mask = pl.tile.cmps(x, 0.0, cmp_type=4)
            scalar_scratch = pl.create_tile([1, 64], dtype=pl.FP32)
            selected = pl.tile.sels(mask, x, scalar_scratch, -3.0)
        else:
            mask = pl.tile.cmp(x, y, cmp_type=4)
            tile_scratch = pl.create_tile([1, 16], dtype=pl.UINT32)
            selected = pl.tile.sel(mask, x, y, tile_scratch)
        result = pl.cast(selected, pl.FP16, mode="rint")
        out = pl.store(result, [0, 0], out)
        out = pl.store(a_tile, [8, 0], out)
        return pl.store(b_tile, [16, 0], out)

    @pl.jit
    def entry(a: pl.Tensor, b: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return kernel(a, b, out)

    return entry


def _cast_select_golden(tensors, scalar, valid):
    rows, cols = valid
    a, b = tensors["a"], tensors["b"]
    out = tensors["out"].clone()
    lhs, rhs = a[:rows, :cols].float(), b[:rows, :cols].float()
    condition = lhs > (0.0 if scalar else rhs)
    out[:rows, :cols] = torch.where(condition, lhs, -3.0 if scalar else rhs).half()
    out[8 : 8 + rows, :cols] = a[:rows, :cols]
    out[16 : 16 + rows, :cols] = b[:rows, :cols]
    return out


@st.cases(
    *[
        st.case(
            _cast_select_entry(scalar, valid),
            ((torch.arange(512).reshape(8, 64) % 17) - 8).half() / 4,
            torch.flip(((torch.arange(512).reshape(8, 64) % 17) - 8).half() / 4, [1]),
            torch.full((24, 64), -777.0, dtype=torch.float16),
            name=f"buffer_cast_select_{'scalar' if scalar else 'tile'}_{valid[1]}_{planner.name.lower()}",
            golden=lambda tensors, scalar=scalar, valid=valid: _cast_select_golden(tensors, scalar, valid),
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for scalar in (False, True)
        for valid in ((8, 64), (7, 61))
        for planner in _PLANNERS
    ]
)
def test_default_cast_compare_select(case_run, request):
    _assert_buffer_result(case_run, request)


def _half_rounding_entry(valid, mode):
    rows, cols = valid

    @pl.jit.incore
    def kernel(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
        source = pl.load(a, [0, 0], [8, 64], valid_shape=[rows, cols])
        result = pl.cast(source, pl.FP16, mode=mode, saturation_mode="off")
        return pl.store(result, [0, 0], out)

    return _softmax_entry(kernel)


def _half_rounding_data():
    data = torch.linspace(-3, 3, 512).reshape(8, 64)
    # Consecutive half-way values round down/up under ties-to-even. Negative
    # values and tiny subnormals expose sign and underflow handling as well.
    data[0, :8] = torch.tensor(
        [1 + 2**-11, 1 + 3 * 2**-11, -1 - 2**-11, -1 - 3 * 2**-11, 2**-25, 3 * 2**-25, -(2**-25), -3 * 2**-25]
    )
    return data


def _half_rounding_golden(tensors, valid):
    rows, cols = valid
    result = tensors["out"].clone()
    result[:rows, :cols] = tensors["a"][:rows, :cols].half()
    return result


@st.cases(
    *[
        st.case(
            _half_rounding_entry(valid, mode),
            _half_rounding_data(),
            torch.full((8, 64), -777.0, dtype=torch.float16),
            name=f"buffer_half_rounding_{mode}_{valid[1]}_{planner.name.lower()}",
            golden=lambda tensors, valid=valid: _half_rounding_golden(tensors, valid),
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for mode in ("none", "rint")
        for valid in ((8, 64), (7, 61))
        for planner in _PLANNERS
    ]
)
def test_default_half_rounding(case_run, request):
    _assert_buffer_result(case_run, request)


def _indexed_gather_entry():
    @pl.jit.incore
    def kernel(a: pl.Tensor, idx: pl.Tensor, out: pl.InOut[pl.Tensor]):
        source = pl.load(a, [0, 0], [8, 64])
        indices = pl.load(idx, [0, 0], [4, 32])
        scratch = pl.create_tile([4, 32], dtype=pl.INT32)
        gathered = pl.tile.gather(source, indices, scratch)
        out = pl.store(gathered, [0, 0], out)
        return pl.store(source, [4, 0], out)

    @pl.jit
    def entry(a: pl.Tensor, idx: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return kernel(a, idx, out)

    return entry


def _indexed_gather_golden(tensors):
    result = tensors["out"].clone()
    result[:4, :32] = tensors["a"].reshape(-1)[tensors["idx"].long()]
    result[4:] = tensors["a"]
    return result


@st.cases(
    *[
        st.case(
            _indexed_gather_entry(),
            torch.arange(512).reshape(8, 64).to(dtype),
            ((torch.arange(128).reshape(4, 32) * 37) % 512).to(torch.int32),
            torch.full((12, 64), -777, dtype=dtype),
            name=f"buffer_indexed_gather_{dtype}_{planner.name.lower()}",
            golden=_indexed_gather_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for dtype in (torch.float16, torch.float32, torch.int32)
        for planner in _PLANNERS
    ]
)
def test_default_indexed_gather(case_run, request):
    _assert_buffer_result(case_run, request)


def _indexed_scatter_entry(valid):
    rows, cols = valid

    @pl.jit.incore
    def kernel(a: pl.Tensor, values: pl.Tensor, idx: pl.Tensor, out: pl.InOut[pl.Tensor]):
        old = pl.load(a, [0, 0], [8, 64])
        source = pl.load(values, [0, 0], [4, 32], valid_shape=[rows, cols])
        indices = pl.load(idx, [0, 0], [4, 32], valid_shape=[rows, cols])
        zero = pl.tile.full([8, 64], pl.FP32, 0.0)
        scattered = pl.tile.scatter(zero, source, indices)
        ones = pl.tile.full([4, 32], pl.FP32, 1.0)
        active_ones = pl.tile.set_validshape(ones, rows, cols)
        mask_zero = pl.tile.full([8, 64], pl.FP32, 0.0)
        written = pl.tile.scatter(mask_zero, active_ones, indices)
        predicate = pl.tile.cmps(written, 0.0, cmp_type=1)
        scratch = pl.create_tile([1, 16], dtype=pl.UINT32)
        result = pl.tile.sel(predicate, scattered, old, scratch)
        out = pl.store(result, [0, 0], out)
        out = pl.store(old, [8, 0], out)
        return pl.store(source, [16, 0], out)

    @pl.jit
    def entry(a: pl.Tensor, values: pl.Tensor, idx: pl.Tensor, out: pl.InOut[pl.Tensor]):
        return kernel(a, values, idx, out)

    return entry


def _indexed_scatter_golden(tensors, valid):
    rows, cols = valid
    result = tensors["out"].clone()
    result[:8] = tensors["a"]
    result[8:16] = tensors["a"]
    result[16 : 16 + rows, :cols] = tensors["values"][:rows, :cols]
    # Distinct values at repeated indices expose scan order; negative base
    # sentinels expose lost preservation or an accidentally skipped scatter.
    for row in range(rows):
        for col in range(cols):
            result[:8].view(-1)[tensors["idx"][row, col]] = tensors["values"][row, col]
    return result


@st.cases(
    *[
        st.case(
            _indexed_scatter_entry(valid),
            -torch.arange(1, 513).reshape(8, 64).float(),
            torch.arange(1, 129).reshape(4, 32).float(),
            ((torch.arange(128).reshape(4, 32) // 2 * 7) % 512).to(torch.int32),
            torch.full((20, 64), -777.0),
            name=f"buffer_indexed_scatter_{valid[1]}_{planner.name.lower()}",
            golden=lambda tensors, valid=valid: _indexed_scatter_golden(tensors, valid),
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for valid in ((4, 32), (3, 29))
        for planner in _PLANNERS
    ]
)
def test_default_indexed_scatter(case_run, request):
    _assert_buffer_result(case_run, request)


def _tensor_scatter_entry():
    @pl.jit.incore
    def kernel(base: pl.Tensor, index: pl.Tensor, values: pl.Tensor, out: pl.Out[pl.Tensor]):
        result = pl.tensor.scatter(base, dim=-1, index=index, src=values)
        return pl.assemble(out, result, [0, 0])

    @pl.jit
    def entry(base: pl.Tensor, index: pl.Tensor, values: pl.Tensor, out: pl.Out[pl.Tensor]):
        return kernel(base, index, values, out)

    return entry


def _tensor_scatter_golden(tensors):
    result = tensors["base"].clone()
    for col in range(32):
        result[0, tensors["index"][0, col]] = tensors["values"][0, col]
    return result


@st.cases(
    *[
        st.case(
            _tensor_scatter_entry(),
            -torch.arange(1, 65).reshape(1, 64).float(),
            (torch.arange(32).reshape(1, 32) // 2 * 3).to(torch.int32),
            torch.arange(1, 33).reshape(1, 32).float(),
            torch.empty((1, 64)),
            name=f"buffer_tensor_scatter_{planner.name.lower()}",
            golden=_tensor_scatter_golden,
            memory_planner=planner,
            rtol=0,
            atol=0,
        )
        for planner in _PLANNERS
    ]
)
def test_default_tensor_scatter(case_run, request):
    _assert_buffer_result(case_run, request)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
