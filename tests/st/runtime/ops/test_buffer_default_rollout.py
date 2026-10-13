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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
