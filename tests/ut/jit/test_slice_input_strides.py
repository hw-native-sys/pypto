# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Row addressing for identically shaped views passed to the same computation."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.runtime import RunConfig


@pl.jit.inline
def add_one(x: pl.Tensor, out: pl.Tensor):
    for r in pl.parallel(0, 32, 8):
        with pl.at(level=pl.Level.CORE_GROUP):
            for c in pl.range(0, 4096, 256):
                out = pl.assemble(out, pl.add(x[r : r + 8, c : c + 256], 1.0), [r, c])
    return out


@pl.jit
def stride_views(
    backing: pl.Tensor[[32, 32768], pl.FP32],
    dense_out: pl.Out[pl.Tensor[[32, 4096], pl.FP32]],
    strided_out: pl.Out[pl.Tensor[[32, 4096], pl.FP32]],
):
    dense = pl.reshape(backing[:4, :], [32, 4096])
    dense_out = add_one(dense, dense_out)
    strided = backing[:, :4096]
    strided_out = add_one(strided, strided_out)
    return dense_out, strided_out


def test_outlined_calls_use_each_input_pitch(tmp_path):
    program = stride_views.lower()

    class InputPitches(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.pitches = []

        def visit_call(self, op):
            if isinstance(op.op, ir.GlobalVar):
                kernel = program.get_function(op.op.name)
                for param, direction in zip(kernel.params, kernel.param_directions, strict=True):
                    if direction == ir.ParamDirection.In and isinstance(param.type, ir.TensorType):
                        view = param.type.tensor_view
                        pitch = view.stride[0] if view and view.stride else param.type.shape[1]
                        assert isinstance(pitch, ir.ConstInt)
                        self.pitches.append(pitch.value)
            super().visit_call(op)

    calls = InputPitches()
    calls.visit_function(program.get_function("stride_views"))
    # Second-row element offsets in backing storage, in dispatch order.
    assert calls.pitches == [4096, 32768]
    compiled = stride_views.compile(config=RunConfig(save_kernels_dir=str(tmp_path)))
    assert compiled is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
