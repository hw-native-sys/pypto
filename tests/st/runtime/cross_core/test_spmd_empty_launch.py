# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Zero-block multi-output launches preserve buffers and explicit task chains."""

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, DataType, PTOTestCase, TensorSpec
from pypto import ir, passes


@pl.jit.incore
def empty_spmd_initialize(
    a: pl.Out[pl.Tensor[[8, 128], pl.FP32]], b: pl.Out[pl.Tensor[[8, 128], pl.FP32]]
) -> tuple[pl.Tensor[[8, 128], pl.FP32], pl.Tensor[[8, 128], pl.FP32]]:
    ta = pl.tile.full([8, 128], dtype=pl.FP32, value=11.0)
    tb = pl.tile.full([8, 128], dtype=pl.FP32, value=23.0)
    a = pl.store(ta, [0, 0], a)
    b = pl.store(tb, [0, 0], b)
    return a, b


@pl.jit.incore
def empty_spmd_producer(
    a: pl.Out[pl.Tensor[[8, 128], pl.FP32]], b: pl.Out[pl.Tensor[[8, 128], pl.FP32]]
) -> tuple[pl.Tensor[[8, 128], pl.FP32], pl.Tensor[[8, 128], pl.FP32]]:
    row = pl.tile.get_block_idx()
    ta = pl.tile.full([1, 128], dtype=pl.FP32, value=3.0)
    tb = pl.tile.full([1, 128], dtype=pl.FP32, value=7.0)
    a = pl.store(ta, [row, 0], a)
    b = pl.store(tb, [row, 0], b)
    return b, a


@pl.jit.incore
def empty_spmd_consumer(
    a: pl.Tensor[[8, 128], pl.FP32],
    b: pl.Tensor[[8, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 128], pl.FP32]],
) -> pl.Tensor[[8, 128], pl.FP32]:
    ta = pl.load(a, [0, 0], [8, 128])
    tb = pl.load(b, [0, 0], [8, 128])
    return pl.store(pl.tile.sub(ta, tb), [0, 0], out)


@pl.jit
def empty_spmd_main(
    ctrl: pl.Tensor[[3], pl.INT32],
    a: pl.Tensor[[8, 128], pl.FP32],
    b: pl.Tensor[[8, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[8, 128], pl.FP32]],
) -> pl.Tensor[[8, 128], pl.FP32]:
    with pl.manual_scope():
        with pl.spmd(1) as prior:
            a, b = empty_spmd_initialize(a, b)
        tid = prior
        oa = a
        ob = b
        for step in pl.range(3):
            n = pl.tensor.read(ctrl, [step])
            with pl.spmd(n, deps=[tid]) as step_tid:
                ob, oa = empty_spmd_producer(oa, ob)
            tid = step_tid
        with pl.spmd(1, deps=[tid]):
            out = empty_spmd_consumer(oa, ob, out)
    return out


class EmptySpmdCase(PTOTestCase):
    __test__ = False

    def __init__(self, count, platform):
        super().__init__(platform=platform)
        self.counts = (count, count, count) if isinstance(count, int) else count

    def get_name(self):
        return "empty_spmd_multi_out_" + "_".join(str(n) for n in self.counts)

    def get_program(self):
        return empty_spmd_main.specialize()

    def define_tensors(self):
        return [
            TensorSpec("ctrl", [3], DataType.INT32, init_value=torch.tensor(self.counts, dtype=torch.int32)),
            TensorSpec("a", [8, 128], DataType.FP32, init_value=-99.0),
            TensorSpec("b", [8, 128], DataType.FP32, init_value=-77.0),
            TensorSpec("out", [8, 128], DataType.FP32, init_value=0.0, is_output=True),
        ]

    def compute_expected(self, tensors, params=None):
        tensors["out"][:] = -12.0
        maximum = max(self.counts)
        if maximum > 0:
            tensors["out"][:maximum] = -4.0


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("count", [-1, 0, 1, 3, 8, (1, 0, 1), (3, 0, 0)])
def test_empty_spmd_multi_output(test_runner, platform, count):
    result = test_runner.run(EmptySpmdCase(count, platform))
    assert result.passed, result.error


class _UseAutomaticDependencies(ir.IRMutator):
    """Exercise the same program using only TensorMap/compiler-derived edges."""

    def visit_runtime_scope_stmt(self, op):
        return ir.RuntimeScopeStmt(False, op.name_hint, body=self.visit_stmt(op.body), span=op.span)

    def visit_submit(self, op):
        return ir.Submit(
            op.op,
            op.args,
            [],
            kwargs=op.kwargs,
            attrs=op.attrs,
            type=op.type,
            span=op.span,
            core_num=op.core_num,
            sync_start=op.sync_start,
            allow_early_resolve=op.allow_early_resolve,
            predicate=op.predicate,
        )


class AutomaticEmptySpmdCase(EmptySpmdCase):
    __test__ = False

    def get_name(self):
        return super().get_name() + "_auto"

    def get_program(self):
        program = passes.convert_to_ssa()(empty_spmd_main.specialize())
        program = passes.outline_incore_scopes()(program)
        program = passes.outline_cluster_scopes()(program)
        return _UseAutomaticDependencies().visit_program(program)


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("count", [0, 3, (1, 0, 1), (3, 0, 0)])
def test_empty_spmd_automatic_dependencies(test_runner, platform, count):
    result = test_runner.run(AutomaticEmptySpmdCase(count, platform))
    assert result.passed, result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
