# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for NormalizeTupleReturnOrder pass."""

import re

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def test_normalize_tuple_return_order_runs_immediately_before_init_mem_ref():
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    names = pm.get_pass_names()
    i_norm = names.index("NormalizeTupleReturnOrder")
    i_init = names.index("InitMemRef")
    assert i_norm == i_init - 1, (i_norm, i_init, names)


def test_allocate_memory_addr_is_last_tile_stage_pass():
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    names = pm.get_pass_names()
    assert names[-1] == "AllocateMemoryAddr"


def test_normalize_tuple_return_order_same_position_in_debug_tile_pipeline():
    pm = PassManager.get_strategy(OptimizationStrategy.DebugTileOptimization)
    names = pm.get_pass_names()
    assert names.index("NormalizeTupleReturnOrder") == names.index("InitMemRef") - 1


@pytest.fixture(autouse=True)
def _reset_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def test_default_pipeline_runs_on_incore_single_out():
    @pl.program
    class TileOnlyProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            tile_a = pl.load(a, [0, 0], [16, 16])
            tile_b = pl.load(b, [0, 0], [16, 16])
            result = pl.add(tile_a, tile_b)
            out = pl.store(result, [0, 0], out)
            return out

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    pm.run_passes(TileOnlyProgram)


def test_mixed_kernel_incore_return_order_matches_out_params():
    """Regression for #814 / #811: tuple return slots align with Out param order after the pass.

    Same pattern as ``test_mixed_loop_carried_and_full_tuple_return`` in
    ``test_orchestration_codegen.py``.
    Outlined first incore kernel is ``main_incore_0`` (not AIC/AIV split — two separate incore scopes).
    """

    @pl.program
    class MixedReturnProgram:
        @pl.function
        def main(
            self,
            src: pl.Tensor[[4, 16], pl.FP32],
            final_out: pl.Out[pl.Tensor[[4, 16], pl.FP32]],
        ) -> pl.Tensor[[4, 16], pl.FP32]:
            dst = pl.create_tensor([4, 16], dtype=pl.FP32)
            acc = pl.create_tensor([4, 16], dtype=pl.FP32)
            with pl.incore():
                for i in pl.range(4):
                    row = pl.slice(src, [1, 16], [i, 0])
                    dst = pl.assemble(dst, row, [i, 0])
                full_view = pl.slice(src, [4, 16], [0, 0])
                acc = pl.assemble(acc, full_view, [0, 0])
            with pl.incore():
                dst_tile = pl.slice(dst, [4, 16], [0, 0])
                acc_tile = pl.slice(acc, [4, 16], [0, 0])
                result = pl.add(dst_tile, acc_tile)
                final_out = pl.assemble(final_out, result, [0, 0])
            return final_out

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    prog = pm.run_passes(MixedReturnProgram)

    fn = prog.get_function("main_incore_0")
    assert fn is not None

    out_param_names = [
        fn.params[i].name_hint
        for i in range(len(fn.params))
        if fn.param_directions[i] in (ir.ParamDirection.Out, ir.ParamDirection.InOut)
    ]
    assert len(out_param_names) == 2, out_param_names

    printed = fn.as_python()
    ret_line_m = re.search(r"^\s*return\s+(.+)$", printed, re.MULTILINE)
    assert ret_line_m is not None, printed[:2000]
    ret_rest = ret_line_m.group(1).strip()
    if ret_rest.startswith("("):
        inner = ret_rest[1 : ret_rest.rindex(")")]
        parts = inner.replace("\n", " ")
        return_vars = [x.strip() for x in parts.split(",") if x.strip()]
    else:
        return_vars = [x.strip() for x in ret_rest.split(",") if x.strip()]
    assert len(return_vars) == 2, return_vars

    def _base(name: str) -> str:
        return name.split("__", 1)[0]

    assert _base(return_vars[0]) == _base(out_param_names[0]), (return_vars, out_param_names)
    assert _base(return_vars[1]) == _base(out_param_names[1]), (return_vars, out_param_names)


def test_split_vector_then_normalize_then_init_mem_ref_chain():
    """Run SplitVectorKernel → NormalizeTupleReturnOrder → InitMemRef on post-tile IR.

    Apply the same Default-strategy passes that run *before* ``SplitVectorKernel`` so the
    program is in the SSA / mixed-kernel / tile shape this trio expects, then compose the
    three passes explicitly (mirrors the tail of the full pipeline).
    """

    @pl.program
    class TileOnlyProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            tile_a = pl.load(a, [0, 0], [16, 16])
            tile_b = pl.load(b, [0, 0], [16, 16])
            result = pl.add(tile_a, tile_b)
            out = pl.store(result, [0, 0], out)
            return out

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    i_split = pm.get_pass_names().index("SplitVectorKernel")
    p = TileOnlyProgram
    for i in range(i_split):
        p = pm.passes[i](p)
    p = passes.split_vector_kernel()(p)
    p2 = passes.normalize_tuple_return_order()(p)
    passes.init_mem_ref()(p2)
