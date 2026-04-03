# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""a2a3-specific regression tests for ExpandMixedKernel cross-core handling."""

from typing import cast

import pypto.language as pl
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.printer import python_print


def test_gm_pipe_injection_preserves_split_mode_for_a2a3_cross_core_functions():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class CrossCoreProgram:
        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out)
            self.vector_producer(a, out)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.group_func(a, out)
            return updated

    with passes.PassContext([], ir.VerificationLevel.NONE):
        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(CrossCoreProgram))
        )
    vector_producer = transformed.get_function("vector_producer")
    cube_consumer = transformed.get_function("cube_consumer")
    assert vector_producer is not None
    assert cube_consumer is not None
    assert vector_producer.func_type == ir.FunctionType.AIV
    assert cube_consumer.func_type == ir.FunctionType.AIC
    assert vector_producer.split == ir.SplitMode.UP_DOWN
    assert cube_consumer.split == ir.SplitMode.UP_DOWN

    printed = python_print(transformed)
    assert "__gm_pipe_buffer" in printed


def test_gm_pipe_injection_handles_nested_initialize_pipe_ops():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class NestedPipeProgram:
        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            if 1:
                pl.aiv_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=v2c_peer)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            if 1:
                pl.aic_initialize_pipe(dir_mask=2, slot_size=512, v2c_consumer_buf=pipe_buf)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out)
            self.vector_producer(a, out)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.group_func(a, out)
            return updated

    with passes.PassContext([], ir.VerificationLevel.NONE):
        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(NestedPipeProgram))
        )

    vector_producer = transformed.get_function("vector_producer")
    cube_consumer = transformed.get_function("cube_consumer")
    assert vector_producer is not None
    assert cube_consumer is not None
    assert vector_producer.params[-1].name_hint == "__gm_pipe_buffer"
    assert cube_consumer.params[-1].name_hint == "__gm_pipe_buffer"
    vector_buffer_type = cast(ir.TensorType, vector_producer.params[-1].type)
    cube_buffer_type = cast(ir.TensorType, cube_consumer.params[-1].type)
    vector_buffer_dim = cast(ir.ConstInt, vector_buffer_type.shape[0])
    cube_buffer_dim = cast(ir.ConstInt, cube_buffer_type.shape[0])
    assert vector_buffer_dim.value == 1024
    assert cube_buffer_dim.value == 1024
