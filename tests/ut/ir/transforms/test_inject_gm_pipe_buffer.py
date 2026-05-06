# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the InjectGMPipeBuffer pass (Ascend910B-gated)."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _run_inject(program: ir.Program) -> ir.Program:
    """Run convert_to_ssa + inject_gm_pipe_buffer on already-split AIC/AIV input.

    These tests exercise InjectGMPipeBuffer in isolation: Before programs are
    written with explicit AIC/AIV/Group functions plus existing pipe setup,
    so ExpandMixedKernel is not in the loop and does not contribute to the
    Expected output.
    """
    with passes.PassContext([], ir.VerificationLevel.NONE):
        return passes.inject_gm_pipe_buffer()(passes.convert_to_ssa()(program))


def test_gm_pipe_injection_preserves_split_mode_for_a2a3_cross_core_functions():
    @pl.program
    class Before:
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

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), pipe_buf, dir_mask=2, slot_size=512)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=512)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out, gm_pipe_buffer)
            self.vector_producer(a, out, gm_pipe_buffer)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                [1],
                dtype=pl.FP32,
                layout=pl.TensorLayout.ND,
                manual_dep=True,
            )
            updated = self.group_func(a, out, gm_pipe_buffer_0)
            return updated

    After = _run_inject(Before)
    ir.assert_structural_equal(After, Expected)


def test_gm_pipe_injection_sizes_multiple_single_direction_pipes():
    """InjectGMPipeBuffer keeps GM workspace sizing as a codegen concern."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_kernel(self):
            c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_kernel")
            v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=32768, base=0x1000)
            pl.aic_initialize_pipe(c2v_peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), v2c_buf, dir_mask=2, slot_size=4096, id=1)

        @pl.function(type=pl.FunctionType.AIV)
        def vector_kernel(self):
            c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=65536, base=0x2000)
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_kernel")
            pl.aiv_initialize_pipe(c2v_buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=4096, id=1)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(self):
            self.cube_kernel()
            self.vector_kernel()

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self):
            self.group_func()

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC)
        def cube_kernel(self, gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]]):
            c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_kernel")
            v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=32768, base=0x1000)
            pl.aic_initialize_pipe(c2v_peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), v2c_buf, dir_mask=2, slot_size=4096, id=1)

        @pl.function(type=pl.FunctionType.AIV)
        def vector_kernel(self, gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]]):
            c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=65536, base=0x2000)
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_kernel")
            pl.aiv_initialize_pipe(c2v_buf, pl.const(0, pl.INT32), dir_mask=1, slot_size=8192, id=0)
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=4096, id=1)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(self, gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]]):
            self.cube_kernel(gm_pipe_buffer)
            self.vector_kernel(gm_pipe_buffer)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self):
            gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                [1],
                dtype=pl.FP32,
                layout=pl.TensorLayout.ND,
                manual_dep=True,
            )
            self.group_func(gm_pipe_buffer_0)

    After = _run_inject(Before)
    ir.assert_structural_equal(After, Expected)


def test_gm_pipe_injection_handles_nested_initialize_pipe_ops():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            if True:
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
            if True:
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

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            if True:
                pl.aic_initialize_pipe(pl.const(0, pl.INT32), pipe_buf, dir_mask=2, slot_size=512)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            if True:
                pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=512)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out, gm_pipe_buffer)
            self.vector_producer(a, out, gm_pipe_buffer)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                [1],
                dtype=pl.FP32,
                layout=pl.TensorLayout.ND,
                manual_dep=True,
            )
            updated = self.group_func(a, out, gm_pipe_buffer_0)
            return updated

    After = _run_inject(Before)
    ir.assert_structural_equal(After, Expected)


def test_gm_pipe_buffer_per_call_allocation():
    """Each cross-core Group call gets its own gm_pipe_buffer tensor.create.

    When an orchestration function calls multiple Group functions that use
    cross-core pipes, each call must independently allocate its own
    gm_pipe_buffer via a separate tensor.create.  Sharing a single buffer
    causes scope escape and synchronization conflicts.
    """

    @pl.program
    class Before:
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
            out = self.group_func(a, out)
            out = self.group_func(a, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), pipe_buf, dir_mask=2, slot_size=512)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=512)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out, gm_pipe_buffer)
            self.vector_producer(a, out, gm_pipe_buffer)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                [1],
                dtype=pl.FP32,
                layout=pl.TensorLayout.ND,
                manual_dep=True,
            )
            out_1: pl.Tensor[[16, 16], pl.FP16] = self.group_func(a, out, gm_pipe_buffer_0)
            gm_pipe_buffer_1: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                [1],
                dtype=pl.FP32,
                layout=pl.TensorLayout.ND,
                manual_dep=True,
            )
            out_2: pl.Tensor[[16, 16], pl.FP16] = self.group_func(a, out_1, gm_pipe_buffer_1)
            return out_2

    After = _run_inject(Before)
    ir.assert_structural_equal(After, Expected)


def test_gm_pipe_buffer_per_call_inside_for_loop():
    """Per-call gm_pipe_buffer must also work inside for-loops."""

    @pl.program
    class Before:
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
            for b in pl.range(4):
                out = self.group_func(a, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), pipe_buf, dir_mask=2, slot_size=512)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=512)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out, gm_pipe_buffer)
            self.vector_producer(a, out, gm_pipe_buffer)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            for b, (out_iter,) in pl.range(4, init_values=(out,)):
                gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                    [1],
                    dtype=pl.FP32,
                    layout=pl.TensorLayout.ND,
                    manual_dep=True,
                )
                out_new: pl.Tensor[[16, 16], pl.FP16] = self.group_func(a, out_iter, gm_pipe_buffer_0)
                out_yield: pl.Tensor[[16, 16], pl.FP16] = pl.yield_(out_new)
            return out_yield

    After = _run_inject(Before)
    ir.assert_structural_equal(After, Expected)


def test_gm_pipe_buffer_param_direction_is_out():
    """The gm_pipe_buffer parameter must have Out direction.

    Verified structurally by declaring `gm_pipe_buffer: pl.Out[...]` in the
    Expected functions, which makes assert_structural_equal check the
    parameter direction.
    """

    @pl.program
    class Before:
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

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIC, attrs={"split": pl.SplitMode.UP_DOWN})
        def cube_consumer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            pipe_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
            pl.aic_initialize_pipe(pl.const(0, pl.INT32), pipe_buf, dir_mask=2, slot_size=512)
            received: pl.Tile[[16, 16], pl.FP16, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
            pl.tfree_to_aiv(received)
            updated: pl.Tensor[[16, 16], pl.FP16] = pl.store(received, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.AIV, attrs={"split": pl.SplitMode.UP_DOWN})
        def vector_producer(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ):
            v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_consumer")
            pl.aiv_initialize_pipe(pl.const(0, pl.INT32), v2c_peer, dir_mask=2, slot_size=512)
            tile_a: pl.Tile[[16, 16], pl.FP16] = pl.load(a, [0, 0], [16, 16])
            pl.tpush_to_aic(tile_a, split=0)

        @pl.function(type=pl.FunctionType.Group)
        def group_func(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
            gm_pipe_buffer: pl.Out[pl.Tensor[[1], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube_consumer(a, out, gm_pipe_buffer)
            self.vector_producer(a, out, gm_pipe_buffer)
            return updated

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            gm_pipe_buffer_0: pl.Tensor[[1], pl.FP32] = pl.tensor.create(
                [1],
                dtype=pl.FP32,
                layout=pl.TensorLayout.ND,
                manual_dep=True,
            )
            updated = self.group_func(a, out, gm_pipe_buffer_0)
            return updated

    After = _run_inject(Before)
    ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
