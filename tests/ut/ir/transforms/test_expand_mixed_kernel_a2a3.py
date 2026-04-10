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
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.printer import python_print


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure Ascend910B backend before each test and reset afterward."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def test_gm_pipe_injection_preserves_split_mode_for_a2a3_cross_core_functions():
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


def test_v2c_boundary_uses_nz_layout_on_a2a3():
    """On Ascend910B, cross-core push needs no layout adaptation on the AIV side.

    Ascend910B routes push/pop through ub → gm → mat. The ub → gm transfer
    uses ND layout directly, so no tile.move is needed before tpush_to_aic.
    The AIC tpop still lands in Mat with NZ layout (col_major blayout), and a
    subsequent Mat → Left tile.move resolves the final layout.

    The expected behavior on 910B:
      - AIV side: `tpush_to_aic(source_tile)` directly, no NZ/ZN tmov.
      - AIC side: `tpop_from_aiv()` lands in Mat with NZ layout (col_major
        blayout), followed by a `Mat → Left` `tile.move`.
    """

    @pl.program
    class MixedMatmul:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            x_tile = pl.load(x, [0, 0], [16, 128])
            # Direct Vec -> Left boundary (exercises BuildCrossCoreTransferView with Left)
            x_left = pl.move(x_tile, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            z_vec = pl.move(
                z_tile,
                target_memory=pl.MemorySpace.Vec,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.none_box,
            )
            out_0: pl.Tensor[[16, 64], pl.FP32] = pl.store(z_vec, [0, 0], out_0)
            return out_0

    with passes.PassContext([], ir.VerificationLevel.NONE):
        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(MixedMatmul))
        )

    aic_func = transformed.get_function("main_incore_0_aic")
    aiv_func = transformed.get_function("main_incore_0_aiv")
    assert aic_func is not None
    assert aiv_func is not None

    aic_printed = python_print(aic_func)
    aiv_printed = python_print(aiv_func)

    # AIV side: push directly with no NZ/ZN layout adaptation (910B ub->gm uses ND).
    assert "pl.tile.tpush_to_aic(" in aiv_printed
    assert "_nz" not in aiv_printed
    assert "_zn" not in aiv_printed

    # AIC side: tpop lands in Mat with NZ layout (col_major blayout), and
    # the subsequent Mat -> Left move is present.
    assert "pl.tile.tpop_from_aiv(" in aic_printed
    assert "pl.Mem.Mat" in aic_printed
    assert "pl.Mem.Left" in aic_printed
    assert "target_memory=pl.Mem.Left" in aic_printed
    # The tpop result type carries NZ layout (col_major blayout) because
    # 910B transfers go through Mat which only supports NZ.
    assert "blayout=pl.TileLayout.col_major" in aic_printed


def test_c2v_boundary_preserves_vec_pop_layout_on_a2a3():
    """On Ascend910B, C2V Vec pops must stay in the final Vec layout.

    The A2A3 GM-backed pipe consumer materializes the popped tile through an ND
    GlobalTensor. PTO-ISA does not support loading that ND buffer into an NZ
    Vec tile, so ExpandMixedKernel must not introduce an NZ Vec bridge tile
    here.
    """

    @pl.program
    class MixedMatmul:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 128], pl.BF16],
            y: pl.Tensor[[128, 64], pl.BF16],
            out_0: pl.Out[pl.Tensor[[16, 64], pl.FP32]],
        ) -> pl.Tensor[[16, 64], pl.FP32]:
            x_mat = pl.load(x, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            x_left = pl.move(x_mat, target_memory=pl.MemorySpace.Left)
            y_mat = pl.load(y, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            y_right = pl.move(y_mat, target_memory=pl.MemorySpace.Right)
            z_tile = pl.matmul(x_left, y_right)
            z_vec = pl.move(
                z_tile,
                target_memory=pl.MemorySpace.Vec,
                blayout=pl.TileLayout.row_major,
                slayout=pl.TileLayout.none_box,
            )
            out_0 = pl.store(z_vec, [0, 0], out_0)
            return out_0

    with passes.PassContext([], ir.VerificationLevel.NONE):
        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(MixedMatmul))
        )

    aiv_func = transformed.get_function("main_incore_0_aiv")
    assert aiv_func is not None

    aiv_printed = python_print(aiv_func)
    assert "pl.tile.tpop_from_aic(" in aiv_printed
    assert "pl.tile.move(" not in aiv_printed
    assert "blayout=pl.TileLayout.col_major" not in aiv_printed
    assert "slayout=pl.TileLayout.row_major" not in aiv_printed


def test_gm_pipe_buffer_per_call_allocation():
    """Each cross-core Group call gets its own gm_pipe_buffer tensor.create.

    When an orchestration function calls multiple Group functions that use
    cross-core pipes, each call must independently allocate its own
    gm_pipe_buffer via a separate tensor.create.  Sharing a single buffer
    causes scope escape and synchronization conflicts.
    """

    @pl.program
    class MultiCallProgram:
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

    with passes.PassContext([], ir.VerificationLevel.NONE):
        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(MultiCallProgram))
        )

    orch_func = transformed.get_function("main")
    assert orch_func is not None
    orch_printed = python_print(orch_func)

    # Each call should have its own tensor.create with a unique name
    creates = [
        line for line in orch_printed.split("\n") if "gm_pipe_buffer" in line and "tensor.create" in line
    ]
    assert len(creates) == 2, f"Expected 2 tensor.creates, got {len(creates)}: {creates}"
    assert "gm_pipe_buffer_0" in orch_printed
    assert "gm_pipe_buffer_1" in orch_printed


def test_gm_pipe_buffer_per_call_inside_for_loop():
    """Per-call gm_pipe_buffer must also work inside for-loops."""

    @pl.program
    class LoopedCrossCore:
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

    with passes.PassContext([], ir.VerificationLevel.NONE):
        transformed = passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(passes.convert_to_ssa()(LoopedCrossCore))
        )

    orch_func = transformed.get_function("main")
    assert orch_func is not None
    orch_printed = python_print(orch_func)
    lines = orch_printed.strip().split("\n")

    # tensor.create must appear inside the for-loop (after the for header)
    for_line_idx = None
    create_line_idx = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if for_line_idx is None and stripped.startswith("for ") and "pl.range" in stripped:
            for_line_idx = idx
        if create_line_idx is None and "gm_pipe_buffer" in stripped and "tensor.create" in stripped:
            create_line_idx = idx

    assert for_line_idx is not None, "for-loop not found in orchestration body"
    assert create_line_idx is not None, "gm_pipe_buffer tensor.create not found"
    assert for_line_idx < create_line_idx, "gm_pipe_buffer tensor.create must appear inside the for-loop body"


def test_gm_pipe_buffer_param_direction_is_out():
    """The __gm_pipe_buffer parameter must have Out direction."""

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

    group_func = transformed.get_function("group_func")
    assert group_func is not None
    gm_idx = next(i for i, p in enumerate(group_func.params) if p.name_hint == "__gm_pipe_buffer")
    assert group_func.param_directions[gm_idx] == ir.ParamDirection.Out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
