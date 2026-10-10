# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Joint one-way planning and conservative endpoint matching."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.text_parser import parse

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIC, strict_ssa=True)
    def producer(self, out: pl.Tensor[[64, 16], pl.FP32]):
        peer: pl.Scalar[pl.INT32] = pl.system.import_peer_buffer(name="fifo", peer_func="consumer")
        pl.system.aic_initialize_pipe(peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=3)
        for i in pl.pipeline(4, stage=3, attrs={"software_pipeline_family": 1}):
            acc: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            pl.tile.tpush_to_aiv(acc, split=0)

    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def consumer(self, out: pl.Tensor[[64, 16], pl.FP32]):
        fifo: pl.Scalar[pl.INT32] = pl.system.reserve_buffer(name="fifo", size=3072, base=-1)
        pl.system.aiv_initialize_pipe(fifo, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=3)
        for i in pl.pipeline(4, stage=3, attrs={"software_pipeline_family": 1}):
            received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.tpop_from_aic(split=0)
            result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(received, 1.0)
            pl.system.tfree_to_aic(received, split=0)
            _stored: pl.Tensor[[64, 16], pl.FP32] = pl.tile.store(result, [i * 16, 0], out)

    @pl.function(type=pl.FunctionType.Group, strict_ssa=True)
    def group(self, out: pl.Tensor[[64, 16], pl.FP32]):
        self.producer(out)
        self.consumer(out)


@pl.program
class Expected:
    @pl.function(type=pl.FunctionType.AIC, strict_ssa=True)
    def producer(self, out: pl.Tensor[[64, 16], pl.FP32]):
        peer: pl.Scalar[pl.INT32] = pl.const(0, pl.INT32)
        pl.system.aic_initialize_pipe(peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=3)
        for i in pl.pipeline(4, stage=3, attrs={"software_pipeline_family": 1}):
            acc: pl.Tile[[16, 16], pl.FP32, pl.Mem.Acc] = pl.tile.create(
                [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Acc
            )
            pl.tile.tpush_to_aiv(acc, split=0)

    @pl.function(type=pl.FunctionType.AIV, strict_ssa=True)
    def consumer(self, out: pl.Tensor[[64, 16], pl.FP32]):
        fifo: pl.Scalar[pl.INT32] = pl.const(0, pl.INT32)
        pl.system.aiv_initialize_pipe(fifo, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=3)
        if pl.const(0, pl.INDEX) < pl.const(4, pl.INDEX):
            _prime0: pl.Tile[[16, 16], pl.FP32, pl.MemRef("received", slots=3)[0], pl.Mem.Vec] = (
                pl.tile.tpop_from_aic(split=0, attrs={"software_pipeline_slots": True})
            )
        if pl.const(1, pl.INDEX) < pl.const(4, pl.INDEX):
            _prime1: pl.Tile[[16, 16], pl.FP32, pl.MemRef("received", slots=3)[1], pl.Mem.Vec] = (
                pl.tile.tpop_from_aic(split=0, attrs={"software_pipeline_slots": True})
            )
        for i in pl.range(4, attrs={"software_pipeline_slots": 3}):
            slot: pl.Scalar[pl.INDEX] = i % 3
            future_slot: pl.Scalar[pl.INDEX] = (i + 2) % 3
            if i + 2 < 4:
                _future: pl.Tile[
                    [16, 16], pl.FP32, pl.MemRef("received", slots=3)[future_slot], pl.Mem.Vec
                ] = pl.tile.tpop_from_aic(split=0, attrs={"software_pipeline_slots": True})
            current: pl.Tile[[16, 16], pl.FP32, pl.MemRef("received", slots=3)[slot], pl.Mem.Vec] = (
                pl.tile.create(
                    [16, 16], dtype=pl.FP32, target_memory=pl.Mem.Vec, attrs={"software_pipeline_slots": True}
                )
            )
            result: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.adds(current, 1.0)
            _stored: pl.Tensor[[64, 16], pl.FP32] = pl.tile.store(result, [i * 16, 0], out)


def _run(program, enabled=True):
    with passes.PassContext([], enable_software_pipeline=enabled):
        after = passes.skew_cross_core_pipeline()(program)
    ir.assert_structural_equal(parse(ir.python_print(after)), after)
    return after


def _function(program, name):
    return next(f for f in program.functions.values() if f.name == name)


def test_owned_receive_is_prefetched_without_a_local_fifo_reservation():
    after = _run(Before)
    ir.assert_structural_equal(_function(after, "consumer"), _function(Expected, "consumer"))
    ir.assert_structural_equal(_function(after, "producer"), _function(Expected, "producer"))


def test_flag_off_preserves_one_way_program():
    ir.assert_structural_equal(_run(Before, enabled=False), Before)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('peer_func="consumer"', 'peer_func="unknown"'),
        ("size=3072", "size=1024"),
        ('"software_pipeline_family": 1', '"unproven_loop": 1'),
        ("slot_num=3", "slot_num=1"),
    ],
)
def test_unproven_pair_retains_existing_path(old, new):
    original = ir.python_print(Before)
    assert old in original
    before = parse(original.replace(old, new), filename="<joint-fallback>")
    ir.assert_structural_equal(_run(before), before)


@pytest.mark.parametrize("stage", [2, 3, 4])
def test_fifo_auxiliary_load_uses_the_declared_parent_stage(stage):
    source = ir.python_print(Before)
    source = (
        source.replace(
            "out: pl.Tensor[[64, 16], pl.FP32]",
            "source: pl.Tensor[[64, 16], pl.FP32], out: pl.Tensor[[64, 16], pl.FP32]",
        )
        .replace("self.producer(out)", "self.producer(source, out)")
        .replace("self.consumer(out)", "self.consumer(source, out)")
        .replace("stage=3", f"stage={stage}")
    )
    source = source.replace(
        "received: pl.Tile",
        "aux: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tile.load(source, [i * 16, 0], [16, 16])\n"
        "            received: pl.Tile",
    ).replace("pl.tile.adds(received, 1.0)", "pl.tile.add(received, aux)")
    after = _function(_run(parse(source)), "consumer")
    loads = []

    class Loads(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.load").name:
                assert isinstance(op.var.type, ir.TileType)
                loads.append(op.var.type.memref)
            super().visit_assign_stmt(op)

    Loads().visit_stmt(after.body)
    # stage-1 guarded prologue loads plus one future-load site in the loop.
    assert len(loads) == stage
    assert all(ref is not None and ref.slot_count_ == stage for ref in loads)
    assert all(ref.base_.same_as(loads[0].base_) for ref in loads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
