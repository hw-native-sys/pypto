# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Register stores establish initialized byte intervals, not Tile snapshots."""

import pytest
from pypto import DataType, backend, ir, passes
from pypto.ir.builder import IRBuilder
from pypto.ir.op import tile

SPAN = ir.Span("register_memory.py", 7, 1, 7, 70)


@pytest.fixture(autouse=True)
def a5_backend():
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.Ascend950)
    yield
    backend.reset_for_testing()


def memory_program(
    *, writes=(), active=None, read_lanes=64, read_offset=0, loop_stop=None, view=False, branch_writes=None
):
    ib = IRBuilder()
    with ib.function("memory", span=SPAN, type=ir.FunctionType.AIV) as f:
        condition = f.param("condition", ir.ScalarType(DataType.BOOL), SPAN)
        storage = ib.let("storage", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        target = ib.let("view", tile.slice(storage, [64], [32])) if view else storage
        with ib.reg_scope(SPAN):
            value = ib.emit_reg(
                "value",
                "reg.broadcast",
                [ir.ConstFloat(1, DataType.FP32, SPAN)],
                {"lanes": 64, "dtype": DataType.FP32},
                SPAN,
            )
            mask = None
            if active is not None:
                mask = ib.emit_reg(
                    "mask",
                    "reg.active_mask",
                    [ir.ConstInt(active, DataType.INDEX, SPAN)],
                    {"lanes": 64},
                    SPAN,
                )
            if branch_writes is not None:
                with ib.if_stmt(condition, SPAN) as branch:
                    for offset in branch_writes[0]:
                        ib.emit_reg_store("then_out", target, value, offset=offset, span=SPAN)
                    branch.else_(SPAN)
                    for offset in branch_writes[1]:
                        ib.emit_reg_store("else_out", target, value, offset=offset, span=SPAN)
            elif loop_stop is None:
                for offset in writes:
                    ib.emit_reg_store("updated", target, value, offset=offset, mask=mask, span=SPAN)
            else:
                i = ib.var("i", ir.ScalarType(DataType.INDEX), SPAN)
                with ib.for_loop(i, 0, loop_stop, 1, span=SPAN):
                    ib.emit_reg_store("updated", target, value, mask=mask, span=SPAN)
            ib.emit_reg(
                "read",
                "reg.load",
                [storage, ir.ConstInt(read_offset, DataType.INDEX, SPAN)],
                {"lanes": read_lanes},
                SPAN,
            )
    return ir.Program([f.get_result()], "RegisterMemory", SPAN)


def initialize(program):
    with passes.PassContext([]):
        return passes.init_mem_ref()(program)


@pytest.mark.parametrize("dynamic_write", [False, True])
@pytest.mark.parametrize("unrelated_static_read", [False, True])
def test_dynamic_access_passes_through_without_disabling_other_roots(dynamic_write, unrelated_static_read):
    ib = IRBuilder()
    with ib.function("dynamic_memory", span=SPAN, type=ir.FunctionType.AIV) as f:
        offset = f.param("offset", ir.ScalarType(DataType.INDEX), SPAN)
        storage = ib.let("storage", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        other = ib.let("other", tile.create([128], dtype=DataType.FP32, target_memory=ir.MemorySpace.Vec))
        with ib.reg_scope(SPAN):
            if dynamic_write:
                value = ib.emit_reg(
                    "value",
                    "reg.broadcast",
                    [ir.ConstFloat(1, DataType.FP32, SPAN)],
                    {"lanes": 64, "dtype": DataType.FP32},
                    SPAN,
                )
                ib.emit_reg_store("updated", storage, value, offset=offset, span=SPAN)
            # A dynamic write may initialize this static read. This root is
            # outside static analysis; an unrelated static allocation is not.
            target = other if unrelated_static_read else storage
            read_offset = (
                ir.ConstInt(0, DataType.INDEX, SPAN) if dynamic_write or unrelated_static_read else offset
            )
            ib.emit_reg("read", "reg.load", [target, read_offset], {"lanes": 64}, SPAN)
    program = ir.Program([f.get_result()], "RegisterMemory", SPAN)
    if unrelated_static_read:
        with pytest.raises(ValueError, match="uninitialized"):
            initialize(program)
    else:
        initialize(program)


def test_fresh_tile_is_not_initialized_by_valid_shape():
    with pytest.raises(ValueError, match="uninitialized") as error:
        initialize(memory_program())
    assert "register_memory.py" in str(error.value)


@pytest.mark.parametrize("writes,lanes,offset", [((0,), 64, 0), ((0, 64), 128, 0), ((32,), 32, 48)])
def test_store_intervals_initialize_only_written_elements(writes, lanes, offset):
    initialize(memory_program(writes=writes, read_lanes=lanes, read_offset=offset))


def test_partial_store_does_not_initialize_remaining_elements():
    with pytest.raises(ValueError, match="uninitialized"):
        initialize(memory_program(writes=(0,), read_lanes=128))


@pytest.mark.parametrize(
    "active,read_lanes,valid", [(0, 1, False), (32, 32, True), (32, 64, False), (64, 64, True)]
)
def test_mask_false_lanes_are_not_initialized(active, read_lanes, valid):
    program = memory_program(writes=(0,), active=active, read_lanes=read_lanes)
    if valid:
        initialize(program)
    else:
        with pytest.raises(ValueError, match="uninitialized"):
            initialize(program)


def test_overlapping_view_updates_root_interval():
    initialize(memory_program(writes=(0,), view=True, read_lanes=64, read_offset=32))
    with pytest.raises(ValueError, match="uninitialized"):
        initialize(memory_program(writes=(0,), view=True, read_lanes=64, read_offset=0))


@pytest.mark.parametrize("stop,valid", [(0, False), (1, True)])
def test_zero_trip_loop_does_not_initialize_tile(stop, valid):
    program = memory_program(writes=(0,), loop_stop=stop)
    if valid:
        initialize(program)
    else:
        with pytest.raises(ValueError, match="uninitialized"):
            initialize(program)


@pytest.mark.parametrize(
    "branches,offset,lanes,valid",
    [
        (((0,), (0,)), 0, 64, True),
        (((0,), (32,)), 32, 32, True),
        (((0,), (32,)), 0, 64, False),
        (((0,), ()), 0, 64, False),
        (((0, 64), (0, 64)), 0, 128, True),
    ],
)
def test_exhaustive_branches_preserve_only_common_initialized_bytes(branches, offset, lanes, valid):
    # Reads after a dynamic if are safe exactly where both paths wrote bytes.
    program = memory_program(branch_writes=branches, read_offset=offset, read_lanes=lanes)
    if valid:
        initialize(program)
    else:
        with pytest.raises(ValueError, match="uninitialized"):
            initialize(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
