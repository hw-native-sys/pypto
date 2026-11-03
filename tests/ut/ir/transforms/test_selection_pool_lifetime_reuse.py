# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native If selections preserve independent storage and operand alias constraints."""

from textwrap import indent

import pypto.language as pl
import pytest
from pypto import ir, passes


def _storage(program):
    result = {}

    class Collector(ir.IRVisitor):
        def visit_var(self, op):
            if isinstance(op.type, ir.TileType):
                base = op.type.memref.base_ if op.type.memref is not None else None
                assert result.setdefault(op.name_hint, base) is base

    Collector().visit_program(program)
    return result


def _materialize(program):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        before = passes.init_mem_ref()(program)
        return passes.materialize_semantic_aliases()(before)


def _placed_vec_windows(program, instruments):
    """Run ordinary PYPTO packing and inspect the resulting physical Vec ranges."""
    with passes.PassContext(instruments, memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        initialized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
        placed = passes.allocate_memory_addr()(passes.memory_reuse()(initialized))
    windows: dict[str, tuple[int, int]] = {}

    class Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            tile = op.var.type
            if isinstance(tile, ir.TileType) and tile.memory_space == ir.MemorySpace.Vec:
                assert tile.memref is not None
                offset = tile.memref.byte_offset_
                assert isinstance(offset, ir.ConstInt)
                windows[op.var.name_hint] = (offset.value, offset.value + tile.memref.size_)
            super().visit_assign_stmt(op)

        def visit_call(self, op):
            assert op.op.name != ir.get_op("tile.move").name, "branch packing must not insert payload copies"
            super().visit_call(op)

    Collector().visit_program(placed)
    return windows


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("two_tile_arm", ["then", "else"])
def test_vec_if_branch_reuse_is_pairwise_and_fits_two_buffers(
    nested, two_tile_arm, ascend_backend, pass_verification_instruments
):
    """Two coexisting allocations cannot merge transitively through the opposite arm."""
    selection = (
        """if other > 0:
    first, second = pl.yield_(a, b)
else:
    first, second = pl.yield_(b, a)"""
        if nested
        else "first = a\nsecond = b"
    )
    two_tiles = (
        "a = pl.load(x, [0, 0], [64, 256], target_memory=pl.Mem.Vec)\n"
        "b = pl.load(x, [64, 0], [64, 256], target_memory=pl.Mem.Vec)\n"
        + selection
        + "\nselected_first, selected_second = pl.yield_(first, second)"
    )
    single_tile = (
        "c = pl.load(x, [128, 0], [64, 256], target_memory=pl.Mem.Vec)\n"
        "selected_first, selected_second = pl.yield_(c, c)"
    )
    then_body, else_body = (two_tiles, single_tile) if two_tile_arm == "then" else (single_tile, two_tiles)
    program = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[192, 256], pl.FP32], flag: pl.Scalar[pl.INT32],
               other: pl.Scalar[pl.INT32],
               out: pl.Out[pl.Tensor[[128, 256], pl.FP32]]) -> pl.Tensor[[128, 256], pl.FP32]:
        if flag > 0:
{indent(then_body, "            ")}
        else:
{indent(else_body, "            ")}
        pl.store(selected_first, [0, 0], out)
        return pl.store(selected_second, [64, 0], out)
""")
    windows = _placed_vec_windows(program, pass_verification_instruments)
    a, b, c = (windows[name] for name in ("a", "b", "c"))
    assert all(end - start == 64 * 1024 for start, end in (a, b, c))
    assert a[1] <= b[0] or b[1] <= a[0], windows
    assert c in (a, b), windows
    # When C is defined first, FFD may add A to C's bucket; B must still
    # conflict with A even though B is individually exclusive with C.
    # Three 64-KiB allocations exceed Vec if treated as one conflicting family.
    # Pairwise branch packing retains two real buffers, including through the
    # nested If's descriptors, without copying either branch's payload.
    assert max(end for _, end in windows.values()) == 128 * 1024, windows


def test_vec_if_branch_alias_cannot_donate_live_external_storage(
    ascend_backend, pass_verification_instruments
):
    """A branch-local name does not make its pre-If allocation branch-private."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[48, 256], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[48, 256], pl.FP32]],
        ) -> pl.Tensor[[48, 256], pl.FP32]:
            outside = pl.load(x, [0, 0], [16, 256], target_memory=pl.Mem.Vec)
            if flag > 0:
                alias = outside
                b = pl.load(x, [16, 0], [16, 256], target_memory=pl.Mem.Vec)
                first, second = pl.yield_(alias, b)
            else:
                c = pl.load(x, [32, 0], [16, 256], target_memory=pl.Mem.Vec)
                first, second = pl.yield_(c, c)
            pl.store(outside, [0, 0], out)
            pl.store(first, [16, 0], out)
            return pl.store(second, [32, 0], out)

    windows = _placed_vec_windows(Before, pass_verification_instruments)
    outside, b, c = (windows[name] for name in ("outside", "b", "c"))
    for branch in (b, c):
        assert outside[1] <= branch[0] or branch[1] <= outside[0], windows
    assert b == c, windows


def test_vec_if_branch_escape_stays_live_across_an_independent_if(
    ascend_backend, pass_verification_instruments
):
    """Opposite arms of different Ifs can execute together and cannot share live storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[192, 256], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            other: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
        ) -> pl.Tensor[[128, 256], pl.FP32]:
            if flag > 0:
                a = pl.load(x, [0, 0], [64, 256], target_memory=pl.Mem.Vec)
                saved = pl.yield_(a)
            else:
                b = pl.load(x, [64, 0], [64, 256], target_memory=pl.Mem.Vec)
                saved = pl.yield_(b)
            if other > 0:
                c = pl.load(x, [128, 0], [64, 256], target_memory=pl.Mem.Vec)
                selected = pl.yield_(c)
            else:
                selected = pl.yield_(saved)
            pl.store(saved, [0, 0], out)
            return pl.store(selected, [64, 0], out)

    windows = _placed_vec_windows(Before, pass_verification_instruments)
    a, b, c = (windows[name] for name in ("a", "b", "c"))
    assert a == b, windows
    for source in (a, b):
        assert source[1] <= c[0] or c[1] <= source[0], windows
    assert max(end for _, end in windows.values()) == 128 * 1024, windows


def test_independently_live_old_input_is_not_coalesced(ascend_backend):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[4, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update = pl.tile.adds(a, 1.0)
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(a)
            pl.store(a, [4, 0], out)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is not storage["a"]


def test_same_arm_tail_read_keeps_old_input_independent(ascend_backend):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[4, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update = pl.tile.adds(a, 1.0)
                if flag > 1:
                    pl.store(a, [4, 0], out)
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(a)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is not storage["a"]


def test_unresolved_phi_hint_is_not_fixed_storage_proof(ascend_backend):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            b = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                picked = pl.yield_(a)
            else:
                picked = pl.yield_(b)
            if flag > 1:
                update = pl.tile.adds(picked, 1.0)
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(picked)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is not storage["a"]
    assert storage["update"] is not storage["b"]


def test_distinct_result_slots_keep_escaping_old_value(ascend_backend):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            b = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update = pl.tile.adds(a, 1.0)
                selected, saved = pl.yield_(update, a)
            else:
                selected, saved = pl.yield_(a, b)
            pl.store(saved, [4, 0], out)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is not storage["a"]


def test_inplace_unsafe_branch_producer_is_not_retargeted(ascend_backend):
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 128], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
        ) -> pl.Tensor[[1, 128], pl.FP32]:
            a = pl.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                update = pl.tile.mrgsort(a, block_len=64)
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(a)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is not storage["a"]


def test_inplace_unsafe_update_after_selected_phi_keeps_separate_storage(ascend_backend):
    """The sort output must remain separate from both selected input candidates."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 128], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 128], pl.FP32]],
        ) -> pl.Tensor[[1, 128], pl.FP32]:
            a = pl.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                update_a = pl.tile.adds(a, 1.0)
                first = pl.yield_(update_a)
            else:
                first = pl.yield_(a)
            if flag > 1:
                sorted_value = pl.tile.mrgsort(first, block_len=64)
                selected = pl.yield_(sorted_value)
            else:
                selected = pl.yield_(first)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["first"] is None
    assert storage["update_a"] is not None
    assert storage["update_a"] is not storage["a"]
    assert storage["sorted_value"] is not storage["first"]
    assert storage["sorted_value"] is not storage["a"]


def test_conditional_addc_update_keeps_carry_storage_separate(ascend_backend):
    """TADDC reads carry after writing its partial sum, even when carry is dead."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            carry = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update = pl.tile.addc(a, a, carry)
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(carry)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is not storage["carry"]


def test_addc_update_after_selected_phi_keeps_carry_candidates_separate(ascend_backend):
    """Neither possible carry input may share the addc output storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            source = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update_a = pl.tile.adds(a, 1.0)
                first = pl.yield_(update_a)
            else:
                first = pl.yield_(a)
            if flag > 1:
                update = pl.tile.addc(source, source, first)
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(first)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["first"] is None
    assert storage["update_a"] is not None
    assert storage["update_a"] is not storage["a"]
    assert storage["update"] is not storage["first"]
    assert storage["update"] is not storage["a"]


def test_loop_addc_update_keeps_carry_storage_separate(ascend_backend):
    """Loop retargeting must preserve the same operand-specific restriction."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            carry = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            for _i, (current,) in pl.range(2, init_values=(carry,)):
                update = pl.tile.addc(a, a, current)
                result = pl.yield_(update)
            return pl.store(result, [0, 0], out)

    storage = _storage(_materialize(Before))
    # The loop has a fixed destination and copies update into it afterwards.
    assert storage["update"] is not storage["result"]


def test_selected_addc_carry_hint_does_not_prove_loop_destination_disjoint(ascend_backend):
    """The non-representative carry candidate can be the loop's destination."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[12, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            other = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            carry = pl.load(x, [8, 0], [4, 8], target_memory=pl.Mem.Vec)
            for _i, (current,) in pl.range(2, init_values=(carry,)):
                if flag > 0:
                    picked = pl.yield_(other)
                else:
                    picked = pl.yield_(current)
                picked_alias = picked
                update = pl.tile.addc(a, a, picked_alias)
                result = pl.yield_(update)
            return pl.store(result, [0, 0], out)

    storage = _storage(_materialize(Before))
    # Both selected aliases remain dynamic; their else arm can read current.
    assert storage["picked"] is storage["picked_alias"] is None
    assert storage["other"] is not None
    assert storage["update"] is not storage["result"]


def test_value_escaping_through_another_selection_keeps_old_storage(ascend_backend):
    """A saved native choice observes the first update's value before a later update."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ) -> pl.Tensor[[8, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            other = pl.load(x, [4, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update_a = pl.tile.adds(a, 1.0)
                first = pl.yield_(update_a)
            else:
                first = pl.yield_(a)
            if flag > 1:
                saved = pl.yield_(first)
            else:
                saved = pl.yield_(other)
            if flag > 2:
                update_b = pl.tile.adds(first, 2.0)
                selected = pl.yield_(update_b)
            else:
                selected = pl.yield_(first)
            pl.store(saved, [4, 0], out)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["first"] is None
    assert storage["update_a"] is not None
    assert storage["update_a"] is not storage["a"]
    # saved may retain either update_a or the original a through first.
    # A later update must preserve both possible saved payloads.
    assert storage["update_b"] is not storage["first"]
    assert storage["update_b"] is not storage["a"]


def test_bare_rename_of_unresolved_phi_does_not_authorize_reuse(ascend_backend):
    """An extra handle must not turn a conditional MemRef hint into fixed storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[4, 8], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[4, 8], pl.FP32]],
        ) -> pl.Tensor[[4, 8], pl.FP32]:
            a = pl.load(x, [0, 0], [4, 8], target_memory=pl.Mem.Vec)
            if flag > 0:
                update_a = pl.tile.adds(a, 1.0)
                first = pl.yield_(update_a)
            else:
                first = pl.yield_(a)
            renamed = first
            if flag > 1:
                update_b = pl.tile.adds(renamed, 2.0)
                second = pl.yield_(update_b)
            else:
                second = pl.yield_(renamed)
            return pl.store(second, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["first"] is storage["renamed"] is None
    assert storage["update_a"] is not None
    assert storage["update_a"] is not storage["a"]
    assert storage["second"] is None
    assert storage["update_b"] is not None
    assert storage["update_b"] is not storage["first"]
    assert storage["update_b"] is not storage["a"]


def test_nested_assemble_chain_preserves_dynamic_storage_after_reuse(ascend_backend):
    """Nested descriptors and their writebacks never acquire phantom allocations."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 128], pl.INT8],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 256], pl.INT8]],
        ) -> pl.Tensor[[1, 256], pl.INT8]:
            a = pl.tile.full([1, 256], dtype=pl.INT8, value=0)
            piece = pl.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                if flag > 1:
                    update_a = pl.tile.assemble(a, piece, [0, 0])
                    inner_a = pl.yield_(update_a)
                else:
                    inner_a = pl.yield_(a)
                first = pl.yield_(inner_a)
            else:
                first = pl.yield_(a)
            if flag > 2:
                if flag > 3:
                    update_b = pl.tile.assemble(first, piece, [0, 128])
                    inner_b = pl.yield_(update_b)
                else:
                    inner_b = pl.yield_(first)
                selected = pl.yield_(inner_b)
            else:
                selected = pl.yield_(first)
            flat = pl.tile.reshape(selected, [1, 256])
            return pl.store(flat, [0, 0], out)

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        after = passes.memory_reuse()(passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before)))
    storage = _storage(after)
    assert storage["update_a"] is storage["a"]
    assert all(
        storage[name] is None for name in ("inner_a", "first", "update_b", "inner_b", "selected", "flat")
    )


def test_nested_selected_assemble_preserves_dynamic_storage_without_merging_candidates(ascend_backend):
    """A writeback to a selected tile follows its runtime descriptor without a fixed MemRef."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[3, 256], pl.INT8],
            flag: pl.Scalar[pl.INT32],
            out: pl.Out[pl.Tensor[[1, 256], pl.INT8]],
        ) -> pl.Tensor[[1, 256], pl.INT8]:
            a = pl.load(x, [0, 0], [1, 256], target_memory=pl.Mem.Vec)
            b = pl.load(x, [1, 0], [1, 256], target_memory=pl.Mem.Vec)
            c = pl.load(x, [2, 0], [1, 256], target_memory=pl.Mem.Vec)
            piece = pl.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Vec)
            if flag > 0:
                if flag > 1:
                    picked = pl.yield_(a)
                else:
                    picked = pl.yield_(b)
                update = pl.tile.assemble(picked, piece, [0, 0])
                selected = pl.yield_(update)
            else:
                selected = pl.yield_(c)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["picked"] is storage["selected"] is storage["update"] is None
    assert storage["a"] is not storage["b"]
    assert storage["a"] is not storage["c"]
    assert storage["b"] is not storage["c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
