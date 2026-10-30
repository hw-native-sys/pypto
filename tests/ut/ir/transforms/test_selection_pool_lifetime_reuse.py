# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Conditional Vec updates reuse a proven dead sibling without merge copies."""

import pypto.language as pl
import pytest
from pypto import ir, passes


def _storage(program):
    result = {}

    class Collector(ir.IRVisitor):
        def visit_var(self, op):
            if isinstance(op.type, ir.TileType) and op.type.memref is not None:
                base = op.type.memref.base_
                assert result.setdefault(op.name_hint, base) is base

    Collector().visit_program(program)
    return result


def _materialize(program):
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS, enable_buffer_ir=False):
        before = passes.init_mem_ref()(program)
        return passes.materialize_semantic_aliases()(before)


def test_conditional_update_chain_reuses_one_proven_storage(ascend_backend):
    """Online-softmax style chains must not retain a new historical tile per If."""

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
            if flag > 1:
                update_b = pl.tile.adds(first, 2.0)
                second = pl.yield_(update_b)
            else:
                second = pl.yield_(first)
            if flag > 2:
                update_c = pl.tile.adds(second, 3.0)
                selected = pl.yield_(update_c)
            else:
                selected = pl.yield_(second)
            return pl.store(selected, [0, 0], out)

    after = _materialize(Before)
    storage = _storage(after)
    assert all(
        storage[name] is storage["a"]
        for name in ("update_a", "first", "update_b", "second", "update_c", "selected")
    )
    assert "tile.move" not in ir.python_print(after)


def test_sibling_only_observation_does_not_block_update(ascend_backend):
    """The opposite arm reads the original value only when the update does not run."""

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
                if flag < 0:
                    pl.store(a, [4, 0], out)
                selected = pl.yield_(a)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is storage["a"]


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


def test_inplace_unsafe_update_after_coalesced_phi_keeps_separate_storage(ascend_backend):
    """The first phi's original hint must not hide the sort's actual input allocation."""

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
    assert storage["update_a"] is storage["first"] is storage["a"]
    # RetargetAssign used the original first.type when checking the sort's
    # nonaliasing contract. Its current allocation has already changed to a.
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


def test_addc_update_after_coalesced_phi_checks_actual_carry_storage(ascend_backend):
    """A rewritten phi's abandoned hint must not hide its forbidden carry input."""

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
    assert storage["update_a"] is storage["first"] is storage["a"]
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
    # picked's representative hint names other, but its else arm reads current.
    assert storage["picked"] is storage["picked_alias"] is storage["other"]
    assert storage["update"] is not storage["result"]


def test_conditional_addc_may_reuse_source_with_disjoint_carry(ascend_backend):
    """The carry restriction still permits overwriting a dead ordinary source."""

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
                selected = pl.yield_(a)
            return pl.store(selected, [0, 0], out)

    storage = _storage(_materialize(Before))
    assert storage["update"] is storage["selected"] is storage["a"]
    assert storage["update"] is not storage["carry"]


def test_coalesced_value_escaping_through_another_selection_keeps_old_storage(ascend_backend):
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
    assert storage["update_a"] is storage["first"] is storage["a"]
    # saved's original MemRef hint names update_a's abandoned allocation. The
    # later update must still preserve the actual first/a bytes saved may read.
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
    assert storage["update_a"] is storage["first"] is storage["renamed"]
    assert storage["first"] is not storage["a"]
    assert storage["update_b"] is storage["second"]
    assert storage["update_b"] is not storage["first"]
    assert storage["update_b"] is not storage["a"]


def test_nested_assemble_chain_refreshes_every_phi_hint_after_reuse(ascend_backend):
    """A reused assemble root must not leave phantom allocations on later phis."""

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
    assert all(
        storage[name] is storage["a"]
        for name in ("update_a", "inner_a", "first", "update_b", "inner_b", "selected", "flat")
    )


def test_nested_selected_assemble_refreshes_hint_without_merging_candidates(ascend_backend):
    """Updating a nested chosen tile changes only hints, not either possible allocation."""

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
    assert storage["selected"] is storage["update"] is storage["a"]
    assert storage["a"] is not storage["b"]
    assert storage["a"] is not storage["c"]
    assert storage["b"] is not storage["c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
