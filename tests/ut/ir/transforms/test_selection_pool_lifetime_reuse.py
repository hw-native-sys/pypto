# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Native If selections preserve independent storage and operand alias constraints."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


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
    assert storage["update_a"] is storage["first"]
    assert storage["first"] is not storage["a"]
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
    assert storage["update_a"] is storage["first"]
    assert storage["first"] is not storage["a"]
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
    # picked's representative hint names other, but its else arm reads current.
    assert storage["picked"] is storage["picked_alias"] is storage["other"]
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
    assert storage["update_a"] is storage["first"]
    assert storage["first"] is not storage["a"]
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


@pytest.mark.parametrize("width", [64, 256])
@pytest.mark.parametrize("keep_original", [False, True])
@pytest.mark.parametrize("valid_rows", [64, 63, "dynamic"])
def test_optional_updates_materialize_only_to_resolve_capacity(
    width, keep_original, valid_rows, ascend_backend, pass_verification_instruments
):
    """Three 64-KiB updates need two buffers, unless the original stays live.

    Reciprocal forbids output/input aliasing: the capacity repair must insert
    real copies, rather than restoring optional in-place update coalescing.
    Alternating arms covers a forwarded value on either side of the If. The
    physical height stays 64 while valid rows may be 63 or one shared runtime
    SSA value: partial validity must not disable the capacity repair.
    """
    columns = width
    dynamic_valid = valid_rows == "dynamic"
    partial_valid = valid_rows == 63

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[64, columns], pl.FP32],
            flag0: pl.Scalar[pl.INT32],
            flag1: pl.Scalar[pl.INT32],
            flag2: pl.Scalar[pl.INT32],
            runtime_rows: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[128, columns], pl.FP32]],
        ) -> pl.Tensor[[128, columns], pl.FP32]:
            selected_rows = runtime_rows
            if not dynamic_valid:
                selected_rows = pl.const(64, pl.INDEX)
            if partial_valid:
                selected_rows = pl.const(63, pl.INDEX)
            initial = pl.tile.load(
                x, [0, 0], [64, columns], [selected_rows, columns], target_memory=pl.Mem.Vec
            )
            if flag0 > 0:
                update0 = pl.tile.recip(initial)
                first = pl.yield_(update0)
            else:
                first = pl.yield_(initial)
            if flag1 > 0:
                second = pl.yield_(first)
            else:
                update1 = pl.tile.recip(first)
                second = pl.yield_(update1)
            if flag2 > 0:
                update2 = pl.tile.recip(second)
                third = pl.yield_(update2)
            else:
                third = pl.yield_(second)
            if keep_original:
                pl.store(initial, [64, 0], out)
            return pl.store(third, [0, 0], out)

    with passes.PassContext(
        pass_verification_instruments, memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False
    ):
        if width == 256 and keep_original:
            # Two update buffers plus a retained original genuinely exceed UB.
            with pytest.raises(ValueError, match="Vec buffer usage.*exceeds platform limit"):
                PassManager(OptimizationStrategy.Default).run_passes(Before)
            return
        after = PassManager(OptimizationStrategy.Default).run_passes(Before)

    text = after.as_python()
    moves = []
    initial_types = []

    class MoveCollector(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if op.var.name_hint.startswith("initial__") and isinstance(op.var.type, ir.TileType):
                initial_types.append(op.var.type)
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.move").name:
                moves.append(op)
            super().visit_assign_stmt(op)

    MoveCollector().visit_program(after)
    assert len(initial_types) == 1
    assert isinstance(initial_types[0].shape[0], ir.ConstInt)
    assert initial_types[0].shape[0].value == 64
    initial_valid_rows = initial_types[0].get_effective_tile_view().valid_shape[0]
    if dynamic_valid:
        assert isinstance(initial_valid_rows, ir.Var)
    else:
        assert isinstance(initial_valid_rows, ir.ConstInt)
        assert initial_valid_rows.value == valid_rows
    if width == 64:
        # The native plan fits; preserve its zero-copy selection behavior.
        assert "vec_selection_copy_" not in text
        return
    assert len(moves) == 3
    for move in moves:
        source_valid = move.value.args[0].type.get_effective_tile_view().valid_shape
        target_valid = move.var.type.get_effective_tile_view().valid_shape
        assert ir.structural_equal(source_valid[0], target_valid[0])
        if dynamic_valid:
            assert isinstance(source_valid[0], ir.Var)
        else:
            assert isinstance(source_valid[0], ir.ConstInt)
            assert source_valid[0].value == valid_rows
    storage = _storage(after)
    bases = [
        next(base for name, base in storage.items() if name.startswith(prefix + "__"))
        for prefix in ("initial", "update0", "update1", "update2")
    ]
    assert all(left is not right for left, right in zip(bases, bases[1:]))
    assert len(set(bases)) == 2


def test_capacity_materialization_preserves_shared_sibling_result(ascend_backend):
    """A mutually exclusive shared allocation is not private to its first arm.

    The small If returns (b, a) or (a, c), with b/c already sharing a legal
    compiler-owned allocation. Copying a into b's allocation on the else path
    would destroy c, the second result. A separate large update chain forces
    the capacity retry to run and fit, so declining the unsafe copy is tested.
    """

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            small: pl.Tensor[[16, 256], pl.FP32],
            large: pl.Tensor[[64, 256], pl.FP32],
            flag: pl.Scalar[pl.INT32],
            small_out: pl.Out[pl.Tensor[[32, 256], pl.FP32]],
            large_out: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
        ) -> pl.Tensor[[64, 256], pl.FP32]:
            a = pl.load(small, [0, 0], [16, 256], target_memory=pl.Mem.Vec)
            if flag > 0:
                b = pl.tile.adds(a, 1.0)
                small_first, small_second = pl.yield_(b, a)
            else:
                c = pl.tile.adds(a, 2.0)
                small_first, small_second = pl.yield_(a, c)
            initial = pl.load(large, [0, 0], [64, 256], target_memory=pl.Mem.Vec)
            if flag > 1:
                updated0 = pl.tile.recip(initial)
                first = pl.yield_(updated0)
            else:
                first = pl.yield_(initial)
            if flag > 2:
                updated1 = pl.tile.recip(first)
                second = pl.yield_(updated1)
            else:
                second = pl.yield_(first)
            if flag > 3:
                updated2 = pl.tile.recip(second)
                third = pl.yield_(updated2)
            else:
                third = pl.yield_(second)
            pl.store(small_first, [0, 0], small_out)
            pl.store(small_second, [16, 0], small_out)
            return pl.store(third, [0, 0], large_out)

    definitions = {}

    class DefinitionCollector(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if op.var.name_hint in {"b", "c"}:
                definitions[op.var.name_hint] = op.var
            super().visit_assign_stmt(op)

    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PYPTO, enable_buffer_ir=False):
        initialized = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before))
        DefinitionCollector().visit_program(initialized)
        b, c = definitions["b"], definitions["c"]
        assert isinstance(b.type, ir.TileType) and b.type.memref is not None
        assert isinstance(c.type, ir.TileType) and c.type.memref is not None
        # Model legal earlier compiler placement, without an author-pinned
        # declaration: b and c execute on opposite paths of the same If.
        shared_c = ir.Var(
            c.name_hint,
            ir.TileType(c.type.shape, c.type.dtype, b.type.memref, c.type.tile_view, c.type.memory_space),
            c.span,
        )

        class ShareBranchAllocation(ir.IRMutator):
            def visit_var(self, op):
                if op is c:
                    return shared_c
                return super().visit_var(op)

        shared = ShareBranchAllocation().visit_program(initialized)
        after = passes.allocate_memory_addr()(passes.memory_reuse()(shared))

    moves = []

    class MoveCollector(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.value, ir.Call) and op.value.op.name == ir.get_op("tile.move").name:
                moves.append(op)
            super().visit_assign_stmt(op)

    MoveCollector().visit_program(after)
    # All three wide selections must materialize for allocation to fit. The
    # sibling result forbids a fourth copy from the small value a into b/c.
    assert len(moves) == 3
    assert all(op.value.args[0].name_hint != "a" for op in moves)
    storage = _storage(after)
    assert storage["b"] is storage["c"]
    assert storage["a"] is not storage["b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
