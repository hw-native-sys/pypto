# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor use validation across explicit AUTO runtime scopes."""

import re

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    _generate_orch_code,
    _generate_orch_full_pipeline,
    _out_of_scope_tensor_refs,
)
from pypto import backend, passes
from pypto.backend import BackendType


def test_nested_scope_tensor_carry_snapshot_survives_scope_exit():
    """An escaping carry copy remains a snapshot, with its declaration outside the scope."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def consume(
            self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.load(x, [0], [16]), [0], out)

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, x: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32]):
            acc = pl.create_tensor([16], dtype=pl.FP32)
            with pl.scope():
                for i, (carry,) in pl.range(2, init_values=(acc,)):
                    with pl.manual_scope():
                        snap = carry
                    fresh = pl.create_tensor([16], dtype=pl.FP32)
                    next_acc = self.consume(x, fresh)
                    out = self.consume(snap, out)
                    result = pl.yield_(next_acc)
                acc = result
            return out

    code = _generate_orch_full_pipeline(Program)
    assert not _out_of_scope_tensor_refs(code), code
    assert re.search(r"Tensor\s+snap\s*=\s*\w+;", code), code
    assert "add_input(snap)" in code, code


@pytest.mark.parametrize("inout", [False, True])
@pytest.mark.parametrize("local_buffer", [False, True])
@pytest.mark.parametrize("use_view", [False, True])
def test_caller_allocated_output_scope(local_buffer, use_view, inout):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def fill(self, out: pl.Out[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.tile.full([16], pl.FP32, 1.0), [0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def update(self, out: pl.InOut[pl.Tensor[[16], pl.FP32]]) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.tile.add(pl.load(out, [0], [16]), 1.0), [0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def consume(
            self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            return pl.store(pl.load(x, [0], [16]), [0], out)

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self, a: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32]
        ) -> pl.Tensor[[16], pl.FP32]:
            with pl.scope():
                if local_buffer:
                    scratch = pl.create_tensor([16], dtype=pl.FP32)
                else:
                    scratch = a
                result = self.fill(scratch)
                if inout:
                    result = self.update(result)
            with pl.scope():
                if use_view:
                    view = pl.reshape(result, [4, 4])
                    out = self.consume(pl.reshape(view, [16]), out)
                else:
                    out = self.consume(result, out)
            return out

    if local_buffer:
        with pytest.raises(ValueError, match="used after its AUTO runtime scope has closed"):
            _generate_orch_full_pipeline(Program)
    else:
        code = _generate_orch_full_pipeline(Program)
        assert not _out_of_scope_tensor_refs(code), code


def test_loop_yield_cannot_read_closed_scope_tensor():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            for i in pl.range(2):
                n = pl.min(i + 1, 2)
                with pl.scope():
                    scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                    for bi in pl.spmd(n):
                        scratch[0:16, 0:16] = pl.add(a, 1.0)
                    a = scratch
            return a

    with pytest.raises(ValueError, match="used after its AUTO runtime scope has closed"):
        _generate_orch_full_pipeline(Program)


def test_sibling_scopes_can_reuse_tensor_source_name():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.scope():
                scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                for bi in pl.spmd(1):
                    scratch[0:16, 0:16] = pl.add(a, 1.0)
                for bj in pl.spmd(1):
                    a[0:16, 0:16] = pl.add(scratch, 1.0)
            with pl.scope():
                scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                for bk in pl.spmd(1):
                    scratch[0:16, 0:16] = pl.add(a, 1.0)
                for bl in pl.spmd(1):
                    a[0:16, 0:16] = pl.add(scratch, 1.0)
            return a

    code = _generate_orch_full_pipeline(Program)
    assert code.count("alloc_tensors(") == 2, code
    assert not _out_of_scope_tensor_refs(code), code


def test_nested_auto_carry_cannot_escape_outer_allocation():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self, a: pl.Tensor[[16, 16], pl.FP32], n: pl.Scalar[pl.INDEX]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.scope():
                scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                with pl.scope():
                    for layer in pl.range(2):
                        for bi in pl.spmd(n):
                            scratch = pl.assemble(scratch, pl.add(a, 1.0), [0, 0])
            with pl.scope():
                for bi in pl.spmd(1):
                    a = pl.assemble(a, pl.add(scratch, 1.0), [0, 0])
            return a

    with pytest.raises(ValueError, match="used after its AUTO runtime scope has closed"):
        _generate_orch_full_pipeline(Program)


@pytest.mark.parametrize("use_view", [False, True])
@pytest.mark.parametrize("use_loop", [False, True])
def test_nested_auto_carry_cannot_hide_inner_allocation(use_view, use_loop):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self, a: pl.Tensor[[16, 16], pl.FP32], n: pl.Scalar[pl.INDEX]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.scope():
                with pl.scope():
                    scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                    if use_view:
                        scratch = pl.reshape(pl.reshape(scratch, [256]), [16, 16])
                    if use_loop:
                        for layer in pl.range(n):
                            a = scratch
                    elif n > 0:
                        a = scratch
            with pl.scope():
                for bi in pl.spmd(1):
                    a = pl.assemble(a, pl.add(a, 1.0), [0, 0])
            return a

    with pytest.raises(ValueError, match="allocation from a closed runtime scope"):
        _generate_orch_full_pipeline(Program)


@pytest.mark.parametrize("use_loop", [False, True])
def test_single_auto_carry_cannot_hide_local_allocation(use_loop):
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self, a: pl.Tensor[[16, 16], pl.FP32], n: pl.Scalar[pl.INDEX]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            with pl.scope():
                scratch = pl.create_tensor([16, 16], dtype=pl.FP32)
                if use_loop:
                    for layer in pl.range(n):
                        a = scratch
                elif n > 0:
                    a = scratch
            with pl.scope():
                for bi in pl.spmd(1):
                    a = pl.assemble(a, pl.add(a, 1.0), [0, 0])
            return a

    with pytest.raises(ValueError, match="allocation from a closed runtime scope"):
        _generate_orch_full_pipeline(Program)


@pytest.mark.parametrize("trip_count", [0, 1, 2, pytest.param(None, id="dynamic")])
@pytest.mark.parametrize("consume_fresh", [False, True])
@pytest.mark.parametrize("reverse_yield", [False, True])
@pytest.mark.parametrize("full_pipeline", [False, True], ids=["direct", "default"])
def test_carry_snapshot_lifetime_uses_value_at_capture(
    trip_count, consume_fresh, reverse_yield, full_pipeline
):
    """A snapshot keeps the incoming carry, including simultaneous yield ordering."""
    carry_names = "prior, carry" if reverse_yield else "carry, prior"
    yield_stmt = (
        "old, result = pl.yield_(snapshot, fresh)"
        if reverse_yield
        else ("result, old = pl.yield_(fresh, snapshot)")
    )
    consumed = "result" if consume_fresh else "old"
    upper_bound = "n" if trip_count is None else str(trip_count)
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def consume(
        self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
    ) -> pl.Tensor[[16], pl.FP32]:
        return pl.tile.store(pl.tile.load(x, [0], [16]), [0], out)
    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self, a: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32], n: pl.Scalar[pl.INDEX]):
        acc = pl.create_tensor([16], dtype=pl.FP32)
        acc = self.consume(a, acc)
        with pl.scope():
            for i, ({carry_names}) in pl.range({upper_bound}, init_values=(acc, acc)):
                snapshot = carry
                fresh = pl.create_tensor([16], dtype=pl.FP32)
                {yield_stmt}
        out = self.consume({consumed}, out)
        return out
""")

    if full_pipeline:
        generate_code = _generate_orch_full_pipeline
    else:
        # Keep the original loop for carry tracking; Default also exercises
        # the ordinary snapshot alias left after folding a one-iteration loop.
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910)
        program = passes.convert_to_ssa()(program)
        generate_code = _generate_orch_code
    if trip_count is None or trip_count > 1 or (trip_count == 1 and consume_fresh):
        with pytest.raises(
            ValueError,
            match="allocation from a closed runtime scope|used after its AUTO runtime scope has closed",
        ):
            generate_code(program)
    else:
        code = generate_code(program)
        if not full_pipeline:
            assert "for (" in code, code
        assert not _out_of_scope_tensor_refs(code), code


@pytest.mark.parametrize("capture_fresh", [False, True])
def test_branch_carry_snapshot_does_not_follow_later_rebinding(capture_fresh):
    """Merge each branch's captured storage without borrowing later carry sources."""
    snapshot_source = "fresh" if capture_fresh else "carry"
    program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIV)
    def consume(
        self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
    ) -> pl.Tensor[[16], pl.FP32]:
        return pl.tile.store(pl.tile.load(x, [0], [16]), [0], out)
    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(
        self, a: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32], active: pl.Scalar[pl.INDEX]
    ):
        acc = pl.create_tensor([16], dtype=pl.FP32)
        acc = self.consume(a, acc)
        with pl.scope():
            if active > 0:
                for i, (carry, prior) in pl.range(1, init_values=(acc, acc)):
                    fresh = pl.create_tensor([16], dtype=pl.FP32)
                    snapshot = {snapshot_source}
                    result, old = pl.yield_(fresh, snapshot)
            else:
                old = acc
        out = self.consume(old, out)
        return out
""")

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    program = passes.convert_to_ssa()(program)
    if capture_fresh:
        with pytest.raises(ValueError, match="allocation from a closed runtime scope"):
            _generate_orch_code(program)
    else:
        code = _generate_orch_code(program)
        assert "if (" in code, code
        assert not _out_of_scope_tensor_refs(code), code


@pytest.mark.parametrize("trip_count", [1, 2])
def test_loop_carry_read_rejects_allocation_closed_by_previous_iteration(trip_count):
    """A later backedge must invalidate a carry read already visited by codegen."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def consume(
            self, x: pl.Tensor[[16], pl.FP32], out: pl.Out[pl.Tensor[[16], pl.FP32]]
        ) -> pl.Tensor[[16], pl.FP32]:
            return pl.tile.store(pl.tile.load(x, [0], [16]), [0], out)

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, a: pl.Tensor[[16], pl.FP32], out: pl.Tensor[[16], pl.FP32]):
            acc = pl.create_tensor([16], dtype=pl.FP32)
            acc = self.consume(a, acc)
            with pl.scope():
                for i, (carry,) in pl.range(trip_count, init_values=(acc,)):
                    with pl.scope():
                        out = self.consume(carry, out)
                        fresh = pl.create_tensor([16], dtype=pl.FP32)
                        _result = pl.yield_(fresh)
            return out

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    program = passes.convert_to_ssa()(Program)
    if trip_count > 1:
        with pytest.raises(ValueError, match="allocation from a closed runtime scope"):
            _generate_orch_code(program)
    else:
        code = _generate_orch_code(program)
        assert "for (" in code, code
        assert not _out_of_scope_tensor_refs(code), code
