# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the MaterializeRuntimeScopes pass.

The pass inserts explicit AUTO ``RuntimeScopeStmt`` (``manual=False``) nodes
into every Orchestration function: wrapping the function body and each ForStmt
/ IfStmt branch body, while skipping insertion inside a manual scope. This is
the IR form of the orchestration codegen's former implicit ``PTO2_SCOPE()``
wrappers, expressed via ``with pl.auto_scope():`` in the DSL.

Tests follow the Before/Expected pattern: ``derive_call_directions`` runs on
both programs (the pass requires CallDirectionsResolved), and the pass runs on
Before only. ``Expected`` is hand-written with explicit ``pl.auto_scope()``.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _derive(program):
    return passes.derive_call_directions()(program)


def _materialize(program):
    return passes.materialize_runtime_scopes()(_derive(program))


def test_function_body_and_for_body_wrapped():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            for i in pl.range(4):
                out = self.kernel(a, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.auto_scope():
                for i in pl.range(4):
                    with pl.auto_scope():
                        out = self.kernel(a, out)
                return out

    after = _materialize(Before)
    ir.assert_structural_equal(after, _derive(Expected))


def test_if_branches_wrapped():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            flag: pl.Scalar[pl.INT64],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], flag: pl.Scalar[pl.INT64]):
            out: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            if flag == 0:
                out = self.kernel(a, flag, out)
            else:
                out = self.kernel(a, flag, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            flag: pl.Scalar[pl.INT64],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], flag: pl.Scalar[pl.INT64]):
            with pl.auto_scope():
                out: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                if flag == 0:
                    with pl.auto_scope():
                        out = self.kernel(a, flag, out)
                else:
                    with pl.auto_scope():
                        out = self.kernel(a, flag, out)
                return out

    after = _materialize(Before)
    ir.assert_structural_equal(after, _derive(Expected))


def test_manual_scope_suppresses_inner_for_wrap():
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.manual_scope():
                for i in pl.range(4):
                    out = self.kernel(a, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            # Function body still gets the outermost AUTO scope, but the for body
            # inside the manual scope is NOT wrapped (AUTO forbidden in MANUAL).
            with pl.auto_scope():
                with pl.manual_scope():
                    for i in pl.range(4):
                        out = self.kernel(a, out)
                return out

    after = _materialize(Before)
    ir.assert_structural_equal(after, _derive(Expected))


def test_idempotent():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            for i in pl.range(4):
                out = self.kernel(a, out)
            return out

    once = _materialize(Prog)
    twice = passes.materialize_runtime_scopes()(once)
    ir.assert_structural_equal(once, twice)


def test_user_written_auto_scope_not_double_wrapped():
    # A user-written `with pl.auto_scope():` for-body must not be wrapped again
    # (the body may arrive as a single-statement SeqStmts around the scope).
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            for i in pl.range(4):
                with pl.auto_scope():
                    out = self.kernel(a, out)
            return out

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            # Function body wrapped once; the existing for-body auto_scope is kept
            # as-is — NOT nested inside a second auto_scope.
            with pl.auto_scope():
                for i in pl.range(4):
                    with pl.auto_scope():
                        out = self.kernel(a, out)
                return out

    after = _materialize(Before)
    ir.assert_structural_equal(after, _derive(Expected))


def test_non_orchestration_function_untouched():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
            r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
            return r

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            out = self.kernel(a, out)
            return out

    after = _materialize(Prog)
    # The AIV kernel body is unchanged: no RuntimeScopeStmt anywhere in it.
    kernel = after.get_function("kernel")
    assert kernel is not None

    class _ScopeFinder(ir.IRVisitor):
        def __init__(self):
            super().__init__()
            self.found = False

        def visit_runtime_scope_stmt(self, op):
            self.found = True

    finder = _ScopeFinder()
    finder.visit_stmt(kernel.body)
    assert not finder.found, "AIV (non-Orchestration) function must not be wrapped in a RuntimeScopeStmt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
