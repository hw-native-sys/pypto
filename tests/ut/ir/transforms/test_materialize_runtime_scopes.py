# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the MaterializeRuntimeScopes pass and the auto_scope opt-out.

In the default ``auto_scope=True`` mode the compiler inserts AUTO
``RuntimeScopeStmt`` nodes (function body + each ForStmt / IfStmt branch body)
and marks the function ``auto_scope=False`` (scopes materialized). Under
``@pl.function(auto_scope=False)`` the user places scopes by hand with
``with pl.scope()`` and the pass inserts nothing.

Auto-mode tests compare the pass output against an ``auto_scope=False`` Expected
that spells the same scopes explicitly (the pass output ends up auto_scope=False
too). ``derive_call_directions`` runs on both (the pass requires it).
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


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


def _count_runtime_scopes(func) -> int:
    """Count RuntimeScopeStmt nodes anywhere in a function body."""

    def walk(stmt) -> int:
        if stmt is None:
            return 0
        count = 1 if isinstance(stmt, ir.RuntimeScopeStmt) else 0
        if isinstance(stmt, ir.SeqStmts):
            count += sum(walk(s) for s in stmt.stmts)
        elif isinstance(stmt, (ir.RuntimeScopeStmt, ir.ForStmt, ir.WhileStmt)):
            count += walk(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            count += walk(stmt.then_body) + walk(stmt.else_body)
        return count

    return walk(func.body)


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

        # The pass marks the function auto_scope=False once scopes are materialized.
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.scope():
                for i in pl.range(4):
                    with pl.scope():
                        out = self.kernel(a, out)
                return out

    ir.assert_structural_equal(_materialize(Before), _derive(Expected))


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

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], flag: pl.Scalar[pl.INT64]):
            with pl.scope():
                out: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
                if flag == 0:
                    with pl.scope():
                        out = self.kernel(a, flag, out)
                else:
                    with pl.scope():
                        out = self.kernel(a, flag, out)
                return out

    ir.assert_structural_equal(_materialize(Before), _derive(Expected))


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

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            # Function body wrapped in an AUTO scope; the for body inside the
            # manual scope is NOT wrapped (AUTO forbidden in MANUAL).
            with pl.scope():
                with pl.scope(mode=pl.ScopeMode.MANUAL):
                    for i in pl.range(4):
                        out = self.kernel(a, out)
                return out

    ir.assert_structural_equal(_materialize(Before), _derive(Expected))


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


def test_opt_out_inserts_no_scopes():
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

        # Opt out and write no scope: relies on the runtime's implicit top scope.
        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            out = self.kernel(a, out)
            return out

    after = _materialize(Prog)
    # The pass is a no-op: no scope inserted, structurally equal to pre-pass.
    ir.assert_structural_equal(after, _derive(Prog))
    assert _count_runtime_scopes(after.get_function("orch")) == 0


def test_opt_out_user_scope_preserved_not_double_wrapped():
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

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            with pl.scope():
                out = self.kernel(a, out)
            return out

    after = _materialize(Prog)
    # Exactly one user scope, untouched (pass inserts nothing in opt-out mode).
    ir.assert_structural_equal(after, _derive(Prog))
    assert _count_runtime_scopes(after.get_function("orch")) == 1


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
    kernel = after.get_function("kernel")
    assert kernel is not None
    assert _count_runtime_scopes(kernel) == 0, "AIV (non-Orchestration) function must not be scope-wrapped"


def test_opt_out_scopes_survive_full_pipeline():
    # User-placed scopes appear at parse time and flow through the whole Default
    # pipeline. Run it with ROUNDTRIP verification (print→parse→structural-equal
    # after every pass) to guard against any pass mishandling scope stmts.
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

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
            # A per-iteration scope that even wraps the loop-carried pl.yield_:
            # RuntimeScopeStmt is transparent to SSA (ConvertToSSA / SSAVerify see
            # the carry-yield through the scope), so this is supported.
            with pl.scope():
                for i, (acc,) in pl.range(4, init_values=(out,)):
                    with pl.scope():
                        nxt: pl.Tensor[[16, 16], pl.FP32] = self.kernel(a, acc)
                        acc = pl.yield_(nxt)
                return acc

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    # Real print→parse→structural-equal check after every pass.
    with passes.PassContext([ir.make_roundtrip_instrument()]):
        out = pm.run_passes(Prog)
    orch = out.get_function("orch")
    assert orch is not None
    # Both user scopes (outer + per-iter) survive the full pipeline.
    assert _count_runtime_scopes(orch) == 2


def test_hand_placed_auto_scope_rejected_in_default_mode():
    # In the default auto_scope=True mode the compiler owns AUTO placement, so a
    # hand-placed `with pl.scope()` is rejected (set auto_scope=False instead).
    with pytest.raises(Exception):  # noqa: B017 — parser raises ParserSyntaxError

        @pl.program
        class _Prog:
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
                with pl.scope():
                    out = self.kernel(a, out)
                return out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
