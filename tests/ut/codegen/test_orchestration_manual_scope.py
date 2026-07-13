# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual-scope orchestration-codegen tests."""

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from _orchestration_codegen_common import (
    _generate_orch_code,
    _run_default_pipeline_with_auto_scope_deps,
)
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


class TestManualScopeCodegen:
    """Codegen for ``with pl.manual_scope():`` and the ``deps=[var]`` submit kwarg.

    Runs under the default (roundtrip) verification: ``Submit::deps_`` is a
    first-class typed field that survives print -> parse.
    """

    def test_submit_dumps_emits_toggle_and_per_task_dump(self):
        """``pl.submit(..., dumps=[x])`` (explicit kwarg) marks one arg slot of
        one task launch.

        Demonstrates per-launch granularity the forward-sticky ``pl.dump_tag``
        cannot express: ``x`` is dumped on the first submit only, never on the
        second. Matched by Var identity, not name.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class DumpPerSubmitProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch_per_submit(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.k1, x, dumps=[x])  # dump x on task 0 only
                    b, _ = pl.submit(self.k2, x, deps=[a_tid])  # task 1 dumps nothing
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(DumpPerSubmitProgram)
        code = _generate_orch_code(transformed)

        # No orch-body toggle (simpler#953): the runtime latches the dump level
        # (off / partial / full) host-side; codegen only emits ``.dump(...)``.
        assert "enable_dump_args_selective" not in code

        # Only task 0 dumps ext_x; task 1 dumps nothing.
        assert code.count("params_t0.dump(ext_x);") == 1
        assert "params_t1.dump(" not in code

    def test_submit_runtime_out_after_materialized_comm_ctx_aliases_task_output(self):
        """Runtime Out aliases must ignore trailing materialized CommCtx args."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class SubmitRuntimeOutWithCtxProgram:
            @pl.function(type=pl.FunctionType.InCore)
            def producer(
                self,
                x: pl.Tensor[[64], pl.FP32],
                signal: pld.DistributedTensor[[1], pl.INT32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                tile = pl.load(x, [0], [64])
                return pl.store(tile, [0], out)

            @pl.function(type=pl.FunctionType.InCore)
            def consumer(self, y: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return y

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                signal: pld.DistributedTensor[[1], pl.INT32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    y, _tid = pl.submit(self.producer, x, signal)
                z = self.consumer(y)
                return z

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(SubmitRuntimeOutWithCtxProgram)
        code = _generate_orch_code(transformed)

        assert code.count("params_t0.add_scalar(signal_ctx);") == 1, code
        assert "const Tensor& y = task_0_outs.get_ref(0);" in code, code
        assert "params_t1.add_input(y);" in code, code
        assert "signal_ctx.get_ref" not in code, code

    def test_manual_scope_emits_manual_pto2_scope_and_task_id_capture(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.k1, x)
                    b, _ = pl.submit(self.k2, a, deps=[a_tid])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code
        # Every kernel submit inside a manual scope captures the submit's
        # outputs handle so downstream code can call ``.task_id()`` on it.
        # We no longer pre-emit a ``PTO2TaskId task_<n>`` variable — the
        # ``pl.submit`` producer-TaskId tuple element binds it on demand.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code
        assert "TaskOutputTensors task_1_outs = rt_submit_aiv_task(" in code
        # The ``pl.submit`` producer TaskId binds to ``task_0_outs.task_id()``
        # and is wired into the consumer via ``set_dependencies``.
        assert "task_0_outs.task_id()" in code
        assert ".set_dependencies(" in code

    def test_manual_scope_emits_explicit_user_deps(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, a_tid = pl.submit(self.k1, x)
                    b, b_tid = pl.submit(self.k2, x)
                    # Explicit deps even though `c` doesn't consume `a`/`b` data.
                    c, _ = pl.submit(self.k3, x, deps=[a_tid, b_tid])
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Each ``pl.submit`` producer TaskId binds to ``task_<n>_outs.task_id()``.
        assert "PTO2TaskId a_tid = task_0_outs.task_id();" in code
        assert "PTO2TaskId b_tid = task_1_outs.task_id();" in code
        # The consumer's deps are packed into a fixed-size stack array and
        # attached with a single ``set_dependencies(arr, count)`` call. Both
        # entries are unconditionally filled (plain TaskId bindings, not
        # iter-arg carries, so no is_valid() guard).
        assert "L0TaskArgs params_t2;" in code
        assert "PTO2TaskId params_t2_deps[2];" in code
        assert "params_t2_deps[params_t2_deps_count++] = a_tid;" in code
        assert "params_t2_deps[params_t2_deps_count++] = b_tid;" in code
        assert "params_t2.set_dependencies(params_t2_deps, params_t2_deps_count);" in code

    def test_user_written_task_dummy_lowers_to_dummy_submit(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N_BRANCHES = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                    for j in pl.parallel(N_BRANCHES):
                        _branch_out, tid = pl.submit(self.k1, x)
                        tids[j] = tid
                    barrier = pl.system.task_dummy(deps=[tids])
                    b, _consumer_tid = pl.submit(self.k2, x, deps=[barrier])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
        assert f"PTO2TaskId params_phase_fence_barrier_0_deps[{N_BRANCHES}];" in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code
        assert re.search(
            r"if \(barrier.*\.is_valid\(\)\) (params_t\d+)_deps\[\1_deps_count\+\+\] = barrier",
            code,
        ), code

    def test_user_written_task_dummy_accepts_scalar_dep_codegen(self):
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, tid = pl.submit(self.k1, x)
                    barrier = pl.system.task_dummy(deps=[tid])
                    b, _consumer_tid = pl.submit(self.k2, a, deps=[barrier])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
        assert "PTO2TaskId params_phase_fence_barrier_0_deps[1];" in code, code
        # ``tid`` is a fresh direct-producer TaskId (issue #1966): its dep-array
        # insert is emitted WITHOUT the redundant is_valid() guard.
        assert re.search(
            r"params_phase_fence_barrier_0_deps"
            r"\[params_phase_fence_barrier_0_deps_count\+\+\] = tid",
            code,
        ), code
        assert "if (tid.is_valid())" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code

    def test_user_written_empty_task_dummy_submits_unconditionally(self):
        # A user-written ``task_dummy(deps=[])`` has no producers but must still
        # materialize as a real, ready-immediately task whose id is valid and is
        # added to each consumer's fanin. It is submitted unconditionally — an
        # ``if (deps_count > 0)`` guard would statically elide it (issue #1976).
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    barrier = pl.system.task_dummy(deps=[])
                    y, _tid = pl.submit(self.k1, x, deps=[barrier])
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The dummy is submitted unconditionally and its id captured directly.
        assert (
            "TaskOutputTensors phase_fence_barrier_0_outs = "
            "rt_submit_dummy_task(params_phase_fence_barrier_0);" in code
        ), code
        assert "PTO2TaskId barrier = phase_fence_barrier_0_outs.task_id();" in code, code
        # No runtime deps-count guard for the empty-deps path — that guard is
        # what statically elided the barrier before the fix.
        assert "if (params_phase_fence_barrier_0_deps_count > 0)" not in code, code
        assert "PTO2TaskId barrier = PTO2TaskId::invalid();" not in code, code
        # The consumer still lists the (now valid) barrier in its fanin.
        assert re.search(
            r"if \(barrier.*\.is_valid\(\)\) (params_t\d+)_deps\[\1_deps_count\+\+\] = barrier",
            code,
        ), code

    def test_manual_scope_preserves_user_deps_without_compiler_deps(self):
        """Auto-deps: user deps stay intact while manual scopes are skipped."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def unrelated(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                scratch: pl.Tensor[[64], pl.FP32],
                other: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    produced, _producer_tid = pl.submit(self.fill, scratch)
                    _unused, user_tid = pl.submit(self.unrelated, other)
                    out, _ = pl.submit(self.consume, produced, deps=[user_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2TaskId params_t2_deps[1];" in code
        assert "params_t2_deps[params_t2_deps_count++] = user_tid;" in code
        assert "task_0_outs.task_id()" not in code
        assert code.count("params_t2.set_dependencies(") == 1

    def test_auto_scope_does_not_emit_task_id_capture(self):
        """Sanity: plain ``self.kernel(...)`` in auto scope stays fire-and-forget."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = self.k1(x)
                return a

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)
        assert "PTO2ScopeMode::MANUAL" not in code
        assert "TaskOutputTensors task_0_outs" not in code
        assert "set_dependencies(" not in code

    def test_pl_at_with_deps_and_as_tid_emits_submit_call(self):
        """``with pl.at(..., deps=[tid]) as tid:`` extends the dep interface
        to the ``pl.at``-block style (no explicit ``self.kernel(...)`` calls).

        Each block outlines to a first-class ``Submit`` whose return type
        carries the trailing ``Scalar[TASK_ID]``: codegen captures a
        ``TaskOutputTensors`` handle, binds the producer TaskId Var, and the
        downstream ``deps=[tid]`` flows through ``Submit::deps_`` into the
        same stack-array + ``set_dependencies`` codegen path used by
        ``pl.submit(...)``. Equivalent to writing two ``pl.submit`` calls,
        but matches the ``pl.at``-block programming style.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                # Two back-to-back pl.at blocks writing *disjoint* output
                # tensors (scratch / out) so the test pins the deps wiring
                # without tripping the pre-existing SSA rename limitation
                # for two pl.at blocks writing the same buffer.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1") as t1:
                    t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    r: pl.Tile[[64], pl.FP32] = pl.add(t, t)
                    scratch = pl.store(r, [0], scratch)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[t1]) as _t2:
                    t2t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    r2: pl.Tile[[64], pl.FP32] = pl.add(t2t, t2t)
                    out = pl.store(r2, [0], out)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The outlined ``stage1`` Call captures the TaskOutputTensors handle
        # and binds ``t1`` to the producer TaskId.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        assert "PTO2TaskId t1 = task_0_outs.task_id();" in code, code
        # ``stage2`` carries the explicit dep on ``t1`` via the stack-array
        # + set_dependencies path.
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        # ``t1`` is a fresh direct-producer TaskId (issue #1966): unguarded insert.
        assert "params_t1_deps[params_t1_deps_count++] = t1;" in code, code
        assert "if (t1.is_valid())" not in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code
        # The parser-emitted ``t1 = system.task_invalid()`` placeholder is
        # dropped by the outliner once the real TupleGetItem binding is
        # generated.
        assert "PTO2TaskId t1 = PTO2TaskId::invalid();" not in code, code

    def test_inline_pl_at_task_id_return_feeds_downstream_deps(self):
        """Regression for issue #1456: a ``TaskId`` produced inside an inline
        helper and returned to the caller must remain a valid scheduling
        dependency for a later ``pl.at(..., deps=[tid])`` in the caller.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.Inline)
            def stage1(
                self,
                x: pl.Tensor[[64], pl.FP32],
                scratch: pl.Tensor[[64], pl.FP32],
            ) -> pl.Scalar[pl.TASK_ID]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="inline_stage1") as stage1_tid:
                    v: pl.Scalar[pl.FP32] = pl.read(x, [0])
                    pl.write(scratch, [0], v)
                return stage1_tid

            @pl.function(type=pl.FunctionType.Opaque)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                scratch = pl.create_tensor([64], dtype=pl.FP32)
                dep_tid: pl.Scalar[pl.TASK_ID] = self.stage1(x, scratch)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2", deps=[dep_tid]):
                    t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                    r: pl.Tile[[64], pl.FP32] = pl.add(t, t)
                    out = pl.store(r, [0], out)

                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        assert "task_0_outs.task_id()" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code
        assert "params_t1_deps[params_t1_deps_count++]" in code, code

    def test_submit_with_deps_in_auto_scope_emits_set_dependencies(self):
        """``pl.submit(..., deps=[tid])`` works in auto scope too.

        The runtime's ``Arg::set_dependencies`` is orthogonal to OverlapMap
        auto-tracking (final fanin = auto ∪ explicit), so the codegen emits
        the task-output capture and the deps stack array without requiring a
        ``with pl.manual_scope():`` wrapper. The implicit ``PTO2_SCOPE()``
        (auto OverlapMap) stays on.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a, a_tid = pl.submit(self.k1, x)
                b, _ = pl.submit(self.k2, x, deps=[a_tid])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Stays in auto scope — no MANUAL wrapper.
        assert "PTO2ScopeMode::MANUAL" not in code, code
        # Both submits capture their TaskOutputTensors handle for downstream
        # ``.task_id()``.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        assert "PTO2TaskId a_tid = task_0_outs.task_id();" in code, code
        # k2's explicit dep on a_tid is wired through a stack deps array +
        # set_dependencies, exactly like in manual scope.
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        # ``a_tid`` is a fresh direct-producer TaskId (issue #1966): unguarded insert.
        assert "params_t1_deps[params_t1_deps_count++] = a_tid;" in code, code
        assert "if (a_tid.is_valid())" not in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_compiler_derived_deps_in_auto_scope_emit_set_dependencies_when_enabled_without_manual_scope(
        self,
    ):
        """AutoDeriveTaskDependencies may add explicit deps inside AUTO scopes.

        The scope must stay AUTO (``PTO2_SCOPE()`` / OverlapMap still enabled),
        while compiler-derived edges use the same ``set_dependencies`` codegen
        path as user-written deps when AUTO-scope analysis is explicitly enabled.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.scope(mode=pl.ScopeMode.AUTO):
                    produced = self.fill(scratch)
                    out = self.consume(produced)
                return out

        transformed = _run_default_pipeline_with_auto_scope_deps(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" not in code, code
        assert "PTO2_SCOPE() {" in code, code
        producer_tid = re.search(
            r"PTO2TaskId (\w+_tid) = PTO2TaskId::invalid\(\);[\s\S]*\1 = task_0_outs\.task_id\(\);",
            code,
        )
        assert producer_tid, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert (
            f"if ({producer_tid.group(1)}.is_valid()) "
            f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};"
        ) in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_compiler_derived_deps_in_default_auto_scope_do_not_emit_set_dependencies(self):
        """Default auto_scope=True is skipped unless AUTO-scope analysis is enabled."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced = self.fill(scratch)
                out = self.consume(produced)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" not in code, code
        assert "PTO2_SCOPE() {" in code, code
        assert "PTO2TaskId producer_tid = task_0_outs.task_id();" not in code, code
        assert "params_t1.set_dependencies(" not in code, code

    def test_compiler_derived_deps_in_default_auto_scope_emit_set_dependencies_when_enabled(self):
        """Default auto_scope=True can become MANUAL when analysis covers the body."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced = self.fill(scratch)
                out = self.consume(produced)
                return out

        transformed = _run_default_pipeline_with_auto_scope_deps(Prog)
        code = _generate_orch_code(transformed)

        assert code.count("PTO2_SCOPE(PTO2ScopeMode::MANUAL)") == 1, code
        assert "PTO2_SCOPE() {" not in code, code
        producer_tid = re.search(
            r"PTO2TaskId (\w+_tid) = PTO2TaskId::invalid\(\);[\s\S]*\1 = task_0_outs\.task_id\(\);",
            code,
        )
        assert producer_tid, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert (
            f"if ({producer_tid.group(1)}.is_valid()) "
            f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};"
        ) in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_compiler_derived_deps_for_plain_auto_call_do_not_capture_task_id_by_default(self):
        """pl.at-style ordinary calls stay fire-and-forget by default."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced = self.fill(scratch)
                out = self.consume(produced)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" not in code, code
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" not in code, code
        assert "set_dependencies(" not in code, code

    def test_compiler_derived_deps_for_plain_auto_call_capture_task_id_when_enabled(self):
        """pl.at-style ordinary calls are captured when deps need them."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def fill(
                self,
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.InCore)
            def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                produced = self.fill(scratch)
                out = self.consume(produced)
                return out

        transformed = _run_default_pipeline_with_auto_scope_deps(Prog)
        code = _generate_orch_code(transformed)

        assert code.count("PTO2_SCOPE(PTO2ScopeMode::MANUAL)") == 1, code
        assert "PTO2_SCOPE() {" not in code, code
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(
            r"PTO2TaskId (\w+_tid) = PTO2TaskId::invalid\(\);[\s\S]*\1 = task_0_outs\.task_id\(\);",
            code,
        )
        assert producer_tid, code
        assert (
            f"if ({producer_tid.group(1)}.is_valid()) "
            f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};"
        ) in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_auto_scope_task_id_array_slot_dep_uses_scalar_snapshot(self):
        """A TaskId array slot read is a valid explicit dep in auto scope."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                tids = pl.array.create(1, pl.TASK_ID)
                a, first_tid = pl.submit(self.k1, x)
                tids[0] = first_tid
                prev = tids[0]
                b, second_tid = pl.submit(self.k2, x)
                tids[0] = second_tid
                c, _ = pl.submit(self.k3, x, deps=[prev])
                return c

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "PTO2ScopeMode::MANUAL" not in code, code
        assert re.search(r"PTO2TaskId\s+prev\s*=\s*tids\[0\];", code), code
        assert "tids[0] = second_tid;" in code, code
        assert "PTO2TaskId params_t2_deps[1];" in code, code
        assert "if (prev.is_valid()) params_t2_deps[params_t2_deps_count++] = prev;" in code, code
        assert "params_t2.set_dependencies(params_t2_deps, params_t2_deps_count);" in code, code

    def test_manual_scope_seq_outer_parallel_inner_two_stage_pipeline(self):
        """End-to-end: ``with pl.manual_scope():`` wrapping
        ``for i in pl.range(M): for j in pl.parallel(N): stage1; stage2``.

        Each iteration runs a 2-stage pipeline (stage2 reads stage1's output
        on the same tile). This shapes a verifiable dependency graph where
        manual_scope must:

        1. **Establish** the stage1 → stage2 dependency *within* an iteration
           (otherwise stage2 races stage1's write on the shared scratch
           buffer); and
        2. **Avoid serializing** across iterations: different ``(i, j)`` tiles
           write disjoint regions of ``out`` and ``scratch``, so the inner
           ``pl.parallel(N)`` loop and the outer ``pl.range(M)`` loop should
           submit tasks at maximum concurrency.

        Because MANUAL mode skips the runtime OverlapMap, the only ordering
        comes from explicit ``set_dependencies`` calls. The codegen output
        therefore proves correct parallelism: present where required
        (stage2→stage1 within each iteration), absent everywhere else.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        M, N = 4, 8
        TILE_R, TILE_C = 32, 32
        ROWS, COLS = M * TILE_R, N * TILE_C

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def stage1(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def stage2(
                self,
                scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.range(M):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(N):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            scratch, stage1_tid = pl.submit(self.stage1, x, scratch, row, col)
                            out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[stage1_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        # Property-only verification: the explicit Submit dep covers the
        # intra-iteration hazard, so the manual scope can stay manual.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The manual scope remains manual; ordering comes only from the user
        # dep below, preserving cross-iteration parallelism.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        assert "for (int64_t i = 0; i < 4; i += 1)" in code, code
        assert "for (int64_t j = 0; j < 8; j += 1)" in code, code

        # The producer TaskId is preserved through windowed rewriting and is
        # threaded into the consumer dependency edge.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        assert "TaskOutputTensors task_1_outs = rt_submit_aiv_task(" in code, code

        # *** Manual dep correctly established WITHIN each iteration ***
        # stage2 reads what stage1 just wrote to ``scratch``: this dep is
        # required for correctness.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

        # *** Correct parallelism ACROSS iterations ***
        # The only ``set_dependencies`` in the manual scope is the one above;
        # there is no cross-iteration serialization. Different (i, j) tiles
        # run in parallel as MANUAL mode allows.
        assert code.count("set_dependencies(") == 1, code

    def test_manual_scope_parallel_outer_seq_inner_two_stage_pipeline(self):
        """End-to-end: outer ``pl.parallel`` + inner ``pl.range`` inside
        ``with pl.manual_scope():`` with the same 2-stage pipeline.

        Mirror of the previous case with the loop kinds swapped. The same
        manual-dep contract holds: stage1 → stage2 within an iteration is
        explicit; cross-iteration is unconstrained so the parallel outer
        loop dispatches tiles concurrently.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        M, N = 8, 4
        TILE_R, TILE_C = 32, 32
        ROWS, COLS = M * TILE_R, N * TILE_C

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def stage1(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def stage2(
                self,
                scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.parallel(M):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.range(N):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            scratch, stage1_tid = pl.submit(self.stage1, x, scratch, row, col)
                            out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[stage1_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        # Property-only verification: the explicit Submit dep covers the
        # intra-iteration hazard, so the manual scope can stay manual.
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The manual scope remains manual; ordering comes only from the user
        # dep below, preserving cross-iteration parallelism.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        assert "for (int64_t i = 0; i < 8; i += 1)" in code, code
        assert "for (int64_t j = 0; j < 4; j += 1)" in code, code

        # The producer TaskId is preserved through windowed rewriting and is
        # threaded into the consumer dependency edge.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        assert "TaskOutputTensors task_1_outs = rt_submit_aiv_task(" in code, code

        # Manual dep WITHIN each iteration: stage2 follows stage1.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

        # Cross-iteration parallel: the ONLY set_dependencies is the
        # intra-iteration one.
        assert code.count("set_dependencies(") == 1, code

    def test_manual_scope_tid_loop_carried_into_array_carry(self):
        """Regression for #1811: a ``manual_scope``-produced TaskId loop-carried
        into an ``Array[TASK_ID]`` across an enclosing ``pl.range`` loop.

        Shape: an outer ``pl.range`` carries a ``pl.array(TASK_ID)``; each
        iteration a ``with pl.manual_scope():`` wraps a ``pl.parallel`` that
        submits one task per slot and writes its TaskId back into the carry
        (``carry[n] = prod_tid``). The next iteration's consumer fences on the
        whole carry, so the carry is genuinely loop-carried.

        The ``pl.parallel`` array carry registered INSIDE the manual scope
        reuses the enclosing loop's backing array. Orchestration codegen used
        to blindly restore ``array_carry_vars_`` at manual-scope exit, wiping
        that registration; the enclosing loop's yield (emitted AFTER the block)
        then could not resolve the carry and tripped an ``INTERNAL_CHECK``
        (``scalar yield to array carry must resolve to a TaskId variable``).
        The fix preserves array carries whose backing storage is
        enclosing-scope-valid, so the per-slot write threads through one array.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N, LOOP, ROWS, COLS = 4, 2, 16, 256
        TILE = COLS // N

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def seed(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(x, [0, 0], [ROWS, COLS])
                return pl.store(t, [0, 0], y)

            @pl.function(type=pl.FunctionType.InCore)
            def consume(
                self, y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]]
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[ROWS, COLS], pl.FP32] = pl.load(y, [0, 0], [ROWS, COLS])
                r: pl.Tile[[ROWS, COLS], pl.FP32] = pl.add(t, t)
                return pl.store(r, [0, 0], y)

            @pl.function(type=pl.FunctionType.InCore)
            def prod(
                self,
                y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[ROWS, TILE], pl.FP32] = pl.load(y, [0, col], [ROWS, TILE])
                r: pl.Tile[[ROWS, TILE], pl.FP32] = pl.add(t, t)
                return pl.store(r, [0, col], y)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                y: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                carry = pl.array.create(N, pl.TASK_ID)
                y, seed_tid = pl.submit(self.seed, x, y)
                for s in pl.unroll(N):
                    carry[s] = seed_tid
                for _i in pl.range(LOOP):
                    y, _ = pl.submit(self.consume, y, deps=[carry[i] for i in range(N)])
                    with pl.manual_scope():
                        for n in pl.parallel(N):
                            col: pl.Scalar[pl.INDEX] = n * TILE
                            y, prod_tid = pl.submit(self.prod, y, col, deps=[carry[n]])
                            carry[n] = prod_tid
                return y

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Identify the loop-carried array by its seed broadcast: the seed
        # TaskId is written to every slot before the loop. Deps-buffer arrays
        # (``params_t*_deps`` / ``_submit_deps_buf``) are also ``PTO2TaskId[N]``
        # but are never seeded this way, so this name is unambiguous.
        seed_writes = re.findall(r"(\w+)\[\d+\] = seed_tid;", code)
        assert len(seed_writes) == N, code
        carry_arr = seed_writes[0]
        assert all(name == carry_arr for name in seed_writes), code
        # Exactly ONE backing declaration for the carry — the single-array
        # threading invariant is the heart of the fix. A regression that wiped
        # the carry would either crash or allocate a distinct array.
        assert len(re.findall(rf"PTO2TaskId\s+{carry_arr}\[{N}\]", code)) == 1, code
        # The manual-scope parallel loop writes each producer TaskId back into
        # its slot of the SAME backing array (the per-slot loop-carry write).
        # Scope the assertion to the MANUAL block region (everything after the
        # marker) so it cannot be satisfied by the constant-index seed writes
        # that precede the loop, and require a variable slot index (the loop
        # var) to pin the per-iteration write.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        manual_region = code[code.index("PTO2_SCOPE(PTO2ScopeMode::MANUAL)") :]
        assert re.search(rf"{carry_arr}\[[A-Za-z_]\w*\]\s*=\s*\w+;", manual_region), code
        # The carry is consumed by per-slot reads (the consumer fences on the
        # previous iteration's producers, and the parallel body reads its own
        # slot for its dep) — proof the carry is genuinely loop-carried. The
        # consumer alone contributes N reads, so require at least N.
        slot_reads = re.findall(rf"=\s*{carry_arr}\[\w+\];", code)
        assert len(slot_reads) >= N, code

    def test_manual_scope_parallel_dynamic_trip_count_rejected(self):
        """``pl.parallel(<dynamic>)`` carrying a manual_scope dep must error.

        Array-carry codegen needs a const trip count to allocate a fixed-size
        ``PTO2TaskId[N]`` fence array. With a dynamic trip count we cannot
        emit correct multi-deps lowering; silently falling back to a scalar
        ``last-dispatched`` fence would be wrong. ``ClassifyIterArgCarry`` (the
        pipeline's last pass, which sizes the array carry) surfaces this as a
        clear user-facing CHECK — before codegen ever runs.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                n_branches: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    prev_tid = None
                    for i in pl.range(4):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(n_branches):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            out, prev_tid = pl.submit(self.kern, x, out, row, col, deps=[prev_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        with pytest.raises(Exception, match="statically-known trip count"):
            _generate_orch_code(pm.run_passes(Prog))

    def test_manual_scope_double_buffered_array_carry_above_legacy_16_cap(self):
        """A stable full-array dep with ``N > 16`` lowers through a dummy barrier.

        The runtime's ``Arg::set_dependencies(ptr, count)`` primitive has no
        upper bound on explicit deps. The phase-fence compression keeps the
        N-slot dependency fanin on a synthetic dummy barrier, then makes each
        real downstream task depend on the barrier's single TaskId. The witness
        uses a double-buffered carrier so the dependency source is read-only
        inside the parallel body.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32
        ABOVE_LEGACY_CAP = 17  # past the legacy 16-dep cap that no longer exists

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    tids = pl.array.create(ABOVE_LEGACY_CAP, pl.TASK_ID)
                    for i, (tids_iter,) in pl.range(4, init_values=(tids,)):
                        tids_next = pl.array.create(ABOVE_LEGACY_CAP, pl.TASK_ID)
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(ABOVE_LEGACY_CAP):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids_iter])
                            tids_next[j] = tid
                        tids = pl.yield_(tids_next)
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)
        assert "rt_submit_dummy_task(params_phase_fence_barrier_0)" in code, code
        assert f"PTO2TaskId params_phase_fence_barrier_0_deps[{ABOVE_LEGACY_CAP}];" in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[1\];", code), code
        assert re.search(
            r"if \(phase_fence_barrier_0_tid\.is_valid\(\)\) "
            r"(params_t\d+)_deps\[\1_deps_count\+\+\] = phase_fence_barrier_0_tid;",
            code,
        ), code
        assert not re.search(rf"PTO2TaskId params_t\d+_deps\[{ABOVE_LEGACY_CAP}\];", code), code

    def test_manual_scope_phase_fence_scalar_dep_does_not_emit_dummy_barrier(self):
        """Scalar TaskId deps remain on the legacy single-edge lowering path."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    a, tid = pl.submit(self.k1, x)
                    b, _ = pl.submit(self.k2, a, deps=[tid])
                return b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task" not in code, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_manual_scope_phase_fence_mixed_deps_do_not_emit_dummy_barrier(self):
        """Mixed array + scalar deps are intentionally outside first-version scope."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32
        N_BRANCHES = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    seed_out, seed_tid = pl.submit(self.kern, x, out, 0, 0)
                    tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                    for i in pl.range(2):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(N_BRANCHES):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            seed_out, tid = pl.submit(
                                self.kern,
                                x,
                                seed_out,
                                row,
                                col,
                                deps=[tids, seed_tid],
                            )
                            tids[j] = tid
                return seed_out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task" not in code, code
        # 4 user deps (tids[0..3]) + 1 user dep (seed_tid) = 5
        assert re.search(r"PTO2TaskId params_t\d+_deps\[5\];", code), code

    def test_auto_scope_array_dep_does_not_emit_dummy_barrier(self):
        """Array deps outside manual_scope keep the existing explicit-deps lowering."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        ROWS, COLS = 128, 128
        TILE_R, TILE_C = 32, 32
        N_BRANCHES = 4

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                tids = pl.array.create(N_BRANCHES, pl.TASK_ID)
                for i in pl.range(2):
                    row: pl.Scalar[pl.INDEX] = i * TILE_R
                    for j in pl.parallel(N_BRANCHES):
                        col: pl.Scalar[pl.INDEX] = j * TILE_C
                        out, tid = pl.submit(self.kern, x, out, row, col, deps=[tids])
                        tids[j] = tid
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "rt_submit_dummy_task" not in code, code
        assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code

    def test_manual_scope_submit_task_id_dep(self):
        """The producer TaskId of a ``pl.submit(...)`` threaded into a later
        submit's ``deps=[...]`` reaches the dependency edge.

        Pattern:
            scratch, tid = pl.submit(self.stage1, x, scratch, row, col)
            out, _       = pl.submit(self.stage2, scratch, out, row, col, deps=[tid])

        Expected codegen for the dep chain:
            L0TaskArgs params_t1;
            PTO2TaskId params_t1_deps[1];
            uint32_t params_t1_deps_count = 0;
            params_t1_deps[params_t1_deps_count++] = <producer TaskId>;
            params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        M, N = 4, 4
        TILE_R, TILE_C = 32, 32
        ROWS, COLS = M * TILE_R, N * TILE_C

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def stage1(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(x, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], scratch)
                return ret

            @pl.function(type=pl.FunctionType.InCore)
            def stage2(
                self,
                scratch: pl.Tensor[[ROWS, COLS], pl.FP32],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                row: pl.Scalar[pl.INDEX],
                col: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                t: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.load(scratch, [row, col], [TILE_R, TILE_C])
                r: pl.Tile[[TILE_R, TILE_C], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(r, [row, col], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[ROWS, COLS], pl.FP32],
                scratch: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
                out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
            ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
                with pl.manual_scope():
                    for i in pl.range(M):
                        row: pl.Scalar[pl.INDEX] = i * TILE_R
                        for j in pl.parallel(N):
                            col: pl.Scalar[pl.INDEX] = j * TILE_C
                            scratch, tid = pl.submit(self.stage1, x, scratch, row, col)
                            out, _ = pl.submit(self.stage2, scratch, out, row, col, deps=[tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The producer TaskId is attached to the consumer dependency edge.
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        # The dep edge is filled into the consumer's stack deps array and
        # attached with a single ``set_dependencies`` call.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_manual_scope_submit_iter_arg_taskid_carry(self):
        """A ``pl.submit`` producer TaskId threaded through a ``pl.range``
        iter_arg seeds the next iteration's ``deps=[...]``.

        Pattern (per-branch linear chain):
            prev_tid = None
            for step in pl.range(N):
                out, prev_tid = pl.submit(self.kern, ..., deps=[prev_tid])

        ``prev_tid`` starts as the ``None`` sentinel (``PTO2TaskId::invalid()``)
        and is rebound each iteration to the submit's producer TaskId. Because
        it is a loop iter_arg, the dep fill is guarded by ``is_valid()`` so the
        first iteration contributes no edge.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        N_BRANCHES, N_STEPS = 4, 4
        TILE_M, BIG_N = 32, 32
        BIG_M = N_BRANCHES * N_STEPS * TILE_M

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
                row: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
            ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
                t: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.load(data, [row, 0], [TILE_M, BIG_N])
                r: pl.Tile[[TILE_M, BIG_N], pl.FP32] = pl.add(t, t)
                ret: pl.Tensor[[BIG_M, BIG_N], pl.FP32] = pl.store(r, [row, 0], out)
                return ret

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                data: pl.Tensor[[BIG_M, BIG_N], pl.FP32],
                out: pl.Out[pl.Tensor[[BIG_M, BIG_N], pl.FP32]],
            ) -> pl.Tensor[[BIG_M, BIG_N], pl.FP32]:
                with pl.manual_scope():
                    for branch in pl.parallel(N_BRANCHES):
                        prev_tid = None
                        for step in pl.range(N_STEPS):
                            row: pl.Scalar[pl.INDEX] = (step * N_BRANCHES + branch) * TILE_M
                            out, prev_tid = pl.submit(self.kern, data, row, out, deps=[prev_tid])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # The ``None`` seed lowers to an invalid PTO2TaskId sentinel.
        assert "PTO2TaskId::invalid()" in code, code
        # The iter-arg TaskId carry is filled into the deps array under an
        # ``is_valid()`` guard (first iteration's sentinel is skipped).
        assert ".is_valid())" in code, code
        assert ".set_dependencies(" in code, code

    def test_manual_scope_inner_deps_can_reference_outer_scope_tid(self):
        """Regression: a kernel inside ``with pl.manual_scope():`` can carry
        ``deps=[outer_tid]`` referencing a TaskId produced by an *outer*
        (auto-scope) task.

        Previously ``OrchestrationCodegen::VisitStmt_(RuntimeScopeStmtPtr)``
        wiped ``manual_task_id_map_`` on manual-scope entry (``std::move``
        the outer map into ``saved_map`` then ``clear()``), so any
        ``deps=`` edge pointing to an outer-scope producer was silently
        dropped in ``CountManualDeps`` (early return on the
        ``manual_task_id_map_.find(edge) == end()`` miss) and no
        ``set_dependencies(...)`` call was emitted for that inner task.
        With no explicit edge and no auto-dep (manual scope disables
        OverlapMap), the inner kernel could race the outer producer.

        The fix snapshots the outer map by *copy* instead of moving + clearing,
        so the live map keeps the outer entries visible inside the inner scope.
        The outer-only snapshot is restored on exit so the inner-scope adds —
        which name C++ identifiers that die with the inner ``{`` block — do
        not leak back into the parent scope.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k_outer(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.InCore)
            def k_inner(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                # Outer producer in auto scope — captures TaskId for the
                # cross-scope fence. ``outer_a`` is bound but unused; the
                # only edge from outer to inner is the explicit TaskId dep.
                outer_a, outer_tid = pl.submit(self.k_outer, x)
                with pl.manual_scope():
                    # Inner consumer in manual scope. The two kernels share
                    # no data input (both read ``x``), so the ``deps=`` edge
                    # is the *only* thing fencing inner behind outer.
                    inner_b, _ = pl.submit(self.k_inner, x, deps=[outer_tid])
                return inner_b

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        # Outer producer (Task 0, in auto scope) captures its TaskId. Use the
        # same regex-discovery idiom as nearby tests (lines 3730, 3819, 4010)
        # so the assertion is not brittle to any future SSA renaming /
        # suffixing the codegen may apply to the producer TaskId emit name.
        assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
        producer_tid = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert producer_tid, code
        # Inner kernel runs inside a MANUAL scope.
        assert "PTO2_SCOPE(PTO2ScopeMode::MANUAL)" in code, code
        # The inner kernel's deps array MUST include the producer TaskId edge —
        # this is what the bug used to silently drop. Without these lines the
        # inner ``rt_submit_*_task`` would have no explicit fence on the outer
        # task and the regression would re-emerge.
        assert "L0TaskArgs params_t1;" in code, code
        assert "PTO2TaskId params_t1_deps[1];" in code, code
        # ``outer_tid`` is a fresh direct-producer TaskId declared in the outer
        # C++ scope (issue #1966): its cross-scope dep insert is unguarded.
        assert f"params_t1_deps[params_t1_deps_count++] = {producer_tid.group(1)};" in code, code
        assert f"if ({producer_tid.group(1)}.is_valid())" not in code, code
        assert "params_t1.set_dependencies(params_t1_deps, params_t1_deps_count);" in code, code

    def test_submit_dumps_emits_per_task_dump(self):
        """``pl.submit(..., dumps=[x])`` feeds the same dump_vars path as a
        ``pl.dump_tag`` declaration: codegen emits the selective-dump toggle
        and a per-task ``.dump(...)`` for the listed arg. Confirms the existing
        codegen path consumes a Submit's dump_vars from the kwarg surface.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    out, _ = pl.submit(self.k, x, dumps=[x])
                return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        transformed = pm.run_passes(Prog)
        code = _generate_orch_code(transformed)

        assert "enable_dump_args_selective" not in code, code
        dump_lines = [ln for ln in code.split("\n") if ".dump(" in ln]
        assert dump_lines, code
        assert any("ext_x" in ln for ln in dump_lines), code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
