# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Task-id / submit / spmd orchestration-codegen tests."""

import re

import pypto.language as pl
import pytest
from _orchestration_codegen_common import (
    _generate_orch_code,
    _generate_orch_from_transformed_program,
    _generate_orch_full_pipeline,
)
from pypto import backend, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _assert_dep_edge_rejected(program, edge_name: str) -> str:
    """Assert the full pipeline rejects ``program`` over its ``edge_name`` dep edge.

    A user-written ``deps=[...]`` edge whose TaskId is no longer bound — its
    producer sits in a scope that has since closed — must raise a
    ``pypto::ValueError`` naming that TaskId. Returns the diagnostic so callers
    can assert on its remaining details.
    """
    with pytest.raises(ValueError) as excinfo:
        _generate_orch_full_pipeline(program, allow_relaxed_verification=True)

    msg = str(excinfo.value)
    assert edge_name in msg, msg
    assert "cannot be honored" in msg, msg
    return msg


def test_array_slot_task_id_usable_as_submit_dep():
    """Regression for issue #1577.

    ``prev = tids[k]`` reads one slot of a ``pl.array.create(N, pl.TASK_ID)``
    into a ``Scalar[TASK_ID]``. Using ``prev`` as a ``pl.submit(..., deps=[prev])``
    source must wire the consumer's ``set_dependencies`` array from that snapshot
    local. Before the fix, orchestration codegen aborted with "manual_dep_edge
    var '...' has no producer task in current manual scope".
    """

    @pl.program
    class P:
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
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            tids = pl.array.create(1, pl.TASK_ID)
            _seed, seed_tid = pl.submit(self.k1, x)
            _a, tid = pl.submit(self.k2, x)
            tids[0] = tid
            prev = tids[0]
            b, _ = pl.submit(self.k3, x, deps=[seed_tid, prev])
            return b

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)

    # ``prev = tids[0]`` lowers to a scalar TaskId snapshot local read from
    # the array slot (the dep site references the snapshot, not a slot re-read).
    assert re.search(r"TaskId\s+prev\w*\s*=\s*tids\w*\[0\];", code), code
    # The consumer gets exactly one dependency array, filled from BOTH the direct
    # producer tid (seed_tid) and the array-slot snapshot (prev).
    assert code.count("set_dependencies(") == 1, code
    # ``seed_tid`` is a fresh direct-producer TaskId (issue #1966): its insert is
    # unguarded. ``prev`` is an array-slot snapshot that may hold the invalid
    # sentinel, so it keeps the is_valid() guard.
    assert not re.search(r"if \(seed_tid\w*\.is_valid\(\)\)", code), code
    assert re.search(r"\] = seed_tid\w*;", code), code
    assert re.search(r"if \(prev\w*\.is_valid\(\)\)", code), code


def test_direct_producer_dep_skips_is_valid_guard():
    """Regression for issue #1966.

    A dependency TaskId produced by a ``pl.submit(...)`` earlier in the same
    straight-line scope is statically always-valid — the runtime never hands it
    the ``TaskId::invalid()`` sentinel. Orchestration codegen must therefore
    emit its ``set_dependencies`` insert *without* the redundant ``is_valid()``
    guard. Loop-carried ``iter_arg`` / ``None``-seed / array-slot TaskIds still
    keep the guard (see ``test_compiler_derived_deps_for_fixed_trip_loop_fan_in_``
    ``capture_task_ids`` and ``test_array_slot_task_id_usable_as_submit_dep``).
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            _a, prod_tid = pl.submit(self.k1, x)
            b, _ = pl.submit(self.k2, x, deps=[prod_tid])
            return b

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)

    # The producer TaskId (task 0) is a fresh direct-producer local.
    producer = re.search(r"TaskId\s+(\w+)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert producer, code
    name = producer.group(1)
    # Its dependency-array insert is UNGUARDED (the #1966 optimization) ...
    assert re.search(rf"\w+_deps\[[^\]]*\] = {re.escape(name)};", code), code
    # ... with no redundant is_valid() branch emitted for it.
    assert f"if ({name}.is_valid())" not in code, code
    assert re.search(r"\.set_dependencies\(", code), code


def test_user_dep_edge_across_closed_scope_is_rejected():
    """A user ``deps=[tid]`` naming a TaskId from a closed scope is an error.

    Regression for the issue #1577 lifetime hazard *and* for the silent-drop
    that replaced it. A producer TaskId declared inside a nested AUTO
    ``pl.scope()`` names a ``TaskId`` C++ local that dies at the block's
    closing brace, so codegen cannot reference it afterwards. Emitting the
    reference produced uncompilable C++ (#1577); silently dropping the edge
    instead left the consumer unordered against its producer, which surfaces at
    runtime as a stale read.

    Codegen now rejects the program with an actionable ``pypto::ValueError``
    rather than emitting either. Compiler-derived edges keep the tolerant
    skip — see ``test_cross_scope_task_id_is_hoisted_for_set_dependencies``.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                _a, scoped_tid = pl.submit(self.k1, x)
            b, _ = pl.submit(self.k2, x, deps=[scoped_tid])
            return b

    msg = _assert_dep_edge_rejected(P, "scoped_tid")
    # The diagnostic also explains the fix.
    assert "inside the scope that produces it" in msg, msg


def test_cross_scope_task_id_is_hoisted_for_set_dependencies():
    """A compiler-derived dependency can legally cross a generated SIMPLER_SCOPE.

    The producer TaskId must be declared in the outer C++ scope, assigned inside
    the producer block, and used by the later consumer's set_dependencies call.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def k1(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                produced, scoped_tid = pl.submit(self.k1, scratch)
            with pl.scope():
                b = self.k2(produced, attrs={"compiler_manual_dep_edges": [scoped_tid]})
            return b

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        transformed = passes.derive_call_directions()(P)
        code = _generate_orch_from_transformed_program(transformed)

    assert re.search(r"TaskId\s+scoped_tid_tid\s*=\s*TaskId::invalid\(\);", code), code
    assert "scoped_tid_tid = task_0_outs.task_id();" in code
    assert "TaskId scoped_tid = task_0_outs.task_id();" not in code
    assert re.search(
        r"if \(scoped_tid_tid\.is_valid\(\)\) "
        r"params_t\d+_deps\[params_t\d+_deps_count\+\+\] = scoped_tid_tid;",
        code,
    ), code
    assert re.search(r"params_t\d+\.set_dependencies\(params_t\d+_deps, params_t\d+_deps_count\);", code), (
        code
    )


def test_cross_scope_plain_auto_call_captures_task_id_when_hoisted():
    """Plain auto-scope calls capture TaskOutputTensors when their TaskId is hoisted."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def k1(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            local = pl.create_tensor([64], dtype=pl.FP32)
            with pl.scope():
                produced = self.k1(local)
            with pl.scope():
                out = self.k2(produced)
            return out

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        transformed = passes.derive_call_directions()(P)
        transformed = passes.auto_derive_task_dependencies(analyze_auto_scopes=True)(transformed)
        code = _generate_orch_from_transformed_program(transformed)

    assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code
    assert "produced_tid = task_0_outs.task_id();" in code
    assert ".set_dependencies(" in code


def test_compiler_derived_deps_for_fire_and_forget_call_capture_task_id_when_hoisted():
    """Fire-and-forget calls must capture TaskIds when compiler deps use their output.

    Qwen copy_hidden writes an already-created tensor and does not bind the call
    result.  A later compiler-derived dependency on that tensor must still
    resolve to the producer task id instead of the hoisted invalid sentinel.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self) -> pl.Tensor[[64], pl.FP32]:
            local = pl.create_tensor([64], dtype=pl.FP32)
            with pl.scope():
                self.fill(local)
            with pl.scope():
                out = self.consume(local)
            return out

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    # This intentionally models a lowered fire-and-forget output-existing
    # producer shape. Source-level code should normally read the returned
    # post-call tensor, but codegen still has to handle this IR form because
    # earlier passes can erase the binding while compiler deps still reference
    # the written tensor.
    with passes.PassContext([], passes.VerificationLevel.NONE):
        transformed = passes.derive_call_directions()(P)
        transformed = passes.auto_derive_task_dependencies(analyze_auto_scopes=True)(transformed)
        code = _generate_orch_from_transformed_program(transformed)

    assert "TaskOutputTensors task_0_outs = rt_submit_aiv_task(" in code, code
    assert "local_tid = task_0_outs.task_id();" in code, code
    assert re.search(
        r"if \(local_tid\.is_valid\(\)\) params_t\d+_deps\[params_t\d+_deps_count\+\+\] = local_tid;",
        code,
    ), code


def test_compiler_derived_deps_for_mixed_group_call_capture_task_id_when_hoisted():
    """MixedKernels producers must capture TaskIds when compiler deps use their output."""

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIC)
        def cube(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            tile = pl.load(x, [0, 0], [16, 16])
            updated = pl.store(tile, [0, 0], out)
            return updated

        @pl.function(type=pl.FunctionType.AIV)
        def vector(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ):
            tile = pl.load(x, [0, 0], [16, 16])
            pl.store(tile, [0, 0], out)

        @pl.function(type=pl.FunctionType.Group)
        def mixed(
            self,
            x: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP16]],
        ) -> pl.Tensor[[16, 16], pl.FP16]:
            updated = self.cube(x, out)
            self.vector(x, out)
            return updated

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP16]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(self, x: pl.Tensor[[16, 16], pl.FP16]) -> pl.Tensor[[16, 16], pl.FP16]:
            scratch = pl.create_tensor([16, 16], dtype=pl.FP16)
            with pl.scope():
                produced = self.mixed(x, scratch)
            with pl.scope():
                out = self.consume(produced)
            return out

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    assert "MixedKernels mixed_0" in code, code
    assert "TaskOutputTensors task_0_outs = rt_submit_task(mixed_0, params_t0);" in code, code
    assert "produced_tid = task_0_outs.task_id();" in code, code
    assert re.search(
        r"if \(produced_tid\.is_valid\(\)\) params_t\d+_deps\[params_t\d+_deps_count\+\+\] = produced_tid;",
        code,
    ), code


def test_compiler_derived_deps_for_fixed_trip_loop_fan_in_capture_task_ids():
    """Fixed-trip loop producers should fan in all iteration TaskIds to a later consumer."""

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            carried = scratch
            last_tid = pl.system.task_invalid()
            for _i, (carried, last_tid) in pl.range(
                0,
                4,
                init_values=(carried, last_tid),  # pyright: ignore[reportArgumentType]
            ):
                carried, last_tid = pl.submit(self.fill, carried)
                carried, last_tid = pl.yield_(carried, last_tid)
            out = self.consume(carried)
            return out

    code = _generate_orch_full_pipeline(
        P,
        analyze_auto_scopes_for_deps=True,
        allow_relaxed_verification=True,
    )

    fan_in = re.search(r"TaskId\s+(last_tid\w*)\[4\];", code)
    assert fan_in, code
    fan_in_name = fan_in.group(1)
    assert f"{fan_in_name}[_i] = last_tid__ssa_v0;" in code, code
    assert re.search(r"TaskId params_t\d+_deps\[4\];", code), code
    assert len(re.findall(rf"if \({fan_in_name}\[\d+\]\.is_valid\(\)\)", code)) == 4, code
    assert re.search(r"params_t\d+\.set_dependencies\(params_t\d+_deps, params_t\d+_deps_count\);", code), (
        code
    )


def test_compiler_derived_deps_for_dynamic_trip_loop_fan_in_falls_back():
    """Dynamic-trip loop fan-in should fall back instead of allocating vectors."""

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            scratch: pl.Tensor[[64], pl.FP32],
            n_steps: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            carried = scratch
            last_tid = pl.system.task_invalid()
            for _i, (carried, last_tid) in pl.range(
                0,
                n_steps,
                init_values=(carried, last_tid),  # pyright: ignore[reportArgumentType]
            ):
                carried, last_tid = pl.submit(self.fill, carried)
                carried, last_tid = pl.yield_(carried, last_tid)
            out = self.consume(carried)
            return out

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    assert "std::vector<TaskId>" not in code
    assert "#include <vector>" not in code
    assert ".set_dependencies(" not in code
    assert "SIMPLER_SCOPE(ScopeMode::MANUAL)" not in code


def test_compiler_derived_deps_for_dynamic_trip_tensor_carrier_falls_back():
    """Dynamic-trip tensor carriers should fall back instead of allocating vectors."""

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            scratch: pl.Tensor[[64], pl.FP32],
            n_steps: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            carried = scratch
            for _i, (carried,) in pl.range(
                0,
                n_steps,
                init_values=(carried,),
            ):
                carried = self.fill(carried)
                carried = pl.yield_(carried)
            out = self.consume(carried)
            return out

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    assert "std::vector<TaskId>" not in code
    assert "#include <vector>" not in code
    assert ".set_dependencies(" not in code


def test_descending_parallel_loop_emits_descending_cpp_loop():
    """Regression: a descending ``pl.parallel`` must not compile to a dead loop.

    ``EvalConstTripCount`` returned 0 for any ``step <= 0``, and the emitted C++
    repeated the same positive-step assumption twice more: the fan-in capacity
    expression evaluated to 0, and ``EmitForLoopHeader`` hard-coded ``i < stop``
    so ``for (i = 4; i < 0; i += -1)`` ran zero times. The kernel silently
    submitted nothing.

    Both directions run 4 times, so both must take the static ``TaskId[4]``
    path and emit a loop whose condition actually admits iterations.
    """

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    def build(start, stop, step):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.AIV)
            def fill(self, out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                return out

            @pl.function(type=pl.FunctionType.AIV)
            def consume(self, q: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return q

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, q: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                q_c = q
                for _i, (q_c,) in pl.parallel(start, stop, step, init_values=(q_c,)):
                    q_c = self.fill(q_c)
                    q_c = pl.yield_(q_c)
                return self.consume(q_c)

        return P

    up = _generate_orch_full_pipeline(build(0, 4, 1), analyze_auto_scopes_for_deps=True)
    down = _generate_orch_full_pipeline(build(4, 0, -1), analyze_auto_scopes_for_deps=True)

    # The ascending case must be byte-for-byte unchanged in its loop header:
    # a constant positive step still emits the plain ``<`` form.
    assert re.search(r"for \(int64_t \w+ = 0; \w+ < 4; \w+ \+= 1\)", up), up

    # The descending case must test ``>``, not ``<`` — otherwise it never runs.
    down_header = re.search(r"for \(int64_t (\w+) = 4; \1 ([<>]) 0; \1 \+= -1\)", down)
    assert down_header, down
    assert down_header.group(2) == ">", down

    # Both directions have a static trip count of 4, so both must size their
    # compiler-dep fan-in as a fixed TaskId[4] and neither may fall back to
    # the dynamic vector path (whose capacity expression also evaluated to 0).
    for label, code in (("ascending", up), ("descending", down)):
        assert re.search(r"TaskId \w+\[4\]", code), (label, code)
        assert "std::vector<TaskId>" not in code, (label, code)


def test_zero_step_loop_emits_a_terminating_condition():
    """A zero step must produce zero iterations, never a non-terminating loop.

    Review catch on #2427: picking the comparison from `step < 0` alone sent a
    constant zero step down the `<` branch, emitting `for (i = 0; i < N; i += 0)`
    — an infinite loop in the generated orchestration function. A zero step is
    zero-trip per `ComputeStaticTripCount`, so the emitted condition must agree.
    """

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill(self, out: pl.Out[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, q: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return q

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, q: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            q_c = q
            for _i, (q_c,) in pl.parallel(0, 4, 0, init_values=(q_c,)):
                q_c = self.fill(q_c)
                q_c = pl.yield_(q_c)
            return self.consume(q_c)

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    # Whatever else it does, it must not emit a loop that both steps by zero and
    # has a condition that can hold.
    for header in re.findall(r"for \(int64_t \w+ = [^;]+; ([^;]+); \w+ \+= 0\)", code):
        assert header.strip() == "false", code


def test_compiler_derived_deps_for_dynamic_parallel_tensor_carriers_share_phase_barrier():
    """Dynamic parallel tensor producers in one phase should share one dummy barrier."""

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill_q(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def fill_k(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def fill_v(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def consume(
            self,
            q: pl.Tensor[[64], pl.FP32],
            k: pl.Tensor[[64], pl.FP32],
            v: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            return q

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            q: pl.Tensor[[64], pl.FP32],
            k: pl.Tensor[[64], pl.FP32],
            v: pl.Tensor[[64], pl.FP32],
            n_steps: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            q_carried = q
            k_carried = k
            v_carried = v
            for _i, (q_carried, k_carried, v_carried) in pl.parallel(
                0,
                n_steps,
                init_values=(q_carried, k_carried, v_carried),
            ):
                q_carried = self.fill_q(q_carried)
                k_carried = self.fill_k(k_carried)
                v_carried = self.fill_v(v_carried)
                q_carried, k_carried, v_carried = pl.yield_(q_carried, k_carried, v_carried)
            out = self.consume(q_carried, k_carried, v_carried)
            return out

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    assert "#include <vector>" in code
    collection = re.search(
        r"std::vector<TaskId>\s+(\w+)\(static_cast<size_t>\((\w+)\)\);\n\s+uint32_t\s+(\w+)\s*=\s*0;",
        code,
    )
    assert collection, code
    buffer_name, capacity_name, count_name = collection.groups()
    capacity_init = re.search(rf"const int64_t {capacity_name} = ([^\n]+);", code)
    assert capacity_init, code
    assert " * 3" in capacity_init.group(1), code
    assert ".push_back(" not in code
    assert code.count(f"{buffer_name}[{count_name}++] =") == 3, code
    assert code.count("rt_orch_profile_add_dynamic_dep_vector") == 4, code
    assert "#if SIMPLER_ORCH_PROFILING" in code, code
    assert "PTO2_ORCH_PROFILING" not in code, code
    assert code.count("Dynamic compiler-dependency barrier") == 1, code
    assert code.count("rt_submit_dummy_task") == 1, code
    assert f".set_dependencies({buffer_name}.data(), {count_name});" in code, code
    assert re.search(r"TaskId params_t\d+_deps\[1\];", code), code
    assert not re.search(r"TaskId params_t\d+_deps\[[23]\];", code), code
    assert code.count(".add_no_dep(") >= 3, code


def test_compiler_derived_deps_for_dynamic_trip_tuple_output_tensor_carrier_falls_back():
    """Dynamic-trip tuple-output tensor carriers should fall back instead of allocating vectors."""

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def fill_pair(
            self,
            left: pl.Out[pl.Tensor[[64], pl.FP32]],
            right: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            return left, right

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            left: pl.Tensor[[64], pl.FP32],
            right: pl.Tensor[[64], pl.FP32],
            n_steps: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            for _i, (left_carried, right_carried) in pl.range(
                0,
                n_steps,
                init_values=(left, right),
            ):
                left_carried, right_carried = self.fill_pair(left_carried, right_carried)
                left_carried, right_carried = pl.yield_(left_carried, right_carried)
            out = self.consume(right_carried)
            return out

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    assert "std::vector<TaskId>" not in code
    assert "#include <vector>" not in code
    assert ".set_dependencies(" not in code


def test_compiler_auto_manual_scope_is_not_tied_to_function_name():
    """A loop-local tuple producer must not overwrite an outer tuple producer TaskId.

    This mirrors the Qwen decode qk_norm -> cache-update shape.  The loop body
    should depend on the stable qk_norm producer every iteration, while the
    loop-local producer TaskIds are collected separately for the later fan-in.
    The compiler-created MANUAL wrapper must come from dependency analysis, not
    from a callee-name substring.
    """

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def qk_norm(
            self,
            q_src: pl.Tensor[[64], pl.FP32],
            k_src: pl.Tensor[[64], pl.FP32],
            q_out: pl.Out[pl.Tensor[[64], pl.FP32]],
            k_out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            return q_out, k_out

        @pl.function(type=pl.FunctionType.AIV)
        def cache_update(
            self,
            q_norm: pl.Tensor[[64], pl.FP32],
            k_norm: pl.Tensor[[64], pl.FP32],
            all_q: pl.Out[pl.Tensor[[64], pl.FP32]],
            k_cache: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
            return all_q, k_cache

        @pl.function(type=pl.FunctionType.AIV)
        def consume(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            q_src: pl.Tensor[[64], pl.FP32],
            k_src: pl.Tensor[[64], pl.FP32],
            q_norm_buf: pl.Tensor[[64], pl.FP32],
            k_norm_buf: pl.Tensor[[64], pl.FP32],
            all_q: pl.Tensor[[64], pl.FP32],
            k_cache: pl.Tensor[[64], pl.FP32],
            n_steps: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[64], pl.FP32]:
            q_norm_buf, k_norm_buf = self.qk_norm(q_src, k_src, q_norm_buf, k_norm_buf)
            for _i, (all_q_carried, k_cache_carried) in pl.parallel(
                0,
                n_steps,
                init_values=(all_q, k_cache),
            ):
                all_q_carried, k_cache_carried = self.cache_update(
                    q_norm_buf, k_norm_buf, all_q_carried, k_cache_carried
                )
                all_q_carried, k_cache_carried = pl.yield_(all_q_carried, k_cache_carried)
            out = self.consume(k_cache_carried)
            return out

    code = _generate_orch_full_pipeline(
        P,
        analyze_auto_scopes_for_deps=True,
        allow_relaxed_verification=True,
    )

    assert code.count("SIMPLER_SCOPE(ScopeMode::MANUAL)") == 1, code
    qk_tid = re.search(r"\b(\w+_tid)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert qk_tid, code
    rope_tid = re.search(r"(?:TaskId\s+)?(\w+_tid\w*)\s*=\s*task_1_outs\.task_id\(\);", code)
    assert rope_tid, code
    assert qk_tid.group(1) != rope_tid.group(1), code
    collection = re.search(
        r"std::vector<TaskId>\s+(\w+)\(static_cast<size_t>\(\w+\)\);\n\s+uint32_t\s+(\w+)\s*=\s*0;",
        code,
    )
    assert collection, code
    buffer_name, count_name = collection.groups()
    assert code.count(f"{buffer_name}[{count_name}++] =") == 1, code
    manual_scope_idx = code.index("SIMPLER_SCOPE(ScopeMode::MANUAL)")
    qk_submit_idx = code.index("TaskOutputTensors task_0_outs")
    assert manual_scope_idx < qk_submit_idx, code
    rope_submit_idx = code.index("TaskOutputTensors task_1_outs", manual_scope_idx)
    rope_deps_idx = code.rfind("params_t1.set_dependencies", manual_scope_idx, rope_submit_idx)
    assert rope_deps_idx > manual_scope_idx, code
    assert re.search(
        rf"if \({qk_tid.group(1)}\.is_valid\(\)\) "
        rf"params_t1_deps\[params_t1_deps_count\+\+\] = {qk_tid.group(1)};",
        code,
    ), code
    assert f"{qk_tid.group(1)} = task_1_outs.task_id();" not in code, code


def test_mixed_in_and_out_of_scope_deps_reports_user_error():
    """A deps= list mixing an in-scope TaskId with one scoped out of a closed
    ``pl.scope()`` is reported as a clean user error.

    The partially-satisfiable case must not degrade into either failure mode the
    codegen has had historically: an ``INTERNAL_CHECK`` abort (compiler crash),
    or silently wiring only the in-scope edge while dropping the other (the
    consumer then races its out-of-scope producer). It raises a
    ``pypto::ValueError`` naming the unsatisfiable edge — a ``ValueError`` in
    Python, distinct from ``pypto.InternalError`` which derives from
    ``RuntimeError``.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k3(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                _a, scoped_tid = pl.submit(self.k1, x)
            _seed, seed_tid = pl.submit(self.k2, x)
            b, _ = pl.submit(self.k3, x, deps=[seed_tid, scoped_tid])
            return b

    # The unsatisfiable edge is named, not the satisfiable ``seed_tid``.
    msg = _assert_dep_edge_rejected(P, "scoped_tid")
    # A user-facing diagnostic, not an internal-invariant abort.
    assert "Internal error" not in msg, msg


def test_user_dep_edge_out_of_compiler_inserted_loop_scope_is_rejected():
    """The boundary need not be user-written to swallow a ``deps=[...]`` edge.

    ``MaterializeRuntimeScopes`` wraps every ``ForStmt`` body in its own AUTO
    ``SIMPLER_SCOPE``, so a TaskId produced in the loop body and depended on after
    the loop crosses a closed scope even though the source names no scope at
    all. The loop-carry machinery does hoist a valid C++ ``TaskId`` to the
    outer level, but the dep binding is not visible there, so the edge would be
    dropped — reject instead.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            loop_tid = pl.system.task_invalid()
            for _i in pl.range(2):
                _a, loop_tid = pl.submit(self.k1, x)
            b, _ = pl.submit(self.k2, x, deps=[loop_tid])
            return b

    msg = _assert_dep_edge_rejected(P, "loop_tid")
    # The message points at loop bodies / branches as scope openers.
    assert "pl.range" in msg, msg


def test_spmd_grid_tid_dep_across_manual_scope_boundary_is_rejected():
    """A captured ``with pl.spmd(N) as tid:`` grid TaskId depended on from the
    auto region after the enclosing ``pl.manual_scope()`` is rejected.

    The grid TaskId is a genuine whole-grid join (the runtime completes the slot
    only after all N blocks retire), so the edge is meaningful — it just cannot
    be wired from outside the scope that produced it. Silently dropping it left
    the reader racing all N blocks.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            t = pl.load(a, [0, 0], [64, 64])
            return pl.store(pl.add(t, t), [0, 0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def reader(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            t = pl.load(a, [0, 0], [64, 64])
            return pl.store(pl.add(t, t), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[64, 64], pl.FP32],
            scratch: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            with pl.manual_scope():
                with pl.spmd(4) as grid_tid:
                    scratch = self.producer(x, scratch)
            out, _ = pl.submit(self.reader, scratch, out, deps=[grid_tid])
            return out

    _assert_dep_edge_rejected(P, "grid_tid")


def test_grid_tid_dep_inside_producing_manual_scope_still_wires():
    """Control for the two rejections above: the same grid/reader pair kept
    inside one ``pl.manual_scope()`` still emits the ``set_dependencies`` edge.

    Pins that the new check rejects only genuinely unsatisfiable edges and does
    not regress the supported in-scope wiring.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            t = pl.load(a, [0, 0], [64, 64])
            return pl.store(pl.add(t, t), [0, 0], out)

        @pl.function(type=pl.FunctionType.InCore)
        def reader(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            t = pl.load(a, [0, 0], [64, 64])
            return pl.store(pl.add(t, t), [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[64, 64], pl.FP32],
            scratch: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            with pl.manual_scope():
                with pl.spmd(4) as grid_tid:
                    scratch = self.producer(x, scratch)
                out, _ = pl.submit(self.reader, scratch, out, deps=[grid_tid])
            return out

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)
    assert re.search(r"TaskId grid_tid = task_0_outs\.task_id\(\);", code), code
    assert re.search(r"\] = grid_tid;", code), code
    assert ".set_dependencies(" in code, code


def test_spmd_submit_emits_launch_spec_and_captures_task_id():
    """``pl.spmd_submit`` lowers to a single submit carrying the SPMD launch
    spec (set_block_num / set_require_sync_start) and a captured producer
    TaskId that a downstream submit depends on.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def producer(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            bi = pl.tile.get_block_idx()
            off = bi * 128
            t = pl.load(x, [off, 0], [128, 128])
            out = pl.store(t, [off, 0], out)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def consumer(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            t = pl.load(x, [0, 0], [128, 128])
            out = pl.store(t, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            with pl.manual_scope():
                scratch = pl.create_tensor([512, 128], dtype=pl.FP32)
                scratch, tid = pl.spmd_submit(self.producer, x, scratch, core_num=4, sync_start=True)
                out, _ = pl.spmd_submit(self.consumer, scratch, out, core_num=2, deps=[tid])
            return out

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)
    # Producer carries core_num=4 + sync_start; consumer carries core_num=2.
    assert "params_t0.launch_spec.set_block_num(4);" in code, code
    assert "params_t0.launch_spec.set_require_sync_start(true);" in code, code
    assert "params_t1.launch_spec.set_block_num(2);" in code, code
    # sync_start defaults False on the consumer — emitted exactly once.
    assert code.count("set_require_sync_start(true)") == 1, code
    # Producer TaskId is captured and the consumer depends on it.
    assert re.search(r"TaskId\s+\w+\s*=\s*task_0_outs\.task_id\(\);", code), code
    assert "set_dependencies(" in code, code
    # Direct-call dispatch emits set_dependencies BEFORE the launch spec (deps
    # -> launch_spec -> early_resolve). This locks the byte-identical per-path
    # emission order; wrapper paths (Spmd/Group/Mixed) use the opposite order.
    deps_pos = code.index("params_t1.set_dependencies(")
    launch_pos = code.index("params_t1.launch_spec.set_block_num(2);")
    assert deps_pos < launch_pos, code


def test_spmd_submit_group_emits_mixed_kernels_and_launch_spec():
    """``pl.spmd_submit`` of a split (cube+vector) kernel routes through the
    Group dispatch path and still emits the SPMD launch spec.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
        def kernel(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            bias: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            tile_mm = pl.matmul(tile_a_l0a, tile_b_l0b)
            tile_bias = pl.load(bias, [0, 0], [64, 64])
            tile_out = pl.add(tile_mm, tile_bias)
            out = pl.store(tile_out, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            bias: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.spmd_submit(self.kernel, a, b, bias, out, core_num=4, sync_start=True)
            return out

    code = _generate_orch_full_pipeline(P)
    assert "MixedKernels mixed_0" in code, code
    assert "rt_submit_task(mixed_0, params_t0);" in code, code
    assert "params_t0.launch_spec.set_block_num(4);" in code, code
    assert "params_t0.launch_spec.set_require_sync_start(true);" in code, code


def test_plain_submit_emits_no_launch_spec():
    """Regression: a plain ``pl.submit`` (no SPMD launch spec) must not emit
    any launch_spec calls — the spmd_submit path is fully opt-in.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            t = pl.load(x, [0], [128])
            out = pl.store(t, [0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.k, x, out)
            return out

    code = _generate_orch_full_pipeline(P)
    assert "launch_spec" not in code, code


def test_submit_allow_early_resolve_emits_hint():
    """``pl.submit(..., allow_early_resolve=True)`` lowers to a
    ``set_allow_early_resolve(true)`` call on the producer task's Arg before its
    rt_submit_* (simpler#1065). A submit without the flag emits nothing.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            t = pl.load(x, [0], [128])
            out = pl.store(t, [0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            with pl.manual_scope():
                # First submit opts in; second does not.
                scratch = pl.create_tensor([128], dtype=pl.FP32)
                scratch, tid = pl.submit(self.k, x, scratch, allow_early_resolve=True)
                out, _ = pl.submit(self.k, scratch, out, deps=[tid])
            return out

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)
    # The flagged producer (task 0) gets the hint; the plain consumer (task 1)
    # does not — so the call appears exactly once.
    assert "params_t0.set_allow_early_resolve(true);" in code, code
    assert code.count("set_allow_early_resolve(true)") == 1, code


def test_plain_submit_emits_no_allow_early_resolve():
    """Regression: a plain ``pl.submit`` (no hint) must not emit any
    ``set_allow_early_resolve`` call — the feature is fully opt-in.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            t = pl.load(x, [0], [128])
            out = pl.store(t, [0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.submit(self.k, x, out)
            return out

    code = _generate_orch_full_pipeline(P)
    assert "set_allow_early_resolve" not in code, code


def test_at_allow_early_resolve_emits_hint():
    """``pl.at(..., allow_early_resolve=True)`` outlines into a Submit carrying
    the flag, so codegen emits ``set_allow_early_resolve(true)`` on the
    synthesized dispatch's Arg (simpler#1065).
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
                t = pl.load(x, [0, 0], [128, 128])
                out = pl.store(t, [0, 0], out)
            return out

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)
    assert "set_allow_early_resolve(true);" in code, code
    assert code.count("set_allow_early_resolve(true)") == 1, code


def test_spmd_submit_aic_direct_dispatch():
    """spmd_submit of a directly-declared AIC (cube) kernel routes through the
    direct dispatch path: rt_submit_aic_task + launch spec (910B)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIC)
        def cube(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            ta = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            tb = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
            la = pl.move(ta, target_memory=pl.MemorySpace.Left)
            lb = pl.move(tb, target_memory=pl.MemorySpace.Right)
            mm = pl.matmul(la, lb)
            out = pl.store(mm, [0, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.spmd_submit(self.cube, a, b, out, core_num=8)
            return out

    code = _generate_orch_code(P)
    assert "rt_submit_aic_task" in code, code
    assert "params_t0.launch_spec.set_block_num(8);" in code, code


def test_spmd_submit_core_num_variable_emits_var_reference():
    """A non-constant core_num (an orchestration scalar parameter) is emitted as
    a variable reference in the launch spec, not constant-folded."""

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k(
            self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            bi = pl.tile.get_block_idx()
            off = bi * 128
            t = pl.load(x, [off, 0], [128, 128])
            out = pl.store(t, [off, 0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[512, 128], pl.FP32],
            n: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.spmd_submit(self.k, x, out, core_num=n)
            return out

    code = _generate_orch_full_pipeline(P)
    m = re.search(r"launch_spec\.set_block_num\(([^)]*)\)", code)
    assert m, f"no set_block_num in:\n{code}"
    # The argument is a variable reference (a name), not a constant-folded literal.
    assert not m.group(1).strip().isdigit(), f"core_num was constant-folded, expected a var: {m.group(1)!r}"


def test_spmd_submit_950_backend_emits_set_core_num():
    """On Ascend950 the launch spec uses set_core_num (vs set_block_num on 910B),
    via the backend handler's GetLaunchSpecCoreCountMethod()."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.AIV)
        def k(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            t = pl.load(x, [0], [128])
            out = pl.store(t, [0], out)
            return out

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self, x: pl.Tensor[[128], pl.FP32], out: pl.Out[pl.Tensor[[128], pl.FP32]]
        ) -> pl.Tensor[[128], pl.FP32]:
            with pl.manual_scope():
                out, _ = pl.spmd_submit(self.k, x, out, core_num=4)
            return out

    # Property-only verification (see test_spmd_submit_aic_direct_dispatch).
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        code = _generate_orch_code(P)
    assert "params_t0.launch_spec.set_core_num(4);" in code, code
    assert "set_block_num" not in code, code


def test_compiler_dep_carry_array_sized_by_outer_loop_trip_count():
    """Compiler-dep TaskId carry arrays must be sized by the *outer* Sequential loop.

    When a Sequential outer loop (trip M) wraps a Parallel inner loop (trip N)
    inside a ``pl.manual_scope``, and the outer loop's TaskId iter_arg receives a
    compiler-derived dependency edge, the carry array must declare
    ``TaskId arr[M]`` (outer trip), NOT ``arr[N]`` (inner trip).

    ``ResolveArrayCarrySize`` would otherwise recurse into the inner Parallel loop
    and return N, producing a mis-sized fan-in array that over- or under-fences
    (YunjiQin review, PR #1813).  The codegen post-process must unconditionally
    use ``EvalConstTripCount`` of the outer loop for compiler-dep carries.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    M, N = 4, 8

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIV)
        def k1(
            self,
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            return out

        @pl.function(type=pl.FunctionType.AIV)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, scratch: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.manual_scope():
                prev = pl.system.task_invalid()
                for _i in pl.range(M):
                    for _j in pl.parallel(N):
                        _, _tid = pl.submit(self.k1, scratch)
                    # compiler_manual_dep_edges on prev: makes it an
                    # iter_arg carry of the outer Sequential loop,
                    # and triggers NeedsCompilerDepTaskId for its
                    # return_var.
                    self.k2(scratch, attrs={"compiler_manual_dep_edges": [prev]})
                    prev = pl.system.task_invalid()
                return scratch

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    transformed = pm.run_passes(Prog)
    code = _generate_orch_code(transformed)

    # The compiler-dep carry array must be sized by the outer loop trip M=4.
    # The emit name is derived from prev's return_var (SSA-base "prev").
    outer_arr = re.search(r"TaskId\s+(prev\w*)\[(" + str(M) + r")\];", code)
    assert outer_arr, f"Expected TaskId <prev...>[{M}] (outer trip) in:\n{code}"
    # The init loop also iterates M times, not N.
    assert f"for (int64_t __init_i = 0; __init_i < {M}; ++__init_i)" in code, (
        f"Expected init loop bound {M} in:\n{code}"
    )
    # No array declaration should be sized by the inner trip N=8.
    assert not re.search(r"TaskId\s+\w+\[" + str(N) + r"\];", code), (
        f"Unexpected TaskId array sized [{N}] (inner trip) in:\n{code}"
    )


def test_user_written_task_dummy_dep_across_closed_scope_is_rejected():
    """A user ``pl.system.task_dummy(deps=[tid])`` is enforced like ``pl.submit``.

    ``task_dummy`` is a user-facing DSL construct, and the parser stamps
    ``attrs["dummy_task"]`` on it exactly as ``ExpandManualPhaseFence`` does on
    the barriers it synthesises. Treating ``dummy_task`` as a compiler-authorship
    marker would therefore silently drop this user edge, leaving the barrier with
    no fanin and every consumer gated on it unordered — the failure this change
    exists to prevent.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
        def main(
            self,
            x: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64], pl.FP32]],
        ) -> pl.Tensor[[64], pl.FP32]:
            with pl.scope():
                _a, scoped_tid = pl.submit(self.k1, x)
            # The barrier's fanin names a TaskId whose scope has closed.
            barrier = pl.system.task_dummy(deps=[scoped_tid])
            b, _ = pl.submit(self.k2, x, deps=[barrier])
            return b

    _assert_dep_edge_rejected(P, "scoped_tid")


def test_user_written_task_dummy_with_in_scope_dep_still_wires():
    """Control for the rejection above: an in-scope ``task_dummy`` fanin wires.

    Proves the enforcement rejects only genuinely unresolvable edges and does not
    over-reject the supported ``pl.system.task_dummy`` path.
    """

    @pl.program
    class P:
        @pl.function(type=pl.FunctionType.InCore)
        def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.InCore)
        def k2(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            with pl.manual_scope():
                _a, tid = pl.submit(self.k1, x)
                barrier = pl.system.task_dummy(deps=[tid])
                b, _ = pl.submit(self.k2, x, deps=[barrier])
            return b

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    code = _generate_orch_code(pm.run_passes(P))

    assert "rt_submit_dummy_task(" in code, code
    assert code.count("set_dependencies(") >= 2, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
