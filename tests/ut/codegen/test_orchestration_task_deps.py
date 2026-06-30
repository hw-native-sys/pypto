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

    # ``prev = tids[0]`` lowers to a scalar PTO2TaskId snapshot local read from
    # the array slot (the dep site references the snapshot, not a slot re-read).
    assert re.search(r"PTO2TaskId\s+prev\w*\s*=\s*tids\w*\[0\];", code), code
    # The consumer gets exactly one dependency array, filled from BOTH the direct
    # producer tid (seed_tid) and the array-slot snapshot (prev), each guarded.
    assert code.count("set_dependencies(") == 1, code
    assert re.search(r"if \(seed_tid\w*\.is_valid\(\)\)", code), code
    assert re.search(r"if \(prev\w*\.is_valid\(\)\)", code), code


def test_task_id_binding_does_not_leak_past_pl_scope():
    """Regression for the issue #1577 lifetime hazard.

    A producer TaskId declared inside a nested AUTO ``pl.scope()`` names a
    ``PTO2TaskId`` C++ local that dies at the block's closing brace. Codegen must
    not reference it after the block — before the fix it emitted a
    ``set_dependencies`` fill from the out-of-scope local (uncompilable C++).
    The TaskId binding is now scoped to the ``PTO2_SCOPE`` it is produced in, so
    its identifier appears exactly once (its declaration) in the generated code.
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

    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)

    # The producer tid is declared inside the inner PTO2_SCOPE block.
    m = re.search(r"PTO2TaskId\s+(\w+)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert m, code
    tid_name = m.group(1)
    # It must NOT be referenced after its block closes — the binding is scoped
    # out, so the only occurrence is its declaration (no dangling dep fill).
    assert code.count(tid_name) == 1, f"TaskId local '{tid_name}' leaked past its pl.scope():\n{code}"


def test_cross_scope_task_id_is_hoisted_for_set_dependencies():
    """A compiler-derived dependency can legally cross a generated PTO2_SCOPE.

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

    assert re.search(r"PTO2TaskId\s+scoped_tid_tid\s*=\s*PTO2TaskId::invalid\(\);", code), code
    assert "scoped_tid_tid = task_0_outs.task_id();" in code
    assert "PTO2TaskId scoped_tid = task_0_outs.task_id();" not in code
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

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    fan_in = re.search(r"PTO2TaskId\s+(last_tid\w*)\[4\];", code)
    assert fan_in, code
    fan_in_name = fan_in.group(1)
    assert f"{fan_in_name}[_i] = last_tid__ssa_v0;" in code, code
    assert re.search(r"PTO2TaskId params_t\d+_deps\[4\];", code), code
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

    assert "std::vector<PTO2TaskId>" not in code
    assert "#include <vector>" not in code
    assert ".set_dependencies(" not in code


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

    assert "std::vector<PTO2TaskId>" not in code
    assert "#include <vector>" not in code
    assert ".set_dependencies(" not in code


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

    assert "std::vector<PTO2TaskId>" not in code
    assert "#include <vector>" not in code
    assert ".set_dependencies(" not in code


def test_compiler_derived_deps_keep_outer_tuple_producer_task_id_stable_in_dynamic_parallel_loop():
    """A loop-local tuple producer must not overwrite an outer tuple producer TaskId.

    This mirrors the Qwen decode qk_norm -> rope_kv_cache pattern.  The loop
    body should depend on the stable qk_norm producer every iteration, while the
    rope producer TaskIds are collected separately for the later fan-in.
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
        def rope_kv_cache(
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
                all_q_carried, k_cache_carried = self.rope_kv_cache(
                    q_norm_buf, k_norm_buf, all_q_carried, k_cache_carried
                )
                all_q_carried, k_cache_carried = pl.yield_(all_q_carried, k_cache_carried)
            out = self.consume(k_cache_carried)
            return out

    code = _generate_orch_full_pipeline(P, analyze_auto_scopes_for_deps=True)

    qk_tid = re.search(r"\b(\w+_tid)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert qk_tid, code
    rope_tid = re.search(r"(?:PTO2TaskId\s+)?(\w+_tid\w*)\s*=\s*task_1_outs\.task_id\(\);", code)
    assert rope_tid, code
    assert qk_tid.group(1) != rope_tid.group(1), code
    assert re.search(
        rf"if \({qk_tid.group(1)}\.is_valid\(\)\) "
        rf"params_t1_deps\[params_t1_deps_count\+\+\] = {qk_tid.group(1)};",
        code,
    ), code
    assert f"{qk_tid.group(1)} = task_1_outs.task_id();" not in code, code


def test_mixed_in_and_out_of_scope_deps_does_not_crash_codegen():
    """A deps= list mixing an in-scope TaskId with one scoped out of a closed
    ``pl.scope()`` must not abort codegen.

    ``CountManualDeps`` skips the out-of-scope edge when sizing the dep stack
    array, so the count is non-zero (the in-scope edge). ``EmitManualDeps`` must
    skip the same out-of-scope edge rather than asserting on it — otherwise the
    mixed case trips an INTERNAL_CHECK and crashes the compiler. The in-scope
    edge is still wired; the out-of-scope edge is dropped.
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

    # Must not raise (regression: EmitManualDeps used to INTERNAL_CHECK here).
    code = _generate_orch_full_pipeline(P, allow_relaxed_verification=True)

    # The in-scope edge is wired; the out-of-scope ``scoped_tid`` is dropped.
    assert code.count("set_dependencies(") == 1, code
    assert re.search(r"if \(seed_tid\w*\.is_valid\(\)\)", code), code
    m = re.search(r"PTO2TaskId\s+(\w+)\s*=\s*task_0_outs\.task_id\(\);", code)
    assert m, code
    assert code.count(m.group(1)) == 1, code


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
    assert re.search(r"PTO2TaskId\s+\w+\s*=\s*task_0_outs\.task_id\(\);", code), code
    assert "set_dependencies(" in code, code


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
