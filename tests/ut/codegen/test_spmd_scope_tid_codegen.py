# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Orchestration codegen for ``with pl.spmd(...) as tid:`` — the SPMD producer-TaskId capture.

A captured SPMD dispatch lowers to an ``ir.Submit`` whose own ``core_num`` is None
(it rides on the outlined ``Spmd`` Function attrs), so the launch spec must come
through ``EffectiveLaunchSpec``'s function-attr fallback. These tests pin that
fallback plus producer-TaskId capture and explicit ``deps=`` emission.
"""

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.pypto_core import ir


class TestSpmdScopeTaskIdCodegen:
    """Codegen for the captured ``with pl.spmd(...) as tid:`` SPMD dispatch."""

    @staticmethod
    def _mixed_spmd_pipeline(program):
        return passes.expand_mixed_kernel()(
            passes.infer_tile_memory_space()(
                passes.outline_cluster_scopes()(passes.convert_to_ssa()(program))
            )
        )

    @staticmethod
    def _codegen(program):
        """Run the codegen-entry passes on a hand-built program, then codegen.

        NormalizeReturnOrder first: orchestration codegen reads each callee's
        return->param map straight off its ReturnStmt (IRProperty
        ReturnParamsExplicit), which is a declared codegen precondition. It is a
        no-op for programs that already went through the pass pipeline.


        Runs under the repo conftest's default ``PYPTO_VERIFY_LEVEL=roundtrip``
        instrument (print -> parse -> structural_equal after each pass):
        DeriveCallDirections now PRESERVES the captured ``as tid`` dispatch as a
        Submit (printed as ``pl.submit(...)``), so it round-trips — no
        VerificationLevel.NONE bypass is needed.
        """
        program = passes.normalize_return_order()(program)
        program = passes.derive_call_directions()(program)
        program = passes.materialize_dist_tensor_ctx()(program)
        program = passes.materialize_runtime_scopes()(program)
        program = passes.classify_iter_arg_carry()(program)
        for func in program.functions.values():
            if func.func_type == ir.FunctionType.Orchestration:
                return codegen.generate_orchestration(program, func).code
        raise ValueError("No orchestration function found in program")

    def test_as_tid_launch_spec_via_function_attr_fallback(self):
        """A tid-bearing Spmd dispatch still emits set_block_num / set_require_sync_start.

        The Submit's ``core_num`` is None (SubmitToCallView emits no core_num attr),
        so this proves ``EffectiveLaunchSpec`` falls back to the Spmd Function's
        ``core_num`` / ``sync_start`` attrs.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

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
                with pl.spmd(4, sync_start=True) as tid:  # captured grid dispatch TaskId
                    out = self.kernel(a, b, bias, out)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        spmd_func = transformed.get_function("main_spmd_0")
        assert spmd_func is not None
        assert spmd_func.func_type == pl.FunctionType.Spmd
        assert "core_num" in spmd_func.attrs  # launch spec rides on the Spmd Function

        code = self._codegen(transformed)
        assert "params_t0.launch_spec.set_block_num(4);" in code, code
        assert "params_t0.launch_spec.set_require_sync_start(true);" in code, code

    def test_as_tid_deps_emit_set_dependencies_and_capture_task_id(self):
        """A captured dispatch feeding a downstream ``deps=[tid]`` dispatch emits a
        ``TaskOutputTensors`` capture (``.task_id()``) and ``set_dependencies(...)``."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class P:
            # Plain vector InCore kernel (no mixed wrapper) keeps the test focused
            # on the dep wiring. Two separate Out buffers avoid feeding one
            # dispatch's tuple-return into the next dispatch's args — the edge is
            # carried purely by deps=[tid0].
            @pl.function(type=pl.FunctionType.InCore)
            def vkernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(pl.add(t, t), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out1: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
                out2: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid0:
                    out1 = self.vkernel(a, out1)
                with pl.spmd(4, name_hint="stage2", deps=[tid0]) as tid1:
                    out2 = self.vkernel(a, out2)
                return out2

        transformed = self._mixed_spmd_pipeline(P)
        spmd_fns = [f for f in transformed.functions.values() if f.func_type == pl.FunctionType.Spmd]
        assert len(spmd_fns) == 2

        code = self._codegen(transformed)
        # Bind the producer TaskId alias captured from the first dispatch ...
        m = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert m is not None, f"first dispatch's producer TaskId not captured\n{code}"
        alias = m.group(1)
        # ... assert THAT alias (not just any TaskId) is pushed into a deps array ...
        # A fresh direct-producer TaskId (issue #1966) is statically valid, so its
        # dep-array insert is emitted WITHOUT the is_valid() guard.
        m2 = re.search(rf"(\w+)\[[^\]]*\] = {re.escape(alias)};", code)
        assert m2 is not None, f"captured TaskId {alias!r} not wired into a deps array\n{code}"
        deps_arr = m2.group(1)
        # ... and that the same deps array is handed to the consumer's set_dependencies.
        assert re.search(rf"\.set_dependencies\({re.escape(deps_arr)},", code) is not None, (
            f"expected set_dependencies({deps_arr}, ...) tying the dep to {alias!r}\n{code}"
        )

    def test_mixed_spmd_in_equals_out_aliases_shared_buffer(self):
        """A MIXED (split=) ``pl.spmd`` dispatch passing one buffer as both an
        input AND the output must codegen — and both lanes of the aliased buffer
        must resolve to the SAME external tensor.

        Regression for the OOB read in ``BuildWrapperReorderedParams``: the Spmd
        outliner deduplicates the repeated ``out`` arg, so ``main_spmd_0`` has 3
        params while the Group keeps 4. Before the bridge fix, codegen indexed
        ``outer_call->args_[3]`` (a Group-param index) on the 3-arg Spmd-wrapper
        call → ``std::vector::operator[]`` UB → SIGSEGV, or the
        ``BuildWrapperReorderedParams`` "neither a variable nor a recognized
        constant literal" InternalError. No captured ``as tid`` / TupleGetItem is
        involved — a single dispatch reproduces it.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

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
                tile_out = pl.add(tile_mm, pl.load(bias, [0, 0], [64, 64]))
                return pl.store(tile_out, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    # `out` is BOTH the input (param a) and the output (param out)
                    out = self.kernel(out, b, bias, out)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        # The Spmd wrapper deduplicated the aliased `out` to one fewer param than
        # the Group it wraps — the precondition for the original crash.
        spmd_func = transformed.get_function("main_spmd_0")
        group_func = transformed.get_function("kernel")
        assert spmd_func is not None and group_func is not None
        assert len(spmd_func.params) < len(group_func.params)

        code = self._codegen(transformed)  # must not raise
        assert "rt_submit_task(mixed_0, params_t0);" in code, code
        # Both the input lane (Group param 0 = `a` ← out) and the output lane
        # (Group param 3 = `out`) of the deduped buffer resolve to the SAME
        # external tensor — proof that Group param 3 maps to outer arg 0, not OOB.
        shared = re.findall(r"params_t0\.add_\w+\(ext_out\);", code)
        assert len(shared) == 2, f"aliased buffer must appear on both lanes\n{code}"
        # The other (distinct) args are each emitted exactly once, unaffected.
        assert code.count("params_t0.add_input(ext_b);") == 1, code
        assert code.count("params_t0.add_input(ext_bias);") == 1, code

    def test_mixed_spmd_forwards_materialized_comm_ctx_scalar(self):
        """DistributedTensor ctx params flow through Spmd -> Group wrappers as ordinary scalars.

        Regression coverage for the #1913 family: wrapper codegen must not rely
        on a side-channel ctx-synthesis helper. Once
        MaterializeDistTensorCtx has appended ``signal_ctx`` to the wrapper
        signature and inner call, BuildWrapperReorderedParams should forward it
        with the normal scalar path.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                signal: pld.DistributedTensor[[1], pl.INT32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_mm = pl.matmul(tile_a_l0a, tile_b_l0b)
                tile_out = pl.add(tile_mm, pl.load(bias, [0, 0], [64, 64]))
                return pl.store(tile_out, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                signal: pld.DistributedTensor[[1], pl.INT32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, b, bias, out, signal)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        code = self._codegen(transformed)
        assert "uint64_t signal_ctx = orch_args.scalar(0);" in code, code
        assert code.count("params_t0.add_scalar(signal_ctx);") == 1, code
        assert "ext_signal_ctx" not in code, code

    def test_mixed_spmd_distinct_args_codegen_unchanged(self):
        """Control / no-regression: a MIXED ``pl.spmd`` dispatch with all-distinct
        args (no dedup) emits the same correct param block as before the fix.

        Here the Spmd wrapper and the Group have equal param counts, so the new
        bridge path collapses to the identity mapping — emitted text must be the
        canonical 4-arg mixed dispatch.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

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
                tile_out = pl.add(tile_mm, pl.load(bias, [0, 0], [64, 64]))
                return pl.store(tile_out, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, b, bias, out)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        # Distinct args => no dedup => equal param counts.
        spmd_func = transformed.get_function("main_spmd_0")
        group_func = transformed.get_function("kernel")
        assert spmd_func is not None and group_func is not None
        assert len(spmd_func.params) == len(group_func.params)

        code = self._codegen(transformed)
        for line in (
            "params_t0.add_input(ext_a);",
            "params_t0.add_input(ext_b);",
            "params_t0.add_input(ext_bias);",
            "params_t0.add_output(ext_out);",
            "rt_submit_task(mixed_0, params_t0);",
        ):
            assert line in code, f"missing {line!r}\n{code}"

    def test_captured_dispatch_mixed_consumer_in_equals_out(self):
        """The original KNOWN_ISSUES repro: a captured ``as tid`` MIXED dispatch
        whose tensor output feeds a downstream MIXED dispatch that also reuses it
        as its own output. The consumer's Spmd wrapper deduplicates to fewer
        params than its Group (same root cause), so it crashed at codegen.

        After the fix the consumer must (a) not crash, (b) alias both lanes of
        its deduped buffer to the producer's tuple output, and (c) wire
        ``deps=[tid0]`` from the captured producer TaskId.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

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
                tile_out = pl.add(tile_mm, pl.load(bias, [0, 0], [64, 64]))
                return pl.store(tile_out, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4, sync_start=True) as tid0:
                    out = self.kernel(a, b, bias, out)
                with pl.spmd(4, deps=[tid0]) as tid1:
                    out = self.kernel(out, b, bias, out)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        code = self._codegen(transformed)  # must not raise

        # Producer (task 0) captures its task output handle + producer TaskId.
        assert "TaskOutputTensors task_0_outs = rt_submit_task(mixed_0, params_t0);" in code, code
        m = re.search(r"PTO2TaskId (\w+) = task_0_outs\.task_id\(\);", code)
        assert m is not None, f"producer TaskId not captured\n{code}"
        tid = m.group(1)

        # Consumer (task 1) is a SECOND mixed dispatch — previously the crash site.
        assert "rt_submit_task(mixed_1, params_t1);" in code, code
        consumer_args = re.findall(r"params_t1\.add_\w+\((\w+)\);", code)
        assert len(consumer_args) == 4, f"{consumer_args}\n{code}"
        # Exactly one buffer is aliased on both lanes (the producer's tensor
        # output, reused as the consumer's in-place output).
        aliased = [n for n in set(consumer_args) if consumer_args.count(n) == 2]
        assert len(aliased) == 1, f"expected one buffer aliased twice, got {consumer_args}\n{code}"
        # And the deps edge is wired from the captured producer TaskId.
        # Fresh direct-producer TaskId (issue #1966): unguarded dep insert, no
        # redundant is_valid() branch.
        assert re.search(rf"if \({re.escape(tid)}\.is_valid\(\)\)", code) is None, code
        assert re.search(rf"\[[^\]]*\] = {re.escape(tid)};", code) is not None, code
        assert re.search(r"params_t1\.set_dependencies\(", code) is not None, code

    def test_allow_early_resolve_emits_set_allow_early_resolve(self):
        """``pl.spmd(..., allow_early_resolve=True) as tid`` emits the codegen hint.

        End-to-end proof that the parser-level scope attr threads through the Spmd
        outliner onto the ``ir.Submit`` and surfaces in orchestration codegen as
        ``Arg::set_allow_early_resolve(true)`` (same rail as ``pl.submit`` /
        ``pl.at``).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def vkernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(pl.add(t, t), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1", allow_early_resolve=True) as tid:
                    out = self.vkernel(a, out)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        code = self._codegen(transformed)
        assert re.search(r"\w+\.set_allow_early_resolve\(true\);", code) is not None, code

    def test_no_allow_early_resolve_omits_hint(self):
        """An ordinary captured Spmd dispatch never emits ``set_allow_early_resolve``."""
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def vkernel(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                t = pl.load(a, [0, 0], [512, 128])
                out = pl.store(pl.add(t, t), [0, 0], out)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                with pl.spmd(4, name_hint="stage1") as tid:
                    out = self.vkernel(a, out)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        code = self._codegen(transformed)
        assert "set_allow_early_resolve" not in code, code

    def test_spmd_dist_tensor_threads_comm_ctx(self):
        """A ``pl.spmd`` dispatch of an InCore kernel taking a ``DistributedTensor``
        must thread the per-tensor CommContext scalar into the task (issue #1913).

        MaterializeDistTensorCtx appends one explicit CommContext scalar arg per
        DistributedTensor formal; the L2 Spmd orchestration must forward the
        matching ``signal_ctx`` exactly like the InCore path, or the kernel reads
        a garbage CommContext and the cross-rank notify/wait deadlocks. Mirrors
        ``tests/st/distributed/test_l3_notify_wait.py`` but
        wraps the call in ``pl.spmd`` (the failing scope).
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def barrier_step(
                self,
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
                signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
                peer: pl.Scalar[pl.INT32],
                tag: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=tag, op=pld.NotifyOp.Set)
                pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)
                val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
                pl.write(out, [0, 0], val)
                return out

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
                signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
                peer: pl.Scalar[pl.INT32],
                tag: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[1, 1], pl.INT32]:
                with pl.spmd(2):
                    out = self.barrier_step(out, signal, peer, tag)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        # Plain vector InCore kernel -> AIV Spmd dispatch (GenerateSpmdCallCode's
        # non-Group branch), not a mixed cube+vector Group.
        code = self._codegen(transformed)
        assert "rt_submit_aiv_task" in code, code
        # The DistributedTensor ``signal`` threads its explicit CommContext scalar last.
        assert code.count("params_t0.add_scalar(signal_ctx);") == 1, code
        assert "ext_signal_ctx" not in code, code

    def test_mixed_spmd_dist_tensor_threads_comm_ctx_through_group_bridge(self):
        """A MIXED (``split=``) ``pl.spmd`` dispatch carrying a ``DistributedTensor``
        threads the CommContext scalar through the Spmd-wrapped-Group bridge too
        (issue #1913).

        The Spmd wrapper dispatches a cube+vector Group, so codegen routes through
        ``GenerateGroupCallCode``'s MixedKernels branch (via ``WrapperBridge``) —
        a different emit site than the plain-Spmd path above. Both must forward
        the explicit ``signal_ctx`` scalar for the DistributedTensor formal.
        """
        backend.reset_for_testing()
        backend.set_backend_type(BackendType.Ascend910B)

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                tile_a_l1 = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_b_l1 = pl.load(b, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
                tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
                tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
                tile_mm = pl.matmul(tile_a_l0a, tile_b_l0b)
                tile_out = pl.add(tile_mm, pl.load(bias, [0, 0], [64, 64]))
                pld.system.notify(target=signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)
                return pl.store(tile_out, [0, 0], out)

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                bias: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[1, 1], pl.INT32]],
                peer: pl.Scalar[pl.INT32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.spmd(4):
                    out = self.kernel(a, b, bias, out, signal, peer)
                return out

        transformed = self._mixed_spmd_pipeline(P)
        code = self._codegen(transformed)
        # Mixed cube+vector dispatch through the Group bridge ...
        assert "rt_submit_task(mixed_0, params_t0);" in code, code
        # ... still threads the DistributedTensor CommContext scalar.
        assert code.count("params_t0.add_scalar(signal_ctx);") == 1, code
        assert "ext_signal_ctx" not in code, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
