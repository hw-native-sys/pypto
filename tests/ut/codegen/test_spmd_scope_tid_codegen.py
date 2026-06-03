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

import pypto.language as pl
import pytest
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.pypto_core import ir


class TestSpmdScopeTaskIdCodegen:
    """Codegen for the captured ``with pl.spmd(...) as tid:`` SPMD dispatch."""

    @staticmethod
    def _mixed_spmd_pipeline(program):
        with passes.PassContext([], passes.VerificationLevel.NONE):
            return passes.expand_mixed_kernel()(
                passes.infer_tile_memory_space()(
                    passes.outline_cluster_scopes()(passes.convert_to_ssa()(program))
                )
            )

    @staticmethod
    def _codegen(program):
        """Run DeriveCallDirections + MaterializeRuntimeScopes + orchestration codegen.

        Pinned to VerificationLevel.NONE: an outlined auto-scope ``as tid`` dispatch
        is lowered Submit -> Call (with a TASK_ID-augmented Tuple return) by
        DeriveCallDirections, and that Call form does not survive a print -> reparse
        roundtrip (the printed Tuple-annotated plain call reparses to the callee's
        scalar return). That is a pre-existing limitation shared by the
        ``pl.at(...) as tid:`` rail and is orthogonal to this test, which exercises
        the codegen output itself — the contract under test.
        """
        with passes.PassContext([], passes.VerificationLevel.NONE):
            program = passes.derive_call_directions()(program)
            program = passes.materialize_runtime_scopes()(program)
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
        # The first dispatch's producer TaskId is captured ...
        assert "task_0_outs.task_id()" in code, code
        # ... and threaded as the second dispatch's explicit dependency.
        assert "set_dependencies(" in code, code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
