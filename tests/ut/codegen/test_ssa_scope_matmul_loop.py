# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Minimal repro: FlattenTileNdTo2D breaks SSA scope on for-loop with matmul_acc.

Reproduces the 'Variable used outside its defining scope' error seen in
paged_attention_64_example's kernel_pv_matmul_64 function.

Root cause: FlattenTileNdTo2D pass rebuilds Var objects for iter_args / init_values
but does not preserve pointer identity, causing the SSA verifier to treat them
as variables from a different scope.
"""

import pypto.language as pl
import pytest
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core import passes


def _run_passes_before_flatten(program):
    """Run all default pipeline passes up to (but not including) FlattenTileNdTo2D."""
    pass_sequence = [
        passes.unroll_loops,
        passes.ctrl_flow_transform,
        passes.convert_to_ssa,
        passes.flatten_call_expr,
        passes.split_chunked_loops,
        passes.interchange_chunk_loops,
        passes.outline_hierarchy_scopes,
        passes.outline_incore_scopes,
        passes.outline_cluster_scopes,
        passes.convert_tensor_to_tile_ops,
    ]
    for pass_fn in pass_sequence:
        program = pass_fn()(program)
    return program


def _build_matmul_loop_program():
    """Build minimal program: matmul before loop + matmul_acc inside loop."""

    @pl.program
    class MatmulLoop:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[16, 128], pl.FP32]],
            n_iters: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            # First iteration: matmul creates accumulator
            a_l1 = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
            b_l1 = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            a_l0 = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
            b_l0 = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(a_l0, b_l0)

            # Remaining iterations: matmul_acc in loop
            for i, (acc_iter,) in pl.range(1, n_iters, init_values=(acc,)):
                a_l1_i = pl.load(a, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
                b_l1_i = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                a_l0_i = pl.move(a_l1_i, target_memory=pl.MemorySpace.Left)
                b_l0_i = pl.move(b_l1_i, target_memory=pl.MemorySpace.Right)
                acc_new = pl.matmul_acc(acc_iter, a_l0_i, b_l0_i)
                (acc_out,) = pl.yield_(acc_new)

            result = pl.store(acc_out, [0, 0], out)
            return result

        @pl.function
        def main(
            self,
            a: pl.Tensor[[16, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            out = pl.create_tensor([16, 128], dtype=pl.FP32)
            result = self.kernel(a, b, out, 4)  # type: ignore[reportArgumentType]
            return result

    return MatmulLoop


def test_before_flatten_ssa_is_valid():
    """SSA is valid after ConvertTensorToTileOps (before FlattenTileNdTo2D)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    before = _run_passes_before_flatten(_build_matmul_loop_program())

    ssa_props = passes.IRPropertySet()
    ssa_props.insert(passes.IRProperty.SSAForm)
    # Should NOT raise — SSA is intact before FlattenTileNdTo2D
    passes.verify_properties(ssa_props, before, "BeforeFlatten")


def test_flatten_tile_nd_to_2d_breaks_ssa_scope():
    """FlattenTileNdTo2D breaks SSA scope for iter_arg variables in for-loops.

    Before (SSA valid):
        acc_0 = pl.tile.matmul(a_l0_0, b_l0_0)
        for i_0, (acc_iter,) in pl.range(1, n_iters_0, init_values=(acc_0,)):
            acc_new_0 = pl.tile.matmul_acc(acc_iter, a_l0_i_0, b_l0_i_0)
            acc_out = pl.yield_(acc_new_0)
        result_0 = pl.tile.store(acc_out, [0, 0], out_0)

    After FlattenTileNdTo2D (SSA broken):
        Same textual IR, but internal Var pointers for iter_args/init_values
        are recreated without preserving scope identity, so the verifier sees
        'acc_iter' and 'acc_0' as used outside their defining scope.
    """
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    before = _run_passes_before_flatten(_build_matmul_loop_program())

    # The pass's built-in post-verification catches the SSA scope violation
    with pytest.raises(ValueError, match="used outside its defining scope"):
        passes.flatten_tile_nd_to_2d()(before)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
