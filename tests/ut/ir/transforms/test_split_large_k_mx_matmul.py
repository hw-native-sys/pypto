# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for ExpandMxPackedQuant Phase-1 K-split of large-K MX matmul."""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes


@pytest.fixture(autouse=True)
def _reset_backend():
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


class TestExpandMxPackedQuantKSplit:
    """Phase-1 K-split rewrites static K>64 MX matmul into K=64 chunks."""

    @staticmethod
    def _run(program):
        return passes.expand_mx_packed_quant()(program)

    @staticmethod
    def _collect_calls(program, op_name: str):
        found = []
        registered_op_name = ir.get_op(op_name).name

        class _Collect(ir.IRVisitor):
            def visit_call(self, op):
                if op.op.name == registered_op_name:
                    found.append(op)
                super().visit_call(op)

        _Collect().visit_program(program)
        return found

    @staticmethod
    def _static_k(call) -> int:
        # matmul_mx: args[0]=lhs [M,K]; matmul_mx_acc: args[1]=lhs
        lhs = call.args[0] if call.op.name == ir.get_op("tile.matmul_mx").name else call.args[1]
        k_dim = lhs.type.shape[1]
        assert isinstance(k_dim, ir.ConstInt)
        return k_dim.value

    def test_splits_matmul_mx_k128_into_base_and_acc(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 128], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 4], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                ta = pl.load(a, [0, 0], [16, 128], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [16, 4], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [128, 32], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [4, 32], target_memory=pl.Mem.Mat)
                c = pl.matmul_mx(ta, tas, tb, tbs)
                return pl.store(c, [0, 0], out)

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 128], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 4], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[128, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[4, 32], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, a_s, b, b_s, out)

        after = self._run(Before)
        mx = self._collect_calls(after, "tile.matmul_mx")
        mx_acc = self._collect_calls(after, "tile.matmul_mx_acc")
        slices = self._collect_calls(after, "tile.slice")

        assert len(mx) == 1
        assert len(mx_acc) == 1
        assert self._static_k(mx[0]) == 64
        assert self._static_k(mx_acc[0]) == 64
        # 2 chunks × (lhs, lhs_scale, rhs, rhs_scale)
        assert len(slices) == 8

        # Idempotent: second run must not invent more chunks.
        twice = passes.expand_mx_packed_quant()(after)
        assert len(self._collect_calls(twice, "tile.matmul_mx")) == 1
        assert len(self._collect_calls(twice, "tile.matmul_mx_acc")) == 1

    def test_skips_k64(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                ta = pl.load(a, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
                tas = pl.load(a_s, [0, 0], [16, 2], target_memory=pl.Mem.Mat)
                tb = pl.load(b, [0, 0], [64, 32], target_memory=pl.Mem.Mat)
                tbs = pl.load(b_s, [0, 0], [2, 32], target_memory=pl.Mem.Mat)
                c = pl.matmul_mx(ta, tas, tb, tbs)
                return pl.store(c, [0, 0], out)

            @pl.function
            def main(
                self,
                a: pl.Tensor[[16, 64], pl.FP8E4M3FN],
                a_s: pl.Tensor[[16, 2], pl.FP8E8M0, pl.MX_A_ZZ],
                b: pl.Tensor[[64, 32], pl.FP8E4M3FN],
                b_s: pl.Tensor[[2, 32], pl.FP8E8M0, pl.MX_B_NN],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                out = pl.create_tensor([16, 32], dtype=pl.FP32)
                return self.kernel(a, a_s, b, b_s, out)

        after = self._run(Before)
        assert len(self._collect_calls(after, "tile.matmul_mx")) == 1
        assert len(self._collect_calls(after, "tile.matmul_mx_acc")) == 0
        assert len(self._collect_calls(after, "tile.slice")) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
