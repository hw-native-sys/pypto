# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The normal pass pipeline keeps TIMG2COL as a direct L1-to-L0A operation."""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager


def test_img2col_matmul_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64, 32], pl.FP16],
                w: pl.Tensor[[32, 32], pl.FP16],
                out: pl.Out[pl.Tensor[[16, 32], pl.FP32]],
            ) -> pl.Tensor[[16, 32], pl.FP32]:
                src = pl.load(x, [0, 0], [64, 32], target_memory=pl.MemorySpace.Mat)
                weight = pl.load(w, [0, 0], [32, 32], target_memory=pl.MemorySpace.Mat)
                lhs = pl.img2col(
                    src, 16, 32, [16, 32], image_shape=(8, 8), kernel_size=(3, 3), padding=(1, 1, 1, 1)
                )
                rhs = pl.tile.extract(weight, 0, 0, [32, 32], target_memory=pl.MemorySpace.Right)
                result = pl.tile.matmul(lhs, rhs)
                return pl.store(result, [0, 0], out)

        program = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
        generated = codegen.PTOCodegen().generate(program)
        mlir = generated if isinstance(generated, str) else "".join(generated.values())
        assert mlir.count("pto.timg2col ins(") == 1
        assert "pto.tmatmul" in mlir
        assert mlir.count("pto.textract ins(") == 1
        for attr in ("fmap_h = 8", "kernel_h = 3", "pad_top = 1", "stride_h = 1", "dilation_w = 1"):
            assert attr in mlir
    finally:
        backend.reset_for_testing()


@pytest.mark.parametrize("source", ["x", "pl.slice(x, [64, 32], [0, 0])", "pl.add(x, 1.0)"])
def test_tensor_img2col_matmul_codegen(source):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    try:
        program = pl.parse_program(f"""
@pl.program
class Program:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 32], pl.FP16], w: pl.Tensor[[32, 32], pl.FP16],
               out: pl.Out[pl.Tensor[[16, 32], pl.FP32]]) -> pl.Tensor[[16, 32], pl.FP32]:
        src = {source}
        lhs = pl.img2col(src, 16, 32, [16, 32], image_shape=(8, 8), kernel_size=(3, 3), padding=(1, 1, 1, 1))
        result = pl.matmul(lhs, w)
        return pl.assemble(out, result, [0, 0])
""")
        ir.assert_structural_equal(pl.parse_program(ir.python_print(program)), program)
        lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        kernels = ir.Program(
            [func for func in lowered.functions.values() if ir.is_incore_type(func.func_type)],
            lowered.name,
            lowered.span,
        )
        generated = codegen.PTOCodegen().generate(kernels)
        mlir = generated if isinstance(generated, str) else "".join(generated.values())
        assert mlir.count("pto.timg2col ins(") == 1
        assert "pto.tmatmul" in mlir
        assert mlir.count("pto.tload ins(") == 2
        assert "kernel_h = 3" in mlir
        assert "pad_top = 1" in mlir
    finally:
        backend.reset_for_testing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
