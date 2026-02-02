# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from pypto import DataType, ir
from pypto.ir import IRBuilder
from pypto.ir.op import block
from pypto.ir.pass_manager import PassManager
from pypto.pypto_core import codegen


def build_matmul():
    """Build a matmul function using IRBuilder."""
    ib = IRBuilder()
    span = ir.Span.unknown()

    with ib.function("matmul") as f:
        a = f.param("a", ir.TensorType([64, 64], DataType.FP16))
        b = f.param("b", ir.TensorType([64, 64], DataType.FP16))
        c = f.param("c", ir.TensorType([64, 64], DataType.FP32))
        f.return_type(ir.ScalarType(DataType.INT32))

        tile_a = ib.let("tile_a", block.load(a, 0, 0, 64, 64, 2))
        tile_b = ib.let("tile_b", block.load(b, 0, 0, 64, 64, 2))
        tile_a_l0a = ib.let("tile_a_l0a", block.move(tile_a, target_memory=3))
        tile_b_l0b = ib.let("tile_b_l0b", block.move(tile_b, target_memory=4))
        tile_c_l0c = ib.let("tile_c_l0c", block.matmul(tile_a_l0a, tile_b_l0b))
        tile_c = ib.let("tile_c", block.move(tile_c_l0c, target_memory=2))
        ib.let("result", block.store(tile_c, 0, 0, 64, 64, c))
        ib.return_stmt(ir.ConstInt(0, DataType.INT32, span))

    return f.get_result()


if __name__ == "__main__":
    func = build_matmul()
    program = ir.Program([func], "matmul", ir.Span.unknown())
    pm = PassManager.get_strategy()
    optimized_program = pm.run_passes(program)
    optimized_func = list(optimized_program.functions.values())[0]
    print(optimized_func)
    generator = codegen.CceCodegen()
    code = generator.Generate(optimized_func)
    print(code)
