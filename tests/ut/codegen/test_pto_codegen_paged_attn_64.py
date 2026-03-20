# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for PTO backend codegen for 64-tier paged attention operations.

Tests that the 64-tier paged attention program (build_paged_attention_64_program)
compiles through the full pass pipeline and generates non-empty MLIR for each function.
"""

import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

from examples.ir_parser.paged_attention_64_example import build_paged_attention_64_program

# Small parameters for fast UT execution
_BATCH = 1
_NUM_HEADS = 16
_HEAD_DIM = 128
_BLOCK_SIZE = 128
_MAX_NUM_BLOCKS_PER_REQ = 1
_CONTEXT_LEN = 128


def test_paged_attention_64_codegen():
    """Verify 64-tier paged attention compiles and generates MLIR for all functions."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    program = build_paged_attention_64_program(
        batch=_BATCH,
        num_heads=_NUM_HEADS,
        head_dim=_HEAD_DIM,
        block_size=_BLOCK_SIZE,
        max_num_blocks_per_req=_MAX_NUM_BLOCKS_PER_REQ,
        context_len=_CONTEXT_LEN,
    )

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    # ConvertTensorToTileOps pass produces SSA scope violations and the DSL
    # generates single-child SeqStmts — both are infrastructure-level issues,
    # not example bugs. Disable post-pass verification to unblock codegen testing.
    # ctx = passes.PassContext([], passes.VerificationLevel.NONE)
    # with ctx:
    #     optimized_program = pm.run_passes(program)
    # codegen_instance = codegen.PTOCodegen()
    optimized_program = pm.run_passes(program)
    codegen_instance = codegen.PTOCodegen()

    for func in optimized_program.functions.values():
        if not ir.is_incore_type(func.func_type):
            continue
        func_name = func.name
        single_func_program = ir.Program([func], func_name, optimized_program.span)
        mlir_code = codegen_instance.generate(single_func_program)
        assert mlir_code, f"Generated MLIR code for {func_name} should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
