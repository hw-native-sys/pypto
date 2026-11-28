# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A5 explicit matrix and FIX hardware buffer token codegen."""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def _generate_mlir(program):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend950)
    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    functions = [f for f in lowered.functions.values() if ir.is_incore_type(f.func_type)]
    return codegen.PTOCodegen().generate(ir.Program(functions, "Kernel", lowered.span))


def test_buffer_tokens_emit_exact_matrix_and_fix_calls(tmp_path):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.AIC)
        def kernel(self, slot: pl.Scalar[pl.INDEX]):
            pl.system.get_buf(14, pipe=pl.PipeType.M)
            pl.system.rls_buf(14, pipe=pl.PipeType.M)
            pl.system.get_buf(slot + 14, pipe=pl.PipeType.FIX)
            pl.system.rls_buf(slot + 14, pipe=pl.PipeType.FIX)

    mlir = _generate_mlir(Prog)
    assert 'pto.get_buf "TMATMUL", 14, 0' in mlir
    assert 'pto.rls_buf "TMATMUL", 14, 0' in mlir
    assert 'pto.get_buf_dyn "TSTORE_ACC", %' in mlir
    assert 'pto.rls_buf_dyn "TSTORE_ACC", %' in mlir

    if find_ptoas_binary() is not None:
        cpp = pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path))
        assert "get_buf(PIPE_M, 14, 0)" in cpp
        assert "rls_buf(PIPE_M, 14, 0)" in cpp
        assert "get_buf(PIPE_FIX," in cpp
        assert "rls_buf(PIPE_FIX," in cpp


@pytest.mark.parametrize("op", [pl.system.get_buf, pl.system.rls_buf])
def test_buffer_tokens_reject_invalid_static_id_and_pipe(op):
    with pytest.raises(ValueError, match="buffer ID must be in"):
        op(32, pipe=pl.PipeType.M)
    with pytest.raises(ValueError, match="only PIPE_M and PIPE_FIX"):
        op(14, pipe=pl.PipeType.V)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
