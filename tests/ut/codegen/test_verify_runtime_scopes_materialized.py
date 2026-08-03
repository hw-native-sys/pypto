# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Negative and positive tests for the RuntimeScopesMaterialized property verifier."""

import pypto
import pypto.language as pl
import pytest
from pypto import backend, codegen, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


@pl.program
class _OrchWithSubmit:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y, _ = pl.submit(self.kernel, x)
        return y


def _program_without_materialized_scopes():
    """Stop after DeriveCallDirections — omit MaterializeRuntimeScopes."""
    return passes.derive_call_directions()(passes.convert_to_ssa()(_OrchWithSubmit))


def _program_with_materialized_scopes():
    return passes.classify_iter_arg_carry()(
        passes.materialize_runtime_scopes()(
            passes.derive_call_directions()(passes.convert_to_ssa()(_OrchWithSubmit))
        )
    )


def _orch_func(program):
    for func in program.functions.values():
        if func.func_type == pl.FunctionType.Orchestration:
            return func
    pytest.fail("No orchestration function found in program")
    raise AssertionError  # unreachable


def test_runtime_scopes_materialized_registry_rejects_unmaterialized_orchestration():
    program = _program_without_materialized_scopes()
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.RuntimeScopesMaterialized)
    with pytest.raises(pypto.Error, match=r"RuntimeScopesMaterialized|auto_scope"):
        passes.PropertyVerifierRegistry.verify_or_throw(props, program)


def test_runtime_scopes_materialized_codegen_precondition_rejects_unmaterialized():
    program = _program_without_materialized_scopes()
    with pytest.raises(pypto.Error, match=r"RuntimeScopesMaterialized|auto_scope|Verification failed"):
        codegen.generate_orchestration(program, _orch_func(program))


def test_runtime_scopes_materialized_rejects_stale_function_handle():
    program = passes.derive_call_directions()(passes.convert_to_ssa()(_OrchWithSubmit))
    stale_func = _orch_func(program)
    program = passes.classify_iter_arg_carry()(passes.materialize_runtime_scopes()(program))
    with pytest.raises(pypto.Error, match=r"stale function handle|stale handle"):
        codegen.generate_orchestration(program, stale_func)


@pl.program
class _OtherOrchProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def other_orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y, _ = pl.submit(self.kernel, x)
        return y


def test_generate_orchestration_rejects_function_not_in_program():
    """func must belong to the supplied program, not merely share a finalized shape."""
    program = _program_with_materialized_scopes()
    other_program = passes.classify_iter_arg_carry()(
        passes.materialize_runtime_scopes()(
            passes.derive_call_directions()(passes.convert_to_ssa()(_OtherOrchProgram))
        )
    )
    foreign_func = _orch_func(other_program)
    with pytest.raises(ValueError, match="not present in the supplied program"):
        codegen.generate_orchestration(program, foreign_func)


def test_runtime_scopes_materialized_registry_accepts_materialized_orchestration():
    program = _program_with_materialized_scopes()
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.RuntimeScopesMaterialized)
    passes.PropertyVerifierRegistry.verify_or_throw(props, program)


def test_runtime_scopes_materialized_in_get_verified_properties():
    props = passes.get_verified_properties()
    assert props.contains(passes.IRProperty.RuntimeScopesMaterialized)


@pl.program
class _UserOptOutScopes:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], out)
        return r

    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def orch(self, a: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
        with pl.scope():
            out = self.kernel(a, out)
        return out


def test_runtime_scopes_materialized_user_opt_out_without_pass():
    """User auto_scope=False is valid without running MaterializeRuntimeScopes."""
    program = passes.derive_call_directions()(passes.convert_to_ssa()(_UserOptOutScopes))
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.RuntimeScopesMaterialized)
    passes.PropertyVerifierRegistry.verify_or_throw(props, program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
