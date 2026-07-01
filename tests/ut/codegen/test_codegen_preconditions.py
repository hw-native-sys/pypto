# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import codegen, passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Match distributed codegen tests: allow CommDomain materialization-only flows."""
    with passes.PassContext([passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
        yield


def test_distributed_codegen_requires_comm_domain_materialization_when_distributed_tensors_present():
    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            x: pl.Tensor[[64], pl.FP32],
            data: pld.DistributedTensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            x: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            buf = pld.alloc_window_buffer(64 * 4)
            data = pld.window(buf, [64], dtype=pl.FP32)
            return self.chip_orch(x, data, device=0)

    # Deliberately omit MaterializeCommDomainScopes.
    program = passes.convert_to_ssa()(Input)
    cg = codegen.DistributedCodegen()
    with pytest.raises(Exception, match="DistributedCodegen preconditions"):
        cg.generate(program)


def test_orchestration_codegen_precondition_entry_point_is_wired():
    """Verify that GenerateOrchestration calls the precondition barrier.

    The precondition must be the first thing that runs in
    ``codegen.generate_orchestration`` — before any IR traversal or emission.
    This smoke test confirms the entry point is wired: a valid program reaches
    the precondition (which passes for well-formed IR from convert_to_ssa)
    and proceeds to codegen, which fails later on a missing pass
    (ExpandMixedKernel) with a distinct error — proving the precondition did
    NOT block and the codegen pipeline advanced past it.

    Note: triggering the IR-property checks (SplitIncoreOrch,
    OrchestrationReferencesResolved, RuntimeScopesMaterialized) from Python
    tests is currently limited because (a) SplitIncoreOrch is included in
    convert_to_ssa, (b) OrchestrationReferencesResolved is enforced by the
    DSL parser, and (c) RuntimeScopesMaterialized lacks a registered verifier.
    Follow-up work should register a verifier for RuntimeScopesMaterialized
    and add a targeted failure test.
    """

    @pl.program
    class Input:
        @pl.function(type=pl.FunctionType.InCore)
        def k(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return x

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            y, _ = pl.submit(self.k, x)
            return y

    program = passes.convert_to_ssa()(Input)
    for func in program.functions.values():
        if func.func_type == pl.FunctionType.Orchestration:
            # The precondition passes (properties are satisfied by
            # convert_to_ssa). Codegen proceeds and fails at
            # InferFunctionCoreType because ExpandMixedKernel was not run —
            # proving the precondition did not block execution.
            with pytest.raises(Exception, match="InferFunctionCoreType"):
                codegen.generate_orchestration(program, func)
            return
    pytest.fail("No orchestration function found in program")
