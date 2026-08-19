# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto
import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, codegen, passes
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


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
    with pytest.raises(pypto.InternalError, match="DistributedCodegen preconditions"):
        cg.generate(program)


def test_device_kernel_rejects_descending_loop():
    """A descending loop in a device function must fail loudly, not silently.

    Regression: device code lowers to MLIR ``scf.for``, which counts upward
    only. A ``pl.range(8, 0, -1)`` was transcribed verbatim into
    ``scf.for %i = 8 to 0 step -1`` — a zero-trip loop that ptoas folded away,
    dropping the entire loop body with no diagnostic from any stage. The kernel
    loaded its input, computed nothing, and stored.

    The PTOAS team has confirmed they will not support a non-positive
    ``scf.for`` step in the foreseeable future — hw-native-sys/PTOAS#1288 will
    be closed by adding the missing assertion only. So this rejection is
    permanent, and the test stays: even once that assertion ships, the PyPTO
    check fires earlier and names the user's loop instead of generated ``.pto``.
    """
    n = 64

    @pl.program
    class Descending:
        @pl.function(type=pl.FunctionType.AIV)
        def kern(
            self,
            x: pl.Tensor[[n, n], pl.FP32],
            out: pl.Out[pl.Tensor[[n, n], pl.FP32]],
        ) -> pl.Tensor[[n, n], pl.FP32]:
            acc: pl.Tile[[n, n], pl.FP32] = pl.load(x, [0, 0], [n, n])
            for _i, (acc,) in pl.range(8, 0, -1, init_values=(acc,)):
                acc = pl.add(acc, acc)
                acc = pl.yield_(acc)
            return pl.store(acc, [0, 0], out)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Descending)

    with pytest.raises(ValueError, match="must have a positive step"):
        codegen.PTOCodegen().generate(optimized)


def test_device_kernel_accepts_ascending_loop():
    """The ascending counterpart of the descending-loop rejection still lowers.

    Guards the rejection above against over-reach: only a non-positive constant
    step is refused.
    """
    n = 64

    @pl.program
    class Ascending:
        @pl.function(type=pl.FunctionType.AIV)
        def kern(
            self,
            x: pl.Tensor[[n, n], pl.FP32],
            out: pl.Out[pl.Tensor[[n, n], pl.FP32]],
        ) -> pl.Tensor[[n, n], pl.FP32]:
            acc: pl.Tile[[n, n], pl.FP32] = pl.load(x, [0, 0], [n, n])
            for _i, (acc,) in pl.range(0, 8, 1, init_values=(acc,)):
                acc = pl.add(acc, acc)
                acc = pl.yield_(acc)
            return pl.store(acc, [0, 0], out)

    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Ascending)

    mlir = codegen.PTOCodegen().generate(optimized)
    assert "scf.for" in mlir, mlir


def test_orchestration_codegen_precondition_entry_point():
    """Verify that GenerateOrchestration calls the precondition barrier.

    The precondition must be the first thing that runs in
    ``codegen.generate_orchestration`` — before any IR traversal or emission.
    This smoke test confirms the entry point is wired: a program that satisfies
    all IR-property preconditions (including RuntimeScopesMaterialized) reaches
    codegen and fails later on a missing pass (ExpandMixedKernel) with a
    distinct error — proving the precondition barrier did not spuriously block.

    Note: triggering every IR-property check from Python tests is still
    partially limited because SplitIncoreOrch is included in convert_to_ssa and
    OrchestrationReferencesResolved is enforced by the DSL parser.
    RuntimeScopesMaterialized is registered — see
    test_verify_runtime_scopes_materialized.py for targeted failure tests.
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

    program = passes.classify_iter_arg_carry()(
        passes.materialize_runtime_scopes()(passes.derive_call_directions()(passes.convert_to_ssa()(Input)))
    )
    for func in program.functions.values():
        if func.func_type == pl.FunctionType.Orchestration:
            # All codegen preconditions pass (including RuntimeScopesMaterialized).
            # Codegen proceeds and fails at InferFunctionCoreType because
            # ExpandMixedKernel was not run — proving the precondition did not
            # block execution.
            with pytest.raises(pypto.InternalError, match="InferFunctionCoreType"):
                codegen.generate_orchestration(program, func)
            return
    pytest.fail("No orchestration function found in program")


def _finalize(program):
    """Run the codegen-entry passes, deliberately omitting NormalizeReturnOrder."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    return passes.classify_iter_arg_carry()(
        passes.materialize_runtime_scopes()(passes.derive_call_directions()(program))
    )


def _orch_func(program):
    for func in program.functions.values():
        if func.func_type == pl.FunctionType.Orchestration:
            return func
    pytest.fail("No orchestration function found in program")
    raise AssertionError  # unreachable, satisfies type checkers


@pl.program
class _MultiOutProgram:
    """A kernel with two Out params that returns only one of them.

    Aliasing the orchestration result to the wrong Out param would silently
    route every downstream consumer into the scratch buffer (#1702/#1573).
    """

    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        scratch: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        t: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        s: pl.Tensor[[16, 16], pl.FP32] = pl.store(t, [0, 0], scratch)
        t2: pl.Tile[[16, 16], pl.FP32] = pl.load(s, [0, 0], [16, 16])
        r: pl.Tensor[[16, 16], pl.FP32] = pl.store(t2, [0, 0], out)
        return r

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        d: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        sc: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
        d = self.kernel(a, sc, d)
        return d


def test_orchestration_codegen_requires_return_params_explicit_for_multi_out_callee():
    """Codegen reads the return->param map off the ReturnStmt, so it needs the property.

    Without NormalizeReturnOrder the kernel returns an SSA alias of `out` rather
    than `out` itself. Codegen must refuse to guess which of the two Out params
    the result aliases.
    """
    program = _finalize(_MultiOutProgram)
    with pytest.raises(pypto.Error, match="ReturnParamsExplicit"):
        codegen.generate_orchestration(program, _orch_func(program))


def test_orchestration_codegen_aliases_the_returned_out_param_not_the_scratch():
    """With the property established, the result aliases `out`, never `scratch`."""
    program = _finalize(passes.normalize_return_order()(_MultiOutProgram))
    code = codegen.generate_orchestration(program, _orch_func(program)).code

    # The task's outputs are (scratch, out) in param order; `d` is the real
    # output and must be the one the orchestration result binds to.
    add_outputs = [line.strip() for line in code.splitlines() if "add_output" in line]
    assert add_outputs == ["params_t0.add_output(sc);", "params_t0.add_output(ext_d);"]
