# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen: dynamic shape vars on DistributedTensor parameters."""

# pyright: reportUndefinedVariable=false

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes as _core_passes

NR = pl.dynamic("NR")


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _generate_mlir(func_name: str = "touch") -> str:
    func = DistDynSignal.get_function(func_name)
    assert func is not None
    program = ir.Program([func], "test_dist_dyn_signal", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    # DistributedTensor[[NR, …]] round-trips without a module-level NR decl today;
    # skip RoundtripInstrument (see test_orchestration_codegen tuple NoDep pattern).
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


@pl.program
class DistDynSignal:
    """InCore kernel with rank-count dynamic dim on a DistributedTensor."""

    @pl.function(type=pl.FunctionType.InCore)
    def touch(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        out: pl.Tensor[[1, 1], pl.INT32],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        pld.system.wait(signal, offsets=[0, 0], expected=0, cmp=pld.WaitCmp.Ge)
        val: pl.Scalar[pl.INT32] = pl.read(signal, [0, 0])
        pl.write(out, [0, 0], val)
        return out


def test_distributed_tensor_dynamic_nr_pto_codegen():
    """NR on DistributedTensor[[NR, 1]] must appear as a trailing index param."""
    mlir_code = _generate_mlir()

    # Locate the trailing index param in the touch function signature without
    # hardcoding its ordinal — arg position can shift if param layout changes.
    func_sig_match = re.search(r"func\.func @touch[^{]+?(%arg\d+): index", mlir_code, re.DOTALL)
    assert func_sig_match is not None, "touch must declare a trailing index param for NR"
    nr_ssa = func_sig_match.group(1)

    # The signal tensor view must use that param as its first (dynamic) dim.
    # This positive assertion already rules out a zeroed/missing NR.
    assert f"shape = [{nr_ssa}, %c1_index]" in mlir_code, (
        f"signal view shape must be [{nr_ssa}, %c1_index]; NR param was dropped"
    )


def test_collect_vars_from_shape_expr_finds_nr():
    """C++ shape walker sees NR inside DistributedTensor shape annotations."""
    func = DistDynSignal.get_function("touch")
    assert func is not None
    signal_param = func.params[0]
    assert isinstance(signal_param.type, ir.DistributedTensorType)
    nr_dim = signal_param.type.shape[0]
    dyn_vars = codegen.collect_vars_from_shape_expr(nr_dim)
    assert len(dyn_vars) == 1
    assert dyn_vars[0].name_hint == "NR"


# ---------------------------------------------------------------------------
# scf.for bound auto-cast: pld.nranks(ctx) used as pl.range() stop
# ---------------------------------------------------------------------------


@pl.program
class DistNranksLoop:
    """InCore kernel that uses pld.nranks(ctx) as a pl.range() bound."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        nranks = pld.nranks(ctx)
        # pld.nranks returns INT32 (i32 in MLIR); scf.for requires index.
        # Codegen must insert arith.index_cast automatically — no pl.cast here.
        for src in pl.range(nranks):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        return out


def _generate_range_mlir() -> str:
    func = DistNranksLoop.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_nranks_range", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_nranks_range_bound_auto_cast_to_index():
    """pld.nranks(ctx) used as pl.range() bound must produce arith.index_cast.

    Regression guard: before the VisitStmt_(ForStmtPtr) fix, scf.for would
    receive an i32 stop value and fail MLIR verification in ptoas.  After the
    fix, codegen auto-inserts arith.index_cast so users can write
    ``pl.range(pld.nranks(ctx))`` without a manual ``pl.cast(..., pl.INDEX)``.
    """
    mlir_code = _generate_range_mlir()

    # Codegen must have emitted arith.index_cast on the i32 nranks value.
    assert "arith.index_cast" in mlir_code
    # scf.for must be present (the range lowered to a loop, not constant-folded).
    assert "scf.for" in mlir_code
    # Verify the arith.index_cast result is used as a scf.for bound operand.
    # This is tighter than checking "i32" is absent from the scf.for line, which
    # would false-positive on iter_args type annotations such as "-> (i32)".
    cast_match = re.search(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    assert cast_match is not None, "expected arith.index_cast i32 -> index in generated MLIR"
    index_result = cast_match.group(1)
    scf_for_match = re.search(r"scf\.for \S+ = (\S+) to (\S+) step (\S+)", mlir_code)
    assert scf_for_match is not None, "expected scf.for in generated MLIR"
    bounds = scf_for_match.groups()
    assert index_result in bounds, (
        f"arith.index_cast result {index_result!r} must be a scf.for bound operand; got {bounds}"
    )


# ---------------------------------------------------------------------------
# scf.for bound auto-cast: non-index start bound
# ---------------------------------------------------------------------------


@pl.program
class DistNranksStartBound:
    """InCore kernel with a non-index (INT32) start bound on pl.range()."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        start_offset: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        nranks = pld.nranks(ctx)
        # start_offset is INT32; codegen must insert arith.index_cast for the
        # lower bound of scf.for — same treatment as the stop bound.
        for src in pl.range(start_offset, nranks):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        out: pl.Tensor[[1, 1], pl.INT32] = pl.create_tensor([1, 1], dtype=pl.INT32)
        return out


def _generate_start_bound_mlir() -> str:
    func = DistNranksStartBound.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_nranks_start_bound", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_nranks_start_bound_auto_cast_to_index():
    """Non-index start bound on pl.range() must produce arith.index_cast.

    Regression guard: before the VisitStmt_(ForStmtPtr) fix, scf.for would
    receive an i32 lower bound and fail MLIR verification in ptoas.  After the
    fix, codegen auto-inserts arith.index_cast for start just as it does for stop.
    """
    mlir_code = _generate_start_bound_mlir()

    assert "arith.index_cast" in mlir_code
    assert "scf.for" in mlir_code

    cast_results = re.findall(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    assert len(cast_results) >= 2, f"expected >=2 arith.index_cast (start + stop), got {len(cast_results)}"

    scf_for_match = re.search(r"scf\.for \S+ = (\S+) to (\S+) step (\S+)", mlir_code)
    assert scf_for_match is not None, "expected scf.for in generated MLIR"
    lower, upper, _step = scf_for_match.groups()
    assert lower in cast_results, (
        f"start bound cast result must be scf.for lower bound; lower={lower!r}, casts={cast_results}"
    )
    assert upper in cast_results, (
        f"stop bound cast result must be scf.for upper bound; upper={upper!r}, casts={cast_results}"
    )


# ---------------------------------------------------------------------------
# scf.for bound auto-cast: non-index step bound
# ---------------------------------------------------------------------------


@pl.program
class DistNranksStepBound:
    """InCore kernel with a non-index (INT32) step bound on pl.range()."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        stride: pl.Scalar[pl.INT32],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        nranks = pld.nranks(ctx)
        # stride is INT32; codegen must insert arith.index_cast for the step
        # of scf.for — same treatment as start/stop bounds.
        for src in pl.range(0, nranks, stride):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        out: pl.Tensor[[1, 1], pl.INT32] = pl.create_tensor([1, 1], dtype=pl.INT32)
        return out


def _generate_step_bound_mlir() -> str:
    func = DistNranksStepBound.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_nranks_step_bound", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_nranks_step_bound_auto_cast_to_index():
    """Non-index step bound on pl.range() must produce arith.index_cast.

    Regression guard: before the VisitStmt_(ForStmtPtr) fix, scf.for would
    receive an i32 step and fail MLIR verification in ptoas.  After the fix,
    codegen auto-inserts arith.index_cast for step just as it does for start/stop.
    """
    mlir_code = _generate_step_bound_mlir()

    assert "arith.index_cast" in mlir_code
    assert "scf.for" in mlir_code

    cast_results = re.findall(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    assert len(cast_results) >= 2, f"expected >=2 arith.index_cast (stop + step), got {len(cast_results)}"

    scf_for_match = re.search(r"scf\.for \S+ = (\S+) to (\S+) step (\S+)", mlir_code)
    assert scf_for_match is not None, "expected scf.for in generated MLIR"
    _lower, upper, step = scf_for_match.groups()
    assert upper in cast_results, (
        f"stop bound cast result must be scf.for upper bound; upper={upper!r}, casts={cast_results}"
    )
    assert step in cast_results, (
        f"step bound cast result must be scf.for step; step={step!r}, casts={cast_results}"
    )


# ---------------------------------------------------------------------------
# scf.for bound auto-cast: index-typed constants (no unnecessary cast)
# ---------------------------------------------------------------------------


@pl.program
class DistIndexConstants:
    """InCore kernel with index-typed start constant — must not produce spurious casts."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        nranks = pld.nranks(ctx)
        # start is already index-typed; EmitCastToIndex must be a no-op here.
        start: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX)
        for src in pl.range(start, nranks):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        out: pl.Tensor[[1, 1], pl.INT32] = pl.create_tensor([1, 1], dtype=pl.INT32)
        return out


def _generate_index_constants_mlir() -> str:
    func = DistIndexConstants.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_index_constants", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_index_constants_no_unnecessary_cast():
    """Index-typed start/stop/step must not produce arith.index_cast on those operands.

    Regression guard: if EmitCastToIndex does not check ``dtype == INDEX``
    before emitting, it would insert a redundant index_cast on values that are
    already index-typed.  Only the stop bound (pld.nranks, i32) should trigger
    a cast.
    """
    mlir_code = _generate_index_constants_mlir()

    assert "arith.index_cast" in mlir_code, "stop bound (i32 nranks) must still be cast"
    assert "scf.for" in mlir_code

    # At least one arith.index_cast must exist (stop bound).  We don't assert an
    # exact count — future codegen changes may add unrelated casts in the same
    # function — but we verify that the scf.for upper bound is the correct cast.
    cast_count = mlir_code.count("arith.index_cast")
    assert cast_count >= 1, f"expected at least 1 arith.index_cast (stop bound), got {cast_count}"

    cast_match = re.search(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    assert cast_match is not None, "expected arith.index_cast i32 -> index for nranks stop bound"
    index_result = cast_match.group(1)

    scf_for_match = re.search(r"scf\.for \S+ = (\S+) to (\S+) step (\S+)", mlir_code)
    assert scf_for_match is not None, "expected scf.for in generated MLIR"
    lower, upper, _step = scf_for_match.groups()
    assert upper == index_result, (
        f"cast result {index_result!r} must be scf.for upper bound; got upper={upper!r}"
    )
    # Lower bound must NOT be the cast result (it's already index-typed).
    assert lower != index_result, f"lower bound {lower!r} must not be the cast result (already index-typed)"
