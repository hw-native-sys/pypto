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
class DistStartBoundLoop:
    """InCore kernel with a non-index (i32) start bound on pl.range()."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        for src in pl.range(my_rank, nranks):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        return out


def _generate_start_bound_mlir() -> str:
    func = DistStartBoundLoop.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_start_bound", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_non_index_start_bound_auto_cast_to_index():
    """Non-index start bound (pld.rank → i32) must produce arith.index_cast."""
    mlir_code = _generate_start_bound_mlir()

    assert "arith.index_cast" in mlir_code
    assert "scf.for" in mlir_code
    # Both start and stop should have index_cast if non-index.
    casts = re.findall(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    assert len(casts) >= 2, (
        f"expected ≥2 arith.index_cast for start+stop bounds; got {len(casts)}"
    )
    scf_for_match = re.search(r"scf\.for \S+ = (\S+) to (\S+) step (\S+)", mlir_code)
    assert scf_for_match is not None, "expected scf.for in generated MLIR"
    bounds = scf_for_match.groups()
    for cast_result in casts:
        assert cast_result in bounds, (
            f"arith.index_cast result {cast_result!r} must be a scf.for bound operand; got {bounds}"
        )


# ---------------------------------------------------------------------------
# scf.for bound auto-cast: non-index step bound
# ---------------------------------------------------------------------------


@pl.program
class DistStepBoundLoop:
    """InCore kernel with a non-index (i32) step bound on pl.range()."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        my_rank = pld.rank(ctx)
        nranks = pld.nranks(ctx)
        step_size: pl.Scalar[pl.INT32] = pl.const(1, pl.INT32)
        for src in pl.range(my_rank, nranks, step_size):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        return out


def _generate_step_bound_mlir() -> str:
    func = DistStepBoundLoop.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_step_bound", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_non_index_step_bound_auto_cast_to_index():
    """Non-index step bound (i32) must produce arith.index_cast for the step."""
    mlir_code = _generate_step_bound_mlir()

    assert "arith.index_cast" in mlir_code
    assert "scf.for" in mlir_code
    # All three bounds may have index_cast.
    casts = re.findall(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    assert len(casts) >= 3, (
        f"expected ≥3 arith.index_cast for start+stop+step bounds; got {len(casts)}"
    )
    scf_for_match = re.search(r"scf\.for \S+ = (\S+) to (\S+) step (\S+)", mlir_code)
    assert scf_for_match is not None, "expected scf.for in generated MLIR"
    bounds = scf_for_match.groups()
    for cast_result in casts:
        assert cast_result in bounds, (
            f"arith.index_cast result {cast_result!r} must be a scf.for bound operand; got {bounds}"
        )


# ---------------------------------------------------------------------------
# scf.for bound: index-typed constant → no unnecessary cast
# ---------------------------------------------------------------------------


@pl.program
class DistIndexConstLoop:
    """InCore kernel with index-typed start → codegen must NOT insert extra cast."""

    @pl.function(type=pl.FunctionType.InCore)
    def scan(
        self,
        signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        out: pl.Out[pl.Tensor[[1, 1], pl.INT32]],
    ) -> pl.Tensor[[1, 1], pl.INT32]:
        ctx = pld.get_comm_ctx(signal)
        nranks = pld.nranks(ctx)
        start: pl.Scalar[pl.INDEX] = pl.const(0, pl.INDEX)
        for src in pl.range(start, nranks):
            pld.system.wait(signal, offsets=[src, 0], expected=1, cmp=pld.WaitCmp.Ge)
        return out


def _generate_index_const_mlir() -> str:
    func = DistIndexConstLoop.get_function("scan")
    assert func is not None
    program = ir.Program([func], "test_index_const", ir.Span.unknown())
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = pm.run_passes(program)
    return codegen.PTOCodegen().generate(optimized)


def test_index_const_start_no_unnecessary_cast():
    """Index-typed start constant must NOT introduce a redundant arith.index_cast.

    The start is already index; only the stop (nranks → i32) needs a cast.
    """
    mlir_code = _generate_index_const_mlir()

    assert "arith.index_cast" in mlir_code, "stop bound should have index_cast"
    assert "scf.for" in mlir_code
    casts = re.findall(r"(%\S+) = arith\.index_cast %\S+ : i32 to index", mlir_code)
    # Only the stop bound should be cast; start is already index.
    assert len(casts) == 1, (
        f"expected exactly 1 arith.index_cast (stop only); got {len(casts)}: {casts}"
    )
