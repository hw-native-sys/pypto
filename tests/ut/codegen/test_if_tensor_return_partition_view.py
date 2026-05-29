# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Regression test for issues #1533 and #1569.

A scratch tensor that is written (``pl.store`` / ``pl.assemble``) inside an
``if`` branch becomes an SCF-if phi after control-flow lowering.

The bug: PTO codegen used to route such TensorType IfStmt return_vars through
``scf.if`` *results*, retyping them to a fully-dynamic ``!pto.tensor_view<?x?>``
and dropping the concrete memref dims. A later ``pto.partition_view`` on that
phi tensor is then rejected by ptoas:

    'pto.partition_view' op operand #0 must be , but got
    'memref<?x?xf32, strided<[?, ?], offset: ?>>'

The fix treats tensors like the other in-place mutable references (ArrayType /
TileType, and loop-carried tensors in ForStmt): they are kept OUT of the
``scf.if`` results and bound to the shared backing tensor_view both branches
mutate in place. The concrete ``make_tensor_view`` then flows straight into
``partition_view``.

``IfStmt`` codegen is shared across control structures, so the same fix covers
the ``pl.spmd`` variant from #1569: a conditional write to an outer tensor
inside a ``pl.spmd`` body is outlined into a per-block kernel whose ``if``
produces the same TensorType phi. ``SpmdIfTensorReturnScratch`` below guards
that outlined path.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level names.
# pyright: reportUndefinedVariable=false

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pl.program
class IfTensorReturnScratch:
    """Output tensor reassigned inside two sequential ``if`` branches.

    The first ``if``'s merged ``output`` feeds the store inside the second
    ``if`` — exactly the cross-if phi chain from issue #1533. Each ``if`` has no
    explicit ``else``, so control-flow lowering synthesises an else that yields
    the unchanged tensor, producing a TensorType IfStmt return_var that the
    second ``if`` then partition_views.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pl.Tensor[[64, 64], pl.FP32],
        cond0: pl.Scalar[pl.BOOL],
        cond1: pl.Scalar[pl.BOOL],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        t: pl.Tile[[64, 64], pl.FP32] = pl.load(data, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
        if cond0:
            output: pl.Tensor[[64, 64], pl.FP32] = pl.store(t, [0, 0], output)
        if cond1:
            output: pl.Tensor[[64, 64], pl.FP32] = pl.store(t, [0, 0], output)
        return output


SPMD_BLOCKS = 32
SPMD_COLS = 16


@pl.program
class SpmdIfTensorReturnScratch:
    """Conditional slice-assign to an outer tensor inside a ``pl.spmd`` body (issue #1569).

    The ``pl.spmd`` body is outlined into a per-block kernel whose ``if`` (no
    explicit ``else``) conditionally writes the outer ``out`` tensor. Control-flow
    lowering turns ``out`` into a TensorType IfStmt phi — the same shape as #1533,
    but reached through the spmd outlining path. The then-branch yields the
    stored-into tensor; the synthesised else yields the unchanged ``out`` param.
    Codegen must keep the phi out of ``scf.if`` results so the partition_view
    sources the concrete make_tensor_view rather than a fully-dynamic phi.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def spmd_kernel(
        self,
        flag: pl.Tensor[[SPMD_BLOCKS], pl.INT32],
        out: pl.Out[pl.Tensor[[SPMD_BLOCKS, SPMD_COLS], pl.FP32]],
    ) -> pl.Tensor[[SPMD_BLOCKS, SPMD_COLS], pl.FP32]:
        block_idx = pl.tile.get_block_idx()
        cond = pl.read(flag, [block_idx])
        if cond > 0:
            tile = pl.full([1, SPMD_COLS], dtype=pl.FP32, value=1.0)
            out[block_idx : block_idx + 1, :] = tile
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        flag: pl.Tensor[[SPMD_BLOCKS], pl.INT32],
        out: pl.Out[pl.Tensor[[SPMD_BLOCKS, SPMD_COLS], pl.FP32]],
    ) -> pl.Tensor[[SPMD_BLOCKS, SPMD_COLS], pl.FP32]:
        with pl.spmd(SPMD_BLOCKS, name_hint="spmd_if_scope"):
            out = self.spmd_kernel(flag, out)
        return out


def _compile_and_codegen(program_cls, func_name: str) -> str:
    """Run pass pipeline + PTO codegen on a single function, return MLIR string."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized = pm.run_passes(program_cls)

    func = None
    for f in optimized.functions.values():
        if f.name == func_name:
            func = f
            break
    assert func is not None, f"Function '{func_name}' not found in optimized program"

    single_func_program = ir.Program([func], func_name, optimized.span)
    gen = codegen.PTOCodegen()
    return gen.generate(single_func_program)


def _scf_if_header_lines(mlir: str) -> list[str]:
    """Return the ``... = scf.if ... -> (...)`` result-signature header lines."""
    return [line.strip() for line in mlir.split("\n") if "scf.if" in line and "-> (" in line]


@pytest.fixture(scope="module")
def scratch_mlir() -> str:
    return _compile_and_codegen(IfTensorReturnScratch, "kernel")


def test_compiles(scratch_mlir: str):
    """The phi-scratch pattern compiles through the pipeline + PTO codegen."""
    assert scratch_mlir, "Generated MLIR code should not be empty"
    assert "scf.if" in scratch_mlir, f"Expected scf.if in MLIR output:\n{scratch_mlir}"


def test_no_tensor_view_scf_if_result(scratch_mlir: str):
    """Tensors must NOT be routed through scf.if results (issue #1533).

    Routing a TensorType through scf.if retypes it to a fully-dynamic
    !pto.tensor_view<?x?>, which ptoas rejects under partition_view. Tensors are
    in-place mutable references, so the merged value is bound to the shared
    backing tensor_view instead of an scf.if result.
    """
    offending = [line for line in _scf_if_header_lines(scratch_mlir) if "tensor_view" in line]
    assert not offending, (
        "scf.if must not yield !pto.tensor_view results for in-place tensors; "
        "found:\n" + "\n".join(offending) + f"\n\nfull MLIR:\n{scratch_mlir}"
    )


def _partition_view_source(line: str) -> str:
    """Extract the source operand of a ``pto.partition_view`` line.

    Format: ``%result = pto.partition_view %source, offsets = [...] ...``.
    The source is the operand between ``partition_view`` and the first comma.
    """
    after_op = line.split("pto.partition_view", 1)[1].strip()
    return after_op.split(",", 1)[0].strip()


def test_partition_view_not_on_scf_phi(scratch_mlir: str):
    """partition_view *source operands* must be the concrete base view, not a phi.

    SCF-if phi tensor_views (``__phi_v*``) carry fully-dynamic dims that ptoas
    rejects. The fix binds tensor return_vars to the shared backing
    make_tensor_view, so every partition_view should source a concrete view
    (``__ssa_v*_view`` / ``__rv_*_view``), never a ``__phi`` SSA value. The
    partition_view *result* may still be named after the phi var — only the
    source operand matters.
    """
    sources = [
        _partition_view_source(line) for line in scratch_mlir.split("\n") if "pto.partition_view" in line
    ]
    assert sources, f"Expected at least one pto.partition_view:\n{scratch_mlir}"
    bad = [src for src in sources if "__phi" in src]
    assert not bad, (
        "pto.partition_view should source the concrete base tensor_view, not an "
        f"scf.if phi result; offending sources: {bad}\n\nfull MLIR:\n{scratch_mlir}"
    )


@pytest.fixture(scope="module")
def spmd_scratch_mlir() -> str:
    return _compile_and_codegen(SpmdIfTensorReturnScratch, "spmd_kernel")


def test_spmd_compiles(spmd_scratch_mlir: str):
    """The spmd-if-write pattern compiles through the pipeline + PTO codegen (issue #1569)."""
    assert spmd_scratch_mlir, "Generated MLIR code should not be empty"
    assert "scf.if" in spmd_scratch_mlir, f"Expected scf.if in MLIR output:\n{spmd_scratch_mlir}"


def test_spmd_no_tensor_view_scf_if_result(spmd_scratch_mlir: str):
    """The spmd-outlined IfStmt must not route the tensor through scf.if results (issue #1569).

    This is the spmd variant of ``test_no_tensor_view_scf_if_result``: routing
    the conditionally-written outer tensor through ``scf.if`` would force the
    empty else branch to yield the raw ``!pto.ptr`` function arg against an
    ``!pto.tensor_view`` result type, which ptoas rejects.
    """
    offending = [line for line in _scf_if_header_lines(spmd_scratch_mlir) if "tensor_view" in line]
    assert not offending, (
        "scf.if must not yield !pto.tensor_view results for in-place tensors; "
        "found:\n" + "\n".join(offending) + f"\n\nfull MLIR:\n{spmd_scratch_mlir}"
    )


def test_spmd_partition_view_not_on_scf_phi(spmd_scratch_mlir: str):
    """partition_view inside the spmd kernel must source the concrete base view, not a phi."""
    sources = [
        _partition_view_source(line) for line in spmd_scratch_mlir.split("\n") if "pto.partition_view" in line
    ]
    assert sources, f"Expected at least one pto.partition_view:\n{spmd_scratch_mlir}"
    bad = [src for src in sources if "__phi" in src]
    assert not bad, (
        "pto.partition_view should source the concrete base tensor_view, not an "
        f"scf.if phi result; offending sources: {bad}\n\nfull MLIR:\n{spmd_scratch_mlir}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
