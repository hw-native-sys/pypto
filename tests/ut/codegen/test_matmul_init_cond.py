# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``matmul_acc(init_cond=...)`` — conditional accumulator initialization.

``init_cond`` makes the accumulator's initial value conditional: where the
predicate holds, the accumulator is overwritten with ``lhs @ rhs`` rather than
accumulated into. This is the split-K ``k == 0`` idiom, and it keeps the
accumulator single-def where a hand-written if/else would put a phi on an
in-place Acc buffer.

The ISA carries this as one bit of the MAD's Xt register, but ``pto.tmatmul`` and
``pto.tmatmul.acc`` are distinct ops, so a *runtime* predicate lowers to a branch
over the two while a literal one selects a single op at compile time.
"""

import pypto.language as pl
import pytest
from pypto import backend, codegen
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager
from pypto.language.parser.diagnostics import InvalidOperationError

PTOCodegen = codegen.PTOCodegen


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _generate_default_mlir(program_cls) -> str:
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    program = pm.run_passes(program_cls)
    result = PTOCodegen().generate(program)
    return result if isinstance(result, str) else "".join(result.values())


@pl.program
class MatmulAccPlain:
    """No predicate — the accumulating form, unchanged."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
            [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc
        )
        out_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(acc_tile, lhs_tile, rhs_tile)
        return pl.store(out_tile, [0, 0], output)


@pl.program
class MatmulAccInitTrue:
    """Literal ``True`` — folds to the non-accumulating form."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
            [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc
        )
        out_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(
            acc_tile, lhs_tile, rhs_tile, init_cond=True
        )
        return pl.store(out_tile, [0, 0], output)


@pl.program
class MatmulAccInitFalse:
    """Literal ``False`` — folds to the accumulating form."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 16], pl.FP32],
        rhs: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
            rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
        )
        acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
            [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc
        )
        out_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(
            acc_tile, lhs_tile, rhs_tile, init_cond=False
        )
        return pl.store(out_tile, [0, 0], output)


@pl.program
class MatmulAccSplitK:
    """Runtime predicate — the split-K ``k == 0`` idiom.

    Spelled through the type-dispatched ``pl.matmul_acc`` rather than
    ``pl.tile.matmul_acc`` so the unified wrapper's Tile path is covered too.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        lhs: pl.Tensor[[16, 64], pl.FP32],
        rhs: pl.Tensor[[64, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
            [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc
        )
        for k0 in pl.range(0, 64, 16):
            a: pl.Tile[[16, 16], pl.FP32] = pl.load(lhs, [0, k0], [16, 16], target_memory=pl.MemorySpace.Mat)
            b: pl.Tile[[16, 16], pl.FP32] = pl.load(rhs, [k0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
            acc_tile = pl.matmul_acc(acc_tile, a, b, init_cond=(k0 == 0))
        return pl.store(acc_tile, [0, 0], output)


def test_no_init_cond_emits_only_the_accumulating_form():
    mlir = _generate_default_mlir(MatmulAccPlain)
    assert "pto.tmatmul.acc" in mlir, mlir
    # The accumulating op is the only matmul, and nothing is branched over.
    assert mlir.count("pto.tmatmul") == 1, mlir
    assert "scf.if" not in mlir, mlir


def test_literal_true_folds_to_the_non_accumulating_form():
    mlir = _generate_default_mlir(MatmulAccInitTrue)
    assert "pto.tmatmul.acc" not in mlir, (
        "init_cond=True must overwrite the accumulator, not accumulate into it:\n" + mlir
    )
    assert "pto.tmatmul " in mlir, mlir
    # A compile-time predicate must not leave a branch behind.
    assert "scf.if" not in mlir, mlir


def test_literal_false_folds_to_the_accumulating_form():
    mlir = _generate_default_mlir(MatmulAccInitFalse)
    assert "pto.tmatmul.acc" in mlir, mlir
    assert mlir.count("pto.tmatmul") == 1, mlir
    assert "scf.if" not in mlir, mlir


def test_runtime_predicate_branches_over_both_forms():
    mlir = _generate_default_mlir(MatmulAccSplitK)
    assert "scf.if" in mlir, "a runtime init_cond must lower to a branch:\n" + mlir
    # Both arms are present: initialize on the guarded path, accumulate otherwise.
    assert "pto.tmatmul.acc" in mlir, mlir
    assert mlir.count("pto.tmatmul") == 2, (
        "expected exactly the initializing and accumulating forms:\n" + mlir
    )
    # Both arms write the same in-place accumulator, so no phi is materialized
    # for the tile — scf.if carries no results.
    assert "scf.if" in mlir and "= scf.if" not in mlir, (
        "the accumulator is written in place, so scf.if must not yield a value:\n" + mlir
    )


def test_non_boolean_init_cond_is_rejected():
    with pytest.raises(InvalidOperationError, match="init_cond to have dtype BOOL"):

        @pl.program
        class BadInitCond:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                lhs: pl.Tensor[[16, 16], pl.FP32],
                rhs: pl.Tensor[[16, 16], pl.FP32],
                acc: pl.Tensor[[16, 16], pl.FP32],
                output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ) -> pl.Tensor[[16, 16], pl.FP32]:
                lhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    lhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                rhs_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(
                    rhs, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat
                )
                acc_tile: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
                    [16, 16], pl.FP32, target_memory=pl.MemorySpace.Acc
                )
                # An index, not a predicate — must be rejected rather than
                # silently reinterpreted as a truth value.
                out_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.matmul_acc(
                    acc_tile, lhs_tile, rhs_tile, init_cond=pl.read(acc, [0, 0])
                )
                return pl.store(out_tile, [0, 0], output)

        _ = BadInitCond


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
