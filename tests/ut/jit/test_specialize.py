# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``JITFunction.specialize`` and the parameter-direction accessors.

``specialize()`` is the pre-pass half of ``lower()``: it returns the parsed
program before any pass has run, which is what a consumer driving the pass
pipeline itself needs — ``ir.compile(program, output_dir=...)`` runs passes and
code generation together, so handing it ``lower()``'s output would run the
pipeline twice.

The system-test harness is that consumer: it builds every case's IR through
this method, whichever surface authored the kernel.
"""

import pypto.language as pl
import pytest
import torch
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.decorator import jit
from pypto.pypto_core import ir

M = 16
N = 16


@jit.incore
def _abs_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@jit
def _abs_entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    out = _abs_kernel(a, out)
    return out


@jit.incore
def _shaped_kernel(a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    tile_a = pl.load(a, [0, 0], [M, N])
    return pl.store(pl.tile.abs(tile_a), [0, 0], out)


@jit
def _shaped_entry(a: pl.Tensor[[M, N], pl.FP32], out: pl.Out[pl.Tensor[[M, N], pl.FP32]]):
    out = _shaped_kernel(a, out)
    return out


@jit
def _mixed_directions(
    x: pl.Tensor, acc: pl.InOut[pl.Tensor], y: pl.Tensor, z: pl.Out[pl.Tensor]
):  # pragma: no cover - never specialized, only its signature is read
    return acc


@pl.program
class _AbsRef:
    """Hand-written equivalent of ``_abs_entry``."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a = pl.load(a, [0, 0], [M, N])
        out = pl.store(pl.tile.abs(tile_a), [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


class TestSpecialize:
    """``specialize()`` returns a usable pre-pass program."""

    def test_returns_pre_pass_program(self):
        """The entry and its dep are both present, untransformed."""
        program = _abs_entry.specialize(torch.randn(M, N), torch.zeros(M, N))
        assert isinstance(program, ir.Program)
        assert len(list(program.functions)) == 2, "entry + its @pl.jit.incore dep"

    def test_matches_hand_written_program_after_passes(self):
        """The specialized program lowers to the same IR as the reference.

        Asserted after the pass pipeline, not before: the specializer renames
        SSA-rebound locals (``out`` becomes ``out_v1``), so the two programs are
        only textually different beforehand. Running both through the same
        strategy is the equivalence that matters — it is what the harness
        compiles.

        The JIT functions are declared here, rather than at module level,
        because function names are part of the IR: they must match the
        reference's ``kernel`` / ``orchestrator`` for the comparison to be
        about structure rather than about naming.
        """

        @jit.incore
        def kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            tile_a = pl.load(a, [0, 0], [M, N])
            return pl.store(pl.tile.abs(tile_a), [0, 0], out)

        @jit
        def orchestrator(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            out = kernel(a, out)
            return out

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        got = pm.run_passes(orchestrator.specialize(torch.randn(M, N), torch.zeros(M, N)))
        ir.assert_structural_equal(got, pm.run_passes(_AbsRef))

    def test_agrees_with_lower(self):
        """Running passes over ``specialize()`` reproduces ``lower()`` exactly."""
        a, out = torch.randn(M, N), torch.zeros(M, N)
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        ir.assert_structural_equal(pm.run_passes(_abs_entry.specialize(a, out)), _abs_entry.lower(a, out))

    def test_does_not_populate_the_compiled_cache(self):
        """No compilation happens, so the L1 cache stays empty."""
        _abs_entry._cache.clear()
        _abs_entry.specialize(torch.randn(M, N), torch.zeros(M, N))
        assert len(_abs_entry._cache) == 0

    def test_signature_mode_needs_full_shapes(self):
        """A bare ``pl.Tensor`` cannot be specialized without a sample."""
        with pytest.raises(TypeError, match="bare 'pl.Tensor' annotation with no shape"):
            _abs_entry.specialize()

    def test_signature_mode_works_with_shaped_annotations(self):
        """Fully-shaped annotations specialize with no sample tensors at all."""
        program = _shaped_entry.specialize()
        assert isinstance(program, ir.Program)
        assert len(list(program.functions)) == 2


class TestParamAccessors:
    """``param_names`` / ``output_param_names`` describe the signature."""

    def test_param_names_in_declaration_order(self):
        assert _abs_entry.param_names == ("a", "out")
        assert _mixed_directions.param_names == ("x", "acc", "y", "z")

    def test_output_param_names_covers_out_and_inout(self):
        """Both directions are outputs, reported in declaration order."""
        assert _mixed_directions.output_param_names == ("acc", "z")

    def test_output_param_names_excludes_pure_inputs(self):
        assert _abs_entry.output_param_names == ("out",)
        assert _abs_kernel.output_param_names == ("out",)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
