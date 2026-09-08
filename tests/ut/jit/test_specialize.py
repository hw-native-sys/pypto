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

from typing import Any

import pypto.language as pl
import pytest
import torch
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit.decorator import jit
from pypto.language.parser.diagnostics import ParserTypeError
from pypto.pypto_core import DataType, ir
from pypto.pypto_core.ir import MemorySpace
from pypto.runtime.runner import RunConfig

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


# ---------------------------------------------------------------------------
# Free names a body inherits from its module or an enclosing function
# ---------------------------------------------------------------------------

_CAST_MODE = "trunc"
_CAST_DTYPE = pl.INT8
_LOAD_MEM = pl.Mem.Vec
_TILE_SHAPE = [1, N]


@jit.incore
def _free_name_kernel(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:
    tile_a = pl.load(a, [0, 0], _TILE_SHAPE, target_memory=_LOAD_MEM)
    quantized = pl.cast(tile_a, _CAST_DTYPE, mode=_CAST_MODE)
    return pl.store(quantized, [0, 0], out)


@jit
def _free_name_entry(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:
    return _free_name_kernel(a, out)


# Deliberately typed ``Any``: this stands for any value the renderer cannot write
# as source, and the point of the test is what the *specializer* does with it.
_OPAQUE: Any = object()


@jit.incore
def _opaque_name_kernel(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:  # pragma: no cover - specialization is expected to fail
    tile_a = pl.load(a, [0, 0], [1, N], target_memory=_OPAQUE)
    quantized = pl.cast(tile_a, pl.INT8, mode="trunc")
    return pl.store(quantized, [0, 0], out)


@jit
def _opaque_name_entry(
    a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
) -> pl.Tensor[[1, N], pl.INT8]:  # pragma: no cover - specialization is expected to fail
    return _opaque_name_kernel(a, out)


def _make_parameterized_entry(mode: str, dtype):
    """A kernel factory — the shape every parameterized test suite wants to write."""

    @jit.incore
    def kernel(
        a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
    ) -> pl.Tensor[[1, N], pl.INT8]:
        tile_a = pl.load(a, [0, 0], [1, N])
        quantized = pl.cast(tile_a, dtype, mode=mode)
        return pl.store(quantized, [0, 0], out)

    @jit
    def entry(
        a: pl.Tensor[[1, N], pl.FP16], out: pl.Out[pl.Tensor[[1, N], pl.INT8]]
    ) -> pl.Tensor[[1, N], pl.INT8]:
        return kernel(a, out)

    return entry


def _find_call(program: ir.Program, op_name: str) -> ir.Call:
    """The single call to ``op_name`` in ``program``."""
    found: list[ir.Call] = []
    target = ir.get_op(op_name).name

    class _Collector(ir.IRVisitor):
        def visit_call(self, op: ir.Call) -> None:
            if op.op.name == target:
                found.append(op)
            super().visit_call(op)

    _Collector().visit_program(program)
    assert len(found) == 1, f"expected exactly one {op_name}, got {len(found)}"
    return found[0]


class TestFreeNameResolution:
    """A body may name a value defined in its module or an enclosing function.

    The generated ``@pl.program`` source is parsed in a namespace holding only
    ``pl`` and ``pld``, so each such name has to be replaced by source text that
    evaluates back to the same value. Before that covered strings, dtypes, enums
    and sequences, only ``int``/``float``/``bool`` folded and everything else
    failed with "Cannot resolve expression" / "Undefined variable" — which made a
    parameterized kernel factory impossible to write.
    """

    def test_module_constants_of_every_renderable_kind_resolve(self):
        """One kernel naming a str, a DataType, an enum and a list constant."""
        program = _free_name_entry.specialize()

        cast_call = _find_call(program, "tile.cast")
        assert cast_call.kwargs["mode"] == 5, "the str constant reached the op as trunc"
        assert cast_call.kwargs["target_type"] == DataType.INT8, "the DataType constant survived"

        load_call = _find_call(program, "tile.load")
        assert load_call.kwargs["target_memory"] == MemorySpace.Vec, "the enum constant survived"
        loaded = load_call.type
        assert isinstance(loaded, ir.TileType)
        shape = [dim.value for dim in loaded.shape if isinstance(dim, ir.ConstInt)]
        assert shape == [1, N], "the list constant became the load shape"

    def test_an_unrenderable_constant_still_fails_loudly(self):
        """A value with no source form must keep reporting the name, not fold to something wrong."""
        with pytest.raises(ParserTypeError, match="Cannot resolve expression '_OPAQUE'"):
            _opaque_name_entry.specialize()

    @pytest.mark.parametrize(
        "mode, dtype, expected_mode, expected_dtype",
        [
            ("trunc", pl.INT8, 5, DataType.INT8),
            ("round", pl.INT16, 2, DataType.INT16),
        ],
    )
    def test_a_factory_can_parameterize_its_kernel(self, mode, dtype, expected_mode, expected_dtype):
        """Values captured from the factory's frame resolve like module ones."""
        program = _make_parameterized_entry(mode, dtype).specialize()
        cast_call = _find_call(program, "tile.cast")
        assert cast_call.kwargs["mode"] == expected_mode
        assert cast_call.kwargs["target_type"] == expected_dtype


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

    def test_rejects_config(self):
        """``config=`` is refused rather than silently discarded.

        No pass runs here, so nothing would read a ``RunConfig``. Consuming it
        quietly would let a caller believe a strategy or diagnostics setting
        shaped the returned IR.
        """
        with pytest.raises(TypeError, match="specialize\\(\\) does not accept config="):
            _abs_entry.specialize(torch.randn(M, N), torch.zeros(M, N), config=RunConfig())

    def test_rejects_config_in_signature_mode(self):
        """The same refusal applies with no sample arguments at all."""
        with pytest.raises(TypeError, match="specialize\\(\\) does not accept config="):
            _shaped_entry.specialize(config=RunConfig(strategy=OptimizationStrategy.Default))


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
