# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for the CompiledProgram callable API.

Verifies that ``ir.compile()`` returns a ``CompiledProgram`` that can
be called directly with ``torch.Tensor`` arguments (Triton-like API).

Tests exercise both calling conventions:

- **In-place**: ``compiled(a, b, c)`` — output tensor passed as argument.
- **Return-style**: ``c = compiled(a, b)`` — output allocated and returned.

Compiled artifacts are saved under ``build_output/test_compiled_program/``
for post-mortem inspection.
"""

import os
from datetime import datetime
from pathlib import Path

import pypto.language as pl
import pytest
import torch
from examples.kernels.elementwise import TileAddProgram, TileMulProgram
from pypto import ir
from pypto.ir.compiled_program import CompiledProgram

_BUILD_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "build_output" / "test_compiled_program"


@pl.program
class TileAddInOutProgram:
    """Program with both InOut and Out params.

    - ``a``: input
    - ``acc``: InOut — initial value provided by caller, updated in-place
    - ``out``: pure Out — can be auto-allocated in return-style calls

    Computes: acc += a; out = acc
    """

    @pl.function(type=pl.FunctionType.InCore)
    def tile_add_acc(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        acc: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> tuple[pl.Tensor[[128, 128], pl.FP32], pl.Tensor[[128, 128], pl.FP32]]:
        a_tile = pl.load(a, [0, 0], [128, 128])
        acc_tile = pl.load(acc, [0, 0], [128, 128])
        sum_tile = pl.add(a_tile, acc_tile)
        acc_new = pl.store(sum_tile, [0, 0], acc)
        out_new = pl.store(sum_tile, [0, 0], out)
        return acc_new, out_new

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[128, 128], pl.FP32],
        acc: pl.InOut[pl.Tensor[[128, 128], pl.FP32]],
        out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
    ) -> pl.Tensor[[128, 128], pl.FP32]:
        _, out_ret = self.tile_add_acc(a, acc, out)
        return out_ret


@pytest.fixture(scope="session")
def output_root() -> Path:
    """Session-scoped output directory under build_output/ (persists after tests)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = _BUILD_OUTPUT_DIR / timestamp
    root.mkdir(parents=True, exist_ok=True)
    return root


class TestCompiledProgramCallable:
    """Test CompiledProgram in-place and return-style calling conventions."""

    def test_compile_returns_compiled_program(self, output_root):
        """ir.compile() should return a CompiledProgram instance."""
        result = ir.compile(TileAddProgram, output_dir=str(output_root / "add"))
        assert isinstance(result, CompiledProgram)

    def test_inplace_add(self, output_root, test_config):
        """In-place call: compiled(a, b, c) modifies c on device."""
        compiled = ir.compile(
            TileAddProgram,
            output_dir=str(output_root / "add_inplace"),
            platform=test_config.platform,
        )

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        compiled(a, b, c, config=test_config)

        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"In-place add failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_return_style_add(self, output_root, test_config):
        """Return-style call: c = compiled(a, b) allocates and returns output."""
        compiled = ir.compile(
            TileAddProgram,
            output_dir=str(output_root / "add_return"),
            platform=test_config.platform,
        )

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)

        c = compiled(a, b, config=test_config)

        assert c is not None, "Return-style call should return a tensor"
        assert isinstance(c, torch.Tensor)
        assert c.shape == (128, 128)
        expected = torch.full((128, 128), 5.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"Return-style add failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_inplace_mul(self, output_root, test_config):
        """In-place multiplication: compiled(a, b, c) with c = a * b."""
        compiled = ir.compile(
            TileMulProgram,
            output_dir=str(output_root / "mul_inplace"),
            platform=test_config.platform,
        )

        a = torch.full((128, 128), 4.0, dtype=torch.float32)
        b = torch.full((128, 128), 3.0, dtype=torch.float32)
        c = torch.zeros((128, 128), dtype=torch.float32)

        compiled(a, b, c, config=test_config)

        expected = torch.full((128, 128), 12.0, dtype=torch.float32)
        assert torch.allclose(c, expected, rtol=1e-5, atol=1e-5), (
            f"In-place mul failed: max diff = {(c - expected).abs().max().item()}"
        )

    def test_compile_once_run_twice(self, output_root, test_config):
        """Compile once, execute multiple times with different inputs."""
        compiled = ir.compile(
            TileAddProgram,
            output_dir=str(output_root / "add_reuse"),
            platform=test_config.platform,
        )

        # First execution: 1.0 + 2.0 = 3.0
        a1 = torch.full((128, 128), 1.0, dtype=torch.float32)
        b1 = torch.full((128, 128), 2.0, dtype=torch.float32)
        c1 = torch.zeros((128, 128), dtype=torch.float32)
        compiled(a1, b1, c1, config=test_config)
        assert torch.allclose(c1, torch.full((128, 128), 3.0), rtol=1e-5, atol=1e-5)

        # Second execution: 10.0 + 20.0 = 30.0
        a2 = torch.full((128, 128), 10.0, dtype=torch.float32)
        b2 = torch.full((128, 128), 20.0, dtype=torch.float32)
        c2 = torch.zeros((128, 128), dtype=torch.float32)
        compiled(a2, b2, c2, config=test_config)
        assert torch.allclose(c2, torch.full((128, 128), 30.0), rtol=1e-5, atol=1e-5)

    def test_wrong_arg_count_raises(self, output_root):
        """Passing wrong number of arguments should raise TypeError."""
        compiled = ir.compile(TileAddProgram, output_dir=str(output_root / "add_err"))
        a = torch.randn(128, 128)
        with pytest.raises(TypeError, match="expects"):
            compiled(a)

    def test_backward_compat_path(self, output_root):
        """str(compiled) and os.path.join should still work."""
        compiled = ir.compile(TileAddProgram, output_dir=str(output_root / "add_compat"))
        assert os.path.isdir(str(compiled))
        assert os.path.isdir(os.path.join(compiled, "orchestration"))

    def test_metadata_extraction(self, output_root):
        """CompiledProgram should expose correct param metadata."""
        compiled = ir.compile(TileAddProgram, output_dir=str(output_root / "add_meta"))
        assert compiled.param_names == ["a", "b", "out_c"]
        assert compiled.output_indices == [2]
        assert compiled.has_return is True

    def test_inout_param_excluded_from_output_indices(self, output_root):
        """InOut params must not appear in output_indices (no auto-allocation).

        Program has params (a: In, acc: InOut, out: Out). Only ``out`` is
        auto-allocated in return-style calls.
        """
        compiled = ir.compile(
            TileAddInOutProgram,
            output_dir=str(output_root / "add_inout_meta"),
        )
        assert compiled.param_names == ["a", "acc", "out"]
        # Only pure Out (index 2) is auto-allocated; InOut (index 1) is not
        assert compiled.output_indices == [2]
        assert compiled.has_return is True

    def test_inout_return_style_preserves_acc_initial(self, output_root, test_config):
        """Return-style with InOut: caller supplies ``a`` and ``acc``; ``out`` is auto-allocated.

        Verifies that InOut ``acc`` keeps its caller-provided initial value
        (not silently zero-allocated like a pure Out).
        """
        compiled = ir.compile(
            TileAddInOutProgram,
            output_dir=str(output_root / "add_inout_return"),
            platform=test_config.platform,
        )

        a = torch.full((128, 128), 2.0, dtype=torch.float32)
        acc = torch.full((128, 128), 10.0, dtype=torch.float32)  # Initial value

        # Return-style: pass In + InOut (2 args), Out is allocated & returned
        out = compiled(a, acc, config=test_config)

        expected = torch.full((128, 128), 12.0, dtype=torch.float32)
        # acc was in-place updated: 10 + 2 = 12
        assert torch.allclose(acc, expected, rtol=1e-5, atol=1e-5), (
            f"InOut acc not updated: max diff = {(acc - expected).abs().max().item()}"
        )
        # out was allocated & returned, and equals acc
        assert out is not None
        assert isinstance(out, torch.Tensor)
        assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
