# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for dynamic valid_shape across if/else branches.

Verifies the pattern described in the paged-attention design discussion:

At the PTO level the pattern is:
  tile = alloc_tile<row=R, col=C, v_row=?, v_col=?, pad=min>
  if (...) { set_validshape(tile, vrow1, vcol1) }
  else     { set_validshape(tile, vrow2, vcol2) }

In the DSL, this translates to computing the valid length as a scalar in the
if/else, then performing a single load+fillpad with that computed length:
  if is_last:
      vlen = last_valid_len
  else:
      vlen = full_len
  s_tile = pl.load(..., valid_shapes=[rows, vlen])
  s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)

The tile buffer type is uniform (same v_row=?, v_col=?, pad=min) regardless
of which branch executed. Only the runtime valid-shape value differs.
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

# ---------------------------------------------------------------------------
# Program 1: Simple if/else with different valid_shapes (no loop)
# ---------------------------------------------------------------------------


@pl.program
class DynValidShapeIfElse:
    """Compute valid length in if/else, then load+fillpad with uniform tile type.

    The if/else only selects the scalar valid length. The load and fillpad
    happen once, producing a single tile type with dynamic valid_shape and pad.min.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        data: pl.Tensor[[64, 64], pl.FP32],
        scale: pl.Scalar[pl.FP32],
        is_last: pl.Scalar[pl.BOOL],
        valid_len: pl.Scalar[pl.INDEX],
        full_len: pl.Scalar[pl.INDEX],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        if is_last:
            vlen: pl.Scalar[pl.INDEX] = valid_len
        else:
            vlen: pl.Scalar[pl.INDEX] = full_len
        s_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
            data, [0, 0], [64, 64], valid_shapes=[64, vlen], target_memory=pl.MemorySpace.Vec
        )
        s_padded: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
        scaled: pl.Tile[[64, 64], pl.FP32] = pl.mul(s_padded, scale)
        out: pl.Tensor[[64, 64], pl.FP32] = pl.store(scaled, [0, 0], output)
        return out


# ---------------------------------------------------------------------------
# Program 2: Loop with if/else — the full paged-attention conversation pattern
# ---------------------------------------------------------------------------


@pl.program
class DynValidShapeLoopIfElse:
    """Loop over blocks, selecting valid length per iteration via if/else.

    On the last iteration: vlen = last_valid_len (partial block)
    On other iterations:   vlen = block_size     (full block)

    After the if/else, the single load+fillpad uses the computed vlen.
    This produces a uniform tile type across all iterations.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        sij_buf: pl.Tensor[[128, 64], pl.FP32],
        scale: pl.Scalar[pl.FP32],
        n_blocks: pl.Scalar[pl.INDEX],
        last_valid_len: pl.Scalar[pl.INDEX],
        block_size: pl.Scalar[pl.INDEX],
        output: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
    ) -> pl.Tensor[[128, 64], pl.FP32]:
        for i, (out,) in pl.range(n_blocks, init_values=(output,)):
            if i == n_blocks - 1:
                vlen: pl.Scalar[pl.INDEX] = last_valid_len
            else:
                vlen: pl.Scalar[pl.INDEX] = block_size
            s_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
                sij_buf, [i * 64, 0], [64, 64], valid_shapes=[64, vlen], target_memory=pl.MemorySpace.Vec
            )
            s_padded: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
            scaled: pl.Tile[[64, 64], pl.FP32] = pl.mul(s_padded, scale)
            updated: pl.Tensor[[128, 64], pl.FP32] = pl.store(scaled, [i * 64, 0], out)
            loop_result = pl.yield_(updated)
        return loop_result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


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


@pytest.fixture(scope="module")
def if_else_mlir() -> str:
    """Compile if/else program once for all tests in this module."""
    return _compile_and_codegen(DynValidShapeIfElse, "kernel")


@pytest.fixture(scope="module")
def loop_mlir() -> str:
    """Compile loop program once for all tests in this module."""
    return _compile_and_codegen(DynValidShapeLoopIfElse, "kernel")


def test_if_else_dyn_valid_shape_compiles(if_else_mlir: str):
    """Verify that if/else with dynamic valid_shape compiles through the pipeline."""
    assert if_else_mlir, "Generated MLIR code should not be empty"


def test_if_else_dyn_valid_shape_has_dynamic_alloc(if_else_mlir: str):
    """Verify the generated code has dynamic valid-shape tile allocations."""
    alloc_lines = [line.strip() for line in if_else_mlir.split("\n") if "pto.alloc_tile" in line]
    s_tile_allocs = [line for line in alloc_lines if "s_tile" in line]
    assert len(s_tile_allocs) >= 1, f"Expected s_tile alloc, got alloc_lines: {alloc_lines}"
    assert "v_col=?" in s_tile_allocs[0], f"Expected dynamic v_col=? in s_tile alloc: {s_tile_allocs[0]}"


def test_if_else_dyn_valid_shape_alloc_carries_runtime_valid_col(if_else_mlir: str):
    """Verify the s_tile alloc carries the runtime valid_col directly.

    With unified always-dynamic alloc_tile, the runtime valid_col flows through
    the alloc_tile valid_col operand instead of a separate pto.set_validshape op
    after the tload.
    """
    assert "pto.set_validshape" not in if_else_mlir, (
        "alloc_tile already carries valid_row/valid_col; "
        f"did not expect a separate pto.set_validshape:\n{if_else_mlir}"
    )
    alloc_lines = [line.strip() for line in if_else_mlir.split("\n") if "pto.alloc_tile" in line]
    s_tile_allocs = [line for line in alloc_lines if "s_tile" in line]
    assert len(s_tile_allocs) >= 1, f"Expected s_tile alloc, got alloc_lines: {alloc_lines}"
    # The runtime valid_col is materialised as %vlen* (the scf.if-yielded scalar)
    # and threaded through the alloc_tile.
    assert "valid_col = %vlen" in s_tile_allocs[0], (
        f"Expected valid_col operand sourced from %vlen in s_tile alloc: {s_tile_allocs[0]}"
    )


def test_if_else_dyn_valid_shape_has_fillpad(if_else_mlir: str):
    """Verify the generated code emits pto.fillpad with pad=min."""
    assert "pto.tfillpad" in if_else_mlir, f"Expected pto.tfillpad in MLIR output:\n{if_else_mlir}"


def test_if_else_dyn_valid_shape_padded_alloc_has_pad_min(if_else_mlir: str):
    """Verify the padded tile alloc has pad=3 (PadValue.min)."""
    alloc_lines = [line.strip() for line in if_else_mlir.split("\n") if "pto.alloc_tile" in line]
    padded_allocs = [line for line in alloc_lines if "s_padded" in line]
    assert len(padded_allocs) >= 1, f"Expected s_padded alloc, got alloc_lines: {alloc_lines}"
    assert "pad=3>" in padded_allocs[0], f"Expected pad=3 (PadValue.min) in padded alloc: {padded_allocs[0]}"


def test_loop_if_else_dyn_valid_shape_compiles(loop_mlir: str):
    """Verify the loop + if/else pattern with dynamic valid_shapes compiles."""
    assert loop_mlir, "Generated MLIR code should not be empty"


def test_loop_if_else_dyn_valid_shape_has_scf_for(loop_mlir: str):
    """Verify the loop generates scf.for in the MLIR output."""
    assert "scf.for" in loop_mlir, f"Expected scf.for loop in MLIR output:\n{loop_mlir}"


def test_loop_if_else_dyn_valid_shape_has_scf_if(loop_mlir: str):
    """Verify the if/else generates scf.if in the MLIR output."""
    assert "scf.if" in loop_mlir, f"Expected scf.if in MLIR output:\n{loop_mlir}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
