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
  s_tile = pl.load(..., valid_shape=[rows, vlen])
  s_padded = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)

The tile buffer type is uniform (same v_row=?, v_col=?, pad=min) regardless
of which branch executed. Only the runtime valid-shape value differs.

The programs below are written with ``@pl.program`` and explicit
``pl.Scalar[...]`` parameters. The final section covers the same pattern
reached through ``@pl.jit``, where the specializer must leave the in-DSL
``if``/``else`` intact for the branch to survive to an ``scf.if``.
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors
# from type-checking annotations that reference module-level names.
# pyright: reportUndefinedVariable=false

import importlib
import re

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen

# ---------------------------------------------------------------------------
# Program 1: Simple if/else with different valid_shape values (no loop)
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
            data, [0, 0], [64, 64], valid_shape=[64, vlen], target_memory=pl.MemorySpace.Vec
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
                sij_buf, [i * 64, 0], [64, 64], valid_shape=[64, vlen], target_memory=pl.MemorySpace.Vec
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
    """Verify the loop + if/else pattern with dynamic valid_shape compiles."""
    assert loop_mlir, "Generated MLIR code should not be empty"


def test_loop_if_else_dyn_valid_shape_has_scf_for(loop_mlir: str):
    """Verify the loop generates scf.for in the MLIR output."""
    assert "scf.for" in loop_mlir, f"Expected scf.for loop in MLIR output:\n{loop_mlir}"


def test_loop_if_else_dyn_valid_shape_has_scf_if(loop_mlir: str):
    """Verify the if/else generates scf.if in the MLIR output."""
    assert "scf.if" in loop_mlir, f"Expected scf.if in MLIR output:\n{loop_mlir}"


# ---------------------------------------------------------------------------
# The same patterns reached through @pl.jit
# ---------------------------------------------------------------------------
#
# The JIT specializer rewrites the user's function into the @pl.program form
# above. It must leave an in-DSL if/else that rebinds one name in both branches
# alone: aliasing the else-branch binding to a distinct name (`vlen_v1`) would
# make ConvertToSSA reject the read as "used outside its defining scope".
# `tests/ut/jit/test_specializer.py::test_if_else_branch_rebind` guards the
# rewrite at source level; these tests guard the whole path down to PTO.


def _jit_device_mlir(jit_func, *args) -> str:
    """Lower a @pl.jit kernel and run PTO codegen on its device function."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    program = jit_func.lower(*args)
    device_funcs = [f for f in program.functions.values() if f.func_type != ir.FunctionType.Orchestration]
    assert len(device_funcs) == 1, (
        f"expected exactly one device function, got {[f.name for f in device_funcs]}"
    )
    single_func_program = ir.Program([device_funcs[0]], device_funcs[0].name, program.span)
    return codegen.PTOCodegen().generate(single_func_program)


def _s_tile_alloc(mlir: str) -> str:
    """Return the ``pto.alloc_tile`` line that allocates ``s_tile``."""
    alloc_lines = [line.strip() for line in mlir.split("\n") if "pto.alloc_tile" in line]
    s_tile_allocs = [line for line in alloc_lines if "s_tile" in line]
    assert len(s_tile_allocs) >= 1, f"Expected s_tile alloc, got alloc_lines: {alloc_lines}"
    return s_tile_allocs[0]


def _valid_col_operand(alloc_line: str) -> str:
    """Extract the ``valid_col`` operand from a ``pto.alloc_tile`` line."""
    match = re.search(r"valid_col = (\S+)", alloc_line)
    assert match is not None, f"No valid_col operand in alloc: {alloc_line}"
    return match.group(1)


def _dyn_valid_shape_example():
    """Import the numbered example module.

    The package registers an unnumbered ``dyn_valid_shape`` alias in
    ``sys.modules``, but that alias exists only at runtime, so a static import
    of it does not resolve. Import the real module name instead, as
    ``tests/st/runtime/framework_and_models/test_paged_attention_spmd.py`` does.
    """
    return importlib.import_module("examples.intermediate.06_dyn_valid_shape")


@pytest.fixture(scope="module")
def jit_if_else_mlir() -> str:
    """Codegen the @pl.jit if/else kernel once for all tests in this module."""
    torch = pytest.importorskip("torch")
    example = _dyn_valid_shape_example()

    data = torch.zeros((example.Q_TILE, example.BLOCK_COL), dtype=torch.float32)
    out = torch.zeros((example.Q_TILE, example.BLOCK_COL), dtype=torch.float32)
    cfg = torch.tensor([1, 48, example.BLOCK_COL], dtype=torch.int64)
    return _jit_device_mlir(example.dyn_valid_shape_if_else, data, cfg, out)


@pytest.fixture(scope="module")
def jit_loop_mlir() -> str:
    """Codegen the @pl.jit loop + if/else kernel once for all tests in this module."""
    torch = pytest.importorskip("torch")
    example = _dyn_valid_shape_example()

    sij_buf = torch.zeros((example.N_ROW, example.BLOCK_COL), dtype=torch.float32)
    out = torch.zeros((example.N_ROW, example.BLOCK_COL), dtype=torch.float32)
    cfg = torch.tensor([2, 48, example.BLOCK_COL], dtype=torch.int64)
    return _jit_device_mlir(example.dyn_valid_shape_loop, sij_buf, cfg, out)


@pytest.fixture(scope="module")
def jit_scalar_param_mlir() -> str:
    """Codegen the @pl.jit scalar-parameter kernel (the specialized-constant form)."""
    torch = pytest.importorskip("torch")
    example = _dyn_valid_shape_example()

    data = torch.zeros((example.Q_TILE, example.BLOCK_COL), dtype=torch.float32)
    out = torch.zeros((example.Q_TILE, example.BLOCK_COL), dtype=torch.float32)
    return _jit_device_mlir(example.dyn_valid_shape, data, 2.0, 48, out)


def test_jit_if_else_survives_specialization(jit_if_else_mlir: str):
    """The in-DSL if/else must reach PTO as a real scf.if.

    Regression guard: the specializer previously alpha-renamed the else-branch
    rebinding of ``vlen``, which failed ConvertToSSA outright.
    """
    assert "scf.if" in jit_if_else_mlir, f"Expected scf.if in MLIR output:\n{jit_if_else_mlir}"


def test_jit_if_else_valid_col_is_runtime(jit_if_else_mlir: str):
    """The s_tile alloc's valid_col comes from the branch, not a folded constant.

    ``cfg`` is read at runtime, so neither branch value is known at
    specialization time and ``valid_col`` must be an SSA operand rather than a
    ``%c<n>`` literal.
    """
    alloc = _s_tile_alloc(jit_if_else_mlir)
    operand = _valid_col_operand(alloc)
    assert not re.fullmatch(r"%c\d+(_\w+)?", operand), (
        f"valid_col was constant-folded to {operand}; expected a runtime operand: {alloc}"
    )
    assert "v_col=?" in alloc, f"Expected dynamic v_col=? in s_tile alloc: {alloc}"


def test_jit_if_else_has_fillpad_with_pad_min(jit_if_else_mlir: str):
    """The padded tile keeps pad=3 (PadValue.min) through the JIT path."""
    assert "pto.tfillpad" in jit_if_else_mlir, f"Expected pto.tfillpad in MLIR output:\n{jit_if_else_mlir}"
    alloc_lines = [line.strip() for line in jit_if_else_mlir.split("\n") if "pto.alloc_tile" in line]
    padded_allocs = [line for line in alloc_lines if "s_padded" in line]
    assert len(padded_allocs) >= 1, f"Expected s_padded alloc, got alloc_lines: {alloc_lines}"
    assert "pad=3>" in padded_allocs[0], f"Expected pad=3 (PadValue.min) in padded alloc: {padded_allocs[0]}"


def test_jit_loop_has_scf_for_and_scf_if(jit_loop_mlir: str):
    """The runtime trip count and the per-iteration branch both survive."""
    assert "scf.for" in jit_loop_mlir, f"Expected scf.for in MLIR output:\n{jit_loop_mlir}"
    assert "scf.if" in jit_loop_mlir, f"Expected scf.if in MLIR output:\n{jit_loop_mlir}"


def test_jit_loop_valid_col_is_runtime(jit_loop_mlir: str):
    """The per-iteration valid_col is the scf.if result, not a constant."""
    alloc = _s_tile_alloc(jit_loop_mlir)
    operand = _valid_col_operand(alloc)
    assert not re.fullmatch(r"%c\d+(_\w+)?", operand), (
        f"valid_col was constant-folded to {operand}; expected a runtime operand: {alloc}"
    )


def test_jit_scalar_param_valid_col_is_constant(jit_scalar_param_mlir: str):
    """A scalar *parameter* is a specialization constant, so valid_col folds.

    Counterpart to :func:`test_jit_if_else_valid_col_is_runtime`: the
    specializer inlines scalar arguments at their use sites, so ``vlen=48``
    reaches codegen as a literal and each distinct value compiles its own
    kernel. Runtime selection requires reading the value from a tensor.
    """
    alloc = _s_tile_alloc(jit_scalar_param_mlir)
    operand = _valid_col_operand(alloc)
    assert re.fullmatch(r"%c48(_\w+)?", operand), (
        f"expected valid_col folded to the constant 48, got {operand}: {alloc}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
