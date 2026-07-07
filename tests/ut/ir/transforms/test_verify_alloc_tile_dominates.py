# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the AllocTileDominatesUses structural property verifier (issue #1956).

The verifier enforces that after MaterializeAllocTiles every PTO tile buffer is
declared by an explicit ``alloc_tile`` op at a scope that dominates all its uses,
so no branch-local handle is read from another branch (the pre-#1956 miscompile
under ``memory_planner=PTOAS``).
"""

# DSL function bodies are parsed as AST, not executed — suppress pyright errors.
# pyright: reportUndefinedVariable=false, reportMissingImports=false

import pypto.language as pl
import pytest
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager

from pypto import DataType, backend, ir, passes


def _alloc_tile_props() -> passes.IRPropertySet:
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AllocTileDominatesUses)
    return props


def _errors(diagnostics: list[passes.Diagnostic]) -> list[passes.Diagnostic]:
    return [d for d in diagnostics if d.severity == passes.DiagnosticSeverity.Error]


def test_pipeline_output_satisfies_alloc_tile_dominance():
    """A real if/else-yield (phi) kernel through the Default pipeline — which runs
    MaterializeAllocTiles — must satisfy AllocTileDominatesUses: every tile buffer,
    including the phi buffer written across both branches, has a dominating
    alloc_tile handle."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)

    @pl.program
    class IfPhi:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            c: pl.Scalar[pl.BOOL],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            ta: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Vec)
            if c:
                r1: pl.Tile[[64, 64], pl.FP32] = pl.mul(ta, ta)
                res: pl.Tile[[64, 64], pl.FP32] = pl.yield_(r1)
            else:
                r2: pl.Tile[[64, 64], pl.FP32] = pl.add(ta, ta)
                res: pl.Tile[[64, 64], pl.FP32] = pl.yield_(r2)
            out: pl.Tensor[[64, 64], pl.FP32] = pl.store(res, [0, 0], output)
            return out

    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    program = pm.run_passes(IfPhi)
    diags = passes.PropertyVerifierRegistry.verify(_alloc_tile_props(), program)
    assert len(_errors(diags)) == 0, [d.message for d in _errors(diags)]


def test_tile_use_without_alloc_tile_is_flagged():
    """A tile buffer that is loaded and stored with no alloc_tile handle at all has
    no dominating handle, so the verifier must flag the use."""
    span = ir.Span.unknown()
    zero = ir.ConstInt(0, DataType.INDEX, span)
    size = ir.ConstInt(64, DataType.INDEX, span)
    byte_offset = ir.ConstInt(0, DataType.INT64, span)

    input_tensor = ir.Var("a", ir.TensorType([64, 64], DataType.FP32), span)
    output_tensor = ir.Var("output", ir.TensorType([64, 64], DataType.FP32), span)
    memref = ir.MemRef(ir.MemorySpace.Vec, byte_offset, 64 * 64 * 4, 0)
    tile_type = ir.TileType([64, 64], DataType.FP32, memref, None, ir.MemorySpace.Vec)
    tile_a = ir.Var("tile_a", tile_type, span)
    result_var = ir.Var("result", ir.TensorType([64, 64], DataType.FP32), span)

    offsets = ir.MakeTuple([zero, zero], span)
    shapes = ir.MakeTuple([size, size], span)
    load_call = ir.Call(ir.Op("tile.load"), [input_tensor, offsets, shapes], {}, tile_type, span)
    store_call = ir.Call(ir.Op("tile.store"), [tile_a, offsets, output_tensor], result_var.type, span)

    body = ir.SeqStmts(
        [
            ir.AssignStmt(tile_a, load_call, span),
            ir.AssignStmt(result_var, store_call, span),
            ir.ReturnStmt([result_var], span),
        ],
        span,
    )
    func = ir.Function(
        "no_alloc_tile",
        [(input_tensor, ir.ParamDirection.In), (output_tensor, ir.ParamDirection.Out)],
        [ir.TensorType([64, 64], DataType.FP32)],
        body,
        span,
        ir.FunctionType.InCore,
    )
    program = ir.Program([func], "no_alloc_tile_program", span)

    errors = _errors(passes.PropertyVerifierRegistry.verify(_alloc_tile_props(), program))
    assert len(errors) >= 1, "Expected AllocTileDominatesUses to flag the un-materialized tile buffer"
    assert "tile_a" in errors[0].message and "alloc_tile" in errors[0].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
