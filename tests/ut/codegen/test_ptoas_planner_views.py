# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen under memory_planner=PTOAS: reserved buffers and view def/use types.

With the PTOAS planner, `MemoryReuse` + `AllocateMemoryAddr` are skipped and ptoas
`PlanMemory` owns on-chip placement (--pto-level=level2). Two things that the
default PyPTO planner hides then have to be emitted correctly:

* `system.reserve_buffer(base=AUTO)` never gets a resolved base, so PTO must emit
  ptoas's `auto = true` form (base absent) instead of the manual `base = <n>` one.
* A view chain (`tile.slice` -> `tile.reshape`) no longer folds into per-variable
  `pto.alloc_tile` re-views at one baked address, so it survives as a real
  `pto.subview` + `pto.treshape` pair whose def/use type strings must agree.
"""

import re

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


def _run_passes(program, planner: passes.MemoryPlanner):
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=planner):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _emit_pto(program, planner: passes.MemoryPlanner) -> str:
    """Run the default pipeline under `planner` and return the emitted PTO MLIR."""
    optimized = _run_passes(program, planner)
    emit_tile_addr = planner == passes.MemoryPlanner.PYPTO
    result = codegen.PTOCodegen().generate(optimized, emit_tile_addr=emit_tile_addr)
    return result if isinstance(result, str) else "".join(result.values())


def _emit_incore_pto(program, planner: passes.MemoryPlanner) -> str:
    """Same, for a program whose kernel is outlined into a single in-core function.

    `PTOCodegen.generate` only accepts in-core functions, so the Orchestration
    parent left behind by `pl.at` outlining has to be dropped first.
    """
    optimized = _run_passes(program, planner)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(incore) == 1, f"expected one in-core function, got {[f.name for f in incore]}"
    single = _ir.Program([incore[0]], incore[0].name, optimized.span)
    return codegen.PTOCodegen().generate(single, emit_tile_addr=planner == passes.MemoryPlanner.PYPTO)


def _sole_line(mlir: str, needle: str) -> str:
    lines = [ln for ln in mlir.splitlines() if needle in ln]
    assert len(lines) == 1, f"expected exactly one {needle!r} line, got {lines}:\n{mlir}"
    return lines[0]


# ── reserve_buffer: base resolution deferred to ptoas ────────────────────────


@pl.program
class AutoReserveBufferProgram:
    """Cross-core pipe whose slot buffers are declared with `base=AUTO`."""

    @pl.function(type=pl.FunctionType.AIV)
    def vector_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096)
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_producer")
        pl.aiv_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_buf, v2c_consumer_buf=v2c_peer
        )

        tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        pl.tpush_to_aic(tile_a, split=0)

        t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)
        out: pl.Tile[[16, 16], pl.FP32] = pl.exp(t)
        pl.tfree_to_aic(t)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(out, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.AIC)
    def cube_producer(self, arg: pl.Tensor[[16, 16], pl.FP32]):
        v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096)
        c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
        pl.aic_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buf
        )
        received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
        pl.tpush_to_aiv(received, split=0)
        pl.tfree_to_aiv(received)


def test_reserve_buffer_defers_base_to_ptoas():
    """PTOAS planner: `base` is never resolved, so emit ptoas's auto-placement form.

    ptoas rejects `auto = true` alongside a `base` attribute (and `auto = false`
    without one), so the two must move together.
    """
    mlir = _emit_pto(AutoReserveBufferProgram, passes.MemoryPlanner.PTOAS)
    for name in ("c2v_slot_buffer", "v2c_slot_buffer"):
        line = _sole_line(mlir, f'pto.reserve_buffer {{name = "{name}"')
        assert "auto = true" in line, line
        assert "base" not in line, line


def test_reserve_buffer_bakes_resolved_base_under_pypto_planner():
    """Default planner: AllocateMemoryAddr resolves `base`, emitted as manual mode."""
    mlir = _emit_pto(AutoReserveBufferProgram, passes.MemoryPlanner.PYPTO)
    for name in ("c2v_slot_buffer", "v2c_slot_buffer"):
        line = _sole_line(mlir, f'pto.reserve_buffer {{name = "{name}"')
        assert "auto = false" in line, line
        assert "base = 0" in line, line


# ── reshape of a subview: def/use tile_buf types must agree ──────────────────

PAD, VALID, D = 16, 5, 128


@pl.program
class SubviewReshapeProgram:
    """Slice the padded rows off a vec tile, then reshape the slice to one row."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[PAD, D], pl.FP32],
        out: pl.Out[pl.Tensor[[1, VALID * D], pl.FP32]],
    ) -> pl.Tensor[[1, VALID * D], pl.FP32]:
        t: pl.Tile[[PAD, D], pl.FP32] = pl.load(x, [0, 0], [PAD, D])
        v: pl.Tile[[VALID, D], pl.FP32] = pl.tile.slice(t, [VALID, D], [0, 0])
        r: pl.Tile[[1, VALID * D], pl.FP32] = pl.reshape(v, [1, VALID * D])
        return pl.store(r, [0, 0], out)


def _result_type(op_line: str) -> str:
    """The type right of `->` in an `op : <src> -> <dst>` annotation."""
    assert " -> " in op_line, f"expected a src -> dst annotation in: {op_line}"
    return op_line.split(" -> ", 1)[1].strip()


def _operand_type(op_line: str) -> str:
    """The type left of `->` in an `op : <src> -> <dst>` annotation."""
    assert " : " in op_line and " -> " in op_line, f"expected a src -> dst annotation in: {op_line}"
    return op_line.split(" : ", 1)[1].split(" -> ", 1)[0].strip()


def test_reshape_of_subview_annotates_the_subview_def_type():
    """A `pto.treshape` reading a `pto.subview` must annotate the subview's DEF type.

    `pto.subview` infers static valid dims (`v_row=5, v_col=128`) from its slice
    `sizes`, while every IR TileType renders as `v_row=?, v_col=?`. Deriving the
    treshape operand type from the TileType therefore prints `valid=?x?` at the
    use, and MLIR rejects the def/use mismatch.
    """
    mlir = _emit_pto(SubviewReshapeProgram, passes.MemoryPlanner.PTOAS)
    subview = _sole_line(mlir, "pto.subview")
    treshape = _sole_line(mlir, "pto.treshape")

    assert f"v_row={VALID}, v_col={D}" in _result_type(subview), subview
    assert _operand_type(treshape) == _result_type(subview), f"{subview}\n{treshape}"


def test_reshape_of_subview_folds_away_under_pypto_planner():
    """Default planner: the reshape result is pre-declared at the shared baked
    address, so it is a re-view and no `pto.treshape` is emitted at all."""
    mlir = _emit_pto(SubviewReshapeProgram, passes.MemoryPlanner.PYPTO)
    assert "pto.treshape" not in mlir, mlir


# ── reinterpret_view: byte-preserving treshape ──────────────────────────────

REINTERPRET_ROWS, REINTERPRET_COLS = 8, 16


@pl.program
class SameShapeReinterpretProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.INT32]],
    ) -> pl.Tensor[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.INT32]:
        src: pl.Tile[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.FP32] = pl.load(
            x, [0, 0], [REINTERPRET_ROWS, REINTERPRET_COLS]
        )
        view: pl.Tile[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.INT32] = pl.tile.reinterpret_view(
            src, pl.INT32
        )
        return pl.store(view, [0, 0], out)


@pl.program
class WidthChangingReinterpretProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[REINTERPRET_ROWS, REINTERPRET_COLS * 2], pl.INT16]],
    ) -> pl.Tensor[[REINTERPRET_ROWS, REINTERPRET_COLS * 2], pl.INT16]:
        src: pl.Tile[[REINTERPRET_ROWS, REINTERPRET_COLS], pl.FP32] = pl.load(
            x, [0, 0], [REINTERPRET_ROWS, REINTERPRET_COLS]
        )
        view: pl.Tile[[REINTERPRET_ROWS, REINTERPRET_COLS * 2], pl.INT16] = pl.tile.reinterpret_view(
            src, pl.INT16
        )
        return pl.store(view, [0, 0], out)


@pl.program
class SubviewReinterpretProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[16, REINTERPRET_COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[5, REINTERPRET_COLS], pl.INT32]],
    ) -> pl.Tensor[[5, REINTERPRET_COLS], pl.INT32]:
        src: pl.Tile[[16, REINTERPRET_COLS], pl.FP32] = pl.load(x, [0, 0], [16, REINTERPRET_COLS])
        sub: pl.Tile[[5, REINTERPRET_COLS], pl.FP32] = pl.tile.slice(src, [5, REINTERPRET_COLS], [0, 0])
        view: pl.Tile[[5, REINTERPRET_COLS], pl.INT32] = pl.tile.reinterpret_view(sub, pl.INT32)
        return pl.store(view, [0, 0], out)


def test_same_shape_reinterpret_uses_treshape_under_ptoas():
    mlir = _emit_pto(SameShapeReinterpretProgram, passes.MemoryPlanner.PTOAS)
    treshape = _sole_line(mlir, "pto.treshape")

    assert "pto.bitcast" not in mlir, mlir
    assert "dtype=f32" in _operand_type(treshape), treshape
    assert "dtype=i32" in _result_type(treshape), treshape
    assert f"rows={REINTERPRET_ROWS}, cols={REINTERPRET_COLS}" in _result_type(treshape), treshape


def test_width_changing_reinterpret_uses_treshape_under_ptoas():
    mlir = _emit_pto(WidthChangingReinterpretProgram, passes.MemoryPlanner.PTOAS)
    treshape = _sole_line(mlir, "pto.treshape")

    assert "pto.bitcast" not in mlir, mlir
    assert "dtype=f32" in _operand_type(treshape), treshape
    assert "dtype=i16" in _result_type(treshape), treshape
    assert f"rows={REINTERPRET_ROWS}, cols={REINTERPRET_COLS * 2}" in _result_type(treshape), treshape


def test_subview_reinterpret_treshape_uses_subview_definition_type():
    mlir = _emit_pto(SubviewReinterpretProgram, passes.MemoryPlanner.PTOAS)
    subview = _sole_line(mlir, "pto.subview")
    treshape = _sole_line(mlir, "pto.treshape")

    assert _operand_type(treshape) == _result_type(subview), f"{subview}\n{treshape}"
    assert "dtype=i32" in _result_type(treshape), treshape


@pytest.mark.parametrize(
    ("program", "target_dtype"),
    [(SameShapeReinterpretProgram, "dtype=i32"), (WidthChangingReinterpretProgram, "dtype=i16")],
)
def test_reinterpret_is_alloc_backed_alias_under_pypto_planner(program, target_dtype):
    mlir = _emit_pto(program, passes.MemoryPlanner.PYPTO)

    assert "pto.bitcast" not in mlir and "pto.treshape" not in mlir, mlir
    source_allocs = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln and "dtype=f32" in ln]
    target_allocs = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln and target_dtype in ln]
    assert len(source_allocs) == 1 and len(target_allocs) == 1, mlir
    source_alloc = source_allocs[0]
    target_alloc = target_allocs[0]
    source_addr = re.search(r"addr = ([^, ]+)", source_alloc)
    target_addr = re.search(r"addr = ([^, ]+)", target_alloc)
    assert source_addr is not None and target_addr is not None, f"{source_alloc}\n{target_alloc}"
    assert source_addr.group(1) == target_addr.group(1), f"{source_alloc}\n{target_alloc}"


# ── transposed matmul operand: the reinterpret needs its own SSA ─────────────

QM, QK, KN = 16, 128, 128


@pl.program
class MatmulBTransProgram:
    """`b_trans=True` views the Mat tile transposed, then tmovs it into Right."""

    @pl.function
    def kernel(
        self,
        q: pl.Tensor[[QM, QK], pl.BF16],
        k: pl.Tensor[[KN, QK], pl.BF16],
        out: pl.Out[pl.Tensor[[QM, KN], pl.FP32]],
    ) -> pl.Tensor[[QM, KN], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
            out[:, :] = pl.matmul(q[:, :], k[:, :], b_trans=True, out_dtype=pl.FP32)
        return out


def _tmov_into(mlir: str, dst_loc: str) -> str:
    """The single `pto.tmov` whose `outs(...)` targets a `dst_loc` tile_buf."""
    movs = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and dst_loc in ln.split("outs(", 1)[1]]
    assert len(movs) == 1, f"expected one tmov into {dst_loc}, got {movs}:\n{mlir}"
    return movs[0]


def test_transposed_matmul_operand_materializes_a_reinterpret_under_ptoas():
    """The transposed view must get its own SSA, and the tmov must read it.

    Under the PyPTO planner the view is a second `pto.alloc_tile` at the source's
    baked address carrying the transposed layout. The PTOAS planner bakes no
    address, so aliased vars collapse onto ONE tile_buf handle and that second
    declaration is never emitted — `tile.transpose_view` must then materialize the
    reinterpret as a `pto.treshape`. Otherwise the tmov annotated the *source*
    handle with the *transposed* layout, and MLIR rejected the def/use mismatch.
    """
    mlir = _emit_incore_pto(MatmulBTransProgram, passes.MemoryPlanner.PTOAS)
    treshape = _sole_line(mlir, "pto.treshape")

    # The reinterpret swaps blayout/slayout relative to its (mat) source.
    assert "blayout=col_major, slayout=row_major" in _operand_type(treshape), treshape
    assert "blayout=row_major, slayout=col_major" in _result_type(treshape), treshape

    # The tmov into the Right buffer reads the reinterpret, not the raw handle.
    reinterpret = treshape.split("=", 1)[0].strip()
    right_mov = _tmov_into(mlir, "loc=right")
    assert f"ins({reinterpret} " in right_mov, f"{treshape}\n{right_mov}"


def test_transposed_matmul_operand_is_a_re_view_under_pypto_planner():
    """Default planner: the transposed view owns an `alloc_tile` at the source's
    address, so no `pto.treshape` is needed and the tmov reads that decl."""
    mlir = _emit_incore_pto(MatmulBTransProgram, passes.MemoryPlanner.PYPTO)
    assert "pto.treshape" not in mlir, mlir

    right_mov = _tmov_into(mlir, "loc=right")
    src = right_mov.split("ins(", 1)[1].split(" ", 1)[0]
    decl = _sole_line(mlir, f"{src} = pto.alloc_tile")
    assert "blayout=row_major, slayout=col_major" in decl, decl


# ── pto.treshape results must carry STATIC valid dims ────────────────────────

COLVEC_ROWS = 16


@pl.program
class ColVectorMulProgram:
    """`[N, 1]` elementwise. pypto lowers it on the `[1, N]` row-major view, so both
    operands reach `tile.mul` through a reshape."""

    @pl.function
    def kernel(
        self,
        x: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        y: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]],
    ) -> pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="colvec_mul"):
            out[:, :] = pl.mul(x[:, :], y[:, :])
        return out


def _rows_cols(type_str: str) -> tuple[int, int]:
    """Extract (rows, cols) from a tile_buf type string."""
    m = re.search(r"rows=(\d+), cols=(\d+)", type_str)
    assert m is not None, f"no rows/cols in {type_str}"
    return int(m.group(1)), int(m.group(2))


def test_treshape_result_carries_static_valid_dims():
    """`pto.treshape` takes no valid_row / valid_col operands.

    ptoas builds the destination tile from the result type alone (an empty EmitC
    initializer) and `TRESHAPE_IMPL` only copies the address, never the valid
    extent. A `v_row=?, v_col=?` result therefore default-constructs to a valid
    extent of ZERO and every consumer silently writes nothing — this made a plain
    `[16, 1]` elementwise multiply return all zeros on device. A view result must
    render its valid dims statically, unlike `alloc_tile`, whose dynamic type is
    fine precisely because it passes explicit valid_row / valid_col operands.
    """
    mlir = _emit_incore_pto(ColVectorMulProgram, passes.MemoryPlanner.PTOAS)

    treshapes = [ln for ln in mlir.splitlines() if "pto.treshape" in ln]
    assert treshapes, f"the [N, 1] lowering must reshape onto the row-major view:\n{mlir}"
    # Both directions appear: [N, 1] -> [1, N] for the operands, [1, N] -> [N, 1]
    # for the result. Each view's valid extent must equal its own shape.
    seen_shapes = set()
    for ln in treshapes:
        result = _result_type(ln)
        assert "v_row=?" not in result and "v_col=?" not in result, (
            f"treshape result must carry static valid dims, got {result}\n{ln}"
        )
        rows, cols = _rows_cols(result)
        assert f"v_row={rows}, v_col={cols}" in result, (
            f"a [{rows}, {cols}] view must declare v_row={rows}, v_col={cols}:\n{ln}"
        )
        seen_shapes.add((rows, cols))
    assert (1, COLVEC_ROWS) in seen_shapes, f"expected the [1, N] row-major view:\n{mlir}"

    # alloc_tile handles keep the dynamic form — they carry explicit valid operands.
    for ln in mlir.splitlines():
        if "= pto.alloc_tile" in ln:
            assert "v_row=?, v_col=?" in ln, f"alloc_tile stays dynamic-valid:\n{ln}"
            assert "valid_row = " in ln and "valid_col = " in ln, ln


def test_colvec_reshape_folds_away_under_pypto_planner():
    """Default planner: the `[1, N]` view is a second alloc_tile at the source's
    baked address, so no `pto.treshape` is emitted at all."""
    mlir = _emit_incore_pto(ColVectorMulProgram, passes.MemoryPlanner.PYPTO)
    assert "pto.treshape" not in mlir, mlir
    assert "= pto.alloc_tile addr = " in mlir, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
