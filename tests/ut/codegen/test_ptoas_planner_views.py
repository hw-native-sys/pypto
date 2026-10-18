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
`PlanMemory` owns on-chip placement (--pto-level=level2). Contracts that the
default PyPTO planner hides then have to be emitted correctly:

* `system.reserve_buffer(base=AUTO)` never gets a resolved base, so PTO must emit
  ptoas's `auto = true` form (base absent) instead of the manual `base = <n>` one.
* A view chain (`tile.slice` -> `tile.reshape`) no longer folds into per-variable
  `pto.alloc_tile` re-views at one baked address, so it survives as a real
  `pto.subview` + `pto.treshape` pair whose def/use type strings must agree.
* Scratch-backed A2/A3 casts need static-valid views even when PlanMemory
  materializes the scratch implicitly from a one-input `pto.tcvt`.
"""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from _pto_loc_common import strip_loc
from pypto import ir as _ir
from pypto.backend import BackendType, pto_backend, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


def _run_passes(program, planner: passes.MemoryPlanner):
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    # These regressions exercise the legacy Tile emitter's planner-dependent
    # alias forms. Default Buffer views have separate selection/native tests.
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)


def _emit_pto(program, planner: passes.MemoryPlanner) -> str:
    """Run the default pipeline under `planner` and return the emitted PTO MLIR."""
    optimized = _run_passes(program, planner)
    emit_tile_addr = planner != passes.MemoryPlanner.PTOAS
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
    return codegen.PTOCodegen().generate(single, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)


def _sole_line(mlir: str, needle: str) -> str:
    """The unique line containing `needle`, without its trailing MLIR location.

    `_result_type` / `_operand_type` slice from the end of the line, so the
    `loc("file":line:col)` suffix codegen appends must come off here.
    """
    lines = [ln for ln in mlir.splitlines() if needle in ln]
    assert len(lines) == 1, f"expected exactly one {needle!r} line, got {lines}:\n{mlir}"
    return strip_loc(lines[0])


def _tile_buf_types(op_line: str) -> list[str]:
    """Return every tile_buf type annotation carried by one PTO operation."""
    return re.findall(r"!pto\.tile_buf<[^>]+>", op_line)


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


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_reserve_buffer_bakes_resolved_base_under_pypto_planner(planner):
    """PyPTO-owned planners resolve `base` and emit manual mode."""
    mlir = _emit_pto(AutoReserveBufferProgram, planner)
    for name in ("c2v_slot_buffer", "v2c_slot_buffer"):
        line = _sole_line(mlir, f'pto.reserve_buffer {{name = "{name}"')
        assert "auto = false" in line, line
        assert "base = 0" in line, line


# Shared in-place handle: the definition type is immutable


@pl.program
class InplaceFillPadProgram:
    """Fill a dynamically valid tile in place, then consume its shared handle."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[16, 16], pl.FP32],
        valid_rows: pl.Scalar[pl.INDEX],
        valid_cols: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        src: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16], valid_shape=[valid_rows, valid_cols])
        padded: pl.Tile[[16, 16], pl.FP32] = pl.tile.fillpad_inplace(src, pad_value=0)
        return pl.store(padded, [0, 0], out)


def _assert_fillpad_view(mlir: str, pad: int) -> None:
    """The input handle stays immutable; a typed alias carries the full result."""
    fillpad = _sole_line(mlir, "pto.tfillpad")
    src_type, dst_type = _tile_buf_types(fillpad)
    assert "pad=0>" in src_type, fillpad
    assert f"pad={pad}>" in dst_type, fillpad
    assert "v_row=?, v_col=?" in dst_type, fillpad
    src = re.search(r"ins\((%[\w.$]+)", fillpad)
    dst = re.search(r"outs\((%[\w.$]+)", fillpad)
    assert src and dst, fillpad
    assert src[1] != dst[1], fillpad
    view = _sole_line(mlir, f"{dst[1]} = pto.treshape")
    assert f"pto.treshape {src[1]} : {src_type} -> {dst_type}" in view, view
    initialized = f"pto.set_validshape {dst[1]}, %c64_index, %c64_index : {dst_type}"
    assert initialized in mlir and mlir.index(initialized) < mlir.index(fillpad.strip()), mlir
    store = _sole_line(mlir, "pto.tstore")
    assert f"ins({dst[1]} : {dst_type})" in store, store
    assert mlir.count("= pto.alloc_tile") == 1, mlir


@pytest.mark.ptoas_compile
def test_inplace_alias_preserves_output_view(tmp_path: Path):
    """Dynamic input validity must not erase the padded result's type (#2869)."""
    mlir = _emit_pto(InplaceFillPadProgram, passes.MemoryPlanner.PTOAS)
    fillpad = _sole_line(mlir, "pto.tfillpad")
    src_type, dst_type = _tile_buf_types(fillpad)
    assert "pad=0>" in src_type, fillpad
    assert "pad=1>" in dst_type, fillpad
    assert "v_row=?, v_col=?" in src_type, fillpad
    assert "v_row=?, v_col=?" in dst_type, fillpad
    view = _sole_line(mlir, "= pto.treshape")
    assert view.endswith(f"-> {dst_type}"), view
    dst = re.search(r"outs\((%[\w.$]+)", fillpad)
    assert dst and f"pto.set_validshape {dst[1]}, %c16_index, %c16_index" in mlir, mlir
    store = _sole_line(mlir, "pto.tstore")
    assert _tile_buf_types(store) == [dst_type], store
    assert mlir.count("= pto.alloc_tile") == 1, mlir
    _compile_fillpad(mlir, tmp_path, passes.MemoryPlanner.PTOAS)


def _compile_fillpad(mlir: str, tmp_path: Path, planner: passes.MemoryPlanner) -> None:
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    cpp = pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path), memory_planner=planner)
    assert "TFILLPAD<pto::TFillPadMode::InPlace>" in cpp, cpp


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("pad_value", [pl.PadValue.zero, pl.PadValue.max, pl.PadValue.min])
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]
)
def test_fillpad_inplace_native(
    tmp_path: Path, pad_value: pl.PadValue, backend_type: BackendType, planner: passes.MemoryPlanner
):
    """The original partial-row fill compiles to the in-place ISA form."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[48, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
            padded = pl.tile.fillpad_inplace(src, pad_value=pad_value)
            return pl.store(padded, [0, 0], out)

    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    if planner == passes.MemoryPlanner.PTOAS:
        _assert_fillpad_view(mlir, pad_value.value)
    _compile_fillpad(mlir, tmp_path, planner)


@pl.program
class FillpadSubviewProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[128, 64], pl.FP32],
        rows: pl.Scalar[pl.INDEX],
        cols: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        src = pl.load(x, [0, 0], [128, 64])
        window = pl.tile.slice(src, [64, 64], [32, 0], valid_shape=[rows, cols])
        padded = pl.tile.fillpad_inplace(window, pad_value=pl.PadValue.zero)
        alias = padded
        return pl.store(alias, [0, 0], out)


@pl.program
class FillpadLoopProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[16, 16], pl.FP32],
        rows: pl.Scalar[pl.INDEX],
        cols: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        src = pl.load(x, [0, 0], [16, 16], valid_shape=[rows, cols])
        seed = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
        for i, (carry,) in pl.range(2, init_values=(seed,)):
            padded = pl.tile.fillpad_inplace(carry, pad_value=pl.PadValue.zero)
            result = pl.yield_(padded)
        return pl.store(result, [0, 0], out)


@pl.program
class FillpadBranchProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[16, 16], pl.FP32],
        rows: pl.Scalar[pl.INDEX],
        cols: pl.Scalar[pl.INDEX],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        src = pl.load(x, [0, 0], [16, 16], valid_shape=[rows, cols])
        if rows < 8:
            padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
            result = pl.yield_(padded)
        else:
            narrowed = pl.tile.set_validshape(src, 8, cols)
            other = pl.tile.fillpad_inplace(narrowed, pad_value=pl.PadValue.zero)
            result = pl.yield_(other)
        return pl.store(result, [0, 0], out)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("program", [FillpadSubviewProgram, FillpadLoopProgram, FillpadBranchProgram])
def test_fillpad_inplace_composed_views(tmp_path: Path, program):
    """Rebound padded handles survive subviews, valid updates, and control flow."""
    mlir = _emit_pto(program, passes.MemoryPlanner.PTOAS)
    fillpads = [line for line in mlir.splitlines() if "pto.tfillpad" in line]
    assert fillpads, mlir
    for line in fillpads:
        dst_type = _tile_buf_types(line)[1]
        assert "pad=1>" in dst_type, line
        if "v_row=?" in dst_type:
            dst = re.search(r"outs\((%[\w.$]+)", line)
            assert dst and f"pto.set_validshape {dst[1]}, %c16_index, %c16_index" in mlir, mlir
        else:
            assert "v_col=?" not in dst_type, line
    _compile_fillpad(mlir, tmp_path, passes.MemoryPlanner.PTOAS)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("dynamic_bounds", [False, True])
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]
)
def test_fillpad_inplace_then_set_validshape(tmp_path: Path, dynamic_bounds, backend_type, planner):
    """Narrowing the padded handle must also affect uses of its original alias."""

    @pl.program
    class DynamicBounds:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[48, 64], pl.FP32],
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
            padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
            _narrowed = pl.tile.set_validshape(padded, rows, cols)
            # Use the pre-update alias: set_validshape mutates that same handle.
            repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
            return pl.store(repadded, [0, 0], out)

    @pl.program
    class StaticBounds:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[48, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
            padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
            _narrowed = pl.tile.set_validshape(padded, 32, 48)
            repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
            return pl.store(repadded, [0, 0], out)

    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            DynamicBounds if dynamic_bounds else StaticBounds
        )
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    fillpads = [line for line in mlir.splitlines() if "pto.tfillpad" in line]
    assert len(fillpads) == 2, mlir
    first_dst = re.search(r"outs\((%[\w.$]+)", fillpads[0])
    second_src = re.search(r"ins\((%[\w.$]+)", fillpads[1])
    assert first_dst and second_src and first_dst[1] == second_src[1], mlir
    assert "v_row=?, v_col=?" in _tile_buf_types(fillpads[0])[1], mlir
    assert f"pto.set_validshape {first_dst[1]}," in mlir, mlir
    _compile_fillpad(mlir, tmp_path, planner)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]
)
def test_fillpad_branch_result_then_set_validshape(tmp_path: Path, backend_type, planner):
    """A phi buffer declared at function entry must retain mutable validity."""

    @pl.program
    class BranchResult:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[48, 64], pl.FP32],
            rows: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            if rows < 32:
                left = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
                merged = pl.yield_(left)
            else:
                right = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
                merged = pl.yield_(right)
            padded = pl.tile.fillpad_inplace(merged, pad_value=pl.PadValue.zero)
            _narrowed = pl.tile.set_validshape(padded, rows, 48)
            repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.max)
            return pl.store(repadded, [0, 0], out)

    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(BranchResult)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert "scf.if" in mlir, mlir
    fillpads = [line for line in mlir.splitlines() if "pto.tfillpad" in line]
    assert len(fillpads) == 2, mlir
    first_dst = re.search(r"outs\((%[\w.$]+)", fillpads[0])
    second_src = re.search(r"ins\((%[\w.$]+)", fillpads[1])
    assert first_dst and second_src and first_dst[1] == second_src[1], mlir
    assert "v_row=?, v_col=?" in _tile_buf_types(fillpads[0])[1], mlir
    _compile_fillpad(mlir, tmp_path, planner)


# Reshape of a subview: def/use tile_buf types must agree

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


@pl.program
class NonzeroSubviewProgram:
    """A nonzero row slice whose MemRef keeps a relative byte offset.

    Its only consumer runs `init_mem_ref` directly, without `InferTileMemorySpace`,
    so the load pins its own memory space instead of relying on the pass to place it.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[8, 8], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 8], pl.FP32]],
    ) -> pl.Tensor[[2, 8], pl.FP32]:
        src: pl.Tile[[8, 8], pl.FP32] = pl.load(x, [0, 0], [8, 8], target_memory=pl.MemorySpace.Vec)
        sub: pl.Tile[[2, 8], pl.FP32] = pl.tile.slice(src, [2, 8], [3, 0])
        return pl.store(sub, [0, 0], out)


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


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP])
def test_reshape_of_subview_folds_away_under_pypto_planner(planner):
    """PyPTO-owned planners pre-declare the result at the shared baked
    address, so it is a re-view and no `pto.treshape` is emitted at all."""
    mlir = _emit_pto(SubviewReshapeProgram, planner)
    assert "pto.treshape" not in mlir, mlir


def test_dsa_rp_writeback_preserves_nonzero_view_offset():
    """Physical placement shifts a view by its original relative byte offset."""

    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.DSA_RP):
        optimized = passes.allocate_memory_addr()(
            passes.materialize_semantic_aliases()(passes.init_mem_ref()(NonzeroSubviewProgram))
        )
    function = next(f for f in optimized.functions.values() if f.name == "kernel")
    offsets: dict[str, int] = {}

    class _Collector(_ir.IRVisitor):
        def visit_assign_stmt(self, stmt):  # type: ignore[override]
            tile_type = stmt.var.type
            if isinstance(tile_type, _ir.TileType) and tile_type.memref is not None:
                offset = tile_type.memref.byte_offset_
                assert isinstance(offset, _ir.ConstInt)
                offsets[stmt.var.name_hint] = offset.value
            super().visit_assign_stmt(stmt)

    _Collector().visit_stmt(function.body)
    assert offsets["sub"] - offsets["src"] == 3 * 8 * 4


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


# ── a transposed operand may not come from a Mat *window* ────────────────────

PARENT_ROWS, WINDOW_ROWS = 512, 64


@pl.program
class MatmulBTransOverSliceProgram:
    """`b_trans=True` on a `tile.slice` of a Mat-resident parent.

    The window is selected per iteration, so its offset is a runtime value and the
    slice reaches codegen as a `pto.subview`.
    """

    @pl.function
    def kernel(
        self,
        query: pl.Tensor[[PARENT_ROWS, QK], pl.INT8],
        key: pl.Tensor[[QM, QK], pl.INT8],
        out: pl.Out[pl.Tensor[[QM, WINDOW_ROWS], pl.INT32]],
    ) -> pl.Tensor[[QM, WINDOW_ROWS], pl.INT32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_window"):
            query_all = pl.slice(query, [PARENT_ROWS, QK], [0, 0])
            for q in pl.range(PARENT_ROWS // WINDOW_ROWS):
                window = pl.slice(query_all, [WINDOW_ROWS, QK], [q * WINDOW_ROWS, 0])
                out[:, :] = pl.matmul(key[:, :], window, out_dtype=pl.INT32, b_trans=True)
        return out


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_transposed_matmul_operand_over_a_mat_slice_is_rejected(planner):
    """A transposed view of a Mat window has no lowering, and must not be aliased.

    `tile.transpose_view` is a zero-copy relabel of a whole buffer. Its source
    here is a `pto.subview`, which carries a runtime offset and the *parent's* row
    pitch — and neither of the op's two lowerings can express that. ptoas refuses
    a mat-source `pto.tmov` on a view ("matching src/dst shapes") and refuses a
    `pto.treshape` on one at every destination size ("same total byte size").

    Left alone, the PyPTO planner took the no-op branch and the result's own
    `pto.alloc_tile` landed at the *parent's base* with the *window's* extent: a
    dynamic slice offset cannot fold into a constant `addr`, and the parent pitch
    is absent from the alloc's type. The matmul then read the same wrong bytes on
    every iteration, silently. Reject at codegen instead, under both planners.
    """
    with pytest.raises(ValueError, match="cannot be taken from a slice of an on-chip Mat tile"):
        _emit_incore_pto(MatmulBTransOverSliceProgram, planner)


@pl.program
class MatmulBTransOverIdentitySliceProgram:
    """`b_trans=True` on a slice covering its Mat parent's FULL extent at [0, 0].

    Same bytes, same pitch, same address as the parent — the one window a
    transpose loses nothing on.
    """

    @pl.function
    def kernel(
        self,
        query: pl.Tensor[[KN, QK], pl.INT8],
        key: pl.Tensor[[QM, QK], pl.INT8],
        out: pl.Out[pl.Tensor[[QM, KN], pl.INT32]],
    ) -> pl.Tensor[[QM, KN], pl.INT32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk_identity"):
            query_all = pl.slice(query, [KN, QK], [0, 0])
            key_tile = pl.slice(key, [QM, QK], [0, 0])
            window = pl.slice(query_all, [KN, QK], [0, 0])
            out[:, :] = pl.matmul(key_tile, window, out_dtype=pl.INT32, b_trans=True)
        return out


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_transposed_matmul_operand_over_an_identity_mat_slice_is_folded(planner):
    """An identity window keeps working, and folds back to its parent.

    A full-extent slice at a constant `[0, 0]` offset names exactly the source's
    bytes with the source's own pitch, so the guard above must not reject it --
    `FlattenTileNdTo2D` emits one per page for a leading-dim-1 batch, making this
    ordinary input rather than a degenerate case.

    Folding to the parent (rather than merely exempting the window) is what keeps
    the PTOAS planner correct too: with no baked address the transpose falls back
    to `pto.treshape`, which ptoas refuses against a `pto.subview` source however
    identity it is, but accepts against the parent's dense handle.
    """
    mlir = _emit_incore_pto(MatmulBTransOverIdentitySliceProgram, planner)

    right_mov = _tmov_into(mlir, "loc=right")
    transposed = "blayout=row_major, slayout=col_major"

    if planner is passes.MemoryPlanner.PTOAS:
        treshape = _sole_line(mlir, "pto.treshape")
        # Reads the parent's own handle, never the `pto.subview` SSA.
        source = treshape.split("pto.treshape ", 1)[1].split(" ", 1)[0]
        assert not source.startswith("%slice_view"), (
            f"the identity window must fold back to its parent, got {source}:\n{mlir}"
        )
        assert "blayout=col_major, slayout=row_major" in _operand_type(treshape), treshape
        assert transposed in _result_type(treshape), treshape
        assert f"ins({treshape.split('=', 1)[0].strip()} " in right_mov, f"{treshape}\n{right_mov}"
    else:
        # Default planner: the transposed alias is a second alloc_tile, as for a
        # whole Mat load — no reinterpret is needed at all.
        assert "pto.treshape" not in mlir, mlir
        src = right_mov.split("ins(", 1)[1].split(" ", 1)[0]
        assert transposed in _sole_line(mlir, f"{src} = pto.alloc_tile"), mlir


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


# ── lane-1 replay sentinel: a static zero valid has no pto-isa overload ──────

SENTINEL_ROWS, SENTINEL_COLS = 16, 8


@pl.program
class ZeroValidSentinelReinterpretProgram:
    """A view over SplitVectorKernel's lane-1 sentinel, whose valid dims are both 0.

    `WithZeroValidShape` stamps that `[0, 0]` onto every cloned lane-1 op, so any
    view taken in the replay lane reaches codegen with an all-zero valid_shape.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        out: pl.Out[pl.Tensor[[SENTINEL_ROWS, SENTINEL_COLS], pl.INT32]],
    ) -> pl.Tensor[[SENTINEL_ROWS, SENTINEL_COLS], pl.INT32]:
        sentinel: pl.Tile[
            [SENTINEL_ROWS, SENTINEL_COLS], pl.FP32, pl.MemorySpace.Vec, pl.TileView(valid_shape=[0, 0])
        ] = pl.tile.create([SENTINEL_ROWS, SENTINEL_COLS], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        view: pl.Tile[[SENTINEL_ROWS, SENTINEL_COLS], pl.INT32] = pl.tile.reinterpret_view(sentinel, pl.INT32)
        return pl.store(view, [0, 0], out)


def test_zero_valid_sentinel_treshape_keeps_a_dynamic_valid():
    """Regression for #2870: a `pto.treshape` view must not bake the lane-1 zero.

    ptoas accepts a static `v_row=0, v_col=0` and default-constructs the tile from
    the result type, which lands in the generated C++ as `pto::Tile<..., 0, 0, ...>`.
    pto-isa declares `GetValidRow` / `GetValidCol` only for a static mask `> 0` and
    for `DYNAMIC`, so that instantiation matches no overload and ccec cannot compile
    it. Rendering the valid dynamic gives a runtime-zero extent that compiles, and
    the replay lane discards the result either way.

    The physical `rows` / `cols` must stay static — only the valid extent defers.
    """
    mlir = _emit_pto(ZeroValidSentinelReinterpretProgram, passes.MemoryPlanner.PTOAS)
    treshape = _sole_line(mlir, "pto.treshape")
    result = _result_type(treshape)

    assert "v_row=0" not in result and "v_col=0" not in result, (
        "the lane-1 [0, 0] sentinel must not become a static zero valid on the "
        f"treshape result (pto-isa has no GetValidRow overload for it); got:\n{treshape}"
    )
    assert "v_row=?" in result and "v_col=?" in result, (
        f"expected a dynamic valid on the sentinel view; got:\n{treshape}"
    )
    assert f"rows={SENTINEL_ROWS}, cols={SENTINEL_COLS}" in result, (
        f"the physical shape must stay static; got:\n{treshape}"
    )


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
def test_fillpad_same_mode_after_validshape_update(tmp_path: Path, backend_type: BackendType):
    """Equal tile types must not merge independently initialized valid extents."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[48, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            src = pl.load(x, [0, 0], [64, 64], valid_shape=[48, 64])
            padded = pl.tile.fillpad_inplace(src, pad_value=pl.PadValue.zero)
            _narrowed = pl.tile.set_validshape(padded, 32, 48)
            # The source handle is still narrowed even though its SSA type did
            # not change. A second zero fill needs its own full-valid result.
            repadded = pl.tile.fillpad_inplace(padded, pad_value=pl.PadValue.zero)
            return pl.store(repadded, [0, 0], out)

    reset_for_testing()
    set_backend_type(backend_type)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Program)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=False)
    fillpads = [strip_loc(line) for line in mlir.splitlines() if "pto.tfillpad" in line]
    assert len(fillpads) == 2, mlir
    first_dst = re.search(r"outs\((%[\w.$]+)", fillpads[0])
    second_src = re.search(r"ins\((%[\w.$]+)", fillpads[1])
    second_dst = re.search(r"outs\((%[\w.$]+)", fillpads[1])
    assert first_dst and second_src and second_dst, mlir
    assert first_dst[1] == second_src[1] != second_dst[1], mlir
    src_type, dst_type = _tile_buf_types(fillpads[1])
    assert src_type == dst_type, fillpads[1]
    assert "pad=1>" in dst_type and "v_row=?, v_col=?" in dst_type, dst_type
    narrowed = f"pto.set_validshape {second_src[1]}, %c32_index, %c48_index : {src_type}"
    initialized = f"pto.set_validshape {second_dst[1]}, %c64_index, %c64_index : {dst_type}"
    assert narrowed in mlir and initialized in mlir, mlir
    assert mlir.index(narrowed) < mlir.index(initialized) < mlir.index(fillpads[1]), mlir
    assert f"ins({second_dst[1]} : {dst_type})" in _sole_line(mlir, "pto.tstore"), mlir
    assert mlir.count("= pto.alloc_tile") == 1, mlir
    _compile_fillpad(mlir, tmp_path, passes.MemoryPlanner.PTOAS)


@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950], ids=["a2a3", "a5"])
@pytest.mark.parametrize("planner", list(passes.MemoryPlanner), ids=lambda planner: planner.name)
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [(pl.FP32, pl.INT16), (pl.FP16, pl.INT16), (pl.FP16, pl.INT8), (pl.FP16, pl.UINT8)],
    ids=["f32-i16", "f16-i16", "f16-i8", "f16-u8"],
)
@pytest.mark.parametrize("saturation_mode", ["off", "on"])
@pytest.mark.parametrize("rows,valid_rows,valid_cols", [(1, 1, 128), (4, 3, 96)], ids=["full", "partial"])
def test_tcvt_planner_scratch_views(
    tmp_path, backend_type, planner, src_dtype, dst_dtype, saturation_mode, rows, valid_rows, valid_cols
):
    """Implicit scratch needs the same static validity as explicit level-3 scratch.

    The original regression is the full f16-i8/OFF case on A2/A3 with PTOAS.
    Partial tiles distinguish actual validity from physical capacity; neither
    padding nor a larger physical extent may become input to the conversion.
    """

    @pl.program
    class CastProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, src: pl.Tensor[[rows, 128], src_dtype], out: pl.Out[pl.Tensor[[rows, 128], dst_dtype]]
        ):
            tile = pl.load(src, [0, 0], [rows, 128], valid_shape=[valid_rows, valid_cols])
            converted = pl.cast(tile, dst_dtype, mode="trunc", saturation_mode=saturation_mode)
            return pl.store(converted, [0, 0], out)

    native = find_ptoas_binary() is not None
    with passes.PassContext([], memory_planner=planner):
        _ir.compile(
            CastProgram,
            output_dir=str(tmp_path),
            backend_type=backend_type,
            skip_ptoas=not native,
            dump_passes=False,
        )
    # Without PTOAS, compile writes raw MLIR as the kernel output rather than
    # an assembler intermediate. Keep the structural assertions in both modes.
    mlir_path = "ptoas/kernel.pto" if native else "kernels/aiv/kernel.pto"
    mlir = (tmp_path / mlir_path).read_text()
    tcvt = _sole_line(mlir, "pto.tcvt")
    needs_scratch = backend_type == BackendType.Ascend910B and saturation_mode == "off"
    explicit = needs_scratch and planner != passes.MemoryPlanner.PTOAS
    assert ("tcvt_tmp_view" in tcvt) == explicit, tcvt
    assert ("tcvt_src_view" in tcvt) == needs_scratch, tcvt
    assert mlir.count("pto.alloc_tile") == (3 if explicit else 2), mlir
    assert (" addr = " in mlir) == (planner != passes.MemoryPlanner.PTOAS), mlir
    if needs_scratch:
        assert "v_row=?" not in tcvt and "v_col=?" not in tcvt, tcvt
        types = _tile_buf_types(tcvt)
        for tile_type in (types[0], types[-1]):
            assert f"v_row={valid_rows}, v_col={valid_cols}" in tile_type, tcvt
    if native:
        assert "TCVT(" in (tmp_path / "ptoas/kernel.cpp").read_text()


@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PTOAS, passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP]
)
@pytest.mark.parametrize("valid_cols", [128, 96])
def test_tcvt_rank_one_with_explicit_scratch(tmp_path: Path, planner, valid_cols):
    """Caller-owned scratch must not inherit compiler scratch's 2D restriction."""

    @pl.program
    class CastProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[128], pl.FP16],
            out: pl.Out[pl.Tensor[[1, 128], pl.INT8]],
        ) -> pl.Tensor[[1, 128], pl.INT8]:
            tile = pl.tile.load(src, [0], [128], valid_shape=[valid_cols], target_memory=pl.Mem.Vec)
            # Deliberately larger and a different dtype than synthesized INT8 scratch.
            tmp = pl.tile.create([1, 256], dtype=pl.FP32, target_memory=pl.Mem.Vec)
            converted = pl.tile.cast(tile, pl.INT8, mode="trunc", tmp=tmp, saturation_mode="off")
            # Store requires 2D validity; normalize only after the rank-one cast.
            converted_2d = pl.tile.reshape(converted, [1, 128])
            return pl.tile.store(converted_2d, [0, 0], out)

    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    # Keep rank-one tiles at codegen entry; the default pipeline would first
    # normalize their geometry and conceal this explicit-scratch regression.
    with passes.PassContext([], memory_planner=planner):
        optimized = passes.init_mem_ref()(CastProgram)
        if planner != passes.MemoryPlanner.PTOAS:
            optimized = passes.allocate_memory_addr()(optimized)

    casts: list[_ir.Call] = []

    class CastCollector(_ir.IRVisitor):
        def visit_call(self, op: _ir.Call) -> None:
            if op.op.name == _ir.get_op("tile.cast").name:
                casts.append(op)
            super().visit_call(op)

    CastCollector().visit_program(optimized)
    assert len(casts) == 1 and len(casts[0].args) == 2
    source_type = casts[0].args[0].type
    assert isinstance(source_type, _ir.TileType) and source_type.shape == [128]
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    tcvt = _sole_line(mlir, "pto.tcvt")
    assert "tcvt_tmp_view" in tcvt, tcvt
    scratch_allocs = [line for line in mlir.splitlines() if "pto.alloc_tile" in line and "dtype=f32" in line]
    assert len(scratch_allocs) == 1, mlir
    src_type, tmp_type, dst_type = _tile_buf_types(tcvt)
    for tile_type in (src_type, dst_type):
        assert "rows=1, cols=128" in tile_type, tcvt
        assert f"v_row=1, v_col={valid_cols}" in tile_type, tcvt
    assert "dtype=f32" in tmp_type and "rows=1, cols=256" in tmp_type, tcvt
    assert "v_row=1, v_col=256" in tmp_type, tcvt
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    cpp = pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path), memory_planner=planner)
    assert "TCVT(" in cpp, cpp


@pytest.mark.parametrize(
    "backend_type,dst_dtype,saturation_mode",
    [
        (BackendType.Ascend910B, pl.INT8, "on"),
        (BackendType.Ascend910B, pl.FP32, "off"),
        (BackendType.Ascend950, pl.INT8, "off"),
        (BackendType.Ascend910B, pl.INT8, "off"),
    ],
    ids=["a2a3-saturating", "a2a3-widening", "a5-nonsaturating", "a2a3-scratch"],
)
def test_tcvt_dynamic_validity_contract(tmp_path, backend_type, dst_dtype, saturation_mode):
    """Casts without A2/A3 scratch must not acquire a static-validity restriction."""

    @pl.program
    class DynamicCast:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tensor[[4, 128], pl.FP16],
            out: pl.Out[pl.Tensor[[4, 128], dst_dtype]],
            cols: pl.Scalar[pl.INDEX],
        ):
            tile = pl.load(src, [0, 0], [4, 128], valid_shape=[4, cols])
            converted = pl.cast(tile, dst_dtype, mode="trunc", saturation_mode=saturation_mode)
            return pl.store(converted, [0, 0], out)

    if backend_type == BackendType.Ascend910B and dst_dtype == pl.INT8 and saturation_mode == "off":
        # Runtime validity cannot size A2/A3's implicit scratch. Fail at the
        # source cast with a useful diagnostic, rather than inside PTOAS.
        with pytest.raises(ValueError, match="requires compile-time static valid_shape"):
            _emit_pto(DynamicCast, passes.MemoryPlanner.PTOAS)
        return

    native = find_ptoas_binary() is not None
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        _ir.compile(
            DynamicCast,
            output_dir=str(tmp_path),
            backend_type=backend_type,
            skip_ptoas=not native,
            dump_passes=False,
        )
    mlir_path = "ptoas/kernel.pto" if native else "kernels/aiv/kernel.pto"
    mlir = (tmp_path / mlir_path).read_text()
    tcvt = _sole_line(mlir, "pto.tcvt")
    assert "v_col=?" in tcvt and "tcvt_src_view" not in tcvt, tcvt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
