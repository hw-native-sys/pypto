# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTOAS loop-carry storage and explicit shared-storage move regressions.

Check accumulator storage, carry write-back, and each move operand's descriptor.
Native If selection is covered by the native If transform tests.
"""

# pyright: reportUndefinedVariable=false

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir as _ir
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes


def _ptoas_pto(program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span), emit_tile_addr=False)


def _planner_pto(program, planner: passes.MemoryPlanner) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    return codegen.PTOCodegen().generate(
        _ir.Program([func], "kernel", optimized.span),
        emit_tile_addr=planner == passes.MemoryPlanner.PYPTO,
    )


def _sole_line(mlir: str, needle: str) -> str:
    lines = [ln for ln in mlir.splitlines() if needle in ln]
    assert len(lines) == 1, f"expected exactly one {needle!r} line, got {lines}:\n{mlir}"
    return lines[0]


def _first_group(pattern: str, text: str) -> str:
    m = re.search(pattern, text)
    assert m is not None, f"pattern {pattern!r} not found in:\n{text}"
    return m.group(1)


@pl.program
class LoopCarriedIfPhiProgram:
    """A `pl.range` accumulator whose per-iteration update is an if/else. The loop
    init, the iter_arg, the if-phi and the loop result all denote one buffer, so
    under PTOAS they must all collapse onto a single `tile_buf` handle."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[64, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        acc: pl.Tile[[64, 64], pl.FP32] = pl.load(x, [0, 0], [64, 64])
        for i in pl.range(4):  # noqa: B007 - loop index drives the branch
            if i == 0:
                acc = pl.add(acc, 1.0)
            else:
                acc = pl.mul(acc, 2.0)
        return pl.store(acc, [0, 0], out)


def test_loop_carried_if_phi_shares_the_accumulator_handle():
    """The if-phi must reuse the accumulator's handle, not declare a second buffer.

    Under PTOAS no `addr` is baked, so two addr-less `pto.alloc_tile` are two
    independent buffers to ptoas PlanMemory. Head-declaring a fresh handle for the
    phi therefore stranded the accumulator: the branch producers wrote the phi
    buffer while the loop-carried read and the post-loop store used the accumulator
    handle, which no branch ever wrote — a silent miscompile (issue: the loop-carried
    counterpart of the `scf.if` phi fix #1956/#1985).
    """
    mlir = _ptoas_pto(LoopCarriedIfPhiProgram)

    store = next(ln for ln in mlir.splitlines() if "pto.tstore" in ln)
    acc = _first_group(r"pto\.tstore ins\((%[A-Za-z0-9_]+)", store)

    # Exactly one tile_buf is declared for the accumulator, and nothing else.
    decls = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln]
    assert len(decls) == 1, f"expected a single alloc_tile, got {len(decls)}:\n{mlir}"
    assert f"{acc} = pto.alloc_tile" in decls[0], f"the sole alloc_tile must be {acc}:\n{mlir}"

    # Both branch producers write that handle, and both read it back (the carry).
    producers = [ln for ln in mlir.splitlines() if "pto.tadds" in ln or "pto.tmuls" in ln]
    assert len(producers) == 2, f"expected two branch producers, got {producers}:\n{mlir}"
    for prod in producers:
        assert _first_group(r"outs\((%[A-Za-z0-9_]+)", prod) == acc, f"producer must write {acc}:\n{prod}"
        assert _first_group(r"ins\((%[A-Za-z0-9_]+)", prod) == acc, f"producer must read {acc}:\n{prod}"

    assert "pto.tmov" not in mlir, f"a shared handle needs no copy:\n{mlir}"


@pytest.mark.parametrize("enable_buffer_ir", [False, True])
def test_loop_carried_if_phi_keeps_baked_addr_aliasing_under_pypto(enable_buffer_ir):
    """The carry shares storage in both Tile and Buffer IR."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    with passes.PassContext([], enable_buffer_ir=enable_buffer_ir):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(LoopCarriedIfPhiProgram)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    mlir = codegen.PTOCodegen().generate(_ir.Program([func], "kernel", optimized.span))

    decls = [ln for ln in mlir.splitlines() if "= pto.alloc_tile" in ln]
    if enable_buffer_ir:
        assert len(decls) == 1, f"Buffer IR uses one shared carry handle:\n{mlir}"
    else:
        assert len(decls) > 1, f"Tile IR declares one alloc_tile per var:\n{mlir}"
    addrs = {_first_group(r"addr = (%[A-Za-z0-9_]+)", ln) for ln in decls}
    assert len(addrs) == 1, f"the aliased tiles must share one baked addr, got {addrs}:\n{mlir}"


# ── [N, 1] col-vector carry write-back ──

COLVEC_ROWS, COLVEC_STEPS = 16, 4


@pl.program
class ColVectorCarryProgram:
    """A `[N, 1]` col-vector loop carry updated through an if/else — the shape of an
    online-softmax `mi` / `li` carry.

    pypto runs `[N, 1]` elementwise on the `[1, N]` row-major view, so each branch
    yields a *reshape view* of its producer, not the producer itself.
    """

    @pl.function
    def kernel(
        self,
        x: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        y: pl.Tensor[[COLVEC_ROWS, 1], pl.FP32],
        out: pl.Out[pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]],
    ) -> pl.Tensor[[COLVEC_ROWS, 1], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="colvec_carry"):
            li = x[:, :]
            for i in pl.range(COLVEC_STEPS):  # noqa: B007 - index drives the branch
                b = y[:, :]
                if i == 0:
                    li = pl.mul(li, b)
                else:
                    li = pl.add(pl.mul(li, b), b)
            out[:, :] = li
        return out


def _outs_of(line: str) -> str:
    return _first_group(r"outs\((%[A-Za-z0-9_]+)", line)


def _ins_of(line: str) -> str:
    return _first_group(r"ins\((%[A-Za-z0-9_]+)", line)


def test_colvec_carry_is_written_back_under_pypto_planner():
    """Default planner: MemoryReuse's YieldFixupMutator inserts the same write-back.

    Aliasing is by baked address, not by SSA name — each var keeps its own
    `alloc_tile` — so the write-back's destination is a *different* handle sitting
    at the carry's address.
    """
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(ColVectorCarryProgram)
    incore = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    mlir = codegen.PTOCodegen().generate(_ir.Program([incore[0]], incore[0].name, optimized.span))

    assert "pto.treshape" not in mlir, f"PYPTO expresses views as re-declared allocs:\n{mlir}"

    def addr_of(handle: str) -> str:
        return _first_group(r"addr = (%[A-Za-z0-9_]+)", _sole_line(mlir, f"{handle} = pto.alloc_tile"))

    carry_addr = addr_of(_ins_of(_sole_line(mlir, "pto.tstore")))
    written = [ln for ln in mlir.splitlines() if "pto.tmov" in ln and addr_of(_outs_of(ln)) == carry_addr]
    assert written, f"the carry's buffer must be written before the store:\n{mlir}"


@pl.program
class RowInitializedColumnCarryProgram:
    """A column carry whose original allocation was defined as a row (#3017)."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        flag: pl.Scalar[pl.INT32],
        out: pl.Out[pl.Tensor[[32, 1], pl.FP32]],
    ) -> pl.Tensor[[32, 1], pl.FP32]:
        row_base = pl.tile.full([1, 32], dtype=pl.FP32, value=0.0)
        base = pl.tile.reshape(row_base, [32, 1])
        for i, (carry,) in pl.range(2, init_values=(base,)):  # noqa: B007 - explicit carry binding
            if flag != 0:
                row = pl.reshape(carry, [1, 32])
                updated = pl.add(row, 1.0)
                column = pl.reshape(updated, [32, 1])
                value = column
            else:
                value = carry
            result = pl.yield_(value)
        output = pl.store(result, [0, 0], out)
        return output


def _tile_buf_types(line: str) -> list[str]:
    return re.findall(r"!pto\.tile_buf<[^>]+>", line)


def _assert_tmov_operand_types(mlir: str) -> None:
    """TMOV must preserve each SSA definition and copy matching tile geometry."""
    definitions = {}
    for line in mlir.splitlines():
        match = re.match(r"\s*(%[A-Za-z0-9_]+) = (?:pto\.(?:alloc_tile|treshape)|scf\.if)\b", line)
        if match:
            definitions[match.group(1)] = _tile_buf_types(line)[-1]
    moves = [line for line in mlir.splitlines() if "pto.tmov" in line]
    assert moves, mlir
    for move in moves:
        source, destination = _tile_buf_types(move)
        assert source == definitions[_ins_of(move)], move
        assert destination == definitions[_outs_of(move)], move
        for field in ("rows", "cols", "blayout", "slayout"):
            pattern = rf"\b{field}=([^,>]+)"
            assert _first_group(pattern, source) == _first_group(pattern, destination), move


def test_row_initialized_column_carry_writes_a_view_of_its_original_storage():
    """The loop write-back must reinterpret the row allocation before TMOV."""
    mlir = _ptoas_pto(RowInitializedColumnCarryProgram)
    _assert_tmov_operand_types(mlir)
    root = _outs_of(_sole_line(mlir, "pto.texpands"))
    root_alloc = _sole_line(mlir, f"{root} = pto.alloc_tile")
    assert "rows=1, cols=32" in root_alloc and "v_row=?, v_col=?" in root_alloc, root_alloc
    # The initializer, updated row and writeback snapshot own storage. Native If
    # adds no phi allocation; the snapshot prevents a self-move on the unchanged arm.
    assert mlir.count("= pto.alloc_tile") == 3, mlir

    # Both the post-loop store and the loop write-back address the original bytes.
    store_handle = _ins_of(_sole_line(mlir, "pto.tstore"))
    assert f"{store_handle} = pto.treshape {root} " in mlir, mlir
    writes_to_root = []
    for move in (line for line in mlir.splitlines() if "pto.tmov" in line):
        destination = _outs_of(move)
        views = [line for line in mlir.splitlines() if f"{destination} = pto.treshape {root} " in line]
        if views:
            assert len(views) == 1, mlir
            view = views[0]
            assert _tile_buf_types(view)[0] == _tile_buf_types(root_alloc)[0], view
            assert "rows=32, cols=1, v_row=32, v_col=1" in _tile_buf_types(view)[-1], view
            writes_to_root.append(move)
    assert len(writes_to_root) == 1, mlir


def test_row_initialized_column_carry_keeps_baked_address_codegen():
    mlir = _planner_pto(RowInitializedColumnCarryProgram, passes.MemoryPlanner.PYPTO)
    assert "pto.treshape" not in mlir, mlir
    assert "= pto.alloc_tile addr = " in mlir, mlir
    _assert_tmov_operand_types(mlir)


@pytest.mark.ptoas_compile
def test_row_initialized_column_carry_assembles_with_ptoas(tmp_path: Path):
    """PTOAS must accept the real loop write-back, beyond matching annotations."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    mlir = _ptoas_pto(RowInitializedColumnCarryProgram)
    source, output = tmp_path / "column_carry.pto", tmp_path / "column_carry.cpp"
    source.write_text(mlir)
    _run_ptoas(str(source), str(output), ["--pto-arch=a2", "--pto-level=level2"])
    assert "TMOV(" in output.read_text()


def _move_to_shared_storage_program(
    root_shape: tuple[int, int],
    logical_shape: tuple[int, int],
    *,
    valid_shape: str = "full",
) -> _ir.Program:
    """Build the lowered shared-MemRef state without high-level layout repair."""
    span = _ir.Span.unknown()
    zero = _ir.ConstInt(0, pl.INDEX, span)
    root_memref = _ir.MemRef(_ir.Var("root_base", _ir.PtrType(), span), zero, 128, span)
    source_memref = _ir.MemRef(_ir.Var("source_base", _ir.PtrType(), span), zero, 128, span)
    valid_rows = _ir.Var("valid_rows", _ir.ScalarType(pl.INDEX), span)
    layout = _ir.TileLayout.col_major if logical_shape[1] == 1 else _ir.TileLayout.row_major
    valid = [valid_rows, 1] if valid_shape == "dynamic" else [0, 0] if valid_shape == "zero" else None
    view = _ir.TileView(valid_shape=valid, blayout=layout)
    root_type = _ir.TileType(root_shape, pl.FP32, root_memref, None, _ir.MemorySpace.Vec)
    source_type = _ir.TileType(logical_shape, pl.FP32, source_memref, view, _ir.MemorySpace.Vec)
    destination_type = _ir.TileType(logical_shape, pl.FP32, root_memref, view, _ir.MemorySpace.Vec)
    root = _ir.Var("root", root_type, span)
    source = _ir.Var("source", source_type, span)
    destination = _ir.Var("destination", destination_type, span)
    output = _ir.Var("output", _ir.TensorType(logical_shape, pl.FP32), span)
    stored = _ir.Var("stored", output.type, span)
    offsets = _ir.MakeTuple([zero, zero], span)
    body = _ir.SeqStmts(
        [
            _ir.AssignStmt(root, _ir.Call(_ir.Op("tile.create"), [], root_type, span), span),
            _ir.AssignStmt(source, _ir.Call(_ir.Op("tile.create"), [], source_type, span), span),
            _ir.AssignStmt(
                destination,
                _ir.Call(_ir.Op("tile.move"), [source], destination_type, span),
                span,
            ),
            _ir.AssignStmt(
                stored,
                _ir.Call(_ir.Op("tile.store"), [destination, offsets, output], output.type, span),
                span,
            ),
            _ir.ReturnStmt([stored], span),
        ],
        span,
    )
    params = [(output, _ir.ParamDirection.Out)]
    if valid_shape == "dynamic":
        params.append((valid_rows, _ir.ParamDirection.In))
    function = _ir.Function("kernel", params, [output.type], body, span, _ir.FunctionType.InCore)
    return _ir.Program([function], "SharedStorageMove", span)


def _shared_storage_move_pto(program: _ir.Program) -> str:
    reset_for_testing()
    set_backend_type(BackendType.Ascend910)
    return codegen.PTOCodegen().generate(program, emit_tile_addr=False)


@pytest.mark.parametrize(
    ("root_shape", "logical_shape"),
    [((1, 32), (32, 1)), ((32, 1), (1, 32)), ((1, 32), (1, 32))],
    ids=["row-storage-column-result", "column-storage-row-result", "matching-row-types"],
)
def test_shared_storage_move_uses_the_logical_destination_type(root_shape, logical_shape):
    mlir = _shared_storage_move_pto(_move_to_shared_storage_program(root_shape, logical_shape))
    _assert_tmov_operand_types(mlir)
    assert mlir.count("= pto.alloc_tile") == 2, mlir
    move = _sole_line(mlir, "pto.tmov")
    assert _ins_of(_sole_line(mlir, "pto.tstore")) == _outs_of(move), mlir
    if root_shape == logical_shape:
        assert "pto.treshape" not in mlir, mlir
    else:
        view = _sole_line(mlir, "pto.treshape")
        assert f"{_outs_of(move)} = pto.treshape %root " in view, mlir
        rows, cols = logical_shape
        assert f"rows={rows}, cols={cols}, v_row={rows}, v_col={cols}" in view, view


def test_shared_storage_move_rejects_a_dynamic_valid_destination_view():
    program = _move_to_shared_storage_program((1, 32), (32, 1), valid_shape="dynamic")
    with pytest.raises(ValueError, match=r"tile\.move.*static valid_shape"):
        _shared_storage_move_pto(program)


def test_shared_storage_move_keeps_the_zero_valid_sentinel_dynamic():
    program = _move_to_shared_storage_program((1, 32), (32, 1), valid_shape="zero")
    mlir = _shared_storage_move_pto(program)
    _assert_tmov_operand_types(mlir)
    assert mlir.count("= pto.alloc_tile") == 2, mlir
    view = _sole_line(mlir, "pto.treshape")
    assert "rows=32, cols=1, v_row=?, v_col=?" in _tile_buf_types(view)[-1], view
    assert "v_row=0" not in view and "v_col=0" not in view, view
    assert f"{_outs_of(_sole_line(mlir, 'pto.tmov'))} = pto.treshape %root " in view, mlir


def test_parameter_reinterpret_view_keeps_actual_caller_storage(ascend_backend):
    """Reinterpret must pass caller ownership through the common no-op guard."""

    @pl.program
    class ParameterReinterpret:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            source: pl.Tile[[1, 128], pl.FP32, pl.Mem.Vec],
            out: pl.Out[pl.Tensor[[1, 128], pl.INT32]],
        ) -> pl.Tensor[[1, 128], pl.INT32]:
            reinterpreted = pl.tile.reinterpret_view(source, pl.INT32)
            return pl.store(reinterpreted, [0, 0], out)

    with passes.PassContext([]):
        initialized = passes.init_mem_ref()(ParameterReinterpret)
        placed = passes.allocate_memory_addr()(initialized)
    mlir = codegen.PTOCodegen().generate(placed, emit_source_loc=False)
    source = _first_group(r"(%[A-Za-z0-9_]+) = pto.alloc_tile addr = %source_addr", mlir)
    view = _sole_line(mlir, "pto.treshape ")
    assert f"pto.treshape {source} :" in view, mlir
    assert len([line for line in mlir.splitlines() if "= pto.alloc_tile" in line]) == 1, mlir


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
