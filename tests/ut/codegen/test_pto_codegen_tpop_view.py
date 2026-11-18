# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PTO codegen for a zero-copy view (reshape) over a cross-core tpop result.

A ``tile.tpop_from_aic`` result has no general-pool address: its data lives in
the reserved C2V slot, delivered via the pipe. Since InitMemRef now leaves the
tpop result MemRef-less, a ``pl.reshape`` over it inherits no MemRef and codegen
lowers it to ``pto.treshape`` reading the popped tile directly — instead of a
fresh, disconnected ``pto.alloc_tile`` the popped data is never moved into.
"""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir, passes
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir import OptimizationStrategy, PassManager

PTOCodegen = codegen.PTOCodegen


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    yield
    backend.reset_for_testing()


def _generate_default_mlir(program_cls) -> str:
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    program = pm.run_passes(program_cls)
    result = PTOCodegen().generate(program)
    return result if isinstance(result, str) else "".join(result.values())


def _function_body(mlir: str, func_name: str) -> str:
    """Return the text of a single ``func.func @<name>`` block from a module."""
    lines = mlir.splitlines()
    out: list[str] = []
    depth = 0
    capturing = False
    for line in lines:
        if not capturing and f"func.func @{func_name}" in line:
            capturing = True
        if capturing:
            out.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0 and "{" in "\n".join(out):
                break
    return "\n".join(out)


def _generate_mixed_tile_mlir(program: ir.Program, planner: passes.MemoryPlanner) -> str:
    """Lower a mixed InCore program through the legacy Tile pipeline."""
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    # Group is a host dispatch wrapper; PTO codegen takes its device members.
    members = [
        func
        for func in optimized.functions.values()
        if func.func_type in (ir.FunctionType.AIC, ir.FunctionType.AIV)
    ]
    assert len(members) == 2
    device_program = ir.Program(members, optimized.name, optimized.span)
    mlir = PTOCodegen().generate(device_program, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert isinstance(mlir, str)
    return mlir


def _compile_borrowed_program(
    program: ir.Program, planner: passes.MemoryPlanner, tmp_path: Path, backend_type: BackendType
) -> str:
    """Check the legacy Tile path with both address and PTOAS memory planning."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)

    @pl.program
    class Peer:
        @pl.function(type=pl.FunctionType.AIC)
        def producer(self):
            peer = pl.import_peer_buffer(name="c2v", peer_func="kernel")
            pl.aic_initialize_pipe(peer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)

    # PTOAS validates both ends of a reserved FIFO even in compile-only tests.
    program = ir.Program([*program.functions.values(), *Peer.functions.values()], program.name, program.span)
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    mlir = PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    assert isinstance(mlir, str)
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path), memory_planner=planner)
    return mlir


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_borrowed_if_result_codegen(tmp_path: Path, planner: passes.MemoryPlanner, backend_type: BackendType):
    """An ordinary Tile if-result must retain the received handle without an allocation."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self, flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            buffer = pl.reserve_buffer(name="c2v", size=2048, base=0)
            pl.aiv_initialize_pipe(buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
            if flag:
                selected = pl.yield_(received)
            else:
                selected = pl.yield_(received)
            out = pl.store(selected, [0, 0], out)
            pl.tfree_to_aic(received)
            return out

    mlir = _compile_borrowed_program(Program, planner, tmp_path, backend_type)
    assert "scf.if" in mlir
    assert "pto.alloc_tile" not in mlir
    assert "pto.tmov" not in mlir


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_borrowed_if_view_result_codegen(
    tmp_path: Path, planner: passes.MemoryPlanner, backend_type: BackendType
):
    """Branch-local views must remain defined at the merged borrowed result."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self, flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]
        ) -> pl.Tensor[[8, 32], pl.FP32]:
            buffer = pl.reserve_buffer(name="c2v", size=2048, base=0)
            pl.aiv_initialize_pipe(buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
            if flag:
                flat = pl.reshape(received, [8, 32])
                selected = pl.yield_(flat)
            else:
                other = pl.reshape(received, [8, 32])
                selected = pl.yield_(other)
            out = pl.store(selected, [0, 0], out)
            pl.tfree_to_aic(received)
            return out

    mlir = _compile_borrowed_program(Program, planner, tmp_path, backend_type)
    assert "scf.if" in mlir
    assert "pto.alloc_tile" not in mlir
    assert "pto.tmov" not in mlir


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_different_borrowed_if_sources_keep_phi_copy(
    tmp_path: Path, planner: passes.MemoryPlanner, backend_type: BackendType
):
    """Selecting distinct slots must copy before their FIFO lifetimes end."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(
            self, flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            buffer = pl.reserve_buffer(name="c2v", size=2048, base=0)
            pl.aiv_initialize_pipe(buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            first: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
            second: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
            if flag:
                selected = pl.yield_(first)
            else:
                selected = pl.yield_(second)
            pl.tfree_to_aic(first)
            pl.tfree_to_aic(second)
            out = pl.store(selected, [0, 0], out)
            return out

    mlir = _compile_borrowed_program(Program, planner, tmp_path, backend_type)
    assert mlir.count("pto.alloc_tile") == 1
    assert mlir.count("pto.tmov") == 2
    assert mlir.rindex("pto.tmov") < mlir.index("pto.tfree_from_aic") < mlir.index("pto.tstore")


@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_outer_mat_fifo_if_merge_rejected(planner: passes.MemoryPlanner, backend_type: BackendType):
    """Automatic V2C receives before an if must not fall through to Mat-to-Mat copies."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            flag: pl.Scalar[pl.BOOL],
            x: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            vector = pl.load(x, [0, 0], [16, 16])
            first = pl.move(
                vector,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            vector2 = pl.add(vector, 1.0)
            second = pl.move(
                vector2,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            if flag:
                selected = pl.yield_(first)
            else:
                selected = pl.yield_(second)
            left = pl.move(selected, target_memory=pl.Mem.Left)
            right_mat = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            right = pl.move(right_mat, target_memory=pl.Mem.Right)
            product = pl.matmul(left, right)
            out = pl.store(product, [0, 0], out)
            return out

    # Expansion succeeds because both receives dominate the if. Codegen must
    # diagnose its unsupported phi copy before handing invalid MLIR to PTOAS.
    with pytest.raises(ValueError, match="Merging a FIFO-backed Tile.*only supported in Vec memory"):
        _generate_mixed_tile_mlir(Program, planner)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_same_mat_fifo_if_result_codegen(
    tmp_path: Path, planner: passes.MemoryPlanner, backend_type: BackendType
):
    """Selecting one shared Mat receive remains legal because no phi copy is needed."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            flag: pl.Scalar[pl.BOOL],
            x: pl.Tensor[[16, 16], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            vector = pl.load(x, [0, 0], [16, 16])
            received = pl.move(
                vector,
                target_memory=pl.Mem.Mat,
                blayout=pl.TileLayout.col_major,
                slayout=pl.TileLayout.row_major,
            )
            if flag:
                selected = pl.yield_(received)
            else:
                selected = pl.yield_(received)
            left = pl.move(selected, target_memory=pl.Mem.Left)
            right_mat = pl.load(x, [0, 0], [16, 16], target_memory=pl.Mem.Mat)
            right = pl.move(right_mat, target_memory=pl.Mem.Right)
            product = pl.matmul(left, right)
            out = pl.store(product, [0, 0], out)
            return out

    mlir = _generate_mixed_tile_mlir(Program, planner)
    body = _function_body(mlir, "kernel_aic")
    assert re.search(r"scf\.if .* -> \(!pto\.tile_buf<loc=mat", body), body
    assert not any("pto.tmov" in line and line.count("loc=mat") == 2 for line in body.splitlines()), body
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path), memory_planner=planner)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("ordinary_else", [False, True])
def test_branch_local_borrowed_result_copied_before_free(
    tmp_path: Path, planner: passes.MemoryPlanner, backend_type: BackendType, ordinary_else: bool
):
    """A branch-local received view must be copied before its slot is released."""
    backend.reset_for_testing()
    backend.set_backend_type(backend_type)

    if ordinary_else:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]
            ) -> pl.Tensor[[8, 32], pl.FP32]:
                cube: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
                pl.tfree_to_aiv(cube)
                if flag:
                    received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
                    view = pl.reshape(received, [8, 32])
                    selected = pl.yield_(view)
                else:
                    ordinary = pl.tile.full([8, 32], pl.FP32, 1.0)
                    selected = pl.yield_(ordinary)
                out = pl.store(selected, [0, 0], out)
                return out

    else:

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self, flag: pl.Scalar[pl.BOOL], out: pl.Out[pl.Tensor[[8, 32], pl.FP32]]
            ) -> pl.Tensor[[8, 32], pl.FP32]:
                cube: pl.Tile[[16, 16], pl.FP16, pl.Mem.Mat] = pl.tpop_from_aiv(split=0)
                pl.tfree_to_aiv(cube)
                if flag:
                    received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
                    view = pl.reshape(received, [8, 32])
                    selected = pl.yield_(view)
                else:
                    other: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
                    other_view = pl.reshape(other, [8, 32])
                    selected = pl.yield_(other_view)
                out = pl.store(selected, [0, 0], out)
                return out

    mlir = _generate_mixed_tile_mlir(Program, planner)
    body = _function_body(mlir, "kernel_aiv")
    # A2/A3 may also emit a secondary-lane handshake replay. Check the first
    # execution of each original branch, where the result feeds the store.
    branch_count = 1 if ordinary_else else 2
    pops = list(re.finditer("pto.tpop_from_aic", body))
    assert len(pops) >= branch_count, body
    for pop in pops[:branch_count]:
        view = body.index("pto.treshape", pop.end())
        copy = body.index("pto.tmov", view)
        release = body.index("pto.tfree_from_aic", pop.end())
        assert pop.start() < view < copy < release, body
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    pto_backend._compile_pto_module(mlir, "kernel", str(tmp_path), memory_planner=planner)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("backend_type", [BackendType.Ascend910B, BackendType.Ascend950])
@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_borrowed_fillpad_writeback_codegen(
    tmp_path: Path, planner: passes.MemoryPlanner, backend_type: BackendType
):
    """Fillpad needs a defined alias with its own pad metadata, under either planner."""

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.AIV)
        def kernel(self, out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]) -> pl.Tensor[[16, 16], pl.FP32]:
            buffer = pl.reserve_buffer(name="c2v", size=2048, base=0)
            pl.aiv_initialize_pipe(buffer, pl.const(0, pl.INT32), dir_mask=1, slot_size=1024, slot_num=2)
            received: pl.Tile[[16, 16], pl.FP32, pl.Mem.Vec] = pl.tpop_from_aic(split=0)
            padded = pl.tile.fillpad_inplace(received, pad_value=pl.PadValue.zero)
            out = pl.store(padded, [0, 0], out)
            pl.tfree_to_aic(received)
            return out

    mlir = _compile_borrowed_program(Program, planner, tmp_path, backend_type)
    fill = next(line for line in mlir.splitlines() if "pto.tfillpad " in line)
    destination = re.search(r"outs\((%[\w.$]+)", fill)
    assert destination is not None, fill
    assert f"{destination[1]} = pto.treshape" in mlir
    types = re.findall(r"!pto\.tile_buf<[^>]+>", fill)
    assert len(types) == 2 and "pad=0>" in types[0] and "pad=1>" in types[1], fill
    assert "pto.alloc_tile" not in mlir


@pl.program
class TpopReshapeView:
    """Vector consumer that reshapes a popped (cross-core) tile in place."""

    @pl.function(type=pl.FunctionType.AIV)
    def vector_consumer(
        self,
        a: pl.Tensor[[16, 16], pl.FP32],
        output: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x2000)
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_producer")
        pl.aiv_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_buf, v2c_consumer_buf=v2c_peer
        )

        tile_a: pl.Tile[[16, 16], pl.FP32] = pl.load(a, [0, 0], [16, 16])
        pl.tpush_to_aic(tile_a, split=0)

        # Popped tile: lives in the reserved C2V slot, no general-pool address.
        t: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)

        v: pl.Tile[[256], pl.FP32, pl.MemorySpace.Vec] = pl.reshape(t, [256])
        pl.tfree_to_aic(t)
        vv: pl.Tile[[16, 16], pl.FP32] = pl.reshape(v, [16, 16])
        out: pl.Tile[[16, 16], pl.FP32] = pl.exp(vv)

        updated: pl.Tensor[[16, 16], pl.FP32] = pl.store(out, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.AIC)
    def cube_producer(self, arg: pl.Tensor[[16, 16], pl.FP32]):
        v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
        c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
        pl.aic_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buf
        )
        received: pl.Tile[[16, 16], pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
        pl.tpush_to_aiv(received, split=0)
        pl.tfree_to_aiv(received)


def test_pto_codegen_reshape_over_tpop_lowers_to_treshape():
    mlir = _generate_default_mlir(TpopReshapeView)
    consumer = _function_body(mlir, "vector_consumer")

    # The popped tile is the source of the reshape view.
    assert "pto.tpop_from_aic" in consumer, consumer

    # Both the direct view (reshape(t)) and the chained view (reshape(reshape(t)))
    # reinterpret the popped tile in place via pto.treshape reading the source SSA.
    assert consumer.count("pto.treshape") == 2, (
        "both reshapes over the tpop result must lower to pto.treshape; got:\n" + consumer
    )

    # The views own no buffer: the only alloc_tiles are the loaded input and the
    # exp output — neither reshape result gets a (disconnected) alloc_tile.
    assert consumer.count("pto.alloc_tile") == 2, (
        "a view over a tpop result must not allocate a buffer; got:\n" + consumer
    )

    # Each treshape carries its `: src_type -> dst_type` annotation (the source is
    # MemRef-less, so the type must come from the TileType, not the MemRef).
    for line in consumer.splitlines():
        if "pto.treshape" in line:
            assert " : " in line and " -> " in line, (
                "pto.treshape must carry its source/result type annotation; got:\n" + line
            )


@pl.program
class TpopTransposeView:
    """Vector consumer that transpose_views a popped (cross-core) tile in place."""

    @pl.function(type=pl.FunctionType.AIV)
    def vector_consumer(
        self,
        a: pl.Tensor[[16, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 16], pl.FP32]],
    ) -> pl.Tensor[[32, 16], pl.FP32]:
        c2v_buf = pl.reserve_buffer(name="c2v_slot_buffer", size=4096, base=0x2000)
        v2c_peer = pl.import_peer_buffer(name="v2c_slot_buffer", peer_func="cube_producer")
        pl.aiv_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_buf, v2c_consumer_buf=v2c_peer
        )

        tile_a: pl.Tile[[16, 32], pl.FP32] = pl.load(a, [0, 0], [16, 32])
        pl.tpush_to_aic(tile_a, split=0)

        # Popped tile: lives in the reserved C2V slot, no general-pool address.
        t: pl.Tile[[16, 32], pl.FP32, pl.MemorySpace.Vec] = pl.tpop_from_aic(split=0)

        # Zero-copy NZ<->ZN reinterpret of the popped tile (DSL-exposed op).
        vt: pl.Tile[[32, 16], pl.FP32, pl.MemorySpace.Vec] = pl.tile.transpose_view(t)
        out: pl.Tile[[32, 16], pl.FP32] = pl.exp(vt)
        pl.tfree_to_aic(t)

        updated: pl.Tensor[[32, 16], pl.FP32] = pl.store(out, [0, 0], output)
        return updated

    @pl.function(type=pl.FunctionType.AIC)
    def cube_producer(self, arg: pl.Tensor[[16, 32], pl.FP32]):
        v2c_buf = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=0x1000)
        c2v_peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
        pl.aic_initialize_pipe(
            dir_mask=3, slot_size=1024, c2v_consumer_buf=c2v_peer, v2c_consumer_buf=v2c_buf
        )
        received: pl.Tile[[16, 32], pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
        pl.tpush_to_aiv(received, split=0)
        pl.tfree_to_aiv(received)


def test_pto_codegen_transpose_view_over_tpop_lowers_to_treshape():
    mlir = _generate_default_mlir(TpopTransposeView)
    consumer = _function_body(mlir, "vector_consumer")

    assert "pto.tpop_from_aic" in consumer, consumer

    # The transpose_view over the MemRef-less popped tile reinterprets the slot in
    # place via pto.treshape reading the source SSA — NOT a real data-moving
    # transpose (pto.ttrans) and NOT a fresh, disconnected alloc_tile.
    assert "pto.treshape" in consumer, "transpose_view over a tpop must lower to pto.treshape:\n" + consumer
    assert "pto.ttrans" not in consumer, "must not materialise a real transpose:\n" + consumer

    # The view owns no buffer: its result comes from pto.treshape, not a fresh
    # (disconnected) alloc_tile. The treshape must READ the popped tile's SSA so
    # the data is connected, and carry its `: src -> dst` type annotation
    # (the MemRef-less source's type comes from the TileType, not a MemRef).
    tpop_lines = [ln for ln in consumer.splitlines() if "= pto.tpop_from_aic" in ln]
    assert len(tpop_lines) == 1, consumer
    tpop_ssa = tpop_lines[0].split("=", 1)[0].strip()
    treshape_lines = [ln for ln in consumer.splitlines() if "pto.treshape" in ln]
    assert treshape_lines, consumer
    for line in treshape_lines:
        assert "pto.alloc_tile" not in line, "view must not allocate a buffer:\n" + line
        assert f"pto.treshape {tpop_ssa}" in line, "treshape must read the popped tile:\n" + line
        assert " : " in line and " -> " in line, "pto.treshape needs src/dst annotation:\n" + line

    # tfree must follow the consuming op (lifetime extended through the view),
    # else the FIFO slot is freed before the transpose view is read.
    assert consumer.find("pto.tfree_from_aic") > consumer.find("pto.treshape") > 0, (
        "tfree must come after the treshape view:\n" + consumer
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
