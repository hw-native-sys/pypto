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


@pytest.mark.parametrize("view_op", ["reshape", "reinterpret_view", "transpose_view"])
@pytest.mark.parametrize("valid_shape", ["runtime", (0, 0), (0, 8), (8, 0), (8, 8)])
@pytest.mark.parametrize("emit_tile_addr", [False, True])
def test_tpop_view_dynamic_validity_requires_local_storage(view_op, valid_shape, emit_tile_addr):
    """TPOP owns a FIFO handle, so a view cannot initialize dynamic metadata on it.

    Build lowered IR directly to include the all-zero marker before replay
    legalization. Static positive views remain supported on that same source.
    """
    span = ir.Span.unknown()
    rows = ir.Var("rows", ir.ScalarType(pl.INDEX), span)
    shape = [ir.ConstInt(16, pl.INDEX, span) for _ in range(2)]
    valid = (
        [rows, ir.ConstInt(8, pl.INDEX, span)]
        if valid_shape == "runtime"
        else [ir.ConstInt(v, pl.INDEX, span) for v in valid_shape]
    )
    source_type = ir.TileType(shape, pl.FP32, None, ir.TileView(valid_shape=valid), ir.MemorySpace.Vec)
    result_type = ir.TileType(
        shape,
        pl.INT32 if view_op == "reinterpret_view" else pl.FP32,
        None,
        ir.TileView(valid_shape=list(reversed(valid)) if view_op == "transpose_view" else valid),
        ir.MemorySpace.Vec,
    )
    source = ir.Var("popped", source_type, span)
    result = ir.Var("view", result_type, span)
    args = [source, ir.MakeTuple(shape, span)] if view_op == "reshape" else [source]
    body = ir.SeqStmts(
        [
            ir.AssignStmt(
                source, ir.Call(ir.Op("tile.tpop_from_aic"), [], {"split": 0}, source_type, span), span
            ),
            ir.AssignStmt(result, ir.Call(ir.Op(f"tile.{view_op}"), args, {}, result_type, span), span),
        ],
        span,
    )
    function = ir.Function("consumer", [rows], [], body, span, ir.FunctionType.AIV)
    program = ir.Program([function], "TpopView", span)
    if valid_shape == (8, 8):
        mlir = PTOCodegen().generate(program, emit_tile_addr=emit_tile_addr)
        assert "pto.treshape" in mlir
        assert "pto.set_validshape" not in mlir
    else:
        with pytest.raises(ValueError, match="dynamic-valid tile view requires a locally allocated source"):
            PTOCodegen().generate(program, emit_tile_addr=emit_tile_addr)


@pytest.mark.parametrize("program", [TpopReshapeView, TpopTransposeView])
@pytest.mark.parametrize("planner", list(passes.MemoryPlanner), ids=lambda planner: planner.name)
def test_tpop_view_replay_compiles(tmp_path, program, planner):
    """Replay must preserve the pipe and compile without setters on its popped views."""
    consumer = program.get_function("vector_consumer")
    assert consumer is not None
    replay_consumer = ir.Function(
        consumer.name,
        list(zip(consumer.params, consumer.param_directions)),
        consumer.return_types,
        consumer.body,
        consumer.span,
        consumer.func_type,
        attrs={**consumer.attrs, "dual_aiv_dispatch": True},
    )
    # Use a supported Acc -> Vec producer for native compilation. The older
    # text-only fixtures push a Mat tile, which PTOAS does not accept as C2V.
    shape = [dim.value for dim in consumer.params[0].type.shape]
    producer_program = pl.parse(f"""
@pl.program
class Producer:
    @pl.function(type=pl.FunctionType.AIC)
    def cube_producer(self):
        local = pl.reserve_buffer(name="v2c_slot_buffer", size=4096, base=4096)
        peer = pl.import_peer_buffer(name="c2v_slot_buffer", peer_func="vector_consumer")
        pl.aic_initialize_pipe(dir_mask=3, slot_size=1024, c2v_consumer_buf=peer, v2c_consumer_buf=local)
        received: pl.Tile[{shape}, pl.FP32, pl.MemorySpace.Mat] = pl.tpop_from_aiv(split=0)
        source: pl.Tile[{shape}, pl.FP32, pl.MemorySpace.Acc] = pl.tile.create(
            {shape}, dtype=pl.FP32, target_memory=pl.MemorySpace.Acc
        )
        pl.tpush_to_aiv(source, split=0)
        pl.tfree_to_aiv(received)
""")
    assert isinstance(producer_program, ir.Program)
    producer = producer_program.get_function("cube_producer")
    assert producer is not None
    functions = [replay_consumer, producer]
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=False):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
            ir.Program(functions, program.name, program.span)
        )
    mlir = PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    body = _function_body(mlir, "vector_consumer")
    assert body.count("pto.tpop_from_aic") == 2, body
    assert body.count("pto.tfree_from_aic") == 2, body
    # The lane-0 views remain; lane 1 now gets initialized empty allocations.
    assert body.count("pto.treshape") == (2 if program is TpopReshapeView else 1), body
    assert "pto.set_validshape" not in body, body
    if find_ptoas_binary() is not None:
        pto_backend._compile_pto_module(mlir, "vector_consumer", str(tmp_path), memory_planner=planner)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
