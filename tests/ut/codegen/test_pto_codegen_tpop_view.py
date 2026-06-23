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
from pypto import backend, codegen
from pypto.backend import BackendType
from pypto.ir import OptimizationStrategy, PassManager

PTOCodegen = codegen.PTOCodegen


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
