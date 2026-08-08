# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the LowerCompositeOps pass.

The LowerCompositeOps pass decomposes composite tile ops into primitive
arithmetic tile ops. Today it covers ``tile.sin`` / ``tile.cos`` (Cody-Waite
range reduction + degree-9 odd Horner polynomial). The decomposition uses
only ``tile.muls``, ``tile.adds``, ``tile.add``, ``tile.sub``, ``tile.mul``
and ``tile.cast`` — no sin/cos remain after the pass.

Decomposition tests use the Before/Expected pattern: the ``Expected`` program
pins the full decomposed primitive tree so any change to the lowering surfaces
as a structural diff.
"""

import re

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics.exceptions import ParserError

_OP_PLD_TILE_REMOTE_LOAD = ir.get_op("pld.tile.remote_load").name
_OP_TILE_LOAD = ir.get_op("tile.load").name

# Primitive tile ops the decomposition is allowed to emit (besides framework
# infrastructure ops like tile.load / tile.store / tile.move that wrap the
# decomposed body).
_DECOMP_PRIMITIVES = {
    ir.get_op("tile.muls").name,
    ir.get_op("tile.adds").name,
    ir.get_op("tile.add").name,
    ir.get_op("tile.sub").name,
    ir.get_op("tile.mul").name,
    ir.get_op("tile.cast").name,
}


class _OpNameCollector(ir.IRVisitor):
    """Walk the IR and record the ``op.name`` of every Call encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.op_names: list[str] = []

    def visit_call(self, op: ir.Call) -> None:
        self.op_names.append(op.op.name)
        super().visit_call(op)


def _collect_op_names(prog) -> list[str]:
    collector = _OpNameCollector()
    collector.visit_program(prog)
    return collector.op_names


def test_lower_composite_ops_pass_factory_exists():
    """The factory returns a Pass instance with the expected name."""
    p = passes.lower_composite_ops()
    assert p is not None
    assert p.get_name() == "LowerCompositeOps"


def test_lower_composite_ops_noop_on_no_trig():
    """Pass must leave programs without sin/cos unchanged."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.exp(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_composite_ops()(Before)
    ir.assert_structural_equal(After, Before)


def test_sin_is_decomposed_to_primitives():
    """``tile.sin`` is decomposed into the full Cody-Waite + Horner primitive tree."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sin(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def main_incore_0(
            x: pl.Tensor[[16, 16], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile = pl.tile.load(x, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
            y_tile__pi_inv_x_tmp_v0 = pl.tile.muls(x_tile, 0.31830987334251404)
            y_tile__k_i_tmp_v1 = pl.tile.cast(y_tile__pi_inv_x_tmp_v0, target_type=pl.INT32, mode="round")
            y_tile__k_f_tmp_v2 = pl.tile.cast(y_tile__k_i_tmp_v1, target_type=pl.FP32, mode="none")
            y_tile__k_pi_v2_tmp_v3 = pl.tile.muls(y_tile__k_f_tmp_v2, 3.140625)
            y_tile__t0_tmp_v4 = pl.tile.sub(x_tile, y_tile__k_pi_v2_tmp_v3)
            y_tile__k_pi_c1_tmp_v5 = pl.tile.muls(y_tile__k_f_tmp_v2, 0.0009670257568359375)
            y_tile__t1_tmp_v6 = pl.tile.sub(y_tile__t0_tmp_v4, y_tile__k_pi_c1_tmp_v5)
            y_tile__k_pi_c2_tmp_v7 = pl.tile.muls(y_tile__k_f_tmp_v2, 6.2771141529083252e-07)
            y_tile__t2_tmp_v8 = pl.tile.sub(y_tile__t1_tmp_v6, y_tile__k_pi_c2_tmp_v7)
            y_tile__k_pi_c3_tmp_v9 = pl.tile.muls(y_tile__k_f_tmp_v2, 1.2164491636212915e-10)
            y_tile__t3_tmp_v10 = pl.tile.sub(y_tile__t2_tmp_v8, y_tile__k_pi_c3_tmp_v9)
            y_tile__k_pi_c4_tmp_v11 = pl.tile.muls(y_tile__k_f_tmp_v2, -1.0290622927356871e-13)
            y_tile__t4_tmp_v12 = pl.tile.sub(y_tile__t3_tmp_v10, y_tile__k_pi_c4_tmp_v11)
            y_tile__half_k_tmp_v13 = pl.tile.muls(y_tile__k_f_tmp_v2, 0.5)
            y_tile__floor_hk_i_tmp_v14 = pl.tile.cast(
                y_tile__half_k_tmp_v13, target_type=pl.INT32, mode="floor"
            )
            y_tile__floor_hk_f_tmp_v15 = pl.tile.cast(
                y_tile__floor_hk_i_tmp_v14, target_type=pl.FP32, mode="none"
            )
            y_tile__floor_x4_tmp_v16 = pl.tile.muls(y_tile__floor_hk_f_tmp_v15, 4.0)
            y_tile__neg2_k_tmp_v17 = pl.tile.muls(y_tile__k_f_tmp_v2, -2.0)
            y_tile__sign_pre_tmp_v18 = pl.tile.add(y_tile__floor_x4_tmp_v16, y_tile__neg2_k_tmp_v17)
            y_tile__sign_tmp_v19 = pl.tile.adds(y_tile__sign_pre_tmp_v18, 1.0)
            y_tile__t2sq_tmp_v20 = pl.tile.mul(y_tile__t4_tmp_v12, y_tile__t4_tmp_v12)
            y_tile__p_r0_tmp_v21 = pl.tile.muls(y_tile__t2sq_tmp_v20, 2.6049265215988271e-06)
            y_tile__p_r1_tmp_v22 = pl.tile.adds(y_tile__p_r0_tmp_v21, -0.00019808944489341229)
            y_tile__p_t2_r1_tmp_v23 = pl.tile.mul(y_tile__p_r1_tmp_v22, y_tile__t2sq_tmp_v20)
            y_tile__p_r2_tmp_v24 = pl.tile.adds(y_tile__p_t2_r1_tmp_v23, 0.0083330497145652771)
            y_tile__p_t2_r2_tmp_v25 = pl.tile.mul(y_tile__p_r2_tmp_v24, y_tile__t2sq_tmp_v20)
            y_tile__p_r3_tmp_v26 = pl.tile.adds(y_tile__p_t2_r2_tmp_v25, -0.16666658222675323)
            y_tile__p_t2_r3_tmp_v27 = pl.tile.mul(y_tile__p_r3_tmp_v26, y_tile__t2sq_tmp_v20)
            y_tile__p_one_tmp_v28 = pl.tile.adds(y_tile__p_t2_r3_tmp_v27, 1.0)
            y_tile__t_p_tmp_v29 = pl.tile.mul(y_tile__t4_tmp_v12, y_tile__p_one_tmp_v28)
            y_tile = pl.tile.mul(y_tile__sign_tmp_v19, y_tile__t_p_tmp_v29)
            out_0 = pl.tile.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0 = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            r = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_composite_ops()(Before)
    ir.assert_structural_equal(After, Expected)


def test_cos_is_decomposed_to_primitives():
    """``tile.cos`` is decomposed into the full Cody-Waite + Horner primitive tree."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.cos(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def main_incore_0(
            x: pl.Tensor[[16, 16], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile = pl.tile.load(x, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
            y_tile__pi_inv_x_tmp_v0 = pl.tile.muls(x_tile, 0.31830987334251404)
            y_tile__k_pre_tmp_v1 = pl.tile.adds(y_tile__pi_inv_x_tmp_v0, 0.5)
            y_tile__k_i_tmp_v2 = pl.tile.cast(y_tile__k_pre_tmp_v1, target_type=pl.INT32, mode="rint")
            y_tile__k_f_tmp_v3 = pl.tile.cast(y_tile__k_i_tmp_v2, target_type=pl.FP32, mode="none")
            y_tile__k_pi_v2_tmp_v4 = pl.tile.muls(y_tile__k_f_tmp_v3, 3.140625)
            y_tile__t0_tmp_v5 = pl.tile.sub(x_tile, y_tile__k_pi_v2_tmp_v4)
            y_tile__k_pi_c1_tmp_v6 = pl.tile.muls(y_tile__k_f_tmp_v3, 0.0009670257568359375)
            y_tile__t1_tmp_v7 = pl.tile.sub(y_tile__t0_tmp_v5, y_tile__k_pi_c1_tmp_v6)
            y_tile__t1h_tmp_v8 = pl.tile.adds(y_tile__t1_tmp_v7, 1.5707963705062866)
            y_tile__k_pi_c2_tmp_v9 = pl.tile.muls(y_tile__k_f_tmp_v3, 6.2771141529083252e-07)
            y_tile__t2_tmp_v10 = pl.tile.sub(y_tile__t1h_tmp_v8, y_tile__k_pi_c2_tmp_v9)
            y_tile__k_pi_c3_tmp_v11 = pl.tile.muls(y_tile__k_f_tmp_v3, 1.2164491636212915e-10)
            y_tile__t3_tmp_v12 = pl.tile.sub(y_tile__t2_tmp_v10, y_tile__k_pi_c3_tmp_v11)
            y_tile__k_pi_c4_tmp_v13 = pl.tile.muls(y_tile__k_f_tmp_v3, -1.0290622927356871e-13)
            y_tile__t4_tmp_v14 = pl.tile.sub(y_tile__t3_tmp_v12, y_tile__k_pi_c4_tmp_v13)
            y_tile__t4t_tmp_v15 = pl.tile.adds(y_tile__t4_tmp_v14, -4.3711388286737929e-08)
            y_tile__half_k_tmp_v16 = pl.tile.muls(y_tile__k_f_tmp_v3, 0.5)
            y_tile__floor_hk_i_tmp_v17 = pl.tile.cast(
                y_tile__half_k_tmp_v16, target_type=pl.INT32, mode="floor"
            )
            y_tile__floor_hk_f_tmp_v18 = pl.tile.cast(
                y_tile__floor_hk_i_tmp_v17, target_type=pl.FP32, mode="none"
            )
            y_tile__floor_x4_tmp_v19 = pl.tile.muls(y_tile__floor_hk_f_tmp_v18, 4.0)
            y_tile__neg2_k_tmp_v20 = pl.tile.muls(y_tile__k_f_tmp_v3, -2.0)
            y_tile__sign_pre_tmp_v21 = pl.tile.add(y_tile__floor_x4_tmp_v19, y_tile__neg2_k_tmp_v20)
            y_tile__sign_tmp_v22 = pl.tile.adds(y_tile__sign_pre_tmp_v21, 1.0)
            y_tile__t2sq_tmp_v23 = pl.tile.mul(y_tile__t4t_tmp_v15, y_tile__t4t_tmp_v15)
            y_tile__p_r0_tmp_v24 = pl.tile.muls(y_tile__t2sq_tmp_v23, 2.6049265215988271e-06)
            y_tile__p_r1_tmp_v25 = pl.tile.adds(y_tile__p_r0_tmp_v24, -0.00019808944489341229)
            y_tile__p_t2_r1_tmp_v26 = pl.tile.mul(y_tile__p_r1_tmp_v25, y_tile__t2sq_tmp_v23)
            y_tile__p_r2_tmp_v27 = pl.tile.adds(y_tile__p_t2_r1_tmp_v26, 0.0083330497145652771)
            y_tile__p_t2_r2_tmp_v28 = pl.tile.mul(y_tile__p_r2_tmp_v27, y_tile__t2sq_tmp_v23)
            y_tile__p_r3_tmp_v29 = pl.tile.adds(y_tile__p_t2_r2_tmp_v28, -0.16666658222675323)
            y_tile__p_t2_r3_tmp_v30 = pl.tile.mul(y_tile__p_r3_tmp_v29, y_tile__t2sq_tmp_v23)
            y_tile__p_one_tmp_v31 = pl.tile.adds(y_tile__p_t2_r3_tmp_v30, 1.0)
            y_tile__t_p_tmp_v32 = pl.tile.mul(y_tile__t4t_tmp_v15, y_tile__p_one_tmp_v31)
            y_tile = pl.tile.mul(y_tile__sign_tmp_v22, y_tile__t_p_tmp_v32)
            out_0 = pl.tile.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0 = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            r = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_composite_ops()(Before)
    ir.assert_structural_equal(After, Expected)


def test_sin_lowering_is_idempotent():
    """Running the pass twice gives the same IR as running it once."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.sin(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    once = passes.lower_composite_ops()(Prog)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_cos_lowering_is_idempotent():
    """Running the pass twice on a cos program gives the same IR as once."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.cos(x_tile)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    once = passes.lower_composite_ops()(Prog)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_both_sin_and_cos_in_same_function():
    """Verify sin and cos lowering don't interfere when both appear in one function."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[16, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile: pl.Tile[[16, 16], pl.FP32] = pl.load(x, [0, 0], [16, 16])
            a: pl.Tile[[16, 16], pl.FP32] = pl.tile.sin(x_tile)
            b: pl.Tile[[16, 16], pl.FP32] = pl.tile.cos(x_tile)
            y_tile: pl.Tile[[16, 16], pl.FP32] = pl.tile.add(a, b)
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0: pl.Tensor[[16, 16], pl.FP32] = pl.create_tensor([16, 16], dtype=pl.FP32)
            r: pl.Tensor[[16, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    @pl.program
    class Expected:
        @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
        def main_incore_0(
            x: pl.Tensor[[16, 16], pl.FP32], out_0: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
        ) -> pl.Tensor[[16, 16], pl.FP32]:
            x_tile = pl.tile.load(x, [0, 0], [16, 16], [16, 16], target_memory=pl.Mem.Vec)
            a__pi_inv_x_tmp_v0 = pl.tile.muls(x_tile, 0.31830987334251404)
            a__k_i_tmp_v1 = pl.tile.cast(a__pi_inv_x_tmp_v0, target_type=pl.INT32, mode="round")
            a__k_f_tmp_v2 = pl.tile.cast(a__k_i_tmp_v1, target_type=pl.FP32, mode="none")
            a__k_pi_v2_tmp_v3 = pl.tile.muls(a__k_f_tmp_v2, 3.140625)
            a__t0_tmp_v4 = pl.tile.sub(x_tile, a__k_pi_v2_tmp_v3)
            a__k_pi_c1_tmp_v5 = pl.tile.muls(a__k_f_tmp_v2, 0.0009670257568359375)
            a__t1_tmp_v6 = pl.tile.sub(a__t0_tmp_v4, a__k_pi_c1_tmp_v5)
            a__k_pi_c2_tmp_v7 = pl.tile.muls(a__k_f_tmp_v2, 6.2771141529083252e-07)
            a__t2_tmp_v8 = pl.tile.sub(a__t1_tmp_v6, a__k_pi_c2_tmp_v7)
            a__k_pi_c3_tmp_v9 = pl.tile.muls(a__k_f_tmp_v2, 1.2164491636212915e-10)
            a__t3_tmp_v10 = pl.tile.sub(a__t2_tmp_v8, a__k_pi_c3_tmp_v9)
            a__k_pi_c4_tmp_v11 = pl.tile.muls(a__k_f_tmp_v2, -1.0290622927356871e-13)
            a__t4_tmp_v12 = pl.tile.sub(a__t3_tmp_v10, a__k_pi_c4_tmp_v11)
            a__half_k_tmp_v13 = pl.tile.muls(a__k_f_tmp_v2, 0.5)
            a__floor_hk_i_tmp_v14 = pl.tile.cast(a__half_k_tmp_v13, target_type=pl.INT32, mode="floor")
            a__floor_hk_f_tmp_v15 = pl.tile.cast(a__floor_hk_i_tmp_v14, target_type=pl.FP32, mode="none")
            a__floor_x4_tmp_v16 = pl.tile.muls(a__floor_hk_f_tmp_v15, 4.0)
            a__neg2_k_tmp_v17 = pl.tile.muls(a__k_f_tmp_v2, -2.0)
            a__sign_pre_tmp_v18 = pl.tile.add(a__floor_x4_tmp_v16, a__neg2_k_tmp_v17)
            a__sign_tmp_v19 = pl.tile.adds(a__sign_pre_tmp_v18, 1.0)
            a__t2sq_tmp_v20 = pl.tile.mul(a__t4_tmp_v12, a__t4_tmp_v12)
            a__p_r0_tmp_v21 = pl.tile.muls(a__t2sq_tmp_v20, 2.6049265215988271e-06)
            a__p_r1_tmp_v22 = pl.tile.adds(a__p_r0_tmp_v21, -0.00019808944489341229)
            a__p_t2_r1_tmp_v23 = pl.tile.mul(a__p_r1_tmp_v22, a__t2sq_tmp_v20)
            a__p_r2_tmp_v24 = pl.tile.adds(a__p_t2_r1_tmp_v23, 0.0083330497145652771)
            a__p_t2_r2_tmp_v25 = pl.tile.mul(a__p_r2_tmp_v24, a__t2sq_tmp_v20)
            a__p_r3_tmp_v26 = pl.tile.adds(a__p_t2_r2_tmp_v25, -0.16666658222675323)
            a__p_t2_r3_tmp_v27 = pl.tile.mul(a__p_r3_tmp_v26, a__t2sq_tmp_v20)
            a__p_one_tmp_v28 = pl.tile.adds(a__p_t2_r3_tmp_v27, 1.0)
            a__t_p_tmp_v29 = pl.tile.mul(a__t4_tmp_v12, a__p_one_tmp_v28)
            a = pl.tile.mul(a__sign_tmp_v19, a__t_p_tmp_v29)
            b__pi_inv_x_tmp_v30 = pl.tile.muls(x_tile, 0.31830987334251404)
            b__k_pre_tmp_v31 = pl.tile.adds(b__pi_inv_x_tmp_v30, 0.5)
            b__k_i_tmp_v32 = pl.tile.cast(b__k_pre_tmp_v31, target_type=pl.INT32, mode="rint")
            b__k_f_tmp_v33 = pl.tile.cast(b__k_i_tmp_v32, target_type=pl.FP32, mode="none")
            b__k_pi_v2_tmp_v34 = pl.tile.muls(b__k_f_tmp_v33, 3.140625)
            b__t0_tmp_v35 = pl.tile.sub(x_tile, b__k_pi_v2_tmp_v34)
            b__k_pi_c1_tmp_v36 = pl.tile.muls(b__k_f_tmp_v33, 0.0009670257568359375)
            b__t1_tmp_v37 = pl.tile.sub(b__t0_tmp_v35, b__k_pi_c1_tmp_v36)
            b__t1h_tmp_v38 = pl.tile.adds(b__t1_tmp_v37, 1.5707963705062866)
            b__k_pi_c2_tmp_v39 = pl.tile.muls(b__k_f_tmp_v33, 6.2771141529083252e-07)
            b__t2_tmp_v40 = pl.tile.sub(b__t1h_tmp_v38, b__k_pi_c2_tmp_v39)
            b__k_pi_c3_tmp_v41 = pl.tile.muls(b__k_f_tmp_v33, 1.2164491636212915e-10)
            b__t3_tmp_v42 = pl.tile.sub(b__t2_tmp_v40, b__k_pi_c3_tmp_v41)
            b__k_pi_c4_tmp_v43 = pl.tile.muls(b__k_f_tmp_v33, -1.0290622927356871e-13)
            b__t4_tmp_v44 = pl.tile.sub(b__t3_tmp_v42, b__k_pi_c4_tmp_v43)
            b__t4t_tmp_v45 = pl.tile.adds(b__t4_tmp_v44, -4.3711388286737929e-08)
            b__half_k_tmp_v46 = pl.tile.muls(b__k_f_tmp_v33, 0.5)
            b__floor_hk_i_tmp_v47 = pl.tile.cast(b__half_k_tmp_v46, target_type=pl.INT32, mode="floor")
            b__floor_hk_f_tmp_v48 = pl.tile.cast(b__floor_hk_i_tmp_v47, target_type=pl.FP32, mode="none")
            b__floor_x4_tmp_v49 = pl.tile.muls(b__floor_hk_f_tmp_v48, 4.0)
            b__neg2_k_tmp_v50 = pl.tile.muls(b__k_f_tmp_v33, -2.0)
            b__sign_pre_tmp_v51 = pl.tile.add(b__floor_x4_tmp_v49, b__neg2_k_tmp_v50)
            b__sign_tmp_v52 = pl.tile.adds(b__sign_pre_tmp_v51, 1.0)
            b__t2sq_tmp_v53 = pl.tile.mul(b__t4t_tmp_v45, b__t4t_tmp_v45)
            b__p_r0_tmp_v54 = pl.tile.muls(b__t2sq_tmp_v53, 2.6049265215988271e-06)
            b__p_r1_tmp_v55 = pl.tile.adds(b__p_r0_tmp_v54, -0.00019808944489341229)
            b__p_t2_r1_tmp_v56 = pl.tile.mul(b__p_r1_tmp_v55, b__t2sq_tmp_v53)
            b__p_r2_tmp_v57 = pl.tile.adds(b__p_t2_r1_tmp_v56, 0.0083330497145652771)
            b__p_t2_r2_tmp_v58 = pl.tile.mul(b__p_r2_tmp_v57, b__t2sq_tmp_v53)
            b__p_r3_tmp_v59 = pl.tile.adds(b__p_t2_r2_tmp_v58, -0.16666658222675323)
            b__p_t2_r3_tmp_v60 = pl.tile.mul(b__p_r3_tmp_v59, b__t2sq_tmp_v53)
            b__p_one_tmp_v61 = pl.tile.adds(b__p_t2_r3_tmp_v60, 1.0)
            b__t_p_tmp_v62 = pl.tile.mul(b__t4t_tmp_v45, b__p_one_tmp_v61)
            b = pl.tile.mul(b__sign_tmp_v52, b__t_p_tmp_v62)
            y_tile = pl.tile.add(a, b)
            out_0 = pl.tile.store(y_tile, [0, 0], out_0)
            return out_0

        @pl.function
        def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
            out_0 = pl.tensor.create([16, 16], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            r = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_composite_ops()(Before)
    ir.assert_structural_equal(After, Expected)


def test_sin_in_return_stmt_is_decomposed():
    """A ``tile.sin`` Call placed directly inside ``ReturnStmt::value_`` (i.e.
    not pre-bound to an AssignStmt — the shape pre-SSA / standalone callers can
    surface) must still be decomposed by the pass.

    SSA-form programs never produce this shape (every Call is bound to an
    AssignStmt), so the test constructs the IR programmatically via the IR
    builder API to exercise the ``VisitStmt_(ReturnStmtPtr)`` override.
    """
    span = ir.Span.unknown()
    tile_type = ir.TileType([16, 16], ir.DataType.FP32)

    x_param = ir.Var("x", tile_type, span)
    sin_call = ir.create_op_call("tile.sin", [x_param], {}, span)
    body = ir.ReturnStmt([sin_call], span)
    func = ir.Function("trig_return", [x_param], [tile_type], body, span, ir.FunctionType.InCore)
    prog = ir.Program([func], "test_program", span)

    after = passes.lower_composite_ops()(prog)
    op_names = set(_collect_op_names(after))

    # The trig op embedded directly in ReturnStmt must be lowered.
    assert "tile.sin" not in op_names

    # Decomposition primitives must appear in the lowered IR.
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


def test_cos_in_return_stmt_is_decomposed():
    """Mirror of ``test_sin_in_return_stmt_is_decomposed`` for ``tile.cos``."""
    span = ir.Span.unknown()
    tile_type = ir.TileType([16, 16], ir.DataType.FP32)

    x_param = ir.Var("x", tile_type, span)
    cos_call = ir.create_op_call("tile.cos", [x_param], {}, span)
    body = ir.ReturnStmt([cos_call], span)
    func = ir.Function("trig_return", [x_param], [tile_type], body, span, ir.FunctionType.InCore)
    prog = ir.Program([func], "test_program", span)

    after = passes.lower_composite_ops()(prog)
    op_names = set(_collect_op_names(after))

    assert "tile.cos" not in op_names
    assert _DECOMP_PRIMITIVES & op_names, "lowering produced no primitive ops"


# ============================================================================
# pld.tensor.allreduce lowering
#
# The allreduce rule is the first composite-op rule that uses LoweringBuilder's
# structured control-flow primitives (EmitFor / EmitForReduce / EmitIf /
# EmitIfExpr). These tests pin the invariants of the lowering — primitive op
# set, presence of For / If structure, in-place rebind semantics, and
# idempotency — without hand-mirroring every temp name.
# ============================================================================

_ALLREDUCE_SIZE = 16
_ALLREDUCE_NRANKS = 2
_ALLREDUCE_FP32_CHUNK = 4096
_ALLREDUCE_REDUCE_CASES = [
    (pld.ReduceOp.Sum, ir.get_op("tile.add").name),
    (pld.ReduceOp.Max, ir.get_op("tile.maximum").name),
    (pld.ReduceOp.Min, ir.get_op("tile.minimum").name),
    (pld.ReduceOp.Prod, ir.get_op("tile.mul").name),
]

# Ops the chunked mesh decomposition must emit.
_ALLREDUCE_REQUIRED_OPS = {
    ir.get_op("pld.system.get_comm_ctx").name,
    ir.get_op("pld.system.nranks").name,
    ir.get_op("pld.system.rank").name,
    ir.get_op("pld.system.notify").name,  # Ready and per-chunk read-complete barriers
    ir.get_op("pld.system.wait").name,  # Ready and per-chunk read-complete barriers
    ir.get_op("pld.tile.remote_load").name,  # Peer chunk load
    ir.get_op("tile.fillpad_inplace").name,  # Zero ragged padding before accumulation
    ir.get_op("tile.add").name,  # Accumulate peer chunks
    ir.get_op("tile.load").name,  # Self chunk load + user-side load
    ir.get_op("tile.set_validshape").name,  # Narrow the final ragged chunk
    ir.get_op("tile.store").name,  # Per-chunk result + user-side store
}


class _StmtKindCollector(ir.IRVisitor):
    """Walk IR and tally the kinds of every Stmt encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.for_count = 0
        self.if_count = 0

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        self.for_count += 1
        self._walk_stmt(op.body)

    def visit_if_stmt(self, op: ir.IfStmt) -> None:
        self.if_count += 1
        self._walk_stmt(op.then_body)
        if op.else_body is not None:
            self._walk_stmt(op.else_body)

    def _walk_stmt(self, stmt: ir.Stmt) -> None:
        # The nanobind trampoline's base implementation does not redispatch
        # nested statement callbacks to Python overrides.
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                self._walk_stmt(child)
        elif isinstance(stmt, ir.ForStmt):
            self.visit_for_stmt(stmt)
        elif isinstance(stmt, ir.IfStmt):
            self.visit_if_stmt(stmt)


def _build_allreduce_before(
    size: int = _ALLREDUCE_SIZE,
    reduce_op: pld.ReduceOp = pld.ReduceOp.Sum,
):
    """Build a minimal Before program that calls ``pld.tensor.allreduce``."""
    SIZE = size
    nr = _ALLREDUCE_NRANKS
    REDUCE_OP = reduce_op

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            local = pl.load(inp, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)
            data = pld.tensor.allreduce(data, signal, op=REDUCE_OP)
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

    return Before


def test_allreduce_is_decomposed_to_primitives():
    """The composite call is replaced by its chunked mesh decomposition."""
    Before = _build_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.allreduce" not in op_names, (
        "lower_composite_ops must remove the composite allreduce call entirely"
    )
    missing = _ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_allreduce_in_host_orchestrator_is_left_for_host_collective_lowering():
    """Host-level allreduce is lowered by LowerHostTensorCollectives, not here."""
    SIZE = _ALLREDUCE_SIZE

    @pl.program
    class HostAllreduce:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            data: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[2, 1], pl.INT32],
        ):
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return 0

    After = passes.lower_composite_ops()(HostAllreduce)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.allreduce" in op_names
    assert "pld.system.notify" not in op_names
    assert "pld.tile.remote_load" not in op_names


def test_new_host_collectives_in_host_orchestrator_are_left_for_host_collective_lowering():
    """HOST collectives are skipped by LowerCompositeOps (left for host lower)."""
    SIZE = 64
    NR = 2

    @pl.program
    class HostCollectives:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            ag_stage: pld.DistributedTensor[[1, SIZE], pl.FP32],
            stage: pld.DistributedTensor[[NR, SIZE], pl.FP32],
            data: pld.DistributedTensor[[NR, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[NR, 1], pl.INT32],
        ):
            pld.tensor.barrier(signal)
            data = pld.tensor.broadcast(data, signal, root=0)
            data = pld.tensor.allgather(ag_stage, data, signal)
            data = pld.tensor.all_to_all(stage, data, signal)
            data = pld.tensor.reduce_scatter(data, signal)
            return 0

    After = passes.lower_composite_ops()(HostCollectives)
    op_names = set(_collect_op_names(After))

    for op_name in (
        "pld.tensor.barrier",
        "pld.tensor.broadcast",
        "pld.tensor.allgather",
        "pld.tensor.all_to_all",
        "pld.tensor.reduce_scatter",
    ):
        assert op_name in op_names, f"HOST collective {op_name!r} should survive LowerCompositeOps"
    assert "pld.system.notify" not in op_names


class _StmtProbe(ir.IRVisitor):
    """Collect ForStmt ``stop`` expressions and AssignStmt values."""

    def __init__(self) -> None:
        super().__init__()
        self.for_stops: list[ir.Expr] = []
        self.assign_values: list[ir.Expr] = []

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        self.for_stops.append(op.stop)
        self._walk_stmt(op.body)

    def visit_if_stmt(self, op: ir.IfStmt) -> None:
        self._walk_stmt(op.then_body)
        if op.else_body is not None:
            self._walk_stmt(op.else_body)

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        self.assign_values.append(op.value)

    def _walk_stmt(self, stmt: ir.Stmt) -> None:
        # The nanobind trampoline does not redispatch nested statement
        # callbacks to Python overrides (see _StmtKindCollector).
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                self._walk_stmt(child)
        elif isinstance(stmt, ir.ForStmt):
            self.visit_for_stmt(stmt)
        elif isinstance(stmt, ir.IfStmt):
            self.visit_if_stmt(stmt)
        elif isinstance(stmt, ir.AssignStmt):
            self.visit_assign_stmt(stmt)


def _probe_stmts(prog) -> _StmtProbe:
    probe = _StmtProbe()
    probe.visit_program(prog)
    return probe


_AAV_SIZE = 16
_AAV_NRANKS = 2
_AAV_MAX_RECV = 2
_AAV_TOTAL = _AAV_NRANKS * _AAV_MAX_RECV


def _build_all_to_all_v_before():
    """InCore program calling ``pld.tensor.all_to_all_v`` with runtime counts."""
    SIZE = _AAV_SIZE
    nr = _AAV_NRANKS
    total = _AAV_TOTAL

    @pl.program
    class AllToAllV:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1], pl.INT32],
            out: pl.Out[pl.Tensor[[total, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pl.Tensor[[total, SIZE], pl.FP32]:
            result = pld.tensor.all_to_all_v(inp, data, signal, counts, recv_counts)
            row = pl.load(result, [0, 0], [1, SIZE])
            return pl.store(row, [0, 0], out)

    return AllToAllV


def test_all_to_all_v_push_loop_is_bounded_by_runtime_send_counts():
    """The push loop is bounded by ``send_counts[dest]`` read at runtime, not by
    the compile-time MAX_RECV capacity — a capacity-bounded loop would transfer
    padding rows the destination never asked for."""
    After = passes.lower_composite_ops()(_build_all_to_all_v_before())
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.all_to_all_v" not in op_names, "composite op must be fully lowered"
    assert "tensor.read" in op_names, "send_counts must be read at runtime to bound the push loop"
    assert "pld.tile.put" in op_names, "rows are pushed to peers via TPUT"
    assert "pld.system.notify" in op_names
    assert "pld.system.wait" in op_names

    probe = _probe_stmts(After)

    # No loop may be bounded by the MAX_RECV capacity constant: the row loop's
    # bound is the (clamped) runtime count.
    const_stops = [s.value for s in probe.for_stops if isinstance(s, ir.ConstInt)]
    assert _AAV_MAX_RECV not in const_stops, (
        f"a loop is still bounded by the MAX_RECV capacity ({_AAV_MAX_RECV}); "
        f"constant loop bounds found: {const_stops}"
    )

    # The runtime count is clamped against the capacity, so a count larger than
    # MAX_RECV cannot push into the next destination's slice of the peer window.
    clamps = [v for v in probe.assign_values if isinstance(v, ir.Min)]
    assert clamps, "the runtime send count must be clamped (min) against the MAX_RECV capacity"
    clamp_operands = [
        operand.value
        for clamp in clamps
        for operand in (clamp.left, clamp.right)
        if isinstance(operand, ir.ConstInt)
    ]
    assert _AAV_MAX_RECV in clamp_operands, (
        f"the clamp must bound the count by MAX_RECV ({_AAV_MAX_RECV}); "
        f"constant clamp operands found: {clamp_operands}"
    )


def test_all_to_all_v_in_host_orchestrator_is_left_for_host_collective_lowering():
    """Host-level all_to_all_v is lowered by LowerHostTensorCollectives (via its
    builtin.tensor.all_to_all_v rule), not here — LowerCompositeOps must defer
    it unchanged rather than reject it."""
    SIZE = _AAV_SIZE
    nr = _AAV_NRANKS
    total = _AAV_TOTAL

    @pl.program
    class HostAllToAllV:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1], pl.INT32],
            data: pld.DistributedTensor[[total, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[nr, 1], pl.INT32],
            recv_counts: pld.DistributedTensor[[nr, 1], pl.INT32],
        ):
            data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv_counts)  # type: ignore[arg-type]
            return 0

    After = passes.lower_composite_ops()(HostAllToAllV)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.all_to_all_v" in op_names
    assert "pld.system.notify" not in op_names
    assert "pld.tile.put" not in op_names


def test_allreduce_without_signal_is_rejected_outside_host_orchestrator():
    SIZE = _ALLREDUCE_SIZE

    @pl.program
    class MissingSignal:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            local = pl.load(inp, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)
            data = pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

    with pytest.raises(ValueError, match="requires an explicit signal outside host orchestrator"):
        passes.lower_composite_ops()(MissingSignal)


def test_allreduce_eval_stmt_without_signal_is_rejected_outside_host_orchestrator():
    SIZE = _ALLREDUCE_SIZE

    @pl.program
    class MissingSignalEval:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            pld.tensor.allreduce(data, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(ValueError, match="requires an explicit signal outside host orchestrator"):
        passes.lower_composite_ops()(MissingSignalEval)


def test_allreduce_eval_stmt_with_signal_is_decomposed():
    SIZE = _ALLREDUCE_SIZE
    nr = _ALLREDUCE_NRANKS

    @pl.program
    class EvalAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    After = passes.lower_composite_ops()(EvalAllreduce)
    op_names = set(_collect_op_names(After))

    assert ir.get_op("pld.tensor.allreduce").name not in op_names
    missing = _ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_incore_allreduce_rejects_local_multicore_request():
    SIZE = _ALLREDUCE_SIZE
    nr = _ALLREDUCE_NRANKS

    @pl.program
    class MultiCoreAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 4], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            data = pld.tensor.allreduce(data, signal, core_num=4)
            return data

    with pytest.raises(ValueError, match="supported only in a HOST orchestrator"):
        passes.lower_composite_ops()(MultiCoreAllreduce)


def test_allreduce_in_for_loop_now_succeeds():
    """A dynamic trip count used to have no compile-time generation to wait for,
    so this was rejected. The self-clearing epilogue (each call restarts at
    all-zero) removes that restriction — see ``lower_composite_ops_pass.cpp``'s
    deleted ``CheckCollectiveLoopUse``.
    """
    SIZE = _ALLREDUCE_SIZE
    nr = _ALLREDUCE_NRANKS

    @pl.program
    class LoopAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            for _ in pl.range(2):
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    After = passes.lower_composite_ops()(LoopAllreduce)
    assert _static_wait_expectations(After) == [1]


def test_allreduce_in_while_loop_now_succeeds():
    SIZE = _ALLREDUCE_SIZE
    nr = _ALLREDUCE_NRANKS

    @pl.program
    class LoopAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            while True:
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    After = passes.lower_composite_ops()(LoopAllreduce)
    assert _static_wait_expectations(After) == [1]


def test_all_to_all_v_in_for_loop_now_succeeds():
    """The self-clearing epilogue (each call restarts at all-zero) removes the
    old loop restriction for all collectives.  all_to_all_v lowers through
    the same barrier path as the other six — see the deleted
    ``CheckCollectiveLoopUse`` in ``lower_composite_ops_pass.cpp``.
    """
    SIZE = _AAV_SIZE
    nr = _AAV_NRANKS
    total = _AAV_TOTAL

    @pl.program
    class LoopAllToAllV:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1], pl.INT32],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[total, SIZE], pl.FP32]:
            for _ in pl.range(2):
                data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv_counts)
            return data

    passes.lower_composite_ops()(LoopAllToAllV)


def test_all_to_all_v_in_while_loop_now_succeeds():
    """Same as above — while loops are legal under the self-clearing protocol."""
    SIZE = _AAV_SIZE
    nr = _AAV_NRANKS
    total = _AAV_TOTAL

    @pl.program
    class LoopAllToAllV:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_step(
            self,
            inp: pl.Tensor[[total, SIZE], pl.FP32],
            counts: pl.Tensor[[nr, 1], pl.INT32],
            data: pl.InOut[pld.DistributedTensor[[total, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            recv_counts: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[total, SIZE], pl.FP32]:
            while True:
                data = pld.tensor.all_to_all_v(inp, data, signal, counts, recv_counts)
            return data

    passes.lower_composite_ops()(LoopAllToAllV)


def test_allreduce_emits_for_and_if_control_flow():
    """The chunked recipe emits six ForStmts and five IfStmts:

    * Phase 2a (notify all peers) — for + if
    * Phase 2b (wait on all peers) — for + if
    * Phase 3 chunk traversal — one for loop
    * Phase 3 peer reduction — for + if
    * Per-chunk Phase 3.5 re-notify — for + if
    * Per-chunk Phase 3.5 re-wait — for + if

    Phase 3.5 is a second cross-rank barrier inserted between Phase 3
    (read peers via ``pld.tile.remote_load``) and Phase 4 (write reduced
    value back into ``target``). Without it, a fast rank could overwrite
    its slot while slower ranks are still reading the staged Phase-1 data
    — a write-after-read race that manifests as off-by-N*peer drift on
    slower ranks at P>=4.

    This pins the structured control-flow shape so a refactor that
    collapses or drops any of the loops surfaces here."""
    Before = _build_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    assert collector.for_count == 7, (
        f"expected 7 ForStmts (notify,wait,chunk,reduce,renotify,rewait,epilogue), got {collector.for_count}"
    )
    assert collector.if_count == 6, f"expected 6 peer-filter IfStmts, got {collector.if_count}"


def test_allreduce_flattens_target_to_2d_view_for_mesh_lowering():
    """Fully-valid mesh allreduce treats packed ND storage as one linear stream."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[2, 3, 4], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[2, 3, 4], pl.FP32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    After = passes.lower_composite_ops()(Before)
    op_names = _collect_op_names(After)
    assert "pld.tensor.allreduce" not in op_names
    assert "tensor.view" in op_names

    func = After.get_function("reduce_step")
    assert func is not None
    body = func.body
    assert isinstance(body, ir.SeqStmts)
    view_stmt = next(
        stmt
        for stmt in body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tensor.view").name
    )
    view_type = view_stmt.var.type
    assert isinstance(view_type, ir.DistributedTensorType)
    assert view_type.shape == [1, 24]


def test_allreduce_mesh_lowering_preserves_partial_valid_shape():
    """Mesh allreduce operates only on the target's representable valid rectangle."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, 3, 2], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(valid_shape=[1, 3, 2], stride=[], layout=pl.TensorLayout.ND),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    After = passes.lower_composite_ops()(Before)
    func = After.get_function("reduce_step")
    assert func is not None
    assert isinstance(func.body, ir.SeqStmts)

    view_stmt = next(
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tensor.view").name
    )
    view_type = view_stmt.var.type
    assert isinstance(view_stmt.value, ir.Call)
    assert len(view_stmt.value.args) == 3
    assert isinstance(view_type, ir.DistributedTensorType)
    assert view_type.shape == [6, 4]
    assert view_type.tensor_view is not None
    assert view_type.tensor_view.valid_shape == [3, 2]

    class CallCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            self.calls.append(op)
            super().visit_call(op)

    collector = CallCollector()
    collector.visit_program(After)
    load = next(call for call in collector.calls if call.op.name == _OP_TILE_LOAD)
    remote_load = next(call for call in collector.calls if call.op.name == _OP_PLD_TILE_REMOTE_LOAD)
    load_shape = load.args[2]
    load_valid_shape = load.args[3]
    remote_shape = remote_load.args[3]
    assert isinstance(load_shape, ir.MakeTuple)
    assert isinstance(load_valid_shape, ir.MakeTuple)
    assert isinstance(remote_shape, ir.MakeTuple)
    assert load_shape.elements == [3, 2]
    assert load_valid_shape.elements == [3, 2]
    assert remote_shape.elements == [3, 2]


@pytest.mark.parametrize("size", [1, 3, 17, 4096, 4097, 65537])
def test_allreduce_mesh_chunks_non_aligned_and_larger_than_ub(size):
    """Chunk bounds, widths, and offsets cover every logical element exactly."""
    Before = _build_allreduce_before(size)
    After = passes.lower_composite_ops()(Before)
    aligned_chunk = min(_ALLREDUCE_FP32_CHUNK, ((size + 7) // 8) * 8)
    expected_step = aligned_chunk
    expected_rows = 1
    expected_cols = aligned_chunk
    assert expected_rows * expected_cols * 4 % 32 == 0

    class CallCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            self.calls.append(op)
            super().visit_call(op)

    collector = CallCollector()
    collector.visit_program(After)
    lowered_load = next(
        call
        for call in collector.calls
        if call.op.name == ir.get_op("tile.load").name
        and isinstance(call.args[0].type, ir.DistributedTensorType)
        and isinstance(call.args[2], ir.MakeTuple)
        and isinstance(call.args[2].elements[0], ir.ConstInt)
        and call.args[2].elements[0].value == expected_rows
        and isinstance(call.args[2].elements[1], ir.ConstInt)
        and call.args[2].elements[1].value == expected_cols
        and isinstance(call.args[3], ir.MakeTuple)
        and isinstance(call.args[3].elements[1], ir.Min)
    )
    remote_load = next(
        call for call in collector.calls if call.op.name == ir.get_op("pld.tile.remote_load").name
    )

    loops: list[ir.ForStmt] = []

    def collect_loops(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                collect_loops(child)
        elif isinstance(stmt, ir.ForStmt):
            loops.append(stmt)
            collect_loops(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            collect_loops(stmt.then_body)
            if stmt.else_body is not None:
                collect_loops(stmt.else_body)

    func = After.get_function("reduce_step")
    assert func is not None
    collect_loops(func.body)
    chunk_loop = next(
        loop for loop in loops if isinstance(loop.stop, ir.ConstInt) and loop.stop.value == size
    )

    assert len(remote_load.args) == 5
    assert isinstance(chunk_loop.start, ir.ConstInt) and chunk_loop.start.value == 0
    assert isinstance(chunk_loop.step, ir.ConstInt) and chunk_loop.step.value == expected_step
    assert isinstance(lowered_load.args[3], ir.MakeTuple)
    assert isinstance(remote_load.args[4], ir.MakeTuple)
    assert isinstance(lowered_load.args[3].elements[1], ir.Min)
    ir.assert_structural_equal(remote_load.args[4], lowered_load.args[3])
    assert isinstance(lowered_load.args[1], ir.MakeTuple)
    assert isinstance(remote_load.args[2], ir.MakeTuple)
    ir.assert_structural_equal(remote_load.args[2], lowered_load.args[1])
    assert isinstance(lowered_load.args[1].elements[0], ir.ConstInt)
    assert lowered_load.args[1].elements[0].value == 0
    ir.assert_structural_equal(lowered_load.args[1].elements[1], chunk_loop.loop_var)


def test_allreduce_rejects_oversized_partial_valid_shape():
    """A partial rectangle cannot use a single tile larger than the chunk budget."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 65537],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, 65537], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 65537],
            pl.FP32,
            pl.TensorView(valid_shape=[1, 65537], stride=[], layout=pl.TensorLayout.ND),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(ValueError, match="partial valid_shape must fit within one 16384-byte mesh chunk"):
        passes.lower_composite_ops()(Before)


def test_allreduce_dynamic_partial_valid_shape_uses_bounded_physical_rectangle(
    default_pass_manager, ascend_backend
):
    """A symbolic valid extent reaches PTO with a bounded physical rectangle."""
    m = pl.dynamic("ALLREDUCE_VALID_M")

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [64, 64],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, m], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            shape_anchor: pl.Tensor[[1, m], pl.FP32],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [64, 64],
            pl.FP32,
            pl.TensorView(valid_shape=[1, m], stride=[], layout=pl.TensorLayout.ND),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    # First pin the LowerCompositeOps contract directly: the source rectangle
    # stays statically bounded while only its valid width remains symbolic.
    After = passes.lower_composite_ops()(Before)

    class CallCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            self.calls.append(op)
            super().visit_call(op)

    collector = CallCollector()
    collector.visit_program(After)
    load = next(
        call
        for call in collector.calls
        if call.op.name == ir.get_op("tile.load").name
        and isinstance(call.args[0].type, ir.DistributedTensorType)
    )
    remote_load = next(
        call for call in collector.calls if call.op.name == ir.get_op("pld.tile.remote_load").name
    )

    assert isinstance(load.args[2], ir.MakeTuple)
    assert load.args[2].elements == [64, 64]
    assert isinstance(load.args[3], ir.MakeTuple)
    assert load.args[3].elements[0] == 1
    assert isinstance(load.args[3].elements[1], ir.Var)
    assert load.args[3].elements[1].name_hint == "ALLREDUCE_VALID_M"
    assert len(remote_load.args) == 5
    ir.assert_structural_equal(remote_load.args[3], load.args[2])
    ir.assert_structural_equal(remote_load.args[4], load.args[3])

    # Also exercise the real default pass pipeline and PTO codegen. The
    # shape_anchor parameter supplies the runtime binding for ``m``.
    from pypto import codegen  # noqa: PLC0415

    optimized = default_pass_manager.run_passes(Before)
    func = optimized.get_function("reduce_step")
    assert func is not None
    single = ir.Program([func], func.name, optimized.span)
    mlir = codegen.PTOCodegen().generate(single)
    assert "pto.tload" in mlir
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert re.search(r"sizes = \[%c1_index, %[A-Za-z0-9_.$]+\]", remote_partition), remote_partition


def test_allreduce_mesh_lowering_rejects_noncontiguous_partial_valid_shape():
    """A partial box spanning disjoint flattened row ranges cannot use one 2D view."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(valid_shape=[2, 2, 4], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(valid_shape=[2, 2, 4], stride=[], layout=pl.TensorLayout.ND),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(ValueError, match="valid_shape cannot be represented by a single 2D view"):
        passes.lower_composite_ops()(Before)


def test_allreduce_mesh_lowering_rejects_strided_target_collapse():
    """Flattening must not replace a legal strided-family view with packed storage."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(stride=[100, 10, 1], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(stride=[100, 10, 1], layout=pl.TensorLayout.ND),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(ValueError, match="requires a packed source"):
        passes.lower_composite_ops()(Before)


def test_allreduce_mesh_lowering_accepts_explicit_packed_nd_stride():
    """An explicitly materialized packed ND stride is still flattenable."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(stride=[12, 4, 1], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(stride=[12, 4, 1], layout=pl.TensorLayout.ND),
        ]:
            return pld.tensor.allreduce(data, signal)

    After = passes.lower_composite_ops()(Before)
    op_names = _collect_op_names(After)
    assert ir.get_op("pld.tensor.allreduce").name not in op_names
    assert ir.get_op("tensor.view").name in op_names


def test_allreduce_mesh_lowering_rejects_fully_valid_dn_target():
    """A fully-valid DN view is not a row-major linear stream."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(stride=[12, 1, 3], layout=pl.TensorLayout.DN),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(stride=[12, 1, 3], layout=pl.TensorLayout.DN),
        ]:
            return pld.tensor.allreduce(data, signal)

    with pytest.raises(ValueError, match="only supports ND layout"):
        passes.lower_composite_ops()(Before)


def test_allreduce_mesh_lowering_rejects_partial_dn_target_collapse():
    """The row-major partial-valid collapse is not valid for DN addresses."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, 3, 2], stride=[12, 1, 3], layout=pl.TensorLayout.DN),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(valid_shape=[1, 3, 2], stride=[12, 1, 3], layout=pl.TensorLayout.DN),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    with pytest.raises(ValueError, match="only supports ND layout"):
        passes.lower_composite_ops()(Before)


def test_allreduce_flattened_mesh_lowering_reaches_pto_codegen(default_pass_manager, ascend_backend):
    """Default pipeline codegens the three-arg partial-valid target view."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, 3, 2], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(valid_shape=[1, 3, 2], stride=[], layout=pl.TensorLayout.ND),
        ]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    from pypto import codegen  # noqa: PLC0415

    optimized = default_pass_manager.run_passes(Before)
    func = optimized.get_function("reduce_step")
    assert func is not None
    view_stmt = next(
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tensor.view").name
    )
    assert isinstance(view_stmt.value, ir.Call)
    assert len(view_stmt.value.args) == 3
    assert isinstance(view_stmt.var.type, ir.DistributedTensorType)
    assert view_stmt.var.type.shape == [6, 4]
    assert view_stmt.var.type.tensor_view is not None
    assert view_stmt.var.type.tensor_view.valid_shape == [3, 2]
    single = ir.Program([func], func.name, optimized.span)
    mlir = codegen.PTOCodegen().generate(single)
    assert "tile.store tile valid_shape must be 2D" not in mlir


def test_allreduce_large_ragged_mesh_lowering_reaches_pto_codegen(default_pass_manager, ascend_backend):
    """A larger-than-UB ragged length emits dominance-safe PTO."""
    from pypto import codegen  # noqa: PLC0415

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, 65537], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, 65537], pl.FP32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    optimized = default_pass_manager.run_passes(Before)
    func = optimized.get_function("reduce_step")
    assert func is not None
    single = ir.Program([func], func.name, optimized.span)
    mlir = codegen.PTOCodegen().generate(single)

    assert "scf.for" in mlir
    assert "4096" in mlir
    assert "pto.tload" in mlir
    assert "pto.tfillpad" in mlir
    assert "pto.set_validshape" in mlir
    assert "pto.tstore" in mlir

    # Regression: alloc_tile declarations are sometimes emitted outside the
    # control-flow region that produced the corresponding tile. A dynamic
    # valid_col from inside the chunk/peer loop must never leak into such a
    # hoisted declaration (PTOAS reports "operand does not dominate this use").
    definitions: dict[str, int] = {}
    lines = mlir.splitlines()
    for line_number, line in enumerate(lines):
        for argument in re.findall(r"(%[A-Za-z0-9_.$]+)\s*:", line) if "func.func" in line else ():
            definitions.setdefault(argument, line_number)
        definition = re.search(r"(?:^\s*|scf\.for\s+)(%[A-Za-z0-9_.$]+)\s*=", line)
        if definition:
            definitions.setdefault(definition.group(1), line_number)

    for line_number, line in enumerate(lines):
        if "pto.alloc_tile" not in line:
            continue
        for operand in re.findall(r"valid_(?:row|col)\s*=\s*(%[A-Za-z0-9_.$]+)", line):
            assert operand in definitions, f"missing definition for {operand}: {line}"
            assert definitions[operand] < line_number, (
                f"{operand} is defined at line {definitions[operand] + 1} after alloc_tile "
                f"use at line {line_number + 1}: {line}"
            )


def test_allreduce_dynamic_mesh_lowering_reaches_pto_codegen(default_pass_manager, ascend_backend):
    """A fully dynamic packed target is chunked by the real default pipeline."""
    from pypto import codegen  # noqa: PLC0415

    n = pl.dynamic("ALLREDUCE_DYNAMIC_N")

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, n], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, n], pl.FP32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            return data

    # A dynamic dimension that appears only on DistributedTensor is not yet
    # self-contained in Python printer roundtrips. Keep pass verification, but
    # skip the unrelated RoundtripInstrument limitation for this pipeline test.
    from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = default_pass_manager.run_passes(Before)
    func = optimized.get_function("reduce_step")
    assert func is not None
    single = ir.Program([func], func.name, optimized.span)
    mlir = codegen.PTOCodegen().generate(single)

    assert "scf.for" in mlir
    assert "4096" in mlir
    assert "arith.minsi" in mlir
    assert "pto.tload" in mlir
    remote_partition = next(
        line for line in mlir.splitlines() if "pto.partition_view" in line and "_peer" in line
    )
    assert re.search(r"sizes = \[%c1_index, %[A-Za-z0-9_.$]+\]", remote_partition), remote_partition


def test_allreduce_lowering_is_idempotent():
    """Running the pass on already-lowered IR is a no-op — the second pass
    has nothing left to rewrite."""
    Before = _build_allreduce_before()
    once = passes.lower_composite_ops()(Before)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_allreduce_noop_when_only_user_call_chain():
    """Programs that never call ``pld.tensor.allreduce`` are left
    structurally unchanged (sanity check the dispatch table)."""

    @pl.program
    class NoAllreduce:
        @pl.function(type=pl.FunctionType.InCore)
        def main_incore_0(
            self,
            x: pl.Tensor[[1, 16], pl.FP32],
            out_0: pl.Out[pl.Tensor[[1, 16], pl.FP32]],
        ) -> pl.Tensor[[1, 16], pl.FP32]:
            tile = pl.load(x, [0, 0], [1, 16])
            return pl.store(tile, [0, 0], out_0)

        @pl.function
        def main(self, x: pl.Tensor[[1, 16], pl.FP32]) -> pl.Tensor[[1, 16], pl.FP32]:
            out_0: pl.Tensor[[1, 16], pl.FP32] = pl.create_tensor([1, 16], dtype=pl.FP32)
            r: pl.Tensor[[1, 16], pl.FP32] = self.main_incore_0(x, out_0)
            return r

    After = passes.lower_composite_ops()(NoAllreduce)
    ir.assert_structural_equal(After, NoAllreduce)


def test_allreduce_deducer_rejects_plain_tensor():
    """Passing a plain :class:`pl.Tensor` as the ``target`` argument must
    fail at IR-construction time — the deducer enforces window-bound
    semantics so misuse cannot reach the lowering pass."""
    SIZE = _ALLREDUCE_SIZE

    with pytest.raises((ValueError, TypeError, ParserError)):

        @pl.program
        class Bad:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                local: pl.Tensor[[1, SIZE], pl.FP32],  # plain tensor — not distributed
                signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
            ) -> pl.Tensor[[1, SIZE], pl.FP32]:
                # Intentional type misuse — verifies the runtime deducer
                # rejects a plain Tensor where a DistributedTensor is expected.
                local = pld.tensor.allreduce(local, signal, op=pld.ReduceOp.Sum)  # pyright: ignore[reportArgumentType]
                return local


@pytest.mark.parametrize(("reduce_op", "expected_tile_op"), _ALLREDUCE_REDUCE_CASES)
def test_allreduce_lowers_every_reduce_op(reduce_op, expected_tile_op):
    Before = _build_allreduce_before(reduce_op=reduce_op)
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert ir.get_op("pld.tensor.allreduce").name not in op_names
    assert expected_tile_op in op_names
    for _, other_tile_op in _ALLREDUCE_REDUCE_CASES:
        if other_tile_op != expected_tile_op:
            assert other_tile_op not in op_names


# ============================================================================
# pld.tensor.barrier lowering
# ============================================================================

_BARRIER_NRANKS = 2
_BARRIER_REQUIRED_OPS = {
    ir.get_op("pld.system.get_comm_ctx").name,
    ir.get_op("pld.system.nranks").name,
    ir.get_op("pld.system.rank").name,
    ir.get_op("pld.system.notify").name,
    ir.get_op("pld.system.wait").name,
}


def _build_barrier_before():
    """Build a minimal Before program that calls ``pld.tensor.barrier``."""
    nr = _BARRIER_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def barrier_step(
            self,
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            signal = pld.tensor.barrier(signal)
            return signal

    return Before


def test_barrier_is_decomposed_to_primitives():
    """The composite ``pld.tensor.barrier`` Call is replaced by notify-all +
    wait-all; no occurrence survives the pass."""
    Before = _build_barrier_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.barrier" not in op_names, (
        "lower_composite_ops must remove the composite barrier call entirely"
    )
    missing = _BARRIER_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_barrier_emits_for_and_if_control_flow():
    """Barrier emits 2 ForStmts + 2 IfStmts: notify-all + wait-all."""
    Before = _build_barrier_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    assert collector.for_count == 3, (
        f"expected 3 ForStmts (notify, wait, epilogue), got {collector.for_count}"
    )
    assert collector.if_count == 3, f"expected 3 IfStmts (one per ForStmt body), got {collector.if_count}"


def test_barrier_lowering_is_idempotent():
    """Running the pass on already-lowered barrier IR is a no-op."""
    Before = _build_barrier_before()
    once = passes.lower_composite_ops()(Before)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


# ============================================================================
# pld.tensor.broadcast lowering
# ============================================================================

_BROADCAST_SIZE = 16
_BROADCAST_NRANKS = 2
_BROADCAST_REQUIRED_OPS = {
    ir.get_op("pld.system.get_comm_ctx").name,
    ir.get_op("pld.system.nranks").name,
    ir.get_op("pld.system.rank").name,
    ir.get_op("pld.system.notify").name,
    ir.get_op("pld.system.wait").name,
    ir.get_op("tile.create").name,
    ir.get_op("pld.tile.get").name,
}


def _build_broadcast_before():
    """Build a minimal Before program that calls ``pld.tensor.broadcast``."""
    SIZE = _BROADCAST_SIZE
    nr = _BROADCAST_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def broadcast_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            data = pld.tensor.broadcast(data, signal, root=0)
            return data

    return Before


def test_broadcast_is_decomposed_to_primitives():
    """The composite ``pld.tensor.broadcast`` Call is replaced by its
    3-phase decomposition; no occurrence survives the pass."""
    Before = _build_broadcast_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.broadcast" not in op_names, (
        "lower_composite_ops must remove the composite broadcast call entirely"
    )
    missing = _BROADCAST_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_broadcast_emits_for_and_if_control_flow():
    """Broadcast emits 2 ForStmts + 2 IfStmts: notify-all + wait-all.
    Phase 3 (tile.create + pld.tile.get) has no loop."""
    Before = _build_broadcast_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    assert collector.for_count == 3, (
        f"expected 3 ForStmts (notify, wait, epilogue), got {collector.for_count}"
    )
    assert collector.if_count == 3, f"expected 3 IfStmts (one per ForStmt body), got {collector.if_count}"


def test_broadcast_lowering_is_idempotent():
    """Running the pass on already-lowered broadcast IR is a no-op."""
    Before = _build_broadcast_before()
    once = passes.lower_composite_ops()(Before)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


# ============================================================================
# pld.tensor.allgather lowering
# ============================================================================

_ALLGATHER_SIZE = 16
_ALLGATHER_NRANKS = 2
_ALLGATHER_REQUIRED_OPS = {
    ir.get_op("pld.system.get_comm_ctx").name,
    ir.get_op("pld.system.nranks").name,
    ir.get_op("pld.system.rank").name,
    ir.get_op("pld.system.notify").name,
    ir.get_op("pld.system.wait").name,
    ir.get_op("pld.tile.put").name,
    ir.get_op("tile.create").name,
}


def _build_allgather_before():
    """Build a minimal Before program that calls ``pld.tensor.allgather``."""
    SIZE = _ALLGATHER_SIZE
    nr = _ALLGATHER_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            result = pld.tensor.allgather(inp, data, signal)
            return result

    return Before


def test_allgather_is_decomposed_to_primitives():
    """The composite ``pld.tensor.allgather`` Call is replaced by its
    decompose; no occurrence survives the pass."""
    Before = _build_allgather_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.allgather" not in op_names, (
        "lower_composite_ops must remove the composite allgather call entirely"
    )
    missing = _ALLGATHER_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_allgather_emits_for_and_if_control_flow():
    """Push-based allgather emits 3 ForStmts + 2 IfStmts: push loop, notify-all, wait-all.

    Phase 1 (push) uses a runtime ForStmt over nranks_idx — every peer gets a
    pld.tile.put (self-store via HCCL identity mapping, no per-rank IfStmt).
    Phase 2a/2b are the standard notify-all/wait-all loops with per-peer IfStmts."""
    Before = _build_allgather_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    assert collector.for_count == 4, (
        f"expected 4 ForStmts (push, notify, wait, epilogue), got {collector.for_count}"
    )
    assert collector.if_count == 3, (
        f"expected 3 IfStmts (notify-all + wait-all + epilogue), got {collector.if_count}"
    )


def test_allgather_lowering_is_idempotent():
    """Running the pass on already-lowered allgather IR is a no-op."""
    Before = _build_allgather_before()
    once = passes.lower_composite_ops()(Before)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


# ============================================================================
# pld.tensor.reduce_scatter lowering
# ============================================================================

_REDUCE_SCATTER_SIZE = 16
_REDUCE_SCATTER_NRANKS = 2
_REDUCE_SCATTER_REQUIRED_OPS = {
    ir.get_op("pld.system.get_comm_ctx").name,
    ir.get_op("pld.system.nranks").name,
    ir.get_op("pld.system.rank").name,
    ir.get_op("pld.system.notify").name,
    ir.get_op("pld.system.wait").name,
    ir.get_op("pld.tile.remote_load").name,
    ir.get_op("tile.add").name,
    ir.get_op("tile.load").name,
    ir.get_op("tile.store").name,
}


def _build_reduce_scatter_before():
    """Build a minimal Before program that calls ``pld.tensor.reduce_scatter``."""
    SIZE = _REDUCE_SCATTER_SIZE
    nr = _REDUCE_SCATTER_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return data

    return Before


def test_reduce_scatter_is_decomposed_to_primitives():
    """The composite ``pld.tensor.reduce_scatter`` Call is replaced by its
    5-phase decomposition; no occurrence survives the pass."""
    Before = _build_reduce_scatter_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.reduce_scatter" not in op_names, (
        "lower_composite_ops must remove the composite reduce_scatter call entirely"
    )
    missing = _REDUCE_SCATTER_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_reduce_scatter_emits_for_and_if_control_flow():
    """Reduce-scatter emits 5 ForStmts + 5 IfStmts (same shape as allreduce):
    notify, wait, reduce, re-notify, re-wait."""
    Before = _build_reduce_scatter_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    assert collector.for_count == 6, (
        f"expected 6 ForStmts (notify, wait, reduce, re-notify, re-wait, epilogue), got {collector.for_count}"
    )
    assert collector.if_count == 6, f"expected 6 IfStmts (one per ForStmt body), got {collector.if_count}"


def test_reduce_scatter_lowering_is_idempotent():
    """Running the pass on already-lowered reduce_scatter IR is a no-op."""
    Before = _build_reduce_scatter_before()
    once = passes.lower_composite_ops()(Before)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_reduce_scatter_deducer_rejects_unsupported_reduce_op():
    """First-version lowering supports ``ReduceOp.Sum`` only — the deducer
    must reject other variants."""
    SIZE = _REDUCE_SCATTER_SIZE
    nr = _REDUCE_SCATTER_NRANKS

    with pytest.raises((ValueError, TypeError, ParserError)):

        @pl.program
        class BadOp:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
            ) -> pl.Tensor[[nr, SIZE], pl.FP32]:
                data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Max)
                return data


# ============================================================================
# pld.tensor.allreduce ring mode lowering
#
# Ring allreduce decomposes ``pld.tensor.allreduce(data, signal, mode="ring")``
# into an NCCL-style chunked reduce-scatter + allgather schedule with 2(P−1)
# per-round barriers.  The signal shape is [2*(NR−1), NR] (one row per ring
# round, one cell per rank).  These tests pin the ring-specific invariants
# without hand-mirroring every temp name.
# ============================================================================

_RING_ALLREDUCE_SIZE = 16
_RING_ALLREDUCE_NRANKS = 2

# Ops the ring decomposition must emit.
_RING_ALLREDUCE_REQUIRED_OPS = {
    ir.get_op(name).name
    for name in (
        "pld.system.get_comm_ctx",
        "pld.system.nranks",
        "pld.system.rank",
        "pld.system.notify",  # per-round barrier (2(P−1) rounds)
        "pld.system.wait",  # per-round barrier
        "pld.tile.remote_load",  # per-ring-step chunk receive
        "tile.add",  # reduce-scatter accumulation
        "tile.load",  # reduce-scatter local accumulation
        "tile.fillpad_inplace",  # promote ragged subchunks for fixed-shape arithmetic
        "tile.set_validshape",  # narrow the tail again before storing
        "tile.store",  # reduce-scatter + allgather chunk writes
    )
}


def _build_ring_allreduce_before(
    size: int = _RING_ALLREDUCE_SIZE,
    n_ranks: int = _RING_ALLREDUCE_NRANKS,
    reduce_op: pld.ReduceOp = pld.ReduceOp.Sum,
    dtype=pl.FP32,
):
    """Build a minimal Before program that calls allreduce(mode="ring")."""
    SIZE = size
    nr = n_ranks
    total_rounds = 2 * (nr - 1)
    REDUCE_OP = reduce_op
    DTYPE = dtype

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, SIZE], DTYPE],
            out: pl.Out[pl.Tensor[[1, SIZE], DTYPE]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], DTYPE]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], DTYPE]:
            local = pl.load(inp, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)
            data = pld.tensor.allreduce(data, signal, op=REDUCE_OP, mode="ring")
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

    return Before


def test_ring_allreduce_is_decomposed_to_primitives():
    """The composite ring allreduce Call is replaced by the ring primitive
    tree; no ``pld.tensor.allreduce`` survives."""
    Before = _build_ring_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert ir.get_op("pld.tensor.allreduce").name not in op_names, (
        "lower_composite_ops must remove the composite allreduce call entirely"
    )
    assert ir.get_op("tile.create").name not in op_names, (
        "inactive ring segments must not leave allocation-only placeholders"
    )
    missing = _RING_ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"ring-lowered IR missing expected ops: {missing}"


def test_ring_allreduce_emits_ring_control_flow():
    """Ring lowering emits phase, subchunk, and two-phase barrier loops.

    For P=2 and one subchunk, each RS/AG phase has one outer loop, one
    subchunk loop, four peer loops, one guarded data path, and one guarded
    store."""
    Before = _build_ring_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    assert collector.for_count == 14, f"expected 14 ForStmts for P=2 ring, got {collector.for_count}"
    assert collector.if_count == 13, f"expected 13 IfStmts for P=2 ring, got {collector.if_count}"


@pytest.mark.parametrize("size", [1, 3, 17, 8193, 65537])
@pytest.mark.parametrize("n_ranks", [2, 4])
def test_ring_allreduce_accepts_arbitrary_lengths(size, n_ranks):
    """Non-divisible, shorter-than-rank, and larger-than-UB sizes lower safely."""
    Before = _build_ring_allreduce_before(size=size, n_ranks=n_ranks)
    After = passes.lower_composite_ops()(Before)
    ssa_before = passes.convert_to_ssa()(Before)
    ssa_lowered = passes.lower_composite_ops()(ssa_before)
    flattened = passes.flatten_tile_nd_to_2d()(ssa_lowered)
    assert "__FREE_VAR" not in ir.python_print(flattened)

    class CallCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            self.calls.append(op)
            super().visit_call(op)

    collector = CallCollector()
    collector.visit_program(After)
    remote_loads = [
        call for call in collector.calls if call.op.name == ir.get_op("pld.tile.remote_load").name
    ]
    loads = [call for call in collector.calls if call.op.name == ir.get_op("tile.load").name]
    set_valid_shapes = [
        call for call in collector.calls if call.op.name == ir.get_op("tile.set_validshape").name
    ]

    assert remote_loads
    assert loads
    assert set_valid_shapes
    assert all(len(call.args) == 5 for call in remote_loads)

    loops: list[ir.ForStmt] = []

    def collect_loops(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                collect_loops(child)
        elif isinstance(stmt, ir.ForStmt):
            loops.append(stmt)
            collect_loops(stmt.body)
        elif isinstance(stmt, ir.IfStmt):
            collect_loops(stmt.then_body)
            if stmt.else_body is not None:
                collect_loops(stmt.else_body)

    func = After.get_function("reduce_step")
    assert func is not None
    collect_loops(func.body)

    max_segment = (size + n_ranks - 1) // n_ranks
    expected_chunk = min(4096, ((max_segment + 7) // 8) * 8)
    subchunk_loops = [
        loop
        for loop in loops
        if ("rs_col" in loop.loop_var.name_hint or "ag_col" in loop.loop_var.name_hint)
        and isinstance(loop.step, ir.ConstInt)
        and loop.step.value == expected_chunk
    ]
    assert len(subchunk_loops) == 2
    for loop in subchunk_loops:
        assert isinstance(loop.start, ir.ConstInt) and loop.start.value == 0
        assert isinstance(loop.stop, ir.ConstInt) and loop.stop.value == max_segment

    chunk_shapes = [call.args[3] for call in remote_loads]
    for shape in chunk_shapes:
        assert isinstance(shape, ir.MakeTuple)
        chunk_rows = shape.elements[0]
        chunk_cols = shape.elements[1]
        assert isinstance(chunk_rows, ir.ConstInt)
        assert isinstance(chunk_cols, ir.ConstInt)
        chunk_bytes = chunk_rows.value * chunk_cols.value * 4
        assert chunk_bytes <= 16 * 1024
        assert chunk_bytes % 32 == 0


@pytest.mark.parametrize("size", [1, 17, 33, 8193, 65537])
@pytest.mark.parametrize("n_ranks", [2, 4])
def test_ring_allreduce_fp16_uses_aligned_ring_schedule(size, n_ranks):
    """FP16 stays on the ring path and marks every remote tail as padded."""
    Before = _build_ring_allreduce_before(
        size=size,
        n_ranks=n_ranks,
        dtype=pl.FP16,
    )
    After = passes.lower_composite_ops()(Before)

    class CallCollector(ir.IRVisitor):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[ir.Call] = []

        def visit_call(self, op: ir.Call) -> None:
            self.calls.append(op)
            super().visit_call(op)

    collector = CallCollector()
    collector.visit_program(After)
    remote_loads = [
        call for call in collector.calls if call.op.name == ir.get_op("pld.tile.remote_load").name
    ]
    assert remote_loads
    assert all(call.kwargs.get("allow_physical_tail_padding") is True for call in remote_loads)
    assert all(len(call.args) == 5 for call in remote_loads)

    stmt_collector = _StmtKindCollector()
    stmt_collector.visit_program(After)
    assert stmt_collector.for_count == 14, (
        f"FP16 mode=ring must use the ring schedule, got {stmt_collector.for_count} loops"
    )

    max_segment = min(size, (size + n_ranks - 1) // n_ranks + 15)
    expected_chunk = min(8192, ((max_segment + 15) // 16) * 16)
    chunk_shapes = [call.args[3] for call in remote_loads]
    for shape in chunk_shapes:
        assert isinstance(shape, ir.MakeTuple)
        chunk_cols = shape.elements[1]
        assert isinstance(chunk_cols, ir.ConstInt)
        assert chunk_cols.value == expected_chunk
        assert chunk_cols.value * 2 <= 16 * 1024
        assert chunk_cols.value % 16 == 0


def test_ring_allreduce_fp16_lowered_ir_round_trips():
    """The compiler-only aligned remote tail survives print and reparse."""
    Before = _build_ring_allreduce_before(size=17, n_ranks=2, dtype=pl.FP16)
    After = passes.lower_composite_ops()(Before)

    text = ir.python_print(After)
    assert "pld.tile._remote_load_with_physical_tail_padding(" in text
    assert "allow_physical_tail_padding=" not in text
    reparsed = pl.parse_program(text)
    ir.assert_structural_equal(After, reparsed)


def test_ring_allreduce_flattens_packed_nd_target():
    """Ring mode reinterprets packed ND storage as one linear stream."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[2, 3, 17], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 2], pl.INT32]],
        ) -> pld.DistributedTensor[[2, 3, 17], pl.FP32]:
            return pld.tensor.allreduce(data, signal, mode="ring")

    After = passes.lower_composite_ops()(Before)
    func = After.get_function("reduce_step")
    assert func is not None
    assert isinstance(func.body, ir.SeqStmts)

    view_stmt = next(
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tensor.view").name
    )
    assert isinstance(view_stmt.var.type, ir.DistributedTensorType)
    assert view_stmt.var.type.shape == [1, 102]


def test_ring_allreduce_preserves_contiguous_partial_valid_prefix(
    default_pass_manager,
    ascend_backend,
):
    """A row-major prefix stays valid through the default PTO pipeline."""
    from pypto import codegen  # noqa: PLC0415
    from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, 3, 4], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 2], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(valid_shape=[1, 3, 4], stride=[], layout=pl.TensorLayout.ND),
        ]:
            return pld.tensor.allreduce(data, signal, mode="ring")

    After = passes.lower_composite_ops()(Before)
    func = After.get_function("reduce_step")
    assert func is not None
    assert isinstance(func.body, ir.SeqStmts)
    assert ir.get_op("tensor.slice").name not in _collect_op_names(After)
    view_stmt = next(
        stmt
        for stmt in func.body.stmts
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tensor.view").name
    )
    view_type = view_stmt.var.type
    assert isinstance(view_type, ir.DistributedTensorType)
    assert view_type.shape == [1, 24]
    assert view_type.tensor_view is not None
    assert view_type.tensor_view.valid_shape == [1, 12]
    before_func = Before.get_function("reduce_step")
    assert before_func is not None
    before_type = before_func.params[0].type
    assert isinstance(before_type, ir.DistributedTensorType)
    assert view_type.window_buffer == before_type.window_buffer

    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = default_pass_manager.run_passes(Before)
    optimized_func = optimized.get_function("reduce_step")
    assert optimized_func is not None
    mlir = codegen.PTOCodegen().generate(ir.Program([optimized_func], optimized_func.name, optimized.span))
    assert "pto.tload" in mlir
    assert "pto.tstore" in mlir


def test_ring_allreduce_rejects_noncontiguous_partial_valid_box():
    """A rectangular valid box with row gaps is not one linear prefix."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(valid_shape=[1, 3, 2], stride=[], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 2], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(valid_shape=[1, 3, 2], stride=[], layout=pl.TensorLayout.ND),
        ]:
            return pld.tensor.allreduce(data, signal, mode="ring")

    with pytest.raises(ValueError, match="contiguous row-major prefix"):
        passes.lower_composite_ops()(Before)


def test_ring_allreduce_rejects_strided_target():
    """Linear ring addressing must not discard explicit row gaps."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(stride=[100, 10, 1], layout=pl.TensorLayout.ND),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 2], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(stride=[100, 10, 1], layout=pl.TensorLayout.ND),
        ]:
            return pld.tensor.allreduce(data, signal, mode="ring")

    with pytest.raises(ValueError, match="requires a packed source"):
        passes.lower_composite_ops()(Before)


def test_ring_allreduce_rejects_dn_target():
    """A DN view cannot be flattened with row-major ring offsets."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[
                pld.DistributedTensor[
                    [2, 3, 4],
                    pl.FP32,
                    pl.TensorView(stride=[12, 1, 3], layout=pl.TensorLayout.DN),
                ]
            ],
            signal: pl.InOut[pld.DistributedTensor[[2, 2], pl.INT32]],
        ) -> pld.DistributedTensor[
            [2, 3, 4],
            pl.FP32,
            pl.TensorView(stride=[12, 1, 3], layout=pl.TensorLayout.DN),
        ]:
            return pld.tensor.allreduce(data, signal, mode="ring")

    with pytest.raises(ValueError, match="only supports ND layout"):
        passes.lower_composite_ops()(Before)


def test_ring_allreduce_dynamic_nd_reaches_pto_codegen(default_pass_manager, ascend_backend):
    """The default pipeline binds a dynamic packed ND ring extent."""
    from pypto import codegen  # noqa: PLC0415
    from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

    n = pl.dynamic("RING_ALLREDUCE_DYNAMIC_N")

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[2, 3, n], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[2, 2], pl.INT32]],
        ) -> pld.DistributedTensor[[2, 3, n], pl.FP32]:
            return pld.tensor.allreduce(data, signal, mode="ring")

    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = default_pass_manager.run_passes(Before)
    func = optimized.get_function("reduce_step")
    assert func is not None
    mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))

    assert "scf.for" in mlir
    assert "pto.tload" in mlir
    assert "pto.tstore" in mlir
    assert "pto.partition_view" in mlir


def test_ring_allreduce_fp16_ragged_tail_reaches_pto_codegen(
    default_pass_manager,
    ascend_backend,
):
    """The default pipeline preserves an aligned physical FP16 ring tail."""
    from pypto import codegen  # noqa: PLC0415
    from pypto.pypto_core import passes as _core_passes  # noqa: PLC0415

    Before = _build_ring_allreduce_before(size=17, n_ranks=2, dtype=pl.FP16)
    ctx = _core_passes.PassContext(
        [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
    )
    with ctx:
        optimized = default_pass_manager.run_passes(Before)
    func = optimized.get_function("reduce_step")
    assert func is not None
    mlir = codegen.PTOCodegen().generate(ir.Program([func], func.name, optimized.span))

    assert "scf.for" in mlir
    assert "pto.tload" in mlir
    assert "pto.tstore" in mlir
    assert "f16" in mlir


def test_ring_allreduce_lowering_is_idempotent():
    """Running the pass on already-lowered ring IR is a no-op."""
    Before = _build_ring_allreduce_before()
    once = passes.lower_composite_ops()(Before)
    twice = passes.lower_composite_ops()(once)
    ir.assert_structural_equal(twice, once)


def test_ring_allreduce_invalid_signal_shape_is_rejected():
    """Ring mode validates signal type — rejects non-DistributedTensor or
    non-INT32 signals at lowering time.  The exact shape [2*(NR−1), NR]
    is checked for dimensionality (must be 2D) but exact dimension values
    are validated at runtime when NR is dynamic."""
    SIZE = _RING_ALLREDUCE_SIZE
    nr = _RING_ALLREDUCE_NRANKS

    # Wrong dtype: signal must be INT32 for notify/wait counters.
    with pytest.raises((ValueError, TypeError, ParserError)):

        @pl.program
        class BadDtype:
            @pl.function(type=pl.FunctionType.InCore)
            def reduce_step(
                self,
                data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[2 * (nr - 1), nr], pl.FP32]],
            ) -> pl.Tensor[[1, SIZE], pl.FP32]:
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
                return data

        passes.lower_composite_ops()(BadDtype)


def test_ring_allreduce_mesh_default_unchanged():
    """Existing mesh allreduce (mode omitted) still decomposes to the mesh
    recipe — no ring primitives leak in."""
    Before = _build_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert ir.get_op("pld.tensor.allreduce").name not in op_names
    missing = _ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"mesh-lowered IR missing expected ops: {missing}"

    # Mesh-specific: ready barrier + chunk traversal + peer reduction +
    # per-chunk read-complete barrier.
    collector = _StmtKindCollector()
    collector.visit_program(After)
    assert collector.for_count == 7, f"mesh allreduce must produce 7 ForStmts, got {collector.for_count}"


@pytest.mark.parametrize(("reduce_op", "expected_tile_op"), _ALLREDUCE_REDUCE_CASES)
def test_ring_allreduce_lowers_every_reduce_op(reduce_op, expected_tile_op):
    Before = _build_ring_allreduce_before(reduce_op=reduce_op)
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert ir.get_op("pld.tensor.allreduce").name not in op_names
    assert expected_tile_op in op_names
    for _, other_tile_op in _ALLREDUCE_REDUCE_CASES:
        if other_tile_op != expected_tile_op:
            assert other_tile_op not in op_names


# ============================================================================
# Self-clearing credit-barrier protocol (issue #2156)
#
# Every ``pld.tensor.*`` collective barriers through one shared protocol:
# ``AtomicAdd(1)`` into each peer's cell for every barrier this call issues,
# then ``Wait(>= g)`` where ``g`` counts up *within this call only* — every
# fresh call restarts at 1. A per-call epilogue then subtracts the total
# credit count back out of every cell it touched (``AtomicAdd(-N)``), so the
# signal is provably all-zero again once every rank has finished its
# epilogue. The next call on the same signal therefore starts over at
# generation 1 too, with no cross-call bookkeeping required.
#
# These tests pin the protocol itself, which is the deterministic regression
# guard: functional back-to-back coverage lives in the distributed STs, where a
# stale-signal barrier only *races* rather than reliably failing.
# ============================================================================

_PROTOCOL_SIZE = 16
_PROTOCOL_NRANKS = 2


# Route the operator literals through the registry getter so a typo raises at
# import instead of silently never matching.
_NOTIFY_OP_NAME = ir.get_op("pld.system.notify").name
_WAIT_OP_NAME = ir.get_op("pld.system.wait").name


class _CollectiveCallCollector(ir.IRVisitor):
    """Collect every ``pld.system.notify`` / ``pld.system.wait`` Call in order."""

    def __init__(self) -> None:
        super().__init__()
        self.notifies: list[ir.Call] = []
        self.waits: list[ir.Call] = []

    def visit_call(self, op: ir.Call) -> None:
        if op.op.name == _NOTIFY_OP_NAME:
            self.notifies.append(op)
        elif op.op.name == _WAIT_OP_NAME:
            self.waits.append(op)
        super().visit_call(op)


def _collect_barrier_calls(prog) -> _CollectiveCallCollector:
    collector = _CollectiveCallCollector()
    collector.visit_program(prog)
    return collector


def _static_wait_expectations(prog) -> list[int]:
    """Expected values of every ``pld.system.wait`` with a constant expectation.

    The mesh allreduce's per-chunk barrier derives its expectation from the
    chunk loop variable, so it is skipped here — only compile-time constants
    (one per straight-line barrier) are returned, in emission order.
    """
    values = []
    for wait in _collect_barrier_calls(prog).waits:
        expected = wait.args[2]
        if isinstance(expected, ir.ConstInt):
            values.append(expected.value)
    return values


def _build_two_barriers_program():
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def barrier_twice(
            self,
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            signal = pld.tensor.barrier(signal)
            signal = pld.tensor.barrier(signal)
            return signal

    return Before


def _build_two_all_to_all_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def exchange_twice(
            self,
            first: pl.Tensor[[nr, SIZE], pl.FP32],
            second: pl.Tensor[[nr, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            data = pld.tensor.all_to_all(first, data, signal)
            data = pld.tensor.all_to_all(second, data, signal)
            return data

    return Before


def _build_two_allgather_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def gather_twice(
            self,
            first: pl.Tensor[[1, SIZE], pl.FP32],
            second: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            data = pld.tensor.allgather(first, data, signal)
            data = pld.tensor.allgather(second, data, signal)
            return data

    return Before


def _build_two_reduce_scatter_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def scatter_twice(
            self,
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return data

    return Before


def _build_two_broadcast_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def broadcast_twice(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            data = pld.tensor.broadcast(data, signal, root=0)
            data = pld.tensor.broadcast(data, signal, root=0)
            return data

    return Before


def _build_two_ring_allreduce_program():
    SIZE = _RING_ALLREDUCE_SIZE
    nr = _RING_ALLREDUCE_NRANKS
    total_rounds = 2 * (nr - 1)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_twice(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            return data

    return Before


# One builder per collective, with the wait expectations each back-to-back pair
# must produce. Every call restarts at generation 1 — the epilogue resets the
# signal to all-zero at the end of each call — so two calls repeat the same
# sequence rather than accumulating. reduce_scatter barriers twice per call
# (ready + post-reduce): [1, 2, 1, 2]. The ring allreduce's arbitrary-length
# schedule (#2161) barriers twice per subchunk (ready + read-complete); at
# NR=2 there is one reduce-scatter round and one allgather round, each with
# one subchunk, so one call already produces [1, 2, 1, 2] and two calls repeat
# it.
_BACK_TO_BACK_CASES = {
    "barrier": (_build_two_barriers_program, [1, 1]),
    "all_to_all": (_build_two_all_to_all_program, [1, 1]),
    "allgather": (_build_two_allgather_program, [1, 1]),
    "broadcast": (_build_two_broadcast_program, [1, 1]),
    "reduce_scatter": (_build_two_reduce_scatter_program, [1, 2, 1, 2]),
    "ring_allreduce": (_build_two_ring_allreduce_program, []),
}


@pytest.mark.parametrize("collective", sorted(_BACK_TO_BACK_CASES))
def test_back_to_back_collective_restarts_at_generation_one(collective):
    """A signal reused by two consecutive collectives always restarts at 1.

    Regression guard for issue #2156: the self-clearing epilogue
    (``AtomicAdd(-N)``) restores every cell to zero at the end of each call,
    so the *next* call's first barrier safely waits for ``>= 1`` again —
    unlike the old ``Set(1)`` / ``Wait(>= 1)`` protocol, which left every cell
    at 1 and let the second call's waits pass on stale state.
    """
    build, expected_generations = _BACK_TO_BACK_CASES[collective]
    After = passes.lower_composite_ops()(build())

    assert _static_wait_expectations(After) == expected_generations, (
        f"{collective} back-to-back must restart at generation 1 each call, expecting "
        f"{expected_generations}, got {_static_wait_expectations(After)}"
    )


@pytest.mark.parametrize("collective", sorted(_BACK_TO_BACK_CASES))
def test_collective_barriers_use_atomic_add_and_ge(collective):
    """Every barrier notifies with ``AtomicAdd`` and waits with ``Ge``.

    ``Set`` must never appear: mixing it with ``AtomicAdd`` on the same cells
    can clobber an already-advanced counter. ``Ge`` (not ``Eq``) is required
    because a fast peer can advance a cell past the value the waiter is
    looking for. The credit epilogue's reset notify is itself an ``AtomicAdd``
    (of a negative value) — never a ``Set`` — so this invariant covers it too.
    """
    build, _ = _BACK_TO_BACK_CASES[collective]
    After = passes.lower_composite_ops()(build())
    calls = _collect_barrier_calls(After)

    assert calls.notifies, f"{collective} lowering emitted no notify"
    assert calls.waits, f"{collective} lowering emitted no wait"
    for notify in calls.notifies:
        assert notify.kwargs["op"] == int(pld.NotifyOp.AtomicAdd), (
            f"{collective} must notify with AtomicAdd, got op={notify.kwargs['op']}"
        )
    for wait in calls.waits:
        assert wait.kwargs["cmp"] == int(pld.WaitCmp.Ge), (
            f"{collective} must wait with Ge, got cmp={wait.kwargs['cmp']}"
        )


def test_barrier_chain_restarts_generation_through_the_rebind_idiom():
    """``sig = pld.tensor.barrier(sig)`` renames the signal on every call.

    Generations are call-local under the self-clearing protocol, so a chain
    of rebound barriers restarts at 1 on every call instead of continuing to
    count up — the rebind idiom no longer needs cross-call alias tracking.
    """
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def barrier_thrice(
            self,
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            signal = pld.tensor.barrier(signal)
            signal = pld.tensor.barrier(signal)
            signal = pld.tensor.barrier(signal)
            return signal

    After = passes.lower_composite_ops()(Before)
    assert _static_wait_expectations(After) == [1, 1, 1]


def test_mesh_allreduce_epilogue_resets_signal_for_the_next_collective():
    """A later collective is unaffected by an earlier allreduce's barriers.

    The [1, 16] FP32 target is one chunk, so the mesh recipe issues one ready
    barrier (generation 1) plus one chunk-complete barrier (a runtime-derived
    expected value, not a compile-time constant) and then subtracts both
    credits back out via the epilogue. The following barrier therefore waits
    for 1 again, not for a value that accounts for the allreduce's barriers.
    """
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_then_barrier(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            signal = pld.tensor.barrier(signal)
            return signal

    After = passes.lower_composite_ops()(Before)
    # The ready barrier waits for 1; the chunk barrier's expectation is derived
    # from the chunk loop variable (not a constant, so _static_wait_expectations
    # skips it); the epilogue resets the signal, so the trailing barrier waits
    # for 1 again rather than accounting for the allreduce's barriers.
    assert _static_wait_expectations(After) == [1, 1]


def test_dynamic_mesh_allreduce_signal_is_reusable():
    """A symbolic reduction extent no longer poisons the signal for reuse.

    The self-clearing epilogue's credit total may itself be a runtime-computed
    scalar expression (``pld.system.notify``'s value only requires
    ``ScalarType``), so a mesh allreduce over a symbolic extent can still reset
    its signal correctly even though the chunk count is unknown at compile
    time.
    """
    n = pl.dynamic("REUSE_DYNAMIC_N")
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_then_barrier(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, n], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
            signal = pld.tensor.barrier(signal)
            return signal

    After = passes.lower_composite_ops()(Before)
    assert _static_wait_expectations(After) == [1, 1]

    # The allreduce's epilogue reset must carry a symbolic (non-ConstInt)
    # credit total, since the chunk count depends on the runtime extent n.
    notifies = _collect_barrier_calls(After).notifies
    symbolic_value_notifies = [call for call in notifies if not isinstance(call.args[3], ir.ConstInt)]
    assert symbolic_value_notifies, (
        "expected at least one symbolic-value notify (the allreduce epilogue reset)"
    )


def test_mixing_ring_and_mesh_barrier_protocols_on_one_signal_is_rejected():
    """Ring's signal is [2*(NR-1), NR] (one row per round); mesh's is [NR, 1]
    (one cell per rank). Sharing one buffer between the two is now a plain
    shape mismatch rather than a generation-table protocol conflict."""
    SIZE = _RING_ALLREDUCE_SIZE
    nr = _RING_ALLREDUCE_NRANKS
    total_rounds = 2 * (nr - 1)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_then_barrier(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pld.DistributedTensor[[total_rounds, nr], pl.INT32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            signal = pld.tensor.barrier(signal)
            return signal

    with pytest.raises(ValueError, match=r"signal shape\[1\] must be 1"):
        passes.lower_composite_ops()(Before)


def _build_barrier_in_loop_program():
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_step(
            self,
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            for _ in pl.range(2):
                signal = pld.tensor.barrier(signal)
            return signal

    return Before


def _build_broadcast_in_loop_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            for _ in pl.range(2):
                data = pld.tensor.broadcast(data, signal, root=0)
            return data

    return Before


def _build_allgather_in_loop_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            for _ in pl.range(2):
                data = pld.tensor.allgather(inp, data, signal)
            return data

    return Before


def _build_all_to_all_in_loop_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_step(
            self,
            inp: pl.Tensor[[nr, SIZE], pl.FP32],
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            for _ in pl.range(2):
                data = pld.tensor.all_to_all(inp, data, signal)
            return data

    return Before


def _build_reduce_scatter_in_loop_program():
    SIZE = _PROTOCOL_SIZE
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_step(
            self,
            data: pl.InOut[pld.DistributedTensor[[nr, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, SIZE], pl.FP32]:
            for _ in pl.range(2):
                data = pld.tensor.reduce_scatter(data, signal, op=pld.ReduceOp.Sum)
            return data

    return Before


_IN_LOOP_BUILDERS = {
    "barrier": _build_barrier_in_loop_program,
    "broadcast": _build_broadcast_in_loop_program,
    "allgather": _build_allgather_in_loop_program,
    "all_to_all": _build_all_to_all_in_loop_program,
    "reduce_scatter": _build_reduce_scatter_in_loop_program,
}


_IN_LOOP_EXPECTED = {
    "barrier": [1],
    "broadcast": [1],
    "allgather": [1],
    "all_to_all": [1],
    "reduce_scatter": [1, 2],
}


@pytest.mark.parametrize("collective", sorted(_IN_LOOP_BUILDERS))
def test_collective_in_for_loop_now_succeeds(collective):
    """Collectives are legal inside for/while/if now.

    Each call is a self-contained, stateless cycle starting from all-zero, so
    the compiler lowers the loop body's collective call once and the same
    compile-time expected values are safely reused on every runtime iteration
    — unlike the old compile-time generation scheme, which had no fixed
    generation for a dynamic trip count to wait for.
    """
    Before = _IN_LOOP_BUILDERS[collective]()
    After = passes.lower_composite_ops()(Before)
    assert _static_wait_expectations(After) == _IN_LOOP_EXPECTED[collective]


def test_unrolled_loop_collective_restarts_each_call():
    """``pl.unroll`` still produces straight-line back-to-back calls.

    ``UnrollLoops`` runs before ``LowerCompositeOps`` in the default pipeline,
    so a compile-time trip count becomes 3 independent calls on one signal —
    each restarting at generation 1, since the self-clearing epilogue resets
    the signal between them.
    """
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def barrier_unrolled(
            self,
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            for _ in pl.unroll(3):
                signal = pld.tensor.barrier(signal)
            return signal

    After = passes.lower_composite_ops()(passes.unroll_loops()(Before))
    assert _static_wait_expectations(After) == [1, 1, 1]


def test_stateless_signal_round_trip_across_many_calls():
    """A dynamic (non-unrolled) loop issuing many barrier calls on one signal
    compiles to exactly one barrier + epilogue reset, reused at runtime for
    every iteration — proof that the protocol needs no per-iteration or
    per-call compile-time state.
    """
    nr = _PROTOCOL_NRANKS

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def loop_step(
            self,
            signal: pl.InOut[pld.DistributedTensor[[nr, 1], pl.INT32]],
        ) -> pld.DistributedTensor[[nr, 1], pl.INT32]:
            for _ in pl.range(5):
                signal = pld.tensor.barrier(signal)
            return signal

    After = passes.lower_composite_ops()(Before)
    calls = _collect_barrier_calls(After)
    assert len(calls.waits) == 1, (
        "the loop body's barrier must be lowered exactly once, regardless of trip count"
    )
    expected = calls.waits[0].args[2]
    assert isinstance(expected, ir.ConstInt)
    assert expected.value == 1


def test_epilogue_emits_negative_atomic_add():
    """Every collective's final notify is its self-clearing epilogue reset:

    an ``AtomicAdd`` of a negated credit total — either ``Neg(wrapped)`` for
    runtime expressions, or ``ConstInt(negative)`` for compile-time constants
    (the lowering helper folds ``Neg(ConstInt(N))`` to ``ConstInt(-N)`` so
    the PyPTO printer→parser roundtrip stays stable).
    """
    for collective, (build, _) in sorted(_BACK_TO_BACK_CASES.items()):
        After = passes.lower_composite_ops()(build())
        notifies = _collect_barrier_calls(After).notifies
        assert notifies, f"{collective} emitted no notify"
        last_notify = notifies[-1]
        assert last_notify.kwargs["op"] == int(pld.NotifyOp.AtomicAdd), (
            f"{collective} epilogue must notify with AtomicAdd"
        )
        value = last_notify.args[3]
        is_negative = isinstance(value, ir.Neg) or (isinstance(value, ir.ConstInt) and value.value < 0)
        assert is_negative, f"{collective} epilogue must carry a negative value, got {type(value)}"


def test_ring_allreduce_epilogue_resets_every_row():
    """Ring's epilogue must reset every one of the signal's rows, not just a
    single cell — each round credited a distinct row.
    """
    SIZE = _RING_ALLREDUCE_SIZE
    nr = _RING_ALLREDUCE_NRANKS
    total_rounds = 2 * (nr - 1)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_once(
            self,
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pld.DistributedTensor[[1, SIZE], pl.FP32]:
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            return data

    After = passes.lower_composite_ops()(Before)
    notifies = _collect_barrier_calls(After).notifies
    # The epilogue resets are the AtomicAdd notifies whose value is negative
    # (either Neg(wrapped) for runtime expressions, or ConstInt(negative) when
    # the lowering helper folds Neg(ConstInt(N)) to ConstInt(-N) for roundtrip
    # stability). Every earlier round-barrier notify uses a plain +1 ConstInt.
    epilogue_notifies = [
        call
        for call in notifies
        if isinstance(call.args[3], ir.Neg)
        or (isinstance(call.args[3], ir.ConstInt) and call.args[3].value < 0)
    ]
    assert epilogue_notifies, "expected at least one epilogue reset notify"
    # Every epilogue notify targets a 2-element [row, rank] offset tuple (the
    # 2D signal overload), confirming the reset is row-indexed rather than a
    # single [rank, 0] cell.
    for notify in epilogue_notifies:
        offsets = notify.args[2]
        assert isinstance(offsets, ir.MakeTuple)
        assert len(offsets.elements) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
