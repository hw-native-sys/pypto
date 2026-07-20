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

# Primitive tile ops the decomposition is allowed to emit (besides framework
# infrastructure ops like tile.load / tile.store / tile.move that wrap the
# decomposed body).
_DECOMP_PRIMITIVES = {
    "tile.muls",
    "tile.adds",
    "tile.add",
    "tile.sub",
    "tile.mul",
    "tile.cast",
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

# Ops the chunked mesh decomposition must emit.
_ALLREDUCE_REQUIRED_OPS = {
    "pld.system.get_comm_ctx",
    "pld.system.nranks",
    "pld.system.rank",
    "pld.system.notify",  # Ready and per-chunk read-complete barriers
    "pld.system.wait",  # Ready and per-chunk read-complete barriers
    "pld.tile.remote_load",  # Peer chunk load
    "tile.fillpad_inplace",  # Zero ragged padding before accumulation
    "tile.add",  # Accumulate peer chunks
    "tile.load",  # Self chunk load + user-side load
    "tile.set_validshape",  # Narrow the final ragged chunk
    "tile.store",  # Per-chunk result + user-side store
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


def _build_allreduce_before(size: int = _ALLREDUCE_SIZE):
    """Build a minimal Before program that calls ``pld.tensor.allreduce``."""
    SIZE = size
    nr = _ALLREDUCE_NRANKS

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
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum)
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

    with pytest.raises(Exception, match="requires an explicit signal outside host orchestrator"):
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

    with pytest.raises(Exception, match="requires an explicit signal outside host orchestrator"):
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

    assert "pld.tensor.allreduce" not in op_names
    missing = _ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"lowered IR missing expected ops: {missing}"


def test_allreduce_in_for_loop_is_rejected():
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

    with pytest.raises(Exception, match="allreduce is not supported inside a for/while loop"):
        passes.lower_composite_ops()(LoopAllreduce)


def test_allreduce_in_while_loop_is_rejected():
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

    with pytest.raises(Exception, match="allreduce is not supported inside a for/while loop"):
        passes.lower_composite_ops()(LoopAllreduce)


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

    assert collector.for_count == 6, (
        f"expected 6 ForStmts (notify, wait, chunk, reduce, re-notify, re-wait), got {collector.for_count}"
    )
    assert collector.if_count == 5, f"expected 5 peer-filter IfStmts, got {collector.if_count}"


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
        and stmt.value.op.name == "tensor.view"
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
        and stmt.value.op.name == "tensor.view"
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
    load = next(call for call in collector.calls if call.op.name == "tile.load")
    remote_load = next(call for call in collector.calls if call.op.name == "pld.tile.remote_load")
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
    expected_chunk = min(_ALLREDUCE_FP32_CHUNK, ((size + 7) // 8) * 8)
    assert expected_chunk * 4 % 32 == 0

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
        and call.args[2].elements[0].value == 1
        and isinstance(call.args[2].elements[1], ir.ConstInt)
        and call.args[2].elements[1].value == expected_chunk
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
    assert isinstance(chunk_loop.step, ir.ConstInt) and chunk_loop.step.value == expected_chunk
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

    with pytest.raises(Exception, match="partial valid_shape must fit within one 16384-byte mesh chunk"):
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
    with passes.PassContext([]):
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

    with pytest.raises(Exception, match="valid_shape cannot be represented by a single 2D view"):
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

    with pytest.raises(Exception, match="requires a packed source"):
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

    with pytest.raises(Exception, match="only supports ND layout"):
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
        and stmt.value.op.name == "tensor.view"
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


def test_allreduce_deducer_rejects_unsupported_reduce_op():
    """First-version lowering supports ``ReduceOp.Sum`` only — the deducer
    must reject other variants so users get a clear error rather than
    silently wrong codegen."""
    SIZE = _ALLREDUCE_SIZE

    with pytest.raises((ValueError, TypeError, ParserError)):

        @pl.program
        class BadOp:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[2, 1], pl.INT32]],
            ) -> pl.Tensor[[1, SIZE], pl.FP32]:
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Max)
                return data


# ============================================================================
# pld.tensor.barrier lowering
# ============================================================================

_BARRIER_NRANKS = 2
_BARRIER_REQUIRED_OPS = {
    "pld.system.get_comm_ctx",
    "pld.system.nranks",
    "pld.system.rank",
    "pld.system.notify",
    "pld.system.wait",
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

    assert collector.for_count == 2, f"expected 2 ForStmts (notify, wait), got {collector.for_count}"
    assert collector.if_count == 2, f"expected 2 IfStmts (one per ForStmt body), got {collector.if_count}"


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
    "pld.system.get_comm_ctx",
    "pld.system.nranks",
    "pld.system.rank",
    "pld.system.notify",
    "pld.system.wait",
    "tile.create",
    "pld.tile.get",
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

    assert collector.for_count == 2, f"expected 2 ForStmts (notify, wait), got {collector.for_count}"
    assert collector.if_count == 2, f"expected 2 IfStmts (one per ForStmt body), got {collector.if_count}"


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
    "pld.system.get_comm_ctx",
    "pld.system.nranks",
    "pld.system.rank",
    "pld.system.notify",
    "pld.system.wait",
    "pld.tile.put",
    "tile.create",
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

    assert collector.for_count == 3, f"expected 3 ForStmts (push, notify, wait), got {collector.for_count}"
    assert collector.if_count == 2, f"expected 2 IfStmts (notify-all + wait-all), got {collector.if_count}"


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
    "pld.system.get_comm_ctx",
    "pld.system.nranks",
    "pld.system.rank",
    "pld.system.notify",
    "pld.system.wait",
    "pld.tile.remote_load",
    "tile.add",
    "tile.load",
    "tile.store",
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

    assert collector.for_count == 5, (
        f"expected 5 ForStmts (notify, wait, reduce, re-notify, re-wait), got {collector.for_count}"
    )
    assert collector.if_count == 5, f"expected 5 IfStmts (one per ForStmt body), got {collector.if_count}"


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
    "pld.system.get_comm_ctx",
    "pld.system.nranks",
    "pld.system.rank",
    "pld.system.notify",  # per-round barrier (2(P−1) rounds)
    "pld.system.wait",  # per-round barrier
    "pld.tile.remote_load",  # per-ring-step chunk receive
    "tile.add",  # reduce-scatter accumulation
    "tile.load",  # reduce-scatter local accumulation
    "tile.store",  # reduce-scatter + allgather chunk writes
}


def _build_ring_allreduce_before():
    """Build a minimal Before program that calls allreduce(mode="ring")."""
    SIZE = _RING_ALLREDUCE_SIZE
    nr = _RING_ALLREDUCE_NRANKS
    total_rounds = 2 * (nr - 1)

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def reduce_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
            signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            local = pl.load(inp, [0, 0], [1, SIZE])
            data = pl.store(local, [0, 0], data)
            data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Sum, mode="ring")
            acc = pl.load(data, [0, 0], [1, SIZE])
            return pl.store(acc, [0, 0], out)

    return Before


def test_ring_allreduce_is_decomposed_to_primitives():
    """The composite ring allreduce Call is replaced by the ring primitive
    tree; no ``pld.tensor.allreduce`` survives."""
    Before = _build_ring_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    op_names = set(_collect_op_names(After))

    assert "pld.tensor.allreduce" not in op_names, (
        "lower_composite_ops must remove the composite allreduce call entirely"
    )
    missing = _RING_ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"ring-lowered IR missing expected ops: {missing}"


def test_ring_allreduce_emits_ring_control_flow():
    """Ring lowering emits two phase loops plus notify/wait peer loops.

    For P=2 each inner notify/wait peer loop has one direct IfStmt body."""
    Before = _build_ring_allreduce_before()
    After = passes.lower_composite_ops()(Before)
    collector = _StmtKindCollector()
    collector.visit_program(After)

    # Each phase contains notify and wait peer loops: 2 + 2*2 = 6 loops.
    assert collector.for_count == 6, (
        f"expected 6 ForStmts for P=2 ring (2 phase + 4 peer loops), got {collector.for_count}"
    )
    assert collector.if_count == 4, (
        f"expected 4 IfStmts (RS notify+wait + AG notify+wait), got {collector.if_count}"
    )


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

    assert "pld.tensor.allreduce" not in op_names
    missing = _ALLREDUCE_REQUIRED_OPS - op_names
    assert not missing, f"mesh-lowered IR missing expected ops: {missing}"

    # Mesh-specific: ready barrier + chunk traversal + peer reduction +
    # per-chunk read-complete barrier.
    collector = _StmtKindCollector()
    collector.visit_program(After)
    assert collector.for_count == 6, f"mesh allreduce must produce 6 ForStmts, got {collector.for_count}"


def test_ring_allreduce_deducer_rejects_unsupported_reduce_op():
    """Ring mode inherits the kSum-only restriction from mesh."""
    SIZE = _RING_ALLREDUCE_SIZE
    nr = _RING_ALLREDUCE_NRANKS
    total_rounds = 2 * (nr - 1)

    with pytest.raises((ValueError, TypeError, ParserError)):

        @pl.program
        class BadRingOp:
            @pl.function(type=pl.FunctionType.InCore)
            def f(
                self,
                data: pl.InOut[pld.DistributedTensor[[1, SIZE], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[total_rounds, nr], pl.INT32]],
            ) -> pl.Tensor[[1, SIZE], pl.FP32]:
                data = pld.tensor.allreduce(data, signal, op=pld.ReduceOp.Max, mode="ring")
                return data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
