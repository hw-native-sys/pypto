# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


from pypto import DataType, ir
from pypto.ir import IRBuilder


def build_flash_attention():
    """Build flash attention IR using IRBuilder."""
    ib = IRBuilder()
    span = ir.Span.unknown()

    with ib.function("flash_attn") as f:
        # Create function parameters
        q = f.param("q", ir.TensorType([64, 128], DataType.FP16))
        k = f.param("k", ir.TensorType([1024, 128], DataType.FP16))
        v = f.param("v", ir.TensorType([1024, 128], DataType.FP16))
        f.return_type(ir.TensorType([64, 128], DataType.FP32))

        # Create initial tensors
        attn_init = ib.let("attn_init", ir.op.tensor.create([64, 128], DataType.FP32))
        oi_init = ib.let("oi_init", ir.op.tensor.create([64, 128], DataType.FP32))
        li_init = ib.let("li_init", ir.op.tensor.create([64, 1], DataType.FP32))
        mi_init = ib.let("mi_init", ir.op.tensor.create([64, 1], DataType.FP32))

        # Create loop variable
        i = ib.var("i", ir.ScalarType(DataType.INT64))
        # For loop with iteration arguments
        with ib.for_loop(i, 0, 16, 1) as loop:
            # Add iteration arguments (loop-carried values)
            mi_update = loop.iter_arg("mi_update", mi_init)
            li_update = loop.iter_arg("li_update", li_init)
            attn_update = loop.iter_arg("attn_update", attn_init)
            oi_update = loop.iter_arg("oi_update", oi_init)

            # Add return variables to capture final values
            loop.return_var("mi_final")
            loop.return_var("li_final")
            loop.return_var("attn_final")
            loop.return_var("oi_final")

            # Loop body - common operations
            kj = ib.let("kj", ir.op.tensor.view(k, [64, 128], [i * 64, 0]))
            vj = ib.let("vj", ir.op.tensor.view(v, [64, 128], [i * 64, 0]))

            # sij = matmul(q, kj^T)
            sij = ib.let("sij", ir.op.tensor.matmul(q, kj, out_dtype=DataType.FP16, b_trans=True))

            # sij_1 = sij * scale_factor
            scale = ir.ConstFloat(0.0883883, DataType.FP16, span)
            sij_1 = ib.let("sij_1", ir.op.tensor.mul(sij, scale))

            # row_max = rowmax(sij_1, axis=-1, keep_dim=1)
            row_max = ib.let("row_max", ir.op.tensor.row_max(sij_1, axis=-1, keep_dim=1))
            sub = ib.let("sub", ir.op.tensor.sub(sij_1, row_max))
            p_ij = ib.let("p_ij", ir.op.tensor.exp(sub))
            l_ij = ib.let("l_ij", ir.op.tensor.row_sum(p_ij, axis=-1, keep_dim=1))
            tildaPij_83 = ib.let("tildaPij_83", ir.op.tensor.cast(p_ij, DataType.FP16))

            with ib.if_stmt(i == 0) as if_builder:
                oiUpdate_87 = ib.let(
                    "oiUpdate_87",
                    ir.op.tensor.matmul(tildaPij_83, vj, out_dtype=DataType.FP16),
                )

                # oiUpdate_90 = assemble(oi_update, oiUpdate_87, offset=[0, 0])
                oiUpdate_90 = ib.let("oiUpdate_90", ir.op.tensor.assemble(oi_update, oiUpdate_87, [0, 0]))

                with ib.if_stmt(i == 15) as inner_if_1:
                    inner_if_1.return_var("attn_95", ir.TensorType([64, 128], DataType.FP32))
                    attn_94 = ib.let("attn_94", ir.op.tensor.div(oiUpdate_90, l_ij))
                    ib.emit(ir.YieldStmt([attn_94], span))

                    inner_if_1.else_()
                    ib.emit(ir.YieldStmt([attn_update], span))

                attn_95 = inner_if_1.output()
                liUpdate_98 = ib.let("liUpdate_98", ir.op.tensor.assemble(li_update, l_ij, [0, 0]))
                miUpdate_101 = ib.let("miUpdate_101", ir.op.tensor.assemble(mi_update, row_max, [0, 0]))
                yield_stmt = ir.YieldStmt([miUpdate_101, liUpdate_98, attn_95, oiUpdate_90], span)
                ib.emit(yield_stmt)

                # Else branch: i != 0
                if_builder.else_()
                mi_102 = ib.let("mi_102", ir.op.tensor.create([64, 1], DataType.FP32))
                miUpdate_103 = ib.let("miUpdate_103", ir.op.tensor.maximum(mi_102, row_max))
                t1_104 = ib.let("t1_104", ir.op.tensor.sub(mi_102, miUpdate_103))
                t2_105 = ib.let("t2_105", ir.op.tensor.exp(t1_104))
                t3_106 = ib.let("t3_106", ir.op.tensor.sub(row_max, miUpdate_103))
                t4_107 = ib.let("t4_107", ir.op.tensor.exp(t3_106))
                t5_108 = ib.let("t5_108", ir.op.tensor.mul(t4_107, l_ij))
                t6_109 = ib.let("t6_109", ir.op.tensor.mul(t2_105, li_update))
                liUpdate_110 = ib.let("liUpdate_110", ir.op.tensor.add(t6_109, t5_108))
                liUpdate_113 = ib.let(
                    "liUpdate_113",
                    ir.op.tensor.assemble(li_update, liUpdate_110, [0, 0]),
                )
                q3_114 = ib.let("q3_114", ir.op.tensor.mul(oi_update, t2_105))
                q1_115 = ib.let(
                    "q1_115",
                    ir.op.tensor.matmul(tildaPij_83, vj, out_dtype=DataType.FP16),
                )
                q2_116 = ib.let("q2_116", ir.op.tensor.mul(q1_115, t4_107))
                oiUpdate_117 = ib.let("oiUpdate_117", ir.op.tensor.add(q3_114, q2_116))
                oiUpdate_120 = ib.let(
                    "oiUpdate_120",
                    ir.op.tensor.assemble(oi_update, oiUpdate_117, [0, 0]),
                )

                # Nested if in else branch: if i == 15
                with ib.if_stmt(i == 15) as inner_if_2:
                    inner_if_2.return_var("attn_125", ir.TensorType([64, 128], DataType.FP32))

                    attn_124 = ib.let("attn_124", ir.op.tensor.div(oiUpdate_120, liUpdate_113))
                    yield_stmt = ir.YieldStmt([attn_124], span)
                    ib.emit(yield_stmt)

                    inner_if_2.else_()
                    yield_stmt = ir.YieldStmt([attn_update], span)
                    ib.emit(yield_stmt)

                (attn_125,) = inner_if_2.outputs()

                yield_stmt = ir.YieldStmt([miUpdate_103, liUpdate_113, attn_125, oiUpdate_120], span)
                ib.emit(yield_stmt)

                if_builder.return_var("miUpdate_126", ir.TensorType([64, 1], DataType.FP32))
                if_builder.return_var("liUpdate_127", ir.TensorType([64, 1], DataType.FP32))
                if_builder.return_var("attn_128", ir.TensorType([64, 128], DataType.FP32))
                if_builder.return_var("oiUpdate_129", ir.TensorType([64, 128], DataType.FP32))

            yield_stmt = ir.YieldStmt([mi_update, li_update, attn_update, oi_update], span)
            ib.emit(yield_stmt)

        # Return the final attention output
        ib.return_stmt(loop.get_result().return_vars)

    return f.get_result()


if __name__ == "__main__":
    # Build the flash attention IR
    func = build_flash_attention()

    print("Successfully built flash attention IR!")
    print(f"Function name: {func.name}")
    print(f"Number of parameters: {len(func.params)}")
    print(f"Return types: {len(func.return_types)}")

    # Optionally serialize it
    print(func)
    data = ir.serialize(func)
    restored = ir.deserialize(data)
    restored_data = ir.serialize(restored)
    restored_restored = ir.deserialize(restored_data)
    ir.assert_structural_equal(func, restored_restored)
