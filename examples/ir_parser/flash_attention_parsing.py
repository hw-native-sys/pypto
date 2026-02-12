# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl


@pl.function
def flash_attn(
    q_13: pl.Tensor[[64, 128], pl.FP16],
    k_16: pl.Tensor[[1024, 128], pl.FP16],
    v_19: pl.Tensor[[1024, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP32]:
    attn_initial: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
    oi_update_initial: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
    li_update_initial: pl.Tensor[[64, 1], pl.FP32] = pl.create_tensor([64, 1], dtype=pl.FP32)
    mi_update_initial: pl.Tensor[[64, 1], pl.FP32] = pl.create_tensor([64, 1], dtype=pl.FP32)

    # statement.for with iter_args → pl.range with tuple unpacking
    for i, (mi_update, li_update, attn_update, oi_update) in pl.range(
        16,
        init_values=(
            mi_update_initial,
            li_update_initial,
            attn_initial,
            oi_update_initial,
        ),
    ):
        # Inner statement.block
        kj: pl.Tensor[[64, 128], pl.FP16] = pl.view(k_16, [64, 128], [i * 64, 0])
        vj: pl.Tensor[[64, 128], pl.FP16] = pl.view(v_19, [64, 128], [i * 64, 0])
        sij: pl.Tensor[[64, 128], pl.FP16] = pl.matmul(
            q_13, kj, out_dtype=pl.FP16, a_trans=False, b_trans=True, c_matrix_nz=False
        )
        sij_1: pl.Tensor[[64, 128], pl.FP16] = pl.mul(sij, 0.0883883)
        row_max: pl.Tensor[[64, 1], pl.FP16] = pl.row_max(sij_1)
        sub: pl.Tensor[[64, 128], pl.FP16] = pl.sub(sij_1, row_max)
        p_ij: pl.Tensor[[64, 128], pl.FP16] = pl.exp(sub)
        l_ij: pl.Tensor[[64, 1], pl.FP16] = pl.row_sum(p_ij)
        tildaPij_83: pl.Tensor[[64, 128], pl.FP16] = pl.cast(p_ij, target_type=pl.FP16, mode="round")

        # Nested if with yield (SSA phi node)
        if i == 0:
            # Inner statement.block
            oiUpdate_87: pl.Tensor[[64, 128], pl.FP16] = pl.matmul(tildaPij_83, vj, out_dtype=pl.FP16)
            oiUpdate_90: pl.Tensor[[64, 128], pl.FP32] = pl.assemble(oi_update, oiUpdate_87, offset=[0, 0])

            # Nested if inside first branch
            if i == 15:
                attn_94: pl.Tensor[[64, 128], pl.FP32] = pl.div(oiUpdate_90, l_ij)
                attn_95: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(attn_94)
            else:
                attn_95: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(attn_update)

            # More statements in first branch
            liUpdate_98: pl.Tensor[[64, 1], pl.FP32] = pl.assemble(li_update, l_ij, offset=[0, 0])
            miUpdate_101: pl.Tensor[[64, 1], pl.FP32] = pl.assemble(mi_update, row_max, offset=[0, 0])

            # statement.yield → pl.yield_ with assignment
            miUpdate_126, liUpdate_127, attn_128, oiUpdate_129 = pl.yield_(
                miUpdate_101, liUpdate_98, attn_95, oiUpdate_90
            )
        else:
            # Else branch
            mi_102: pl.Tensor[[64, 1], pl.FP32] = pl.create_tensor(shape=[64, 1], dtype=pl.FP32)
            miUpdate_103: pl.Tensor[[64, 1], pl.FP32] = pl.maximum(mi_102, row_max)
            t1_104: pl.Tensor[[64, 1], pl.FP32] = pl.sub(mi_102, miUpdate_103)
            t2_105: pl.Tensor[[64, 1], pl.FP32] = pl.exp(t1_104)
            t3_106: pl.Tensor[[64, 1], pl.FP16] = pl.sub(row_max, miUpdate_103)
            t4_107: pl.Tensor[[64, 1], pl.FP16] = pl.exp(t3_106)
            t5_108: pl.Tensor[[64, 1], pl.FP16] = pl.mul(t4_107, l_ij)
            t6_109: pl.Tensor[[64, 1], pl.FP32] = pl.mul(t2_105, li_update)
            liUpdate_110: pl.Tensor[pl.FP32, 64, 1] = pl.add(t6_109, t5_108)
            liUpdate_113: pl.Tensor[[64, 1], pl.FP32] = pl.assemble(li_update, liUpdate_110, offset=[0, 0])
            q3_114: pl.Tensor[[64, 128], pl.FP32] = pl.mul(oi_update, t2_105)
            q1_115: pl.Tensor[[64, 128], pl.FP16] = pl.matmul(
                tildaPij_83,
                vj,
                out_dtype=pl.FP16,
                a_trans=False,
                b_trans=False,
                c_matrix_nz=False,
            )
            q2_116: pl.Tensor[[64, 128], pl.FP16] = pl.mul(q1_115, t4_107)
            oiUpdate_117: pl.Tensor[[64, 128], pl.FP32] = pl.add(q3_114, q2_116)
            oiUpdate_120: pl.Tensor[[64, 128], pl.FP32] = pl.assemble(oi_update, oiUpdate_117, offset=[0, 0])

            # Nested if in else branch
            if i == 15:
                attn_124: pl.Tensor[[64, 128], pl.FP32] = pl.div(oiUpdate_120, liUpdate_113)
                attn_125: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(attn_124)
            else:
                attn_125: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(attn_update)

            miUpdate_126, liUpdate_127, attn_128, oiUpdate_129 = pl.yield_(
                miUpdate_103, liUpdate_113, attn_125, oiUpdate_120
            )

        # For loop yield (updates iter_args for next iteration)
        mi_final, li_final, attn_final, oi_final = pl.yield_(
            miUpdate_126, liUpdate_127, attn_128, oiUpdate_129
        )
    return attn_final


print(flash_attn)
