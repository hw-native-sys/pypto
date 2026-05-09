# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Flash Attention with loop-level iter_args.

Demonstrates advanced DSL control flow:
  - pl.range with init_values for loop-carried state (iter_args)
  - Nested if/else with pl.yield_ for SSA phi nodes
  - Online softmax with running max/sum accumulators
  - Multi-output yield for updating loop state

Algorithm:
  For each KV block j in [0, 16):
    sij = Q @ Kj^T (scaled)
    pij = softmax(sij) with online max/sum correction
    Oi = update(Oi, pij @ Vj)
  Return Oi / li (final normalization)

Run:  python examples/models/03_flash_attention.py
Next: examples/models/04_paged_attention.py
"""

import pypto.language as pl


@pl.jit
def flash_attention(q_13: pl.Tensor, k_16: pl.Tensor, v_19: pl.Tensor):
    with pl.incore():
        attn_initial = pl.create_tensor([64, 128], dtype=pl.FP32)
        oi_update_initial = pl.create_tensor([64, 128], dtype=pl.FP32)
        li_update_initial = pl.create_tensor([64, 1], dtype=pl.FP32)
        mi_update_initial = pl.create_tensor([64, 1], dtype=pl.FP32)

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
            kj = pl.slice(k_16, [64, 128], [i * 64, 0])
            vj = pl.slice(v_19, [64, 128], [i * 64, 0])
            sij = pl.matmul(q_13, kj, out_dtype=pl.FP16, a_trans=False, b_trans=True, c_matrix_nz=False)
            sij_1 = pl.mul(sij, 0.0883883)
            row_max = pl.row_max(sij_1)
            sub = pl.sub(sij_1, row_max)
            p_ij = pl.exp(sub)
            l_ij = pl.row_sum(p_ij)
            tildaPij_83 = pl.cast(p_ij, target_type=pl.FP16, mode="round")

            # Nested if with yield (SSA phi node)
            if i == 0:
                # Inner statement.block
                oiUpdate_87 = pl.matmul(tildaPij_83, vj, out_dtype=pl.FP16)
                oiUpdate_90 = pl.assemble(oi_update, oiUpdate_87, offset=[0, 0])

                # Nested if inside first branch
                if i == 15:
                    attn_94 = pl.div(oiUpdate_90, l_ij)
                    attn_95 = pl.yield_(attn_94)
                else:
                    attn_95 = pl.yield_(attn_update)

                # More statements in first branch
                liUpdate_98 = pl.assemble(li_update, l_ij, offset=[0, 0])
                miUpdate_101 = pl.assemble(mi_update, row_max, offset=[0, 0])

                # statement.yield → pl.yield_ with assignment
                miUpdate_126, liUpdate_127, attn_128, oiUpdate_129 = pl.yield_(
                    miUpdate_101, liUpdate_98, attn_95, oiUpdate_90
                )
            else:
                # Else branch
                mi_102 = pl.create_tensor(shape=[64, 1], dtype=pl.FP32)
                miUpdate_103 = pl.maximum(mi_102, row_max)
                t1_104 = pl.sub(mi_102, miUpdate_103)
                t2_105 = pl.exp(t1_104)
                t3_106 = pl.sub(row_max, miUpdate_103)
                t4_107 = pl.exp(t3_106)
                t5_108 = pl.mul(t4_107, l_ij)
                t6_109 = pl.mul(t2_105, li_update)
                liUpdate_110 = pl.add(t6_109, t5_108)
                liUpdate_113 = pl.assemble(li_update, liUpdate_110, offset=[0, 0])
                q3_114 = pl.mul(oi_update, t2_105)
                q1_115 = pl.matmul(
                    tildaPij_83, vj, out_dtype=pl.FP16, a_trans=False, b_trans=False, c_matrix_nz=False
                )
                q2_116 = pl.mul(q1_115, t4_107)
                oiUpdate_117 = pl.add(q3_114, q2_116)
                oiUpdate_120 = pl.assemble(oi_update, oiUpdate_117, offset=[0, 0])

                # Nested if in else branch
                if i == 15:
                    attn_124 = pl.div(oiUpdate_120, liUpdate_113)
                    attn_125 = pl.yield_(attn_124)
                else:
                    attn_125 = pl.yield_(attn_update)

                miUpdate_126, liUpdate_127, attn_128, oiUpdate_129 = pl.yield_(
                    miUpdate_103, liUpdate_113, attn_125, oiUpdate_120
                )

            # For loop yield (updates iter_args for next iteration)
            mi_final, li_final, attn_final, oi_final = pl.yield_(
                miUpdate_126, liUpdate_127, attn_128, oiUpdate_129
            )
    return attn_final


if __name__ == "__main__":
    # The body currently fails IR verification at pipeline_input due to a
    # pre-existing IfStmt yield/return_vars structural mismatch in the original
    # @pl.function example (which only ever called print() and never went
    # through the pass pipeline).  See KNOWN_ISSUES.md for the tracking entry.
    # Until that is fixed, this entry only verifies that the JIT decorator
    # wraps and the Python parser accepts the source -- it does NOT execute.
    print(flash_attention)
    print("SKIPPED: flash_attention body fails IR verification (see KNOWN_ISSUES.md)")
