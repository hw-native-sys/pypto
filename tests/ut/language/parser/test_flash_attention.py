# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration test for flash_attention.py parsing."""

import pypto
import pypto.language as pl
import pytest
from pypto.pypto_core import ir


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


class TestFlashAttention:
    """Integration tests using flash_attention.py as example."""

    def test_flash_attention_parses(self):
        """Test that flash_attention.py parses successfully."""
        # Import the flash_attention module which has @pl.function decorator
        assert isinstance(flash_attn, ir.Function)
        assert flash_attn.name == "flash_attn"

    def test_flash_attention_has_correct_params(self):
        """Test flash_attention has correct parameters."""

        assert len(flash_attn.params) == 3
        # Should have q, k, v parameters

    def test_flash_attention_has_return_type(self):
        """Test flash_attention has return type."""

        assert len(flash_attn.return_types) == 1

    def test_flash_attention_serializes(self):
        """Test flash_attention can be serialized."""

        # Should be able to serialize
        data = pypto.ir.serialize(flash_attn)
        assert len(data) > 0

        # Should be able to deserialize
        restored = pypto.ir.deserialize(data)
        assert isinstance(restored, ir.Function)
        assert restored.name == "flash_attn"
        assert len(restored.params) == 3

    def test_flash_attention_round_trip(self):
        """Test flash_attention serialization round-trip."""

        # Serialize, deserialize, serialize again
        data1 = pypto.ir.serialize(flash_attn)
        restored = pypto.ir.deserialize(data1)
        data2 = pypto.ir.serialize(restored)

        # Second restoration
        restored2 = pypto.ir.deserialize(data2)

        assert isinstance(restored2, ir.Function)

        # Should have same structure
        assert restored2.name == flash_attn.name
        assert len(restored2.params) == len(flash_attn.params)

    def test_flash_attention_with_multiple_iter_args(self):
        """Test flash attention pattern with multiple iteration arguments."""

        @pl.function
        def multi_iter_attn(
            q: pl.Tensor[[64, 128], pl.FP16],
        ) -> pl.Tensor[[64, 128], pl.FP32]:
            attn: pl.Tensor[[64, 128], pl.FP32] = pl.create_tensor([64, 128], dtype=pl.FP32)
            scale: pl.Tensor[[64, 1], pl.FP32] = pl.create_tensor([64, 1], dtype=pl.FP32)

            for i, (attn_val, scale_val) in pl.range(4, init_values=(attn, scale)):  # type: ignore
                # Update both
                new_attn: pl.Tensor[[64, 128], pl.FP32] = pl.mul(attn_val, 1.1)
                new_scale: pl.Tensor[[64, 1], pl.FP32] = pl.add(scale_val, 0.1)

                attn_out, scale_out = pl.yield_(new_attn, new_scale)

            return attn_out

        assert isinstance(multi_iter_attn, ir.Function)

    def test_flash_attention_ops(self):
        """Test individual operations used in flash attention."""

        @pl.function
        def test_ops(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP32]:
            # Test operations used in flash attention

            # Cast
            fp32: pl.Tensor[[64, 128], pl.FP32] = pl.cast(x, target_type=pl.FP32, mode="round")

            # Mul
            scaled: pl.Tensor[[64, 128], pl.FP32] = pl.mul(fp32, 0.5)

            # Row max
            row_max_val: pl.Tensor[[64, 1], pl.FP32] = pl.row_max(scaled)

            # Sub
            normalized: pl.Tensor[[64, 128], pl.FP32] = pl.sub(scaled, row_max_val)

            # Exp
            exped: pl.Tensor[[64, 128], pl.FP32] = pl.exp(normalized)

            # Row sum
            row_sum_val: pl.Tensor[[64, 1], pl.FP32] = pl.row_sum(exped)

            # Div
            result: pl.Tensor[[64, 128], pl.FP32] = pl.div(exped, row_sum_val)

            return result

        assert isinstance(test_ops, ir.Function)


class TestParserRobustness:
    """Tests for parser robustness and edge cases."""

    def test_large_tensor_shapes(self):
        """Test parsing with large tensor shapes."""

        @pl.function
        def large_shapes(
            x: pl.Tensor[[1024, 2048, 512], pl.FP32],
        ) -> pl.Tensor[[1024, 2048, 512], pl.FP32]:
            return x

        assert isinstance(large_shapes, ir.Function)

    def test_many_parameters(self):
        """Test function with many parameters."""

        @pl.function
        def many_params(
            a: pl.Tensor[[64], pl.FP32],
            b: pl.Tensor[[64], pl.FP32],
            c: pl.Tensor[[64], pl.FP32],
            d: pl.Tensor[[64], pl.FP32],
            e: pl.Tensor[[64], pl.FP32],
        ) -> pl.Tensor[[64], pl.FP32]:
            temp1: pl.Tensor[[64], pl.FP32] = pl.add(a, b)
            temp2: pl.Tensor[[64], pl.FP32] = pl.add(c, d)
            temp3: pl.Tensor[[64], pl.FP32] = pl.add(temp1, temp2)
            result: pl.Tensor[[64], pl.FP32] = pl.add(temp3, e)
            return result

        assert len(many_params.params) == 5

    def test_deep_nesting(self):
        """Test deeply nested control flow."""

        @pl.function
        def deep_nesting(n: pl.Tensor[[1], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
            init: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)

            for i, (v1,) in pl.range(2, init_values=(init,)):
                for j, (v2,) in pl.range(2, init_values=(v1,)):
                    if i == 0:
                        if j == 0:
                            inner: pl.Tensor[[64], pl.FP32] = pl.mul(v2, 2.0)
                            ir: pl.Tensor[[64], pl.FP32] = pl.yield_(inner)
                        else:
                            ir: pl.Tensor[[64], pl.FP32] = pl.yield_(v2)
                        or_: pl.Tensor[[64], pl.FP32] = pl.yield_(ir)
                    else:
                        or_: pl.Tensor[[64], pl.FP32] = pl.yield_(v2)

                    jr = pl.yield_(or_)

                ir_out = pl.yield_(jr)

            return ir_out

        assert isinstance(deep_nesting, ir.Function)

    def test_long_function(self):
        """Test function with many statements."""

        @pl.function
        def long_function(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            v1: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
            v2: pl.Tensor[[64], pl.FP32] = pl.add(v1, 1.0)
            v3: pl.Tensor[[64], pl.FP32] = pl.add(v2, 1.0)
            v4: pl.Tensor[[64], pl.FP32] = pl.add(v3, 1.0)
            v5: pl.Tensor[[64], pl.FP32] = pl.add(v4, 1.0)
            v6: pl.Tensor[[64], pl.FP32] = pl.add(v5, 1.0)
            v7: pl.Tensor[[64], pl.FP32] = pl.add(v6, 1.0)
            v8: pl.Tensor[[64], pl.FP32] = pl.add(v7, 1.0)
            v9: pl.Tensor[[64], pl.FP32] = pl.add(v8, 1.0)
            v10: pl.Tensor[[64], pl.FP32] = pl.add(v9, 1.0)
            return v10

        assert isinstance(long_function, ir.Function)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
