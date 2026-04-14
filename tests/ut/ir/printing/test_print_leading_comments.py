# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test that python_print emits leading_comments as `# ...` lines above stmts."""

import pypto.language as pl
import pytest
from pypto import ir


class TestPrintLeadingComments:
    def test_assign_leading_comment_printed(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # annotate
                y = x
                return y

        out = ir.python_print(P)
        assert "# annotate" in out
        # Comment should appear above the assignment line
        assign_idx = out.index("y: pl.Scalar[pl.FP32] = x")
        comment_idx = out.index("# annotate")
        assert comment_idx < assign_idx

    def test_for_loop_header_comment_printed(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # main loop
                for i in pl.range(16):  # tiles
                    x = x + 1.0
                return x

        out = ir.python_print(P)
        # Both comments appear above the for header
        assert "# main loop" in out
        assert "# tiles" in out
        for_idx = out.index("for i in pl.range(16)")
        assert out.index("# main loop") < for_idx
        assert out.index("# tiles") < for_idx

    def test_if_else_comments_printed(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                # cond
                if x > 0.0:  # positive
                    y = x
                # fallback
                else:
                    y = -x
                return y

        out = ir.python_print(P)
        assert "# cond" in out
        assert "# positive" in out
        assert "# fallback" in out
        # `# fallback` appears below the then body's closing
        fallback_idx = out.index("# fallback")
        else_idx = out.index("else:")
        assert fallback_idx > else_idx

    def test_docstring_prints_as_comment(self):
        @pl.program
        class P:
            @pl.function
            def main(self, x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
                """preserved docstring"""
                y = x
                return y

        out = ir.python_print(P)
        assert "# preserved docstring" in out
        # Printed `# preserved docstring` should not also appear as a raw docstring
        assert '"""preserved docstring"""' not in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
