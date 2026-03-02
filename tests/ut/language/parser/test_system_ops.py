# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for system operation DSL parsing and round-trip."""

import pypto
import pypto.language as pl
import pytest
from pypto import ir


class TestSystemOpsParsing:
    """Tests for parsing pl.system.* operations in the DSL."""

    def test_sync_src_round_trip(self):
        """Test round-trip for pl.system.sync_src."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.system.sync_src(" in printed
        assert "set_pipe=pl.PipeType.MTE2" in printed
        assert "wait_pipe=pl.PipeType.V" in printed
        assert "event_id=0" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_sync_dst_round_trip(self):
        """Test round-trip for pl.system.sync_dst."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.system.sync_dst(" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_bar_v_round_trip(self):
        """Test round-trip for pl.system.bar_v."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.bar_v()
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.system.bar_v()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_bar_m_round_trip(self):
        """Test round-trip for pl.system.bar_m."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.bar_m()
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.system.bar_m()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_bar_all_round_trip(self):
        """Test round-trip for pl.system.bar_all."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.bar_all()
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.system.bar_all()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_multiple_system_ops_round_trip(self):
        """Test round-trip with multiple system ops in a single function."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.bar_v()
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.bar_all()
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.system.sync_src(" in printed
        assert "pl.system.bar_v()" in printed
        assert "pl.system.sync_dst(" in printed
        assert "pl.system.bar_all()" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)

    def test_sync_with_different_pipe_types(self):
        """Test sync ops with various PipeType enum values."""

        @pl.program
        class Before:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.S, event_id=2)
                return x

        printed = pypto.ir.python_print(Before)
        assert "pl.PipeType.MTE1" in printed
        assert "pl.PipeType.M" in printed
        assert "pl.PipeType.MTE3" in printed
        assert "pl.PipeType.S" in printed

        reparsed = pl.parse_program(printed)
        assert isinstance(reparsed, ir.Program)
        ir.assert_structural_equal(Before, reparsed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
