# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for print-parse roundtrip verification helpers."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.ir import OptimizationStrategy, PassManager
from pypto.pypto_core import passes


class TestVerifyRoundtrip:
    """Tests for ir.verify_roundtrip()."""

    def test_simple_program_passes(self):
        """A well-formed program should pass roundtrip verification."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        ir.verify_roundtrip(Prog)

    def test_program_with_multiple_functions_passes(self):
        """Programs with multiple functions should roundtrip."""

        @pl.program
        class Prog:
            @pl.function
            def add(
                self, x: pl.Tensor[[64], pl.FP32], y: pl.Tensor[[64], pl.FP32]
            ) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return result

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                one: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                result: pl.Tensor[[64], pl.FP32] = self.add(one, x)
                return result

        ir.verify_roundtrip(Prog)

    def test_ctrl_flow_transformed_program_passes(self):
        """Lowered while-style IR should still pass roundtrip verification."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore, strict_ssa=True)
            def main(self, x_0: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i, (x_iter,) in pl.range(0, 10, 1, init_values=(x_0,)):
                    if i > 5:
                        break
                    y: pl.Tensor[[64], pl.FP32] = pl.add(x_iter, x_iter)
                    x_iter = pl.yield_(y)
                return x_iter

        transformed = passes.ctrl_flow_transform()(Prog)
        ir.verify_roundtrip(transformed)


class TestRoundtripInstrument:
    """Tests for ir.RoundtripInstrument()."""

    def test_instrument_name(self):
        """RoundtripInstrument should expose a stable instrument name."""
        instrument = ir.RoundtripInstrument()
        assert isinstance(instrument, passes.CallbackInstrument)
        assert instrument.get_name() == "RoundtripInstrument"

    def test_instrument_runs_in_pass_context(self):
        """RoundtripInstrument should run successfully in a pass context."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.add(x, 1.0)
                return result

        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        with passes.PassContext([ir.RoundtripInstrument(passes.VerificationMode.BEFORE_AND_AFTER)]):
            result = pm.run_passes(Prog)

        assert isinstance(result, ir.Program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
