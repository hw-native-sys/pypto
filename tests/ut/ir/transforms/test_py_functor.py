# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for Python IRVisitor and IRMutator."""

import pypto.language as pl
import pytest
from pypto import ir, passes


class TestIRVisitor:
    """Tests for the Python IRVisitor base class."""

    def test_count_for_stmts(self):
        """Visitor subclass counts ForStmt nodes."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 64):
                    x = pl.add(x, 1.0)
                return x

        class Counter(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.count = 0

            def visit_for_stmt(self, op):
                self.count += 1
                super().visit_for_stmt(op)

        counter = Counter()
        counter.visit_program(Prog)
        assert counter.count == 1

    def test_collect_call_op_names(self):
        """Visitor can collect all Call op names."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y = pl.add(x, 1.0)
                z = pl.mul(y, 2.0)
                return z

        class CallCollector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.op_names: list[str] = []

            def visit_call(self, op):
                self.op_names.append(op.op.name)
                super().visit_call(op)

        collector = CallCollector()
        collector.visit_program(Prog)
        assert "tensor.adds" in collector.op_names
        assert "tensor.muls" in collector.op_names

    def test_super_delegation_recurses(self):
        """super().visit_for_stmt() correctly recurses into children."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 64):
                    x = pl.add(x, 1.0)
                return x

        class ForAndCallCounter(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.for_count = 0
                self.call_count = 0

            def visit_for_stmt(self, op):
                self.for_count += 1
                super().visit_for_stmt(op)  # Must recurse to find calls inside

            def visit_call(self, op):
                self.call_count += 1
                super().visit_call(op)

        counter = ForAndCallCounter()
        counter.visit_program(Prog)
        assert counter.for_count == 1
        assert counter.call_count == 1  # add call inside loop body

    def test_visit_binary_expr_grouped(self):
        """visit_binary_expr handles all binary expression types."""
        span = ir.Span("test", 0, 0)
        a = ir.ConstInt(1, ir.DataType.INT32, span)
        b = ir.ConstInt(2, ir.DataType.INT32, span)
        add = ir.Add(a, b, ir.DataType.INT32, span)

        class BinaryCounter(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.binary_count = 0

            def visit_binary_expr(self, op):
                self.binary_count += 1
                super().visit_binary_expr(op)

        counter = BinaryCounter()
        counter.visit_expr(add)
        assert counter.binary_count == 1

    def test_default_visitor_no_crash(self):
        """Default IRVisitor traverses without crashing."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 64):
                    x = pl.add(x, 1.0)
                return x

        visitor = ir.IRVisitor()
        visitor.visit_program(Prog)  # Should not raise


class TestIRMutator:
    """Tests for the Python IRMutator base class."""

    def test_identity_returns_same_object(self):
        """Default mutator returns the exact same program (identity transform)."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y = pl.add(x, 1.0)
                return y

        mutator = ir.IRMutator()
        result = mutator.visit_program(Prog)
        assert result is Prog

    def test_default_mutator_no_crash(self):
        """Default IRMutator traverses complex IR without crashing."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                for i in pl.range(0, 64):
                    x = pl.add(x, 1.0)
                return x

        mutator = ir.IRMutator()
        result = mutator.visit_program(Prog)
        assert result is Prog
        ir.assert_structural_equal(result, Prog)


class TestPipelineIntegration:
    """Tests for integrating Python visitors/mutators with the C++ pass pipeline."""

    def test_python_pass_in_pipeline(self):
        """Python mutator works as a C++ pass via create_program_pass."""

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y = pl.add(x, 1.0)
                return y

        # Identity mutator wrapped as a pass
        my_pass = passes.create_program_pass(
            lambda prog: ir.IRMutator().visit_program(prog),
            "IdentityPythonPass",
        )
        pipeline = passes.PassPipeline()
        pipeline.add_pass(my_pass)
        result = pipeline.run(Prog)
        assert result is not None
        ir.assert_structural_equal(result, Prog)

    def test_analysis_pass_via_callback(self):
        """Python visitor can be used as analysis via create_program_pass."""

        call_count_holder = [0]

        def analysis_transform(prog):
            class Counter(ir.IRVisitor):
                def __init__(self):
                    super().__init__()
                    self.count = 0

                def visit_call(self, op):
                    self.count += 1
                    super().visit_call(op)

            counter = Counter()
            counter.visit_program(prog)
            call_count_holder[0] = counter.count
            return prog  # analysis only, return unchanged

        @pl.program
        class Prog:
            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                y = pl.add(x, 1.0)
                z = pl.mul(y, 2.0)
                return z

        my_pass = passes.create_program_pass(analysis_transform, "AnalysisPass")
        my_pass(Prog)
        assert call_count_holder[0] >= 2  # at least add and mul


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
