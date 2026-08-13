# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``out = <expr>`` on an Out/InOut parameter writes the whole tensor (#2352).

Rebinding the name alone used to re-point the parameter Var at a freshly
computed tensor, leaving the caller's buffer untouched — the program compiled,
ran, and handed back uninitialised memory with no diagnostic anywhere. The
parser now threads the value through a whole-tensor ``assemble``, so the natural
spelling builds the same IR as the explicit ``pl.assemble`` call.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import passes

_OP_TENSOR_ASSEMBLE = ir.get_op("tensor.assemble").name
_OP_TILE_ASSEMBLE = ir.get_op("tile.assemble").name


def _get_func(program: ir.Program, name: str) -> ir.Function:
    """Look up a function by name, failing the test when it is missing."""
    func = program.get_function(name)
    assert func is not None, f"program has no function '{name}'"
    return func


def _count_calls(program: ir.Program, func_name: str, op_name: str) -> int:
    """Count calls to ``op_name`` in a printed function body."""
    return _get_func(program, func_name).as_python().count(op_name)


class TestOrchestrationOutParam:
    """Bare assignment to an Out param in an orchestration body."""

    def test_bare_assignment_matches_explicit_assemble(self):
        """``out = pl.add(a, b)`` builds the same IR as the explicit form."""

        @pl.program
        class Bare:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, b)
                return out

        @pl.program
        class Explicit:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.add(a, b), [0, 0])
                return out

        ir.assert_structural_equal(Bare, Explicit)

    def test_bare_assignment_matches_subscript_write(self):
        """``out = expr`` and ``out[:, :] = expr`` desugar identically."""

        @pl.program
        class Bare:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, b)
                return out

        @pl.program
        class Subscript:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out[0:128, 0:128] = pl.add(a, b)
                return out

        ir.assert_structural_equal(Bare, Subscript)

    def test_inout_param_also_desugars(self):
        """An InOut parameter gets the same whole-tensor write."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                acc: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    acc = pl.add(a, b)
                return acc

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 1

    def test_in_param_is_not_rewritten(self):
        """A plain In parameter keeps today's rebinding semantics."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    a = pl.add(a, b)
                return a

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 0

    def test_local_variable_is_not_rewritten(self):
        """A non-parameter local keeps today's rebinding semantics."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tmp = pl.add(a, b)
                    tmp = pl.mul(tmp, b)
                    out = tmp
                return out

        # Only the final write into ``out`` is wrapped; the two ``tmp`` rebinds
        # are untouched.
        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 1


class TestNoDoubleWrapping:
    """Values already threaded through the parameter are left alone."""

    def test_explicit_assemble_is_not_rewrapped(self):
        """``out = pl.assemble(out, ...)`` stays a single assemble."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.add(a, b), [0, 0])
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 1

    def test_partial_assemble_is_not_rewrapped(self):
        """A real partial write keeps its offset and is not wrapped again."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, a, [64, 0])
                return out

        printed = _get_func(Prog, "main").as_python()
        assert printed.count(_OP_TENSOR_ASSEMBLE) == 1
        assert "[64, 0]" in printed

    def test_subscript_write_is_not_rewrapped(self):
        """The subscript sugar's own assemble is not wrapped a second time."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out[0:64, 0:128] = a
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 1

    def test_value_reading_the_param_is_left_alone(self):
        """A value that merely *reads* the param keeps today's behaviour.

        ``out = pl.add(out, b)`` reads the buffer and returns a fresh tensor, so
        it drops the write exactly like the #2352 case does. It is deliberately
        not repaired: which argument slot a call writes through differs per op
        (``assemble`` at 0, ``store`` at 2, the scatter and accumulate families
        at 0) and no single registry answers that, so any mention of the param
        is treated as "the user is threading it". Wrapping a genuine
        write-through a second time would corrupt a correct program, which is
        the worse failure.
        """

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(out, b)
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 0

    def test_store_into_param_is_not_rewrapped(self):
        """``out = pl.store(t, off, out)`` writes through arg 2, not arg 0."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                inp: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                local = pl.load(inp, [0, 0], [1, 64])
                out = pl.store(local, [0, 0], out)
                return out

        assert _count_calls(Prog, "kernel", _OP_TENSOR_ASSEMBLE) == 0
        assert _count_calls(Prog, "kernel", _OP_TILE_ASSEMBLE) == 0

    def test_self_assignment_is_not_rewritten(self):
        """``out = out`` writes nothing new and must not emit a copy."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, a, [0, 0])
                out = out  # noqa: PLW0127  # deliberate: the construct under test
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 1


class TestAnnotatedFormPassesThrough:
    """``out: T = <expr>`` is the printer's form and is left unchanged."""

    def test_annotated_assignment_is_not_rewritten(self):
        """Rewriting it would break print -> parse round-tripping."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out: pl.Tensor[[64, 64], pl.FP32] = pl.add(a, b)
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 0

    def test_desugared_program_round_trips(self):
        """print -> parse of a desugared program reproduces it exactly."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, b)
                return out

        reparsed = pl.parse(ir.python_print(Prog))
        ir.assert_structural_equal(reparsed, Prog)


class TestCreateTensorStillRejected:
    """The #889 / PR #910 shadowing error survives the new sugar."""

    def test_create_tensor_shadowing_still_flagged(self):
        """``out = pl.create_tensor(...)`` is not silently desugared."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out = pl.create_tensor([64, 64], dtype=pl.FP32)
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 0

        props = passes.IRPropertySet()
        props.insert(passes.IRProperty.OutParamNotShadowed)
        diagnostics = passes.PropertyVerifierRegistry.verify(props, Prog)
        assert any(d.rule_name == "OutParamNotShadowed" for d in diagnostics)


class TestIncoreOutParam:
    """The same sugar applies inside InCore kernels, not just ``pl.at`` bodies."""

    def test_incore_tensor_out_param_desugars(self):
        """An InCore Out param gets the same whole-tensor write."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out = pl.add(a, b)
                return out

        assert _count_calls(Prog, "kernel", _OP_TENSOR_ASSEMBLE) == 1

    def test_tile_out_param_matches_subscript_write(self):
        """A tile-typed Out param routes through ``tile.assemble``."""

        @pl.program
        class Bare:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tile[[64, 64], pl.FP32],
                out: pl.Out[pl.Tile[[64, 64], pl.FP32]],
            ) -> pl.Tile[[64, 64], pl.FP32]:
                out = a
                return out

        @pl.program
        class Subscript:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tile[[64, 64], pl.FP32],
                out: pl.Out[pl.Tile[[64, 64], pl.FP32]],
            ) -> pl.Tile[[64, 64], pl.FP32]:
                out[0:64, 0:64] = a
                return out

        assert _count_calls(Bare, "kernel", _OP_TILE_ASSEMBLE) == 1
        ir.assert_structural_equal(Bare, Subscript)


class TestLoopBodyWrite:
    """A rebind inside a loop body is rewritten like any other."""

    def test_bare_assignment_in_loop_writes_through(self):
        """The loop body writes the whole tensor rather than dropping it."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                for _ in pl.range(4):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        out = pl.add(a, b)
                return out

        assert _count_calls(Prog, "main", _OP_TENSOR_ASSEMBLE) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
