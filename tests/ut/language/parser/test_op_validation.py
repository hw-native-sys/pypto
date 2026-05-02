# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Coverage for the DSL-wrapper-based parser dispatch.

After moving op type-checking from the parser into the DSL wrappers, the
parser surfaces wrapper ``TypeError`` / ``ValueError`` as
``InvalidOperationError`` with the call-site span. These tests pin the
behavior we explicitly want from the new path: clean error messages, the
parser-pinned span flowing into IR nodes constructed inside wrappers, and
direct-Python wrapper calls staying functional.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import InvalidOperationError
from pypto.language.typing import Scalar
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir


class TestWrapperErrorsThroughParser:
    """Wrapper errors surface as InvalidOperationError with op name + span."""

    def test_unknown_pl_op_clean_error(self):
        with pytest.raises(InvalidOperationError, match=r"Unknown operation 'pl\.does_not_exist'"):

            @pl.function
            def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                result: pl.Tensor[[64], pl.FP32] = pl.does_not_exist(x)  # type: ignore[attr-defined]
                return result

    def test_pl_exp_on_scalar_emits_python_operators_hint(self):
        with pytest.raises(InvalidOperationError) as exc_info:

            @pl.function
            def main(
                config: pl.Tensor[[1], pl.FP32],
                out: pl.Tensor[[1], pl.FP32],
            ) -> pl.Tensor[[1], pl.FP32]:
                a: pl.Scalar[pl.FP32] = pl.tensor.read(config, [0])
                _ = pl.exp(a)  # pyright: ignore[reportArgumentType]
                return out

        msg = exc_info.value.message  # type: ignore[attr-defined]
        assert "pl.exp" in msg
        assert "Scalar" in msg
        assert "Python operators" in (exc_info.value.hint or "")  # type: ignore[attr-defined]

    def test_invalid_value_in_wrapper_names_op(self):
        """A ValueError raised deep in a wrapper is rewrapped with the op name."""
        with pytest.raises(
            InvalidOperationError,
            match=r"pl\.tensor operation 'cast'.*Invalid rounding mode",
        ):

            @pl.function
            def main(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.BF16]:
                result: pl.Tensor[[64], pl.BF16] = pl.tensor.cast(
                    x,
                    target_type=pl.BF16,
                    mode="bogus_mode",  # type: ignore[arg-type]
                )
                return result


class TestSpanPropagatesIntoWrapperConstructedNodes:
    """Span pinned by parser surfaces on IR nodes constructed inside wrappers."""

    def test_call_span_matches_parse_site(self):
        """``Call`` span must point at the user's source line, not the wrapper file."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                z: pl.Tensor[[64], pl.FP32] = pl.add(x, y)
                return z

        # Walk the IR: locate the pl.add Call and verify its span filename
        # matches this test file (the parse site), not tile_ops.py /
        # tensor_ops.py / unified_ops.py.
        found_calls: list[ir.Call] = []

        class _Collect(ir.IRVisitor):
            def visit_call(self, op):
                found_calls.append(op)
                super().visit_call(op)

        _Collect().visit_program(Prog)
        add_calls = [c for c in found_calls if c.op.name in ("tensor.add", "tile.add")]
        assert add_calls, "expected at least one tensor.add or tile.add Call in IR"

        for call in add_calls:
            span = call.span
            # The call site lives in this test file; wrapper file paths
            # would contain "tile_ops.py" or "unified_ops.py".
            assert span is not None
            assert "tile_ops.py" not in span.filename
            assert "tensor_ops.py" not in span.filename
            assert "unified_ops.py" not in span.filename


class TestDirectWrapperCallsStillWork:
    """Wrappers invoked outside the parser keep working — no contextvar set."""

    def test_unified_add_scalar_scalar_via_python(self):
        """Direct-Python ``pl.add(scalar, scalar)`` lowers via ``Scalar.__add__``."""
        # Build minimal scalars from constants — exercises the lhs + rhs path
        # in unified_ops.add without going through the parser.
        s1 = Scalar(expr=_ir.ConstInt(3, DataType.INT32, _ir.Span.unknown()))
        s2 = Scalar(expr=_ir.ConstInt(4, DataType.INT32, _ir.Span.unknown()))
        result = pl.add(s1, s2)
        assert isinstance(result, Scalar)


class TestFullPythonCallingConvention:
    """Parser delegates to DSL wrappers, so Python's full arg-binding works."""

    def test_mixed_positional_and_keyword_styles_all_parse(self):
        """All-positional, mixed, all-keyword, and unified-op-kwargs forms parse."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP16],
            ) -> pl.Tensor[[16, 16], pl.FP16]:
                # All positional
                t_pos: pl.Tile[[16, 16], pl.FP16] = pl.tile.load(x, [0, 0], [16, 16])
                # Mixed positional + keyword
                t_mix: pl.Tile[[16, 16], pl.FP16] = pl.tile.load(x, [0, 0], shapes=[16, 16])
                # All keyword
                t_kw: pl.Tile[[16, 16], pl.FP16] = pl.tile.load(tensor=x, offsets=[0, 0], shapes=[16, 16])
                # Unified-op kwargs
                t_add: pl.Tile[[16, 16], pl.FP16] = pl.add(lhs=t_pos, rhs=t_mix)
                # Suppress "unused" warnings while exercising every call site
                _ = (t_kw, t_add)
                return x

        # Each pl.tile.load lowers to one tile.load Call; the program should
        # contain exactly three of them regardless of how the source spelled
        # the args.
        load_calls: list[ir.Call] = []
        add_calls: list[ir.Call] = []

        class _Collect(ir.IRVisitor):
            def visit_call(self, op):
                if op.op.name == "tile.load":
                    load_calls.append(op)
                elif op.op.name == "tile.add":
                    add_calls.append(op)
                super().visit_call(op)

        _Collect().visit_program(Prog)
        assert len(load_calls) == 3, f"expected 3 tile.load calls, got {len(load_calls)}"
        assert len(add_calls) == 1, f"expected 1 tile.add call, got {len(add_calls)}"

    def test_dtype_attribute_works_positional_and_keyword(self):
        """``pl.cast(x, pl.BF16)`` and ``pl.cast(x, target_type=pl.BF16)`` both parse."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                x: pl.Tensor[[16, 16], pl.FP32],
            ) -> pl.Tensor[[16, 16], pl.BF16]:
                # Positional dtype attribute
                a: pl.Tensor[[16, 16], pl.BF16] = pl.cast(x, pl.BF16)
                # Keyword dtype attribute
                b: pl.Tensor[[16, 16], pl.BF16] = pl.cast(x, target_type=pl.BF16)
                _ = a
                return b

        cast_calls: list[ir.Call] = []

        class _Collect(ir.IRVisitor):
            def visit_call(self, op):
                if op.op.name in ("tensor.cast", "tile.cast"):
                    cast_calls.append(op)
                super().visit_call(op)

        _Collect().visit_program(Prog)
        assert len(cast_calls) == 2, f"expected 2 cast calls, got {len(cast_calls)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
