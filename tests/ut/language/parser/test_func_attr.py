# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.func_attr({...})`` — body-position function attributes.

A decorator is evaluated *before* the signature binds any name, so
``@pl.function(attrs={...})`` can only ever spell values that reference nothing.
``pl.func_attr`` sits in the body prologue, after the parameters are bound, which
makes a function attribute that references a parameter expressible for the first
time. Both spellings parse; the printer emits the prologue form.
"""

import warnings

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError


def _function(program, name="kernel"):
    """Fetch a function from a program, asserting it exists."""
    func = program.get_function(name)
    assert func is not None, f"program has no function '{name}'"
    return func


def _source(prologue):
    """A minimal kernel whose prologue is ``prologue``.

    Negatives that are deliberately ill-typed Python (a list argument, an int
    key) are written as source text rather than a decorated class, so the type
    checker is not asked to accept a call the parser exists to reject.
    """
    return f"""
import pypto.language as pl


@pl.program
class P:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
        {prologue}
        out[0:64] = x[0:64]
"""


def _param_index(func, attr_value):
    """Index of the param ``attr_value`` references, or None if it references none."""
    for index, param in enumerate(func.params):
        if attr_value.same_as(param):
            return index
    return None


class TestReferenceValuedAttrs:
    """The capability the decorator form cannot express at all."""

    def test_attr_references_a_parameter(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ):
                pl.func_attr({"stationary": w})
                out[0:64, 0:64] = x[0:64, 0:64]

        func = _function(P)
        # Resolves to `w` by identity — param index 1, not a copy and not a name.
        assert _param_index(func, func.attrs["stationary"]) == 1

    def test_reference_valued_attr_round_trips_through_text(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ):
                pl.func_attr({"stationary": w})
                out[0:64, 0:64] = x[0:64, 0:64]

        text = ir.python_print(P)
        assert 'pl.func_attr({"stationary": w})' in text, text

        reparsed = pl.parse_program(text)
        ir.assert_structural_equal(reparsed, P)
        func = _function(reparsed)
        assert _param_index(func, func.attrs["stationary"]) == 1

    def test_reference_valued_attr_round_trips_through_serialization(self):
        """Serialization dispatches on value type, so no serializer change was needed."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64, 64], pl.FP32],
                w: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ):
                pl.func_attr({"stationary": w})
                out[0:64, 0:64] = x[0:64, 0:64]

        restored = ir.deserialize(ir.serialize(P))
        ir.assert_structural_equal(restored, P)
        func = _function(restored)
        # Identity must survive: the Var resolves back to the param, not a clone.
        assert _param_index(func, func.attrs["stationary"]) == 1


class TestEquivalenceWithDecoratorForm:
    def test_static_attr_matches_decorator_spelling(self):
        @pl.program
        class Body:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                out[0:64] = x[0:64]

        @pl.program
        class Decorator:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                out[0:64] = x[0:64]

        ir.assert_structural_equal(Body, Decorator)

    def test_split_keeps_its_enum_type_through_the_prologue(self):
        """``split`` stores an int but is spelled as the enum on both paths."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                out[0:64] = x[0:64]

        func = _function(P)
        assert func.attrs["split"] == ir.SplitMode.UP_DOWN.value
        assert func.split == ir.SplitMode.UP_DOWN

    def test_multiple_calls_merge(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                pl.func_attr({"windowize": True})
                out[0:64] = x[0:64]

        attrs = _function(P).attrs
        assert attrs["split"] == ir.SplitMode.UP_DOWN.value
        assert attrs["windowize"] is True

    def test_body_and_decorator_attrs_combine(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore, attrs={"windowize": True})
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                out[0:64] = x[0:64]

        attrs = _function(P).attrs
        assert attrs["windowize"] is True
        assert attrs["split"] == ir.SplitMode.UP_DOWN.value


class TestParserRejections:
    def test_rejects_placement_after_a_statement(self):
        with pytest.raises(ParserSyntaxError, match="must appear before every other statement"):

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    out[0:64] = x[0:64]
                    pl.func_attr({"split": pl.SplitMode.UP_DOWN})

    def test_rejects_duplicate_key_across_two_calls(self):
        with pytest.raises((ParserSyntaxError, ValueError), match="[Dd]uplicate.*split"):

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                    pl.func_attr({"split": pl.SplitMode.LEFT_RIGHT})
                    out[0:64] = x[0:64]

    def test_duplicate_key_error_does_not_leak_builder_check_tail(self):
        """The duplicate-key CHECK lives in C++; its FatalLogger tail must not reach the user.

        ``IRBuilder::AddFunctionAttrs`` rejects the second key with a ``CHECK``, whose message
        carries "Check failed: <C++ expr> at <absolute path>/builder.cpp:<line>". The parser
        strips that tail so it does not render inside the bold ``Error:`` header.
        """
        with pytest.raises(ParserSyntaxError) as exc_info:

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                    pl.func_attr({"split": pl.SplitMode.LEFT_RIGHT})
                    out[0:64] = x[0:64]

        message = exc_info.value.message
        assert "Check failed" not in message
        assert "builder.cpp" not in message
        # The actionable half must survive the strip.
        assert "Each attribute may be declared only once." in message

    def test_rejects_duplicate_key_within_one_call(self):
        with pytest.raises((ParserSyntaxError, ValueError), match="[Dd]uplicate.*windowize"):
            pl.parse_program(_source('pl.func_attr({"windowize": True, "windowize": False})'))

    def test_rejects_key_duplicated_between_decorator_and_body(self):
        with pytest.raises((ParserSyntaxError, ValueError), match="[Dd]uplicate.*split"):

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    pl.func_attr({"split": pl.SplitMode.LEFT_RIGHT})
                    out[0:64] = x[0:64]

    def test_rejects_non_dict_argument(self):
        with pytest.raises(ParserSyntaxError, match="must be a dict literal"):
            pl.parse_program(_source('pl.func_attr(["split"])'))

    def test_rejects_keyword_arguments(self):
        with pytest.raises(ParserSyntaxError, match="exactly one positional dict argument"):
            pl.parse_program(_source("pl.func_attr(split=1)"))

    def test_rejects_non_string_key(self):
        with pytest.raises(ParserSyntaxError, match="key must be a string literal"):
            pl.parse_program(_source("pl.func_attr({1: True})"))

    @pytest.mark.parametrize("key", ["auto_scope", "external_source"])
    def test_rejects_decorator_only_attrs(self, key):
        """The parser reads these before it walks the body, so body position is too late."""
        with pytest.raises(ParserSyntaxError, match=f"'{key}' cannot be set with pl.func_attr"):
            pl.parse_program(_source(f'pl.func_attr({{"{key}": True}})'))


class TestDeprecation:
    def test_decorator_attrs_warns(self):
        with pytest.warns(DeprecationWarning, match="cannot reference parameters"):

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    out[0:64] = x[0:64]

    def test_func_attr_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore)
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                    out[0:64] = x[0:64]

        assert _function(P).attrs["split"] == ir.SplitMode.UP_DOWN.value

    def test_empty_decorator_attrs_do_not_warn(self):
        """``attrs={}`` is deliberately silent, not an oversight.

        It produces IR identical to omitting ``attrs=`` entirely — zero attrs —
        so warning on one spelling but not the other would be arbitrary. The
        warning also tells the reader to use ``pl.func_attr({...})``, which for
        an empty dict would mean writing a no-op prologue. The guard therefore
        fires exactly when an attr actually exists to migrate.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            @pl.program
            class P:
                @pl.function(type=pl.FunctionType.InCore, attrs={})
                def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                    out[0:64] = x[0:64]

        assert dict(_function(P).attrs) == {}

    def test_printed_ir_never_warns_on_reparse(self):
        """The compiler must not warn against its own output.

        Printed IR is reparsed on every ``VerificationLevel.Roundtrip`` pass
        boundary, so any attr still printed in the deprecated ``attrs=`` spelling
        would warn once per pass. Every attr must reach a non-deprecated
        spelling: a dedicated keyword, or the ``pl.func_attr`` prologue.
        """

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore, attrs={"split": pl.SplitMode.UP_DOWN})
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                out[0:64] = x[0:64]

        text = ir.python_print(P)
        assert "attrs=" not in text, text

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            reparsed = pl.parse_program(text)
        ir.assert_structural_equal(reparsed, P)


class TestPrinter:
    def test_auto_scope_still_prints_as_its_own_kwarg(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                pl.func_attr({"windowize": True})
                with pl.scope():
                    a = self.k1(x)
                return a

        text = ir.python_print(P)
        assert "auto_scope=False" in text, text
        # auto_scope stays on the decorator (the parser reads it before the body
        # walk); the ordinary attr moves to the prologue.
        body = text.split("def main")[1]
        assert "auto_scope" not in body, text
        assert 'pl.func_attr({"windowize": True})' in body, text
        ir.assert_structural_equal(pl.parse_program(text), P)

    def test_empty_attrs_emit_no_prologue(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                out[0:64] = x[0:64]

        assert "func_attr" not in ir.python_print(P)


class TestProloguePlacementAndValueShapes:
    """Regressions from review of the original change."""

    def test_docstring_may_precede_the_prologue(self):
        """A docstring is not "another statement" — it precedes the prologue."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                """Kernel docstring."""
                pl.func_attr({"split": pl.SplitMode.UP_DOWN})
                out[0:64] = x[0:64]

        assert _function(P).attrs["split"] == ir.SplitMode.UP_DOWN.value
        ir.assert_structural_equal(pl.parse_program(ir.python_print(P)), P)

    def test_split_none_is_dropped_like_the_decorator_path(self):
        """Storing the 0 would be a spelling the printer filters, losing the key."""

        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]):
                pl.func_attr({"split": pl.SplitMode.NONE})
                out[0:64] = x[0:64]

        assert "split" not in _function(P).attrs
        assert _function(P).split is None
        ir.assert_structural_equal(pl.parse_program(ir.python_print(P)), P)

    def test_int_list_under_an_open_world_key_names_what_is_supported(self):
        """An unsupported list shape is rejected, and the message says what fits.

        Integer lists stay reserved-key-only: accepting them under any key adds
        a function-attr value type, which is a capability change rather than a
        syntax one. The diagnostic used to name ``ArgDirection`` alone, which no
        caller writing ``[16, 32]`` would recognise as the rule they broke.
        """
        with pytest.raises(ParserSyntaxError) as excinfo:
            pl.parse_program(_source('pl.func_attr({"tile_sizes": [16, 32]})'))

        message = str(excinfo.value)
        assert "Unsupported list element type for key: tile_sizes" in message
        assert "Var" in message and "ArgDirection" in message
        assert "arg_direction_overrides" in message

    def test_abstract_subworker_keeps_its_attrs_printable(self):
        """An abstract SubWorker's attrs print as a prologue before the bare ``...``.

        The body still reads as abstract on reparse: the directive is skipped
        when classifying abstractness, exactly as a docstring is.
        """

        @pl.program
        class P:
            @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker, attrs={"tag": 1})
            def cb(x: pl.Tensor[[4], pl.FP16]) -> pl.Tensor[[4], pl.FP16]: ...

        func = _function(P, "cb")
        assert func.attrs["tag"] == 1
        assert func.requires_runtime_binding
        ir.assert_structural_equal(pl.parse_program(ir.python_print(P)), P)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
