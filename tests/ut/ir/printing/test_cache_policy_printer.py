# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Printer tests for the ``pl.set_cache_policy(...)`` scope declaration.

The declaration rides on ``ScopeStmt.attrs['cache_policy_vars']`` and is
surfaced as leading marker *statements* in the scope body rather than as a
header kwarg the way ``no_dep_args=`` / ``dumps=`` do, because a statement is
the DSL surface the parser accepts. These tests pin the emitted text and assert
the print -> parse -> print fixpoint for the three shapes that exercise
distinct printer paths:

1. a ``with pl.at(...)`` scope, which prints its own header and body;
2. a ``for i in pl.spmd(...)`` kernel, whose printer inlines the nested
   ``pl.at(level=CORE_GROUP)`` carrier away — the declaration must be printed
   from that inner InCore scope or it is lost;
3. a ``for i in pl.spmd(...)`` kernel whose body holds nothing else, where the
   printer would otherwise emit a ``pass`` filler and drop the marker.

The parser-side counterpart (what is recovered onto which scope) lives in
``tests/ut/ir/parser/test_cache_policy_roundtrip.py``.
"""

import pypto.language as pl
import pytest
from pypto import ir

BYPASS_MARKER = "pl.set_cache_policy(b, pl.CachePolicy.BYPASS)"


def _at_scope_program():
    """A ``with pl.at(...)`` scope declaring BYPASS on the matmul's rhs."""

    @pl.program
    class AtScope:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                out = pl.assemble(out, c, [0, 0])
            return out

    return AtScope


def _spmd_for_program():
    """A ``for i in pl.spmd(...)`` kernel: the inlined-InCore carrier hole."""

    @pl.program
    class SpmdFor:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            b: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for i in pl.spmd(4):
                pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                offset = i * 128
                tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                out = pl.store(tile_b, [offset, 0], out)
            return out

    return SpmdFor


def _empty_spmd_body_program(*, declare: bool):
    """A ``for i in pl.spmd(...)`` whose body is empty but for the declaration.

    With ``declare=False`` the same kernel is the baseline that shows the
    printer really does fall back to ``pass`` here — otherwise the "no ``pass``"
    assertion would pass vacuously.
    """
    if declare:

        @pl.program
        class EmptyBody:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                return out

    else:

        @pl.program
        class EmptyBody:
            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                b: pl.Tensor[[512, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
            ) -> pl.Tensor[[512, 128], pl.FP32]:
                for i in pl.spmd(4):
                    pass
                return out

    return EmptyBody


def _assert_print_parse_print_stable(prog) -> str:
    """Assert the print -> parse -> print fixpoint and return the printed text."""
    text = ir.python_print(prog)
    reprinted = ir.python_print(pl.parse_program(text))
    assert reprinted == text, (
        f"round-trip is not a fixpoint\n--- first ---\n{text}\n--- second ---\n{reprinted}"
    )
    return text


def _line_index(text: str, needle: str) -> int:
    """Index of the first line containing ``needle``; -1 when absent."""
    for i, line in enumerate(text.splitlines()):
        if needle in line:
            return i
    return -1


# ─── (i) a ``with pl.at(...)`` scope ──────────────────────────────────────


def test_print_at_scope_emits_marker_statement():
    """The declaration prints as a statement inside the body, not a header kwarg."""
    text = ir.python_print(_at_scope_program())

    header = _line_index(text, "with pl.at(level=pl.Level.CORE_GROUP")
    marker = _line_index(text, BYPASS_MARKER)
    body = _line_index(text, "pl.tensor.matmul(")
    assert header >= 0, text
    assert marker == header + 1, text
    assert body > marker, text

    # The marker is indented one level deeper than the ``with`` that owns it.
    header_indent = len(text.splitlines()[header]) - len(text.splitlines()[header].lstrip())
    marker_indent = len(text.splitlines()[marker]) - len(text.splitlines()[marker].lstrip())
    assert marker_indent > header_indent, text

    # Never a header kwarg, and never the raw attr key.
    assert "cache_policy" not in text.splitlines()[header], text
    assert "cache_policy_vars" not in text, text


def test_roundtrip_at_scope():
    """``with pl.at(...)`` carrying a declaration is a print -> parse -> print fixpoint."""
    text = _assert_print_parse_print_stable(_at_scope_program())
    assert BYPASS_MARKER in text, text


# ─── (ii) a ``for i in pl.spmd(...)`` kernel ──────────────────────────────


def test_print_spmd_for_emits_marker_from_inlined_incore_carrier():
    """The Spmd for-form inlines the InCore header away; the marker must survive.

    The declaration attaches to the nested ``pl.at(level=CORE_GROUP)`` carrier
    that this printer branch does not spell out, so printing only from the Spmd
    scope would silently drop it.
    """
    text = ir.python_print(_spmd_for_program())

    loop = _line_index(text, "for i in pl.spmd(4):")
    marker = _line_index(text, BYPASS_MARKER)
    assert loop >= 0, text
    assert marker == loop + 1, text
    # The carrier really is inlined away — there is no nested ``pl.at`` to hang
    # the marker on, which is exactly why the Spmd branch has to print it.
    assert "with pl.at(" not in text, text


def test_roundtrip_spmd_for():
    """``for i in pl.spmd(...)`` carrying a declaration is a fixpoint."""
    text = _assert_print_parse_print_stable(_spmd_for_program())
    assert BYPASS_MARKER in text, text


# ─── (iii) a scope whose body is otherwise empty ──────────────────────────


def test_empty_spmd_body_prints_pass_without_a_declaration():
    """Baseline for the next test: an empty Spmd body does get a ``pass`` filler."""
    text = ir.python_print(_empty_spmd_body_program(declare=False))
    assert _line_index(text, "for i in pl.spmd(4):") >= 0, text
    assert _line_index(text, "pass") >= 0, text


def test_empty_spmd_body_prints_marker_instead_of_pass():
    """A body holding only the declaration prints the marker and no ``pass``.

    The marker counts as body content: emitting ``pass`` alongside it would be
    redundant, and emitting ``pass`` *instead* of it would discard a contract
    the author stated.
    """
    text = ir.python_print(_empty_spmd_body_program(declare=True))

    loop = _line_index(text, "for i in pl.spmd(4):")
    marker = _line_index(text, BYPASS_MARKER)
    assert loop >= 0, text
    assert marker == loop + 1, text
    assert _line_index(text, "pass") == -1, text


def test_roundtrip_empty_spmd_body():
    """The otherwise-empty scope is a fixpoint, marker intact on the second print."""
    text = _assert_print_parse_print_stable(_empty_spmd_body_program(declare=True))
    assert BYPASS_MARKER in text, text
    assert _line_index(text, "pass") == -1, text


# ─── Enum members, ordering and multiplicity ──────────────────────────────


def test_roundtrip_multiple_declarations_and_both_policies():
    """Both enum members render by name, and several declarations keep their order."""

    @pl.program
    class Multi:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                pl.set_cache_policy(a, pl.CachePolicy.DEFAULT)
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                out = pl.assemble(out, c, [0, 0])
            return out

    text = _assert_print_parse_print_stable(Multi)
    first = _line_index(text, BYPASS_MARKER)
    second = _line_index(text, "pl.set_cache_policy(a, pl.CachePolicy.DEFAULT)")
    assert first >= 0, text
    assert second == first + 1, text


def test_declaration_written_late_prints_first():
    """The round-trip is position-normalising: markers lead the scope body.

    The declaration describes the whole scope, so where the author wrote it in
    the body carries no meaning — the printer emits it first, and the fixpoint
    is reached on the very first reprint rather than drifting.
    """

    @pl.program
    class Late:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                pl.set_cache_policy(b, pl.CachePolicy.BYPASS)
                out = pl.assemble(out, c, [0, 0])
            return out

    text = _assert_print_parse_print_stable(Late)
    assert _line_index(text, BYPASS_MARKER) < _line_index(text, "pl.tensor.matmul("), text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
