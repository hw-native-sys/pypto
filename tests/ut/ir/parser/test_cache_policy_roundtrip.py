# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser round-trip tests for the ``pl.set_cache_policy(...)`` declaration.

``pl.set_cache_policy(t, policy)`` emits no IR statement of its own: the parser
hoists it onto the enclosing scope as ``attrs['cache_policy_vars']``, a list of
``(Var, CachePolicy-as-int)`` pairs. These tests assert what a *reparse* of the
printed program recovers — which scope carries the declaration, which tensor it
names, and which policy — for the three shapes with distinct printer paths:

1. a ``with pl.at(...)`` scope;
2. a ``for i in pl.spmd(...)`` kernel, where the declaration attaches to the
   nested InCore carrier whose header the printer inlines away;
3. a ``for i in pl.spmd(...)`` kernel whose body is otherwise empty, where the
   printer would otherwise fall back to a ``pass`` filler.

Each case also asserts the print -> parse -> print fixpoint. The printer-side
counterpart (what text is emitted, and where) lives in
``tests/ut/ir/printing/test_cache_policy_printer.py``.

Note these assert textual stability rather than
``ir.assert_structural_equal``: the ``cache_policy_vars`` attr value type is not
yet handled by the structural comparator.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError

BYPASS = int(ir.CachePolicy.BYPASS)
DEFAULT = int(ir.CachePolicy.DEFAULT)

_CACHE_POLICY_VARS_ATTR = "cache_policy_vars"


def _collect_decls(node) -> list[tuple[str, list[tuple[str, int]]]]:
    """Walk an IR body and collect every scope carrying a cache-policy declaration.

    Returns ``(scope class name, [(tensor name, policy int), ...])`` per
    carrying scope, in traversal order, so a test can assert both *what* was
    declared and *which* scope kind ended up holding it.
    """
    found: list[tuple[str, list[tuple[str, int]]]] = []

    def walk(n) -> None:
        if isinstance(n, ir.ScopeStmt):
            entries = dict(n.attrs).get(_CACHE_POLICY_VARS_ATTR)
            if entries is not None:
                found.append((type(n).__name__, [(var.name_hint, int(policy)) for var, policy in entries]))
        if isinstance(n, ir.SeqStmts):
            for s in n.stmts:
                walk(s)
        elif isinstance(n, ir.IfStmt):
            walk(n.then_body)
            if n.else_body is not None:
                walk(n.else_body)
        elif getattr(n, "body", None) is not None:
            walk(n.body)

    walk(node)
    return found


def _program_decls(prog) -> list[tuple[str, list[tuple[str, int]]]]:
    decls: list[tuple[str, list[tuple[str, int]]]] = []
    for func in prog.functions.values():
        decls.extend(_collect_decls(func.body))
    return decls


def _reparse(prog):
    """print -> parse, asserting the reprint is a fixpoint. Returns the reparsed program."""
    text = ir.python_print(prog)
    reparsed = pl.parse_program(text)
    reprinted = ir.python_print(reparsed)
    assert reprinted == text, (
        f"round-trip is not a fixpoint\n--- first ---\n{text}\n--- second ---\n{reprinted}"
    )
    return reparsed


# ─── (i) a ``with pl.at(...)`` scope ──────────────────────────────────────


def _at_scope_program():
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


def test_parse_at_scope_records_declaration_on_the_scope():
    """The marker leaves no statement behind; it lands on the enclosing scope's attrs."""
    prog = _at_scope_program()
    assert _program_decls(prog) == [("InCoreScopeStmt", [("b", BYPASS)])]


def test_roundtrip_at_scope_recovers_declaration():
    """A reparse of the printed ``with pl.at(...)`` recovers the same declaration."""
    prog = _at_scope_program()
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


# ─── (ii) a ``for i in pl.spmd(...)`` kernel ──────────────────────────────


def _spmd_for_program():
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


def test_parse_spmd_for_attaches_declaration_to_the_incore_carrier():
    """The declaration rides the InCore carrier, not the Spmd scope around it.

    The carrier is what the outliner turns into the per-block kernel whose
    params the policy resolves against, so attaching it anywhere else would
    leave nothing downstream to consume it.
    """
    assert _program_decls(_spmd_for_program()) == [("InCoreScopeStmt", [("b", BYPASS)])]


def test_roundtrip_spmd_for_recovers_declaration():
    """The printer inlines the carrier's header away; the reparse still recovers it."""
    prog = _spmd_for_program()
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


# ─── (iii) a scope whose body is otherwise empty ──────────────────────────


def _empty_spmd_body_program():
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

    return EmptyBody


def test_roundtrip_empty_spmd_body_recovers_declaration():
    """A body holding only the declaration survives; the ``pass`` filler would erase it."""
    prog = _empty_spmd_body_program()
    assert _program_decls(prog) == [("InCoreScopeStmt", [("b", BYPASS)])]
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


# ─── Policies, multiplicity and spelling ──────────────────────────────────


def test_roundtrip_preserves_both_policies_and_their_order():
    """DEFAULT and BYPASS both survive, distinctly, in the order they were written."""

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

    expected = [("InCoreScopeStmt", [("b", BYPASS), ("a", DEFAULT)])]
    assert _program_decls(Multi) == expected
    assert _program_decls(_reparse(Multi)) == expected


def test_roundtrip_normalises_position_of_a_late_declaration():
    """A declaration written mid-body reparses identically — position carries no meaning."""

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

    assert _program_decls(_reparse(Late)) == [("InCoreScopeStmt", [("b", BYPASS)])]


def test_parse_accepts_the_submodule_spelling_and_normalises_it():
    """``pl.tensor.set_cache_policy`` parses too, and prints as ``pl.set_cache_policy``.

    The marker function lives in ``tensor_ops``, so both spellings reach the
    parser; normalising on print keeps the round-trip a fixpoint from either.
    """
    source = """
import pypto.language as pl


@pl.program
class Alt:
    @pl.function
    def main(self, a: pl.Tensor[[256, 128], pl.FP32], b: pl.Tensor[[128, 256], pl.FP32],
             out: pl.Out[pl.Tensor[[256, 256], pl.FP32]]) -> pl.Tensor[[256, 256], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.tensor.set_cache_policy(b, pl.CachePolicy.BYPASS)
            c: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.matmul(a, b, out_dtype=pl.FP32)
            out: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.assemble(out, c, [0, 0])
        return out
"""
    prog = pl.parse_program(source)
    assert _program_decls(prog) == [("InCoreScopeStmt", [("b", BYPASS)])]

    text = ir.python_print(prog)
    assert "pl.set_cache_policy(b, pl.CachePolicy.BYPASS)" in text, text
    assert "pl.tensor.set_cache_policy(" not in text, text
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


def test_repeating_the_same_policy_for_one_tensor_is_accepted():
    """A restated declaration is redundant, not an error: one entry survives."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k"):
                pl.set_cache_policy(a, pl.CachePolicy.BYPASS)
                pl.set_cache_policy(a, pl.CachePolicy.BYPASS)
                t: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                out = pl.store(t, [0, 0], out)
            return out

    decls = _program_decls(Prog)
    assert sum(len(entries) for _scope, entries in decls) == 1, decls


def test_conflicting_policies_for_one_tensor_are_rejected():
    """Two different policies for one tensor in one scope contradict each other.

    Keeping the first silently resolved this toward BYPASS — the direction that
    also asserts the coherency contract the second statement retracts — so the
    parser rejects it instead of picking a winner.
    """
    with pytest.raises(ParserSyntaxError, match="conflicting policies"):

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k"):
                    pl.set_cache_policy(a, pl.CachePolicy.BYPASS)
                    pl.set_cache_policy(a, pl.CachePolicy.DEFAULT)
                    t: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                    out = pl.store(t, [0, 0], out)
                return out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
