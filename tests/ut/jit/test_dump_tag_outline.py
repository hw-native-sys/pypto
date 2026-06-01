# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Post-outline ``pl.dump_tag`` resolution for the ``@pl.jit`` +
``@pl.jit.inline`` + ``with pl.incore()`` style (simpler#844).

Unlike the explicit ``self.kernel(...)`` orchestration style (covered by
``tests/ut/language/parser/test_dump_tag_dsl.py``), here the kernel dispatches
are synthesised by the outline passes, not written at parse time. The dump
intent therefore rides a scope-level ``kAttrDumpVars`` carrier:

  - ``pl.dump_tag`` inside an inline helper (forward-sticky) seeds the enclosing
    ``with pl.incore()`` scope's dump list at parse;
  - ``pl.dump_tag`` at the inline call site lands on the
    inline call's ``dump_vars``, which ``InlineFunctions`` transfers onto the
    spliced scope;
  - the outliner translates the captured scope dump Vars into the synthesised
    dispatch's ``dump_vars`` by Var identity.

These run the full Default pass pipeline via ``compile_for_test`` (no device),
so they also exercise the print -> reparse roundtrip after every pass (the
``tests/ut/conftest.py`` autouse fixture). The companion device/manifest checks
live in ``tests/st/codegen/dsl/test_dump_tag.py``.
"""

import pypto.language as pl
import pytest
from pypto import ir


@pl.jit.inline
def _add_inline(a: pl.Tensor, c: pl.Tensor):
    """c = a + 1.0. Inline-scope ``pl.dump_tag(a)`` -> the incore scope's dump
    list -> kernel1 dumps its ``a`` arg (input role)."""
    pl.dump_tag(a)
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit.inline
def _mul_inline(a: pl.Tensor, c: pl.Tensor):
    """c = a * 2.0. No dump_tag — kernel2 must dump nothing."""
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.mul(tile_a, 2.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def _add_mul_with_dump_tags(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Entry: c = (a + 1) * 2, with mixed-scope dump_tag markers."""
    intermediate = pl.create_tensor([128, 128], dtype=pl.FP32)
    pl.dump_tag(intermediate)  # entry-scope tag -> inline call dump_vars -> kernel1 inout
    intermediate = _add_inline(a, intermediate)
    c = _mul_inline(intermediate, c)
    return c


@pl.jit
def _add_mul_no_tags(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Same shape, no dump markers — every dispatch must dump nothing."""
    intermediate = pl.create_tensor([128, 128], dtype=pl.FP32)
    intermediate = _add_inline_untagged(a, intermediate)
    c = _mul_inline(intermediate, c)
    return c


@pl.jit.inline
def _add_inline_untagged(a: pl.Tensor, c: pl.Tensor):
    with pl.incore():
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit.inline
def _passthrough_outer(a: pl.Tensor, c: pl.Tensor):
    """Forwards ``a`` into a deeper inline; owns no scope that consumes ``a``."""
    c = _add_inline_untagged(a, c)
    return c


@pl.jit
def _entry_tag_through_two_inline_levels(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Entry tags ``a``, which is consumed two inline levels deep
    (_passthrough_outer -> _add_inline_untagged -> incore scope)."""
    pl.dump_tag(a)
    c = _passthrough_outer(a, c)
    return c


def _dispatch_dump_vars(program: ir.Program) -> dict[str, list[str]]:
    """Map each synthesised kernel-dispatch callee name to its sorted dump_vars
    name_hints. Dispatches are the cross-function Calls to the outlined incore
    functions (``*_incore_*``)."""
    out: dict[str, list[str]] = {}

    class _Collector(ir.IRVisitor):
        def visit_call(self, op):
            name = getattr(getattr(op, "op", None), "name", "")
            if "_incore_" in name:
                dv = (op.attrs or {}).get("dump_vars")
                out[name] = sorted(v.name_hint for v in dv) if dv else []
            super().visit_call(op)

    _Collector().visit_program(program)
    return out


def _base(name: str) -> str:
    """Strip the SSA ``__ssa_vN`` suffix from a Var name_hint."""
    return name.split("__", 1)[0]


def test_dump_tag_reaches_outlined_dispatch_single_func():
    """Only the tagged kernel dumps, and it dumps both its tagged input and its
    tagged inout — the single-``func_id`` invariant the scene test asserts."""
    torch = pytest.importorskip("torch")
    _add_mul_with_dump_tags._cache.clear()

    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    program = _add_mul_with_dump_tags.compile_for_test(a, c)

    dumps = _dispatch_dump_vars(program)
    assert len(dumps) == 2, f"expected two outlined dispatches, got {sorted(dumps)}"

    dumping = {name: dv for name, dv in dumps.items() if dv}
    assert len(dumping) == 1, f"selective dump should retain exactly one kernel, got {dumps}"

    (only_dv,) = dumping.values()
    # kernel1 dumps its input ``a`` and its inout ``intermediate`` (the tagged
    # create result), tracked by Var identity through inline + SSA + outline.
    assert {_base(n) for n in only_dv} == {"a", "intermediate"}, only_dv


def test_tag_survives_multi_level_inline_passthrough():
    """A tag on a tensor forwarded through a nested inline (no scope of the
    outer inline consumes it) still reaches the deep dispatch — InlineFunctions
    carries it on the nested dispatch Call across each splice iteration."""
    torch = pytest.importorskip("torch")
    _entry_tag_through_two_inline_levels._cache.clear()

    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    program = _entry_tag_through_two_inline_levels.compile_for_test(a, c)

    dumps = _dispatch_dump_vars(program)
    dumping = {name: dv for name, dv in dumps.items() if dv}
    assert len(dumping) == 1, f"expected the deep dispatch to dump a, got {dumps}"
    (only_dv,) = dumping.values()
    assert {_base(n) for n in only_dv} == {"a"}, only_dv


def test_no_dump_tag_yields_no_dispatch_dump_vars():
    """Without any marker, no dispatch carries dump_vars (selective dump off)."""
    torch = pytest.importorskip("torch")
    _add_mul_no_tags._cache.clear()

    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    program = _add_mul_no_tags.compile_for_test(a, c)

    dumps = _dispatch_dump_vars(program)
    assert dumps, "expected outlined dispatches"
    assert all(dv == [] for dv in dumps.values()), f"unexpected dump_vars: {dumps}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
