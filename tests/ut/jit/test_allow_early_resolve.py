# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end ``allow_early_resolve`` coverage through the ``@pl.jit`` pipeline.

JIT is a specialization layer, not a distinct dispatch surface: a JIT function
body is specialized then parsed exactly like ``@pl.program`` source, so the
``pl.at(..., allow_early_resolve=True)`` hint flows through the full Default
pass pipeline (specialize -> parse -> outline -> codegen) unchanged. These run
``compile_for_test`` (no device) and assert the synthesized dispatch's Arg
carries ``set_allow_early_resolve(true)`` (simpler#1065).
"""

import pypto.language as pl
import pytest
from pypto import codegen, ir


@pl.jit
def _at_early_resolve_entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """A JIT entry whose outlined ``pl.at`` block opts into early-dispatch."""
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


@pl.jit
def _at_no_flag_entry(a: pl.Tensor, c: pl.Out[pl.Tensor]):
    """Same shape, no hint — must emit no early-resolve call."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [128, 128])
        tile_c = pl.add(tile_a, 1.0)
        pl.store(tile_c, [0, 0], c)
    return c


def _orch_code(program: ir.Program) -> str:
    orch = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Orchestration)
    return codegen.generate_orchestration(program, orch).code


def test_jit_at_allow_early_resolve_emits_hint():
    torch = pytest.importorskip("torch")
    _at_early_resolve_entry._cache.clear()

    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    program = _at_early_resolve_entry.compile_for_test(a, c)

    code = _orch_code(program)
    assert "set_allow_early_resolve(true);" in code, code
    assert code.count("set_allow_early_resolve(true)") == 1, code


def test_jit_no_flag_emits_no_hint():
    torch = pytest.importorskip("torch")
    _at_no_flag_entry._cache.clear()

    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    program = _at_no_flag_entry.compile_for_test(a, c)

    assert "set_allow_early_resolve" not in _orch_code(program)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
