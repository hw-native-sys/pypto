# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the EliminateRedundantVarCopy pass.

The pass copy-propagates pure ``X = Y`` AssignStmts (value is a Var/IterArg)
in Orchestration functions, rewriting uses of ``X`` to ``Y`` and dropping the
copy. It runs after DeriveCallDirections; these lineage-redundant rebinds do
not survive to the frontend, so the tests build the post-DeriveCallDirections
IR shape directly via ``pl.parse_program`` and run the pass in isolation.

Safety guards verified below: neither side may be a loop/branch carry lvalue
(codegen manages those as reassigned C++ locals across iterations/phases), and
the source must be visible at every read of the folded name — a source defined
inside a ``manual_scope`` dies at the block's closing brace.
"""

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType


@pytest.fixture(autouse=True)
def _setup_backend():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    yield
    backend.reset_for_testing()


def _fold(src: str) -> ir.Program:
    return passes.eliminate_redundant_var_copy()(pl.parse_program(src))


def test_folds_pure_param_copy():
    """``x = a`` (a is a read-only param) is dropped; its use rewrites to ``a``."""
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        x__ssa_v1: pl.Tensor[[64], pl.FP32] = a
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.kernel(x__ssa_v1, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    expected = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.kernel(a, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(expected))


def test_folds_ssa_copy_chain():
    """``x = a; z = x`` collapses so the kernel call reads ``a`` directly."""
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        x__ssa_v1: pl.Tensor[[64], pl.FP32] = a
        z__ssa_v1: pl.Tensor[[64], pl.FP32] = x__ssa_v1
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.kernel(z__ssa_v1, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    expected = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.kernel(a, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(expected))


def test_folds_in_place_written_source():
    """An in-place-written source still folds: ``x`` and ``p`` are the same buffer.

    Orchestration tensors are handles, so a later in-place write through either
    name is observed through both. Only carry lvalues and scope visibility gate
    the fold, not the number of writers.
    """
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def prod(self, o: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return o

    @pl.function(type=pl.FunctionType.InCore)
    def wr(self, o: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return o

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        p__ssa_v0: pl.Tensor[[64], pl.FP32] = self.prod(a, attrs={"arg_directions": [pl.adir.inout]})
        x__ssa_v1: pl.Tensor[[64], pl.FP32] = p__ssa_v0
        w__ssa_v0: pl.Tensor[[64], pl.FP32] = self.wr(x__ssa_v1, attrs={"arg_directions": [pl.adir.inout]})
        return w__ssa_v0
"""
    expected = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def prod(self, o: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return o

    @pl.function(type=pl.FunctionType.InCore)
    def wr(self, o: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return o

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        p__ssa_v0: pl.Tensor[[64], pl.FP32] = self.prod(a, attrs={"arg_directions": [pl.adir.inout]})
        w__ssa_v0: pl.Tensor[[64], pl.FP32] = self.wr(p__ssa_v0, attrs={"arg_directions": [pl.adir.inout]})
        return w__ssa_v0
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(expected))


def test_folds_outer_base_across_manual_scope():
    """Issue #1713: a manual_scope-internal rebind of an outer base folds away.

    The scope-internal SSA rebind chain (``x__rv_v2 = base``; ``x__rv_v5 =
    x__rv_v2``) emits block-local ``Tensor`` names that die at the closing
    brace, so the after-scope reader fails to C++-compile. Resolving the chain
    to the outer ``base`` removes every block-local alias.
    """
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def rd(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, base: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            x__rv_v2: pl.Tensor[[64], pl.FP32] = base
            x__rv_v5: pl.Tensor[[64], pl.FP32] = x__rv_v2
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.rd(x__rv_v5, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    expected = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def rd(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, base: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            pass
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.rd(base, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(expected))


def test_keeps_scope_local_source():
    """A source defined *inside* a manual_scope is not visible to an after-scope
    reader, so the copy is kept — folding would merely move the out-of-scope bug.
    """
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def rd(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, base: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            p__ssa_v0: pl.Tensor[[64], pl.FP32] = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            x__ssa_v1: pl.Tensor[[64], pl.FP32] = p__ssa_v0
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.rd(x__ssa_v1, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(before))


def test_keeps_loop_carry():
    """A copy of a loop ``return_var`` (mutable carry) is kept."""
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32], o: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32], out: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for i__idx_v0, (out__iter_v1,) in pl.range(4, init_values=(out,)):
            out__ssa_v3: pl.Tensor[[64], pl.FP32] = self.kernel(a, out__iter_v1, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
            out__rv_v2: pl.Tensor[[64], pl.FP32] = pl.yield_(out__ssa_v3)
        c__ssa_v1: pl.Tensor[[64], pl.FP32] = out__rv_v2
        return c__ssa_v1
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(before))


def test_keeps_call_rhs():
    """A non-copy AssignStmt (value is a Call) is left untouched."""
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return a

    @pl.function(type=pl.FunctionType.Orchestration)
    def orch(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        r__ssa_v0: pl.Tensor[[64], pl.FP32] = self.kernel(a, attrs={"arg_directions": [pl.adir.input]})
        return r__ssa_v0
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(before))


def test_keeps_incore_function():
    """The pass only rewrites Orchestration functions; InCore bodies are untouched."""
    before = """
import pypto.language as pl

@pl.program
class M:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        x__ssa_v1: pl.Tensor[[64], pl.FP32] = a
        return x__ssa_v1
"""
    ir.assert_structural_equal(_fold(before), pl.parse_program(before))


def test_pass_metadata():
    """The factory produces a Pass named EliminateRedundantVarCopy."""
    p = passes.eliminate_redundant_var_copy()
    assert p.get_name() == "EliminateRedundantVarCopy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
