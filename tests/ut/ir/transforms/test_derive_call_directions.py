# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DeriveCallDirections pass and its CallDirectionsResolved verifier."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.pypto_core import passes as _core_passes


def _verify_call_directions(program):
    """Run the CallDirectionsResolved property verifier on *program*.

    Replaces the now-deleted ``passes.verify_call_directions()`` pass: the
    integrity of ``Call.attrs['arg_directions']`` is now a verifiable IR property
    (``IRProperty.CallDirectionsResolved``) auto-checked by the pipeline.
    """
    props = _core_passes.IRPropertySet()
    props.insert(_core_passes.IRProperty.CallDirectionsResolved)
    _core_passes.PropertyVerifierRegistry.verify_or_throw(props, program)


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Override the global roundtrip-verification fixture for this module.

    The python_printer/parser do not yet emit Call.arg_directions, so the
    print -> parse -> structural_equal roundtrip fails immediately after
    DeriveCallDirections. We fall back to the lighter BEFORE_AND_AFTER
    property-verification mode here.
    """
    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _UserCallCollector(ir.IRVisitor):
    """Collect every non-builtin Call from a Program for inspection."""

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def visit_call(self, op):
        name = op.op.name
        if not (name.startswith("tile.") or name.startswith("tensor.") or name.startswith("system.")):
            self.calls.append(op)
        super().visit_call(op)


def _user_calls(program):
    collector = _UserCallCollector()
    collector.visit_program(program)
    return collector.calls


def _dirs(call):
    return [d for d in call.arg_directions]


# ---------------------------------------------------------------------------
# Derive pass: per-direction matrix
# ---------------------------------------------------------------------------


class TestDeriveDirectionMatrix:
    """One test per cell of the (callee_dir, arg_origin) mapping table."""

    def test_in_param_tensor_to_input(self):
        """Callee In + tensor argument → Input."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                # `x` is a function param: callee In keeps Input.
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
                return r

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        # Position 0 is callee In + tensor → Input.
        assert _dirs(calls[0])[0] == ir.ArgDirection.Input

    def test_inout_param_tensor_to_inout(self):
        """Callee InOut + tensor argument → InOut."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                t2: pl.Tile[[64], pl.FP32] = pl.tile.add(t, t)
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t2, [0], x)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x)
                return r

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.InOut]

    def test_out_param_external_buffer_to_output_existing(self):
        """Callee Out + arg rooted at a function param → OutputExisting."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
                return r

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]

    def test_out_param_local_buffer_kept_output_existing(self):
        """Callee Out + single-write locally allocated buffer → OutputExisting.

        A buffer that is allocated locally and written to by exactly one Call at
        top level (no sequential ancestor, no prior writer-unit in the same
        scope) does not need the WAW chaining that ``InOut`` provides; keeping
        it as ``OutputExisting`` lets the runtime treat the slot as an ordinary
        output and avoids the spurious dependency that would otherwise serialize
        the task with subsequent siblings.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, local)
                return r

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]

    def test_two_calls_top_level_second_promoted(self):
        """Two consecutive top-level calls writing the same local root.

        First writer keeps ``OutputExisting`` (no prior writes); the second
        writer hits R-prior and is promoted to ``InOut`` so the runtime can
        chain WAW dependencies on the shared buffer.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                local = self.kernel(x, local)
                local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 2
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        assert _dirs(calls[1]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_out_local_in_parallel_keeps_output_existing(self):
        """Single ``pl.parallel`` writer of a local buffer → ``OutputExisting``.

        Regression test for issue #1086: tiled writes inside a ``pl.parallel``
        loop should not be promoted to ``InOut`` just because they happen
        inside a loop, because doing so injects a spurious dependency that
        serializes otherwise independent iterations.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]

    def test_two_parallel_loops_promote_only_second(self):
        """Two consecutive ``pl.parallel`` loops writing the same root.

        The first loop is the only writer-unit at its scope and stays
        ``OutputExisting``; the second loop hits R-prior and is promoted to
        ``InOut`` so the cross-loop WAW dependency is preserved.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    local = self.kernel(x, local)
                for _j in pl.parallel(4):
                    local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 2
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        assert _dirs(calls[1]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_seq_inside_parallel_keeps_inout(self):
        """``pl.range`` (sequential) inside ``pl.parallel`` triggers R-seq.

        Even if the inner sequential loop is the only writer-unit, the
        sequential ancestor forces ``InOut`` so cross-iteration WAW chains in
        the inner loop body are preserved.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for _i in pl.parallel(4):
                    for _j in pl.range(4):
                        local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_parallel_inside_seq_keeps_inout(self):
        """``pl.parallel`` inside ``pl.range`` still triggers R-seq.

        The outer sequential loop is enough for R-seq, regardless of the kind
        of inner loops it contains.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                for _i in pl.range(4):
                    for _j in pl.parallel(4):
                        local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_top_level_call_then_parallel_promoted(self):
        """Top-level writer followed by ``pl.parallel`` writer hits R-prior.

        Mirror of the ``k2(local) for _ in pl.parallel: k1(local)`` scenario:
        the first call is the sole writer-unit, the parallel loop sees a
        prior writer-unit at sibling scope and is therefore promoted.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                local = self.kernel(x, local)
                for _i in pl.parallel(4):
                    local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 2
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        assert _dirs(calls[1]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_while_keeps_inout(self):
        """``while`` loop body triggers R-seq (sequential writer-unit).

        ``WhileStmt`` is treated like a sequential for loop: the body may run
        any number of iterations, so cross-iteration WAW dependencies must be
        preserved by promoting Out → InOut.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                i: pl.Scalar[pl.INDEX] = 0
                while i < n:
                    local = self.kernel(x, local)
                    i = i + 1
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_if_first_writer_keeps_output_existing(self):
        """First writer inside an ``if`` branch is the only writer-unit.

        With no prior writer and no sequential ancestor, the call inside the
        branch keeps ``OutputExisting``. Each branch is analyzed against an
        independent ``seen_roots`` snapshot from the enclosing scope.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                flag: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                if flag:
                    local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]

    def test_if_after_top_level_writer_promoted(self):
        """``if`` branch following a top-level writer hits R-prior.

        The outer scope's prior-writer set already contains the local root
        when the ``if`` is entered, so the branch's snapshot starts with the
        root in ``seen``; the call inside is no longer the first writer.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                flag: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                local = self.kernel(x, local)
                if flag:
                    local = self.kernel(x, local)
                return local

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 2
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        assert _dirs(calls[1]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_out_param_external_buffer_in_seq_loop_promoted(self):
        """R-seq on external root: writes inside ``pl.range`` promote to ``InOut``.

        ``dst`` is rooted at the enclosing ``main`` parameter (not locally
        allocated), but the sequential ancestor still requires WAW chaining
        across iterations — same as for local roots.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for _i in pl.range(4):
                    dst = self.kernel(x, dst)
                return dst

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_out_param_variable_offset_store_in_seq_loop_promoted(self):
        """R-seq: a callee Out written via a parameter-dependent ``tile.store``
        offset is still promoted to ``InOut`` inside a sequential loop.

        An earlier "disjoint variable-offset store" exception kept such a call
        as ``OutputExisting``, assuming a parameter-keyed offset implies the
        per-iteration writes are disjoint. That exception was unsound — it never
        checked offset stride vs. tile extent, offset injectivity, or other
        write paths to the same buffer — so it was removed. R-seq now promotes
        unconditionally; any genuinely-disjoint optimization must be
        reintroduced behind a sound dependence analysis.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256], pl.FP32]],
            ) -> pl.Tensor[[256], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[256], pl.FP32] = pl.store(t, [offset], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[256], pl.FP32],
            ) -> pl.Tensor[[256], pl.FP32]:
                for _i in pl.range(4):
                    dst = self.kernel(x, _i * 64, dst)
                return dst

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [
            ir.ArgDirection.Input,
            ir.ArgDirection.Scalar,
            ir.ArgDirection.InOut,
        ]

    def test_out_param_external_buffer_two_writes_second_promoted(self):
        """R-prior on external root: a prior writer-unit promotes the second to ``InOut``.

        Two consecutive top-level calls writing into the same enclosing-param
        destination. The first stays ``OutputExisting`` (no prior writer); the
        second sees the first as a prior writer and is promoted, mirroring the
        ``test_two_calls_top_level_second_promoted`` semantics for local roots.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                dst = self.kernel(x, dst)
                dst = self.kernel(x, dst)
                return dst

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 2
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]
        assert _dirs(calls[1]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_out_param_enclosing_inout_declaration_promoted(self):
        """R-enclosing: explicit ``pl.InOut`` on the enclosing param promotes to ``InOut``.

        Even when neither R-seq nor R-prior fire (single call, no sequential
        ancestor, first writer in scope), an explicit ``pl.InOut`` declaration
        on the enclosing function's parameter must be honored — the function
        effectively reads the prior caller-supplied value and writes back.

        Regression test for the KV-cache scenario where ``pl.InOut`` declared
        at top level was being collapsed to ``add_output`` in cpp codegen.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
                return r

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_out_param_enclosing_inout_in_parallel_loop_promoted(self):
        """R-enclosing wins even when wrapped in ``pl.parallel``.

        Mirrors the qwen3 KV-cache call site: the kernel is invoked once
        inside an outer ``pl.parallel`` loop, the buffer root traces back
        through the loop's iter binding to a ``pl.InOut`` parameter on the
        enclosing function. Neither R-seq nor R-prior fire here, so this
        case is the canonical motivator for R-enclosing.
        """

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for _i in pl.parallel(4):
                    dst = self.kernel(x, dst)
                return dst

        out = passes.derive_call_directions()(Prog)
        calls = [c for c in _user_calls(out) if c.op.name == "kernel"]
        assert len(calls) == 1
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.InOut]

    def test_builtin_calls_left_untouched(self):
        """tensor.create / tile.* are builtin and keep arg_directions empty."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, local)
                return r

        out = passes.derive_call_directions()(Prog)

        class _BuiltinCallChecker(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.builtin_calls: list = []

            def visit_call(self, op):
                if op.op.name.startswith(("tile.", "tensor.", "system.")):
                    self.builtin_calls.append(op)
                super().visit_call(op)

        checker = _BuiltinCallChecker()
        checker.visit_program(out)
        # Every builtin keeps the legacy empty arg_directions.
        assert len(checker.builtin_calls) > 0
        assert all(list(c.arg_directions) == [] for c in checker.builtin_calls)


# ---------------------------------------------------------------------------
# Derive pass: idempotency and stability
# ---------------------------------------------------------------------------


class TestDeriveIdempotent:
    """Running derive twice produces structurally identical IR."""

    def test_idempotent(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, local)
                return r

        once = passes.derive_call_directions()(Prog)
        twice = passes.derive_call_directions()(once)
        ir.assert_structural_equal(once, twice)


# ---------------------------------------------------------------------------
# Derive pass: explicit call-site directions are preserved
# ---------------------------------------------------------------------------


class TestDerivePreservesExplicit:
    """Pre-populated Call.attrs['arg_directions'] is treated as authoritative."""

    def test_explicit_directions_not_overwritten(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
                return r

        # Pre-populate the Out-param slot with `Output` (runtime-allocation
        # semantics). The derive pass would otherwise emit `OutputExisting`
        # for an external/param-rooted destination, so this checks that the
        # explicit call-site choice survives instead of being overwritten.
        explicit = [ir.ArgDirection.Input, ir.ArgDirection.Output]
        prog = _RewriteUserCall(explicit).run(Prog)

        before = _user_calls(prog)
        assert len(before) == 1
        assert _dirs(before[0]) == explicit

        derived = passes.derive_call_directions()(prog)

        after = _user_calls(derived)
        assert len(after) == 1
        assert _dirs(after[0]) == explicit


# ---------------------------------------------------------------------------
# Verify pass: positive case
# ---------------------------------------------------------------------------


class TestVerifyPositive:
    """Verify pass accepts the output of derive."""

    def test_verify_succeeds_after_derive(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, local)
                return r

        out = passes.derive_call_directions()(Prog)
        # Should not raise.
        _verify_call_directions(out)


# ---------------------------------------------------------------------------
# Verify pass: negative cases (manually mutated IR)
# ---------------------------------------------------------------------------


class TestVerifyNegative:
    """Verify pass rejects ill-formed Call.attrs['arg_directions'] assignments."""

    @staticmethod
    def _build_program(call_dirs):
        """Build a tiny program whose single user call uses *call_dirs*."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, dst)
                return r

        return _RewriteUserCall(call_dirs).run(Prog)

    def test_input_with_output_rejected(self):
        # Position 0 is callee In; using Output there must fail.
        prog = self._build_program([ir.ArgDirection.Output, ir.ArgDirection.OutputExisting])
        with pytest.raises(Exception, match=r"(?i)arg_direction|CallDirectionsResolved"):  # noqa: PT011
            _verify_call_directions(prog)

    def test_out_with_input_rejected(self):
        # Position 1 is callee Out; using Input there must fail.
        prog = self._build_program([ir.ArgDirection.Input, ir.ArgDirection.Input])
        with pytest.raises(Exception, match=r"(?i)arg_direction|CallDirectionsResolved"):  # noqa: PT011
            _verify_call_directions(prog)


# ---------------------------------------------------------------------------
# Helper: rewrite the user call with a custom arg_directions vector.
# ---------------------------------------------------------------------------


class _RewriteUserCall(ir.IRMutator):
    """Replace every non-builtin Call's arg_directions with *new_dirs*."""

    def __init__(self, new_dirs):
        super().__init__()
        self._new_dirs = list(new_dirs)

    def visit_call(self, op):
        name = op.op.name
        if name.startswith(("tile.", "tensor.", "system.")):
            return super().visit_call(op)
        new_args = [self.visit_expr(a) for a in op.args]
        attrs = {"arg_directions": list(self._new_dirs)}
        return ir.Call(op.op, new_args, op.kwargs, attrs, op.type, op.span)

    def run(self, program):
        return self.visit_program(program)


# ---------------------------------------------------------------------------
# pl.no_dep override
# ---------------------------------------------------------------------------


class TestNoDepOverride:
    """``pl.no_dep(arg)`` at a kernel call site sets ArgDirection.NoDep at that slot."""

    @pl.program
    class _Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            b: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            return c

        @pl.function(type=pl.FunctionType.Orchestration)
        def orch(
            self,
            a: pl.Tensor[[16, 16], pl.FP32],
            shared: pl.Tensor[[16, 16], pl.FP32],
            c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        ):
            c = self.kernel(a, pl.no_dep(shared), c)
            return c

    def test_no_dep_at_marked_slot(self):
        new_prog = passes.derive_call_directions()(self._Prog)
        calls = _user_calls(new_prog)
        assert len(calls) == 1
        # 0=a (Input), 1=shared marked NoDep, 2=c (OutputExisting first writer at top level).
        assert _dirs(calls[0]) == [
            ir.ArgDirection.Input,
            ir.ArgDirection.NoDep,
            ir.ArgDirection.OutputExisting,
        ]

    def test_no_no_dep_keeps_input(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ):
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                shared: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ):
                c = self.kernel(a, shared, c)
                return c

        new_prog = passes.derive_call_directions()(P)
        calls = _user_calls(new_prog)
        assert len(calls) == 1
        assert _dirs(calls[0]) == [
            ir.ArgDirection.Input,
            ir.ArgDirection.Input,
            ir.ArgDirection.OutputExisting,
        ]

    def test_multiple_no_dep_slots(self):
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ):
                return c

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
                c: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ):
                c = self.kernel(pl.no_dep(a), pl.no_dep(b), c)
                return c

        new_prog = passes.derive_call_directions()(P)
        calls = _user_calls(new_prog)
        assert _dirs(calls[0]) == [
            ir.ArgDirection.NoDep,
            ir.ArgDirection.NoDep,
            ir.ArgDirection.OutputExisting,
        ]

    def test_no_dep_on_inout_param_rejected(self):
        # The verifier forbids NoDep on Out/InOut params: NoDep is a read-only
        # opt-out that would suppress producer registration if applied to a
        # writer. ``derive_call_directions()`` runs the property verifier as
        # post-condition, so the override surfaces as a build-time failure.
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ):
                return b

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
            ):
                b = self.kernel(a, pl.no_dep(b))
                return b

        with pytest.raises(Exception, match="NoDep|InOut"):  # noqa: PT011
            passes.derive_call_directions()(P)
