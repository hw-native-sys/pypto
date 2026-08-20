# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""InParamWritten: warn when a parameter declared In is written by its body.

The check reads the two declarations direction inference reads — an operator's
registry effects and a callee's own ``param_directions_`` — and reports where
they contradict a parameter's own ``In``. It is a consistency check over the
*declared* write semantics, not an independent discovery of them: an operator
that never declared an effect is invisible to it, because ``CallWriteTargets``
returns nothing and the check is reading the very declaration that is absent.
``ValidateArgEffects`` narrows that gap but does not close it: it fires only on
an operator that declares a reuse contract without classifying the reused
argument, or a write channel without a write. An operator with neither — the
shape ``pld.system.notify`` had in #2391 — passes both.

What this check does buy: once an operator carries a declaration, it stops a
caller from re-declaring its destination ``In``, and it covers every
cross-function call, where the callee's signature is the declaration.

The check is run directly here rather than through the pipeline: the pipeline
*upgrades* a written ``In`` parameter, so a program that reaches it already
consistent proves nothing. Calling it on unlowered IR is what exercises the
"declared In, body writes it" state it exists to report.

**Best-effort, not a property.** This is a warning and nothing more. It runs
`PostPipeline`, which is the earliest point after ``DeriveCallDirections``
(pass 37) — and ``InitMemRef`` (pass 31) invalidates ``SSAForm`` with nothing
re-establishing it, so the IR it sees is not in SSA form and no pipeline
position satisfies both. The buffer lineage has no merging at a join, which is
exact only when each name has one definition, so it can both miss a write and
attribute one to a buffer it reaches on no path.
``test_branch_local_view_does_not_leak_past_the_join`` pins that second case as a
strict ``xfail`` rather than asserting it away.

**Views are followed.** ``BufferRootCollector`` maps ``tensor.slice`` to a fresh
root and skips the tile views entirely, so the verifier resolves the chain itself
from two shared declarations: ``ResultAliasedArgIndex`` (the operator returns the
argument it updated) and ``op_predicates::IsBufferAliasingViewOp``
(``OutputMemoryInheritsInput() && IsInplaceSafe()`` — the zero-copy views, which
update nothing and so declare no reuse contract). ``tile.transpose`` is excluded
by that predicate: it permutes into a fresh buffer, so its output is not an alias
of its input.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes


def _verify(prog):
    """Diagnostics from the InParamWritten check alone.

    Run through the diagnostic registry, which is the only way it is reachable:
    it is a warning, not an ``IRProperty``.
    """
    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.InParamWritten)
    return passes.DiagnosticCheckRegistry.run_checks(checks, passes.DiagnosticPhase.POST_PIPELINE, prog)


def _messages(prog):
    return [d.message for d in _verify(prog)]


class TestWrittenInParamIsRejected:
    def test_store_into_in_param(self):
        """A ``tile.store`` destination declared ``In`` is rejected, and the
        message names the parameter and the operator that writes it."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
                t = pl.load(x, [0, 0], [16, 16])
                pl.store(t, [0, 0], out)

        messages = _messages(Prog)
        assert len(messages) == 1
        assert "'out'" in messages[0]
        assert "tile.store" in messages[0]
        assert "declared In" in messages[0]

    def test_scatter_into_in_param(self):
        """The operator whose missing declaration this whole check exists for:
        ``tile.mscatter`` writes a GM tensor and was in none of the write
        tables, so its destination silently kept ``In``."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                src: pl.Tensor[[16, 16], pl.FP32],
                idx: pl.Tensor[[16, 16], pl.INT32],
                out: pl.Tensor[[16, 16], pl.FP32],
            ):
                s = pl.load(src, [0, 0], [16, 16])
                i = pl.load(idx, [0, 0], [16, 16])
                pl.mscatter(s, i, out)

        messages = _messages(Prog)
        assert len(messages) == 1
        assert "'out'" in messages[0]
        assert "tile.mscatter" in messages[0]

    def test_notify_into_in_signal(self):
        """``pld.system.notify`` deposits into the peer's slot of its signal.
        Reading that signal as an input is what dropped the RAW edge a waiter
        needs and deadlocked the communication card."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                signal: pld.DistributedTensor[[2, 1], pl.INT32],
                peer: pl.Scalar[pl.INT32],
            ):
                pld.system.notify(signal, peer=peer, offsets=[0, 0], value=1, op=pld.NotifyOp.Set)

        messages = _messages(Prog)
        assert len(messages) == 1
        assert "'signal'" in messages[0]

    def test_cross_function_out_arg(self):
        """A caller passing its own ``In`` parameter into a callee's ``Out``
        slot writes it just as surely as a builtin would."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def callee(self, x: pl.Tensor[[16, 16], pl.FP32], dst: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
                t = pl.load(x, [0, 0], [16, 16])
                pl.store(t, [0, 0], dst)

            @pl.function(type=pl.FunctionType.InCore)
            def caller(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
                self.callee(x, out)

        messages = _messages(Prog)
        assert any("'out'" in m and "'caller'" in m for m in messages)

    def test_one_diagnostic_per_parameter(self):
        """A loop writing the same parameter every iteration is one bug, not one
        per write site."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Tensor[[16, 16], pl.FP32]):
                t = pl.load(x, [0, 0], [16, 16])
                pl.store(t, [0, 0], out)
                pl.store(t, [0, 0], out)

        assert len(_messages(Prog)) == 1


class TestSoundDeclarationsPass:
    """Nothing is reported when the declaration already covers the write.

    These are the over-triggering guards: a check that rejects correct programs
    is worse than the silence it replaces, since it blocks compilations that
    would have worked.
    """

    def test_out_param_is_accepted(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
                t = pl.load(x, [0, 0], [16, 16])
                pl.store(t, [0, 0], out)

        assert _messages(Prog) == []

    def test_inout_param_is_accepted(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, out: pl.InOut[pl.Tensor[[16, 16], pl.FP32]]):
                t = pl.load(out, [0, 0], [16, 16])
                pl.store(pl.tile.add(t, t), [0, 0], out)

        assert _messages(Prog) == []

    def test_read_only_param_is_accepted(self):
        """A parameter only ever loaded from is exactly what ``In`` means."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]):
                t = pl.load(x, [0, 0], [16, 16])
                pl.store(t, [0, 0], out)

        assert all("'x'" not in m for m in _messages(Prog))

    def test_wait_does_not_write_its_signal(self):
        """``pld.system.wait`` polls a signal it never writes — declared
        read-only on the registry, and the check must respect that rather than
        assuming every side-effect operator writes."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, signal: pld.DistributedTensor[[2, 1], pl.INT32]):
                pld.system.wait(signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

        assert _messages(Prog) == []

    def test_scalar_param_is_not_a_buffer(self):
        """A scalar is passed by value; its direction makes no aliasing claim,
        so it is never a candidate."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                n: pl.Scalar[pl.INT32],
                out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ):
                t = pl.load(out, [0, 0], [16, 16])
                pl.store(t, [0, 0], out)

        assert all("'n'" not in m for m in _messages(Prog))


def test_written_tile_param_is_reported():
    """A ``Tile`` parameter is a buffer too, and can be declared ``Out``/``InOut``.

    The filter that selects candidate parameters once used ``AsTensorTypeLike``,
    which matches ``TensorType`` and ``DistributedTensorType`` only — its own
    documentation says to use ``As<ShapedType>()`` for the wider union. A whole
    parameter kind was therefore invisible to a check whose entire job is to
    notice a written parameter declared ``In``.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            acc: pl.Tile[[16, 128], pl.FP32],
            src: pl.Tile[[16, 128], pl.FP32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            # `tile.assemble` declares ReadWrite on argument 0.
            acc = pl.tile.assemble(acc, src, [0, 0])
            return acc

    messages = _messages(Prog)
    assert any("'acc'" in m for m in messages), messages
    assert all("'src'" not in m for m in messages), messages


def test_written_param_reaching_a_submit_out_slot_is_reported():
    """A task launch writes its callee's ``Out`` slots just as a call does.

    The base visitor does not forward ``Submit`` to the ``Call`` handler
    (`.claude/rules/pass-submit-awareness.md`), so a ``pl.submit`` inside
    ``pl.manual_scope`` needs its own hook or every launch is unchecked.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def writer(self, dst: pl.Out[pl.Tensor[[16, 128], pl.FP32]]) -> pl.Tensor[[16, 128], pl.FP32]:
            dst = pl.assemble(dst, pl.full([16, 128], dtype=pl.FP32, value=0.0), [0, 0])
            return dst

        @pl.function
        def main(self, buf: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            with pl.manual_scope():
                res, tid = pl.submit(self.writer, buf)
            return res

    messages = _messages(Prog)
    assert any("'buf'" in m for m in messages), messages


def test_write_through_a_zero_copy_view_is_reported():
    """A write through a view of a parameter is a write to the parameter.

    ``BufferRootCollector`` records nothing for a builtin view — it maps
    ``tensor.slice`` to a *fresh* root and skips the tile views entirely — so a
    store through one used to reach no parameter at all. The verifier now
    follows the chain itself, using the registry's own
    ``OutputMemoryInheritsInput()`` view set.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            acc: pl.Tile[[16, 128], pl.FP32],
            src: pl.Tile[[8, 128], pl.FP32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            view: pl.Tile[[8, 128], pl.FP32] = pl.tile.slice(acc, [8, 128], [0, 0])
            view = pl.tile.assemble(view, src, [0, 0])
            return acc

    messages = _messages(Prog)
    assert any("'acc'" in m for m in messages), messages


def test_transpose_output_is_not_an_alias_of_its_input():
    """``tile.transpose`` inherits the memory *space*, never the buffer.

    ``pto.ttrans`` is not in-place safe, so ``InitMemRef`` gives the transpose
    output a fresh buffer. Treating every inherit-input op as an alias would
    blame ``src`` for a write into a buffer of the transpose's own.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src: pl.Tile[[16, 128], pl.FP32],
            patch: pl.Tile[[128, 16], pl.FP32],
        ) -> pl.Tile[[128, 16], pl.FP32]:
            t: pl.Tile[[128, 16], pl.FP32] = pl.tile.transpose(src, 0, 1)
            t = pl.tile.assemble(t, patch, [0, 0])
            return t

    messages = _messages(Prog)
    assert all("'src'" not in m for m in messages), messages


def test_write_through_a_tensor_slice_view_is_reported():
    """``tensor.slice`` is a buffer-aliasing view too, despite the collector.

    ``BufferRootCollector`` deliberately maps it to a *fresh* root, so this case
    depends entirely on the verifier resolving the lineage itself.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            buf: pl.Tensor[[16, 128], pl.FP32],
            src: pl.Tensor[[8, 128], pl.FP32],
        ) -> pl.Tensor[[16, 128], pl.FP32]:
            view: pl.Tensor[[8, 128], pl.FP32] = pl.slice(buf, [8, 128], [0, 0])
            view = pl.assemble(view, src, [0, 0])
            return buf

    messages = _messages(Prog)
    assert any("'buf'" in m for m in messages), messages


def test_rebinding_a_view_name_drops_the_stale_lineage():
    """A rebind must not leave the old buffer on the hook.

    These tests run on pre-SSA IR, where a rebind reuses the same ``Var``. After
    ``t`` is re-pointed at a transpose — a fresh buffer — a write through ``t``
    is a write to that buffer, not to ``src1`` whose view ``t`` used to be.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            src1: pl.Tile[[16, 128], pl.FP32],
            src2: pl.Tile[[128, 16], pl.FP32],
            patch: pl.Tile[[16, 128], pl.FP32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            t: pl.Tile[[16, 128], pl.FP32] = pl.tile.slice(src1, [16, 128], [0, 0])
            t = pl.tile.transpose(src2, 0, 1)
            t = pl.tile.assemble(t, patch, [0, 0])
            return t

    messages = _messages(Prog)
    assert all("'src1'" not in m for m in messages), messages


def test_may_write_through_a_branch_is_reported():
    """A write on *some* path is a write.

    On the ``cond > 0`` path ``v`` is a view of ``bufA`` and the assemble writes
    it, so naming ``bufA`` is the correct may-write answer — not, as an earlier
    version of this test claimed, a join false positive. The lineage surviving
    the branch is what makes the union over paths conservative in the right
    direction here.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            bufA: pl.Tile[[16, 128], pl.FP32],
            src: pl.Tile[[128, 16], pl.FP32],
            patch: pl.Tile[[16, 128], pl.FP32],
            cond: pl.Scalar[pl.INT32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            v: pl.Tile[[16, 128], pl.FP32] = pl.tile.transpose(src, 0, 1)
            if cond > 0:
                v = pl.tile.slice(bufA, [16, 128], [0, 0])
            v = pl.tile.assemble(v, patch, [0, 0])
            return v

    messages = _messages(Prog)
    assert any("'bufA'" in m for m in messages), messages


@pytest.mark.xfail(
    reason="known limitation: BufferRootCollector scans the whole body up front, so a "
    "rebound name carries one final mapping that is applied to earlier writes too. "
    "Reports the buffer bound last and misses the one actually written.",
    strict=True,
)
def test_a_rebound_name_attributes_the_write_to_the_right_buffer():
    """The real misattribution, in both directions at once.

    ``t`` names ``buf1`` when the assemble writes it and ``buf2`` only
    afterwards, but the collector's single final mapping says ``t -> buf2``. So
    ``buf2`` is reported although nothing writes it, and ``buf1`` is missed
    although it is written. This is the shape that needs SSA or a per-access
    environment; ``strict=True`` so fixing it forces the marker off.
    """

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            buf1: pl.Tile[[16, 128], pl.FP32],
            buf2: pl.Tile[[16, 128], pl.FP32],
            patch: pl.Tile[[16, 128], pl.FP32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            t: pl.Tile[[16, 128], pl.FP32] = buf1
            t = pl.tile.assemble(t, patch, [0, 0])
            t = buf2
            return t

    reported = {m.split("'")[1] for m in _messages(Prog)}
    assert reported == {"buf1"}, reported


def test_write_inside_a_branch_is_reported():
    """Control flow does not hide a write, only the lineage across a phi."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            acc: pl.Tile[[16, 128], pl.FP32],
            patch: pl.Tile[[16, 128], pl.FP32],
            cond: pl.Scalar[pl.INT32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            if cond > 0:
                acc = pl.tile.assemble(acc, patch, [0, 0])
            return acc

    messages = _messages(Prog)
    assert any("'acc'" in m for m in messages), messages


def test_write_inside_a_loop_is_reported():
    """One diagnostic for a loop body that writes an ``In`` parameter."""

    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            acc: pl.Tile[[16, 128], pl.FP32],
            patch: pl.Tile[[16, 128], pl.FP32],
        ) -> pl.Tile[[16, 128], pl.FP32]:
            for _ in pl.range(4):
                acc = pl.tile.assemble(acc, patch, [0, 0])
            return acc

    messages = _messages(Prog)
    assert len([m for m in messages if "'acc'" in m]) == 1, messages


def test_check_is_registered():
    """The check must be reachable by name, or the pipeline silently skips it."""
    assert _ir is not None
    checks = passes.DiagnosticCheckSet()
    checks.insert(passes.DiagnosticCheck.InParamWritten)
    assert checks.contains(passes.DiagnosticCheck.InParamWritten)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
