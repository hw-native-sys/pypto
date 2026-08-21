# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ParamDirectionsSound: a parameter declared In must not be written by its body.

This is the check direction inference never had. Every pass that derives
directions builds a set of "which argument does this call write", and an
operator missing from that set reads as a pure consumer: the write disappears,
the parameter stays ``In``, no RAW edge is emitted against it, and the failure
surfaces on device as a race or a scheduler deadlock rather than at compile
time. ``pld.system.notify`` shipped that way; ``tile.mscatter`` was still in that
state when this check was written.

The verifier is run directly here rather than through the pipeline: the pipeline
*upgrades* a written ``In`` parameter, so a program that reaches the check
already sound proves nothing about the check. Calling it on unlowered IR is what
exercises the "declared In, body writes it" state it exists to reject.

**Known boundary.** A write reaches its parameter through ``BufferRootCollector``,
which treats ``tensor.slice`` as a fresh root rather than an alias of its source
(``src/ir/transforms/utils/buffer_root_collector.cpp``). A store into a slice of
a parameter is therefore *not* reported. That is under-reporting, which restores
today's behaviour for that shape; over-reporting would reject programs that
compile correctly. Unifying the three alias models in the tree is separate work.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.pypto_core import ir as _ir
from pypto.pypto_core import passes


def _verify(prog):
    """Diagnostics from the ParamDirectionsSound verifier alone."""
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.ParamDirectionsSound)
    return passes.PropertyVerifierRegistry.verify(props, prog)


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


def test_property_is_registered():
    """The property must be reachable by name, or the pipeline silently skips it."""
    assert _ir is not None
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.ParamDirectionsSound)
    assert props.contains(passes.IRProperty.ParamDirectionsSound)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
