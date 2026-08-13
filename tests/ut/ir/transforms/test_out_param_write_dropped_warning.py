# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Warning for a rebind that drops an Out/InOut parameter's write (#2352).

``out = pl.add(a, b)`` rebinds the Python name only: the parameter Var is
re-pointed at a freshly computed tensor and the caller's buffer is never
written. The program compiled, ran, and handed back uninitialised memory with
no diagnostic anywhere. It now raises an ``OutParamWriteDropped`` warning
naming the parameter and the ``out[:] = <expr>`` form that does write.

The check is data-flow based, not syntactic: a value that reaches the parameter
through a loop carry is a legitimate write-through and must stay silent.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import passes

_RULE = "OutParamWriteDropped"


def _warnings(program) -> list:
    """Run the pre-pipeline warning checks and keep only this rule's output."""
    checks = passes.DiagnosticCheckRegistry.get_warning_checks()
    diagnostics = passes.DiagnosticCheckRegistry.run_checks(
        checks, passes.DiagnosticPhase.PRE_PIPELINE, program
    )
    return [d for d in diagnostics if d.rule_name == _RULE]


class TestDroppedWriteIsReported:
    """A rebind whose value never flows through the parameter."""

    def test_operator_result_assigned_to_out_param(self):
        """The exact repro from #2352."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(a, b)
                return out

        found = _warnings(Prog)
        assert len(found) == 1
        message = found[0].message
        assert "'out'" in message
        assert "out[:] = <expr>" in message
        assert found[0].severity == passes.DiagnosticSeverity.Warning

    def test_fresh_local_assigned_to_out_param(self):
        """``out = tmp`` drops the write exactly like an inline operator result."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tmp = pl.add(a, b)
                    out = tmp
                return out

        assert len(_warnings(Prog)) == 1

    def test_inout_param_is_reported_too(self):
        """An InOut parameter drops the write the same way."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                acc: pl.InOut[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    acc = pl.add(a, b)
                return acc

        found = _warnings(Prog)
        assert len(found) == 1
        assert "InOut parameter 'acc'" in found[0].message

    def test_incore_kernel_is_reported(self):
        """The same drop inside an InCore kernel, not just a ``pl.at`` body."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                out = pl.add(a, b)
                return out

        assert len(_warnings(Prog)) == 1


class TestWriteThroughIsSilent:
    """A value that flows through the parameter really does write it."""

    def test_explicit_whole_tensor_assemble(self):
        """``out = pl.assemble(out, expr, [0, 0])`` names the parameter directly."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.assemble(out, pl.add(a, b), [0, 0])
                return out

        assert _warnings(Prog) == []

    def test_subscript_write(self):
        """The recommended ``out[:] = <expr>`` spelling."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[128, 128], pl.FP32],
                b: pl.Tensor[[128, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out[:] = pl.add(a, b)
                return out

        assert _warnings(Prog) == []

    def test_partial_subscript_write(self):
        """A sub-window write is a write, not a drop."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 128], pl.FP32],
                out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
            ) -> pl.Tensor[[128, 128], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out[0:64, 0:128] = a
                return out

        assert _warnings(Prog) == []

    def test_store_through_the_param(self):
        """``out = pl.store(tile, offset, out)`` writes through argument 2."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                inp: pl.Tensor[[1, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            ) -> pl.Tensor[[1, 64], pl.FP32]:
                local = pl.load(inp, [0, 0], [1, 64])
                out = pl.store(local, [0, 0], out)
                return out

        assert _warnings(Prog) == []


class TestLoopCarryAliasing:
    """The value reaches the parameter through a loop carry, never by name.

    This is the shape that made a purely syntactic "does the value mention the
    parameter?" test wrong: ``staged`` never names the parameter, yet it *is*
    the parameter threaded through ``init_values`` -> ``pl.store`` ->
    ``pl.yield_``.
    """

    def test_plain_tensor_carry_is_silent(self):
        """A plain Out tensor threaded through a carry and rebound."""

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (o,) in pl.range(4, init_values=(out,)):
                    t = pl.load(a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec)
                    o = pl.store(t, [i * 64, 0], o)
                    staged = pl.yield_(o)
                out = staged
                return out

        assert _warnings(Prog) == []

    def test_distributed_collective_rebind_is_silent(self):
        """Regression guard for the ring-allreduce InCore kernels.

        ``pld.tensor.allreduce`` reduces its window-bound target in place and
        returns it. The target reaches the call as ``staged_data``, a loop-carry
        alias of the ``data`` parameter, so the rebind is a real write-through.
        """
        sz = 64
        nr = 2

        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def reduce_step(
                self,
                inp: pl.Tensor[[1, sz], pl.FP32],
                out: pl.Out[pl.Tensor[[1, sz], pl.FP32]],
                data: pl.InOut[pld.DistributedTensor[[1, sz], pl.FP32]],
                signal: pl.InOut[pld.DistributedTensor[[2, nr], pl.INT32]],
            ) -> pl.Tensor[[1, sz], pl.FP32]:
                for col, (data_iter,) in pl.range(0, sz, 8, init_values=(data,)):
                    local = pl.load(inp, [0, col], [8, 8], valid_shape=[1, 8])
                    data_iter = pl.store(local, [0, col], data_iter)
                    staged_data: pld.DistributedTensor[[1, sz], pl.FP32] = pl.yield_(data_iter)

                data = pld.tensor.allreduce(staged_data, signal, op=pld.ReduceOp.Sum, mode="ring")

                for col2, (out_iter,) in pl.range(0, sz, 8, init_values=(out,)):
                    acc = pl.load(data, [0, col2], [8, 8], valid_shape=[1, 8])
                    out_iter = pl.store(acc, [0, col2], out_iter)
                    staged_out = pl.yield_(out_iter)
                return staged_out

        assert _warnings(Prog) == []


class TestMultipleOutParams:
    """Taint is per parameter — one written output must not cover a dropped one.

    All parameters are tainted in one shared pass over one shared reverse index,
    each carrying its own bit, so a cross-contaminated mask would silently
    suppress the warning on the dropped output.
    """

    def test_one_written_one_dropped(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                written: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                dropped: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    written = pl.assemble(written, pl.add(a, b), [0, 0])
                    dropped = pl.mul(a, b)  # noqa: F841 — the dropped write under test
                return written

        found = _warnings(Prog)
        assert len(found) == 1
        assert "'dropped'" in found[0].message

    def test_both_dropped_are_both_reported(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                first: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
                second: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    first = pl.add(a, b)
                    second = pl.mul(a, b)  # noqa: F841 — the second dropped write under test
                return first

        found = _warnings(Prog)
        assert len(found) == 2
        # Reported in statement order, so the report is stable run to run.
        assert "'first'" in found[0].message
        assert "'second'" in found[1].message


class TestConservativeSuppression:
    """Documented false negative: a value that only *reads* the parameter.

    ``out = pl.add(out, b)`` drops the write just like the reported cases, but
    it is indistinguishable from a genuine write-through without per-op write
    semantics the registry does not record. Suppressing here is the safe
    direction — a false warning on correct collective code is worse.
    """

    def test_read_only_use_of_the_param_is_not_reported(self):
        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    out = pl.add(out, b)
                return out

        assert _warnings(Prog) == []


class TestScope:
    """Parameters and spellings the check deliberately leaves alone."""

    def test_in_param_is_not_reported(self):
        """A plain In parameter has no caller buffer to write."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    a = pl.add(a, b)
                return a

        assert _warnings(Prog) == []

    def test_local_rebinding_is_not_reported(self):
        """Rebinding a local is ordinary SSA, not a dropped write."""

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                b: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    tmp = pl.add(a, b)
                    tmp = pl.mul(tmp, b)
                    out[:] = tmp
                return out

        assert _warnings(Prog) == []


class TestDiagnosticWiring:
    """The check is registered as a pre-pipeline warning and can be disabled."""

    def test_registered_as_a_warning_check(self):
        checks = passes.DiagnosticCheckRegistry.get_warning_checks()
        assert checks.contains(passes.DiagnosticCheck.OutParamWriteDropped)

    def test_can_be_disabled(self):
        """``disabled_diagnostics`` removes it from the effective set."""
        disabled = passes.DiagnosticCheckSet()
        disabled.insert(passes.DiagnosticCheck.OutParamWriteDropped)
        effective = passes.DiagnosticCheckRegistry.get_all_checks().difference(disabled)
        assert not effective.contains(passes.DiagnosticCheck.OutParamWriteDropped)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
