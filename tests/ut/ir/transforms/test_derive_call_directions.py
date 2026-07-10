# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DeriveCallDirections pass and its CallDirectionsResolved verifier.

Transform tests follow the project-standard Before/After/Expected pattern: the
``Before`` program is run through ``passes.derive_call_directions()`` to produce
``After``, and the result is compared with ``Expected`` via
``ir.assert_structural_equal``. The derived ``Call.attrs['arg_directions']``
vector is faithfully emitted by the python printer (as
``attrs={"arg_directions": [pl.adir.<name>, ...]}``) and round-trips through the
parser, so ``Expected`` can spell out the derived directions directly. The
kernel bodies in ``Expected`` are written in their post-lowering form
(``pl.tile.load`` / ``pl.tensor.create`` / explicit ``level`` and ``role``)
because the DSL frontend lowers ``pl.load`` / ``pl.create_tensor`` and infers
the function ``level`` / ``role`` before the pass runs.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir, passes
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


# ---------------------------------------------------------------------------
# Derive pass: per-direction matrix
# ---------------------------------------------------------------------------


class TestDeriveDirectionMatrix:
    """One test per cell of the (callee_dir, arg_origin) mapping table."""

    def test_in_param_tensor_to_input(self):
        """Callee In + tensor argument → Input.

        Position 0 is callee In + tensor; ``x`` is a ``main`` parameter, so the
        callee In keeps ``Input``.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_inout_param_tensor_to_inout(self):
        """Callee InOut + tensor argument → InOut."""

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                t2 = pl.tile.add(t, t)
                ret = pl.tile.store(t2, [0], x)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, attrs={"arg_directions": [pl.adir.inout]})
                return r

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_out_param_external_buffer_to_output_existing(self):
        """Callee Out + arg rooted at a function param → OutputExisting."""

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

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
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                r = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_calls_top_level_second_promoted(self):
        """Two consecutive top-level calls writing the same local root.

        First writer keeps ``OutputExisting`` (no prior writes); the second
        writer hits R-prior and is promoted to ``InOut`` so the runtime can
        chain WAW dependencies on the shared buffer.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                local = self.kernel(
                    x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
                )
                local = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_out_local_in_parallel_keeps_output_existing(self):
        """Single ``pl.parallel`` writer of a local buffer → ``OutputExisting``.

        Regression test for issue #1086: tiled writes inside a ``pl.parallel``
        loop should not be promoted to ``InOut`` just because they happen
        inside a loop, because doing so injects a spurious dependency that
        serializes otherwise independent iterations.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for _i in pl.parallel(4):
                    local = self.kernel(
                        x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
                    )
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_two_parallel_loops_promote_only_second(self):
        """Two consecutive ``pl.parallel`` loops writing the same root.

        The first loop is the only writer-unit at its scope and stays
        ``OutputExisting``; the second loop hits R-prior and is promoted to
        ``InOut`` so the cross-loop WAW dependency is preserved.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for _i in pl.parallel(4):
                    local = self.kernel(
                        x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
                    )
                for _j in pl.parallel(4):
                    local = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_seq_inside_parallel_keeps_inout(self):
        """``pl.range`` (sequential) inside ``pl.parallel`` triggers R-seq.

        Even if the inner sequential loop is the only writer-unit, the
        sequential ancestor forces ``InOut`` so cross-iteration WAW chains in
        the inner loop body are preserved.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for _i in pl.parallel(4):
                    for _j in pl.range(4):
                        local = self.kernel(
                            x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]}
                        )
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_parallel_inside_seq_keeps_inout(self):
        """``pl.parallel`` inside ``pl.range`` still triggers R-seq.

        The outer sequential loop is enough for R-seq, regardless of the kind
        of inner loops it contains.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for _i in pl.range(4):
                    for _j in pl.parallel(4):
                        local = self.kernel(
                            x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]}
                        )
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_top_level_call_then_parallel_promoted(self):
        """Top-level writer followed by ``pl.parallel`` writer hits R-prior.

        Mirror of the ``k2(local) for _ in pl.parallel: k1(local)`` scenario:
        the first call is the sole writer-unit, the parallel loop sees a
        prior writer-unit at sibling scope and is therefore promoted.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                local = self.kernel(
                    x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
                )
                for _i in pl.parallel(4):
                    local = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_while_keeps_inout(self):
        """``while`` loop body triggers R-seq (sequential writer-unit).

        ``WhileStmt`` is treated like a sequential for loop: the body may run
        any number of iterations, so cross-iteration WAW dependencies must be
        preserved by promoting Out → InOut.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                n: pl.Scalar[pl.INDEX],
            ) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                i: pl.Scalar[pl.INDEX] = 0
                while i < n:
                    local = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                    i = i + 1
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_first_writer_keeps_output_existing(self):
        """First writer inside an ``if`` branch is the only writer-unit.

        With no prior writer and no sequential ancestor, the call inside the
        branch keeps ``OutputExisting``. Each branch is analyzed against an
        independent ``seen_roots`` snapshot from the enclosing scope.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                flag: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                if flag:
                    local = self.kernel(
                        x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
                    )
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_if_after_top_level_writer_promoted(self):
        """``if`` branch following a top-level writer hits R-prior.

        The outer scope's prior-writer set already contains the local root
        when the ``if`` is entered, so the branch's snapshot starts with the
        root in ``seen``; the call inside is no longer the first writer.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                flag: pl.Scalar[pl.BOOL],
            ) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                local = self.kernel(
                    x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]}
                )
                if flag:
                    local = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return local

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_out_param_external_buffer_in_seq_loop_promoted(self):
        """R-seq on external root: writes inside ``pl.range`` promote to ``InOut``.

        ``dst`` is rooted at the enclosing ``main`` parameter (not locally
        allocated), but the sequential ancestor still requires WAW chaining
        across iterations — same as for local roots.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                for _i in pl.range(4):
                    dst = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return dst

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

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
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                offset: pl.Scalar[pl.INDEX],
                out: pl.Out[pl.Tensor[[256], pl.FP32]],
            ) -> pl.Tensor[[256], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [offset], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[256], pl.FP32],
            ) -> pl.Tensor[[256], pl.FP32]:
                for _i in pl.range(4):
                    dst = self.kernel(
                        x,
                        _i * 64,
                        dst,
                        attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.inout]},
                    )
                return dst

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_out_param_external_buffer_two_writes_second_promoted(self):
        """R-prior on external root: a prior writer-unit promotes the second to ``InOut``.

        Two consecutive top-level calls writing into the same enclosing-param
        destination. The first stays ``OutputExisting`` (no prior writer); the
        second sees the first as a prior writer and is promoted, mirroring the
        ``test_two_calls_top_level_second_promoted`` semantics for local roots.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                dst = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                dst = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return dst

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

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
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return r

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_out_param_enclosing_inout_in_parallel_loop_promoted(self):
        """R-enclosing wins even when wrapped in ``pl.parallel``.

        Mirrors the qwen3 KV-cache call site: the kernel is invoked once
        inside an outer ``pl.parallel`` loop, the buffer root traces back
        through the loop's iter binding to a ``pl.InOut`` parameter on the
        enclosing function. Neither R-seq nor R-prior fire here, so this
        case is the canonical motivator for R-enclosing.
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.InOut[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                for _i in pl.parallel(4):
                    dst = self.kernel(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.inout]})
                return dst

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_builtin_calls_left_untouched(self):
        """tensor.create / tile.* are builtin and keep arg_directions empty.

        Only the user ``kernel`` call gets an ``arg_directions`` vector; the
        ``tensor.create`` / ``tile.load`` / ``tile.store`` builtins keep their
        legacy empty ``arg_directions`` (no ``attrs`` is emitted for them).
        """

        @pl.program
        class Before:
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

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                local = pl.tensor.create([64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                r = self.kernel(x, local, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)


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
        # ``Before`` pre-populates the Out-param slot with ``Output``
        # (runtime-allocation semantics). The derive pass would otherwise emit
        # ``OutputExisting`` for an external/param-rooted destination, so the
        # ``After == Before`` check confirms the explicit call-site choice
        # survives instead of being overwritten.
        @pl.program
        class Before:
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
                r: pl.Tensor[[64], pl.FP32] = self.kernel(
                    x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output]}
                )
                return r

        After = passes.derive_call_directions()(Before)
        # Explicit directions survive untouched: After is structurally Before.
        ir.assert_structural_equal(After, Before)


# ---------------------------------------------------------------------------
# Derive pass: Submit (pl.submit) call-like nodes
# ---------------------------------------------------------------------------
#
# These cases exercise the dedicated VisitExpr_(SubmitPtr) handler
# (derive_call_directions_pass.cpp:341). They are built with raw ``ir.*``
# builders rather than the ``@pl.program`` DSL because the DSL frontend
# rejects a ``pl.submit`` whose positional args are a strict *prefix* of the
# callee signature ("Function 'stage1' expects 2 argument(s), got 1") — the
# runtime-allocated tail Out form is synthesised by an internal IR builder
# (e.g. ConvertTensorToTileOps appends the Out param at the tail), never
# written by hand in the DSL. So both Before and Expected are assembled
# directly as ``ir.Submit`` nodes.
#
# Key semantic facts derived from the pass / the pass-submit-awareness rule:
#   * The handler PRESERVES Submit-ness (pass-submit-awareness.md rule 3):
#     After carries an ``ir.Submit`` with ``arg_directions`` added to its
#     attrs — never a lowered plain ``ir.Call``. The TASK_ID-augmented
#     return tuple and ``deps_`` stay first-class on the rewritten Submit.
#   * Submit.args_ is a positional *prefix* of callee->params_. The size guard
#     is relaxed to ``args.size() <= params.size()`` for Submit
#     (derive_call_directions_pass.cpp:404-407), so only the caller-supplied
#     prefix args receive an ArgDirection. Tail callee Out params
#     (runtime-allocated outputs, materialised via the Submit return tuple)
#     are NOT in args_ and therefore get NO direction — the derived
#     ``arg_directions`` vector length equals ``len(args)``, not
#     ``len(callee params)``.
#   * Submit.deps_ is a typed field, not an attr: a Submit never stores a
#     ``manual_dep_edges`` attr and plain GlobalVar Calls never carry one
#     (ManualDepsOnSubmitOnly invariant), so the only attr the pass adds is
#     ``arg_directions``. Downstream consumers that need a Call-shaped view
#     funnel through ``SubmitToCallView``.


_SUBMIT_SPAN = ir.Span.unknown()


def _t256():
    """Shared [16, 256] FP32 tensor type for the Submit cases."""
    return ir.TensorType([16, 256], DataType.FP32)


def _t64():
    """Shared [64] FP32 tensor type for the Submit deps case."""
    return ir.TensorType([64], DataType.FP32)


class TestDeriveSubmit:
    """Before/After coverage for the dedicated Submit handler.

    The pass derives ``arg_directions`` for a ``Submit`` but PRESERVES the
    Submit kind (pass-submit-awareness.md rule 3): a plain ``Call`` must carry
    its callee's declared return type, whereas a Submit's type is the
    TASK_ID-augmented ``Tuple[<outputs>..., Scalar[TASK_ID]]``. Lowering
    Submit -> Call here used to leave a malformed plain Call carrying that
    Tuple type, which could not survive print -> reparse. Keeping the Submit
    makes the printer emit ``pl.submit(...)`` — which round-trips — so these
    tests run under the default ``PassContext`` (the repo conftest's
    ``PYPTO_VERIFY_LEVEL=roundtrip`` print -> parse -> structural_equal
    instrument is exercised, not suppressed). ``deps_`` stays a typed field on
    the rewritten Submit; downstream consumers funnel through
    ``SubmitToCallView`` where they need the Call-shaped view.
    """

    def test_submit_runtime_allocated_tail_out(self):
        """A ``pl.submit`` whose only declared-Out callee param is a
        runtime-allocated *tail* output.

        Callee ``stage1(x: In, scratch: Out)`` is submitted with a single
        positional arg ``a`` — ``scratch`` is NOT passed (it is the
        runtime-allocated tail Out, materialised as a return-tuple element).
        The pass KEEPS the Submit and derives directions only for the prefix
        args: ``x`` is callee In + tensor -> ``Input``. ``scratch`` gets no slot
        in the vector, so ``arg_directions == [Input]`` (length 1, matching
        ``args``), not length 2. This is the args-side dual of the Submit
        return-type asymmetry — the canonical FOCUS scenario.
        """
        span = _SUBMIT_SPAN

        def build_stage1():
            x = ir.Var("x", _t256(), span)
            scratch = ir.Var("scratch", _t256(), span)
            return ir.Function(
                "stage1",
                [(x, ir.ParamDirection.In), (scratch, ir.ParamDirection.Out)],
                [_t256()],
                ir.ReturnStmt([scratch], span),
                span,
                ir.FunctionType.InCore,
            )

        # Submit return tuple: [runtime-allocated scratch, callee return, TASK_ID].
        submit_ret = ir.TupleType([_t256(), _t256(), ir.ScalarType(DataType.TASK_ID)])

        # --- Before: pl.submit(self.stage1, a) with prefix args [a] ---
        a = ir.Var("a", _t256(), span)
        res = ir.Var("res", submit_ret, span)
        submit = ir.Submit(ir.GlobalVar("stage1"), [a], [], submit_ret, span)
        before_main = ir.Function(
            "main",
            [(a, ir.ParamDirection.In)],
            [submit_ret],
            ir.SeqStmts([ir.AssignStmt(res, submit, span), ir.ReturnStmt([res], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Before = ir.Program([build_stage1(), before_main], "submit_runtime_out", span)

        # --- Expected: Submit preserved, arg_directions=[Input] (prefix only) ---
        a2 = ir.Var("a", _t256(), span)
        res2 = ir.Var("res", submit_ret, span)
        submit_after = ir.Submit(
            ir.GlobalVar("stage1"),
            [a2],
            [],
            {},
            {"arg_directions": [ir.ArgDirection.Input]},
            submit_ret,
            span,
        )
        exp_main = ir.Function(
            "main",
            [(a2, ir.ParamDirection.In)],
            [submit_ret],
            ir.SeqStmts([ir.AssignStmt(res2, submit_after, span), ir.ReturnStmt([res2], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Expected = ir.Program([build_stage1(), exp_main], "submit_runtime_out", span)

        # A prefix-args Submit (runtime-allocated tail Out) hits a SEPARATE,
        # pre-existing parser limitation: ``pl.submit(self.stage1, a)`` cannot
        # reparse because the callee declares more params than the Submit passes
        # (the tail Out is runtime-allocated, materialised as a return-tuple
        # element). This is independent of the TASK_ID-Tuple bug fixed by
        # preserving Submit-ness — suppress only the print->parse roundtrip and
        # still assert the exact Submit structure via structural_equal.
        with _core_passes.PassContext(
            [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
        ):
            After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_submit_caller_allocated_out_plus_runtime_tail_out(self):
        """A ``pl.submit`` mixing a caller-allocated Out arg with a
        runtime-allocated tail Out.

        Callee ``stage1(x: In, out: Out, scratch: Out)`` is submitted with
        ``[a, dst]`` — ``scratch`` is the runtime-allocated tail Out (index 2,
        not passed). Derivation over the 2-arg prefix:
          * ``x``  -> ``Input`` (callee In + tensor).
          * ``dst`` -> ``OutputExisting``: callee Out, rooted at ``main``'s
            param ``dst``, single call (no R-seq: no sequential ancestor; no
            R-prior: first writer; no R-enclosing: ``dst`` declared Out, not
            InOut). So it stays at the OutputExisting default.
        The tail ``scratch`` gets no slot, so ``arg_directions ==
        [Input, OutputExisting]`` (length 2). Confirms the prefix invariant
        holds even when a real caller-allocated Out is present in the prefix.
        """
        span = _SUBMIT_SPAN

        def build_stage1():
            x = ir.Var("x", _t256(), span)
            out = ir.Var("out", _t256(), span)
            scratch = ir.Var("scratch", _t256(), span)
            return ir.Function(
                "stage1",
                [
                    (x, ir.ParamDirection.In),
                    (out, ir.ParamDirection.Out),
                    (scratch, ir.ParamDirection.Out),
                ],
                [_t256()],
                ir.ReturnStmt([scratch], span),
                span,
                ir.FunctionType.InCore,
            )

        submit_ret = ir.TupleType([_t256(), _t256(), ir.ScalarType(DataType.TASK_ID)])

        # --- Before: pl.submit(self.stage1, a, dst) — prefix args [a, dst] ---
        a = ir.Var("a", _t256(), span)
        dst = ir.Var("dst", _t256(), span)
        res = ir.Var("res", submit_ret, span)
        submit = ir.Submit(ir.GlobalVar("stage1"), [a, dst], [], submit_ret, span)
        before_main = ir.Function(
            "main",
            [(a, ir.ParamDirection.In), (dst, ir.ParamDirection.Out)],
            [submit_ret],
            ir.SeqStmts([ir.AssignStmt(res, submit, span), ir.ReturnStmt([res], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Before = ir.Program([build_stage1(), before_main], "submit_caller_runtime_out", span)

        # --- Expected: Submit preserved, arg_directions=[Input, OutputExisting] ---
        a2 = ir.Var("a", _t256(), span)
        dst2 = ir.Var("dst", _t256(), span)
        res2 = ir.Var("res", submit_ret, span)
        submit_after = ir.Submit(
            ir.GlobalVar("stage1"),
            [a2, dst2],
            [],
            {},
            {"arg_directions": [ir.ArgDirection.Input, ir.ArgDirection.OutputExisting]},
            submit_ret,
            span,
        )
        exp_main = ir.Function(
            "main",
            [(a2, ir.ParamDirection.In), (dst2, ir.ParamDirection.Out)],
            [submit_ret],
            ir.SeqStmts([ir.AssignStmt(res2, submit_after, span), ir.ReturnStmt([res2], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Expected = ir.Program([build_stage1(), exp_main], "submit_caller_runtime_out", span)

        # Prefix-args Submit (runtime-allocated tail Out) — see
        # test_submit_runtime_allocated_tail_out for why the print->parse
        # roundtrip is suppressed (separate parser limitation); structural_equal
        # still verifies the Submit is preserved with the derived directions.
        with _core_passes.PassContext(
            [_core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)]
        ):
            After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)

    def test_submit_with_deps_preserves_submit_with_arg_directions(self):
        """A ``pl.submit(..., deps=[t])`` keeps its dep edge AND the Submit kind
        while getting ``arg_directions`` derived.

        Callee ``consumer(x: In)`` returns its input. The Submit carries one
        TaskId dep ``t``. The handler (pass-submit-awareness rule items 2-3):
          * preserves Submit-ness, keeping ``deps_=[t]`` as the typed field
            (rule 3) rather than re-encoding it as a ``manual_dep_edges`` Call
            attr; and
          * derives ``arg_directions=[Input]`` for the single In tensor arg.
        The rewritten Submit round-trips as ``pl.submit(self.consumer, a,
        deps=[t], attrs={"arg_directions": [...]})``.
        """
        span = _SUBMIT_SPAN

        def build_consumer():
            x = ir.Var("x", _t64(), span)
            return ir.Function(
                "consumer",
                [(x, ir.ParamDirection.In)],
                [_t64()],
                ir.ReturnStmt([x], span),
                span,
                ir.FunctionType.InCore,
            )

        submit_ret = ir.TupleType([_t64(), ir.ScalarType(DataType.TASK_ID)])

        # --- Before: res = pl.submit(self.consumer, a, deps=[t]) ---
        a = ir.Var("a", _t64(), span)
        t = ir.Var("t", ir.ScalarType(DataType.TASK_ID), span)
        res = ir.Var("res", submit_ret, span)
        submit = ir.Submit(ir.GlobalVar("consumer"), [a], [t], submit_ret, span)
        before_main = ir.Function(
            "main",
            [(a, ir.ParamDirection.In), (t, ir.ParamDirection.In)],
            [submit_ret],
            ir.SeqStmts([ir.AssignStmt(res, submit, span), ir.ReturnStmt([res], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Before = ir.Program([build_consumer(), before_main], "submit_with_deps", span)

        # --- Expected: Submit preserved with deps=[t] and arg_directions=[Input] ---
        a2 = ir.Var("a", _t64(), span)
        t2 = ir.Var("t", ir.ScalarType(DataType.TASK_ID), span)
        res2 = ir.Var("res", submit_ret, span)
        submit_after = ir.Submit(
            ir.GlobalVar("consumer"),
            [a2],
            [t2],
            {},
            {"arg_directions": [ir.ArgDirection.Input]},
            submit_ret,
            span,
        )
        exp_main = ir.Function(
            "main",
            [(a2, ir.ParamDirection.In), (t2, ir.ParamDirection.In)],
            [submit_ret],
            ir.SeqStmts([ir.AssignStmt(res2, submit_after, span), ir.ReturnStmt([res2], span)], span),
            span,
            ir.FunctionType.Orchestration,
        )
        Expected = ir.Program([build_consumer(), exp_main], "submit_with_deps", span)

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)


# ---------------------------------------------------------------------------
# pl.spmd_submit: Submit + launch spec preserved through DeriveCallDirections
# ---------------------------------------------------------------------------


class TestDeriveSpmdSubmit:
    """Before/Expected coverage for the explicit ``pl.spmd_submit(...)`` form.

    A ``pl.spmd_submit(self.kernel, ..., core_num=N, sync_start=...)`` is an
    ``ir.Submit`` carrying an SPMD launch spec (``core_num`` / ``sync_start``).
    DeriveCallDirections derives ``arg_directions`` for it but MUST keep the
    node a Submit (pass-submit-awareness.md rule 3) — preserving the launch
    spec, the typed ``deps_`` field, and the TASK_ID-augmented Tuple return.
    Lowering to a plain Call would fold the launch spec into attrs and leave a
    Tuple-annotated plain call that cannot survive print -> reparse. Runs under
    the repo conftest's default roundtrip instrument, so ``pl.spmd_submit(...)``
    also round-trips after the pass (sibling roundtrip-asymmetry tests in
    ``TestDeriveSubmit`` cover the prefix-args case that does not).
    """

    def test_spmd_submit_launch_spec_and_deps_survive_with_derived_directions(self):
        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                r: pl.Tile[[64], pl.FP32] = pl.add(t, t)
                o: pl.Tensor[[64], pl.FP32] = pl.store(r, [0], out)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Out[pl.Tensor[[64], pl.FP32]],
                z: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    o1, t1 = pl.spmd_submit(self.kernel, x, y, core_num=8, sync_start=True)
                    o2, t2 = pl.spmd_submit(self.kernel, x, z, core_num=8, deps=[t1])
                return z

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                r: pl.Tile[[64], pl.FP32] = pl.add(t, t)
                o: pl.Tensor[[64], pl.FP32] = pl.store(r, [0], out)
                return o

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                y: pl.Out[pl.Tensor[[64], pl.FP32]],
                z: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                with pl.manual_scope():
                    # Submit kind, launch spec (core_num / sync_start) and deps_
                    # are preserved; only arg_directions=[Input, OutputExisting]
                    # is added (x -> In tensor, y/z -> first-writer Out).
                    o1, t1 = pl.spmd_submit(
                        self.kernel,
                        x,
                        y,
                        core_num=8,
                        sync_start=True,
                        attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]},
                    )
                    o2, t2 = pl.spmd_submit(
                        self.kernel,
                        x,
                        z,
                        core_num=8,
                        deps=[t1],
                        attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]},
                    )
                return z

        After = passes.derive_call_directions()(Before)
        ir.assert_structural_equal(After, Expected)


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


class _RewriteUserCall(ir.IRMutator):
    """Replace every non-builtin Call's arg_directions with *new_dirs*.

    Used only to build deliberately ill-formed IR for the negative verifier
    tests below; well-formed derived directions are exercised via the
    Before/Expected transform tests above.
    """

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


class TestVerifyAutoDepsPositive:
    """Verify pass accepts direction rewrites produced by later dependency passes."""

    def test_inout_with_output_existing_is_allowed_after_auto_dep_rewrite(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, x: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x)
                return r

        prog = _RewriteUserCall([ir.ArgDirection.OutputExisting]).run(Prog)
        _verify_call_directions(prog)


# ---------------------------------------------------------------------------
# pl.no_dep override
# ---------------------------------------------------------------------------
#
# These tests are NOT expressed with the Before/Expected + assert_structural_equal
# pattern. Reason: ``pl.no_dep(arg)`` records its marker as a separate
# ``Call.attrs['arg_direction_overrides']`` entry, and the python printer
# (src/ir/transforms/python_printer.cpp:643-658) only emits the derived
# ``arg_directions`` vector — it never emits ``arg_direction_overrides``.
# After DeriveCallDirections the marked call therefore carries TWO attrs
# (``arg_direction_overrides`` + ``arg_directions``), but any program written
# from / round-tripped through the printer keeps only ONE, so
# ``assert_structural_equal`` fails with "Kwargs size mismatch (2 != 1)" on the
# call's ``attrs``. Until the printer round-trips ``arg_direction_overrides``,
# these stay as direction-vector inspection tests, and the class overrides the
# global verification fixture to property-verification-only (no print/parse
# roundtrip).


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
    return list(call.arg_directions)


class TestNoDepOverride:
    """``pl.no_dep(arg)`` at a kernel call site sets ArgDirection.NoDep at that slot."""

    @pytest.fixture(autouse=True)
    def _no_roundtrip(self):
        """Override the global roundtrip fixture for this class only.

        The ``arg_direction_overrides`` attr left by ``pl.no_dep`` now round-trips
        cleanly (see ``TestOutlineNoDepArgs`` which runs under full roundtrip).
        The remaining blocker for THESE fixtures is unrelated: their ``kernel``
        functions declare no return-type annotation, so the printed kernel has
        no ``-> ...`` and the reparsed call's return type degrades to
        ``UnknownType`` (TensorType on the original) — a separate print/reparse
        gap. Fall back to the lighter BEFORE_AND_AFTER property-verification mode.
        """
        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

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

    def test_no_dep_on_inout_param_accepted(self):
        # ``NoDep`` is legal on callee ``InOut`` params: the user opts the slot
        # out of OverlapMap tracking for both the read and the write side,
        # asserting out-of-band that there is no RaW / WaW conflict on the
        # slot. Typical use: paged-attention writes whose offset is
        # data-dependent (so the compiler cannot prove disjointness) but are
        # guaranteed disjoint by the runtime allocation protocol.
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

        new_prog = passes.derive_call_directions()(P)
        calls = _user_calls(new_prog)
        assert len(calls) == 1
        # 0=a (Input), 1=b marked NoDep (overrides the auto-derived InOut).
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.NoDep]
        # The post-pass verifier must also accept the resulting IR.
        _verify_call_directions(new_prog)

    def test_no_dep_on_out_param_accepted(self):
        # ``NoDep`` is also legal on callee ``Out`` params (the write-side
        # analogue of the InOut case). The auto-deriver would otherwise pick
        # ``OutputExisting`` for a first writer at top level; the override
        # forces ``NoDep`` and the verifier accepts it.
        @pl.program
        class P:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
            ):
                return b

            @pl.function(type=pl.FunctionType.Orchestration)
            def orch(
                self,
                a: pl.Tensor[[16, 16], pl.FP32],
                b: pl.Tensor[[16, 16], pl.FP32],
            ):
                b = self.kernel(a, pl.no_dep(b))
                return b

        new_prog = passes.derive_call_directions()(P)
        calls = _user_calls(new_prog)
        assert len(calls) == 1
        # 0=a (Input), 1=b marked NoDep (overrides the auto-derived
        # OutputExisting). The verifier must accept the resulting IR.
        assert _dirs(calls[0]) == [ir.ArgDirection.Input, ir.ArgDirection.NoDep]
        _verify_call_directions(new_prog)


class TestMaterializeWrapperDirections:
    """Phase 0: Group/Spmd wrapper signatures are rewritten to their effective directions.

    A wrapper forwards its params 1:1 to an inner kernel call, but its own
    ``param_directions_`` can still read ``In`` for a param the inner kernel
    writes. ``DeriveCallDirections`` recovers the true direction once and stores
    it in the signature, so every downstream consumer reads
    ``callee.param_directions`` instead of recomputing it.
    """

    def _dirs_of(self, program, func_name):
        for _gv, func in program.functions.items():
            if func.name == func_name:
                return list(func.param_directions)
        raise AssertionError(f"function '{func_name}' not found in program")

    def test_group_wrapper_out_direction_materialized(self):
        """A Group declaring both params In gains Out on the param its kernel writes."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function(type=pl.FunctionType.Group)
            def group(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, out)
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.group(x, dst)
                return r

        @pl.program
        class Expected:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            # `out` is now declared Out: the pass wrote the effective direction back.
            @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
            def group(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, out, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                # `dst` resolves to OutputExisting because the Group's `out` is
                # now Out; a stale `In` would have left this as Input.
                r = self.group(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

        assert self._dirs_of(Before, "group") == [ir.ParamDirection.In, ir.ParamDirection.In]
        After = passes.derive_call_directions()(Before)
        assert self._dirs_of(After, "group") == [ir.ParamDirection.In, ir.ParamDirection.Out]
        ir.assert_structural_equal(After, Expected)
        _verify_call_directions(After)

        # Re-running on already-materialized IR is a no-op.
        ir.assert_structural_equal(passes.derive_call_directions()(After), After)

    def test_spmd_wrapper_inout_direction_materialized(self):
        """An Spmd wrapper param feeding an InOut kernel param becomes InOut, not In."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(self, acc: pl.InOut[pl.Tensor[[64], pl.FP32]]) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(acc, [0], [64])
                t2: pl.Tile[[64], pl.FP32] = pl.tile.add(t, t)
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t2, [0], acc)
                return ret

            @pl.function(type=pl.FunctionType.Spmd)
            def shard(self, acc: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(acc)
                return r

            @pl.function
            def main(self, acc: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.shard(acc)
                return r

        After = passes.derive_call_directions()(Before)
        assert self._dirs_of(After, "shard") == [ir.ParamDirection.InOut]
        _verify_call_directions(After)

    def test_nested_group_chain_propagates_out(self):
        """Group → Group → InCore: the Out direction reaches the outermost wrapper."""

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function(type=pl.FunctionType.Group)
            def inner(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.kernel(x, out)
                return r

            @pl.function(type=pl.FunctionType.Group)
            def outer(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.inner(x, out)
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.outer(x, dst)
                return r

        After = passes.derive_call_directions()(Before)
        assert self._dirs_of(After, "inner") == [ir.ParamDirection.In, ir.ParamDirection.Out]
        assert self._dirs_of(After, "outer") == [ir.ParamDirection.In, ir.ParamDirection.Out]
        _verify_call_directions(After)

    def test_declared_out_never_demoted_when_no_inner_call_writes_it(self):
        """The merge is monotone: a declared Out survives even with no inner-call evidence.

        ``scratch`` is declared Out but the wrapper's only inner call takes it
        as In. A non-monotone merge would infer ``In`` and write that loss back
        into the signature, silently dropping the write dependency at every
        call site.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.InCore)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                ret: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return ret

            @pl.function(type=pl.FunctionType.Group)
            def group(
                self,
                scratch: pl.Out[pl.Tensor[[64], pl.FP32]],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                # `scratch` is only ever read by the inner call.
                r: pl.Tensor[[64], pl.FP32] = self.kernel(scratch, out)
                return r

            @pl.function
            def main(
                self,
                a: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.group(a, dst)
                return r

        After = passes.derive_call_directions()(Before)
        # scratch keeps its declared Out; out is promoted In -> Out.
        assert self._dirs_of(After, "group") == [ir.ParamDirection.Out, ir.ParamDirection.Out]
        _verify_call_directions(After)

    def test_mutually_recursive_wrappers_reach_a_fixed_point(self):
        """A wrapper cycle must not lose a write dependency, whatever the visit order.

        ``a`` forwards ``p`` to an Out kernel *and* to wrapper ``z``; ``z`` calls
        back into ``a``. Wrappers are visited in program-map (name) order, so
        ``a`` is seeded first. A recursive walk with a cycle guard would memoize
        ``z`` from ``a``'s *declared* directions — before ``a`` itself is
        promoted — leaving ``z.p`` as ``In`` and dropping its write dependency.
        Iterating the monotone merge to a fixed point cannot.

        Runs under an empty ``PassContext`` because the repo conftest's
        roundtrip instrument cannot handle this program: the printer emits
        functions in dependency order, which a wrapper cycle makes impossible,
        so ``a`` is printed before ``z`` and the reparsed forward reference
        drops the call's ``arg_directions`` attrs. That is a printer/parser
        limitation, orthogonal to the direction analysis under test.
        """

        @pl.program
        class Before:
            @pl.function(type=pl.FunctionType.Group)
            def a(
                self,
                x: pl.Tensor[[64], pl.FP32],
                p: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                _r0: pl.Tensor[[64], pl.FP32] = self.z(x, p)
                r1: pl.Tensor[[64], pl.FP32] = self.kern(x, p)
                return r1

            @pl.function(type=pl.FunctionType.InCore)
            def kern(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t: pl.Tile[[64], pl.FP32] = pl.load(x, [0], [64])
                r: pl.Tensor[[64], pl.FP32] = pl.store(t, [0], out)
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                d: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.a(x, d)
                return r

            @pl.function(type=pl.FunctionType.Group)
            def z(
                self,
                x: pl.Tensor[[64], pl.FP32],
                p: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r: pl.Tensor[[64], pl.FP32] = self.a(x, p)
                return r

        with _core_passes.PassContext([]):
            After = passes.derive_call_directions()(Before)
        # `p` is written through the kernel, so it is Out on both wrappers.
        assert self._dirs_of(After, "a") == [ir.ParamDirection.In, ir.ParamDirection.Out]
        assert self._dirs_of(After, "z") == [ir.ParamDirection.In, ir.ParamDirection.Out]
        # A stale wrapper signature would make the pass fail its own verifier.
        _verify_call_directions(After)


class TestVerifyWrapperDirections:
    """The CallDirectionsResolved verifier rejects a wrapper left with stale directions."""

    def test_stale_group_directions_rejected(self):
        # `group` declares `out` as In, but its kernel call writes it and the
        # call sites already carry the matching arg_directions — exactly the
        # inconsistency Phase 0 exists to prevent.
        @pl.program
        class Stale:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
            def group(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, out, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.group(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.input]})
                return r

        with pytest.raises(Exception, match="stale param_directions_"):
            _verify_call_directions(Stale)

    def test_materialized_group_directions_accepted(self):
        @pl.program
        class Fresh:
            @pl.function(type=pl.FunctionType.InCore, level=pl.Level.CHIP_DIE, role=pl.Role.SubWorker)
            def kernel(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec)
                ret = pl.tile.store(t, [0], out)
                return ret

            @pl.function(type=pl.FunctionType.Group, level=pl.Level.CORE_GROUP, role=pl.Role.SubWorker)
            def group(
                self,
                x: pl.Tensor[[64], pl.FP32],
                out: pl.Out[pl.Tensor[[64], pl.FP32]],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.kernel(x, out, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

            @pl.function
            def main(
                self,
                x: pl.Tensor[[64], pl.FP32],
                dst: pl.Tensor[[64], pl.FP32],
            ) -> pl.Tensor[[64], pl.FP32]:
                r = self.group(x, dst, attrs={"arg_directions": [pl.adir.input, pl.adir.output_existing]})
                return r

        _verify_call_directions(Fresh)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
