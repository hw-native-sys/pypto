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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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
                t = pl.tile.load(x, [0], [64], [64], target_memory=pl.Mem.Vec, transpose=False)
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

        ``pl.no_dep`` leaves an ``arg_direction_overrides`` attr that the python
        printer does not emit, so the print -> parse -> structural_equal
        roundtrip fails on the call's ``attrs``. Fall back to the lighter
        BEFORE_AND_AFTER property-verification mode here.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
