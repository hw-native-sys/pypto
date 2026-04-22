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

    def test_out_param_local_buffer_promoted_to_inout(self):
        """Callee Out + locally allocated buffer → InOut (WAW promotion)."""

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
