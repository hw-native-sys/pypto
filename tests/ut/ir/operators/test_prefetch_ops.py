# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the ``prefetch.*`` async GM->L2 prefetch op family."""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.pypto_core import ir as _ir_core


class TestPrefetchOpTypes:
    """Type deduction for the four prefetch ops and their opaque handle types."""

    def test_handle_types_are_singletons(self):
        """Each handle type getter returns a structurally equal singleton."""
        assert ir.PrefetchAsyncContextType.get() == ir.PrefetchAsyncContextType.get()
        assert ir.AsyncEventType.get() == ir.AsyncEventType.get()
        assert ir.AsyncSessionType.get() == ir.AsyncSessionType.get()

    def test_ops_are_registered(self):
        """All four ops resolve through the registry (get_op raises on a typo)."""
        for name in (
            "prefetch.make_context",
            "prefetch.async_prefetch",
            "prefetch.session",
            "prefetch.wait",
        ):
            assert _ir_core.get_op(name).name == name

    def test_prefetch_sequence_deduces_handle_types(self):
        """A full make_context -> async_prefetch -> session -> wait chain types correctly."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[4096], pl.FP32],
                ws: pl.Tensor[[1024], pl.INT8],
            ) -> pl.Tensor[[4096], pl.FP32]:
                ctx = pl.prefetch.make_context(ws)
                evt = pl.prefetch.async_prefetch(x, ctx)
                session = pl.prefetch.session(ctx)
                pl.prefetch.wait(evt, session)
                return x

        ir_str = str(Program)
        assert "prefetch.make_context" in ir_str
        assert "prefetch.async_prefetch" in ir_str
        assert "prefetch.session" in ir_str
        assert "prefetch.wait" in ir_str
        # Handle-typed bindings are annotated with the singleton marker types.
        assert "pl.PrefetchAsyncContextType" in ir_str
        assert "pl.AsyncEventType" in ir_str
        assert "pl.AsyncSessionType" in ir_str

    def test_roundtrip_through_printer(self):
        """Printed IR re-parses to a structurally identical program."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[1, 256], pl.FP32],
                ws: pl.Tensor[[512], pl.INT8],
            ) -> pl.Tensor[[1, 256], pl.FP32]:
                ctx = pl.prefetch.make_context(ws)
                evt = pl.prefetch.async_prefetch(x, ctx)
                session = pl.prefetch.session(ctx)
                pl.prefetch.wait(evt, session)
                return x

        reparsed = pl.parse_program(str(Program))
        ir.assert_structural_equal(reparsed, Program)

    def test_logical_1d_multi_dim_source_accepted(self):
        """A ``[1, 1, N]`` source is logically 1D and is accepted."""

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(
                self,
                x: pl.Tensor[[1, 1, 256], pl.FP32],
                ws: pl.Tensor[[512], pl.INT8],
            ) -> pl.Tensor[[1, 1, 256], pl.FP32]:
                ctx = pl.prefetch.make_context(ws)
                pl.prefetch.async_prefetch(x, ctx)
                return x

        assert "prefetch.async_prefetch" in str(Program)


class TestPrefetchOpVerification:
    """The IR-level verifier mirrors the PTOAS ``verify()`` input checks."""

    def test_non_1d_source_rejected(self):
        """A ``[4, 32]`` source is not a flat contiguous logical-1D region."""
        with pytest.raises(Exception, match="flat contiguous logical 1D"):

            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    x: pl.Tensor[[4, 32], pl.FP32],
                    ws: pl.Tensor[[512], pl.INT8],
                ) -> pl.Tensor[[4, 32], pl.FP32]:
                    ctx = pl.prefetch.make_context(ws)
                    pl.prefetch.async_prefetch(x, ctx)
                    return x

    def test_non_int8_workspace_rejected(self):
        """The SDMA workspace must be raw INT8 bytes."""
        with pytest.raises(Exception, match="INT8 workspace"):

            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    x: pl.Tensor[[256], pl.FP32],
                    ws: pl.Tensor[[512], pl.FP32],
                ) -> pl.Tensor[[256], pl.FP32]:
                    pl.prefetch.make_context(ws)
                    return x

    def test_wait_rejects_mismatched_handle(self):
        """``wait`` requires an AsyncSession, not the context it was projected from."""
        with pytest.raises(Exception, match="session to be an AsyncSession"):

            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    x: pl.Tensor[[256], pl.FP32],
                    ws: pl.Tensor[[512], pl.INT8],
                ) -> pl.Tensor[[256], pl.FP32]:
                    ctx = pl.prefetch.make_context(ws)
                    evt = pl.prefetch.async_prefetch(x, ctx)
                    # Deliberately wrong handle type — the runtime verifier must reject it,
                    # so the static type error here is the point of the test.
                    pl.prefetch.wait(evt, ctx)  # type: ignore[arg-type]
                    return x

    def test_async_prefetch_rejects_non_context(self):
        """``async_prefetch`` requires a PrefetchAsyncContext as its second operand."""
        with pytest.raises(Exception, match="ctx to be a PrefetchAsyncContext"):

            @pl.program
            class Program:
                @pl.function(type=pl.FunctionType.InCore)
                def main(
                    self,
                    x: pl.Tensor[[256], pl.FP32],
                    ws: pl.Tensor[[512], pl.INT8],
                ) -> pl.Tensor[[256], pl.FP32]:
                    ctx = pl.prefetch.make_context(ws)
                    evt = pl.prefetch.async_prefetch(x, ctx)
                    # Deliberately wrong handle type — see note above.
                    pl.prefetch.async_prefetch(x, evt)  # type: ignore[arg-type]
                    return x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
