# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for the DeriveManualScopeDeps pass."""

import pypto.language as pl
import pytest
from pypto import passes
from pypto.pypto_core import passes as _core_passes


@pytest.fixture(autouse=True)
def pass_verification_context():
    """Skip the global print -> parse -> assert_structural_equal roundtrip.

    The python_printer does not surface ``Call.attrs['manual_dep_edges']`` (an
    internal post-pass attr), so the roundtrip would always fail after this
    pass. Property verification still runs.
    """
    instruments: list[_core_passes.PassInstrument] = [
        _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
    ]
    with _core_passes.PassContext(instruments):
        yield


class TestDeriveManualScopeDeps:
    def test_no_manual_scope_is_noop(self):
        @pl.program
        class Prog:
            @pl.function(type=pl.FunctionType.InCore)
            def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                return x

            @pl.function(type=pl.FunctionType.Orchestration)
            def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                a = self.k1(x)
                return a

        ssa = passes.convert_to_ssa()(Prog)
        ddir = passes.derive_call_directions()(ssa)
        ddep = passes.derive_manual_scope_deps()(ddir)
        # No manual scope ⇒ pass is a no-op (returns same Program).
        assert ddep.same_as(ddir)


class TestManualScopeNesting:
    @pytest.fixture(autouse=True)
    def pass_verification_context(self):
        instruments: list[_core_passes.PassInstrument] = [
            _core_passes.VerificationInstrument(_core_passes.VerificationMode.BEFORE_AND_AFTER)
        ]
        with _core_passes.PassContext(instruments):
            yield

    def test_nested_manual_scope_rejected(self):
        """The runtime forbids MANUAL inside MANUAL; reject at parse time."""
        with pytest.raises(Exception, match="manual_scope"):  # noqa: B017

            @pl.program
            class _Prog:
                @pl.function(type=pl.FunctionType.InCore)
                def k1(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    return x

                @pl.function(type=pl.FunctionType.Orchestration)
                def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
                    with pl.manual_scope():
                        a = self.k1(x)
                        with pl.manual_scope():  # nested — must error
                            b = self.k1(a)
                    return b


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
