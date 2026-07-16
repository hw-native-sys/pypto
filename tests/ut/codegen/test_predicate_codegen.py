# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestration-codegen tests for the dispatch predicate.

Drives ``pl.spmd_submit(..., predicate=pl.dispatch_pred(...))`` through the full
Default pass pipeline and asserts the orchestration C++ emits the runtime
``L0TaskPredicate`` + ``Arg::set_predicate(...)`` sequence. Also proves the
predicate operand tensor survives inlining / SSA / outlining and resolves to its
``ext_<name>`` orchestration reference (exercising the Submit pass-walk safety).
"""

import pypto.language as pl
import pytest
from _orchestration_codegen_common import _generate_orch_full_pipeline


@pl.program
class _WithPredicate:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(
        self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.InCore)
    def gate(self, g: pl.Out[pl.Tensor[[512, 128], pl.INT32]]) -> pl.Tensor[[512, 128], pl.INT32]:
        t = pl.load(g, [0, 0], [128, 128])
        g = pl.store(t, [0, 0], g)
        return g

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self,
        x: pl.Tensor[[512, 128], pl.FP32],
        out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        rc: pl.Out[pl.Tensor[[512, 128], pl.INT32]],
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.manual_scope():
            rc, g_tid = pl.spmd_submit(self.gate, rc, core_num=1)
            out, _ = pl.spmd_submit(
                self.expert, x, out, core_num=1, deps=[g_tid],
                predicate=pl.dispatch_pred(rc, [0, 0], ">", 0),
            )
        return out


@pl.program
class _NoPredicate:
    @pl.function(type=pl.FunctionType.InCore)
    def expert(
        self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        t = pl.load(x, [0, 0], [128, 128])
        out = pl.store(t, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[512, 128], pl.FP32], out: pl.Out[pl.Tensor[[512, 128], pl.FP32]]
    ) -> pl.Tensor[[512, 128], pl.FP32]:
        with pl.manual_scope():
            out, _ = pl.spmd_submit(self.expert, x, out, core_num=1)
        return out


def test_predicate_emits_set_predicate_block():
    code = _generate_orch_full_pipeline(_WithPredicate)
    assert "L0TaskPredicate" in code, code
    # Operand resolves to the orchestration ext_ reference; op/target/indices emitted.
    assert ".operand.tensor = &ext_rc;" in code, code
    assert ".operand.ndims = 2;" in code, code
    assert ".operand.indices[0] = 0;" in code, code
    assert ".operand.indices[1] = 0;" in code, code
    assert ".op = PredicateOp::GT;" in code, code
    assert ".target = 0;" in code, code
    assert ".set_predicate(" in code, code
    # Exactly one predicated task (the expert), not the gate.
    assert code.count("set_predicate(") == 1, code


def test_no_predicate_emits_no_set_predicate():
    code = _generate_orch_full_pipeline(_NoPredicate)
    assert "set_predicate" not in code
    assert "L0TaskPredicate" not in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
