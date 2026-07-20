# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Orchestration-codegen tests for the dispatch predicate.

Drives ``pl.spmd_submit(..., predicate=(rc[0, 0] > 0))`` through the full
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
                self.expert,
                x,
                out,
                core_num=1,
                deps=[g_tid],
                predicate=(rc[0, 0] > 0),
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


# ---------------------------------------------------------------------------
# Operator / operand-order coverage
# ---------------------------------------------------------------------------
#
# The runtime predicate always stores `operand OP target` with the tensor on the
# left, so codegen flips the operator whenever the constant is written on the
# left (`0 < t[i]` records LT in the IR but must emit GT). That flip is
# load-bearing for ordinary code — the pipeline canonicalizes comparisons, so a
# plainly-written `t[i] > 3` can reach codegen in mirrored form — yet a single
# `PredicateOp::GT` assertion cannot catch a wrong flip table: dropping the
# Ge/Le flip still leaves such a suite green while emitting LE for `>=`, which
# retires exactly the tasks that should dispatch.
#
# Cover every (operator, operand order) combination end-to-end instead.

_PREDICATE_SRC = """
import pypto.language as pl


@pl.program
class Prog:
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
                self.expert, x, out, core_num=1, deps=[g_tid], predicate=({PRED})
            )
        return out
"""


def _emit_predicate(predicate_src: str) -> str:
    """Compile a program whose expert submit carries ``predicate_src``."""
    return _generate_orch_full_pipeline(pl.parse_program(_PREDICATE_SRC.replace("{PRED}", predicate_src)))


@pytest.mark.parametrize(
    "written,expected_op",
    [
        # Tensor on the left — recorded as written.
        ("rc[1, 2] == 3", "EQ"),
        ("rc[1, 2] != 3", "NE"),
        ("rc[1, 2] > 3", "GT"),
        ("rc[1, 2] < 3", "LT"),
        ("rc[1, 2] >= 3", "GE"),
        ("rc[1, 2] <= 3", "LE"),
        # Constant on the left — codegen must flip so the tensor is the operand.
        ("3 == rc[1, 2]", "EQ"),
        ("3 != rc[1, 2]", "NE"),
        ("3 > rc[1, 2]", "LT"),
        ("3 < rc[1, 2]", "GT"),
        ("3 >= rc[1, 2]", "LE"),
        ("3 <= rc[1, 2]", "GE"),
    ],
)
def test_operator_and_operand_order_mapping(written, expected_op):
    code = _emit_predicate(written)
    assert f".op = PredicateOp::{expected_op};" in code, (
        f"predicate=({written}) should emit PredicateOp::{expected_op}\n{code}"
    )
    assert ".target = 3;" in code, code


def test_index_order_is_preserved():
    """Distinct indices — equal ones cannot catch a transposed emission."""
    code = _emit_predicate("rc[1, 2] > 0")
    assert ".operand.indices[0] = 1;" in code, code
    assert ".operand.indices[1] = 2;" in code, code
    assert ".operand.ndims = 2;" in code, code


def test_set_predicate_precedes_task_submit():
    """The predicate must be installed before the task is submitted.

    ``rt_submit_*`` resolves ``args.predicate()`` internally, so a
    ``set_predicate`` emitted afterwards would leave ``op == NONE`` — the task
    dispatches unconditionally and the predicate silently does nothing, which a
    substring-only assertion would not catch.
    """
    code = _emit_predicate("rc[1, 2] > 0")
    set_pred_at = code.index(".set_predicate(")
    submit_after = code.index("rt_submit_", set_pred_at)
    assert set_pred_at < submit_after, "set_predicate must be emitted before the task's rt_submit_*"
