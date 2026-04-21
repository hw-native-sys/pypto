# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for torch_codegen cooperative scheduler on bidirectional / loop+sync IR.

Background
----------
The original ``torch_codegen`` cross-core support modeled a pipe as a shared
``deque`` and emitted Group functions as ``aic_fn(); aiv_fn();`` (one side
fully runs before the other).  That works for trivial unidirectional patterns
where pop never precedes its matching push, but it deadlocks (``IndexError``
on empty deque) the moment the IR contains:

  * **bidirectional** flow (V→C and C→V on the same Group), or
  * **same-side feedback** (a single function does both ``tpush`` and
    ``tpop`` on the same pipe direction).

This file pins the cooperative-scheduler path that handles those cases:

  1. ``visit_program`` detects when a Group needs the scheduler.
  2. AIC/AIV members of such Groups are emitted as Python *generators* that
     ``yield`` ``_WaitPop`` / ``_WaitPush`` at each pipe sync point.
  3. The Group function emits ``_run_scheduler([(name, gen, sb), ...])`` which
     round-robin-advances each generator, suspending it on empty pipes and
     resuming it when its peer pushes data.

These tests construct hand-built IR (no compiler lowering involved) so that
failures point directly at the codegen / scheduler layer.
"""

from collections import deque

import pytest
import torch
from pypto import DataType, ir
from pypto.debug import torch_codegen

# ---------------------------------------------------------------------------
# IR construction helpers
# ---------------------------------------------------------------------------


def _span():
    return ir.Span.unknown()


def _tile(name: str, shape: list[int]) -> ir.Var:
    return ir.Var(name, ir.TileType(shape, DataType.FP32), _span())


def _tensor(name: str, shape: list[int]) -> ir.Var:
    return ir.Var(name, ir.TensorType(shape, DataType.FP32), _span())


def _i(val: int) -> ir.ConstInt:
    return ir.ConstInt(val, DataType.INT64, _span())


# ---------------------------------------------------------------------------
# Bidirectional: AIV pushes A → AIC pops A, matmul(A, B), pushes result
# back → AIV pops result, adds residual, stores.
#
# Failure mode under the legacy emitter:
#   Group runs aiv_fn() to completion first.  After aiv_fn pushes A it
#   immediately tries to pop from the to_aiv pipe (the result coming back
#   from AIC), but AIC has not run yet → ``IndexError: pop from empty deque``.
# ---------------------------------------------------------------------------


def _build_bidirectional_program():
    shape = [4, 4]

    # AIV: push A, pop result, store(result + residual)
    a = _tile("a", shape)
    residual = _tile("residual", shape)
    out = _tensor("out", shape)
    push_a = ir.create_op_call("tile.tpush_to_aic", [a], {"split": 0}, _span())
    pop_result = ir.Call(
        ir.get_op("tile.tpop_from_aic"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    add_r = _tile("add_r", shape)
    add_call = ir.create_op_call("tile.add", [pop_result, residual], _span())
    offsets = ir.MakeTuple([_i(0), _i(0)], _span())
    store_call = ir.create_op_call("tile.store", [add_r, offsets, out], _span())
    aiv_body = ir.SeqStmts(
        [
            ir.EvalStmt(push_a, _span()),
            ir.AssignStmt(add_r, add_call, _span()),
            ir.EvalStmt(store_call, _span()),
        ],
        _span(),
    )
    aiv_func = ir.Function("aiv_fn", [a, residual, out], [], aiv_body, _span(), type=ir.FunctionType.AIV)

    # AIC: pop A, matmul(A, B), push result
    b = _tile("b", shape)
    pop_a = ir.Call(
        ir.get_op("tile.tpop_from_aiv"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    mm = _tile("mm", shape)
    mm_call = ir.create_op_call("tile.matmul", [pop_a, b], _span())
    push_mm = ir.create_op_call("tile.tpush_to_aiv", [mm], {"split": 0}, _span())
    aic_body = ir.SeqStmts([ir.AssignStmt(mm, mm_call, _span()), ir.EvalStmt(push_mm, _span())], _span())
    aic_func = ir.Function("aic_fn", [b], [], aic_body, _span(), type=ir.FunctionType.AIC)

    aiv_call = ir.Call(ir.GlobalVar("aiv_fn"), [a, residual, out], _span())
    aic_call = ir.Call(ir.GlobalVar("aic_fn"), [b], _span())
    group_body = ir.SeqStmts([ir.EvalStmt(aiv_call, _span()), ir.EvalStmt(aic_call, _span())], _span())
    group_func = ir.Function(
        "bidir_grp",
        [a, b, residual, out],
        [],
        group_body,
        _span(),
        type=ir.FunctionType.Group,
    )

    program = ir.Program([aiv_func, aic_func, group_func], "bidir_test", _span())
    specs = [
        ("a", shape),
        ("b", shape),
        ("residual", shape),
        ("out", shape),
    ]

    def golden(tensors):
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"]) + tensors["residual"]

    return program, specs, golden


# ---------------------------------------------------------------------------
# Same-side feedback: a single AIV function pushes to AIC then pops back from
# AIC on the same pipe direction.  This forces same-side feedback detection
# (one function uses both tpush_to_aic and tpop_from_aiv) and exercises the
# scheduler under a single-task-pair feedback pattern.
# ---------------------------------------------------------------------------


def _build_same_side_feedback_program():
    """AIV does: push(A) -> AIC pops, doubles, pushes -> AIV pops, stores."""
    shape = [4, 4]

    a = _tile("a", shape)
    out = _tensor("out", shape)
    push_a = ir.create_op_call("tile.tpush_to_aic", [a], {"split": 0}, _span())
    pop_back = ir.Call(
        ir.get_op("tile.tpop_from_aic"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    doubled = _tile("doubled", shape)
    bind_doubled = ir.AssignStmt(doubled, pop_back, _span())
    offsets = ir.MakeTuple([_i(0), _i(0)], _span())
    store_call = ir.create_op_call("tile.store", [doubled, offsets, out], _span())
    aiv_body = ir.SeqStmts(
        [
            ir.EvalStmt(push_a, _span()),
            bind_doubled,
            ir.EvalStmt(store_call, _span()),
        ],
        _span(),
    )
    aiv_func = ir.Function("aiv_fb", [a, out], [], aiv_body, _span(), type=ir.FunctionType.AIV)

    pop_a = ir.Call(
        ir.get_op("tile.tpop_from_aiv"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    captured = _tile("captured", shape)
    twice = _tile("twice", shape)
    twice_call = ir.create_op_call("tile.add", [captured, captured], _span())
    push_twice = ir.create_op_call("tile.tpush_to_aiv", [twice], {"split": 0}, _span())
    aic_body = ir.SeqStmts(
        [
            ir.AssignStmt(captured, pop_a, _span()),
            ir.AssignStmt(twice, twice_call, _span()),
            ir.EvalStmt(push_twice, _span()),
        ],
        _span(),
    )
    aic_func = ir.Function("aic_fb", [], [], aic_body, _span(), type=ir.FunctionType.AIC)

    aiv_call = ir.Call(ir.GlobalVar("aiv_fb"), [a, out], _span())
    aic_call = ir.Call(ir.GlobalVar("aic_fb"), [], _span())
    group_body = ir.SeqStmts([ir.EvalStmt(aiv_call, _span()), ir.EvalStmt(aic_call, _span())], _span())
    group_func = ir.Function(
        "fb_grp",
        [a, out],
        [],
        group_body,
        _span(),
        type=ir.FunctionType.Group,
    )

    program = ir.Program([aiv_func, aic_func, group_func], "feedback_test", _span())
    specs = [("a", shape), ("out", shape)]

    def golden(tensors):
        tensors["out"][:] = tensors["a"] + tensors["a"]

    return program, specs, golden


# ---------------------------------------------------------------------------
# Test runner helpers
# ---------------------------------------------------------------------------


def _build_inputs(specs, seed=42):
    torch.manual_seed(seed)
    out: dict = {}
    for name, shape in specs:
        if name == "out":
            out[name] = torch.zeros(shape, dtype=torch.float32)
        else:
            out[name] = torch.randn(shape, dtype=torch.float32)
    return out


def _run_codegen(program, specs, seed=42):
    code = torch_codegen(program, check_shapes=False)
    tensors = _build_inputs(specs, seed=seed)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    # Reset pipes (the generated code uses module-level deques).
    ns["_pipes"] = {"to_aiv": deque(), "to_aic": deque()}
    args = [tensors[name] for name, _ in specs]
    assert "run" in ns and callable(ns["run"]), "torch_codegen entry point missing"
    ns["run"](*args)
    return tensors, code


def _run_golden(specs, golden, seed=42):
    tensors = _build_inputs(specs, seed=seed)
    golden(tensors)
    return tensors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bidirectional_emits_scheduler_and_matches_golden():
    """V↔C bidirectional Group must use the cooperative scheduler.

    Pins both the codegen contract (scheduler markers present, generator-style
    helpers used) and the runtime numerical correctness against a torch
    golden.  Without the scheduler this case raises IndexError.
    """
    program, specs, golden = _build_bidirectional_program()
    tensors, code = _run_codegen(program, specs)

    # Codegen contract: scheduler-form markers must appear.
    assert "_run_scheduler([" in code, "scheduler call not emitted for bidirectional Group"
    assert "_reset_pipes()" in code, "pipe reset not emitted in scheduled Group"
    assert "yield from _tpush_to_aic_g" in code, "AIV tpush_to_aic not in generator form"
    assert "yield from _tpop_from_aiv_g" in code, "AIC tpop_from_aiv not in generator form"
    assert "yield from _tpush_to_aiv_g" in code, "AIC tpush_to_aiv not in generator form"
    assert "yield from _tpop_from_aic_g" in code, "AIV tpop_from_aic not in generator form"

    # Numerical correctness vs torch golden.
    gd = _run_golden(specs, golden)
    diff = (tensors["out"] - gd["out"]).abs().max().item()
    assert torch.allclose(tensors["out"], gd["out"], rtol=1e-5, atol=1e-5), (
        f"bidirectional: max abs diff = {diff:.3e}"
    )


def test_same_side_feedback_uses_scheduler_and_matches_golden():
    """AIV does push+pop on the same pipe direction; needs the scheduler."""
    program, specs, golden = _build_same_side_feedback_program()
    tensors, code = _run_codegen(program, specs)

    assert "_run_scheduler([" in code, "scheduler call not emitted for same-side feedback"
    assert "yield from _tpush_to_aic_g" in code
    assert "yield from _tpop_from_aic_g" in code

    gd = _run_golden(specs, golden)
    diff = (tensors["out"] - gd["out"]).abs().max().item()
    assert torch.allclose(tensors["out"], gd["out"], rtol=1e-5, atol=1e-5), (
        f"same-side feedback: max abs diff = {diff:.3e}"
    )


def test_unidirectional_v2c_keeps_legacy_emission():
    """Single-direction V→C (no feedback) must still use the legacy path.

    Regression guard: the scheduler must not be triggered for trivial
    unidirectional Groups so that the existing 60+ codegen tests are
    unaffected.
    """
    shape = [4, 4]
    a = _tile("a", shape)
    b = _tile("b", shape)
    result = _tile("result", shape)
    matmul_call = ir.create_op_call("tile.matmul", [a, b], _span())
    push_call = ir.create_op_call("tile.tpush_to_aiv", [result], {"split": 0}, _span())
    aic_body = ir.SeqStmts(
        [ir.AssignStmt(result, matmul_call, _span()), ir.EvalStmt(push_call, _span())],
        _span(),
    )
    aic_func = ir.Function("aic_only", [a, b], [], aic_body, _span(), type=ir.FunctionType.AIC)

    out = _tensor("out", shape)
    pop_call = ir.Call(
        ir.get_op("tile.tpop_from_aic"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    popped = _tile("popped", shape)
    offsets = ir.MakeTuple([_i(0), _i(0)], _span())
    store_call = ir.create_op_call("tile.store", [popped, offsets, out], _span())
    aiv_body = ir.SeqStmts(
        [ir.AssignStmt(popped, pop_call, _span()), ir.EvalStmt(store_call, _span())],
        _span(),
    )
    aiv_func = ir.Function("aiv_only", [out], [], aiv_body, _span(), type=ir.FunctionType.AIV)

    aic_call = ir.Call(ir.GlobalVar("aic_only"), [a, b], _span())
    aiv_call = ir.Call(ir.GlobalVar("aiv_only"), [out], _span())
    group_body = ir.SeqStmts([ir.EvalStmt(aic_call, _span()), ir.EvalStmt(aiv_call, _span())], _span())
    group_func = ir.Function("uni_grp", [a, b, out], [], group_body, _span(), type=ir.FunctionType.Group)

    prog = ir.Program([aic_func, aiv_func, group_func], "uni_test", _span())
    code = torch_codegen(prog, check_shapes=False)

    # The preamble defines _run_scheduler; check for the call-site form ([) only.
    assert "_run_scheduler([" not in code, "unidirectional V2C must NOT trigger the scheduler"
    assert "yield from _tpush_to_aiv_g" not in code, (
        "unidirectional V2C must use legacy non-generator helpers"
    )
    # Legacy helpers still expected.
    assert "_tpush_to_aiv(_pipes['to_aiv']," in code
    assert "_tpop_from_aic(_pipes['to_aiv']," in code


def test_scheduler_deadlock_raises():
    """Two AIC/AIV functions that both pop before either pushes must deadlock."""
    shape = [4, 4]
    a = _tile("a", shape)
    b = _tile("b", shape)
    out = _tensor("out", shape)

    pop_back = ir.Call(
        ir.get_op("tile.tpop_from_aic"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    pop_v = _tile("pop_v", shape)
    push_pop_v = ir.create_op_call("tile.tpush_to_aic", [pop_v], {"split": 0}, _span())
    aiv_body = ir.SeqStmts(
        [ir.AssignStmt(pop_v, pop_back, _span()), ir.EvalStmt(push_pop_v, _span())],
        _span(),
    )
    aiv_func = ir.Function("aiv_dl", [a, out], [], aiv_body, _span(), type=ir.FunctionType.AIV)

    pop_fwd = ir.Call(
        ir.get_op("tile.tpop_from_aiv"),
        [],
        {"split": 0},
        ir.TileType(shape, DataType.FP32),
        _span(),
    )
    pop_c = _tile("pop_c", shape)
    push_pop_c = ir.create_op_call("tile.tpush_to_aiv", [pop_c], {"split": 0}, _span())
    aic_body = ir.SeqStmts(
        [ir.AssignStmt(pop_c, pop_fwd, _span()), ir.EvalStmt(push_pop_c, _span())],
        _span(),
    )
    aic_func = ir.Function("aic_dl", [b], [], aic_body, _span(), type=ir.FunctionType.AIC)

    aiv_call = ir.Call(ir.GlobalVar("aiv_dl"), [a, out], _span())
    aic_call = ir.Call(ir.GlobalVar("aic_dl"), [b], _span())
    group_body = ir.SeqStmts([ir.EvalStmt(aiv_call, _span()), ir.EvalStmt(aic_call, _span())], _span())
    group_func = ir.Function("dl_grp", [a, b, out], [], group_body, _span(), type=ir.FunctionType.Group)

    prog = ir.Program([aiv_func, aic_func, group_func], "deadlock_test", _span())
    code = torch_codegen(prog, check_shapes=False)
    assert "_run_scheduler([" in code, "deadlock test should still go through scheduler"

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    ns["_pipes"] = {"to_aiv": deque(), "to_aic": deque()}
    ta = torch.randn(*shape)
    tb = torch.randn(*shape)
    tout = torch.zeros(*shape)
    with pytest.raises(RuntimeError, match="deadlock"):
        ns["run"](ta, tb, tout)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
