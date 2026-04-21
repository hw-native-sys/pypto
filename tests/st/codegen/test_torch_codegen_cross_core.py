# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for torch_codegen on hand-built cross-core (tpush/tpop) IR.

Constructs minimal AIC + AIV + Group programs that directly use
``tile.tpush_to_aiv`` / ``tile.tpop_from_aic`` (V2C direction) and
``tile.tpush_to_aic`` / ``tile.tpop_from_aiv`` (C2V direction) for all three
split modes (NoSplit / UpDown / LeftRight), then compares torch_codegen output
against a hand-written torch golden.

Why hand-built IR rather than ``pl.SplitMode.*`` lowering:

* The ``tests/st/runtime/test_cross_core.py`` ST goes through the full lowering
  pipeline (``pl.chunked_loop_optimizer`` → mixed-kernel expansion → tpush/tpop)
  and validates the *end-to-end sim* output.  It exercises tpush/tpop indirectly
  but never compares torch_codegen against a reference per-tile.
* This test pins the torch_codegen contract for the four cross-core ops
  (split=0/1/2 on each of the four ops) at the level the codegen actually
  emits.  Failures here point at the codegen helpers / split semantics
  directly instead of at the lowering pipeline.
"""

from collections import deque

import pytest
import torch
from pypto import DataType, ir
from pypto.debug import torch_codegen

# ---------------------------------------------------------------------------
# IR builders
# ---------------------------------------------------------------------------


def _span():
    return ir.Span.unknown()


def _tile(name: str, shape: list[int]) -> ir.Var:
    return ir.Var(name, ir.TileType(shape, DataType.FP32), _span())


def _tensor(name: str, shape: list[int]) -> ir.Var:
    return ir.Var(name, ir.TensorType(shape, DataType.FP32), _span())


def _int(val: int) -> ir.ConstInt:
    return ir.ConstInt(val, DataType.INT64, _span())


def _aiv_subblock_shape(full_shape: list[int], split: int) -> list[int]:
    """Subblock shape that each AIV subblock processes after a tpop_from_aic."""
    if split == 1:  # UpDown → row half
        return [full_shape[0] // 2, full_shape[1]]
    if split == 2:  # LeftRight → column half
        return [full_shape[0], full_shape[1] // 2]
    return list(full_shape)


def _build_v2c_program(split: int):
    """Build V2C: AIC matmul → tpush_to_aiv → AIV tpop + add residual → store.

    For ``split == 0`` the AIV consumer receives the full tile, adds residual,
    and stores it.

    For ``split > 0`` the program runs in scheduler mode: AIC pushes the
    matmul result split, both AIV subblocks pop their half and add the
    matching half of residual (sliced via ``tile.get_subblock_idx``), then
    push the half-result back to AIC.  AIC pops with the same split (which
    reassembles via ``cat``) and stores the full tile, so the final output
    equals ``matmul(a, b) + residual`` regardless of split.
    """
    full_shape = [4, 4]

    if split == 0:
        # Legacy unidirectional path (no scheduler).
        a = _tile("a", full_shape)
        b = _tile("b", full_shape)
        result = _tile("result", full_shape)
        matmul_call = ir.create_op_call("tile.matmul", [a, b], _span())
        push_call = ir.create_op_call("tile.tpush_to_aiv", [result], {"split": split}, _span())
        aic_body = ir.SeqStmts(
            [
                ir.AssignStmt(result, matmul_call, _span()),
                ir.EvalStmt(push_call, _span()),
            ],
            _span(),
        )
        aic_func = ir.Function("aic_matmul", [a, b], [], aic_body, _span(), type=ir.FunctionType.AIC)

        residual = _tile("residual", full_shape)
        out = _tensor("out", full_shape)
        pop_call = ir.Call(
            ir.get_op("tile.tpop_from_aic"),
            [],
            {"split": 0},
            ir.TileType(full_shape, DataType.FP32),
            _span(),
        )
        add_result = _tile("add_result", full_shape)
        add_call = ir.create_op_call("tile.add", [pop_call, residual], _span())
        offsets = ir.MakeTuple([_int(0), _int(0)], _span())
        store_call = ir.create_op_call("tile.store", [add_result, offsets, out], _span())
        aiv_body = ir.SeqStmts(
            [
                ir.AssignStmt(add_result, add_call, _span()),
                ir.EvalStmt(store_call, _span()),
            ],
            _span(),
        )
        aiv_func = ir.Function(
            "aiv_add_residual",
            [residual, out],
            [],
            aiv_body,
            _span(),
            type=ir.FunctionType.AIV,
        )

        aic_call = ir.Call(ir.GlobalVar("aic_matmul"), [a, b], _span())
        aiv_call = ir.Call(ir.GlobalVar("aiv_add_residual"), [residual, out], _span())
        group_body = ir.SeqStmts(
            [ir.EvalStmt(aic_call, _span()), ir.EvalStmt(aiv_call, _span())],
            _span(),
        )
        group_func = ir.Function(
            "v2c_matmul_add_group",
            [a, b, residual, out],
            [],
            group_body,
            _span(),
            type=ir.FunctionType.Group,
        )
        program = ir.Program([aic_func, aiv_func, group_func], "v2c_cross_core_test", _span())

        specs = [("a", full_shape), ("b", full_shape), ("residual", full_shape), ("out", full_shape)]

        def golden(tensors: dict[str, torch.Tensor]) -> None:
            tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"]) + tensors["residual"]

        return program, specs, golden

    # split > 0: bidirectional roundtrip with reassembly at AIC.
    half_shape = _aiv_subblock_shape(full_shape, split)

    # AIC: matmul -> push split -> pop reassembled -> store full.
    a = _tile("a", full_shape)
    b = _tile("b", full_shape)
    out = _tensor("out", full_shape)
    matmul_call = ir.create_op_call("tile.matmul", [a, b], _span())
    result = _tile("result", full_shape)
    push_call = ir.create_op_call("tile.tpush_to_aiv", [result], {"split": split}, _span())
    pop_back_call = ir.Call(
        ir.get_op("tile.tpop_from_aiv"),
        [],
        {"split": split},
        ir.TileType(full_shape, DataType.FP32),
        _span(),
    )
    reassembled = _tile("reassembled", full_shape)
    offsets = ir.MakeTuple([_int(0), _int(0)], _span())
    store_call = ir.create_op_call("tile.store", [reassembled, offsets, out], _span())
    aic_body = ir.SeqStmts(
        [
            ir.AssignStmt(result, matmul_call, _span()),
            ir.EvalStmt(push_call, _span()),
            ir.AssignStmt(reassembled, pop_back_call, _span()),
            ir.EvalStmt(store_call, _span()),
        ],
        _span(),
    )
    aic_func = ir.Function("aic_matmul", [a, b, out], [], aic_body, _span(), type=ir.FunctionType.AIC)

    # AIV (runs once per subblock): pop half, add matching half of residual,
    # push back.  Per-subblock offset uses tile.get_subblock_idx.
    if split == 1:
        offset_dim0 = ir.Mul(
            ir.create_op_call("tile.get_subblock_idx", [], _span()),
            _int(half_shape[0]),
            DataType.INT64,
            _span(),
        )
        slice_offsets = ir.MakeTuple([offset_dim0, _int(0)], _span())
    else:
        offset_dim1 = ir.Mul(
            ir.create_op_call("tile.get_subblock_idx", [], _span()),
            _int(half_shape[1]),
            DataType.INT64,
            _span(),
        )
        slice_offsets = ir.MakeTuple([_int(0), offset_dim1], _span())

    residual_full = _tile("residual", full_shape)
    pop_call = ir.Call(
        ir.get_op("tile.tpop_from_aic"),
        [],
        {"split": split},
        ir.TileType(half_shape, DataType.FP32),
        _span(),
    )
    popped_half = _tile("popped_half", half_shape)
    half_shape_tuple = ir.MakeTuple([_int(d) for d in half_shape], _span())
    residual_half_call = ir.create_op_call(
        "tile.slice", [residual_full, half_shape_tuple, slice_offsets], _span()
    )
    residual_half = _tile("residual_half", half_shape)
    add_half_call = ir.create_op_call("tile.add", [popped_half, residual_half], _span())
    add_half = _tile("add_half", half_shape)
    push_back_call = ir.create_op_call("tile.tpush_to_aic", [add_half], {"split": split}, _span())
    aiv_body = ir.SeqStmts(
        [
            ir.AssignStmt(popped_half, pop_call, _span()),
            ir.AssignStmt(residual_half, residual_half_call, _span()),
            ir.AssignStmt(add_half, add_half_call, _span()),
            ir.EvalStmt(push_back_call, _span()),
        ],
        _span(),
    )
    aiv_func = ir.Function(
        "aiv_add_residual",
        [residual_full],
        [],
        aiv_body,
        _span(),
        type=ir.FunctionType.AIV,
    )

    aic_call = ir.Call(ir.GlobalVar("aic_matmul"), [a, b, out], _span())
    aiv_call = ir.Call(ir.GlobalVar("aiv_add_residual"), [residual_full], _span())
    group_body = ir.SeqStmts(
        [ir.EvalStmt(aic_call, _span()), ir.EvalStmt(aiv_call, _span())],
        _span(),
    )
    group_func = ir.Function(
        "v2c_matmul_add_group",
        [a, b, residual_full, out],
        [],
        group_body,
        _span(),
        type=ir.FunctionType.Group,
    )
    program = ir.Program([aic_func, aiv_func, group_func], "v2c_cross_core_test", _span())

    specs = [("a", full_shape), ("b", full_shape), ("residual", full_shape), ("out", full_shape)]

    def golden(tensors: dict[str, torch.Tensor]) -> None:
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"]) + tensors["residual"]

    return program, specs, golden


def _build_c2v_program(split: int):
    """Build C2V: AIV tpush_to_aic(a) → AIC tpop_from_aiv → matmul(b) → store.

    For ``split == 0`` the AIV pushes the full tile and AIC pops the full tile.

    For ``split > 0`` each AIV subblock pushes only its own slice of ``a``
    (selected via ``tile.get_subblock_idx`` and ``tile.slice``); AIC's
    ``tpop_from_aiv`` with the same split reassembles the two halves into a
    full tile via ``cat``.  The matmul output therefore equals
    ``matmul(a, b)`` for every split mode.
    """
    full_shape = [4, 4]

    if split == 0:
        a = _tile("a", full_shape)
        push_call = ir.create_op_call("tile.tpush_to_aic", [a], {"split": 0}, _span())
        aiv_body = ir.SeqStmts([ir.EvalStmt(push_call, _span())], _span())
        aiv_func = ir.Function("aiv_push_a", [a], [], aiv_body, _span(), type=ir.FunctionType.AIV)
    else:
        # AIV (per-subblock): slice ``a`` along the split axis at offset
        # subblock_idx * half_size, push that half to AIC.
        half_shape = _aiv_subblock_shape(full_shape, split)
        a = _tile("a", full_shape)
        if split == 1:
            offset_dim0 = ir.Mul(
                ir.create_op_call("tile.get_subblock_idx", [], _span()),
                _int(half_shape[0]),
                DataType.INT64,
                _span(),
            )
            slice_offsets = ir.MakeTuple([offset_dim0, _int(0)], _span())
        else:
            offset_dim1 = ir.Mul(
                ir.create_op_call("tile.get_subblock_idx", [], _span()),
                _int(half_shape[1]),
                DataType.INT64,
                _span(),
            )
            slice_offsets = ir.MakeTuple([_int(0), offset_dim1], _span())
        half_shape_tuple = ir.MakeTuple([_int(d) for d in half_shape], _span())
        slice_call = ir.create_op_call("tile.slice", [a, half_shape_tuple, slice_offsets], _span())
        a_half = _tile("a_half", half_shape)
        push_call = ir.create_op_call("tile.tpush_to_aic", [a_half], {"split": split}, _span())
        aiv_body = ir.SeqStmts(
            [
                ir.AssignStmt(a_half, slice_call, _span()),
                ir.EvalStmt(push_call, _span()),
            ],
            _span(),
        )
        aiv_func = ir.Function("aiv_push_a", [a], [], aiv_body, _span(), type=ir.FunctionType.AIV)

    # AIC: pop full tile (split-aware reassembly inside the helper); matmul; store.
    b = _tile("b", full_shape)
    out = _tensor("out", full_shape)
    pop_call = ir.Call(
        ir.get_op("tile.tpop_from_aiv"),
        [],
        {"split": split},
        ir.TileType(full_shape, DataType.FP32),
        _span(),
    )
    mm = _tile("mm", full_shape)
    mm_call = ir.create_op_call("tile.matmul", [pop_call, b], _span())
    offsets = ir.MakeTuple([_int(0), _int(0)], _span())
    store_call = ir.create_op_call("tile.store", [mm, offsets, out], _span())
    aic_body = ir.SeqStmts(
        [
            ir.AssignStmt(mm, mm_call, _span()),
            ir.EvalStmt(store_call, _span()),
        ],
        _span(),
    )
    aic_func = ir.Function(
        "aic_matmul_b",
        [b, out],
        [],
        aic_body,
        _span(),
        type=ir.FunctionType.AIC,
    )

    aiv_call = ir.Call(ir.GlobalVar("aiv_push_a"), [a], _span())
    aic_call = ir.Call(ir.GlobalVar("aic_matmul_b"), [b, out], _span())
    group_body = ir.SeqStmts(
        [ir.EvalStmt(aiv_call, _span()), ir.EvalStmt(aic_call, _span())],
        _span(),
    )
    group_func = ir.Function(
        "c2v_push_matmul_group",
        [a, b, out],
        [],
        group_body,
        _span(),
        type=ir.FunctionType.Group,
    )

    program = ir.Program([aiv_func, aic_func, group_func], "c2v_cross_core_test", _span())

    specs = [("a", full_shape), ("b", full_shape), ("out", full_shape)]

    def golden(tensors: dict[str, torch.Tensor]) -> None:
        tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"])

    return program, specs, golden


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_inputs(specs: list[tuple[str, list[int]]], seed: int = 42) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    for name, shape in specs:
        if name == "out":
            out[name] = torch.zeros(shape, dtype=torch.float32)
        else:
            out[name] = torch.randn(shape, dtype=torch.float32)
    return out


def _run_codegen(program, specs, seed=42) -> dict[str, torch.Tensor]:
    code = torch_codegen(program, check_shapes=False)
    tensors = _build_inputs(specs, seed=seed)
    ns: dict = {}
    exec(code, ns)  # noqa: S102
    # Reset pipe state per test (the generated code creates module-level deques).
    ns["_pipes"] = {"to_aiv": deque(), "to_aic": deque()}
    args = [tensors[name] for name, _ in specs]
    assert "run" in ns and callable(ns["run"]), "torch_codegen entry point `run(...)` was not generated"
    ns["run"](*args)
    return tensors


def _run_golden(specs, golden, seed=42) -> dict[str, torch.Tensor]:
    tensors = _build_inputs(specs, seed=seed)
    golden(tensors)
    return tensors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "label"),
    [(0, "nosplit"), (1, "updown"), (2, "leftright")],
)
def test_v2c_tpush_tpop_codegen_vs_golden(split, label):
    """V2C (tpush_to_aiv / tpop_from_aic) torch_codegen ≈ torch golden."""
    program, specs, golden = _build_v2c_program(split)

    code = torch_codegen(program, check_shapes=False)
    # Pin codegen contract for the V2C ops at this split.
    if split == 0:
        # Legacy unidirectional path: direct helper calls.
        assert "_tpush_to_aiv(_pipes['to_aiv'], result, 0)" in code, (
            f"[{label}] tpush_to_aiv split kwarg not forwarded"
        )
        assert "_tpop_from_aic(_pipes['to_aiv'], 0)" in code, (
            f"[{label}] tpop_from_aic split kwarg not forwarded"
        )
    else:
        # Bidirectional split>0 path: scheduler-mode generator wrappers.
        assert "_run_scheduler([" in code, f"[{label}] expected scheduler emission for split>0"
        assert f"_tpush_to_aiv_g(_pipes['to_aiv'], result, {split})" in code
        assert f"_tpop_from_aic_g(_pipes['to_aiv'], {split})" in code
        assert f"_tpop_from_aiv_g(_pipes['to_aic'], {split})" in code

    cg = _run_codegen(program, specs)
    gd = _run_golden(specs, golden)
    diff = (cg["out"] - gd["out"]).abs()
    assert torch.allclose(cg["out"], gd["out"], rtol=1e-5, atol=1e-5), (
        f"[{label}] V2C codegen vs golden max abs diff = {diff.max().item():.3e}"
    )


@pytest.mark.parametrize(
    ("split", "label"),
    [(0, "nosplit"), (1, "updown"), (2, "leftright")],
)
def test_c2v_tpush_tpop_codegen_vs_golden(split, label):
    """C2V (tpush_to_aic / tpop_from_aiv) torch_codegen ≈ torch golden.

    For ``split == 0`` AIV pushes the full tile and AIC pops the full tile.
    For ``split > 0`` each AIV subblock pushes its own slice of ``a`` and
    AIC's ``tpop_from_aiv`` reassembles via ``cat``; the matmul output
    therefore equals ``matmul(a, b)`` for every split mode.
    """
    program, specs, golden = _build_c2v_program(split)

    code = torch_codegen(program, check_shapes=False)
    if split == 0:
        assert "_tpush_to_aic(_pipes['to_aic'], a, 0)" in code, (
            f"[{label}] tpush_to_aic split kwarg not forwarded"
        )
        assert "_tpop_from_aiv(_pipes['to_aic'], 0)" in code, (
            f"[{label}] tpop_from_aiv split kwarg not forwarded"
        )
    else:
        assert "_run_scheduler([" in code, f"[{label}] expected scheduler emission for split>0"
        assert f"_tpush_to_aic_g(_pipes['to_aic'], a_half, {split})" in code
        assert f"_tpop_from_aiv_g(_pipes['to_aic'], {split})" in code

    cg = _run_codegen(program, specs)
    gd = _run_golden(specs, golden)
    diff = (cg["out"] - gd["out"]).abs()
    assert torch.allclose(cg["out"], gd["out"], rtol=1e-5, atol=1e-5), (
        f"[{label}] C2V codegen vs golden max abs diff = {diff.max().item():.3e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
