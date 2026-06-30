# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""System tests for torch codegen on cross-core tpush/tpop scenarios.

Generates executable PyTorch code from cross-core IR via torch_codegen,
runs it with test tensors, and compares outputs to the golden reference.
"""

import tempfile

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, platform_to_backend
from pypto import ir as _ir
from pypto.backend import BackendType, pto_backend, reset_for_testing, set_backend_type
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

M = 32
K = 64
N = 512
N_BLOCK = 64
N_BLOCKS = N // N_BLOCK


@pl.program
class V2CUDProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[32, 32], pl.FP32],
        b: pl.Tensor[[32, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            a_plus_b = pl.add(a, b)
            sub = pl.sub(a, b)
            out = pl.matmul(a_plus_b, sub)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class V2CLRProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[32, 32], pl.FP32],
        b: pl.Tensor[[32, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)],
        ):
            a_plus_b = pl.add(a, b)
            sub = pl.sub(a, b)
            out = pl.matmul(a_plus_b, sub)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class V2CNoSplitProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[32, 32], pl.FP32],
        b: pl.Tensor[[32, 32], pl.FP32],
        output: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            a_plus_b = pl.add(a, b)
            sub = pl.sub(a, b)
            out = pl.matmul(a_plus_b, sub)
            output = pl.assemble(output, out, [0, 0])
        return output


@pl.program
class C2VLRProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)],
        ):
            for nb in pl.range(0, N_BLOCKS):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


@pl.program
class C2VUDProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            for nb in pl.range(0, N_BLOCKS):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


@pl.program
class C2VNoSplitProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            for nb in pl.range(0, N_BLOCKS):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


@pl.program
class BiDirectUDProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
        ):
            for nb in pl.range(0, N_BLOCKS):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


@pl.program
class BiDirectLRProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimizations=[pl.split(pl.SplitMode.LEFT_RIGHT)],
        ):
            for nb in pl.range(0, N_BLOCKS):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


@pl.program
class BiDirectNoSplitProgram:
    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        c: pl.Tensor[[M, N], pl.FP32],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            for nb in pl.range(0, N_BLOCKS):
                n0 = nb * N_BLOCK
                c_prev = pl.slice(c, [M, N_BLOCK], [0, n0])
                a_add = pl.add(a, 1.0)
                b_chunk = pl.slice(b, [K, N_BLOCK], [0, n0])
                c_next = pl.add(c_prev, pl.matmul(a_add, b_chunk))
                c = pl.assemble(c, c_next, [0, n0])
        return c


def _build_v2c_tensors() -> dict[str, torch.Tensor]:
    return {
        "a": torch.randn(32, 32, dtype=torch.float32),
        "b": torch.randn(32, 32, dtype=torch.float32),
        "output": torch.zeros(32, 32, dtype=torch.float32),
    }


def _build_c2v_tensors() -> dict[str, torch.Tensor]:
    return {
        "a": torch.randn(M, K, dtype=torch.float32),
        "b": torch.randn(K, N, dtype=torch.float32),
        "c": torch.randn(M, N, dtype=torch.float32),
    }


def _golden_v2c(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors["output"][:] = torch.matmul(tensors["a"] + tensors["b"], tensors["a"] - tensors["b"])
    return tensors["output"]


def _golden_c2v(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    c_prev = tensors["c"].clone()
    tensors["c"][:] = c_prev + torch.matmul(tensors["a"], tensors["b"])
    return tensors["c"]


def _golden_bidirect(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    c_prev = tensors["c"].clone()
    tensors["c"][:] = c_prev + torch.matmul(tensors["a"] + 1, tensors["b"])
    return tensors["c"]


def _run_codegen_and_check(
    program: _ir.Program | _ir.Function,
    tensors: dict[str, torch.Tensor],
    arg_order: list[str],
    out_name: str,
    expected: torch.Tensor,
) -> None:
    code = torch_codegen(program, check_shapes=True)
    ns: dict = {}
    exec(code, ns)  # noqa: S102

    args = [tensors[name] for name in arg_order]
    result = ns["main"](*args)
    actual = tensors[out_name]

    assert torch.allclose(actual, expected, rtol=5e-2, atol=5e-2), (
        f"max abs diff = {(expected - actual).abs().max().item():.6e}"
    )
    if isinstance(result, torch.Tensor):
        assert torch.allclose(result, expected, rtol=5e-2, atol=5e-2), (
            f"returned tensor max abs diff = {(expected - result).abs().max().item():.6e}"
        )


def _run_codegen_after_default_pass_and_check(
    program: _ir.Program | _ir.Function,
    tensors: dict[str, torch.Tensor],
    arg_order: list[str],
    out_name: str,
    expected: torch.Tensor,
    platform: str,
) -> None:
    backend_type = platform_to_backend(platform)

    reset_for_testing()
    set_backend_type(backend_type)
    try:
        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        code = torch_codegen(transformed, check_shapes=True)
    finally:
        reset_for_testing()

    assert "_cross_core_rt.push_to_" in code
    assert "_cross_core_rt.pop_from_" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102

    args = [tensors[name] for name in arg_order]
    result = ns["main"](*args)
    actual = tensors[out_name]

    assert torch.allclose(actual, expected, rtol=5e-2, atol=5e-2), (
        f"max abs diff = {(expected - actual).abs().max().item():.6e}"
    )
    if isinstance(result, torch.Tensor):
        assert torch.allclose(result, expected, rtol=5e-2, atol=5e-2), (
            f"returned tensor max abs diff = {(expected - result).abs().max().item():.6e}"
        )


_SCENARIOS = [
    ("v2c_updown", V2CUDProgram, _build_v2c_tensors, ["a", "b", "output"], "output", _golden_v2c),
    ("v2c_leftright", V2CLRProgram, _build_v2c_tensors, ["a", "b", "output"], "output", _golden_v2c),
    ("v2c_nosplit", V2CNoSplitProgram, _build_v2c_tensors, ["a", "b", "output"], "output", _golden_v2c),
    ("c2v_leftright", C2VLRProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_c2v),
    ("c2v_updown", C2VUDProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_c2v),
    ("c2v_nosplit", C2VNoSplitProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_c2v),
    ("bidirect_updown", BiDirectUDProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_bidirect),
    ("bidirect_leftright", BiDirectLRProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_bidirect),
    ("bidirect_nosplit", BiDirectNoSplitProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_bidirect),
]


@pytest.mark.parametrize(
    ("_name", "program", "builder", "arg_order", "out_name", "golden_fn"),
    _SCENARIOS,
    ids=[s[0] for s in _SCENARIOS],
)
def test_cross_core_codegen_vs_golden(
    _name: str,
    program: _ir.Program | _ir.Function,
    builder,
    arg_order: list[str],
    out_name: str,
    golden_fn,
):
    """Torch codegen of cross-core scenarios should match golden references."""
    torch.manual_seed(42)
    base_tensors = builder()
    golden_tensors = {k: v.clone() for k, v in base_tensors.items()}
    expected = golden_fn(golden_tensors)
    codegen_tensors = {k: v.clone() for k, v in base_tensors.items()}
    _run_codegen_and_check(program, codegen_tensors, arg_order, out_name, expected)


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    ("_name", "program", "builder", "arg_order", "out_name", "golden_fn"),
    _SCENARIOS,
    ids=[s[0] for s in _SCENARIOS],
)
def test_cross_core_codegen_after_default_pass_vs_golden(
    platform: str,
    _name: str,
    program: _ir.Program | _ir.Function,
    builder,
    arg_order: list[str],
    out_name: str,
    golden_fn,
):
    """Pass-expanded cross-core IR should codegen to correct tpush/tpop behavior."""
    torch.manual_seed(42)
    base_tensors = builder()
    golden_tensors = {k: v.clone() for k, v in base_tensors.items()}
    expected = golden_fn(golden_tensors)
    codegen_tensors = {k: v.clone() for k, v in base_tensors.items()}
    _run_codegen_after_default_pass_and_check(
        program,
        codegen_tensors,
        arg_order,
        out_name,
        expected,
        platform,
    )


# ---------------------------------------------------------------------------
# Explicit AIV-split (pl.split_aiv) fused-mixed kernels.
#
# A ``for aiv_id in pl.split_aiv(...)`` loop nested inside a ``pl.at(CORE_GROUP)``
# scope is FLATTENED by the parser onto the enclosing InCore scope (no nested
# sub-scope). The cube ops (load Mat -> move Left/Right -> matmul -> Acc tile)
# live outside the loop; the vector ops shard the Acc tile DIRECTLY via
# ``pl.aiv_shard`` (the shard IS the C->V boundary — no intermediate move->Vec,
# which would create a double boundary and a FREE_VAR codegen crash).
# ExpandMixedKernel folds aiv_shard / aic_gather into the cross-core tpush/tpop
# machinery, producing one AIC lane and one AIV lane.
# ---------------------------------------------------------------------------

SA_M = 64
SA_K = 64
SA_N = 64


@pl.program
class SplitAivShardProgram:
    """Direct-shard aiv_shard-only kernel: out = (a @ b) * 2, sharded UP_DOWN."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[SA_M, SA_K], pl.FP32],
        b: pl.Tensor[[SA_K, SA_N], pl.FP32],
        out: pl.Out[pl.Tensor[[SA_M, SA_N], pl.FP32]],
    ) -> pl.Tensor[[SA_M, SA_N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
            a_l1 = pl.load(a, [0, 0], [SA_M, SA_K], target_memory=pl.MemorySpace.Mat)
            b_l1 = pl.load(b, [0, 0], [SA_K, SA_N], target_memory=pl.MemorySpace.Mat)
            a_left = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
            b_right = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
            qk = pl.matmul(a_left, b_right)  # Acc tile [M, N]
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                qk_h = pl.aiv_shard(qk)  # DIRECT shard of the Acc tile -> [M/2, N] Vec
                sc = pl.mul(qk_h, 2.0)  # vector op
                offset = aiv_id * (SA_M // 2)
                out = pl.store(sc, [offset, 0], out)
        return out


@pl.program
class SplitAivQkPvProgram:
    """aic_gather round-trip: out = ((a @ b) * 2) @ v, gather half -> full -> cube."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[SA_M, SA_K], pl.FP32],
        b: pl.Tensor[[SA_K, SA_N], pl.FP32],
        v: pl.Tensor[[SA_N, SA_N], pl.FP32],
        out: pl.Out[pl.Tensor[[SA_M, SA_N], pl.FP32]],
    ) -> pl.Tensor[[SA_M, SA_N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="qkpv"):
            a_l1 = pl.load(a, [0, 0], [SA_M, SA_K], target_memory=pl.MemorySpace.Mat)
            b_l1 = pl.load(b, [0, 0], [SA_K, SA_N], target_memory=pl.MemorySpace.Mat)
            a_left = pl.move(a_l1, target_memory=pl.MemorySpace.Left)
            b_right = pl.move(b_l1, target_memory=pl.MemorySpace.Right)
            qk = pl.matmul(a_left, b_right)  # AIC (cube): Acc tile [M, N]
            # The split_aiv loop is the AIV (vector) per-lane region on HALF tiles.
            for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                qk_h = pl.aiv_shard(qk)  # C->V: full [M,N] -> this lane's half [M/2, N] Vec
                sc = pl.mul(qk_h, 2.0)  # AIV vector work on the half
                full = pl.aic_gather(sc)  # V->C: gather halves -> full [M, N] (leaks out of the loop)
            # Cube ops live OUTSIDE the loop and consume the gathered full tile.
            full_left = pl.move(full, target_memory=pl.MemorySpace.Left)
            v_l1 = pl.load(v, [0, 0], [SA_N, SA_N], target_memory=pl.MemorySpace.Mat)
            v_right = pl.move(v_l1, target_memory=pl.MemorySpace.Right)
            pv = pl.matmul(full_left, v_right)  # AIC (cube): post-gather matmul on the full tile
            out = pl.store(pv, [0, 0], out)
        return out


def _build_split_aiv_shard_tensors() -> dict[str, torch.Tensor]:
    return {
        "a": torch.randn(SA_M, SA_K, dtype=torch.float32),
        "b": torch.randn(SA_K, SA_N, dtype=torch.float32),
        "out": torch.zeros(SA_M, SA_N, dtype=torch.float32),
    }


def _build_split_aiv_qk_pv_tensors() -> dict[str, torch.Tensor]:
    return {
        "a": torch.randn(SA_M, SA_K, dtype=torch.float32),
        "b": torch.randn(SA_K, SA_N, dtype=torch.float32),
        "v": torch.randn(SA_N, SA_N, dtype=torch.float32),
        "out": torch.zeros(SA_M, SA_N, dtype=torch.float32),
    }


def _golden_split_aiv_shard(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    tensors["out"][:] = torch.matmul(tensors["a"], tensors["b"]) * 2.0
    return tensors["out"]


def _golden_split_aiv_qk_pv(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    qk = torch.matmul(tensors["a"], tensors["b"]) * 2.0
    tensors["out"][:] = torch.matmul(qk, tensors["v"])
    return tensors["out"]


def _iter_func_bodies_op_names(program: _ir.Program) -> list[str]:
    """Collect every op name appearing in any function body of the program."""
    names: list[str] = []

    def walk(node) -> None:
        if isinstance(node, _ir.Call) and isinstance(node.op, _ir.Op):
            names.append(node.op.name)
        if isinstance(node, _ir.SeqStmts):
            for s in node.stmts:
                walk(s)
            return
        if isinstance(node, _ir.AssignStmt):
            walk(node.value)
        for sub in ("body", "then_body", "else_body", "expr"):
            inner = getattr(node, sub, None)
            if inner is not None:
                walk(inner)

    for func in program.functions.values():
        walk(func.body)
    return names


def _run_split_aiv_default_and_check(
    program: _ir.Program,
    tensors: dict[str, torch.Tensor],
    arg_order: list[str],
    base_name: str,
    expected: torch.Tensor,
) -> None:
    """Run Default + codegen on a pl.split_aiv kernel and assert full e2e health.

    Pinned to Ascend910B (the split_aiv / GM-pipe-buffer path is 910B-gated).
    Asserts: pipeline survives Default; both AIC/AIV lanes exist and the split
    is folded (no surviving tile.aiv_shard / tile.aic_gather op); PTO codegen
    emits both lanes with no ``__FREE_VAR``; torch golden matches.
    """
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    try:
        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)

        funcs = list(transformed.functions.values())
        names = [f.name for f in funcs]
        assert f"{base_name}_aic" in names, f"missing AIC lane in {names}"
        assert f"{base_name}_aiv" in names, f"missing AIV lane in {names}"
        assert any(f.attrs.get("split_aiv", False) for f in funcs), "no function carries split_aiv marker"

        # AssertNoSplitReshapeSurvives: ExpandMixedKernel must fold the explicit
        # split-reshape ops into tpush/tpop on each lane.
        op_names = _iter_func_bodies_op_names(transformed)
        assert "tile.aiv_shard" not in op_names, "tile.aiv_shard survived ExpandMixedKernel"
        assert "tile.aic_gather" not in op_names, "tile.aic_gather survived ExpandMixedKernel"

        # PTO codegen: both lanes emitted, no FREE_VAR placeholder.
        with tempfile.TemporaryDirectory() as td:
            files = pto_backend.generate(transformed, td, skip_ptoas=True)
        pto_code = "\n".join(v for k, v in files.items() if k.endswith(".pto"))
        assert f"@{base_name}_aic(" in pto_code, "AIC lane not emitted in PTO codegen"
        assert f"@{base_name}_aiv(" in pto_code, "AIV lane not emitted in PTO codegen"
        assert "__FREE_VAR" not in pto_code, "PTO codegen emitted a __FREE_VAR placeholder"

        # Torch golden: split-reshape boundary folds to push_to_/pop_from_.
        code = torch_codegen(transformed, check_shapes=True)
    finally:
        reset_for_testing()

    assert "_cross_core_rt.push_to_" in code
    assert "_cross_core_rt.pop_from_" in code

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    args = [tensors[name] for name in arg_order]
    result = ns["main"](*args)
    actual = tensors["out"]
    assert torch.allclose(actual, expected, rtol=5e-2, atol=5e-2), (
        f"max abs diff = {(expected - actual).abs().max().item():.6e}"
    )
    if isinstance(result, torch.Tensor):
        assert torch.allclose(result, expected, rtol=5e-2, atol=5e-2), (
            f"returned tensor max abs diff = {(expected - result).abs().max().item():.6e}"
        )


_SPLIT_AIV_SCENARIOS = [
    (
        "split_aiv_shard_updown",
        SplitAivShardProgram,
        _build_split_aiv_shard_tensors,
        ["a", "b", "out"],
        "qk",
        _golden_split_aiv_shard,
    ),
    (
        "split_aiv_qk_pv",
        SplitAivQkPvProgram,
        _build_split_aiv_qk_pv_tensors,
        ["a", "b", "v", "out"],
        "qkpv",
        _golden_split_aiv_qk_pv,
    ),
]


@pytest.mark.parametrize(
    ("_name", "program", "builder", "arg_order", "base_name", "golden_fn"),
    _SPLIT_AIV_SCENARIOS,
    ids=[s[0] for s in _SPLIT_AIV_SCENARIOS],
)
def test_split_aiv_fused_mixed_codegen_vs_golden(
    _name: str,
    program: _ir.Program,
    builder,
    arg_order: list[str],
    base_name: str,
    golden_fn,
):
    """Explicit pl.split_aiv fused-mixed kernels compile end-to-end on Ascend910B.

    The flatten fix lets a high-level split_aiv kernel (cube outside the loop,
    direct Acc-tile shard) survive all Default passes and codegen on both
    AIC/AIV lanes with no __FREE_VAR, matching the torch golden.
    """
    torch.manual_seed(42)
    base_tensors = builder()
    golden_tensors = {k: v.clone() for k, v in base_tensors.items()}
    expected = golden_fn(golden_tensors)
    codegen_tensors = {k: v.clone() for k, v in base_tensors.items()}
    _run_split_aiv_default_and_check(program, codegen_tensors, arg_order, base_name, expected)


# ---------------------------------------------------------------------------
# LowerAutoVectorSplit auto-split golden correctness (RFC #1300 convergence).
#
# LowerAutoVectorSplit is the live auto-split lowering path: it converts an AUTO
# ``pl.split`` mixed InCore function into the explicit ``split_aiv`` form BEFORE
# ExpandMixedKernel, so the op-driven boundary arm folds tile.aiv_shard /
# tile.aic_gather into split-stamped tpush/tpop. After it runs, SplitVectorKernel
# only stamps attrs (its split_aiv arm).
#
# These tests pin functional correctness on REAL ``pl.split`` kernels (cube->vector,
# vector->cube, and bidirectional; UP_DOWN and LEFT_RIGHT): each runs the full
# Default pipeline and asserts the codegen'd kernel matches its torch golden. A
# regression in the lowering (e.g. a divergent tpop TileView or a mis-positioned
# get_subblock_idx binding) surfaces as a numerical mismatch here.
# ---------------------------------------------------------------------------


def _run_split_auto_golden(
    program: _ir.Program,
    tensors: dict[str, torch.Tensor],
    arg_order: list[str],
    out_name: str,
    expected: torch.Tensor,
) -> None:
    """Run the Default pipeline (Ascend910B) on an AUTO ``pl.split`` kernel +
    torch codegen and assert the lowered kernel is functionally correct against
    the golden."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    try:
        transformed = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
        code = torch_codegen(transformed, check_shapes=True)
    finally:
        reset_for_testing()

    ns: dict = {}
    exec(code, ns)  # noqa: S102
    args = [tensors[name] for name in arg_order]
    result = ns["main"](*args)
    actual = tensors[out_name]
    assert torch.allclose(actual, expected, rtol=5e-2, atol=5e-2), (
        f"auto-split golden mismatch: max abs diff = {(expected - actual).abs().max().item():.6e}"
    )
    if isinstance(result, torch.Tensor):
        assert torch.allclose(result, expected, rtol=5e-2, atol=5e-2), (
            f"auto-split returned tensor max abs diff = {(expected - result).abs().max().item():.6e}"
        )


_SPLIT_PARITY_SCENARIOS = [
    ("v2c_updown", V2CUDProgram, _build_v2c_tensors, ["a", "b", "output"], "output", _golden_v2c),
    ("v2c_leftright", V2CLRProgram, _build_v2c_tensors, ["a", "b", "output"], "output", _golden_v2c),
    ("c2v_updown", C2VUDProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_c2v),
    ("c2v_leftright", C2VLRProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_c2v),
    ("bidirect_updown", BiDirectUDProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_bidirect),
    ("bidirect_leftright", BiDirectLRProgram, _build_c2v_tensors, ["a", "b", "c"], "c", _golden_bidirect),
]


@pytest.mark.parametrize(
    ("_name", "program", "builder", "arg_order", "out_name", "golden_fn"),
    _SPLIT_PARITY_SCENARIOS,
    ids=[s[0] for s in _SPLIT_PARITY_SCENARIOS],
)
def test_lower_auto_vector_split_golden(
    _name: str,
    program: _ir.Program,
    builder,
    arg_order: list[str],
    out_name: str,
    golden_fn,
):
    """LowerAutoVectorSplit golden correctness on a real ``pl.split`` kernel.

    Runs the full Default pipeline — in which LowerAutoVectorSplit is the live
    auto-split lowering path (LowerAutoVectorSplit -> ExpandMixedKernel ->
    SplitVectorKernel attr-stamping) — and asserts the lowered kernel codegens and
    matches its torch golden (``torch.allclose``). Covers cube->vector,
    vector->cube, and bidirectional boundaries under both UP_DOWN and LEFT_RIGHT.
    """
    torch.manual_seed(42)
    base_tensors = builder()
    golden_tensors = {k: v.clone() for k, v in base_tensors.items()}
    expected = golden_fn(golden_tensors)
    codegen_tensors = {k: v.clone() for k, v in base_tensors.items()}
    _run_split_auto_golden(program, codegen_tensors, arg_order, out_name, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
