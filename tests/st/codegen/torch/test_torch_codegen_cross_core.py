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

import pypto.language as pl
import pytest
import torch
from harness.core.harness import PLATFORMS, platform_to_backend
from pypto import ir as _ir
from pypto.backend import reset_for_testing, set_backend_type
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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)
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
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
        ):
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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
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
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.UP_DOWN)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
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
            level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.LEFT_RIGHT)
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
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
        with pl.at(
            level=pl.Level.CORE_GROUP,
            optimization=pl.chunked_loop_optimizer(split=pl.SplitMode.NONE),
        ):
            for nb in pl.parallel(0, N_BLOCKS, 1, chunk=4, chunk_policy="leading_full"):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
