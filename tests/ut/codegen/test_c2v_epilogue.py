# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit pipe epilogues work without matching or rewriting cast/ReLU chains."""

import pypto.language as pl
import pytest
import torch
from pypto import backend, codegen, ir, passes
from pypto.backend import pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.debug import torch_codegen
from pypto.ir.pass_manager import OptimizationStrategy, PassManager


def make_program(
    dtype=pl.BF16,
    relu=True,
    *,
    pipe_id=0,
    peer_relu=None,
    pop_dtype=None,
    split=0,
    input_dtype=pl.BF16,
    slot_size=None,
):
    """Keep the protocol explicit; no Vector cast or ReLU exists to fold."""
    names = {pl.FP32: "FP32", pl.FP16: "FP16", pl.BF16: "BF16", pl.INT8: "INT8", pl.INT32: "INT32"}
    target = names[dtype]
    inputs = names[input_dtype]
    popped = names[pop_dtype or dtype]
    stored = "value" if popped == target else f'pl.cast(value, pl.{target}, mode="rint")'
    peer_relu = relu if peer_relu is None else peer_relu
    if slot_size is None:
        slot_size = 64 * 128 * dtype.get_bit() // 8
    # Integer inputs are used only for malformed-producer tests. Avoid an
    # unrelated unscaled INT32-to-FP32 GM store obscuring the pipe diagnostic.
    raw_store = "" if input_dtype == pl.INT8 else "pl.store(acc, [0, 0], raw)"
    source = f"""@pl.program
class Program:
    @pl.function(type=pl.FunctionType.AIC)
    def producer(self, a: pl.Tensor[[64, 128], pl.{inputs}],
                 b: pl.Tensor[[128, 128], pl.{inputs}],
                 out: pl.Out[pl.Tensor[[64, 128], pl.{target}]],
                 raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        peer = pl.import_peer_buffer(name="scores", peer_func="consumer")
        pl.aic_initialize_pipe(c2v_consumer_buf=peer, dir_mask=1,
            slot_size={slot_size}, slot_num=2, id={pipe_id},
            c2v_dtype=pl.{target}, pre_relu={relu})
        for iteration in pl.range(3):
            a_mat = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
            b_mat = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
            left = pl.move(a_mat, target_memory=pl.Mem.Left)
            right = pl.move(b_mat, target_memory=pl.Mem.Right)
            acc = pl.matmul(left, right)
            pl.tpush_to_aiv(acc, split={split}, id={pipe_id})
            {raw_store}

    @pl.function(type=pl.FunctionType.AIV)
    def consumer(self, a: pl.Tensor[[64, 128], pl.{inputs}],
                 b: pl.Tensor[[128, 128], pl.{inputs}],
                 out: pl.Out[pl.Tensor[[64, 128], pl.{target}]],
                 raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        buf = pl.reserve_buffer(name="scores", size={slot_size * 2})
        pl.aiv_initialize_pipe(c2v_consumer_buf=buf, dir_mask=1,
            slot_size={slot_size}, slot_num=2, id={pipe_id},
            c2v_dtype=pl.{target}, pre_relu={peer_relu})
        for iteration in pl.range(3):
            value = pl.tpop_from_aic(shape=[64, 128], dtype=pl.{popped},
                                     split={split}, id={pipe_id})
            pl.store({stored}, [0, 0], out)
            pl.tfree_to_aic(value, id={pipe_id})

    @pl.function(type=pl.FunctionType.Group)
    def group(self, a: pl.Tensor[[64, 128], pl.{inputs}],
              b: pl.Tensor[[128, 128], pl.{inputs}],
              out: pl.Out[pl.Tensor[[64, 128], pl.{target}]],
              raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        self.producer(a, b, out, raw)
        self.consumer(a, b, out, raw)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[64, 128], pl.{inputs}],
             b: pl.Tensor[[128, 128], pl.{inputs}],
             out: pl.Out[pl.Tensor[[64, 128], pl.{target}]],
             raw: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        self.group(a, b, out, raw)
"""
    program = pl.parse(source)
    assert isinstance(program, ir.Program)
    return program


def lower(program):
    backend.set_backend_type(backend.BackendType.Ascend950)
    with passes.PassContext([], enable_buffer_ir=False):
        after = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    kernels = [f for f in after.functions.values() if ir.is_incore_type(f.func_type)]
    return after, codegen.PTOCodegen().generate(ir.Program(kernels, "kernels", after.span))


@pytest.mark.parametrize("dtype", [pl.FP32, pl.FP16, pl.BF16])
@pytest.mark.parametrize("relu", [False, True])
def test_explicit_parameters_roundtrip(dtype, relu):
    before = make_program(dtype, relu)
    printed = ir.python_print(before)
    assert "c2v_dtype=" in printed
    assert "fixpipe_bf16" not in printed
    ir.assert_structural_equal(before, pl.parse(printed))


# The marker selects compiler coverage; the hosted unit-test job has no PTOAS.
# Compiler CI uses --require-ptoas to reject an unavailable assembler up front.
@pytest.mark.skipif(find_ptoas_binary() is None, reason="PTOAS is not available")
@pytest.mark.ptoas_compile
@pytest.mark.parametrize(
    "dtype,quant,isa_quant",
    [(pl.FP32, "no_convert", "NoQuant"), (pl.FP16, "f32_f16", "F322F16"), (pl.BF16, "f32_bf16", "F322BF16")],
)
@pytest.mark.parametrize("relu", [False, True])
def test_explicit_epilogue_compiles(tmp_path, dtype, quant, isa_quant, relu):
    _, text = lower(make_program(dtype, relu, pipe_id=3))
    expected = f"quant = {quant}, relu = {'normal_relu' if relu else 'no_relu'}"
    assert text.count(expected) == 2
    assert "nosplit = true" in text
    assert "id = 3" in text
    cpp = pto_backend._compile_pto_module(text, "producer", str(tmp_path))
    assert f"QuantMode_t::{isa_quant}" in cpp
    assert f"ReluPreMode::{'NormalRelu' if relu else 'NoRelu'}" in cpp
    assert "TPUSH" in cpp


@pytest.mark.skipif(find_ptoas_binary() is None, reason="PTOAS is not available")
@pytest.mark.ptoas_compile
def test_same_function_cannot_reinitialize_an_explicit_pipe(tmp_path):
    _, text = lower(make_program())
    initializer = next(line for line in text.splitlines() if "pto.aic_initialize_pipe " in line)
    text = text.replace(initializer, initializer + "\n" + initializer)
    # A pipe ID has one static contract per function, not sequential contracts.
    with pytest.raises(RuntimeError, match="requires 'id' to be unique across frontend initialize_pipe"):
        pto_backend._compile_pto_module(text, "producer", str(tmp_path))


@pytest.mark.skipif(find_ptoas_binary() is None, reason="PTOAS is not available")
@pytest.mark.ptoas_compile
def test_explicit_dtype_does_not_leak_to_a_later_function(tmp_path):
    kernels = []
    for prefix, dtype in (("first", pl.BF16), ("second", pl.FP32)):
        # Equal slots isolate dtype metadata from the module's GM pipe layout.
        source = ir.python_print(make_program(dtype, False, slot_size=32768))
        for name in ("producer", "consumer", "group", "main"):
            source = source.replace(name, f"{prefix}_{name}")
        if dtype == pl.FP32:
            source = source.replace(", c2v_dtype=pl.FP32", "")
        after, _ = lower(pl.parse(source))
        kernels.extend(f for f in after.functions.values() if ir.is_incore_type(f.func_type))
    text = codegen.PTOCodegen().generate(ir.Program(kernels, "independent_functions", after.span))
    cpp = pto_backend._compile_pto_module(text, "producer", str(tmp_path))
    assert "first_consumer" in cpp and "second_consumer" in cpp


@pytest.mark.parametrize("dtype", [pl.FP32, pl.FP16, pl.BF16])
@pytest.mark.parametrize("relu", [False, True])
def test_reference_uses_pipe_configuration_and_preserves_accumulator(dtype, relu):
    # Use FP32 operands to isolate the pipe conversion: the existing Torch
    # reference matmul otherwise rounds BF16 inputs' output before widening.
    # Repeated pushes must leave the producer's FP32 accumulator unchanged.
    a = torch.zeros(64, 128)
    a[:, :2] = 1
    b = torch.zeros(128, 128)
    b[0] = torch.tensor([1.0, 1.0078125, -1.0, -1.0078125] * 32).to(torch.bfloat16)
    b[1] = torch.tensor([2**-8, 2**-11, -(2**-8), -(2**-11)] * 32).to(torch.bfloat16)
    torch_dtype = {pl.FP32: torch.float32, pl.FP16: torch.float16, pl.BF16: torch.bfloat16}[dtype]
    out = torch.empty(64, 128, dtype=torch_dtype)
    raw = torch.empty(64, 128)
    namespace = {}
    exec(torch_codegen(make_program(dtype, relu, pipe_id=3, input_dtype=pl.FP32)), namespace)
    namespace["main"](a, b, out, raw)
    expected = a.float() @ b.float()
    torch.testing.assert_close(raw, expected, rtol=0, atol=0)
    if relu:
        expected = expected.clamp_min(0)
    torch.testing.assert_close(out, expected.to(torch_dtype), rtol=0, atol=0)


def test_reference_keeps_independent_epilogues_when_popped_in_reverse_order():
    # The pipe ID is part of both the epilogue and queue contract. A shared
    # queue returns pipe 0's activated BF16 payload for pipe 7's FP16 pop.
    @pl.program
    class IndependentPipes:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, acc: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc]):
            pl.aic_initialize_pipe(dir_mask=1, slot_size=16384, id=0, c2v_dtype=pl.BF16, pre_relu=True)
            pl.aic_initialize_pipe(dir_mask=1, slot_size=16384, id=7, c2v_dtype=pl.FP16)
            pl.tpush_to_aiv(acc, split=0, id=0)
            pl.tpush_to_aiv(acc, split=0, id=7)
            plain = pl.tpop_from_aic(shape=[64, 128], dtype=pl.FP16, split=0, id=7)
            activated = pl.tpop_from_aic(shape=[64, 128], dtype=pl.BF16, split=0, id=0)
            return plain, activated

    namespace = {}
    exec(torch_codegen(IndependentPipes), namespace)
    acc = torch.full((64, 128), -1.00390625)
    plain, activated = namespace["main"](acc)
    assert plain.dtype == torch.float16
    assert activated.dtype == torch.bfloat16
    torch.testing.assert_close(plain, acc.to(torch.float16), rtol=0, atol=0)
    torch.testing.assert_close(activated, torch.zeros_like(acc, dtype=torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [pl.FP32, pl.FP16, pl.BF16])
@pytest.mark.parametrize("relu", [False, True])
def test_reference_applies_pipe_epilogue_in_inline_push_helper(dtype, relu):
    @pl.program
    class InlinePush:
        @pl.function(type=pl.FunctionType.Inline)
        def push(self, acc: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc]):
            pl.tpush_to_aiv(acc, split=0, id=7)

        @pl.function(type=pl.FunctionType.InCore)
        def main(self, acc: pl.Tile[[64, 128], pl.FP32, pl.Mem.Acc]):
            pl.aic_initialize_pipe(dir_mask=1, slot_size=32768, id=7, c2v_dtype=dtype, pre_relu=relu)
            self.push(acc)
            value = pl.tpop_from_aic(shape=[64, 128], dtype=dtype, split=0, id=7)
            return value

    namespace = {}
    exec(torch_codegen(InlinePush), namespace)
    acc = torch.full((64, 128), -1.00390625)
    result = namespace["main"](acc)
    torch_dtype = {pl.FP32: torch.float32, pl.FP16: torch.float16, pl.BF16: torch.bfloat16}[dtype]
    expected = acc.clamp_min(0) if relu else acc
    assert result.dtype == torch_dtype
    torch.testing.assert_close(result, expected.to(torch_dtype), rtol=0, atol=0)


@pytest.mark.parametrize("initializer", [pl.aic_initialize_pipe, pl.aiv_initialize_pipe])
def test_defaults_preserve_existing_pipe(initializer):
    call = initializer(dir_mask=1, slot_size=1024, pre_relu=False)
    assert "c2v_dtype" not in call.kwargs
    assert "pre_relu" not in call.kwargs
    with pytest.raises(ValueError, match="requires an explicit c2v_dtype"):
        initializer(dir_mask=1, slot_size=1024, pre_relu=True)
    with pytest.raises(TypeError, match="must be a DataType"):
        initializer(dir_mask=1, slot_size=1024, c2v_dtype="bf16")


@pytest.mark.parametrize(
    "target,direction,dtype,message",
    [
        (backend.BackendType.Ascend910, 1, "BF16", "A5 C2V-only"),
        (backend.BackendType.Ascend950, 3, "BF16", "A5 C2V-only"),
        (backend.BackendType.Ascend950, 1, "INT32", "must be FP32, FP16, or BF16"),
    ],
)
def test_unsupported_epilogue_is_rejected(target, direction, dtype, message):
    backend.set_backend_type(target)
    program = pl.parse(f"""@pl.program
class Invalid:
    @pl.function(type=pl.FunctionType.AIC)
    def main(self):
        pl.aic_initialize_pipe(dir_mask={direction}, slot_size=1024, c2v_dtype=pl.{dtype})
""")
    assert isinstance(program, ir.Program)
    with pytest.raises(ValueError, match=message):
        codegen.PTOCodegen().generate(program)


@pytest.mark.skipif(find_ptoas_binary() is None, reason="PTOAS is not available")
@pytest.mark.ptoas_compile
@pytest.mark.parametrize("kwargs", [{"peer_relu": False}, {"split": 1}])
def test_inconsistent_pipe_contract_is_rejected(tmp_path, kwargs):
    _, text = lower(make_program(**kwargs))
    with pytest.raises(RuntimeError, match="(?i)epilogue|consumer|peer|split|element type"):
        pto_backend._compile_pto_module(text, "producer", str(tmp_path))


@pytest.mark.parametrize(
    "dtype,pop_dtype,input_dtype",
    [(pl.BF16, pl.FP32, pl.BF16), (pl.FP32, pl.INT32, pl.INT8), (pl.FP32, pl.FP32, pl.INT8)],
)
def test_explicit_dtype_contract_is_enforced(dtype, pop_dtype, input_dtype):
    # NoConvert admits both integer and float pipes in PTOAS. The explicit
    # c2v_dtype must nevertheless constrain this producer/consumer pair.
    with pytest.raises(ValueError, match="c2v_dtype|FP32 Acc"):
        lower(make_program(dtype, pop_dtype=pop_dtype, input_dtype=input_dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
