# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Explicit DDR/SRAM copy contracts and lowering through bounded MTE chunks."""

import pypto.language as pl
import pytest
from pypto import DataType, backend, codegen, ir
from pypto.ir.op import tensor
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import testing


@pytest.fixture(autouse=True)
def reset_backend():
    backend.reset_for_testing()
    yield
    backend.reset_for_testing()


def var(name, shape=(4, 600), dtype=DataType.FP32):
    return ir.Var(name, ir.TensorType(list(shape), dtype), ir.Span.unknown())


@pytest.mark.parametrize("reverse", [False, True])
def test_copy_void_and_pipe(reverse):
    backend.set_backend_type(backend.BackendType.Ascend910B)
    dst, src = var("dst"), var("src")
    source, target = (ir.Mem.SRAM, ir.Mem.DDR) if reverse else (ir.Mem.DDR, ir.Mem.SRAM)
    call = tensor.copy(dst, src, [1, 4], [0, 8], [2, 500], source_memory=source, target_memory=target)
    assert isinstance(call.type, ir.UnknownType)
    assert testing.try_infer_pipe(call) == int(ir.PipeType.MTE3 if reverse else ir.PipeType.MTE2)


@pytest.mark.parametrize("source,target", [(ir.Mem.SRAM, ir.Mem.SRAM), (ir.Mem.Vec, ir.Mem.SRAM)])
def test_invalid_endpoints(source, target):
    with pytest.raises(ValueError, match="DDR -> DDR, DDR -> SRAM or SRAM -> DDR"):
        tensor.copy(
            var("dst"), var("src"), [0, 0], [0, 0], [4, 600], source_memory=source, target_memory=target
        )


def test_ddr_whole_tensor_copy():
    backend.set_backend_type(backend.BackendType.Ascend910B)
    dst, src = var("dst"), var("src")
    call = tensor.copy(dst, src, target_memory=ir.Mem.DDR)
    assert isinstance(call.type, ir.UnknownType)
    assert len(call.args) == 2
    assert testing.try_infer_pipe(call) == int(ir.PipeType.MTE2)
    with pytest.raises(ValueError, match="same static shape"):
        tensor.copy(var("other", (4, 601)), src, target_memory=ir.Mem.DDR)


@pytest.mark.parametrize("region", [{"dst_offsets": [0, 0]}, {"shape": [4, 600]}])
def test_partial_region_arguments_rejected(region):
    with pytest.raises(ValueError, match="together"):
        tensor.copy(var("dst"), var("src"), target_memory=ir.Mem.DDR, **region)


@pytest.mark.parametrize("memory_type", [ir.Mem.DDR, ir.Mem.SRAM])
def test_create_memory_type(memory_type):
    call = tensor.create([4, 600], DataType.FP32, memory_type=memory_type)
    assert dict(call.kwargs).get("memory_type", ir.Mem.DDR) == memory_type
    assert call.type.memory_space == ir.Mem.DDR
    if memory_type == ir.Mem.DDR:
        ir.assert_structural_equal(call, tensor.create([4, 600], DataType.FP32))
    else:
        assert dict(call.kwargs)["memory_type"] == ir.Mem.SRAM


def test_create_rejects_tile_memory():
    with pytest.raises(ValueError, match="memory_type must be DDR or SRAM"):
        tensor.create([4, 600], DataType.FP32, memory_type=ir.Mem.Vec)
    shape = ir.MakeTuple([ir.ConstInt(4, DataType.INDEX, ir.Span.unknown())], ir.Span.unknown())
    with pytest.raises(ValueError, match="memory_type must be DDR or SRAM"):
        ir.create_op_call(
            "tensor.create", [shape], {"dtype": DataType.FP32, "memory_type": ir.Mem.Vec}, ir.Span.unknown()
        )


@pytest.mark.parametrize(
    "dst,src,offsets,shape,message",
    [
        (var("d"), var("s", dtype=DataType.FP16), [0, 0], [4, 600], "matching dtypes"),
        (var("d"), var("s", (600,)), [0, 0], [4, 600], "matching nonzero ranks"),
        (var("d"), var("s"), [-1, 0], [4, 600], "non-negative"),
        (var("d"), var("s"), [0, 1], [4, 600], "exceeds dst shape"),
        (var("d"), var("s"), [0], [4, 600], "dst_offsets rank"),
        (var("d"), var("s"), [0, 0], [0, 600], "must be positive"),
    ],
)
def test_invalid_regions(dst, src, offsets, shape, message):
    with pytest.raises(ValueError, match=message):
        tensor.copy(dst, src, offsets, [0, 0], shape)


def test_dynamic_extent_and_discarded_result():
    backend.set_backend_type(backend.BackendType.Ascend910B)

    @pl.program
    class Copy:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self,
            src: pl.Tensor[[4, 600], pl.FP32],
            dst: pl.Out[pl.Tensor[[4, 600], pl.FP32]],
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[4, 600], pl.FP32]:
            pl.copy(dst, src, [0, 0], [0, 0], [rows, cols])
            return dst

    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Copy)
    functions = [f for f in lowered.functions.values() if ir.is_incore_type(f.func_type)]
    text = codegen.PTOCodegen().generate(ir.Program(functions, functions[0].name, lowered.span))
    assert "pto.tload " in text and "pto.tstore " in text
    assert 'target_memory = "sram"' in text


@pytest.mark.parametrize("target", [backend.BackendType.Ascend910B, backend.BackendType.Ascend950])
@pytest.mark.parametrize("endpoints", [("DDR", "DDR"), ("DDR", "SRAM"), ("SRAM", "DDR")])
@pytest.mark.parametrize("shape", [[600], [4, 600], [2, 4, 600]])
@pytest.mark.parametrize("repeat", [False, True])
def test_full_pipeline(target, endpoints, shape, repeat):
    backend.set_backend_type(target)
    source, destination = endpoints
    offsets = [0] * len(shape)
    first_copy = (
        f"pl.copy(dst, src, {offsets}, {offsets}, {shape}, "
        f"source_memory=pl.Mem.{source}, target_memory=pl.Mem.{destination})"
        if repeat
        else ""
    )
    program = pl.parse(f"""
import pypto.language as pl
@pl.program
class Copy:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, src: pl.Tensor[{shape}, pl.FP32],
             dst: pl.Out[pl.Tensor[{shape}, pl.FP32]]) -> pl.Tensor[{shape}, pl.FP32]:
        {first_copy}
        pl.copy(dst, src, {offsets}, {offsets}, {shape},
                       source_memory=pl.Mem.{source}, target_memory=pl.Mem.{destination})
        return dst
""")
    ir.assert_structural_equal(program, pl.parse(ir.python_print(program)))
    lowered = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    functions = [f for f in lowered.functions.values() if ir.is_incore_type(f.func_type)]
    assert len(functions) == 1
    text = codegen.PTOCodegen().generate(ir.Program(functions, functions[0].name, lowered.span))
    assert "pto.tload " in text and "pto.tstore " in text
    assert f'source_memory = "{"sram" if source == "SRAM" else "gm"}"' in text
    assert f'target_memory = "{"sram" if destination == "SRAM" else "gm"}"' in text
    assert "scf.for" in text
    assert "tensor.copy" not in text
    assert "loc=sram" not in text


@pytest.mark.parametrize("target", [backend.BackendType.Ascend910B, backend.BackendType.Ascend950])
def test_chip_sram_metadata(target, tmp_path):
    backend.set_backend_type(target)
    instance = backend.get_backend_instance(target)
    path = str(tmp_path / "backend.pto")
    instance.export_to_file(path)
    assert (tmp_path / "backend.pto").stat().st_size > 0
    assert instance.get_mem_size(ir.Mem.SRAM) == 256 * 1024 * 1024
    for soc in (instance.soc,):
        assert len(soc.mems) == 1
        assert soc.mems[0].mem_type == ir.Mem.SRAM
        assert soc.mems[0].mem_size == 256 * 1024 * 1024
        for die in soc.die_counts:
            for cluster in die.cluster_counts:
                for core in cluster.core_counts:
                    assert all(mem.mem_type != ir.Mem.SRAM for mem in core.mems)
