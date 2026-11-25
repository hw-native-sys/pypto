# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Distributed entry ABI persistence and execution of generated HOST dispatch."""

import json

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.ir.param_info import _extract_func_param_infos
from pypto.pypto_core import codegen, passes


@pl.program
class Program:
    @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(
        self,
        a: pld.DistributedTensor[[65], pl.UINT8],
        n: pl.Scalar[pl.INT32],
        b: pld.DistributedTensor[[65], pl.UINT8],
    ):
        return

    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def host(
        self,
        a: pld.DistributedTensor[[65], pl.UINT8],
        n: pl.Scalar[pl.INT32],
        b: pld.DistributedTensor[[65], pl.UINT8],
        x: pl.Tensor[[4], pl.FP32],
    ):
        self.chip(a, n, b, device=2)


def plan():
    program = passes.materialize_dist_tensor_ctx()(Program)
    name = program.get_global_var("host")
    assert name is not None
    host = program.functions[name]
    params, _, _ = _extract_func_param_infos(host, host_entry=True)
    return program, params


def test_public_parameter_metadata_follows_source_order():
    program, params = plan()
    host = codegen.DistributedCodegen().generate(program)
    assert "__index_0 = _args.scalar(0)" in host
    assert "__index_2 = _args.scalar(2)" in host
    assert "__ref_0 = decode_domain_ref(_args, 3)" in host
    assert "__ref_2 = decode_domain_ref(_args, 6)" in host
    assert "__slot_2 = __domain_2.buffers[__index_2].name" in host
    assert "_entry_bindings" not in host
    assert [(p.name, p.is_distributed) for p in params] == [
        ("a", True),
        ("n", False),
        ("b", True),
        ("x", False),
    ]


@pl.jit.incore
def copy_window(
    src: pld.DistributedTensor[[2, 32], pl.UINT8], dst: pl.Out[pl.Tensor[[2, 32], pl.UINT8]]
) -> pl.Tensor[[2, 32], pl.UINT8]:
    return pl.store(pl.load(src, [0, 0], [2, 32]), [0, 0], dst)


@pl.jit
def read_window_chip(
    src: pld.DistributedTensor[[2, 32], pl.UINT8], dst: pl.Out[pl.Tensor[[2, 32], pl.UINT8]]
) -> pl.Tensor[[2, 32], pl.UINT8]:
    return copy_window(src, dst)


@pl.jit.host
def read_window(
    src: pld.DistributedTensor[[2, 32], pl.UINT8, pl.MemRef("src", 0, 64)],
    dst: pl.Out[pl.Tensor[[2, 32], pl.UINT8]],
) -> pl.Tensor[[2, 32], pl.UINT8]:
    return read_window_chip(src, dst, device=0)


def test_host_slice_rejects_external_window(tmp_path):
    from pypto.runtime import RunConfig  # noqa: PLC0415

    @pl.jit.host
    def slice_host(
        src: pld.DistributedTensor[[4, 32], pl.UINT8], dst: pl.Out[pl.Tensor[[2, 32], pl.UINT8]]
    ) -> pl.Tensor[[2, 32], pl.UINT8]:
        return read_window_chip(pl.slice(src, [2, 32], [1, 0]), dst, device=0)

    with pytest.raises(ValueError, match="tensor.slice.*supports plain Tensor operands only"):
        slice_host.compile(config=RunConfig(save_kernels_dir=str(tmp_path)))


def test_external_window_cold_load(tmp_path):
    from pypto.runtime import RunConfig  # noqa: PLC0415

    compiled = read_window.compile(config=RunConfig(save_kernels_dir=str(tmp_path)))
    metadata = tmp_path / "distributed_meta.json"
    raw = json.loads(metadata.read_text())
    assert raw["schema"] == 4
    assert [p.get("is_distributed", False) for p in raw["params"]] == [True, False]
    assert "l3_entry" not in raw
    assert type(compiled).from_dir(tmp_path)._get_metadata()[0] == compiled._get_metadata()[0]
    raw["params"][0]["shape"][0] += 1
    metadata.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="parameter metadata"):
        type(compiled).from_dir(tmp_path)
    raw["schema"] = 7
    metadata.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Incompatible .* schema 7"):
        type(compiled).from_dir(tmp_path)


def test_external_parameter_and_local_scratch_have_separate_bindings():
    @pl.program
    class Mixed:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self,
            table: pld.DistributedTensor[[64], pl.UINT8],
            scratch: pl.InOut[pld.DistributedTensor[[64], pl.UINT8]],
        ):
            return

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, table: pld.DistributedTensor[[64], pl.UINT8]):
            storage = pld.alloc_window_buffer(64)
            scratch = pld.window(storage, [64], dtype=pl.UINT8)
            self.chip(table, scratch, device=0)

    program = passes.materialize_comm_domain_scopes()(Mixed)
    program = passes.materialize_dist_tensor_ctx()(program)
    generated = codegen.DistributedCodegen().generate(program)
    assert generated.count("with (_domain_provider or orch.allocate_domain)(") == 1
    assert 'CommBufferSpec(name="storage"' in generated
    assert "__domain_0[0].buffers[__slot_0].tensor(" in generated
    assert '__comm_d0[0].buffers["storage"].tensor(' in generated
