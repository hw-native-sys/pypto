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
import py_compile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto import DataType, InternalError, ir
from pypto.ir.param_info import _extract_func_param_infos
from pypto.pypto_core import codegen, passes
from pypto.runtime import DomainBufferSpec
from pypto.runtime.comm_domain import CommDomain, DomainState, build_domain_layout
from pypto.runtime.distributed_args import decode_domain_ref, encode_domain_ref, pack_window_args


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


def test_codegen_requires_materialized_context_suffix():
    # A direct codegen call bypasses the pass that appends the HOST window contexts.
    with pytest.raises(InternalError, match="Expected one input CommCtx suffix per DistributedTensor"):
        codegen.DistributedCodegen().generate(Program)


def test_codegen_rejects_context_before_materialized_suffix():
    @pl.program
    class ExtraContext:
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host(self, extra: pld.CommCtx, window: pld.DistributedTensor[[4], pl.FP32], ctx: pld.CommCtx):
            return

    with pytest.raises(InternalError, match="CommCtx must form the entry suffix"):
        codegen.DistributedCodegen().generate(ExtraContext)


def test_actual_dispatch_values_follow_runtime_slots():
    CommDomainRef = pytest.importorskip("simpler.comm_domain").CommDomainRef

    program, params = plan()
    module = {}
    exec(codegen.DistributedCodegen().generate(program), module)
    worker = MagicMock(_external_domains={}, dc=SimpleNamespace(device_ids=[4, 2, 5]))
    orch = MagicMock()
    domains = []
    for id_ in (1, 2):
        specs = [DomainBufferSpec(k, (65,), DataType.UINT8) for k in ("p", "q")]
        owner = CommDomain(worker, (2, 0), build_domain_layout(specs)[0], "device", ())
        owner._ref = CommDomainRef(b"instance", id_, 1)
        owner._state = DomainState.LIVE
        worker._external_domains[owner.ref] = owner
        native = MagicMock()
        native.workers = (2, 0)
        native.buffers = [SimpleNamespace(name=k, count=65, nbytes=128) for k in ("p", "q")]
        native.device_context.side_effect = lambda rank, base=id_: base * 100 + rank
        native[2].buffers = {k: MagicMock() for k in ("p", "q")}
        for k in ("p", "q"):
            native[2].buffers[k].tensor.return_value = f"{id_}/{k}"
        owner._tokens[2] = MagicMock()
        owner._tokens[2].tensor.return_value = f"order/{id_}"
        domains.append((owner, native))
    orch.get_domain.side_effect = dict((o.ref, n) for o, n in domains).__getitem__
    module["TaskArgs"] = MagicMock
    module["_submit_chip"] = submit = MagicMock()
    (d, _), (e, _) = domains
    # One ordering token per domain, but one context per window parameter, even for the same buffer twice.
    for a, b, expected in [
        (d.tensor("q"), d.tensor("p"), ["1/q", "1/p", "order/1", -7, 102, 102]),
        (e.tensor("p"), d.tensor("q"), ["2/p", "1/q", "order/2", "order/1", -7, 202, 102]),
        (e.tensor("p"), e.tensor("p"), ["2/p", "2/p", "order/2", -7, 202, 202]),
    ]:
        entry_args, owners = pack_window_args(
            params, [a, -7, b, object()], worker, module["_PYPTO_WINDOW_REQUIREMENTS"]
        )
        scalars = tuple(entry_args.scalar(i) for i in range(entry_args.scalar_count()))
        assert scalars == (
            a.buffer_index,
            2**32 - 7,
            b.buffer_index,
            *encode_domain_ref(a.domain_ref),
            *encode_domain_ref(b.domain_ref),
        )
        assert entry_args.tensor_count() == 0
        tensors = {"n": -7, "x": object(), ".domain_tokens": {owner.ref: owner._tokens for owner in owners}}
        orch.get_domain.reset_mock()
        module["host"](
            orch,
            entry_args,
            None,
            tensors=tensors,
            callables={"chip": 1},
            sub_ids={},
            _keep=[],
            world_size=3,
        )
        assert orch.get_domain.call_count == len({a.domain_ref, b.domain_ref})
        calls = submit.call_args.args[2].mock_calls
        assert [call.args[0] for call in calls] == expected
        submit.reset_mock()
        for pos in (0, 2):
            invalid_args = type(entry_args)()
            for i, value in enumerate(scalars):
                invalid_args.add_scalar(2 if i == pos else value)  # Both domains have two buffers.
            with pytest.raises(IndexError):
                module["host"](
                    orch,
                    invalid_args,
                    None,
                    tensors=tensors,
                    callables={"chip": 1},
                    sub_ids={},
                    _keep=[],
                    world_size=3,
                )
        submit.assert_not_called()
    reqs = module["_PYPTO_WINDOW_REQUIREMENTS"]
    args = [d.tensor("p"), 7, e.tensor("q"), object()]
    with pytest.raises(ValueError, match="different worker"):
        pack_window_args(params, args, MagicMock(_external_domains={}, dc=worker.dc), reqs)
    with pytest.raises(ValueError, match="target worker 1"):
        pack_window_args(params, args, worker, {**reqs, 0: (True, (), None)})
    with pytest.raises(ValueError, match="requires urma"):
        pack_window_args(params, args, worker, {**reqs, 2: (False, (2,), "urma")})
    with pytest.raises(TypeError, match="requires DomainTensor"):
        pack_window_args(params, [object(), *args[1:]], worker, reqs)
    ref = CommDomainRef(b"\xff" * 8, 2**64 - 1, 2**63)
    encoded = encode_domain_ref(ref)
    from simpler.task_interface import TaskArgs  # noqa: PLC0415

    args = TaskArgs()
    for value in encoded:
        args.add_scalar(value)
    assert decode_domain_ref(args, 0) == ref


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


def test_recompile_external_window_retires_previous_chip_binaries(tmp_path):
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415
    from pypto.runtime._binary_cache import binary_context_path  # noqa: PLC0415
    from pypto.runtime._prebuilt import BINARY_MANIFEST  # noqa: PLC0415
    from pypto.runtime.distributed_runner import _assemble_chip_callables  # noqa: PLC0415

    # An earlier prebuild and an ordinary dispatch both left reusable binaries.
    chip = tmp_path / "next_levels" / "read_window_chip"
    stamp = binary_context_path(chip)
    stamp.parent.mkdir(parents=True)
    stamp.write_text("{}")
    config = chip / "kernel_config.py"
    config.write_text("KERNELS = []")
    bytecode = chip / "__pycache__" / "kernel_config.previous.pyc"
    py_compile.compile(str(config), cfile=str(bytecode), doraise=True)
    marker = tmp_path / BINARY_MANIFEST
    marker.write_text('{"schema":2,"platform":"a5","chips":["read_window_chip"]}')

    # A source-only build emits new PTO but no executable CHIP configuration.
    ir.compile(read_window.specialize(), output_dir=str(tmp_path), skip_ptoas=True)
    with pytest.raises(RuntimeError, match="No chip-level tasks"):
        _assemble_chip_callables(DistributedCompiledProgram.from_dir(tmp_path))
    assert not marker.exists()
    assert not stamp.exists()
    assert not bytecode.exists()


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
