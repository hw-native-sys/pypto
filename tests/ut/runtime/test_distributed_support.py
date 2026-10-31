# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-neutral embedded ABI contracts, including real optional CPU wire tests."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import codegen, passes


@pytest.fixture
def mixed_embedded_source():
    @pl.program
    class Program:
        @pl.function(level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT8],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
            enabled: pl.Scalar[pl.BOOL],
            limit: pl.Scalar[pl.INDEX],
            gain: pl.Scalar[pl.FP32],
        ):
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def node(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT8],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
            enabled: pl.Scalar[pl.BOOL],
            limit: pl.Scalar[pl.INDEX],
            gain: pl.Scalar[pl.FP32],
        ):
            self.chip(x, count, out, enabled, limit, gain)

    program = passes.convert_to_ssa()(Program)
    compile(codegen.DistributedCodegen().generate(program), "host_orch.py", "exec")
    source = codegen.DistributedCodegen()._generate_embedded(program, entry_name="node")
    compile(source, "embedded_node.py", "exec")
    assert "def node(orch, args, config):" in source and "_submit_chip(" in source
    return source


@pytest.fixture
def canonical_support_path(mixed_embedded_source):
    # Native generation and its full verification context must succeed before
    # a missing canonical helper can be classified as the intended RED failure.
    assert mixed_embedded_source
    path = Path(__file__).resolve().parents[3] / "python/pypto/runtime/_distributed_support.py"
    return path


@pytest.fixture
def normalized_return_copy_program():
    program = pl.parse_program("""
@pl.program
class Program:
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, x: pl.Tensor[[2, 2], pl.FP32], count: pl.Scalar[pl.INT32]) -> pl.Tensor[[2, 2], pl.FP32]:
        return x
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def cluster(self, x: pl.Tensor[[2, 2], pl.FP32], count: pl.Scalar[pl.INT32]):
        result = self.node(x, count, worker=0)
        return 0
""")
    program = passes.normalize_return_order()(passes.convert_to_ssa()(program))
    properties = passes.IRPropertySet()
    properties.insert(passes.IRProperty.ReturnParamsExplicit)
    assert passes.PropertyVerifierRegistry.verify(properties, program) == []
    compile(codegen.DistributedCodegen().generate(program), "return_cluster.py", "exec")
    return program


_RETURN_COPY_CPU_PROOF = r"""
import importlib.abc
import struct
import sys
import time
from types import SimpleNamespace

class RejectPyPTO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pypto" or fullname.startswith("pypto."):
            raise ModuleNotFoundError("Forbidden runtime-only import: " + fullname)
        return None

assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
sys.meta_path.insert(0, RejectPyPTO())

import torch
from return_proof_namespace import entry as generated
from return_proof_namespace import cluster as caller
from return_proof_namespace import _distributed_support as support
from simpler.buffer import (
    AccessMode, BufferCapability, ImportContext, ImportRegistry, MappedArg, MappedArgs,
    create_host_shared_buffer, mint_owner_instance_id,
)
from simpler.comm_endpoints import HOST_CPU
from simpler.task_interface import CallConfig, DataType, TaskArgs, TensorArgType
from simpler.worker import Worker

strided = sys.argv[1] == "strided"
nbytes = 35 if strided else 32
owner = mint_owner_instance_id()
buffers = [create_host_shared_buffer(nbytes, owner, index + 1) for index in range(2)]
buffers.append(create_host_shared_buffer(1, owner, 3))
buffers.append(create_host_shared_buffer(nbytes, owner, 4, access=AccessMode.READ))
buffers.append(create_host_shared_buffer(nbytes, owner, 5, access=AccessMode.WRITE))
worker = None
try:
    buffers[0].shm.buf[:] = bytes(nbytes)
    buffers[1].shm.buf[:] = bytes([0xCD]) * nbytes
    buffers[2].shm.buf[0] = 0
    ready = buffers[2].tensor(shapes=(1,), dtype=DataType.UINT8)
    source = buffers[0].tensor(
        shapes=(2, 2), dtype=DataType.FLOAT32,
        strides=(4, 1) if strided else (2, 1), byte_offset=4 if strided else 0,
    )
    destination = buffers[1].tensor(
        shapes=(2, 2), dtype=DataType.FLOAT32,
        strides=(4, 1) if strided else (2, 1), byte_offset=8 if strided else 0,
    )

    def producer(args):
        # CPU SUB is a stand-in producer for the CHIP result. Dependency
        # inference and final copy execute through the actual runtime engine.
        # This real shared flag is opened only after the generated return
        # schedules its copy. An inline copy would therefore read stale zeros.
        deadline = time.monotonic() + 10
        while args[1].buffer[0] == 0:
            if time.monotonic() > deadline:
                raise RuntimeError("Return-copy graph construction never released the producer")
            time.sleep(0.001)
        arg = args[0]
        extent = 1 + sum((dim - 1) * stride for dim, stride in zip(arg.shapes, arg.strides))
        torch.frombuffer(arg.buffer, dtype=torch.float32, count=extent).as_strided(
            arg.shapes, arg.strides
        ).fill_(7.0)

    worker = Worker(level=3, num_sub_workers=2)
    producer_handle = worker.register(producer)
    copy_handle = worker.register(support._copy_return_tensor)
    state = support._EmbeddedEntryState(
        generated.node._pypto_embedded_layout, callables={}, sub_ids={}, world_size=1,
        prepared_tensors={}, domain_provider=None, return_copy_callable=copy_handle,
    )
    generated.node._pypto_embedded_state = state
    # Registered callable, state and named HOST storage all exist before fork.
    worker.init()

    routed = []
    packed = []
    node_handle = object()
    def io_arg(binding, tensor, direction):
        routed.append((tensor, direction))
        return tensor
    caller._io_arg = io_arg

    def orchestration(orch, args, config):
        producer_args = TaskArgs()
        producer_args.add_tensor(source, TensorArgType.INOUT)
        producer_args.add_tensor(ready, TensorArgType.INPUT)
        orch.submit_sub(producer_handle, producer_args)

        def submit_node(handle, task_args, actual_config, *, worker):
            assert handle is node_handle and actual_config is config and worker == 7
            packed.append(task_args)
            # This adapter bridges the future L4 transport only. Both emitted
            # bodies and their TaskArgs are real; the inner SUB DAG is native.
            generated.node(orch, task_args, config)

        caller.cluster(
            SimpleNamespace(submit_next_level=submit_node), args, config,
            tensors={"x__ssa_v0": source, "count__ssa_v0": 3, "result__ssa_v0": destination},
            callables={"node": [node_handle]}, nodes=[7], node_sizes=[1], node_bindings=[object()],
            _keep=[], _retain=False,
        )
        buffers[2].shm.buf[0] = 1

    worker.run(orchestration, args=None, config=CallConfig())
    state.complete_invocation()
    assert len(packed) == 1
    assert packed[0].tensor_count() == 2 and packed[0].scalar_count() == 1
    assert packed[0].tensor(0) == source and packed[0].tensor(1) == destination
    assert packed[0].scalar(0) == 3
    assert routed == [(source, TensorArgType.INPUT), (destination, TensorArgType.OUTPUT_EXISTING)]
    assert packed[0].tensor(0).buffer == source.buffer
    assert packed[0].tensor(1).buffer == destination.buffer
    # Source and destination views use different offsets. The final partial
    # element in a 35-byte backing must not make torch.frombuffer reject it.
    expected = bytearray([0xCD]) * nbytes
    positions = (2, 3, 6, 7) if strided else (0, 1, 2, 3)
    for index in positions:
        expected[index * 4 : index * 4 + 4] = struct.pack("<f", 7.0)
    assert bytes(buffers[1].shm.buf) == bytes(expected)
    worker.close()
    worker = None

    # Exercise validation through real mapped arguments and capability rights,
    # not invented convenience attributes or synthetic pointer-backed storage.
    registry = ImportRegistry(ImportContext(deployment=HOST_CPU))
    try:
        def mapped(tensor):
            return MappedArg(
                registry.materialize(tensor.buffer), tensor.byte_offset,
                tuple(tensor.shapes), tuple(tensor.strides), tensor.dtype,
            )

        mapped_source, mapped_destination = mapped(source), mapped(destination)
        wrong_dtype = buffers[1].tensor(
            shapes=destination.shapes, dtype=DataType.INT32,
            strides=destination.strides, byte_offset=destination.byte_offset,
        )
        readonly = mapped(buffers[3].tensor(shapes=(2, 2), dtype=DataType.FLOAT32))
        writeonly = mapped(buffers[4].tensor(shapes=(2, 2), dtype=DataType.FLOAT32))
        assert BufferCapability.DIRECT_STORE not in readonly.imported.capabilities
        assert BufferCapability.DIRECT_LOAD not in writeonly.imported.capabilities
        for arguments, diagnostic in (
            ((mapped_source, mapped(wrong_dtype)), ("dtype", "type")),
            ((mapped_source, readonly), ("store", "write")),
            ((writeonly, mapped_destination), ("load", "read")),
        ):
            try:
                support._copy_return_tensor(MappedArgs(list(arguments), ()))
                raise AssertionError("Invalid return-copy dtype or mapped rights were accepted")
            except ValueError as error:
                assert any(word in str(error).lower() for word in diagnostic), str(error)
        assert bytes(buffers[1].shm.buf) == bytes(expected)
    finally:
        registry.close()
    state.close()
    assert not any(name == "pypto" or name.startswith("pypto.") for name in sys.modules)
    print("C3B_RETURN_COPY_CPU_" + sys.argv[1].upper() + "_PASS")
finally:
    if worker is not None:
        worker.close()
    for buffer in buffers:
        buffer.close()
"""


@pytest.mark.parametrize("view", ["contiguous", "strided"])
def test_c3b_return_copy_uses_real_cpu_dag_and_mapped_views(
    view, normalized_return_copy_program, canonical_support_path, tmp_path, monkeypatch
):
    # This check follows normalized, verified native caller generation. Its
    # current missing callable is a feature RED, not an invalid IR fixture.
    name = "_c3b_return_copy_support_contract"
    spec = importlib.util.spec_from_file_location(name, canonical_support_path)
    assert spec is not None and spec.loader is not None
    support = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, support)
    spec.loader.exec_module(support)
    assert callable(getattr(support, "_copy_return_tensor", None)), (
        "Missing canonical return-copy SUB callable"
    )
    runtime_python = os.environ.get("L4_RUNTIME_ONLY_PYTHON")
    runtime_pythonpath = os.environ.get("L4_RUNTIME_ONLY_PYTHONPATH")
    if not runtime_python or not runtime_pythonpath:
        pytest.skip("Set L4_RUNTIME_ONLY_PYTHON and L4_RUNTIME_ONLY_PYTHONPATH for real Simpler CPU proof")
    package = tmp_path / "return_proof_namespace"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = codegen.DistributedCodegen()._generate_embedded(
        normalized_return_copy_program, entry_name="node"
    )
    caller = codegen.DistributedCodegen().generate(normalized_return_copy_program)
    (package / "entry.py").write_text(source, encoding="utf-8")
    (package / "cluster.py").write_text(caller, encoding="utf-8")
    shutil.copyfile(canonical_support_path, package / "_distributed_support.py")
    result = subprocess.run(
        [runtime_python, "-c", _RETURN_COPY_CPU_PROOF, view],
        cwd=tmp_path,
        env=os.environ
        | {
            "PYTHONPATH": os.pathsep.join((str(tmp_path), runtime_pythonpath)),
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, f"Real CPU return-copy proof failed:\n{result.stdout}\n{result.stderr}"
    assert f"C3B_RETURN_COPY_CPU_{view.upper()}_PASS" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
