# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the real Path B kernels without creating a device context."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import DataType
from pypto._artifact_contract import ArtifactExecutionMode
from pypto.ir._compile_contract import analyze_contract
from pypto.ir.compiled_artifact import CompiledArtifact

from examples.distributed.gloo_shmem_utils import compile_eager_kernel, load_eager_kernel
from examples.models.qwen3_tp import runner as qwen_runner


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[4] / "examples/distributed/08_gloo_shmem_collectives.py"
    spec = importlib.util.spec_from_file_location("gloo_shmem_collectives", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(path.parent))
        spec.loader.exec_module(module)
        yield module
    sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(
    "name,tensors,data_indices",
    [
        ("tstore_chip", 2, (1,)),
        ("tput_chip", 3, (1, 2)),
        ("publish_chip", 2, (1,)),
        ("consume_chip", 2, (0,)),
        ("allreduce_chip", 2, (0,)),
        ("local_compute_chip", 2, ()),
    ],
)
def test_eager_compile_preserves_storage_and_explicit_contexts(
    example, tmp_path, monkeypatch, name, tensors, data_indices
):
    monkeypatch.setattr(CompiledArtifact, "load", lambda self: None)
    meta = torch.empty((1, example.SIZE), dtype=torch.float16)
    artifact = compile_eager_kernel(getattr(example, name), *([meta] * tensors), output_dir=str(tmp_path))
    artifact.execution_capabilities.require(ArtifactExecutionMode.KERNEL)
    with pytest.raises(ValueError):
        artifact.execution_capabilities.require(ArtifactExecutionMode.PROGRAM)
    restored = load_eager_kernel(tmp_path)
    assert restored.metadata == artifact.metadata
    contexts = len(data_indices)
    assert artifact.metadata.comm_context_bindings == tuple(enumerate(data_indices, start=tensors))
    params = artifact.metadata.parameters
    assert len(params) == tensors + contexts
    assert all(p.shape == (1, example.SIZE) for p in params[:tensors])
    assert all(p.shape is None and p.dtype == DataType.UINT64 for p in params[tensors:])
    config = (tmp_path / "kernel_config.py").read_text()
    orchestration = config.split("ORCHESTRATION = {", 1)[1].split("}", 1)[0]
    # KERNEL lowering adds the context slots once; C++ codegen must not add them again.
    assert orchestration.count("_D.SCALAR") == contexts
    assert list(tmp_path.rglob("*.pto"))


def test_ordinary_kernel_contract_explains_missing_communication_context(example):
    jit = example.tput_chip
    spec, _ = jit._resolve_specialization((), {}, allow_signature_mode=True)
    program, _ = jit._compile_to_program_with_rename_map(
        spec.tensor_meta, spec.scalar_dtypes, spec.constexpr_values, spec.per_func_dyn, pl
    )
    with pytest.raises(ValueError, match="require explicit communication contexts"):
        analyze_contract(program, ArtifactExecutionMode.KERNEL)


def test_qwen_runner_rejects_legacy_artifact_before_assembly(example, monkeypatch, tmp_path):
    # Put an incompatible entry after a valid one. The Runner must validate
    # the entire selected bundle before loading even the first callable.
    with monkeypatch.context() as patch:
        patch.setattr(CompiledArtifact, "load", lambda self: None)
        meta = torch.empty((1, example.SIZE), dtype=torch.float16)
        fresh = compile_eager_kernel(
            example.local_compute_chip, meta, meta, output_dir=str(tmp_path / "fresh")
        )
    assert fresh.metadata is not None
    legacy = fresh.metadata.record()
    del legacy["comm_context_bindings"]
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "compiled_meta.json").write_text(json.dumps(legacy))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "world_size": qwen_runner.k.TP_WORLD,
                "source_sha256": hashlib.sha256(Path(qwen_runner.k.__file__).read_bytes()).hexdigest(),
                "entries": ["fresh", "legacy"],
            }
        )
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A legacy artifact must be rejected before callable or device initialization")

    monkeypatch.setattr(CompiledArtifact, "load", forbidden)
    monkeypatch.setattr(qwen_runner.pypto.torch, "init", forbidden)
    monkeypatch.setattr(qwen_runner, "acquire_gloo_shmem_window", forbidden)
    monkeypatch.setattr(qwen_runner.dist, "barrier", forbidden)
    with pytest.raises(ValueError, match="legacy requires explicit communication-context metadata"):
        qwen_runner.Runner(tmp_path, tmp_path, rank=0, local_rank=0, capture=False)


@pytest.mark.parametrize("capture", [False, True])
def test_qwen_runner_clears_signal_before_rank_rendezvous(monkeypatch, tmp_path, capture):
    events = []
    device = "npu:0"
    signal = SimpleNamespace(device=device, zero_=lambda: events.append("zero"))

    def synchronize(target):
        assert target == device
        events.append("sync")

    def shmem_view(slot, shape, dtype):
        return signal if slot == "signal" else object()

    def barrier():
        assert events == ["zero", "sync"]
        events.append("barrier")

    artifact = SimpleNamespace(runtime_name="host_build_graph")
    session = SimpleNamespace(require_config=lambda: None, ensure_callable=lambda *_: None)
    monkeypatch.setattr(qwen_runner, "get_process_kernel_state", lambda: session)
    monkeypatch.setattr(qwen_runner, "load_eager_kernel", lambda *_: artifact)
    monkeypatch.setattr(qwen_runner.pypto.torch, "init", lambda **_: None)
    monkeypatch.setattr(
        qwen_runner.torch,
        "npu",
        SimpleNamespace(synchronize=synchronize, Stream=lambda **_: None),
        raising=False,
    )
    monkeypatch.setattr(
        qwen_runner, "acquire_gloo_shmem_window", lambda **_: SimpleNamespace(as_torch_tensor=shmem_view)
    )
    monkeypatch.setattr(qwen_runner.dist, "barrier", barrier)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "world_size": qwen_runner.k.TP_WORLD,
                "source_sha256": hashlib.sha256(Path(qwen_runner.k.__file__).read_bytes()).hexdigest(),
                "entries": ["local"],
            }
        )
    )
    qwen_runner.Runner(tmp_path, tmp_path, rank=0, local_rank=0, capture=capture)
    assert events == ["zero", "sync", "barrier"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
