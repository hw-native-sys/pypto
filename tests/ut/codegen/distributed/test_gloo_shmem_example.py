# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile the real Path B kernels without creating a device context."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pypto import DataType
from pypto.runtime import DeviceTensor, RunConfig


@pytest.fixture(scope="module")
def example():
    path = Path(__file__).resolve().parents[4] / "examples/distributed/08_gloo_shmem_collectives.py"
    spec = importlib.util.spec_from_file_location("gloo_shmem_collectives", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(
    "name, tensors, contexts",
    [
        ("tstore_chip", 2, 1),
        ("tput_chip", 3, 2),
        ("publish_chip", 2, 1),
        ("consume_chip", 2, 1),
        ("allreduce_chip", 2, 1),
    ],
)
def test_chip_compile_materializes_context_scalars(example, tmp_path, name, tensors, contexts):
    meta = torch.empty((1, example.SIZE), dtype=torch.float16)
    compiled = getattr(example, name).compile(
        *([meta] * tensors),
        config=RunConfig(platform="a2a3", save_kernels=True, save_kernels_dir=str(tmp_path)),
    )
    infos, _, _ = compiled._get_metadata()
    # Public metadata may hide compiler-inserted parameters; the assembled
    # signature must still carry every DistributedTensor's context pointer.
    lowered = compiled._lowered_param_infos
    assert sum(info.shape is not None for info in infos) == tensors
    scalar_infos = [info for info in lowered if info.shape is None]
    assert len(scalar_infos) == contexts
    assert all(info.dtype == DataType.UINT64 for info in scalar_infos)
    assert list(Path(compiled.output_dir).rglob("*.pto"))
    # Native borrowed execution validates scalar counts from the exported
    # ChipCallable signature, independently of ParamInfo metadata.
    config = ast.parse((Path(compiled.output_dir) / "kernel_config.py").read_text())
    orchestration = next(
        node.value
        for node in config.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ORCHESTRATION" for target in node.targets)
    )
    assert isinstance(orchestration, ast.Dict)
    signature = next(
        value
        for key, value in zip(orchestration.keys, orchestration.values)
        if isinstance(key, ast.Constant) and key.value == "signature"
    )
    assert isinstance(signature, ast.List)
    assert all(isinstance(entry, ast.Attribute) for entry in signature.elts)
    assert [entry.attr for entry in signature.elts if isinstance(entry, ast.Attribute)][-contexts:] == [
        "SCALAR"
    ] * contexts
    assert len(signature.elts) == tensors + contexts


@pytest.mark.parametrize("shared_helper", [False, True, "cached", "timed"])
def test_direct_dispatch_uses_buffer_taskargs_and_caller_stream(example, monkeypatch, shared_helper):
    calls = []

    class Args:
        def add_tensor(self, tensor, tag):
            calls.append(("tensor", tensor, tag))

        def add_scalar(self, value):
            calls.append(("scalar", value))

    def wrap(**kwargs):
        address = kwargs["address"]
        return SimpleNamespace(
            tensor=lambda shape, dtype: (address, shape, dtype),
            close=lambda: calls.append(("close", address)),
        )

    monkeypatch.setitem(
        sys.modules,
        "pypto.runtime.task_interface",
        SimpleNamespace(
            TaskArgs=Args,
            torch_dtype_to_datatype=lambda dtype: dtype,
        ),
    )
    monkeypatch.setitem(sys.modules, "simpler.buffer", SimpleNamespace(Buffer=SimpleNamespace(wrap=wrap)))
    monkeypatch.setitem(
        sys.modules,
        "simpler.task_interface",
        SimpleNamespace(
            TensorArgType=SimpleNamespace(INPUT="input", OUTPUT_EXISTING="output", INOUT="inout"),
        ),
    )
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: calls.append(("sync",)),
            current_stream=lambda: SimpleNamespace(npu_stream=123),
        ),
        raising=False,
    )
    infos = [SimpleNamespace(shape=(1, 8), direction=SimpleNamespace(name=name)) for name in ("In", "Out")]
    built_configs = []
    compiled = SimpleNamespace(
        _get_metadata=lambda: (infos, [], []),
        chip_callable="chip",
        platform="a2a3",
        _build_call_config=lambda config: built_configs.append(config) or "config",
    )
    totals = None
    chip = SimpleNamespace(
        _impl=SimpleNamespace(device_location=object()),
        _run_chip=lambda *args, **kwargs: calls.append(("run", kwargs["caller_stream"])),
    )
    dispatch = example._run_kernel
    if shared_helper:
        path = Path(example.__file__).with_name("gloo_shmem_utils.py")
        spec = importlib.util.spec_from_file_location("gloo_shmem_utils", path)
        assert spec is not None and spec.loader is not None
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        dispatch = helper.run_kernel
        if shared_helper in ("cached", "timed"):
            plan = helper.DispatchPlan(compiled)

            totals = {} if shared_helper == "timed" else None

            def dispatch(*args):
                return helper.run_kernel(*args, plan=plan, timings=totals)

    dispatch(
        chip,
        compiled,
        DeviceTensor(0x1000, (1, 8), torch.float16),
        DeviceTensor(0x2000, (1, 8), torch.float16),
        0x3000,
    )
    assert calls == [
        ("tensor", (0x1000, (1, 8), torch.float16), "input"),
        ("tensor", (0x2000, (1, 8), torch.float16), "output"),
        ("scalar", 0x3000),
        ("sync",),
        ("run", 123),
        ("close", 0x1000),
        ("close", 0x2000),
    ]

    if shared_helper in ("cached", "timed"):
        calls.clear()
        dispatch(
            chip,
            compiled,
            DeviceTensor(0x4000, (1, 8), torch.float16),
            DeviceTensor(0x5000, (1, 8), torch.float16),
            0x6000,
        )
        assert len(built_configs) == 1
        assert calls[0][1][0] == 0x4000
        assert calls[1][1][0] == 0x5000
        assert calls[2] == ("scalar", 0x6000)
        assert calls[-2:] == [("close", 0x4000), ("close", 0x5000)]
        if shared_helper == "timed":
            assert totals is not None
            assert set(totals) == {
                "plan",
                "taskargs_init",
                "tensor_inspect",
                "buffer_register",
                "tensor_bind",
                "scalar_bind",
                "torch_sync",
                "stream_lookup",
                "native_dispatch",
                "buffer_release",
            }
            assert all(value >= 0 for value in totals.values())
        with pytest.raises(ValueError, match="different compiled"):
            helper.run_kernel(chip, SimpleNamespace(), plan=plan)

    def fail_dispatch(*args, **kwargs):
        calls.append(("failed_run",))
        raise RuntimeError("dispatch failed")

    def close_worker():
        calls.append(("worker_close",))

    chip._run_chip = fail_dispatch
    chip.close = close_worker
    calls.clear()
    with pytest.raises(RuntimeError, match="dispatch failed"):
        dispatch(
            chip,
            compiled,
            DeviceTensor(0x1000, (1, 8), torch.float16),
            DeviceTensor(0x2000, (1, 8), torch.float16),
            0x3000,
        )
    assert calls[-4:] == [
        ("failed_run",),
        ("worker_close",),
        ("close", 0x1000),
        ("close", 0x2000),
    ]

    def fail_close():
        calls.append(("worker_close_failed",))
        raise RuntimeError("teardown failed")

    chip.close = fail_close
    calls.clear()
    with pytest.raises(RuntimeError, match="teardown failed"):
        dispatch(
            chip,
            compiled,
            DeviceTensor(0x1000, (1, 8), torch.float16),
            DeviceTensor(0x2000, (1, 8), torch.float16),
            0x3000,
        )
    assert calls[-2:] == [("failed_run",), ("worker_close_failed",)]
    assert not any(call[0] == "close" for call in calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
