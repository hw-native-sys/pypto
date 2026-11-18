# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime-only direct controller contract, with the external worker boundary mocked."""

import ast
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def controller_module():
    path = Path(__file__).parents[3] / "python/pypto/runtime/_cluster_controller.py"
    assert path.is_file(), "portable direct controller is missing"
    spec = importlib.util.spec_from_file_location("portable_cluster_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_controller(controller_module, monkeypatch):
    events = []

    class Worker:
        def register(self, target, *, workers):
            events.append(("register", target, workers))
            return "remote-callable"

        def init(self):
            events.append("init")

        def run(self, callback, *, config):
            events.append("run")
            callback(SimpleNamespace(), None, config)
            events.append("fence")

        def close(self):
            events.append("close")

    worker = Worker()

    def create_worker(**kwargs):
        events.append(("create", kwargs))
        return worker

    context = SimpleNamespace(
        create_worker=create_worker,
        topology=SimpleNamespace(executors=[SimpleNamespace(worker_id=7), SimpleNamespace(worker_id=9)]),
    )
    generated = SimpleNamespace(
        root=lambda *args, **kwargs: events.append(("entry", kwargs["tensors"], kwargs["nodes"])),
    )
    monkeypatch.setattr(controller_module.importlib, "import_module", lambda name: generated)
    import simpler.task_interface as interface  # noqa: PLC0415
    import simpler.worker as runtime  # noqa: PLC0415

    monkeypatch.setattr(runtime, "RemoteCallable", lambda target: target)
    monkeypatch.setattr(interface, "CallConfig", SimpleNamespace)
    spec = dict(
        platform="a2a3sim",
        runtime="host_build_graph",
        startup_timeout_s=10,
        enable_sdma=False,
        persistent=True,
        reset=True,
        node_sizes=[1, 1],
        programs=[
            dict(
                entry_name="root",
                module_prefix="bundle.programs.p0",
                param_names=["count"],
                entries=[dict(entry_name="host", module="orchestration.host", qualname="host")],
                plans=[],
            )
        ],
    )
    return controller_module.ClusterController(context, spec), worker, events, spec


def test_plain_spec_uses_direct_context_and_registers_before_init(setup_controller):
    controller, _, events, _ = setup_controller
    assert events[0][0] == "create"
    assert events[0][1]["num_sub_workers"] == 0
    assert events[1] == ("register", "bundle.programs.p0.orchestration.host:host", [7, 9])
    assert events[2] == "init"
    controller.run(0, [13], {"block_dim": 2})
    assert events[-2:] == [("entry", {"count": 13}, [7, 9]), "fence"]
    controller.close()
    assert events[-1] == "close"


def test_failed_fence_does_not_publish_and_retains_journal_if_close_fails(setup_controller, monkeypatch):
    controller, worker, _, _ = setup_controller

    def failed_run(*args, **kwargs):
        raise RuntimeError("fence failed")

    def failed_close():
        raise RuntimeError("drain unconfirmed")

    monkeypatch.setattr(worker, "run", failed_run)
    monkeypatch.setattr(worker, "close", failed_close)
    with pytest.raises(RuntimeError):
        controller.run(0, [1], {})
    assert controller._active_io is not None
    with pytest.raises(RuntimeError, match="closed|close"):
        controller.run(0, [2], {})
    monkeypatch.setattr(worker, "close", lambda: None)
    controller.close()
    assert controller._active_io is None


def test_runtime_only_imports(controller_module):
    tree = ast.parse(Path(controller_module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not item.name.startswith("pypto") for item in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("pypto")
    source = Path(controller_module.__file__).read_text()
    assert "add_mpirun_worker_group" not in source
    assert "MpiL3GroupSpec" not in source


def test_io_preserves_alias_offsets_and_rejects_cross_rank_writes(controller_module, monkeypatch):
    import simpler.task_interface as interface  # noqa: PLC0415
    from simpler.task_interface import TensorArgType  # noqa: PLC0415

    monkeypatch.setattr(interface, "RemoteTensorRef", lambda handle, **kwargs: SimpleNamespace(**kwargs))
    events = []
    worker = SimpleNamespace(
        remote_malloc=lambda **kwargs: 100 + kwargs["worker"],
        remote_copy_to=lambda *args: None,
        remote_free=lambda handle: events.append(handle),
    )
    tensor = torch.arange(8, dtype=torch.float32)
    io = controller_module._InvocationIO(worker)
    io.prepare({"x": tensor}, (7, 9))
    ref = io.argument(7, tensor[2:6], TensorArgType.INOUT)
    assert ref.offset == 8
    with pytest.raises(NotImplementedError, match="Cross-node"):
        io.argument(9, tensor[4:], TensorArgType.INPUT)
    io.release()
    assert events == [107, 109]


def test_domains_share_across_programs_and_reset_once_per_invocation(setup_controller):
    controller, _, _, _ = setup_controller
    handle = SimpleNamespace(
        members=[SimpleNamespace(node_worker_id=7, local_worker_id=0, global_device_rank=0, domain_rank=0)],
        mapping_size=16,
        released=False,
        freed=False,
    )
    allocations = []
    copies = []
    releases = []
    handle.release = lambda: releases.append(True)
    orch = SimpleNamespace(
        allocate_global_domain=lambda **kwargs: allocations.append(kwargs) or handle,
        copy_to_global_domain=lambda *args, **kwargs: copies.append((args, kwargs)),
    )
    plan = dict(
        pool_key=("shared", "state"),
        name="pypto:shared:0",
        members=[[0, 0, 0]],
        canonical=[["canonical", 8, 16, []]],
        domain=dict(
            generated_name="first",
            slots=[
                dict(
                    generated_name="slot0",
                    logical_nbytes=8,
                    allocated_nbytes=16,
                )
            ],
        ),
    )
    first = controller._domain_provider({"plans": [plan]}, orch)
    from simpler.task_interface import CommBufferSpec  # noqa: PLC0415

    with first(
        name="first",
        members=[(7, 0)],
        window_size=16,
        buffers=[CommBufferSpec(name="slot0", dtype="opaque", count=8, nbytes=16)],
        retain_after_run=True,
    ) as actual:
        assert actual is handle
    assert len(allocations) == 1
    assert copies == []
    assert releases == []
    second = dict(plan, domain=dict(plan["domain"], generated_name="second"))
    controller._domain_provider({"plans": [second, second]}, orch)
    assert len(allocations) == 1
    assert len(copies) == 1
    assert copies[0][0][2] == bytes(16)
    controller.close()
    assert releases == [True]


def test_remote_descriptors_reject_lazy_view_flags(controller_module):
    worker = SimpleNamespace(remote_malloc=lambda **kwargs: 7, remote_copy_to=lambda *args: None)
    view = torch._neg_view(torch.ones(2))
    io = controller_module._InvocationIO(worker)
    with pytest.raises(ValueError, match="lazy"):
        io.prepare({"value": view}, (7,))
    assert not io.storages


def test_call_config_restores_native_runtime_environment(setup_controller, monkeypatch):
    import simpler.task_interface as interface  # noqa: PLC0415

    controller, worker, _, _ = setup_controller
    captured = []

    class CallConfig:
        def __init__(self):
            self._env = SimpleNamespace(ring_heap=0)

        @property
        def runtime_env(self):
            return self._env

    monkeypatch.setattr(interface, "CallConfig", CallConfig)
    monkeypatch.setattr(worker, "run", lambda callback, *, config: captured.append(config))
    controller.run(0, [1], {"runtime_env": {"ring_heap": 65536}})
    assert captured[0].runtime_env.ring_heap == 65536


def test_dispatches_all_prepared_programs_without_recreating_worker(controller_module, setup_controller):
    first, worker, events, spec = setup_controller
    first.close()
    events.clear()
    spec["programs"].append(
        dict(spec["programs"][0], module_prefix="bundle.programs.p1", param_names=["other"])
    )
    context = SimpleNamespace(
        topology=SimpleNamespace(executors=[SimpleNamespace(worker_id=7), SimpleNamespace(worker_id=9)]),
        create_worker=lambda **kwargs: worker,
    )
    controller = controller_module.ClusterController(context, spec)
    assert events[1] == ("register", "bundle.programs.p1.orchestration.host:host", [7, 9])
    controller.run(0, [1], {})
    controller.run(1, [2], {})
    controller.run(0, [3], {})
    assert [event[1] for event in events if isinstance(event, tuple) and event[0] == "entry"] == [
        {"count": 1},
        {"other": 2},
        {"count": 3},
    ]
    assert events.count("init") == 1
    controller.close()


@pytest.mark.parametrize("fail", [False, True])
def test_session_done_follows_mapping_close_and_errors_stop_session(controller_module, monkeypatch, fail):
    events = []
    requests = iter(
        [
            {"op": "run", "program": 0, "call_config": {}},
            {"op": "close"},
        ]
    )

    @contextmanager
    def map_values(request, directory):
        events.append("map")
        try:
            yield [17]
        finally:
            events.append("unmap")

    class Controller:
        def __init__(self, context, spec):
            self._nodes = (7, 9)

        def run(self, program, values, config):
            assert program == 0 and values == [17]
            events.append("fence")
            if fail:
                raise RuntimeError("native fence failed")

        def close(self):
            events.append("close")

    channel = SimpleNamespace(
        connect_session=lambda *args: SimpleNamespace(close=lambda: events.append("socket-close")),
        send_record=lambda sock, record: events.append(record["type"]),
        recv_record=lambda sock: next(requests),
        map_values=map_values,
    )
    monkeypatch.setattr(controller_module, "__package__", "portable_bundle")
    monkeypatch.setitem(sys.modules, "portable_bundle._cluster_channel", channel)
    monkeypatch.setattr(controller_module, "ClusterController", Controller)
    context = SimpleNamespace(
        startup_parameters={
            "address": ["127.0.0.1", 1234],
            "authkey": "00",
            "data_dir": "/private",
            "session": {"startup_timeout_s": 10},
        }
    )
    if fail:
        with pytest.raises(RuntimeError, match="native fence failed"):
            controller_module.main(context)
        assert "done" not in events
        assert events == ["ready", "map", "fence", "unmap", "error", "close", "socket-close"]
    else:
        controller_module.main(context)
        assert events == [
            "ready",
            "map",
            "fence",
            "unmap",
            "done",
            "close",
            "closed",
            "close",
            "socket-close",
        ]
