# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Recursive CLUSTER artifact generation, cold contracts and runtime source closure."""

import ast
import ctypes
import hashlib
import importlib
import io
import json
import tarfile
from types import SimpleNamespace

import pypto.language as pl
import pytest
import torch
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.jit._artifact_manifest import BuildKind
from pypto.runtime import _prebuilt


def _config():
    return ir.ClusterConfig(
        [ir.ClusterNodeSpec("node-a", [0, 1]), ir.ClusterNodeSpec("node-b", [2, 3])],
        "/env/bin/python",
    )


def _program(*, lower=True):
    @pl.program
    class Recursive:
        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def chip(
            self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
        ):
            return 0

        @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
        def unused_chip(self, x: pl.Tensor[[8], pl.FP32]):
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.SubWorker)
        def observe(x: pl.Tensor[[8], pl.FP32]):
            assert x.numel() == 8

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def first(
            self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
        ) -> pl.Tensor[[8], pl.FP32]:
            self.observe(x)
            self.chip(x, count, out, worker=0)
            return x

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def second(
            self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32], out: pl.Out[pl.Tensor[[8], pl.FP32]]
        ):
            self.chip(x, count, out, worker=1)
            return 0

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def unused_host(self, x: pl.Tensor[[8], pl.FP32]):
            self.unused_chip(x, worker=0)
            return 0

        @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
        def root(
            self,
            x: pl.Tensor[[8], pl.FP32],
            count: pl.Scalar[pl.INT32],
            left: pl.Out[pl.Tensor[[8], pl.FP32]],
            right: pl.Out[pl.Tensor[[8], pl.FP32]],
        ):
            self.first(x, count, left, worker=0)
            self.second(x, count, right, worker=1)
            return 0

    if not lower:
        return Recursive
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        program = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(Recursive)
        assert program.functions[program.get_global_var("chip")].func_type == ir.FunctionType.Orchestration
        # Verify the real native complete-program source before testing packaging.
        compile(codegen.DistributedCodegen().generate(program), "cluster_orch.py", "exec")
        for name in ("first", "second"):
            source = codegen.DistributedCodegen()._generate_embedded(program, entry_name=name)
            compile(source, f"{name}.py", "exec")
        return program
    finally:
        backend.reset_for_testing()


def _emit(program, root):
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        files = pto_backend.generate(program, str(root), skip_ptoas=True)
    finally:
        backend.reset_for_testing()
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    compiled = ir.DistributedCompiledProgram(program, str(root), cluster_config=_config())
    return compiled, files, json.loads((root / "distributed_meta.json").read_text())


@pytest.fixture(scope="module")
def cluster_program():
    return _program()


@pytest.fixture
def cluster_artifact(tmp_path, cluster_program):
    root = tmp_path / "generated"
    return root, *_emit(cluster_program, root)


def _layout(source):
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_pypto_embedded_layout"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    return json.loads(json.dumps(ast.literal_eval(assignment.value.args[0])))


def test_c3c_recursive_entries_keep_full_logical_chip_paths(cluster_artifact):
    root, _, files, metadata = cluster_artifact
    assert "orchestration/cluster_orch.py" in files
    assert "orchestration/host_orch.py" not in files
    assert not any("unused_" in path for path in files)
    closure = metadata["l3_entries"]
    assert closure["version"] == 1
    entries = closure["entries"]
    assert [entry["entry_name"] for entry in entries] == ["first", "second"]
    expected = {}
    for entry in entries:
        name = entry["entry_name"]
        prefix = f"next_levels/{name}"
        assert entry["logical_path"] == prefix
        assert entry["module"] == f"next_levels.{name}.orchestration.host_orch"
        assert entry["qualname"] == name
        assert entry["chip_paths"] == {"chip": f"{prefix}/next_levels/chip"}
        assert entry["layout"] == _layout(files[f"{prefix}/orchestration/host_orch.py"])
        assert entry["layout"]["version"] == 2
        assert [row[0:3:2] for row in entry["layout"]["tensor_slots"]] == [[0, 0], [2, 1]]
        assert [row[0:3:2] for row in entry["layout"]["scalar_slots"]] == [[1, 0]]
        assert entry["host_tensor_indices"] == [0, 1]
        assert entry["local_device_count"] == 2
        assert entry["minimum_sub_workers"] == (2 if name == "first" else 0)
        assert entry["sub_worker_paths"] == (
            {"observe": f"{prefix}/sub_workers/observe.py"} if name == "first" else {}
        )
        assert entry["return_copy_callable"] == (
            {"module": "orchestration._distributed_support", "qualname": "_copy_return_tensor"}
            if name == "first"
            else None
        )
        assert entry["view_geometry"] == {"version": 1, "window_slots": []}
        expected[f"{prefix}/next_levels/chip"] = root / prefix / "next_levels/chip"
    # Equal CHIP basenames are two distinct registration paths, not one flattened key.
    assert _prebuilt.chip_directories(root, BuildKind.DISTRIBUTED) == expected
    # A valid-looking stale subtree must never become part of this build.
    stale = root / "next_levels/stale/next_levels/chip/kernel_config.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("raise AssertionError('stale config executed')\n")
    assert _prebuilt.chip_directories(root, BuildKind.DISTRIBUTED) == expected


def test_cluster_artifact_cold_reload_validates_config_and_layout(cluster_artifact):
    root, compiled, _, metadata = cluster_artifact
    restored = ir.DistributedCompiledProgram.from_dir(root)
    assert restored.entry_name == compiled.entry_name
    assert restored.cluster_config == compiled.cluster_config
    assert restored.l3_entries == compiled.l3_entries

    incompatible = ir.ClusterConfig(
        [ir.ClusterNodeSpec("other-a", [0, 1]), ir.ClusterNodeSpec("other-b", [2, 3])],
        "/env/bin/python",
    )
    with pytest.raises(ValueError, match="must match"):
        ir.DistributedCompiledProgram.from_dir(root, cluster_config=incompatible)

    metadata["l3_entries"]["entries"][0]["layout"]["version"] = 99
    (root / "distributed_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid l3_entries"):
        ir.DistributedCompiledProgram.from_dir(root)


def test_cluster_compile_then_cold_reload_rejects_missing_child(tmp_path):
    root = tmp_path / "build"
    built = ir.compile(_program(lower=False), output_dir=str(root), cluster_config=_config(), skip_ptoas=True)
    restored = ir.DistributedCompiledProgram.from_dir(root)
    assert restored.entry_name == built.entry_name
    entry = restored.l3_entries["entries"][0]
    (root / entry["logical_path"] / "orchestration/host_orch.py").unlink()
    with pytest.raises(ValueError, match="invalid l3_entries"):
        ir.DistributedCompiledProgram.from_dir(root)


@pytest.fixture(scope="session")
def cluster_startup_cache(tmp_path_factory):
    """Keep activated content-addressed roots stable for this test process."""
    return tmp_path_factory.mktemp("cluster-startup-cache")


@pytest.fixture
def cluster_runner_harness(cluster_artifact, cluster_startup_cache, tmp_path, monkeypatch):
    """Keep native generated root/metadata and replace only remote process/device I/O."""
    runner = importlib.import_module("pypto.runtime.cluster_runner")
    from simpler.callable_identity import CallableHandle  # noqa: PLC0415
    from simpler.mpi_direct_bootstrap import decode_startup, encode_startup  # noqa: PLC0415
    from simpler.startup_artifacts import (  # noqa: PLC0415
        StartupArtifactSpec,
        activate_startup_artifacts,
        install_startup_artifacts,
    )
    from simpler.task_interface import RemoteAddressSpace, RemoteBufferHandle  # noqa: PLC0415
    from simpler.worker import RemoteCallable  # noqa: PLC0415

    root, compiled, _, metadata = cluster_artifact
    events, registrations, tasks = [], [], []
    chips = {
        path: object() for entry in metadata["l3_entries"]["entries"] for path in entry["chip_paths"].values()
    }
    state = SimpleNamespace(events=events, registrations=registrations, tasks=tasks, fail=False)

    class Worker:
        def __init__(self, *, level, **kwargs):
            assert level == 4
            self.memory = {}
            self.in_graph = False
            self.initialized = False
            state.worker = self

        def register(self, target, *, workers=None):
            registrations.append((target, workers))
            remote = isinstance(target, RemoteCallable)
            return CallableHandle(
                "sha256:" + hashlib.sha256(str(target).encode()).hexdigest(),
                "PYTHON_IMPORT" if remote else "CHIP_CALLABLE",
                "REMOTE_TASK_DISPATCHER" if remote else "LOCAL_CHIP",
            )

        def add_mpirun_worker_group(self, spec):
            pytest.fail("CLUSTER preparation must launch Direct MPI, not an MPI worker group")

        def _inner_registry_entries_for_spec(self, spec):
            return []

        def init(self, **kwargs):
            self.initialized = True
            events.append("ready")

        def remote_malloc(self, *, worker, nbytes):
            assert not self.in_graph, "Remote allocation must precede graph submission"
            handle = RemoteBufferHandle._from_remote_allocation(
                worker_id=worker,
                buffer_id=len(self.memory) + 1,
                generation=1,
                address_space=RemoteAddressSpace.REMOTE_DEVICE,
                nbytes=nbytes,
            )
            self.memory[handle] = bytearray(nbytes)
            events.append(("malloc", worker))
            return handle

        def remote_copy_to(self, handle, ptr, nbytes, *, offset=0):
            assert not self.in_graph, "Remote upload must precede graph submission"
            events.append(("upload", handle.worker_id))
            self.memory[handle][offset : offset + nbytes] = ctypes.string_at(ptr, nbytes)

        def remote_copy_from(self, handle, ptr, nbytes, *, offset=0):
            events.append(("copyback", handle.worker_id))
            ctypes.memmove(ptr, bytes(self.memory[handle][offset : offset + nbytes]), nbytes)

        def remote_free(self, handle):
            events.append(("free", handle.worker_id))
            del self.memory[handle]
            handle._mark_released()

        def run(self, fn, args=None, config=None):
            def submit(handle, args, config, *, worker):
                tasks.append((handle, args, worker))

            self.in_graph = True
            try:
                fn(SimpleNamespace(submit_next_level=submit), args, config)
            finally:
                self.in_graph = False
            if state.fail:
                events.append("failed")
                raise RuntimeError("remote DAG failed")
            events.append("drained")

        def close(self):
            if self.initialized:
                events.append("closed")
            self.memory.clear()

    class DirectSession:
        """Run the installed portable engine across the replaced process boundary."""

        def __init__(self, topology, startup_by_rank, controller_target, config):
            startup_by_rank = {
                rank: decode_startup(encode_startup(record)) for rank, record in startup_by_rank.items()
            }
            records = startup_by_rank[0]["startup_artifacts"]
            installed = install_startup_artifacts(records, cluster_startup_cache)
            activate_startup_artifacts(installed)
            artifacts = []
            for record in records:
                archive = tmp_path / (record["sha256"] + ".tar.gz")
                archive.write_bytes(record["payload"])
                artifacts.append(StartupArtifactSpec(archive, record["sha256"], record["module_prefix"]))
            state.spec = SimpleNamespace(
                startup_artifacts=tuple(artifacts),
                startup_by_rank=startup_by_rank,
            )
            parameters = startup_by_rank[0]["parameters"]
            self.data_dir = parameters["data_dir"]
            module = importlib.import_module(controller_target.split(":")[0])
            self.channel = importlib.import_module(module.__package__ + "._cluster_channel")
            context = SimpleNamespace(
                topology=SimpleNamespace(
                    executors=tuple(SimpleNamespace(worker_id=worker_id) for worker_id in (7, 13))
                ),
                create_worker=lambda **kwargs: runner.Worker(level=4, **kwargs),
            )
            self.controller = module.ClusterController(context, parameters["session"])

        def request(self, record):
            with self.channel.map_values(record, self.data_dir) as values:
                try:
                    self.controller.run(record["program"], values, record["call_config"])
                finally:
                    del values
            return {"type": "done"}

        def close(self):
            self.controller.close()

    monkeypatch.setattr(runner, "Worker", Worker)
    monkeypatch.setattr(runner, "_DirectSession", DirectSession)
    monkeypatch.setattr(
        runner,
        "_assemble_cluster_chips",
        lambda compiled: {path: (chip, compiled.cluster_config.runtime, {}) for path, chip in chips.items()},
    )
    # C3c's metadata fixture intentionally uses the real A3 fabric profile.
    # The process/device boundary is replaced here, so select its matching
    # real platform rather than weakening MpiL3GroupSpec's profile validation.
    state.compiled = ir.DistributedCompiledProgram.from_dir(root, platform="a2a3")
    state.chips = chips
    return runner, state


def _i2_compile(root, *, second=False, two_slots=False, reverse=False, dtype="FP32", nbytes=32, repeat=False):
    """Compile native roots with identical callable names and distinct logical WBs."""
    elements = nbytes // 4  # FP32/INT32 views must cover the whole logical window.
    payload = "decode_payload" if second else "prefill_payload"
    signal = "decode_credit" if second else "prefill_credit"
    allocations = [f"{payload} = pld.alloc_window_buffer({nbytes})"]
    if two_slots:
        allocations.append(f"{signal} = pld.alloc_window_buffer(4)")
    if reverse:
        allocations.reverse()
    allocation_source = "\n        ".join(allocations)
    credit_formal = ", credit: pld.DistributedTensor[[1], pl.INT32]" if two_slots else ""
    credit_arg = ", credit" if two_slots else ""
    credit_window = f"credit = pld.window({signal}, [1], dtype=pl.INT32)" if two_slots else ""
    intermediate = "scratch = pl.create_tensor([8], dtype=pl.FP32)" if repeat else ""
    target = "scratch" if repeat else "out"
    repeat_dispatch = f"out = self.chip(scratch, count, data{credit_arg}, out, worker=1)" if repeat else ""
    program = pl.parse_program(f"""
@pl.program
class SharedEntries:
    @pl.function(type=pl.FunctionType.Orchestration, level=pl.Level.CHIP, role=pl.Role.Orchestrator)
    def chip(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
             data: pld.DistributedTensor[[{elements}], pl.{dtype}]{credit_formal},
             out: pl.Out[pl.Tensor[[8], pl.FP32]]) -> pl.Tensor[[8], pl.FP32]:
        return out
    @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
    def node(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
             data: pld.DistributedTensor[[{elements}], pl.{dtype}]{credit_formal},
             out: pl.Out[pl.Tensor[[8], pl.FP32]]) -> pl.Tensor[[8], pl.FP32]:
        {intermediate}
        {target} = self.chip(x, count, data{credit_arg}, {target}, worker=1)
        {repeat_dispatch}
        return out
    @pl.function(level=pl.Level.CLUSTER_0, role=pl.Role.Orchestrator)
    def root(self, x: pl.Tensor[[8], pl.FP32], count: pl.Scalar[pl.INT32],
             out: pl.Out[pl.Tensor[[8], pl.FP32]]) -> pl.Tensor[[8], pl.FP32]:
        {allocation_source}
        data = pld.window({payload}, [{elements}], dtype=pl.{dtype})
        {credit_window}
        scratch = pl.create_tensor([8], dtype=pl.FP32)
        scratch = self.node(x, count, data{credit_arg}, scratch, worker=0)
        out = self.node(x, count, data{credit_arg}, out, worker=1)
        return out
""")
    return ir.compile(
        program, output_dir=str(root), skip_ptoas=True, cluster_config=_config(), platform="a2a3"
    )


@pytest.fixture
def i2_programs(tmp_path):
    return (
        _i2_compile(tmp_path / "prefill"),
        _i2_compile(tmp_path / "decode", second=True),
    )


@pytest.fixture
def direct_adapter_harness(cluster_runner_harness, monkeypatch):
    """Replace the external session while forbidding the legacy MPI group path."""
    from simpler.mpi_direct_bootstrap import decode_startup, encode_startup  # noqa: PLC0415

    runner, state = cluster_runner_harness
    state.sessions = []
    base = runner.Worker

    class Worker(base):
        def add_mpirun_worker_group(self, spec):
            pytest.fail("CLUSTER preparation must launch Direct MPI, not an MPI worker group")

        def _inner_registry_entries_for_spec(self, spec):
            return []

    class DirectSession:
        def __init__(self, topology, startup_by_rank, controller_target, config):
            self.topology = topology
            self.startup_by_rank = {
                rank: decode_startup(encode_startup(record)) for rank, record in startup_by_rank.items()
            }
            self.controller_target = controller_target
            self.config = config
            self.requests = []
            self.closed = False
            state.sessions.append(self)

        def request(self, record):
            assert not self.closed
            self.requests.append(record)
            if state.fail:
                raise RuntimeError("direct controller DAG failed")
            return {"type": "done"}

        def close(self):
            self.closed = True

    monkeypatch.setattr(runner, "Worker", Worker)
    monkeypatch.setattr(runner, "_DirectSession", DirectSession, raising=False)
    monkeypatch.setattr(
        runner,
        "_assemble_cluster_chips",
        lambda compiled: {
            path: (object(), compiled.cluster_config.runtime, {})
            for entry in compiled.l3_entries["entries"]
            for path in entry["chip_paths"].values()
        },
    )
    return runner, state


def test_direct_adapter_single_program_still_uses_session_source_bundle(i2_programs, direct_adapter_harness):
    _, state = direct_adapter_harness
    left, _ = i2_programs
    with left.prepare():
        session = state.sessions[0]
        startup = session.startup_by_rank[1]
        assert "startup_hook" not in startup
        artifact = startup["startup_artifacts"][0]
        with tarfile.open(fileobj=io.BytesIO(artifact["payload"]), mode="r:gz") as archive:
            for entry in left.l3_entries["entries"]:
                source = archive.extractfile(
                    f"programs/p0/{entry['logical_path']}/orchestration/host_orch.py"
                ).read()
                assert b"'domain_bindings': []" in source


def test_direct_adapter_registers_packaged_subworkers_on_executor_ranks(cluster_runner_harness):
    _, state = cluster_runner_harness
    with state.compiled.prepare():
        records = state.spec.startup_by_rank
        prefix = records[0]["startup_artifacts"][0]["module_prefix"]
        assert records[0]["inner_l3_worker"] == []
        for rank in (1, 2):
            registered = records[rank]["inner_l3_worker"]
            assert registered
            assert all(entry["kind"] == "PYTHON_IMPORT" for entry in registered)
            assert any(
                entry["target"].startswith(f"{prefix}.programs.p0.") and entry["target"].endswith(":observe")
                for entry in registered
            )
            assert any(
                entry["target"] == f"{prefix}.orchestration._distributed_support:_copy_return_tensor"
                for entry in registered
            )


def test_direct_adapter_run_failure_keeps_caller_outputs(i2_programs, direct_adapter_harness):
    _, state = direct_adapter_harness
    left, right = i2_programs
    with left.prepare(extra_compiled=[right]) as prepared:
        session = state.sessions[0]
        output = torch.full((8,), -29.0)
        original = output.clone()
        state.fail = True
        with pytest.raises(RuntimeError, match="direct controller DAG failed"):
            prepared.run(right, torch.ones(8), 23, output)
        assert session.requests
        assert torch.equal(output, original)


def test_direct_adapter_reuses_session_for_programs_and_late_tensors(i2_programs, direct_adapter_harness):
    _, state = direct_adapter_harness
    left, right = i2_programs
    with left.prepare(extra_compiled=[right]) as prepared:
        session = state.sessions[0]
        outputs = [torch.zeros(8), torch.zeros(8)]
        for program, output in zip((left, right), outputs, strict=True):
            assert prepared.run(program, torch.ones(8), 7, output) is None
        assert len(session.requests) == 2 and len(state.sessions) == 1
        assert not session.closed
    assert session.closed
